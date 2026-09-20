#!/usr/bin/env python3
"""ZCP-driven GA search baselines on the FULL LoNAS LLaMA-7B 2x5 search space.

Search space per layer:
  - LoRA rank ∈ {32, 28}   (applied to all 7 LoRA targets: q,k,v,o,gate,up,down)
  - FFN intermediate ∈ {5504, 6880, 8256, 9632, 11008}

Methods:
  - snip      : sum |grad * weight| over FFN, real wikitext calibration
  - gradnorm  : sum ||grad||_2 over FFN, real wikitext calibration
  - synflow   : abs-weight forward on ones, sum |grad * weight| over FFN (data-free)
  - wpca      : forward-only PCA over gate_proj activations (Wang et al., ICLR 2025)

Implementation notes:
  - Model kept as PEFT (un-merged). LoRA matrices A,B held in original positions.
  - Rank switching: zero rows/cols of A,B beyond r; adjust scaling = scl * max_r / r.
  - FFN switching: slice base_layer weight + lora_A/lora_B in/out_features.
  - All slicing uses GPU views; restore points data references back to full tensors.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

OUT = Path(os.environ.get("NSC_OUT", "outputs"))

# ── Search space ───────────────────────────────────────────────────
N_LAYERS = 32
HIDDEN = 4096
FFN_FULL = 11008
VOCAB_SIZE = 32000
RANK_CANDIDATES = [32, 28]
FFN_CANDIDATES = [11008, 9632, 8256, 6880, 5504]
PARAM_BUDGET_B = 5.7

LORA_PROJS_ATTN = ("q_proj", "k_proj", "v_proj")  # LoNAS adapter has no o_proj LoRA
LORA_PROJS_FFN = ("gate_proj", "up_proj", "down_proj")
LORA_PROJS_ALL = LORA_PROJS_ATTN + LORA_PROJS_FFN

OUT_DIR = Path(__file__).parent


# ── Utils ──────────────────────────────────────────────────────────

def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def base_param_count() -> int:
    return 2 * VOCAB_SIZE * HIDDEN + N_LAYERS * (4 * HIDDEN * HIDDEN + 2 * HIDDEN)


def count_params_from_subnet(subnet) -> float:
    return (base_param_count() + 3 * HIDDEN * sum(subnet["ffn_dims"])) / 1e9


def make_subnet(lora_ranks, ffn_dims):
    return {"lora_ranks": list(lora_ranks), "ffn_dims": list(ffn_dims)}


def get_decoder_layer(model, li: int):
    return model.base_model.model.model.layers[li]


def get_lora_mod(layer, proj_name: str):
    if proj_name in LORA_PROJS_ATTN:
        return getattr(layer.self_attn, proj_name)
    return getattr(layer.mlp, proj_name)


# ── Model loading (PEFT, un-merged) ────────────────────────────────

def load_peft_supernet(base_path: str, adapter_path: str, dtype):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tok = AutoTokenizer.from_pretrained(base_path, use_fast=False, local_files_only=True)
    tok.padding_side = "left"
    tok.pad_token_id = 0
    base = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=dtype, device_map="cpu", local_files_only=True
    )
    base.config.pad_token_id = 0
    model = PeftModel.from_pretrained(base, adapter_path, torch_dtype=dtype)
    return model, tok


# ── State snapshots ────────────────────────────────────────────────

def snapshot_state(model):
    """Snapshot full LoRA A/B/scaling and base_layer weights (GPU references)."""
    state = {}
    for li in range(N_LAYERS):
        layer = get_decoder_layer(model, li)
        state[li] = {}
        for pn in LORA_PROJS_ALL:
            mod = get_lora_mod(layer, pn)
            A_param = mod.lora_A["default"].weight
            B_param = mod.lora_B["default"].weight
            state[li][pn] = {
                "A_full": A_param.data.clone(),
                "B_full": B_param.data.clone(),
                "scl_orig": float(mod.scaling["default"]),
                "max_r": int(A_param.shape[0]),
                "base_full": mod.base_layer.weight.data,  # GPU reference
                "base_full_shape": tuple(mod.base_layer.weight.data.shape),
            }
    return state


# ── Rank + FFN apply / restore ─────────────────────────────────────

def apply_subnet(model, subnet, state):
    ranks = subnet["lora_ranks"]
    ffns = subnet["ffn_dims"]
    for li in range(N_LAYERS):
        layer = get_decoder_layer(model, li)
        r = ranks[li]
        k = ffns[li]

        for pn in LORA_PROJS_ALL:
            mod = get_lora_mod(layer, pn)
            s = state[li][pn]
            A_full = s["A_full"]
            B_full = s["B_full"]
            scl_orig = s["scl_orig"]
            max_r = s["max_r"]
            if r < max_r:
                A_masked = A_full.clone()
                A_masked[r:, :] = 0
                B_masked = B_full.clone()
                B_masked[:, r:] = 0
                mod.lora_A["default"].weight.data = A_masked
                mod.lora_B["default"].weight.data = B_masked
                mod.scaling["default"] = scl_orig * max_r / r
            else:
                mod.lora_A["default"].weight.data = A_full
                mod.lora_B["default"].weight.data = B_full
                mod.scaling["default"] = scl_orig

        for pn in ("gate_proj", "up_proj"):
            mod = getattr(layer.mlp, pn)
            s = state[li][pn]
            mod.base_layer.weight.data = s["base_full"][:k, :]
            mod.base_layer.out_features = k
            cur_B = mod.lora_B["default"].weight.data
            mod.lora_B["default"].weight.data = cur_B[:k, :]
            mod.lora_B["default"].out_features = k

        mod = layer.mlp.down_proj
        s = state[li]["down_proj"]
        mod.base_layer.weight.data = s["base_full"][:, :k]
        mod.base_layer.in_features = k
        cur_A = mod.lora_A["default"].weight.data
        mod.lora_A["default"].weight.data = cur_A[:, :k]
        mod.lora_A["default"].in_features = k


def restore_subnet(model, state):
    for li in range(N_LAYERS):
        layer = get_decoder_layer(model, li)
        for pn in LORA_PROJS_ALL:
            mod = get_lora_mod(layer, pn)
            s = state[li][pn]
            mod.lora_A["default"].weight.data = s["A_full"]
            mod.lora_A["default"].in_features = s["A_full"].shape[1]
            mod.lora_B["default"].weight.data = s["B_full"]
            mod.lora_B["default"].out_features = s["B_full"].shape[0]
            mod.scaling["default"] = s["scl_orig"]
            mod.base_layer.weight.data = s["base_full"]
            if pn in ("gate_proj", "up_proj"):
                mod.base_layer.out_features = s["base_full_shape"][0]
            elif pn == "down_proj":
                mod.base_layer.in_features = s["base_full_shape"][1]


# ── Grad mode for ZCPs ─────────────────────────────────────────────

def set_ffn_grad_mode(model, enabled: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    if enabled:
        for li in range(N_LAYERS):
            layer = get_decoder_layer(model, li)
            for pn in LORA_PROJS_FFN:
                mod = getattr(layer.mlp, pn)
                mod.base_layer.weight.requires_grad_(True)


def iter_ffn_weights(model):
    for li in range(N_LAYERS):
        layer = get_decoder_layer(model, li)
        for pn in LORA_PROJS_FFN:
            mod = getattr(layer.mlp, pn)
            yield li, pn, mod.base_layer.weight


# ── Proxy data ─────────────────────────────────────────────────────

def make_real_proxy_batch(tokenizer, batch_size: int, seq_len: int, device):
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    import datasets as hf_ds
    ds = hf_ds.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts, chars = [], 0
    target_chars = batch_size * seq_len * 8
    for row in ds:
        text = row.get("text", "").strip()
        if not text:
            continue
        texts.append(text)
        chars += len(text)
        if chars >= target_chars:
            break
    ids = tokenizer("\n\n".join(texts), return_tensors="pt", add_special_tokens=True,
                    truncation=False)["input_ids"][0]
    need = batch_size * seq_len
    if ids.numel() < need:
        ids = ids.repeat(math.ceil(need / ids.numel()))
    input_ids = ids[:need].reshape(batch_size, seq_len).to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, device=device),
        "labels": input_ids.clone(),
    }


def make_ones_batch(batch_size: int, seq_len: int, device):
    input_ids = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids, device=device)}


# ── Scoring kernels ────────────────────────────────────────────────

def backward_for_proxy(model, batch) -> float:
    model.zero_grad(set_to_none=True)
    out = model(**batch)
    loss = out.loss
    if loss is None:
        loss = out.logits.float().sum()
    loss.backward()
    return float(loss.detach().float().item())


def aggregate_grads(model, method: str) -> float:
    score = 0.0
    for _li, _n, w in iter_ffn_weights(model):
        g = w.grad
        if g is None:
            continue
        gf = g.detach().float()
        wf = w.detach().float()
        if method in ("snip", "synflow"):
            score += torch.sum(torch.abs(gf * wf)).item()
        elif method == "gradnorm":
            score += torch.linalg.vector_norm(gf).item()
        else:
            raise ValueError(method)
    return float(score)


_SYNFLOW_PATCHED = False
_ORIG_LINEAR_FORWARD = nn.Linear.forward


def _abs_linear_forward(self, x):
    return F.linear(x, self.weight.abs(), self.bias.abs() if self.bias is not None else None)


def patch_linear_for_synflow():
    global _SYNFLOW_PATCHED
    if not _SYNFLOW_PATCHED:
        nn.Linear.forward = _abs_linear_forward
        _SYNFLOW_PATCHED = True


def unpatch_linear():
    global _SYNFLOW_PATCHED
    if _SYNFLOW_PATCHED:
        nn.Linear.forward = _ORIG_LINEAR_FORWARD
        _SYNFLOW_PATCHED = False


def score_synflow(model, ones_batch):
    set_ffn_grad_mode(model, True)
    patch_linear_for_synflow()
    try:
        model.zero_grad(set_to_none=True)
        out = model(**ones_batch)
        loss = out.logits.float().abs().sum()
        loss.backward()
        score = aggregate_grads(model, "synflow")
        meta = {"loss": float(loss.detach().float().item())}
    finally:
        unpatch_linear()
        model.zero_grad(set_to_none=True)
    return score, meta


def score_snip_gradnorm(model, batch, method: str):
    set_ffn_grad_mode(model, True)
    loss = backward_for_proxy(model, batch)
    score = aggregate_grads(model, method)
    model.zero_grad(set_to_none=True)
    return score, {"loss": loss}


@torch.no_grad()
def score_wpca(model, proxy_batch, threshold: float, params_B: float):
    """W-PCA (Wang et al., ICLR 2025).

    Captures gate_proj output H_i [B, N, D'_i], centers per-feature, computes
    SVD, picks smallest k_i with cumulative explained-variance ratio >= threshold.
    Score = params_B * sum_i k_i.
    """
    captured = [None] * N_LAYERS
    handles = []

    def make_hook(li):
        def hook(_module, _inp, out):
            captured[li] = out.detach()
        return hook

    for li in range(N_LAYERS):
        layer = get_decoder_layer(model, li)
        gate = layer.mlp.gate_proj
        handles.append(gate.register_forward_hook(make_hook(li)))
    try:
        model(**proxy_batch)
    finally:
        for h in handles:
            h.remove()

    total_components = 0
    details = []
    for li in range(N_LAYERS):
        H = captured[li]
        captured[li] = None
        if H is None:
            details.append(0)
            continue
        H2 = H.reshape(-1, H.shape[-1]).float()
        H2 = H2 - H2.mean(dim=0, keepdim=True)
        # PCA via Gram trick: eigvals of X^T X (or X X^T, smaller) equal sigma_i^2.
        # Avoids torch.linalg.svdvals CPU fallback path on tall/wide matrices.
        try:
            n, d = H2.shape
            if n <= d:
                M = H2 @ H2.T
            else:
                M = H2.T @ H2
            eigvals = torch.linalg.eigvalsh(M)
            eigvals = eigvals.clamp(min=0)
            var, _ = torch.sort(eigvals, descending=True)
            ratio = torch.cumsum(var, 0) / (var.sum() + 1e-12)
            k = int((ratio < threshold).sum().item() + 1)
            k = min(k, H2.shape[1])
        except Exception:
            k = H2.shape[1]
        total_components += k
        details.append(k)

    score = float(params_B) * float(total_components)
    return score, {
        "pca_components": details,
        "sum_components": total_components,
        "params_B": float(params_B),
        "threshold": threshold,
    }


def score_candidate(model, state, subnet, method: str, real_batch, ones_batch,
                    wpca_threshold: float):
    apply_subnet(model, subnet, state)
    try:
        if method == "synflow":
            return score_synflow(model, ones_batch)
        if method == "wpca":
            params_B = count_params_from_subnet(subnet)
            return score_wpca(model, real_batch, wpca_threshold, params_B)
        return score_snip_gradnorm(model, real_batch, method)
    finally:
        restore_subnet(model, state)
        torch.cuda.empty_cache()


# ── GA helpers ─────────────────────────────────────────────────────

def random_valid_subnet(budget_b: float):
    while True:
        ranks = [random.choice(RANK_CANDIDATES) for _ in range(N_LAYERS)]
        ffns = [random.choice(FFN_CANDIDATES) for _ in range(N_LAYERS)]
        sub = make_subnet(ranks, ffns)
        if count_params_from_subnet(sub) <= budget_b:
            return sub


def mutate_subnet(parent, prob: float):
    ranks = [
        random.choice(RANK_CANDIDATES) if random.random() < prob else r
        for r in parent["lora_ranks"]
    ]
    ffns = [
        random.choice(FFN_CANDIDATES) if random.random() < prob else k
        for k in parent["ffn_dims"]
    ]
    return make_subnet(ranks, ffns)


def crossover_subnet(a, b):
    ranks = [a["lora_ranks"][i] if random.random() < 0.5 else b["lora_ranks"][i] for i in range(N_LAYERS)]
    ffns = [a["ffn_dims"][i] if random.random() < 0.5 else b["ffn_dims"][i] for i in range(N_LAYERS)]
    return make_subnet(ranks, ffns)


def generate_child(population, budget_b: float, top_k: int, mutation_prob: float):
    if len(population) < 2:
        return random_valid_subnet(budget_b), 1
    parents = population[: max(2, min(top_k, len(population)))]
    attempts = 0
    while True:
        attempts += 1
        p1, p2 = random.sample(parents, 2)
        child = crossover_subnet(p1["subnet"], p2["subnet"])
        child = mutate_subnet(child, mutation_prob)
        if count_params_from_subnet(child) <= budget_b:
            return child, attempts


# ── Main GA loop ───────────────────────────────────────────────────

def run_ga(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    total_t0 = time.time()
    setup_t0 = time.time()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[setup] loading PEFT supernet (un-merged): {args.adapter}", flush=True)
    model, tokenizer = load_peft_supernet(args.base_model, args.adapter, dtype)
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "sdpa"
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "sdpa"
    model = model.to(device)
    if args.gradient_checkpointing and args.method != "wpca":
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
        print("[setup] gradient_checkpointing ENABLED", flush=True)
    else:
        model.eval()

    state = snapshot_state(model)
    torch.cuda.empty_cache()
    print(f"[setup] state snapshot done, GPU mem={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    real_batch = None
    ones_batch = None
    if args.method in ("snip", "gradnorm", "wpca"):
        real_batch = make_real_proxy_batch(tokenizer, args.proxy_batch_size, args.proxy_seq_len, device)
    if args.method == "synflow":
        ones_batch = make_ones_batch(args.proxy_batch_size, args.proxy_seq_len, device)

    del tokenizer
    cuda_sync()
    setup_time_s = time.time() - setup_t0
    print(f"[setup] dtype={args.dtype} setup_time={setup_time_s:.1f}s method={args.method}", flush=True)
    print(f"[setup] proxy: b={args.proxy_batch_size} s={args.proxy_seq_len}", flush=True)
    print(f"[setup] search space: rank ∈ {RANK_CANDIDATES} × ffn ∈ {FFN_CANDIDATES}", flush=True)

    population = []
    all_evals = []
    n_proxy_evals = 0
    n_attempts = 0

    cuda_sync()
    search_t0 = time.time()
    for gen in range(args.generations):
        gen_rows = []
        while len(gen_rows) < args.population:
            if gen == 0:
                subnet = random_valid_subnet(args.param_budget)
                n_attempts += 1
            else:
                subnet, attempts = generate_child(population, args.param_budget, args.parent_top_k, args.mutation_prob)
                n_attempts += attempts
            cuda_sync()
            score_t0 = time.time()
            score, meta = score_candidate(model, state, subnet, args.method, real_batch, ones_batch,
                                          args.wpca_threshold)
            cuda_sync()
            n_proxy_evals += 1
            row = {
                "eval_index": n_proxy_evals,
                "generation": gen,
                "subnet": subnet,
                "params_B": round(count_params_from_subnet(subnet), 6),
                "score": score,
                "score_meta": meta,
                "score_time_s": round(time.time() - score_t0, 4),
            }
            gen_rows.append(row)
            all_evals.append(row)
        candidates = population + gen_rows
        candidates.sort(key=lambda x: x["score"], reverse=True)
        population = candidates[: args.population]
        elapsed = time.time() - search_t0
        rate = n_proxy_evals / max(elapsed, 1e-6)
        best = population[0]
        n_r28 = sum(1 for r in best["subnet"]["lora_ranks"] if r == 28)
        print(
            f"[{args.method}] gen {gen+1}/{args.generations}: "
            f"best={best['score']:.5g} params={best['params_B']:.4f}B "
            f"#r=28: {n_r28}/32 evals={n_proxy_evals} elapsed={elapsed:.0f}s rate={rate:.2f}/s "
            f"peak_mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB",
            flush=True,
        )
        write_payload(args, setup_time_s, time.time() - search_t0, time.time() - total_t0,
                      n_proxy_evals, n_attempts, population, all_evals, "running")

    cuda_sync()
    search_time_s = time.time() - search_t0
    return write_payload(args, setup_time_s, search_time_s, time.time() - total_t0,
                         n_proxy_evals, n_attempts, population, all_evals, "complete")


def write_payload(args, setup_time_s, search_time_s, total_wall_time_s,
                  n_proxy_evals, n_attempts, population, all_evals, status):
    best = population[0] if population else None
    proxy_input = {"batch_size": args.proxy_batch_size, "seq_len": args.proxy_seq_len}
    if args.method == "synflow":
        proxy_input["data"] = "ones (data-free)"
    else:
        proxy_input["data"] = "wikitext-2-raw-v1 train"
    formula = {
        "snip": "sum |grad * weight| over FFN, real calibration",
        "gradnorm": "sum ||grad||_2 over FFN, real calibration",
        "synflow": "abs-weight forward on ones, sum |grad * weight| over FFN",
        "wpca": "params_B * sum_i k_i, k_i = #PCA components for cumvar>=threshold of gate_proj act",
    }[args.method]
    payload = {
        "status": status,
        "method": f"{args.method.upper()}-GA",
        "proxy_method": args.method,
        "proxy_formula": formula,
        "search_space": {
            "type": "LoNAS LLaMA full 2x5",
            "rank_candidates": RANK_CANDIDATES,
            "ffn_candidates": FFN_CANDIDATES,
            "n_layers": N_LAYERS,
        },
        "ga_config": {
            "population": args.population,
            "generations": args.generations,
            "target_proxy_evals": args.population * args.generations,
            "parent_top_k": args.parent_top_k,
            "mutation_prob": args.mutation_prob,
            "seed": args.seed,
        },
        "proxy_input": proxy_input,
        "wpca_threshold": args.wpca_threshold if args.method == "wpca" else None,
        "param_budget_B": args.param_budget,
        "setup_time_s": round(setup_time_s, 4),
        "search_time_s": round(search_time_s, 4),
        "total_wall_time_s": round(total_wall_time_s, 4),
        "n_proxy_evals": n_proxy_evals,
        "n_attempts": n_attempts,
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 4) if torch.cuda.is_available() else None,
        "rank1": None,
        "top10": [],
        "all_evals": all_evals,
    }
    if best:
        payload["rank1"] = {
            "rank": 1,
            "subnet": best["subnet"],
            "params_B": best["params_B"],
            "score": best["score"],
            "score_meta": best["score_meta"],
            "n_rank28": sum(1 for r in best["subnet"]["lora_ranks"] if r == 28),
            "n_rank32": sum(1 for r in best["subnet"]["lora_ranks"] if r == 32),
        }
        for idx, row in enumerate(population[:10], 1):
            payload["top10"].append({
                "rank": idx,
                "subnet": row["subnet"],
                "params_B": row["params_B"],
                "score": row["score"],
                "n_rank28": sum(1 for r in row["subnet"]["lora_ranks"] if r == 28),
                "n_rank32": sum(1 for r in row["subnet"]["lora_ranks"] if r == 32),
            })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out)
    return payload


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["snip", "gradnorm", "synflow", "wpca"])
    p.add_argument("--base_model", default=os.environ.get("LLAMA_PATH", "yahma/llama-7b-hf"))
    p.add_argument("--adapter", default=os.environ.get("LONAS_ADAPTER", "lonas-llama-7b-adapter"))
    p.add_argument("--output", required=True)
    p.add_argument("--population", type=int, default=50)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--parent_top_k", type=int, default=10)
    p.add_argument("--mutation_prob", type=float, default=0.2)
    p.add_argument("--param_budget", type=float, default=PARAM_BUDGET_B)
    p.add_argument("--proxy_batch_size", type=int, default=2)
    p.add_argument("--proxy_seq_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--wpca_threshold", type=float, default=0.99)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = run_ga(args)
    print(json.dumps({
        "status": payload["status"],
        "method": payload["method"],
        "setup_time_s": payload["setup_time_s"],
        "search_time_s": payload["search_time_s"],
        "total_wall_time_s": payload["total_wall_time_s"],
        "n_proxy_evals": payload["n_proxy_evals"],
        "peak_gpu_memory_gb": payload["peak_gpu_memory_gb"],
        "rank1": payload["rank1"],
        "output": args.output,
    }, indent=2), flush=True)
