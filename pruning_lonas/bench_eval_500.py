#!/usr/bin/env python3
"""Batched evaluation of LoNAS LLaMA-7B subnets on the eight commonsense tasks.

Implementation:
  1. Merge LoRA once and keep the full-size merged weights in CPU RAM
  2. Per subnet: slice FFN weights in place on GPU
  3. Batched log-likelihood scoring for all 8 tasks
  4. Pre-tokenize every (prompt, continuation) pair once
  5. Dynamic left-pad batching, sorted by sequence length
  6. Resume: skip already-evaluated subnets (JSONL append)
"""

import argparse
import gc
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ─── Constants ────────────────────────────────────────────────────
N_LAYERS = 32
HIDDEN = 4096
FFN_FULL = 11008
DATA_DIR = Path(__file__).parent / "llamabench" / "data"

INSTRUCT_PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)
INSTRUCT_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


# ─── Dataset preparation ──────────────────────────────────────────

def build_instruct_items(task_key):
    n_choices = {"ARC-Easy": 4, "ARC-Challenge": 4, "openbookqa": 4, "social_i_qa": 3}
    nc = n_choices[task_key]
    data_path = DATA_DIR / task_key / "test.json"
    with open(data_path) as f:
        dataset = json.load(f)

    items = []
    for row in dataset:
        inp = row.get("input", "")
        if inp:
            prompt = INSTRUCT_PROMPT_WITH_INPUT.format(
                instruction=row["instruction"], input=inp)
        else:
            prompt = INSTRUCT_PROMPT_NO_INPUT.format(
                instruction=row["instruction"])
        templates = [f"the correct answer is answer{i}" for i in range(1, nc + 1)]
        keys = [f"answer{i}" for i in range(1, nc + 1)]
        label = row["answer"].strip().lower()
        items.append((prompt, templates, keys, label))
    return items


def build_lmeval_items(task_name):
    import datasets as hf_ds
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    if task_name == "boolq":
        ds = hf_ds.load_dataset("aps/super_glue", "boolq", split="validation")
        return [(f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:",
                 [" no", " yes"], r["label"]) for r in ds]

    elif task_name == "piqa":
        ds = hf_ds.load_dataset("baber/piqa", split="validation")
        return [(f"Question: {r['goal']}\nAnswer:",
                 [" " + r["sol1"], " " + r["sol2"]], r["label"]) for r in ds]

    elif task_name == "hellaswag":
        ds = hf_ds.load_dataset("Rowan/hellaswag", split="validation")
        items = []
        for r in ds:
            ctx = (r["activity_label"] + ": " if r["activity_label"] else "") + r["ctx"]
            items.append((ctx, [" " + e for e in r["endings"]], int(r["label"])))
        return items

    elif task_name == "winogrande":
        ds = hf_ds.load_dataset("allenai/winogrande", "winogrande_xl",
                                split="validation")
        items = []
        for r in ds:
            idx = r["sentence"].index("_")
            ctx = r["sentence"][:idx]
            suffix = r["sentence"][idx + 1:]
            items.append((ctx,
                          [r["option1"] + suffix, r["option2"] + suffix],
                          int(r["answer"]) - 1))
        return items

    raise ValueError(f"Unknown task: {task_name}")


# ─── Tokenization (unified format) ───────────────────────────────
# Each sample → ([(full_ids, ctx_len, meta)], gold)
# We keep full_ids intact to avoid subword boundary misalignment.

def pretokenize_instruct(items, tokenizer):
    """Returns list of ([(full_ids, ctx_len, key)], label)."""
    results = []
    for prompt, templates, keys, label in items:
        ctx_len = tokenizer(prompt, return_tensors="pt",
                            add_special_tokens=True)["input_ids"].shape[1]
        choices = []
        for tmpl, key in zip(templates, keys):
            full_ids = tokenizer(prompt + tmpl, return_tensors="pt",
                                 add_special_tokens=True)["input_ids"][0]
            choices.append((full_ids, ctx_len, key))
        results.append((choices, label))
    return results


def pretokenize_lmeval(items, tokenizer):
    """Returns list of ([(full_ids, ctx_len, choice_idx)], label_idx)."""
    results = []
    for ctx, conts, label_idx in items:
        ctx_len = tokenizer(ctx, return_tensors="pt",
                            add_special_tokens=True)["input_ids"].shape[1]
        choices = []
        for ci, cont in enumerate(conts):
            full_ids = tokenizer(ctx + cont, return_tensors="pt",
                                 add_special_tokens=True)["input_ids"][0]
            choices.append((full_ids, ctx_len, ci))
        results.append((choices, label_idx))
    return results


# ─── Batched LL computation ───────────────────────────────────────

@torch.inference_mode()
def batched_ll(model, all_seqs, batch_size=24, device="cuda"):
    """
    Compute per-sequence log-likelihood of the 'answer' portion.
    all_seqs: list of (full_ids: 1D tensor, ctx_len: int)
    Returns: list of float.
    """
    results = [0.0] * len(all_seqs)
    sorted_indices = sorted(range(len(all_seqs)),
                            key=lambda i: all_seqs[i][0].shape[0])

    for batch_start in range(0, len(sorted_indices), batch_size):
        batch_idx = sorted_indices[batch_start:batch_start + batch_size]
        batch_items = [all_seqs[i] for i in batch_idx]
        max_len = max(ids.shape[0] for ids, _ in batch_items)

        input_ids_list = []
        attn_mask_list = []
        for ids, _ in batch_items:
            pad_len = max_len - ids.shape[0]
            input_ids_list.append(F.pad(ids, (pad_len, 0), value=0))
            attn_mask_list.append(F.pad(torch.ones_like(ids), (pad_len, 0), value=0))

        input_ids = torch.stack(input_ids_list).to(device)
        attn_mask = torch.stack(attn_mask_list).to(device)
        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits

        for j, orig_i in enumerate(batch_idx):
            full_ids_j, ctx_len = batch_items[j]
            pad_len = max_len - full_ids_j.shape[0]
            ans_start = pad_len + ctx_len
            ans_logits = logits[j, ans_start - 1: max_len - 1, :]
            ans_targets = input_ids[j, ans_start:]
            if ans_targets.numel() == 0:
                results[orig_i] = -1e9
                continue
            ll = F.log_softmax(ans_logits, dim=-1)
            results[orig_i] = ll.gather(1, ans_targets.unsqueeze(1)).sum().item()

    return results


# ─── Task evaluation (flat-batch approach) ────────────────────────

def _flatten_to_seqs(pretok_data):
    """Flatten pre-tokenized data to (full_ids, ctx_len) per choice."""
    all_seqs = []
    seq_map = []  # (sample_idx, choice_idx)
    for si, (choices, _gold) in enumerate(pretok_data):
        for ci, (full_ids, ctx_len, _meta) in enumerate(choices):
            all_seqs.append((full_ids, ctx_len))
            seq_map.append((si, ci))
    return all_seqs, seq_map


def eval_task(model, pretok_data, batch_size=24, device="cuda",
              length_normalize=False, is_instruct=False):
    """Unified evaluation for both instruct and lm_eval tasks."""
    all_seqs, seq_map = _flatten_to_seqs(pretok_data)
    all_ll = batched_ll(model, all_seqs, batch_size, device)

    sample_lls = defaultdict(dict)
    for seq_id, (si, ci) in enumerate(seq_map):
        sample_lls[si][ci] = all_ll[seq_id]

    correct = 0
    for si, (choices, gold) in enumerate(pretok_data):
        lls = dict(sample_lls[si])
        if length_normalize:
            for ci in lls:
                full_ids, ctx_len, _ = choices[ci]
                ans_len = full_ids.shape[0] - ctx_len
                if ans_len > 0:
                    lls[ci] /= ans_len
        best_ci = max(lls, key=lls.get)
        if is_instruct:
            pred = choices[best_ci][2]  # key string
            correct += (pred == gold)
        else:
            correct += (best_ci == gold)
    return correct / len(pretok_data)


# ─── Model management ────────────────────────────────────────────

def load_merged_model(base_path, adapter_path, dtype=torch.float16):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    print("Loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(base_path, use_fast=False,
                                        local_files_only=True)
    tok.padding_side = "left"
    tok.pad_token_id = 0

    print("Loading base model...", flush=True)
    m = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=dtype,
                                             device_map="cpu",
                                             local_files_only=True)
    m.config.pad_token_id = 0

    print("Loading LoRA adapter...", flush=True)
    m = PeftModel.from_pretrained(m, adapter_path, torch_dtype=dtype)

    print("Merging LoRA (rank=32)...", flush=True)
    m = m.merge_and_unload()
    print("Model merged.", flush=True)
    return m, tok


def save_full_weights(model):
    weights = {}
    for li in range(N_LAYERS):
        layer = model.model.layers[li]
        weights[li] = {
            "gate": layer.mlp.gate_proj.weight.data.clone(),
            "up":   layer.mlp.up_proj.weight.data.clone(),
            "down": layer.mlp.down_proj.weight.data.clone(),
        }
    return weights


def apply_subnet_ffn(model, ffn_dims, full_weights):
    for li in range(N_LAYERS):
        k = ffn_dims[li]
        layer = model.model.layers[li]
        fw = full_weights[li]
        layer.mlp.gate_proj.weight.data = fw["gate"][:k, :]
        layer.mlp.gate_proj.out_features = k
        layer.mlp.up_proj.weight.data = fw["up"][:k, :]
        layer.mlp.up_proj.out_features = k
        layer.mlp.down_proj.weight.data = fw["down"][:, :k]
        layer.mlp.down_proj.in_features = k


def restore_full_ffn(model, full_weights):
    for li in range(N_LAYERS):
        layer = model.model.layers[li]
        fw = full_weights[li]
        layer.mlp.gate_proj.weight.data = fw["gate"]
        layer.mlp.gate_proj.out_features = FFN_FULL
        layer.mlp.up_proj.weight.data = fw["up"]
        layer.mlp.up_proj.out_features = FFN_FULL
        layer.mlp.down_proj.weight.data = fw["down"]
        layer.mlp.down_proj.in_features = FFN_FULL


# ─── Main ─────────────────────────────────────────────────────────

LMEVAL_TASKS = ["boolq", "piqa", "winogrande"]
INSTRUCT_TASKS = ["ARC-Easy", "ARC-Challenge", "openbookqa", "social_i_qa"]
ALL_TASKS = LMEVAL_TASKS + INSTRUCT_TASKS
LMEVAL_NORM = set()


def main():
    parser = argparse.ArgumentParser(
        description="Batch-evaluate 500 subnets for NAS benchmark")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--bench", default="output/bench_500.json")
    parser.add_argument("--output", default="output/bench_500_results.jsonl")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1,
                        help="End index exclusive (-1 = all)")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16"])
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = "cuda"

    # ── Load bench ──
    with open(args.bench) as f:
        bench = json.load(f)
    samples = bench["samples"]
    end = args.end if args.end > 0 else len(samples)
    samples = samples[args.start:end]
    print(f"Subnet range [{args.start}, {end}) → {len(samples)} subnets", flush=True)

    # ── Resume ──
    done_ids = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
        print(f"Resuming: {len(done_ids)} done, "
              f"{sum(1 for s in samples if s['id'] not in done_ids)} remaining",
              flush=True)

    # ── Load model once ──
    model, tokenizer = load_merged_model(args.base_model, args.adapter, dtype)
    model = model.to(device)
    model.eval()

    print("Saving full FFN weights...", flush=True)
    full_weights = save_full_weights(model)

    # ── Pre-tokenize all datasets once ──
    print("\nPre-tokenizing datasets...", flush=True)
    t0 = time.time()

    instruct_data = {}
    for tk in INSTRUCT_TASKS:
        raw = build_instruct_items(tk)
        instruct_data[tk] = pretokenize_instruct(raw, tokenizer)
        print(f"  {tk}: {len(instruct_data[tk])} samples", flush=True)

    lmeval_data = {}
    for tk in LMEVAL_TASKS:
        raw = build_lmeval_items(tk)
        lmeval_data[tk] = pretokenize_lmeval(raw, tokenizer)
        print(f"  {tk}: {len(lmeval_data[tk])} samples", flush=True)

    print(f"Pre-tokenization: {time.time() - t0:.1f}s\n", flush=True)

    # ── Evaluate ──
    remaining = [s for s in samples if s["id"] not in done_ids]
    total = len(remaining)
    elapsed_times = []

    for idx, subnet in enumerate(remaining):
        sid = subnet["id"]
        t_sub = time.time()

        eta_str = ""
        if elapsed_times:
            avg_t = sum(elapsed_times) / len(elapsed_times)
            eta_s = avg_t * (total - idx)
            eta_h = eta_s / 3600
            eta_str = f"  ETA={eta_h:.1f}h"

        print(f"[{idx+1}/{total}] id={sid} params={subnet['params']:.3f}B "
              f"nsc_mp={subnet['nsc_mp']:.1f}{eta_str}", flush=True)

        apply_subnet_ffn(model, subnet["ffn_dims"], full_weights)

        scores = {}

        for tk in LMEVAL_TASKS:
            t0 = time.time()
            acc = eval_task(model, lmeval_data[tk], args.batch_size, device,
                            length_normalize=(tk in LMEVAL_NORM), is_instruct=False)
            scores[tk] = round(acc, 4)
            print(f"  {tk}: {acc:.4f} ({time.time()-t0:.1f}s)", flush=True)

        for tk in INSTRUCT_TASKS:
            t0 = time.time()
            acc = eval_task(model, instruct_data[tk], args.batch_size, device,
                            length_normalize=False, is_instruct=True)
            scores[tk] = round(acc, 4)
            print(f"  {tk}: {acc:.4f} ({time.time()-t0:.1f}s)", flush=True)

        avg7 = sum(scores[t] for t in ALL_TASKS) / 7
        scores["avg7"] = round(avg7, 4)
        elapsed = time.time() - t_sub
        elapsed_times.append(elapsed)
        print(f"  avg7={avg7:.4f}  ({elapsed:.0f}s)\n", flush=True)

        result = {
            "id": sid,
            "ffn_dims": subnet["ffn_dims"],
            "params": subnet["params"],
            "nsc_mp": subnet["nsc_mp"],
            "scores": scores,
            "eval_time_s": round(elapsed, 1),
        }
        with open(args.output, "a") as f:
            f.write(json.dumps(result) + "\n")

        restore_full_ffn(model, full_weights)

    if elapsed_times:
        total_h = sum(elapsed_times) / 3600
        avg_m = sum(elapsed_times) / len(elapsed_times) / 60
        print(f"Finished {len(elapsed_times)} subnets in {total_h:.1f}h "
              f"(avg {avg_m:.1f}min/subnet)", flush=True)
    print(f"Results → {args.output}", flush=True)


if __name__ == "__main__":
    main()
