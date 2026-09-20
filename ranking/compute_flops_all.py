#!/usr/bin/env python3
"""Compute FLOPs for all six benchmarks used in the paper.

Outputs six JSONs in $NSC_OUT/flops/:
  - flops_flexibert.json
  - flops_gpt2.json
  - flops_autoformer_tiny.json
  - flops_autoformer_small.json
  - flops_nats_sss_c100.json
  - flops_mnv3_r224.json

Each file is a list of {arch_id/idx/arch_key, flops_mac, flops_2x, n_params}.

Conventions:
  - flops_mac    : MACs (one MAC = one mul-add); follows OFA/MobileNet papers.
  - flops_2x     : 2 * MACs (FLOPs in the lit. that count add+mul separately).
  - Reference inputs (fixed across each benchmark):
      FlexiBERT      : seq_len = 128, batch = 1
      GPT-2 (TXL bench): seq_len = 192, batch = 1 (matches eval batch)
      AutoFormer-T/S : ImageNet 224x224, read directly from bench JSON
      NATS-SSS C100  : CIFAR-100 32x32, batch = 1
      MNV3 r=224     : ImageNet 224x224, batch = 1
"""
from __future__ import annotations
import json, os, sys, math, time
from pathlib import Path
import importlib.util
import warnings

import numpy as np
import torch
import torch.nn as nn
from thop import profile

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("NSC_ROOT", "."))
PAPER = ROOT / "paper"
OUT = Path(os.environ.get("NSC_OUT", "outputs")) / "flops"
OUT.mkdir(parents=True, exist_ok=True)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def thop_flops(model: nn.Module, x: torch.Tensor) -> tuple[float, float]:
    """Return (macs, n_params) using thop."""
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(x,), verbose=False)
    return float(macs), float(params)


# ──────────────────────────────────────────────────────────────────────
# 1. FlexiBERT  (analytical formula)
# ──────────────────────────────────────────────────────────────────────

def flexibert_layer_macs(lc: dict, H: int, S: int) -> float:
    """MACs for one FlexiBERT encoder layer at sequence length S, hidden H."""
    op = lc["operation_type"]
    nff = int(lc.get("num_feed_forward", 1))
    DFF = int(lc.get("feed_forward_dimension", 1024))
    macs = 0.0

    if op == "SA":
        # QKV projection (merged 3H, H) + attention scores + softmax*V + output proj.
        macs += S * H * (3 * H)            # QKV proj
        macs += S * S * H                  # Q @ K^T  (per head sums to S*S*H)
        macs += S * S * H                  # softmax @ V
        macs += S * H * H                  # output projection
    elif op == "LT":
        # Linear transform (DFT/DCT family); FlexiBERT models it as 2 * (H, H) ops.
        macs += 2 * S * H * H
    elif op == "DSC":
        # Depth-separable conv attention proxy: one (H, H) op as in nsc_utils.
        macs += S * H * H
    else:
        # Conv branches (rarely used) -> approximate as one HxH op
        macs += S * H * H

    # FFN: nff parallel branches, each H -> ff_dim/nff -> H
    ff_dim = DFF // nff
    for _ in range(nff):
        macs += S * H * ff_dim     # up
        macs += S * ff_dim * H     # down
    return macs


def compute_flexibert():
    bench = json.load(open(ROOT / "NAS/FlexiBERT/BERT_benchmark.json"))
    scores = json.load(open(Path(os.environ.get("NSC_OUT", "outputs")) / "scores/flexibert_corrected_realdata_scores.json"))
    bench_by_id = {a["id"]: a for a in bench}
    S = 128
    out = []
    for rec in scores:
        aid = rec["arch_id"]
        hpo = bench_by_id[aid]["hparams"]["model_hparam_overrides"]
        H = int(hpo["hidden_size"])
        layers = hpo["nas_config"]["encoder_layers"]
        macs = sum(flexibert_layer_macs(lc, H, S) for lc in layers)
        # Classifier head + pooler (small but include for completeness)
        macs += S * H * H            # pooler (linear H->H on [CLS])
        macs += H * 2                # binary classifier head (negligible)
        out.append({
            "arch_id": aid,
            "n_params": rec.get("n_params"),
            "n_layers": rec.get("n_layers", len(layers)),
            "seq_len": S,
            "hidden_size": H,
            "flops_mac": macs,
            "flops_2x": 2 * macs,
        })
    fp = OUT / "flops_flexibert.json"
    json.dump(out, open(fp, "w"), indent=2)
    print(f"[FlexiBERT] {len(out)} archs  →  {fp}")
    print(f"  MACs range: {min(r['flops_mac'] for r in out):.3e} – {max(r['flops_mac'] for r in out):.3e}")


# ──────────────────────────────────────────────────────────────────────
# 2. GPT-2 (Transformer-XL bench, GPT2Flex via eval_gpt2_baselines.build_model)
# ──────────────────────────────────────────────────────────────────────

def compute_gpt2():
    mod = _load_module("gpt2_baselines", Path(__file__).parent / "eval_gpt2_baselines.py")
    bench = json.load(open(PAPER / "dataset/gpt2_txl/gpt2_benchmark.json"))
    scores = json.load(open(Path(os.environ.get("NSC_OUT", "outputs")) / "scores/gpt2_realdata_scores.json"))
    keys = [r["arch_key"] for r in scores]
    S = 192
    out = []
    for i, key in enumerate(keys):
        cfg = bench[key]["config"]
        torch.manual_seed(0)
        model = mod.build_model(cfg)
        x = torch.randint(0, cfg["n_token"], (1, S))
        try:
            macs, params = thop_flops(model, x)
        except Exception as e:
            print(f"  [warn] {key}: {e}")
            macs, params = float("nan"), float("nan")
        out.append({
            "arch_key": key,
            "n_params": float(bench[key]["params"]["total"]),
            "n_params_thop": params,
            "n_layer": cfg["n_layer"],
            "d_model": cfg["d_model"],
            "seq_len": S,
            "flops_mac": macs,
            "flops_2x": 2 * macs,
        })
        del model
        if (i + 1) % 25 == 0:
            print(f"  [GPT-2] {i+1}/{len(keys)}  macs={macs:.3e}")
    fp = OUT / "flops_gpt2.json"
    json.dump(out, open(fp, "w"), indent=2)
    print(f"[GPT-2] {len(out)} archs  →  {fp}")


# ──────────────────────────────────────────────────────────────────────
# 3 + 4. AutoFormer Tiny / Small  (FLOPs already in bench JSON, in MFLOPs)
# ──────────────────────────────────────────────────────────────────────

def compute_autoformer(scale: str):
    bench = json.load(open(PAPER / f"dataset/autoformer/data/autoformer_{scale}_1k.json"))
    scores = json.load(open(Path(os.environ.get("NSC_OUT", "outputs")) / f"scores/autoformer_{scale}_realdata_scores.json"))
    out = []
    for rec in scores:
        idx = rec["idx"]
        e = bench[str(idx)]
        macs_mflops = float(e["flops"])  # the bench reports FLOPs in MFLOPs (≈ MACs)
        out.append({
            "idx": idx,
            "n_params_M": float(e["params"]),
            "n_params": int(round(float(e["params"]) * 1e6)),
            "depth": e["net_setting"]["layer_num"],
            "embed_dim": e["net_setting"]["embed_dim"][0] if isinstance(e["net_setting"]["embed_dim"], list) else e["net_setting"]["embed_dim"],
            "img_size": 224,
            "flops_mac": macs_mflops * 1e6,
            "flops_2x": 2 * macs_mflops * 1e6,
        })
    fp = OUT / f"flops_autoformer_{scale}.json"
    json.dump(out, open(fp, "w"), indent=2)
    print(f"[AutoFormer-{scale}] {len(out)} archs  →  {fp}")


# ──────────────────────────────────────────────────────────────────────
# 5. NATS-SSS C100  (use NATSBenchSSS class from run_realdata_proxy.py)
# ──────────────────────────────────────────────────────────────────────

def compute_nats():
    rdp = _load_module("rdp", Path(__file__).parent / "run_realdata_proxy.py")
    NATSBenchSSS = rdp.NATSBenchSSS
    scores = json.load(open(Path(os.environ.get("NSC_OUT", "outputs")) / "scores/nats_sss_c100_realdata_scores.json"))
    out = []
    for i, rec in enumerate(scores):
        idx = rec["idx"]
        ch = rec["channels"]
        torch.manual_seed(0)
        model = NATSBenchSSS(ch, nc=100)
        x = torch.randn(1, 3, 32, 32)
        macs, params = thop_flops(model, x)
        out.append({
            "idx": idx,
            "channels": ch,
            "n_params": int(params),
            "img_size": 32,
            "flops_mac": macs,
            "flops_2x": 2 * macs,
        })
        del model
        if (i + 1) % 200 == 0:
            print(f"  [NATS] {i+1}/{len(scores)}  macs={macs:.3e}")
    fp = OUT / "flops_nats_sss_c100.json"
    json.dump(out, open(fp, "w"), indent=2)
    print(f"[NATS-SSS C100] {len(out)} archs  →  {fp}")


# ──────────────────────────────────────────────────────────────────────
# 6. MNV3 r=224  (use MNV3ProxyNet class from run_realdata_proxy.py)
# ──────────────────────────────────────────────────────────────────────

def compute_mnv3():
    rdp = _load_module("rdp2", Path(__file__).parent / "run_realdata_proxy.py")
    MNV3ProxyNet = rdp.MNV3ProxyNet
    scores = json.load(open(Path(os.environ.get("NSC_OUT", "outputs")) / "scores/mnv3_r224_realdata_scores.json"))
    out = []
    for i, rec in enumerate(scores):
        idx = rec["local_idx"]
        ph = rec["phenotype"]
        torch.manual_seed(0)
        model = MNV3ProxyNet(ph, nc=1000)
        x = torch.randn(1, 3, 224, 224)
        try:
            macs, params = thop_flops(model, x)
        except Exception as e:
            print(f"  [warn] mnv3 idx={idx}: {e}")
            macs, params = float("nan"), float("nan")
        out.append({
            "local_idx": idx,
            "phenotype": ph,
            "n_params": int(params) if not math.isnan(params) else None,
            "img_size": 224,
            "flops_mac": macs,
            "flops_2x": 2 * macs,
        })
        del model
        if (i + 1) % 50 == 0:
            print(f"  [MNV3] {i+1}/{len(scores)}  macs={macs:.3e}")
    fp = OUT / "flops_mnv3_r224.json"
    json.dump(out, open(fp, "w"), indent=2)
    print(f"[MNV3 r=224] {len(out)} archs  →  {fp}")


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    only = set(sys.argv[1:])
    def run(name, fn):
        if only and name not in only:
            return
        t0 = time.time()
        fn()
        print(f"  ({name} done in {time.time()-t0:.1f}s)\n")

    run("flexibert",        compute_flexibert)
    run("autoformer_tiny",  lambda: compute_autoformer("tiny"))
    run("autoformer_small", lambda: compute_autoformer("small"))
    run("nats",             compute_nats)
    run("gpt2",             compute_gpt2)
    run("mnv3",             compute_mnv3)
    print("All done. Outputs in", OUT)
