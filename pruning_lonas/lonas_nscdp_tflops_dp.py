#!/usr/bin/env python3
"""NSC-DP with inference TFLOPs as the resource.

lonas_nscdp_full_pareto.py runs the dynamic-programming search with parameter
count as the layer-additive cost; this script uses inference TFLOPs instead
(LoNAS convention, seq_len=256). The DP state is total quantized TFLOPs across
processed layers; layer-level options carry per-layer TFLOPs as their cost.

Steps:
  1. Load the psi_MP cache.
  2. Run the TFLOPs-state DP across the full achievable TFLOPs range.
  3. Save the per-TFLOPs-bin top-1 Pareto front.
  4. Compare the selected architectures with the parameter-based NSC-DP Pareto front.
"""
from __future__ import annotations

import os
import ast
import json
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

OUT = Path(os.environ.get("NSC_OUT", "outputs"))

from lonas_nscdp_original_space import (
    FFN_CANDIDATES,
    HIDDEN,
    N_LAYERS,
    RANK_CANDIDATES,
    base_param_count,
    load_cache,
    option_score,
    params_from_ffn_sum,
)

CACHE_PATH = OUT / "nsch_mp_cache.json"
PARAMS_PARETO_PATH = OUT / "nscdp_full_pareto.json"
OUT_DIR = Path(__file__).parent
OUT_PATH = OUT_DIR / "nscdp_tflops_pareto.json"
COMPARE_PATH = OUT_DIR / "tflops_vs_params_equivalence.json"

VOCAB = 32000
SEQ_LEN = 256
ONE_TFLOP = Fraction(10) ** 12


def per_layer_constant_tflops_frac() -> Fraction:
    flops = 4 * 2 * HIDDEN * HIDDEN
    flops += 2 * 2 * SEQ_LEN * HIDDEN
    return Fraction(flops * SEQ_LEN, 2) / ONE_TFLOP


def per_layer_ffn_tflops_frac(k: int) -> Fraction:
    return Fraction(3 * 2 * HIDDEN * k * SEQ_LEN, 2) / ONE_TFLOP


def total_const_tflops_frac() -> Fraction:
    head = Fraction(2 * HIDDEN * VOCAB * SEQ_LEN, 2) / ONE_TFLOP
    return N_LAYERS * per_layer_constant_tflops_frac() + head


def push_top(bucket: list, item: dict, top_k: int = 1) -> None:
    bucket.append(item)
    bucket.sort(key=lambda x: x["score"], reverse=True)
    if len(bucket) > top_k:
        del bucket[top_k:]


def run_tflops_dp(cache: dict) -> tuple[dict[Fraction, list], float]:
    """DP with state = exact rational cumulative FFN TFLOPs (Fraction).

    Using Fraction avoids the floating-point spurious bin-splitting that occurs
    when two ffn_dim sequences with the same ffn_sum compound differently after
    rounding. With exact rationals, two configs with the same ffn_sum map to
    the same DP state by construction.
    """
    start = time.time()

    ffn_cost = {k: per_layer_ffn_tflops_frac(k) for k in FFN_CANDIDATES}

    states: dict[Fraction, list] = {Fraction(0): [{"score": 0.0, "ranks": [], "ffns": []}]}
    for layer_idx in range(N_LAYERS):
        layer_options = [
            {
                "rank": r,
                "ffn": k,
                "cost": ffn_cost[k],
                "score": option_score(cache, layer_idx, r, k),
            }
            for r in RANK_CANDIDATES
            for k in FFN_CANDIDATES
        ]
        nxt: dict[Fraction, list] = {}
        for cum_tf, partials in states.items():
            for partial in partials:
                for opt in layer_options:
                    new_tf = cum_tf + opt["cost"]
                    item = {
                        "score": partial["score"] + opt["score"],
                        "ranks": partial["ranks"] + [opt["rank"]],
                        "ffns": partial["ffns"] + [opt["ffn"]],
                    }
                    push_top(nxt.setdefault(new_tf, []), item, top_k=1)
        states = nxt
        print(f"Layer {layer_idx + 1}/{N_LAYERS}: {len(states)} TFLOPs bins (exact)", flush=True)
    return states, time.time() - start


def build_pareto(states: dict[Fraction, list]) -> list[dict]:
    pareto = []
    const_tf = total_const_tflops_frac()
    for cum_tf in sorted(states.keys()):
        best = max(states[cum_tf], key=lambda x: x["score"])
        total_tf = const_tf + cum_tf
        ffn_sum = sum(best["ffns"])
        pareto.append({
            "tflops": round(float(total_tf), 6),
            "params_B": round(params_from_ffn_sum(ffn_sum), 6),
            "ffn_sum": ffn_sum,
            "score": best["score"],
            "n_rank32": best["ranks"].count(32),
            "subnet": {
                "lora_ranks": list(best["ranks"]),
                "ffn_dims": list(best["ffns"]),
            },
        })
    return pareto


def compare_with_params_pareto(tflops_pareto: list[dict]) -> dict:
    with open(PARAMS_PARETO_PATH) as f:
        params_pareto = json.load(f)["pareto"]

    by_ffn_sum_tflops = {p["ffn_sum"]: p for p in tflops_pareto}
    by_ffn_sum_params = {p["ffn_sum"]: p for p in params_pareto}

    common = sorted(set(by_ffn_sum_tflops) & set(by_ffn_sum_params))
    n = len(common)

    score_match = 0
    rank_match = 0
    ffn_match = 0
    full_match = 0
    diffs = []
    for fs in common:
        a = by_ffn_sum_tflops[fs]
        b = by_ffn_sum_params[fs]
        score_eq = abs(a["score"] - b["score"]) < 1e-6
        rank_eq = tuple(a["subnet"]["lora_ranks"]) == tuple(b["subnet"]["lora_ranks"])
        ffn_eq = tuple(a["subnet"]["ffn_dims"]) == tuple(b["subnet"]["ffn_dims"])
        score_match += int(score_eq)
        rank_match += int(rank_eq)
        ffn_match += int(ffn_eq)
        full_match += int(score_eq and rank_eq and ffn_eq)
        if not (score_eq and rank_eq and ffn_eq):
            diffs.append({
                "ffn_sum": fs,
                "score_eq": score_eq,
                "rank_eq": rank_eq,
                "ffn_eq": ffn_eq,
                "score_tflops": a["score"],
                "score_params": b["score"],
            })

    return {
        "n_common_pareto_points": n,
        "n_only_in_tflops_pareto": len(set(by_ffn_sum_tflops) - set(by_ffn_sum_params)),
        "n_only_in_params_pareto": len(set(by_ffn_sum_params) - set(by_ffn_sum_tflops)),
        "exact_score_match": score_match,
        "exact_rank_match": rank_match,
        "exact_ffn_match": ffn_match,
        "exact_full_match": full_match,
        "exact_full_match_pct": round(full_match / n * 100, 4) if n else 0.0,
        "diff_records": diffs,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading psi_MP cache from {CACHE_PATH}...")
    cache = load_cache(str(CACHE_PATH))
    print(f"  {len(cache)} entries")

    print("\n=== TFLOPs-direct NSC-DP ===")
    states, elapsed = run_tflops_dp(cache)
    pareto = build_pareto(states)
    print(f"\nDone: {len(pareto)} Pareto points in {elapsed*1000:.1f} ms")
    print(
        f"  TFLOPs range: [{pareto[0]['tflops']:.4f}, {pareto[-1]['tflops']:.4f}]\n"
        f"  params range: [{pareto[0]['params_B']:.4f}, {pareto[-1]['params_B']:.4f}] B"
    )

    out = {
        "method": "NSC-DP with TFLOPs as the layer-additive resource",
        "cost_function": "TFLOPs per layer (seq_len=256), exact rational arithmetic",
        "score": "Sum of psi_MP closed-form (architectural NSC)",
        "n_pareto_points": len(pareto),
        "search_time_seconds": round(elapsed, 6),
        "search_space": {
            "rank_candidates": RANK_CANDIDATES,
            "ffn_candidates": FFN_CANDIDATES,
            "n_layers": N_LAYERS,
        },
        "pareto": pareto,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {OUT_PATH}")

    print("\n=== Equivalence check vs params-based Pareto ===")
    cmp = compare_with_params_pareto(pareto)
    for k, v in cmp.items():
        if k == "diff_records":
            continue
        print(f"  {k}: {v}")
    if cmp["diff_records"]:
        print("  --- DIFFERENCES (first 5) ---")
        for d in cmp["diff_records"][:5]:
            print(f"    {d}")
    else:
        print("  ✓ All Pareto points match exactly across both DP variants.")
    with open(COMPARE_PATH, "w") as f:
        json.dump(cmp, f, indent=2)
    print(f"\nSaved {COMPARE_PATH}")


if __name__ == "__main__":
    main()
