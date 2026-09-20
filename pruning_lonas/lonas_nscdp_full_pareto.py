#!/usr/bin/env python3
"""Run NSCDP across the full param range and emit a per-ffn_sum Pareto front."""

import json
import math
import os
import sys
import time
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
OUT_PATH = OUT / "nscdp_full_pareto.json"
TOP_K = 1


def push_top(bucket, item, k):
    bucket.append(item)
    bucket.sort(key=lambda x: x["score"], reverse=True)
    if len(bucket) > k:
        del bucket[k:]


def run_full_dp(cache, top_k=TOP_K):
    """DP without budget cap: keep top_k partial solutions per ffn_sum bin."""
    start = time.time()
    states = {0: [{"score": 0.0, "ranks": [], "ffns": []}]}
    for layer_idx in range(N_LAYERS):
        layer_options = [
            {
                "rank": rank,
                "ffn": ffn_dim,
                "score": option_score(cache, layer_idx, rank, ffn_dim),
            }
            for rank in RANK_CANDIDATES
            for ffn_dim in FFN_CANDIDATES
        ]
        next_states = {}
        for ffn_sum, partials in states.items():
            for partial in partials:
                for opt in layer_options:
                    new_sum = ffn_sum + opt["ffn"]
                    item = {
                        "score": partial["score"] + opt["score"],
                        "ranks": partial["ranks"] + [opt["rank"]],
                        "ffns": partial["ffns"] + [opt["ffn"]],
                    }
                    push_top(next_states.setdefault(new_sum, []), item, top_k)
        states = next_states
        print(
            f"Layer {layer_idx + 1}/{N_LAYERS}: {len(states)} ffn-sum bins",
            flush=True,
        )
    elapsed = time.time() - start
    return states, elapsed


def build_pareto(states):
    """For each ffn_sum, take the top-1 → params-vs-score Pareto."""
    pareto = []
    for ffn_sum in sorted(states.keys()):
        best = max(states[ffn_sum], key=lambda x: x["score"])
        pareto.append(
            {
                "ffn_sum": ffn_sum,
                "params_B": round(params_from_ffn_sum(ffn_sum), 6),
                "score": best["score"],
                "n_rank32": best["ranks"].count(32),
                "subnet": {
                    "lora_ranks": list(best["ranks"]),
                    "ffn_dims": list(best["ffns"]),
                },
            }
        )
    return pareto


def main():
    print(f"Loading proxy cache: {CACHE_PATH}", flush=True)
    cache = load_cache(str(CACHE_PATH))
    print(f"  {len(cache)} entries", flush=True)

    states, elapsed = run_full_dp(cache, top_k=TOP_K)
    pareto = build_pareto(states)

    payload = {
        "method": "NSCDP full-range exact DP (per-ffn_sum top-1)",
        "score": "NSC-H psi_MP closed-form (paper eq. 3-4), flat sum",
        "cache": str(CACHE_PATH),
        "search_time_seconds": round(elapsed, 6),
        "search_space": {
            "rank_candidates": RANK_CANDIDATES,
            "ffn_candidates": FFN_CANDIDATES,
            "n_layers": N_LAYERS,
        },
        "n_pareto_points": len(pareto),
        "pareto": pareto,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {OUT_PATH}", flush=True)
    print(
        f"Pareto: {len(pareto)} points, params [{pareto[0]['params_B']:.3f}, "
        f"{pareto[-1]['params_B']:.3f}] B, time {elapsed:.3f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
