#!/usr/bin/env python3
"""Exact NSC-DP search on the LoNAS LLaMA-7B search space.
"""

import ast
import json
import os
import time


N_LAYERS = 32
HIDDEN = 4096
VOCAB = 32000
RANK_CANDIDATES = [32, 28]
FFN_CANDIDATES = [11008, 9632, 8256, 6880, 5504]
PARAM_BUDGET_B = 5.7
TOP_K = 10
CACHE_PATH = os.path.join(os.environ.get("NSC_OUT", "outputs"), "nsch_mp_cache.json")
OUT_PATH = os.path.join(os.environ.get("NSC_OUT", "outputs"), "nscdp_original_space.json")


def load_cache(path):
    raw = json.load(open(path))
    cache = {}
    for key, value in raw.items():
        cache[ast.literal_eval(key)] = float(value)
    return cache


def base_param_count():
    return 2 * VOCAB * HIDDEN + N_LAYERS * (4 * HIDDEN * HIDDEN + 2 * HIDDEN)


def params_from_ffn_sum(ffn_sum):
    return (base_param_count() + 3 * HIDDEN * ffn_sum) / 1e9


def option_score(cache, layer_idx, rank, ffn_dim):
    total = 0.0
    for proj in ("q_proj", "k_proj", "v_proj"):
        total += cache[(layer_idx, proj, rank)]
    total += cache[(layer_idx, "o_proj")]
    for proj in ("gate_proj", "up_proj", "down_proj"):
        total += cache[(layer_idx, proj, ffn_dim)]
    return total


def push_top(bucket, item, top_k=TOP_K):
    bucket.append(item)
    bucket.sort(key=lambda x: x["score"], reverse=True)
    if len(bucket) > top_k:
        del bucket[top_k:]


def exact_search(cache):
    max_ffn_sum = int((PARAM_BUDGET_B * 1e9 - base_param_count()) // (3 * HIDDEN))
    start = time.time()

    # DP state: ffn_sum -> top-K partial solutions for processed layers.
    states = {0: [{"score": 0.0, "ranks": [], "ffns": []}]}
    for layer_idx in range(N_LAYERS):
        layer_options = []
        for rank in RANK_CANDIDATES:
            for ffn_dim in FFN_CANDIDATES:
                layer_options.append({
                    "rank": rank,
                    "ffn": ffn_dim,
                    "score": option_score(cache, layer_idx, rank, ffn_dim),
                })

        next_states = {}
        for ffn_sum, partials in states.items():
            for partial in partials:
                for opt in layer_options:
                    new_sum = ffn_sum + opt["ffn"]
                    if new_sum > max_ffn_sum:
                        continue
                    item = {
                        "score": partial["score"] + opt["score"],
                        "ranks": partial["ranks"] + [opt["rank"]],
                        "ffns": partial["ffns"] + [opt["ffn"]],
                    }
                    push_top(next_states.setdefault(new_sum, []), item)
        states = next_states
        print(f"Layer {layer_idx + 1}/{N_LAYERS}: {len(states)} ffn-sum states", flush=True)

    all_items = []
    for ffn_sum, items in states.items():
        for item in items:
            all_items.append((ffn_sum, item))
    all_items.sort(key=lambda x: x[1]["score"], reverse=True)

    top = []
    seen = set()
    for ffn_sum, item in all_items:
        key = (tuple(item["ranks"]), tuple(item["ffns"]))
        if key in seen:
            continue
        seen.add(key)
        top.append({
            "rank": len(top) + 1,
            "subnet": {
                "lora_ranks": item["ranks"],
                "ffn_dims": item["ffns"],
            },
            "score": item["score"],
            "params": round(params_from_ffn_sum(ffn_sum), 6),
            "ffn_sum": ffn_sum,
            "n_rank32": item["ranks"].count(32),
        })
        if len(top) >= TOP_K:
            break

    return {
        "search_time_seconds": round(time.time() - start, 6),
        "method": "NSC-DP exact dynamic programming",
        "score": "NSC-H no-rescale cache, flat sum of psi",
        "search_space": {
            "rank_candidates": RANK_CANDIDATES,
            "ffn_candidates": FFN_CANDIDATES,
            "n_layers": N_LAYERS,
        },
        "param_budget": PARAM_BUDGET_B,
        "max_ffn_sum": max_ffn_sum,
        "top": top,
    }


def main():
    cache = load_cache(CACHE_PATH)
    result = exact_search(cache)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {OUT_PATH}", flush=True)
    print(json.dumps(result["top"][:3], indent=2), flush=True)


if __name__ == "__main__":
    main()
