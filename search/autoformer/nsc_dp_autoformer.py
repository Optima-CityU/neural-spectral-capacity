#!/usr/bin/env python3
"""NSC-DP for the AutoFormer search space.

Macro variables: g = (embed_dim, depth)
Micro variables: per-layer (mlp_ratio, num_heads) type
Solver: GCD-compressed bounded knapsack DP (exact)

Search spaces:
  Tiny:  E∈{192},          D∈{12,13,14}, mlp∈{3.5,4.0}, heads∈{3,4}
  Small: E∈{320,384,448},  D∈{12,13,14}, mlp∈{3.0,3.5,4.0}, heads∈{5,6,7}
  Base:  E∈{528,576,624},  D∈{14,15,16}, mlp∈{3.0,3.5,4.0}, heads∈{8,9,10}

Usage:
  python nsc_dp_autoformer.py --space tiny
  python nsc_dp_autoformer.py --space small
  python nsc_dp_autoformer.py --space base
  python nsc_dp_autoformer.py --space all
  python nsc_dp_autoformer.py --space all --top_k 5
"""

import os
import math, time, argparse, json
from functools import reduce
import numpy as np
from scipy import integrate

# ─── Constants ────────────────────────────────────────────────────────────────

HEAD_DIM = 64
PATCH_SIZE = 16
IMG_SIZE = 224
N_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2   # 196
SEQ_LEN = N_PATCHES + 1                     # 197

# ─── AutoFormer Search Spaces ────────────────────────────────────────────────

SUPERNETS = {
    "tiny": {
        "embed_dims":    [192],
        "depth_choices": [12, 13, 14],
        "mlp_choices":   [3.5, 4.0],
        "head_choices":  [3, 4],
        "param_limit_M": 5.9,
    },
    "small": {
        "embed_dims":    [320, 384, 448],
        "depth_choices": [12, 13, 14],
        "mlp_choices":   [3.0, 3.5, 4.0],
        "head_choices":  [5, 6, 7],
        "param_limit_M": 23.5,
    },
    "base": {
        "embed_dims":    [528, 576, 624],
        "depth_choices": [14, 15, 16],
        "mlp_choices":   [3.0, 3.5, 4.0],
        "head_choices":  [8, 9, 10],
        "param_limit_M": 54.5,
    },
}

# ─── NSC-MP (Marchenko-Pastur, Xavier-init σ) ────────────────────────────────────

_PSI_CACHE = {}


def _mp_density(x, gamma):
    lp = (1 + np.sqrt(gamma)) ** 2
    lm = (1 - np.sqrt(gamma)) ** 2
    if x < lm or x > lp:
        return 0.0
    return np.sqrt((lp - x) * (x - lm)) / (2 * np.pi * gamma * x)


def psi_mp(m, n):
    """ψ_MP(m, n) with Xavier-init σ = √(2/(m+n))."""
    if m == 0 or n == 0:
        return 0.0
    key = (m, n)
    if key in _PSI_CACHE:
        return _PSI_CACHE[key]
    if m < n:
        m, n = n, m
    gamma = n / m
    sigma = np.sqrt(2.0 / (m + n))
    s2m = sigma ** 2 * m
    lp = (1 + np.sqrt(gamma)) ** 2
    lm = (1 - np.sqrt(gamma)) ** 2

    def integrand(x):
        d = _mp_density(x, gamma)
        return np.log(1 + s2m * x) * d if d > 0 else 0.0

    val = n * integrate.quad(integrand, lm, lp, limit=200)[0]
    _PSI_CACHE[key] = val
    return val


# ─── Parameter counting ─────────────────────────────────────────────────────────

def layer_params(E, mlp_ratio, num_heads):
    mlp_dim = int(E * mlp_ratio)
    qkv_dim = num_heads * HEAD_DIM
    p = 0
    p += E * 3 * qkv_dim + 3 * qkv_dim     # QKV + bias
    p += qkv_dim * E + E                     # output proj + bias
    p += 2 * E                               # LN (attn)
    p += E * mlp_dim + mlp_dim               # FFN up + bias
    p += mlp_dim * E + E                     # FFN down + bias
    p += 2 * E                               # LN (FFN)
    return p


def fixed_params(E):
    p = 0
    p += 3 * PATCH_SIZE * PATCH_SIZE * E + E  # patch embed + bias
    p += E                                     # CLS token
    p += SEQ_LEN * E                           # pos embed
    p += 2 * E                                 # final LN
    return p


# ─── Per-layer NSC-MP Scores ─────────────────────────────────────────────────

def nscmp_layer_merged(E, mlp_ratio, num_heads):
    qkv_dim = num_heads * HEAD_DIM
    mlp_dim = int(E * mlp_ratio)
    return (psi_mp(3 * qkv_dim, E) + psi_mp(E, qkv_dim) +
            psi_mp(E, mlp_dim) + psi_mp(mlp_dim, E))


def nscmp_layer_split(E, mlp_ratio, num_heads):
    qkv_dim = num_heads * HEAD_DIM
    mlp_dim = int(E * mlp_ratio)
    return (3 * psi_mp(qkv_dim, E) + psi_mp(E, qkv_dim) +
            psi_mp(E, mlp_dim) + psi_mp(mlp_dim, E))


def nscmp_fixed(E):
    return psi_mp(E, 3 * PATCH_SIZE * PATCH_SIZE)


# ─── GCD-compressed Bounded Knapsack DP ──────────────────────────────────────

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _solve_dp(D, layer_types, costs, values_m, values_s, fp, param_limit,
              top_k=1):
    """Bounded Knapsack DP for per-layer type allocation.

    Each of D layers picks one of K types. Maximize total NSC-MP (merged)
    subject to fp + sum(layer_costs) <= param_limit.

    Uses GCD compression on shifted costs to keep the DP state space small.
    Returns top_k solutions (different compositions) for this (E, D).

    Returns:
        list of (merged_score, split_score, n_counts, total_params), or []
    """
    K = len(layer_types)
    budget = param_limit - fp
    if budget < 0:
        return []

    c_min = min(costs)
    shifts = [c - c_min for c in costs]

    if all(s == 0 for s in shifts):
        best_k = int(np.argmax(values_m))
        n_counts = [0] * K
        n_counts[best_k] = D
        total_p = fp + D * costs[best_k]
        if total_p > param_limit:
            return []
        return [(D * values_m[best_k], D * values_s[best_k],
                 n_counts, total_p)]

    g = reduce(_gcd, [s for s in shifts if s > 0])
    compressed = [s // g for s in shifts]

    raw_budget = int(budget) - D * c_min
    if raw_budget < 0:
        return []
    S_max = raw_budget // g
    S = int(S_max) + 1

    dp = np.full(S, -np.inf, dtype=np.float64)
    dp[0] = 0.0
    bt = np.full((D, S), -1, dtype=np.int16)

    for layer in range(D):
        new_dp = np.full(S, -np.inf, dtype=np.float64)
        new_bt = np.full(S, -1, dtype=np.int16)
        for k in range(K):
            sh = compressed[k]
            v = values_m[k]
            if sh == 0:
                cand = dp + v
                mask = cand > new_dp
                new_dp[mask] = cand[mask]
                new_bt[mask] = k
            else:
                end = S - sh
                if end <= 0:
                    continue
                cand = dp[:end] + v
                mask = cand > new_dp[sh:]
                new_dp[sh:][mask] = cand[mask]
                new_bt[sh:][mask] = k
        dp = new_dp
        bt[layer] = new_bt

    valid = np.where(~np.isinf(dp))[0]
    if len(valid) == 0:
        return []

    # Top-K: pick states with highest dp values, backtrack each
    order = valid[np.argsort(dp[valid])[::-1]]
    results = []
    seen = set()

    for s_val in order:
        if len(results) >= top_k:
            break
        s_val = int(s_val)

        choices = [0] * D
        s = s_val
        for layer in range(D - 1, -1, -1):
            k = int(bt[layer, s])
            choices[layer] = k
            s -= compressed[k]

        n_counts = [0] * K
        total_split = 0.0
        for k_idx in choices:
            n_counts[k_idx] += 1
            total_split += values_s[k_idx]

        key = tuple(n_counts)
        if key in seen:
            continue
        seen.add(key)

        total_params = fp + sum(costs[k_idx] for k_idx in choices)
        merged_score = float(dp[s_val])
        results.append((merged_score, total_split, list(n_counts),
                         total_params))

    return results


# ─── Composition → per-layer config ──────────────────────────────────────────

def compose_config(D, n_counts, layer_types):
    """Convert composition counts to per-layer (mlp_ratio, num_heads) lists.

    Spreads higher-head layers evenly across depth.
    """
    layers = []
    for ci, count in enumerate(n_counts):
        layers.extend([layer_types[ci]] * count)

    max_h = max(h for _, h in layer_types)
    hi_indices = [i for i, l in enumerate(layers) if l[1] == max_h]
    lo_indices = [i for i, l in enumerate(layers) if l[1] != max_h]
    n_hi = len(hi_indices)

    if 0 < n_hi < D:
        reordered = [None] * D
        step = D / n_hi
        positions = [int(i * step + step / 2) for i in range(n_hi)]
        hi_pool = [layers[i] for i in hi_indices]
        lo_pool = [layers[i] for i in lo_indices]
        for pos in positions:
            reordered[pos] = hi_pool.pop(0)
        for i in range(D):
            if reordered[i] is None:
                reordered[i] = lo_pool.pop(0)
        layers = reordered

    mlp_ratios = [m for m, h in layers]
    num_heads = [h for m, h in layers]
    return mlp_ratios, num_heads


# ─── NSC-DP Main Search ──────────────────────────────────────────────────────

def nsc_dp_search(space_name, top_k_per_macro=10, verbose=True):
    """NSC-DP search for one AutoFormer supernet variant.

    Enumerates all feasible macro (embed_dim, depth), solves the micro
    allocation via Bounded Knapsack DP (returning top_k_per_macro
    compositions per macro config), returns all solutions ranked by NSC-MP.
    """
    cfg = SUPERNETS[space_name]
    param_limit = cfg["param_limit_M"] * 1e6

    layer_types = [(r, h)
                   for r in cfg["mlp_choices"]
                   for h in cfg["head_choices"]]
    K = len(layer_types)

    if verbose:
        print(f"\n{'='*72}")
        print(f"  NSC-DP + NSC-MP: AutoFormer-{space_name.upper()}")
        print(f"{'='*72}")
        print(f"  embed_dim  ∈ {cfg['embed_dims']}")
        print(f"  depth      ∈ {cfg['depth_choices']}")
        print(f"  mlp_ratio  ∈ {cfg['mlp_choices']}")
        print(f"  num_heads  ∈ {cfg['head_choices']}")
        print(f"  Layer types ({K}): {layer_types}")
        print(f"  Param limit: {param_limit/1e6:.1f}M")
        print(f"  Top-K per macro: {top_k_per_macro}")

    solutions = []
    t0 = time.perf_counter()

    for E in cfg["embed_dims"]:
        fp = fixed_params(E)

        costs = [layer_params(E, r, h) for r, h in layer_types]
        vals_m = [nscmp_layer_merged(E, r, h) for r, h in layer_types]
        vals_s = [nscmp_layer_split(E, r, h) for r, h in layer_types]
        fixed_nsc = nscmp_fixed(E)

        if verbose:
            print(f"\n  E={E}, fixed_params={fp/1e6:.4f}M, "
                  f"fixed_nscmp={fixed_nsc:.4f}")
            for i, lt in enumerate(layer_types):
                print(f"    (r={lt[0]}, h={lt[1]}): merged={vals_m[i]:.4f}, "
                      f"split={vals_s[i]:.4f}, "
                      f"params={costs[i]/1e6:.4f}M")

        for D in cfg["depth_choices"]:
            results = _solve_dp(D, layer_types, costs, vals_m, vals_s,
                                fp, param_limit, top_k=top_k_per_macro)
            if not results:
                if verbose:
                    print(f"    D={D}: infeasible")
                continue

            for rank, (best_m, best_s, n_counts, total_p) in enumerate(results):
                nsc_m = fixed_nsc + best_m
                nsc_s = fixed_nsc + best_s

                mlp_ratios, num_heads = compose_config(D, n_counts, layer_types)

                sol = {
                    "embed_dim":    E,
                    "depth":        D,
                    "n_counts":     n_counts,
                    "mlp_ratio":    mlp_ratios,
                    "num_heads":    num_heads,
                    "params":       total_p,
                    "nscmp_merged": nsc_m,
                    "nscmp_split":  nsc_s,
                }
                solutions.append(sol)

            if verbose:
                top1_m, _, top1_n, top1_p = results[0]
                desc = ", ".join(
                    f"{layer_types[i]}x{top1_n[i]}"
                    for i in range(K) if top1_n[i] > 0)
                print(f"    D={D}: {len(results)} compositions, "
                      f"best=[{desc}]  "
                      f"params={top1_p/1e6:.3f}M  "
                      f"NSC_m={fixed_nsc+top1_m:.2f}")

    elapsed = time.perf_counter() - t0

    solutions.sort(key=lambda x: x["nscmp_merged"], reverse=True)

    return {
        "space": space_name,
        "param_limit_M": cfg["param_limit_M"],
        "layer_types": layer_types,
        "search_time_s": round(elapsed, 4),
        "n_solutions": len(solutions),
        "solutions": solutions,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="NSC-DP + NSC-MP for AutoFormer")
    ap.add_argument("--space", type=str, default="all",
                    choices=["tiny", "small", "base", "all"])
    ap.add_argument("--top_k", type=int, default=10,
                    help="Number of top solutions to display (global)")
    ap.add_argument("--top_k_per_macro", type=int, default=20,
                    help="Top-K compositions per (E,D) macro config")
    ap.add_argument("--save", type=str, default=None,
                    help="Save results to JSON (auto-named if not given)")
    args = ap.parse_args()

    spaces = (["tiny", "small", "base"] if args.space == "all"
              else [args.space])
    all_output = {}

    for sp in spaces:
        sr = nsc_dp_search(sp, top_k_per_macro=args.top_k_per_macro,
                           verbose=True)
        solutions = sr["solutions"]

        print(f"\n{'='*72}")
        print(f"  NSC-DP Top-{args.top_k}: AutoFormer-{sp.upper()}")
        print(f"  Search time: {sr['search_time_s']:.4f}s")
        print(f"  Total feasible: {sr['n_solutions']} macro configs")
        print(f"{'='*72}")
        print(f"  {'Rank':<5s} {'E':>4s} {'D':>3s} {'Composition':>36s} "
              f"{'Params':>8s} {'NSC_m':>10s} {'NSC_s':>10s}")
        print("  " + "-" * 80)

        for i, sol in enumerate(solutions[:args.top_k]):
            K = len(sr["layer_types"])
            desc = ", ".join(
                f"{sr['layer_types'][j]}x{sol['n_counts'][j]}"
                for j in range(K) if sol["n_counts"][j] > 0)
            print(f"  {i+1:<5d} {sol['embed_dim']:>4d} {sol['depth']:>3d} "
                  f"{desc:>36s} "
                  f"{sol['params']/1e6:>7.3f}M "
                  f"{sol['nscmp_merged']:>10.4f} "
                  f"{sol['nscmp_split']:>10.4f}")

        if solutions:
            best = solutions[0]
            print(f"\n  Best architecture:")
            print(f"    embed_dim  = {best['embed_dim']}")
            print(f"    depth      = {best['depth']}")
            print(f"    num_heads  = {best['num_heads']}")
            print(f"    mlp_ratio  = {best['mlp_ratio']}")
            print(f"    n_counts   = {best['n_counts']}")
            print(f"    params     = {best['params']/1e6:.3f}M")
            print(f"    NSC-MP (merged) = {best['nscmp_merged']:.4f}")
            print(f"    NSC-MP (split)  = {best['nscmp_split']:.4f}")

        all_output[sp] = sr

    # Save
    save_path = args.save
    if save_path is None:
        tag = args.space
        save_path = os.path.join(os.environ.get("NSC_OUT", "outputs"), f"nsc_dp_autoformer_{tag}_results.json")

    serializable = {}
    for sp, sr in all_output.items():
        ser_solutions = []
        for sol in sr["solutions"][:args.top_k]:
            ser_solutions.append({
                "embed_dim": sol["embed_dim"],
                "depth": sol["depth"],
                "n_counts": sol["n_counts"],
                "mlp_ratio": sol["mlp_ratio"],
                "num_heads": sol["num_heads"],
                "params": sol["params"],
                "nscmp_merged": sol["nscmp_merged"],
                "nscmp_split": sol["nscmp_split"],
            })
        serializable[sp] = {
            "method": "NSC-DP",
            "search_space": f"AutoFormer-{sp.upper()}",
            "param_limit_M": sr["param_limit_M"],
            "layer_types": [list(lt) for lt in sr["layer_types"]],
            "search_time_s": sr["search_time_s"],
            "n_solutions": sr["n_solutions"],
            "top_solutions": ser_solutions,
        }

    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved → {save_path}")
    print()


if __name__ == "__main__":
    main()
