#!/usr/bin/env python3
"""NSC-DP for the Transformer-XL search space.

Macro variables: g = (d_model, n_layer)
Micro variables: x_l = d_inner_l per layer (21 choices)
Solver: bounded knapsack DP for each macro configuration.

Usage:
  python nsc_dp_search.py                          # default 38.4M ±5%
  python nsc_dp_search.py --target 38.4e6 --tol 0.05
  python nsc_dp_search.py --top_k 20               # show top-20 solutions
"""

import math, time, argparse, json
import numpy as np
from scipy import integrate

# ─── Search Space ─────────────────────────────────────────────────────────────

D_MODEL = [128, 256, 384, 512, 640, 768, 1024]
N_LAYER = [12, 14, 16, 18]
D_INNER = list(range(512, 2049, 128)) + list(range(2304, 4097, 256))  # 21 choices
N_HEAD  = [1, 2, 4, 8, 16]
SIGMA   = 0.02


def valid_heads(dm):
    return [h for h in N_HEAD if dm % h == 0]


def default_nhead(dm):
    for h in sorted(valid_heads(dm), reverse=True):
        if dm // h >= 32:
            return h
    return valid_heads(dm)[0]


def compute_params(dm, nl, di_list):
    """Non-embedding params: per layer = 4·dm² + 9·dm + di·(2·dm+1)."""
    return nl * (4 * dm * dm + 9 * dm) + sum(di * (2 * dm + 1) for di in di_list)


# ─── NSC-MP (Marchenko-Pastur) ────────────────────────────────────────────────

_PSI_CACHE = {}


def _mp_density(x, gamma):
    lp = (1 + np.sqrt(gamma)) ** 2
    lm = (1 - np.sqrt(gamma)) ** 2
    if x < lm or x > lp:
        return 0.0
    return np.sqrt((lp - x) * (x - lm)) / (2 * np.pi * gamma * x)


def psi_mp(m, n, sigma=SIGMA):
    """Marchenko-Pastur spectral capacity ψ_MP(m, n, σ)."""
    key = (m, n, sigma)
    if key in _PSI_CACHE:
        return _PSI_CACHE[key]
    if m < n:
        m, n = n, m
    if n == 0 or m == 0:
        return 0.0
    gamma = n / m
    lp = (1 + np.sqrt(gamma)) ** 2
    lm = (1 - np.sqrt(gamma)) ** 2
    s2m = sigma ** 2 * m

    def integrand(x):
        d = _mp_density(x, gamma)
        return np.log(1 + s2m * x) * d if d > 0 else 0.0

    val = n * integrate.quad(integrand, lm, lp, limit=200)[0]
    _PSI_CACHE[key] = val
    return val


def nsc_mp_score(dm, nl, di_list):
    """Full NSC-MP flat-sum score for a complete architecture."""
    psi_qkv = psi_mp(3 * dm, dm)
    psi_o   = psi_mp(dm, dm)
    return sum(psi_qkv + psi_o + 2 * psi_mp(di, dm) for di in di_list)


# ─── Bounded Knapsack DP (numpy-vectorized) ──────────────────────────────────

_SHIFTED = np.array([d - D_INNER[0] for d in D_INNER], dtype=np.int64)


def _solve_dp(nl, min_sum, max_sum, psi_arr):
    """Solve the per-layer d_inner allocation via Bounded Knapsack DP.

    Numpy-vectorized: the inner loop over s_max states is replaced by
    array slicing, so only the 21 d_inner choices are iterated in Python.

    Args:
        nl:      number of layers
        min_sum: minimum allowed sum of d_inner values
        max_sum: maximum allowed sum of d_inner values
        psi_arr: np.array of shape (21,) — ψ_FFN values for each d_inner

    Returns:
        (best_psi, actual_sum, di_list) or None if infeasible
    """
    d_min = D_INNER[0]
    s_min = max(0, min_sum - nl * d_min)
    s_max = max_sum - nl * d_min
    if s_max < 0:
        return None

    S = s_max + 1
    dp = np.full(S, -np.inf)
    dp[0] = 0.0
    bt = np.full((nl, S), -1, dtype=np.int16)

    for layer in range(nl):
        new_dp = np.full(S, -np.inf)
        new_bt = np.full(S, -1, dtype=np.int16)
        for idx in range(len(D_INNER)):
            sh = int(_SHIFTED[idx])
            pv = psi_arr[idx]
            if sh == 0:
                cand = dp + pv
                mask = cand > new_dp
                new_dp[mask] = cand[mask]
                new_bt[mask] = idx
            else:
                end = S - sh
                if end <= 0:
                    continue
                cand = dp[:end] + pv
                slc = slice(sh, S)
                mask = cand > new_dp[slc]
                new_dp[sh:S][mask] = cand[mask]
                new_bt[sh:S][mask] = idx
        dp = new_dp
        bt[layer] = new_bt

    # Best feasible state in [s_min, s_max]
    feasible = dp[s_min:]
    if np.all(np.isinf(feasible)):
        return None
    best_s = s_min + int(np.argmax(feasible))
    best_psi = float(dp[best_s])

    # Backtrack
    di_list = [0] * nl
    s = best_s
    for layer in range(nl - 1, -1, -1):
        idx = int(bt[layer, s])
        di_list[layer] = D_INNER[idx]
        s -= int(_SHIFTED[idx])

    actual_sum = best_s + nl * d_min
    return best_psi, actual_sum, di_list


# ─── NSC-DP Main Search ──────────────────────────────────────────────────────

def nsc_dp_search(target, tol=0.05, verbose=True):
    """NSC-DP: search for the best architecture under a parameter budget.

    Enumerates all feasible macro configurations (d_model, n_layer),
    solves the micro-level d_inner allocation via Bounded Knapsack DP,
    and returns all feasible solutions ranked by NSC-MP score.

    Args:
        target: target non-embedding parameter count (e.g. 38.4e6)
        tol:    tolerance (e.g. 0.05 for ±5%)

    Returns:
        list of solution dicts, sorted by NSC-MP score descending
    """
    solutions = []

    for dm in D_MODEL:
        # Precompute macro-level constants
        psi_qkv = psi_mp(3 * dm, dm)
        psi_o   = psi_mp(dm, dm)
        c_attn  = psi_qkv + psi_o           # C(g) per layer
        di_psi  = np.array([psi_mp(di, dm) for di in D_INNER])
        cost_unit = 2 * dm + 1

        for nl in N_LAYER:
            # Macro-level fixed params: R_0(g)
            fixed_params = nl * (4 * dm * dm + 9 * dm)

            # Derive d_inner sum bounds from budget
            var_upper = target * (1 + tol) - fixed_params
            var_lower = target * (1 - tol) - fixed_params
            if var_upper < 0:
                continue

            max_sum = int(var_upper / cost_unit)
            min_sum = max(0, math.ceil(max(0, var_lower) / cost_unit))

            # Clip to feasible d_inner range
            if max_sum < nl * D_INNER[0]:
                continue
            max_sum = min(max_sum, nl * D_INNER[-1])
            min_sum = max(min_sum, nl * D_INNER[0])
            if min_sum > max_sum:
                continue

            # Solve micro-level allocation
            result = _solve_dp(nl, min_sum, max_sum, di_psi)
            if result is None:
                continue

            total_ffn_psi, di_sum, di_list = result
            nsc    = nl * c_attn + 2 * total_ffn_psi
            params = fixed_params + di_sum * cost_unit
            nh     = default_nhead(dm)

            solutions.append({
                'd_model':  dm,
                'n_layer':  nl,
                'd_inner':  di_list,
                'n_head':   [nh] * nl,
                'params':   params,
                'nsc':      nsc,
            })

            if verbose:
                di_uniq = sorted(set(di_list))
                di_desc = ', '.join(f'{d}x{di_list.count(d)}' for d in di_uniq)
                print(f"  dm={dm:4d}  nl={nl:2d}  "
                      f"d_inner=[{di_desc}]  "
                      f"params={params/1e6:.2f}M  NSC={nsc:.2f}")

    solutions.sort(key=lambda x: x['nsc'], reverse=True)
    return solutions


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='NSC-DP: NSC-guided Global Optimization for Transformer-XL')
    ap.add_argument('--target', type=float, default=38.4e6,
                    help='Target non-embedding param count (default: 38.4M)')
    ap.add_argument('--tol', type=float, default=0.05,
                    help='Tolerance (default: 0.05 = ±5%%)')
    ap.add_argument('--top_k', type=int, default=10,
                    help='Number of top solutions to display')
    ap.add_argument('--save', type=str, default=None,
                    help='Save results to JSON')
    args = ap.parse_args()

    target = args.target

    print("=" * 72)
    print("  NSC-DP: NSC-guided Global Optimization")
    print("  Search Space: Transformer-XL")
    print("=" * 72)
    print(f"  Macro:  d_model ∈ {D_MODEL}  ({len(D_MODEL)} choices)")
    print(f"          n_layer ∈ {N_LAYER}  ({len(N_LAYER)} choices)")
    print(f"  Micro:  d_inner ∈ [{D_INNER[0]}..{D_INNER[-1]}]  "
          f"({len(D_INNER)} choices/layer)")
    print(f"  Budget: {target/1e6:.1f}M ± {args.tol*100:.0f}%  "
          f"= [{target*(1-args.tol)/1e6:.2f}M, {target*(1+args.tol)/1e6:.2f}M]")
    print()

    print("  Solving each (d_model, n_layer) sub-problem...")
    t0 = time.time()
    solutions = nsc_dp_search(target, args.tol, verbose=True)
    elapsed = time.time() - t0

    print(f"\n  Total: {len(solutions)} feasible solutions found in {elapsed:.2f}s")

    # TXL Base reference
    txl_nsc = nsc_mp_score(410, 16, [2100] * 16)
    txl_params = compute_params(410, 16, [2100] * 16)

    # Top-K
    print(f"\n{'='*72}")
    print(f"  Top-{args.top_k} Architectures by NSC-MP Score")
    print(f"{'='*72}")
    print(f"  {'Rank':<5s} {'dm':>4s} {'nl':>3s} {'d_inner (unique)':>30s} "
          f"{'Params':>8s} {'NSC-MP':>10s}")
    print("  " + "-" * 65)

    # Reference line
    print(f"  {'TXL':>5s} {410:>4d} {16:>3d} {'2100x16':>30s} "
          f"{txl_params/1e6:>7.2f}M {txl_nsc:>10.2f}")
    print("  " + "-" * 65)

    for i, sol in enumerate(solutions[:args.top_k]):
        di_uniq = sorted(set(sol['d_inner']))
        di_desc = ', '.join(f'{d}x{sol["d_inner"].count(d)}' for d in di_uniq)
        print(f"  {i+1:<5d} {sol['d_model']:>4d} {sol['n_layer']:>3d} "
              f"{di_desc:>30s} "
              f"{sol['params']/1e6:>7.2f}M {sol['nsc']:>10.2f}")

    if solutions:
        best = solutions[0]
        print(f"\n  Best architecture:")
        print(f"    d_model  = {best['d_model']}")
        print(f"    n_layer  = {best['n_layer']}")
        print(f"    n_head   = {best['n_head'][0]} (per layer)")
        print(f"    d_inner  = {best['d_inner']}")
        print(f"    params   = {best['params']/1e6:.2f}M")
        print(f"    NSC-MP   = {best['nsc']:.4f}")
        print(f"    vs TXL Base NSC-MP = {txl_nsc:.4f} "
              f"(Δ = {best['nsc'] - txl_nsc:+.4f})")
        print(f"    Search time: {elapsed:.2f}s")

    if args.save:
        out = {
            'method': 'NSC-DP',
            'search_space': 'Transformer-XL',
            'target_params': target,
            'tolerance': args.tol,
            'search_time_sec': elapsed,
            'n_solutions': len(solutions),
            'reference': {
                'name': 'TXL Base',
                'd_model': 410, 'n_layer': 16,
                'd_inner': [2100] * 16,
                'params': txl_params,
                'nsc': txl_nsc,
            },
            'solutions': solutions[:args.top_k],
        }
        with open(args.save, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\n  Results saved → {args.save}")

    print()


if __name__ == '__main__':
    main()
