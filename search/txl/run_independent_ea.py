#!/usr/bin/env python3
"""Run SynDiv, W-PCA and SoftmaxConf evolutionary search independently (separate
model builds) to measure per-method wall time on the Transformer-XL search space.
Each method: pop=50, gen=25, ~1250 evaluations.
"""
import sys, os

import json, time, random
import numpy as np
import torch

from search_txl_space import (
    D_MODEL, N_LAYER, D_INNER, N_HEAD,
    valid_heads, compute_params, _feasible_pairs,
    _rand_arch, _mutate, eval_model_zcps, default_nhead,
    nsc_mp_sum, knapsack_nsc_mp,
)

TARGET = 38_400_000
TOL    = 0.05
POP    = 50
NGEN   = 25
SEED   = 42


def run_single_method_ea(method_key, target, tol, pop_sz, n_gen, seed):
    """Run EA for a single ZCP method with independent model builds."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pairs = _feasible_pairs(target, tol)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_eval = 0
    t_start = time.time()

    def score_one(arch):
        nonlocal n_eval
        dm, nl = arch['d_model'], arch['n_layer']
        nh_list = [default_nhead(dm)] * nl
        scores = eval_model_zcps(dm, nl, arch['d_inner'], nh_list, device,
                                 seed=seed + n_eval)
        n_eval += 1
        if device == 'cuda' and n_eval % 100 == 0:
            torch.cuda.empty_cache()
        return scores[method_key]

    pop = []
    for _ in range(pop_sz * 10):
        a = _rand_arch(pairs, target, tol)
        if a:
            a['score'] = score_one(a)
            pop.append(a)
        if len(pop) >= pop_sz:
            break

    best = max(pop, key=lambda x: x['score'])

    for gen in range(n_gen):
        offspring = []
        for _ in range(pop_sz):
            cands = random.sample(pop, min(3, len(pop)))
            parent = max(cands, key=lambda x: x['score'])
            child = _mutate(parent, pairs, target, tol)
            if child:
                child['score'] = score_one(child)
                offspring.append(child)

        combined = pop + offspring
        combined.sort(key=lambda x: x['score'], reverse=True)
        pop = combined[:pop_sz]
        if pop[0]['score'] > best['score']:
            best = pop[0]

        if (gen + 1) % 5 == 0:
            elapsed = time.time() - t_start
            rate = n_eval / (elapsed + 1e-9)
            eta_s = max(0, (pop_sz + pop_sz * n_gen - n_eval)) / rate
            print(f"  [{method_key}] gen {gen+1:3d}/{n_gen}  "
                  f"evals={n_eval}  {elapsed:.1f}s  {rate:.1f}e/s  "
                  f"ETA {eta_s:.0f}s  best={best['score']:.4g}",
                  flush=True)

    total_time = time.time() - t_start
    best['wall_time'] = total_time
    best['n_eval'] = n_eval
    best['nsc_mp_sum'] = nsc_mp_sum(best['d_model'], best['n_layer'],
                                     best['d_inner'])
    return best, total_time


def main():
    methods = {
        'syndiv':       'SynDiv',
        'wpca':         'W-PCA',
        'softmax_conf': 'SoftmaxConf',
    }

    results = {}

    # NSC-MP first (instant)
    print("=" * 70)
    print("  NSC-MP (Knapsack DP)")
    print("=" * 70)
    t0 = time.time()
    r = knapsack_nsc_mp(TARGET, TOL)
    t_nsc = time.time() - t0
    print(f"  Time: {t_nsc:.3f}s")
    print(f"  dm={r['d_model']}  nl={r['n_layer']}  params={r['params']/1e6:.2f}M")
    results['NSC-MP'] = {
        'wall_time': t_nsc,
        'd_model': r['d_model'],
        'n_layer': r['n_layer'],
        'params': r['params'],
        'nsc': r['nsc'],
    }

    # Run each method independently
    for key, label in methods.items():
        print(f"\n{'=' * 70}")
        print(f"  {label}  (independent EA, pop={POP}, gen={NGEN})")
        print(f"{'=' * 70}", flush=True)

        best, wall = run_single_method_ea(key, TARGET, TOL, POP, NGEN, SEED)

        print(f"\n  {label} done:")
        print(f"    Wall time : {wall:.1f}s  ({wall/60:.1f} min)")
        print(f"    Evals     : {best['n_eval']}")
        print(f"    dm={best['d_model']}  nl={best['n_layer']}  "
              f"params={best['params']/1e6:.2f}M")
        print(f"    Score     : {best['score']:.4g}")
        print(f"    NSC-sum   : {best['nsc_mp_sum']:.1f}", flush=True)

        results[label] = {
            'wall_time': wall,
            'wall_time_min': round(wall / 60, 2),
            'n_eval': best['n_eval'],
            'd_model': best['d_model'],
            'n_layer': best['n_layer'],
            'params': best['params'],
            'score': best['score'],
            'nsc_mp_sum': best['nsc_mp_sum'],
            'd_inner': best['d_inner'],
        }

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    for name, r in results.items():
        wt = r['wall_time']
        print(f"  {name:<15s}  {wt:8.1f}s  ({wt/60:5.1f} min)")

    out_path = os.path.join(os.path.dirname(__file__),
                            'independent_ea_times.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved → {out_path}")


if __name__ == '__main__':
    main()
