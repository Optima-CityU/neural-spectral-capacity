#!/usr/bin/env python3
"""Compute NSC-MP for GPT-2 benchmark (200 archs). Uses sigma=0.02 (trunc_normal init)."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nsc_utils import psi_mp
from scipy.stats import kendalltau, spearmanr

SIGMA = 0.02

ROOT = os.environ.get("NSC_ROOT", ".")

with open(os.path.join(ROOT, "NAS/gpt2_bench/gpt2_benchmark.json")) as f:
    benchmark = json.load(f)

results = []
for key, arch in benchmark.items():
    cfg = arch["config"]
    d_model = cfg["d_model"]
    n_layer = cfg["n_layer"]
    n_head = cfg["n_head"]
    d_inner_list = cfg.get("d_inner", cfg.get("d_inner_list", []))
    if isinstance(d_inner_list, int):
        d_inner_list = [d_inner_list] * n_layer

    if isinstance(n_head, int):
        n_head_list = [n_head] * n_layer
    else:
        n_head_list = n_head

    nsc = 0.0
    for i in range(n_layer):
        nh = n_head_list[i] if i < len(n_head_list) else n_head_list[-1]
        d_inner = d_inner_list[i] if i < len(d_inner_list) else d_inner_list[-1]
        hd = d_model // nh

        psi_attn = 3 * nh * psi_mp(d_model, hd, SIGMA)
        psi_attn += psi_mp(d_model, d_model, SIGMA)
        psi_ffn = psi_mp(d_inner, d_model, SIGMA) + psi_mp(d_model, d_inner, SIGMA)
        nsc += psi_attn + psi_ffn

    neg_ppl = -arch.get("valid_ppl", arch.get("test_ppl", 0))
    n_params = sum(v for k, v in arch.get("params", {}).items() if isinstance(v, (int, float)))
    if n_params == 0:
        n_params = arch.get("total_params", 0)

    results.append({
        "arch_key": key,
        "neg_ppl": neg_ppl,
        "n_params": n_params,
        "nsc_mp": nsc,
        "d_model": d_model,
        "n_layer": n_layer,
    })

neg_ppl = [r["neg_ppl"] for r in results]
nsc = [r["nsc_mp"] for r in results]
params = [r["n_params"] for r in results]
tau, _ = kendalltau(nsc, neg_ppl)
rho, _ = spearmanr(nsc, neg_ppl)
tau_p, _ = kendalltau(params, neg_ppl)
rho_p, _ = spearmanr(params, neg_ppl)
unique = len(set(round(v, 10) for v in nsc))

print(f"GPT-2 (N={len(results)})")
print(f"  NSC-MP perhead_sum: tau={tau:.4f}, rho={rho:.4f}, unique={unique}")
print(f"  #Params:            tau={tau_p:.4f}, rho={rho_p:.4f}")
print(f"  (Paper target: tau=0.849, rho=0.968)")

out_path = os.path.join(os.path.dirname(__file__), "gpt2_scores.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_path}")
