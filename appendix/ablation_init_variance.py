#!/usr/bin/env python3
"""
Ablation: NSC ranking robustness across 3 init-variance (σ) conventions.

Benchmarks: FlexiBERT (spec-faithful), GPT-2.
σ conventions:
  - Xavier:      s = sqrt(2 / (m + n))
  - Kaiming:     s = sqrt(2 / n)        (fan-in)
  - TruncNormal: s = 0.02               (GPT-2 / HuggingFace default)

For each (benchmark, σ) pair we recompute NSC via psi_mp(m, n, s) and report
Kendall τ, Spearman ρ vs ground truth, plus pairwise Spearman ρ between the
NSC orderings produced by different σ choices.
"""
import json, sys, os
import numpy as np
from scipy.stats import kendalltau, spearmanr
from scipy import integrate

sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.environ.get("NSC_ROOT", ".")

# ── Fresh psi_mp with explicit sigma ──

def _mp_density(x, gamma):
    lp = (1 + np.sqrt(gamma)) ** 2
    lm = (1 - np.sqrt(gamma)) ** 2
    if x < lm or x > lp:
        return 0.0
    return np.sqrt((lp - x) * (x - lm)) / (2 * np.pi * gamma * x)

_cache = {}

def psi_mp(m, n, s):
    key = (m, n, round(s, 12))
    if key in _cache:
        return _cache[key]
    if m < n:
        m, n = n, m
    if n == 0 or m == 0:
        _cache[key] = 0.0
        return 0.0
    gamma = n / m
    lp = (1 + np.sqrt(gamma)) ** 2
    lm = (1 - np.sqrt(gamma)) ** 2
    s2m = s ** 2 * m
    def integrand(x):
        d = _mp_density(x, gamma)
        return np.log(1 + s2m * x) * d if d > 0 else 0.0
    val = n * integrate.quad(integrand, lm, lp, limit=200)[0]
    _cache[key] = val
    return val


SIGMA_CONVS = {
    "Xavier":      lambda m, n: np.sqrt(2.0 / (m + n)),
    "Kaiming":     lambda m, n: np.sqrt(2.0 / n),
    "TruncNorm02": lambda m, n: 0.02,
}


# ── FlexiBERT scoring (flat-QKV, same as compute_flexibert.py) ──

with open(os.path.join(ROOT, "NAS/FlexiBERT/BERT_benchmark.json")) as f:
    fb_bench = json.load(f)
with open(os.path.join(ROOT, "paper/FlexiBERT/results/flexibert_corrected.json")) as f:
    fb_corrected = json.load(f)
fb_by_id = {a["id"]: a for a in fb_bench}


def flexibert_nsc(gt_data, get_H, get_DFF, sigma_fn):
    scores = []
    for rec in gt_data:
        hpo = fb_by_id[rec["arch_id"]]["hparams"]["model_hparam_overrides"]
        layers = hpo["nas_config"]["encoder_layers"]
        H = get_H(hpo)
        DFF_g = get_DFF(hpo, layers)
        total = 0.0
        for lc in layers:
            op = lc["operation_type"]
            nff = lc.get("num_feed_forward", 1)
            DFF = lc.get("feed_forward_dimension", DFF_g)
            ff_dim = DFF // nff
            if op == "SA":
                total += psi_mp(3 * H, H, sigma_fn(3 * H, H))
                total += psi_mp(H, H, sigma_fn(H, H))
            elif op == "LT":
                total += psi_mp(H, H, sigma_fn(H, H))
                total += psi_mp(H, H, sigma_fn(H, H))
            elif op == "DSC":
                total += psi_mp(H, H, sigma_fn(H, H))
            for _ in range(nff):
                total += psi_mp(ff_dim, H, sigma_fn(ff_dim, H))
                total += psi_mp(H, ff_dim, sigma_fn(H, ff_dim))
        scores.append(total)
    return np.array(scores)


# ── GPT-2 scoring (per-head, same as compute_gpt2.py) ──

with open(os.path.join(ROOT, "NAS/gpt2_bench/gpt2_benchmark.json")) as f:
    gpt2_bench = json.load(f)


def gpt2_nsc(sigma_fn):
    scores, neg_ppls = [], []
    for key, arch in gpt2_bench.items():
        cfg = arch["config"]
        dm = cfg["d_model"]; nl = cfg["n_layer"]
        nh = cfg["n_head"]
        di = cfg.get("d_inner", cfg.get("d_inner_list", []))
        if isinstance(di, int): di = [di] * nl
        if isinstance(nh, int): nh = [nh] * nl
        nsc = 0.0
        for i in range(nl):
            h = nh[i] if i < len(nh) else nh[-1]
            d = di[i] if i < len(di) else di[-1]
            hd = dm // h
            nsc += 3 * h * psi_mp(dm, hd, sigma_fn(dm, hd))
            nsc += psi_mp(dm, dm, sigma_fn(dm, dm))
            nsc += psi_mp(d, dm, sigma_fn(d, dm))
            nsc += psi_mp(dm, d, sigma_fn(dm, d))
        scores.append(nsc)
        neg_ppls.append(-arch.get("valid_ppl", arch.get("test_ppl", 0)))
    return np.array(scores), np.array(neg_ppls)


# ── Run ──

results = []
benchmarks = {
    "FlexiBERT-Corrected": {
        "data": fb_corrected,
        "gt": np.array([r["glue"] for r in fb_corrected]),
        "get_H": lambda hpo: hpo["hidden_size"],
        "get_DFF": lambda hpo, lc: lc[0].get("feed_forward_dimension", 1024),
    },
}

all_scores = {}  # (bench, sigma_name) -> score vector

for bname, binfo in benchmarks.items():
    for sname, sfn in SIGMA_CONVS.items():
        _cache.clear()
        s = flexibert_nsc(binfo["data"], binfo["get_H"], binfo["get_DFF"], sfn)
        gt = binfo["gt"]
        tau, _ = kendalltau(s, gt)
        rho, _ = spearmanr(s, gt)
        all_scores[(bname, sname)] = s
        results.append({"benchmark": bname, "sigma": sname,
                        "tau": round(tau, 4), "rho": round(rho, 4)})

for sname, sfn in SIGMA_CONVS.items():
    _cache.clear()
    s, gt = gpt2_nsc(sfn)
    tau, _ = kendalltau(s, gt)
    rho, _ = spearmanr(s, gt)
    all_scores[("GPT-2", sname)] = s
    results.append({"benchmark": "GPT-2", "sigma": sname,
                    "tau": round(tau, 4), "rho": round(rho, 4)})

# Pairwise rank agreement between σ choices (per benchmark)
pairwise = []
for bname in ["FlexiBERT-Corrected", "GPT-2"]:
    snames = list(SIGMA_CONVS.keys())
    for i in range(len(snames)):
        for j in range(i + 1, len(snames)):
            a = all_scores[(bname, snames[i])]
            b = all_scores[(bname, snames[j])]
            rho_ab = spearmanr(a, b).correlation
            tau_ab, _ = kendalltau(a, b)
            pairwise.append({
                "benchmark": bname, "pair": f"{snames[i]} vs {snames[j]}",
                "rho": round(rho_ab, 4), "tau": round(tau_ab, 4),
            })

# ── Print ──
print(f"\n{'='*75}")
print(f" Init-Variance Ablation")
print(f"{'='*75}")
print(f"  {'Benchmark':<24s} {'σ convention':<16s} {'τ':>7s} {'ρ':>7s}")
print(f"  {'-'*24} {'-'*16} {'-'*7} {'-'*7}")
for r in results:
    print(f"  {r['benchmark']:<24s} {r['sigma']:<16s} {r['tau']:7.4f} {r['rho']:7.4f}")

print(f"\n  Pairwise rank agreement (between σ choices):")
print(f"  {'Benchmark':<24s} {'Pair':<28s} {'ρ':>7s} {'τ':>7s}")
print(f"  {'-'*24} {'-'*28} {'-'*7} {'-'*7}")
for p in pairwise:
    print(f"  {p['benchmark']:<24s} {p['pair']:<28s} {p['rho']:7.4f} {p['tau']:7.4f}")

# ── Save ──
out = {"ranking": results, "pairwise": pairwise}
out_path = os.path.join(os.path.dirname(__file__), "ablation_init_variance.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved → {out_path}")
