#!/usr/bin/env python3
"""
Compute NSC-MP for FlexiBERT 500 architectures.

QKV modeling: merged (3H, H) — treats concatenated Q/K/V as one matrix.
Aggregation:  flat_sum and harmonic.
Setting:      spec-faithful (per-architecture H, per-layer D_FF).
"""
import json, sys, os
import numpy as np
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from nsc_utils import psi_mp

ROOT = os.environ.get("NSC_ROOT", ".")

with open(os.path.join(ROOT, "NAS/FlexiBERT/BERT_benchmark.json")) as f:
    benchmark = json.load(f)
with open(os.path.join(ROOT, "paper/FlexiBERT/results/flexibert_corrected.json")) as f:
    corrected_data = json.load(f)

bench_by_id = {a["id"]: a for a in benchmark}
corrected_by_id = {r["arch_id"]: r for r in corrected_data}


def compute_layer_psis(layers_cfg, H, D_FF_global):
    """For each layer, return list of per-matrix psi values.
    
    Args:
        layers_cfg: encoder_layers from nas_config
        H: hidden_size to use
        D_FF_global: global FFN dim (only used when per-layer field is absent)
    Returns:
        list of dicts, each with 'attn' and 'ffn' psi lists
    """
    layer_psis = []
    for lc in layers_cfg:
        op = lc["operation_type"]
        nff = lc.get("num_feed_forward", 1)
        D_FF = lc.get("feed_forward_dimension", D_FF_global)
        ff_dim = D_FF // nff

        attn_psis = []
        if op == "SA":
            attn_psis.append(psi_mp(3 * H, H))   # merged Q,K,V
            attn_psis.append(psi_mp(H, H))        # output proj
        elif op == "LT":
            attn_psis.append(psi_mp(H, H))
            attn_psis.append(psi_mp(H, H))
        elif op == "DSC":
            attn_psis.append(psi_mp(H, H))

        ffn_psis = []
        for _ in range(nff):
            ffn_psis.append(psi_mp(ff_dim, H))
            ffn_psis.append(psi_mp(H, ff_dim))

        layer_psis.append({"attn": attn_psis, "ffn": ffn_psis})
    return layer_psis


def agg_flat(layer_psis):
    """Flat sum of all psi values across all layers."""
    return sum(v for lp in layer_psis for v in lp["attn"] + lp["ffn"])


def agg_harmonic(layer_psis):
    """Harmonic mean per layer (scaled by count), then sum across layers."""
    s = 0.0
    for lp in layer_psis:
        vals = lp["attn"] + lp["ffn"]
        if not vals:
            continue
        n = len(vals)
        hm = n / sum(1.0 / (v + 1e-30) for v in vals)
        s += n * hm
    return s


AGGS = {
    "flat": agg_flat,
    "harmonic": agg_harmonic,
}


# ── ZeroLM (analytical): ||W||²_F / min(m,n) ──
# For W ~ N(0, σ²): E[||W||²_F] = m*n*σ², so score = m*n*σ² / min(m,n).
# ZeroLM uses Xavier init σ = sqrt(2/(fan_in+fan_out)).

def _zerolm_matrix(m, n):
    sigma = np.sqrt(2.0 / (m + n))
    return m * n * sigma ** 2 / min(m, n)


def compute_zerolm(layers_cfg, H, D_FF_global):
    """ZeroLM: S_ffn + α·S_attn.  Returns (S_attn, S_ffn) so caller can sweep α."""
    s_attn, s_ffn = 0.0, 0.0
    for lc in layers_cfg:
        op = lc["operation_type"]
        nh = lc["num_operation_heads"]
        nff = lc.get("num_feed_forward", 1)
        D_FF = lc.get("feed_forward_dimension", D_FF_global)
        ff_dim = D_FF // nff

        if op == "SA":
            hd = H // nh
            s_attn += 3 * nh * _zerolm_matrix(H, hd)  # Q,K,V per head
            s_attn += _zerolm_matrix(H, H)              # output proj
        elif op == "LT":
            s_attn += 2 * _zerolm_matrix(H, H)
        elif op == "DSC":
            s_attn += _zerolm_matrix(H, H)

        for _ in range(nff):
            s_ffn += _zerolm_matrix(ff_dim, H)
            s_ffn += _zerolm_matrix(H, ff_dim)
    return s_attn, s_ffn


def run_setting(setting_name, gt_data, get_H, get_D_FF_global):
    """Run one setting, compute all aggregation variants, print and return results."""
    arch_ids = [r["arch_id"] for r in gt_data]
    glues = np.array([r["glue"] for r in gt_data])
    n_params = np.array([r["n_params"] for r in gt_data])

    scores = {name: [] for name in AGGS}
    zl_attn_list, zl_ffn_list = [], []

    results = []
    for rec in gt_data:
        aid = rec["arch_id"]
        arch = bench_by_id[aid]
        hpo = arch["hparams"]["model_hparam_overrides"]
        layers_cfg = hpo["nas_config"]["encoder_layers"]

        H = get_H(hpo)
        D_FF_global = get_D_FF_global(hpo, layers_cfg)

        lp = compute_layer_psis(layers_cfg, H, D_FF_global)
        zl_a, zl_f = compute_zerolm(layers_cfg, H, D_FF_global)
        zl_attn_list.append(zl_a)
        zl_ffn_list.append(zl_f)

        row = {
            "arch_id": aid,
            "glue": rec["glue"],
            "n_params": rec["n_params"],
            "n_layers": rec["n_layers"],
        }
        for agg_name, agg_fn in AGGS.items():
            val = agg_fn(lp)
            row[f"nsc_mp_{agg_name}"] = val
            scores[agg_name].append(val)
        row["zerolm_attn"] = zl_a
        row["zerolm_ffn"] = zl_f

        baseline_keys = [
            "synflow", "snip", "naswot", "fisher", "grasp", "gradnorm",
            "pca", "wpca", "head_importance", "head_softmax_conf",
            "act_distance", "jacobian_cosine", "log_synflow", "nsc_w", "nsc_jd",
        ]
        for k in baseline_keys:
            if k in rec:
                row[k] = rec[k]
        results.append(row)

    # ── Print ──
    print(f"\n{'='*65}")
    print(f" {setting_name}  (N={len(results)})")
    print(f"{'='*65}")
    print(f"  {'method':<22s}  {'τ':>7s}  {'ρ':>7s}  {'unique':>6s}")
    print(f"  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*6}")

    for agg_name in AGGS:
        s = np.array(scores[agg_name])
        tau, _ = kendalltau(s, glues)
        rho, _ = spearmanr(s, glues)
        uniq = len(set(round(v, 10) for v in s))
        print(f"  NSC-MP {agg_name:<13s}  {tau:7.4f}  {rho:7.4f}  {uniq:6d}")

    tau_p, _ = kendalltau(n_params, glues)
    rho_p, _ = spearmanr(n_params, glues)
    print(f"  {'#Params':<22s}  {tau_p:7.4f}  {rho_p:7.4f}")

    for bname, bkey in [("NSC-W (SVD)", "nsc_w"), ("W-PCA", "wpca")]:
        vals = np.array([rec.get(bkey, np.nan) for rec in gt_data])
        if np.isnan(vals).all():
            continue
        tau_b, _ = kendalltau(vals, glues)
        rho_b, _ = spearmanr(vals, glues)
        print(f"  {bname:<22s}  {tau_b:7.4f}  {rho_b:7.4f}")

    # ZeroLM: sweep α from 0 to 3 in steps of 0.05
    zl_a = np.array(zl_attn_list)
    zl_f = np.array(zl_ffn_list)
    best_alpha, best_rho = 0.0, -1.0
    for a100 in range(0, 301, 5):
        alpha = a100 / 100.0
        s = zl_f + alpha * zl_a
        rho = spearmanr(s, glues).correlation
        if rho > best_rho:
            best_rho = rho
            best_alpha = alpha
    zl_best = zl_f + best_alpha * zl_a
    tau_zl, _ = kendalltau(zl_best, glues)
    rho_zl, _ = spearmanr(zl_best, glues)
    zl_sum = zl_f + zl_a
    tau_zl1, _ = kendalltau(zl_sum, glues)
    rho_zl1, _ = spearmanr(zl_sum, glues)
    print(f"  {'ZeroLM (α=1)':<22s}  {tau_zl1:7.4f}  {rho_zl1:7.4f}")
    print(f"  ZeroLM (α={best_alpha:<4.2f})      {tau_zl:7.4f}  {rho_zl:7.4f}")

    return results


# ── Spec-faithful setting: per-arch H, per-layer D_FF ──
corrected_results = run_setting(
    "Spec-faithful setting (per-arch H, per-layer D_FF)",
    corrected_data,
    get_H=lambda hpo: hpo["hidden_size"],
    get_D_FF_global=lambda hpo, lc: lc[0].get("feed_forward_dimension", 1024),
)

# ── Save ──
out_dir = os.environ.get("NSC_OUT", "outputs")
os.makedirs(out_dir, exist_ok=True)
for name, data in [("corrected", corrected_results)]:
    path = os.path.join(out_dir, f"flexibert_{name}_scores.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved {name} → {path}")
