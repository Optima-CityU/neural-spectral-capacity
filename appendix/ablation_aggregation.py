#!/usr/bin/env python3
"""
Aggregation functions (sum, min, geometric, harmonic, product, ...) over
per-layer psi_MP vectors, plus the GPT-2 per-layer loader. Shared by
ablation_aggregation_specfaithful.py and ablation_min_generalization.py.
"""
import json, sys, os, math
import numpy as np
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from nsc_utils import psi_mp

ROOT = os.environ.get("NSC_ROOT", ".")
SIGMA_GPT2 = 0.02




def compute_gpt2_layers():
    with open(os.path.join(ROOT, "NAS/gpt2_bench/gpt2_benchmark.json")) as f:
        benchmark = json.load(f)

    data = []
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

        layer_psis = []
        all_matrix_psis = []
        for i in range(n_layer):
            nh = n_head_list[i] if i < len(n_head_list) else n_head_list[-1]
            d_inner = d_inner_list[i] if i < len(d_inner_list) else d_inner_list[-1]
            hd = d_model // nh

            attn_psis = []
            for _ in range(nh):
                for _ in range(3):
                    attn_psis.append(psi_mp(d_model, hd, SIGMA_GPT2))
            attn_psis.append(psi_mp(d_model, d_model, SIGMA_GPT2))

            ffn_psis = [
                psi_mp(d_inner, d_model, SIGMA_GPT2),
                psi_mp(d_model, d_inner, SIGMA_GPT2),
            ]

            layer_psis.append({
                "attn": attn_psis,
                "ffn": ffn_psis,
                "all": attn_psis + ffn_psis,
            })
            all_matrix_psis.extend(attn_psis + ffn_psis)

        neg_ppl = -arch.get("valid_ppl", arch.get("test_ppl", 0))
        data.append({
            "glue": neg_ppl,
            "layers": layer_psis,
            "all_psis": all_matrix_psis,
        })
    return data


# ── Aggregation functions ──────────────────────────────────────────────
# Each takes a list of layer dicts and returns a scalar score.

def agg_flat_sum(layers):
    return sum(sum(L["all"]) for L in layers)

def agg_min_per_layer(layers):
    return sum(min(L["all"]) for L in layers)

def agg_min_sum_per_layer(layers):
    return sum(min(L["attn"]) if L["attn"] else 0 for L in layers) + \
           sum(sum(L["ffn"]) for L in layers)

def agg_geometric_per_layer(layers):
    s = 0.0
    for L in layers:
        vals = L["all"]
        gm = np.exp(np.mean(np.log(np.array(vals) + 1e-30)))
        s += len(vals) * gm
    return s

def agg_harmonic_per_layer(layers):
    s = 0.0
    for L in layers:
        vals = L["all"]
        hm = len(vals) / sum(1.0 / (v + 1e-30) for v in vals)
        s += len(vals) * hm
    return s

def agg_log_sum_per_layer(layers):
    return sum(math.log(1 + sum(L["all"])) for L in layers)

def agg_sqrt_sum_per_layer(layers):
    return sum(math.sqrt(sum(L["all"])) for L in layers)

def agg_log_min_per_layer(layers):
    return sum(math.log(1 + min(L["all"])) for L in layers)

def agg_per_matrix_log(layers):
    return sum(sum(math.log(1 + v) for v in L["all"]) for L in layers)

def agg_log_of_log(layers):
    return sum(math.log(1 + sum(math.log(1 + v) for v in L["all"])) for L in layers)

def agg_min_plus_sum(layers):
    """min dominates, sum as tiebreaker (auto-scaled)."""
    min_score = sum(min(L["all"]) for L in layers)
    sum_score = sum(sum(L["all"]) for L in layers)
    return min_score + 0.01 * sum_score

def agg_min_plus_sum_auto(layers):
    """min + sum with auto epsilon = 1/(n_layers * mean_psi)."""
    all_vals = [v for L in layers for v in L["all"]]
    mean_psi = np.mean(all_vals) if all_vals else 1.0
    eps = 1.0 / (len(layers) * mean_psi + 1e-10)
    min_score = sum(min(L["all"]) for L in layers)
    sum_score = sum(sum(L["all"]) for L in layers)
    return min_score + eps * sum_score

def agg_softmin_per_layer(layers, beta=0.1):
    """Softmin with temperature beta."""
    s = 0.0
    for L in layers:
        vals = np.array(L["all"])
        weights = np.exp(-vals / (beta * np.mean(vals) + 1e-10))
        weights /= weights.sum() + 1e-30
        s += np.dot(weights, vals)
    return s

def agg_softmin_beta05(layers):
    return agg_softmin_per_layer(layers, beta=0.5)

def agg_softmin_beta02(layers):
    return agg_softmin_per_layer(layers, beta=0.2)

def agg_softmin_beta01(layers):
    return agg_softmin_per_layer(layers, beta=0.1)

def agg_softmin_beta005(layers):
    return agg_softmin_per_layer(layers, beta=0.05)

def agg_attn_min_ffn_sum(layers):
    """Attention: min per layer; FFN: sum per layer."""
    s = 0.0
    for L in layers:
        attn = min(L["attn"]) if L["attn"] else 0
        ffn = sum(L["ffn"])
        s += attn + ffn
    return s

def agg_attn_harmonic_ffn_sum(layers):
    """Attention: harmonic mean; FFN: sum."""
    s = 0.0
    for L in layers:
        if L["attn"]:
            n = len(L["attn"])
            hm = n / sum(1.0 / (v + 1e-30) for v in L["attn"])
            s += n * hm
        s += sum(L["ffn"])
    return s

def agg_depth_weighted_sum(layers):
    """Linearly decreasing weights: shallower layers weighted more."""
    L_total = len(layers)
    return sum((L_total - i) / L_total * sum(L["all"]) for i, L in enumerate(layers))

def agg_depth_weighted_exp(layers):
    """Exponentially decaying depth weights (alpha=0.95)."""
    alpha = 0.95
    return sum(alpha ** i * sum(L["all"]) for i, L in enumerate(layers))

def agg_min_of_layer_sums(layers):
    """Global bottleneck: min over layer-level sums."""
    return min(sum(L["all"]) for L in layers) if layers else 0

def agg_prod_of_layer_sums(layers):
    """Product of per-layer sums (log domain for stability)."""
    return sum(math.log(sum(L["all"]) + 1e-30) for L in layers)

def agg_min_attn_times_sum_ffn(layers):
    """Product: min(attn) * sum(ffn), summed over layers."""
    s = 0.0
    for L in layers:
        attn_min = min(L["attn"]) if L["attn"] else 1.0
        ffn_sum = sum(L["ffn"]) if L["ffn"] else 1.0
        s += attn_min * ffn_sum
    return s

def agg_log_min_plus_sqrt_sum(layers):
    """log(1+min) + sqrt(sum), per layer."""
    s = 0.0
    for L in layers:
        s += math.log(1 + min(L["all"])) + math.sqrt(sum(L["all"]))
    return s

def agg_quadratic_mean(layers):
    """RMS per-layer, then sum."""
    s = 0.0
    for L in layers:
        vals = L["all"]
        s += math.sqrt(sum(v ** 2 for v in vals) / len(vals)) * len(vals)
    return s

def agg_entropy_weighted(layers):
    """Weight each layer by spectral entropy of its psi distribution."""
    s = 0.0
    for L in layers:
        vals = np.array(L["all"]) + 1e-30
        p = vals / vals.sum()
        entropy = -np.sum(p * np.log(p))
        s += entropy * vals.sum()
    return s

def agg_cv_penalized_sum(layers):
    """Sum minus penalty for within-layer variance (CV)."""
    s = 0.0
    for L in layers:
        vals = np.array(L["all"])
        mean = vals.mean()
        cv = vals.std() / (mean + 1e-10)
        s += vals.sum() * (1 - 0.5 * cv)
    return s

def agg_min_head_sum(layers):
    """Per head: take min over Q,K,V; then sum all heads + FFN."""
    s = 0.0
    for L in layers:
        attn = L["attn"]
        if len(attn) > 1 and (len(attn) - 1) % 3 == 0:
            n_heads = (len(attn) - 1) // 3
            for h in range(n_heads):
                qkv = attn[h*3:(h+1)*3]
                s += min(qkv)
            s += attn[-1]
        else:
            s += sum(attn)
        s += sum(L["ffn"])
    return s

def agg_min_block(layers):
    """min over (attn_total, ffn_total) per layer, then sum layers. (ablation 'min')"""
    s = 0.0
    for L in layers:
        attn_total = sum(L["attn"])
        ffn_total = sum(L["ffn"])
        s += min(attn_total, ffn_total)
    return s

def agg_log_min_block(layers):
    """log(1 + min(attn, ffn)) per layer, then sum."""
    s = 0.0
    for L in layers:
        attn_total = sum(L["attn"])
        ffn_total = sum(L["ffn"])
        s += math.log(1 + min(attn_total, ffn_total))
    return s

def agg_min_block_plus_sum(layers):
    """min(attn,ffn) + eps*total_sum."""
    min_s = sum(min(sum(L["attn"]), sum(L["ffn"])) for L in layers)
    total_s = sum(sum(L["all"]) for L in layers)
    return min_s + 0.001 * total_s

def agg_sqrt_min_block(layers):
    """sqrt(min(attn,ffn)) per layer, then sum."""
    return sum(math.sqrt(min(sum(L["attn"]), sum(L["ffn"]))) for L in layers)

def agg_min_block_times_nlayers(layers):
    """min(attn,ffn) averaged × n_layers^2."""
    vals = [min(sum(L["attn"]), sum(L["ffn"])) for L in layers]
    return sum(vals) * len(layers)

def agg_harmonic_block(layers):
    """Harmonic mean of (attn_total, ffn_total), then sum layers."""
    s = 0.0
    for L in layers:
        a, f = sum(L["attn"]), sum(L["ffn"])
        if a > 0 and f > 0:
            s += 2 * a * f / (a + f)
        else:
            s += 0
    return s

def agg_geometric_block(layers):
    """Geometric mean of (attn_total, ffn_total), then sum layers."""
    s = 0.0
    for L in layers:
        a, f = sum(L["attn"]), sum(L["ffn"])
        s += math.sqrt(a * f) if a > 0 and f > 0 else 0
    return s

def agg_min_block_log_sum(layers):
    """sum_l [min(attn,ffn)] + sum_l [log(1+sum_all)]."""
    s1 = sum(min(sum(L["attn"]), sum(L["ffn"])) for L in layers)
    s2 = sum(math.log(1 + sum(L["all"])) for L in layers)
    return s1 + s2

def agg_ratio_penalized(layers):
    """flat_sum penalized by attn/ffn imbalance."""
    s = 0.0
    for L in layers:
        a, f = sum(L["attn"]), sum(L["ffn"])
        total = a + f
        ratio = min(a, f) / (max(a, f) + 1e-10)
        s += total * ratio
    return s

def agg_weighted_harmonic_sum(layers):
    """Weighted: harmonic of blocks (for bottleneck) + 0.1*sum (for scale)."""
    h_s = 0.0
    flat_s = 0.0
    for L in layers:
        a, f = sum(L["attn"]), sum(L["ffn"])
        if a > 0 and f > 0:
            h_s += 2 * a * f / (a + f)
        flat_s += a + f
    return h_s + 0.1 * flat_s


ALL_AGGS = {
    "flat_sum": agg_flat_sum,
    "min_per_layer": agg_min_per_layer,
    "log_min_per_layer": agg_log_min_per_layer,
    "min+sum(0.01)": agg_min_plus_sum,
    "min+sum(auto)": agg_min_plus_sum_auto,
    "softmin(β=0.5)": agg_softmin_beta05,
    "softmin(β=0.2)": agg_softmin_beta02,
    "softmin(β=0.1)": agg_softmin_beta01,
    "softmin(β=0.05)": agg_softmin_beta005,
    "geometric": agg_geometric_per_layer,
    "harmonic": agg_harmonic_per_layer,
    "per_layer_log": agg_log_sum_per_layer,
    "per_layer_sqrt": agg_sqrt_sum_per_layer,
    "per_matrix_log": agg_per_matrix_log,
    "log_of_log": agg_log_of_log,
    "prod_layers": agg_prod_of_layer_sums,
    "attn_min+ffn_sum": agg_attn_min_ffn_sum,
    "attn_harmonic+ffn_sum": agg_attn_harmonic_ffn_sum,
    "min_head_sum": agg_min_head_sum,
    "depth_linear": agg_depth_weighted_sum,
    "depth_exp(0.95)": agg_depth_weighted_exp,
    "min_layer_sum": agg_min_of_layer_sums,
    "min_attn×ffn": agg_min_attn_times_sum_ffn,
    "log_min+sqrt_sum": agg_log_min_plus_sqrt_sum,
    "quadratic_mean": agg_quadratic_mean,
    "entropy_weighted": agg_entropy_weighted,
    "cv_penalized": agg_cv_penalized_sum,
    "min_block": agg_min_block,
    "log_min_block": agg_log_min_block,
    "min_block+sum": agg_min_block_plus_sum,
    "sqrt_min_block": agg_sqrt_min_block,
    "harmonic_block": agg_harmonic_block,
    "geometric_block": agg_geometric_block,
    "min_block_log_sum": agg_min_block_log_sum,
    "ratio_penalized": agg_ratio_penalized,
    "w_harmonic+sum": agg_weighted_harmonic_sum,
}

# ── Linear combination sweep: α*method_A + (1-α)*flat_sum ────────────
def make_combo(fn_a, fn_b, alpha):
    def combo(layers):
        return alpha * fn_a(layers) + (1 - alpha) * fn_b(layers)
    return combo



def evaluate(data, agg_fn):
    gt = [d["glue"] for d in data]
    scores = [agg_fn(d["layers"]) for d in data]
    tau, _ = kendalltau(scores, gt)
    rho, _ = spearmanr(scores, gt)
    unique = len(set(round(v, 8) for v in scores))
    return tau, rho, unique


