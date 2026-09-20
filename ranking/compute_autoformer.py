#!/usr/bin/env python3
"""Compute NSC-MP perhead_sum for AutoFormer 3003 (Tiny+Small+Base) + load baseline proxies."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nsc_utils import psi_mp
from scipy.stats import kendalltau, spearmanr

ROOT = os.environ.get("NSC_ROOT", ".")

SCALES = [
    ("Tiny",  os.path.join(ROOT, "NAS/autoformer_bench/ood_vit_nas_data/autoformer_tiny_1k.json")),
    ("Small", os.path.join(ROOT, "NAS/autoformer_bench/ood_vit_nas_data/autoformer_small_1k.json")),
    ("Base",  os.path.join(ROOT, "NAS/autoformer_bench/ood_vit_nas_data/autoformer_base_1k.json")),
]

PROXY_DIR = os.path.join(ROOT, "ood_vit_nas_full")
PROXY_METHODS = {
    "dss": "compute_dss",
    "snip": "compute_snip",
    "grasp": "compute_grasp",
    "autoprox": "computeautoProx_P",
    "croze": "computeautocroze",
    "jacobian": "computeautojacobian",
    "meco": "compute_meco",
}


def nsc_mp_perhead(arch):
    ns = arch["net_setting"]
    depth = ns["layer_num"]
    ed_list = ns["embed_dim"]
    nh_list = ns["num_heads"]
    mr_list = ns["mlp_ratio"]

    nsc = 0.0
    for i in range(depth):
        ed = ed_list[i] if isinstance(ed_list, list) else ed_list
        nh = nh_list[i] if isinstance(nh_list, list) else nh_list
        mr = mr_list[i] if isinstance(mr_list, list) else mr_list
        hd = ed // nh
        mlp_dim = int(ed * mr)

        psi_attn = 3 * nh * psi_mp(ed, hd)   # per-head Q,K,V
        psi_attn += psi_mp(ed, ed)             # output proj
        psi_ffn = psi_mp(mlp_dim, ed) + psi_mp(ed, mlp_dim)
        nsc += psi_attn + psi_ffn
    return nsc


def load_proxies(scale_name):
    """Load all baseline proxy scores for a given scale."""
    scale_dir_map = {"Tiny": "Autoformer-Tiny", "Small": "Autoformer-Small", "Base": "Autoformer-Base"}
    proxy_base = os.path.join(PROXY_DIR, scale_dir_map[scale_name], "proxy")
    if not os.path.isdir(proxy_base):
        return {}

    proxy_data = {}
    for method_key, file_pattern in PROXY_METHODS.items():
        for fname in os.listdir(proxy_base):
            if file_pattern in fname and fname.endswith(".json"):
                with open(os.path.join(proxy_base, fname)) as f:
                    data = json.load(f)
                scores = {}
                for idx_str, rec in data.items():
                    score_val = None
                    for k in [method_key, "dss", "snip", "grasp", "meco",
                              "autoProx_P", "autoCroZe", "autoJacobian"]:
                        if k in rec:
                            score_val = rec[k]
                            break
                    if score_val is None:
                        for k, v in rec.items():
                            if k not in ("net_setting", "OOD-Accuracy") and isinstance(v, (int, float)):
                                score_val = v
                                break
                    scores[idx_str] = score_val
                proxy_data[method_key] = scores
                break
    return proxy_data


all_results = []

for scale_name, data_path in SCALES:
    with open(data_path) as f:
        bench = json.load(f)

    proxies = load_proxies(scale_name)

    scale_results = []
    for idx_str in sorted(bench.keys(), key=int):
        arch = bench[idx_str]
        acc = arch["performance"]["Imagenet"]["clean"]
        params = arch["params"]

        nsc = nsc_mp_perhead(arch)

        row = {
            "idx": int(idx_str),
            "scale": scale_name,
            "acc": acc,
            "params": params,
            "nsc_mp": nsc,
        }
        for method_key, scores_dict in proxies.items():
            row[method_key] = scores_dict.get(idx_str)

        scale_results.append(row)

    accs = [r["acc"] for r in scale_results]
    nscs = [r["nsc_mp"] for r in scale_results]
    params_list = [r["params"] for r in scale_results]
    tau, _ = kendalltau(nscs, accs)
    rho, _ = spearmanr(nscs, accs)
    tau_p, _ = kendalltau(params_list, accs)
    rho_p, _ = spearmanr(params_list, accs)
    unique = len(set(round(v, 10) for v in nscs))

    print(f"AutoFormer-{scale_name} (N={len(scale_results)})")
    print(f"  NSC-MP perhead_sum: tau={tau:.4f}, rho={rho:.4f}, unique={unique}")
    print(f"  #Params:            tau={tau_p:.4f}, rho={rho_p:.4f}")

    for mk in proxies:
        vals = [r[mk] for r in scale_results if r[mk] is not None]
        accs_sub = [r["acc"] for r in scale_results if r[mk] is not None]
        if len(vals) > 50:
            t, _ = kendalltau(vals, accs_sub)
            r, _ = spearmanr(vals, accs_sub)
            print(f"  {mk:12s}: tau={t:.4f}, rho={r:.4f}")

    all_results.extend(scale_results)

print(f"\nBucket analysis (perhead_sum):")
BUCKETS = [(5, 7), (18, 22), (24, 28)]
for lo, hi in BUCKETS:
    bucket = [r for r in all_results if lo <= r["params"] <= hi]
    if len(bucket) < 10:
        print(f"  {lo}-{hi}M: N={len(bucket)}, too few")
        continue
    accs = [r["acc"] for r in bucket]
    nscs = [r["nsc_mp"] for r in bucket]
    params_list = [r["params"] for r in bucket]
    tau, _ = kendalltau(nscs, accs)
    rho, _ = spearmanr(nscs, accs)
    tau_p, _ = kendalltau(params_list, accs)
    rho_p, _ = spearmanr(params_list, accs)
    print(f"  {lo}-{hi}M (N={len(bucket)}): NSC-MP tau={tau:.4f}/rho={rho:.4f}, "
          f"#Params tau={tau_p:.4f}/rho={rho_p:.4f}")

out_path = os.path.join(os.path.dirname(__file__), "autoformer_scores.json")
with open(out_path, "w") as f:
    json.dump(all_results, f)
print(f"\nSaved to {out_path}")
