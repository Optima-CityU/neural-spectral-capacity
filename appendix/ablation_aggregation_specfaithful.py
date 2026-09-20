"""Aggregation ablation (Appendix: choice of aggregation) on FlexiBERT
(spec-faithful protocol) and GPT-2. Compares sum / mean / harmonic / geometric /
min / product / global-bottleneck aggregation of per-matrix psi_MP scores.
Construction mirrors compute_flexibert.py (spec-faithful setting): per-arch H,
per-layer D_FF, merged QKV as one 3H x H matrix."""
import os
import json
import sys

import numpy as np
from scipy.stats import spearmanr

from nsc_utils import psi_mp
import ablation_aggregation as A

with open(os.path.join(os.environ.get("FLEXIBERT_DIR", "FlexiBERT"), "BERT_benchmark.json")) as f:
    benchmark = json.load(f)
with open(os.path.join(os.environ.get("NSC_OUT", "outputs"), 'flexibert_corrected_scores.json')) as f:
    corrected = json.load(f)
bench_by_id = {a['id']: a for a in benchmark}


def layers_for(aid):
    hpo = bench_by_id[aid]['hparams']['model_hparam_overrides']
    H = hpo['hidden_size']
    out = []
    for lc in hpo['nas_config']['encoder_layers']:
        op = lc['operation_type']
        nff = lc.get('num_feed_forward', 1)
        ffd = lc.get('feed_forward_dimension', 1024) // nff
        if op == 'SA':
            attn = [psi_mp(3 * H, H), psi_mp(H, H)]
        elif op == 'LT':
            attn = [psi_mp(H, H), psi_mp(H, H)]
        else:
            attn = [psi_mp(H, H)]
        ffn = []
        for _ in range(nff):
            ffn.append(psi_mp(ffd, H))
            ffn.append(psi_mp(H, ffd))
        out.append({'attn': attn, 'ffn': ffn, 'all': attn + ffn})
    return out


fb = [{'glue': r['glue'], 'layers': layers_for(r['arch_id'])}
      for r in corrected]
g2 = A.compute_gpt2_layers()

aggs = {
    'sum (ours)': A.agg_flat_sum,
    'harmonic-mean/layer': A.agg_harmonic_per_layer,
    'geometric-mean/layer': A.agg_geometric_per_layer,
    'min/layer': A.agg_min_per_layer,
    'product over layers': A.agg_prod_of_layer_sums,
    'sum of log': A.agg_per_matrix_log,
    'global bottleneck': A.agg_min_of_layer_sums,
}
print(f"{'aggregation':28s} {'FB rho':>8s} {'G2 rho':>8s} {'mean':>8s}  FB-uniq")
for name, fn in aggs.items():
    sf = [fn(d['layers']) for d in fb]
    gf = [fn(d['layers']) for d in g2]
    r1 = spearmanr(sf, [d['glue'] for d in fb]).correlation
    r2 = spearmanr(gf, [d['glue'] for d in g2]).correlation
    print(f"{name:28s} {r1:8.4f} {r2:8.4f} {(r1+r2)/2:8.4f}  "
          f"{len(set(np.round(sf, 6)))}")
