"""Aggregation appendix: why the min rule collapses, on FlexiBERT (spec-faithful),
GPT-2 and NATS-Bench-SSS. Reports the number of distinct scores per rule and the
rank correlation of min vs sum, and which layer attains the minimum."""
import csv, json, sys
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
from nsc_utils import psi_mp, xavier_sigma
import ablation_aggregation as A

def conv_psi(c_in, c_out, k):
    fan_in = c_in * k * k
    return psi_mp(c_out, fan_in, xavier_sigma(c_out, fan_in))

def nats_blocks(ch, nc=100):
    c0, c1, c2, c3, c4 = ch
    b = {'stem': [conv_psi(3, c0, 3)]}
    cell0 = [conv_psi(c0, c1, 3)] * 2 + [conv_psi(c1, c1, 3)] * 3
    if c0 != c1:
        cell0.append(conv_psi(c0, c1, 1))
    b['cell0'] = cell0
    b['cell1'] = [conv_psi(c1, c2, 3), conv_psi(c2, c2, 3), conv_psi(c1, c2, 1)]
    cell2 = [conv_psi(c2, c3, 3)] * 2 + [conv_psi(c3, c3, 3)] * 3
    if c2 != c3:
        cell2.append(conv_psi(c2, c3, 1))
    b['cell2'] = cell2
    b['cell3'] = [conv_psi(c3, c4, 3), conv_psi(c4, c4, 3), conv_psi(c3, c4, 1)]
    b['cell4'] = [conv_psi(c4, c4, 3)] * 5
    b['classifier'] = [psi_mp(nc, c4)]
    return b

def report(name, s, y):
    rho = spearmanr(s, y).correlation
    print(f"{name:36s} rho={rho:.3f}  distinct={len(set(np.round(s, 6)))}/{len(s)}")

print('--- GPT-2: depth decomposition ---')
g2 = A.compute_gpt2_layers()
tot = [[sum(l['all']) for l in d['layers']] for d in g2]
y = [d['glue'] for d in g2]
mins = [min(t) for t in tot]
dep = [len(t) for t in tot]
report('sum', [sum(t) for t in tot], y)
report('min', mins, y)
report('min * n_layer', [m * k for m, k in zip(mins, dep)], y)
print('rho(min,depth) =', round(spearmanr(mins, dep).correlation, 3),
      '  rho(depth,y) =', round(spearmanr(dep, y).correlation, 3))

print('--- NATS-SSS: paper protocol ---')
d = json.load(open(os.path.join(os.environ.get("NSC_OUT", "outputs"), 'scores', 'nats_sss_c100_realdata_scores.json')))
acc = [r['acc'] for r in d]
sums, m7, m5, argn = [], [], [], Counter()
for r in d:
    b = nats_blocks(r['channels'])
    tots = {k: sum(v) for k, v in b.items()}
    sums.append(sum(tots.values()))
    m7.append(min(tots.values()))
    m5.append(min(v for k, v in tots.items() if k.startswith('cell')))
    argn[min(tots, key=tots.get)] += 1
assert np.allclose(sums, [r['nsc_mp'] for r in d])
report('sum (== paper nsc_mp)', sums, acc)
report('min over 7 blocks', m7, acc)
report('min over cells only', m5, acc)
print('argmin block:', dict(argn))

# ---------------------------------------------------------------------------
