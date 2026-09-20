#!/usr/bin/env python3
"""
Compute NSC-MP (flatsum) on OFA-MobileNetV3 search space.

Conv2d(Cin, Cout, k) -> equivalent matrix (Cout, Cin*k*k)
Aggregation: flatsum = sum of psi_MP over all conv/linear layers.

OFA-MobileNetV3 search space:
  - 5 stages, base channels [16, 24, 40, 80, 112, 160]
  - ks ∈ {3,5,7} per layer (depthwise kernel size)
  - e ∈ {3,4,6} per layer (expand ratio)
  - d ∈ {2,3,4} per stage (depth)
  - Each MBConv block: 1x1 expand -> kxk depthwise -> 1x1 project
"""

import json
import numpy as np
from scipy import integrate
from scipy.stats import kendalltau, spearmanr


# ============================================================
# NSC-MP core
# ============================================================
_PSI = {}

def _mp_density(x, gamma):
    lp = (1 + np.sqrt(gamma)) ** 2
    lm = (1 - np.sqrt(gamma)) ** 2
    if x < lm or x > lp:
        return 0.0
    return np.sqrt((lp - x) * (x - lm)) / (2 * np.pi * gamma * x)

def psi_mp(m, n, sigma=None):
    if sigma is None:
        sigma = np.sqrt(2.0 / (m + n))  # Xavier init
    key = (m, n, round(sigma, 8))
    if key in _PSI:
        return _PSI[key]
    if m < n:
        m, n = n, m
    if n == 0 or m == 0:
        _PSI[key] = 0.0
        return 0.0
    gamma = n / m
    lp = (1 + np.sqrt(gamma)) ** 2
    lm = (1 - np.sqrt(gamma)) ** 2
    s2 = sigma ** 2 * m

    def integrand(x):
        d = _mp_density(x, gamma)
        return np.log(1 + s2 * x) * d if d > 0 else 0.0

    val = n * integrate.quad(integrand, lm, lp, limit=200)[0]
    _PSI[key] = val
    return val


# ============================================================
# OFA-MobileNetV3
# ============================================================

# stage 0: first conv (3 -> 16)
# stages 1-5: MBConv blocks with base channels below
# stage 6: final layers (conv 1x1 + pool + fc)
MNV3_BASE_CHANNELS = [16, 16, 24, 40, 80, 112, 160]


def nsc_mp_mnv3(phenotype):
    """Compute NSC-MP(flatsum) for OFA-MobileNetV3."""
    ks_list = phenotype['ks']   # len=20: kernel size per layer
    e_list = phenotype['e']     # len=20: expand ratio per layer
    d_list = phenotype['d']     # len=5: depth per stage

    # First conv: Conv2d(3, 16, 3, stride=2)
    score = psi_mp(16, 3 * 3 * 3)

    layer_idx = 0
    for stage in range(5):
        c_in = MNV3_BASE_CHANNELS[stage + 1]
        c_out = (MNV3_BASE_CHANNELS[stage + 2]
                 if stage + 1 < len(MNV3_BASE_CHANNELS) - 1
                 else MNV3_BASE_CHANNELS[-1])
        depth = d_list[stage]

        for block in range(depth):
            if layer_idx >= len(ks_list):
                break
            ks = ks_list[layer_idx]
            expand = e_list[layer_idx]
            layer_idx += 1

            block_cin = c_in if block == 0 else c_out
            mid = block_cin * expand

            # 1x1 expand: (block_cin -> mid)
            if expand != 1:
                score += psi_mp(mid, block_cin)

            # Depthwise conv: (mid -> mid, groups=mid)
            # Each group: Conv2d(1, 1, ks) -> matrix (1, ks*ks)
            # Total contribution: mid * psi_mp(1, ks*ks)
            score += mid * psi_mp(1, ks * ks)

            # 1x1 project: (mid -> c_out)
            score += psi_mp(c_out, mid)

    # Final layers: Conv2d(160, 960, 1) -> pool -> Linear(960, 1280) -> Linear(1280, 1000)
    score += psi_mp(960, MNV3_BASE_CHANNELS[-1])
    score += psi_mp(1280, 960)
    score += psi_mp(1000, 1280)

    return score


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    import os
    data_path = os.path.join(os.path.dirname(__file__), '..', 'mnv3_data.json')
    with open(data_path) as f:
        mnv3 = json.load(f)
    print(f"Loaded {len(mnv3)} MNV3 architectures")

    print("Computing NSC-MP scores...")
    scores = [nsc_mp_mnv3(d['phenotype']) for d in mnv3]
    accs = [d['test_acc'] for d in mnv3]
    params = [d['params'] for d in mnv3]

    rho_n = spearmanr(scores, accs).correlation
    rho_p = spearmanr(params, accs).correlation
    print(f"\nGlobal (N={len(mnv3)}):")
    print(f"  NSC-MP  Spearman rho = {rho_n:.4f}")
    print(f"  #Params Spearman rho = {rho_p:.4f}")

    resolutions = sorted(set(d['phenotype']['r'] for d in mnv3))
    print(f"\nPer resolution:")
    for r in resolutions:
        mask = [i for i, d in enumerate(mnv3) if d['phenotype']['r'] == r]
        rn = spearmanr([scores[i] for i in mask], [accs[i] for i in mask]).correlation
        rp = spearmanr([params[i] for i in mask], [accs[i] for i in mask]).correlation
        print(f"  r={r}: NSC-MP rho={rn:.4f}, #Params rho={rp:.4f} (N={len(mask)})")
