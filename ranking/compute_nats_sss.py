#!/usr/bin/env python3
"""
Compute NSC-MP scores for NATS-Bench SSS (Size Search Space).
All 32768 architectures share the same cell genotype, only channel widths differ.

Architecture: channels = [c0, c1, c2, c3, c4]
Network layout (N=1 stage):
  stem:      Conv2d(3, c0, 3x3)
  cell_0:    InferCell(c0 -> c1)   [normal]
  cell_1:    ResNetBasicblock(c1 -> c2, stride=2)  [reduction]
  cell_2:    InferCell(c2 -> c3)   [normal]
  cell_3:    ResNetBasicblock(c3 -> c4, stride=2)  [reduction]
  cell_4:    InferCell(c4 -> c4)   [normal]  # last cell in/out same
  classifier: Linear(c4, num_classes)

InferCell genotype (fixed, best from NAS-Bench-201):
  |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|
  This means 5 conv operations total in each cell (skip_connect has no weights when C_in==C_out).

ResNetBasicblock(in, out, stride=2):
  conv_a: Conv(in, out, 3x3)
  conv_b: Conv(out, out, 3x3)
  downsample: Conv(in, out, 1x1) [when stride=2 or in!=out]

For NSC-MP, each Conv2d(C_in, C_out, k, k) has weight shape (C_out, C_in*k*k),
so the matrix is (C_out) x (C_in * k * k).
Xavier init: sigma = sqrt(2 / (C_out + C_in * k * k)).
"""

import csv
import sys
import os
import math
import numpy as np
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from nsc_utils import psi_mp, xavier_sigma


def conv_psi(c_in, c_out, kernel_size=3):
    """PSI for a Conv2d(c_in, c_out, k, k) with Xavier init."""
    fan_in = c_in * kernel_size * kernel_size
    m, n = c_out, fan_in
    sigma = xavier_sigma(m, n)
    return psi_mp(m, n, sigma)


def infer_cell_psis(c_in, c_out):
    """
    Compute PSI values for one InferCell.
    Fixed genotype: |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|

    Node 0 = input (channels = c_in)
    Node 1: nor_conv_3x3 from node 0 -> Conv(c_in, c_out, 3x3)
    Node 2: nor_conv_3x3 from node 0 -> Conv(c_in, c_out, 3x3)
            nor_conv_3x3 from node 1 -> Conv(c_out, c_out, 3x3)
    Node 3: skip_connect from node 0 -> Identity if c_in==c_out, else FactorizedReduce(c_in, c_out)
            nor_conv_3x3 from node 1 -> Conv(c_out, c_out, 3x3)
            nor_conv_3x3 from node 2 -> Conv(c_out, c_out, 3x3)
    """
    psis = []
    # Node 1: nor_conv_3x3 from 0
    psis.append(conv_psi(c_in, c_out, 3))
    # Node 2: nor_conv_3x3 from 0, nor_conv_3x3 from 1
    psis.append(conv_psi(c_in, c_out, 3))
    psis.append(conv_psi(c_out, c_out, 3))
    # Node 3: skip from 0 (no weights if c_in==c_out), nor_conv_3x3 from 1, nor_conv_3x3 from 2
    if c_in != c_out:
        # FactorizedReduce: two Conv(c_in, c_out//2, 1x1) + Conv(c_in, c_out-c_out//2, 1x1)
        c_half1 = c_out // 2
        c_half2 = c_out - c_half1
        psis.append(conv_psi(c_in, c_half1, 1))
        psis.append(conv_psi(c_in, c_half2, 1))
    psis.append(conv_psi(c_out, c_out, 3))
    psis.append(conv_psi(c_out, c_out, 3))
    return psis


def resnet_basicblock_psis(c_in, c_out):
    """ResNetBasicblock with stride=2: conv_a(3x3) + conv_b(3x3) + downsample(1x1)."""
    psis = []
    psis.append(conv_psi(c_in, c_out, 3))   # conv_a
    psis.append(conv_psi(c_out, c_out, 3))   # conv_b
    # downsample: AvgPool + Conv(c_in, c_out, 1x1) when stride=2
    psis.append(conv_psi(c_in, c_out, 1))
    return psis


def compute_nsc_for_arch(channels):
    """
    Compute NSC-MP for a NATS-Bench SSS architecture.
    channels: [c0, c1, c2, c3, c4]

    Returns dict with per-layer psi vectors and various aggregated scores.
    """
    c0, c1, c2, c3, c4 = channels

    all_layer_psis = []

    # stem: Conv(3, c0, 3x3)
    stem_psis = [conv_psi(3, c0, 3)]
    all_layer_psis.append({"name": "stem", "psis": stem_psis})

    # cell_0: InferCell(c0 -> c1) [normal]
    cell0_psis = infer_cell_psis(c0, c1)
    all_layer_psis.append({"name": "cell0_normal", "psis": cell0_psis})

    # cell_1: ResNetBasicblock(c1 -> c2, stride=2) [reduction]
    cell1_psis = resnet_basicblock_psis(c1, c2)
    all_layer_psis.append({"name": "cell1_reduce", "psis": cell1_psis})

    # cell_2: InferCell(c2 -> c3) [normal]
    cell2_psis = infer_cell_psis(c2, c3)
    all_layer_psis.append({"name": "cell2_normal", "psis": cell2_psis})

    # cell_3: ResNetBasicblock(c3 -> c4, stride=2) [reduction]
    cell3_psis = resnet_basicblock_psis(c3, c4)
    all_layer_psis.append({"name": "cell3_reduce", "psis": cell3_psis})

    # cell_4: InferCell(c4 -> c4) [normal, last]
    cell4_psis = infer_cell_psis(c4, c4)
    all_layer_psis.append({"name": "cell4_normal", "psis": cell4_psis})

    # classifier: Linear(c4, num_classes) - small, usually negligible
    # We include it for completeness: psi_mp(num_classes, c4)
    cls_psi = psi_mp(10, c4)  # CIFAR-10 has 10 classes
    all_layer_psis.append({"name": "classifier", "psis": [cls_psi]})

    return all_layer_psis


def agg_flat_sum(layer_psis_list):
    return sum(sum(lp["psis"]) for lp in layer_psis_list)


def agg_harmonic(layer_psis_list):
    """Harmonic mean per layer, scaled by count, then sum across layers."""
    s = 0.0
    for lp in layer_psis_list:
        vals = lp["psis"]
        if not vals:
            continue
        n = len(vals)
        hm = n / sum(1.0 / (v + 1e-30) for v in vals)
        s += n * hm
    return s


def agg_geometric(layer_psis_list):
    """Geometric mean per layer, scaled by count, then sum across layers."""
    s = 0.0
    for lp in layer_psis_list:
        vals = lp["psis"]
        if not vals:
            continue
        n = len(vals)
        gm = np.exp(np.mean(np.log(np.array(vals) + 1e-30)))
        s += n * gm
    return s


def agg_harmonic_block(layer_psis_list):
    """For each layer, treat all psis as one block, compute harmonic mean of layer sums."""
    layer_sums = [sum(lp["psis"]) for lp in layer_psis_list if lp["psis"]]
    if not layer_sums:
        return 0
    n = len(layer_sums)
    return n / sum(1.0 / (v + 1e-30) for v in layer_sums) * n


def zerolm_score(channels, sigma=None):
    """
    ZeroLM metric: sum of (||W||_F^2 / min(m,n)) for each weight matrix.
    For Xavier init: ||W||_F^2 = m * n * sigma^2, so metric = m*n*sigma^2 / min(m,n).
    If sigma is None, use Xavier init.
    """
    c0, c1, c2, c3, c4 = channels

    def zerolm_conv(c_in, c_out, ks):
        fan_in = c_in * ks * ks
        m, n = c_out, fan_in
        if sigma is None:
            s = xavier_sigma(m, n)
        else:
            s = sigma
        return m * n * s**2 / min(m, n)

    score = 0.0
    # stem
    score += zerolm_conv(3, c0, 3)
    # cell0: InferCell(c0->c1)
    score += zerolm_conv(c0, c1, 3)  # node1
    score += zerolm_conv(c0, c1, 3)  # node2 from 0
    score += zerolm_conv(c1, c1, 3)  # node2 from 1
    if c0 != c1:
        score += zerolm_conv(c0, c1//2, 1)
        score += zerolm_conv(c0, c1 - c1//2, 1)
    score += zerolm_conv(c1, c1, 3)  # node3 from 1
    score += zerolm_conv(c1, c1, 3)  # node3 from 2
    # cell1: ResNetBasicblock(c1->c2)
    score += zerolm_conv(c1, c2, 3)
    score += zerolm_conv(c2, c2, 3)
    score += zerolm_conv(c1, c2, 1)
    # cell2: InferCell(c2->c3)
    score += zerolm_conv(c2, c3, 3)
    score += zerolm_conv(c2, c3, 3)
    score += zerolm_conv(c3, c3, 3)
    if c2 != c3:
        score += zerolm_conv(c2, c3//2, 1)
        score += zerolm_conv(c2, c3 - c3//2, 1)
    score += zerolm_conv(c3, c3, 3)
    score += zerolm_conv(c3, c3, 3)
    # cell3: ResNetBasicblock(c3->c4)
    score += zerolm_conv(c3, c4, 3)
    score += zerolm_conv(c4, c4, 3)
    score += zerolm_conv(c3, c4, 1)
    # cell4: InferCell(c4->c4)
    score += zerolm_conv(c4, c4, 3)
    score += zerolm_conv(c4, c4, 3)
    score += zerolm_conv(c4, c4, 3)
    # skip_connect: c4==c4, identity, no weights
    score += zerolm_conv(c4, c4, 3)
    score += zerolm_conv(c4, c4, 3)
    return score


def estimate_params(channels):
    """Rough param count for ranking (ignoring BN params)."""
    c0, c1, c2, c3, c4 = channels
    p = 0
    # stem: 3*c0*3*3
    p += 3 * c0 * 9
    # cell0: InferCell(c0->c1): 2x Conv(c0,c1,3x3) + 3x Conv(c1,c1,3x3) + skip
    p += 2 * c0 * c1 * 9 + 3 * c1 * c1 * 9
    if c0 != c1:
        p += c0 * c1  # factorized reduce ~
    # cell1: ResNetBasicblock(c1->c2)
    p += c1 * c2 * 9 + c2 * c2 * 9 + c1 * c2
    # cell2: InferCell(c2->c3)
    p += 2 * c2 * c3 * 9 + 3 * c3 * c3 * 9
    if c2 != c3:
        p += c2 * c3
    # cell3: ResNetBasicblock(c3->c4)
    p += c3 * c4 * 9 + c4 * c4 * 9 + c3 * c4
    # cell4: InferCell(c4->c4): 5x Conv(c4,c4,3x3)
    p += 5 * c4 * c4 * 9
    # classifier
    p += c4 * 10
    return p


def main():
    csv_path = "/tmp/cifar10_sss.csv"
    print(f"Loading {csv_path} ...")
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  {len(rows)} architectures")

    gt_accs = []
    nsc_flat = []
    nsc_harmonic = []
    nsc_geometric = []
    nsc_harm_block = []
    zlm_he = []
    zlm_fixed = []
    params_list = []
    sum_ch_list = []

    for i, row in enumerate(rows):
        channels = [int(x) for x in row["arch"].split(":")]
        acc = float(row["acc"])
        gt_accs.append(acc)

        layer_psis = compute_nsc_for_arch(channels)
        nsc_flat.append(agg_flat_sum(layer_psis))
        nsc_harmonic.append(agg_harmonic(layer_psis))
        nsc_geometric.append(agg_geometric(layer_psis))
        nsc_harm_block.append(agg_harmonic_block(layer_psis))
        zlm_he.append(zerolm_score(channels, sigma=None))
        zlm_fixed.append(zerolm_score(channels, sigma=0.01))

        params_list.append(estimate_params(channels))
        sum_ch_list.append(sum(channels))

        if (i + 1) % 5000 == 0:
            print(f"  processed {i+1}/{len(rows)} ...")

    print(f"\nDone! Computing correlations...\n")

    methods = {
        "NSC-MP (flat_sum)": nsc_flat,
        "NSC-MP (harmonic)": nsc_harmonic,
        "NSC-MP (geometric)": nsc_geometric,
        "NSC-MP (harm_block)": nsc_harm_block,
        "ZeroLM (Xavier init)": zlm_he,
        "ZeroLM (σ=0.01)": zlm_fixed,
        "#Params (estimated)": params_list,
        "Sum of channels": sum_ch_list,
    }

    header = f"{'Method':<28s} {'τ':>8s} {'ρ':>8s} {'unique':>8s}"
    print(header)
    print("-" * len(header))

    for name, scores in methods.items():
        tau, _ = kendalltau(scores, gt_accs)
        rho, _ = spearmanr(scores, gt_accs)
        unique = len(set(round(v, 8) for v in scores))
        print(f"{name:<28s} {tau:8.4f} {rho:8.4f} {unique:8d}")

    # ZeroLM degeneracy analysis
    print(f"\n=== ZeroLM Degeneracy Analysis ===")
    zlm_arr = np.array(zlm_he)
    print(f"ZeroLM (He) range: {zlm_arr.min():.4f} ~ {zlm_arr.max():.4f}")
    print(f"ZeroLM (He) std:   {zlm_arr.std():.6f}")
    print(f"ZeroLM (He) CV:    {zlm_arr.std()/zlm_arr.mean():.6f}")
    print("  -> With Xavier init, ZeroLM metric = 2*m*n/((m+n)*min(m,n)) ≈ constant per layer!")
    print("  -> Total score mostly determined by #layers (fixed), not channel widths.")

    zlm_f = np.array(zlm_fixed)
    print(f"\nZeroLM (σ=0.01) range: {zlm_f.min():.4f} ~ {zlm_f.max():.4f}")
    print(f"ZeroLM (σ=0.01) std:   {zlm_f.std():.4f}")

    # Quartile analysis
    print(f"\n\n=== Accuracy Quartile Analysis ===")
    accs_arr = np.array(gt_accs)
    q25, q50, q75 = np.percentile(accs_arr, [25, 50, 75])
    print(f"Quartiles: Q25={q25:.4f}, Q50={q50:.4f}, Q75={q75:.4f}")

    for qname, mask in [
        ("Top 25% (acc>=Q75)", accs_arr >= q75),
        ("Middle 50%", (accs_arr >= q25) & (accs_arr < q75)),
        ("Bottom 25% (acc<Q25)", accs_arr < q25),
    ]:
        idx = np.where(mask)[0]
        if len(idx) < 10:
            continue
        gt_sub = accs_arr[idx]
        print(f"\n  {qname} ({len(idx)} archs, acc {gt_sub.min():.4f}~{gt_sub.max():.4f}):")
        for mname, scores in methods.items():
            s_sub = np.array(scores)[idx]
            tau, _ = kendalltau(s_sub, gt_sub)
            rho, _ = spearmanr(s_sub, gt_sub)
            print(f"    {mname:<28s} τ={tau:.4f}  ρ={rho:.4f}")


if __name__ == "__main__":
    main()
