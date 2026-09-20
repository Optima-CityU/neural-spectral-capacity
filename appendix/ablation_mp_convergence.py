#!/usr/bin/env python3
"""
Finite-size convergence of psi_MP (Marchenko-Pastur integral)
versus psi_W (empirical SVD on sampled matrices).

Sweep:
  d (= min(m, n))  in {32, 64, 128, 256, 512, 1024, 2048, 4096}
  gamma = n / m     in {1.0, 0.5, 0.25}
  init convention   in {Xavier, Kaiming, TruncNorm-0.02}

For each (d, gamma, init) we:
  1. Compute psi_MP analytically via closed-form quadrature.
  2. Sample K=20 random matrices W ~ N(0, sigma^2), compute
     psi_W = sum log(1 + s_i^2) from singular values, average over K.
  3. Record relative error |psi_MP - psi_W| / psi_W.

Outputs:
  - mp_convergence.json
  - mp_convergence.pdf (3-panel figure, one per gamma)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import integrate

OUT_DIR = os.path.dirname(__file__)

# ── psi_mp (analytical) ──

def _mp_density(x, gamma):
    lp = (1 + np.sqrt(gamma)) ** 2
    lm = (1 - np.sqrt(gamma)) ** 2
    if x < lm or x > lp:
        return 0.0
    return np.sqrt((lp - x) * (x - lm)) / (2 * np.pi * gamma * x)

def psi_mp(m, n, sigma):
    if m < n:
        m, n = n, m
    if n == 0 or m == 0:
        return 0.0
    gamma = n / m
    lp = (1 + np.sqrt(gamma)) ** 2
    lm = (1 - np.sqrt(gamma)) ** 2
    s2m = sigma ** 2 * m
    def integrand(x):
        d = _mp_density(x, gamma)
        return np.log(1 + s2m * x) * d if d > 0 else 0.0
    return n * integrate.quad(integrand, lm, lp, limit=200)[0]


# ── psi_W (empirical SVD) ──

def psi_svd(m, n, sigma, K=20):
    vals = []
    for _ in range(K):
        W = np.random.randn(m, n) * sigma
        sv = np.linalg.svd(W, compute_uv=False)
        vals.append(np.sum(np.log(1.0 + sv ** 2)))
    return float(np.mean(vals))


# ── Configuration ──

D_VALUES = [32, 64, 128, 256, 512, 1024, 2048, 4096]
GAMMAS = [1.0, 0.5, 0.25]
INITS = {
    "Xavier":      lambda m, n: np.sqrt(2.0 / (m + n)),
    "Kaiming":     lambda m, n: np.sqrt(2.0 / n),
    "TruncNorm02": lambda m, n: 0.02,
}

def _k_for_d(d):
    """Fewer SVD samples for large matrices to keep runtime manageable."""
    if d <= 512:
        return 20
    if d <= 1024:
        return 10
    if d <= 2048:
        return 5
    return 3

np.random.seed(42)

results = []

for gamma in GAMMAS:
    for d in D_VALUES:
        n = d
        m = int(round(n / gamma))
        for init_name, sigma_fn in INITS.items():
            sigma = sigma_fn(m, n)
            v_mp = psi_mp(m, n, sigma)
            v_svd = psi_svd(m, n, sigma, K=_k_for_d(d))
            rel_err = abs(v_mp - v_svd) / max(abs(v_svd), 1e-30)
            results.append({
                "gamma": gamma, "d": d, "m": m, "n": n,
                "init": init_name, "sigma": round(sigma, 8),
                "psi_mp": round(v_mp, 6), "psi_svd": round(v_svd, 6),
                "rel_error": round(rel_err, 6),
            })
            print(f"  gamma={gamma:.2f}  d={d:>5d}  {init_name:<12s}  "
                  f"σ={sigma:.6f}  ψ_MP={v_mp:10.4f}  ψ_W={v_svd:10.4f}  "
                  f"rel_err={rel_err:.6f}")

# ── Save JSON ──
json_path = os.path.join(OUT_DIR, "mp_convergence.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved JSON → {json_path}")

# ── Plot: 3-panel figure (one per gamma) ──

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 10,
})

INIT_STYLES = {
    "Xavier":      {"color": "#2196F3", "marker": "o",  "ls": "-"},
    "Kaiming":     {"color": "#FF5722", "marker": "s",  "ls": "--"},
    "TruncNorm02": {"color": "#4CAF50", "marker": "^",  "ls": "-."},
}

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

ERR_FLOOR = 1e-8  # below this we have only quadrature noise; flooring keeps log-scale plottable.

for ax_idx, gamma in enumerate(GAMMAS):
    ax = axes[ax_idx]
    for init_name, style in INIT_STYLES.items():
        ds = [r["d"] for r in results if r["gamma"] == gamma and r["init"] == init_name]
        errs = [max(r["rel_error"], ERR_FLOOR)
                for r in results if r["gamma"] == gamma and r["init"] == init_name]
        ax.plot(ds, errs, label=init_name.replace("02", "-0.02"),
                color=style["color"], marker=style["marker"],
                linestyle=style["ls"], markersize=5, linewidth=1.5)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e0)
    ax.set_yticks([10 ** k for k in range(-8, 1, 2)])  # 10^-8, 10^-6, ..., 10^0
    ax.set_xlabel(r"$\min(m,n)$")
    ax.set_title(rf"$\gamma = {gamma}$")
    ax.grid(True, which="major", alpha=0.18, linewidth=0.5)
    ax.grid(True, which="minor", alpha=0.07, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.set_xticks(D_VALUES)
    ax.set_xticklabels([str(d) for d in D_VALUES], fontsize=7, rotation=45)
    ax.tick_params(axis="both", which="both", length=2.5)

axes[0].set_ylabel("Relative error  " + r"$|\psi_\mathrm{MP} - \psi_W| \,/\, |\psi_W|$")
axes[-1].legend(fontsize=8, loc="upper right", framealpha=0.9)

fig.suptitle(r"Convergence of $\psi_\mathrm{MP}$ (analytical) to $\psi_W$ (empirical SVD)",
             fontsize=12, y=1.02)
fig.tight_layout()

pdf_path = os.path.join(OUT_DIR, "mp_convergence.pdf")
fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
print(f"Saved figure → {pdf_path}")


