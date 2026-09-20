#!/usr/bin/env python3
"""Plot the marginal gain d psi_MP / d d_ff as a function of d_ff,
one curve per d_model in {128, 256, 384, 512, 640, 768, 1024}.
Concavity of psi_MP in d_ff manifests as monotonically decreasing curves.
"""
import os, sys, math, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from run_realdata_proxy import psi_mp

D_MODELS = [128, 256, 384, 512, 640, 768, 1024]
# d_ff sweep: anchor on TXL FFN search range 512..4096, step 128
D_FF = np.arange(512, 4096 + 1, 128)
SIGMA = 0.02  # Transformer-XL truncated-normal init

fig, ax = plt.subplots(figsize=(5.6, 3.6))
cmap = plt.get_cmap("viridis")
for i, dm in enumerate(D_MODELS):
    psi_vals = np.array([psi_mp(d, dm, SIGMA) for d in D_FF])
    # forward differences as proxy for d psi / d d_ff
    grad = np.diff(psi_vals) / np.diff(D_FF)
    d_mid = (D_FF[:-1] + D_FF[1:]) / 2
    ax.plot(d_mid, grad,
            color=cmap(i / max(1, len(D_MODELS)-1)),
            lw=1.6, label=f"$d_{{\\mathrm{{model}}}}={dm}$")

ax.set_xlabel(r"$d_{\mathrm{ff}}$")
ax.set_ylabel(r"$\partial \psi_{\mathrm{MP}} / \partial d_{\mathrm{ff}}$")
ax.set_title("Marginal gain of $\\psi_{\\mathrm{MP}}$ in $d_{\\mathrm{ff}}$")
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.legend(fontsize=8, ncol=2, loc="upper right", frameon=False)
plt.tight_layout()
out = Path(os.environ.get("NSC_OUT", "outputs")) / "psi_concavity.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, bbox_inches="tight")
print(f"saved {out}")

# Also print a concavity check (all 2nd diffs negative?)
print("\nSecond-diff sign check:")
for dm in D_MODELS:
    psi_vals = np.array([psi_mp(d, dm, SIGMA) for d in D_FF])
    d2 = np.diff(psi_vals, n=2)
    n_pos = int((d2 > 0).sum())
    print(f"  d_model={dm:>4d}: n_pairs={len(d2)}, n_2nd_diff>0 = {n_pos}, max 2nd diff = {d2.max():.3e}")
