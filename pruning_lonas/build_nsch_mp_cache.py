#!/usr/bin/env python3
"""Build the closed-form NSC cache (psi_MP, Marchenko-Pastur) for the LoNAS LLaMA-7B search space.

    psi_MP(m, n, s) = N * integrate(ln(1 + M*s^2*lambda) * f_MP(lambda; gamma), lm, lp)
    where M = max(m,n), N = min(m,n), gamma = N/M.

NSC = sum of psi_MP over all weight matrices, with per-head decomposition for
attention (each head is treated as [d_h, d]) and per-matrix scoring for FFN:
  - attention q/k/v at LoRA rank r: per-head base + LoRA-A [r,d] + LoRA-B [d,r]
  - attention o (fixed rank): per-head base + LoRA-A + LoRA-B
  - ffn gate/up at width k: base [k,d] + LoRA-A [r,d] + LoRA-B [k,r]
  - ffn down at width k:    base [d,k] + LoRA-A [r,k] + LoRA-B [d,r]

Xavier variance: s^2 = 2 / (m + n) per matrix.
"""
import os
import json
import math
from pathlib import Path

import numpy as np
from scipy import integrate

N_LAYERS = 32
HIDDEN = 4096
NUM_HEADS = 32
HEAD_DIM = HIDDEN // NUM_HEADS  # 128
RANK_CANDIDATES = [32, 28]
FFN_CANDIDATES = [11008, 9632, 8256, 6880, 5504]
LORA_FIXED = 32  # fixed LoRA rank for o_proj and FFN

OUT = Path(os.path.join(os.environ.get("NSC_OUT", "outputs"), "nsch_mp_cache.json"))

_PSI_MP_CACHE: dict[tuple[int, int], float] = {}


def psi_mp(m: int, n: int, sigma2: float | None = None) -> float:
    """Closed-form Marchenko-Pastur spectral capacity ."""
    if m == 0 or n == 0:
        return 0.0
    if m < n:
        m, n = n, m
    s2 = 2.0 / (m + n) if sigma2 is None else sigma2
    key = (m, n, round(s2, 12))
    if key in _PSI_MP_CACHE:
        return _PSI_MP_CACHE[key]
    gamma = n / m
    lp = (1.0 + math.sqrt(gamma)) ** 2
    lm = (1.0 - math.sqrt(gamma)) ** 2
    Ms2 = m * s2

    def integrand(lam: float) -> float:
        if lam <= lm or lam >= lp:
            return 0.0
        density = math.sqrt((lp - lam) * (lam - lm)) / (2.0 * math.pi * gamma * lam)
        return math.log1p(Ms2 * lam) * density

    val, _ = integrate.quad(integrand, lm, lp, limit=400)
    res = n * val
    _PSI_MP_CACHE[key] = res
    return res


def attn_proj_psi(lora_rank: int) -> float:
    """q/k/v/o_proj: H heads of [d_h, d] base + LoRA-A [r, d] + LoRA-B [d, r]."""
    base = NUM_HEADS * psi_mp(HEAD_DIM, HIDDEN)
    lora_a = psi_mp(lora_rank, HIDDEN)
    lora_b = psi_mp(HIDDEN, lora_rank)
    return base + lora_a + lora_b


def gate_up_psi(ffn_dim: int) -> float:
    """gate/up_proj at width k: base [k, d] + LoRA-A [r, d] + LoRA-B [k, r]."""
    base = psi_mp(ffn_dim, HIDDEN)
    lora_a = psi_mp(LORA_FIXED, HIDDEN)
    lora_b = psi_mp(ffn_dim, LORA_FIXED)
    return base + lora_a + lora_b


def down_psi(ffn_dim: int) -> float:
    """down_proj at width k: base [d, k] + LoRA-A [r, k] + LoRA-B [d, r]."""
    base = psi_mp(HIDDEN, ffn_dim)
    lora_a = psi_mp(LORA_FIXED, ffn_dim)
    lora_b = psi_mp(HIDDEN, LORA_FIXED)
    return base + lora_a + lora_b


def main() -> None:
    cache: dict[str, float] = {}
    for li in range(N_LAYERS):
        for proj in ("q_proj", "k_proj", "v_proj"):
            for r in RANK_CANDIDATES:
                cache[f"({li}, '{proj}', {r})"] = attn_proj_psi(r)
        cache[f"({li}, 'o_proj')"] = attn_proj_psi(LORA_FIXED)
        for proj in ("gate_proj", "up_proj"):
            for k in FFN_CANDIDATES:
                cache[f"({li}, '{proj}', {k})"] = gate_up_psi(k)
        for k in FFN_CANDIDATES:
            cache[f"({li}, 'down_proj', {k})"] = down_psi(k)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(cache, f)

    print(f"Saved {OUT}: {len(cache)} entries", flush=True)
    print(f"Distinct (m,n,s2) tuples integrated: {len(_PSI_MP_CACHE)}", flush=True)
    print("\nSample (m, n) -> psi_MP:")
    for key, val in sorted(_PSI_MP_CACHE.items()):
        m, n, s2 = key
        print(f"  m={m:5d}  n={n:5d}  s^2={s2:.5e}  psi_MP={val:.4f}")
    print("\nSample (layer 0) cache values:")
    for proj in ("q_proj", "k_proj", "v_proj"):
        for r in RANK_CANDIDATES:
            k_str = f"(0, '{proj}', {r})"
            print(f"  {k_str} = {cache[k_str]:.4f}")
    k_str = "(0, 'o_proj')"
    print(f"  {k_str} = {cache[k_str]:.4f}")
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for k in [FFN_CANDIDATES[0], FFN_CANDIDATES[-1]]:
            k_str = f"(0, '{proj}', {k})"
            print(f"  {k_str} = {cache[k_str]:.4f}")


if __name__ == "__main__":
    main()
