#!/usr/bin/env python3
"""
Compute W-PCA, ZeroLM, SNIP for AutoFormer-Tiny (1001 architectures).
"""
import json, os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.environ.get("NSC_ROOT", ".")


# ═══════════════════════════════════════════════
# AutoFormer ViT model
# ═══════════════════════════════════════════════

HEAD_DIM = 64  # AutoFormer uses fixed head_dim=64

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = HEAD_DIM
        self.inner_dim = num_heads * HEAD_DIM
        self.scale = HEAD_DIM ** -0.5
        self.qkv = nn.Linear(dim, self.inner_dim * 3)
        self.proj = nn.Linear(self.inner_dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.inner_dim)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AutoFormerViT(nn.Module):
    def __init__(self, layer_num, embed_dims, num_heads_list, mlp_ratios,
                 img_size=224, patch_size=16, num_classes=1000, init_std=0.02):
        super().__init__()
        self.embed_dim = embed_dims[0]  # all layers same embed_dim in Small
        self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=patch_size,
                                     stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

        self.blocks = nn.ModuleList()
        for i in range(layer_num):
            dim = embed_dims[i] if isinstance(embed_dims, list) else embed_dims
            nh = num_heads_list[i] if isinstance(num_heads_list, list) else num_heads_list
            mr = mlp_ratios[i] if isinstance(mlp_ratios, list) else mlp_ratios
            self.blocks.append(Block(dim, nh, mr))

        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes)

        self._init_weights(init_std)

    def _init_weights(self, std=0.02):
        nn.init.trunc_normal_(self.pos_embed, std=std)
        nn.init.trunc_normal_(self.cls_token, std=std)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def build_model(arch):
    ns = arch["net_setting"]
    return AutoFormerViT(
        layer_num=ns["layer_num"],
        embed_dims=ns["embed_dim"],
        num_heads_list=ns["num_heads"],
        mlp_ratios=ns["mlp_ratio"],
    )


# ═══════════════════════════════════════════════
# W-PCA: hook FFN fc1, SVD for effective PCA dim
# ═══════════════════════════════════════════════

def compute_wpca(model, img_size=224, batch_size=2, threshold=0.99):
    model.eval()
    torch.manual_seed(0)
    data = torch.randn(batch_size, 3, img_size, img_size, device=DEVICE)

    activations = []
    hooks = []
    def make_hook(idx):
        def hook_fn(m, inp, out):
            activations.append(out.detach())
        return hook_fn

    for i, blk in enumerate(model.blocks):
        hooks.append(blk.mlp.fc1.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        model(data)
    for h in hooks:
        h.remove()

    pca_sum = 0.0
    for act in activations:
        act_2d = act.reshape(-1, act.size(-1)).float()
        act_2d = act_2d - act_2d.mean(dim=0, keepdim=True)
        try:
            _, s, _ = torch.linalg.svd(act_2d, full_matrices=False)
            cumvar = torch.cumsum(s ** 2, 0) / (s ** 2).sum()
            pca_dim = (cumvar < threshold).sum().item() + 1
        except Exception:
            pca_dim = act_2d.size(1)
        pca_sum += pca_dim

    n_params = sum(p.numel() for p in model.parameters())
    return float(n_params) * pca_sum


# ═══════════════════════════════════════════════
# ZeroLM: S = α·S_attn + (1-α)·S_ffn
# S_x = Σ_layer Σ_weight ||W||²_F / min(m,n)
# ═══════════════════════════════════════════════

def compute_zerolm_parts(model):
    s_attn_total = 0.0
    s_ffn_total = 0.0
    for blk in model.blocks:
        s_attn = 0.0
        for mod in [blk.attn.qkv, blk.attn.proj]:
            W = mod.weight.data.float()
            m, n = W.shape
            s_attn += (W ** 2).sum().item() / min(m, n)
        s_attn_total += s_attn

        s_ffn = 0.0
        for mod in [blk.mlp.fc1, blk.mlp.fc2]:
            W = mod.weight.data.float()
            m, n = W.shape
            s_ffn += (W ** 2).sum().item() / min(m, n)
        s_ffn_total += s_ffn

    return s_attn_total, s_ffn_total


# ═══════════════════════════════════════════════
# SNIP: |W * grad| summed over all params
# ═══════════════════════════════════════════════

def compute_snip(model, img_size=224, batch_size=2):
    model.train()
    model.zero_grad()
    for p in model.parameters():
        p.requires_grad_(True)

    data = torch.randn(batch_size, 3, img_size, img_size, device=DEVICE)
    target = torch.randint(0, 1000, (batch_size,), device=DEVICE)

    logits = model(data)
    loss = F.cross_entropy(logits, target)
    loss.backward()

    score = 0.0
    for p in model.parameters():
        if p.grad is not None:
            score += (p.data * p.grad.data).abs().sum().item()
    model.zero_grad()
    return score


# ═══════════════════════════════════════════════
# NSC-MP (harmonic aggregation)
# ═══════════════════════════════════════════════

from nsc_utils import psi_mp


def compute_nsc_mp(arch):
    ns = arch["net_setting"]
    depth = ns["layer_num"]
    ed_list = ns["embed_dim"]
    nh_list = ns["num_heads"]
    mr_list = ns["mlp_ratio"]

    nsc_flat = 0.0
    nsc_harmonic = 0.0
    for i in range(depth):
        ed = ed_list[i] if isinstance(ed_list, list) else ed_list
        nh = nh_list[i] if isinstance(nh_list, list) else nh_list
        mr = mr_list[i] if isinstance(mr_list, list) else mr_list
        hd = HEAD_DIM
        inner = nh * hd
        mlp_dim = int(ed * mr)

        # qkv: Linear(ed, inner*3) -> treat as 3*nh separate (ed, hd) projections
        psi_qkv = 3 * nh * psi_mp(ed, hd)
        # proj: Linear(inner, ed)
        psi_proj = psi_mp(ed, inner)
        psi_fc1 = psi_mp(mlp_dim, ed)
        psi_fc2 = psi_mp(ed, mlp_dim)

        layer_flat = psi_qkv + psi_proj + psi_fc1 + psi_fc2
        nsc_flat += layer_flat

        psis = []
        for _ in range(3 * nh):
            psis.append(psi_mp(ed, hd))
        psis.append(psi_proj)
        psis.append(psi_fc1)
        psis.append(psi_fc2)

        psis_pos = [p for p in psis if p > 0]
        if psis_pos:
            harmonic = len(psis_pos) / sum(1.0 / p for p in psis_pos)
        else:
            harmonic = 0.0
        nsc_harmonic += harmonic

    return nsc_flat, nsc_harmonic


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    data_path = os.path.join(ROOT, "paper/dataset/autoformer/data/autoformer_tiny_1k.json")
    with open(data_path) as f:
        bench = json.load(f)
    print(f"Loaded {len(bench)} Tiny architectures")

    out_path = os.path.join(os.environ.get("NSC_OUT", "outputs"), "autoformer_tiny_scores.json")

    results = []
    n = len(bench)
    torch.manual_seed(42)

    t_wpca, t_zerolm, t_snip, t_nsc = 0, 0, 0, 0
    total_start = time.time()

    for i, idx_str in enumerate(sorted(bench.keys(), key=int)):
        arch = bench[idx_str]
        idx = int(idx_str)
        acc = arch["performance"]["Imagenet"]["clean"]

        t0 = time.time()
        nsc_flat, nsc_harmonic = compute_nsc_mp(arch)
        t_nsc += time.time() - t0

        model = build_model(arch).to(DEVICE)

        t0 = time.time()
        wpca = compute_wpca(model)
        t_wpca += time.time() - t0

        t0 = time.time()
        s_attn, s_ffn = compute_zerolm_parts(model)
        t_zerolm += time.time() - t0

        t0 = time.time()
        snip = compute_snip(model)
        t_snip += time.time() - t0

        n_params = sum(p.numel() for p in model.parameters())

        row = {
            'idx': idx,
            'scale': 'Tiny',
            'acc': acc,
            'params': arch['params'],
            'n_params': n_params,
            'nsc_mp': nsc_flat,
            'nsc_mp_harmonic': nsc_harmonic,
            'wpca': wpca,
            'zerolm_attn': s_attn,
            'zerolm_ffn': s_ffn,
            'snip': snip,
        }
        results.append(row)

        del model
        torch.cuda.empty_cache()

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - total_start
            eta = elapsed / (i + 1) * (n - i - 1) if i > 0 else 0
            print(f"[{i+1:4d}/{n}] idx={idx} acc={acc:.2f} "
                  f"nsc={nsc_flat:.0f} wpca={wpca:.2e} snip={snip:.1f} "
                  f"[{elapsed:.0f}s, ETA {eta:.0f}s]", flush=True)

    print(f"\n=== Timing ===")
    print(f"  NSC-MP : {t_nsc:.2f}s ({t_nsc/n*1000:.1f}ms/arch)")
    print(f"  W-PCA  : {t_wpca:.2f}s ({t_wpca/n*1000:.1f}ms/arch)")
    print(f"  ZeroLM : {t_zerolm:.2f}s ({t_zerolm/n*1000:.1f}ms/arch)")
    print(f"  SNIP   : {t_snip:.2f}s ({t_snip/n*1000:.1f}ms/arch)")

    # ZeroLM: use alpha=1.0 (same as Small)
    for r in results:
        r['zerolm'] = r['zerolm_attn']

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} entries to {out_path}")

    # Print correlations
    accs = [r['acc'] for r in results]
    print("\n=== Correlations (Tiny, N=1001) ===")
    methods = {
        'NSC-MP flat': [r['nsc_mp'] for r in results],
        'NSC-MP harmonic': [r['nsc_mp_harmonic'] for r in results],
        'W-PCA': [r['wpca'] for r in results],
        'ZeroLM (a=1)': [r['zerolm'] for r in results],
        'SNIP': [r['snip'] for r in results],
        '#Params': [r['params'] for r in results],
    }

    # Also sweep alpha for reference
    attns = np.array([r['zerolm_attn'] for r in results])
    ffns = np.array([r['zerolm_ffn'] for r in results])
    accs_np = np.array(accs)
    best_alpha, best_tau = 0, -1
    for alpha_10x in range(-15, 31):
        alpha = alpha_10x / 10.0
        scores = alpha * attns + (1 - alpha) * ffns
        tau, _ = kendalltau(scores, accs_np)
        if tau > best_tau:
            best_tau = tau
            best_alpha = alpha
    print(f"\n  ZeroLM alpha sweep: best alpha={best_alpha:.1f}, tau={best_tau:.4f}")

    for name, vals in methods.items():
        tau, _ = kendalltau(vals, accs)
        rho, _ = spearmanr(vals, accs)
        print(f"  {name:20s}: tau={tau:.4f}, rho={rho:.4f}")


if __name__ == "__main__":
    main()
