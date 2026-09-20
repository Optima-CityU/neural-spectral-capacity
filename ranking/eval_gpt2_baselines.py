#!/usr/bin/env python3
"""
Compute W-PCA, ZeroLM, SNIP, and LPZero for GPT-2 benchmark.
Model aligned with LPZero/ZeroLM's GPT2LMHeadModelFlex (HF GPT-2 style):
  - Attention: c_attn (QKV merged) + c_proj, NO r_net
  - Position: absolute position embedding (wpe)
  - Init: N(0, 0.02), c_proj scaled by 1/sqrt(2*n_layer)
"""
import json, os, sys, time, gc, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.environ.get("NSC_ROOT", ".")

# ═══════════════════════════════════════════════
# GPT-2 Flex model (aligned with LPZero/ZeroLM)
# ═══════════════════════════════════════════════

class GPT2Attention(nn.Module):
    """Matches GPT2AttentionFlex: c_attn(d, 3d) + c_proj(d, d), no r_net."""
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(causal == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class GPT2MLP(nn.Module):
    """Matches GPT2MLPFlex: c_fc(d, d_inner) + c_proj(d_inner, d)."""
    def __init__(self, d_model, d_inner):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_inner)
        self.c_proj = nn.Linear(d_inner, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class GPT2Block(nn.Module):
    def __init__(self, d_model, n_head, d_inner):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = GPT2Attention(d_model, n_head)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = GPT2MLP(d_model, d_inner)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Flex(nn.Module):
    """Lightweight GPT-2 Flex aligned with GPT2LMHeadModelFlex structure.
    
    Key differences from LightTXL:
    - No r_net (absolute position embedding instead of relative)
    - c_attn merges QKV into single Linear(d, 3d)
    - Initialization: N(0, 0.02), c_proj scaled by 1/sqrt(2*n_layer)
    """
    def __init__(self, n_token, d_model, d_embed, n_layer, layers_cfg, 
                 max_pos=192, init_std=0.02):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.wte = nn.Embedding(n_token, d_model)
        self.wpe = nn.Embedding(max_pos, d_model)
        self.blocks = nn.ModuleList()
        for cfg in layers_cfg:
            self.blocks.append(GPT2Block(d_model, cfg['n_head'], cfg['d_inner']))
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, n_token, bias=False)

        self._init_weights(init_std)

    def _init_weights(self, std):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        for block in self.blocks:
            block.attn.c_proj.weight.data.normal_(
                mean=0.0, std=std / math.sqrt(2 * self.n_layer))
            block.mlp.c_proj.weight.data.normal_(
                mean=0.0, std=std / math.sqrt(2 * self.n_layer))

    def forward(self, input_ids):
        B, T = input_ids.size()
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.wte(input_ids) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def build_model(arch_cfg):
    d_model = arch_cfg['d_model']
    n_layer = arch_cfg['n_layer']
    n_token = arch_cfg['n_token']
    tgt_len = arch_cfg.get('tgt_len', 192)
    init_std = arch_cfg.get('weight_init_std', 0.02)

    n_head_list = arch_cfg['n_head']
    if isinstance(n_head_list, int):
        n_head_list = [n_head_list] * n_layer
    d_inner_list = arch_cfg.get('d_inner', arch_cfg.get('d_inner_list', []))
    if isinstance(d_inner_list, int):
        d_inner_list = [d_inner_list] * n_layer

    layers_cfg = []
    for i in range(n_layer):
        nh = n_head_list[i] if i < len(n_head_list) else n_head_list[-1]
        di = d_inner_list[i] if i < len(d_inner_list) else d_inner_list[-1]
        layers_cfg.append({'n_head': nh, 'd_inner': di})

    d_embed = arch_cfg.get('d_embed', d_model)
    return GPT2Flex(n_token, d_model, d_embed, n_layer, layers_cfg,
                    max_pos=tgt_len, init_std=init_std)


# ═══════════════════════════════════════════════
# W-PCA (hook FFN c_fc activations, SVD for PCA dims)
# ═══════════════════════════════════════════════

def compute_wpca(model, seq_len=192, batch_size=4, threshold=0.99):
    model.eval()
    n_token = model.lm_head.weight.size(0)
    torch.manual_seed(0)
    data = torch.randint(0, n_token, (batch_size, seq_len), device=DEVICE)

    activations = []
    hooks = []
    def make_hook(idx):
        def hook_fn(m, inp, out):
            activations.append(out.detach())
        return hook_fn

    for i, block in enumerate(model.blocks):
        hooks.append(block.mlp.c_fc.register_forward_hook(make_hook(i)))

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
    return pca_sum, float(n_params) * pca_sum


# ═══════════════════════════════════════════════
# ZeroLM: S = Σ_layer (α·S_attn + (1-α)·S_ffn)
# S_x = ||W||²_F / min(m,n) on actual initialized weights
# Attn weights: c_attn + c_proj  (matches GPT2AttentionFlex)
# FFN weights:  c_fc + c_proj    (matches GPT2MLPFlex)
# ═══════════════════════════════════════════════

def compute_zerolm_parts(model):
    """Return (s_attn_total, s_ffn_total) for later alpha combination."""
    s_attn_total = 0.0
    s_ffn_total = 0.0
    for block in model.blocks:
        s_attn = 0.0
        for mod in [block.attn.c_attn, block.attn.c_proj]:
            W = mod.weight.data.float()
            m, n = W.shape
            s_attn += (W ** 2).sum().item() / min(m, n)
        s_attn_total += s_attn

        s_ffn = 0.0
        for mod in [block.mlp.c_fc, block.mlp.c_proj]:
            W = mod.weight.data.float()
            m, n = W.shape
            s_ffn += (W ** 2).sum().item() / min(m, n)
        s_ffn_total += s_ffn

    return s_attn_total, s_ffn_total


# ═══════════════════════════════════════════════
# SNIP
# ═══════════════════════════════════════════════

def compute_snip(model, seq_len=192, batch_size=4):
    model.train()
    model.zero_grad()
    for p in model.parameters():
        p.requires_grad_(True)

    n_token = model.lm_head.weight.size(0)
    data = torch.randint(0, n_token, (batch_size, seq_len), device=DEVICE)
    target = torch.randint(0, n_token, (batch_size, seq_len), device=DEVICE)

    logits = model(data)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
    loss.backward()

    score = 0.0
    for p in model.parameters():
        if p.grad is not None:
            score += (p.data * p.grad.data).abs().sum().item()
    model.zero_grad()
    return score


# ═══════════════════════════════════════════════
# LPZero (searched proxy from EMNLP 2024)
# genotype = {input: [weight, grad], ops: [[3,10],[11,3], 0]}
# Iterates ALL Linear layers (matching LPZero's Conv1D/Linear traversal)
# ═══════════════════════════════════════════════

def compute_lpzero(model, seq_len=192, batch_size=4):
    model.train()
    model.zero_grad()
    for p in model.parameters():
        p.requires_grad_(True)

    n_token = model.lm_head.weight.size(0)
    data = torch.randint(0, n_token, (batch_size, seq_len), device=DEVICE)
    target = torch.randint(0, n_token, (batch_size, seq_len), device=DEVICE)

    logits = model(data)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
    loss.backward()

    total_score = 0.0
    for block in model.blocks:
        layer_score = 0.0
        linears = [block.attn.c_attn, block.attn.c_proj,
                    block.mlp.c_fc, block.mlp.c_proj]
        for lin in linears:
            W = lin.weight.data.float()
            G = lin.weight.grad.float() if lin.weight.grad is not None else torch.zeros_like(W)

            left = torch.pow(W, 2)
            left = torch.sum(torch.abs(left)) / (left.numel() + 1e-9)

            right = F.softmax(G.flatten(), dim=0)
            right = torch.pow(right, 2).sum()

            layer_score += (left + right).item()

        total_score += layer_score

    model.zero_grad()
    return total_score


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    bm_path = os.path.join(ROOT, "paper/dataset/gpt2_txl/gpt2_benchmark.json")
    with open(bm_path) as f:
        benchmark = json.load(f)

    scores_path = os.path.join(ROOT, "paper/dataset/gpt2_txl/gpt2_scores.json")
    with open(scores_path) as f:
        existing = json.load(f)

    key_to_idx = {r['arch_key']: i for i, r in enumerate(existing)}

    N = len(benchmark)
    print(f"Computing W-PCA + ZeroLM + SNIP + LPZero for {N} GPT-2 archs on {DEVICE}")
    print(f"Model: GPT2Flex (aligned with LPZero/ZeroLM GPT2LMHeadModelFlex)")

    times = {m: 0.0 for m in ['wpca', 'zerolm', 'snip', 'lpzero']}
    all_attn, all_ffn = [], []
    t0 = time.time()

    for idx, (key, arch) in enumerate(benchmark.items()):
        cfg = arch['config']
        tgt_len = cfg.get('tgt_len', 192)

        torch.manual_seed(42)
        model = build_model(cfg).to(DEVICE)

        # ZeroLM: collect attn/ffn parts (no forward/backward needed)
        t_z = time.time()
        sa, sf = compute_zerolm_parts(model)
        all_attn.append(sa)
        all_ffn.append(sf)
        times['zerolm'] += time.time() - t_z

        # W-PCA
        t_w = time.time()
        pca_sum, wpca_val = compute_wpca(model, seq_len=tgt_len, batch_size=4)
        times['wpca'] += time.time() - t_w

        # SNIP
        torch.manual_seed(42 + idx)
        t_s = time.time()
        snip_val = compute_snip(model, seq_len=tgt_len, batch_size=4)
        times['snip'] += time.time() - t_s

        # LPZero
        torch.manual_seed(42 + idx)
        t_l = time.time()
        lpzero_val = compute_lpzero(model, seq_len=tgt_len, batch_size=4)
        times['lpzero'] += time.time() - t_l

        if key in key_to_idx:
            ri = key_to_idx[key]
            existing[ri]['wpca'] = wpca_val
            existing[ri]['pca'] = pca_sum
            existing[ri]['zerolm_attn'] = sa
            existing[ri]['zerolm_ffn'] = sf
            existing[ri]['snip'] = snip_val
            existing[ri]['lpzero'] = lpzero_val

        del model
        torch.cuda.empty_cache()
        gc.collect()

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{N}] {elapsed:.1f}s ({elapsed/(idx+1):.2f}s/arch)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/N:.2f}s/arch)")
    print(f"\nPer-method total time:")
    for m, t in times.items():
        print(f"  {m:10s}: {t:.1f}s ({t/N:.3f}s/arch)")

    # ZeroLM α sweep: [-1.5, 1.5] step 0.1 (paper Algorithm 1)
    neg_ppl = [r['neg_ppl'] for r in existing]
    all_attn = np.array(all_attn)
    all_ffn = np.array(all_ffn)
    neg_ppl_arr = np.array(neg_ppl)

    print(f"\nZeroLM α sweep [-1.5, 3.0]:")
    results = []
    for a10 in range(-15, 31):
        a = a10 / 10.0
        proxy = a * all_attn + (1 - a) * all_ffn
        tau, _ = kendalltau(proxy, neg_ppl_arr)
        rho, _ = spearmanr(proxy, neg_ppl_arr)
        results.append((a, tau, rho))
    results.sort(key=lambda x: -abs(x[1]))
    for a, t, r in results[:10]:
        print(f"  α={a:5.1f}: τ={t:.4f}, ρ={r:.4f}")

    top1_a, top2_a = results[0][0], results[1][0]
    alpha_star = (top1_a + top2_a) / 2
    proxy_star = alpha_star * all_attn + (1 - alpha_star) * all_ffn
    tau_star, _ = kendalltau(proxy_star, neg_ppl_arr)
    rho_star, _ = spearmanr(proxy_star, neg_ppl_arr)
    print(f"  α* = ({top1_a}+{top2_a})/2 = {alpha_star:.2f} → τ={tau_star:.4f}, ρ={rho_star:.4f}")

    for i, r in enumerate(existing):
        r['zerolm'] = float(alpha_star * all_attn[i] + (1 - alpha_star) * all_ffn[i])

    # Final correlations
    print(f"\nCorrelations (N={len(existing)}):")
    for mname, mkey in [('NSC-MP', 'nsc_mp'), ('#Params', 'n_params'),
                        ('SNIP', 'snip'), ('W-PCA', 'wpca'),
                        ('ZeroLM', 'zerolm'), ('LPZero', 'lpzero')]:
        vals = [r.get(mkey, 0) for r in existing]
        tau, _ = kendalltau(vals, neg_ppl)
        rho, _ = spearmanr(vals, neg_ppl)
        print(f"  {mname:12s}: τ={tau:.4f}, ρ={rho:.4f}")

    with open(scores_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved to {scores_path}")


if __name__ == '__main__':
    main()
