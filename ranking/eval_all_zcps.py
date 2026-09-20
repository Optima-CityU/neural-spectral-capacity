#!/usr/bin/env python3
"""Zero-cost proxies on the FlexiBERT benchmark.

Methods:
  Gradient-based:    SNIP, GraSP, SynFlow, LogSynflow, GradNorm, Fisher, Jacobian cosine
  Activation-based:  NASWOT, synaptic diversity, activation distance
  Attention-based:   head importance, attention confidence
  Structural/PCA:    #Params, PCA, W-PCA
  NSC variants:      NSC-W (weight-based), NSC-Jd (activation-gated)

Outputs per-architecture scores (JSON) and correlation tables.
"""
import os, sys, json, time, math, gc
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
from collections import defaultdict

sys.path.insert(0, os.environ.get("FLEXIBERT_DIR", "FlexiBERT"))   # ELECTRA modeling code
from configuration_electra import ElectraConfig
from modeling_electra import ElectraModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════
#  Core primitives
# ═══════════════════════════════════════════════════════════════

def psi_val(M):
    if M.dim() != 2 or min(M.shape) == 0:
        return 0.0
    try:
        s = torch.linalg.svdvals(M.float().to(DEVICE))
        return torch.log(s + 1).sum().item()
    except Exception:
        return 0.0


def activation_derivative(z, act_fn):
    z_var = z.detach().clone().requires_grad_(True)
    a = act_fn(z_var)
    g = torch.autograd.grad(a.sum(), z_var, create_graph=False)[0]
    return g.detach()


# ═══════════════════════════════════════════════════════════════
#  GraSP  (Wang et al., ICLR 2020)
#  Score = -tr(H · θ⊙∇θL) ≈ -(∇θL)^T · Hv  where v = θ⊙∇θL
#  Implemented via two backward passes.
# ═══════════════════════════════════════════════════════════════

def compute_grasp(model, head, cfg):
    BS, SL = 4, 64
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (BS, SL), device=DEVICE)
    labels = torch.randint(0, cfg.vocab_size, (BS, SL), device=DEVICE)

    model.zero_grad()
    head.zero_grad()
    model.train()

    # First forward-backward: get gradients
    out = model(input_ids)
    logits = head(out.last_hidden_state)
    loss = nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), labels.view(-1))
    loss.backward(create_graph=True)

    # Collect θ⊙∇θ as the vector v for Hessian-vector product
    grads = []
    params_with_grad = []
    for p in list(model.parameters()) + list(head.parameters()):
        if p.grad is not None and p.dim() >= 2:
            grads.append((p.grad * p.data).flatten())
            params_with_grad.append(p)

    if not grads:
        model.zero_grad()
        head.zero_grad()
        return 0.0

    v = torch.cat(grads)

    # Hessian-vector product: Hv = d/dθ (∇θL · v)
    flat_grads = torch.cat([p.grad.flatten() for p in params_with_grad])
    hvp = torch.autograd.grad(flat_grads, params_with_grad,
                              grad_outputs=v.split([p.numel() for p in params_with_grad]),
                              retain_graph=False, allow_unused=True)

    # GraSP score = - Σ (θ⊙∇θ) · Hv
    score = 0.0
    for p, hv in zip(params_with_grad, hvp):
        if hv is not None:
            score -= ((p.data * p.grad.data).flatten() * hv.flatten()).sum().item()

    model.zero_grad()
    head.zero_grad()
    return score


# ═══════════════════════════════════════════════════════════════
#  NASWOT  (Mellor et al., ICML 2021)
#  Score = log|K| where K_ij = Hamming similarity of activation patterns
#  Adapted for GELU: threshold at 0 (positive = active)
# ═══════════════════════════════════════════════════════════════

def compute_naswot(model, cfg):
    N_SAMPLES = 32
    SL = 64
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (N_SAMPLES, SL), device=DEVICE)

    # Collect post-activation binary patterns via hooks
    activations = []
    hooks = []

    def make_hook(store_list):
        def hook_fn(m, inp, out):
            t = out if isinstance(out, torch.Tensor) else out[0]
            # Binary pattern: active (>0) vs inactive (<=0)
            store_list.append((t.detach() > 0).float())
        return hook_fn

    for module in model.modules():
        if isinstance(module, (nn.GELU, nn.SiLU, nn.ReLU)):
            hooks.append(module.register_forward_hook(make_hook(activations)))

    # If no activation modules found, hook after intermediate dense layers
    if not hooks:
        for li, blk in enumerate(model.encoder.layer):
            hooks.append(blk.intermediate.dense_in.register_forward_hook(make_hook(activations)))

    model.eval()
    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    if not activations:
        return 0.0

    # Build binary code per sample: concatenate all activation patterns
    codes = []
    for act in activations:
        # act: [N_SAMPLES, SL, hidden_dim] -> flatten per sample
        codes.append(act.reshape(N_SAMPLES, -1))
    codes = torch.cat(codes, dim=1)  # [N_SAMPLES, total_neurons]

    # Kernel matrix: K_ij = fraction of neurons with same activation
    # K_ij = (codes_i · codes_j + (1-codes_i)·(1-codes_j)) / total_neurons
    # Simplified: K_ij = 1 - hamming_distance(i,j)
    K = codes @ codes.T + (1 - codes) @ (1 - codes).T
    K = K / codes.shape[1]

    # Numerical stability: add small diagonal
    K = K + 1e-5 * torch.eye(N_SAMPLES, device=DEVICE)

    try:
        score = torch.logdet(K).item()
    except Exception:
        score = 0.0

    if not math.isfinite(score):
        score = 0.0

    return score


# ═══════════════════════════════════════════════════════════════
#  Synaptic Diversity  (W-PCA baseline)
#  Binary activation pattern diversity across samples.
#  Score = log(rank of binary activation matrix)
# ═══════════════════════════════════════════════════════════════

def compute_synaptic_diversity(model, cfg):
    N_SAMPLES = 32
    SL = 64
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (N_SAMPLES, SL), device=DEVICE)

    activations = []
    hooks = []

    def make_hook(lst):
        def fn(m, inp, out):
            t = out if isinstance(out, torch.Tensor) else out[0]
            lst.append((t.detach() > 0).float().reshape(N_SAMPLES, -1))
        return fn

    for blk in model.encoder.layer:
        hooks.append(blk.intermediate.dense_in.register_forward_hook(make_hook(activations)))

    model.eval()
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()

    if not activations:
        return 0.0

    codes = torch.cat(activations, dim=1)  # [N_SAMPLES, total_neurons]
    try:
        s = torch.linalg.svdvals(codes.float())
        n_effective = (s > 1e-5).sum().item()
        return n_effective / N_SAMPLES
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════
#  Activation Distance  (W-PCA baseline)
#  Average pairwise L2 distance of hidden representations.
# ═══════════════════════════════════════════════════════════════

def compute_activation_distance(model, cfg):
    N_SAMPLES = 16
    SL = 64
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (N_SAMPLES, SL), device=DEVICE)

    model.eval()
    with torch.no_grad():
        out = model(input_ids)
    h = out.last_hidden_state.mean(dim=1)  # [N_SAMPLES, hidden_size]

    dists = torch.cdist(h, h, p=2)
    mask = torch.triu(torch.ones(N_SAMPLES, N_SAMPLES, device=DEVICE), diagonal=1).bool()
    return dists[mask].mean().item()


# ═══════════════════════════════════════════════════════════════
#  Head Importance  (Michel et al., 2019)
#  I_h = |Att_h^T · ∂L/∂Att_h|   (per-head saliency)
#  Total = Σ over all heads in all layers
#  Only SA/DSC layers have attention heads; LT layers are skipped.
# ═══════════════════════════════════════════════════════════════

def compute_head_importance(model, head, cfg):
    BS, SL = 4, 64
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (BS, SL), device=DEVICE)
    labels = torch.randint(0, cfg.vocab_size, (BS, SL), device=DEVICE)

    attn_outputs = []
    n_heads_list = []
    hooks = []

    def make_hook(lst):
        def fn(m, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            t.retain_grad()
            lst.append(t)
        return fn

    layers_cfg = cfg.nas_config['encoder_layers']
    for li, blk in enumerate(model.encoder.layer):
        op_type = layers_cfg[li].get('operation_type', '')
        if op_type in ('SA', 'DSC'):
            attn_op = blk.operation
            if hasattr(attn_op, 'operation'):
                hooks.append(attn_op.operation.register_forward_hook(make_hook(attn_outputs)))
                n_heads_list.append(layers_cfg[li]['num_operation_heads'])

    model.train()
    model.zero_grad()
    head.zero_grad()
    out = model(input_ids)
    logits = head(out.last_hidden_state)
    loss = nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), labels.view(-1))
    loss.backward()

    for h in hooks:
        h.remove()

    importance = 0.0
    for t, n_h in zip(attn_outputs, n_heads_list):
        if t.grad is None:
            continue
        act = t.float()           # [B, SL, all_head_size]
        grad = t.grad.float()     # [B, SL, all_head_size]
        head_dim = act.shape[-1] // n_h
        act_h = act.view(act.shape[0], act.shape[1], n_h, head_dim)
        grad_h = grad.view(grad.shape[0], grad.shape[1], n_h, head_dim)
        for hi in range(n_h):
            a = act_h[:, :, hi, :]    # [B, SL, head_dim]
            g = grad_h[:, :, hi, :]   # [B, SL, head_dim]
            importance += abs((a * g).sum().item())

    model.zero_grad()
    head.zero_grad()
    return importance


# ═══════════════════════════════════════════════════════════════
#  Head Softmax Confidence  (W-PCA baseline)
#  Average max attention probability across all heads/layers.
# ═══════════════════════════════════════════════════════════════

def compute_head_softmax_confidence(model, cfg):
    BS, SL = 4, 64
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (BS, SL), device=DEVICE)

    attn_weights = []
    hooks = []

    def make_hook(lst):
        def fn(m, inp, out):
            if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                lst.append(out[1].detach())  # attention weights [BS, n_heads, SL, SL]
        return fn

    for blk in model.encoder.layer:
        hooks.append(blk.operation.register_forward_hook(make_hook(attn_weights)))

    model.eval()
    # Try with output_attentions if the model supports it
    try:
        with torch.no_grad():
            out = model(input_ids, output_attentions=True)
        if hasattr(out, 'attentions') and out.attentions is not None:
            attn_weights = list(out.attentions)
    except Exception:
        with torch.no_grad():
            model(input_ids)

    for h in hooks:
        h.remove()

    if not attn_weights:
        # Fallback: manually compute attention from Q, K weights
        return _compute_softmax_confidence_manual(model, cfg, input_ids)

    confidences = []
    for aw in attn_weights:
        # aw: [BS, n_heads, SL, SL]
        max_probs = aw.max(dim=-1).values  # [BS, n_heads, SL]
        confidences.append(max_probs.mean().item())

    return float(np.mean(confidences)) if confidences else 0.0


def _compute_softmax_confidence_manual(model, cfg, input_ids):
    """Fallback: compute Q·K^T softmax confidence manually."""
    model.eval()
    hidden_states = []
    hooks = []

    def make_hook(lst):
        def fn(m, inp, out):
            t = out if isinstance(out, torch.Tensor) else out[0]
            lst.append(t.detach())
        return fn

    # Hook the input to each attention block to get hidden states
    for blk in model.encoder.layer:
        hooks.append(blk.operation.register_forward_hook(make_hook(hidden_states)))

    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()

    # Try to compute attention from Q, K projections
    confidences = []
    for li, blk in enumerate(model.encoder.layer):
        # Get Q, K weight matrices
        qkv_found = False
        for name, param in blk.operation.named_parameters():
            if 'query' in name.lower() and 'weight' in name.lower():
                W_Q = param.data
                qkv_found = True
            elif 'key' in name.lower() and 'weight' in name.lower():
                W_K = param.data

        if not qkv_found:
            continue

        if li < len(hidden_states):
            h = hidden_states[li].float()
            try:
                Q = h @ W_Q.T.float()
                K = h @ W_K.T.float()
                d_k = Q.shape[-1]
                attn_logits = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
                attn_probs = torch.softmax(attn_logits, dim=-1)
                max_probs = attn_probs.max(dim=-1).values
                confidences.append(max_probs.mean().item())
            except Exception:
                pass

    return float(np.mean(confidences)) if confidences else 0.0


# ═══════════════════════════════════════════════════════════════
#  PCA & W-PCA  (Wang et al., ICLR 2025)
#  PCA = Σ_l PCA_dim(H_l, η),  where H_l = X·W1 + b1 in FFN
#  W-PCA = n_params × PCA
#  η = 0.99 (cumulative variance threshold)
# ═══════════════════════════════════════════════════════════════

def _collect_ffn_intermediate_activations(model, cfg):
    """Hook into each ElectraIntermediate.dense_in to capture FFN
    first-linear outputs (before activation), matching W-PCA paper."""
    BS, SL = 4, 64
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (BS, SL), device=DEVICE)

    activations = []
    hooks = []

    def hook_fn(m, inp, out):
        t = out if isinstance(out, torch.Tensor) else out[0]
        activations.append(t.detach())

    for blk in model.encoder.layer:
        hooks.append(blk.intermediate.dense_in.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    return activations


def compute_pca_and_wpca(model, cfg, n_params=None, threshold=0.99):
    """PCA = sum of PCA_dim across FFN layers;  W-PCA = n_params × PCA."""
    activations = _collect_ffn_intermediate_activations(model, cfg)

    pca_sum = 0.0
    for act in activations:
        a = act.reshape(-1, act.shape[-1]).float()
        if a.shape[0] < 2 or a.shape[1] < 2:
            continue
        try:
            a_centered = a - a.mean(dim=0, keepdim=True)
            s = torch.linalg.svdvals(a_centered.to(DEVICE))
            cumvar = torch.cumsum(s ** 2, 0) / (s ** 2).sum()
            pca_dim = (cumvar < threshold).sum().item() + 1
            pca_sum += pca_dim
        except Exception:
            pass

    if n_params is None:
        n_params = sum(p.numel() for p in model.parameters())
    wpca_score = float(n_params) * pca_sum
    return pca_sum, wpca_score


# ═══════════════════════════════════════════════════════════════
#  Jacobian Cosine  (Mellor et al., 2021 variant)
#  Cosine similarity between input-output Jacobians of different
#  mini-batches. Low similarity → high expressivity.
#  Score = -mean(cos_sim), so higher = more diverse = better.
# ═══════════════════════════════════════════════════════════════

def compute_jacobian_cosine(model, head, cfg):
    SL = 32
    torch.manual_seed(0)
    x1 = torch.randint(0, cfg.vocab_size, (2, SL), device=DEVICE)
    x2 = torch.randint(0, cfg.vocab_size, (2, SL), device=DEVICE)

    model.eval()

    def get_jacobian_flat(x):
        model.zero_grad()
        head.zero_grad()
        for p in model.parameters():
            p.requires_grad_(True)
        for p in head.parameters():
            p.requires_grad_(True)
        out = model(x)
        logits = head(out.last_hidden_state)
        scalar = logits.sum()
        scalar.backward()
        grads = []
        for p in list(model.parameters()) + list(head.parameters()):
            if p.grad is not None:
                grads.append(p.grad.data.float().flatten())
        model.zero_grad()
        head.zero_grad()
        if not grads:
            return None
        return torch.cat(grads)

    j1 = get_jacobian_flat(x1)
    j2 = get_jacobian_flat(x2)

    if j1 is None or j2 is None:
        return 0.0

    cos = torch.nn.functional.cosine_similarity(j1.unsqueeze(0), j2.unsqueeze(0)).item()
    return -cos


# ═══════════════════════════════════════════════════════════════
#  NSC-W (weight-based)
# ═══════════════════════════════════════════════════════════════

def compute_nsc_w(model, cfg):
    nsc = 0.0
    for blk in model.encoder.layer:
        ffn_mods = [blk.intermediate.dense_in] + list(blk.intermediate.dense_list) + [blk.output.dense]
        psi_ffn = sum(psi_val(m.weight.data) for m in ffn_mods)
        psi_attn = 0.0
        for name, p in blk.operation.named_parameters():
            if 'LayerNorm' not in name and 'bias' not in name and p.dim() == 2:
                psi_attn += psi_val(p.data)
        nsc += math.log(1 + psi_ffn) + math.log(1 + psi_attn)
    return nsc


# ═══════════════════════════════════════════════════════════════
#  NSC-Jd (activation-gated)
# ═══════════════════════════════════════════════════════════════

def compute_nsc_jd(model, cfg):
    BS, SL = 4, 64
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (BS, SL), device=DEVICE)

    act_fn = nn.GELU()
    for m in model.modules():
        if isinstance(m, (nn.GELU, nn.SiLU, nn.ReLU, nn.Mish)):
            act_fn = m
            break

    pre_acts = {}
    hooks = []
    for li, blk in enumerate(model.encoder.layer):
        def mkhook(idx):
            def fn(m, inp, out):
                pre_acts[idx] = out.detach()
            return fn
        hooks.append(blk.intermediate.dense_in.register_forward_hook(mkhook(li)))

    model.eval()
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()

    nsc = 0.0
    for li, blk in enumerate(model.encoder.layer):
        W_in = blk.intermediate.dense_in
        W_mids = list(blk.intermediate.dense_list)
        W_out = blk.output.dense

        psi_ffn = 0.0
        if li in pre_acts:
            z = pre_acts[li].reshape(-1, pre_acts[li].shape[-1]).float().to(DEVICE)
            g = activation_derivative(z, act_fn).mean(0)
            eff_W = g.unsqueeze(1) * W_in.weight.data.float().to(DEVICE)
            psi_ffn += psi_val(eff_W)
            for Wm in W_mids:
                psi_ffn += psi_val(Wm.weight.data)
        else:
            for mod in [W_in] + W_mids:
                psi_ffn += psi_val(mod.weight.data)
        psi_ffn += psi_val(W_out.weight.data)

        psi_attn = 0.0
        for name, p in blk.operation.named_parameters():
            if 'LayerNorm' not in name and 'bias' not in name and p.dim() == 2:
                psi_attn += psi_val(p.data)
        nsc += math.log(1 + psi_ffn) + math.log(1 + psi_attn)
    return nsc


# ═══════════════════════════════════════════════════════════════
#  Shared forward+backward for gradient-based ZCPs
# ═══════════════════════════════════════════════════════════════

def _do_forward_backward(model, head, cfg):
    BS, SL = 4, 64
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (BS, SL), device=DEVICE)
    labels = torch.randint(0, cfg.vocab_size, (BS, SL), device=DEVICE)
    model.train()
    model.zero_grad()
    head.zero_grad()
    out = model(input_ids)
    logits = head(out.last_hidden_state)
    loss = nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), labels.view(-1))
    loss.backward()


# ═══════════════════════════════════════════════════════════════
#  SNIP  (Lee et al., ICLR 2019)  = Synaptic Saliency
#  Score = Σ |∂L/∂θ ⊙ θ|
# ═══════════════════════════════════════════════════════════════

def compute_snip(model, head, cfg):
    _do_forward_backward(model, head, cfg)
    score = 0.0
    for p in list(model.parameters()) + list(head.parameters()):
        if p.grad is not None and p.dim() >= 2:
            score += (p.grad.data.float() * p.data.float()).abs().sum().item()
    model.zero_grad()
    head.zero_grad()
    return score


# ═══════════════════════════════════════════════════════════════
#  GradNorm  (Abdelfattah et al., 2021)
#  Score = Σ_l ||∇_θ_l L||
# ═══════════════════════════════════════════════════════════════

def compute_gradnorm(model, head, cfg):
    _do_forward_backward(model, head, cfg)
    score = 0.0
    for p in list(model.parameters()) + list(head.parameters()):
        if p.grad is not None and p.dim() >= 2:
            score += p.grad.data.float().norm().item()
    model.zero_grad()
    head.zero_grad()
    return score


# ═══════════════════════════════════════════════════════════════
#  Fisher  (Turner et al., 2020)
#  Score = Σ (∂L/∂θ)²
# ═══════════════════════════════════════════════════════════════

def compute_fisher(model, head, cfg):
    _do_forward_backward(model, head, cfg)
    score = 0.0
    for p in list(model.parameters()) + list(head.parameters()):
        if p.grad is not None and p.dim() >= 2:
            score += (p.grad.data.float() ** 2).sum().item()
    model.zero_grad()
    head.zero_grad()
    return score


# ═══════════════════════════════════════════════════════════════
#  SynFlow  (Tanaka et al., ICML 2020)
#  Score = Σ_l (||W_l||_1)²
# ═══════════════════════════════════════════════════════════════

def compute_synflow(model, cfg):
    score = 0.0
    for n, p in model.named_parameters():
        if p.dim() >= 2 and 'LayerNorm' not in n:
            score += p.data.float().abs().sum().item() ** 2
    return score


# ═══════════════════════════════════════════════════════════════
#  FlexiBERT model builder
# ═══════════════════════════════════════════════════════════════

def build_flexibert(arch):
    """Build a FlexiBERT model from its architecture specification
    (per-arch hidden_size, per-layer FFN dims via nas_config)."""
    hpo = arch["hparams"]["model_hparam_overrides"]
    nas = hpo["nas_config"]
    layers = nas["encoder_layers"]

    hidden = hpo["hidden_size"]
    intermediate = layers[0]["feed_forward_dimension"]

    cfg = ElectraConfig(
        vocab_size=30522, hidden_size=hidden,
        num_hidden_layers=len(layers),
        num_attention_heads=layers[0]["num_operation_heads"],
        intermediate_size=intermediate,
        nas_config=nas,
    )
    torch.manual_seed(42)
    model = ElectraModel(cfg).to(DEVICE)
    head = nn.Linear(hidden, 30522, bias=False).to(DEVICE)
    return model, head, cfg


# ═══════════════════════════════════════════════════════════════
#  Main evaluation loop
# ═══════════════════════════════════════════════════════════════

RESUME_KEY = 'time_jacobian_cosine'

# Each entry: (result_key(s), function, needs_head)
# If result_key is a tuple, the function returns a tuple of the same length.
ZCP_METHODS = [
    ('nsc_w',            compute_nsc_w,                    False),
    ('nsc_jd',           compute_nsc_jd,                   False),
    ('snip',             compute_snip,                     True),
    ('synflow',          compute_synflow,                  False),
    ('gradnorm',         compute_gradnorm,                 True),
    ('fisher',           compute_fisher,                   True),
    ('grasp',            compute_grasp,                    True),
    ('naswot',           compute_naswot,                   False),
    (('pca', 'wpca'),    compute_pca_and_wpca,             False),
    ('syn_diversity',    compute_synaptic_diversity,       False),
    ('act_distance',     compute_activation_distance,      False),
    ('head_importance',  compute_head_importance,          True),
    ('head_softmax_conf', compute_head_softmax_confidence, False),
    ('jacobian_cosine',  compute_jacobian_cosine,          True),
]


def _run_one_setting(bench, setting_name, out_file):
    """Evaluate all ZCPs on the FlexiBERT benchmark."""

    print(f"\n{'#'*70}", flush=True)
    print(f"  SETTING: {setting_name}", flush=True)
    print(f"  →  {out_file.name}", flush=True)
    print(f"{'#'*70}\n", flush=True)

    existing = {}
    if out_file.exists():
        with open(out_file) as f:
            for r in json.load(f):
                existing[r['arch_id']] = r
        print(f"Loaded {len(existing)} existing results", flush=True)

    results = []
    t0 = time.time()
    n_total = len(bench)

    for i, arch in enumerate(bench):
        glue_data = arch.get('metrics', arch.get('scores', {}))
        glue = glue_data.get('glue_avg', glue_data.get('glue'))
        if glue is None:
            continue

        arch_id = arch.get('id', i)

        if arch_id in existing and RESUME_KEY in existing[arch_id]:
            results.append(existing[arch_id])
            continue

        try:
            model_tmp, _, cfg_tmp = build_flexibert(arch)
            r = {
                'arch_id': arch_id,
                'glue': float(glue),
                'n_params': sum(p.numel() for p in model_tmp.parameters()),
                'n_layers': cfg_tmp.num_hidden_layers,
                'hidden_size': cfg_tmp.hidden_size,
            }
            del model_tmp

            for keys, fn, needs_head in ZCP_METHODS:
                model, head, cfg = build_flexibert(arch)
                t_start = time.time()
                if needs_head:
                    val = fn(model, head, cfg)
                else:
                    val = fn(model, cfg)
                elapsed_ms = (time.time() - t_start) * 1000
                del model, head

                if isinstance(keys, tuple):
                    for k, v in zip(keys, val):
                        r[k] = v
                    r[f'time_{"_".join(keys)}'] = elapsed_ms
                else:
                    r[keys] = val
                    r[f'time_{keys}'] = elapsed_ms

            r['log_synflow'] = math.log(r['synflow'] + 1e-30)

            results.append(r)
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"  [{i}] ERR: {e}", flush=True)
            import traceback
            traceback.print_exc()

        if (i + 1) % 25 == 0 or i == n_total - 1:
            elapsed = time.time() - t0
            rate = elapsed / max(len(results), 1)
            eta = rate * (n_total - i - 1)
            print(f"  [{i+1}/{n_total}] {len(results)} ok "
                  f"({elapsed:.0f}s, ~{rate:.1f}s/arch, ETA {eta:.0f}s)", flush=True)
            with open(out_file, 'w') as f:
                json.dump(results, f, indent=2)

    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTotal: {len(results)} archs, {time.time()-t0:.1f}s\n", flush=True)
    return results


def evaluate_all():
    with open(Path(os.environ.get("FLEXIBERT_DIR", "FlexiBERT")) / "BERT_benchmark.json") as f:
        bench = json.load(f)

    out_dir = Path(os.environ.get("NSC_OUT", "outputs"))
    out_dir.mkdir(exist_ok=True)

    results_corrected = _run_one_setting(
        bench, "per-arch h, per-layer ffn",
        out_file=out_dir / "flexibert_corrected.json",
    )
    print("\n" + "="*70)
    print("  RESULTS")
    print("="*70)
    analyze_results(results_corrected)

    return results_corrected


def analyze_results(results):
    if not results:
        return

    glues = np.array([r['glue'] for r in results])

    metric_labels = {
        'n_params':          '#Params',
        'nsc_w':             'NSC-W',
        'nsc_jd':            'NSC-Jd',
        'snip':              'SNIP (Syn.Sal.)',
        'grasp':             'GraSP',
        'synflow':           'SynFlow',
        'log_synflow':       'LogSynflow',
        'gradnorm':          'GradNorm',
        'fisher':            'Fisher',
        'jacobian_cosine':   'Jacobian Cosine',
        'naswot':            'NASWOT',
        'pca':               'PCA',
        'wpca':              'W-PCA',
        'syn_diversity':     'Syn. Diversity',
        'act_distance':      'Act. Distance',
        'head_importance':   'Head Importance',
        'head_softmax_conf': 'Attn. Confidence',
    }

    available = [(k, v) for k, v in metric_labels.items()
                 if k in results[0] and results[0][k] is not None]

    def print_table(name, idxs):
        g = glues[idxs]
        if len(g) < 5:
            return
        print(f"\n{'═'*70}")
        print(f"  {name}  (n={len(g)}, GLUE=[{g.min():.1f}, {g.max():.1f}])")
        print(f"{'═'*70}")
        print(f"  {'Method':<20s} {'Spearman ρ':>12s} {'Kendall τ':>12s} {'p-value':>12s}")
        print(f"  {'─'*56}")
        rows = []
        for key, label in available:
            vals = np.array([float(results[i][key]) for i in idxs])
            ok = np.isfinite(vals) & np.isfinite(g) & (vals != 0)
            if ok.sum() < 5:
                rows.append((label, float('nan'), float('nan'), 1.0))
                continue
            spr, _ = spearmanr(vals[ok], g[ok])
            tau, p_tau = kendalltau(vals[ok], g[ok])
            rows.append((label, spr, tau, p_tau))

        # Sort by Spearman ρ descending
        rows.sort(key=lambda x: -x[1] if math.isfinite(x[1]) else 999)
        for label, spr, tau, p_tau in rows:
            if not math.isfinite(spr):
                print(f"  {label:<20s} {'N/A':>12s} {'N/A':>12s} {'N/A':>12s}")
            else:
                star = ' ***' if p_tau < 0.001 else ' **' if p_tau < 0.01 else ' *' if p_tau < 0.05 else ''
                print(f"  {label:<20s} {spr:>12.4f} {tau:>12.4f} {p_tau:>12.2e}{star}")

    # Table 1: Overall (all 500)
    print_table("TABLE 1: FlexiBERT ALL", np.arange(len(results)))

    # Table 2: hidden_size=256 subgroup
    h256_idxs = [i for i, r in enumerate(results) if r['hidden_size'] == 256]
    if len(h256_idxs) >= 15:
        print_table("TABLE 2: FlexiBERT h=256", np.array(h256_idxs))

    # Table 3: 5% params pairwise accuracy
    print(f"\n{'═'*70}")
    print(f"  TABLE 3: Pairwise Ranking Accuracy (params within 5%)")
    print(f"{'═'*70}")
    params = np.array([r['n_params'] for r in results])

    for key, label in available:
        vals = np.array([float(r[key]) for r in results])
        ok = np.isfinite(vals) & (vals != 0)

        n_pairs = 0
        n_correct = 0
        for i in range(len(results)):
            if not ok[i]:
                continue
            for j in range(i + 1, len(results)):
                if not ok[j]:
                    continue
                # Only pairs with <5% param difference
                p_diff = abs(params[i] - params[j]) / max(params[i], params[j])
                if p_diff >= 0.05:
                    continue
                n_pairs += 1
                # Check if ZCP ranking agrees with GLUE ranking
                zcp_order = (vals[i] > vals[j])
                glue_order = (glues[i] > glues[j])
                if zcp_order == glue_order:
                    n_correct += 1

        if n_pairs > 0:
            acc = n_correct / n_pairs * 100
            print(f"  {label:<20s}  {acc:6.1f}%  ({n_pairs} pairs)")
        else:
            print(f"  {label:<20s}  N/A  (0 pairs)")

    # By (hidden_size, n_layers) subgroup
    groups = defaultdict(list)
    for i, r in enumerate(results):
        groups[(r['hidden_size'], r['n_layers'])].append(i)
    for (hs, nl), idxs in sorted(groups.items()):
        if len(idxs) >= 15:
            print_table(f"Subgroup h={hs}, L={nl}", np.array(idxs))

    # ── Timing summary ──
    time_keys = sorted([k for k in results[0] if k.startswith('time_')])
    if time_keys:
        print(f"\n{'═'*70}")
        print(f"  TIMING: Average per-architecture cost (ms)")
        print(f"{'═'*70}")
        print(f"  {'Method':<25s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
        print(f"  {'─'*65}")
        timing_rows = []
        for tk in time_keys:
            vals = [r[tk] for r in results if tk in r]
            if not vals:
                continue
            arr = np.array(vals)
            label = tk.replace('time_', '')
            timing_rows.append((label, arr.mean(), arr.std(), arr.min(), arr.max()))
        timing_rows.sort(key=lambda x: x[1])
        for label, mean, std, mn, mx in timing_rows:
            print(f"  {label:<25s} {mean:>10.1f} {std:>10.1f} {mn:>10.1f} {mx:>10.1f}")


if __name__ == "__main__":
    print(f"Device: {DEVICE}", flush=True)
    if DEVICE == 'cuda':
        torch.randn(256, 256, device=DEVICE) @ torch.randn(256, 256, device=DEVICE)

    evaluate_all()
