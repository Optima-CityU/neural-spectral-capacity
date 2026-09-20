#!/usr/bin/env python3
import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import integrate
from scipy.stats import kendalltau, spearmanr

ROOT = Path(os.environ.get("NSC_ROOT", ".")) / "paper"
OUT = Path(os.environ.get("NSC_OUT", "outputs"))
SAMPLES = Path(os.environ.get("NSC_ROOT", ".")) / "sample_inputs"   # small proxy input batches
SCORES = OUT / "scores"
TIMING = OUT / "timing"
LOGS = OUT / "logs"
TABLES = OUT / "tables"
SUBSETS = OUT / "subsets"
FIGURES = OUT / "figures"
for d in [SCORES, TIMING, LOGS, TABLES, SUBSETS, FIGURES]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 20260426
SUBSET_FRAC = 0.15
N_BINS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GLUE_BATCH = SAMPLES / "sample_text/glue_sst2_val64_electra_maxlen128.pt"
WT103_CACHE = ROOT / "transformer-xl/data/cache/wt103_gpt2_seq512"
IMAGENET_BATCH = SAMPLES / "sample_images/imagenet_3imgs.pt"
CIFAR100_BATCH = SAMPLES / "sample_images/cifar100_3imgs.pt"
IMAGENET_LABELS = SAMPLES / "sample_images/imagenet_3labels.pt"
CIFAR100_LABELS = SAMPLES / "sample_images/cifar100_3labels.pt"
FLEXIBERT_SRC = Path(os.environ.get("FLEXIBERT_DIR", "FlexiBERT"))


def log(msg):
    print(msg, flush=True)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def append_csv(path, row, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def install_flexibert_compat():
    """Minimal ELECTRA API used by the FlexiBERT proxy implementations:
    encoder.layer, layer.operation, intermediate.dense_in/dense_list, output.dense."""
    import types

    if "configuration_electra" in sys.modules and "modeling_electra" in sys.modules:
        return

    class ElectraConfig:
        def __init__(self, **kwargs):
            self.vocab_size = kwargs.get("vocab_size", 30522)
            self.hidden_size = kwargs.get("hidden_size", 256)
            self.num_hidden_layers = kwargs.get("num_hidden_layers", 4)
            self.num_attention_heads = kwargs.get("num_attention_heads", 4)
            self.intermediate_size = kwargs.get("intermediate_size", 1024)
            self.nas_config = kwargs.get("nas_config", {"encoder_layers": []})
            self.max_position_embeddings = kwargs.get("max_position_embeddings", 512)
            self.type_vocab_size = kwargs.get("type_vocab_size", 2)
            self.hidden_act = kwargs.get("hidden_act", "gelu")

    class CompatOutput:
        def __init__(self, last_hidden_state):
            self.last_hidden_state = last_hidden_state
            self.attentions = None

    class CompatAttention(nn.Module):
        def __init__(self, hidden, heads):
            super().__init__()
            self.query = nn.Linear(hidden, hidden)
            self.key = nn.Linear(hidden, hidden)
            self.value = nn.Linear(hidden, hidden)
            self.out = nn.Linear(hidden, hidden)
            self.heads = max(1, int(heads))
            self.head_dim = max(1, hidden // self.heads)

        def forward(self, x):
            q = self.query(x)
            k = self.key(x)
            v = self.value(x)
            b, t, h = q.shape
            usable = self.heads * self.head_dim
            q = q[..., :usable].view(b, t, self.heads, self.head_dim).transpose(1, 2)
            k = k[..., :usable].view(b, t, self.heads, self.head_dim).transpose(1, 2)
            v = v[..., :usable].view(b, t, self.heads, self.head_dim).transpose(1, 2)
            att = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
            y = (att @ v).transpose(1, 2).contiguous().view(b, t, usable)
            if usable < h:
                y = F.pad(y, (0, h - usable))
            return self.out(y)

    class CompatOperation(nn.Module):
        def __init__(self, hidden, heads):
            super().__init__()
            self.operation = CompatAttention(hidden, heads)

        def forward(self, x, *args, **kwargs):
            return self.operation(x)

    class CompatIntermediate(nn.Module):
        def __init__(self, hidden, intermediate, nff=1):
            super().__init__()
            self.dense_in = nn.Linear(hidden, intermediate)
            self.dense_list = nn.ModuleList()
            self.intermediate_act_fn = nn.GELU()
            for _ in range(max(0, int(nff) - 1)):
                self.dense_list.append(nn.Linear(intermediate, intermediate))

        def forward(self, x):
            x = self.intermediate_act_fn(self.dense_in(x))
            for dense in self.dense_list:
                x = self.intermediate_act_fn(dense(x))
            return x

    class CompatLayerOutput(nn.Module):
        def __init__(self, hidden, intermediate):
            super().__init__()
            self.dense = nn.Linear(intermediate, hidden)
            self.LayerNorm = nn.LayerNorm(hidden)

        def forward(self, x, residual):
            return self.LayerNorm(self.dense(x) + residual)

    class ElectraLayer(nn.Module):
        def __init__(self, hidden, heads, intermediate, nff=1):
            super().__init__()
            self.operation = CompatOperation(hidden, heads)
            self.attn_norm = nn.LayerNorm(hidden)
            self.intermediate = CompatIntermediate(hidden, intermediate, nff=nff)
            self.output = CompatLayerOutput(hidden, intermediate)

        def forward(self, x):
            attn = self.operation(x)
            x = self.attn_norm(x + attn)
            return self.output(self.intermediate(x), x)

    class CompatEncoder(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            layers_cfg = cfg.nas_config.get("encoder_layers", [])
            if not layers_cfg:
                layers_cfg = [{} for _ in range(cfg.num_hidden_layers)]
            self.layer = nn.ModuleList()
            for lc in layers_cfg:
                heads = lc.get("num_operation_heads", cfg.num_attention_heads)
                inter = lc.get("feed_forward_dimension", cfg.intermediate_size)
                nff = lc.get("num_feed_forward", 1)
                self.layer.append(ElectraLayer(cfg.hidden_size, heads, inter, nff=nff))

        def forward(self, x):
            for layer in self.layer:
                x = layer(x)
            return x

    class CompatEmbeddings(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
            self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
            self.token_type_embeddings = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
            self.LayerNorm = nn.LayerNorm(cfg.hidden_size)

        def forward(self, input_ids, token_type_ids=None):
            b, t = input_ids.shape
            pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
            return self.LayerNorm(
                self.word_embeddings(input_ids)
                + self.position_embeddings(pos)
                + self.token_type_embeddings(token_type_ids)
            )

    class ElectraModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.config = cfg
            self.embeddings = CompatEmbeddings(cfg)
            self.encoder = CompatEncoder(cfg)

        def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, **kwargs):
            x = self.embeddings(input_ids, token_type_ids=token_type_ids)
            return CompatOutput(self.encoder(x))

    cfg_mod = types.ModuleType("configuration_electra")
    cfg_mod.ElectraConfig = ElectraConfig
    mdl_mod = types.ModuleType("modeling_electra")
    mdl_mod.ElectraModel = ElectraModel
    mdl_mod.ElectraLayer = ElectraLayer
    sys.modules["configuration_electra"] = cfg_mod
    sys.modules["modeling_electra"] = mdl_mod


def load_existing(path):
    if path.exists():
        return read_json(path)
    return []


def load_glue_batch():
    if not GLUE_BATCH.exists():
        raise FileNotFoundError(GLUE_BATCH)
    batch = torch.load(GLUE_BATCH, map_location="cpu")
    req = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    for k in req:
        if k not in batch:
            raise KeyError(f"GLUE batch missing {k}")
    return {k: batch[k].to(DEVICE) for k in req}


def load_gpt2_batch(batch_size=4, seq_len=192):
    if not WT103_CACHE.exists():
        raise FileNotFoundError(WT103_CACHE)
    from datasets import load_from_disk
    ds = load_from_disk(str(WT103_CACHE))
    rows = [ds["validation"][i]["input_ids"] for i in range(batch_size)]
    x = torch.tensor([r[:seq_len + 1] for r in rows], dtype=torch.long, device=DEVICE)
    if x.shape != (batch_size, seq_len + 1):
        raise RuntimeError(f"bad GPT2 batch shape {tuple(x.shape)}")
    return x[:, :-1].contiguous(), x[:, 1:].contiguous()


def image_label_path(path):
    p = Path(path)
    s = str(p)
    if s.endswith("_3imgs.pt"):
        return Path(s.replace("_3imgs.pt", "_3labels.pt"))
    return p.with_name(p.stem.replace("imgs", "labels") + p.suffix)


def load_image_batch(path, num_classes):
    if not path.exists():
        raise FileNotFoundError(path)
    labels_path = image_label_path(path)
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    data = torch.load(path, map_location="cpu")
    labels = torch.load(labels_path, map_location="cpu")
    if not torch.is_tensor(data):
        raise TypeError(f"{path} must be a tensor, got {type(data)}")
    if not torch.is_tensor(labels):
        raise TypeError(f"{labels_path} must be a tensor, got {type(labels)}")
    data = data.float().to(DEVICE)
    if data.ndim != 4 or data.shape[1] != 3:
        raise RuntimeError(f"bad image batch shape {tuple(data.shape)}")
    labels = labels.long().to(DEVICE)
    if labels.ndim != 1 or labels.shape[0] != data.shape[0]:
        raise RuntimeError(f"bad image labels shape {tuple(labels.shape)} for {tuple(data.shape)}")
    if int(labels.max().item()) >= int(num_classes) or int(labels.min().item()) < 0:
        raise RuntimeError(f"labels out of range for {num_classes} classes: {labels.detach().cpu().tolist()}")
    return data, labels


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def time_call(fn):
    start = time.time()
    val = fn()
    return val, time.time() - start


def safe_corr(scores, gt):
    x = np.asarray(scores, dtype=float)
    y = np.asarray(gt, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or len(np.unique(x[mask])) < 2 or len(np.unique(y[mask])) < 2:
        return None, None
    return float(kendalltau(x[mask], y[mask]).correlation), float(spearmanr(x[mask], y[mask]).correlation)


def stratified_indices(values, frac=SUBSET_FRAC, bins=N_BINS, seed=SEED):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    order = np.argsort(values)
    selected, stats = [], []
    for bi, idx in enumerate(np.array_split(order, bins)):
        n = max(1, int(round(len(idx) * frac)))
        chosen = rng.choice(idx, size=n, replace=False)
        selected.extend(int(i) for i in chosen)
        stats.append({
            "bin": bi,
            "available": int(len(idx)),
            "selected": int(n),
            "min_gt": float(values[idx].min()),
            "max_gt": float(values[idx].max()),
        })
    return sorted(selected), stats


_PSI_CACHE = {}


def mp_density(x, gamma):
    lp = (1 + np.sqrt(gamma)) ** 2
    lm = (1 - np.sqrt(gamma)) ** 2
    if x < lm or x > lp:
        return 0.0
    return np.sqrt((lp - x) * (x - lm)) / (2 * np.pi * gamma * x)


def psi_mp(m, n, sigma=None):
    m, n = int(m), int(n)
    if m <= 0 or n <= 0:
        return 0.0
    if sigma is None:
        sigma = np.sqrt(2.0 / (m + n))
    key = (m, n, round(float(sigma), 12))
    if key in _PSI_CACHE:
        return _PSI_CACHE[key]
    mm, nn = (m, n) if m >= n else (n, m)
    gamma = nn / mm
    lp = (1 + np.sqrt(gamma)) ** 2
    lm = (1 - np.sqrt(gamma)) ** 2
    s2m = sigma ** 2 * mm

    def integrand(x):
        d = mp_density(x, gamma)
        return np.log(1 + s2m * x) * d if d > 0 else 0.0

    val = nn * integrate.quad(integrand, lm, lp, limit=200)[0]
    _PSI_CACHE[key] = float(val)
    return float(val)


def zerolm_svd_matrix(weight):
    W = weight.detach().float().to(DEVICE)
    s = torch.linalg.svdvals(W)
    return float(((s ** 2).sum() / max(1, min(W.shape))).item())


def flexibert_nsc_mp_from_arch(arch):
    hpo = arch["hparams"]["model_hparam_overrides"]
    layers_cfg = hpo["nas_config"]["encoder_layers"]
    H = int(hpo["hidden_size"])
    D_FF_global = int(layers_cfg[0].get("feed_forward_dimension", 1024))
    total = 0.0
    for lc in layers_cfg:
        op = lc["operation_type"]
        nff = int(lc.get("num_feed_forward", 1))
        D_FF = int(lc.get("feed_forward_dimension", D_FF_global))
        ff_dim = D_FF // nff
        if op == "SA":
            total += psi_mp(3 * H, H) + psi_mp(H, H)
        elif op == "LT":
            total += 2 * psi_mp(H, H)
        elif op == "DSC":
            total += psi_mp(H, H)
        for _ in range(nff):
            total += psi_mp(ff_dim, H) + psi_mp(H, ff_dim)
    return float(total)


def flexibert_zerolm_svd_parts(model):
    s_attn_total = 0.0
    s_ffn_total = 0.0
    for layer in model.encoder.layer:
        for mod in layer.operation.modules():
            if isinstance(mod, nn.Linear):
                s_attn_total += zerolm_svd_matrix(mod.weight)
        for sub in [layer.intermediate, layer.output]:
            for mod in sub.modules():
                if isinstance(mod, nn.Linear):
                    s_ffn_total += zerolm_svd_matrix(mod.weight)
    return s_attn_total, s_ffn_total


def gpt2_zerolm_svd_parts(model):
    s_attn_total = 0.0
    s_ffn_total = 0.0
    for block in model.blocks:
        for mod in [block.attn.c_attn, block.attn.c_proj]:
            s_attn_total += zerolm_svd_matrix(mod.weight)
        for mod in [block.mlp.c_fc, block.mlp.c_proj]:
            s_ffn_total += zerolm_svd_matrix(mod.weight)
    return s_attn_total, s_ffn_total


def autoformer_zerolm_svd_parts(model):
    s_attn_total = 0.0
    s_ffn_total = 0.0
    for blk in model.blocks:
        for mod in [blk.attn.qkv, blk.attn.proj]:
            s_attn_total += zerolm_svd_matrix(mod.weight)
        for mod in [blk.mlp.fc1, blk.mlp.fc2]:
            s_ffn_total += zerolm_svd_matrix(mod.weight)
    return s_attn_total, s_ffn_total


def conv_psi(c_in, c_out, k=3):
    return psi_mp(c_out, c_in * k * k)


def wpca_from_activations(activations, n_params, threshold=0.99):
    pca_sum = 0.0
    for act in activations:
        if act.ndim == 4:
            a = act.permute(0, 2, 3, 1).reshape(-1, act.shape[1]).float()
        else:
            a = act.reshape(-1, act.shape[-1]).float()
        if a.shape[0] < 2 or a.shape[1] < 2:
            continue
        a = a - a.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(a.to(DEVICE))
        denom = (s ** 2).sum().clamp(min=1e-12)
        cumvar = torch.cumsum(s ** 2, 0) / denom
        pca_sum += int((cumvar < threshold).sum().item() + 1)
    return float(pca_sum), float(n_params) * float(pca_sum)


def compute_wpca_hooks(model, data, modules, forward_fn):
    model.eval()
    activations, hooks = [], []

    def hook(_m, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        activations.append(t.detach())

    for m in modules:
        hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        forward_fn(model, data)
    for h in hooks:
        h.remove()
    return wpca_from_activations(activations, count_params(model))


def snip_gradnorm_from_loss(model, loss_fn):
    model.train()
    model.zero_grad(set_to_none=True)
    for p in model.parameters():
        p.requires_grad_(True)
    loss = loss_fn()
    loss.backward()
    snip = 0.0
    grad_sq = 0.0
    for p in model.parameters():
        if p.grad is not None and p.dim() >= 2:
            snip += float((p.data.float() * p.grad.data.float()).abs().sum().item())
            grad_sq += float((p.grad.data.float() ** 2).sum().item())
    model.zero_grad(set_to_none=True)
    return snip, math.sqrt(grad_sq)


def run_flexibert(args):
    log("=== FlexiBERT real SST-2 run ===")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if str(FLEXIBERT_SRC) not in sys.path:
        sys.path.insert(0, str(FLEXIBERT_SRC))
    flex = load_module("flex_all_zcps_real", Path(__file__).parent / "eval_all_zcps.py")
    flex.DEVICE = DEVICE
    glue = load_glue_batch()
    meta = {
        "batch": str(GLUE_BATCH),
        "input_ids_shape": list(glue["input_ids"].shape),
        "labels_shape": list(glue["labels"].shape),
        "device": DEVICE,
        "loss": "SST-2 CLS cross_entropy with 2-way linear head for head/gradient proxies",
    }
    write_json(LOGS / "flexibert_batch_meta.json", meta)

    def real_forward_backward(model, head, cfg):
        model.train()
        head.train()
        model.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        out = model(
            input_ids=glue["input_ids"],
            token_type_ids=glue["token_type_ids"],
            attention_mask=glue["attention_mask"],
        )
        pooled = out.last_hidden_state[:, 0]
        logits = head(pooled)
        loss = F.cross_entropy(logits, glue["labels"])
        loss.backward()

    def real_collect_ffn(model, cfg):
        activations, hooks = [], []

        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            activations.append(t.detach())

        for blk in model.encoder.layer:
            hooks.append(blk.intermediate.dense_in.register_forward_hook(hook))
        model.eval()
        with torch.no_grad():
            model(
                input_ids=glue["input_ids"],
                token_type_ids=glue["token_type_ids"],
                attention_mask=glue["attention_mask"],
            )
        for h in hooks:
            h.remove()
        return activations

    def real_syn_diversity(model, cfg):
        activations, hooks = [], []

        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            activations.append((t.detach() > 0).float().reshape(t.shape[0], -1))

        for blk in model.encoder.layer:
            hooks.append(blk.intermediate.dense_in.register_forward_hook(hook))
        model.eval()
        with torch.no_grad():
            model(input_ids=glue["input_ids"], token_type_ids=glue["token_type_ids"], attention_mask=glue["attention_mask"])
        for h in hooks:
            h.remove()
        if not activations:
            return 0.0
        codes = torch.cat(activations, dim=1)
        s = torch.linalg.svdvals(codes.float())
        return float((s > 1e-5).sum().item() / codes.shape[0])

    def real_act_distance(model, cfg):
        model.eval()
        with torch.no_grad():
            out = model(input_ids=glue["input_ids"], token_type_ids=glue["token_type_ids"], attention_mask=glue["attention_mask"])
        h = out.last_hidden_state[:, 0]
        d = torch.cdist(h, h, p=2)
        mask = torch.triu(torch.ones(h.shape[0], h.shape[0], device=DEVICE), diagonal=1).bool()
        return float(d[mask].mean().item())

    def real_naswot(model, cfg):
        acts, hooks = [], []

        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            acts.append((t.detach() > 0).float())

        for module in model.modules():
            if isinstance(module, (nn.GELU, nn.SiLU, nn.ReLU)):
                hooks.append(module.register_forward_hook(hook))
        if not hooks:
            for blk in model.encoder.layer:
                hooks.append(blk.intermediate.dense_in.register_forward_hook(hook))
        model.eval()
        with torch.no_grad():
            model(input_ids=glue["input_ids"], token_type_ids=glue["token_type_ids"], attention_mask=glue["attention_mask"])
        for h in hooks:
            h.remove()
        if not acts:
            return 0.0
        codes = torch.cat([a.reshape(a.shape[0], -1) for a in acts], dim=1)
        K = codes @ codes.T + (1 - codes) @ (1 - codes).T
        K = K / codes.shape[1] + 1e-5 * torch.eye(codes.shape[0], device=DEVICE)
        val = torch.logdet(K).item()
        return float(val) if math.isfinite(val) else 0.0

    def real_jacobian_cosine(model, head, cfg):
        model.eval()
        halves = [
            {k: v[:2] for k, v in glue.items() if k != "labels"},
            {k: v[2:4] for k, v in glue.items() if k != "labels"},
        ]

        def flat_grad(inputs):
            model.zero_grad(set_to_none=True)
            head.zero_grad(set_to_none=True)
            out = model(**inputs)
            logits = head(out.last_hidden_state[:, 0])
            logits.sum().backward()
            grads = []
            for p in list(model.parameters()) + list(head.parameters()):
                if p.grad is not None:
                    grads.append(p.grad.detach().float().flatten())
            return torch.cat(grads) if grads else None

        j1, j2 = flat_grad(halves[0]), flat_grad(halves[1])
        model.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        if j1 is None or j2 is None:
            return 0.0
        return float(-F.cosine_similarity(j1.unsqueeze(0), j2.unsqueeze(0)).item())

    def real_head_importance(model, head, cfg):
        attn_outputs, n_heads_list, hooks = [], [], []

        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            t.retain_grad()
            attn_outputs.append(t)

        layers_cfg = cfg.nas_config["encoder_layers"]
        for li, blk in enumerate(model.encoder.layer):
            if layers_cfg[li].get("operation_type", "") in ("SA", "DSC"):
                attn_op = blk.operation
                if hasattr(attn_op, "operation"):
                    hooks.append(attn_op.operation.register_forward_hook(hook))
                    n_heads_list.append(layers_cfg[li]["num_operation_heads"])
        real_forward_backward(model, head, cfg)
        for h in hooks:
            h.remove()
        importance = 0.0
        for t, n_h in zip(attn_outputs, n_heads_list):
            if t.grad is None:
                continue
            head_dim = t.shape[-1] // n_h
            a = t.float().view(t.shape[0], t.shape[1], n_h, head_dim)
            g = t.grad.float().view(t.shape[0], t.shape[1], n_h, head_dim)
            importance += float((a * g).sum(dim=(0, 1, 3)).abs().sum().item())
        model.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        return importance

    def real_head_softmax_conf(model, cfg):
        return flex._compute_softmax_confidence_manual(
            model, cfg, glue["input_ids"][:4]
        )

    def real_grasp(model, head, cfg):
        weights = []
        for p in list(model.parameters()) + list(head.parameters()):
            if p.dim() >= 2:
                p.requires_grad_(True)
                weights.append(p)

        def loss_once(create_graph=False):
            out = model(
                input_ids=glue["input_ids"],
                token_type_ids=glue["token_type_ids"],
                attention_mask=glue["attention_mask"],
            )
            logits = head(out.last_hidden_state[:, 0])
            return F.cross_entropy(logits, glue["labels"])

        model.train()
        head.train()
        model.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        grad_w = torch.autograd.grad(loss_once(), weights, allow_unused=True)
        grad_f = torch.autograd.grad(loss_once(), weights, create_graph=True, allow_unused=True)
        z = 0.0
        for gw, gf in zip(grad_w, grad_f):
            if gw is not None and gf is not None:
                z = z + (gw.detach() * gf).sum()
        z.backward()
        score = 0.0
        for p in weights:
            if p.grad is not None:
                score -= float((p.data.float() * p.grad.data.float()).sum().item())
        model.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        return score

    flex._do_forward_backward = real_forward_backward
    flex._collect_ffn_intermediate_activations = real_collect_ffn
    flex.compute_synaptic_diversity = real_syn_diversity
    flex.compute_activation_distance = real_act_distance
    flex.compute_naswot = real_naswot
    flex.compute_jacobian_cosine = real_jacobian_cosine
    flex.compute_head_importance = real_head_importance
    flex.compute_head_softmax_confidence = real_head_softmax_conf
    flex.compute_grasp = real_grasp

    methods = [
        ("params", lambda m, h, c: count_params(m)),
        ("nsc_mp", lambda m, h, c, a=None: flexibert_nsc_mp_from_arch(a)),
        ("snip", lambda m, h, c: flex.compute_snip(m, h, c)),
        ("synflow", lambda m, h, c: flex.compute_synflow(m, c)),
        ("gradnorm", lambda m, h, c: flex.compute_gradnorm(m, h, c)),
        ("fisher", lambda m, h, c: flex.compute_fisher(m, h, c)),
        ("grasp", lambda m, h, c: flex.compute_grasp(m, h, c)),
        ("naswot", lambda m, h, c: flex.compute_naswot(m, c)),
        ("pca_wpca", lambda m, h, c: flex.compute_pca_and_wpca(m, c)),
        ("syn_diversity", lambda m, h, c: flex.compute_synaptic_diversity(m, c)),
        ("act_distance", lambda m, h, c: flex.compute_activation_distance(m, c)),
        ("head_importance", lambda m, h, c: flex.compute_head_importance(m, h, c)),
        ("head_softmax_conf", lambda m, h, c: flex.compute_head_softmax_confidence(m, c)),
        ("jacobian_cosine", lambda m, h, c: flex.compute_jacobian_cosine(m, h, c)),
        ("zerolm", lambda m, h, c: flexibert_zerolm_svd_parts(m)),
    ]
    if args.smoke:
        keep = {"params", "nsc_mp", "pca_wpca", "snip", "gradnorm", "zerolm"}
        methods = [(n, f) for n, f in methods if n in keep]

    bench = read_json(ROOT / "dataset/flexibert/BERT_benchmark.json")
    if args.limit:
        bench = bench[: args.limit]
    elif args.smoke:
        bench = bench[:2]

    for setting in ["corrected"]:
        gt_rows = read_json(ROOT / f"dataset/flexibert/flexibert_{setting}.json")
        gt_by_id = {int(r["arch_id"]): r for r in gt_rows}
        out_path = SCORES / f"flexibert_{setting}_realdata_scores.json"
        results = {str(r["arch_id"]): r for r in load_existing(out_path)}
        timing_rows = []
        for i, arch in enumerate(bench):
            arch_id = str(arch.get("id", i))
            gt = gt_by_id[int(arch_id)]
            row = results.get(arch_id, {
                "arch_id": int(arch_id),
                "glue": float(gt["glue"]),
                "setting": setting,
            })
            row["glue"] = float(gt["glue"])
            if "n_params" in gt:
                row.setdefault("n_params", int(gt["n_params"]))
            if all(("wpca" in row if name == "pca_wpca" else name in row or (name == "zerolm" and "zerolm_attn" in row) or (name == "params" and "n_params" in row)) for name, _ in methods):
                continue
            log(f"[FlexiBERT {setting}] arch {i+1}/{len(bench)} id={arch_id}")
            for name, fn in methods:
                done = ("wpca" in row) if name == "pca_wpca" else (("zerolm_attn" in row) if name == "zerolm" else (("n_params" in row) if name == "params" else (name in row)))
                if done:
                    continue
                model, _old_head, cfg = flex.build_flexibert(arch)
                head = nn.Linear(cfg.hidden_size, 2, bias=True).to(DEVICE)
                torch.manual_seed(42)
                try:
                    if name == "nsc_mp":
                        val, sec = time_call(lambda: fn(model, head, cfg, arch))
                    else:
                        val, sec = time_call(lambda: fn(model, head, cfg))
                    if name == "params":
                        row["n_params"] = int(val)
                    elif name == "nsc_mp":
                        row["nsc_mp_flat"] = float(val)
                        row["nsc_mp"] = float(val)
                    elif name == "pca_wpca":
                        row["pca"], row["wpca"] = float(val[0]), float(val[1])
                    elif name == "zerolm":
                        row["zerolm_attn"], row["zerolm_ffn"] = float(val[0]), float(val[1])
                        row["zerolm"] = float(val[0] + val[1])
                    else:
                        row[name] = float(val)
                    row.pop(f"{name}_error", None)
                    row[f"time_{name}_sec"] = sec
                    timing_rows.append({"benchmark": f"flexibert_{setting}", "arch_id": arch_id, "method": name, "time_sec": sec, "status": "ok"})
                except Exception as e:
                    row[f"{name}_error"] = repr(e)
                    row[f"time_{name}_sec"] = None
                    timing_rows.append({"benchmark": f"flexibert_{setting}", "arch_id": arch_id, "method": name, "time_sec": None, "status": "failed", "error": repr(e)})
                    traceback.print_exc()
                del model, head, _old_head
                clear_cuda()
            results[arch_id] = row
            write_json(out_path, list(results.values()))
        write_json(TIMING / f"flexibert_{setting}_per_arch_timing.json", timing_rows)


def run_gpt2(args):
    log("=== GPT-2 real WikiText-103 run ===")
    mod = load_module("gpt2_base_real", Path(__file__).parent / "eval_gpt2_baselines.py")
    mod.DEVICE = DEVICE
    x, y = load_gpt2_batch(batch_size=4, seq_len=192)
    write_json(LOGS / "gpt2_batch_meta.json", {
        "batch": str(WT103_CACHE),
        "input_shape": list(x.shape),
        "target_shape": list(y.shape),
        "target": "WikiText-103 validation shifted next-token labels",
        "device": DEVICE,
    })

    def wpca(model):
        return compute_wpca_hooks(model, x, [b.mlp.c_fc for b in model.blocks], lambda m, d: m(d))

    def snip_grad(model):
        return snip_gradnorm_from_loss(model, lambda: F.cross_entropy(model(x).reshape(-1, model.lm_head.weight.size(0)), y.reshape(-1)))

    def lpzero_real(model):
        model.train()
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).reshape(-1, model.lm_head.weight.size(0)), y.reshape(-1))
        loss.backward()
        total = 0.0
        for block in model.blocks:
            for lin in [block.attn.c_attn, block.attn.c_proj, block.mlp.c_fc, block.mlp.c_proj]:
                W = lin.weight.data.float()
                G = lin.weight.grad.float() if lin.weight.grad is not None else torch.zeros_like(W)
                left = torch.pow(W, 2).abs().sum() / (W.numel() + 1e-9)
                right = torch.pow(F.softmax(G.flatten(), dim=0), 2).sum()
                total += float((left + right).item())
        model.zero_grad(set_to_none=True)
        return total

    bench = read_json(ROOT / "dataset/gpt2_txl/gpt2_benchmark.json")
    old_scores_path = OUT / "gpt2_scores.json"   # optional cache from compute_gpt2.py
    old_nsc = {}
    if old_scores_path.exists():
        for r in read_json(old_scores_path):
            old_nsc[str(r.get("arch_key"))] = r.get("nsc_mp")
    items = list(bench.items())
    if args.limit:
        items = items[: args.limit]
    elif args.smoke:
        items = items[:2]
    out_path = SCORES / "gpt2_realdata_scores.json"
    results = {str(r["arch_key"]): r for r in load_existing(out_path)}
    timing_rows = []
    for i, (key, arch) in enumerate(items):
        row = results.get(str(key), {
            "arch_key": key,
            "neg_ppl": -float(arch["valid_ppl"]),
            "n_params": float(arch["params"]["total"]),
        })
        row["neg_ppl"] = -float(arch["valid_ppl"])
        nsc_existing = row.get("nsc_mp")
        nsc_missing = nsc_existing is None or not np.isfinite(float(nsc_existing))
        if nsc_missing and old_nsc.get(str(key)) is not None:
            row["nsc_mp"] = float(old_nsc[str(key)])
            row.setdefault("time_nsc_mp_sec", 0.0)
        methods_needed = ["nsc_mp", "pca", "wpca", "zerolm", "snip", "gradnorm", "lpzero", "time_params_sec"]
        if all(k in row for k in methods_needed):
            continue
        log(f"[GPT2] arch {i+1}/{len(items)} key={key}")
        torch.manual_seed(42)
        model = mod.build_model(arch["config"]).to(DEVICE)
        method_fns = [
            ("zerolm", lambda: gpt2_zerolm_svd_parts(model)),
            ("pca_wpca", lambda: wpca(model)),
            ("snip_gradnorm", lambda: snip_grad(model)),
            ("lpzero", lambda: lpzero_real(model)),
            ("params", lambda: count_params(model)),
        ]
        for name, fn in method_fns:
            try:
                val, sec = time_call(fn)
                if name == "zerolm":
                    row["zerolm_attn"], row["zerolm_ffn"] = float(val[0]), float(val[1])
                    row["zerolm"] = float(val[0] + val[1])
                elif name == "pca_wpca":
                    row["pca"], row["wpca"] = float(val[0]), float(val[1])
                elif name == "snip_gradnorm":
                    row["snip"], row["gradnorm"] = float(val[0]), float(val[1])
                elif name == "params":
                    row["n_params"] = int(val)
                else:
                    row[name] = float(val)
                row.pop(f"{name}_error", None)
                row[f"time_{name}_sec"] = sec
                timing_rows.append({"benchmark": "gpt2", "arch_id": key, "method": name, "time_sec": sec, "status": "ok"})
            except Exception as e:
                row[f"{name}_error"] = repr(e)
                timing_rows.append({"benchmark": "gpt2", "arch_id": key, "method": name, "time_sec": None, "status": "failed", "error": repr(e)})
                traceback.print_exc()
        del model
        clear_cuda()
        results[str(key)] = row
        write_json(out_path, list(results.values()))
    write_json(TIMING / "gpt2_per_arch_timing.json", timing_rows)


def autoformer_gradnorm(model, x, y):
    return snip_gradnorm_from_loss(model, lambda: F.cross_entropy(model(x), y))


class CorrectAutoFormerAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"AutoFormer dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = int(dim) // int(num_heads)
        self.inner_dim = int(dim)
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.inner_dim)
        return self.proj(x)


class CorrectAutoFormerMlp(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class CorrectAutoFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CorrectAutoFormerAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = CorrectAutoFormerMlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CorrectAutoFormerViT(nn.Module):
    def __init__(self, layer_num, embed_dims, num_heads_list, mlp_ratios, img_size=224, patch_size=16, num_classes=1000, init_std=0.02):
        super().__init__()
        self.embed_dim = embed_dims[0] if isinstance(embed_dims, list) else embed_dims
        self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.blocks = nn.ModuleList()
        for i in range(layer_num):
            dim = embed_dims[i] if isinstance(embed_dims, list) else embed_dims
            nh = num_heads_list[i] if isinstance(num_heads_list, list) else num_heads_list
            mr = mlp_ratios[i] if isinstance(mlp_ratios, list) else mlp_ratios
            self.blocks.append(CorrectAutoFormerBlock(dim, nh, mr))
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes)
        self._init_weights(init_std)

    def _init_weights(self, std=0.02):
        nn.init.trunc_normal_(self.pos_embed, std=std)
        nn.init.trunc_normal_(self.cls_token, std=std)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
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


def build_correct_autoformer_model(arch):
    ns = arch["net_setting"]
    return CorrectAutoFormerViT(
        layer_num=ns["layer_num"],
        embed_dims=ns["embed_dim"],
        num_heads_list=ns["num_heads"],
        mlp_ratios=ns["mlp_ratio"],
    )


def autoformer_nsc_mp_original(arch):
    ns = arch["net_setting"]
    depth = ns["layer_num"]
    ed_list = ns["embed_dim"]
    nh_list = ns["num_heads"]
    mr_list = ns["mlp_ratio"]
    total = 0.0
    for i in range(depth):
        ed = ed_list[i] if isinstance(ed_list, list) else ed_list
        nh = nh_list[i] if isinstance(nh_list, list) else nh_list
        mr = mr_list[i] if isinstance(mr_list, list) else mr_list
        hd = ed // nh
        mlp_dim = int(ed * mr)
        total += 3 * nh * psi_mp(ed, hd)
        total += psi_mp(ed, ed)
        total += psi_mp(mlp_dim, ed) + psi_mp(ed, mlp_dim)
    return float(total)


def autoformer_zerolm_init_expected(arch):
    ns = arch["net_setting"]
    total = 0.0
    for D, r in zip(ns["embed_dim"], ns["mlp_ratio"]):
        D = float(D)
        r = float(r)
        total += 3 * D + D + r * D + r * D
    return float(total)


def run_autoformer(args):
    scales = getattr(args, "autoformer_scales", None) or ["tiny"]
    mod = load_module("autoformer_real", Path(__file__).parent / "eval_autoformer_tiny.py")
    mod.DEVICE = DEVICE
    x, y = load_image_batch(IMAGENET_BATCH, 1000)
    write_json(LOGS / "autoformer_batch_meta.json", {
        "batch": str(IMAGENET_BATCH),
        "labels_path": str(IMAGENET_LABELS),
        "input_shape": list(x.shape),
        "labels_shape": list(y.shape),
        "labels": y.detach().cpu().tolist(),
        "device": DEVICE,
        "scales": scales,
    })
    for scale in scales:
        scale = scale.lower()
        if scale not in {"tiny", "small", "base"}:
            raise ValueError(f"unsupported AutoFormer scale: {scale}")
        bench_name = f"autoformer_{scale}"
        log(f"=== AutoFormer {scale.capitalize()} real ImageNet run ===")
        bench = read_json(ROOT / f"dataset/autoformer/data/autoformer_{scale}_1k.json")
        items = sorted(bench.items(), key=lambda kv: int(kv[0]))
        if args.limit:
            items = items[: args.limit]
        elif args.smoke:
            items = items[:2]
        out_path = SCORES / f"autoformer_{scale}_realdata_scores.json"
        results = {str(r["idx"]): r for r in load_existing(out_path)}
        timing_path = TIMING / f"autoformer_{scale}_per_arch_timing.json"
        timing_rows = load_existing(timing_path)
        for i, (idx_str, arch) in enumerate(items):
            idx = int(idx_str)
            row = results.get(str(idx), {"idx": idx, "acc": float(arch["performance"]["Imagenet"]["clean"]), "params": arch["params"]})
            row["acc"] = float(arch["performance"]["Imagenet"]["clean"])
            row["params"] = arch["params"]
            if all(k in row for k in ["nsc_mp", "zerolm_init_expected", "wpca", "snip", "gradnorm", "n_params"]):
                continue
            log(f"[AutoFormer {scale.capitalize()}] arch {i+1}/{len(items)} idx={idx}")
            torch.manual_seed(42)
            model = mod.build_model(arch).to(DEVICE)
            method_fns = [
                ("nsc_mp", lambda: autoformer_nsc_mp_original(arch)),
                ("zerolm_init_expected", lambda: autoformer_zerolm_init_expected(arch)),
                ("pca_wpca", lambda: compute_wpca_hooks(model, x, [b.mlp.fc1 for b in model.blocks], lambda m, d: m(d))),
                ("snip_gradnorm", lambda: autoformer_gradnorm(model, x, y)),
                ("params", lambda: count_params(model)),
            ]
            for name, fn in method_fns:
                try:
                    val, sec = time_call(fn)
                    if name == "nsc_mp":
                        row["nsc_mp"] = float(val)
                    elif name == "zerolm_init_expected":
                        row["zerolm_init_expected"] = float(val)
                    elif name == "pca_wpca":
                        row["pca"], row["wpca"] = float(val[0]), float(val[1])
                    elif name == "snip_gradnorm":
                        row["snip"], row["gradnorm"] = float(val[0]), float(val[1])
                    elif name == "params":
                        row["n_params"] = int(val)
                    row.pop(f"{name}_error", None)
                    row[f"time_{name}_sec"] = sec
                    timing_rows.append({"benchmark": bench_name, "arch_id": idx, "method": name, "time_sec": sec, "status": "ok"})
                except Exception as e:
                    row[f"{name}_error"] = repr(e)
                    timing_rows.append({"benchmark": bench_name, "arch_id": idx, "method": name, "time_sec": None, "status": "failed", "error": repr(e)})
                    traceback.print_exc()
            del model
            clear_cuda()
            results[str(idx)] = row
            write_json(out_path, list(results.values()))
            write_json(timing_path, timing_rows)
        write_json(timing_path, timing_rows)


class ReLUConvBN(nn.Module):
    def __init__(self, c_in, c_out, ks, stride=1, padding=0):
        super().__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(c_in, c_out, ks, stride=stride, padding=padding, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        return self.op(x)


class ResConv1x1(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        return self.bn(self.conv(x))


class InferCell(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.op_1_0 = ReLUConvBN(c_in, c_out, 3, padding=1)
        self.op_2_0 = ReLUConvBN(c_in, c_out, 3, padding=1)
        self.op_2_1 = ReLUConvBN(c_out, c_out, 3, padding=1)
        self.skip_0 = nn.Identity() if c_in == c_out else ResConv1x1(c_in, c_out)
        self.op_3_1 = ReLUConvBN(c_out, c_out, 3, padding=1)
        self.op_3_2 = ReLUConvBN(c_out, c_out, 3, padding=1)

    def forward(self, x):
        n1 = self.op_1_0(x)
        n2 = self.op_2_0(x) + self.op_2_1(n1)
        return self.skip_0(x) + self.op_3_1(n1) + self.op_3_2(n2)


class ResNetBasicblock(nn.Module):
    def __init__(self, c_in, c_out, stride=2):
        super().__init__()
        self.conv_a = ReLUConvBN(c_in, c_out, 3, stride=stride, padding=1)
        self.conv_b = ReLUConvBN(c_out, c_out, 3, padding=1)
        self.downsample = nn.Sequential(nn.AvgPool2d(stride, stride, count_include_pad=False), nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        return self.conv_b(self.conv_a(x)) + self.downsample(x)


class NATSBenchSSS(nn.Module):
    def __init__(self, channels, nc=100):
        super().__init__()
        c0, c1, c2, c3, c4 = [int(x) for x in channels]
        self.stem = nn.Sequential(nn.Conv2d(3, c0, 3, padding=1, bias=False), nn.BatchNorm2d(c0))
        self.blocks = nn.ModuleList([InferCell(c0, c1), ResNetBasicblock(c1, c2), InferCell(c2, c3), ResNetBasicblock(c3, c4), InferCell(c4, c4)])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c4, nc)

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return self.classifier(self.gap(x).flatten(1))


MNV3_BASE_CH = [16, 16, 24, 40, 80, 112, 160]


class InvertedResidual(nn.Module):
    def __init__(self, c_in, c_out, ks, expand, stride):
        super().__init__()
        mid = int(c_in * expand)
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(c_in, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=False)]
        layers += [nn.Conv2d(mid, mid, ks, stride=stride, padding=ks // 2, groups=mid, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=False)]
        layers += [nn.Conv2d(mid, c_out, 1, bias=False), nn.BatchNorm2d(c_out)]
        self.op = nn.Sequential(*layers)
        self.use_res = stride == 1 and c_in == c_out

    def forward(self, x):
        y = self.op(x)
        return x + y if self.use_res else y


class MNV3ProxyNet(nn.Module):
    def __init__(self, pheno, nc=1000):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=False))
        blocks = []
        ks_list, e_list, d_list = pheno["ks"], pheno["e"], pheno["d"]
        li = 0
        strides = [1, 2, 2, 2, 1]
        for stage in range(5):
            c_in = MNV3_BASE_CH[stage + 1]
            c_out = MNV3_BASE_CH[stage + 2] if stage + 2 < len(MNV3_BASE_CH) else MNV3_BASE_CH[-1]
            for blk in range(int(d_list[stage])):
                if li >= len(ks_list):
                    break
                ks, expand = int(ks_list[li]), int(e_list[li])
                li += 1
                bc = c_in if blk == 0 else c_out
                stride = strides[stage] if blk == 0 else 1
                blocks.append(InvertedResidual(bc, c_out, ks, expand, stride))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Conv2d(160, 960, 1, bias=False), nn.BatchNorm2d(960), nn.ReLU(inplace=False), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(960, 1280), nn.ReLU(inplace=False), nn.Linear(1280, nc))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def nats_nsc(channels, nc=100):
    c0, c1, c2, c3, c4 = [int(x) for x in channels]
    score = conv_psi(3, c0, 3)
    score += 2 * conv_psi(c0, c1, 3) + 3 * conv_psi(c1, c1, 3)
    if c0 != c1:
        score += conv_psi(c0, c1, 1)
    score += conv_psi(c1, c2, 3) + conv_psi(c2, c2, 3) + conv_psi(c1, c2, 1)
    score += 2 * conv_psi(c2, c3, 3) + 3 * conv_psi(c3, c3, 3)
    if c2 != c3:
        score += conv_psi(c2, c3, 1)
    score += conv_psi(c3, c4, 3) + conv_psi(c4, c4, 3) + conv_psi(c3, c4, 1)
    score += 5 * conv_psi(c4, c4, 3)
    score += psi_mp(nc, c4)
    return score


def mnv3_nsc(pheno):
    score = conv_psi(3, 16, 3)
    li = 0
    for stage in range(5):
        c_in = MNV3_BASE_CH[stage + 1]
        c_out = MNV3_BASE_CH[stage + 2] if stage + 2 < len(MNV3_BASE_CH) else MNV3_BASE_CH[-1]
        for blk in range(int(pheno["d"][stage])):
            if li >= len(pheno["ks"]):
                break
            ks, expand = int(pheno["ks"][li]), int(pheno["e"][li])
            li += 1
            bc = c_in if blk == 0 else c_out
            mid = bc * expand
            if expand != 1:
                score += psi_mp(mid, bc)
            score += mid * psi_mp(1, ks * ks)
            score += psi_mp(c_out, mid)
    score += psi_mp(960, 160) + psi_mp(1280, 960) + psi_mp(1000, 1280)
    return score


def conv_wpca(model, x):
    modules = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    return compute_wpca_hooks(model, x, modules, lambda m, d: m(d))


def run_nats(args):
    log("=== NATS-SSS C100 real CIFAR-100 run ===")
    x, y = load_image_batch(CIFAR100_BATCH, 100)
    rows = []
    with open(ROOT / "nats_sss_official_90ep.csv") as f:
        for r in csv.DictReader(f):
            channels = [int(v) for v in r["arch_str"].split(":")]
            rows.append({"idx": int(r["arch_index"]), "channels": channels, "acc": float(r["cifar100_test_acc"])})
    values = [r["acc"] for r in rows]
    subset, stats = stratified_indices(values)
    full_subset = list(subset)
    write_json(SUBSETS / "nats_sss_c100_15pct_indices.json", {"seed": SEED, "frac": SUBSET_FRAC, "indices": full_subset, "bin_stats": stats})
    if args.limit:
        subset = subset[: args.limit]
    elif args.smoke:
        subset = subset[:2]
    write_json(LOGS / "nats_batch_meta.json", {"batch": str(CIFAR100_BATCH), "labels_path": str(CIFAR100_LABELS), "input_shape": list(x.shape), "labels": y.detach().cpu().tolist(), "device": DEVICE})
    out_path = SCORES / "nats_sss_c100_realdata_scores.json"
    results = {str(r["idx"]): r for r in load_existing(out_path)}
    timing_rows = []
    for pos, row0_idx in enumerate(subset):
        base = rows[row0_idx]
        idx = str(base["idx"])
        row = results.get(idx, {"idx": base["idx"], "channels": base["channels"], "acc": base["acc"]})
        row["zerolm_status"] = "unavailable"
        row["zerolm_reason"] = "ZeroLM is transformer-specific; CNN version not evaluated"
        if all(k in row for k in ["nsc_mp", "wpca", "snip", "gradnorm", "n_params"]):
            continue
        log(f"[NATS C100] {pos+1}/{len(subset)} idx={idx}")
        model = NATSBenchSSS(base["channels"], nc=100).to(DEVICE)
        method_fns = [
            ("nsc_mp", lambda: nats_nsc(base["channels"], 100)),
            ("pca_wpca", lambda: conv_wpca(model, x)),
            ("snip_gradnorm", lambda: snip_gradnorm_from_loss(model, lambda: F.cross_entropy(model(x), y))),
            ("params", lambda: count_params(model)),
        ]
        for name, fn in method_fns:
            try:
                val, sec = time_call(fn)
                if name == "pca_wpca":
                    row["pca"], row["wpca"] = float(val[0]), float(val[1])
                elif name == "snip_gradnorm":
                    row["snip"], row["gradnorm"] = float(val[0]), float(val[1])
                elif name == "params":
                    row["n_params"] = int(val)
                else:
                    row[name] = float(val)
                row.pop(f"{name}_error", None)
                row[f"time_{name}_sec"] = sec
                timing_rows.append({"benchmark": "nats_sss_c100", "arch_id": idx, "method": name, "time_sec": sec, "status": "ok"})
            except Exception as e:
                row[f"{name}_error"] = repr(e)
                timing_rows.append({"benchmark": "nats_sss_c100", "arch_id": idx, "method": name, "time_sec": None, "status": "failed", "error": repr(e)})
                traceback.print_exc()
        del model
        clear_cuda()
        results[idx] = row
        write_json(out_path, list(results.values()))
    write_json(TIMING / "nats_sss_c100_per_arch_timing.json", timing_rows)


def run_mnv3(args):
    log("=== MNV3 r=224 real ImageNet run ===")
    x, y = load_image_batch(IMAGENET_BATCH, 1000)
    data = read_json(ROOT / "dataset/mnv3/mnv3_data.json")
    rows = [r for r in data if int(r["phenotype"].get("r", -1)) == 224]
    values = [r["test_acc"] for r in rows]
    subset, stats = stratified_indices(values)
    full_subset = list(subset)
    write_json(SUBSETS / "mnv3_r224_15pct_indices.json", {"seed": SEED, "frac": SUBSET_FRAC, "indices": full_subset, "bin_stats": stats})
    if args.limit:
        subset = subset[: args.limit]
    elif args.smoke:
        subset = subset[:2]
    write_json(LOGS / "mnv3_batch_meta.json", {"batch": str(IMAGENET_BATCH), "labels_path": str(IMAGENET_LABELS), "input_shape": list(x.shape), "labels": y.detach().cpu().tolist(), "device": DEVICE, "model_note": "repo-local MobileNetV3 proxy network matching old runner structure; #Params uses original dataset params"})
    out_path = SCORES / "mnv3_r224_realdata_scores.json"
    results = {str(r["local_idx"]): r for r in load_existing(out_path)}
    timing_rows = []
    for pos, local_idx in enumerate(subset):
        base = rows[local_idx]
        idx = str(local_idx)
        row = results.get(idx, {"local_idx": int(local_idx), "test_acc": float(base["test_acc"]), "params": base["params"], "phenotype": base["phenotype"]})
        row["n_params"] = float(base["params"])
        row["zerolm_status"] = "unavailable"
        row["zerolm_reason"] = "ZeroLM is transformer-specific; CNN version not evaluated"
        if all(k in row for k in ["nsc_mp", "wpca", "snip", "gradnorm", "n_params"]):
            continue
        log(f"[MNV3 r=224] {pos+1}/{len(subset)} local_idx={idx}")
        model = MNV3ProxyNet(base["phenotype"], nc=1000).to(DEVICE)
        method_fns = [
            ("nsc_mp", lambda: mnv3_nsc(base["phenotype"])),
            ("pca_wpca", lambda: conv_wpca(model, x)),
            ("snip_gradnorm", lambda: snip_gradnorm_from_loss(model, lambda: F.cross_entropy(model(x), y))),
            ("params", lambda: float(base["params"])),
        ]
        for name, fn in method_fns:
            try:
                val, sec = time_call(fn)
                if name == "pca_wpca":
                    row["pca"], row["wpca"] = float(val[0]), float(val[1])
                elif name == "snip_gradnorm":
                    row["snip"], row["gradnorm"] = float(val[0]), float(val[1])
                elif name == "params":
                    row["n_params"] = int(val)
                else:
                    row[name] = float(val)
                row.pop(f"{name}_error", None)
                row[f"time_{name}_sec"] = sec
                timing_rows.append({"benchmark": "mnv3_r224", "arch_id": idx, "method": name, "time_sec": sec, "status": "ok"})
            except Exception as e:
                row[f"{name}_error"] = repr(e)
                timing_rows.append({"benchmark": "mnv3_r224", "arch_id": idx, "method": name, "time_sec": None, "status": "failed", "error": repr(e)})
                traceback.print_exc()
        del model
        clear_cuda()
        results[idx] = row
        write_json(out_path, list(results.values()))
    write_json(TIMING / "mnv3_r224_per_arch_timing.json", timing_rows)


def summarize_tables():
    specs = [
        ("BERT corrected", SCORES / "flexibert_corrected_realdata_scores.json", "glue", {"NSC-MP": "nsc_mp_flat", "ZeroLM": "zerolm", "W-PCA": "wpca", "SNIP": "snip", "GradNorm": "gradnorm", "#Params": "n_params"}, "standard"),
        ("GPT-2", SCORES / "gpt2_realdata_scores.json", "neg_ppl", {"NSC-MP": "nsc_mp", "ZeroLM": "zerolm", "W-PCA": "wpca", "SNIP": "snip", "GradNorm": "gradnorm", "#Params": "n_params"}, "standard"),
        ("ViT Tiny", SCORES / "autoformer_tiny_realdata_scores.json", "acc", {"NSC-MP": "nsc_mp", "ZeroLM": "zerolm_init_expected", "W-PCA": "wpca", "SNIP": "snip", "GradNorm": "gradnorm", "#Params": "n_params"}, "standard"),
        ("ViT Small", SCORES / "autoformer_small_realdata_scores.json", "acc", {"NSC-MP": "nsc_mp", "ZeroLM": "zerolm_init_expected", "W-PCA": "wpca", "SNIP": "snip", "GradNorm": "gradnorm", "#Params": "n_params"}, "standard"),
        ("ViT Base", SCORES / "autoformer_base_realdata_scores.json", "acc", {"NSC-MP": "nsc_mp", "ZeroLM": "zerolm_init_expected", "W-PCA": "wpca", "SNIP": "snip", "GradNorm": "gradnorm", "#Params": "n_params"}, "standard"),
        ("NATS-SSS C100", SCORES / "nats_sss_c100_realdata_scores.json", "acc", {"NSC-MP": "nsc_mp", "ZeroLM": None, "W-PCA": "wpca", "SNIP": "snip", "GradNorm": "gradnorm", "#Params": "n_params"}, "proxy"),
        ("MNV3 r=224", SCORES / "mnv3_r224_realdata_scores.json", "test_acc", {"NSC-MP": "nsc_mp", "ZeroLM": None, "W-PCA": "wpca", "SNIP": "snip", "GradNorm": "gradnorm", "#Params": "n_params"}, "proxy"),
    ]
    avail_rows, corr_rows, per_arch_time_rows = [], [], []
    for dataset, path, gt_key, methods, default_status in specs:
        rows = load_existing(path)
        for m, key in methods.items():
            if key is None:
                avail_rows.append({"dataset": dataset, "method": m, "status": "unavailable", "total_rows": len(rows), "have_scores": 0, "failed_rows": 0, "score_key": "", "file": str(path), "note": "ZeroLM is transformer-specific; CNN version not evaluated"})
                corr_rows.append({"dataset": dataset, "method": m, "kendall_tau": None, "spearman_rho": None, "n": 0, "score_key": "", "status": "unavailable"})
                continue
            have = [r for r in rows if key in r and r.get(key) is not None and np.isfinite(float(r.get(key)))]
            errs = [r for r in rows if any(k.endswith("_error") for k in r)]
            tau, rho = (None, None)
            if have:
                tau, rho = safe_corr([r[key] for r in have], [r[gt_key] for r in have])
            status = default_status
            if len(have) == 0:
                status = "missing"
            elif errs:
                status = "invalid" if len(have) < len(rows) else default_status
            if dataset in ("NATS-SSS C100", "MNV3 r=224") and m in ("NSC-MP", "#Params"):
                status = "standard"
            avail_rows.append({"dataset": dataset, "method": m, "status": status, "total_rows": len(rows), "have_scores": len(have), "failed_rows": len(errs), "score_key": key, "file": str(path), "note": ""})
            corr_rows.append({"dataset": dataset, "method": m, "kendall_tau": tau, "spearman_rho": rho, "n": len(have), "score_key": key, "status": status})
        for r in rows:
            arch_id = r.get("arch_id", r.get("arch_key", r.get("idx", r.get("local_idx", ""))))
            for k, v in r.items():
                if k.startswith("time_") and k.endswith("_sec") and v is not None:
                    per_arch_time_rows.append({"dataset": dataset, "arch_id": arch_id, "time_key": k, "method_group": k[len("time_"):-len("_sec")], "time_sec": v})
    write_json(TABLES / "availability.json", avail_rows)
    write_json(TABLES / "correlation_summary.json", corr_rows)
    with open(TABLES / "availability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(avail_rows[0].keys()))
        w.writeheader()
        w.writerows(avail_rows)
    with open(TABLES / "correlation_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()))
        w.writeheader()
        w.writerows(corr_rows)
    with open(TIMING / "per_arch_method_timing.csv", "w", newline="") as f:
        fields = ["dataset", "arch_id", "time_key", "method_group", "time_sec"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_arch_time_rows)
    totals = {}
    for r in per_arch_time_rows:
        key = (r["dataset"], r["method_group"])
        totals.setdefault(key, {"dataset": r["dataset"], "method_group": r["method_group"], "n_arch": 0, "time_total_sec": 0.0, "time_per_arch_sec": 0.0})
        totals[key]["n_arch"] += 1
        totals[key]["time_total_sec"] += float(r["time_sec"])
    for rec in totals.values():
        rec["time_per_arch_sec"] = rec["time_total_sec"] / max(1, rec["n_arch"])
    total_rows = list(totals.values())
    with open(TIMING / "per_method_total_timing.csv", "w", newline="") as f:
        fields = ["dataset", "method_group", "n_arch", "time_total_sec", "time_per_arch_sec"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(total_rows)
    scale_rows = [r for r in corr_rows if r["dataset"] in ("ViT Tiny", "ViT Small", "ViT Base")]
    with open(TABLES / "correlation_summary_autoformer_scales.csv", "w", newline="") as f:
        fields = ["dataset", "method", "kendall_tau", "spearman_rho", "n", "score_key", "status"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(scale_rows)
    write_json(TABLES / "correlation_summary_autoformer_scales.json", scale_rows)
    log(f"Wrote {TABLES / 'availability.csv'} and correlation_summary.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["all", "flexibert", "gpt2", "autoformer", "nats", "mnv3", "tables"], default="all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--autoformer-scales", nargs="+", choices=["tiny", "small", "base"], default=["tiny"])
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    log(f"OUT={OUT}")
    log(f"DEVICE={DEVICE}")
    log(f"smoke={args.smoke} limit={args.limit}")
    if args.bench in ("all", "flexibert"):
        run_flexibert(args)
    if args.bench in ("all", "gpt2"):
        run_gpt2(args)
    if args.bench in ("all", "autoformer"):
        run_autoformer(args)
    if args.bench in ("all", "nats"):
        run_nats(args)
    if args.bench in ("all", "mnv3"):
        run_mnv3(args)
    if args.bench in ("all", "tables"):
        summarize_tables()


if __name__ == "__main__":
    main()
