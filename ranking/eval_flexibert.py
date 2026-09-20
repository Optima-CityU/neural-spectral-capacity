#!/usr/bin/env python3
"""Evaluate zero-cost proxies and NSC-MP on the FlexiBERT benchmark
(per-architecture hidden_size, per-layer FFN dims).

Usage:
  python eval_flexibert.py [--limit N]
"""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import sys, json, time, math, gc, copy, argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
from scipy import integrate
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
FLEXIBERT_DIR = Path(os.environ.get("FLEXIBERT_DIR", "FlexiBERT"))
sys.path.insert(0, str(FLEXIBERT_DIR))
from configuration_electra import ElectraConfig
from modeling_electra import ElectraModel, ElectraLayer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════
#  NSC-MP (analytical, Marchenko-Pastur)
# ═══════════════════════════════════════════════════════════════

_PSI_CACHE = {}

def mp_density(x, gamma):
    lp = (1 + np.sqrt(gamma)) ** 2
    lm = (1 - np.sqrt(gamma)) ** 2
    if x < lm or x > lp:
        return 0.0
    return np.sqrt((lp - x) * (x - lm)) / (2 * np.pi * gamma * x)

def psi_mp(m, n, sigma=None):
    if sigma is None:
        sigma = np.sqrt(2.0 / (m + n))
    key = (m, n, round(sigma, 10))
    if key in _PSI_CACHE:
        return _PSI_CACHE[key]
    if m < n:
        m, n = n, m
    if n == 0 or m == 0:
        _PSI_CACHE[key] = 0.0
        return 0.0
    gamma = n / m
    lp = (1 + np.sqrt(gamma)) ** 2
    lm = (1 - np.sqrt(gamma)) ** 2
    s2m = sigma ** 2 * m
    def integrand(x):
        d = mp_density(x, gamma)
        return np.log(1 + s2m * x) * d if d > 0 else 0.0
    val = n * integrate.quad(integrand, lm, lp, limit=200)[0]
    _PSI_CACHE[key] = val
    return val

def compute_nsc_mp(layers_cfg, H, D_FF_global):
    """Returns (flat_sum, harmonic) scores."""
    layer_psis = []
    for lc in layers_cfg:
        op = lc["operation_type"]
        nff = lc.get("num_feed_forward", 1)
        D_FF = lc.get("feed_forward_dimension", D_FF_global)
        ff_dim = D_FF // nff

        attn_psis = []
        if op == "SA":
            attn_psis.append(psi_mp(3 * H, H))
            attn_psis.append(psi_mp(H, H))
        elif op == "LT":
            attn_psis.append(psi_mp(H, H))
            attn_psis.append(psi_mp(H, H))
        elif op == "DSC":
            attn_psis.append(psi_mp(H, H))

        ffn_psis = []
        for _ in range(nff):
            ffn_psis.append(psi_mp(ff_dim, H))
            ffn_psis.append(psi_mp(H, ff_dim))
        layer_psis.append({"attn": attn_psis, "ffn": ffn_psis})

    flat = sum(v for lp in layer_psis for v in lp["attn"] + lp["ffn"])
    hm_sum = 0.0
    for lp in layer_psis:
        vals = lp["attn"] + lp["ffn"]
        if not vals:
            continue
        n = len(vals)
        hm = n / sum(1.0 / (v + 1e-30) for v in vals)
        hm_sum += n * hm
    return flat, hm_sum

# ═══════════════════════════════════════════════════════════════
#  Real text inputs (LPZero-aligned: wikitext, bs=128, sl=512)
# ═══════════════════════════════════════════════════════════════

_REAL_INPUTS_CACHE = None

def _prepare_real_inputs(bs=128, sl=512):
    global _REAL_INPUTS_CACHE
    if _REAL_INPUTS_CACHE is not None:
        return _REAL_INPUTS_CACHE
    from transformers import ElectraTokenizerFast
    from datasets import load_dataset
    print("  Loading wikitext-2 + ElectraTokenizer ...", flush=True)
    tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-small-discriminator')
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1')
    texts = [x['text'] for x in ds['train'] if len(x['text'].strip()) > 50]
    enc = tokenizer(texts[:bs], truncation=True, padding='max_length',
                    max_length=sl, return_tensors='pt')
    inputs = {k: v.to(DEVICE) for k, v in enc.items()}
    print(f"  Real inputs ready: input_ids {inputs['input_ids'].shape}", flush=True)
    _REAL_INPUTS_CACHE = inputs
    return inputs

def _make_inputs(cfg, bs=128, sl=512):
    return _prepare_real_inputs(bs=bs, sl=sl)

# ═══════════════════════════════════════════════════════════════
#  Helper: get_layer_metric_array
# ═══════════════════════════════════════════════════════════════

def get_layer_metric_array(net, metric, mode='param'):
    metric_array = []
    for layer in net.modules():
        if isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))
    return metric_array

# ═══════════════════════════════════════════════════════════════
#  ZCP Methods (LPZero-aligned implementations)
# ═══════════════════════════════════════════════════════════════

def compute_snip(model, cfg):
    model.zero_grad()
    inputs = _make_inputs(cfg)
    output = model(**inputs).last_hidden_state
    output.backward(torch.ones_like(output))
    metric_array = []
    for layer in model.modules():
        if isinstance(layer, ElectraLayer):
            for sub in layer.operation.modules():
                if isinstance(sub, nn.Linear) and sub.weight.grad is not None:
                    metric_array.append(torch.abs(sub.weight * sub.weight.grad).sum())
            for sub in layer.intermediate.modules():
                if isinstance(sub, nn.Linear) and sub.weight.grad is not None:
                    metric_array.append(torch.abs(sub.weight * sub.weight.grad).sum())
            for sub in layer.output.modules():
                if isinstance(sub, nn.Linear) and sub.weight.grad is not None:
                    metric_array.append(torch.abs(sub.weight * sub.weight.grad).sum())
    return sum(v.item() for v in metric_array)

def compute_gradnorm(model, cfg):
    model.zero_grad()
    inputs = _make_inputs(cfg)
    output = model(**inputs).last_hidden_state
    output.backward(torch.ones_like(output))
    score = 0.0
    for layer in model.modules():
        if isinstance(layer, nn.Linear) and layer.weight.grad is not None:
            score += layer.weight.grad.float().norm().item()
    return score

def compute_fisher(model, cfg):
    model.zero_grad()
    inputs = _make_inputs(cfg)
    output = model(**inputs).last_hidden_state
    output.backward(torch.ones_like(output))
    score = 0.0
    for layer in model.modules():
        if isinstance(layer, nn.Linear) and layer.weight.grad is not None:
            score += (layer.weight.grad.float() ** 2).sum().item()
    return score

def compute_synflow(model, cfg):
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    signs = linearize(model)
    model.zero_grad()
    model.double()
    inputs = _make_inputs(cfg)
    outputs = model(**inputs).last_hidden_state
    outputs.sum().backward()

    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        return torch.zeros_like(layer.weight)
    grads_abs = get_layer_metric_array(model, synflow, mode='param')
    nonlinearize(model, signs)
    model.float()
    return sum(torch.sum(g).item() for g in grads_abs)

def compute_log_synflow(model, cfg):
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    signs = linearize(model)
    model.zero_grad()
    model.double()
    inputs = _make_inputs(cfg)
    output = model(**inputs).last_hidden_state
    output.backward(torch.ones_like(output))

    def logsynflow(layer):
        if layer.weight.grad is not None:
            g = torch.abs(layer.weight.grad).clamp(min=1e-30)
            return layer.weight * torch.abs(torch.log(g))
        return torch.zeros_like(layer.weight)
    grads_abs = get_layer_metric_array(model, logsynflow, mode='param')
    nonlinearize(model, signs)
    model.float()
    return sum(torch.sum(g).item() for g in grads_abs)

def compute_grasp(model, cfg):
    weights = []
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            weights.append(layer.weight)
            layer.weight.requires_grad_(True)
    model.zero_grad()
    inputs = _make_inputs(cfg)
    outputs = model(**inputs).last_hidden_state
    loss = outputs.sum()
    grad_w_p = torch.autograd.grad(loss, weights, allow_unused=True)
    grad_w = list(grad_w_p)
    outputs = model(**inputs).last_hidden_state
    loss = outputs.sum()
    grad_f = torch.autograd.grad(loss, weights, create_graph=True, allow_unused=True)
    z, count = 0, 0
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            if grad_w[count] is not None:
                z += (grad_w[count].data * grad_f[count]).sum()
            count += 1
    z.backward()
    score = 0.0
    count = 0
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            if layer.weight.grad is not None:
                score -= (layer.weight.data * layer.weight.grad).sum().item()
            count += 1
    return score

def compute_naswot(model, cfg):
    model.zero_grad()
    inputs = _make_inputs(cfg)
    output = model(**inputs).last_hidden_state
    output.backward(torch.ones_like(output))
    jacobs = model.embeddings.position_embeddings.weight.grad.detach()
    SL = inputs['input_ids'].shape[1]
    jacobs = jacobs[:SL]
    try:
        jacob = torch.transpose(jacobs, 0, 1).reshape(jacobs.size(1), -1).cpu().numpy()
        corr = np.corrcoef(jacob)
        v, _ = np.linalg.eig(corr)
        k = 1e-5
        return float(-np.sum(np.log(v + k) + 1.0 / (v + k)))
    except Exception:
        return 0.0

def compute_jacobian_cosine(model, cfg):
    model.zero_grad()
    inputs = _make_inputs(cfg)
    output = model(**inputs).last_hidden_state
    output.backward(torch.ones_like(output))
    jacobs = model.embeddings.position_embeddings.weight.grad.detach()
    SL = inputs['input_ids'].shape[1]
    jacobs = jacobs[:SL]
    try:
        from sklearn.metrics import pairwise_distances
        jacob = torch.transpose(jacobs, 0, 1).reshape(jacobs.size(1), -1).cpu().numpy()
        norm = np.linalg.norm(jacob, axis=1)
        normed = jacob / (norm[:, None] + 1e-10)
        cosines = (-pairwise_distances(normed, metric="cosine") + 1) - np.identity(normed.shape[0])
        summed = np.sum(np.power(np.absolute(cosines.flatten()), 1.0 / 20)) / 2
        return float(1 - (1 / (pow(cosines.shape[0], 2) - cosines.shape[0]) * summed))
    except Exception:
        return 0.0

def compute_pca_and_wpca(model, cfg):
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
    pca_sum = 0.0
    for act in activations:
        a = act.reshape(-1, act.shape[-1]).float()
        if a.shape[0] < 2 or a.shape[1] < 2:
            continue
        try:
            a_centered = a - a.mean(dim=0, keepdim=True)
            s = torch.linalg.svdvals(a_centered.to(DEVICE))
            cumvar = torch.cumsum(s ** 2, 0) / (s ** 2).sum()
            pca_dim = (cumvar < 0.99).sum().item() + 1
            pca_sum += pca_dim
        except Exception:
            pass
    n_params = sum(p.numel() for p in model.parameters())
    return pca_sum, float(n_params) * pca_sum

def compute_synaptic_diversity(model, cfg):
    model.zero_grad()
    inputs = _make_inputs(cfg)
    output = model(**inputs).last_hidden_state
    output.backward(torch.ones_like(output))
    metric_array = []
    for layer in model.modules():
        if isinstance(layer, ElectraLayer):
            for sub in layer.operation.modules():
                if isinstance(sub, nn.Linear) and sub.weight is not None and sub.weight.grad is not None:
                    metric_array.append(torch.abs(
                        torch.norm(sub.weight, 'nuc') * torch.norm(sub.weight.grad, 'nuc')))
    return sum(torch.nansum(v).item() for v in metric_array)

def compute_activation_distance(model, cfg):
    model.train()
    activation_outputs = []
    def activation_hook(module, input, output):
        activation_outputs.append(output)
    hooks = []
    for layer in model.modules():
        if isinstance(layer, ElectraLayer):
            hooks.append(layer.intermediate.intermediate_act_fn.register_forward_hook(activation_hook))
    with torch.no_grad():
        inputs = _make_inputs(cfg)
        model(**inputs).last_hidden_state
    for h in hooks:
        h.remove()
    summed = torch.tensor(0.0, device=DEVICE)
    for out in activation_outputs:
        if isinstance(out, tuple):
            out = out[0]
        out = out[0].view(out.size(1), -1)
        x = (out > 0).float()
        K = x @ x.t()
        K2 = (1.0 - x) @ (1.0 - x).t()
        summed += torch.nansum(K + K2)
    return summed.detach().item()

def compute_head_importance(model, cfg):
    model.zero_grad()
    inputs = _make_inputs(cfg)
    output = model(**inputs).last_hidden_state
    output.backward(torch.ones_like(output))
    metric_array = []
    for layer in model.modules():
        if isinstance(layer, ElectraLayer):
            for sub in layer.operation.operation.modules():
                if isinstance(sub, nn.Linear) and sub.weight is not None and sub.weight.grad is not None:
                    if sub.weight.shape[0] >= 128:
                        metric_array.append(torch.abs(sub.weight.data * sub.weight.grad).sum())
    return sum(v.item() for v in metric_array)

def compute_attention_confidence(model, cfg):
    model.zero_grad()
    inputs = _make_inputs(cfg)
    output = model(**inputs).last_hidden_state
    output.backward(torch.ones_like(output))
    head_outputs = []
    hooks = []
    def head_hook(module, input, output):
        head_outputs.append(output)
    for layer in model.modules():
        if isinstance(layer, ElectraLayer):
            sub = layer.operation.operation
            for attr in ('query', 'key', 'value'):
                if hasattr(sub, attr):
                    getattr(sub, attr).register_forward_hook(head_hook)
    with torch.no_grad():
        model(**inputs)
    metric_array = []
    for out in head_outputs:
        if isinstance(out, tuple):
            out = out[0]
        metric_array.append(torch.mean(torch.max(out, 1)[0]))
    return sum(torch.nansum(v).item() for v in metric_array)

# ═══════════════════════════════════════════════════════════════
#  Model builder — per-arch hidden size, per-layer FFN dims
# ═══════════════════════════════════════════════════════════════

def build_model(arch):
    hpo = arch["hparams"]["model_hparam_overrides"]
    nas = hpo["nas_config"]
    layers = nas["encoder_layers"]
    hidden = hpo["hidden_size"]
    intermediate = layers[0]["feed_forward_dimension"]

    cfg = ElectraConfig(
        vocab_size=30522, hidden_size=hidden,
        num_hidden_layers=len(layers),
        num_attention_heads=layers[0]["num_operation_heads"],
        intermediate_size=intermediate, nas_config=nas,
    )
    torch.manual_seed(42)
    model = ElectraModel(cfg).to(DEVICE)
    return model, cfg

# ═══════════════════════════════════════════════════════════════
#  ZCP method registry
# ═══════════════════════════════════════════════════════════════

ZCP_METHODS = [
    (('pca', 'wpca'),    compute_pca_and_wpca,           False),
    ('snip',             compute_snip,                    True),
    ('gradnorm',         compute_gradnorm,                True),
    ('fisher',           compute_fisher,                  True),
    ('synflow',          compute_synflow,                 True),
    ('log_synflow',      compute_log_synflow,             True),
    ('grasp',            compute_grasp,                   True),
    ('naswot',           compute_naswot,                  True),
    ('jacobian_cosine',  compute_jacobian_cosine,         True),
    ('head_importance',  compute_head_importance,         True),
    ('attn_confidence',  compute_attention_confidence,    True),
    ('syn_diversity',    compute_synaptic_diversity,      True),
    ('act_distance',     compute_activation_distance,     True),
]

# ═══════════════════════════════════════════════════════════════
#  Main loop
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    bench_path = Path(os.environ.get("FLEXIBERT_DIR", "FlexiBERT")) / "BERT_benchmark.json"
    with open(bench_path) as f:
        bench = json.load(f)
    if args.limit > 0:
        bench = bench[:args.limit]

    out_dir = Path(os.environ.get("NSC_OUT", "outputs"))
    out_dir.mkdir(exist_ok=True)
    suffix = f"_first{args.limit}" if args.limit > 0 else ""
    out_file = out_dir / f"corrected{suffix}.json"

    existing = {}
    if out_file.exists():
        with open(out_file) as f:
            for r in json.load(f):
                existing[r['arch_id']] = r
        print(f"Loaded {len(existing)} existing results", flush=True)

    results = []
    t0 = time.time()

    for i, arch in enumerate(bench):
        glue_data = arch.get('metrics', arch.get('scores', {}))
        glue = glue_data.get('glue_avg', glue_data.get('glue'))
        if glue is None:
            continue
        arch_id = arch.get('id', i)

        if arch_id in existing and 'nsc_mp_flat' in existing[arch_id]:
            results.append(existing[arch_id])
            continue

        try:
            hpo = arch["hparams"]["model_hparam_overrides"]
            nas_cfg = hpo["nas_config"]["encoder_layers"]

            H = hpo["hidden_size"]
            D_FF_global = nas_cfg[0].get("feed_forward_dimension", 1024)
            nsc_flat, nsc_hm = compute_nsc_mp(nas_cfg, H=H, D_FF_global=D_FF_global)

            model_tmp, cfg_tmp = build_model(arch)
            r = {
                'arch_id': arch_id,
                'glue': float(glue),
                'n_params': sum(p.numel() for p in model_tmp.parameters()),
                'n_layers': cfg_tmp.num_hidden_layers,
                'hidden_size': cfg_tmp.hidden_size,
                'nsc_mp_flat': nsc_flat,
                'nsc_mp_harmonic': nsc_hm,
            }
            del model_tmp

            for keys, fn, needs_fresh in ZCP_METHODS:
                model, cfg = build_model(arch)
                t_start = time.time()
                try:
                    val = fn(model, cfg)
                except Exception as e:
                    print(f"  [{i}] {keys} ERR: {e}", flush=True)
                    val = (0.0, 0.0) if isinstance(keys, tuple) else 0.0
                elapsed_ms = (time.time() - t_start) * 1000
                del model
                if isinstance(keys, tuple):
                    for k, v in zip(keys, val):
                        r[k] = v
                    r[f'time_{"_".join(keys)}'] = elapsed_ms
                else:
                    r[keys] = val
                    r[f'time_{keys}'] = elapsed_ms

                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

            results.append(r)

        except Exception as e:
            print(f"  [{i}] ERR: {e}", flush=True)
            import traceback; traceback.print_exc()

        if (i + 1) % 10 == 0 or i == len(bench) - 1:
            elapsed = time.time() - t0
            rate = elapsed / max(len(results) - len(existing), 1)
            eta = rate * (len(bench) - i - 1)
            print(f"  [{i+1}/{len(bench)}] {len(results)} done "
                  f"({elapsed:.0f}s, ~{rate:.1f}s/arch, ETA {eta:.0f}s)", flush=True)
            with open(out_file, 'w') as f:
                json.dump(results, f, indent=2)

    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results → {out_file}", flush=True)

    # Print correlation table
    if results:
        glues = np.array([r['glue'] for r in results])
        labels = {
            'n_params': '#Params', 'nsc_mp_flat': 'NSC-MP flat', 'nsc_mp_harmonic': 'NSC-MP harmonic',
            'pca': 'PCA', 'wpca': 'W-PCA', 'snip': 'SNIP', 'gradnorm': 'GradNorm',
            'fisher': 'Fisher', 'synflow': 'SynFlow', 'log_synflow': 'LogSynflow',
            'grasp': 'GraSP', 'naswot': 'NASWOT', 'jacobian_cosine': 'Jacobian Cosine',
            'head_importance': 'Head Importance', 'attn_confidence': 'Attn. Confidence',
            'syn_diversity': 'Syn. Diversity', 'act_distance': 'Act. Distance',
        }
        print(f"\n{'='*70}")
        print(f"  per-arch h, per-layer ffn  N={len(results)}")
        print(f"{'='*70}")
        print(f"  {'Method':<22s} {'Spearman ρ':>12s} {'Kendall τ':>12s}")
        print(f"  {'─'*48}")
        rows = []
        for key, label in labels.items():
            vals = np.array([float(r.get(key, 0)) for r in results])
            ok = np.isfinite(vals) & (vals != 0)
            if ok.sum() < 5:
                continue
            spr = spearmanr(vals[ok], glues[ok])[0]
            tau = kendalltau(vals[ok], glues[ok])[0]
            rows.append((label, spr, tau))
        rows.sort(key=lambda x: -x[1])
        for label, spr, tau in rows:
            print(f"  {label:<22s} {spr:>12.4f} {tau:>12.4f}")

        time_keys = sorted([k for k in results[0] if k.startswith('time_')])
        if time_keys:
            print(f"\n  {'Method':<22s} {'Mean ms':>10s}")
            print(f"  {'─'*34}")
            for tk in sorted(time_keys, key=lambda k: np.mean([r.get(k, 0) for r in results])):
                vals = [r.get(tk, 0) for r in results]
                print(f"  {tk.replace('time_',''):<22s} {np.mean(vals):>10.1f}")


if __name__ == "__main__":
    print(f"Device: {DEVICE}", flush=True)
    print(f"Setting: per-arch h, per-layer ffn", flush=True)
    main()
