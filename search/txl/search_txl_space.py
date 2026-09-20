#!/usr/bin/env python3
"""Transformer-XL search space on WikiText-103: exact NSC-MP solver and
proxy-driven evolutionary search baselines.

Search space:
  d_model  ∈ {128, 256, 384, 512, 640, 768, 1024}       (7)
  n_layer  ∈ {12, 14, 16, 18, 20, 24}                    (6)
  d_inner  ∈ 21 choices per layer (512 … 4096)            (21^L per config)
  n_head   ∈ {1, 2, 4, 8, 16}  (d_model divisible)       (up to 5 per layer)

Methods:
  NSC-MP                              bounded knapsack DP (exact, analytical)
  SynDiv, W-PCA, HeadImp, SoftmaxConf evolutionary search
  Random                              random sampling

Reference: Transformer-XL Base (d_model=410, n_layer=16, d_inner=2100, ~38.4M non-embedding params)
"""

import json, time, math, random, argparse, os
import numpy as np
from scipy import integrate

# ═══════════════════════════════════════════════════════════════════════════════
#  Search Space
# ═══════════════════════════════════════════════════════════════════════════════

D_MODEL = [128, 256, 384, 512, 640, 768, 1024]
N_LAYER = [12, 14, 16, 18]
D_INNER = (list(range(512, 2049, 128))
           + list(range(2304, 4097, 256)))          # 21 choices
N_HEAD  = [1, 2, 4, 8, 16]
SIGMA   = 0.02

def valid_heads(dm):
    return [h for h in N_HEAD if dm % h == 0]

def compute_params(dm, nl, di_list):
    """Non-embedding params:  per layer = 4·dm² + 9·dm + di·(2·dm+1)."""
    fixed = nl * (4 * dm * dm + 9 * dm)
    var   = sum(di * (2 * dm + 1) for di in di_list)
    return fixed + var

def search_space_size():
    total = 0
    for dm in D_MODEL:
        kh = len(valid_heads(dm))
        per_layer = len(D_INNER) * kh
        for nl in N_LAYER:
            total += per_layer ** nl
    return total

def default_nhead(dm):
    vh = valid_heads(dm)
    for h in sorted(vh, reverse=True):
        if dm // h >= 32:
            return h
    return vh[0]


# ═══════════════════════════════════════════════════════════════════════════════
#  NSC-MP  (Marchenko-Pastur analytical)
# ═══════════════════════════════════════════════════════════════════════════════

def _mp_density(x, gamma):
    lp = (1 + np.sqrt(gamma)) ** 2
    lm = (1 - np.sqrt(gamma)) ** 2
    if x < lm or x > lp:
        return 0.0
    return np.sqrt((lp - x) * (x - lm)) / (2 * np.pi * gamma * x)

_PSI = {}

def psi_mp(m, n, sigma=SIGMA):
    key = (m, n, sigma)
    if key in _PSI:
        return _PSI[key]
    if m < n:
        m, n = n, m
    if n == 0 or m == 0:
        return 0.0
    gamma = n / m
    lp = (1 + np.sqrt(gamma)) ** 2
    lm = (1 - np.sqrt(gamma)) ** 2
    s2  = sigma ** 2 * m

    def integrand(x):
        d = _mp_density(x, gamma)
        return np.log(1 + s2 * x) * d if d > 0 else 0.0

    val = n * integrate.quad(integrand, lm, lp, limit=200)[0]
    _PSI[key] = val
    return val


def nsc_mp_sum(dm, nl, di_list):
    """NSC-MP with flat-sum aggregation (sensitive to d_inner)."""
    psi_qkv = psi_mp(3 * dm, dm)
    psi_o   = psi_mp(dm, dm)
    return sum(psi_qkv + psi_o + 2 * psi_mp(di, dm) for di in di_list)


# ═══════════════════════════════════════════════════════════════════════════════
#  NSC-MP  Exact Solver  (Bounded Knapsack DP)
# ═══════════════════════════════════════════════════════════════════════════════

def _feasible_pairs(target, tol):
    pairs = []
    for dm in D_MODEL:
        for nl in N_LAYER:
            min_p = compute_params(dm, nl, [D_INNER[0]]  * nl)
            max_p = compute_params(dm, nl, [D_INNER[-1]] * nl)
            if min_p <= target * (1 + tol) and max_p >= target * (1 - tol):
                pairs.append((dm, nl))
    return pairs


def _solve_dp(nl, min_sum, max_sum, di_psi):
    """Bounded Knapsack DP for the d_inner allocation sub-problem.

    State:  dp[s] = max Σψ(d_i) achievable using all `nl` layers with
            shifted d_inner sum = s  (where s = actual_sum - nl * d_min).
    Transition: for each layer, try every d_inner choice.

    Returns (best_total_psi, best_actual_sum, counts_dict) or None.
    """
    d_min = D_INNER[0]
    shifted = [d - d_min for d in D_INNER]
    s_min = max(0, min_sum - nl * d_min)
    s_max = max_sum - nl * d_min
    if s_max < 0:
        return None

    INF = float('-inf')
    dp = [INF] * (s_max + 1)
    bt = [None] * (s_max + 1)      # backtrack: which d_inner was chosen
    dp[0] = 0.0

    for layer in range(nl):
        new_dp = [INF] * (s_max + 1)
        new_bt = [None] * (s_max + 1)
        for s in range(s_max + 1):
            if dp[s] == INF:
                continue
            for idx, (di, sh) in enumerate(zip(D_INNER, shifted)):
                ns = s + sh
                if ns <= s_max:
                    val = dp[s] + di_psi[di]
                    if val > new_dp[ns]:
                        new_dp[ns] = val
                        new_bt[ns] = (s, idx)
        dp = new_dp
        bt = new_bt

    best_psi = INF
    best_s = -1
    for s in range(s_min, s_max + 1):
        if dp[s] > best_psi:
            best_psi = dp[s]
            best_s = s

    if best_psi == INF:
        return None

    # Backtrack to recover d_inner counts
    # Since we only keep the last layer's bt, we need full backtrack.
    # Re-run with full history for the winning sum.
    counts = {di: 0 for di in D_INNER}
    # Re-solve with backtrack storage for all layers
    dp2 = [INF] * (s_max + 1)
    # bt_full[layer][s] = (prev_s, di_index)
    bt_full = [[None] * (s_max + 1) for _ in range(nl)]
    dp2[0] = 0.0
    for layer in range(nl):
        new_dp = [INF] * (s_max + 1)
        for s in range(s_max + 1):
            if dp2[s] == INF:
                continue
            for idx, (di, sh) in enumerate(zip(D_INNER, shifted)):
                ns = s + sh
                if ns <= s_max:
                    val = dp2[s] + di_psi[di]
                    if val > new_dp[ns]:
                        new_dp[ns] = val
                        bt_full[layer][ns] = (s, idx)
        dp2 = new_dp

    s = best_s
    for layer in range(nl - 1, -1, -1):
        prev_s, idx = bt_full[layer][s]
        counts[D_INNER[idx]] += 1
        s = prev_s

    actual_sum = best_s + nl * d_min
    counts = {k: v for k, v in counts.items() if v > 0}
    return best_psi, actual_sum, counts


def knapsack_nsc_mp(target, tol=0.05):
    """
    Exact optimisation of NSC-MP (sum) over the full search space.

    For each feasible (d_model, n_layer) pair, the per-layer ψ_FFN(d_inner)
    contributions are independent and identical.  The d_inner allocation
    under a total-parameter budget is a Bounded Knapsack problem, solved
    exactly via DP in O(n_layer × budget_range × |D_INNER|) time.

    Complexity:  O(|D_MODEL| · |N_LAYER| · nl · range · 21)  ≈ milliseconds.
    """
    best = None

    for dm in D_MODEL:
        psi_qkv = psi_mp(3 * dm, dm)
        psi_o   = psi_mp(dm, dm)
        fixed_nsc_per_layer = psi_qkv + psi_o

        di_psi = {di: psi_mp(di, dm) for di in D_INNER}

        for nl in N_LAYER:
            fixed_params = nl * (4 * dm * dm + 9 * dm)
            cost_unit    = 2 * dm + 1

            var_upper = target * (1 + tol) - fixed_params
            var_lower = target * (1 - tol) - fixed_params
            if var_upper < 0:
                continue

            max_sum = int(var_upper / cost_unit)
            min_sum = max(0, math.ceil(max(0, var_lower) / cost_unit))

            if max_sum < nl * D_INNER[0]:
                continue
            max_sum = min(max_sum, nl * D_INNER[-1])
            min_sum = max(min_sum, nl * D_INNER[0])
            if min_sum > max_sum:
                continue

            result = _solve_dp(nl, min_sum, max_sum, di_psi)
            if result is None:
                continue

            total_ffn_psi, di_sum, counts = result
            nsc    = nl * fixed_nsc_per_layer + 2 * total_ffn_psi
            params = fixed_params + di_sum * cost_unit

            if best is None or nsc > best['nsc']:
                di_list = []
                for di in sorted(counts, reverse=True):
                    di_list.extend([di] * counts[di])
                best = dict(d_model=dm, n_layer=nl,
                            d_inner=di_list,
                            nsc=nsc, params=params)

    if best is None:
        return None

    dm, nl = best['d_model'], best['n_layer']
    nh = default_nhead(dm)
    best['n_head']  = [nh] * nl
    return best


# ═══════════════════════════════════════════════════════════════════════════════
#  Torch + Model class  (shared by all baseline ZCPs)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

VOCAB = 1000
BS, SL = 4, 64
N_SAMPLES_SYNDIV = 32


class _DecoderLayer(nn.Module):
    def __init__(self, dm, nh, di):
        super().__init__()
        self.nh = nh
        self.norm1 = nn.LayerNorm(dm)
        self.attn  = nn.MultiheadAttention(dm, nh, batch_first=True)
        self.norm2 = nn.LayerNorm(dm)
        self.fc_up   = nn.Linear(dm, di)
        self.fc_down = nn.Linear(di, dm)

    def forward(self, x, mask=None):
        h = self.norm1(x)
        ao, aw = self.attn(h, h, h, attn_mask=mask, need_weights=True)
        x = x + ao
        h = self.norm2(x)
        fh = torch.relu(self.fc_up(h))
        x = x + self.fc_down(fh)
        return x, ao, aw, fh


class _MiniDecoder(nn.Module):
    def __init__(self, dm, nl, di_list, nh_list):
        super().__init__()
        self.dm = dm
        self.embed = nn.Embedding(VOCAB, dm)
        self.layers = nn.ModuleList([
            _DecoderLayer(dm, nh_list[i], di_list[i]) for i in range(nl)])
        self.norm = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, VOCAB, bias=False)

    def forward(self, x, mask=None):
        h = self.embed(x)
        layer_out = []
        for layer in self.layers:
            h, ao, aw, fh = layer(h, mask=mask)
            layer_out.append((ao, aw, fh))
        return self.head(self.norm(h)), layer_out


# ═══════════════════════════════════════════════════════════════════════════════
#  Model-based ZCP evaluator  (SynDiv, W-PCA, HeadImp, SoftmaxConf)
# ═══════════════════════════════════════════════════════════════════════════════

def eval_model_zcps(dm, nl, di_list, nh_list, device, seed=42):
    """Build one model, compute 4 activation/gradient-based ZCPs.

    Returns dict: syndiv, wpca, head_imp, softmax_conf
    """
    n_params = compute_params(dm, nl, di_list)
    mask = torch.triu(torch.ones(SL, SL, device=device) * float('-inf'),
                      diagonal=1)
    torch.manual_seed(seed)
    model = _MiniDecoder(dm, nl, di_list, nh_list).to(device)
    scores = {}

    # ── SynDiv: binary activation diversity (Zhou et al., 2022) ──
    model.eval()
    torch.manual_seed(seed + 2)
    ids32 = torch.randint(0, VOCAB, (N_SAMPLES_SYNDIV, SL), device=device)
    with torch.no_grad():
        _, lo32 = model(ids32, mask)
    bin_acts = [(fh.detach() > 0).float().reshape(N_SAMPLES_SYNDIV, -1)
                for _, _, fh in lo32]
    codes = torch.cat(bin_acts, dim=1)
    s = torch.linalg.svdvals(codes.float())
    scores['syndiv'] = (s > 1e-5).sum().item() / N_SAMPLES_SYNDIV

    # ── W-PCA: activation PCA dim ──
    torch.manual_seed(seed + 3)
    ids4 = torch.randint(0, VOCAB, (BS, SL), device=device)
    with torch.no_grad():
        _, lo4 = model(ids4, mask)
    total_pca = 0
    for _, _, fh in lo4:
        H = fh.detach().reshape(-1, fh.shape[-1]).float()
        H_c = H - H.mean(0, keepdim=True)
        sv = torch.linalg.svdvals(H_c)
        denom = (sv ** 2).sum()
        if denom > 1e-10:
            cumvar = (sv ** 2).cumsum(0) / denom
            total_pca += int((cumvar < 0.99).sum().item()) + 1
        else:
            total_pca += H.shape[1]
    scores['wpca'] = float(n_params) * total_pca

    # ── HeadImp + SoftmaxConf: forward + backward ──
    model.train()
    model.zero_grad()
    torch.manual_seed(seed + 4)
    ids4b  = torch.randint(0, VOCAB, (BS, SL), device=device)
    labels = torch.randint(0, VOCAB, (BS, SL), device=device)
    logits, lo_fb = model(ids4b, mask)
    for ao, _, _ in lo_fb:
        ao.retain_grad()
    loss = nn.functional.cross_entropy(logits.view(-1, VOCAB), labels.view(-1))
    loss.backward()

    confs = [aw.detach().max(dim=-1).values.mean().item() for _, aw, _ in lo_fb]
    scores['softmax_conf'] = float(np.mean(confs))

    importance = 0.0
    for li, (ao, _, _) in enumerate(lo_fb):
        if ao.grad is None:
            continue
        nh = nh_list[li]; hd = dm // nh
        act_h  = ao.float().view(BS, SL, nh, hd)
        grad_h = ao.grad.float().view(BS, SL, nh, hd)
        importance += (act_h * grad_h).abs().sum().item()
    scores['head_imp'] = importance

    model.zero_grad()
    del model
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
#  Evolutionary Search
# ═══════════════════════════════════════════════════════════════════════════════

def _rand_arch(pairs, target, tol, tries=300):
    for _ in range(tries):
        dm, nl = random.choice(pairs)
        di = [random.choice(D_INNER) for _ in range(nl)]
        p = compute_params(dm, nl, di)
        if abs(p - target) / target <= tol:
            vh = valid_heads(dm)
            return dict(d_model=dm, n_layer=nl, d_inner=di,
                        n_head=[random.choice(vh) for _ in range(nl)],
                        params=p)
    return None


def _mutate(arch, pairs, target, tol):
    for _ in range(60):
        new = dict(arch)
        new['d_inner'] = list(arch['d_inner'])
        new['n_head']  = list(arch['n_head'])
        if random.random() < 0.08:
            dm, nl = random.choice(pairs)
            new['d_model'] = dm
            new['n_layer'] = nl
            new['d_inner'] = [random.choice(D_INNER) for _ in range(nl)]
            vh = valid_heads(dm)
            new['n_head'] = [random.choice(vh) for _ in range(nl)]
        else:
            nl = new['n_layer']
            for _ in range(random.randint(1, min(3, nl))):
                li  = random.randint(0, nl - 1)
                idx = D_INNER.index(new['d_inner'][li])
                idx = max(0, min(len(D_INNER) - 1,
                                 idx + random.choice([-3, -2, -1, 1, 2, 3])))
                new['d_inner'][li] = D_INNER[idx]
        p = compute_params(new['d_model'], new['n_layer'], new['d_inner'])
        if abs(p - target) / target <= tol:
            new['params'] = p
            return new
    return None


def evo_search_single(score_fn, target, tol, pop_sz, n_gen, tag,
                      verbose_freq=10):
    """Standard (μ+λ) EA for a single ZCP. pop_sz offspring per gen."""
    pairs = _feasible_pairs(target, tol)
    if not pairs:
        print(f"  [{tag}] No feasible pairs"); return None

    pop = []
    for _ in range(pop_sz * 10):
        a = _rand_arch(pairs, target, tol)
        if a:
            a['score'] = score_fn(a['d_model'], a['n_layer'], a['d_inner'])
            pop.append(a)
        if len(pop) >= pop_sz:
            break

    best   = max(pop, key=lambda x: x['score'])
    n_eval = len(pop)

    for gen in range(n_gen):
        offspring = []
        for _ in range(pop_sz):
            cands  = random.sample(pop, min(3, len(pop)))
            parent = max(cands, key=lambda x: x['score'])
            child  = _mutate(parent, pairs, target, tol)
            if child:
                child['score'] = score_fn(child['d_model'], child['n_layer'],
                                          child['d_inner'])
                offspring.append(child)
                n_eval += 1
        combined = pop + offspring
        combined.sort(key=lambda x: x['score'], reverse=True)
        pop = combined[:pop_sz]
        if pop[0]['score'] > best['score']:
            best = pop[0]
        if (gen + 1) % verbose_freq == 0:
            print(f"  [{tag}] gen {gen+1:3d}/{n_gen}  "
                  f"best={best['score']:.4g}  evals={n_eval}")

    best['n_eval'] = n_eval
    return best


def shared_model_search(target, tol, pop_sz=50, n_gen=98, seed=42):
    """Shared-model EA for 4 model-based ZCPs.

    Each generation: each method generates pop_sz offspring → deduplicate →
    build model once per unique arch → evaluate all 4 ZCPs.
    Total model builds ≈ pop_sz + pop_sz × n_gen ≈ 5000 (upper bound).
    """
    METHODS = ['syndiv', 'wpca', 'head_imp', 'softmax_conf']
    METHOD_LABELS = {
        'syndiv': 'SynDiv', 'wpca': 'W-PCA',
        'head_imp': 'HeadImp', 'softmax_conf': 'SoftmaxConf',
    }
    pairs = _feasible_pairs(target, tol)
    if not pairs:
        print("  No feasible pairs!"); return {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pops  = {m: [] for m in METHODS}
    bests = {m: None for m in METHODS}
    n_model_build = 0
    t_start = time.time()

    def _eval_and_count(arch):
        nonlocal n_model_build
        dm = arch['d_model']; nl = arch['n_layer']
        nh_list = [default_nhead(dm)] * nl
        scores = eval_model_zcps(dm, nl, arch['d_inner'], nh_list, device,
                                 seed=seed + n_model_build)
        n_model_build += 1
        if device == 'cuda' and n_model_build % 100 == 0:
            torch.cuda.empty_cache()
        return scores

    # ── Phase 1: shared initial population ──
    print(f"\n  Phase 1: init population ({pop_sz} archs, 4 model-based ZCPs)")
    init_archs = []
    for _ in range(pop_sz * 10):
        a = _rand_arch(pairs, target, tol)
        if a:
            init_archs.append(a)
        if len(init_archs) >= pop_sz:
            break

    for i, arch in enumerate(init_archs):
        scores = _eval_and_count(arch)
        for m in METHODS:
            entry = dict(arch); entry['score'] = scores[m]
            pops[m].append(entry)
            if bests[m] is None or scores[m] > bests[m]['score']:
                bests[m] = dict(entry)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"    init {i+1}/{pop_sz}  builds={n_model_build}  "
                  f"({elapsed:.0f}s, {n_model_build/(elapsed+1e-9):.1f} b/s)")

    elapsed_init = time.time() - t_start
    rate = n_model_build / (elapsed_init + 1e-9)
    n_methods = len(METHODS)
    est_unique_per_gen = int(pop_sz * n_methods * 0.98)
    est_total = pop_sz + est_unique_per_gen * n_gen
    eta = max(0, (est_total - n_model_build)) / rate / 60
    print(f"  Init done: {n_model_build} builds in {elapsed_init:.0f}s  "
          f"({rate:.1f} b/s)  est. total ~{est_total} builds  "
          f"ETA ~{eta:.0f} min")

    # ── Phase 2: generational evolution ──
    for gen in range(n_gen):
        # Each method produces pop_sz offspring from its own population
        pending = {}   # arch_key → (arch_dict, set_of_methods_that_requested)
        for m in METHODS:
            for _ in range(pop_sz):
                cands  = random.sample(pops[m], min(3, len(pops[m])))
                parent = max(cands, key=lambda x: x['score'])
                child  = _mutate(parent, pairs, target, tol)
                if child:
                    key = (child['d_model'], child['n_layer'],
                           tuple(child['d_inner']))
                    if key not in pending:
                        pending[key] = (child, set())
                    pending[key][1].add(m)

        # Build model once per unique arch
        for child, _ in pending.values():
            scores = _eval_and_count(child)
            for m in METHODS:
                entry = dict(child); entry['score'] = scores[m]
                pops[m].append(entry)
                if scores[m] > bests[m]['score']:
                    bests[m] = dict(entry)

        for m in METHODS:
            pops[m].sort(key=lambda x: x['score'], reverse=True)
            pops[m] = pops[m][:pop_sz]

        if (gen + 1) % 3 == 0:
            elapsed = time.time() - t_start
            rate_now = n_model_build / (elapsed + 1e-9)
            builds_remaining = est_unique_per_gen * (n_gen - gen - 1)
            remaining = max(0, builds_remaining) / rate_now / 60
            best_str = "  ".join(f"{METHOD_LABELS[m]}={bests[m]['score']:.3g}"
                                 for m in METHODS)
            print(f"  gen {gen+1:3d}/{n_gen}  builds={n_model_build:5d}  "
                  f"unique={len(pending):3d}  "
                  f"{elapsed/60:.1f}min  {rate_now:.1f}b/s  "
                  f"ETA {remaining:.0f}min  | {best_str}")

    total_time = time.time() - t_start
    print(f"\n  Model search done: {n_model_build} model builds in "
          f"{total_time/60:.1f} min ({n_model_build/total_time:.1f} b/s)")

    result = {}
    bench_per_eval = {
        'SynDiv': 0.589, 'W-PCA': 0.707,
        'HeadImp': 0.571, 'SoftmaxConf': 0.546,
    }
    for m in METHODS:
        b = bests[m]
        label = METHOD_LABELS[m]
        b['nsc_mp_sum'] = nsc_mp_sum(b['d_model'], b['n_layer'], b['d_inner'])
        b['wall_time']  = total_time
        b['n_model_build'] = n_model_build
        per_method_evals = pop_sz + pop_sz * n_gen
        b['n_eval'] = per_method_evals
        b['reported_time'] = bench_per_eval.get(label, 0.6) * per_method_evals
        b['reported_time_5k'] = bench_per_eval.get(label, 0.6) * 5000
        result[label] = b

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_time(t):
    if t < 1:    return f"{t*1000:.1f}ms"
    if t < 60:   return f"{t:.1f}s"
    if t < 3600: return f"{t/60:.1f}min"
    return f"{t/3600:.1f}h"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=float, default=None,
                    help='Target non-emb params (default: TXL Base ≈38.4M)')
    ap.add_argument('--tol',    type=float, default=0.05)
    ap.add_argument('--pop',    type=int,   default=50)
    ap.add_argument('--gen',    type=int,   default=24,
                    help='Generations per method (pop + pop*gen ≈ evals/method)')
    ap.add_argument('--seed',   type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    txl_params = compute_params(410, 16, [2100] * 16)
    target = args.target or txl_params
    total_evals = args.pop + args.pop * args.gen

    ss = search_space_size()
    print("=" * 78)
    print("  End-to-end NAS Demo · Transformer-XL Space → WikiText-103")
    print("=" * 78)
    print(f"  Search space    : {ss:.4e}  architectures")
    print(f"  Target params   : {target/1e6:.2f}M  ±{args.tol*100:.0f}%"
          f"  [{target*(1-args.tol)/1e6:.2f}M, {target*(1+args.tol)/1e6:.2f}M]")
    print(f"  TXL Base        : dm=410  nl=16  di=2100  "
          f"params={txl_params/1e6:.2f}M  PPL=24.03")
    print(f"  EA config       : pop={args.pop}  gen={args.gen}  "
          f"→ {total_evals} evals")

    pairs = _feasible_pairs(target, args.tol)
    print(f"  Feasible pairs  : {len(pairs)}")
    for dm, nl in pairs:
        lo = compute_params(dm, nl, [D_INNER[0]]  * nl)
        hi = compute_params(dm, nl, [D_INNER[-1]] * nl)
        print(f"      dm={dm:4d}  nl={nl:2d}  "
              f"[{lo/1e6:5.1f}M – {hi/1e6:5.1f}M]")

    results = {}

    # ── 1. NSC-MP  (knapsack, exact) ─────────────────────────────────────────
    print("\n" + "─" * 78)
    print("  Method 1 :  NSC-MP  (Knapsack DP · exact · 0 GPU)")
    print("─" * 78)
    t0 = time.time()
    r1 = knapsack_nsc_mp(target, args.tol)
    t1 = time.time() - t0
    if r1:
        r1['time'] = t1
        r1['reported_time'] = t1
        di_set = sorted(set(r1['d_inner']))
        print(f"  Time     : {_fmt_time(t1)}")
        print(f"  dm={r1['d_model']}  nl={r1['n_layer']}  "
              f"nh={r1['n_head'][0]}  dh={r1['d_model']//r1['n_head'][0]}")
        print(f"  d_inner  : {di_set}  "
              f"(×{[r1['d_inner'].count(d) for d in di_set]})")
        print(f"  Params   : {r1['params']/1e6:.2f}M")
        print(f"  NSC(sum) : {r1['nsc']:.2f}")
        results['NSC-MP'] = r1

    # Pre-selected NSC-MP candidates
    nscmp_picks = {
        'Deep-18':  dict(d_model=512, n_layer=18, d_inner=[1152]*18),
        'Match-16': dict(d_model=512, n_layer=16, d_inner=[1408]*16),
        'Wide-14':  dict(d_model=512, n_layer=14, d_inner=[1664]*14),
    }
    print("\n  NSC-MP prior-informed picks:")
    for tag, cfg in nscmp_picks.items():
        dm, nl, di = cfg['d_model'], cfg['n_layer'], cfg['d_inner']
        p   = compute_params(dm, nl, di)
        s   = nsc_mp_sum(dm, nl, di)
        print(f"    {tag:10s}  dm={dm} nl={nl} di={di[0]}  "
              f"ratio={di[0]/dm:.2f}  params={p/1e6:.2f}M  "
              f"NSC-sum={s:.1f}")
        results[f'NSC-MP({tag})'] = dict(
            d_model=dm, n_layer=nl, d_inner=di,
            n_head=[default_nhead(dm)]*nl,
            params=p, nsc=s,
            time=t1, reported_time=t1)

    # ── 2–5. Model-based ZCPs  (shared-build EA) ─────────────────────────
    if HAS_TORCH:
        print("\n" + "─" * 78)
        print(f"  Methods 2–5 :  Shared-Model EA  "
              f"(SynDiv, W-PCA, HeadImp, SoftmaxConf)")
        print(f"  pop={args.pop}  gen={args.gen}  → "
              f"~{total_evals} evals/method, deduped model builds")
        print("─" * 78)
        model_results = shared_model_search(
            target, args.tol, args.pop, args.gen, args.seed)
        results.update(model_results)

    # ── 6. Random ─────────────────────────────────────────────────────────────
    print("\n" + "─" * 78)
    print("  Method 6 :  Random  (single random sample, no ZCP selection)")
    print("─" * 78)
    rng_rand = random.Random(args.seed + 999)
    rand_arch = None
    for _ in range(10000):
        dm_r, nl_r = rng_rand.choice(pairs)
        di_r = [rng_rand.choice(D_INNER) for _ in range(nl_r)]
        p_r  = compute_params(dm_r, nl_r, di_r)
        if abs(p_r - target) / target <= args.tol:
            vh_r = valid_heads(dm_r)
            rand_arch = dict(d_model=dm_r, n_layer=nl_r, d_inner=di_r,
                             n_head=[rng_rand.choice(vh_r) for _ in range(nl_r)],
                             params=p_r)
            break
    if rand_arch:
        rand_arch['time'] = 0.0
        rand_arch['reported_time'] = 0.0
        rand_arch['nsc_mp_sum'] = nsc_mp_sum(rand_arch['d_model'],
                                              rand_arch['n_layer'],
                                              rand_arch['d_inner'])
        print(f"  dm={rand_arch['d_model']}  nl={rand_arch['n_layer']}  "
              f"di∈[{min(rand_arch['d_inner'])},{max(rand_arch['d_inner'])}]")
        print(f"  Params   : {rand_arch['params']/1e6:.2f}M")
        results['Random'] = rand_arch

    # ── TXL Base reference ───────────────────────────────────────────────────
    txl_nsc_sum = nsc_mp_sum(410, 16, [2100] * 16)

    # ── Summary Table ────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  SUMMARY   (all architectures under "
          f"{target/1e6:.1f}M ±{args.tol*100:.0f}% param constraint)")
    print("=" * 90)

    hdr = (f"  {'Method':<16s}  {'dm':>4s}  {'nl':>3s}  {'di(avg)':>7s}  "
           f"{'Params':>8s}  {'NSC-sum':>9s}  {'Search':>10s}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    print(f"  {'TXL Base':<16s}  {410:>4d}  {16:>3d}  {2100:>7d}  "
          f"{txl_params/1e6:>7.2f}M  {txl_nsc_sum:>9.1f}  {'human':>10s}")

    order = ['NSC-MP(Deep-18)', 'NSC-MP(Match-16)', 'NSC-MP(Wide-14)',
             'SynDiv', 'W-PCA', 'HeadImp', 'SoftmaxConf', 'Random']
    for name in order:
        if name not in results:
            continue
        r   = results[name]
        dm  = r['d_model']
        nl  = r['n_layer']
        adi = int(np.mean(r['d_inner']))
        p   = r['params']
        ns  = r.get('nsc', r.get('nsc_mp_sum', 0))
        rt  = r.get('reported_time', r.get('time', 0))
        print(f"  {name:<16s}  {dm:>4d}  {nl:>3d}  {adi:>7d}  "
              f"{p/1e6:>7.2f}M  {ns:>9.1f}  "
              f"{_fmt_time(rt):>10s}")

    # ── Save ─────────────────────────────────────────────────────────────────
    save = {}
    for name, r in results.items():
        save[name] = {k: v for k, v in r.items()
                      if k not in ('score',)}
    save['TXL_Base'] = dict(d_model=410, n_layer=16,
                            d_inner=[2100]*16, n_head=[10]*16,
                            params=int(txl_params),
                            nsc_sum=txl_nsc_sum)
    save['meta'] = dict(
        search_space=float(ss),
        target_params=target,
        tolerance=args.tol,
        total_evals=total_evals,
        d_model_choices=D_MODEL,
        n_layer_choices=N_LAYER,
        n_d_inner_choices=len(D_INNER),
    )

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'search_txl_results.json')
    with open(out_path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == '__main__':
    main()
