#!/usr/bin/env python3
"""Train Transformer-XL architectures on WikiText-103.

Model:
  - Relative position bias (Dai et al., 2019)
  - Segment-level recurrence memory
  - Per-layer d_inner (supports heterogeneous FFN widths)

Training recipe (TXL Base):
  - Adam, lr=2.5e-4, warmup 16K steps, cosine decay
  - batch_size=60, bptt=150, mem_len=150
  - Gradient clipping 0.25, dropout 0.1
  - Number of training steps configurable (--max_steps)

Usage:
  # Train single architecture:
  python train_wt103_txl.py --arch "TXL Base" --data /path/to/wikitext-103

  # Train all architectures sequentially:
  python train_wt103_txl.py --arch all --data /path/to/wikitext-103
"""

import argparse, json, math, os, sys, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ═══════════════════════════════════════════════════════════════════════════════
#  Architecture Configs  (from NAS search + TXL Base reference)
# ═══════════════════════════════════════════════════════════════════════════════

ARCH_CONFIGS = {
    "TXL Base": dict(
        dm=410, nl=16, nh=10,
        di=[2100]*16),
    "NSC-MP(Deep-18)": dict(
        dm=512, nl=18, nh=16,
        di=[1152]*18),
    "NSC-MP(Match-16)": dict(
        dm=512, nl=16, nh=16,
        di=[1408]*16),
    "NSC-MP(Wide-14)": dict(
        dm=512, nl=14, nh=16,
        di=[1664]*14),
    "NSC-W": dict(
        dm=384, nl=20, nh=16,
        di=[2048, 1792, 1792, 2048, 1792, 1792, 1920, 1920, 1792, 1664,
            1792, 2048, 1920, 1920, 1664, 1792, 1792, 2048, 1664, 1792]),
    "SynDiv": dict(
        dm=384, nl=18, nh=8,
        di=[512, 640, 1920, 2816, 1920, 1152, 1792, 2816, 1920, 3072,
            1280, 2048, 1664, 3840, 1024, 3328, 3840, 2048]),
    "W-PCA": dict(
        dm=384, nl=20, nh=16,
        di=[1152, 1792, 3328, 768, 1024, 640, 512, 3328, 3584, 1920,
            1920, 768, 1152, 1536, 1024, 2816, 1408, 1152, 3840, 3328]),
    "HeadImp": dict(
        dm=384, nl=20, nh=4,
        di=[1920, 2560, 640, 1024, 2304, 3840, 896, 1664, 3072, 512,
            1920, 640, 1920, 1664, 896, 1024, 2816, 1408, 1280, 1536]),
    "SoftmaxConf": dict(
        dm=384, nl=16, nh=4,
        di=[3584, 3072, 1152, 2816, 640, 896, 2048, 2304, 1152, 3584,
            1536, 1280, 3328, 2816, 4096, 1152]),
    "Random": dict(
        dm=256, nl=24, nh=4,
        di=[2304, 3072, 4096, 1024, 4096, 3584, 3584, 4096, 4096, 1280,
            3584, 640, 1536, 2048, 1280, 3072, 2816, 3072, 1920, 1408,
            2048, 1152, 3072, 640]),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Positional Encoding (sinusoidal, for relative position bias)
# ═══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, seq_len):
        return self.pe[:seq_len]


# ═══════════════════════════════════════════════════════════════════════════════
#  Relative Multi-Head Attention  (Transformer-XL style)
# ═══════════════════════════════════════════════════════════════════════════════

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.scale  = 1.0 / (d_head ** 0.5)

        self.W_q = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_r = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_o = nn.Linear(n_head * d_head, d_model, bias=False)

        self.drop_attn = nn.Dropout(dropout)

    def _rel_shift(self, x):
        """Relative shift to align position-based attention scores."""
        bsz, n_head, qlen, klen = x.size()
        zero_pad = torch.zeros(bsz, n_head, qlen, 1,
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(bsz, n_head, klen + 1, qlen)
        x = x_padded[:, :, 1:].view_as(x)
        return x

    def forward(self, h, mem, r, u_bias, v_bias, attn_mask=None):
        """
        h:     [bsz, qlen, d_model]  current segment
        mem:   [bsz, mlen, d_model]  cached memory (or empty)
        r:     [klen, d_model]        relative position encoding
        u_bias, v_bias: [n_head, d_head]  global biases
        """
        bsz, qlen, _ = h.size()
        if mem is not None and mem.size(1) > 0:
            cat = torch.cat([mem, h], dim=1)
        else:
            cat = h
        klen = cat.size(1)

        Q = self.W_q(h).view(bsz, qlen, self.n_head, self.d_head)
        K = self.W_k(cat).view(bsz, klen, self.n_head, self.d_head)
        V = self.W_v(cat).view(bsz, klen, self.n_head, self.d_head)
        R = self.W_r(r).view(klen, self.n_head, self.d_head)

        Q = Q.permute(0, 2, 1, 3)   # [bsz, nh, qlen, dh]
        K = K.permute(0, 2, 3, 1)   # [bsz, nh, dh, klen]
        V = V.permute(0, 2, 1, 3)   # [bsz, nh, klen, dh]
        R = R.permute(1, 2, 0)      # [nh, dh, klen]

        # Content-based attention + global content bias
        AC = torch.matmul(Q + u_bias.unsqueeze(1), K)   # [bsz, nh, qlen, klen]

        # Position-based attention + global positional bias
        BD = torch.matmul(Q + v_bias.unsqueeze(1), R.unsqueeze(0).expand(bsz, -1, -1, -1))
        BD = self._rel_shift(BD)

        attn = (AC + BD) * self.scale

        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.drop_attn(attn)

        out = torch.matmul(attn, V)                     # [bsz, nh, qlen, dh]
        out = out.permute(0, 2, 1, 3).reshape(bsz, qlen, -1)
        return self.W_o(out)


# ═══════════════════════════════════════════════════════════════════════════════
#  Transformer-XL Decoder Layer
# ═══════════════════════════════════════════════════════════════════════════════

class TXLDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_head, d_inner, dropout=0.1):
        super().__init__()
        self.attn = RelativeMultiHeadAttention(d_model, n_head, d_head, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, h, mem, r, u_bias, v_bias, attn_mask):
        attn_out = self.attn(h, mem, r, u_bias, v_bias, attn_mask)
        h = self.norm1(h + self.drop(attn_out))
        h = self.norm2(h + self.ff(h))
        return h


# ═══════════════════════════════════════════════════════════════════════════════
#  Transformer-XL Model
# ═══════════════════════════════════════════════════════════════════════════════

class TransformerXL(nn.Module):
    def __init__(self, n_vocab, d_model, n_layer, n_head, d_inner_list,
                 mem_len=150, dropout=0.1, tie_weights=True):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.mem_len = mem_len
        d_head = d_model // n_head

        self.embed = nn.Embedding(n_vocab, d_model)
        self.drop  = nn.Dropout(dropout)
        self.pos_enc = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            TXLDecoderLayer(d_model, n_head, d_head, d_inner_list[i], dropout)
            for i in range(n_layer)
        ])

        self.norm_out = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, n_vocab, bias=False)

        if tie_weights:
            self.proj.weight = self.embed.weight

        # Global biases (shared across layers, as in TXL paper)
        self.u_bias = nn.Parameter(torch.zeros(n_head, d_head))
        self.v_bias = nn.Parameter(torch.zeros(n_head, d_head))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def init_memory(self, bsz, device):
        return [torch.zeros(bsz, self.mem_len, self.d_model, device=device)
                for _ in range(self.n_layer)]

    def _update_memory(self, mems, hiddens):
        with torch.no_grad():
            new_mems = []
            for i in range(self.n_layer):
                cat = torch.cat([mems[i], hiddens[i]], dim=1)
                new_mems.append(cat[:, -self.mem_len:].detach())
            return new_mems

    def forward(self, x, mems=None):
        """
        x:    [bsz, qlen]  token ids
        mems: list of [bsz, mlen, d_model] per layer, or None
        """
        bsz, qlen = x.size()
        device = x.device

        if mems is None:
            mems = self.init_memory(bsz, device)

        mlen = mems[0].size(1)
        klen = qlen + mlen

        # Causal mask: each query position can attend to all memory + past in segment
        attn_mask = torch.triu(
            torch.ones(qlen, klen, device=device) * float('-inf'),
            diagonal=1 + mlen)

        # Position encoding (reversed: farther = larger index)
        r = self.pos_enc(klen).flip(0)  # [klen, d_model]

        h = self.drop(self.embed(x))    # [bsz, qlen, d_model]

        hiddens = []
        for i, layer in enumerate(self.layers):
            hiddens.append(h)
            h = layer(h, mems[i], r, self.u_bias, self.v_bias, attn_mask)

        h = self.norm_out(h)
        logits = self.proj(h)

        new_mems = self._update_memory(mems, hiddens)
        return logits, new_mems

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def count_non_emb_params(self):
        emb_params = self.embed.weight.numel()
        total = self.count_params()
        return total - emb_params


# ═══════════════════════════════════════════════════════════════════════════════
#  WikiText-103 Data
# ═══════════════════════════════════════════════════════════════════════════════

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class WikiText103Dataset:
    def __init__(self, data_dir):
        self.vocab = Vocabulary()
        self.train = self._tokenize(os.path.join(data_dir, 'wiki.train.tokens'))
        self.valid = self._tokenize(os.path.join(data_dir, 'wiki.valid.tokens'))
        self.test  = self._tokenize(os.path.join(data_dir, 'wiki.test.tokens'))
        print(f"  Vocab size  : {len(self.vocab)}")
        print(f"  Train tokens: {len(self.train):,}")
        print(f"  Valid tokens: {len(self.valid):,}")
        print(f"  Test tokens : {len(self.test):,}")

    def _tokenize(self, path):
        assert os.path.exists(path), f"File not found: {path}"
        with open(path, 'r', encoding='utf-8') as f:
            tokens = []
            for line in f:
                words = line.strip().split() + ['<eos>']
                for w in words:
                    self.vocab.add_word(w)
                    tokens.append(self.vocab.word2idx[w])
        return torch.LongTensor(tokens)


def batchify(data, bsz):
    n_batch = data.size(0) // bsz
    data = data[:n_batch * bsz].view(bsz, -1)
    return data


def get_batch(source, i, bptt):
    seq_len = min(bptt, source.size(1) - 1 - i)
    data   = source[:, i:i+seq_len]
    target = source[:, i+1:i+1+seq_len]
    return data, target


# ═══════════════════════════════════════════════════════════════════════════════
#  Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, data_source, bptt, device):
    model.eval()
    total_loss = 0.0
    total_len  = 0
    mems = None
    with torch.no_grad():
        for i in range(0, data_source.size(1) - 1, bptt):
            inp, tgt = get_batch(data_source, i, bptt)
            inp, tgt = inp.to(device), tgt.to(device)
            logits, mems = model(inp, mems)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   tgt.reshape(-1))
            seq_len = tgt.numel()
            total_loss += loss.item() * seq_len
            total_len  += seq_len
    return total_loss / total_len


def train_one_arch(arch_name, cfg, data, args, device):
    """Train a single architecture and return best val PPL."""
    dm, nl, nh = cfg['dm'], cfg['nl'], cfg['nh']
    di_list = cfg['di']
    n_vocab = len(data.vocab)

    print(f"\n{'='*78}")
    print(f"  Training: {arch_name}")
    print(f"  dm={dm}  nl={nl}  nh={nh}  d_head={dm//nh}")
    print(f"  di range: [{min(di_list)}, {max(di_list)}]  "
          f"avg={sum(di_list)//len(di_list)}")
    print(f"{'='*78}")

    model = TransformerXL(
        n_vocab=n_vocab, d_model=dm, n_layer=nl, n_head=nh,
        d_inner_list=di_list, mem_len=args.mem_len,
        dropout=args.dropout, tie_weights=True
    ).to(device)

    n_params = model.count_params()
    n_non_emb = model.count_non_emb_params()
    print(f"  Total params    : {n_params/1e6:.2f}M")
    print(f"  Non-emb params  : {n_non_emb/1e6:.2f}M")

    train_data = batchify(data.train, args.batch_size).to(device)
    val_data   = batchify(data.valid, args.eval_batch_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Cosine schedule with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler('cuda') if args.fp16 and device.type == 'cuda' else None
    amp_dtype = torch.float16 if args.fp16 else torch.float32

    best_val_loss = float('inf')
    best_val_ppl  = float('inf')
    step = 0
    epoch = 0
    t_start = time.time()

    save_dir = os.path.join(args.save_dir, arch_name.replace(' ', '_')
                            .replace('(', '_').replace(')', '_'))
    os.makedirs(save_dir, exist_ok=True)

    while step < args.max_steps:
        model.train()
        mems = None
        epoch += 1
        total_train_loss = 0.0
        total_train_tok  = 0
        n_batches = (train_data.size(1) - 1) // args.bptt

        for batch_i in range(n_batches):
            if step >= args.max_steps:
                break

            inp, tgt = get_batch(train_data, batch_i * args.bptt, args.bptt)

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits, mems = model(inp, mems)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                           tgt.reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, mems = model(inp, mems)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       tgt.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

            # Detach memory to prevent backprop across segments
            if mems is not None:
                mems = [m.detach() for m in mems]

            scheduler.step()
            step += 1

            total_train_loss += loss.item() * tgt.numel()
            total_train_tok  += tgt.numel()

            if step % args.log_interval == 0:
                cur_loss = total_train_loss / total_train_tok
                elapsed  = time.time() - t_start
                lr_now   = optimizer.param_groups[0]['lr']
                eta_h    = elapsed / step * (args.max_steps - step) / 3600
                print(f"  step {step:6d}/{args.max_steps}  "
                      f"epoch {epoch:2d}  "
                      f"loss {cur_loss:.3f}  ppl {math.exp(cur_loss):7.2f}  "
                      f"lr {lr_now:.2e}  "
                      f"{elapsed/60:.0f}min  ETA {eta_h:.1f}h")
                total_train_loss = 0.0
                total_train_tok  = 0

            if step % args.eval_interval == 0:
                val_loss = evaluate(model, val_data, args.bptt, device)
                val_ppl  = math.exp(val_loss)
                elapsed  = time.time() - t_start
                print(f"  >>> val step {step}  "
                      f"loss={val_loss:.3f}  ppl={val_ppl:.2f}  "
                      f"best={best_val_ppl:.2f}  ({elapsed/60:.0f}min)")

                if val_ppl < best_val_ppl:
                    best_val_loss = val_loss
                    best_val_ppl  = val_ppl
                    ckpt_path = os.path.join(save_dir, 'best.pt')
                    torch.save({
                        'step': step, 'val_ppl': val_ppl,
                        'model_state': model.state_dict(),
                    }, ckpt_path)
                    print(f"  >>> saved best → {ckpt_path}")

                model.train()

    # Final evaluation
    val_loss = evaluate(model, val_data, args.bptt, device)
    val_ppl  = math.exp(val_loss)
    if val_ppl < best_val_ppl:
        best_val_ppl = val_ppl
        torch.save({
            'step': step, 'val_ppl': val_ppl,
            'model_state': model.state_dict(),
        }, os.path.join(save_dir, 'best.pt'))

    total_time = time.time() - t_start
    print(f"\n  {arch_name} DONE: best val PPL = {best_val_ppl:.2f}  "
          f"({total_time/3600:.1f}h, {step} steps, {epoch} epochs)")

    # Clean up
    del model, optimizer, scheduler, train_data, val_data
    if scaler: del scaler
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return dict(
        arch_name=arch_name, config=cfg,
        n_params=n_params, n_non_emb=n_non_emb,
        best_val_ppl=best_val_ppl, best_val_loss=best_val_loss,
        total_steps=step, total_epochs=epoch,
        total_time_h=total_time / 3600,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='Path to wikitext-103 directory')
    parser.add_argument('--arch', type=str, default='all',
                        help='Architecture name or "all"')
    parser.add_argument('--save_dir', type=str, default='./txl_checkpoints')

    parser.add_argument('--max_steps', type=int, default=100_000)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--eval_batch_size', type=int, default=10)
    parser.add_argument('--bptt', type=int, default=150)
    parser.add_argument('--mem_len', type=int, default=150)

    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--warmup_steps', type=int, default=16000)
    parser.add_argument('--clip', type=float, default=0.25)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--fp16', action='store_true')

    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB")

    # Load data
    print(f"\nLoading WikiText-103 from {args.data} ...")
    data = WikiText103Dataset(args.data)

    # Select architectures
    if args.arch.lower() == 'all':
        arch_list = list(ARCH_CONFIGS.keys())
    else:
        if args.arch not in ARCH_CONFIGS:
            print(f"Unknown arch: {args.arch}")
            print(f"Available: {list(ARCH_CONFIGS.keys())}")
            sys.exit(1)
        arch_list = [args.arch]

    print(f"\nWill train {len(arch_list)} architecture(s):")
    for name in arch_list:
        c = ARCH_CONFIGS[name]
        print(f"  {name:20s}  dm={c['dm']}  nl={c['nl']}  nh={c['nh']}  "
              f"di_avg={sum(c['di'])//len(c['di'])}")

    print(f"\nTraining config:")
    print(f"  max_steps={args.max_steps}  bs={args.batch_size}  "
          f"bptt={args.bptt}  mem={args.mem_len}")
    print(f"  lr={args.lr}  warmup={args.warmup_steps}  clip={args.clip}  "
          f"dropout={args.dropout}  fp16={args.fp16}")

    # Train
    all_results = []
    for i, name in enumerate(arch_list):
        print(f"\n\n{'#'*78}")
        print(f"#  [{i+1}/{len(arch_list)}]  {name}")
        print(f"{'#'*78}")

        result = train_one_arch(name, ARCH_CONFIGS[name], data, args, device)
        all_results.append(result)

        # Save running results
        results_path = os.path.join(args.save_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    # Final summary
    print(f"\n\n{'='*78}")
    print(f"  FINAL RESULTS  ({args.max_steps//1000}K steps, WikiText-103)")
    print(f"{'='*78}")

    hdr = (f"  {'Method':<20s}  {'dm':>4s}  {'nl':>3s}  {'di_avg':>6s}  "
           f"{'Params':>8s}  {'Val PPL':>8s}  {'Time':>6s}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    print(f"  {'TXL Base (paper)':<20s}  {410:>4d}  {16:>3d}  {2100:>6d}  "
          f"{'38.40M':>8s}  {'24.03':>8s}  {'—':>6s}")

    for r in all_results:
        c = r['config']
        di_avg = sum(c['di']) // len(c['di'])
        print(f"  {r['arch_name']:<20s}  {c['dm']:>4d}  {c['nl']:>3d}  "
              f"{di_avg:>6d}  {r['n_non_emb']/1e6:>7.2f}M  "
              f"{r['best_val_ppl']:>8.2f}  {r['total_time_h']:>5.1f}h")

    print(f"\n  Results saved → {results_path}")


if __name__ == '__main__':
    main()
