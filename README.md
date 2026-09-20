# Neural Spectral Capacity — code

Code for *Neural Spectral Capacity: Measuring and Designing Architectures from
Network Specification Alone* (Chenyu Zhu, Ruoyu Zhao, Zhichao Lu).

NSC-MP scores a network from its specification alone: each weight matrix
contributes a closed-form spectral capacity ψ_MP computed from its shape and
initialization variance via the Marchenko–Pastur law, and the scores are summed
over the network. NSC-DP is an exact dynamic program that returns the
architecture maximizing NSC under a resource budget.

Benchmark data, datasets and model weights are not included; see *External
resources* below.

## Layout

```
nsc/               nsc_utils.py — reference ψ_MP implementation
ranking/           NSC scores on five architecture families; baseline scores
                   (#FLOPs, W-PCA, ZeroLM, SNIP, GradNorm, SynFlow)
search/txl/        NSC-DP on Transformer-XL (WikiText-103), EA baselines, training
search/autoformer/ NSC-DP on AutoFormer (ImageNet-1K)
pruning_lonas/     LLaMA-7B pruning via the LoNAS supernet: NSC-DP, proxy + GA baselines,
                   Avg-8 commonsense evaluation
appendix/          aggregation choice, MP finite-size convergence, init-variance robustness,
                   concavity of ψ_MP
```

## Setup

Python ≥ 3.9 with `numpy scipy matplotlib torch torchvision timm transformers peft thop`.
Scripts import each other by module name; run them from this directory with

```bash
export PYTHONPATH=nsc:ranking:search/txl:search/autoformer:pruning_lonas:appendix
```

Paths are read from environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `NSC_ROOT` | root of the benchmark data tree; small proxy input batches go in `$NSC_ROOT/sample_inputs/` | `.` |
| `NSC_OUT` | where results and figures are written | `outputs` |
| `FLEXIBERT_DIR` | FlexiBERT code + `BERT_benchmark.json` | `FlexiBERT` |
| `AUTOFORMER_DIR` | checkout of microsoft/Cream `AutoFormer/` | `Cream/AutoFormer` |
| `AUTOFORMER_CKPT_DIR` | `supernet-{tiny,small,base}.pth` | `checkpoints` |
| `IMAGENET_DIR` | ImageNet-1K | `data/imagenet` |
| `LLAMA_PATH` | LLaMA-7B (HF format) | `yahma/llama-7b-hf` |
| `LONAS_ADAPTER` | the LoNAS LLaMA-7B commonsense adapter | `lonas-llama-7b-adapter` |
| `LONAS_DATA_DIR` | the eight commonsense-reasoning evaluation sets | `datasets` |

## Paper section → script

**Method**
- ψ_MP: `nsc/nsc_utils.py`

**Ranking across architecture families (§4.1)**
- NSC scores: `ranking/compute_flexibert.py`, `compute_gpt2.py`, `compute_autoformer.py`,
  `compute_nats_sss.py`, `compute_mnv3.py`
- Baselines: `ranking/compute_flops_all.py` (#FLOPs); `ranking/run_realdata_proxy.py`
  (W-PCA, ZeroLM, SNIP, GradNorm, SynFlow), which uses the per-family proxy implementations in
  `ranking/eval_all_zcps.py` (FlexiBERT), `ranking/eval_gpt2_baselines.py` (GPT-2) and
  `ranking/eval_autoformer_tiny.py` (AutoFormer)
- FlexiBERT, NSC and all training-free proxies (per-architecture hidden size, per-layer FFN
  widths): `ranking/eval_flexibert.py`

**Architecture search via NSC-DP (§4.2)**
- Transformer-XL: `search/txl/search_txl_space.py` (search space and DP),
  `search/txl/nsc_dp_search.py` (NSC-DP), `search/txl/run_independent_ea.py` (EA baselines),
  `search/txl/train_wt103_txl.py` (training on WikiText-103; requires kimiyoung/transformer-xl)
- AutoFormer: `search/autoformer/nsc_dp_autoformer.py` (NSC-DP); the returned subnets are evaluated
  with the AutoFormer supernet evaluation in microsoft/Cream

**Structured pruning of LLaMA-7B (§4.3, multi-budget appendix)**
- NSC-DP: `pruning_lonas/build_nsch_mp_cache.py` → `lonas_nscdp_full_pareto.py`;
  TFLOPs-constrained Pareto front (operating points A–G): `lonas_nscdp_tflops_dp.py`;
  search space and DP core: `lonas_nscdp_original_space.py`
- Baselines (proxy + GA): `pruning_lonas/run_zcp_baselines_full.py`
- Avg-8 evaluation: `pruning_lonas/lonas_eval_unified_table_avg8.py`
  (with `lonas_eval_unified_table.py`, `bench_eval_500.py`)

**Appendix ablations**
- Aggregation choice: `appendix/ablation_aggregation_specfaithful.py`, `ablation_min_generalization.py`,
  `aggregation_autoformer.py` (aggregation functions in `appendix/ablation_aggregation.py`)
- MP finite-size convergence: `appendix/ablation_mp_convergence.py`
- Robustness to initialization convention: `appendix/ablation_init_variance.py`
- Concavity of ψ_MP: `appendix/plot_psi_concavity.py`

## External resources

- **FlexiBERT**: `BERT_benchmark.json` and the ELECTRA modeling code from the FlexiBERT release.
- **GPT-2 / LiteTransformerSearch**: `gpt2_benchmark.json`.
- **AutoFormer**: microsoft/Cream (AutoFormer), supernet checkpoints, the 1k-architecture sets; ImageNet-1K.
- **NATS-Bench-SSS**: the official NATS-Bench size-search-space results.
- **MobileNetV3**: OFA MobileNetV3 accuracy table.
- **Transformer-XL**: kimiyoung/transformer-xl; WikiText-103.
- **LoNAS**: LLaMA-7B and the IntelLabs LoNAS commonsense adapter; the eight commonsense
  tasks (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA).

## License

MIT (see `LICENSE`).
