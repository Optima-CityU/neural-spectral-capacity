#!/usr/bin/env python3
"""Avg-8 evaluation of LoNAS LLaMA-7B subnets on eight commonsense tasks
(BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA), following the
LoNAS evaluation protocol (instruction-format log-likelihood, no length normalization).
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch


from bench_eval_500 import (
    FFN_FULL,
    HIDDEN,
    N_LAYERS,
    INSTRUCT_PROMPT_NO_INPUT,
    INSTRUCT_PROMPT_WITH_INPUT,
    eval_task,
    load_merged_model,
    pretokenize_instruct,
)

LONAS_DATA_DIR = Path(os.environ.get("LONAS_DATA_DIR", "datasets"))
from lonas_eval_unified_table import (
    count_params_from_subnet,
    load_rows,
    load_vanilla_model,
    merge_search_metadata,
    resolve_subnet,
)


# ── CPU-resident FFN snapshot ────────────

def save_full_weights_cpu(model):
    """Snapshot FFN weights to CPU pinned memory; frees ~8.6 GB on GPU."""
    weights = {}
    for li in range(N_LAYERS):
        layer = model.model.layers[li]
        weights[li] = {
            "gate": layer.mlp.gate_proj.weight.data.detach().to("cpu", copy=True),
            "up":   layer.mlp.up_proj.weight.data.detach().to("cpu", copy=True),
            "down": layer.mlp.down_proj.weight.data.detach().to("cpu", copy=True),
        }
    torch.cuda.empty_cache()
    gc.collect()
    return weights


def apply_subnet_ffn_from_cpu(model, ffn_dims, full_weights_cpu, device):
    for li in range(N_LAYERS):
        k = ffn_dims[li]
        layer = model.model.layers[li]
        fw = full_weights_cpu[li]
        layer.mlp.gate_proj.weight.data = fw["gate"][:k, :].to(device, non_blocking=True)
        layer.mlp.gate_proj.out_features = k
        layer.mlp.up_proj.weight.data = fw["up"][:k, :].to(device, non_blocking=True)
        layer.mlp.up_proj.out_features = k
        layer.mlp.down_proj.weight.data = fw["down"][:, :k].to(device, non_blocking=True)
        layer.mlp.down_proj.in_features = k


def restore_full_ffn_from_cpu(model, full_weights_cpu, device):
    for li in range(N_LAYERS):
        layer = model.model.layers[li]
        fw = full_weights_cpu[li]
        layer.mlp.gate_proj.weight.data = fw["gate"].to(device, non_blocking=True)
        layer.mlp.gate_proj.out_features = FFN_FULL
        layer.mlp.up_proj.weight.data = fw["up"].to(device, non_blocking=True)
        layer.mlp.up_proj.out_features = FFN_FULL
        layer.mlp.down_proj.weight.data = fw["down"].to(device, non_blocking=True)
        layer.mlp.down_proj.in_features = FFN_FULL


ALL_TASKS_8 = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande",
               "ARC-Easy", "ARC-Challenge", "openbookqa"]

# Per-task (template_keys, n_choices) for LoNAS instruction-LL protocol.
# Templates match LoNAS's `output` field exactly: "the correct answer is <key>"
TASK_KEYS = {
    "boolq":         ["true", "false"],
    "piqa":          ["solution1", "solution2"],
    "winogrande":    ["option1", "option2"],
    "hellaswag":     ["ending1", "ending2", "ending3", "ending4"],
    "ARC-Easy":      ["answer1", "answer2", "answer3", "answer4"],
    "ARC-Challenge": ["answer1", "answer2", "answer3", "answer4"],
    "openbookqa":    ["answer1", "answer2", "answer3", "answer4"],
    "social_i_qa":   ["answer1", "answer2", "answer3"],
}


def build_instruct_items_v2(task_key):
    """Universal LoNAS instruction-LL builder for all 8 tasks.

    Reads datasets/<task>/test.json (LoNAS-format) and emits
    (prompt, [template_per_choice], [key_per_choice], gold_label).
    """
    keys = TASK_KEYS[task_key]
    data_path = LONAS_DATA_DIR / task_key / "test.json"
    with open(data_path) as f:
        dataset = json.load(f)
    items = []
    for row in dataset:
        inp = row.get("input", "")
        if inp:
            prompt = INSTRUCT_PROMPT_WITH_INPUT.format(
                instruction=row["instruction"], input=inp)
        else:
            prompt = INSTRUCT_PROMPT_NO_INPUT.format(
                instruction=row["instruction"])
        templates = [f"the correct answer is {k}" for k in keys]
        gold = row["answer"].strip().lower()
        items.append((prompt, templates, list(keys), gold))
    return items


def prepare_data(tokenizer):
    instruct_data = {}
    for task in ALL_TASKS_8:
        raw = build_instruct_items_v2(task)
        instruct_data[task] = pretokenize_instruct(raw, tokenizer)
        print(f"  {task}: {len(instruct_data[task])}", flush=True)
    return None, instruct_data


def eval_loaded_model(model, lmeval_data, instruct_data, batch_size, device):
    scores = {}
    for task in ALL_TASKS_8:
        t0 = time.time()
        acc = eval_task(model, instruct_data[task], batch_size, device,
                        length_normalize=False, is_instruct=True)
        scores[task] = round(acc, 4)
        print(f"  {task}: {acc:.4f} ({time.time() - t0:.1f}s)", flush=True)
    scores["avg8"] = round(sum(scores[t] for t in ALL_TASKS_8) / len(ALL_TASKS_8), 4)
    return scores


def main():
    parser = argparse.ArgumentParser(description="LoNAS-aligned avg8 eval")
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base_model", default=os.environ.get("LLAMA_PATH", "yahma/llama-7b-hf"))
    parser.add_argument("--adapter", default=os.environ.get("LONAS_ADAPTER", "lonas-llama-7b-adapter"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_rows(args.rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    results = []
    vanilla_rows = [r for r in rows if r.get("model_kind") == "vanilla"]
    adapter_rows = [r for r in rows if r.get("model_kind") != "vanilla"]

    if vanilla_rows:
        print("Loading vanilla LLaMA...", flush=True)
        model, tokenizer = load_vanilla_model(args.base_model, dtype)
        model = model.to(device).eval()
        lmeval_data, instruct_data = prepare_data(tokenizer)
        for row in vanilla_rows:
            print(f"\nEvaluating {row['method']}...", flush=True)
            t0 = time.time()
            scores = eval_loaded_model(model, lmeval_data, instruct_data, args.batch_size, device)
            results.append({**row, "bench500_scores": scores, "avg8": scores["avg8"],
                            "eval_time_s": round(time.time() - t0, 1)})
        del model
        torch.cuda.empty_cache()

    if adapter_rows:
        print("Loading LoNAS merged supernet...", flush=True)
        model, tokenizer = load_merged_model(args.base_model, args.adapter, dtype)
        model = model.to(device).eval()

        any_subnet = any(resolve_subnet(merge_search_metadata(r)) is not None for r in adapter_rows)
        full_weights_cpu = None
        if any_subnet:
            print("Snapshotting full FFN weights to CPU (frees ~8.6 GB on GPU)...", flush=True)
            full_weights_cpu = save_full_weights_cpu(model)
            print(f"  GPU mem after snapshot: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

        lmeval_data, instruct_data = prepare_data(tokenizer)

        for row in adapter_rows:
            row = merge_search_metadata(row)
            subnet = resolve_subnet(row)
            print(f"\nEvaluating {row['method']}...", flush=True)
            t0 = time.time()
            if subnet is not None:
                apply_subnet_ffn_from_cpu(model, subnet["ffn_dims"], full_weights_cpu, device)
            scores = eval_loaded_model(model, lmeval_data, instruct_data, args.batch_size, device)
            if subnet is not None:
                restore_full_ffn_from_cpu(model, full_weights_cpu, device)
            results.append({
                **row,
                "subnet": subnet,
                "params_B": row.get("params_B", count_params_from_subnet(subnet)),
                "bench500_scores": scores,
                "avg8": scores["avg8"],
                "eval_time_s": round(time.time() - t0, 1),
            })
            with open(output, "w") as f:
                json.dump({"status": "running", "tasks": ALL_TASKS_8, "rows": results}, f, indent=2)

    with open(output, "w") as f:
        json.dump({"status": "complete", "tasks": ALL_TASKS_8, "rows": results}, f, indent=2)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
