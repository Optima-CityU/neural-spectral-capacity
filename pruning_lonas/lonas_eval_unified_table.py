#!/usr/bin/env python3
"""Shared evaluation utilities for LoNAS LLaMA-7B subnets.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench_eval_500 import (
    apply_subnet_ffn,
    build_instruct_items,
    build_lmeval_items,
    eval_task,
    load_merged_model,
    pretokenize_instruct,
    pretokenize_lmeval,
    restore_full_ffn,
    save_full_weights,
)


LMEVAL_TASKS_7 = ["boolq", "piqa", "hellaswag", "winogrande"]
INSTRUCT_TASKS_7 = ["ARC-Easy", "ARC-Challenge", "openbookqa"]
ALL_TASKS_7 = LMEVAL_TASKS_7 + INSTRUCT_TASKS_7


def setup_tokenizer(tokenizer):
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 0
    return tokenizer


def load_vanilla_model(base_path, dtype):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tok = setup_tokenizer(AutoTokenizer.from_pretrained(base_path, use_fast=False, local_files_only=True))
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=dtype,
        device_map="cpu",
        local_files_only=True,
    )
    model.config.pad_token_id = 0
    return model, tok


def prepare_data(tokenizer):
    lmeval_data = {}
    for task in LMEVAL_TASKS_7:
        raw = build_lmeval_items(task)
        lmeval_data[task] = pretokenize_lmeval(raw, tokenizer)
        print(f"  {task}: {len(lmeval_data[task])}", flush=True)

    instruct_data = {}
    for task in INSTRUCT_TASKS_7:
        raw = build_instruct_items(task)
        instruct_data[task] = pretokenize_instruct(raw, tokenizer)
        print(f"  {task}: {len(instruct_data[task])}", flush=True)

    return lmeval_data, instruct_data


def eval_loaded_model(model, lmeval_data, instruct_data, batch_size, device):
    scores = {}
    for task in LMEVAL_TASKS_7:
        t0 = time.time()
        acc = eval_task(model, lmeval_data[task], batch_size, device, length_normalize=False, is_instruct=False)
        scores[task] = round(acc, 4)
        print(f"  {task}: {acc:.4f} ({time.time() - t0:.1f}s)", flush=True)
    for task in INSTRUCT_TASKS_7:
        t0 = time.time()
        acc = eval_task(model, instruct_data[task], batch_size, device, length_normalize=False, is_instruct=True)
        scores[task] = round(acc, 4)
        print(f"  {task}: {acc:.4f} ({time.time() - t0:.1f}s)", flush=True)
    scores["avg7"] = round(sum(scores[t] for t in ALL_TASKS_7) / len(ALL_TASKS_7), 4)
    return scores


def load_rows(rows_path):
    with open(rows_path) as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "rows" in payload:
        return payload["rows"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Rows JSON must be a list or contain a top-level 'rows' list")


def resolve_subnet(row):
    if "subnet" in row and row["subnet"] is not None:
        return row["subnet"]
    if "search_json" in row:
        with open(row["search_json"]) as f:
            search = json.load(f)
        return search["rank1"]["subnet"]
    return None


def merge_search_metadata(row):
    if "search_json" not in row:
        return dict(row)
    with open(row["search_json"]) as f:
        search = json.load(f)
    rank1 = search.get("rank1") or {}
    merged = dict(row)
    merged.update({
        "search_status": search.get("status"),
        "search_time_s": search.get("search_time_s"),
        "setup_time_s": search.get("setup_time_s"),
        "n_proxy_evals": search.get("n_proxy_evals"),
        "n_attempts": search.get("n_attempts"),
        "proxy_score": rank1.get("score"),
        "params_B": rank1.get("params_B", row.get("params_B")),
    })
    return merged


def count_params_from_subnet(subnet):
    if subnet is None:
        return None
    hidden = 4096
    vocab = 32000
    n_layers = 32
    total = 2 * vocab * hidden + n_layers * (4 * hidden * hidden + 2 * hidden)
    total += 3 * hidden * sum(subnet["ffn_dims"])
    return round(total / 1e9, 6)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fixed and searched LoNAS rows with bench500 eval7")
    parser.add_argument("--rows", required=True, help="JSON list of rows to evaluate")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base_model", default=os.environ.get("LLAMA_PATH", "yahma/llama-7b-hf"))
    parser.add_argument("--adapter", default=os.environ.get("LONAS_ADAPTER", "lonas-llama-7b-adapter"))
    parser.add_argument("--batch_size", type=int, default=24)
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
        print("Pre-tokenizing eval7 datasets for vanilla model...", flush=True)
        lmeval_data, instruct_data = prepare_data(tokenizer)
        for row in vanilla_rows:
            print(f"\nEvaluating {row['method']}...", flush=True)
            t0 = time.time()
            scores = eval_loaded_model(model, lmeval_data, instruct_data, args.batch_size, device)
            results.append({**row, "bench500_scores": scores, "avg7": scores["avg7"], "eval_time_s": round(time.time() - t0, 1)})
        del model
        torch.cuda.empty_cache()

    if adapter_rows:
        print("Loading LoNAS merged supernet...", flush=True)
        model, tokenizer = load_merged_model(args.base_model, args.adapter, dtype)
        model = model.to(device).eval()
        full_weights = save_full_weights(model)
        print("Pre-tokenizing eval7 datasets for LoNAS model...", flush=True)
        lmeval_data, instruct_data = prepare_data(tokenizer)

        for row in adapter_rows:
            row = merge_search_metadata(row)
            subnet = resolve_subnet(row)
            print(f"\nEvaluating {row['method']}...", flush=True)
            t0 = time.time()
            if subnet is not None:
                apply_subnet_ffn(model, subnet["ffn_dims"], full_weights)
            scores = eval_loaded_model(model, lmeval_data, instruct_data, args.batch_size, device)
            if subnet is not None:
                restore_full_ffn(model, full_weights)
            results.append({
                **row,
                "subnet": subnet,
                "params_B": row.get("params_B", count_params_from_subnet(subnet)),
                "bench500_scores": scores,
                "avg7": scores["avg7"],
                "eval_time_s": round(time.time() - t0, 1),
            })
            with open(output, "w") as f:
                json.dump({"status": "running", "tasks": ALL_TASKS_7, "rows": results}, f, indent=2)

    with open(output, "w") as f:
        json.dump({"status": "complete", "tasks": ALL_TASKS_7, "rows": results}, f, indent=2)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
