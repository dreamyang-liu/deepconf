"""
Fill in per-token confidence scores for bedrock-converted DeepConf pickle files.

Reads pickle files from outputs-bedrock/, does prefill via SGLang server
to recover logprobs, computes confs, and saves updated pickles to output dir.

Supports resumption: skips questions that already have confs filled.

Usage:
    # Start SGLang server first:
    # python -m sglang.launch_server --model-path Qwen/Qwen3-32B --tp 2 --trust-remote-code --mem-fraction-static 0.75

    python prefill-bedrock-confs.py \
        --input-dir outputs-bedrock \
        --output-dir outputs-bedrock-confs \
        --dataset-file aime_2025.jsonl \
        --model-path Qwen/Qwen3-32B \
        --url http://localhost:30000 \
        --batch-size 64 \
        --max-concurrent 2
"""

import asyncio
import aiohttp
import argparse
import glob
import json
import os
import pickle
import sys
import time
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer

sys.path.insert(0, "/sgl-workspace/deepconf")
from helper import (
    extract_answer,
    equal_func,
    prepare_prompt,
    weighted_majority_vote,
    compute_confidence_sglang,
)

TOP_LOGPROBS = 20
MAX_RETRIES = 5
RETRY_BACKOFF = 3


async def check_server_health(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{url}/health", timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    return True
    except Exception as e:
        print(f"Cannot reach SGLang server at {url}: {e}")
    return False


async def prefill_one(session, url, input_ids, trace_idx, semaphore, counters):
    async with semaphore:
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
            "return_logprob": True,
            "top_logprobs_num": TOP_LOGPROBS,
            "logprob_start_len": 0,
        }

        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(f"{url}/generate", json=payload) as resp:
                    resp.raise_for_status()
                    output = await resp.json()
                break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"    [ERROR] Trace {trace_idx} failed after {MAX_RETRIES} retries: {e}")
                    counters["errors"] += 1
                    return None
                await asyncio.sleep(RETRY_BACKOFF ** attempt)

        counters["completed"] += 1
        done = counters["completed"]
        total = counters["total"]
        if done % 100 == 0 or done == total:
            elapsed = time.time() - counters["start_time"]
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"    [{done}/{total}] {done/total:.0%} - {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

        return output


async def prefill_batch(url, input_ids_batch, max_concurrent, timeout):
    """Prefill a batch of traces concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    counters = {
        "completed": 0,
        "errors": 0,
        "total": len(input_ids_batch),
        "start_time": time.time(),
    }

    async with aiohttp.ClientSession(connector=connector, timeout=client_timeout) as session:
        tasks = [
            prefill_one(session, url, ids, i, semaphore, counters)
            for i, ids in enumerate(input_ids_batch)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    outputs = []
    for r in results:
        if isinstance(r, Exception):
            outputs.append(None)
        else:
            outputs.append(r)
    return outputs, counters


def extract_confs_from_prefill(output, prompt_len):
    meta = output.get("meta_info", {})
    input_top_logprobs = meta.get("input_top_logprobs", [])
    if not input_top_logprobs:
        return []
    gen_logprobs = input_top_logprobs[prompt_len:]
    return compute_confidence_sglang(gen_logprobs)


def pick_batch_size(avg_tokens):
    """Pick batch size based on average trace token length."""
    if avg_tokens < 8000:
        return 128
    elif avg_tokens < 16000:
        return 64
    elif avg_tokens < 25000:
        return 32
    else:
        return 16


def main():
    parser = argparse.ArgumentParser(
        description="Fill confs in bedrock-converted DeepConf pickles via SGLang prefill"
    )
    parser.add_argument("--input-dir", type=str, default="outputs-bedrock",
                        help="Directory with bedrock-converted pickle files")
    parser.add_argument("--output-dir", type=str, default="outputs-bedrock-confs",
                        help="Output directory for updated pickle files")
    parser.add_argument("--dataset-file", type=str, default="aime_2025.jsonl",
                        help="JSONL dataset file")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-32B",
                        help="Model path for tokenizer")
    parser.add_argument("--url", type=str, default="http://localhost:30000",
                        help="SGLang server URL")
    parser.add_argument("--max-concurrent", type=int, default=2,
                        help="Max concurrent prefill requests")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-request timeout seconds")
    parser.add_argument("--qids", type=int, nargs="*", default=None,
                        help="Only process these question IDs (default: all)")
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line.strip()) for line in f]

    # Init tokenizer
    print(f"Initializing tokenizer for {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Check server
    if not asyncio.run(check_server_health(args.url)):
        raise RuntimeError(f"SGLang server at {args.url} not reachable")
    print(f"SGLang server at {args.url} is healthy")

    # Find input pickle files
    pkl_files = sorted(glob.glob(os.path.join(args.input_dir, "*.pkl")))
    print(f"Found {len(pkl_files)} pickle files in {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Check which are already done
    existing_outputs = set(os.path.basename(f) for f in glob.glob(os.path.join(args.output_dir, "*.pkl")))

    total_start = time.time()

    for pkl_path in pkl_files:
        basename = os.path.basename(pkl_path)

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        qid = data["question_id"]

        # Filter by qids if specified
        if args.qids is not None and qid not in args.qids:
            continue

        # Check if already done
        out_path = os.path.join(args.output_dir, basename)
        if basename in existing_outputs:
            # Check if confs are actually filled
            with open(out_path, "rb") as f:
                existing = pickle.load(f)
            first_confs = existing["all_traces"][0].get("confs", []) if existing["all_traces"] else []
            if len(first_confs) > 0:
                print(f"[SKIP] qid={qid} already has confs ({len(existing['all_traces'])} traces)")
                continue

        all_traces = data["all_traces"]
        ground_truth = data["ground_truth"]
        question_data = dataset[qid]
        num_traces = len(all_traces)

        print(f"\n[QID {qid}] Processing {num_traces} traces...")

        # Prepare prompt
        prompt_text, _ = prepare_prompt(question_data, tokenizer)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        # Tokenize all trace texts
        print(f"  Tokenizing {num_traces} traces...")
        tok_start = time.time()
        all_gen_ids = []
        all_input_ids = []
        for trace in all_traces:
            gen_ids = tokenizer.encode(trace["text"], add_special_tokens=False)
            all_gen_ids.append(gen_ids)
            all_input_ids.append(prompt_ids + gen_ids)
        tok_time = time.time() - tok_start
        avg_tokens = sum(len(ids) for ids in all_gen_ids) / len(all_gen_ids)
        batch_size = pick_batch_size(avg_tokens)
        print(f"  Tokenized in {tok_time:.1f}s (prompt={prompt_len} tokens, avg_gen={avg_tokens:.0f} tokens, batch_size={batch_size})")

        # Load checkpoint if exists (resume within a question)
        ckpt_path = os.path.join(args.output_dir, f".ckpt_qid{qid}.pkl")
        all_confs = [None] * num_traces
        all_token_ids_resolved = [None] * num_traces
        resume_from = 0
        total_errors = 0

        if os.path.exists(ckpt_path):
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f)
            resume_from = ckpt.get("next_batch_start", 0)
            saved_confs = ckpt.get("confs", {})
            saved_tids = ckpt.get("token_ids", {})
            for idx, c in saved_confs.items():
                all_confs[idx] = c
            for idx, t in saved_tids.items():
                all_token_ids_resolved[idx] = t
            filled_so_far = sum(1 for c in all_confs if c is not None and len(c) > 0)
            print(f"  Resuming from batch at trace {resume_from} ({filled_so_far} traces already done)")

        # Process in batches with per-batch checkpointing
        for batch_start in range(resume_from, num_traces, batch_size):
            batch_end = min(batch_start + batch_size, num_traces)
            batch_ids = all_input_ids[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1
            total_batches = (num_traces + batch_size - 1) // batch_size

            print(f"  Batch {batch_num}/{total_batches} (traces {batch_start}-{batch_end-1})...")

            outputs, counters = asyncio.run(
                prefill_batch(args.url, batch_ids, args.max_concurrent, args.timeout)
            )
            total_errors += counters["errors"]

            for i, output in enumerate(outputs):
                idx = batch_start + i
                if output is None:
                    all_confs[idx] = []
                else:
                    all_confs[idx] = extract_confs_from_prefill(output, prompt_len)
                all_token_ids_resolved[idx] = list(all_gen_ids[idx])

            # Save checkpoint after each batch
            ckpt_data = {
                "next_batch_start": batch_end,
                "confs": {i: c for i, c in enumerate(all_confs) if c is not None},
                "token_ids": {i: t for i, t in enumerate(all_token_ids_resolved) if t is not None},
            }
            with open(ckpt_path, "wb") as f:
                pickle.dump(ckpt_data, f)

        # Update traces with confs and token_ids
        updated_traces = []
        for i, trace in enumerate(all_traces):
            trace_copy = dict(trace)
            trace_copy["confs"] = all_confs[i] if all_confs[i] is not None else []
            trace_copy["token_ids"] = all_token_ids_resolved[i] if all_token_ids_resolved[i] is not None else list(all_gen_ids[i])
            trace_copy["num_tokens"] = len(trace_copy["token_ids"])
            updated_traces.append(trace_copy)

        # Update the result
        data["all_traces"] = updated_traces
        total_tokens = sum(t["num_tokens"] for t in updated_traces)
        data["token_stats"]["total_tokens"] = total_tokens
        data["token_stats"]["avg_tokens_per_trace"] = total_tokens / num_traces if num_traces else 0

        # Save final output
        with open(out_path, "wb") as f:
            pickle.dump(data, f)

        # Remove checkpoint
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

        # Stats
        filled = sum(1 for c in all_confs if c and len(c) > 0)
        print(f"  Done: {filled}/{num_traces} traces with confs, {total_errors} errors")
        print(f"  Saved to {out_path}")

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"All done in {total_time:.0f}s ({total_time/60:.1f}m)")


if __name__ == "__main__":
    main()
