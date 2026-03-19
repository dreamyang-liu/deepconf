"""
Generate DeepConf-compatible offline data from pre-existing inference text.

Given a file where each line is a complete generated text (reasoning trace),
this script does a prefill (forward pass) through the model to recover
per-token logprobs, then computes confidence scores identical to the
DeepConf offline pipeline.

Usage:
    # Start SGLang server first:
    # python -m sglang.launch_server --model-path <MODEL> --tp 8 --trust-remote-code

    python prefill-confs.py \
        --traces-file traces.txt \
        --dataset-file aime_2025.jsonl \
        --qid 0 --rid run0 \
        --url http://localhost:30000 \
        --output-dir outputs-prefill
"""

import asyncio
import aiohttp
import argparse
import json
import os
import pickle
import sys
import time
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer

# Add deepconf to path so we can import helper
sys.path.insert(0, "/sgl-workspace/deepconf")
from helper import (
    extract_answer,
    equal_func,
    prepare_prompt,
    weighted_majority_vote,
    compute_confidence_sglang,
)

MODEL_PATH = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
TOP_LOGPROBS = 20
MAX_RETRIES = 3


async def check_server_health(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{url}/health", timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    print(f"SGLang server at {url} is healthy")
                    return True
    except Exception as e:
        print(f"Cannot reach SGLang server at {url}: {e}")
    return False


async def prefill_one(session, url, input_ids, trace_idx, semaphore, counters):
    """
    Send the full sequence (prompt + generated text) as input_ids,
    request max_new_tokens=1 with return_logprob=True.
    The input token logprobs give us the per-token confidence for the
    generated portion.
    """
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
                    print(f"  [ERROR] Trace {trace_idx} failed after {MAX_RETRIES} retries: {e}")
                    counters["errors"] += 1
                    return None
                await asyncio.sleep(2 ** attempt)

        counters["completed"] += 1
        done = counters["completed"]
        total = counters["total"]
        if done % 50 == 0 or done == total:
            elapsed = time.time() - counters["start_time"]
            print(f"  [{done}/{total}] {done/total:.0%} - {elapsed:.1f}s")

        return output


async def prefill_all(url, all_input_ids, max_concurrent, timeout):
    semaphore = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    counters = {
        "completed": 0,
        "errors": 0,
        "total": len(all_input_ids),
        "start_time": time.time(),
    }

    async with aiohttp.ClientSession(connector=connector, timeout=client_timeout) as session:
        tasks = [
            prefill_one(session, url, ids, i, semaphore, counters)
            for i, ids in enumerate(all_input_ids)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    outputs = []
    for r in results:
        if isinstance(r, Exception):
            print(f"  [ERROR] Exception: {r}")
            outputs.append(None)
        else:
            outputs.append(r)
    return outputs


def extract_confs_from_prefill(output, prompt_len):
    """
    Extract per-token confidence from a prefill output.

    SGLang returns input_top_logprobs for all input tokens.
    We only want the ones corresponding to the generated portion
    (i.e., tokens after prompt_len).

    Each entry in input_top_logprobs is a list of (logprob, token_id, decoded_text).
    """
    meta = output.get("meta_info", {})
    input_top_logprobs = meta.get("input_top_logprobs", [])

    if not input_top_logprobs:
        return []

    # Take only the generation portion
    gen_logprobs = input_top_logprobs[prompt_len:]
    confs = compute_confidence_sglang(gen_logprobs)
    return confs


def build_trace(text, confs, token_ids, ground_truth):
    extracted = extract_answer(text)
    is_correct = False
    if extracted and ground_truth:
        try:
            is_correct = equal_func(extracted, ground_truth)
        except:
            is_correct = str(extracted) == str(ground_truth)

    return {
        "stop_reason": "prefill_scored",
        "text": text,
        "token_ids": token_ids,
        "num_tokens": len(token_ids),
        "confs": confs,
        "extracted_answer": extracted,
        "is_correct": is_correct,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Recover per-token confidence from existing inference text via prefill"
    )
    parser.add_argument("--traces-file", type=str, required=True,
                        help="File with one generated text per line")
    parser.add_argument("--dataset-file", type=str, required=True,
                        help="JSONL dataset file (for question and ground truth)")
    parser.add_argument("--qid", type=int, required=True,
                        help="Question ID (0-based)")
    parser.add_argument("--rid", type=str, default="run0",
                        help="Run ID for naming")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH,
                        help=f"Model path for tokenizer (default: {MODEL_PATH})")
    parser.add_argument("--url", type=str, default="http://localhost:30000",
                        help="SGLang server URL")
    parser.add_argument("--max-concurrent", type=int, default=128,
                        help="Max concurrent requests")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-request timeout seconds")
    parser.add_argument("--output-dir", type=str, default="outputs-prefill",
                        help="Output directory")
    parser.add_argument("--window-size", type=int, default=2048,
                        help="Window size for config (default: 2048)")
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset_file, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]
    if args.qid < 0 or args.qid >= len(data):
        raise ValueError(f"qid {args.qid} out of range (0-{len(data)-1})")
    question_data = data[args.qid]

    # Load traces: supports .jsonl ({"text": "..."} per line) or plain text (one trace per line)
    trace_texts = []
    with open(args.traces_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if args.traces_file.endswith(".jsonl"):
                obj = json.loads(line)
                trace_texts.append(obj["text"])
            else:
                # plain text: unescape literal \n back to real newlines
                trace_texts.append(line.replace("\\n", "\n"))
    print(f"Loaded {len(trace_texts)} traces from {args.traces_file}")

    # Init tokenizer and prepare prompt
    print(f"Initializing tokenizer for {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompt_text, ground_truth = prepare_prompt(question_data, tokenizer)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_len = len(prompt_ids)
    print(f"Prompt length: {prompt_len} tokens")

    # Tokenize each trace and build full input_ids = prompt_ids + gen_ids
    all_input_ids = []
    all_gen_ids = []
    for text in trace_texts:
        gen_ids = tokenizer.encode(text, add_special_tokens=False)
        full_ids = prompt_ids + gen_ids
        all_input_ids.append(full_ids)
        all_gen_ids.append(gen_ids)

    # Check server
    if not asyncio.run(check_server_health(args.url)):
        raise RuntimeError(f"SGLang server at {args.url} not reachable")

    # Prefill all traces
    print(f"\nPrefilling {len(trace_texts)} traces to get logprobs...")
    start_time = time.time()
    outputs = asyncio.run(
        prefill_all(args.url, all_input_ids, args.max_concurrent, args.timeout)
    )
    prefill_time = time.time() - start_time
    print(f"Prefill completed in {prefill_time:.1f}s")

    # Build traces
    all_traces = []
    for i, (text, gen_ids, output) in enumerate(zip(trace_texts, all_gen_ids, outputs)):
        if output is None:
            print(f"  [WARN] Trace {i} failed, using empty confs")
            confs = []
        else:
            confs = extract_confs_from_prefill(output, prompt_len)

        trace = build_trace(text, confs, list(gen_ids), ground_truth)
        all_traces.append(trace)

    # Voting
    voting_answers = [t["extracted_answer"] for t in all_traces if t["extracted_answer"]]
    voting_weights = [1.0] * len(voting_answers)
    voted_answer = weighted_majority_vote(voting_answers, voting_weights)
    is_voted_correct = False
    if voted_answer and ground_truth:
        try:
            is_voted_correct = equal_func(voted_answer, ground_truth)
        except:
            is_voted_correct = str(voted_answer) == str(ground_truth)

    correct_traces = sum(1 for t in all_traces if t["is_correct"])
    total_tokens = sum(t["num_tokens"] for t in all_traces)
    accuracy = correct_traces / len(all_traces) if all_traces else 0

    # Save in the same format as deepconf-offline.py
    problem_result = {
        "question_id": args.qid,
        "run_id": args.rid,
        "question": question_data["question"],
        "ground_truth": ground_truth,
        "all_traces": all_traces,
        "voted_answer": voted_answer,
        "is_voted_correct": is_voted_correct,
        "accuracy": accuracy,
        "correct_traces_count": correct_traces,
        "token_stats": {
            "total_tokens": total_tokens,
            "total_traces_count": len(all_traces),
            "avg_tokens_per_trace": total_tokens / len(all_traces) if all_traces else 0,
        },
        "timing_stats": {
            "prefill_time": prefill_time,
        },
        "config": {
            "model_path": args.model_path,
            "total_budget": len(all_traces),
            "window_size": args.window_size,
        },
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        args.output_dir,
        f"deepconf_simple_qid{args.qid}_rid{args.rid}_{timestamp}.pkl",
    )
    with open(out_path, "wb") as f:
        pickle.dump(problem_result, f)

    print(f"\n=== Summary ===")
    print(f"Traces: {len(all_traces)}")
    print(f"Voted answer: {voted_answer} (gt: {ground_truth}) -> {'CORRECT' if is_voted_correct else 'WRONG'}")
    print(f"Trace accuracy: {correct_traces}/{len(all_traces)} ({accuracy:.1%})")
    print(f"Total tokens: {total_tokens}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
