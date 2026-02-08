"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import asyncio
import aiohttp
from transformers import AutoTokenizer
import json
import time
import pickle
import numpy as np
from datetime import datetime
from helper import (
    equal_func,
    prepare_prompt,
    weighted_majority_vote,
    process_output_offline_sglang,
)
import os
import glob
import argparse

# Configuration
# MODEL_PATH = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
MODEL_PATH = "Qwen/Qwen3-32B"
MAX_TOKENS = 32000
DATASET_FILE = "hmmt_feb_2025.jsonl"

# Algorithm parameters
TOTAL_BUDGET = 4096
WINDOW_SIZE = 2048
REASONING_EFFORT = 'high'

MAX_RETRIES = 3


async def check_server_health(url):
    """Verify the SGLang server is reachable before dispatching requests."""
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


async def generate_one(
    session, url, prompt, trace_id, list_index, semaphore,
    ground_truth, window_size, output_dir,
    results_list, counters, progress_interval,
):
    """Send one /generate request, process the response, and save immediately.

    Args:
        trace_id: Global ID used for the filename (trace_{trace_id:04d}.pkl)
        list_index: Local index into results_list for this batch
    """
    async with semaphore:
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_new_tokens": MAX_TOKENS,
            },
            "return_logprob": True,
            "top_logprobs_num": 20,
        }

        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(f"{url}/generate", json=payload) as resp:
                    resp.raise_for_status()
                    output = await resp.json()
                break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"  [ERROR] Trace {trace_id:04d} failed after {MAX_RETRIES} retries: {e}")
                    counters['errors'] += 1
                    return
                await asyncio.sleep(2 ** attempt)

    # Process using existing helper function
    trace_data = process_output_offline_sglang(output, ground_truth, window_size)

    # Save individual trace immediately to disk
    trace_path = os.path.join(output_dir, f"trace_{trace_id:04d}.pkl")
    with open(trace_path, 'wb') as f:
        pickle.dump(trace_data, f)

    # Accumulate for aggregation (local index into this batch's list)
    results_list[list_index] = trace_data

    # Progress reporting
    counters['completed'] += 1
    completed = counters['completed']
    if completed % progress_interval == 0 or completed == counters['total']:
        elapsed = time.time() - counters['start_time']
        print(f"  Progress: {completed}/{counters['total']} "
              f"({completed / counters['total']:.1%}) - {elapsed:.1f}s elapsed")


async def generate_all(
    url, prompt, total_budget, ground_truth, window_size,
    output_dir, max_concurrent, timeout, progress_interval,
    id_offset=0,
):
    """Fire off total_budget async requests and collect results."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results_list = [None] * total_budget
    counters = {
        'completed': 0,
        'errors': 0,
        'total': total_budget,
        'start_time': time.time(),
    }

    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    client_timeout = aiohttp.ClientTimeout(total=timeout)

    async with aiohttp.ClientSession(connector=connector, timeout=client_timeout) as session:
        tasks = [
            generate_one(
                session, url, prompt, id_offset + i, i, semaphore,
                ground_truth, window_size, output_dir,
                results_list, counters, progress_interval,
            )
            for i in range(total_budget)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out None entries (from failed requests)
    successful_results = [r for r in results_list if r is not None]
    return successful_results, counters


def process_question(qid, rid, question_data, tokenizer, server_url,
                     max_concurrent, timeout, progress_interval, output_dir="outputs"):
    """
    Process a single question: prepare prompt, generate traces, vote, save results.

    Returns:
        dict: problem_result with all traces, voted answer, and statistics
    """
    question_start = time.time()

    print(f"\n{'='*60}")
    print(f"Processing question {qid}: {question_data['question'][:100]}...")
    print(f"{'='*60}")

    # Prepare prompt
    prompt_prep_start = time.time()
    prompt, ground_truth = prepare_prompt(question_data, tokenizer)
    prompt_prep_time = time.time() - prompt_prep_start

    # Stable trace output directory (no timestamp, so we can resume)
    model_name = MODEL_PATH.split('/')[-1]
    dataset_name = DATASET_FILE.split('.')[0]
    trace_dir = f"{output_dir}/{model_name}/{dataset_name}/traces/qid{qid}_rid{rid}"
    os.makedirs(trace_dir, exist_ok=True)

    # Scan for existing traces to support resumption
    existing_files = sorted(glob.glob(os.path.join(trace_dir, "trace_*.pkl")))
    existing_count = len(existing_files)
    remaining = TOTAL_BUDGET - existing_count

    if remaining <= 0:
        print(f"  Already have {existing_count}/{TOTAL_BUDGET} traces, skipping generation.")
        # Load all existing traces
        all_traces = []
        for f in existing_files[:TOTAL_BUDGET]:
            with open(f, 'rb') as fh:
                all_traces.append(pickle.load(fh))
        generation_time = 0.0
        counters = {'completed': existing_count, 'errors': 0, 'total': TOTAL_BUDGET, 'start_time': time.time()}
    else:
        # Load existing traces
        existing_traces = []
        for f in existing_files:
            with open(f, 'rb') as fh:
                existing_traces.append(pickle.load(fh))
        print(f"  Found {existing_count} existing traces, generating {remaining} more...")

        generation_start = time.time()
        new_traces, counters = asyncio.run(
            generate_all(
                url=server_url,
                prompt=prompt,
                total_budget=remaining,
                ground_truth=ground_truth,
                window_size=WINDOW_SIZE,
                output_dir=trace_dir,
                max_concurrent=max_concurrent,
                timeout=timeout,
                progress_interval=progress_interval,
                id_offset=existing_count,
            )
        )
        generation_time = time.time() - generation_start
        all_traces = existing_traces + new_traces
        print(f"  Generation completed: {generation_time:.2f}s "
              f"({len(new_traces)} new, {existing_count} existing, {counters['errors']} errors)")

    total_tokens = sum(t['num_tokens'] for t in all_traces)

    # Voting for final answer
    voting_answers = []
    voting_weights = []

    for trace in all_traces:
        if trace['extracted_answer']:
            voting_answers.append(trace['extracted_answer'])
            voting_weights.append(1.0)

    print(f'  Voting candidates: {len(voting_answers)}')

    voted_answer = weighted_majority_vote(voting_answers, voting_weights)
    is_voted_correct = False
    if voted_answer and ground_truth:
        try:
            is_voted_correct = equal_func(voted_answer, ground_truth)
        except:
            is_voted_correct = str(voted_answer) == str(ground_truth)

    question_time = time.time() - question_start

    # Calculate statistics
    correct_traces = sum(1 for trace in all_traces if trace['is_correct'])
    accuracy = correct_traces / len(all_traces) if all_traces else 0

    problem_result = {
        "question_id": qid,
        "run_id": rid,
        "question": question_data['question'],
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
            "errors_count": counters['errors'],
        },
        "timing_stats": {
            "prompt_prep_time": prompt_prep_time,
            "generation_time": generation_time,
            "question_time": question_time,
        },
        "config": {
            "model_path": MODEL_PATH,
            "total_budget": TOTAL_BUDGET,
            "window_size": WINDOW_SIZE,
            "server_url": server_url,
            "max_concurrent": max_concurrent,
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Save per-question aggregated result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"{output_dir}/{model_name}/{dataset_name}/deepconf_simple_qid{qid}_rid{rid}_{timestamp}.pkl"
    os.makedirs(os.path.dirname(result_filename), exist_ok=True)
    with open(result_filename, 'wb') as f:
        pickle.dump(problem_result, f)

    # Print per-question summary
    status = "CORRECT" if is_voted_correct else "WRONG"
    print(f"  Result: [{status}] voted={voted_answer}, gt={ground_truth}, "
          f"trace_acc={correct_traces}/{len(all_traces)} ({accuracy:.1%}), "
          f"tokens={total_tokens}, time={question_time:.1f}s")
    print(f"  Saved to {result_filename}")

    return problem_result


def main(qids, rid, server_url, max_concurrent, timeout, progress_interval, output_dir="outputs"):
    """
    Main function to process questions.

    Args:
        qids (list[int]): List of question IDs to process
        rid (str): Run ID for file naming
        server_url (str): SGLang server base URL
        max_concurrent (int): Maximum concurrent HTTP requests
        timeout (int): HTTP request timeout in seconds
        progress_interval (int): Print progress every N completions
    """
    os.makedirs(output_dir, exist_ok=True)
    total_start_time = time.time()

    # Load data
    print(f"Loading data from {DATASET_FILE}...")
    with open(DATASET_FILE, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file]
    print(f"Loaded {len(data)} questions")

    # Validate qids
    for qid in qids:
        if qid < 0 or qid >= len(data):
            raise ValueError(f"Question ID {qid} is out of range (0-{len(data)-1})")

    # Initialize tokenizer (once)
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Check server health (once)
    if not asyncio.run(check_server_health(server_url)):
        raise RuntimeError(
            f"SGLang server at {server_url} is not reachable. "
            f"Start it with: python -m sglang.launch_server "
            f"--model-path {MODEL_PATH} --tp 8 --trust-remote-code"
        )

    print(f"\nWill process {len(qids)} questions: {qids}")

    # Process each question
    all_results = []
    for i, qid in enumerate(qids):
        print(f"\n[{i+1}/{len(qids)}] Question {qid}")
        result = process_question(
            qid, rid, data[qid], tokenizer,
            server_url, max_concurrent, timeout, progress_interval,
            output_dir=output_dir,
        )
        all_results.append(result)

    # Overall summary
    total_time = time.time() - total_start_time
    correct_count = sum(1 for r in all_results if r['is_voted_correct'])

    print(f"\n{'='*60}")
    print(f"=== Overall Summary (rid={rid}) ===")
    print(f"{'='*60}")
    print(f"Questions: {len(all_results)}")
    print(f"Voted correct: {correct_count}/{len(all_results)} ({correct_count/len(all_results):.1%})")
    print(f"Total time: {total_time:.1f}s")

    for r in all_results:
        status = "OK" if r['is_voted_correct'] else "X "
        print(f"  [{status}] qid={r['question_id']:3d}  "
              f"voted={str(r['voted_answer'])[:30]:30s}  "
              f"gt={str(r['ground_truth'])[:30]:30s}  "
              f"trace_acc={r['accuracy']:.1%}")

    return all_results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Process questions with DeepConf (SGLang server mode)'
    )
    parser.add_argument('--qid', type=int, nargs='*', default=None,
                        help='Question ID(s) to process. If not specified, process all questions.')
    parser.add_argument('--rid', type=str, required=True,
                        help='Run ID for file naming')
    parser.add_argument('--url', type=str, default='http://localhost:30000',
                        help='SGLang server base URL (default: http://localhost:30000)')
    parser.add_argument('--max-concurrent', type=int, default=256,
                        help='Maximum concurrent HTTP requests (default: 256)')
    parser.add_argument('--timeout', type=int, default=1800,
                        help='Per-request timeout in seconds (default: 1800)')
    parser.add_argument('--progress-interval', type=int, default=100,
                        help='Print progress every N completions (default: 100)')
    parser.add_argument('--output-dir', type=str, default='outputs-final',
                        help='Base output directory (default: outputs)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Determine which questions to process
    if args.qid is not None:
        qids = args.qid
    else:
        # Process all questions
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            num_questions = sum(1 for _ in f)
        qids = list(range(num_questions))

    results = main(
        qids, args.rid,
        server_url=args.url,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
        progress_interval=args.progress_interval,
        output_dir=args.output_dir,
    )
