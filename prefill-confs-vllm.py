"""
Recover per-token confidence from existing inference text via vLLM prefill.

Processes traces in chunks sorted by length (short first). Saves per-trace
checkpoints so crashes on long traces don't lose previous work.

Usage:
    conda run -n deepconf python prefill-confs-vllm.py \
        --input-dir outputs-bedrock/aime24 \
        --dataset-file aime_2024.jsonl \
        --model-path Qwen/Qwen3-32B \
        --tp 2 \
        --output-dir outputs-bedrock-confs/aime24
"""

import argparse
import glob
import json
import os
import pickle
import time
import gc
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from helper import (
    extract_answer,
    equal_func,
    weighted_majority_vote,
    prepare_prompt,
)


TOP_LOGPROBS = 20


def compute_confidence_from_prompt_logprobs(prompt_logprobs, prompt_len):
    gen_logprobs = prompt_logprobs[prompt_len:]
    confs = []
    for token_lps in gen_logprobs:
        if token_lps is None:
            continue
        if token_lps:
            mean_lp = np.mean([lp.logprob for lp in token_lps.values()])
            confs.append(round(-mean_lp, 3))
    return confs


def pretokenize_question(tokenizer, pickle_path, prompt_text, output_dir, max_model_len):
    """Pre-tokenize all traces for a question. Must be called BEFORE vLLM init."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    qid = data["question_id"]
    all_traces = data["all_traces"]
    n_traces = len(all_traces)
    basename = os.path.basename(pickle_path)

    # Check if final output already done
    out_path = os.path.join(output_dir, basename)
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            existing = pickle.load(f)
        if existing["all_traces"] and len(existing["all_traces"][0].get("confs", [])) > 0:
            print(f"  qid={qid}: already done, skipping")
            return None

    # Load existing checkpoints
    ckpt_dir = os.path.join(output_dir, f"ckpt_qid{qid}")
    existing_confs = {}
    if os.path.exists(ckpt_dir):
        for ckpt_file in glob.glob(os.path.join(ckpt_dir, "trace_*.pkl")):
            idx = int(os.path.basename(ckpt_file).replace("trace_", "").replace(".pkl", ""))
            with open(ckpt_file, "rb") as f:
                existing_confs[idx] = pickle.load(f)

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    to_process = [(idx, trace["text"])
                  for idx, trace in enumerate(all_traces) if idx not in existing_confs]

    n_todo = len(to_process)
    n_workers = 16
    print(f"    Pre-tokenizing {n_todo} traces with {n_workers} threads...")
    pretok_start = time.time()

    # Split into chunks and tokenize in parallel using threads
    # (threads share the tokenizer in memory, no HF downloads, safe before vLLM init)
    chunk_size_tok = max(1, (n_todo + n_workers - 1) // n_workers)
    sub_chunks = [to_process[i:i+chunk_size_tok] for i in range(0, n_todo, chunk_size_tok)]

    def _tok_subchunk(sub):
        full_texts = [prompt_text + text for _, text in sub]
        all_ids = tokenizer(full_texts, add_special_tokens=False)["input_ids"]
        res = []
        for (idx, _), full_ids in zip(sub, all_ids):
            if len(full_ids) > max_model_len - 2:
                res.append((idx, None, len(full_ids) - prompt_len))
            else:
                res.append((idx, full_ids, len(full_ids)))
        return res

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_tok_subchunk, sub_chunks))

    candidates = []
    skipped = 0
    for chunk_result in results:
        for idx, full_ids, length in chunk_result:
            if full_ids is None:
                skipped += 1
                all_traces[idx]["num_tokens"] = length
            else:
                candidates.append((idx, full_ids, length))
    pretok_time = time.time() - pretok_start
    print(f"    Pre-tokenized {len(candidates)} traces in {pretok_time:.1f}s ({n_workers} threads)")

    if skipped:
        print(f"    Skipped {skipped} traces (exceed max_model_len)")

    candidates.sort(key=lambda x: x[2])

    return {
        "qid": qid, "data": data, "all_traces": all_traces,
        "candidates": candidates, "existing_confs": existing_confs,
        "prompt_len": prompt_len, "basename": basename,
        "skipped": skipped,
    }


def process_question(llm, prepped, output_dir, chunk_size):
    qid = prepped["qid"]
    data = prepped["data"]
    all_traces = prepped["all_traces"]
    n_traces = len(all_traces)
    candidates = prepped["candidates"]
    existing_confs = prepped["existing_confs"]
    prompt_len = prepped["prompt_len"]
    basename = prepped["basename"]

    out_path = os.path.join(output_dir, basename)
    ckpt_dir = os.path.join(output_dir, f"ckpt_qid{qid}")
    os.makedirs(ckpt_dir, exist_ok=True)

    if existing_confs:
        print(f"    Resuming: {len(existing_confs)}/{n_traces} already done")

    skipped = prepped["skipped"]

    n_to_process = len(candidates)
    if n_to_process == 0:
        print(f"    Nothing to process")
    else:
        print(f"    Processing {n_to_process} traces (sorted by length)...")

        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            prompt_logprobs=TOP_LOGPROBS,
        )

        total_start = time.time()
        processed = 0
        io_executor = ThreadPoolExecutor(max_workers=4)
        io_futures = []

        def _save_checkpoint(ckpt_path, data):
            with open(ckpt_path, "wb") as f:
                pickle.dump(data, f)

        # Process in chunks (using pre-tokenized IDs to skip CPU rendering)
        for chunk_start in range(0, n_to_process, chunk_size):
            # Wait for previous chunk's I/O to finish before reusing memory
            for fut in io_futures:
                fut.result()
            io_futures.clear()

            chunk_end = min(chunk_start + chunk_size, n_to_process)
            chunk = candidates[chunk_start:chunk_end]
            chunk_prompts = [{"prompt_token_ids": c[1]} for c in chunk]
            chunk_indices = [c[0] for c in chunk]
            max_est = chunk[-1][2]

            t0 = time.time()
            outputs = llm.generate(chunk_prompts, sampling_params)
            chunk_time = time.time() - t0

            chunk_tokens = 0
            for i, output in enumerate(outputs):
                trace_idx = chunk_indices[i]
                if output.prompt_logprobs:
                    confs = compute_confidence_from_prompt_logprobs(
                        output.prompt_logprobs, prompt_len
                    )
                else:
                    confs = []
                num_tokens = len(confs)
                chunk_tokens += num_tokens
                ckpt_data = {"confs": confs, "num_tokens": num_tokens}
                existing_confs[trace_idx] = ckpt_data

                # Async checkpoint write
                ckpt_path = os.path.join(ckpt_dir, f"trace_{trace_idx:04d}.pkl")
                io_futures.append(io_executor.submit(_save_checkpoint, ckpt_path, ckpt_data))

            processed += len(chunk)
            del outputs
            gc.collect()

            elapsed = time.time() - total_start
            print(f"    {processed}/{n_to_process} "
                  f"(chunk {chunk_time:.1f}s, max_est ~{max_est} tok, "
                  f"total {elapsed:.0f}s, "
                  f"~{chunk_tokens/max(chunk_time, 0.1):.0f} tok/s)")

        # Wait for remaining I/O
        for fut in io_futures:
            fut.result()
        io_executor.shutdown(wait=False)

    # Assemble final pickle
    total_tokens = 0
    for idx, trace in enumerate(all_traces):
        if idx in existing_confs:
            trace["confs"] = existing_confs[idx]["confs"]
            trace["num_tokens"] = existing_confs[idx]["num_tokens"]
        total_tokens += trace["num_tokens"]

    data["token_stats"]["total_tokens"] = total_tokens
    data["token_stats"]["avg_tokens_per_trace"] = (
        total_tokens / n_traces if n_traces else 0
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    print(f"  qid={qid}: {n_traces} traces, {total_tokens} tokens -> {out_path}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Recover per-token confidence via vLLM prefill"
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--dataset-file", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Traces per generate() call (default: 512)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--qids", type=int, nargs="*", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir.rstrip("/") + "-confs"

    with open(args.dataset_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line.strip()) for line in f]
    print(f"Loaded {len(dataset)} questions from {args.dataset_file}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    prompts = {}
    for i, q in enumerate(dataset):
        prompt_text, ground_truth = prepare_prompt(q, tokenizer)
        prompts[i] = (prompt_text, ground_truth)

    # Phase 1: Pre-tokenize all questions BEFORE loading model (avoids fork+GPU conflicts)
    pickle_files = sorted(glob.glob(os.path.join(args.input_dir, "*.pkl")))
    print(f"Found {len(pickle_files)} pickle files")

    prepped_questions = []
    for pkl_path in pickle_files:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        qid = data["question_id"]

        if args.qids is not None and qid not in args.qids:
            continue
        if qid not in prompts:
            continue

        prompt_text, ground_truth = prompts[qid]
        print(f"\nPre-tokenizing qid={qid} ({len(data['all_traces'])} traces)...")

        prepped = pretokenize_question(
            tokenizer, pkl_path, prompt_text,
            args.output_dir, args.max_model_len,
        )
        if prepped is not None:
            prepped_questions.append(prepped)

    print(f"\nPre-tokenized {len(prepped_questions)} questions")

    if not prepped_questions:
        print("Nothing to process, exiting")
        return

    # Phase 2: Load model and process
    print(f"Loading model: {args.model_path} (tp={args.tp})")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=True,
    )

    total_start = time.time()
    results = []

    for prepped in prepped_questions:
        qid = prepped["qid"]
        print(f"\nProcessing qid={qid} ({len(prepped['all_traces'])} traces)...")

        result = process_question(
            llm, prepped, args.output_dir, args.chunk_size,
        )
        results.append(result)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"DONE: {len(results)} questions in {total_time:.1f}s ({total_time/3600:.1f}h)")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
