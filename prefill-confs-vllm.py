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
import multiprocessing as mp
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from helper import (
    extract_answer,
    equal_func,
    weighted_majority_vote,
    prepare_prompt,
)


from operator import attrgetter

TOP_LOGPROBS = 20
_GET_LOGPROB = attrgetter("logprob")


def compute_confidence_from_prompt_logprobs(prompt_logprobs, prompt_len):
    """Per-trace confs = -mean(top-k logprobs) per generated-token position."""
    gen_logprobs = prompt_logprobs[prompt_len:]
    rows = [[_GET_LOGPROB(v) for v in tlp.values()] for tlp in gen_logprobs if tlp]
    if not rows:
        return []
    arr = np.asarray(rows, dtype=np.float32)
    means = arr.mean(axis=1)
    return np.round(-means, 3).tolist()


def compute_confs_batched(outputs, prompt_len):
    """Run confidence computation once across every trace's prompt_logprobs.

    Some positions may have fewer than top-k logprobs (edge cases in vLLM),
    so we compute the mean per row with pure-Python sum()/len() — still fast
    for 20-element rows and avoids the "inhomogeneous shape" crash from
    np.asarray on jagged lists.

    Returns a list of length len(outputs): confs list per trace (may be []).
    """
    # Flatten each row to just its logprob floats (sum/len doesn't need numpy).
    means = []
    boundaries = [0]
    for output in outputs:
        plp = output.prompt_logprobs
        if plp:
            gen_lps = plp[prompt_len:]
            for tlp in gen_lps:
                if tlp:
                    vals = [_GET_LOGPROB(v) for v in tlp.values()]
                    if vals:
                        means.append(-sum(vals) / len(vals))
        boundaries.append(len(means))

    if not means:
        return [[] for _ in outputs]

    neg_means = np.round(np.asarray(means, dtype=np.float32), 3)
    result = []
    for i in range(len(outputs)):
        start, end = boundaries[i], boundaries[i + 1]
        result.append(neg_means[start:end].tolist() if end > start else [])
    return result


# ---------------------------------------------------------------------------
# Multi-process tokenization (initialized once per child via fork inheritance).
# ---------------------------------------------------------------------------
_PRETOK_TOK = None


def _pretok_init(tokenizer):
    global _PRETOK_TOK
    _PRETOK_TOK = tokenizer


def _pretok_subchunk(task):
    """Tokenize one sub-chunk of traces; returns list of (idx, ids_or_None, length)."""
    prompt_text, sub, max_model_len, prompt_len = task
    full_texts = [prompt_text + text for _, text in sub]
    all_ids = _PRETOK_TOK(full_texts, add_special_tokens=False)["input_ids"]
    res = []
    for (idx, _), full_ids in zip(sub, all_ids):
        if len(full_ids) > max_model_len - 2:
            res.append((idx, None, len(full_ids) - prompt_len))
        else:
            res.append((idx, full_ids, len(full_ids)))
    return res


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
    n_workers = 8
    print(f"    Pre-tokenizing {n_todo} traces with {n_workers} processes...")
    pretok_start = time.time()

    # Split into chunks and tokenize in parallel using true multi-processing
    # (fork inherits the tokenizer from parent; safe BEFORE vLLM init when
    # CUDA has not been touched yet).
    chunk_size_tok = max(1, (n_todo + n_workers - 1) // n_workers)
    sub_chunks = [to_process[i:i+chunk_size_tok] for i in range(0, n_todo, chunk_size_tok)]

    # Hand off (prefix, sub, max_model_len) so the worker has no closure state.
    tasks = [(prompt_text, sub, max_model_len, prompt_len) for sub in sub_chunks]

    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=n_workers, mp_context=ctx,
        initializer=_pretok_init, initargs=(tokenizer,),
    ) as ex:
        results = list(ex.map(_pretok_subchunk, tasks))

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
        # 4 post-processing threads can run in parallel; the main loop never
        # blocks on them — it fires-and-forgets and drains at the end.
        post_executor = ThreadPoolExecutor(max_workers=4)
        post_futures = []

        def _save_checkpoint(ckpt_path, data):
            with open(ckpt_path, "wb") as f:
                pickle.dump(data, f)

        def _post_process_chunk(outputs, chunk_indices_copy, prompt_len_cap):
            """Runs in background: compute confs, save ckpts. Never blocks main."""
            confs_per_trace = compute_confs_batched(outputs, prompt_len_cap)
            for i in range(len(outputs)):
                trace_idx = chunk_indices_copy[i]
                confs = confs_per_trace[i]
                n = len(confs)
                ckpt_data = {"confs": confs, "num_tokens": n}
                existing_confs[trace_idx] = ckpt_data
                ckpt_path = os.path.join(ckpt_dir, f"trace_{trace_idx:04d}.pkl")
                io_executor.submit(_save_checkpoint, ckpt_path, ckpt_data)

        # Process in chunks (using pre-tokenized IDs to skip CPU rendering).
        # Main loop = GPU only. Post-processing is fire-and-forget to the
        # post_executor, never blocking the next llm.generate call.
        prev_loop_end = None
        for chunk_start in range(0, n_to_process, chunk_size):
            loop_start = time.time()
            gap_since_prev = (loop_start - prev_loop_end) if prev_loop_end else 0.0

            chunk_end = min(chunk_start + chunk_size, n_to_process)
            chunk = candidates[chunk_start:chunk_end]
            chunk_prompts = [{"prompt_token_ids": c[1]} for c in chunk]
            chunk_indices = [c[0] for c in chunk]
            max_est = chunk[-1][2]
            prep_time = time.time() - loop_start

            t0 = time.time()
            outputs = llm.generate(chunk_prompts, sampling_params)
            chunk_time = time.time() - t0

            submit_start = time.time()
            post_futures.append(post_executor.submit(
                _post_process_chunk, outputs, chunk_indices, prompt_len
            ))
            submit_time = time.time() - submit_start

            processed += len(chunk)
            elapsed = time.time() - total_start
            in_flight = sum(1 for f in post_futures if not f.done())
            done_posts = sum(1 for f in post_futures if f.done())
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] {processed}/{n_to_process} "
                  f"chunk[prep {prep_time*1000:.0f}ms, GPU {chunk_time:.1f}s, "
                  f"submit {submit_time*1000:.0f}ms, gap-since-prev {gap_since_prev:.1f}s] "
                  f"post[inflight {in_flight}, done {done_posts}] "
                  f"max_est ~{max_est} total {elapsed:.0f}s", flush=True)

            del outputs
            prev_loop_end = time.time()

        # Drain all post-processing tasks and pending I/O saves.
        drain_start = time.time()
        for fut in post_futures:
            fut.result()
        print(f"    drained {len(post_futures)} post tasks in {time.time()-drain_start:.1f}s")
        post_executor.shutdown(wait=True)
        io_executor.shutdown(wait=True)

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
