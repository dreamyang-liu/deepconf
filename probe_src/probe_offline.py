"""
Probe offline traces at regular token intervals.

For each trace, truncates thinking at every `probe_interval` tokens and asks
the model to give a final \\boxed answer. No candidate hints are provided.
Outputs compact NPZ files with probe results + original confidence data.

Architecture:
  - Phase 1: scan questions, skip completed
  - Phase 2: load vLLM, process one question at a time
  - Per-question: load pkl → build prompts on the fly → vLLM generate → checkpoint
  - Final: save NPZ with probe results + original trace data

Multi-GPU: --parallel N launches N workers (each TP=2) on separate GPU pairs.

Usage:
    # Single worker (GPU 0,1):
    python probe_src/probe_offline.py \\
        --input-dir outputs-bedrock-confs/brumo25 \\
        --output-dir probe_results/brumo25

    # 4 workers (GPU 0-7):
    python probe_src/probe_offline.py \\
        --input-dir outputs-bedrock-confs/brumo25 \\
        --output-dir probe_results/brumo25 \\
        --parallel 4
"""

import argparse
import gc
import glob
import os
import pickle
import subprocess
import sys
import time

import numpy as np
from collections import Counter
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper import extract_answer, equal_func

INJECT_SUFFIX = (
    " Considering the limited time by the user, I have to give the"
    " solution based on the thinking directly now.\n</think>\n\n\\boxed{"
)
GPU_PAIRS = ["0", "1", "2", "3", "4", "5", "6", "7"]
CONDA_PYTHON = "/opt/dlami/nvme/miniconda3/envs/deepconf/bin/python"


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def _extract_qid_from_filename(filename):
    """Extract qid from filename like deepconf_simple_qid5_rid..."""
    import re
    m = re.search(r"qid(\d+)", filename)
    return int(m.group(1)) if m else None


def scan_questions(input_dir, output_dir, qids_filter=None):
    """Return [(qid, pkl_path)] for questions that still need processing."""
    pkl_files = sorted(glob.glob(os.path.join(input_dir, "*.pkl")))
    questions = []
    for pkl_path in pkl_files:
        qid = _extract_qid_from_filename(os.path.basename(pkl_path))
        if qid is None:
            continue
        if qids_filter is not None and qid not in qids_filter:
            continue
        npz_path = os.path.join(output_dir, f"q{qid:02d}.npz")
        if os.path.exists(npz_path):
            continue
        questions.append((qid, pkl_path))
    return questions


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path):
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            return pickle.load(f)
    return {}


def save_checkpoint(ckpt_path, results):
    tmp = ckpt_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(results, f)
    os.replace(tmp, ckpt_path)


# ---------------------------------------------------------------------------
# Per-question processing
# ---------------------------------------------------------------------------

def process_question(llm, tokenizer, qid, pkl_path, output_dir,
                     probe_interval, max_model_len, chunk_size):
    """Load one question, probe all traces at all depths, save NPZ."""
    from vllm import SamplingParams

    # ---- Load data ----
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    traces = data["all_traces"]
    n_traces = len(traces)
    ground_truth = str(data.get("ground_truth", ""))
    question_text = data.get("question", "")

    # ---- Tokenize shared parts ----
    messages = [{"role": "user", "content": question_text}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    inject_ids = tokenizer.encode(INJECT_SUFFIX, add_special_tokens=False)
    max_trace_tokens = max_model_len - len(prompt_ids) - len(inject_ids) - 200

    # ---- Load checkpoint ----
    ckpt_dir = os.path.join(output_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"qid{qid}.pkl")
    results = load_checkpoint(ckpt_path)

    # ---- Build job list: (trace_idx, depth, total_len) ----
    jobs = []
    for ti, trace in enumerate(traces):
        tids = trace.get("token_ids", [])
        if not tids:
            continue
        n = len(tids)
        # Probe at every interval
        for depth in range(probe_interval, n + 1, probe_interval):
            if depth > max_trace_tokens:
                break
            if (ti, depth) not in results:
                jobs.append((ti, depth, len(prompt_ids) + depth + len(inject_ids)))
        # Probe at exact length if not already a multiple
        if n % probe_interval != 0 and n <= max_trace_tokens:
            if (ti, n) not in results:
                jobs.append((ti, n, len(prompt_ids) + n + len(inject_ids)))

    # Sort short-first for efficiency
    jobs.sort(key=lambda x: x[2])

    print(f"  qid={qid}: {len(jobs)} probes to run, {len(results)} from checkpoint")

    # ---- Run probes ----
    if jobs:
        sampling_params = SamplingParams(
            max_tokens=200, temperature=0.6, top_p=0.95, top_k=20,
        )
        t_start = time.time()
        done = 0

        for cs in range(0, len(jobs), chunk_size):
            chunk = jobs[cs : cs + chunk_size]

            # Build prompts on the fly (no pre-allocation of all token_ids)
            prompts = []
            for ti, depth, _ in chunk:
                full_ids = prompt_ids + traces[ti]["token_ids"][:depth] + inject_ids
                prompts.append({"prompt_token_ids": full_ids})

            t0 = time.time()
            outputs = llm.generate(prompts, sampling_params)
            ct = time.time() - t0

            for (ti, depth, _), out in zip(chunk, outputs):
                text = out.outputs[0].text or ""
                # Use brace-matching to handle nested {} in answers like \sqrt{2}
                answer = extract_answer("\\boxed{" + text)
                if not answer:
                    answer = text.strip().rstrip("}").strip() or None

                is_correct = False
                if answer and ground_truth:
                    try:
                        is_correct = equal_func(answer, ground_truth)
                    except Exception:
                        is_correct = str(answer) == str(ground_truth)

                results[(ti, depth)] = {
                    "answer": answer, "is_correct": is_correct, "raw_text": text,
                }

            done += len(chunk)
            del outputs, prompts
            gc.collect()

            # Checkpoint every 4 chunks or at the end
            if done % (chunk_size * 4) < chunk_size or cs + chunk_size >= len(jobs):
                save_checkpoint(ckpt_path, results)

            elapsed = time.time() - t_start
            print(f"    {done}/{len(jobs)} ({ct:.1f}s/chunk, {elapsed:.0f}s total)")

        save_checkpoint(ckpt_path, results)

    # ---- Build NPZ ----
    _save_npz(output_dir, qid, question_text, ground_truth,
              traces, n_traces, results, probe_interval, max_trace_tokens)

    del data, traces
    gc.collect()


def _save_npz(output_dir, qid, question_text, ground_truth,
              traces, n_traces, results, probe_interval, max_trace_tokens):
    """Assemble and save the final NPZ for one question."""

    # Collect all depths that exist in results
    all_depths = sorted(set(d for (_, d) in results))
    if not all_depths:
        print(f"  qid={qid}: no results to save")
        return

    depths = np.array(all_depths, dtype=np.int32)
    n_depths = len(depths)
    d2i = {d: i for i, d in enumerate(depths)}

    # Probe result arrays
    probe_answers = np.full((n_traces, n_depths), None, dtype=object)
    probe_correct = np.zeros((n_traces, n_depths), dtype=np.bool_)

    for (ti, depth), r in results.items():
        if depth in d2i:
            probe_answers[ti, d2i[depth]] = r["answer"]
            probe_correct[ti, d2i[depth]] = r["is_correct"]

    # Original trace arrays
    max_tok = max(len(t.get("confs", [])) for t in traces) if traces else 0
    confs = np.zeros((n_traces, max_tok), dtype=np.float32)
    lengths = np.zeros(n_traces, dtype=np.int32)
    is_correct = np.zeros(n_traces, dtype=np.bool_)
    answers = np.array([t.get("extracted_answer") for t in traces], dtype=object)

    for i, t in enumerate(traces):
        c = t.get("confs", [])
        confs[i, : len(c)] = c
        lengths[i] = len(c)
        is_correct[i] = t.get("is_correct", False)

    npz_path = os.path.join(output_dir, f"q{qid:02d}.npz")
    np.savez_compressed(
        npz_path,
        probe_answers=probe_answers,
        probe_correct=probe_correct,
        depths=depths,
        confs=confs,
        lengths=lengths,
        is_correct=is_correct,
        answers=answers,
        question_id=qid,
        ground_truth=ground_truth,
        question_text=question_text,
    )

    # Print depth summary (first 10 + last)
    show = list(depths[:10])
    if len(depths) > 10 and depths[-1] not in show:
        show.append(depths[-1])
    for depth in show:
        di = d2i[depth]
        probed = probe_answers[:, di]
        n_probed = int(np.sum(probed != None))
        n_correct = int(probe_correct[:, di].sum())
        if n_probed > 0:
            depth_ans = [a for a in probed if a is not None]
            maj = Counter(depth_ans).most_common(1)[0][0] if depth_ans else None
            try:
                maj_ok = equal_func(maj, ground_truth) if maj else False
            except Exception:
                maj_ok = str(maj) == str(ground_truth)
            print(f"    @{depth:>6d}: {n_correct:>4d}/{n_probed:<4d} "
                  f"({n_correct/n_probed:5.1%}), "
                  f"maj={'✓' if maj_ok else '✗'}")

    print(f"  -> {npz_path}")


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------

def worker_main(input_dir, output_dir, qid_set, model_path, tp,
                probe_interval, max_model_len, gpu_mem, chunk_size, gpus):
    """Single worker: own GPU pair, own vLLM, process assigned questions."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
        enforce_eager=True,
        enable_prefix_caching=True,
    )

    pkl_files = sorted(glob.glob(os.path.join(input_dir, "*.pkl")))
    for pkl_path in pkl_files:
        qid = _extract_qid_from_filename(os.path.basename(pkl_path))
        if qid is None or qid not in qid_set:
            continue

        npz_path = os.path.join(output_dir, f"q{qid:02d}.npz")
        if os.path.exists(npz_path):
            print(f"[GPU {gpus}] qid={qid}: done, skip")
            continue

        print(f"\n[GPU {gpus}] qid={qid}: starting...")
        process_question(
            llm, tokenizer, qid, pkl_path, output_dir,
            probe_interval, max_model_len, chunk_size,
        )


# ---------------------------------------------------------------------------
# Main / launcher
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Probe offline traces at regular token intervals"
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--probe-interval", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of GPU-pair workers (max 4)")
    parser.add_argument("--qids", type=int, nargs="*", default=None)
    # Internal: subprocess worker mode
    parser.add_argument("--_gpus", type=str, default=None)
    parser.add_argument("--_qids", type=int, nargs="*", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Subprocess worker mode ----
    if args._gpus is not None:
        worker_main(
            args.input_dir, args.output_dir, set(args._qids),
            args.model_path, args.tp, args.probe_interval,
            args.max_model_len, args.gpu_memory_utilization,
            args.chunk_size, args._gpus,
        )
        return

    # ---- Scan ----
    questions = scan_questions(
        args.input_dir, args.output_dir,
        set(args.qids) if args.qids else None,
    )
    if not questions:
        print("Nothing to process")
        return

    print(f"Questions to probe: {len(questions)}")
    n_workers = min(args.parallel, len(GPU_PAIRS), len(questions))

    if n_workers <= 1:
        # Single-process mode
        worker_main(
            args.input_dir, args.output_dir,
            set(q[0] for q in questions),
            args.model_path, args.tp, args.probe_interval,
            args.max_model_len, args.gpu_memory_utilization,
            args.chunk_size, GPU_PAIRS[0],
        )
    else:
        # Multi-process: distribute qids round-robin, launch subprocesses
        buckets = [[] for _ in range(n_workers)]
        for i, (qid, _) in enumerate(questions):
            buckets[i % n_workers].append(qid)

        print(f"Launching {n_workers} workers:")
        for i in range(n_workers):
            print(f"  Worker {i} (GPU {GPU_PAIRS[i]}): qids {buckets[i]}")

        procs = []
        for i in range(n_workers):
            if not buckets[i]:
                continue
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = GPU_PAIRS[i]
            cmd = [
                CONDA_PYTHON, "-u", os.path.abspath(__file__),
                "--input-dir", args.input_dir,
                "--output-dir", args.output_dir,
                "--model-path", args.model_path,
                "--tp", str(args.tp),
                "--probe-interval", str(args.probe_interval),
                "--chunk-size", str(args.chunk_size),
                "--max-model-len", str(args.max_model_len),
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                "--_gpus", GPU_PAIRS[i],
                "--_qids", *[str(q) for q in buckets[i]],
            ]
            log_path = os.path.join(args.output_dir, f"worker_{i}.log")
            log_f = open(log_path, "w")
            p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
            procs.append((i, p, log_f))
            print(f"  Worker {i} started (pid={p.pid}, log={log_path})")

        for i, p, log_f in procs:
            p.wait()
            log_f.close()
            print(f"  Worker {i} finished (rc={p.returncode})")

        # Summary
        done = len(glob.glob(os.path.join(args.output_dir, "q*.npz")))
        print(f"\nAll workers done. {done} NPZ files in {args.output_dir}")


if __name__ == "__main__":
    main()
