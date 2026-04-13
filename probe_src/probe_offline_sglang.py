"""
Probe offline traces using SGLang servers with radix attention KV cache reuse.

Each trace's probes (at increasing depths) are sent sequentially to the same
server, so SGLang reuses KV cache from the shorter prefix — only the new
tokens need computation.

## Quick Start

    # Step 1: Launch 8 SGLang servers (one per GPU, runs in foreground)
    # Do this in a separate terminal / tmux session:
    conda run -n deepconf python probe_src/probe_offline_sglang.py \
        --launch-servers --num-servers 8

    # Step 2: Run probes (in another terminal, after servers are ready):
    conda run -n deepconf python probe_src/probe_offline_sglang.py \
        --input-dir outputs-bedrock-confs/brumo25 \
        --output-dir probe_results/brumo25

## Running All 3 Datasets

    # brumo25
    python probe_src/probe_offline_sglang.py \
        --input-dir outputs-bedrock-confs/brumo25 \
        --output-dir probe_results/brumo25

    # hmmt
    python probe_src/probe_offline_sglang.py \
        --input-dir outputs-bedrock-confs/hmmt \
        --output-dir probe_results/hmmt

    # aime25
    python probe_src/probe_offline_sglang.py \
        --input-dir outputs-bedrock-confs/aime25 \
        --output-dir probe_results/aime25

## Options

    --num-servers N         Number of SGLang servers / GPUs (default: 8)
    --concurrency-per-server N  Max concurrent traces per server (default: 32)
                                Lower this (e.g. 8) for long traces to avoid OOM
    --probe-interval N      Probe every N tokens (default: 2048)
    --max-model-len N       Max sequence length (default: 40960)
    --qids 0 5 10           Only process specific question IDs
    --model-path PATH       Model (default: Qwen/Qwen3-32B)
    --mem-fraction F        GPU memory fraction for SGLang (default: 0.85)

## Architecture

    8 SGLang servers (one per GPU, radix KV cache enabled)
           ↑
    Async HTTP client (one process, asyncio)
      - Loads one question at a time
      - 4096 traces assigned to servers round-robin (sticky routing)
      - Per trace: sends probes at depths [2048, 4096, 6144, ...] sequentially
      - SGLang reuses KV cache from shorter depths automatically
           ↓
    Checkpoint: probe_results/{dataset}/ckpt/qid{N}.pkl
    Final:      probe_results/{dataset}/q{NN}.npz

## Checkpointing & Resume

    - Saves every 256 traces (configurable)
    - Crash-safe: restart the probe command, it skips completed probes
    - Completed questions (NPZ exists) are skipped entirely

## Output Format (NPZ)

    probe_answers:  (n_traces, n_depths) object  — answer at each (trace, depth)
    probe_correct:  (n_traces, n_depths) bool     — whether correct
    depths:         (n_depths,) int32             — [2048, 4096, ...]
    confs:          (n_traces, max_tokens) float32 — original confidence scores
    lengths:        (n_traces,) int32             — original trace lengths
    is_correct:     (n_traces,) bool              — original trace correctness
    answers:        (n_traces,) object            — original extracted answers
    question_id, ground_truth, question_text      — metadata

## Monitoring

    # Check progress:
    ls probe_results/brumo25/q*.npz | wc -l    # completed questions
    tail -1 /tmp/probe_sglang_test.log          # latest throughput

    # Check GPU:
    nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader
"""

import argparse
import asyncio
import gc
import glob
import json
import os
import pickle
import re
import subprocess
import sys
import time

import aiohttp
import numpy as np
from collections import Counter
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper import extract_answer, equal_func

INJECT_SUFFIX = (
    " Considering the limited time by the user, I have to give the"
    " solution based on the thinking directly now.\n</think>\n\n\\boxed{"
)
BASE_PORT = 30000
MODEL_PATH = "Qwen/Qwen3-32B"
CONDA_PYTHON = "/opt/dlami/nvme/miniconda3/envs/deepconf/bin/python"


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def launch_servers(num_servers, model_path, mem_fraction=0.85):
    """Launch SGLang servers, one per GPU."""
    procs = []
    for i in range(num_servers):
        port = BASE_PORT + i
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        conda_bin = os.path.dirname(CONDA_PYTHON)
        env["PATH"] = conda_bin + ":" + env.get("PATH", "")
        cmd = [
            CONDA_PYTHON, "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--port", str(port),
            "--tp", "1",
            "--mem-fraction-static", str(mem_fraction),
        ]
        log_path = f"/tmp/sglang_server_{i}.log"
        log_f = open(log_path, "w")
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((i, p, log_f))
        print(f"  Server {i} (GPU {i}, port {port}): pid={p.pid}, log={log_path}")
    return procs


async def wait_servers_ready(num_servers, timeout=600):
    """Wait until all servers respond to health checks."""
    print(f"Waiting for {num_servers} servers to be ready...")
    start = time.time()
    ready = set()
    async with aiohttp.ClientSession() as session:
        while len(ready) < num_servers and time.time() - start < timeout:
            for i in range(num_servers):
                if i in ready:
                    continue
                try:
                    url = f"http://localhost:{BASE_PORT + i}/health"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status == 200:
                            ready.add(i)
                            print(f"  Server {i} ready ({time.time()-start:.0f}s)")
                except Exception:
                    pass
            if len(ready) < num_servers:
                await asyncio.sleep(5)
    if len(ready) < num_servers:
        missing = set(range(num_servers)) - ready
        raise RuntimeError(f"Servers {missing} not ready after {timeout}s")
    print(f"All {num_servers} servers ready!")


# ---------------------------------------------------------------------------
# Scanning & checkpointing
# ---------------------------------------------------------------------------

def _extract_qid(filename):
    m = re.search(r"qid(\d+)", filename)
    return int(m.group(1)) if m else None


def scan_questions(input_dir, output_dir, qids_filter=None):
    questions = []
    for pkl_path in sorted(glob.glob(os.path.join(input_dir, "*.pkl"))):
        qid = _extract_qid(os.path.basename(pkl_path))
        if qid is None:
            continue
        if qids_filter is not None and qid not in qids_filter:
            continue
        npz_path = os.path.join(output_dir, f"q{qid:02d}.npz")
        if os.path.exists(npz_path):
            continue
        questions.append((qid, pkl_path))
    return questions


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
# Core: async probe for one trace (depths in order → KV cache reuse)
# ---------------------------------------------------------------------------

async def probe_one_trace(
    session, server_url, trace_idx, prompt_ids, trace_token_ids,
    inject_ids, probe_interval, max_model_len, ground_truth, results,
    semaphore, trace_stats,
):
    """Send probes for one trace in depth order to maximize KV cache reuse."""
    async with semaphore:
        n_tokens = len(trace_token_ids)
        max_trace = max_model_len - len(prompt_ids) - len(inject_ids) - 200

        depths = list(range(probe_interval, n_tokens + 1, probe_interval))
        if n_tokens % probe_interval != 0:
            depths.append(n_tokens)

        t_trace_start = time.time()
        n_probed = 0
        n_skipped = 0

        for depth in depths:
            if depth > max_trace:
                break
            if (trace_idx, depth) in results:
                n_skipped += 1
                continue

            full_ids = prompt_ids + trace_token_ids[:depth] + inject_ids

            payload = {
                "input_ids": full_ids,
                "sampling_params": {
                    "max_new_tokens": 200,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                },
            }

            try:
                async with session.post(
                    f"{server_url}/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=600),
                ) as resp:
                    data = await resp.json()
                    text = data.get("text", "")

                answer = extract_answer("\\boxed{" + text)
                if not answer:
                    answer = text.strip().rstrip("}").strip() or None

                is_correct = False
                if answer and ground_truth:
                    try:
                        is_correct = equal_func(answer, ground_truth)
                    except Exception:
                        is_correct = str(answer) == str(ground_truth)

                results[(trace_idx, depth)] = {
                    "answer": answer,
                    "is_correct": is_correct,
                    "raw_text": text,
                }
                n_probed += 1
            except Exception as e:
                print(f"    ERROR trace {trace_idx} @{depth}: {e}")

        elapsed = time.time() - t_trace_start
        trace_stats.append({
            "trace_idx": trace_idx,
            "n_tokens": n_tokens,
            "n_probed": n_probed,
            "n_skipped": n_skipped,
            "elapsed": elapsed,
        })


# ---------------------------------------------------------------------------
# Process one question
# ---------------------------------------------------------------------------

async def process_question_async(
    qid, pkl_path, output_dir, num_servers, probe_interval,
    max_model_len, concurrency_per_server, model_path=MODEL_PATH,
):
    """Process all probes for one question using async HTTP to SGLang servers."""
    # Load data
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    traces = data["all_traces"]
    n_traces = len(traces)
    ground_truth = str(data.get("ground_truth", ""))
    question_text = data.get("question", "")

    # Tokenize shared parts
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    messages = [{"role": "user", "content": question_text}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    inject_ids = tokenizer.encode(INJECT_SUFFIX, add_special_tokens=False)

    # Load checkpoint
    ckpt_dir = os.path.join(output_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"qid{qid}.pkl")
    results = load_checkpoint(ckpt_path)

    # Count total probes needed
    total_probes = 0
    max_trace = max_model_len - len(prompt_ids) - len(inject_ids) - 200
    for trace in traces:
        tids = trace.get("token_ids", [])
        if not tids:
            continue
        n = len(tids)
        for depth in range(probe_interval, n + 1, probe_interval):
            if depth > max_trace:
                break
            total_probes += 1
        if n % probe_interval != 0 and n <= max_trace:
            total_probes += 1

    already_done = len(results)
    remaining = total_probes - already_done
    print(f"  qid={qid}: {remaining} probes to run, {already_done} from checkpoint (total {total_probes})")

    if remaining <= 0:
        _save_npz(output_dir, qid, question_text, ground_truth, traces, results, probe_interval, max_trace)
        return

    # Assign traces to servers round-robin
    server_urls = [f"http://localhost:{BASE_PORT + i}" for i in range(num_servers)]
    semaphore = asyncio.Semaphore(concurrency_per_server * num_servers)

    t_start = time.time()
    trace_stats = []

    async with aiohttp.ClientSession() as session:
        tasks = []
        for ti, trace in enumerate(traces):
            tids = trace.get("token_ids", [])
            if not tids:
                continue
            # Assign to server based on trace index for sticky routing
            server_url = server_urls[ti % num_servers]
            task = probe_one_trace(
                session, server_url, ti, prompt_ids, tids,
                inject_ids, probe_interval, max_model_len,
                ground_truth, results, semaphore, trace_stats,
            )
            tasks.append(task)

        # Process with periodic checkpoint saves
        batch_size = 256
        for batch_start in range(0, len(tasks), batch_size):
            batch = tasks[batch_start:batch_start + batch_size]
            await asyncio.gather(*batch, return_exceptions=True)

            # Checkpoint
            save_checkpoint(ckpt_path, results)
            done_now = len(results) - already_done
            elapsed = time.time() - t_start
            rate = done_now / elapsed if elapsed > 0 else 0

            # Trace timing stats
            recent = [s for s in trace_stats if s["n_probed"] > 0]
            if recent:
                avg_t = sum(s["elapsed"] for s in recent[-batch_size:]) / min(len(recent), batch_size)
                avg_probes = sum(s["n_probed"] for s in recent[-batch_size:]) / min(len(recent), batch_size)
                print(f"    {len(results)}/{total_probes} "
                      f"(+{done_now} in {elapsed:.0f}s, {rate:.0f} probes/s, "
                      f"avg {avg_t:.1f}s/{avg_probes:.0f}probes per trace)")
            else:
                print(f"    {len(results)}/{total_probes} "
                      f"(+{done_now} in {elapsed:.0f}s, {rate:.0f} probes/s)")

    save_checkpoint(ckpt_path, results)

    # Save NPZ
    _save_npz(output_dir, qid, question_text, ground_truth, traces, results, probe_interval, max_trace)

    del data, traces
    gc.collect()


def _save_npz(output_dir, qid, question_text, ground_truth, traces, results, probe_interval, max_trace):
    """Save final NPZ for one question."""
    n_traces = len(traces)

    all_depths = sorted(set(d for (_, d) in results))
    if not all_depths:
        print(f"  qid={qid}: no results to save")
        return

    depths = np.array(all_depths, dtype=np.int32)
    n_depths = len(depths)
    d2i = {d: i for i, d in enumerate(depths)}

    probe_answers = np.full((n_traces, n_depths), None, dtype=object)
    probe_correct = np.zeros((n_traces, n_depths), dtype=np.bool_)

    for (ti, depth), r in results.items():
        if depth in d2i:
            probe_answers[ti, d2i[depth]] = r["answer"]
            probe_correct[ti, d2i[depth]] = r["is_correct"]

    max_tok = max(len(t.get("confs", [])) for t in traces) if traces else 0
    confs = np.zeros((n_traces, max_tok), dtype=np.float32)
    lengths = np.zeros(n_traces, dtype=np.int32)
    is_correct = np.zeros(n_traces, dtype=np.bool_)
    answers = np.array([t.get("extracted_answer") for t in traces], dtype=object)

    for i, t in enumerate(traces):
        c = t.get("confs", [])
        confs[i, :len(c)] = c
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

    # Print depth summary at 2048 multiples
    for depth in all_depths:
        if depth % probe_interval != 0:
            continue
        di = d2i[depth]
        probed = probe_answers[:, di]
        n_probed = int(np.sum(probed != None))
        n_correct = int(probe_correct[:, di].sum())
        if n_probed > 10:
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Probe offline traces via SGLang servers"
    )
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--num-servers", type=int, default=8)
    parser.add_argument("--probe-interval", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--concurrency-per-server", type=int, default=32,
                        help="Max concurrent requests per server")
    parser.add_argument("--qids", type=int, nargs="*", default=None)
    parser.add_argument("--launch-servers", action="store_true",
                        help="Launch SGLang servers and wait for ready")
    parser.add_argument("--mem-fraction", type=float, default=0.85)
    args = parser.parse_args()

    if args.launch_servers:
        print(f"Launching {args.num_servers} SGLang servers...")
        procs = launch_servers(args.num_servers, args.model_path, args.mem_fraction)
        asyncio.run(wait_servers_ready(args.num_servers))
        print("Servers running. Press Ctrl+C to stop.")
        try:
            for _, p, _ in procs:
                p.wait()
        except KeyboardInterrupt:
            for _, p, lf in procs:
                p.terminate()
                lf.close()
        return

    if not args.input_dir or not args.output_dir:
        parser.error("--input-dir and --output-dir required (or use --launch-servers)")

    os.makedirs(args.output_dir, exist_ok=True)

    questions = scan_questions(
        args.input_dir, args.output_dir,
        set(args.qids) if args.qids else None,
    )
    if not questions:
        print("Nothing to process")
        return

    print(f"Questions to probe: {len(questions)}")

    # Verify servers are up
    asyncio.run(wait_servers_ready(args.num_servers, timeout=10))

    for qid, pkl_path in questions:
        print(f"\nProcessing qid={qid}...")
        asyncio.run(process_question_async(
            qid, pkl_path, args.output_dir, args.num_servers,
            args.probe_interval, args.max_model_len,
            args.concurrency_per_server, args.model_path,
        ))

    done = len(glob.glob(os.path.join(args.output_dir, "q*.npz")))
    print(f"\nDone! {done} NPZ files in {args.output_dir}")


if __name__ == "__main__":
    main()
