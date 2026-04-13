"""
Probe offline traces using producer-worker queue-based pipeline with SGLang.

Architecture:
  - Trace Pool: shared mp.Queue of (qid, trace_idx, token_ids)
  - 8 Producer Groups (4 producers each, 1 leader per group):
      Leader pulls trace from pool → distributes depths to 4 producers →
      producers tokenize in parallel → leader packs and sends to input queue
  - 8 GPU Workers (SGLang Engine offline, one per GPU):
      Pull batches from input queue → engine.generate() with radix cache →
      push results to output queue
  - 8 Aggregators:
      Pull from output queues → save checkpoints + NPZ

Usage:
    python probe_src/probe_batch.py \\
        --input-dir outputs-bedrock-confs/brumo25 \\
        --output-dir probe_results/brumo25
"""

import argparse
import gc
import glob
import os
import pickle
import re
import signal
import sys
import time

import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Queue, Event
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper import extract_answer, equal_func

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = "Qwen/Qwen3-32B"
CONDA_PYTHON = "/opt/dlami/nvme/miniconda3/envs/deepconf/bin/python"
INJECT_SUFFIX = (
    " Considering the limited time by the user, I have to give the"
    " solution based on the thinking directly now.\n</think>\n\n\\boxed{"
)
NUM_GPUS = 8
PRODUCERS_PER_GROUP = 4
NUM_PRODUCER_GROUPS = 8
PROBE_INTERVAL = 2048
MAX_MODEL_LEN = 40960
BATCH_SIZE = 64  # probes per GPU generate() call
SENTINEL = None  # poison pill


# ---------------------------------------------------------------------------
# Scanning & checkpointing
# ---------------------------------------------------------------------------
def _qid_from_filename(f):
    m = re.search(r"qid(\d+)", f)
    return int(m.group(1)) if m else None


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
# Producer Group
# ---------------------------------------------------------------------------
def producer_group(group_id, trace_pool, input_queue, max_trace_tokens,
                   done_event):
    """
    Leader process for a producer group.
    Pulls traces from pool, computes depth list, sends lightweight job
    (token_ids + depths only) to GPU worker's input queue.
    GPU worker constructs full prompts locally to avoid heavy queue serialization.
    """
    while not done_event.is_set():
        try:
            item = trace_pool.get(timeout=2)
        except Exception:
            continue

        if item is SENTINEL:
            trace_pool.put(SENTINEL)  # re-poison for other groups
            break

        qid, trace_idx, token_ids = item
        n_tokens = len(token_ids)

        # Generate depth list
        depths = list(range(PROBE_INTERVAL, n_tokens + 1, PROBE_INTERVAL))
        if n_tokens % PROBE_INTERVAL != 0:
            depths.append(n_tokens)
        depths = [d for d in depths if d <= max_trace_tokens]

        if not depths:
            continue

        # Send lightweight job — GPU worker builds prompts locally
        input_queue.put({
            "qid": qid,
            "trace_idx": trace_idx,
            "token_ids": token_ids,
            "depths": depths,
        })

    print(f"  [Producer {group_id}] done")


# ---------------------------------------------------------------------------
# GPU Worker
# ---------------------------------------------------------------------------
def gpu_worker(worker_id, input_queue, output_queue, model_path,
               mem_fraction, prompt_ids, inject_ids, done_event):
    """
    GPU worker: loads SGLang Engine, pulls jobs from input queue,
    builds prompts locally, runs generate(), pushes results to output queue.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    conda_bin = os.path.dirname(CONDA_PYTHON)
    os.environ["PATH"] = conda_bin + ":" + os.environ.get("PATH", "")

    import sglang as sgl

    print(f"  [GPU {worker_id}] loading engine...")
    engine = sgl.Engine(
        model_path=model_path,
        tp_size=1,
        mem_fraction_static=mem_fraction,
    )
    print(f"  [GPU {worker_id}] engine ready")

    sampling_params = {
        "max_new_tokens": 200,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    }

    while not done_event.is_set():
        try:
            job = input_queue.get(timeout=5)
        except Exception:
            continue

        if job is SENTINEL:
            break

        qid = job["qid"]
        trace_idx = job["trace_idx"]
        token_ids = job["token_ids"]
        depths = job["depths"]

        t0 = time.time()

        # Build prompts locally (avoids heavy queue serialization)
        probes = [(d, prompt_ids + token_ids[:d] + inject_ids) for d in depths]

        # Process in sub-batches
        all_results = []
        for batch_start in range(0, len(probes), BATCH_SIZE):
            batch = probes[batch_start:batch_start + BATCH_SIZE]
            batch_ids = [p[1] for p in batch]

            outputs = engine.generate(
                input_ids=batch_ids,
                sampling_params=sampling_params,
            )

            for (depth, _), out in zip(batch, outputs):
                text = out.get("text", "") if isinstance(out, dict) else getattr(out, "text", "")
                answer = extract_answer("\\boxed{" + text)
                if not answer:
                    answer = text.strip().rstrip("}").strip() or None
                all_results.append((depth, answer, text))

        elapsed = time.time() - t0

        output_queue.put({
            "qid": qid,
            "trace_idx": trace_idx,
            "results": all_results,
            "elapsed": elapsed,
        })

    engine.shutdown()
    print(f"  [GPU {worker_id}] done")


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
def aggregator(agg_id, output_queue, output_dir, ground_truths, done_event,
               stats_queue):
    """
    Aggregator: pulls results from output queue, saves checkpoints.
    """
    ckpt_dir = os.path.join(output_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    # In-memory results cache per qid
    results_cache = {}  # qid -> {(trace_idx, depth): {answer, is_correct, raw_text}}
    trace_times = {}  # qid -> [(trace_idx, elapsed)]

    total_probes = 0
    t_start = time.time()

    while not done_event.is_set():
        try:
            item = output_queue.get(timeout=5)
        except Exception:
            continue

        if item is SENTINEL:
            break

        qid = item["qid"]
        trace_idx = item["trace_idx"]
        elapsed = item["elapsed"]
        gt = ground_truths.get(qid, "")

        if qid not in results_cache:
            # Load existing checkpoint
            ckpt_path = os.path.join(ckpt_dir, f"qid{qid}.pkl")
            results_cache[qid] = load_checkpoint(ckpt_path)
            trace_times[qid] = []

        for depth, answer, raw_text in item["results"]:
            is_correct = False
            if answer and gt:
                try:
                    is_correct = equal_func(answer, gt)
                except Exception:
                    is_correct = str(answer) == str(gt)

            results_cache[qid][(trace_idx, depth)] = {
                "answer": answer,
                "is_correct": is_correct,
                "raw_text": raw_text,
            }
            total_probes += 1

        trace_times[qid].append((trace_idx, elapsed, len(item["results"])))

        # Periodic checkpoint
        if total_probes % 500 == 0:
            for q, res in results_cache.items():
                ckpt_path = os.path.join(ckpt_dir, f"qid{q}.pkl")
                save_checkpoint(ckpt_path, res)

            wall = time.time() - t_start
            rate = total_probes / wall if wall > 0 else 0
            recent = trace_times.get(qid, [])[-10:]
            avg_trace_time = (sum(t for _, t, _ in recent) / len(recent)) if recent else 0
            avg_probes = (sum(n for _, _, n in recent) / len(recent)) if recent else 0

            stats_queue.put({
                "agg_id": agg_id,
                "total_probes": total_probes,
                "wall_time": wall,
                "rate": rate,
                "avg_trace_time": avg_trace_time,
                "avg_probes_per_trace": avg_probes,
                "qid": qid,
                "qid_probes": len(results_cache.get(qid, {})),
            })

    # Final checkpoint
    for q, res in results_cache.items():
        ckpt_path = os.path.join(ckpt_dir, f"qid{q}.pkl")
        save_checkpoint(ckpt_path, res)

    print(f"  [Agg {agg_id}] done, {total_probes} probes saved")


# ---------------------------------------------------------------------------
# NPZ saver
# ---------------------------------------------------------------------------
def save_npz_for_qid(output_dir, qid, question_text, ground_truth,
                     traces, results):
    """Save final NPZ for one question."""
    n_traces = len(traces)
    all_depths = sorted(set(d for (_, d) in results))
    if not all_depths:
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

    # Print depth summary
    for depth in all_depths:
        if depth % PROBE_INTERVAL != 0:
            continue
        di = d2i[depth]
        probed = probe_answers[:, di]
        n_probed = int(np.sum(probed != None))
        n_correct = int(probe_correct[:, di].sum())
        if n_probed > 10:
            ans_list = [a for a in probed if a is not None]
            maj = Counter(ans_list).most_common(1)[0][0] if ans_list else None
            try:
                maj_ok = equal_func(maj, ground_truth) if maj else False
            except Exception:
                maj_ok = str(maj) == str(ground_truth)
            print(f"    @{depth:>6d}: {n_correct:>4d}/{n_probed:<4d} "
                  f"({n_correct/n_probed:5.1%}), maj={'✓' if maj_ok else '✗'}")

    print(f"  -> {npz_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(
        description="Probe offline traces with producer-worker pipeline"
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--probe-interval", type=int, default=PROBE_INTERVAL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--mem-fraction", type=float, default=0.85)
    parser.add_argument("--num-gpus", type=int, default=NUM_GPUS)
    parser.add_argument("--qids", type=int, nargs="*", default=None)
    args = parser.parse_args()

    num_gpus = args.num_gpus

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Tokenizer (main process only, before forking) ----
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    inject_ids = tokenizer.encode(INJECT_SUFFIX, add_special_tokens=False)

    # ---- Find questions to process ----
    pkl_files = sorted(glob.glob(os.path.join(args.input_dir, "*.pkl")))
    questions = []
    for pkl_path in pkl_files:
        qid = _qid_from_filename(os.path.basename(pkl_path))
        if qid is None:
            continue
        if args.qids is not None and qid not in args.qids:
            continue
        npz_path = os.path.join(args.output_dir, f"q{qid:02d}.npz")
        if os.path.exists(npz_path):
            continue
        questions.append((qid, pkl_path))

    if not questions:
        print("Nothing to process")
        return

    print(f"Questions to probe: {len(questions)}")

    # ---- Create queues ----
    trace_pool = Queue(maxsize=1000)
    input_queues = [Queue(maxsize=100) for _ in range(num_gpus)]
    output_queues = [Queue(maxsize=1000) for _ in range(num_gpus)]
    stats_queue = Queue()
    done_event = Event()

    # ---- Process questions one at a time ----
    for qi, (qid, pkl_path) in enumerate(questions):
        print(f"\n{'='*60}")
        print(f"[{qi+1}/{len(questions)}] qid={qid}")
        print(f"{'='*60}")

        # Load data
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        traces = data["all_traces"]
        n_traces = len(traces)
        ground_truth = str(data.get("ground_truth", ""))
        question_text = data.get("question", "")
        ground_truths = {qid: ground_truth}

        # Tokenize question prompt
        messages = [{"role": "user", "content": question_text}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        max_trace_tokens = MAX_MODEL_LEN - len(prompt_ids) - len(inject_ids) - 200

        # Load checkpoint to skip already-done traces
        ckpt_dir = os.path.join(args.output_dir, "ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"qid{qid}.pkl")
        existing = load_checkpoint(ckpt_path)
        done_traces = set()
        for (ti, _) in existing:
            done_traces.add(ti)

        # Count expected probes per trace to know which are fully done
        traces_to_process = []
        total_probes = 0
        for ti, trace in enumerate(traces):
            tids = trace.get("token_ids", [])
            if not tids:
                continue
            n = len(tids)
            n_depths = len([d for d in range(PROBE_INTERVAL, n + 1, PROBE_INTERVAL)
                           if d <= max_trace_tokens])
            if n % PROBE_INTERVAL != 0 and n <= max_trace_tokens:
                n_depths += 1
            # Check if all depths for this trace are done
            all_done = all(
                (ti, d) in existing
                for d in range(PROBE_INTERVAL, n + 1, PROBE_INTERVAL)
                if d <= max_trace_tokens
            )
            if n % PROBE_INTERVAL != 0 and n <= max_trace_tokens:
                all_done = all_done and (ti, n) in existing
            if not all_done:
                traces_to_process.append((ti, tids))
                total_probes += n_depths

        print(f"  {len(traces_to_process)} traces to process "
              f"({len(existing)} probes from checkpoint)")

        if not traces_to_process:
            # All done, just save NPZ
            save_npz_for_qid(args.output_dir, qid, question_text,
                             ground_truth, traces, existing)
            del data, traces
            gc.collect()
            continue

        # ---- Start workers for this question ----
        done_event.clear()

        # Start GPU workers (only on first question, keep running)
        if qi == 0:
            gpu_procs = []
            for i in range(num_gpus):
                p = Process(
                    target=gpu_worker,
                    args=(i, input_queues[i], output_queues[i],
                          args.model_path, args.mem_fraction,
                          prompt_ids, inject_ids, done_event),
                )
                p.start()
                gpu_procs.append(p)
            print(f"  Started {num_gpus} GPU workers")

        # Start producer groups
        prod_procs = []
        for g in range(min(NUM_PRODUCER_GROUPS, num_gpus)):
            p = Process(
                target=producer_group,
                args=(g, trace_pool, input_queues[g],
                      max_trace_tokens, done_event),
            )
            p.start()
            prod_procs.append(p)

        # Start aggregators
        agg_procs = []
        for i in range(num_gpus):
            p = Process(
                target=aggregator,
                args=(i, output_queues[i], args.output_dir,
                      ground_truths, done_event, stats_queue),
            )
            p.start()
            agg_procs.append(p)

        # ---- Fill trace pool ----
        t_start = time.time()
        for ti, tids in traces_to_process:
            trace_pool.put((qid, ti, tids))

        # Poison pills for producers
        for _ in range(NUM_PRODUCER_GROUPS + 1):
            trace_pool.put(SENTINEL)

        # Wait for producers to finish
        for p in prod_procs:
            p.join()

        # Poison pills for GPU workers (signal end of this question)
        for q in input_queues:
            q.put(SENTINEL)

        # Wait for GPU workers to drain
        # (don't join them — they persist across questions)
        # Instead, wait for all output to be consumed
        # Send sentinel to aggregators after GPU workers are done
        for p in gpu_procs:
            p.join()

        for q in output_queues:
            q.put(SENTINEL)

        for p in agg_procs:
            p.join()

        # Collect stats
        while not stats_queue.empty():
            try:
                s = stats_queue.get_nowait()
                rate = s.get("rate", 0)
                avg_t = s.get("avg_trace_time", 0)
                avg_p = s.get("avg_probes_per_trace", 0)
                print(f"    [{s['agg_id']}] {s['total_probes']} probes, "
                      f"{rate:.0f}/s, {avg_t:.1f}s/{avg_p:.0f}p per trace")
            except Exception:
                break

        elapsed = time.time() - t_start
        print(f"  qid={qid}: {elapsed:.0f}s")

        # Merge all aggregator checkpoints and save NPZ
        merged = load_checkpoint(ckpt_path)
        save_npz_for_qid(args.output_dir, qid, question_text,
                         ground_truth, traces, merged)

        del data, traces
        gc.collect()

        # Restart GPU workers for next question
        # (need fresh engines since we sent SENTINEL)
        gpu_procs = []
        if qi < len(questions) - 1:
            for i in range(num_gpus):
                p = Process(
                    target=gpu_worker,
                    args=(i, input_queues[i], output_queues[i],
                          args.model_path, args.mem_fraction,
                          prompt_ids, inject_ids, done_event),
                )
                p.start()
                gpu_procs.append(p)

    done = len(glob.glob(os.path.join(args.output_dir, "q*.npz")))
    print(f"\nDone! {done} NPZ files in {args.output_dir}")


if __name__ == "__main__":
    main()
