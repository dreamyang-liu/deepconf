"""
Probe offline traces: load -> truncate -> tokenize -> infer -> aggregate -> save.

Architecture:
  - N Producer processes: load pkl, tokenize, build trace bundles → prep_queue
  - K GPU Worker processes: pull bundles from shared prep_queue, infer → result_queue
  - Writer thread (main process): pull completed bundles, extract answers, write pkl

Usage:
    python probe_src/probe_batch_v2.py \
        --input-dir conf-data/aime25 \
        --output-dir probe_results/aime25

    # 8 GPUs (default)
    python probe_src/probe_batch_v2.py \
        --input-dir conf-data/aime25 \
        --output-dir probe_results/aime25 \
        --num-gpus 8
"""

import argparse
import glob
import os
import pickle
import re
import sys
import threading
import time

# Force unbuffered stdout so child process logs appear immediately
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from multiprocessing import Process, Queue, Value
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper import extract_answer, equal_func

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = "Qwen/Qwen3-32B"
INJECT_SUFFIX = (
    " Considering the limited time by the user, I have to give the"
    " solution based on the thinking directly now.\n</think>\n\n\\boxed{"
)
PROBE_INTERVAL = 2048
MAX_MODEL_LEN = 32768
BATCH_SIZE = 64
NUM_PRODUCERS = 60
NUM_GPUS = 8
SENTINEL = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _qid_from_filename(f):
    m = re.search(r"qid(\d+)", f)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Producer process: load pkl → tokenize → trace bundles → prep_queue
# ---------------------------------------------------------------------------
def producer(
    producer_id: int,
    tasks: list[tuple[int, str]],
    prep_queue: Queue,
    inject_ids: list[int],
    tokenizer_path: str,
    probe_interval: int,
    max_model_len: int,
    output_trace_dir: str = "",
):
    """
    Each trace bundle pushed to prep_queue:
        {
            "qid": int,
            "trace_idx": int,
            "ground_truth": str,
            "probes": [
                {"depth": int, "input_ids": list[int]},
                ...  # sorted by depth ascending
            ],
        }
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    total_bundles = 0
    total_probes = 0
    total_skipped = 0
    t_start = time.time()

    for task_idx, (qid, pkl_path) in enumerate(tasks):
        print(f"  [Producer {producer_id}] loading qid{qid} ({task_idx+1}/{len(tasks)}): {os.path.basename(pkl_path)}")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        ground_truth = str(data.get("ground_truth", ""))
        question_text = data.get("question", "")

        messages = [{"role": "user", "content": question_text}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        max_trace_tokens = max_model_len - len(prompt_ids) - len(inject_ids) - 50

        traces = data["all_traces"]
        print(f"  [Producer {producer_id}] qid{qid}: {len(traces)} traces, "
              f"prompt_len={len(prompt_ids)}, max_trace_tokens={max_trace_tokens}")

        for trace_idx, trace in enumerate(traces):
            token_ids = trace.get("token_ids", [])
            if not token_ids:
                text = trace.get("text", "")
                if not text:
                    continue
                token_ids = tokenizer.encode(text, add_special_tokens=False)

            n_tokens = len(token_ids)
            depths = list(range(probe_interval, n_tokens + 1, probe_interval))
            if n_tokens % probe_interval != 0:
                depths.append(n_tokens)
            depths = [d for d in depths if d <= max_trace_tokens]
            if not depths:
                print(f"  [Producer {producer_id}] qid{qid} trace{trace_idx}: "
                      f"skipped (n_tokens={n_tokens}, max={max_trace_tokens})")
                continue

            # Skip if output already exists
            if output_trace_dir:
                out_path = os.path.join(output_trace_dir, f"qid{qid}_trace{trace_idx}.pkl")
                if os.path.exists(out_path):
                    total_skipped += 1
                    continue

            confs = trace.get("confs", [])

            probes = []
            for depth in depths:
                full_ids = prompt_ids + token_ids[:depth] + inject_ids
                # avg confidence of the last 2048 tokens before truncation
                if confs:
                    start = max(0, depth - 2048)
                    avg_conf = float(np.mean(confs[start:depth]))
                else:
                    avg_conf = None
                probes.append({"depth": depth, "input_ids": full_ids, "avg_conf": avg_conf})

            prep_queue.put({
                "qid": qid,
                "trace_idx": trace_idx,
                "ground_truth": ground_truth,
                "probes": probes,
            })
            total_bundles += 1
            total_probes += len(probes)

        print(f"  [Producer {producer_id}] qid{qid} done: "
              f"{total_bundles} bundles, {total_probes} probes so far")

    elapsed = time.time() - t_start
    print(f"  [Producer {producer_id}] finished: "
          f"{total_bundles} bundles, {total_probes} probes, "
          f"{total_skipped} skipped in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# GPU Worker process: prep_queue → batch inference → result_queue
# ---------------------------------------------------------------------------
MAX_BUFFERED_BUNDLES = 16


def gpu_worker(
    worker_id: int,
    prep_queue: Queue,
    result_queue: Queue,
    model_path: str,
    mem_fraction: float,
    batch_size: int,
    max_buffered_bundles: int = MAX_BUFFERED_BUNDLES,
):
    """
    One per GPU. Pulls trace bundles from shared prep_queue, processes probes
    in depth-tier waves for optimal prefix cache utilization.

    Wave strategy:
      1. Accumulate N bundles
      2. Organize probes by tier (tier 0 = first depth of each bundle, etc.)
      3. Process tier 0 → tier 1 → ... → tier K sequentially
      4. Each tier batch has uniform seq_len, hits prefix cache from prior tiers

    Exits on SENTINEL, then puts SENTINEL into result_queue.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)

    import sglang as sgl

    print(f"  [GPU {worker_id}] loading engine on cuda:{worker_id}...")
    engine = sgl.Engine(
        model_path=model_path,
        tp_size=1,
        mem_fraction_static=mem_fraction,
    )
    print(f"  [GPU {worker_id}] engine ready")

    sampling_params = {
        "max_new_tokens": 50,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    }

    total_probes = 0
    total_batches = 0
    total_bundles_done = 0
    total_waves = 0
    t_start = time.time()

    # Bundle tracking: bid -> {metadata + results}
    bundles = {}
    bundle_counter = 0
    got_sentinel = False

    def run_batch(items, tier, num_tiers):
        """Run inference on a list of (bid, pidx, depth, input_ids, avg_conf)."""
        nonlocal total_probes, total_batches, total_bundles_done

        batch_ids = [item[3] for item in items]
        seq_lens = [len(ids) for ids in batch_ids]
        total_tokens = sum(seq_lens)
        total_batches += 1

        t_batch = time.time()
        outputs = engine.generate(
            input_ids=batch_ids,
            sampling_params=sampling_params,
        )
        batch_elapsed = time.time() - t_batch

        completed_count = 0
        for (bid, pidx, depth, _ids, avg_conf), out in zip(items, outputs):
            text = out.get("text", "") if isinstance(out, dict) else getattr(out, "text", "")
            bundles[bid]["results"][pidx] = {"depth": depth, "generated_text": text, "avg_conf": avg_conf}
            total_probes += 1

            b = bundles[bid]
            if len(b["results"]) == b["num_probes"]:
                completed = {
                    "qid": b["qid"],
                    "trace_idx": b["trace_idx"],
                    "ground_truth": b["ground_truth"],
                    "probes": [b["results"][i] for i in range(b["num_probes"])],
                }
                result_queue.put(completed)
                del bundles[bid]
                completed_count += 1
                total_bundles_done += 1

        tokens_per_sec = total_tokens / batch_elapsed if batch_elapsed > 0 else 0
        elapsed = time.time() - t_start
        print(f"  [GPU {worker_id}] wave#{total_waves} tier {tier}/{num_tiers-1}: "
              f"batch#{total_batches} {len(items)} items, "
              f"seq_lens={min(seq_lens)}-{max(seq_lens)}, "
              f"total_tokens={total_tokens}, {batch_elapsed:.1f}s "
              f"({tokens_per_sec:.0f} tok/s), "
              f"{completed_count} bundles done | "
              f"cumulative: {total_probes} probes, {total_probes / elapsed:.0f} probes/s")

    def process_wave(wave_bundles):
        """Process a set of bundles tier by tier for prefix cache reuse."""
        nonlocal total_waves
        total_waves += 1

        # Find max number of tiers across all bundles in this wave
        max_tiers = max(b["num_probes"] for b in wave_bundles.values())
        total_probes_in_wave = sum(b["num_probes"] for b in wave_bundles.values())
        tier_counts = []
        for tier in range(max_tiers):
            n = sum(1 for b in wave_bundles.values() if tier < b["num_probes"])
            tier_counts.append(n)

        print(f"  [GPU {worker_id}] wave#{total_waves} start: "
              f"{len(wave_bundles)} bundles, {total_probes_in_wave} probes, "
              f"{max_tiers} tiers, items/tier={tier_counts}")

        t_wave = time.time()
        for tier in range(max_tiers):
            # Collect all probes at this tier
            tier_items = []
            for bid, b in wave_bundles.items():
                if tier < b["num_probes"]:
                    probe = b["probes_data"][tier]
                    tier_items.append((bid, tier, probe["depth"], probe["input_ids"], probe["avg_conf"]))

            if not tier_items:
                continue

            # Process in batches of batch_size
            for start in range(0, len(tier_items), batch_size):
                batch = tier_items[start:start + batch_size]
                run_batch(batch, tier, max_tiers)

        wave_elapsed = time.time() - t_wave
        wave_probes_per_sec = total_probes_in_wave / wave_elapsed if wave_elapsed > 0 else 0
        elapsed = time.time() - t_start
        print(f"  [GPU {worker_id}] wave#{total_waves} done: "
              f"{len(wave_bundles)} bundles, {total_probes_in_wave} probes in {wave_elapsed:.1f}s "
              f"({wave_probes_per_sec:.0f} probes/s this wave) | "
              f"cumulative: {total_bundles_done} bundles, "
              f"{total_probes} probes, {total_probes / elapsed:.0f} probes/s")

    print(f"  [GPU {worker_id}] waiting for bundles...")

    while not got_sentinel:
        # --- Accumulate bundles for next wave ---
        wave_bundles = {}
        fill_start = time.time()

        while len(wave_bundles) < max_buffered_bundles:
            remaining = 1.0 - (time.time() - fill_start)
            if remaining <= 0:
                break
            try:
                bundle = prep_queue.get(timeout=min(remaining, 0.1))
            except Exception:
                if wave_bundles:
                    break
                continue

            if bundle is SENTINEL:
                got_sentinel = True
                print(f"  [GPU {worker_id}] received sentinel, "
                      f"{len(wave_bundles)} bundles buffered for final wave")
                break

            bid = bundle_counter
            bundle_counter += 1
            bundles[bid] = {
                "qid": bundle["qid"],
                "trace_idx": bundle["trace_idx"],
                "ground_truth": bundle["ground_truth"],
                "num_probes": len(bundle["probes"]),
                "results": {},
            }
            wave_bundles[bid] = {
                "num_probes": len(bundle["probes"]),
                "probes_data": bundle["probes"],
            }

        # --- Process this wave tier by tier ---
        if wave_bundles:
            process_wave(wave_bundles)

    engine.shutdown()
    result_queue.put(SENTINEL)
    elapsed = time.time() - t_start
    print(f"  [GPU {worker_id}] finished: {total_bundles_done} bundles, "
          f"{total_probes} probes, {total_batches} batches, {total_waves} waves "
          f"in {elapsed:.1f}s ({total_probes / elapsed:.0f} probes/s)")


# ---------------------------------------------------------------------------
# Writer thread: result_queue → extract answer → write pkl
# ---------------------------------------------------------------------------
def writer_fn(result_queue, output_dir, num_gpus):
    """
    Runs in main process as a thread. Pulls completed trace bundles from
    result_queue, extracts answers, checks correctness, writes per-trace pkl.

    Exits after receiving num_gpus SENTINELs (one from each GPU worker).
    """
    trace_dir = os.path.join(output_dir, "traces")
    os.makedirs(trace_dir, exist_ok=True)

    total_probes = 0
    total_traces = 0
    total_correct = 0
    gpus_done = 0
    t_start = time.time()

    while True:
        item = result_queue.get()
        if item is SENTINEL:
            gpus_done += 1
            print(f"  [Writer] GPU sentinel received ({gpus_done}/{num_gpus})")
            if gpus_done >= num_gpus:
                break
            continue

        qid = item["qid"]
        trace_idx = item["trace_idx"]
        gt = item["ground_truth"]
        probe_results = {}
        n_correct_this = 0

        for probe in item["probes"]:
            depth = probe["depth"]
            raw_text = probe["generated_text"]
            avg_conf = probe["avg_conf"]

            answer = extract_answer("\\boxed{" + raw_text)
            if not answer:
                answer = raw_text.strip().rstrip("}").strip() or None

            is_correct = False
            if answer and gt:
                try:
                    is_correct = equal_func(answer, gt)
                except Exception:
                    is_correct = str(answer) == str(gt)

            probe_results[depth] = {
                "answer": answer,
                "is_correct": is_correct,
                "raw_text": raw_text,
                "avg_conf": avg_conf,
            }
            total_probes += 1
            if is_correct:
                n_correct_this += 1
                total_correct += 1

        payload = {
            "qid": qid,
            "trace_idx": trace_idx,
            "ground_truth": gt,
            "probes": probe_results,
        }
        path = os.path.join(trace_dir, f"qid{qid}_trace{trace_idx}.pkl")
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f)
        os.replace(tmp, path)
        total_traces += 1

        if total_traces % 50 == 0:
            elapsed = time.time() - t_start
            acc = total_correct / total_probes * 100 if total_probes else 0
            print(f"  [Writer] {total_traces} traces, {total_probes} probes, "
                  f"acc={acc:.1f}%, {total_probes / elapsed:.0f} probes/s")

    elapsed = time.time() - t_start
    acc = total_correct / total_probes * 100 if total_probes else 0
    print(f"  [Writer] done: {total_traces} traces, {total_probes} probes, "
          f"acc={acc:.1f}%, {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(args):
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # Scan input files
    pkl_files = sorted(glob.glob(os.path.join(args.input_dir, "*.pkl")))
    questions = []
    for pkl_path in pkl_files:
        qid = _qid_from_filename(os.path.basename(pkl_path))
        if qid is None:
            continue
        if args.qids is not None and qid not in args.qids:
            continue
        questions.append((qid, pkl_path))

    if not questions:
        print("Nothing to process")
        return
    print(f"Questions to probe: {len(questions)}")

    # Tokenize inject suffix
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    inject_ids = tokenizer.encode(INJECT_SUFFIX, add_special_tokens=False)

    # Queues
    num_gpus = args.num_gpus
    prep_queue = Queue(maxsize=200 * num_gpus)
    result_queue = Queue(maxsize=200 * num_gpus)

    # --- Start GPU workers (staggered to avoid OOM) ---
    gpu_procs = []
    for gid in range(num_gpus):
        p = Process(
            target=gpu_worker,
            args=(gid, prep_queue, result_queue,
                  args.model_path, args.mem_fraction, args.batch_size),
        )
        p.start()
        gpu_procs.append(p)
    print(f"Started {num_gpus} GPU workers")

    # --- Start writer thread ---
    wt = threading.Thread(
        target=writer_fn,
        args=(result_queue, args.output_dir, num_gpus),
    )
    wt.start()

    # --- Start producer processes ---
    num_producers = min(args.num_producers, len(questions))
    chunks = [[] for _ in range(num_producers)]
    for i, q in enumerate(questions):
        chunks[i % num_producers].append(q)

    prod_procs = []
    for pid in range(num_producers):
        if not chunks[pid]:
            continue
        p = Process(
            target=producer,
            args=(pid, chunks[pid], prep_queue, inject_ids,
                  args.model_path, args.probe_interval, args.max_model_len,
                  os.path.join(args.output_dir, "traces")),
        )
        p.start()
        prod_procs.append(p)
    print(f"Started {len(prod_procs)} producers")

    # --- Wait for producers, then signal GPU workers to stop ---
    for p in prod_procs:
        p.join()
    print("All producers finished, sending stop signals to GPU workers...")
    for _ in range(num_gpus):
        prep_queue.put(SENTINEL)

    # --- Wait for GPU workers ---
    for p in gpu_procs:
        p.join()
    print("All GPU workers finished")

    # --- Wait for writer ---
    wt.join()
    print("Done")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Probe offline traces (producers → GPU workers → writer)"
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--probe-interval", type=int, default=PROBE_INTERVAL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--mem-fraction", type=float, default=0.85)
    parser.add_argument("--num-gpus", type=int, default=NUM_GPUS)
    parser.add_argument("--num-producers", type=int, default=NUM_PRODUCERS)
    parser.add_argument("--qids", type=int, nargs="*", default=None)
    args = parser.parse_args()

    run_pipeline(args)


if __name__ == "__main__":
    main()
