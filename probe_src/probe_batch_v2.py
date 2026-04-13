"""
Probe offline traces: load → truncate → tokenize → infer → aggregate → save.

Architecture:
  - 64 Producer processes: load pkl, truncate traces at each probe depth,
    tokenize, round-robin into per-GPU input queues
  - 8 GPU Workers (SGLang Engine, one per GPU):
    pull from own input queue → engine.generate() → push to shared output queue
  - 1 Aggregator: pull from output queue → extract answer, check correctness,
    async file writes (one file per trace)

Usage:
    python probe_src/probe_batch_v2.py \
        --input-dir outputs-bedrock-confs/brumo25 \
        --output-dir probe_results/brumo25
"""

import argparse
import glob
import os
import pickle
import re
import sys
import time

import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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
NUM_PRODUCERS = 64
PROBE_INTERVAL = 2048
MAX_MODEL_LEN = 32768
BATCH_SIZE = 64
SENTINEL = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _qid_from_filename(f):
    m = re.search(r"qid(\d+)", f)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Stage 1: Producers — load, truncate, tokenize, enqueue
# ---------------------------------------------------------------------------
def producer(
    producer_id: int,
    tasks: list[tuple[int, str]],       # [(qid, pkl_path), ...]
    input_queues: list[Queue],           # one per GPU, round-robin
    inject_ids: list[int],
    tokenizer_path: str,
    done_event: Event,
):
    """
    Load pkl files assigned to this producer, extract traces, truncate to
    each probe depth, prepend prompt_ids + append inject_ids, and push
    fully-formed token-id sequences into the appropriate GPU input queue.

    Each item pushed to input_queue:
        {
            "qid": int,
            "trace_idx": int,
            "depth": int,
            "input_ids": list[int],   # prompt + trace[:depth] + inject
            "ground_truth": str,
        }
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    num_gpus = len(input_queues)
    rr_counter = producer_id  # round-robin starting offset

    for qid, pkl_path in tasks:
        if done_event.is_set():
            break

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        ground_truth = str(data.get("ground_truth", ""))
        question_text = data.get("question", "")

        messages = [{"role": "user", "content": question_text}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        max_trace_tokens = MAX_MODEL_LEN - len(prompt_ids) - len(inject_ids) - 200

        traces = data["all_traces"]
        for trace_idx, trace in enumerate(traces):
            if done_event.is_set():
                break

            token_ids = trace.get("token_ids", [])
            if not token_ids:
                text = trace.get("text", "")
                if not text:
                    continue
                token_ids = tokenizer.encode(text, add_special_tokens=False)

            n_tokens = len(token_ids)

            depths = list(range(PROBE_INTERVAL, n_tokens + 1, PROBE_INTERVAL))
            if n_tokens % PROBE_INTERVAL != 0:
                depths.append(n_tokens)
            depths = [d for d in depths if d <= max_trace_tokens]

            for depth in depths:
                full_ids = prompt_ids + token_ids[:depth] + inject_ids
                target_queue = input_queues[rr_counter % num_gpus]
                rr_counter += 1

                target_queue.put({
                    "qid": qid,
                    "trace_idx": trace_idx,
                    "depth": depth,
                    "input_ids": full_ids,
                    "ground_truth": ground_truth,
                })

    print(f"  [Producer {producer_id}] done")


# ---------------------------------------------------------------------------
# Stage 2: GPU Workers — inference
# ---------------------------------------------------------------------------
def gpu_worker(
    worker_id: int,
    input_queue: Queue,
    output_queue: Queue,
    model_path: str,
    mem_fraction: float,
    done_event: Event,
):
    """
    One per GPU. Pulls items from input_queue, batches them up to BATCH_SIZE,
    calls engine.generate(), pushes results to output_queue.

    Each item pushed to output_queue:
        {
            "qid": int,
            "trace_idx": int,
            "depth": int,
            "generated_text": str,
        }
    """
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

    def drain_batch(max_items: int, timeout: float) -> list[dict]:
        """Collect up to max_items from input_queue, blocking on the first."""
        batch = []
        try:
            first = input_queue.get(timeout=timeout)
            if first is SENTINEL:
                return SENTINEL
            batch.append(first)
        except Exception:
            return batch

        while len(batch) < max_items:
            try:
                item = input_queue.get_nowait()
                if item is SENTINEL:
                    input_queue.put(SENTINEL)
                    break
                batch.append(item)
            except Exception:
                break
        return batch

    while not done_event.is_set():
        batch = drain_batch(BATCH_SIZE, timeout=5)
        if batch is SENTINEL:
            break
        if not batch:
            continue

        batch_ids = [item["input_ids"] for item in batch]

        outputs = engine.generate(
            input_ids=batch_ids,
            sampling_params=sampling_params,
        )

        for item, out in zip(batch, outputs):
            text = out.get("text", "") if isinstance(out, dict) else getattr(out, "text", "")
            output_queue.put({
                "qid": item["qid"],
                "trace_idx": item["trace_idx"],
                "depth": item["depth"],
                "generated_text": text,
                "ground_truth": item["ground_truth"],
            })

    engine.shutdown()
    print(f"  [GPU {worker_id}] done")


# ---------------------------------------------------------------------------
# Stage 3: Aggregator — post-process + async file writes
# ---------------------------------------------------------------------------
def aggregator(
    output_queue: Queue,
    output_dir: str,
    done_event: Event,
):
    """
    Single aggregator process. Pulls results from output_queue, extracts
    answers, checks correctness, accumulates per-(qid, trace_idx), and
    writes per-trace result files asynchronously via a thread pool.

    Output file per trace: {output_dir}/traces/qid{qid}_trace{trace_idx}.pkl
        {
            "qid": int,
            "trace_idx": int,
            "ground_truth": str,
            "probes": {depth: {"answer": str, "is_correct": bool, "raw_text": str}},
        }
    """
    trace_dir = os.path.join(output_dir, "traces")
    os.makedirs(trace_dir, exist_ok=True)

    # In-memory accumulator: (qid, trace_idx) -> {depth -> result}
    accumulator: dict[tuple[int, int], dict[int, dict]] = {}
    # Track expected probe count per (qid, trace_idx) so we know when it's complete
    expected_counts: dict[tuple[int, int], int] = {}

    writer = ThreadPoolExecutor(max_workers=4)
    pending_futures = []
    total_probes = 0
    t_start = time.time()

    def _write_trace_file(path: str, payload: dict):
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f)
        os.replace(tmp, path)

    while not done_event.is_set():
        try:
            item = output_queue.get(timeout=5)
        except Exception:
            continue

        if item is SENTINEL:
            break

        qid = item["qid"]
        trace_idx = item["trace_idx"]
        depth = item["depth"]
        raw_text = item["generated_text"]
        gt = item["ground_truth"]

        # Extract answer
        answer = extract_answer("\\boxed{" + raw_text)
        if not answer:
            answer = raw_text.strip().rstrip("}").strip() or None

        is_correct = False
        if answer and gt:
            try:
                is_correct = equal_func(answer, gt)
            except Exception:
                is_correct = str(answer) == str(gt)

        key = (qid, trace_idx)
        if key not in accumulator:
            accumulator[key] = {}
        accumulator[key][depth] = {
            "answer": answer,
            "is_correct": is_correct,
            "raw_text": raw_text,
        }
        total_probes += 1

        # Check if this trace is complete → async write
        if key in expected_counts and len(accumulator[key]) >= expected_counts[key]:
            payload = {
                "qid": qid,
                "trace_idx": trace_idx,
                "ground_truth": gt,
                "probes": accumulator.pop(key),
            }
            path = os.path.join(trace_dir, f"qid{qid}_trace{trace_idx}.pkl")
            fut = writer.submit(_write_trace_file, path, payload)
            pending_futures.append(fut)

        if total_probes % 1000 == 0:
            wall = time.time() - t_start
            rate = total_probes / wall if wall > 0 else 0
            print(f"  [Aggregator] {total_probes} probes, {rate:.0f}/s")

    # Flush remaining incomplete traces
    for (qid, trace_idx), probes in accumulator.items():
        gt = ground_truths.get(qid, "")
        payload = {
            "qid": qid,
            "trace_idx": trace_idx,
            "ground_truth": gt,
            "probes": probes,
        }
        path = os.path.join(trace_dir, f"qid{qid}_trace{trace_idx}.pkl")
        fut = writer.submit(_write_trace_file, path, payload)
        pending_futures.append(fut)

    # Wait for all writes to finish
    for fut in pending_futures:
        fut.result()

    writer.shutdown(wait=True)
    print(f"  [Aggregator] done, {total_probes} probes written")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def start_gpu_workers(
    num_gpus: int,
    input_queues: list[Queue],
    output_queue: Queue,
    model_path: str,
    mem_fraction: float,
    done_event: Event,
) -> list[Process]:
    """Start one GPU worker per GPU with staggered launches."""
    orig_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_procs = []
    for i in range(num_gpus):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
        p = Process(
            target=gpu_worker,
            args=(i, input_queues[i], output_queue,
                  model_path, mem_fraction, done_event),
        )
        p.start()
        gpu_procs.append(p)
        if i < num_gpus - 1:
            time.sleep(30)
    if orig_cvd is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = orig_cvd
    print(f"Started {num_gpus} GPU workers")
    return gpu_procs


def load_and_enqueue_traces(
    input_dir: str,
    qids: list[int] | None,
    input_queues: list[Queue],
    inject_ids: list[int],
    model_path: str,
    done_event: Event,
) -> list[Process]:
    """Scan input dir, partition files across producers, enqueue traces."""
    pkl_files = sorted(glob.glob(os.path.join(input_dir, "*.pkl")))
    questions = []
    for pkl_path in pkl_files:
        qid = _qid_from_filename(os.path.basename(pkl_path))
        if qid is None:
            continue
        if qids is not None and qid not in qids:
            continue
        questions.append((qid, pkl_path))

    if not questions:
        print("Nothing to process")
        return []

    print(f"Questions to probe: {len(questions)}")

    num_producers = min(NUM_PRODUCERS, len(questions))
    chunks = [[] for _ in range(num_producers)]
    for i, q in enumerate(questions):
        chunks[i % num_producers].append(q)

    prod_procs = []
    for pid in range(num_producers):
        if not chunks[pid]:
            continue
        p = Process(
            target=producer,
            args=(pid, chunks[pid], input_queues,
                  inject_ids, model_path, done_event),
        )
        p.start()
        prod_procs.append(p)
    print(f"Started {len(prod_procs)} producers")
    return prod_procs


def run_pipeline(args):
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    inject_ids = tokenizer.encode(INJECT_SUFFIX, add_special_tokens=False)

    num_gpus = args.num_gpus
    input_queues = [Queue(maxsize=200) for _ in range(num_gpus)]
    output_queue = Queue(maxsize=5000)
    done_event = Event()

    gpu_procs = start_gpu_workers(
        num_gpus, input_queues, output_queue,
        args.model_path, args.mem_fraction, done_event,
    )

    agg_proc = Process(
        target=aggregator,
        args=(output_queue, args.output_dir, done_event),
    )
    agg_proc.start()

    prod_procs = load_and_enqueue_traces(
        args.input_dir, args.qids, input_queues,
        inject_ids, args.model_path, done_event,
    )

    for p in prod_procs:
        p.join()
    print("All producers finished")

    for q in input_queues:
        q.put(SENTINEL)
    for p in gpu_procs:
        p.join()
    print("All GPU workers finished")

    output_queue.put(SENTINEL)
    agg_proc.join()
    print("Aggregator finished")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    global PROBE_INTERVAL, BATCH_SIZE, MAX_MODEL_LEN

    parser = argparse.ArgumentParser(
        description="Probe offline traces (v2: load → infer → aggregate)"
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

    PROBE_INTERVAL = args.probe_interval
    BATCH_SIZE = args.batch_size
    MAX_MODEL_LEN = args.max_model_len

    run_pipeline(args)


if __name__ == "__main__":
    main()
