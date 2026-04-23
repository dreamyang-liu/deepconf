"""
Data-parallel deepconf offline generator.

Runs NUM_GPUS vLLM replicas (TP=1) in parallel, each pulling
(qid, batch_idx) work items from a shared queue so any idle GPU can pick
up the next batch — no GPU stalls waiting on the slowest question.

Each batch produces a partial pkl (256 traces). After all batches for a qid
complete, they are merged into a single bedrock-format pkl matching
`deepconf_simple_qid{qid}_rid{rid}_{timestamp}.pkl`.

Usage:
  python deepconf-offline-dp.py --rid deepseek
  python deepconf-offline-dp.py --rid deepseek --qids 0 1 2  # subset
"""
import os
import json
import time
import pickle
import argparse
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

# Configuration
MODEL_PATH = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
MAX_TOKENS = 64000
DATASET_FILE = "brumo_2025.jsonl"
TOTAL_BUDGET = 4096
BATCH_SIZE = 256
WINDOW_SIZE = 2048
NUM_GPUS = 8


def worker(gpu_id, work_queue, done_queue, prompts_map, gts_map,
           output_dir, model_path):
    """Persistent worker: loads vLLM on one GPU, processes batches forever."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from vllm import LLM, SamplingParams
    from helper import process_output_offline

    print(f"[GPU {gpu_id}] loading model {model_path}...")
    t0 = time.time()
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
    )
    print(f"[GPU {gpu_id}] model loaded in {time.time()-t0:.1f}s")

    while True:
        item = work_queue.get()
        if item is None:
            print(f"[GPU {gpu_id}] received sentinel, shutting down")
            return
        qid, batch_idx = item
        prompt = prompts_map[qid]
        gt = gts_map[qid]

        t0 = time.time()
        params = SamplingParams(
            n=BATCH_SIZE,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens=MAX_TOKENS,
            logprobs=20,
        )
        outputs = llm.generate([prompt], params)

        traces = []
        for out in outputs[0].outputs:
            traces.append(process_output_offline(out, gt, WINDOW_SIZE))

        batch_path = Path(output_dir) / f"qid{qid}_batch{batch_idx:02d}.pkl"
        with open(batch_path, "wb") as f:
            pickle.dump(traces, f)

        done_queue.put({
            "qid": qid,
            "batch_idx": batch_idx,
            "gpu_id": gpu_id,
            "elapsed": time.time() - t0,
            "n_traces": len(traces),
            "total_tokens": sum(t["num_tokens"] for t in traces),
        })


def merge_qid(qid, args, gts_map, data, final_dir, batch_dir, timestamp):
    """Merge all batches for a qid into one bedrock-format pkl."""
    all_traces = []
    for batch_idx in range(TOTAL_BUDGET // BATCH_SIZE):
        bpath = batch_dir / f"qid{qid}_batch{batch_idx:02d}.pkl"
        if not bpath.exists():
            print(f"  [warn] qid {qid}: missing batch {batch_idx}")
            continue
        with open(bpath, "rb") as f:
            all_traces.extend(pickle.load(f))

    correct = sum(1 for t in all_traces if t.get("is_correct"))
    total_tokens = sum(t["num_tokens"] for t in all_traces)
    result = {
        "question_id": qid,
        "run_id": args.rid,
        "question": data[qid]["question"],
        "ground_truth": gts_map[qid],
        "all_traces": all_traces,
        "accuracy": correct / len(all_traces) if all_traces else 0,
        "correct_traces_count": correct,
        "token_stats": {
            "total_tokens": total_tokens,
            "total_traces_count": len(all_traces),
            "avg_tokens_per_trace": total_tokens / len(all_traces) if all_traces else 0,
        },
        "config": {
            "model_path": args.model,
            "total_budget": TOTAL_BUDGET,
            "window_size": WINDOW_SIZE,
            "batch_size": BATCH_SIZE,
            "source": "dp_worker",
        },
        "timestamp": datetime.now().isoformat(),
    }
    out_path = final_dir / f"deepconf_simple_qid{qid}_rid{args.rid}_{timestamp}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  qid {qid:>2}: {len(all_traces):>5} traces, "
          f"{correct}/{len(all_traces)} correct → {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rid", type=str, default="deepseek")
    parser.add_argument("--qids", type=int, nargs="*", default=list(range(30)))
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--dataset", type=str, default=DATASET_FILE)
    parser.add_argument("--output-dir", type=str, default="outputs_brumo25_deepseek")
    parser.add_argument("--num-gpus", type=int, default=NUM_GPUS)
    parser.add_argument("--dry-run-batches", type=int, default=None,
                        help="Only process this many work items (for testing)")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    with open(args.dataset) as f:
        data = [json.loads(l) for l in f]

    from transformers import AutoTokenizer
    from helper import prepare_prompt
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts_map, gts_map = {}, {}
    for qid in args.qids:
        p, gt = prepare_prompt(data[qid], tokenizer)
        prompts_map[qid], gts_map[qid] = p, gt

    final_dir = Path(args.output_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = final_dir / f"batches_{args.rid}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Build work queue, skipping batches already on disk (resume support)
    work_items = []
    for qid in args.qids:
        for batch_idx in range(TOTAL_BUDGET // BATCH_SIZE):
            bpath = batch_dir / f"qid{qid}_batch{batch_idx:02d}.pkl"
            if not bpath.exists():
                work_items.append((qid, batch_idx))

    if args.dry_run_batches is not None:
        work_items = work_items[: args.dry_run_batches]

    total = len(work_items)
    print(f"Qids: {args.qids}")
    print(f"Work items: {total} (batch_size={BATCH_SIZE}, "
          f"total_budget={TOTAL_BUDGET}, num_gpus={args.num_gpus})")

    if total > 0:
        # Interleave batches so all qids make progress in parallel early on
        work_items.sort(key=lambda x: (x[1], x[0]))

        work_queue = mp.Queue()
        done_queue = mp.Queue()
        for item in work_items:
            work_queue.put(item)
        for _ in range(args.num_gpus):
            work_queue.put(None)

        procs = []
        for gpu_id in range(args.num_gpus):
            p = mp.Process(
                target=worker,
                args=(gpu_id, work_queue, done_queue, prompts_map, gts_map,
                      str(batch_dir), args.model),
            )
            p.start()
            procs.append(p)

        t_start = time.time()
        completed = 0
        total_tokens = 0
        while completed < total:
            info = done_queue.get()
            completed += 1
            total_tokens += info["total_tokens"]
            elapsed_total = time.time() - t_start
            eta_s = elapsed_total / completed * (total - completed)
            throughput = total_tokens / elapsed_total if elapsed_total > 0 else 0
            print(f"[{completed:>3}/{total}] GPU {info['gpu_id']} qid{info['qid']:>2} "
                  f"batch{info['batch_idx']:02d} in {info['elapsed']:>5.1f}s "
                  f"({info['total_tokens']:>7} tok) | "
                  f"agg {throughput:>6.0f} tok/s | ETA {eta_s/60:>5.1f} min")

        for p in procs:
            p.join()
        print(f"\nAll batches done in {(time.time()-t_start)/60:.1f} min.")
    else:
        print("No work items — all batches already on disk.")

    # Merge per-qid into bedrock-format pkls
    print("\nMerging batches into per-qid pkls...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for qid in args.qids:
        merge_qid(qid, args, gts_map, data, final_dir, batch_dir, timestamp)

    print("\nDone.")


if __name__ == "__main__":
    main()
