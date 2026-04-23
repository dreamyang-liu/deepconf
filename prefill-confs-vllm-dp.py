"""
Data-parallel prefill confs with producer/gpu/postproc architecture.

Architecture:
  Main:       load dataset, scan ckpts, feed work_queue (batches of 16 traces)
  Producers:  N fork procs; pull from work_queue, batched tokenize, push to input_queue
  GPU:        M spawn procs (TP=tp each); pull from input_queue, buffer to chunk_size
              (or 0.5s timeout), llm.generate, push raw outputs to output_queue
  Post:       K fork procs; pull from output_queue, compute_confs, save ckpt

Ckpt/output format matches run_prefill_all.py's asm() so final assembly is identical.

Usage:
  python prefill-confs-vllm-dp.py \
    --input-dir conf-data-coder-next/brumo25 \
    --dataset-file brumo_2025.jsonl \
    --output-dir conf-data-coder-next-confs/brumo25 \
    --model-path Qwen/Qwen3-Next-80B-A3B-Thinking-FP8 \
    --tp 2
"""
import argparse
import glob
import json
import os
import pickle
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from operator import attrgetter
from pathlib import Path
from queue import Empty
import multiprocessing as mp

import numpy as np

SD = Path(__file__).parent.resolve()
sys.path.insert(0, str(SD / "probe_src"))

TOP_LOGPROBS = 20
_GET_LOGPROB = attrgetter("logprob")


# --------------------------------------------------------------------------
# Shared compute
# --------------------------------------------------------------------------

def compute_confs_from_outputs(outputs_and_pls):
    """Batched -mean(top-k logprob) for variable prompt_lens.

    Expects vLLM's FlatLogprobs (SamplingParams.flat_logprobs=True) — flat
    primitive lists rather than list[dict[int, Logprob]]; 5-10x faster
    because we skip per-Logprob-object attr access.

    outputs_and_pls: list of (output, prompt_len)
    Returns: list of confs-list, one per input (may be []).
    """
    means = []
    boundaries = [0]
    for output, pl in outputs_and_pls:
        plp = output.prompt_logprobs
        if plp is not None:
            logprobs = plp.logprobs
            start = plp.start_indices
            end = plp.end_indices
            for i in range(pl, len(start)):
                s, e = start[i], end[i]
                if e > s:
                    means.append(-sum(logprobs[s:e]) / (e - s))
        boundaries.append(len(means))
    if not means:
        return [[] for _ in outputs_and_pls]
    neg_means = np.round(np.asarray(means, dtype=np.float32), 3)
    return [neg_means[boundaries[i]:boundaries[i + 1]].tolist()
            for i in range(len(outputs_and_pls))]


# --------------------------------------------------------------------------
# Producer (fork): tokenize a batch of 16 traces
# --------------------------------------------------------------------------

def producer_worker(pid, work_queue, input_queue, prompts_map,
                    prompt_lens_map, tokenizer, max_model_len):
    n_done = 0
    while True:
        batch = work_queue.get()
        if batch is None:
            return
        # batch: list of (qid, trace_idx, gen_text)
        full_texts = [prompts_map[qid] + txt for qid, _, txt in batch]
        all_ids = tokenizer(full_texts, add_special_tokens=False)["input_ids"]
        out = []
        for (qid, ti, _), full_ids in zip(batch, all_ids):
            pl = prompt_lens_map[qid]
            gen_len = len(full_ids) - pl
            if len(full_ids) > max_model_len - 2:
                out.append((qid, ti, None, pl, gen_len))  # over max_model_len
            else:
                out.append((qid, ti, full_ids, pl, gen_len))
        input_queue.put(out)
        n_done += len(batch)


# --------------------------------------------------------------------------
# GPU worker (spawn): buffer to chunk_size, llm.generate
# --------------------------------------------------------------------------

def gpu_worker(wid, gpu_str, input_queue, output_queue, model_path, tp,
               max_model_len, gpu_memory_utilization, chunk_size, timeout_s,
               max_num_batched_tokens):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    from vllm import LLM, SamplingParams

    print(f"[GPU {gpu_str}] loading model...", flush=True)
    t0 = time.time()
    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tp,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=True,
        disable_log_stats=True,
    )
    if max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    llm = LLM(**llm_kwargs)
    print(f"[GPU {gpu_str}] model loaded in {time.time()-t0:.0f}s", flush=True)

    sampling = SamplingParams(
        max_tokens=1, temperature=0.0, prompt_logprobs=TOP_LOGPROBS,
        detokenize=False, flat_logprobs=True,
    )

    # Background post thread: computes confs + pushes small data to output_queue.
    # Main thread submits and immediately continues to next llm.generate(),
    # overlapping compute_confs with the next GPU forward pass.
    post_exec = ThreadPoolExecutor(max_workers=2)

    def _bg_compute_and_push(valid_items, outputs, oor_items, chunk_idx, dt):
        """Runs in background thread; pure Python iteration releases some GIL
        time when main thread is inside llm.generate (CUDA calls)."""
        if oor_items:
            output_queue.put(("oor", oor_items))
        if valid_items:
            confs_list = compute_confs_from_outputs(
                [(out, pl) for (_, _, _, pl, _), out in zip(valid_items, outputs)]
            )
            items = [(q, t, confs)
                     for (q, t, _, _, _), confs in zip(valid_items, confs_list)]
            output_queue.put(("confs", items))
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}][GPU {gpu_str}] chunk #{chunk_idx} bg-pushed "
              f"{len(valid_items)} traces (gen {dt:.1f}s)", flush=True)

    buffer = []  # list of (qid, ti, input_ids|None, prompt_len, gen_len)
    buffer_start = None
    chunks_done = 0
    total_traces = 0

    def do_flush():
        nonlocal buffer, buffer_start, chunks_done, total_traces
        if not buffer:
            buffer_start = None
            return
        oor = [(q, t, pl, gl) for q, t, ids, pl, gl in buffer if ids is None]
        valid = [(q, t, ids, pl, gl) for q, t, ids, pl, gl in buffer if ids is not None]
        outs = None
        dt = 0.0
        if valid:
            prompts = [{"prompt_token_ids": ids} for _, _, ids, _, _ in valid]
            t_gen = time.time()
            try:
                outs = llm.generate(prompts, sampling, use_tqdm=False)
            except Exception as e:
                print(f"[GPU {gpu_str}] generate error ({len(valid)} traces): {e}",
                      flush=True)
                output_queue.put(("err", [(q, t, pl, gl) for q, t, _, pl, gl in valid]))
                buffer = []
                buffer_start = None
                return
            dt = time.time() - t_gen
            chunks_done += 1
            total_traces += len(valid)
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}][GPU {gpu_str}] chunk #{chunks_done} "
                  f"{len(valid)} traces ({len(oor)} oor) gen {dt:.1f}s "
                  f"(total {total_traces})", flush=True)
        # Hand off heavy compute+push to background thread; main continues.
        post_exec.submit(_bg_compute_and_push, valid, outs, oor, chunks_done, dt)
        buffer = []
        buffer_start = None

    try:
        while True:
            if buffer_start is not None:
                remaining = timeout_s - (time.time() - buffer_start)
                timeout = max(0.001, remaining)
            else:
                timeout = None  # block forever if buffer empty

            try:
                item = input_queue.get(timeout=timeout)
            except Empty:
                do_flush()
                continue

            if item is None:
                do_flush()
                print(f"[GPU {gpu_str}] sentinel, draining bg post...", flush=True)
                post_exec.shutdown(wait=True)
                print(f"[GPU {gpu_str}] exit chunks={chunks_done} "
                      f"traces={total_traces}", flush=True)
                return

            if not buffer:
                buffer_start = time.time()
            buffer.extend(item)

            while len(buffer) >= chunk_size:
                to_flush = buffer[:chunk_size]
                rest = buffer[chunk_size:]
                buffer = to_flush
                do_flush()
                buffer = rest
                buffer_start = time.time() if buffer else None
    finally:
        post_exec.shutdown(wait=True)


# --------------------------------------------------------------------------
# Post worker (fork): compute_confs + save ckpt
# --------------------------------------------------------------------------

def post_worker(pid, output_queue, ckpt_root):
    saved = 0
    t0 = time.time()
    ckpt_root = Path(ckpt_root)
    while True:
        item = output_queue.get()
        if item is None:
            dt = time.time() - t0
            print(f"[post {pid}] exiting, saved {saved} ckpts in {dt:.0f}s "
                  f"({saved/dt:.1f}/s)", flush=True)
            return
        kind, entries = item
        if kind == "confs":
            # entries: list of (qid, ti, confs)  (compute done in GPU worker)
            for qid, ti, confs in entries:
                d = ckpt_root / f"ckpt_qid{qid}"
                d.mkdir(parents=True, exist_ok=True)
                with open(d / f"trace_{ti:04d}.pkl", "wb") as f:
                    pickle.dump({"confs": confs, "num_tokens": len(confs)}, f)
                saved += 1
        elif kind == "oor":
            # Out of max_model_len: save empty confs with gen_len for stats
            for qid, ti, pl, gl in entries:
                d = ckpt_root / f"ckpt_qid{qid}"
                d.mkdir(parents=True, exist_ok=True)
                with open(d / f"trace_{ti:04d}.pkl", "wb") as f:
                    pickle.dump({"confs": [], "num_tokens": gl}, f)
                saved += 1
        elif kind == "err":
            # Generate failure — don't save ckpt so it can be retried next run
            for qid, ti, pl, gl in entries:
                print(f"[post {pid}] SKIP qid={qid} ti={ti} (gen error)",
                      flush=True)
        if saved and saved % 500 == 0:
            dt = time.time() - t0
            print(f"[post {pid}] saved {saved} in {dt:.0f}s ({saved/dt:.1f}/s)",
                  flush=True)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def compute_gpu_assignments(tp, num_workers, total_gpus=8):
    assert num_workers * tp <= total_gpus, \
        f"{num_workers} workers × tp={tp} > {total_gpus} GPUs"
    return [",".join(str(i * tp + j) for j in range(tp))
            for i in range(num_workers)]


def build_work_list(input_dir, output_dir, tokenizer, dataset_file, qids_filter):
    """Scan input pkls, existing ckpts, dataset jsonl. Return:
    work_items: list of (qid, ti, gen_text) for missing ckpts
    prompts_map: qid -> prompt_text
    prompt_lens_map: qid -> prompt_len (tokenized)
    all_traces_map: qid -> all_traces list
    input_pkl_map: qid -> input pkl Path
    """
    from helper import prepare_prompt

    with open(dataset_file) as f:
        dataset = [json.loads(line) for line in f]

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    prompts_map = {}
    prompt_lens_map = {}
    all_traces_map = {}
    input_pkl_map = {}

    # Discover qids from input dir
    pkls = sorted(input_dir.glob("deepconf_simple_qid*_rid*.pkl"))
    for pkl in pkls:
        try:
            qid = int(pkl.name.split("_qid", 1)[1].split("_", 1)[0])
        except (IndexError, ValueError):
            continue
        if qids_filter is not None and qid not in qids_filter:
            continue
        input_pkl_map[qid] = pkl

    work_items = []
    already_done = 0
    oor_count = 0
    for qid in sorted(input_pkl_map.keys()):
        with open(input_pkl_map[qid], "rb") as f:
            data = pickle.load(f)
        all_traces_map[qid] = data["all_traces"]
        # Build prompt from dataset jsonl (matches prepare_prompt in helper.py)
        prompt_text, _ = prepare_prompt(dataset[qid], tokenizer)
        prompts_map[qid] = prompt_text
        prompt_lens_map[qid] = len(
            tokenizer.encode(prompt_text, add_special_tokens=False)
        )

        ckpt_dir = output_dir / f"ckpt_qid{qid}"
        out_pkl = output_dir / input_pkl_map[qid].name
        if out_pkl.exists():
            # Already assembled -> skip all traces
            already_done += len(data["all_traces"])
            continue

        for ti, trace in enumerate(data["all_traces"]):
            if (ckpt_dir / f"trace_{ti:04d}.pkl").exists():
                already_done += 1
                continue
            work_items.append((qid, ti, trace["text"]))

    return (work_items, prompts_map, prompt_lens_map, all_traces_map,
            input_pkl_map, already_done)


def assemble_qid(input_pkl, ckpt_dir, out_pkl):
    """Merge per-trace ckpts into final pkl (matches run_prefill_all.asm)."""
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)
    tt = 0
    for cf in sorted(ckpt_dir.glob("trace_*.pkl")):
        ti = int(cf.stem.replace("trace_", ""))
        with open(cf, "rb") as f:
            c = pickle.load(f)
        data["all_traces"][ti]["confs"] = c["confs"]
        data["all_traces"][ti]["num_tokens"] = c["num_tokens"]
        tt += c["num_tokens"]
    data["token_stats"]["total_tokens"] = tt
    data["token_stats"]["avg_tokens_per_trace"] = tt / max(len(data["all_traces"]), 1)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(data, f)
    wc = sum(1 for t in data["all_traces"] if len(t.get("confs", [])) > 0)
    return wc, len(data["all_traces"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--dataset-file", required=True, type=Path)
    p.add_argument("--model-path", required=True)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--num-gpu-workers", type=int, default=4)
    p.add_argument("--num-producers", type=int, default=30)
    p.add_argument("--num-post", type=int, default=2)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--timeout-s", type=float, default=0.5)
    p.add_argument("--producer-batch", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=131072)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    p.add_argument("--max-num-batched-tokens", type=int, default=None,
                   help="vLLM scheduler cap (default: vLLM's default 16384)")
    p.add_argument("--qids", type=int, nargs="*", default=None)
    p.add_argument("--dry-run-work", type=int, default=None,
                   help="Only feed N work items (for quick test)")
    p.add_argument("--skip-assemble", action="store_true")
    p.add_argument("--shuffle-seed", type=int, default=0)
    a = p.parse_args()

    mp.set_start_method("spawn", force=True)

    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {a.model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True)

    print("Scanning input pkls and existing ckpts...", flush=True)
    t0 = time.time()
    qids_filter = set(a.qids) if a.qids else None
    (work_items, prompts_map, prompt_lens_map, all_traces_map,
     input_pkl_map, already_done) = build_work_list(
        a.input_dir, a.output_dir, tokenizer, a.dataset_file, qids_filter
    )
    print(f"Found {len(work_items)} traces to process, {already_done} already done "
          f"({time.time()-t0:.1f}s)", flush=True)
    print(f"QIDs: {sorted(input_pkl_map.keys())}", flush=True)

    if a.dry_run_work is not None:
        work_items = work_items[:a.dry_run_work]
        print(f"[dry-run] truncated to {len(work_items)} items", flush=True)

    if work_items:
        random.Random(a.shuffle_seed).shuffle(work_items)
        batches = [work_items[i:i + a.producer_batch]
                   for i in range(0, len(work_items), a.producer_batch)]
        print(f"Feeding {len(batches)} batches × {a.producer_batch}", flush=True)

        fork_ctx = mp.get_context("fork")
        spawn_ctx = mp.get_context("spawn")

        # Queues (spawn-compatible; usable by fork children too)
        work_queue = fork_ctx.Queue(maxsize=200)
        input_queue = spawn_ctx.Queue(maxsize=60)
        output_queue = spawn_ctx.Queue(maxsize=50)

        # Start producers (fork: inherits tokenizer + prompts_map)
        producers = []
        for i in range(a.num_producers):
            proc = fork_ctx.Process(
                target=producer_worker,
                args=(i, work_queue, input_queue, prompts_map,
                      prompt_lens_map, tokenizer, a.max_model_len),
            )
            proc.start()
            producers.append(proc)
        print(f"Started {len(producers)} producers (fork)", flush=True)

        # Start post workers (fork)
        post_procs = []
        for i in range(a.num_post):
            proc = fork_ctx.Process(
                target=post_worker,
                args=(i, output_queue, str(a.output_dir)),
            )
            proc.start()
            post_procs.append(proc)
        print(f"Started {len(post_procs)} post workers (fork)", flush=True)

        # Start GPU workers (spawn)
        gpu_assignments = compute_gpu_assignments(a.tp, a.num_gpu_workers)
        gpu_procs = []
        for i, gpu_str in enumerate(gpu_assignments):
            proc = spawn_ctx.Process(
                target=gpu_worker,
                args=(i, gpu_str, input_queue, output_queue, a.model_path,
                      a.tp, a.max_model_len, a.gpu_memory_utilization,
                      a.chunk_size, a.timeout_s, a.max_num_batched_tokens),
            )
            proc.start()
            gpu_procs.append(proc)
        print(f"Started {len(gpu_procs)} GPU workers (spawn): {gpu_assignments}",
              flush=True)

        # Queue-size monitor (daemon thread, every 10s).
        # qsize() is approximate on mp.Queue but good enough for visibility.
        def _monitor_queues():
            while True:
                time.sleep(10)
                try:
                    print(f"[queue] work={work_queue.qsize()}/200 "
                          f"input={input_queue.qsize()}/60 "
                          f"output={output_queue.qsize()}/50", flush=True)
                except Exception:
                    pass
        threading.Thread(target=_monitor_queues, daemon=True).start()

        # Feed work
        t_start = time.time()
        for batch in batches:
            work_queue.put(batch)
        # Sentinel producers
        for _ in range(a.num_producers):
            work_queue.put(None)

        print("Waiting for producers to finish...", flush=True)
        for proc in producers:
            proc.join()
        print(f"All producers done ({time.time()-t_start:.0f}s). "
              f"Signaling GPU workers...", flush=True)

        for _ in range(a.num_gpu_workers):
            input_queue.put(None)
        for proc in gpu_procs:
            proc.join()
        print(f"GPU workers done ({time.time()-t_start:.0f}s). "
              f"Signaling post workers...", flush=True)

        for _ in range(a.num_post):
            output_queue.put(None)
        for proc in post_procs:
            proc.join()
        print(f"All workers done in {time.time()-t_start:.0f}s.", flush=True)

    if a.skip_assemble:
        print("Skipping assemble (--skip-assemble).", flush=True)
        return

    print("\nAssembling per-qid pkls...", flush=True)
    for qid in sorted(input_pkl_map.keys()):
        in_pkl = input_pkl_map[qid]
        out_pkl = a.output_dir / in_pkl.name
        if out_pkl.exists():
            continue
        ckpt_dir = a.output_dir / f"ckpt_qid{qid}"
        n_ckpts = len(list(ckpt_dir.glob("trace_*.pkl")))
        n_expected = len(all_traces_map[qid])
        if n_ckpts >= n_expected:
            wc, total = assemble_qid(in_pkl, ckpt_dir, out_pkl)
            print(f"  qid {qid}: assembled {wc}/{total} -> {out_pkl.name}",
                  flush=True)
        else:
            print(f"  qid {qid}: {n_ckpts}/{n_expected} ckpts (incomplete, "
                  f"skipping assemble)", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
