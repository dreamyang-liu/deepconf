"""
Backfill token_ids into existing offline pkl files.

Tokenizes each trace's text using the model tokenizer and saves token_ids
back into the pkl. This makes convert_offline_to_online.py work correctly
for early-stopping text truncation.

Usage:
    python backfill_token_ids.py --input-dir outputs-bedrock-confs/brumo25
    python backfill_token_ids.py --input-dir outputs-bedrock-confs/brumo25 --model-path Qwen/Qwen3-32B
    python backfill_token_ids.py --input-dir outputs-bedrock-confs --recursive
"""

import argparse
import glob
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer


def backfill_one_file(pkl_path, tokenizer, n_workers=16):
    """Backfill token_ids for all traces in a single pkl file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    traces = data["all_traces"]
    # Check if already backfilled
    if traces and len(traces[0].get("token_ids", [])) > 0:
        print(f"  SKIP (already has token_ids): {os.path.basename(pkl_path)}")
        return False

    texts = [t["text"] for t in traces]
    n = len(texts)

    # Batch tokenize in parallel threads
    chunk_size = max(1, (n + n_workers - 1) // n_workers)
    chunks = [texts[i:i + chunk_size] for i in range(0, n, chunk_size)]

    def _tok_chunk(chunk):
        return tokenizer(chunk, add_special_tokens=False)["input_ids"]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_tok_chunk, chunks))

    all_ids = []
    for chunk_ids in results:
        all_ids.extend(chunk_ids)

    # Write back
    for trace, ids in zip(traces, all_ids):
        trace["token_ids"] = ids

    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    elapsed = time.time() - t0
    print(f"  OK {os.path.basename(pkl_path)}: {n} traces, {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Backfill token_ids into pkl files")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--recursive", action="store_true",
                        help="Process all subdirectories")
    parser.add_argument("--workers", type=int, default=16,
                        help="Threads for tokenization (default: 16)")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.recursive:
        pkl_files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.pkl"), recursive=True))
    else:
        pkl_files = sorted(glob.glob(os.path.join(args.input_dir, "*.pkl")))

    print(f"Found {len(pkl_files)} pkl files")

    t_start = time.time()
    done = 0
    skipped = 0

    for pkl_path in pkl_files:
        print(f"\n[{done + skipped + 1}/{len(pkl_files)}] {pkl_path}")
        if backfill_one_file(pkl_path, tokenizer, args.workers):
            done += 1
        else:
            skipped += 1

    elapsed = time.time() - t_start
    print(f"\nDone: {done} backfilled, {skipped} skipped, {elapsed:.0f}s total")


if __name__ == "__main__":
    main()
