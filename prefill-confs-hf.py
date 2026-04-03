"""
Recover per-token confidence from existing inference text via HuggingFace forward pass.

Instead of vLLM (which OOMs on prompt_logprobs for long sequences), this uses
HuggingFace transformers directly. We run model.forward() on the full sequence
and extract logprobs from the logits.

Usage:
    conda run -n deepconf python prefill-confs-hf.py \
        --input-dir outputs-bedrock/aime24 \
        --dataset-file aime_2024.jsonl \
        --model-path Qwen/Qwen3-32B \
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
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

from helper import (
    extract_answer,
    equal_func,
    weighted_majority_vote,
    prepare_prompt,
)


TOP_K_LOGPROBS = 5  # number of top logprobs to average for confidence


@torch.no_grad()
def compute_confs_for_trace(model, input_ids, prompt_len, device, max_chunk=4096):
    """
    Run forward pass and compute per-token confidence for the generated portion.

    For very long sequences, process in chunks to avoid OOM.
    Confidence = -mean(top_k logprobs) at each token position.
    """
    seq_len = input_ids.shape[1]
    gen_len = seq_len - prompt_len

    if gen_len <= 0:
        return []

    # For confidence, we need logits at positions [prompt_len-1, ..., seq_len-2]
    # because logit at position i predicts token at position i+1
    # So conf for generated token at position prompt_len is from logit at prompt_len-1

    confs = []

    # Process in chunks if sequence is very long
    # We use a sliding window approach but for simplicity, just do full forward
    # HF models can handle long sequences with flash attention
    ids_tensor = input_ids.to(device)

    # Forward pass — may need chunking for very long seqs
    if seq_len <= max_chunk:
        outputs = model(ids_tensor)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        # Get logits for generated portion: positions [prompt_len-1 .. seq_len-2]
        gen_logits = logits[0, prompt_len - 1 : seq_len - 1, :]  # (gen_len, vocab)

        # Convert to log probs
        log_probs = torch.log_softmax(gen_logits, dim=-1)

        # Top-k log probs at each position
        topk_lps, _ = torch.topk(log_probs, TOP_K_LOGPROBS, dim=-1)

        # Confidence = -mean(top_k_logprobs)
        mean_lps = topk_lps.mean(dim=-1)  # (gen_len,)
        confs = (-mean_lps).float().cpu().numpy().round(3).tolist()

        del outputs, logits, gen_logits, log_probs, topk_lps, mean_lps
    else:
        # Chunk processing for very long sequences
        # We need logits at positions prompt_len-1 to seq_len-2
        # Process overlapping chunks, only use the relevant logits from each
        chunk_confs = []
        stride = max_chunk - 128  # overlap to handle boundary

        for start in range(0, seq_len, stride):
            end = min(start + max_chunk, seq_len)
            chunk = ids_tensor[:, start:end]

            outputs = model(chunk)
            logits = outputs.logits[0]  # (chunk_len, vocab)

            # Which positions in this chunk correspond to generated tokens?
            for pos_in_chunk in range(logits.shape[0] - 1):
                global_pos = start + pos_in_chunk  # this logit predicts token at global_pos+1
                token_pos = global_pos + 1  # the token being predicted

                if token_pos < prompt_len or token_pos >= seq_len:
                    continue
                # Avoid duplicates from overlap
                gen_idx = token_pos - prompt_len
                if gen_idx < len(chunk_confs):
                    continue

                lp = torch.log_softmax(logits[pos_in_chunk], dim=-1)
                topk, _ = torch.topk(lp, TOP_K_LOGPROBS)
                conf = -topk.mean().item()
                chunk_confs.append(round(conf, 3))

            del outputs, logits
            torch.cuda.empty_cache()

            if end >= seq_len:
                break

        confs = chunk_confs

    del ids_tensor
    torch.cuda.empty_cache()

    return confs


def process_question(model, tokenizer, device, pickle_path, prompt_text, ground_truth,
                     output_dir, max_model_len=40960):
    """Process all traces for a single question."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    qid = data["question_id"]
    all_traces = data["all_traces"]
    n_traces = len(all_traces)

    # Check output — skip if already done
    out_path = os.path.join(output_dir, os.path.basename(pickle_path))
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            existing = pickle.load(f)
        if existing["all_traces"] and len(existing["all_traces"][0].get("confs", [])) > 0:
            print(f"  qid={qid}: already processed, skipping")
            return existing

    # Tokenize prompt once
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    total_tokens = 0
    start_time = time.time()
    processed = 0
    skipped = 0

    for idx, trace in enumerate(all_traces):
        full_text = prompt_text + trace["text"]
        input_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"]

        seq_len = input_ids.shape[1]
        if seq_len > max_model_len:
            skipped += 1
            trace["num_tokens"] = seq_len - prompt_len
            continue

        confs = compute_confs_for_trace(model, input_ids, prompt_len, device)

        trace["confs"] = confs
        if trace["num_tokens"] == 0:
            trace["num_tokens"] = len(confs)
        total_tokens += trace["num_tokens"]
        processed += 1

        if processed % 10 == 0 or processed == 1:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (n_traces - processed - skipped) / rate if rate > 0 else 0
            print(f"    {processed}/{n_traces} ({skipped} skipped) "
                  f"- {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    # Update stats
    data["token_stats"]["total_tokens"] = total_tokens
    data["token_stats"]["avg_tokens_per_trace"] = (
        total_tokens / n_traces if n_traces else 0
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    elapsed = time.time() - start_time
    print(f"  qid={qid}: {processed}/{n_traces} traces ({skipped} skipped), "
          f"{total_tokens} tokens, {elapsed:.1f}s -> {out_path}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Recover per-token confidence via HuggingFace forward pass"
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--dataset-file", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--qids", type=int, nargs="*", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir.rstrip("/") + "-confs"

    # Load dataset
    with open(args.dataset_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line.strip()) for line in f]
    print(f"Loaded {len(dataset)} questions from {args.dataset_file}")

    # Init tokenizer
    print(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Prepare prompts
    prompts = {}
    for i, q in enumerate(dataset):
        prompt_text, ground_truth = prepare_prompt(q, tokenizer)
        prompts[i] = (prompt_text, ground_truth)

    # Load model
    print(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    # Find and process pickle files
    pickle_files = sorted(glob.glob(os.path.join(args.input_dir, "*.pkl")))
    print(f"Found {len(pickle_files)} pickle files")

    total_start = time.time()
    results = []

    for pkl_path in pickle_files:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        qid = data["question_id"]

        if args.qids is not None and qid not in args.qids:
            continue
        if qid not in prompts:
            continue

        prompt_text, ground_truth = prompts[qid]
        print(f"\nProcessing qid={qid} ({len(data['all_traces'])} traces)...")

        result = process_question(
            model, tokenizer, device, pkl_path, prompt_text, ground_truth,
            args.output_dir, args.max_model_len,
        )
        results.append(result)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"DONE: {len(results)} questions in {total_time:.1f}s")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
