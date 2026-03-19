"""
Convert Bedrock batch inference outputs to DeepConf offline format.

Reads bedrock-format .jsonl.out files (with reasoningContent),
extracts the full text (reasoning + content), and saves as
DeepConf-compatible pickle files (one per question).

The output format matches deepconf-offline.py exactly, except:
- confs/token_ids are empty (no logprobs from bedrock)
- Use prefill-confs.py to recover confs via SGLang prefill if needed

Usage:
    python convert-bedrock-to-offline.py \
        --input-dir data/qwen32b/qwen3-32b/aime25_output \
        --dataset-file aime_2025.jsonl \
        --output-dir outputs-bedrock \
        --model-name Qwen3-32B
"""

import argparse
import glob
import json
import os
import pickle
from collections import defaultdict
from datetime import datetime

from helper import (
    extract_answer,
    equal_func,
    weighted_majority_vote,
)


def parse_bedrock_record(record):
    """Parse a single bedrock batch output record.

    Returns (qid, trace_idx, reasoning_text, content_text, stop_reason) or None on error.
    """
    record_id = record.get("recordId", "")
    # recordId format: aime25__{qid}__trace_{idx}
    parts = record_id.split("__")
    if len(parts) < 3:
        return None
    qid = int(parts[1])
    trace_idx = int(parts[2].replace("trace_", ""))

    mo = record.get("modelOutput", {})
    if not mo or "output" not in mo:
        return None

    msg = mo["output"]["message"]
    stop_reason = mo.get("stopReason", "unknown")

    reasoning = ""
    content = ""
    for c in msg.get("content", []):
        if c is None:
            continue
        rc = c.get("reasoningContent")
        if rc:
            if isinstance(rc, dict):
                rt = rc.get("reasoningText") or rc.get("text", "")
                if isinstance(rt, dict):
                    reasoning = rt.get("text", "")
                else:
                    reasoning = str(rt)
        t = c.get("text")
        if t:
            content = str(t)

    return qid, trace_idx, reasoning, content, stop_reason


def build_trace(reasoning, content, ground_truth, stop_reason):
    """Build a DeepConf-compatible trace dict from bedrock text."""
    # Full text = reasoning (thinking) + content (final answer)
    # DeepConf expects the full generation including thinking
    full_text = reasoning + "\n" + content if reasoning and content else reasoning or content

    extracted = extract_answer(full_text)
    is_correct = False
    if extracted and ground_truth:
        try:
            is_correct = equal_func(extracted, ground_truth)
        except:
            is_correct = str(extracted) == str(ground_truth)

    return {
        "stop_reason": stop_reason,
        "text": full_text,
        "token_ids": [],       # no token_ids from bedrock
        "num_tokens": 0,       # unknown without tokenizer
        "confs": [],           # no logprobs from bedrock, use prefill-confs.py to recover
        "extracted_answer": extracted,
        "is_correct": is_correct,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert Bedrock batch outputs to DeepConf offline format"
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing batch output folders (e.g. data/qwen32b/qwen3-32b/aime25_output)")
    parser.add_argument("--dataset-file", type=str, required=True,
                        help="JSONL dataset file for ground truth")
    parser.add_argument("--output-dir", type=str, default="outputs-bedrock",
                        help="Output directory for pickle files")
    parser.add_argument("--model-name", type=str, default="Qwen3-32B",
                        help="Model name for metadata")
    parser.add_argument("--rid", type=str, default="bedrock",
                        help="Run ID")
    parser.add_argument("--window-size", type=int, default=2048)
    args = parser.parse_args()

    # Load dataset for ground truth
    with open(args.dataset_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line.strip()) for line in f]
    print(f"Loaded {len(dataset)} questions from {args.dataset_file}")

    # Find all .jsonl.out files
    pattern = os.path.join(args.input_dir, "*", "*", "*.jsonl.out")
    jsonl_files = sorted(glob.glob(pattern))
    jsonl_files = [f for f in jsonl_files if not f.endswith("manifest.json.out")]
    print(f"Found {len(jsonl_files)} batch output files")

    # Parse all records, grouped by question
    traces_by_qid = defaultdict(list)
    total_records = 0
    parse_errors = 0

    for fpath in jsonl_files:
        print(f"  Reading {fpath}...")
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                total_records += 1
                try:
                    record = json.loads(line.strip())
                    parsed = parse_bedrock_record(record)
                    if parsed is None:
                        parse_errors += 1
                        continue
                    qid, trace_idx, reasoning, content, stop_reason = parsed
                    if not reasoning and not content:
                        parse_errors += 1
                        continue
                    traces_by_qid[qid].append((trace_idx, reasoning, content, stop_reason))
                except Exception as e:
                    parse_errors += 1

    print(f"\nParsed {total_records} records, {parse_errors} errors")
    print(f"Questions with traces: {len(traces_by_qid)}")
    for qid in sorted(traces_by_qid.keys()):
        print(f"  qid {qid}: {len(traces_by_qid[qid])} traces")

    # Build and save per-question pickle files
    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for qid in sorted(traces_by_qid.keys()):
        if qid >= len(dataset):
            print(f"  [WARN] qid {qid} out of dataset range, skipping")
            continue

        ground_truth = str(dataset[qid].get("answer", "")).strip()
        raw_traces = sorted(traces_by_qid[qid], key=lambda x: x[0])

        all_traces = []
        for trace_idx, reasoning, content, stop_reason in raw_traces:
            trace = build_trace(reasoning, content, ground_truth, stop_reason)
            all_traces.append(trace)

        # Voting
        voting_answers = [t["extracted_answer"] for t in all_traces if t["extracted_answer"]]
        voting_weights = [1.0] * len(voting_answers)
        voted_answer = weighted_majority_vote(voting_answers, voting_weights)
        is_voted_correct = False
        if voted_answer and ground_truth:
            try:
                is_voted_correct = equal_func(voted_answer, ground_truth)
            except:
                is_voted_correct = str(voted_answer) == str(ground_truth)

        correct_traces = sum(1 for t in all_traces if t["is_correct"])
        accuracy = correct_traces / len(all_traces) if all_traces else 0

        problem_result = {
            "question_id": qid,
            "run_id": args.rid,
            "question": dataset[qid]["question"],
            "ground_truth": ground_truth,
            "all_traces": all_traces,
            "voted_answer": voted_answer,
            "is_voted_correct": is_voted_correct,
            "accuracy": accuracy,
            "correct_traces_count": correct_traces,
            "token_stats": {
                "total_tokens": 0,  # unknown without tokenizer
                "total_traces_count": len(all_traces),
                "avg_tokens_per_trace": 0,
            },
            "timing_stats": {},
            "config": {
                "model_path": args.model_name,
                "total_budget": len(all_traces),
                "window_size": args.window_size,
                "source": "bedrock_batch",
            },
            "timestamp": datetime.now().isoformat(),
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            args.output_dir,
            f"deepconf_simple_qid{qid}_rid{args.rid}_{timestamp}.pkl",
        )
        with open(out_path, "wb") as f:
            pickle.dump(problem_result, f)

        status = "OK" if is_voted_correct else "X "
        print(f"  [{status}] qid={qid:2d}  voted={str(voted_answer)[:20]:20s}  "
              f"gt={ground_truth:10s}  acc={correct_traces}/{len(all_traces)} ({accuracy:.1%})  "
              f"-> {out_path}")
        results.append(problem_result)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    total_correct = sum(1 for r in results if r["is_voted_correct"])
    print(f"Questions: {len(results)}")
    print(f"Voted correct: {total_correct}/{len(results)} ({total_correct/len(results):.1%})")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
