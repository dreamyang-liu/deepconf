"""
Convert Bedrock batch inference outputs (from ~/projects/) to DeepConf offline format.

Handles three datasets with different recordId conventions:
  - aime24:  aime24__{numeric_qid}__trace_{idx}   (qids 60-89, needs fuzzy mapping to aime_2024.jsonl)
  - brumo25: brumo25__problem_{N}__trace_{idx}     (N=0..29, maps directly to brumo_2025.jsonl)
  - hmmt:    hmmt__problem_{N}__trace_{idx}         (N=0..29, maps directly to hmmt_feb_2025.jsonl)

Usage:
    conda run -n deepconf python convert-bedrock-batch.py --dataset aime24
    conda run -n deepconf python convert-bedrock-batch.py --dataset brumo25
    conda run -n deepconf python convert-bedrock-batch.py --dataset hmmt
    conda run -n deepconf python convert-bedrock-batch.py --dataset all
"""

import argparse
import glob
import json
import os
import pickle
import re
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher

from helper import extract_answer, equal_func, weighted_majority_vote

PROJECTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bedrock-data")

DATASET_CONFIG = {
    "aime24": {
        "dataset_file": "aime_2024.jsonl",
        "batch_dir": os.path.join(PROJECTS_DIR, "aime24"),
    },
    "aime25": {
        "dataset_file": "aime_2025.jsonl",
        "batch_dir": os.path.join(PROJECTS_DIR, "aime25"),
    },
    "brumo25": {
        "dataset_file": "brumo_2025.jsonl",
        "batch_dir": os.path.join(PROJECTS_DIR, "brumo25"),
    },
    "hmmt": {
        "dataset_file": "hmmt_feb_2025.jsonl",
        "batch_dir": os.path.join(PROJECTS_DIR, "hmmt"),
    },
}


def build_aime24_qid_mapping(batch_dir, dataset):
    """Build mapping from aime24 numeric qids to dataset indices via fuzzy matching."""
    # Collect one question per qid from batch data
    batch_qs = {}
    first_batch = glob.glob(os.path.join(batch_dir, "0", "*", "*.jsonl.out"))
    first_batch = [f for f in first_batch if "manifest" not in f]
    if not first_batch:
        raise FileNotFoundError(f"No batch files found in {batch_dir}/0/")

    with open(first_batch[0], "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            qid_str = d["recordId"].split("__")[1]
            if qid_str not in batch_qs:
                batch_qs[qid_str] = d["modelInput"]["messages"][0]["content"][0]["text"]

    def extract_numbers(s):
        return set(re.findall(r"\d+", s))

    # Score all pairs and greedily assign best matches
    matches = []
    for qid_str, bq in batch_qs.items():
        bn = extract_numbers(bq)
        for i, d in enumerate(dataset):
            dn = extract_numbers(d["question"])
            num_overlap = len(bn & dn) / max(len(bn | dn), 1)
            seq_score = SequenceMatcher(None, bq[:300], d["question"][:300]).ratio()
            combined = 0.5 * seq_score + 0.5 * num_overlap
            matches.append((combined, qid_str, i))

    matches.sort(reverse=True)
    qid_to_ds = {}
    used_ds = set()
    for score, qid_str, ds_idx in matches:
        if qid_str in qid_to_ds or ds_idx in used_ds:
            continue
        qid_to_ds[qid_str] = ds_idx
        used_ds.add(ds_idx)

    assert len(qid_to_ds) == len(batch_qs), (
        f"Could not map all qids: {len(qid_to_ds)}/{len(batch_qs)}"
    )
    return qid_to_ds


def parse_record_qid(record_id, dataset_name):
    """Parse recordId and return (qid_str, trace_idx)."""
    parts = record_id.split("__")
    qid_str = parts[1]
    trace_idx = int(parts[2].replace("trace_", ""))
    return qid_str, trace_idx


def parse_bedrock_content(record):
    """Extract reasoning text and content text from a Bedrock record."""
    mo = record.get("modelOutput", {})
    if not mo or "output" not in mo:
        return None, None, "unknown"

    msg = mo["output"]["message"]
    stop_reason = mo.get("stopReason", "unknown")

    reasoning = ""
    content = ""
    for c in msg.get("content", []):
        if c is None:
            continue
        rc = c.get("reasoningContent")
        if rc and isinstance(rc, dict):
            rt = rc.get("reasoningText") or rc.get("text", "")
            if isinstance(rt, dict):
                reasoning = rt.get("text", "")
            else:
                reasoning = str(rt)
        t = c.get("text")
        if t:
            content = str(t)

    return reasoning, content, stop_reason


def build_trace(reasoning, content, ground_truth, stop_reason):
    """Build a DeepConf-compatible trace dict."""
    full_text = reasoning + "\n" + content if reasoning and content else reasoning or content

    extracted = extract_answer(full_text)
    is_correct = False
    if extracted and ground_truth:
        try:
            is_correct = equal_func(extracted, ground_truth)
        except Exception:
            is_correct = str(extracted) == str(ground_truth)

    return {
        "stop_reason": stop_reason,
        "text": full_text,
        "token_ids": [],
        "num_tokens": 0,
        "confs": [],
        "extracted_answer": extracted,
        "is_correct": is_correct,
    }


def convert_dataset(dataset_name, output_dir, rid="bedrock"):
    cfg = DATASET_CONFIG[dataset_name]
    batch_dir = cfg["batch_dir"]
    dataset_file = cfg["dataset_file"]

    # Load ground truth
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line.strip()) for line in f]
    print(f"[{dataset_name}] Loaded {len(dataset)} questions from {dataset_file}")

    # Build qid mapping
    if dataset_name == "aime24":
        qid_to_ds = build_aime24_qid_mapping(batch_dir, dataset)
        print(f"[{dataset_name}] Built fuzzy qid mapping: {len(qid_to_ds)} questions")
    elif dataset_name == "aime25":
        # numeric qid -> dataset index directly
        qid_to_ds = {str(i): i for i in range(len(dataset))}
    else:
        # problem_N -> N
        qid_to_ds = {f"problem_{i}": i for i in range(len(dataset))}

    # Find all batch output files
    jsonl_files = sorted(glob.glob(os.path.join(batch_dir, "*", "*", "*.jsonl.out")))
    jsonl_files = [f for f in jsonl_files if "manifest" not in f]
    print(f"[{dataset_name}] Found {len(jsonl_files)} batch files")

    # Parse all records, grouped by question
    traces_by_qid = defaultdict(list)
    total_records = 0
    parse_errors = 0

    for fpath in jsonl_files:
        batch_name = os.path.basename(fpath)
        print(f"  Reading {batch_name}...")
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                total_records += 1
                try:
                    record = json.loads(line.strip())
                    qid_str, trace_idx = parse_record_qid(
                        record["recordId"], dataset_name
                    )
                    reasoning, content, stop_reason = parse_bedrock_content(record)
                    if not reasoning and not content:
                        parse_errors += 1
                        continue
                    traces_by_qid[qid_str].append(
                        (trace_idx, reasoning, content, stop_reason)
                    )
                except Exception:
                    parse_errors += 1

    print(f"[{dataset_name}] Parsed {total_records} records, {parse_errors} errors")
    print(f"[{dataset_name}] Questions with traces: {len(traces_by_qid)}")

    # Build and save per-question pickle files
    ds_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(ds_output_dir, exist_ok=True)
    results = []

    for qid_str in sorted(traces_by_qid.keys(), key=lambda x: qid_to_ds.get(x, -1)):
        if qid_str not in qid_to_ds:
            print(f"  [WARN] qid '{qid_str}' not in mapping, skipping")
            continue

        ds_idx = qid_to_ds[qid_str]
        ground_truth = str(dataset[ds_idx].get("answer", "")).strip()
        raw_traces = sorted(traces_by_qid[qid_str], key=lambda x: x[0])

        all_traces = []
        for trace_idx, reasoning, content, stop_reason in raw_traces:
            trace = build_trace(reasoning, content, ground_truth, stop_reason)
            all_traces.append(trace)

        # Voting
        voting_answers = [
            t["extracted_answer"] for t in all_traces if t["extracted_answer"]
        ]
        voting_weights = [1.0] * len(voting_answers)
        voted_answer = weighted_majority_vote(voting_answers, voting_weights)
        is_voted_correct = False
        if voted_answer and ground_truth:
            try:
                is_voted_correct = equal_func(voted_answer, ground_truth)
            except Exception:
                is_voted_correct = str(voted_answer) == str(ground_truth)

        correct_traces = sum(1 for t in all_traces if t["is_correct"])
        accuracy = correct_traces / len(all_traces) if all_traces else 0

        problem_result = {
            "question_id": ds_idx,
            "run_id": rid,
            "question": dataset[ds_idx]["question"],
            "ground_truth": ground_truth,
            "all_traces": all_traces,
            "voted_answer": voted_answer,
            "is_voted_correct": is_voted_correct,
            "accuracy": accuracy,
            "correct_traces_count": correct_traces,
            "token_stats": {
                "total_tokens": 0,
                "total_traces_count": len(all_traces),
                "avg_tokens_per_trace": 0,
            },
            "timing_stats": {},
            "config": {
                "model_path": "bedrock",
                "total_budget": len(all_traces),
                "window_size": 2048,
                "source": "bedrock_batch",
                "original_qid": qid_str,
            },
            "timestamp": datetime.now().isoformat(),
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            ds_output_dir,
            f"deepconf_simple_qid{ds_idx}_rid{rid}_{timestamp}.pkl",
        )
        with open(out_path, "wb") as f:
            pickle.dump(problem_result, f)

        status = "OK" if is_voted_correct else "X "
        print(
            f"  [{status}] qid={ds_idx:2d} ({qid_str:>12s})  "
            f"traces={len(all_traces):5d}  "
            f"voted={str(voted_answer)[:15]:15s}  "
            f"gt={ground_truth:10s}  "
            f"acc={correct_traces}/{len(all_traces)} ({accuracy:.1%})"
        )
        results.append(problem_result)

    # Summary
    total_correct = sum(1 for r in results if r["is_voted_correct"])
    total_traces = sum(r["token_stats"]["total_traces_count"] for r in results)
    print(f"\n{'='*60}")
    print(f"[{dataset_name}] SUMMARY")
    print(f"{'='*60}")
    print(f"Questions: {len(results)}")
    print(f"Total traces: {total_traces}")
    print(f"Voted correct: {total_correct}/{len(results)} ({total_correct/len(results):.1%})")
    print(f"Output: {ds_output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert Bedrock batch outputs to DeepConf offline format"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["aime24", "aime25", "brumo25", "hmmt", "all"],
        help="Dataset to convert",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs-bedrock",
        help="Output directory (default: outputs-bedrock)",
    )
    parser.add_argument(
        "--rid", type=str, default="bedrock", help="Run ID (default: bedrock)"
    )
    args = parser.parse_args()

    datasets = (
        list(DATASET_CONFIG.keys()) if args.dataset == "all" else [args.dataset]
    )

    for ds_name in datasets:
        print(f"\n{'#'*60}")
        print(f"# Converting {ds_name}")
        print(f"{'#'*60}\n")
        convert_dataset(ds_name, args.output_dir, args.rid)


if __name__ == "__main__":
    main()
