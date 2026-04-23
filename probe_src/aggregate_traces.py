"""
Aggregate all individual trace pickle files into a single summary.

Outputs (saved to <results_dir>/aggregated/):
  - aggregated_traces.pkl: dict keyed by (qid, trace_idx) with all probe data
  - aggregated_summary.csv: flat CSV with one row per (qid, trace_idx, token_position)

If --conf-data-dir is provided, the per-token confidence list (`confs`) and
`extracted_answer` from the matching bedrock trace are merged into each record
in the pickle output. `extracted_answer` is also added to each CSV row (the
`confs` list stays pickle-only to keep CSV size manageable).

Usage:
  python aggregate_traces.py <traces_dir> [--conf-data-dir <conf_data_dir>]

Example:
  python aggregate_traces.py probe_results/aime24/traces --conf-data-dir conf-data/aime24
"""

import argparse
import pickle
import csv
from pathlib import Path

import numpy as np


def load_bedrock_by_qid(conf_data_dir: Path) -> dict:
    """Scan conf_data_dir and return {qid: path} for per-qid pkls.

    Matches any rid (e.g. 'ridbedrock', 'riddeepseek', 'ridcoder_next'), so the
    same aggregator works for bedrock-sourced data and locally-generated data.
    """
    qid_to_path = {}
    for fpath in conf_data_dir.glob("deepconf_simple_qid*_rid*_*.pkl"):
        # filename: deepconf_simple_qid{N}_rid{ID}_{timestamp}.pkl
        stem = fpath.name
        try:
            qid_str = stem.split("_qid", 1)[1].split("_", 1)[0]
            qid = int(qid_str)
        except (IndexError, ValueError):
            print(f"  [warn] could not parse qid from {stem}, skipping")
            continue
        if qid in qid_to_path:
            print(f"  [warn] duplicate source pkl for qid {qid}: {fpath.name}")
        qid_to_path[qid] = fpath
    return qid_to_path


def aggregate(traces_dir: Path, output_dir: Path, conf_data_dir: Path = None):
    files = sorted(traces_dir.glob("*.pkl"))
    print(f"Found {len(files)} trace files")

    # Pre-index bedrock files by qid (loaded lazily, one per qid)
    bedrock_index = {}
    bedrock_cache = {}  # qid -> all_traces list
    if conf_data_dir is not None:
        bedrock_index = load_bedrock_by_qid(conf_data_dir)
        print(f"Found {len(bedrock_index)} bedrock files in {conf_data_dir}")

    all_traces = {}  # (qid, trace_idx) -> full record
    csv_rows = []
    merged_count = 0
    missing_qids = set()
    oob_count = 0

    for i, fpath in enumerate(files):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(files)} files...")

        with open(fpath, "rb") as f:
            data = pickle.load(f)

        qid = data["qid"]
        trace_idx = data["trace_idx"]
        ground_truth = data["ground_truth"]
        probes = data["probes"]

        extracted_answer = None

        # Merge bedrock data if available
        if bedrock_index:
            if qid not in bedrock_cache:
                bpath = bedrock_index.get(qid)
                if bpath is None:
                    bedrock_cache[qid] = None
                    missing_qids.add(qid)
                else:
                    with open(bpath, "rb") as bf:
                        bedrock_cache[qid] = pickle.load(bf)["all_traces"]

            btraces = bedrock_cache[qid]
            if btraces is not None:
                if trace_idx < len(btraces):
                    btrace = btraces[trace_idx]
                    confs = btrace.get("confs")
                    # Store as float16 numpy array: ~8x smaller than list of
                    # np.float64 scalars and ~orders-of-magnitude faster to
                    # pickle/unpickle. Precision is ample for confidence values.
                    data["confs"] = (
                        np.asarray(confs, dtype=np.float16)
                        if confs is not None else None
                    )
                    data["extracted_answer"] = btrace.get("extracted_answer")
                    extracted_answer = data["extracted_answer"]
                    merged_count += 1
                else:
                    oob_count += 1

        all_traces[(qid, trace_idx)] = data

        for token_pos in sorted(probes.keys()):
            p = probes[token_pos]
            csv_rows.append({
                "qid": qid,
                "trace_idx": trace_idx,
                "ground_truth": ground_truth,
                "token_position": token_pos,
                "answer": p["answer"],
                "is_correct": p["is_correct"],
                "raw_text": p["raw_text"],
                "avg_conf": p["avg_conf"],
                "extracted_answer": extracted_answer,
            })

    if bedrock_index:
        print(f"Merged bedrock data into {merged_count} traces")
        if missing_qids:
            print(f"  [warn] no bedrock file for qids: {sorted(missing_qids)}")
        if oob_count:
            print(f"  [warn] trace_idx out of bounds in bedrock for {oob_count} traces")

    # Save pickle
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_dir / "aggregated_traces.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(all_traces, f)
    print(f"Saved aggregated pickle: {pkl_path} ({len(all_traces)} traces)")

    # Save CSV
    csv_path = output_dir / "aggregated_summary.csv"
    fieldnames = ["qid", "trace_idx", "ground_truth", "token_position",
                  "answer", "is_correct", "raw_text", "avg_conf",
                  "extracted_answer"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved aggregated CSV: {csv_path} ({len(csv_rows)} rows)")

    # Print quick stats
    print("\n--- Quick Stats ---")
    qids = sorted(set(k[0] for k in all_traces))
    print(f"Total qids: {len(qids)}")
    print(f"Total traces: {len(all_traces)}")
    print(f"Total probe points: {len(csv_rows)}")

    for qid in qids:
        traces_for_qid = [k for k in all_traces if k[0] == qid]
        correct_final = 0
        for k in traces_for_qid:
            probes = all_traces[k]["probes"]
            last_pos = max(probes.keys())
            if probes[last_pos]["is_correct"]:
                correct_final += 1
        print(f"  qid {qid:>2}: {len(traces_for_qid):>5} traces, "
              f"final-probe accuracy = {correct_final}/{len(traces_for_qid)} "
              f"({correct_final / len(traces_for_qid) * 100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate probe trace files")
    parser.add_argument("traces_dir", type=Path, help="Directory containing trace .pkl files")
    parser.add_argument("--conf-data-dir", type=Path, default=None,
                        help="Optional bedrock conf-data directory to merge "
                             "per-token confs and extracted_answer from")
    args = parser.parse_args()

    output_dir = args.traces_dir.parent / "aggregated"
    aggregate(args.traces_dir, output_dir, conf_data_dir=args.conf_data_dir)
