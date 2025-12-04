"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
import re
from tqdm import tqdm

def parse_filename(filename):
    """Parse filename to extract method, qid, rid, timestamp"""
    # Pattern for simple method
    simple_pattern = r'^deepconf_simple_qid(\d+)_rid([^_]+)_(\d{8}_\d{6})\.pkl$'
    simple_match = re.match(simple_pattern, filename)

    if simple_match:
        qid, rid, timestamp = simple_match.groups()
        return 'simple', int(qid), rid, timestamp

    # Pattern for original method
    original_pattern = r'^deepconf_qid(\d+)_rid([^_]+)_(\d{8}_\d{6})\.pkl$'
    original_match = re.match(original_pattern, filename)

    if original_match:
        qid, rid, timestamp = original_match.groups()
        return 'original', int(qid), rid, timestamp

    # Check if it's just a .pkl file in the folder
    if filename.endswith('.pkl'):
        return 'unknown', 0, '0', '0'

    return None

def extract_key_metrics(result, method_type, filename, qid, rid):
    """Extract only essential metrics from a single result"""
    try:
        # Basic info
        is_correct = result.get('is_voted_correct', False)
        voted_answer = result.get('voted_answer', 'N/A')
        ground_truth = result.get('ground_truth', 'N/A')
        question_id = result.get('question_id', qid)
        run_id = result.get('run_id', rid)

        # Count traces and tokens
        warmup_traces = result.get('warmup_traces', [])
        final_traces = result.get('final_traces', [])

        warmup_count = len([t for t in warmup_traces if t.get('num_tokens', 0) <= 64000])
        final_count = len([t for t in final_traces if t.get('num_tokens', 0) <= 64000])

        warmup_tokens = sum(min(t.get('num_tokens', 0), 64000) for t in warmup_traces)
        final_tokens = sum(min(t.get('num_tokens', 0), 64000) for t in final_traces)

        total_traces = warmup_count + final_count
        total_tokens = warmup_tokens + final_tokens

        # Get confidence metrics from final traces
        min_confs = []
        for trace in final_traces:
            if trace.get('num_tokens', 0) <= 64000:
                min_conf = trace.get('min_conf', None)
                if min_conf is not None:
                    min_confs.append(min_conf)

        avg_min_conf = np.mean(min_confs) if min_confs else None

        return {
            'method': method_type,
            'question_id': question_id,
            'run_id': run_id,
            'filename': filename,
            'is_voted_correct': is_correct,
            'voted_answer': voted_answer,
            'ground_truth': ground_truth,
            'total_tokens': total_tokens,
            'total_traces': total_traces,
            'warmup_traces': warmup_count,
            'final_traces': final_count,
            'warmup_tokens': warmup_tokens,
            'final_tokens': final_tokens,
            'avg_tokens_per_trace': total_tokens / total_traces if total_traces > 0 else 0,
            'avg_min_conf': avg_min_conf
        }
    except Exception as e:
        print(f"Error extracting metrics from {filename}: {e}")
        return None

def process_file(filepath):
    filename = os.path.basename(filepath)
    parsed = parse_filename(filename)
    if not parsed:
        return None, None
    method_type, qid, rid, timestamp = parsed

    try:
        # Load file
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        # Extract only what we need
        metrics = extract_key_metrics(data, method_type, filename, qid, rid)

        # Clean up - delete the large data structure immediately
        del data

        return metrics, None
    except Exception as e:
        return None, (filename, str(e))

def load_and_analyze_results(outputs_dir="outputs-online"):
    """Load results, extract metrics immediately, and clean up memory"""

    if not os.path.exists(outputs_dir):
        print(f"Directory {outputs_dir} not found!")
        return pd.DataFrame()

    all_files = []
    for root, dirs, files in os.walk(outputs_dir):
        for file in files:
            if file.endswith('.pkl'):
                all_files.append(os.path.join(root, file))

    print(f"Found {len(all_files)} pickle files")

    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    all_metrics = []
    load_errors = []

    with ProcessPoolExecutor(max_workers=64) as executor:
        results = list(tqdm(executor.map(process_file, all_files), total=len(all_files), desc="Processing files"))

    for metrics, error in results:
        if metrics:
            all_metrics.append(metrics)
        if error:
            load_errors.append(error)

    if load_errors:
        print(f"Load errors: {len(load_errors)}")
        for fname, error in load_errors[:5]:  # Show first 5 errors
            print(f"  {fname}: {error}")

    df = pd.DataFrame(all_metrics)
    print(f"Successfully loaded {len(df)} experiments")

    return df

def analyze_results(df):
    """Analyze and compare results"""

    if df.empty:
        print("No data to analyze!")
        return df

    print(f"\n" + "="*50)
    print("DEEPCONF ONLINE RESULTS ANALYSIS")
    print("="*50)

    # Basic counts
    simple_df = df[df['method'] == 'simple']
    original_df = df[df['method'] == 'original']
    unknown_df = df[df['method'] == 'unknown']

    print(f"Total experiments: {len(df)}")
    print(f"Self-Consistency method: {len(simple_df)} experiments")
    print(f"DeepConf method: {len(original_df)} experiments")
    print(f"Unknown method: {len(unknown_df)} experiments")
    print(f"Questions covered: {df['question_id'].nunique()}")

    # Overall accuracy
    print(f"\n" + "-"*40)
    print("OVERALL RESULTS")
    print("-"*40)

    overall_accuracy = df['is_voted_correct'].mean()
    print(f"Overall Accuracy: {overall_accuracy:.1%} ({df['is_voted_correct'].sum()}/{len(df)})")

    # Method comparison
    print(f"\n" + "-"*40)
    print("METHOD COMPARISON")
    print("-"*40)

    for method in ['simple', 'original', 'unknown']:
        method_data = df[df['method'] == method]
        if not method_data.empty:
            accuracy = method_data['is_voted_correct'].mean()
            avg_tokens = method_data['total_tokens'].mean()
            avg_traces = method_data['total_traces'].mean()

            method_name = {'simple': 'Self-Consistency', 'original': 'DeepConf', 'unknown': 'Unknown'}[method]
            print(f"{method_name}:")
            print(f"  Accuracy: {accuracy:.1%} ({method_data['is_voted_correct'].sum()}/{len(method_data)})")
            print(f"  Avg tokens: {avg_tokens:.0f}")
            print(f"  Avg traces: {avg_traces:.1f}")
            if method in ['original', 'unknown']:
                avg_conf = method_data['avg_min_conf'].mean()
                if not pd.isna(avg_conf):
                    print(f"  Avg min confidence: {avg_conf:.3f}")

    # Per-question breakdown
    if df['question_id'].nunique() > 1:
        print(f"\n" + "-"*40)
        print("PER-QUESTION BREAKDOWN")
        print("-"*40)

        for qid in sorted(df['question_id'].unique()):
            if qid == 0:  # Skip unknown question IDs
                continue
            qid_data = df[df['question_id'] == qid]
            accuracy = qid_data['is_voted_correct'].mean()
            ground_truth = qid_data['ground_truth'].iloc[0] if len(qid_data) > 0 else 'N/A'
            print(f"Question {qid} (GT: {ground_truth}): {accuracy:.1%} ({qid_data['is_voted_correct'].sum()}/{len(qid_data)})")

            # Show all voted answers for this question
            voted_answers = qid_data['voted_answer'].tolist()
            run_ids = qid_data['run_id'].tolist()
            methods = qid_data['method'].tolist()
            correctness = qid_data['is_voted_correct'].tolist()

            for i, (run_id, method, voted_answer, correct) in enumerate(zip(run_ids, methods, voted_answers, correctness)):
                status = '✓' if correct else '✗'
                print(f"  Run {run_id} ({method}): {voted_answer} {status}")

    # Show some examples
    print(f"\n" + "-"*40)
    print("SAMPLE RESULTS")
    print("-"*40)

    for i, row in df.head(5).iterrows():
        print(f"Q{row['question_id']}_R{row['run_id']}: {row['voted_answer']} (GT: {row['ground_truth']}) - {'✓' if row['is_voted_correct'] else '✗'}")

    return df

import argparse

def main():
    """Main analysis function"""
    print("Loading DeepConf online results...")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze DeepConf online results')
    parser.add_argument('--output_path', type=str, default='outputs-online',
                       help='Path to directory containing output pickle files (default: outputs-online)')
    args = parser.parse_args()

    # Load and process results efficiently
    df = load_and_analyze_results(args.output_path)

    if df.empty:
        print("No valid results found!")
        return df

    # Analyze results
    df = analyze_results(df)

    # Save compact results
    if not df.empty:
        csv_filename = 'deepconf_online_results.csv'
        df.to_csv(csv_filename, index=False)
        print(f"\nResults saved to '{csv_filename}'")

    return df

if __name__ == "__main__":
    results_df = main()
