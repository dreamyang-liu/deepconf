"""
Convert offline DeepConf data format to online format.

Offline format:
- Generates all traces at once
- Stores in 'all_traces' list
- Each trace has: stop_reason, text, token_ids, num_tokens, confs, extracted_answer, is_correct

Online format:
- Has warmup phase and final phase
- Stores in 'warmup_traces' and 'final_traces' separately  
- Each trace has additional fields: avg_conf, max_conf, min_conf, group_confs, answer_token_conf
- Simulates early stopping based on confidence threshold

Usage:
    python probe_src/convert_offline_to_online.py \
        --input outputs-offline/deepconf_simple_qid0_rid1_20251125_070529.pkl \
        --warmup_traces 16 \
        --confidence_percentile 90 \
        --window_size 2048
"""

import pickle
import numpy as np
import argparse
import os
from datetime import datetime
import multiprocessing as mp
import random
from functools import cache
from transformers import AutoTokenizer

MODEL_PATH = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

@cache
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    return tokenizer

def compute_least_grouped(confs, group_size):
    """Compute sliding window mean confidence"""
    if len(confs) < group_size:
        return [sum(confs) / len(confs)] if confs else [0]
    
    sliding_means = []
    for i in range(len(confs) - group_size + 1):
        window = confs[i:i + group_size]
        sliding_means.append(round(sum(window) / len(window), 3))
    return sliding_means


def convert_trace_to_online_format(trace, window_size=2048):
    """
    Convert a single offline trace to online format by adding windowed confidence metrics.
    
    Args:
        trace: Offline trace dictionary
        window_size: Window size for computing confidence metrics
        
    Returns:
        Online format trace dictionary
    """
    online_trace = trace.copy()
    
    # Get confidence scores
    confs = trace.get('confs', [])
    
    # Calculate sliding window confidences
    group_confs = compute_least_grouped(confs, group_size=window_size)
    min_conf = min(group_confs) if group_confs else 0
    
    # Add online-specific fields
    online_trace['group_confs'] = group_confs
    online_trace['min_conf'] = min_conf
    
    # Calculate answer-level confidence metrics (simplified version)
    # In full online version, these are computed by locating answer tokens
    # For conversion, we use approximations based on available data
    if confs:
        online_trace['avg_conf'] = round(np.mean(confs), 3)
        online_trace['max_conf'] = round(np.max(confs), 3)
        # min_conf already computed from sliding windows
    else:
        online_trace['avg_conf'] = 0
        online_trace['max_conf'] = 0
    
    # Add empty answer_token_conf (would require tokenizer to compute properly)
    online_trace['answer_token_conf'] = []
    
    return online_trace


def convert_offline_to_online(offline_data, warmup_traces=16, total_traces=512, confidence_percentile=90, window_size=2048):
    """
    Convert offline DeepConf data to online format.
    
    Args:
        offline_data: Dictionary with offline format data
        warmup_traces: Number of traces to use as warmup (default: 16)
        total_traces: Total number of traces (default: 512)
        confidence_percentile: Percentile for confidence threshold (default: 90)
        window_size: Window size for confidence calculation (default: 2048)
        
    Returns:
        Dictionary with online format data
    """
    all_traces = offline_data['all_traces']
    sampled_traces = random.sample(all_traces, total_traces)

    if len(sampled_traces) < warmup_traces:
        raise ValueError(f"Not enough traces ({len(sampled_traces)}) for warmup ({warmup_traces})")

    # Convert all traces to online format

    # Use parallel processing to convert traces
    with mp.Pool(processes=32) as pool:
        converted_traces = pool.starmap(
            convert_trace_to_online_format,
            [(trace, window_size) for trace in sampled_traces]
        )

    # Split into warmup and final traces
    warmup_traces_data = converted_traces[:warmup_traces]
    final_traces_data = converted_traces[warmup_traces:]
    
    # Calculate confidence threshold from warmup traces
    warmup_min_confs = [trace['min_conf'] for trace in warmup_traces_data]
    conf_bar = float(np.percentile(warmup_min_confs, confidence_percentile))
    
    # Simulate early stopping for final traces below threshold
    for trace in final_traces_data:
        if trace['min_conf'] < conf_bar:
            # Find the position where confidence drops below the threshold
            early_stop_idx = None
            for i, group_conf in enumerate(trace['group_confs']):
                if group_conf < conf_bar:
                    early_stop_idx = i
                    break

            if early_stop_idx is not None:
                # Early stop position is the index in group_confs plus window_size - 1
                # (since each group_conf represents the end of a window)
                stop_position = early_stop_idx + window_size - 1

                # Truncate tokens and confidences
                if 'token_ids' in trace and len(trace['token_ids']) > stop_position:
                    trace['token_ids'] = trace['token_ids'][:stop_position+1]

                if 'confs' in trace and len(trace['confs']) > stop_position:
                    trace['confs'] = trace['confs'][:stop_position+1]

                # Update the text to match truncated tokens
                if 'text' in trace and trace.get('token_ids'):
                    tokenizer = get_tokenizer()
                    trace['text'] = tokenizer.decode(trace['token_ids'])

                # Update num_tokens
                trace['num_tokens'] = len(trace.get('token_ids', []))

                # Update group_confs
                trace['group_confs'] = trace['group_confs'][:early_stop_idx+1]

                # Set stop reason
            trace['stop_reason'] = 'gconf_threshold'
    
    # Calculate token statistics
    warmup_tokens = sum(trace['num_tokens'] for trace in warmup_traces_data)
    final_tokens = sum(trace['num_tokens'] for trace in final_traces_data)
    total_tokens = warmup_tokens + final_tokens
    
    # Create online format result
    online_data = {
        "question_id": offline_data.get('question_id'),
        "run_id": offline_data.get('run_id'),
        "question": offline_data.get('question'),
        "ground_truth": offline_data.get('ground_truth'),
        "conf_bar": conf_bar,
        "warmup_traces": warmup_traces_data,
        "final_traces": final_traces_data,
        "voted_answer": offline_data.get('voted_answer'),
        "is_voted_correct": offline_data.get('is_voted_correct'),
        "token_stats": {
            "warmup_tokens": warmup_tokens,
            "final_tokens": final_tokens,
            "total_tokens": total_tokens,
            "warmup_traces_count": len(warmup_traces_data),
            "final_traces_count": len(final_traces_data),
            "avg_tokens_per_warmup_trace": warmup_tokens / len(warmup_traces_data) if warmup_traces_data else 0,
            "avg_tokens_per_final_trace": final_tokens / len(final_traces_data) if final_traces_data else 0,
        },
        "timing_stats": offline_data.get('timing_stats', {}),
        "config": {
            **offline_data.get('config', {}),
            "warmup_traces": warmup_traces,
            "confidence_percentile": confidence_percentile,
            "window_size": window_size,
        },
        "timestamp": datetime.now().isoformat(),
        "converted_from_offline": True,
    }
    
    return online_data


def main():
    parser = argparse.ArgumentParser(description='Convert offline DeepConf data to online format')
    parser.add_argument('--input', type=str, required=True, help='Input offline pickle file path')
    parser.add_argument('--output', type=str, help='Output online pickle file path (optional, defaults to same name with _online suffix)')
    parser.add_argument('--warmup_traces', type=int, default=16, help='Number of warmup traces (default: 16)')
    parser.add_argument('--confidence_percentile', type=int, default=90, help='Confidence percentile for threshold (default: 90)')
    parser.add_argument('--window_size', type=int, default=2048, help='Window size for confidence calculation (default: 2048)')
    
    args = parser.parse_args()
    
    # Load offline data
    print(f"Loading offline data from {args.input}...")
    with open(args.input, 'rb') as f:
        offline_data = pickle.load(f)
    
    # Print offline data info
    print(f"Offline data loaded:")
    print(f"  Question ID: {offline_data.get('question_id')}")
    print(f"  Run ID: {offline_data.get('run_id')}")
    print(f"  Total traces: {len(offline_data.get('all_traces', []))}")
    print(f"  Total tokens: {offline_data.get('token_stats', {}).get('total_tokens', 0)}")
    
    # Convert to online format
    print(f"\nConverting to online format...")
    print(f"  Warmup traces: {args.warmup_traces}")
    print(f"  Confidence percentile: {args.confidence_percentile}")
    print(f"  Window size: {args.window_size}")
    
    online_data = convert_offline_to_online(
        offline_data,
        warmup_traces=args.warmup_traces,
        confidence_percentile=args.confidence_percentile,
        window_size=args.window_size
    )
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Create output path with more descriptive filename including parameters
        input_filename = os.path.basename(args.input)
        # Extract question ID and run ID from filename
        import re
        qid_match = re.search(r'qid(\d+)', input_filename)
        rid_match = re.search(r'rid(\d+)', input_filename)
        date_match = re.search(r'(\d{8}_\d{6})', input_filename)

        qid = qid_match.group(1) if qid_match else "unknown"
        rid = rid_match.group(1) if rid_match else "unknown"
        old_date = date_match.group(1) if date_match else ""

        # Create new filename with parameters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"deepconf_simple_qid{qid}_rid{rid}_{old_date}_online_w{args.warmup_traces}_p{args.confidence_percentile}_{timestamp}.pkl"

        # Create directory structure
        output_dir = os.path.join('outputs-online', f"qid{qid}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
    
    # Save online data
    print(f"\nSaving online data to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(online_data, f)
    
    # Print online data info
    print(f"\nOnline data saved:")
    print(f"  Confidence bar: {online_data['conf_bar']:.3f}")
    print(f"  Warmup traces: {len(online_data['warmup_traces'])}")
    print(f"  Final traces: {len(online_data['final_traces'])}")
    print(f"  Warmup tokens: {online_data['token_stats']['warmup_tokens']}")
    print(f"  Final tokens: {online_data['token_stats']['final_tokens']}")
    print(f"  Total tokens: {online_data['token_stats']['total_tokens']}")
    
    # Count early stopped traces
    early_stopped = sum(1 for trace in online_data['final_traces'] if trace.get('stop_reason') == 'gconf_threshold')
    print(f"  Early stopped traces: {early_stopped}/{len(online_data['final_traces'])}")
    
    print(f"\nConversion completed successfully!")


if __name__ == "__main__":
    main()
