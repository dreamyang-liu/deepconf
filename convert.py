"""
Parallel conversion script for converting offline DeepConf data to online format.

This script finds all offline pickle files and converts them in parallel using multiprocessing.

Usage:
    python convert.py --input_dir outputs-offline --output_dir outputs-online --workers 8
"""

import argparse
import os
import glob
import multiprocessing as mp
from pathlib import Path
import sys

# Add probe_src to path to import the converter
sys.path.insert(0, 'probe_src')
from convert_offline_to_online import convert_offline_to_online
import pickle


def convert_single_file(args_tuple):
    """
    Convert a single offline file to online format.
    
    Args:
        args_tuple: Tuple of (input_path, output_path, warmup_traces, confidence_percentile, window_size)
        
    Returns:
        Tuple of (input_path, success, message)
    """
    input_path, output_path, warmup_traces, confidence_percentile, window_size = args_tuple
    print(args_tuple)
    
    try:
        # Load offline data
        with open(input_path, 'rb') as f:
            offline_data = pickle.load(f)
        
        # Convert to online format
        online_data = convert_offline_to_online(
            offline_data,
            warmup_traces=warmup_traces,
            confidence_percentile=confidence_percentile,
            window_size=window_size
        )
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save online data
        with open(output_path, 'wb') as f:
            pickle.dump(online_data, f)
        
        qid = offline_data.get('question_id', 'unknown')
        rid = offline_data.get('run_id', 'unknown')
        early_stopped = sum(1 for t in online_data['final_traces'] if t.get('stop_reason') == 'gconf_threshold')
        total_final = len(online_data['final_traces'])
        
        message = f"QID {qid} RID {rid}: {early_stopped}/{total_final} early stopped"
        return (input_path, True, message)
        
    except Exception as e:
        e.with_traceback()
        return (input_path, False, str(e))

def main():
    parser = argparse.ArgumentParser(description='Parallel conversion of offline DeepConf data to online format')
    parser.add_argument('--input_dir', type=str, default='outputs-offline', 
                        help='Input directory containing offline pickle files (default: outputs-offline)')
    parser.add_argument('--output_dir', type=str, default='outputs-online',
                        help='Output directory for online pickle files (default: outputs-online)')
    parser.add_argument('--warmup_traces', type=int, default=16,
                        help='Number of warmup traces (default: 16)')
    parser.add_argument('--confidence_percentile', type=int, default=90,
                        help='Confidence percentile for threshold (default: 90)')
    parser.add_argument('--window_size', type=int, default=2048,
                        help='Window size for confidence calculation (default: 2048)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--pattern', type=str, default='*.pkl',
                        help='File pattern to match (default: *.pkl)')
    parser.add_argument('--qid', type=int, default=None,
                        help='Convert only specific question ID (optional)')
    
    args = parser.parse_args()
    
    # Find all input files
    input_pattern = os.path.join(args.input_dir, args.pattern)
    input_files = sorted(glob.glob(input_pattern))
    
    if not input_files:
        print(f"No files found matching pattern: {input_pattern}")
        return
    
    # Filter by qid if specified
    if args.qid is not None:
        input_files = [f for f in input_files if f'qid{args.qid}_' in f]
        if not input_files:
            print(f"No files found for question ID {args.qid}")
            return
    
    print(f"Found {len(input_files)} files to convert")
    print(f"Using {args.workers} parallel workers")
    print(f"Configuration: warmup={args.warmup_traces}, percentile={args.confidence_percentile}, window={args.window_size}")
    print()
    
    # Prepare conversion arguments
    conversion_args = []
    for input_path in input_files:
        # Create output path by replacing input_dir with output_dir and 'simple_' with ''
        input_filename = os.path.basename(input_path)
        output_filename = input_filename.replace('deepconf_simple_', 'deepconf_')
        # Create output path with more descriptive filename including parameters
        input_filename = os.path.basename(input_path)
        # Extract question ID and run ID from filename
        import re
        qid_match = re.search(r'qid(\d+)', input_filename)
        rid_match = re.search(r'rid(\d+)', input_filename)
        date_match = re.search(r'(\d{8}_\d{6})', input_filename)
        qid = qid_match.group(1) if qid_match else "unknown"
        rid = rid_match.group(1) if rid_match else "unknown"
        old_date = date_match.group(1) if date_match else ""
        # Create new filename with parameters
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"deepconf_simple_qid{qid}_rid{rid}_{old_date}_online_w{args.warmup_traces}_p{args.confidence_percentile}_{timestamp}.pkl"
        # Create directory structure
        output_dir = os.path.join(args.output_dir, f"qid{qid}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        conversion_args.append((
            input_path,
            output_path,
            args.warmup_traces,
            args.confidence_percentile,
            args.window_size
        ))
    
    # Run conversions in parallel
    if args.workers == 1:
        # Single-threaded for debugging
        results = []
        for i, conv_args in enumerate(conversion_args, 1):
            print(f"[{i}/{len(conversion_args)}] Converting {os.path.basename(conv_args[0])}...")
            result = convert_single_file(conv_args)
            results.append(result)
            if result[1]:
                print(f"  ✓ {result[2]}")
            else:
                print(f"  ✗ Error: {result[2]}")
    else:
        # Multi-threaded
        with mp.Pool(processes=args.workers) as pool:
            results = []
            for i, result in enumerate(pool.imap(convert_single_file, conversion_args), 1):
                results.append(result)
                input_file = os.path.basename(result[0])
                if result[1]:
                    print(f"[{i}/{len(conversion_args)}] ✓ {input_file}: {result[2]}")
                else:
                    print(f"[{i}/{len(conversion_args)}] ✗ {input_file}: Error: {result[2]}")
    
    # Print summary
    print()
    print("=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for r in results if r[1])
    failed = len(results) - successful
    
    print(f"Total files: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed conversions:")
        for result in results:
            if not result[1]:
                print(f"  - {os.path.basename(result[0])}: {result[2]}")
    
    print()
    print(f"Output directory: {args.output_dir}")
    print("Conversion completed!")


if __name__ == "__main__":
    main()
