#!/usr/bin/env python3
"""
Monitor a folder for new .pkl files and upload them to S3.
Uses polling-based approach (no extra dependencies beyond boto3).

Usage:
    python watch_and_upload.py

    # Or run in background:
    nohup python watch_and_upload.py > watch_upload.log 2>&1 &
"""

import os
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


WATCH_DIR = "/home/ubuntu/projects/deepconf/outputs-final/Qwen3-32B/hmmt_feb_2025/traces"
S3_BUCKET = "deepconf"  # Change to your actual bucket name
S3_PREFIX = "deepconf/outputs-final/Qwen3-32B/hmmt_feb_2025/traces"
POLL_INTERVAL = 5  # seconds


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def upload_to_s3(local_path: str, s3_key: str) -> bool:
    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
    try:
        result = subprocess.run(
            ["aws", "s3", "cp", local_path, s3_uri],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            log(f"  Uploaded -> {s3_uri}")
            return True
        else:
            log(f"  FAILED: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        log(f"  FAILED: upload timed out for {local_path}")
        return False


def scan_pkl_files(watch_dir: str) -> dict[str, float]:
    """Return a dict of {relative_path: mtime} for all .pkl files."""
    files = {}
    for root, _, filenames in os.walk(watch_dir):
        for fname in filenames:
            if fname.endswith(".pkl"):
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, watch_dir)
                files[rel_path] = os.path.getmtime(full_path)
    return files


def initial_sync(watch_dir: str, known_files: dict[str, float]) -> int:
    """Sync all existing .pkl files to S3 using aws s3 sync."""
    log("Running initial sync of existing .pkl files...")
    s3_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}/"
    result = subprocess.run(
        ["aws", "s3", "sync", f"{watch_dir}/", s3_uri,
         "--exclude", "*", "--include", "*.pkl"],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode == 0:
        # Count uploaded files from output
        uploaded = result.stdout.count("upload:")
        log(f"Initial sync done. {uploaded} file(s) uploaded, {len(known_files)} total .pkl files tracked.")
        return uploaded
    else:
        log(f"Initial sync had errors: {result.stderr.strip()}")
        return 0


def watch(watch_dir: str, poll_interval: int):
    log(f"Watching: {watch_dir}")
    log(f"Target:   s3://{S3_BUCKET}/{S3_PREFIX}/")
    log(f"Polling every {poll_interval}s. Press Ctrl+C to stop.\n")

    # Build initial snapshot
    known_files = scan_pkl_files(watch_dir)
    initial_sync(watch_dir, known_files)
    log("")

    try:
        while True:
            time.sleep(poll_interval)
            current_files = scan_pkl_files(watch_dir)

            # Detect new or modified files
            for rel_path, mtime in current_files.items():
                if rel_path not in known_files or mtime > known_files[rel_path]:
                    full_path = os.path.join(watch_dir, rel_path)
                    s3_key = f"{S3_PREFIX}/{rel_path}"
                    action = "New" if rel_path not in known_files else "Modified"
                    log(f"{action} file: {rel_path}")
                    upload_to_s3(full_path, s3_key)

            known_files = current_files

    except KeyboardInterrupt:
        log("\nStopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch for new .pkl files and upload to S3")
    parser.add_argument("--dir", default=WATCH_DIR, help="Directory to watch")
    parser.add_argument("--bucket", default=S3_BUCKET, help="S3 bucket name")
    parser.add_argument("--prefix", default=S3_PREFIX, help="S3 key prefix")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL, help="Poll interval in seconds")
    args = parser.parse_args()

    WATCH_DIR = args.dir
    S3_BUCKET = args.bucket
    S3_PREFIX = args.prefix

    watch(args.dir, args.interval)
