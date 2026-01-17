#!/usr/bin/env python3
"""
Compute the size of `probe_results` across JSON files under a root directory.

Measures:
- probe_results_count: len(probe_results) if it's a list (or len(keys) if dict)
- probe_results_bytes: UTF-8 byte length of a compact JSON serialization of probe_results
- disk_bytes: file size on disk
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class FileStats:
    path: str
    disk_bytes: int
    probe_results_type: str
    probe_results_count: int
    probe_results_bytes: int
    error: str


def _probe_results_metrics(probe_results: Any) -> tuple[str, int, int]:
    if probe_results is None:
        return "missing", 0, 0

    if isinstance(probe_results, list):
        count = len(probe_results)
        typ = "list"
    elif isinstance(probe_results, dict):
        count = len(probe_results)
        typ = "dict"
    else:
        count = 1
        typ = type(probe_results).__name__

    # Compact serialization for a consistent byte-size estimate.
    b = len(
        json.dumps(
            probe_results,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return typ, count, b


def _iter_json_files(root: Path, glob_pattern: str) -> list[Path]:
    # Using rglob keeps it simple and works well at this scale (~4k files).
    if any(ch in glob_pattern for ch in ["*", "?", "["]):
        return sorted(root.rglob(glob_pattern))
    return sorted(root.rglob("*.json"))


def compute(root: Path, glob_pattern: str) -> list[FileStats]:
    results: list[FileStats] = []
    for p in _iter_json_files(root, glob_pattern):
        if not p.is_file():
            continue
        disk_bytes = p.stat().st_size
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            probe_results = data.get("probe_results") if isinstance(data, dict) else None
            typ, cnt, by = _probe_results_metrics(probe_results)
            results.append(
                FileStats(
                    path=str(p),
                    disk_bytes=disk_bytes,
                    probe_results_type=typ,
                    probe_results_count=cnt,
                    probe_results_bytes=by,
                    error="",
                )
            )
        except Exception as e:  # noqa: BLE001 - want to keep going
            results.append(
                FileStats(
                    path=str(p),
                    disk_bytes=disk_bytes,
                    probe_results_type="error",
                    probe_results_count=0,
                    probe_results_bytes=0,
                    error=f"{type(e).__name__}: {e}",
                )
            )
    return results


def write_csv(rows: list[FileStats], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "path",
                "disk_bytes",
                "probe_results_type",
                "probe_results_count",
                "probe_results_bytes",
                "error",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.path,
                    r.disk_bytes,
                    r.probe_results_type,
                    r.probe_results_count,
                    r.probe_results_bytes,
                    r.error,
                ]
            )


def _fmt_bytes(n: int) -> str:
    # Human-readable (binary, base-1024)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}" if u != "B" else f"{int(x)} {u}"
        x /= 1024
    return f"{n} B"


def print_summary(rows: list[FileStats], top_k: int) -> None:
    n = len(rows)
    ok = [r for r in rows if not r.error]
    err = [r for r in rows if r.error]
    missing = [r for r in ok if r.probe_results_type == "missing"]

    total_disk = sum(r.disk_bytes for r in ok)
    total_pr_bytes = sum(r.probe_results_bytes for r in ok)
    total_pr_count = sum(r.probe_results_count for r in ok)

    print(f"files_total={n}")
    print(f"files_ok={len(ok)} files_error={len(err)} files_missing_probe_results={len(missing)}")
    print(f"disk_bytes_total={total_disk} ({_fmt_bytes(total_disk)})")
    print(f"probe_results_count_total={total_pr_count}")
    print(f"probe_results_bytes_total={total_pr_bytes} ({_fmt_bytes(total_pr_bytes)})")

    if ok:
        cnts = [r.probe_results_count for r in ok]
        bys = [r.probe_results_bytes for r in ok]
        print(f"probe_results_count_min={min(cnts)} max={max(cnts)} avg={sum(cnts)/len(cnts):.2f}")
        print(f"probe_results_bytes_min={min(bys)} max={max(bys)} avg={sum(bys)/len(bys):.2f}")

    if top_k > 0:
        biggest = sorted(ok, key=lambda r: r.probe_results_bytes, reverse=True)[:top_k]
        print(f"\nTop {top_k} by probe_results_bytes:")
        for r in biggest:
            print(f"- {_fmt_bytes(r.probe_results_bytes):>10} | cnt={r.probe_results_count:>4} | {r.path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/Users/wbchen/Workspace-Py/deepconf/outputs-online-real-percent-90-aime-2025-probe_result_raw"
        ),
        help="Directory to recursively scan for JSON files.",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="*.json",
        help='Glob pattern for files under root (default: "*.json").',
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional path to write per-file stats as CSV.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Print top-K largest files by probe_results_bytes (0 disables).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = compute(args.root, args.glob)
    print_summary(rows, args.top_k)
    if args.out_csv is not None:
        write_csv(rows, args.out_csv)
        print(f"\nwrote_csv={args.out_csv}")


if __name__ == "__main__":
    main()

