#!/usr/bin/env python3
"""
Convert raw dp_probe JSON files into a tidy per-trace DataFrame (per qid).

Key behavior (propagation):
1) Missing probe files (typically because a run finished early):
   - Controlled by: --propagate-missing-files / --no-propagate-missing-files
   - Creates a full grid over all configured probe checkpoints, and propagates the
     latest available probe information forward for each trace_idx.

2) Missing trace rows within a probe file (typically because the probed answer
   does not follow the expected format):
   - Controlled by: --propagate-missing-traces / --no-propagate-missing-traces
   - For any missing (trace_idx, prob_token) rows where the file exists but the
     trace is absent, propagates the latest available information for the same
     trace_idx forward.

Output columns for tracking propagation:
  - observed: True if the row was directly observed in JSON
  - is_propagated: True if the row was filled (either file or trace missing)
  - is_propagated_file: True if filled due to missing probe file
  - is_propagated_trace: True if filled due to missing trace row in existing file
  - source_prob_token: The prob_token from which the values were propagated

Usage examples:
  # Process all qids with both propagation modes enabled (default)
  python convert_probe_json_to_df_per_qid.py

  # Process one qid, only propagate missing files (not missing traces)
  python convert_probe_json_to_df_per_qid.py --qid 0 --no-propagate-missing-traces

  # Process all qids, disable all propagation (grid with NaN for missing)
  python convert_probe_json_to_df_per_qid.py --no-propagate-missing-files --no-propagate-missing-traces
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


# Filename format (raw):
# deepconf_simple_qid{qid}_rid{rid}_{base_ts}_online_w16_p90_{run_id}_dp_probe.json
FILENAME_RE = re.compile(
    r"deepconf_simple_qid(?P<qid>\d+)_rid(?P<rid>\d+)_(?P<base_ts>\d{8}_\d{6})_online_w(?P<w>\d+)_p(?P<p>\d+)_(?P<run_id>\d{8}_\d{6})_dp_probe\.json$"
)


@dataclass(frozen=True)
class ProbeFileRef:
    qid: int
    rid: int
    base_ts: str
    run_id: str
    prob_token: int
    path: Path


def parse_filename(filename: str) -> Optional[tuple[int, int, str, str]]:
    """Extract qid, rid, base_ts, run_id from filename."""
    m = FILENAME_RE.search(filename)
    if not m:
        return None
    return int(m.group("qid")), int(m.group("rid")), m.group("base_ts"), m.group("run_id")


def get_token_dirs_and_positions(probe_dir: Path) -> tuple[list[str], list[int]]:
    """Extract probe directories and their token positions.

    Handles both:
    - old format: '4096'
    - new format: '4096-probe'
    """
    token_dirs: list[str] = []
    token_positions: list[int] = []

    for d in os.listdir(probe_dir):
        dir_path = probe_dir / d
        if not dir_path.is_dir():
            continue

        if d.isdigit():
            token_dirs.append(d)
            token_positions.append(int(d))
        elif d.endswith("-probe"):
            num_part = d[: -len("-probe")]
            if num_part.isdigit():
                token_dirs.append(d)
                token_positions.append(int(num_part))

    sorted_pairs = sorted(zip(token_positions, token_dirs))
    token_positions_sorted = [p for p, _ in sorted_pairs]
    token_dirs_sorted = [d for _, d in sorted_pairs]
    return token_dirs_sorted, token_positions_sorted


def normalize_answer(a: Any) -> str:
    if a is None:
        return "<EMPTY>"
    s = str(a).strip()
    return s if s != "" else "<EMPTY>"


def scan_probe_files(
    probe_dir: Path,
    token_dirs: list[str],
    qids: Optional[set[int]] = None,
) -> dict[int, list[ProbeFileRef]]:
    """Scan the raw probe directory and group files by qid."""
    by_qid: dict[int, list[ProbeFileRef]] = {}
    skipped = 0
    total = 0

    for token_dir in token_dirs:
        token_pos = int(token_dir.replace("-probe", ""))
        dir_path = probe_dir / token_dir

        for filename in os.listdir(dir_path):
            if not filename.endswith(".json") or filename == "summary_results.json":
                continue

            parsed = parse_filename(filename)
            if parsed is None:
                skipped += 1
                continue

            qid, rid, base_ts, run_id = parsed
            if qids is not None and qid not in qids:
                continue

            ref = ProbeFileRef(
                qid=qid,
                rid=rid,
                base_ts=base_ts,
                run_id=run_id,
                prob_token=token_pos,
                path=dir_path / filename,
            )
            by_qid.setdefault(qid, []).append(ref)
            total += 1

    LOG.info("Scanned %d files (skipped %d non-matching)", total, skipped)
    return by_qid


def load_probe_file_rows(
    ref: ProbeFileRef,
    include_warmup: bool,
) -> list[dict[str, Any]]:
    """Load one JSON file and return tidy per-trace rows."""
    with ref.path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    pr = obj.get("probe_results") or []
    pr = sorted(pr, key=lambda r: r.get("idx", -1))
    if not include_warmup:
        pr = [r for r in pr if not r.get("is_warmup", False)]

    rows: list[dict[str, Any]] = []
    for r in pr:
        rows.append(
            {
                "qid": ref.qid,
                "run_id": ref.run_id,
                "rid": ref.rid,
                "base_ts": ref.base_ts,
                "prob_token": ref.prob_token,
                "trace_idx": r.get("idx", None),
                "answer": normalize_answer(r.get("answer", "")),
                "confidence": r.get("confidence", None),
                "token_usage": r.get("token_usage", None),
                "is_warmup": bool(r.get("is_warmup", False)),
                "is_deep_conf_truncated": bool(r.get("is_deep_conf_truncated", False)),
                "is_naturally_stopped": bool(r.get("is_naturally_stopped", False)),
                "is_out_of_budget": bool(r.get("is_out_of_budget", False)),
            }
        )
    return rows


def _apply_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply memory-friendly dtypes (best-effort)."""
    if df.empty:
        return df
    df["qid"] = df["qid"].astype("int16")
    df["prob_token"] = df["prob_token"].astype("int32")
    df["trace_idx"] = df["trace_idx"].astype("int16")
    df["token_usage"] = pd.to_numeric(df["token_usage"], errors="coerce").astype("Int32")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").astype("float32")
    df["run_id"] = df["run_id"].astype("category")
    return df


def build_filled_view(
    df_raw: pd.DataFrame,
    token_positions: list[int],
    *,
    expected_n_traces: Optional[int] = 512,
    propagate_missing_files: bool = True,
    propagate_missing_traces: bool = True,
    fill_leading: bool = False,
) -> pd.DataFrame:
    """Create a fully-populated grid and optionally propagate missing values.

    This function builds a complete (trace_idx × prob_token) grid for each (qid, run_id)
    and propagates values from the latest available probe checkpoint for each trace.

    Args:
        df_raw: Observed data with one row per (qid, run_id, trace_idx, prob_token).
        token_positions: All probe token checkpoint positions to include in the grid.
        expected_n_traces: If set (e.g., 512), enforce a fixed trace_idx grid [0..N-1].
            Use None or 0 to use the observed union of trace indices.
        propagate_missing_files: If True, propagate values when an entire probe file
            is missing (i.e., a run finished before reaching a later checkpoint).
            Fills from the latest available probe for each trace_idx.
            If False, rows for missing files are removed from the output (since they
            would otherwise have NaN values and be not useful).
        propagate_missing_traces: If True, propagate values when individual trace rows
            are missing within an existing probe file (e.g., due to format parsing issues).
            Fills from the latest available probe for the same trace_idx.
        fill_leading: If True, also back-fill leading missing values (i.e., if a trace
            only appears at later checkpoints, fill earlier checkpoints from the first
            observed value). Default is forward-fill only.

    Returns:
        DataFrame with the full grid, including:
            - observed: True if the row was directly observed in JSON
            - is_propagated: True if the row was filled (either file or trace missing)
            - is_propagated_file: True if filled due to missing probe file
            - is_propagated_trace: True if filled due to missing trace row in existing file
            - source_prob_token: The prob_token from which the values were propagated
    """
    if df_raw.empty:
        return df_raw.copy()

    token_positions_sorted = list(sorted(token_positions))

    group_keys = ["qid", "run_id"]
    value_cols = [
        "rid",
        "base_ts",
        "answer",
        "confidence",
        "token_usage",
        "is_warmup",
        "is_deep_conf_truncated",
        "is_naturally_stopped",
        "is_out_of_budget",
    ]

    filled_parts: list[pd.DataFrame] = []

    for (qid, run_id), g in df_raw.groupby(group_keys, observed=True, sort=False):
        # --- Step 1: Build the full (trace_idx, prob_token) grid ---
        if expected_n_traces is not None and expected_n_traces > 0:
            trace_ids = np.arange(int(expected_n_traces), dtype=np.int16)
        else:
            trace_ids = np.array(sorted(g["trace_idx"].unique()), dtype=np.int16)

        full_index = pd.MultiIndex.from_product(
            [trace_ids, token_positions_sorted], names=["trace_idx", "prob_token"]
        )

        # --- Step 2: Reindex observed data to full grid ---
        g_idx = g.set_index(["trace_idx", "prob_token"], drop=False)
        orig_index = set(g_idx.index)

        base = g_idx[["trace_idx", "prob_token"] + value_cols].set_index(
            ["trace_idx", "prob_token"]
        )
        base = base.reindex(full_index)

        # --- Step 3: Classify each grid cell ---
        # observed_mask: True if (trace_idx, prob_token) was in the raw data
        observed_mask = pd.Series(
            [idx in orig_index for idx in full_index],
            index=full_index,
            dtype=bool,
        )

        # Identify which prob_token checkpoints have at least one trace (file exists)
        prob_tokens_with_file = set(g["prob_token"].unique())
        file_exists_for_token = pd.Series(
            [pt in prob_tokens_with_file for pt in full_index.get_level_values("prob_token")],
            index=full_index,
            dtype=bool,
        )

        # file_missing_mask: Cell is missing because the entire probe file is absent
        file_missing_mask = ~observed_mask & ~file_exists_for_token

        # trace_missing_mask: Cell is missing but the probe file exists (trace row missing)
        trace_missing_mask = ~observed_mask & file_exists_for_token

        # --- Step 4: Determine which cells should be filled ---
        should_fill = pd.Series(False, index=full_index, dtype=bool)
        if propagate_missing_files:
            should_fill = should_fill | file_missing_mask
        if propagate_missing_traces:
            should_fill = should_fill | trace_missing_mask

        # --- Step 5: Forward-fill values within each trace_idx ---
        # (and optionally back-fill if fill_leading=True)
        base_filled = base.groupby(level=0).ffill()
        if fill_leading:
            base_filled = base_filled.groupby(level=0).bfill()

        # Track source: which prob_token did the value come from?
        src = pd.Series(
            np.where(observed_mask, full_index.get_level_values("prob_token"), np.nan),
            index=full_index,
            dtype=float,
        )
        src_filled = src.groupby(level=0).ffill()
        if fill_leading:
            src_filled = src_filled.groupby(level=0).bfill()

        # --- Step 6: Apply propagation selectively ---
        # Only use filled values for cells that should_fill; keep observed as-is
        for col in value_cols:
            base[col] = np.where(
                observed_mask | should_fill,
                base_filled[col],
                np.nan,
            )

        src_final = np.where(
            observed_mask | should_fill,
            src_filled,
            np.nan,
        )

        # --- Step 7: Assemble output DataFrame ---
        out = base.reset_index()
        out["qid"] = np.int16(qid)
        out["run_id"] = run_id
        out["observed"] = observed_mask.values
        out["is_propagated_file"] = (file_missing_mask & should_fill).values
        out["is_propagated_trace"] = (trace_missing_mask & should_fill).values
        out["is_propagated"] = out["is_propagated_file"] | out["is_propagated_trace"]
        out["source_prob_token"] = src_final

        # Track which rows are for missing files (for optional filtering later)
        out["_file_missing"] = file_missing_mask.values

        filled_parts.append(out)

    # --- Concatenate and finalize ---
    df_filled = pd.concat(filled_parts, ignore_index=True)

    # --- Step 8: Remove rows for missing files if not propagating them ---
    # When propagate_missing_files=False, those rows have NaN values and are not useful.
    if not propagate_missing_files:
        n_before = len(df_filled)
        df_filled = df_filled[~df_filled["_file_missing"]].copy()
        n_removed = n_before - len(df_filled)
        if n_removed > 0:
            LOG.debug("Removed %d rows for missing files (not propagated)", n_removed)

    # Drop the internal helper column
    df_filled = df_filled.drop(columns=["_file_missing"], errors="ignore")

    # Apply dtypes
    df_filled = _apply_dtypes(df_filled)
    df_filled["source_prob_token"] = (
        pd.to_numeric(df_filled["source_prob_token"], errors="coerce").astype("Int32")
    )

    # --- Step 9: Ensure boolean columns are proper bool dtype ---
    # np.where with np.nan can convert bools to object dtype with mixed types.
    # Convert via numeric first (True->1, False->0, NaN->NaN) then to nullable boolean.
    bool_cols = ["is_warmup", "is_deep_conf_truncated", "is_naturally_stopped", "is_out_of_budget"]
    for col in bool_cols:
        if col in df_filled.columns:
            # First convert to numeric to handle mixed object dtype
            numeric_col = pd.to_numeric(df_filled[col], errors="coerce")
            # Then convert to nullable boolean (1->True, 0->False, NaN->NA)
            df_filled[col] = numeric_col.astype("boolean")

    df_filled = (
        df_filled.sort_values(["qid", "run_id", "trace_idx", "prob_token"], kind="stable")
        .reset_index(drop=True)
    )
    return df_filled


def load_qid_dataframes(
    refs: list[ProbeFileRef],
    token_positions: list[int],
    *,
    include_warmup: bool,
    expected_n_traces: Optional[int],
    propagate_missing_files: bool = True,
    propagate_missing_traces: bool = True,
    fill_leading: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load df_raw + df_filled for a single qid.

    Args:
        refs: List of ProbeFileRef pointing to the JSON files for this qid.
        token_positions: All probe token checkpoint positions.
        include_warmup: If True, include warmup traces (is_warmup=True).
        expected_n_traces: If set, enforce trace_idx grid [0..N-1]. Use None to disable.
        propagate_missing_files: If True, propagate values when entire probe files
            are missing (run finished early).
        propagate_missing_traces: If True, propagate values when individual trace
            rows are missing within an existing probe file.
        fill_leading: If True, also back-fill leading missing values.

    Returns:
        (df_raw, df_filled): Raw observed data and the filled/propagated view.
    """
    rows: list[dict[str, Any]] = []
    for ref in sorted(refs, key=lambda r: (r.run_id, r.prob_token, r.path.name)):
        rows.extend(load_probe_file_rows(ref, include_warmup=include_warmup))

    df_raw = pd.DataFrame(rows)
    df_raw = _apply_dtypes(df_raw)
    df_filled = build_filled_view(
        df_raw,
        token_positions,
        expected_n_traces=expected_n_traces,
        propagate_missing_files=propagate_missing_files,
        propagate_missing_traces=propagate_missing_traces,
        fill_leading=fill_leading,
    )
    return df_raw, df_filled


def _can_write_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:
        return False


def write_df(df: pd.DataFrame, out_path: Path, fmt: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(out_path, index=False)
    elif fmt == "csv":
        df.to_csv(out_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--probe-dir",
        type=Path,
        default=Path(
            "/Users/wbchen/Workspace-Py/deepconf/outputs-online-real-percent-90-aime-2025-probe_result_raw"
        ),
        help="Root directory containing token probe subdirectories.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/wbchen/Workspace-Py/deepconf/eval/df_per_qid"),
        help="Output directory for per-qid dataframes.",
    )
    p.add_argument(
        "--qid",
        type=int,
        action="append",
        default=None,
        help="Only process this qid (repeatable). If omitted, process all qids found.",
    )
    p.add_argument(
        "--include-warmup",
        action="store_true",
        default=True,
        help="Include warmup traces (is_warmup=True). Default: True.",
    )
    p.add_argument(
        "--no-include-warmup",
        dest="include_warmup",
        action="store_false",
        help="Exclude warmup traces. Note: this creates empty rows for trace_idx 0-15 "
        "unless --expected-n-traces is adjusted accordingly.",
    )
    p.add_argument(
        "--expected-n-traces",
        type=int,
        default=512,
        help="If set, enforce trace_idx grid [0..N-1]. Use 0 to disable and use observed union.",
    )
    # --- Propagation control ---
    p.add_argument(
        "--propagate-missing-files",
        action="store_true",
        default=True,
        help="Propagate when entire probe files are missing (run finished early). Default: True.",
    )
    p.add_argument(
        "--no-propagate-missing-files",
        dest="propagate_missing_files",
        action="store_false",
        help="Disable propagation for missing probe files.",
    )
    p.add_argument(
        "--propagate-missing-traces",
        action="store_true",
        default=True,
        help="Propagate when individual trace rows are missing within a file (format issues). Default: True.",
    )
    p.add_argument(
        "--no-propagate-missing-traces",
        dest="propagate_missing_traces",
        action="store_false",
        help="Disable propagation for missing trace rows.",
    )
    p.add_argument(
        "--fill-leading",
        action="store_true",
        help="Also back-fill leading missing values (default: forward-fill only).",
    )
    # --- Output options ---
    p.add_argument(
        "--write-raw",
        action="store_true",
        help="Also write the observed-only df_raw per qid.",
    )
    p.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="csv",
        help="Output format for dataframes.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    token_dirs, token_positions = get_token_dirs_and_positions(args.probe_dir)
    if not token_positions:
        raise SystemExit(f"No token probe subdirectories found under {args.probe_dir}")
    LOG.info("Found %d probe points: %s", len(token_positions), token_positions)

    qid_filter = set(args.qid) if args.qid is not None else None
    by_qid = scan_probe_files(args.probe_dir, token_dirs, qids=qid_filter)
    qids = sorted(by_qid.keys())
    if not qids:
        raise SystemExit("No matching probe JSON files found.")
    LOG.info("Processing %d qids: %s", len(qids), qids)

    # Pick a default format if parquet isn't available.
    fmt = args.format
    if fmt == "parquet" and not _can_write_parquet():
        LOG.warning("pyarrow not available; falling back to CSV output")
        fmt = "csv"

    expected_n_traces: Optional[int]
    expected_n_traces = None if args.expected_n_traces == 0 else int(args.expected_n_traces)

    LOG.info(
        "Propagation settings: missing_files=%s, missing_traces=%s, fill_leading=%s",
        args.propagate_missing_files,
        args.propagate_missing_traces,
        args.fill_leading,
    )

    for qid in qids:
        LOG.info("qid=%s: loading %d files", qid, len(by_qid[qid]))
        df_raw, df_filled = load_qid_dataframes(
            by_qid[qid],
            token_positions,
            include_warmup=args.include_warmup,
            expected_n_traces=expected_n_traces,
            propagate_missing_files=args.propagate_missing_files,
            propagate_missing_traces=args.propagate_missing_traces,
            fill_leading=args.fill_leading,
        )

        # Write outputs
        stem = f"qid{qid:02d}"
        out_filled = args.out_dir / "filled" / f"{stem}.{fmt}"
        write_df(df_filled, out_filled, fmt=fmt)
        LOG.info("qid=%s: wrote filled df: %s (rows=%d)", qid, out_filled, len(df_filled))

        if args.write_raw:
            out_raw = args.out_dir / "raw" / f"{stem}.{fmt}"
            write_df(df_raw, out_raw, fmt=fmt)
            LOG.info("qid=%s: wrote raw df: %s (rows=%d)", qid, out_raw, len(df_raw))


if __name__ == "__main__":
    main()

