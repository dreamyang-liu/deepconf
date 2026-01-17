#!/usr/bin/env python3
"""
Compute DeepConf online performance (accuracy vs. tokens) from logged probe CSVs.

Assumptions (matches your clarified setup and Algorithm 2 in the paper):
  - The CSVs were produced from DeepConf *online* runs (so token usage already reflects
    any early-termination / truncation).
  - For evaluation, we DO NOT use "probed answers" at intermediate checkpoints.
    Instead, we use each trace's *final answer* (as recorded in the logs) and perform
    confidence-weighted majority voting on:
      - warmup traces (is_warmup=True) AND
      - naturally-stopped traces (is_naturally_stopped=True)
    (optionally excluding truncated/out-of-budget traces for voting).
  - To simulate different token budgets B (number of traces), we take the first B traces
    in generation order, approximated by ascending trace_idx (0..511).

Outputs:
  - Per-(qid, run_id, budget) tokens + correctness
  - Aggregated curve: mean tokens and mean accuracy for each budget

Paper reference:
  https://arxiv.org/pdf/2508.15260
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project root is importable (so `import helper` works when running from anywhere)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Repo-local fuzzy answer equality (used in notebooks)
from helper import equal_func as math_equal  # noqa: E402

LOG = logging.getLogger(__name__)


FILENAME_RE = re.compile(
    r"deepconf_simple_qid(?P<qid>\d+)_rid(?P<rid>\d+)_(?P<base_ts>\d{8}_\d{6})_online_w(?P<w>\d+)_p(?P<p>\d+)_(?P<run_id>\d{8}_\d{6})_dp_probe\.json$"
)


@dataclass(frozen=True)
class TraceFinal:
    """Collapsed per-trace record used for voting/cost."""

    qid: int
    run_id: str
    trace_idx: int
    final_answer: str
    final_token_usage: int
    # Weight used for confidence-weighted voting / filtering.
    # For LGC-style confidence, this is typically min group confidence across the trace;
    # we approximate it using min over available probe checkpoints in the CSV.
    trace_confidence: float
    # Flags (constant per trace in properly-formed logs)
    is_warmup: bool
    is_naturally_stopped: bool
    is_deep_conf_truncated: bool
    is_out_of_budget: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--df-dir",
        type=Path,
        default=Path("/Users/wbchen/Workspace-Py/deepconf/eval/df_per_qid/filled"),
        help="Directory containing per-qid CSVs (e.g., qid00.csv, qid12.csv).",
    )
    p.add_argument(
        "--probe-dir",
        type=Path,
        default=Path(
            "/Users/wbchen/Workspace-Py/deepconf/outputs-online-real-percent-90-aime-2025-probe_result_raw"
        ),
        help="Raw dp_probe directory used only to fetch ground_truth by qid.",
    )
    p.add_argument(
        "--budgets",
        type=str,
        default="32,64,128,256,512",
        help="Comma-separated list of trace budgets B.",
    )
    p.add_argument(
        "--keep-top",
        type=float,
        default=0.90,
        help="Confidence filtering keep ratio eta in (0,1]. e.g. 0.90 for DeepConf-high.",
    )
    p.add_argument(
        "--exclude-truncated",
        action="store_true",
        default=True,
        help="Exclude is_deep_conf_truncated traces from voting. Default: True.",
    )
    p.add_argument(
        "--include-truncated",
        dest="exclude_truncated",
        action="store_false",
        help="Include truncated traces in voting.",
    )
    p.add_argument(
        "--exclude-out-of-budget",
        action="store_true",
        default=True,
        help="Exclude is_out_of_budget traces from voting. Default: True.",
    )
    p.add_argument(
        "--include-out-of-budget",
        dest="exclude_out_of_budget",
        action="store_false",
        help="Include out-of-budget traces in voting.",
    )
    p.add_argument(
        "--qid",
        type=int,
        action="append",
        default=None,
        help="Only process this qid (repeatable). If omitted, process all qid*.csv found.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("/Users/wbchen/Workspace-Py/deepconf/eval/results/deepconf_online_curve.csv"),
        help="Where to write the aggregated curve CSV.",
    )
    p.add_argument(
        "--out-details-csv",
        type=Path,
        default=None,
        help="Optional: write per-(qid, run_id, budget) details CSV.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def _list_qid_csvs(df_dir: Path) -> dict[int, Path]:
    """Return {qid: csv_path} discovered under df_dir."""
    out: dict[int, Path] = {}
    for p in sorted(df_dir.glob("qid*.csv")):
        m = re.match(r"qid(?P<qid>\d+)\.csv$", p.name)
        if not m:
            continue
        out[int(m.group("qid"))] = p
    return out


def _get_token_dirs_sorted(probe_dir: Path) -> list[Path]:
    """Return probe token directories sorted by token position."""
    dirs: list[tuple[int, Path]] = []
    for name in os.listdir(probe_dir):
        p = probe_dir / name
        if not p.is_dir():
            continue
        if name.endswith("-probe"):
            num = name[: -len("-probe")]
            if num.isdigit():
                dirs.append((int(num), p))
        elif name.isdigit():
            dirs.append((int(name), p))
    return [p for _, p in sorted(dirs)]


def load_ground_truth_by_qid(probe_dir: Path) -> dict[int, str]:
    """Load qid -> ground_truth by scanning ONE probe checkpoint directory.

    The raw dp_probe JSON includes `ground_truth` at the top level, which is constant
    across checkpoints. We read it once per qid to score accuracy.
    """
    token_dirs = _get_token_dirs_sorted(probe_dir)
    if not token_dirs:
        raise RuntimeError(f"No probe subdirectories found under {probe_dir}")

    # Use earliest checkpoint to maximize coverage.
    first_dir = token_dirs[0]
    gt_by_qid: dict[int, str] = {}

    for filename in os.listdir(first_dir):
        if not filename.endswith(".json") or filename == "summary_results.json":
            continue
        m = FILENAME_RE.search(filename)
        if not m:
            continue
        qid = int(m.group("qid"))
        if qid in gt_by_qid:
            continue
        path = first_dir / filename
        try:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            gt_by_qid[qid] = str(obj.get("ground_truth", "")).strip()
        except Exception as e:  # noqa: BLE001
            LOG.warning("Failed reading ground_truth from %s: %s", path, e)

    return gt_by_qid


def _ensure_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([], dtype=float)


def collapse_to_trace_finals(df: pd.DataFrame, qid: int) -> pd.DataFrame:
    """Collapse a per-qid checkpoint DataFrame to per-trace final records.

    Returns a DataFrame with one row per (run_id, trace_idx), including:
      - final_answer (taken from the checkpoint with largest token_usage, tie-broken by prob_token)
      - final_token_usage (max token_usage over checkpoints)
      - trace_confidence (min confidence over checkpoints, approximating LGC)
      - flags (any True over checkpoints)
    """
    if df.empty:
        return df

    # Use observed rows only if present (avoid any propagated rows).
    if "observed" in df.columns:
        df = df[df["observed"] == True].copy()  # noqa: E712

    # Basic required columns
    required = ["run_id", "trace_idx", "prob_token", "answer", "confidence", "token_usage"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in qid{qid}: {missing}")

    df = df.copy()
    df["qid"] = qid
    df["trace_idx"] = pd.to_numeric(df["trace_idx"], errors="coerce").astype("Int32")
    df["prob_token"] = pd.to_numeric(df["prob_token"], errors="coerce").astype("Int32")
    df["token_usage_num"] = pd.to_numeric(df["token_usage"], errors="coerce")
    df["confidence_num"] = pd.to_numeric(df["confidence"], errors="coerce")

    # Compute trace_confidence as min confidence over checkpoints (LGC proxy).
    conf_min = (
        df.groupby(["run_id", "trace_idx"], observed=True)["confidence_num"]
        .min()
        .rename("trace_confidence")
    )

    # Pick "final row": max token_usage, then max prob_token as tie-break.
    # This is only used to select the final_answer and final_token_usage.
    df_sorted = df.sort_values(
        ["run_id", "trace_idx", "token_usage_num", "prob_token"],
        ascending=[True, True, True, True],
        kind="stable",
    )
    final_rows = df_sorted.groupby(["run_id", "trace_idx"], observed=True, sort=False).tail(1)

    # Flags: any True across checkpoints.
    def _any_true(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(False, index=conf_min.index)
        return (
            df.groupby(["run_id", "trace_idx"], observed=True)[col]
            .apply(lambda s: bool(np.any(s == True)))  # noqa: E712
            .rename(col)
        )

    flags = pd.concat(
        [
            _any_true("is_warmup"),
            _any_true("is_naturally_stopped"),
            _any_true("is_deep_conf_truncated"),
            _any_true("is_out_of_budget"),
        ],
        axis=1,
    )

    out = final_rows[["run_id", "trace_idx", "answer", "token_usage_num"]].copy()
    out = out.rename(columns={"answer": "final_answer", "token_usage_num": "final_token_usage"})
    out = out.merge(conf_min.reset_index(), on=["run_id", "trace_idx"], how="left")
    out = out.merge(flags.reset_index(), on=["run_id", "trace_idx"], how="left")
    out["qid"] = qid

    # Clean types
    out["final_token_usage"] = pd.to_numeric(out["final_token_usage"], errors="coerce").fillna(0).astype("int64")
    out["trace_confidence"] = pd.to_numeric(out["trace_confidence"], errors="coerce").fillna(0.0).astype("float64")
    out["trace_idx"] = pd.to_numeric(out["trace_idx"], errors="coerce").fillna(-1).astype("int32")
    out["run_id"] = out["run_id"].astype(str)

    return out


def select_first_b_traces(trace_finals: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Select the first `budget` traces in generation order (trace_idx ascending)."""
    if trace_finals.empty:
        return trace_finals
    return trace_finals.sort_values(["trace_idx"], kind="stable").head(int(budget)).copy()


def _filter_top_eta(traces: pd.DataFrame, keep_top: float) -> pd.DataFrame:
    """Keep top-eta fraction by trace_confidence (descending)."""
    if traces.empty:
        return traces
    keep_top = float(keep_top)
    if keep_top <= 0 or keep_top > 1:
        raise ValueError("--keep-top must be in (0, 1].")
    n = len(traces)
    k = max(1, int(np.ceil(keep_top * n)))
    return traces.sort_values(["trace_confidence"], ascending=False, kind="stable").head(k)


def confidence_weighted_vote(
    traces: pd.DataFrame,
    *,
    keep_top: float,
    exclude_truncated: bool,
    exclude_out_of_budget: bool,
) -> tuple[Optional[str], dict[str, float]]:
    """Confidence-weighted majority voting on final answers.

    Voting pool is:
      is_warmup OR is_naturally_stopped
    with optional exclusions for truncated/out_of_budget traces.

    Returns:
      (predicted_answer, vote_weights_by_answer)
    """
    if traces.empty:
        return None, {}

    # Vote eligibility
    elig = (traces.get("is_warmup", False) == True) | (traces.get("is_naturally_stopped", False) == True)  # noqa: E712
    if exclude_truncated and "is_deep_conf_truncated" in traces.columns:
        elig = elig & (traces["is_deep_conf_truncated"] == False)  # noqa: E712
    if exclude_out_of_budget and "is_out_of_budget" in traces.columns:
        elig = elig & (traces["is_out_of_budget"] == False)  # noqa: E712

    pool = traces[elig].copy()
    if pool.empty:
        return None, {}

    # Confidence filtering (top-eta by confidence)
    pool = _filter_top_eta(pool, keep_top=keep_top)

    # Weighted vote
    vote_weights: dict[str, float] = {}
    for _, r in pool.iterrows():
        ans = str(r["final_answer"])
        w = float(r.get("trace_confidence", 0.0))
        vote_weights[ans] = vote_weights.get(ans, 0.0) + w

    if not vote_weights:
        return None, {}
    pred = max(vote_weights.items(), key=lambda kv: kv[1])[0]
    return pred, vote_weights


def compute_total_tokens(traces: pd.DataFrame) -> int:
    """Total tokens consumed by the selected traces (including non-voting traces)."""
    if traces.empty:
        return 0
    return int(pd.to_numeric(traces["final_token_usage"], errors="coerce").fillna(0).sum())


def score_answer(pred: Optional[str], gt: str) -> Optional[bool]:
    """Return True/False correctness; None if pred is None."""
    if pred is None:
        return None
    try:
        return bool(math_equal(pred, gt))
    except Exception:
        # Fallback: exact match
        return pred.strip() == gt.strip()


def parse_budgets(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(out))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    budgets = parse_budgets(args.budgets)
    if not budgets:
        raise SystemExit("No budgets provided.")
    LOG.info("Budgets: %s", budgets)
    LOG.info("keep_top=%.3f exclude_truncated=%s exclude_out_of_budget=%s", args.keep_top, args.exclude_truncated, args.exclude_out_of_budget)

    gt_by_qid = load_ground_truth_by_qid(args.probe_dir)
    LOG.info("Loaded ground_truth for %d qids from %s", len(gt_by_qid), args.probe_dir)

    qid_csvs = _list_qid_csvs(args.df_dir)
    if args.qid is not None:
        want = set(args.qid)
        qid_csvs = {qid: p for qid, p in qid_csvs.items() if qid in want}
    if not qid_csvs:
        raise SystemExit(f"No qid CSVs found under {args.df_dir}")

    details_rows: list[dict[str, Any]] = []

    for qid, csv_path in sorted(qid_csvs.items()):
        if qid not in gt_by_qid:
            LOG.warning("qid=%s missing ground_truth; skipping", qid)
            continue

        df = pd.read_csv(csv_path)
        trace_finals = collapse_to_trace_finals(df, qid=qid)
        if trace_finals.empty:
            LOG.warning("qid=%s: empty trace table; skipping", qid)
            continue

        # Process each run independently
        for run_id, g in trace_finals.groupby("run_id", sort=False):
            g = g.sort_values(["trace_idx"], kind="stable")

            for b in budgets:
                sel = select_first_b_traces(g, budget=b)
                tokens = compute_total_tokens(sel)
                pred, _ = confidence_weighted_vote(
                    sel,
                    keep_top=args.keep_top,
                    exclude_truncated=args.exclude_truncated,
                    exclude_out_of_budget=args.exclude_out_of_budget,
                )
                correct = score_answer(pred, gt_by_qid[qid])

                details_rows.append(
                    {
                        "qid": qid,
                        "run_id": run_id,
                        "budget_traces": b,
                        "total_tokens": tokens,
                        "pred": pred if pred is not None else "",
                        "ground_truth": gt_by_qid[qid],
                        "correct": (bool(correct) if correct is not None else None),
                        "n_selected_traces": int(len(sel)),
                        "n_voting_traces": int(
                            (
                                (sel.get("is_warmup", False) == True)  # noqa: E712
                                | (sel.get("is_naturally_stopped", False) == True)  # noqa: E712
                            ).sum()
                        ),
                    }
                )

    if not details_rows:
        raise SystemExit("No results produced (check qid selection and ground_truth availability).")

    df_details = pd.DataFrame(details_rows)
    # Aggregate curve over (qid, run_id) trials
    df_curve = (
        df_details.groupby("budget_traces", as_index=False)
        .agg(
            mean_tokens=("total_tokens", "mean"),
            mean_acc=("correct", "mean"),
            n_trials=("correct", "count"),
        )
        .sort_values("budget_traces")
        .reset_index(drop=True)
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_curve.to_csv(args.out_csv, index=False)
    LOG.info("Wrote curve CSV: %s", args.out_csv)

    if args.out_details_csv is not None:
        args.out_details_csv.parent.mkdir(parents=True, exist_ok=True)
        df_details.to_csv(args.out_details_csv, index=False)
        LOG.info("Wrote details CSV: %s", args.out_details_csv)

    # Print a compact summary to stdout
    print(df_curve.to_string(index=False))


if __name__ == "__main__":
    main()

