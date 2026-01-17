#!/usr/bin/env python3
"""
Visualize top-N answer trends across probe token positions.

This script reads the parsed probe checkpoints produced by:
  `eval/convert_probe_json_to_df_per_qid.py`

Specifically it expects per-qid CSVs in:
  `eval/df_per_qid/filled/qidXX.csv`

For each question (qid), each probe token position (prob_token), and each run_id,
we consider the distribution of answers across traces (trace_idx). We compute:
  - per-run answer proportions at each token position
  - mean/std across runs to form error bands

It also supports mathematical-equivalence clustering (canonicalization) using
`helper.equal_func` which delegates to `dynasor.core.evaluator.math_equal` when
available. Canonicalization can optionally "snap" any answer equivalent to the
ground truth into the exact ground truth string (so the correct line is stable).

Two trace-selection modes are supported:
  - all: include all traces
  - non_truncated: include only rows where is_deep_conf_truncated == False

Typical usage (AIME 2025):
  python eval/plot_top_answer_trends.py \
    --df-dir eval/df_per_qid/filled \
    --dataset-jsonl aime_2025.jsonl \
    --top-n 10 \
    --trace-mode non_truncated \
    --out-dir eval/figs_answer_trends_aime_2025
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Ensure project root is importable (so `import helper` works when running from anywhere)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid Matplotlib trying to write cache/config outside the workspace.
# (In some sandboxes, ~/.matplotlib may be non-writable.)
_mpl_cfg = PROJECT_ROOT / ".mplconfig"
_mpl_cfg.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cfg))

# Headless-friendly backend (safe even on macOS; avoids GUI requirements).
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Repo-local fuzzy answer equality (used in notebooks / eval scripts)
from helper import equal_func as math_equal  # noqa: E402

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrendsResult:
    """Computed trends for one qid."""

    qid: int
    token_positions: List[int]
    plot_answers: List[str]
    ground_truth: Optional[str]
    # trends[token_pos][answer] = {"mean": float, "std": float}
    trends: Dict[int, Dict[str, Dict[str, float]]]
    # total_token_usage[token_pos] = {"mean": float, "std": float}
    # Mean/std are computed across runs after summing token_usage across *all traces*
    # within each run at that probe token position.
    total_token_usage: Dict[int, Dict[str, float]]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--df-dir", type=Path, default=Path("eval/df_per_qid/filled"))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("eval/figs/answer_trends"),
        help="Directory to write PNGs into.",
    )
    p.add_argument(
        "--qid",
        type=int,
        nargs="*",
        default=None,
        help="If provided, only plot these qids. Otherwise plot all qids found in df-dir.",
    )
    p.add_argument("--top-n", type=int, default=10, help="Top-N answers to track.")
    p.add_argument(
        "--trace-mode",
        choices=["all", "non_truncated"],
        default="all",
        help="Which traces to include in the distributions.",
    )
    p.add_argument(
        "--no-canonicalize",
        action="store_true",
        help="Disable mathematical-equivalence clustering (math_equal canonicalization).",
    )
    p.add_argument(
        "--dataset-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional dataset JSONL containing {question, answer} per line in qid order. "
            "If provided, ground truth will be shown and used as the canonical label for "
            "any mathematically-equivalent variants."
        ),
    )
    p.add_argument(
        "--fig-dpi",
        type=int,
        default=150,
        help="Output PNG DPI.",
    )
    p.add_argument(
        "--no-error-bands",
        action="store_true",
        help="Disable ±std error bands across runs.",
    )
    p.add_argument(
        "--no-token-usage",
        action="store_true",
        help="Disable secondary axis showing total token_usage across all traces.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def list_qid_csvs(df_dir: Path) -> Dict[int, Path]:
    """Return {qid: path} for all qidXX.csv files found in df_dir."""
    out: Dict[int, Path] = {}
    for p in sorted(df_dir.glob("qid*.csv")):
        stem = p.stem  # e.g. qid00
        if not stem.startswith("qid"):
            continue
        try:
            qid = int(stem[3:])
        except ValueError:
            continue
        out[qid] = p
    return out


def load_ground_truths(dataset_jsonl: Path) -> Dict[int, str]:
    """
    Load {qid -> answer_str} from a dataset JSONL.

    Assumes each line is a JSON object containing an "answer" field, and that qid
    corresponds to the 0-based line index (matching existing notebooks).
    """
    gts: Dict[int, str] = {}
    with dataset_jsonl.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ans = obj.get("answer", "")
            gts[i] = str(ans).strip()
    return gts


def _coerce_bool(df: pd.DataFrame, col: str) -> None:
    """Coerce a True/False column that might be loaded as object/string."""
    if col not in df.columns:
        return
    if pd.api.types.is_bool_dtype(df[col]):
        return
    # Common cases: "True"/"False", 1/0, or NaN.
    df[col] = df[col].astype("string").str.lower().map({"true": True, "false": False})


def load_qid_df(path: Path) -> pd.DataFrame:
    """
    Load a per-qid CSV produced by `convert_probe_json_to_df_per_qid.py`.

    We keep answer/run_id as strings to avoid losing formatting.
    """
    df = pd.read_csv(
        path,
        dtype={
            "answer": "string",
            "run_id": "string",
            "base_ts": "string",
        },
    )
    for c in [
        "is_warmup",
        "is_deep_conf_truncated",
        "is_naturally_stopped",
        "is_out_of_budget",
        "observed",
        "is_propagated_file",
        "is_propagated_trace",
        "is_propagated",
    ]:
        _coerce_bool(df, c)
    return df


def filter_traces(df: pd.DataFrame, trace_mode: str) -> pd.DataFrame:
    """Apply the requested trace selection mode."""
    if trace_mode == "all":
        return df
    if trace_mode == "non_truncated":
        if "is_deep_conf_truncated" not in df.columns:
            LOG.warning("trace-mode=non_truncated requested but column missing; no filtering applied.")
            return df
        return df[df["is_deep_conf_truncated"] == False]  # noqa: E712
    raise ValueError(f"Unknown trace_mode: {trace_mode}")


def build_canonical_map(
    unique_answers: Iterable[str],
    ground_truth: Optional[str],
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Build a mapping from each unique answer to its canonical representative.

    Strategy (matches the notebook):
    - First, map any answer math-equal to ground_truth into the exact ground_truth string.
    - Then, greedily cluster remaining answers by comparing to existing cluster reps.

    Returns:
      canonical_map: {original_answer: canonical_answer}
      clusters: {canonical_answer: [members]}
    """
    answers = [str(a) for a in unique_answers if a is not None]
    canonical_map: Dict[str, str] = {}
    clusters: Dict[str, List[str]] = {}

    @lru_cache(maxsize=200_000)
    def _me(a: str, b: str) -> bool:
        try:
            return bool(math_equal(a, b))
        except Exception:
            return False

    gt = (str(ground_truth).strip() if ground_truth is not None else "") or None

    # Phase 1: snap-to-ground-truth
    if gt is not None:
        clusters.setdefault(gt, [])
        for ans in answers:
            if _me(ans, gt):
                canonical_map[ans] = gt
                clusters[gt].append(ans)

    # Phase 2: cluster remaining greedily
    for ans in answers:
        if ans in canonical_map:
            continue

        matched = False
        for canonical in list(clusters.keys()):
            if gt is not None and canonical == gt:
                continue
            if _me(ans, canonical):
                clusters[canonical].append(ans)
                canonical_map[ans] = canonical
                matched = True
                break

        if not matched:
            clusters[ans] = [ans]
            canonical_map[ans] = ans

    return canonical_map, clusters


def compute_trends(
    df_answers: pd.DataFrame,
    df_all_traces: pd.DataFrame,
    qid: int,
    top_n: int,
    canonicalize: bool,
    ground_truth: Optional[str],
) -> TrendsResult:
    """
    Compute answer trend mean/std across runs for a single qid.

    Output structure mirrors the notebook, but is driven directly by the per-qid CSV.
    """
    if df_answers.empty:
        return TrendsResult(
            qid=qid,
            token_positions=[],
            plot_answers=[],
            ground_truth=ground_truth,
            trends={},
            total_token_usage={},
        )

    # Token positions (x-axis) should be stable across trace modes.
    token_positions = sorted({int(x) for x in df_all_traces["prob_token"].dropna().unique().tolist()})

    # Collect unique raw answers for canonicalization.
    unique_raw_answers = set(df_answers["answer"].dropna().astype(str).tolist())

    if canonicalize and ground_truth:
        canonical_map, _clusters = build_canonical_map(unique_raw_answers, ground_truth)
    else:
        canonical_map = {a: a for a in unique_raw_answers}

    # per_run_props[token_pos][run_id] = {answer: prop}
    per_run_props: Dict[int, Dict[str, Dict[str, float]]] = {}
    # per_run_total_tokens[token_pos][run_id] = sum token_usage across all traces at token_pos
    # NOTE: this is computed from df_all_traces (unfiltered) so it's comparable across trace modes.
    per_run_total_tokens: Dict[int, Dict[str, float]] = {}
    # Used to determine "top answers" by pooled counts at each token_pos.
    pooled_counts_by_token: Dict[int, Counter] = {}

    # Grouping is faster if we pre-split by token position.
    for tp in token_positions:
        df_tp = df_answers[df_answers["prob_token"] == tp]
        if df_tp.empty:
            continue

        per_run_props[tp] = {}
        pooled: List[str] = []

        for run_id, g in df_tp.groupby("run_id", observed=True, sort=False):
            # Answers across traces at this checkpoint.
            raw_answers = g["answer"].dropna().astype(str).tolist()
            if not raw_answers:
                continue

            canon_answers = [canonical_map.get(a, a) for a in raw_answers]
            pooled.extend(canon_answers)

            counts = Counter(canon_answers)
            total = float(len(canon_answers))
            per_run_props[tp][str(run_id)] = {a: c / total for a, c in counts.items()}

        if pooled:
            pooled_counts_by_token[tp] = Counter(pooled)

    # Total token usage (sum over all traces), computed on unfiltered data.
    if "token_usage" in df_all_traces.columns:
        for tp in token_positions:
            g_tp = df_all_traces[df_all_traces["prob_token"] == tp]
            if g_tp.empty:
                continue
            per_run_total_tokens[tp] = {}
            for run_id, g in g_tp.groupby("run_id", observed=True, sort=False):
                toks = pd.to_numeric(g["token_usage"], errors="coerce")
                per_run_total_tokens[tp][str(run_id)] = float(toks.sum(skipna=True))

    # Union of top-N answers across token positions.
    all_top_answers: set[str] = set()
    for tp, cnt in pooled_counts_by_token.items():
        for ans, _c in cnt.most_common(top_n):
            all_top_answers.add(ans)
    if ground_truth:
        all_top_answers.add(ground_truth)

    # Compute mean/std across runs.
    trends: Dict[int, Dict[str, Dict[str, float]]] = {}
    for tp, by_run in per_run_props.items():
        if not by_run:
            continue
        trends[tp] = {}
        for ans in all_top_answers:
            props = [by_run[r].get(ans, 0.0) for r in by_run.keys()]
            trends[tp][ans] = {"mean": float(np.mean(props)), "std": float(np.std(props))}

    total_token_usage: Dict[int, Dict[str, float]] = {}
    for tp, by_run in per_run_total_tokens.items():
        if not by_run:
            continue
        vals = list(by_run.values())
        total_token_usage[tp] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    # Choose plotted answers: prioritize ground truth, then by max mean proportion over time.
    def _max_mean(a: str) -> float:
        mm = 0.0
        for tp in trends.keys():
            mm = max(mm, trends[tp].get(a, {}).get("mean", 0.0))
        return mm

    sorted_answers = sorted(all_top_answers, key=lambda a: (a != ground_truth, -_max_mean(a)))
    plot_answers = sorted_answers[:top_n]
    if ground_truth and ground_truth not in plot_answers:
        plot_answers = [ground_truth] + plot_answers[: max(0, top_n - 1)]

    return TrendsResult(
        qid=qid,
        token_positions=token_positions,
        plot_answers=plot_answers,
        ground_truth=ground_truth,
        trends=trends,
        total_token_usage=total_token_usage,
    )


def plot_trends(
    res: TrendsResult,
    *,
    show_error_bands: bool,
    show_token_usage: bool,
    canonicalize: bool,
    title_suffix: str,
) -> Optional[plt.Figure]:
    """Create a matplotlib figure for one qid; returns None if no data."""
    if not res.trends or not res.token_positions:
        return None

    sorted_tokens = sorted(res.trends.keys())
    gt = res.ground_truth

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.set_xlabel("Token Position", fontsize=12)
    ax1.set_ylabel("Proportion of Answers", fontsize=12, color="black")
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax1.grid(True, alpha=0.3)

    # Enough distinct colors for 10 lines.
    colors = list(plt.cm.tab20.colors)
    answer_colors: Dict[str, Tuple[float, float, float]] = {}
    color_idx = 0
    for ans in res.plot_answers:
        if gt and ans == gt:
            answer_colors[ans] = (0.18, 0.49, 0.20)  # dark-ish green
        else:
            answer_colors[ans] = colors[color_idx % len(colors)]
            color_idx += 1

    for ans in res.plot_answers:
        means = [res.trends[tp].get(ans, {}).get("mean", 0.0) for tp in sorted_tokens]
        stds = [res.trends[tp].get(ans, {}).get("std", 0.0) for tp in sorted_tokens]

        label = ans
        if len(label) > 40:
            label = label[:40] + "..."
        if gt and ans == gt:
            label = f"✓ {label} (correct)"
            linewidth, alpha, band_alpha = 3, 1.0, 0.25
        else:
            linewidth, alpha, band_alpha = 2, 0.75, 0.15

        color = answer_colors[ans]
        ax1.plot(
            sorted_tokens,
            means,
            marker="o",
            markersize=5,
            label=label,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )

        if show_error_bands:
            upper = [min(m + s, 1.0) for m, s in zip(means, stds)]
            lower = [max(m - s, 0.0) for m, s in zip(means, stds)]
            ax1.fill_between(sorted_tokens, lower, upper, color=color, alpha=band_alpha)

    ax2_line = None
    if show_token_usage and res.total_token_usage:
        ax2 = ax1.twinx()
        usage_tokens = [tp for tp in sorted_tokens if tp in res.total_token_usage]
        usage_means = [res.total_token_usage[tp]["mean"] for tp in usage_tokens]
        usage_stds = [res.total_token_usage[tp].get("std", 0.0) for tp in usage_tokens]

        ax2.plot(usage_tokens, usage_means, "k--", linewidth=2, alpha=0.6)
        if show_error_bands:
            upper = [m + s for m, s in zip(usage_means, usage_stds)]
            lower = [max(m - s, 0.0) for m, s in zip(usage_means, usage_stds)]
            ax2.fill_between(usage_tokens, lower, upper, color="gray", alpha=0.15)

        ax2.set_ylabel("Total token_usage across all traces", fontsize=12, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        ax2.spines["right"].set_color("gray")
        ax2_line = plt.Line2D([0], [0], color="black", linestyle="--", linewidth=2, alpha=0.6)

    canon_note = " (math-equiv grouped)" if canonicalize else ""
    gt_note = ""
    if gt:
        gt_short = gt[:60] + ("..." if len(gt) > 60 else "")
        gt_note = f"Ground Truth: {gt_short}"
    else:
        gt_note = "Ground Truth: (unknown; pass --dataset-jsonl to show)"

    tokens_note = ""
    if res.total_token_usage:
        last_tp = max(res.total_token_usage.keys())
        m = res.total_token_usage[last_tp]["mean"]
        tokens_note = f"Total tokens @ {last_tp}: {m:,.0f} (mean over runs)"

    ax1.set_title(
        f"Top {len(res.plot_answers)} Answer Trends for Q{res.qid}{canon_note}{title_suffix}\n{gt_note}"
        + (f"\n{tokens_note}" if tokens_note else ""),
        fontsize=13,
    )

    handles, labels = ax1.get_legend_handles_labels()
    if ax2_line is not None:
        handles.append(ax2_line)
        labels.append("Total token_usage across all traces (right)")
    ax1.legend(handles, labels, bbox_to_anchor=(1.15, 1), loc="upper left", fontsize=9)

    return fig


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    df_dir: Path = args.df_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    qid_paths = list_qid_csvs(df_dir)
    if not qid_paths:
        LOG.error("No qid*.csv files found in %s", df_dir)
        return 2

    qids: List[int]
    if args.qid is None:
        qids = sorted(qid_paths.keys())
    else:
        qids = [q for q in args.qid if q in qid_paths]
        missing = [q for q in args.qid if q not in qid_paths]
        if missing:
            LOG.warning("Requested qids not found in df-dir (skipping): %s", missing)

    ground_truths: Dict[int, str] = {}
    if args.dataset_jsonl is not None:
        ground_truths = load_ground_truths(args.dataset_jsonl)
        LOG.info("Loaded %d ground-truth answers from %s", len(ground_truths), args.dataset_jsonl)

    canonicalize = not args.no_canonicalize
    show_error_bands = not args.no_error_bands
    show_token_usage = not args.no_token_usage

    title_suffix = f" · trace_mode={args.trace_mode}"

    LOG.info(
        "Plotting qids=%d top_n=%d canonicalize=%s trace_mode=%s -> %s",
        len(qids),
        args.top_n,
        canonicalize,
        args.trace_mode,
        out_dir,
    )

    n_done = 0
    for qid in qids:
        csv_path = qid_paths[qid]
        gt = ground_truths.get(qid)

        df = load_qid_df(csv_path)
        df_all = df
        df_answers = filter_traces(df_all, args.trace_mode)

        res = compute_trends(
            df_answers=df_answers,
            df_all_traces=df_all,
            qid=qid,
            top_n=args.top_n,
            canonicalize=canonicalize,
            ground_truth=gt,
        )
        if res.total_token_usage:
            last_tp = max(res.total_token_usage.keys())
            m = res.total_token_usage[last_tp]["mean"]
            s = res.total_token_usage[last_tp].get("std", 0.0)
            LOG.info(
                "qid=%02d total_token_usage@%d = %.0f ± %.0f (mean±std over runs; summed over all traces)",
                qid,
                last_tp,
                m,
                s,
            )

        fig = plot_trends(
            res,
            show_error_bands=show_error_bands,
            show_token_usage=show_token_usage,
            canonicalize=canonicalize,
            title_suffix=title_suffix,
        )
        if fig is None:
            LOG.warning("No data to plot for qid=%s (after filtering).", qid)
            continue

        canon_tag = "canon" if canonicalize else "raw"
        out_path = out_dir / f"top{args.top_n}_trends_{canon_tag}_{args.trace_mode}_qid{qid:02d}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=args.fig_dpi, bbox_inches="tight")
        plt.close(fig)
        n_done += 1
        LOG.info("Saved %s", out_path)

    LOG.info("Done. Wrote %d figures to %s", n_done, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

