# eval scripts: probe JSON → CSV → DeepConf curve

This folder documents two scripts in `eval/`:

- `convert_probe_json_to_df_per_qid.py`: convert raw dp_probe JSONs into per-qid CSVs
- `compute_deepconf_online_curve.py`: compute **accuracy vs total tokens** from those CSVs (DeepConf online)

---

## 1) Convert raw dp_probe JSONs to per-qid CSVs

**Input**: raw JSON tree like:
`outputs-online-real-percent-90-aime-2025-probe_result_raw/{prob_token}-probe/*.json`

**Output**:
`eval/df_per_qid/filled/qidXX.csv` (one file per qid; rows are per `(run_id, trace_idx, prob_token)` checkpoint)

Run:

```bash
python eval/convert_probe_json_to_df_per_qid.py --no-propagate-missing-files
```

Useful flags:
- `--qid <int>` (repeatable): process only some qids
- `--no-propagate-missing-files`: drop rows for missing probe files (recommended if you want only real observations)
- `--no-propagate-missing-traces`: keep missing trace rows as missing (no forward fill)
- `--expected-n-traces 512` (default): enforce `trace_idx=0..511` grid; use `0` to use only observed traces
- `--format csv` (default) or `--format parquet` (if `pyarrow` available)

Notes:
- Warmup traces are included by default; use `--no-include-warmup` to exclude them.
- Propagation-tracking columns include `observed`, `is_propagated_*`, and `source_prob_token`.

---

## 2) Compute DeepConf online accuracy-vs-tokens curve

**What it does** (matches the “online thinking” evaluation logic in the paper):
- Treat each trace as having a **final answer** + **final token usage** (cost).
- For a trace budget \(B\), select the **first B traces** in generation order (approximated by ascending `trace_idx`).
- Sum **tokens over all selected traces**.
- Do **confidence-weighted majority voting** on final answers of:
  - warmup traces (`is_warmup=True`) and
  - naturally-stopped traces (`is_naturally_stopped=True`)
  (optionally excluding truncated / out-of-budget traces from voting).

The script reads the per-qid CSVs from step (1) and loads `ground_truth` from the raw JSONs to compute accuracy.

Run (all qids):

```bash
python eval/compute_deepconf_online_curve.py
```

Outputs:
- Aggregated curve CSV (default): `eval/results/deepconf_online_curve.csv`
- Optional per-trial details: pass `--out-details-csv <path>`

Common flags:
- `--budgets 32,64,128,256,512`, with 16 warm-up traces included. `--budgets 32` means 16 warmup + 16 non-warmup
- `--keep-top 0.90` (DeepConf-high style filtering ratio)
- `--qid <int>` (repeatable): run only some qids

Paper reference: `https://arxiv.org/pdf/2508.15260`

