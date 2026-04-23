# Run Full DeepConf Probe Pipeline

End-to-end flow from converted bedrock conf-data pkls to aggregated probe traces on S3.

## Pipeline Overview

```
Bedrock .jsonl.out    (generation traces from Qwen3-Next-80B-A3B-Thinking etc.)
    │  convert-bedrock-batch.py
    ▼
conf-data-<model>/<dataset>/  (30 deepconf_simple_qid*_rid*.pkl — no confs yet)
    │  prefill-confs-vllm-dp.py  ← Stage 1: PREFILL
    ▼
conf-data-<model>-confs/<dataset>/  (same format, now with per-token confs)
    │  probe_src/probe_batch_v2.py  ← Stage 2: PROBE
    ▼
probe_results/<dataset>/traces/*.pkl  (one pkl per (qid, trace_idx))
    │  probe_src/aggregate_traces.py  ← Stage 3: AGGREGATE
    ▼
probe_results/<dataset>/aggregated/{aggregated_traces.pkl, aggregated_summary.csv}
    │  aws s3 cp  ← Stage 4: UPLOAD
    ▼
s3://drmyang-training-data-241580540779-us-east-2-an/deepconf_probe_traces/<dataset>.pkl
```

## Prerequisites

- Docker container `verl` (vLLM 0.12.0, CUDA 12.9) for Stages 1 and 2 — both require vLLM
- Host conda env `deepconf` for Stages 3 and 4 (numpy + boto3)
- Input pkls in `conf-data-<model>/<dataset>/` produced by `convert-bedrock-batch.py`
- Dataset jsonl files (`brumo_2025.jsonl`, `hmmt_feb_2025.jsonl`, `aime_2024.jsonl`) at repo root

---

## Stage 1: Prefill (per-token confidence scores)

Loads each bedrock-generated trace back through vLLM with `prompt_logprobs=20` to compute per-token `-mean(top-k logprob)` confidence values.

### Launch (via orchestration script)

The `.cortices/launch_prefill.sh` wraps the full command. Usage:

```bash
# Host side, fire-and-forget into docker verl container:
.cortices/launch_prefill.sh brumo25   # or hmmt | aime24 | aime25
```

The script runs this inside the container:

```bash
docker exec -w /workspace/verl -d verl bash -c "nohup python prefill-confs-vllm-dp.py \
  --input-dir conf-data-coder-next/<dataset> \
  --dataset-file <dataset>.jsonl \
  --output-dir conf-data-coder-next-confs/<dataset> \
  --model-path Qwen/Qwen3-Next-80B-A3B-Thinking-FP8 \
  --tp 2 --num-gpu-workers 4 --num-producers 30 --num-post 2 \
  --chunk-size 256 --timeout-s 2.0 \
  --max-model-len 131072 --gpu-memory-utilization 0.75 \
  >> conf-data-coder-next-confs/prefill-dp-<dataset>.log 2>&1 &"
```

### Architecture

- 30 fork producers: batched-tokenize 16 traces → input_queue
- 4 spawn GPU workers (TP=2, uses all 8 GPUs): buffer to chunk_size=256 or 2s timeout → llm.generate → bg thread compute_confs → output_queue
- 2 fork post workers: save per-trace ckpts
- SamplingParams: `prompt_logprobs=20, detokenize=False, flat_logprobs=True` (key for performance — 1.6x speedup vs. Dict[int,Logprob])

### Resume behavior

Automatic — the script scans `ckpt_qid*/trace_*.pkl` on startup and only processes missing traces. Safe to kill + relaunch anytime.

### Monitor

```bash
# Ckpt count (target 122879 for datasets with 30 qids × 4096 traces)
find conf-data-coder-next-confs/<dataset>/ -name 'trace_*.pkl' | wc -l

# Final assembled pkls (target 30)
ls conf-data-coder-next-confs/<dataset>/deepconf_simple_*.pkl | wc -l

# Chunk-level progress
grep "chunk #" conf-data-coder-next-confs/prefill-dp-<dataset>.log | tail -10

# Per-chunk GPU gen time + bg compute time (should see gen ~100-150s, bg ~2-3s)
# Queue utilization (daemon prints every 10s): work=X/200 input=Y/60 output=Z/50
grep "\[queue\]" conf-data-coder-next-confs/prefill-dp-<dataset>.log | tail -5
```

Expected steady-state: **~7-8 traces/s aggregate** (chunk=256, flat_logprobs). ~4.5h per dataset.

### Stop

```bash
docker exec verl pkill -9 -f prefill-confs-vllm-dp
docker exec verl bash -c "pgrep -f 'VLLM::Worker|VLLM::EngineCor' | xargs -r kill -9"
```

---

## Stage 2: Probe

Runs a smaller evaluator model (default Qwen3-32B) at fixed token intervals along each generation to extract intermediate answers. Each probe point records `(token_position, raw_text, answer, is_correct, avg_conf)`.

```bash
# Inside docker verl container (or host w/ vLLM installed):
python probe_src/probe_batch_v2.py \
  --input-dir conf-data-coder-next-confs/<dataset> \
  --output-dir probe_results/<dataset> \
  --model-path Qwen/Qwen3-32B \
  --num-gpus 8 \
  --probe-interval 2048 \
  --batch-size 64 \
  --max-model-len 32768 \
  --mem-fraction 0.85 \
  --num-producers 60
```

### Output layout

```
probe_results/<dataset>/
  traces/           # one .pkl per (qid, trace_idx)
    qid00_trace00000.pkl
    qid00_trace00001.pkl
    ...
```

Each trace pkl contains:
- `qid`, `trace_idx`, `ground_truth`
- `probes`: dict mapping `token_position → {answer, is_correct, raw_text, avg_conf}`

### Resume

probe_batch_v2 skips traces with existing output pkl in `traces/`.

### Monitor

```bash
# Expected total probes = 30 qids × 4096 traces = 122880
ls probe_results/<dataset>/traces/*.pkl | wc -l

tail -30 probe_results/<dataset>/probe.log  # if logging redirected
```

---

## Stage 3: Aggregate

Merges all per-trace probe pkls into a single summary file for downstream analysis. Optionally merges per-token confs (from Stage 1 output) and extracted_answer.

```bash
# Host conda env (CPU-only, needs numpy)
conda activate deepconf
python probe_src/aggregate_traces.py \
  probe_results/<dataset>/traces \
  --conf-data-dir conf-data-coder-next-confs/<dataset>
```

### Output

```
probe_results/<dataset>/aggregated/
  aggregated_traces.pkl     # dict keyed by (qid, trace_idx); includes probes + confs + extracted_answer
  aggregated_summary.csv    # flat CSV, one row per (qid, trace_idx, token_position)
```

Typical size: 50-200 MB for 122880 traces (confs stored as np.float16 arrays to shrink ~8x).

---

## Stage 4: Upload to S3

Convention: final aggregated pkl → `s3://drmyang-training-data-241580540779-us-east-2-an/deepconf_probe_traces/<dataset>.pkl`

```bash
# Example for brumo25 with thinking model
aws s3 cp probe_results/brumo25/aggregated/aggregated_traces.pkl \
  s3://drmyang-training-data-241580540779-us-east-2-an/deepconf_probe_traces/brumo25.pkl

# Or with model-specific suffix if comparing runs
aws s3 cp probe_results/brumo25/aggregated/aggregated_traces.pkl \
  s3://drmyang-training-data-241580540779-us-east-2-an/deepconf_probe_traces/brumo25_thinking.pkl

# Verify
aws s3 ls s3://drmyang-training-data-241580540779-us-east-2-an/deepconf_probe_traces/
```

### Credentials

If using a pre-configured EC2 instance role, `aws s3` works directly. Otherwise need to export temporary credentials first:

```bash
aws sts get-session-token --duration-seconds 43200  # 12h temp creds
# Then export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN
```

---

## Full Sequential Example (single dataset)

```bash
DATASET=brumo25

# Stage 1: prefill (~4.5h on 8× H200)
.cortices/launch_prefill.sh $DATASET

# Wait for 30 final pkls
until [ $(ls conf-data-coder-next-confs/$DATASET/deepconf_simple_*.pkl 2>/dev/null | wc -l) -eq 30 ]; do
  echo "prefill: $(find conf-data-coder-next-confs/$DATASET/ -name 'trace_*.pkl' | wc -l)/122879 ckpts"
  sleep 600
done

# Stage 2: probe (~2-3h on 8× H200)
docker exec -w /workspace/verl -d verl bash -c "nohup python probe_src/probe_batch_v2.py \
  --input-dir conf-data-coder-next-confs/$DATASET \
  --output-dir probe_results/$DATASET \
  --num-gpus 8 \
  >> probe_results/$DATASET/probe.log 2>&1 &"

# Wait for 122880 traces
until [ $(ls probe_results/$DATASET/traces/*.pkl 2>/dev/null | wc -l) -ge 122880 ]; do
  sleep 300
done

# Stage 3: aggregate (~2-5 min)
conda run -n deepconf python probe_src/aggregate_traces.py \
  probe_results/$DATASET/traces \
  --conf-data-dir conf-data-coder-next-confs/$DATASET

# Stage 4: upload
aws s3 cp probe_results/$DATASET/aggregated/aggregated_traces.pkl \
  s3://drmyang-training-data-241580540779-us-east-2-an/deepconf_probe_traces/${DATASET}.pkl
```

## Orchestration via Cortices Plan

For long-running unattended execution (auto-restart, per-stage stall detection), use the existing `coder-next-hmmt-aime24-pipeline` plan as a template:

```bash
cortices plan list              # view plans
cortices plan run <id>          # trigger now
cortices plan update <id> --status active|paused
```

The plan decides stages based on observed disk state (final pkl counts, ckpt counts, running processes) rather than tracking internal flags — more resilient to restarts.
