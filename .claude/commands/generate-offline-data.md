# Generate Offline DeepConf Data from Text

This skill guides how to convert raw reasoning traces (text) into DeepConf-compatible offline data with per-token confidence scores.

## Overview

The pipeline has 3 stages:

1. **Generate traces** (if you don't have them yet) → `generate-traces.py`
2. **Convert external format** (if traces come from Bedrock/other APIs) → `convert-bedrock-to-offline.py`
3. **Recover per-token confidence** (prefill to get logprobs) → `prefill-confs.py` or `prefill-bedrock-confs.py`

## Stage 1: Generate Traces (Optional)

If you need to generate new traces from scratch using SGLang offline engine:

```bash
python generate-traces.py \
    --datasets aime_2025 \
    --budget 64 \
    --rid run0 \
    --tp 2 \
    --output-dir outputs-traces
```

Output: individual trace pickle files under `outputs-traces/<model>/<dataset>/traces/qid{N}_rid{rid}/trace_XXXX.pkl`

## Stage 2a: Convert Bedrock Batch Outputs

If traces come from AWS Bedrock batch inference (`.jsonl.out` files with `reasoningContent`):

```bash
python convert-bedrock-to-offline.py \
    --input-dir data/qwen32b/qwen3-32b/aime25_output \
    --dataset-file aime_2025.jsonl \
    --output-dir outputs-bedrock \
    --model-name Qwen3-32B
```

Output: per-question pickle files in DeepConf format, but with **empty confs** (no logprobs from Bedrock). Proceed to Stage 3.

## Stage 2b: Prepare Text Traces File

If you have traces as raw text, prepare a file with one trace per line:
- **Plain text**: one trace per line, literal `\n` for newlines within a trace
- **JSONL**: one `{"text": "..."}` per line

## Stage 3: Recover Per-Token Confidence via Prefill

This is the key step. SGLang does a forward pass (prefill) on the full sequence `prompt + generated_text` to recover logprobs, then computes confidence scores.

### Prerequisites

Start the SGLang server:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-32B \
    --tp 2 \
    --trust-remote-code \
    --port 30000 \
    --host 0.0.0.0 \
    --mem-fraction-static 0.75
```

Wait until the server is healthy: `curl http://localhost:30000/health`

### Option A: Single question from text file (`prefill-confs.py`)

```bash
python prefill-confs.py \
    --traces-file traces.txt \
    --dataset-file aime_2025.jsonl \
    --qid 0 \
    --rid run0 \
    --model-path Qwen/Qwen3-32B \
    --url http://localhost:30000 \
    --max-concurrent 128 \
    --output-dir outputs-prefill
```

### Option B: Batch process all bedrock-converted pickles (`prefill-bedrock-confs.py`)

```bash
python prefill-bedrock-confs.py \
    --input-dir outputs-bedrock \
    --output-dir outputs-bedrock-confs \
    --dataset-file aime_2025.jsonl \
    --model-path Qwen/Qwen3-32B \
    --url http://localhost:30000 \
    --batch-size 64 \
    --max-concurrent 2 \
    --timeout 1800
```

Features: auto batch-size based on token length, per-batch checkpointing, resumption support.

### After prefill is done

Kill the SGLang server to free GPU memory:

```bash
pkill -9 -f sglang
nvidia-smi  # verify GPU memory freed
```

## Output Format

All scripts produce pickle files with this structure:

```python
{
    "question_id": int,
    "run_id": str,
    "question": str,
    "ground_truth": str,
    "all_traces": [
        {
            "text": str,           # full generated text
            "token_ids": [int],    # token IDs of generated portion
            "num_tokens": int,
            "confs": [float],      # per-token confidence scores
            "extracted_answer": str,
            "is_correct": bool,
            "stop_reason": str,
        },
        ...
    ],
    "voted_answer": str,
    "is_voted_correct": bool,
    "accuracy": float,
    "token_stats": {...},
    "config": {...},
    "timestamp": str,
}
```

## Key Notes

- **Confidence scores** are computed from top-20 logprobs at each token position using `compute_confidence_sglang()` from `helper.py`
- **Batch size tuning**: `prefill-bedrock-confs.py` auto-selects batch size based on avg token length (128 for <8k, 64 for <16k, 32 for <25k, 16 for longer)
- **Memory**: long traces (>16k tokens) need lower `--max-concurrent` (2-4) to avoid OOM
- **Resumption**: both prefill scripts support resuming from where they left off
