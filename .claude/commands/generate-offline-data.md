# Generate Offline DeepConf Data from Bedrock Batch Output

Convert AWS Bedrock batch inference outputs into DeepConf-compatible offline data with per-token confidence scores.

## Pipeline Overview

```
Bedrock .jsonl.out files
    │
    ▼ Stage 1: convert-bedrock-batch.py
Per-question pickles (confs=[], no logprobs)
    │
    ▼ Stage 2: prefill-confs-vllm.py + run_prefill_all.py
Per-question pickles (confs=[float], with logprobs)
    │
    ▼ Ready for analysis_offline.py
```

## Data Layout

Input (Bedrock batch outputs):
```
~/projects/{dataset}/{0..7}/{hash}/{dataset}_batch_{N}.jsonl.out
```

Each `.jsonl.out` record has:
- `recordId`: `{dataset}__{qid}__trace_{idx}` (e.g. `aime24__60__trace_00003`)
- `modelOutput.output.message.content[0].reasoningContent.reasoningText.text` — thinking
- `modelOutput.output.message.content[1].text` — final answer

Dataset files (ground truth):
- `aime_2024.jsonl` — 30 questions (AIME 2024 I+II from MathArena)
- `brumo_2025.jsonl` — 30 questions
- `hmmt_feb_2025.jsonl` — 30 questions

## Stage 1: Convert Bedrock → DeepConf Pickle

```bash
conda run -n deepconf python convert-bedrock-batch.py --dataset aime24 --output-dir outputs-bedrock
conda run -n deepconf python convert-bedrock-batch.py --dataset brumo25 --output-dir outputs-bedrock
conda run -n deepconf python convert-bedrock-batch.py --dataset hmmt --output-dir outputs-bedrock
# Or all at once:
conda run -n deepconf python convert-bedrock-batch.py --dataset all --output-dir outputs-bedrock
```

What it does:
1. Reads all `.jsonl.out` files across 8 batch folders
2. Parses `recordId` to group traces by question (handles different ID formats per dataset)
3. For aime24: fuzzy-matches questions to `aime_2024.jsonl` (qids 60-89 → dataset index 0-29)
4. Extracts `reasoning + content` as full trace text
5. Runs `extract_answer()` + `equal_func()` for correctness
6. Saves per-question pickle with `confs=[]` (no logprobs from Bedrock)

Output: `outputs-bedrock/{dataset}/deepconf_simple_qid{N}_rid{rid}_{timestamp}.pkl`

**After this stage**: majority voting analysis already works. Confidence-weighted voting requires Stage 2.

## Stage 2: Recover Per-Token Confidence via vLLM Prefill

### How it works

For each trace, construct `full_sequence = prompt + trace_text`, send to vLLM with `prompt_logprobs=20, max_tokens=1`. vLLM does a forward pass (prefill) and returns top-20 logprobs at each input token position. Confidence at each generated token = `-mean(top_20_logprobs)`.

### Prerequisites

```bash
conda activate deepconf
# vLLM must be installed (pip install vllm)
# Model weights for Qwen/Qwen3-32B must be accessible
```

### Run single question

```bash
conda run -n deepconf python prefill-confs-vllm.py \
    --input-dir outputs-bedrock/aime24 \
    --dataset-file aime_2024.jsonl \
    --model-path Qwen/Qwen3-32B \
    --tp 2 \
    --chunk-size 512 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.60 \
    --qids 0 \
    --output-dir outputs-bedrock-confs/aime24
```

### Run all questions (with auto-retry on OOM)

```bash
# Foreground:
conda run -n deepconf python run_prefill_all.py aime24
conda run -n deepconf python run_prefill_all.py brumo25
conda run -n deepconf python run_prefill_all.py hmmt

# Background (recommended, ~20-30h per dataset):
nohup conda run -n deepconf python run_prefill_all.py aime24 > prefill_aime24.log 2>&1 &
```

`run_prefill_all.py` auto-handles OOM:
1. Tries chunk sizes `[512, 64, 8, 1]` in sequence
2. Saves per-trace checkpoints to `ckpt_qid{N}/trace_NNNN.pkl` after each chunk
3. On crash, restarts from checkpoints with smaller chunk
4. If all chunks fail, assembles final pickle from whatever checkpoints exist
5. Traces exceeding `max_model_len` are skipped (confs remain empty)

### Key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--gpu-memory-utilization` | 0.60 | Lower = more room for prompt_logprobs softmax. 0.75 OOMs on >20K token traces |
| `--chunk-size` | 512 | Traces per `llm.generate()` call. Smaller = less progress lost on OOM |
| `--max-model-len` | 40960 | Qwen3-32B max. Traces exceeding this are skipped |
| `--tp` | 2 | Tensor parallel GPUs |

### OOM behavior

vLLM V1 allocates `(seq_len × vocab_size × 4 bytes)` for prompt_logprobs softmax. For Qwen3-32B (vocab=152064):
- 20K tokens ≈ 11 GB temporary allocation
- 30K tokens ≈ 17 GB temporary allocation

With `gpu_memory_utilization=0.60` on 96GB GPUs: model=31GB, KV cache=24GB, ~3GB free per GPU. Long traces may still OOM at chunk=1 — these get skipped and their confs remain empty.

### Monitor progress

```bash
# Check checkpoints per question:
for q in $(seq 0 29); do
  d=outputs-bedrock-confs/aime24/ckpt_qid${q}
  [ -d "$d" ] && echo "qid$q: $(find $d -name '*.pkl' | wc -l)/4096"
done

# Check final pickles:
ls outputs-bedrock-confs/aime24/deepconf_simple_*.pkl | wc -l

# GPU usage:
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

## Output Format

All scripts produce pickle files compatible with `analysis_offline.py`:

```python
{
    "question_id": int,
    "run_id": str,
    "question": str,
    "ground_truth": str,
    "all_traces": [
        {
            "text": str,              # full generated text (reasoning + answer)
            "token_ids": list[int],   # token IDs (empty from Bedrock, filled by prefill)
            "num_tokens": int,
            "confs": list[float],     # per-token confidence = -mean(top_20_logprobs)
            "extracted_answer": str,  # from \boxed{}
            "is_correct": bool,
            "stop_reason": str,
        },
        ...
    ],
    "voted_answer": str,
    "is_voted_correct": bool,
    "accuracy": float,
    "token_stats": {"total_tokens": int, "total_traces_count": int, "avg_tokens_per_trace": float},
    "config": {...},
    "timestamp": str,
}
```

## Alternative: SGLang-based prefill

If using SGLang instead of vLLM, use the server-based scripts:

```bash
# Start SGLang server
python -m sglang.launch_server --model-path Qwen/Qwen3-32B --tp 2 --trust-remote-code --port 30000

# Single question from text file
python prefill-confs.py --traces-file traces.txt --dataset-file aime_2025.jsonl --qid 0

# Batch from bedrock-converted pickles
python prefill-bedrock-confs.py --input-dir outputs-bedrock --output-dir outputs-bedrock-confs
```

## Script Reference

| Script | Purpose |
|--------|---------|
| `convert-bedrock-batch.py` | Bedrock .jsonl.out → DeepConf pickle (no confs) |
| `prefill-confs-vllm.py` | vLLM offline prefill to recover logprobs/confs |
| `run_prefill_all.py` | Auto-run prefill for all questions with OOM retry |
| `prefill-confs.py` | SGLang server-based prefill (single question) |
| `prefill-bedrock-confs.py` | SGLang server-based prefill (batch) |
| `convert-bedrock-to-offline.py` | Old single-dir Bedrock converter |
