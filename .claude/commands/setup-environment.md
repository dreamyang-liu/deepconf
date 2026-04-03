# Setup DeepConf Environment

Set up the DeepConf development environment from scratch.

## Prerequisites

- CUDA-compatible GPU(s) (tested on H200 96GB)
- Conda (Miniconda or Anaconda)
- Git

## Step 1: Create Conda Environment

```bash
conda create -n deepconf python=3.12 -y
conda activate deepconf
```

## Step 2: Install Core Python Dependencies

```bash
pip install numpy pandas tqdm matplotlib scipy
```

## Step 3: Install Dynasor (Math Evaluation)

Used for `math_equal()` answer checking.

```bash
git clone https://github.com/hao-ai-lab/Dynasor.git
cd Dynasor && pip install . && cd -
```

## Step 4: Install HuggingFace Libraries

```bash
pip install transformers datasets
```

## Step 5: Install vLLM

Two options depending on the workflow:

### Option A: Offline DeepConf (standard vLLM with logprobs)

```bash
uv pip install --pre vllm==0.10.1+gptoss \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  --index-strategy unsafe-best-match
```

### Option B: Online DeepConf (custom vLLM with conf-stop)

```bash
git clone https://github.com/Viol2000/vllm.git
cd vllm && git checkout conf-stop
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
cd -
```

> Note: Make sure vLLM V1 is enabled.

## Step 6 (Optional): Install SGLang

Only needed if using SGLang-based prefill scripts (`prefill-confs.py`, `prefill-bedrock-confs.py`).

```bash
pip install sglang aiohttp
```

## Step 7: Prepare Datasets

Download from [MathArena on HuggingFace](https://huggingface.co/MathArena):

```bash
conda run -n deepconf python data.py
```

Or manually:

```python
import json
from datasets import load_dataset

for name, outfile in [
    ("MathArena/aime_2024", "aime_2024.jsonl"),
    ("MathArena/BRUMO_2025", "brumo_2025.jsonl"),
    ("MathArena/hmmt_feb_2025", "hmmt_feb_2025.jsonl"),
]:
    dataset = load_dataset(name, split="train")
    with open(outfile, "w", encoding="utf-8") as f:
        for ex in dataset:
            entry = {"question": ex["problem"], "answer": str(ex["answer"])}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Converted {len(dataset)} examples to {outfile}")
```

## Step 8: Verify Installation

```bash
conda run -n deepconf python -c "
import vllm; print('vLLM:', vllm.__version__)
import transformers; print('Transformers:', transformers.__version__)
import numpy; print('NumPy:', numpy.__version__)
from dynasor.core.evaluator import math_equal; print('Dynasor: OK')
print('All dependencies verified.')
"
```

## Model Weights

The following models are used across experiments:

| Model | Used By |
|-------|---------|
| `Qwen/Qwen3-32B` | Confidence prefill (`prefill-confs-vllm.py`) |
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | Online DeepConf experiments |
| `openai/gpt-oss-120b` | Offline DeepConf experiments |

Models are auto-downloaded by vLLM/transformers on first use. Ensure HuggingFace access is configured if models are gated.

## Directory Structure

```
deepconf/
  *.jsonl                    # Dataset files (ground truth)
  deepconf-offline.py        # Offline DeepConf algorithm
  deepconf-online.py         # Online DeepConf with early stopping
  deepconf-baseline.py       # Self-consistency baseline
  analysis_offline.py        # Analyze offline results
  analysis_online.py         # Analyze online results
  convert-bedrock-batch.py   # Bedrock batch -> DeepConf pickle
  prefill-confs-vllm.py      # vLLM-based confidence recovery
  prefill-confs-hf.py        # HuggingFace-based confidence recovery
  run_prefill_all.py         # Orchestrator for prefill with OOM retry
  helper.py                  # Core utilities (answer extraction, voting)
  outputs-bedrock/           # Stage 1 output (no confs)
  outputs-bedrock-confs/     # Stage 2 output (with confs)
```

## Quick Smoke Test

```bash
# Run offline on one question (requires GPU + model weights)
conda run -n deepconf python deepconf-offline.py --qid 0 --rid 0

# Run baseline on one question
conda run -n deepconf python deepconf-baseline.py --qid 0 --rid 0
```
