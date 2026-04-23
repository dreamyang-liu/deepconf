#!/bin/bash
# Convert bedrock batch output (jsonl.out) into per-qid pkls for prefill input.
# Uses conda env 'deepconf' on host (CPU-only).
# Usage: ./convert_dataset.sh <dataset>
#   dataset: brumo25 | hmmt | aime24 | aime25
set -eu
DS=$1
cd /opt/dlami/nvme/projects/deepconf
source /opt/dlami/nvme/miniconda3/etc/profile.d/conda.sh
conda activate deepconf
python convert-bedrock-batch.py \
  --dataset "${DS}" \
  --output-dir conf-data-coder-next \
  --rid coder_next \
  --bedrock-dir bedrock-data/qwen3-coder-next
