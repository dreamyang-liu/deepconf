#!/bin/bash
# Launch DP prefill for a given dataset inside docker verl container (detached).
# Usage: ./launch_prefill.sh <dataset>
#   dataset: brumo25 | hmmt | aime24 | aime25
set -eu
DS=$1
case "$DS" in
  brumo25)  DF="brumo_2025.jsonl" ;;
  hmmt)     DF="hmmt_feb_2025.jsonl" ;;
  aime24)   DF="aime_2024.jsonl" ;;
  aime25)   DF="aime_2025.jsonl" ;;
  *) echo "unknown dataset: $DS" >&2; exit 1 ;;
esac
docker exec -w /workspace/verl -d verl bash -c "nohup python prefill-confs-vllm-dp.py \
  --input-dir conf-data-coder-next/${DS} \
  --dataset-file ${DF} \
  --output-dir conf-data-coder-next-confs/${DS} \
  --model-path Qwen/Qwen3-Next-80B-A3B-Thinking-FP8 \
  --tp 2 --num-gpu-workers 4 --num-producers 30 --num-post 2 \
  --chunk-size 256 --timeout-s 2.0 \
  --max-model-len 131072 --gpu-memory-utilization 0.75 \
  >> conf-data-coder-next-confs/prefill-dp-${DS}.log 2>&1 &"
echo "launched prefill for ${DS}"
