#!/bin/bash
# Aggregate probe traces and upload final pkl to S3.
# Usage: ./aggregate_and_upload.sh <probe_subdir> <conf_data_dir> <s3_name>
#   probe_subdir:  subdir under probe_results/ (e.g. brumo25_thinking)
#   conf_data_dir: subdir under conf-data-coder-next-confs/ (e.g. brumo25) — optional, for confs merge
#   s3_name:       filename under s3://.../deepconf_probe_traces/ (e.g. brumo25_thinking.pkl)
set -eu
SUBDIR=$1
CONF_DS=$2
S3NAME=$3
cd /opt/dlami/nvme/projects/deepconf

source /opt/dlami/nvme/miniconda3/etc/profile.d/conda.sh
conda activate deepconf

echo "[aggregate] merging traces + bedrock confs for $SUBDIR..."
python probe_src/aggregate_traces.py \
  "probe_results/${SUBDIR}/traces" \
  --conf-data-dir "conf-data-coder-next-confs/${CONF_DS}"

PKL="probe_results/${SUBDIR}/aggregated/aggregated_traces.pkl"
if [ ! -f "$PKL" ]; then
  echo "[aggregate] ERROR: $PKL not found" >&2
  exit 1
fi

echo "[upload] $PKL -> s3://drmyang-training-data-241580540779-us-east-2-an/deepconf_probe_traces/${S3NAME}"
aws s3 cp "$PKL" \
  "s3://drmyang-training-data-241580540779-us-east-2-an/deepconf_probe_traces/${S3NAME}"

echo "[done] uploaded ${S3NAME}"
