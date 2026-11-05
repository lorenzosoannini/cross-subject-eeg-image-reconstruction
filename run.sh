#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo 'Usage: bash run.sh CONFIG CHECKPOINT OUTPUT_DIR [SUBJECT=10]' >&2
  exit 2
fi
config="$1"
checkpoint="$2"
output="$3"
subject="${4:-10}"
weights="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["weights_dir"])' "$config")"
python eval/reconstruct.py --config "$config" --checkpoint "$checkpoint" --output-dir "$output" --subject "$subject"
python eval/metrics.py --input-dir "$output" --mode base
python eval/caption_keyframe.py --input_dir "$output" --cache_dir "$weights" --batch_size 1
python eval/enhance.py --input_dir "$output" --weights_dir "$weights"
python eval/metrics.py --input-dir "$output" --mode enhance
