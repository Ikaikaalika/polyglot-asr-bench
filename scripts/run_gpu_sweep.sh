#!/usr/bin/env bash
# Run on a rented CUDA GPU, then STOP THE POD as soon as it finishes.
# Scoring is done later on your laptop from the copied-back results/ dir.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pab.run --config configs/fleurs_multi.yaml
# transcription is the only GPU-bound step; you can report here or locally:
python -m pab.report --results-dir results --gpu-price 0.40
echo
echo ">>> Done. Copy results/ back to your laptop and stop this GPU pod. <<<"
