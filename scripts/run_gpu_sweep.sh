#!/usr/bin/env bash
# Run on a rented CUDA GPU AFTER scripts/setup_pod.sh. STOP THE POD when done.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/_cuda_env.sh

echo ">>> GPU smoke (1 language, 5 samples) — fail fast before the big run"
python3 -m pab.run --config configs/gpu_smoke.yaml

echo ">>> Full multilingual sweep"
python3 -m pab.run --config configs/fleurs_multi.yaml

echo ">>> Leaderboard"
python3 -m pab.report --results-dir results --gpu-price 0.40

echo
echo ">>> Done. Copy results/ back to your laptop, then STOP THIS POD."
