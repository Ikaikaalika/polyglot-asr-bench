#!/usr/bin/env bash
# Long-tail / low-resource language tier. Run after scripts/setup_pod.sh.
# TERMINATE THE POD when done.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/_cuda_env.sh

# huggingface_hub's read timeout defaults to 10s. On an unauthenticated connection a
# slow shard fetch blows past that and the sweep stalls with the GPU idle at 0% — an
# expensive, silent failure (the process stays alive and the log stops advancing).
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

python3 -m pab.run --config configs/fleurs_lowresource.yaml
python3 -m pab.report --results-dir results --gpu-price 0.69 --compare

echo
echo ">>> Done. Copy results/ back to your laptop, then TERMINATE this pod."
