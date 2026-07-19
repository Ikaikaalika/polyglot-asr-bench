#!/usr/bin/env bash
# One-shot setup on a RunPod CUDA pod (RTX 4090 / L4 / A10, any CUDA 12 image).
# From the repo root:  bash scripts/setup_pod.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo ">>> [1/4] GPU visible?"
nvidia-smi -L || { echo "No GPU detected — launch a GPU pod, not CPU."; exit 1; }

PY="${PAB_PY:-python3}"

echo ">>> [2/4] Installing benchmark + CUDA runtime libs"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r requirements.txt
# CTranslate2 (faster-whisper's engine) needs cuBLAS + cuDNN 9 at runtime:
"$PY" -m pip install --quiet nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"

echo ">>> [3/4] Putting CUDA libs on the loader path"
# shellcheck disable=SC1091
source scripts/_cuda_env.sh
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<empty>}"

echo ">>> [4/4] CUDA sanity check (load faster-whisper on GPU)"
"$PY" - <<'PYEOF'
from faster_whisper import WhisperModel
WhisperModel("tiny", device="cuda", compute_type="float16")
print("OK: faster-whisper runs on CUDA")
PYEOF

echo
echo ">>> Setup complete. Next:  bash scripts/run_gpu_sweep.sh"
