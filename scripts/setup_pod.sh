#!/usr/bin/env bash
# One-shot setup on a RunPod CUDA pod (RTX 4090 / L4 / A10, any CUDA 12 image).
# From the repo root:  bash scripts/setup_pod.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo ">>> [1/4] GPU visible?"
nvidia-smi -L || { echo "No GPU detected — launch a GPU pod, not CPU."; exit 1; }

# Unauthenticated HF Hub downloads are rate-limited and can stall mid-sweep. That
# failure is expensive and silent: the process stays alive, the log stops advancing,
# and you keep paying for a GPU sitting at 0% utilization. Set HF_TOKEN to avoid it.
if [ -z "${HF_TOKEN:-}" ]; then
  echo "    WARNING: HF_TOKEN not set — dataset downloads may rate-limit and stall"
  echo "             the sweep with the GPU idle. Export HF_TOKEN before running."
fi

PY="${PAB_PY:-python3}"

# Modern base images (Ubuntu 24.04+) ship PEP 668 EXTERNALLY-MANAGED, which makes pip
# refuse to install outside a venv. In an ephemeral GPU container that guard buys us
# nothing, so opt out rather than layering a venv over the image's CUDA stack.
PIP_FLAGS=""
if "$PY" -c "import os,sys,sysconfig; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()['stdlib'],'EXTERNALLY-MANAGED')) else 1)"; then
  PIP_FLAGS="--break-system-packages"
  echo "    (PEP 668 detected -> using --break-system-packages)"
fi

echo ">>> [2/4] Installing benchmark + CUDA runtime libs"
"$PY" -m pip install --quiet $PIP_FLAGS --upgrade pip
"$PY" -m pip install --quiet $PIP_FLAGS -r requirements.txt
# CTranslate2 (faster-whisper's engine) needs cuBLAS + cuDNN 9 at runtime:
"$PY" -m pip install --quiet $PIP_FLAGS nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"

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
