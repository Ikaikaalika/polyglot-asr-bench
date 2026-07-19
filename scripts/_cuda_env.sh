# Source this (don't execute) to put CTranslate2's CUDA libs on the loader path.
# No-op on a CPU box, so it's safe to source everywhere.
if command -v nvidia-smi >/dev/null 2>&1; then
  _pab_py="${PAB_PY:-python3}"
  _cudnn=$("$_pab_py" -c "import os,nvidia.cudnn;print(os.path.join(os.path.dirname(nvidia.cudnn.__file__),'lib'))" 2>/dev/null || true)
  _cublas=$("$_pab_py" -c "import os,nvidia.cublas;print(os.path.join(os.path.dirname(nvidia.cublas.__file__),'lib'))" 2>/dev/null || true)
  if [ -n "${_cudnn:-}" ]; then
    export LD_LIBRARY_PATH="${_cudnn}:${_cublas:-}:${LD_LIBRARY_PATH:-}"
  fi
fi
