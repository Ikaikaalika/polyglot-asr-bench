#!/usr/bin/env bash
# Laptop smoke test: transcribe 10 clips on CPU and print a leaderboard (~2 min).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pab.run --config configs/smoke.yaml
python -m pab.report --results-dir results
