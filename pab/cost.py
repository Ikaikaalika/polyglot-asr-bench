"""GPU cost modeling.

Turns a measured real-time factor (RTF) into $ per 1000 minutes of audio, and
compares self-hosting against per-minute API pricing.

Self-host: to process 1000 min of audio at real-time-factor RTF, you spend
RTF * 1000 min of GPU time = RTF * 1000/60 GPU-hours. So:
    $/1000min = gpu_price_per_hr * RTF * (1000/60)
This ignores batching/concurrency (which improve throughput) and cold-start, so it
is a conservative single-stream estimate — directional, not an invoice.
"""
from __future__ import annotations

MIN_PER_1000 = 1000.0 / 60.0   # GPU-hours per 1000 audio-min at RTF=1


def self_host_cost_per_1000min(rtf: float, gpu_price_per_hr: float) -> float:
    return gpu_price_per_hr * rtf * MIN_PER_1000


def api_cost_per_1000min(price_per_min: float) -> float:
    return price_per_min * 1000.0


# Rough published list prices ($/min of audio). Update from vendor pages before quoting.
API_PRICES = {
    "openai_whisper_api": 0.006,
    "deepgram_nova": 0.0043,
    "assemblyai": 0.0037,
}
