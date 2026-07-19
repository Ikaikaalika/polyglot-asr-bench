"""Audio degradations that approximate contact-center conditions.

Real call-center audio is narrowband (8 kHz), companded (μ-law), and noisy. Read-speech
benchmarks like FLEURS are none of those, so clean WER overstates production quality.
These transforms let you measure the gap. Proxy, not a substitute for real call data.
"""
from __future__ import annotations
import numpy as np


def _resample(a, src, dst):
    a = np.asarray(a, dtype=np.float32)
    if src == dst:
        return a
    try:
        import librosa
        return librosa.resample(a, orig_sr=src, target_sr=dst).astype(np.float32)
    except Exception:
        n = int(round(len(a) * dst / src))
        return np.interp(np.linspace(0, len(a), n, endpoint=False),
                         np.arange(len(a)), a).astype(np.float32)


def mulaw(a, mu=255):
    """μ-law compand + expand — the lossy companding a phone line applies."""
    a = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)
    comp = np.sign(a) * np.log1p(mu * np.abs(a)) / np.log1p(mu)
    q = np.round((comp + 1) / 2 * mu)                 # quantize to 8 bits
    xq = 2 * q / mu - 1
    exp = np.sign(xq) * (1.0 / mu) * ((1 + mu) ** np.abs(xq) - 1)
    return exp.astype(np.float32)


def telephony(a, sr=16000):
    """16k -> 8k narrowband -> μ-law -> back to 16k."""
    down = _resample(a, sr, 8000)
    return _resample(mulaw(down), 8000, sr)


def add_noise(a, snr_db=10.0, seed=0):
    a = np.asarray(a, dtype=np.float32)
    rng = np.random.default_rng(seed)
    sig_p = float(np.mean(a ** 2)) + 1e-12
    noise_p = sig_p / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_p), size=a.shape).astype(np.float32)
    return (a + noise).astype(np.float32)


AUGS = {
    "none": lambda a, sr: a,
    "telephony": lambda a, sr: telephony(a, sr),
    "noise10": lambda a, sr: add_noise(a, 10.0),
    "telephony_noise": lambda a, sr: add_noise(telephony(a, sr), 10.0),
}
