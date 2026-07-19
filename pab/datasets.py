"""Dataset loaders.

Each loader yields normalized sample dicts:
    {"id": str, "audio": np.float32 mono @16k, "sr": 16000, "ref": str, "language": str}

Audio is decoded with soundfile via `Audio(decode=False)` rather than letting
`datasets` auto-decode. That deliberately sidesteps the torchcodec/ffmpeg backend
dance in recent `datasets` versions — we control decoding, so the smoke test needs
nothing beyond soundfile.
"""
from __future__ import annotations
import io
import numpy as np

TARGET_SR = 16000


def _to_mono16k(array, sr) -> np.ndarray:
    a = np.asarray(array, dtype=np.float32)
    if a.ndim > 1:                       # stereo -> mono
        a = a.mean(axis=1)
    if sr != TARGET_SR:
        try:
            import librosa
            a = librosa.resample(a, orig_sr=sr, target_sr=TARGET_SR)
        except Exception:                # cheap linear fallback if no librosa
            n = int(round(len(a) * TARGET_SR / sr))
            a = np.interp(
                np.linspace(0, len(a), n, endpoint=False),
                np.arange(len(a)), a,
            ).astype(np.float32)
    return a.astype(np.float32)


def load_hf(hf_id, config, split, text_key, language,
            max_samples=None, **_ignored):
    import soundfile as sf
    from datasets import load_dataset, Audio

    ds = load_dataset(hf_id, config, split=split)
    ds = ds.cast_column("audio", Audio(decode=False))   # raw bytes/path, no decode
    n = len(ds) if max_samples is None else min(max_samples, len(ds))
    for i in range(n):
        ex = ds[i]
        a = ex["audio"]
        if a.get("bytes"):
            arr, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        else:
            arr, sr = sf.read(a["path"], dtype="float32")
        yield {
            "id": f"{i}:{ex.get('id', i)}"[:120],
            "audio": _to_mono16k(arr, sr),
            "sr": TARGET_SR,
            "ref": ex.get(text_key) or "",
            "language": language,
        }


def load_dataset_spec(spec):
    kind = spec.get("kind", "hf")
    if kind == "hf":
        yield from load_hf(
            hf_id=spec["hf_id"],
            config=spec.get("config"),
            split=spec.get("split", "test"),
            text_key=spec.get("text_key", "text"),
            language=spec.get("language", "und"),
            max_samples=spec.get("max_samples"),
        )
    else:
        raise ValueError(f"unknown dataset kind: {kind!r}")
