"""ASR backends.

A backend takes 16 kHz mono float32 audio and returns text + wall-clock inference
time. faster-whisper (CTranslate2) is the default because one backend covers both
machines: int8 on a Mac CPU for the smoke test, float16 on a CUDA GPU for the real
sweep. New backends (openai-whisper, MMS, Parakeet, a cloud API) just implement
`.transcribe()` and register in BACKENDS.
"""
from __future__ import annotations
import time
from dataclasses import dataclass


@dataclass
class Transcription:
    text: str
    infer_sec: float


class FasterWhisper:
    def __init__(self, model_size="tiny", device="cpu", compute_type="int8",
                 beam_size=1, **_ignored):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, device=device,
                                  compute_type=compute_type)
        self.beam_size = beam_size

    def transcribe(self, audio, sr, language=None) -> Transcription:
        t0 = time.perf_counter()
        segments, _info = self.model.transcribe(
            audio, language=language, beam_size=self.beam_size)
        text = "".join(seg.text for seg in segments)   # consume the generator
        return Transcription(text=text.strip(), infer_sec=time.perf_counter() - t0)


BACKENDS = {"faster_whisper": FasterWhisper}


def build_backend(spec: dict):
    kind = spec.get("backend", "faster_whisper")
    if kind not in BACKENDS:
        raise ValueError(f"unknown backend {kind!r}; have {list(BACKENDS)}")
    kwargs = {k: v for k, v in spec.items() if k not in ("backend", "name")}
    return BACKENDS[kind](**kwargs)
