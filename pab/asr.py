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


class HFCtc:
    """CTC backend (wav2vec2 / MMS) via transformers.

    The point of having this next to Whisper: CTC and attention seq2seq solve the same
    problem — max log P(y|x) marginalizing a monotonic alignment — with opposite
    tradeoffs. CTC emits one frame-level distribution per timestep over vocab u {blank}
    and sums over all alignments that collapse to y, which makes it non-autoregressive
    and streamable, but assumes outputs are conditionally independent given the audio.
    Decoding here is the greedy CTC path: argmax per frame, then collapse repeats and
    drop blanks (what `batch_decode` does).

    For multilingual MMS, pass `target_lang` (e.g. "eng", "spa") to load its adapter.
    """

    def __init__(self, model_id="facebook/wav2vec2-base-960h", device="cpu",
                 target_lang=None, **_ignored):
        import torch
        from transformers import AutoProcessor, AutoModelForCTC
        self.torch = torch
        self.device = device
        proc_kw = {"target_lang": target_lang} if target_lang else {}
        model_kw = ({"target_lang": target_lang, "ignore_mismatched_sizes": True}
                    if target_lang else {})
        self.processor = AutoProcessor.from_pretrained(model_id, **proc_kw)
        self.model = AutoModelForCTC.from_pretrained(model_id, **model_kw)
        self.model = self.model.to(device).eval()

    def transcribe(self, audio, sr, language=None) -> Transcription:
        t0 = time.perf_counter()
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
        ids = logits.argmax(dim=-1)                       # greedy CTC path
        text = self.processor.batch_decode(ids)[0]
        return Transcription(text=text.strip(), infer_sec=time.perf_counter() - t0)


BACKENDS = {"faster_whisper": FasterWhisper, "hf_ctc": HFCtc}


def build_backend(spec: dict):
    kind = spec.get("backend", "faster_whisper")
    if kind not in BACKENDS:
        raise ValueError(f"unknown backend {kind!r}; have {list(BACKENDS)}")
    kwargs = {k: v for k, v in spec.items() if k not in ("backend", "name")}
    return BACKENDS[kind](**kwargs)
