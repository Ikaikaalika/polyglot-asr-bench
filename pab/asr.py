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


class MlxWhisper:
    """Apple Silicon GPU backend via MLX (Metal).

    CTranslate2 targets CPU and CUDA only, so faster-whisper cannot reach the GPU on a
    Mac — `device: cuda` simply fails there. mlx_whisper runs the same Whisper weights
    through MLX, which does use the Metal GPU. That makes the on-device question
    measurable: how good, how fast, and how cheap is transcription that never leaves
    the laptop? For contact centers this is a real deployment option, since audio that
    stays on-device sidesteps a large class of privacy and data-residency problems.

    Note the comparison is implementation-vs-implementation (CTranslate2 vs MLX), not
    just hardware — quantization schemes differ, so treat quality deltas as a property
    of the whole stack rather than of the GPU alone.
    """

    def __init__(self, model_repo="mlx-community/whisper-large-v3-mlx",
                 warmup=True, **_ignored):
        import mlx_whisper
        import numpy as np
        self._mlx = mlx_whisper
        self.model_repo = model_repo
        # mlx_whisper loads weights lazily on the FIRST transcribe() call, which would
        # charge model-load time to sample #1's latency and make RTF/p50 unfair versus
        # faster-whisper (which loads in its constructor). Burn that cost here, outside
        # the timed loop, so the two backends are measured on equal footing.
        if warmup:
            self._mlx.transcribe(np.zeros(16000, dtype=np.float32),
                                 path_or_hf_repo=model_repo, verbose=False)

    def transcribe(self, audio, sr, language=None) -> Transcription:
        t0 = time.perf_counter()
        res = self._mlx.transcribe(audio, path_or_hf_repo=self.model_repo,
                                   language=language, verbose=False)
        text = (res.get("text") or "") if isinstance(res, dict) else str(res)
        return Transcription(text=text.strip(),
                             infer_sec=time.perf_counter() - t0)


BACKENDS = {
    "faster_whisper": FasterWhisper,
    "hf_ctc": HFCtc,
    "mlx_whisper": MlxWhisper,
}


def build_backend(spec: dict):
    kind = spec.get("backend", "faster_whisper")
    if kind not in BACKENDS:
        raise ValueError(f"unknown backend {kind!r}; have {list(BACKENDS)}")
    kwargs = {k: v for k, v in spec.items() if k not in ("backend", "name")}
    return BACKENDS[kind](**kwargs)
