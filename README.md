# polyglot-asr-bench

A reproducible benchmark for **speech-to-text across many languages**, scored on the
three axes that actually decide production deployment: **quality, latency, and cost**.

Point it at a set of ASR models and a set of multilingual datasets; it transcribes,
scores WER/CER with transparent normalization, measures real-time factor and latency
percentiles, models self-host vs API cost, and emits a leaderboard.

> A focused study of the production speech-model evaluation problem: picking and
> serving transcription models across dozens of languages under quality/cost/latency
> budgets.

## Why it's structured this way

**Compute on the GPU once, analyze on your laptop forever.** Transcription (the only
GPU-bound step) writes predictions to a JSONL cache. All scoring, aggregation, and
plotting run offline from that cache — so the expensive machine only runs while audio
is being transcribed, and every re-run is free and instant. A full multi-language
sweep stays well under ~$50 of rented GPU time.

Two consequences fall out of that design:
- **Idempotent + resumable** — a killed or spot-preempted run picks up exactly where it left off; already-transcribed samples are skipped.
- **One backend, two machines** — `faster-whisper` (CTranslate2) runs int8 on a Mac CPU for the smoke test and float16 on a CUDA GPU for the real sweep, identical code path.

## Quickstart (laptop, ~2 min, no GPU)

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
bash scripts/run_local_smoke.sh
```

Transcribes 10 LibriSpeech clips with `faster-whisper-tiny` (CPU/int8) and prints a
leaderboard — an end-to-end proof of the pipeline.

## Full multilingual sweep (rented GPU)

```bash
# on the GPU box
bash scripts/run_gpu_sweep.sh
# copy results/ back, then locally:
python -m pab.report --gpu-price 0.40
```

Edit `configs/fleurs_multi.yaml` to scale languages up to dozens and add models.

## Metrics

- **WER / CER** — corpus-level, after transparent normalization (`pab/normalize.py`: NFKC, lowercase, strip punctuation, collapse whitespace). Swap the normalizer and every number updates.
- **RTF (real-time factor)** = inference time ÷ audio duration; <1 is faster than real time.
- **Latency** — p50 / p95 per-utterance inference seconds.
- **Cost** — self-host `$/1000min = gpu $/hr × RTF × (1000/60)`; API baselines from published per-minute prices (`pab/cost.py`).

## Contact-center realism

`pab/telephony.py` degrades audio toward call-center conditions — 8 kHz narrowband +
μ-law companding, optional additive noise — so you can measure the WER gap between
clean read speech and telephony-grade audio. Enable via `augmentations:` in a config.

## Layout

```
pab/         library: datasets, asr, metrics, cost, telephony, cache, run, report
configs/     smoke.yaml (laptop) + fleurs_multi.yaml (GPU)
results/     predictions cache + generated leaderboard.md
scripts/     smoke + gpu-sweep runners
docs/        stakeholder report template
```

## Roadmap

- [ ] More model families: Whisper large-v3-turbo, Distil-Whisper, NVIDIA Parakeet/Canary, a commercial API (Deepgram/AssemblyAI) as a cost baseline
- [ ] Quantization sweep: int8 vs fp16 quality/speed tradeoff → Pareto plot
- [ ] Throughput-under-load serving benchmark (vLLM / SGLang)
- [ ] LoRA/PEFT low-resource adaptation track
- [ ] TTS evaluation track (ASR round-trip WER + UTMOS)

## Caveats (read before quoting numbers)

- FLEURS is **read** speech — not conversational or telephony. Clean-FLEURS WER is an upper bound on real-world quality; the telephony augmentation is a proxy, not real call data.
- Normalization choices materially move WER, especially across scripts. The scheme here is deliberately simple and documented — always report which one you used.
- Costs are modeled from measured RTF + list prices (single stream, no batching) — directional, not billed.
```
