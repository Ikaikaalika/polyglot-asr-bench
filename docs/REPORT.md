# Multilingual ASR: Quality, Latency, and Cost on Contact-Center-Style Audio

**Model evaluated:** `whisper-large-v3` via faster-whisper / CTranslate2
**Hardware:** 1× RTX 4090 (24 GB), Secure Cloud, $0.69/hr
**Data:** FLEURS test splits, n=200/language (Tagalog n=150), clean + simulated telephony
**Date:** 2026-07-20 · Repro: `bash scripts/run_gpu_sweep.sh` · raw predictions in `results/`

---

## 1. Executive summary

Three findings drive the recommendations:

1. **int8 quantization is strictly dominated for single-stream decoding.** It costs no
   measurable accuracy (six paired tests, all non-significant) but ran **15–20% slower**,
   making it *more* expensive per minute of audio. Do not quantize for this workload.
2. **Cost per audio-minute varies ~3× across languages** on identical hardware —
   Hindi $1.08 vs Tagalog $0.36 per 1000 min. Language, not just volume, drives spend.
3. **Quality varies 7× across languages** (Spanish 2.4% WER → Hindi 17.5%). A single
   global quality SLA is not achievable with one model; tier by language.

---

## 2. Results

### Quality and cost, clean audio (fp16)

| Language | WER% (95% CI) | CER% | RTF | $/1000 min |
|---|:--|--:|--:|--:|
| Spanish | 2.4 [1.8–3.1] | 1.0 | 0.036 | $0.41 |
| English | 4.7 [3.8–5.5] | 2.4 | 0.038 | $0.43 |
| Tagalog | 12.2 [10.7–13.7] | 3.9 | 0.031 | $0.36 |
| Hindi | 17.5 [15.8–19.2] | 10.5 | 0.094 | $1.08 |

Confidence intervals are bootstrap percentile intervals over utterances (2000 replicates).

### fp16 vs int8 — paired tests on identical utterances

| Language | Aug | ΔWER pp (fp16 − int8) | 95% CI | p | Verdict |
|---|---|--:|:--|--:|---|
| English | clean | −0.05 | [−0.17, +0.09] | 0.543 | n.s. |
| English | telephony | −0.09 | [−0.34, +0.14] | 0.505 | n.s. |
| Spanish | clean | +0.06 | [−0.02, +0.17] | 0.330 | n.s. |
| Spanish | telephony | +0.02 | [−0.06, +0.10] | 0.833 | n.s. |
| Hindi | clean | −0.24 | [−0.96, +0.41] | 0.474 | n.s. |
| Hindi | telephony | −0.61 | [−1.27, +0.00] | 0.051 | borderline |

**No significant quality difference anywhere.** But int8 raised RTF on every language
(en 0.038→0.044, es 0.036→0.043, hi 0.094→0.113), i.e. **+15–20% cost for zero gain**.

> **Why:** at batch size 1 the model is not memory-bandwidth-bound, so int8's smaller
> weights buy nothing, while quantize/dequantize overhead is paid on every op. The
> 4090's fp16 tensor cores are already saturated. Quantization pays off under
> *batched* serving where memory bandwidth dominates — a different regime than this one.

### Telephony degradation (fp16, clean → 8 kHz + μ-law)

| Language | Clean | Telephony | Δ |
|---|--:|--:|--:|
| Spanish | 2.4 | 2.6 | +0.2 |
| English | 4.7 | 5.3 | +0.6 |
| Hindi | 17.5 | 18.8 | +1.3 |

Degradation **compounds with difficulty** — the weakest language loses the most. These
are point estimates; a paired clean-vs-telephony test is listed in §5 as follow-up work.

### The long tail — where quality collapses (large-v3)

Six lower-resource FLEURS languages, run on Apple Silicon GPU (MLX). Khmer and Lao are
written without word spaces, so read **CER**, not WER, for those two.

| Language | n | WER% | CER% | RTF | Failure mode |
|---|--:|--:|--:|--:|---|
| Māori (mi) | 100 | 39.5 | 14.9 | 0.24 | marginal |
| Cebuano (ceb) | 100 | 43.1 | 12.3 | 0.45 | marginal (Whisper has no Cebuano; auto-detected) |
| Yoruba (yo) | 100 | 96.4 | 45.8 | 0.26 | **confident collapse** |
| Amharic (am) | 30 | 188 | 116 | 6.6 | **fallback storm** |
| Khmer (km) | 30 | (n/a) | 101 | 6.3 | **fallback storm** |
| Lao (lo) | 30 | (n/a) | 102 | 1.0 | fallback storm (intermittent) |

**Two distinct ways a model fails on an unsupported language — and they cost differently:**

- **Confident collapse (Yoruba):** normal RTF (0.26), catastrophic quality (96% WER). The
  model doesn't know it's failing, so it transcribes fast, cheap, and almost entirely wrong.
- **Fallback storm (Amharic, Khmer):** RTF explodes to 6–24. Whisper's own quality guard
  retries low-confidence segments at rising temperatures, so a language it can't handle burns
  up to 6 decoding passes per segment — and still produces garbage (often degenerate
  repetition loops). Cost blows up **~40–50×** precisely where quality is worst: Amharic on
  CPU is RTF 23.7 vs Māori's 0.46.

**The operational lesson:** the languages you serve *worst* can be the ones that blow up your
GPU budget. A capacity plan sized on English throughput would be off by an order of magnitude.

### CPU (int8) vs Apple Silicon GPU (MLX) — paired on identical utterances

| Language | ΔWER pp (CPU − GPU) | 95% CI | p | Verdict |
|---|--:|:--|--:|---|
| Cebuano | +0.19 | [−0.35, +0.75] | 0.55 | not significant |
| Māori | +0.39 | [−0.60, +1.32] | 0.49 | not significant |
| Yoruba | +1.39 | [+0.52, +2.23] | 0.002 | significant (small) |
| Amharic | −52.5 | [−100.9, −11.4] | 0.009 | **significant (large)** |

Two takeaways:
- **On languages the model handles, the implementation is interchangeable** — int8-on-CPU and
  MLX-on-GPU are statistically identical (Cebuano, Māori). GPU only buys **speed**: 2.1× on
  healthy Yoruba, 3.4× on the Amharic fallback storm (RTF 23.7 → 6.6).
- **In the degenerate regime they diverge wildly** — a 52-point WER gap on Amharic (both still
  useless, both >100% WER). So *cross-implementation quality comparisons cannot be trusted
  where the model is already failing.* Compare implementations only on inputs the model handles.

**Hardware mitigates the fallback storm; it does not cure it.** Amharic on GPU is still RTF 6.6
— 27× a healthy language. No device (CPU/GPU) or precision (int8/fp16) choice rescues an
unsupported language. The fix is model coverage (MMS, fine-tuning), not compute.

---

## 3. Recommendations

**Serving configuration**
- **Use fp16, not int8**, for single-stream transcription on 4090-class hardware. Revisit
  only if you move to batched serving, where the bandwidth argument reverses.
- Beam size 1 throughout; RTF ≤0.1 across all languages means real-time is comfortable.

**Language tiering** — one SLA does not fit all:

| Tier | Languages | WER | Guidance |
|---|---|---|---|
| Production | Spanish, English | 2–5% | Ship; suitable for automated downstream use |
| Marginal | Tagalog | ~12% | Usable for search/analytics; risky for automated actions |
| Not ready | Hindi | ~18% | Needs adaptation (fine-tuning/LoRA) or a specialist model |

**Cost planning**
- Budget **per language**, not per minute of audio: Hindi costs ~3× Tagalog on the same
  GPU. Token density drives autoregressive decode steps, which drives cost.
- At $0.69/hr and RTF ~0.04, self-hosting is ~$0.41/1000 min for Spanish — below typical
  commercial API list pricing (~$3.70–6.00/1000 min), so self-hosting wins on unit cost
  at sustained volume. Add a commercial API baseline before committing (see §5).

---

## 4. Method notes

- **Corpus WER** = Σ edit distance ÷ Σ reference length, not the mean of per-utterance
  rates (which over-weights short utterances).
- **Normalization** (`pab/normalize.py`): NFKC, lowercase, punctuation stripped,
  whitespace collapsed. Deliberately simple and documented — normalization choices move
  WER by points, so any comparison must state its scheme.
- **Paired testing**: fp16 and int8 were compared on identical utterances, so
  per-utterance difficulty cancels. This matters — in a side experiment, two systems
  with heavily overlapping marginal CIs ([1.9–9.4] vs [5.8–14.9]) showed a *significant*
  paired difference (p=0.025). **Overlapping error bars do not imply no difference.**
- **Cost model**: `$/1000min = gpu $/hr × RTF × (1000/60)`, single-stream, no batching.

---

## 5. Limitations and follow-ups

Read these before quoting any number.

1. **FLEURS is read speech.** It is not conversational, not spontaneous, and not real
   call audio. Treat every WER here as an **upper bound on quality** for contact-center
   traffic. The telephony augmentation is a codec/bandwidth proxy, not real call data.
2. **Telephony is simulated** (8 kHz + μ-law + optional noise), which omits packet loss,
   jitter, echo, crosstalk, and far-field effects.
3. **n=200 gives ±~1 pp intervals.** Differences below ~1 pp are not resolvable at this
   sample size; scale n before making fine-grained claims.
4. **Timing is single-run.** RTF was measured once per configuration, not repeated, so
   the 15–20% int8 slowdown — while consistent across all six configurations — has no
   error bar. Repeat-measure before publishing it as a headline.
5. **No commercial API baseline yet**, so the self-host-vs-buy comparison uses published
   list prices rather than measured ones.
6. **Long-tail sample sizes are uneven.** Cebuano, Māori, and Yoruba ran at n=100;
   Amharic, Khmer, and Lao at n=30 (the fallback-storm languages, where n=30 already
   settles "is this usable at all"). Small-n rows carry wide CIs — e.g. Amharic WER
   [149–232]. Khmer/Lao are word-space-free, so their WER is meaningless; use CER.
   Also: FLEURS shard downloads for these configs are pathologically slow (~30 min for
   Māori's first sample), so this tier runs locally, not on rented GPU — the workload is
   download-bound, not compute-bound.
7. **Hawaiian and Tongan cannot be evaluated here at all** — neither is in FLEURS, and
   no reference-transcript speech corpus is readily available. Benchmarking them would
   require a different approach (MMS warm start + aligned Bible audio), not this harness.

**Highest-value next steps:** (a) paired clean-vs-telephony significance tests,
(b) a commercial API baseline for the buy-vs-build number, (c) repeat-measure timing,
(d) finish the long-tail tier.
