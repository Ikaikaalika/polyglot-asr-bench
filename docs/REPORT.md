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
6. **Long tail incomplete.** Cebuano, Māori, Yoruba, Amharic, Khmer, and Lao are
   configured (`configs/fleurs_lowresource.yaml`) but did not complete — FLEURS shard
   downloads for those configs stalled repeatedly. Khmer and Lao additionally need
   **CER, not WER**, since they are written without word spaces.
7. **Hawaiian and Tongan cannot be evaluated here at all** — neither is in FLEURS, and
   no reference-transcript speech corpus is readily available. Benchmarking them would
   require a different approach (MMS warm start + aligned Bible audio), not this harness.

**Highest-value next steps:** (a) paired clean-vs-telephony significance tests,
(b) a commercial API baseline for the buy-vs-build number, (c) repeat-measure timing,
(d) finish the long-tail tier.
