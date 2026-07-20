# Speech AI — a working roadmap to depth

A study plan for genuinely understanding the models and methods this benchmark
exercises, not just running them. Ordered by dependency: each tier explains a stage
of the pipeline in `pab/`, and each has an exercise you can run **in this repo**.

Assumed background: transformers, autodiff, optimization, probability. The gap this
closes is *speech-specific* structure — the DSP front-end, the alignment paradigms,
and evaluation science.

**Method that actually works:** read the paper → implement the smallest piece from
scratch → run it against a real model in this harness → write down what surprised you.
Reading alone produces recognition, not understanding.

---

## The problem, stated precisely

ASR maps a waveform `x` (16,000 samples/sec) to a token sequence `y`. Two structural
facts drive every design decision:

1. **Length mismatch** — audio frames ≫ text tokens.
2. **Monotonic alignment** — audio and text both advance left-to-right.

Every architecture below is a different answer to: *how do we model `P(y|x)` when the
alignment between them is latent and monotonic?*

---

## Tier A — The front-end (waveform → features)

**Where it lives:** `pab/datasets.py` decodes to 16 kHz mono float32; faster-whisper
computes log-Mel internally.

**Concepts:** sampling & Nyquist, convolution, DFT/FFT, STFT, windowing (Hann),
mel filterbanks, MFCCs, dynamic range compression.

**Core math:**
- STFT: frame into ~25 ms windows, hop 10 ms, window, FFT → magnitude spectrum.
- Mel warping: `mel(f) = 2595 · log10(1 + f/700)`, applied as triangular filterbanks
  (a linear projection of the spectrum).
- Take `log` of filterbank energies.
- Whisper: 30 s chunk → 3000 frames × 80 mel bins (128 for large-v3); two stride-2
  convs downsample time before the Transformer encoder.

**Nuance to own:** the time–frequency uncertainty (Gabor) limit — narrow windows buy
time resolution at the cost of frequency resolution. The `log` both matches perceptual
loudness and converts multiplicative gain/noise into additive shifts (which is why
mean-normalization works).

- [ ] **Exercise:** implement STFT + mel filterbank in NumPy; plot the spectrogram of a
      FLEURS clip pulled through `pab.datasets.load_dataset_spec`. Compare your mel
      matrix to the model's expected input shape.
- [ ] **Exercise:** run the `telephony` augmentation (`pab/telephony.py`) and look at
      what 8 kHz + μ-law does to the spectrogram. Predict the WER hit, then measure it.

---

## Tier B — Alignment paradigms (the heart of ASR)

**Where it lives:** `pab/asr.py` — `faster_whisper` (attention) and `hf_ctc` (CTC).

All three families maximize `log P(y|x)` while marginalizing a latent monotonic
alignment `a`:  **`P(y|x) = Σ_a P(y, a | x)`**. They differ only in how `a` is
structured and whether outputs are conditionally independent.

| Paradigm | Alignment handling | Output independence | Streams? | Examples |
|---|---|---|---|---|
| **CTC** | per-frame softmax over vocab ∪ {blank}; sum over collapsing alignments via forward algorithm | **yes** | yes | wav2vec2, MMS |
| **Attention seq2seq** | soft attention; autoregressive decoder | no | no | Whisper, LAS |
| **RNN-T / Transducer** | encoder + prediction net + joiner; sums alignments *and* conditions on prior output | no | **yes** | Parakeet, most production streaming ASR |

**Core math:**
- **CTC loss:** `P(y|x) = Σ_{π ∈ B⁻¹(y)} Π_t P(π_t | x)` where `B` collapses repeats and
  removes blanks. Computed with a forward–backward DP over an extended label sequence.
  The conditional-independence assumption is the price of the speedup.
- **RNN-T:** same marginalization over a 2-D lattice (time × labels), but the prediction
  network conditions on previously emitted tokens — no independence assumption, and it
  streams. **This is what contact centers run.**
- **Attention:** teacher-forced cross-entropy; no explicit alignment sum. Buys full
  `P(y)` modeling, pays with hallucination/looping failure modes and non-streaming.

**Read:** Graves 2006 (CTC) · distill.pub *"Sequence Modeling with CTC"* (derive the
forward algorithm by hand — worth a full afternoon) · Graves 2012 (RNN-T) ·
Chan 2015 (Listen, Attend & Spell) · Radford 2022 (Whisper), alongside
`whisper/model.py`, which is short and readable.

- [ ] **Exercise:** `python -m pab.run --config configs/ctc_compare.yaml` — CTC
      (wav2vec2) vs attention (Whisper) on identical audio. Inspect the JSONL: where
      does CTC's independence assumption visibly hurt? Where does Whisper hallucinate?
- [ ] **Exercise:** implement the CTC forward algorithm on a toy example (vocab {a,b},
      3 frames) and verify it sums to the same value as brute-force enumeration.
- [ ] **Exercise:** add a transducer backend (NVIDIA Parakeet via NeMo) and compare all
      three paradigms in one leaderboard.

---

## Tier C — Decoding

**Where it lives:** `beam_size` in the model specs.

**Concepts:** greedy vs beam search, length normalization, temperature, shallow fusion
with an external LM (`score = log P(y|x) + λ·log P_LM(y)`), CTC prefix beam search,
Whisper's fallback heuristics (temperature bump on low avg-logprob / bad compression
ratio, to escape loops).

- [ ] **Exercise:** sweep `beam_size` 1→5 and plot WER vs RTF. You'll see the
      accuracy/latency knob that production teams actually turn.

---

## Tier D — Evaluation science *(go deepest here — this IS the job)*

**Where it lives:** `pab/metrics.py`, `pab/normalize.py`, `pab/report.py`.

**Core math:**
- `WER = (S + D + I) / N` — minimum edit (Levenshtein) distance between reference and
  hypothesis word sequences ÷ reference length, computed by DP.
- Corpus aggregation is `Σ edits / Σ ref_len`, **not** the mean of per-utterance WERs
  (the latter over-weights short utterances).

**Nuance that separates an evaluator from a script-runner:**
- WER is unbounded above (insertions), asymmetric, and is an **estimator** — comparing
  two models needs a **bootstrap confidence interval** or a matched-pairs significance
  test (MAPSSWE), not a bare point estimate.
- Normalization dominates. Casing, punctuation, numerals ("21" vs "twenty-one"),
  contractions, and diacritics can move WER by several points. Half of reproducing any
  published number is matching their normalizer.
- CER matters for morphologically rich or space-free languages; per-language WER is not
  comparable across scripts without care.

**Read:** the HF **Open ASR Leaderboard** methodology · the **ESB** benchmark paper ·
Whisper's `EnglishTextNormalizer`.

### Worked example — a WER gap that isn't a quality gap

Running `configs/ctc_compare.yaml` on 10 LibriSpeech clips gives:

| model | WER% | CER% |
|---|--:|--:|
| wav2vec2-base-960h (CTC) | 6.2 | 2.5 |
| faster-whisper-tiny (attention) | 10.4 | 5.1 |

Naive reading: "CTC is 40% better." Look at the actual output and that collapses:

```
REF : MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES ...
CTC : MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES ...
WHSP: Mr. Kfilter is the apostle of the middle classes ...
```

Two confounds, both fatal to the naive conclusion:

1. **Orthographic convention, not recognition error.** Whisper writes `Mr.`; the
   LibriSpeech reference says `MISTER`. Our normalizer strips case and punctuation but
   does *not* map `mr → mister`, so a correct transcription is scored as an error.
   This is exactly what Whisper's `EnglishTextNormalizer` exists to fix.
2. **Train/test domain overlap.** `wav2vec2-base-960h` was fine-tuned *on LibriSpeech*,
   so it reproduces that corpus's exact orthography. Whisper is zero-shot here. The
   benchmark is measuring in-domain advantage as if it were general quality.

**The lesson:** a leaderboard number is a claim about a *pipeline* (model + normalizer +
domain), not about a model. Before you report any comparison, ask what else differs
besides the thing you think you're measuring. This is the single most common way ASR
benchmarks mislead — and catching it is what the job actually is.

- [ ] **Exercise:** add bootstrap 95% CIs to `pab/report.py` by resampling utterances in
      the predictions JSONL. Are your model differences actually significant at n=200?
- [ ] **Exercise:** swap in a stricter normalizer and re-score. Quantify how much WER
      moved. This is why the README insists you report which normalizer you used.
- [ ] **Exercise:** reproduce one Open ASR Leaderboard number exactly. This forces you
      to get normalization *and* decoding right, and is the single best proof of rigor.

---

## Tier E — Efficiency: quantization, distillation, serving

**Where it lives:** `compute_type` in the model specs; `pab/cost.py`.

**Core math:**
- **Quantization** (symmetric int8): `x_q = round(x / s)`, `s = max|x| / 127`. Matmuls in
  int8 give ~2–4× speed and memory wins for a small accuracy cost.
  Nuance: per-tensor vs **per-channel** scales; static (calibrated) vs dynamic;
  activation **outliers** are the hard part (LLM.int8(), SmoothQuant, GPTQ/AWQ).
  CTranslate2's `int8_float16` keeps sensitive ops in fp16.
- **Distillation:** student matches teacher soft logits, `L = KL(teacher ‖ student) +
  task loss`. Distil-Whisper's win comes from cutting *decoder* layers — the
  autoregressive decoder is the latency bottleneck, not the encoder.
- **Serving:** PagedAttention (KV-cache paging) + continuous batching (vLLM/SGLang).
- **Cost:** `$/1000min = gpu $/hr × RTF × (1000/60)` — the bridge from FLOPs to dollars.

- [ ] **Exercise:** compare the `int8` vs `fp16` rows in your leaderboard. Quantify
      ΔWER per unit speedup — that ratio is the whole deployment argument.
- [ ] **Exercise:** take one weight tensor, quantize/dequantize it by hand, and measure
      the reconstruction error per-tensor vs per-channel.

---

## Tier F — TTS, codecs, speech-to-speech

**Concepts:** text → mel (Tacotron2, FastSpeech2, VITS) → waveform (HiFi-GAN), or
end-to-end. **VITS** is the math-rich one: VAE + normalizing flows + adversarial
training + monotonic alignment search. Frontier: neural codec LMs (EnCodec/SoundStream
use **residual vector quantization**; VALL-E is an LM over those tokens) and
**flow-matching TTS** (F5-TTS).

> Your flow-matching background transfers directly here — this is the tier where your
> existing expertise is an edge rather than a gap.

**Evaluation:** MOS (human) · **UTMOS** (learned MOS predictor) · intelligibility via
**ASR round-trip WER** · speaker similarity via embedding cosine.

- [ ] **Exercise:** synthesize a sentence with an open TTS (Kokoro/XTTS), transcribe it
      through this harness, and compute round-trip WER. You've just built a TTS
      intelligibility metric out of an ASR benchmark.

---

## Canonical resources

- **Jurafsky & Martin, _Speech and Language Processing_ (3rd ed. draft, free online)** —
  the textbook. Feature extraction, ASR, TTS chapters.
- **HuggingFace Audio Course** — free, hands-on, and uses this exact stack.
- **distill.pub — Sequence Modeling with CTC** — the best explanation of CTC anywhere.
- **Stanford CS224S** — Spoken Language Processing lectures.
- Papers: Graves 2006 (CTC) · Graves 2012 (RNN-T) · Chan 2015 (LAS) · Radford 2022
  (Whisper) · Baevski 2020 (wav2vec 2.0) · Pratap 2023 (MMS) · Kim 2021 (VITS) ·
  Défossez 2022 (EnCodec) · Kwon 2023 (vLLM/PagedAttention).

---

## Milestones — you know it when you can…

- [ ] Explain why log-Mel and not raw waveform or linear spectrogram, including the
      Gabor limit tradeoff.
- [ ] Derive the CTC forward algorithm and state exactly what its independence
      assumption costs.
- [ ] Explain why transducers stream and attention seq2seq doesn't.
- [ ] Predict, before running it, whether int8 or beam width will hurt WER more.
- [ ] Defend a WER difference with a confidence interval instead of a point estimate.
- [ ] Recommend a model for a given language/latency/cost budget and show the numbers.
