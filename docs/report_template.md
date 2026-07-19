# Multilingual ASR for Contact-Center Audio — Quality / Cost / Latency

*Draft eval report. This is the stakeholder-facing deliverable — the leaderboard is
evidence; this doc is the recommendation.*

## 1. Question
Which speech-to-text model(s) should we deploy for <languages>, given targets of
<WER budget>, <latency budget>, and <cost budget>?

## 2. Setup
- **Models:** …
- **Datasets / languages:** … (note domain: read speech vs conversational vs telephony)
- **Metrics:** corpus WER/CER (normalization: `pab/normalize.py`), RTF, p50/p95 latency, self-host $/1000min (GPU $/hr = …).
- **Hardware:** … (GPU, compute_type)

## 3. Results
<paste results/leaderboard.md>

<Pareto plot: WER vs $/1000min, and WER vs p95 latency>

## 4. Findings
- Quality: …
- Clean vs telephony gap: …
- Quantization (int8 vs fp16): quality delta … for speedup … .
- Self-host vs API crossover: below … min/month, API wins; above, self-host wins.

## 5. Recommendation
- **Default:** use **<model>** for <languages> — <why, tied to the budgets>.
- **Watch-outs:** <languages where WER is unacceptable>, <domain gap caveats>.

## 6. Caveats
- FLEURS is read speech; treat as an upper bound on real-call quality.
- Costs modeled from measured RTF + list prices, single-stream (no batching) — directional.
