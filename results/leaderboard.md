| model | dataset | aug | n | WER% (95% CI) | CER% | RTF | p50 s | p95 s | $/1000min |
|---|---|---|--:|:--|--:|--:|--:|--:|--:|
| faster-whisper-large-v3-fp16 | fleurs-en | none | 200 | 4.7 [3.8–5.5] | 2.4 | 0.038 | 0.33 | 0.42 | 0.43 |
| faster-whisper-large-v3-int8 | fleurs-en | none | 200 | 4.7 [3.9–5.6] | 2.4 | 0.044 | 0.38 | 0.52 | 0.50 |
| faster-whisper-large-v3-fp16 | fleurs-en | telephony | 200 | 5.3 [4.4–6.2] | 2.8 | 0.037 | 0.33 | 0.44 | 0.43 |
| faster-whisper-large-v3-int8 | fleurs-en | telephony | 200 | 5.4 [4.4–6.3] | 2.8 | 0.044 | 0.39 | 0.53 | 0.51 |
| faster-whisper-large-v3-fp16 | fleurs-en-smoke | none | 5 | 5.8 [2.7–9.5] | 1.9 | 0.049 | 0.28 | 0.60 | 0.56 |
| faster-whisper-large-v3-int8 | fleurs-es | none | 200 | 2.3 [1.7–3.0] | 1.0 | 0.043 | 0.49 | 0.73 | 0.50 |
| faster-whisper-large-v3-fp16 | fleurs-es | none | 200 | 2.4 [1.8–3.1] | 1.0 | 0.036 | 0.41 | 0.58 | 0.41 |
| faster-whisper-large-v3-int8 | fleurs-es | telephony | 200 | 2.6 [1.9–3.3] | 1.1 | 0.043 | 0.49 | 0.71 | 0.49 |
| faster-whisper-large-v3-fp16 | fleurs-es | telephony | 200 | 2.6 [2.0–3.3] | 1.0 | 0.035 | 0.41 | 0.58 | 0.40 |
| faster-whisper-large-v3-fp16 | fleurs-hi | none | 200 | 17.5 [15.8–19.2] | 10.5 | 0.094 | 0.90 | 1.88 | 1.08 |
| faster-whisper-large-v3-int8 | fleurs-hi | none | 200 | 17.7 [16.0–19.6] | 10.5 | 0.113 | 1.13 | 2.62 | 1.30 |
| faster-whisper-large-v3-fp16 | fleurs-hi | telephony | 200 | 18.8 [17.2–20.5] | 11.3 | 0.099 | 0.88 | 2.35 | 1.14 |
| faster-whisper-large-v3-int8 | fleurs-hi | telephony | 200 | 19.4 [17.6–21.4] | 11.7 | 0.123 | 1.13 | 3.03 | 1.41 |
| faster-whisper-large-v3-fp16 | fleurs-tl | none | 150 | 12.2 [10.7–13.7] | 3.9 | 0.031 | 0.50 | 0.76 | 0.36 |
| wav2vec2-base-960h-ctc | librispeech-dummy-en | none | 10 | 6.2 [1.9–9.4] | 2.5 | 0.049 | 0.29 | 1.15 | 0.56 |
| faster-whisper-tiny-int8 | librispeech-dummy-en | none | 10 | 10.4 [5.8–14.9] | 5.1 | 0.035 | 0.28 | 0.57 | 0.40 |

## Paired A/B (WER difference, same utterances)

| dataset | aug | n | A | B | ΔWER pp (A−B) | 95% CI | p | verdict |
|---|---|--:|---|---|--:|:--|--:|---|
| fleurs-en | none | 200 | faster-whisper-large-v3-fp16 | faster-whisper-large-v3-int8 | -0.05 | [-0.17, +0.09] | 0.543 | not significant |
| fleurs-en | telephony | 200 | faster-whisper-large-v3-fp16 | faster-whisper-large-v3-int8 | -0.09 | [-0.34, +0.14] | 0.505 | not significant |
| fleurs-es | none | 200 | faster-whisper-large-v3-fp16 | faster-whisper-large-v3-int8 | +0.06 | [-0.02, +0.17] | 0.330 | not significant |
| fleurs-es | telephony | 200 | faster-whisper-large-v3-fp16 | faster-whisper-large-v3-int8 | +0.02 | [-0.06, +0.10] | 0.833 | not significant |
| fleurs-hi | none | 200 | faster-whisper-large-v3-fp16 | faster-whisper-large-v3-int8 | -0.24 | [-0.96, +0.41] | 0.474 | not significant |
| fleurs-hi | telephony | 200 | faster-whisper-large-v3-fp16 | faster-whisper-large-v3-int8 | -0.61 | [-1.27, +0.00] | 0.051 | not significant |
| librispeech-dummy-en | none | 10 | faster-whisper-tiny-int8 | wav2vec2-base-960h-ctc | +4.25 | [+0.70, +9.34] | 0.025 | **significant** |
