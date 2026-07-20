| model | dataset | aug | n | WER% (95% CI) | CER% | RTF | p50 s | p95 s | $/1000min |
|---|---|---|--:|:--|--:|--:|--:|--:|--:|
| large-v3-int8-cpu | fleurs-am-cpu | none | 29 | 129.7 [109.2–152.8] | 123.8 | 23.704 | 247.77 | 464.01 | 272.59 |
| large-v3-mlx-gpu | fleurs-am-cpu | none | 30 | 188.4 [149.0–232.0] | 116.5 | 6.625 | 74.75 | 84.49 | 76.18 |
| large-v3-mlx-gpu | fleurs-ceb-cpu | none | 100 | 43.1 [39.4–47.0] | 12.3 | 0.451 | 5.46 | 10.42 | 5.19 |
| large-v3-int8-cpu | fleurs-ceb-cpu | none | 100 | 43.3 [39.5–47.2] | 12.6 | 0.980 | 12.31 | 18.16 | 11.27 |
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
| large-v3-mlx-gpu | fleurs-km-cpu | none | 30 | 112.3 [106.2–121.3] | 101.2 | 6.336 | 78.24 | 131.94 | 72.86 |
| large-v3-mlx-gpu | fleurs-lo-cpu | none | 30 | 112.3 [103.5–123.8] | 101.9 | 1.009 | 7.57 | 68.10 | 11.60 |
| large-v3-mlx-gpu | fleurs-mi-cpu | none | 100 | 39.5 [35.2–44.2] | 14.9 | 0.242 | 4.35 | 7.93 | 2.78 |
| large-v3-int8-cpu | fleurs-mi-cpu | none | 100 | 39.9 [35.4–44.5] | 15.3 | 0.461 | 8.28 | 15.78 | 5.30 |
| faster-whisper-large-v3-fp16 | fleurs-tl | none | 150 | 12.2 [10.7–13.7] | 3.9 | 0.031 | 0.50 | 0.76 | 0.36 |
| large-v3-mlx-gpu | fleurs-yo-cpu | none | 100 | 96.4 [94.3–98.4] | 45.8 | 0.260 | 3.66 | 4.83 | 2.98 |
| large-v3-int8-cpu | fleurs-yo-cpu | none | 100 | 97.8 [95.6–99.8] | 45.6 | 0.537 | 7.56 | 9.86 | 6.17 |
| wav2vec2-base-960h-ctc | librispeech-dummy-en | none | 10 | 6.2 [1.9–9.4] | 2.5 | 0.049 | 0.29 | 1.15 | 0.56 |
| faster-whisper-tiny-int8 | librispeech-dummy-en | none | 10 | 10.4 [5.8–14.9] | 5.1 | 0.035 | 0.28 | 0.57 | 0.40 |

## Paired A/B (WER difference, same utterances)

| dataset | aug | n | A | B | ΔWER pp (A−B) | 95% CI | p | verdict |
|---|---|--:|---|---|--:|:--|--:|---|
| fleurs-am-cpu | none | 29 | large-v3-int8-cpu | large-v3-mlx-gpu | -52.49 | [-100.91, -11.35] | 0.009 | **significant** |
| fleurs-ceb-cpu | none | 100 | large-v3-int8-cpu | large-v3-mlx-gpu | +0.19 | [-0.35, +0.75] | 0.552 | not significant |
| fleurs-en | none | 200 | faster-whisper-large-v3-fp16 | faster-whisper-large-v3-int8 | -0.05 | [-0.17, +0.09] | 0.543 | not significant |
| fleurs-en | telephony | 200 | faster-whisper-large-v3-fp16 | faster-whisper-large-v3-int8 | -0.09 | [-0.34, +0.14] | 0.505 | not significant |
| fleurs-es | none | 200 | faster-whisper-large-v3-fp16 | faster-whisper-large-v3-int8 | +0.06 | [-0.02, +0.17] | 0.330 | not significant |
| fleurs-es | telephony | 200 | faster-whisper-large-v3-fp16 | faster-whisper-large-v3-int8 | +0.02 | [-0.06, +0.10] | 0.833 | not significant |
| fleurs-hi | none | 200 | faster-whisper-large-v3-fp16 | faster-whisper-large-v3-int8 | -0.24 | [-0.96, +0.41] | 0.474 | not significant |
| fleurs-hi | telephony | 200 | faster-whisper-large-v3-fp16 | faster-whisper-large-v3-int8 | -0.61 | [-1.27, +0.00] | 0.051 | not significant |
| fleurs-mi-cpu | none | 100 | large-v3-int8-cpu | large-v3-mlx-gpu | +0.39 | [-0.60, +1.32] | 0.488 | not significant |
| fleurs-yo-cpu | none | 100 | large-v3-int8-cpu | large-v3-mlx-gpu | +1.39 | [+0.52, +2.23] | 0.002 | **significant** |
| librispeech-dummy-en | none | 10 | faster-whisper-tiny-int8 | wav2vec2-base-960h-ctc | +4.25 | [+0.70, +9.34] | 0.025 | **significant** |
