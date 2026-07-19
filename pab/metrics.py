"""Quality + speed metrics.

WER/CER are computed at the corpus level (sum of edit distances / sum of
reference lengths), which is the standard way to aggregate — averaging per-utterance
rates over-weights short utterances. Edit distance is a small Levenshtein DP so the
project has one fewer dependency and the metric is fully auditable.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
from .normalize import normalize


def _edit_distance(ref: Sequence, hyp: Sequence) -> int:
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def wer(ref: str, hyp: str) -> tuple[int, int]:
    """Return (word errors, reference word count)."""
    r = normalize(ref).split()
    h = normalize(hyp).split()
    return _edit_distance(r, h), len(r)


def cer(ref: str, hyp: str) -> tuple[int, int]:
    """Return (char errors, reference char count), spaces ignored."""
    r = list(normalize(ref).replace(" ", ""))
    h = list(normalize(hyp).replace(" ", ""))
    return _edit_distance(r, h), len(r)


def corpus_rate(pairs) -> float:
    """pairs: iterable of (errors, ref_len). Returns aggregate error rate.

    Materialize first — callers pass generators, and we iterate twice.
    """
    pairs = list(pairs)
    err = sum(e for e, _ in pairs)
    tot = sum(n for _, n in pairs)
    return err / tot if tot else 0.0


def percentile(values, p) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), p))
