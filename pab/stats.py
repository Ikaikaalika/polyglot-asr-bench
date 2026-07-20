"""Uncertainty for corpus error rates.

WER is an estimate from a finite sample of utterances, so a bare point estimate can't
support "model A beats model B". Two tools here:

  bootstrap_ci        - resample utterances with replacement to get a CI for one system.
  paired_bootstrap    - resample utterance *indices* once per replicate and score BOTH
                        systems on that same resample. Because the systems are compared
                        on identical utterances, per-utterance difficulty cancels out,
                        which makes this dramatically more sensitive than eyeballing two
                        independent CIs.

Important: overlapping marginal CIs do NOT imply a non-significant difference. Two
systems can have visibly overlapping CIs while their paired difference is decisively
non-zero, because the marginal intervals are dominated by variance in utterance
difficulty that the pairing removes. Always use the paired test for A/B claims.

Corpus rate is a ratio of sums (total errors / total reference length), not a mean of
per-utterance rates, so the bootstrap resamples utterances and recomputes the ratio.
"""
from __future__ import annotations
import numpy as np


def _rate(errs: np.ndarray, lens: np.ndarray) -> float:
    tot = lens.sum()
    return float(errs.sum() / tot) if tot > 0 else 0.0


def _as_arrays(pairs):
    errs = np.array([p[0] for p in pairs], dtype=float)
    lens = np.array([p[1] for p in pairs], dtype=float)
    return errs, lens


def bootstrap_ci(pairs, n_boot=2000, alpha=0.05, seed=0):
    """pairs: iterable of (errors, ref_len). Returns (point, lo, hi)."""
    errs, lens = _as_arrays(pairs)
    point = _rate(errs, lens)
    n = len(errs)
    if n < 2:
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = errs[idx].sum(axis=1) / np.maximum(lens[idx].sum(axis=1), 1e-9)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_bootstrap(a_pairs, b_pairs, n_boot=2000, alpha=0.05, seed=0):
    """Paired bootstrap of (rate_a - rate_b) over the SAME utterances, in order.

    Returns (diff, lo, hi, p_two_sided). p is the bootstrap two-sided achieved
    significance level: how often replicates land on the opposite side of zero,
    doubled. It is approximate — report it alongside the interval, not instead of it.
    """
    ae, al = _as_arrays(a_pairs)
    be, bl = _as_arrays(b_pairs)
    if len(ae) != len(be):
        raise ValueError("paired_bootstrap requires aligned, equal-length inputs")
    diff = _rate(ae, al) - _rate(be, bl)
    n = len(ae)
    if n < 2:
        return diff, diff, diff, 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    ra = ae[idx].sum(axis=1) / np.maximum(al[idx].sum(axis=1), 1e-9)
    rb = be[idx].sum(axis=1) / np.maximum(bl[idx].sum(axis=1), 1e-9)
    boot = ra - rb
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2.0 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))
    return diff, float(lo), float(hi), min(p, 1.0)
