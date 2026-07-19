"""Lightweight, dependency-free text normalization for WER/CER scoring.

The point of normalization is to score *what was said*, not formatting: casing,
punctuation, and whitespace shouldn't count as errors. This mirrors what
Whisper's own evaluation does (a basic multilingual normalizer), kept simple and
transparent here so the numbers are reproducible and easy to audit. Swap this out
for a stricter/looser scheme and every metric downstream updates.
"""
from __future__ import annotations
import re
import unicodedata

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)   # drop punctuation
    text = _WS_RE.sub(" ", text)      # collapse whitespace
    return text.strip()
