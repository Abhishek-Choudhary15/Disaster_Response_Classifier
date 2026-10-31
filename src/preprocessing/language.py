"""Seeded language guess shared by the EDA notebook and the CrisisLexT26 pipeline."""

import re

from src.preprocessing.text_cleaning import clean_tweet_for_tfidf

_PLACEHOLDERS = re.compile(r"\b(?:urltoken|usertoken)\b")
_NON_LETTERS = re.compile(r"[^\w\s]|\d|_")
MIN_LETTERS = 15          # fewer letters than this -> 'too_short' (not reliable)

_detector_ready = False


def guess_language(text) -> str:
    """ISO code of the dominant language, or 'too_short' / 'unknown'."""
    global _detector_ready
    try:
        from langdetect import DetectorFactory, detect
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError("pip install langdetect") from exc
    if not _detector_ready:
        DetectorFactory.seed = 0                                # deterministic
        _detector_ready = True

    cleaned = _PLACEHOLDERS.sub(" ", clean_tweet_for_tfidf(text))
    cleaned = _NON_LETTERS.sub(" ", cleaned)
    if len(re.findall(r"[^\W\d_]", cleaned)) < MIN_LETTERS:
        return "too_short"
    try:
        return detect(cleaned)
    except Exception:
        return "unknown"
