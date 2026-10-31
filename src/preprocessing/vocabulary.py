"""Training-only vocabulary and sequence utilities for a BiLSTM."""

import re
from collections import Counter
from typing import Iterable

from src.preprocessing.config import (
    BILSTM_MAX_SEQUENCE_LENGTH as MAX_SEQUENCE_LENGTH,
    BILSTM_MIN_FREQUENCY,
)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Words (keeping inner apostrophes: don't, it's), runs of ! or ? as ONE token
# (urgency signal), and every other symbol/emoji as its own token.
_TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[!?]+|[^\w\s]")


def tokenize_for_vocabulary(text: str) -> list[str]:
    """Split cleaned text into word, punctuation and emoji tokens."""
    return _TOKEN_PATTERN.findall(text)


def build_vocabulary(
    training_texts: Iterable[str],
    min_frequency: int = BILSTM_MIN_FREQUENCY,
) -> dict[str, int]:
    """Build a deterministic vocabulary from training texts only."""
    if min_frequency < 1:
        raise ValueError("min_frequency must be at least 1")

    counts = Counter()
    for text in training_texts:
        counts.update(tokenize_for_vocabulary(str(text)))

    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    words = [
        word for word, count in counts.items()
        if count >= min_frequency and word not in vocabulary
    ]
    words.sort(key=lambda word: (-counts[word], word))
    for word in words:
        vocabulary[word] = len(vocabulary)
    return vocabulary


def encode_text(text: str, vocabulary: dict[str, int]) -> list[int]:
    """Convert text to token IDs; map unseen tokens to <UNK>."""
    unknown_id = vocabulary[UNK_TOKEN]
    return [
        vocabulary.get(token, unknown_id)
        for token in tokenize_for_vocabulary(str(text))
    ]


def encode_and_pad(
    text: str,
    vocabulary: dict[str, int],
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> list[int]:
    """Encode text, truncate on the right, and pad on the right."""
    if max_length < 1:
        raise ValueError("max_length must be at least 1")
    token_ids = encode_text(text, vocabulary)[:max_length]
    return token_ids + [vocabulary[PAD_TOKEN]] * (max_length - len(token_ids))
