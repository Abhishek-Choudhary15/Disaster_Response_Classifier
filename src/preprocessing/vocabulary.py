
"""Training-only vocabulary and sequence utilities for a BiLSTM."""

from collections import Counter
from typing import Iterable

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MAX_SEQUENCE_LENGTH = 64


def tokenize_for_vocabulary(text: str) -> list[str]:
    """Basic whitespace tokenization for the initial BiLSTM baseline."""
    return text.split()


def build_vocabulary(
    training_texts: Iterable[str],
    min_frequency: int = 2,
) -> dict[str, int]:
    """Build a deterministic vocabulary from training texts only."""
    if min_frequency < 1:
        raise ValueError("min_frequency must be at least 1")

    counts = Counter()
    for text in training_texts:
        counts.update(tokenize_for_vocabulary(str(text)))

    vocabulary = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    words = [
        word for word, count in counts.items()
        if count >= min_frequency and word not in vocabulary
    ]
    words.sort(key=lambda word: (-counts[word], word))

    for word in words:
        vocabulary[word] = len(vocabulary)

    return vocabulary


def encode_text(
    text: str,
    vocabulary: dict[str, int],
) -> list[int]:
    """Convert text to token IDs; map unseen words to <UNK>."""
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

    token_ids = encode_text(text, vocabulary)
    token_ids = token_ids[:max_length]
    return token_ids + [vocabulary[PAD_TOKEN]] * (
        max_length - len(token_ids)
    )
