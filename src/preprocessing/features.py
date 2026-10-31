
"""Hand-engineered numerical features for HumAID tweets."""

import re

import pandas as pd

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
HASHTAG_PATTERN = re.compile(r"(?<!\w)#\w+")

FEATURE_COLUMNS = [
    "char_length",
    "word_count",
    "exclamation_count",
    "question_count",
    "hashtag_count",
    "mention_count",
    "url_count",
    "uppercase_ratio",
]


def extract_tweet_features(texts):
    """Extract deterministic numerical features from original tweet text.

    Args:
        texts: Iterable of original tweet strings.

    Returns:
        DataFrame with one row per tweet and stable feature columns.
    """
    rows = []

    for value in texts:
        text = "" if pd.isna(value) else str(value)

        letters = [char for char in text if char.isalpha()]
        uppercase_ratio = (
            sum(char.isupper() for char in letters) / len(letters)
            if letters
            else 0.0
        )

        rows.append({
            "char_length": len(text),
            "word_count": len(text.split()),
            "exclamation_count": text.count("!"),
            "question_count": text.count("?"),
            "hashtag_count": len(HASHTAG_PATTERN.findall(text)),
            "mention_count": len(MENTION_PATTERN.findall(text)),
            "url_count": len(URL_PATTERN.findall(text)),
            "uppercase_ratio": uppercase_ratio,
        })

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
