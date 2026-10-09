
"""Reusable text preprocessing for the HumAID disaster classifier.

Policy:
- Preserve the original tweet separately.
- Replace URLs and mentions with stable placeholders.
- Remove leading retweet prefixes.
- Keep hashtag words but remove the # symbol.
- Preserve emojis, punctuation, casing, and elongated spellings.
"""

import re
import html
from typing import Optional

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
RETWEET_PREFIX_PATTERN = re.compile(
    r"^\s*RT\s+@[A-Za-z0-9_]+:\s*",
    re.IGNORECASE,
)
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_tweet_for_tfidf(text: Optional[str]) -> str:
    """Normalize tweet text for a traditional TF-IDF model.

    This function is deterministic and does not learn from the dataset.
    """
    if text is None:
        return ""

    text = html.unescape(str(text)).strip()
    text = RETWEET_PREFIX_PATTERN.sub("", text)
    text = URL_PATTERN.sub(" URLTOKEN ", text)
    text = MENTION_PATTERN.sub(" USERTOKEN ", text)

    # Keep hashtag words, removing only the hash symbol.
    text = re.sub(r"#(?=\w)", "", text)

    # Preserve punctuation and emojis; normalize whitespace only.
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text.lower()


def prepare_tweet_for_transformer(text: Optional[str]) -> str:
    """Light preprocessing for a future tweet-aware transformer.

    Preserve casing, hashtags, punctuation, emojis, and informal spelling.
    Remove only a leading retweet prefix and normalize HTML entities.
    """
    if text is None:
        return ""

    text = html.unescape(str(text)).strip()
    text = RETWEET_PREFIX_PATTERN.sub("", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text
