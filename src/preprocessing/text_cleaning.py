"""Reusable text preprocessing for the HumAID disaster classifier.

ONE implementation used for HumAID, Kaggle and CrisisLexT26 (checklist A15, D2).

Policy
- Keep the original tweet separately (callers store it).
- Unescape HTML entities first, repeatedly, so '&amp;amp;' is fully decoded.
- Remove known encoding-damage ("mojibake") sequences.
- Strip leading retweet prefixes, including chained 'RT @a: RT @b:'.
- TF-IDF / BiLSTM path: URLs -> URLTOKEN, mentions -> USERTOKEN, hashtag words
  kept without '#', lower-cased; punctuation, emojis and elongations preserved.
- Transformer path: keep casing, hashtags, URLs, mentions, emojis and informal
  spelling; BERTweet's own normalizer handles URLs/mentions/emojis.
"""

import html
import re
from typing import Optional

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
RETWEET_PREFIX_PATTERN = re.compile(
    r"^(?:\s*RT\s+@[A-Za-z0-9_]+:?\s*)+",   # also strips chained prefixes
    re.IGNORECASE,
)
WHITESPACE_PATTERN = re.compile(r"\s+")

# Encoding damage seen only in the Kaggle file (0 hits in HumAID/CrisisLexT26):
#   \x89Û? (mangled dashes/quotes), åÊ (mangled non-breaking space), Ì¢ / ã¢ ...
MOJIBAKE_PATTERN = re.compile(r"\x89Û.?|\x89|åÊ|å_|Ì[¢¤Ñ_]|ã¢")


def unescape_html(text: str) -> str:
    """Unescape HTML entities until stable (handles '&amp;amp;')."""
    for _ in range(3):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    return text


def fix_mojibake(text: str) -> str:
    """Replace known encoding-damage sequences with a space."""
    return MOJIBAKE_PATTERN.sub(" ", text)


def _base_clean(text: Optional[str]) -> str:
    """Steps shared by both paths, in the required order."""
    if text is None:
        return ""
    text = unescape_html(str(text))
    text = fix_mojibake(text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return RETWEET_PREFIX_PATTERN.sub("", text)


def clean_tweet_for_tfidf(text: Optional[str]) -> str:
    """Normalize tweet text for TF-IDF / BiLSTM. Deterministic and idempotent."""
    text = _base_clean(text)
    text = URL_PATTERN.sub(" URLTOKEN ", text)
    text = MENTION_PATTERN.sub(" USERTOKEN ", text)
    text = re.sub(r"#(?=\w)", "", text)          # keep the hashtag word
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text.lower()


def prepare_tweet_for_transformer(text: Optional[str]) -> str:
    """Light preprocessing for BERTweet (casing/emoji/punctuation preserved)."""
    return WHITESPACE_PATTERN.sub(" ", _base_clean(text)).strip()


def normalize_for_dedup(text: Optional[str]) -> str:
    """Key that collapses retweets / URL / mention / case / punctuation noise."""
    t = clean_tweet_for_tfidf(text)
    t = re.sub(r"\b(urltoken|usertoken)\b", " ", t)
    return re.sub(r"[^a-z0-9]+", " ", t).strip()
