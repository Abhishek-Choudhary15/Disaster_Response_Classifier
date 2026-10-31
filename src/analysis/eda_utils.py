"""Helpers for notebooks/01_eda_three_datasets.ipynb.

Every dataset goes through the SAME functions, and the same shared cleaning
code the production pipeline uses (checklist E-spot-check, A15).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.run_audits import oov_stats, overlap          # single implementation
from src.preprocessing.config import BILSTM_MAX_SEQUENCE_LENGTH
from src.preprocessing.text_cleaning import (
    MENTION_PATTERN,
    MOJIBAKE_PATTERN,
    URL_PATTERN,
    clean_tweet_for_tfidf,
    normalize_for_dedup,
    prepare_tweet_for_transformer,
)
from src.preprocessing.language import guess_language
from src.preprocessing.vocabulary import tokenize_for_vocabulary

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
EDA_DIR = ROOT / "docs" / "eda"
SPLITS = ["train", "validation", "test"]

ENTITY_RE = re.compile(r"&(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-f]+);", re.IGNORECASE)
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
HASHTAG_RE = re.compile(r"(?<!\w)#\w+")
ELONGATED_RE = re.compile(r"([A-Za-z])\1{2,}")
CONTRACTION_RE = re.compile(r"\b(?:im|dont|cant|wont|isnt|didnt|wasnt|youre|theyre|whos|thats|ive)\b",
                            re.IGNORECASE)
PLACEHOLDER_RE = r"\b(?:urltoken|usertoken)\b"


# ------------------------------------------------------------------ loading
def load_humaid():
    return {s: pd.read_csv(RAW / f"humaid_{s}.csv") for s in SPLITS}


def load_kaggle():
    return pd.read_csv(RAW / "nlpgettingstarted_train")


def load_crisislex():
    """Concatenate the 26 labeled event files; return (frame, per-event schema report)."""
    frames, schemas = [], {}
    for path in sorted((RAW / "crisislext26").rglob("*-tweets_labeled.csv")):
        df = pd.read_csv(path)
        raw_cols = list(df.columns)
        df.columns = [str(c).strip() for c in df.columns]
        schemas[path.parent.name] = {
            "rows": len(df),
            "columns": tuple(df.columns),
            "header_had_extra_whitespace": raw_cols != list(df.columns),
        }
        df["event_name"] = path.parent.name
        df["source_file"] = path.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True), schemas


# ------------------------------------------------------------------ profiling
def length_stats(values):
    v = pd.Series(values)
    return {"min": int(v.min()), "mean": round(float(v.mean()), 1), "p50": float(v.quantile(.5)),
            "p95": float(v.quantile(.95)), "p99": float(v.quantile(.99)), "max": int(v.max())}


def profile_texts(texts: pd.Series, ids: pd.Series = None) -> dict:
    """All 'universal' EDA measurements for one text column."""
    t = texts.fillna("").astype(str)
    cleaned = t.map(clean_tweet_for_tfidf)
    bare = cleaned.str.replace(PLACEHOLDER_RE, " ", regex=True)
    no_text = ~bare.str.contains(r"[^\W_]", regex=True)          # no letter/digit left
    tokens = cleaned.map(lambda x: len(tokenize_for_vocabulary(x)))

    tags = t.str.findall(HASHTAG_RE).explode().dropna().str.lower()
    tag_counts = tags.value_counts()
    mentions = t.str.count(MENTION_PATTERN.pattern)
    urls = t.str.count(URL_PATTERN.pattern)
    emojis = t.map(EMOJI_RE.findall).explode().dropna()
    emoji_counts = emojis.value_counts()

    out = {
        "rows": int(len(texts)),
        "null_text": int(texts.isna().sum()),
        "empty_text": int((t.str.strip() == "").sum()),
        "exact_duplicate_rows": int(t.duplicated().sum()),
        "near_duplicate_rows": int(t.map(normalize_for_dedup).duplicated().sum()),
        "html_entity_rows": int(t.str.contains(ENTITY_RE).sum()),
        "html_entity_rows_after_cleaning": int(cleaned.str.contains(ENTITY_RE).sum()),
        "retweet_prefix_rows": int(t.str.match(r"^\s*RT\s+@", case=False).sum()),
        "encoding_damage_rows": int(t.str.contains(MOJIBAKE_PATTERN).sum()),
        "empty_after_tfidf_cleaning": int((cleaned.str.strip() == "").sum()),
        "empty_after_transformer_cleaning": int((t.map(prepare_tweet_for_transformer).str.strip() == "").sum()),
        "no_letters_or_digits_after_cleaning": int(no_text.sum()),
        "  of which url/mention only": int((no_text & cleaned.str.contains("urltoken|usertoken")).sum()),
        "  of which emoji only": int((no_text & t.str.contains(EMOJI_RE)).sum()),
        "apostrophe_less_contraction_rows": int(t.str.contains(CONTRACTION_RE).sum()),
        "elongated_word_rows": int(t.map(lambda x: bool(ELONGATED_RE.search(x))).sum()),
        "char_length": length_stats(t.str.len()),
        "whitespace_token_length": length_stats(t.str.split().str.len()),
        "project_token_length": length_stats(tokens),
        "pct_over_bilstm_max_len": round(float((tokens > BILSTM_MAX_SEQUENCE_LENGTH).mean() * 100), 3),
        "hashtags": {
            "total": int(len(tags)), "unique": int(tag_counts.size),
            "top1_share_pct": round(float(tag_counts.iloc[0] / len(tags) * 100), 2) if len(tags) else 0.0,
            "top10_share_pct": round(float(tag_counts.head(10).sum() / len(tags) * 100), 2) if len(tags) else 0.0,
            "top20": {k: int(v) for k, v in tag_counts.head(20).items()},
        },
        "mentions": {"total": int(mentions.sum()), "rows_with": int((mentions > 0).sum()),
                     "mean_per_row": round(float(mentions.mean()), 3), "max_in_row": int(mentions.max())},
        "urls": {"total": int(urls.sum()), "rows_with": int((urls > 0).sum()),
                 "mean_per_row": round(float(urls.mean()), 3), "max_in_row": int(urls.max())},
        "emoji": {"total": int(len(emojis)), "unique": int(emoji_counts.size),
                  "rows_with": int(t.str.contains(EMOJI_RE).sum()),
                  "top10": {k: int(v) for k, v in emoji_counts.head(10).items()}},
    }
    if ids is not None:
        ids = ids.astype(str)
        multi = t.groupby(ids).nunique()
        out["ids"] = {"unique": int(ids.nunique()), "duplicate_ids": int(ids.duplicated().sum()),
                      "ids_with_different_texts": int((multi > 1).sum())}
    return out


def scalar_table(profiles: dict) -> pd.DataFrame:
    """Side-by-side table of the scalar metrics for several datasets."""
    rows = {}
    for name, p in profiles.items():
        flat = {k: v for k, v in p.items() if isinstance(v, (int, float))}
        flat["char_len p50/p95/p99/max"] = "/".join(str(int(p["char_length"][k])) for k in ("p50", "p95", "p99", "max"))
        flat["token_len p50/p95/p99/max"] = "/".join(str(int(p["project_token_length"][k])) for k in ("p50", "p95", "p99", "max"))
        rows[name] = flat
    return pd.DataFrame(rows)


def language_profile(texts: pd.Series, n=3000, seed=0) -> dict:
    """Seeded langdetect on a random sample (shared guess_language function)."""
    sample = texts.fillna("").astype(str).sample(min(n, len(texts)), random_state=seed)
    langs = sample.map(guess_language)
    return {"sampled": int(len(sample)),
            "share_pct": (langs.value_counts(normalize=True) * 100).round(2).head(6).to_dict()}


# ------------------------------------------------------------------ HumAID helpers
DISASTER_TYPES = {                     # keyword PROXY - the CSVs carry no event column
    "earthquake": r"earthquake|quake|tremor|aftershock",
    "hurricane/storm": r"hurricane|cyclone|typhoon|storm|irma|harvey|maria|matthew",
    "flood": r"flood",
    "wildfire": r"wildfire|fire|blaze|burn",
}


def disaster_type_proxy(df: pd.DataFrame) -> pd.DataFrame:
    text = df["tweet_text"].str.lower()
    kind = pd.Series("unknown", index=df.index)
    for name, pattern in reversed(list(DISASTER_TYPES.items())):
        kind[text.str.contains(pattern)] = name
    return kind


def class_distribution(humaid: dict) -> pd.DataFrame:
    counts = pd.DataFrame({s: d["class_label"].value_counts() for s, d in humaid.items()}).fillna(0).astype(int)
    return counts.sort_values("train", ascending=False)


def sample_rows(df, text_col, n=15, seed=0, by=None):
    """Random rows (not .head()); one block per class when `by` is given."""
    if by is None:
        return df.sample(min(n, len(df)), random_state=seed)[[text_col]]
    parts = [g.sample(min(n, len(g)), random_state=seed) for _, g in df.groupby(by)]
    return pd.concat(parts)[[by, text_col]]
