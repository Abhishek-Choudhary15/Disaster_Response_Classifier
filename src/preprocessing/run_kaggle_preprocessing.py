"""Preprocess Kaggle 'NLP with Disaster Tweets' for the BINARY pre-filter.

This pipeline is fully independent from the HumAID one: its own vectorizer,
its own split, its own artifact names. Only the labeled training file is read.
"""

import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.config import (
    KAGGLE_TEST_SIZE,
    KAGGLE_TFIDF_ARTIFACT,
    KAGGLE_VALIDATION_SIZE,
    SEED,
)
from src.preprocessing.text_cleaning import (
    MOJIBAKE_PATTERN,
    clean_tweet_for_tfidf,
    normalize_for_dedup,
)
from src.preprocessing.vectorization import TFIDF_TOKEN_PATTERN, fit_tfidf, transform_tfidf

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed" / "kaggle"
MODEL_DIR = ROOT / "models" / "preprocessing"

KAGGLE_TRAIN_FILE = "nlpgettingstarted_train"   # the ONLY Kaggle file used


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RAW_DIR / KAGGLE_TRAIN_FILE)
    required = {"text", "target"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing required columns: {required - set(raw.columns)}")

    # 1. Keep text + target only. keyword/location are deliberately NOT features
    #    (shortcut risk / unusable free text) - see docs/preprocessing_decisions.md.
    df = raw[["text", "target"]].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["target"] = df["target"].astype(int)
    if not set(df["target"].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    # 2. Deduplicate BEFORE splitting so no tweet can sit in two splits.
    #    Key collapses retweets / URL / mention / case / punctuation differences.
    df["dedupe_key"] = df["text"].map(normalize_for_dedup)
    n_labels = df.groupby("dedupe_key")["target"].transform("nunique")
    conflicting = df[n_labels > 1].sort_values("dedupe_key")
    conflicting.to_csv(PROCESSED_DIR / "kaggle_conflicting_label_texts.csv", index=False)

    rows_raw = len(df)
    exact_duplicates = int(df["text"].duplicated().sum())
    mojibake_rows = int(df["text"].str.contains(MOJIBAKE_PATTERN).sum())
    df = df[n_labels == 1].drop_duplicates("dedupe_key").reset_index(drop=True)

    # 3. Clean with the SHARED function (same rules as HumAID / CrisisLexT26).
    df["cleaned_text"] = df["text"].map(clean_tweet_for_tfidf)

    # 4. Stratified 80/10/10 split with a fixed seed.
    train_df, temp_df = train_test_split(
        df, test_size=KAGGLE_TEST_SIZE + KAGGLE_VALIDATION_SIZE,
        random_state=SEED, stratify=df["target"])
    validation_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["target"])

    # 5. Leakage guard: fail loudly if any normalised text is in two splits.
    keys = {n: set(d["dedupe_key"]) for n, d in
            {"train": train_df, "validation": validation_df, "test": test_df}.items()}
    shared = (len(keys["train"] & keys["validation"]) + len(keys["train"] & keys["test"])
              + len(keys["validation"] & keys["test"]))
    if shared:
        raise ValueError(f"Kaggle split leakage: {shared} shared texts between splits")

    # 6. Fit TF-IDF on the Kaggle TRAIN split only (shared configuration).
    vectorizer, X_train = fit_tfidf(train_df["cleaned_text"])
    splits = {
        "train": (train_df, X_train),
        "validation": (validation_df, transform_tfidf(vectorizer, validation_df["cleaned_text"])),
        "test": (test_df, transform_tfidf(vectorizer, test_df["cleaned_text"])),
    }

    for name, (split_df, X) in splits.items():
        split_df[["target"]].to_csv(PROCESSED_DIR / f"{name}_labels.csv", index=False)
        joblib.dump(X, PROCESSED_DIR / f"{name}_tfidf.joblib")
        split_df[["text", "cleaned_text"]].to_csv(PROCESSED_DIR / f"{name}_text.csv", index=False)

    joblib.dump(vectorizer, MODEL_DIR / KAGGLE_TFIDF_ARTIFACT)

    residual = df["cleaned_text"].str.contains(r"[\x80-\xff]", regex=True)
    report = {
        "dataset": "Kaggle NLP with Disaster Tweets",
        "seed": SEED,
        "rows_raw": rows_raw,
        "exact_duplicate_rows_in_raw": exact_duplicates,
        "rows_with_encoding_damage_fixed": mojibake_rows,
        "conflicting_label_rows_dropped": int(len(conflicting)),
        "conflicting_label_groups_dropped": int(conflicting["dedupe_key"].nunique()),
        "rows_after_deduplication": len(df),
        "rows_with_other_non_ascii_for_manual_review": int(residual.sum()),
        "split_sizes": {n: len(d) for n, (d, _) in splits.items()},
        "target_distribution": {
            n: d["target"].value_counts().sort_index().to_dict() for n, (d, _) in splits.items()},
        "shared_texts_between_splits": shared,
        "tfidf_features": int(X_train.shape[1]),
        "tfidf_token_pattern": TFIDF_TOKEN_PATTERN,
        "vectorizer_fit_on": "Kaggle deduplicated training split only",
        "keyword_used": False,
        "location_used": False,
        "competition_test_used_for_fitting": False,
        "target_mapping": {"0": "not disaster-related", "1": "disaster-related"},
    }
    (PROCESSED_DIR / "kaggle_preprocessing_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print("\nKaggle preprocessing completed.")


if __name__ == "__main__":
    main()
