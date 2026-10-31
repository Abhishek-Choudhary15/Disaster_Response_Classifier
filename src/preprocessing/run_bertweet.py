"""Prepare BERTweet inputs from HumAID (tokenizer 'fit' = download + save only).

Text goes through prepare_tweet_for_transformer first (HTML unescape, encoding
repair, RT-prefix removal); BERTweet's own normalization then handles
URLs / @mentions / emojis. Max length is capped at the model's real limit.
"""

import importlib.metadata as metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.preprocessing.config import (
    BERTWEET_HARD_LIMIT,
    BERTWEET_MAX_LENGTH as MAX_LENGTH,
    BERTWEET_MODEL_NAME,
    HUMAID_BERTWEET_TOKENIZER_DIR,
)
from src.preprocessing.text_cleaning import prepare_tweet_for_transformer

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "models" / "preprocessing"

SPLITS = {
    "train": "humaid_train.csv",
    "validation": "humaid_validation.csv",
    "test": "humaid_test.csv",
}


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main():
    if MAX_LENGTH > BERTWEET_HARD_LIMIT:
        raise ValueError(f"BERTWEET_MAX_LENGTH={MAX_LENGTH} exceeds the model limit "
                         f"({BERTWEET_HARD_LIMIT}).")
    if package_version("emoji") is None:
        print("WARNING: 'emoji' is not installed - BERTweet will not convert emojis. "
              "Run: pip install emoji==0.6.0")

    texts = {}
    for split, filename in SPLITS.items():
        path = RAW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}")
        df = pd.read_csv(path)
        if df["tweet_text"].isna().any():
            raise ValueError(f"Missing tweet text in {filename}")
        texts[split] = df["tweet_text"].map(prepare_tweet_for_transformer).tolist()

    tokenizer = AutoTokenizer.from_pretrained(BERTWEET_MODEL_NAME, normalization=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(ARTIFACT_DIR / HUMAID_BERTWEET_TOKENIZER_DIR)

    # Justify the length from the TRAIN distribution: tokenise without truncation.
    train_ids = tokenizer(texts["train"], truncation=False, padding=False,
                          add_special_tokens=True)["input_ids"]
    train_lengths = np.array([len(ids) for ids in train_ids])

    report = {
        "model_name": BERTWEET_MODEL_NAME,
        "max_length": MAX_LENGTH,
        "model_hard_limit": BERTWEET_HARD_LIMIT,
        "padding": "max_length",
        "truncation": True,
        "length_selection_split": "train",
        "text_input": "prepare_tweet_for_transformer(tweet_text)",
        "tokenizer_normalization": True,
        "emoji_package_version": package_version("emoji"),
        "transformers_version": package_version("transformers"),
        "train_token_length_percentiles": {
            f"p{q}": float(np.percentile(train_lengths, q)) for q in (50, 95, 99, 99.9)
        } | {"max": int(train_lengths.max())},
        "train_tweets_over_max_length": int((train_lengths > MAX_LENGTH).sum()),
        "train_tweets_over_max_length_pct": round(float((train_lengths > MAX_LENGTH).mean() * 100), 3),
        "splits": {},
    }

    for split, split_texts in texts.items():
        encoded = tokenizer(split_texts, padding="max_length", truncation=True,
                            max_length=MAX_LENGTH, return_attention_mask=True,
                            return_tensors="np")
        input_ids = encoded["input_ids"].astype(np.int32)
        attention_mask = encoded["attention_mask"].astype(np.int8)
        np.save(OUTPUT_DIR / f"{split}_bertweet_input_ids.npy", input_ids)
        np.save(OUTPUT_DIR / f"{split}_bertweet_attention_mask.npy", attention_mask)
        report["splits"][split] = {
            "rows": len(split_texts),
            "input_ids_shape": list(input_ids.shape),
            "attention_mask_shape": list(attention_mask.shape),
            "tweets_at_max_length_limit": int((attention_mask.sum(axis=1) >= MAX_LENGTH).sum()),
        }

    (OUTPUT_DIR / "bertweet_preprocessing_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\nBERTweet preprocessing completed successfully.")


if __name__ == "__main__":
    main()
