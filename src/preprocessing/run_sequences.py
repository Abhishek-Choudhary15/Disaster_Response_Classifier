"""Build BiLSTM sequences using a TRAIN-only vocabulary."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing.config import (
    BILSTM_MIN_FREQUENCY,
    HUMAID_VOCAB_ARTIFACT,
    HUMAID_VOCAB_JSON,
)
from src.preprocessing.text_cleaning import clean_tweet_for_tfidf
from src.preprocessing.vocabulary import (
    MAX_SEQUENCE_LENGTH,
    build_vocabulary,
    encode_and_pad,
    tokenize_for_vocabulary,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "models" / "preprocessing"

SPLITS = {
    "train": "humaid_train.csv",
    "validation": "humaid_validation.csv",
    "test": "humaid_test.csv",
}


def main():
    datasets = {}
    for split, filename in SPLITS.items():
        path = RAW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}")
        df = pd.read_csv(path)
        if df["tweet_text"].isna().any():
            raise ValueError(f"Missing tweet text in {filename}")
        datasets[split] = df

    cleaned = {s: df["tweet_text"].map(clean_tweet_for_tfidf).tolist()
               for s, df in datasets.items()}

    # Choose/justify the length from the TRAIN token-length distribution (A12).
    lengths = {s: np.array([len(tokenize_for_vocabulary(t)) for t in texts])
               for s, texts in cleaned.items()}
    p99 = float(np.percentile(lengths["train"], 99))
    if MAX_SEQUENCE_LENGTH < p99:
        raise ValueError(
            f"MAX_SEQUENCE_LENGTH={MAX_SEQUENCE_LENGTH} is below the train 99th "
            f"percentile ({p99:.0f}). Raise BILSTM_MAX_SEQUENCE_LENGTH in config.py."
        )

    vocabulary = build_vocabulary(cleaned["train"], min_frequency=BILSTM_MIN_FREQUENCY)

    sequences = {
        split: np.asarray([encode_and_pad(t, vocabulary, MAX_SEQUENCE_LENGTH) for t in texts],
                          dtype=np.int32)
        for split, texts in cleaned.items()
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vocabulary, ARTIFACT_DIR / HUMAID_VOCAB_ARTIFACT)
    (ARTIFACT_DIR / HUMAID_VOCAB_JSON).write_text(
        json.dumps(vocabulary, indent=2, ensure_ascii=False), encoding="utf-8")
    for split, matrix in sequences.items():
        np.save(OUTPUT_DIR / f"{split}_bilstm_sequences.npy", matrix)

    report = {
        "vocabulary_size": len(vocabulary),
        "minimum_frequency": BILSTM_MIN_FREQUENCY,
        "tokenizer": "regex: words(with apostrophes) | !/? runs | other symbols",
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "padding": "post",
        "truncation": "post",
        "vocabulary_fit_split": "train",
        "train_length_percentiles": {
            f"p{q}": float(np.percentile(lengths["train"], q)) for q in (50, 95, 99, 99.9)
        } | {"max": int(lengths["train"].max())},
        "splits": {
            split: {
                "rows": int(matrix.shape[0]),
                "sequence_shape": list(matrix.shape),
                "truncated_tweets": int((lengths[split] > MAX_SEQUENCE_LENGTH).sum()),
                "truncated_pct": round(float((lengths[split] > MAX_SEQUENCE_LENGTH).mean() * 100), 3),
            }
            for split, matrix in sequences.items()
        },
    }
    (OUTPUT_DIR / "bilstm_sequence_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\nBiLSTM sequence preparation completed successfully.")


if __name__ == "__main__":
    main()
