
"""Build BiLSTM sequences using a training-only vocabulary."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing.text_cleaning import clean_tweet_for_tfidf
from src.preprocessing.vocabulary import (
    MAX_SEQUENCE_LENGTH,
    build_vocabulary,
    encode_and_pad,
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

    # Apply the existing TF-IDF cleaning policy consistently.
    cleaned_texts = {
        split: df["tweet_text"].map(clean_tweet_for_tfidf).tolist()
        for split, df in datasets.items()
    }

    # Learn the vocabulary from training tweets only.
    vocabulary = build_vocabulary(
        cleaned_texts["train"],
        min_frequency=2,
    )

    # Use the same vocabulary and sequence length for every split.
    sequences = {
        split: np.asarray(
            [
                encode_and_pad(text, vocabulary, MAX_SEQUENCE_LENGTH)
                for text in texts
            ],
            dtype=np.int32,
        )
        for split, texts in cleaned_texts.items()
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        vocabulary,
        ARTIFACT_DIR / "bilstm_vocabulary.joblib",
    )

    (ARTIFACT_DIR / "bilstm_vocabulary.json").write_text(
        json.dumps(vocabulary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for split, matrix in sequences.items():
        np.save(OUTPUT_DIR / f"{split}_bilstm_sequences.npy", matrix)

    report = {
        "vocabulary_size": len(vocabulary),
        "minimum_frequency": 2,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "padding": "post",
        "truncation": "post",
        "vocabulary_fit_split": "train",
        "splits": {
            split: {
                "rows": int(matrix.shape[0]),
                "sequence_shape": list(matrix.shape),
            }
            for split, matrix in sequences.items()
        },
    }

    (OUTPUT_DIR / "bilstm_sequence_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print("\nBiLSTM sequence preparation completed successfully.")


if __name__ == "__main__":
    main()
