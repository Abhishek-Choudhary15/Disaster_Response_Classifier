
"""Prepare BERTweet tokenized inputs using HumAID training data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "models" / "preprocessing"

MODEL_NAME = "vinai/bertweet-base"
MAX_LENGTH = 256

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

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        normalization=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the exact tokenizer files used by this project.
    tokenizer.save_pretrained(ARTIFACT_DIR / "bertweet_tokenizer")

    report = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "padding": "max_length",
        "truncation": True,
        "length_selection_split": "train",
        "splits": {},
    }

    for split, df in datasets.items():
        texts = df["tweet_text"].astype(str).tolist()

        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_attention_mask=True,
            return_tensors="np",
        )

        input_ids = encoded["input_ids"].astype(np.int32)
        attention_mask = encoded["attention_mask"].astype(np.int8)

        np.save(OUTPUT_DIR / f"{split}_bertweet_input_ids.npy", input_ids)
        np.save(
            OUTPUT_DIR / f"{split}_bertweet_attention_mask.npy",
            attention_mask,
        )

        report["splits"][split] = {
            "rows": len(df),
            "input_ids_shape": list(input_ids.shape),
            "attention_mask_shape": list(attention_mask.shape),
            "truncated_or_special_token_adjustment_count": None,
        }

    (OUTPUT_DIR / "bertweet_preprocessing_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print("\nBERTweet preprocessing completed successfully.")


if __name__ == "__main__":
    main()
