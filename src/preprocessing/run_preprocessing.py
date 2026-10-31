"""Run reproducible HumAID feature engineering (TF-IDF, numeric features, labels).

Everything learned (vectorizer, scaler, label map) is fitted on the TRAIN split
only and then applied unchanged to validation and test.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.preprocessing.config import (
    HUMAID_LABEL_MAP_ARTIFACT,
    HUMAID_SCALER_ARTIFACT,
    HUMAID_TFIDF_ARTIFACT,
)
from src.preprocessing.features import extract_tweet_features
from src.preprocessing.labels import (
    build_label_mapping,
    describe_label_schema,
    encode_labels,
)
from src.preprocessing.text_cleaning import clean_tweet_for_tfidf
from src.preprocessing.vectorization import (
    TFIDF_TOKEN_PATTERN,
    fit_tfidf,
    transform_tfidf,
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

    # 1. Load and validate all three official splits (never re-shuffled).
    for split, filename in SPLITS.items():
        path = RAW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Required dataset not found: {path}")
        df = pd.read_csv(path)
        required = {"tweet_text", "class_label"}
        if not required.issubset(df.columns):
            raise ValueError(f"{filename} must contain columns {sorted(required)}")
        if df["tweet_text"].isna().any() or df["class_label"].isna().any():
            raise ValueError(f"Missing text or labels found in {filename}")
        datasets[split] = df

    # 2. Label mapping from TRAIN labels only; every split must use known labels.
    label2id, id2label = build_label_mapping(datasets["train"]["class_label"])
    for split, df in datasets.items():
        unknown = set(df["class_label"]) - set(label2id)
        if unknown:
            raise ValueError(f"Unknown labels in {split}: {sorted(unknown)}")

    # 3. Clean text, extract numeric features (from tweet text only), encode labels.
    cleaned_texts, numerical_features, encoded_labels = {}, {}, {}
    for split, df in datasets.items():
        cleaned_texts[split] = df["tweet_text"].map(clean_tweet_for_tfidf).tolist()
        numerical_features[split] = extract_tweet_features(df["tweet_text"])
        encoded_labels[split] = encode_labels(df["class_label"], label2id)

    # 4. Fit TF-IDF on TRAIN only; transform the other splits.
    vectorizer, train_matrix = fit_tfidf(cleaned_texts["train"])
    matrices = {
        "train": train_matrix,
        "validation": transform_tfidf(vectorizer, cleaned_texts["validation"]),
        "test": transform_tfidf(vectorizer, cleaned_texts["test"]),
    }

    # 5. Fit the numeric scaler on TRAIN only.
    scaler = StandardScaler().fit(numerical_features["train"])
    scaled = {
        split: pd.DataFrame(
            scaler.transform(numerical_features[split]),
            columns=numerical_features[split].columns,
        )
        for split in datasets
    }

    # 6. Row-alignment guard: every representation must match its split's rows.
    for split, df in datasets.items():
        n = len(df)
        sizes = {
            "tfidf": matrices[split].shape[0],
            "numeric": len(numerical_features[split]),
            "scaled_numeric": len(scaled[split]),
            "labels": len(encoded_labels[split]),
        }
        if set(sizes.values()) != {n}:
            raise ValueError(f"Row misalignment in {split}: expected {n}, got {sizes}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # 7. Save versioned, dataset-tagged artifacts (names live in config.py).
    joblib.dump(vectorizer, ARTIFACT_DIR / HUMAID_TFIDF_ARTIFACT)
    joblib.dump(scaler, ARTIFACT_DIR / HUMAID_SCALER_ARTIFACT)
    (ARTIFACT_DIR / HUMAID_LABEL_MAP_ARTIFACT).write_text(
        json.dumps({"label2id": label2id, "id2label": id2label},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for split in datasets:
        numerical_features[split].to_csv(OUTPUT_DIR / f"{split}_numerical_features.csv", index=False)
        scaled[split].to_csv(OUTPUT_DIR / f"{split}_scaled_numerical_features.csv", index=False)
        pd.DataFrame({"label_id": encoded_labels[split]}).to_csv(
            OUTPUT_DIR / f"{split}_labels.csv", index=False)
        joblib.dump(matrices[split], OUTPUT_DIR / f"{split}_tfidf.joblib")

    # 8. Report.
    report = {
        "fit_split": "train",
        "tfidf_token_pattern": TFIDF_TOKEN_PATTERN,
        "label_schema": describe_label_schema(datasets["train"]["class_label"]),
        "splits": {
            split: {
                "rows": len(datasets[split]),
                "classes": int(datasets[split]["class_label"].nunique()),
                "tfidf_shape": list(matrices[split].shape),
                "numerical_feature_shape": list(numerical_features[split].shape),
                "scaled_numerical_feature_shape": list(scaled[split].shape),
            }
            for split in datasets
        },
    }
    (OUTPUT_DIR / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()
