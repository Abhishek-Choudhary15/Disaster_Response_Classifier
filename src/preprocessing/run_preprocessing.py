from sklearn.preprocessing import StandardScaler
"""Run reproducible HumAID feature engineering and preprocessing."""


"""Run reproducible HumAID feature engineering and preprocessing."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.preprocessing.text_cleaning import clean_tweet_for_tfidf
from src.preprocessing.labels import build_label_mapping, encode_labels
from src.preprocessing.vectorization import fit_tfidf, transform_tfidf
from src.preprocessing.features import extract_tweet_features


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

    # Load and validate all three existing splits.
    for split, filename in SPLITS.items():
        path = RAW_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Required dataset not found: {path}")

        df = pd.read_csv(path)

        required = {"tweet_text", "class_label"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"{filename} must contain columns {sorted(required)}"
            )

        if df["tweet_text"].isna().any() or df["class_label"].isna().any():
            raise ValueError(f"Missing text or labels found in {filename}")

        datasets[split] = df

    # Build the label mapping from training labels only.
    label2id, id2label = build_label_mapping(
        datasets["train"]["class_label"]
    )

    for split, df in datasets.items():
        unknown = set(df["class_label"]) - set(label2id)
        if unknown:
            raise ValueError(
                f"Unknown labels in {split}: {sorted(unknown)}"
            )

    # Clean text, extract numerical features, and encode labels.
    cleaned_texts = {}
    numerical_features = {}
    encoded_labels = {}

    for split, df in datasets.items():
        cleaned_texts[split] = df["tweet_text"].map(
            clean_tweet_for_tfidf
        ).tolist()

        numerical_features[split] = extract_tweet_features(
            df["tweet_text"]
        )

        encoded_labels[split] = encode_labels(
            df["class_label"], label2id
        )

    # Fit TF-IDF on training data only.
    vectorizer, train_matrix = fit_tfidf(cleaned_texts["train"])

    matrices = {
        "train": train_matrix,
        "validation": transform_tfidf(
            vectorizer, cleaned_texts["validation"]
        ),
        "test": transform_tfidf(
            vectorizer, cleaned_texts["test"]
        ),
    }

    # Fit the numerical scaler on training data only.
    scaler = StandardScaler()
    scaler.fit(numerical_features["train"])

    # Apply the same fitted scaler to all splits.
    scaled_numerical_features = {
        split: pd.DataFrame(
            scaler.transform(numerical_features[split]),
            columns=numerical_features[split].columns,
            index=numerical_features[split].index,
        )
        for split in datasets
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Save reusable learned artifacts.
    joblib.dump(
        vectorizer, ARTIFACT_DIR / "tfidf_vectorizer.joblib"
    )
    joblib.dump(
        scaler, ARTIFACT_DIR / "numerical_scaler.joblib"
    )

    (ARTIFACT_DIR / "label_mapping.json").write_text(
        json.dumps(
            {"label2id": label2id, "id2label": id2label},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Save features, labels, and sparse TF-IDF matrices.
    for split, df in datasets.items():
        numerical_features[split].to_csv(
            OUTPUT_DIR / f"{split}_numerical_features.csv",
            index=False,
        )

        scaled_numerical_features[split].to_csv(
            OUTPUT_DIR / f"{split}_scaled_numerical_features.csv",
            index=False,
        )

        pd.DataFrame({
            "label_id": encoded_labels[split]
        }).to_csv(
            OUTPUT_DIR / f"{split}_labels.csv",
            index=False,
        )

        joblib.dump(
            matrices[split],
            OUTPUT_DIR / f"{split}_tfidf.npz",
        )

    # Write a compact report.
    report = {
        split: {
            "rows": len(datasets[split]),
            "classes": int(datasets[split]["class_label"].nunique()),
            "tfidf_shape": list(matrices[split].shape),
            "numerical_feature_shape": list(
                numerical_features[split].shape
            ),
            "scaled_numerical_feature_shape": list(
                scaled_numerical_features[split].shape
            ),
        }
        for split in datasets
    }

    (OUTPUT_DIR / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()
