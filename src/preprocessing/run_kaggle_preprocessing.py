
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from src.preprocessing.text_cleaning import clean_tweet_for_tfidf


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed/kaggle")
MODEL_DIR = Path("models/preprocessing")

SEED = 42
TEST_SIZE = 0.10
VALIDATION_SIZE = 0.10


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_path = RAW_DIR / "nlpgettingstarted_train"
    df = pd.read_csv(train_path)

    required = {"text", "target"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")

    # Use only labeled competition training data.
    df = df[["text", "target"]].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["target"] = df["target"].astype(int)

    if not set(df["target"].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    # Apply the shared TF-IDF cleaning function.
    df["cleaned_text"] = df["text"].map(clean_tweet_for_tfidf)

    # Split labeled data into 80% train, 10% validation, 10% test.
    train_df, temp_df = train_test_split(
        df,
        test_size=TEST_SIZE + VALIDATION_SIZE,
        random_state=SEED,
        stratify=df["target"],
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_df["target"],
    )

    # Fit TF-IDF on Kaggle training text only.
    vectorizer = TfidfVectorizer(
        max_features=50000,
        min_df=2,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )

    X_train = vectorizer.fit_transform(train_df["cleaned_text"])
    X_validation = vectorizer.transform(validation_df["cleaned_text"])
    X_test = vectorizer.transform(test_df["cleaned_text"])

    splits = {
        "train": (train_df, X_train),
        "validation": (validation_df, X_validation),
        "test": (test_df, X_test),
    }

    for name, (split_df, X) in splits.items():
        split_df[["target"]].to_csv(
            PROCESSED_DIR / f"{name}_labels.csv", index=False
        )
        joblib.dump(X, PROCESSED_DIR / f"{name}_tfidf.joblib")
        split_df[["text", "cleaned_text"]].to_csv(
            PROCESSED_DIR / f"{name}_text.csv", index=False
        )

    joblib.dump(
        vectorizer,
        MODEL_DIR / "tfidf_kaggle_prefilter_v1.joblib",
    )

    report = {
        "dataset": "Kaggle NLP with Disaster Tweets",
        "seed": SEED,
        "split_sizes": {
            name: len(split_df)
            for name, (split_df, _) in splits.items()
        },
        "target_distribution": {
            name: split_df["target"].value_counts().sort_index().to_dict()
            for name, (split_df, _) in splits.items()
        },
        "tfidf_features": X_train.shape[1],
        "vectorizer_fit_on": "Kaggle labeled training split only",
        "keyword_used": False,
        "location_used": False,
        "competition_test_used_for_fitting": False,
        "target_mapping": {
            "0": "not disaster-related",
            "1": "disaster-related",
        },
    }

    with open(PROCESSED_DIR / "kaggle_preprocessing_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print("\nKaggle preprocessing completed.")


if __name__ == "__main__":
    main()
