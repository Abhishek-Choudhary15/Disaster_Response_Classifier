"""Transform CrisisLexT26 using FROZEN HumAID preprocessing artifacts.

CrisisLexT26 is an external evaluation set. Nothing is ever fitted on it: this
script only loads saved HumAID objects and calls transform/encode on them.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing.config import (
    HUMAID_LABEL_MAP_ARTIFACT,
    HUMAID_TFIDF_ARTIFACT,
    HUMAID_VOCAB_ARTIFACT,
)
from src.preprocessing.labels import CRISISLEX_LABEL_MAP, CRISISLEX_PREFILTER_MAP
from src.preprocessing.language import guess_language
from src.preprocessing.text_cleaning import clean_tweet_for_tfidf, normalize_for_dedup
from src.preprocessing.vocabulary import (
    MAX_SEQUENCE_LENGTH,
    encode_and_pad,
    tokenize_for_vocabulary,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "crisislext26"
OUTPUT_DIR = ROOT / "data" / "processed" / "crisislext26"
ARTIFACT_DIR = ROOT / "models" / "preprocessing"


def main():
    vectorizer_path = ARTIFACT_DIR / HUMAID_TFIDF_ARTIFACT
    vocabulary_path = ARTIFACT_DIR / HUMAID_VOCAB_ARTIFACT
    mapping_path = ARTIFACT_DIR / HUMAID_LABEL_MAP_ARTIFACT
    for path in (vectorizer_path, vocabulary_path, mapping_path):
        if not path.exists():
            raise FileNotFoundError(f"Required HumAID artifact missing: {path}")

    # Frozen HumAID artifacts (loaded by explicit versioned filename).
    vectorizer = joblib.load(vectorizer_path)
    vocabulary = joblib.load(vocabulary_path)
    label2id = json.loads(mapping_path.read_text(encoding="utf-8"))["label2id"]

    files = sorted(RAW_DIR.rglob("*-tweets_labeled.csv"))
    if not files:
        raise FileNotFoundError(f"No labeled event CSV files found under {RAW_DIR}")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        required = {"Tweet Text", "Information Type"}
        if not required.issubset(df.columns):
            raise ValueError(f"{path} is missing columns: {sorted(required - set(df.columns))}")

        def col(name):
            return df[name].astype("string").str.strip() if name in df.columns else pd.NA

        # Preserve event identity from the source directory (D6).
        frames.append(pd.DataFrame({
            "event_name": path.parent.name,
            "source_file": path.name,
            "tweet_id": df["Tweet ID"] if "Tweet ID" in df.columns else pd.NA,
            "original_text": df["Tweet Text"],
            "information_type": col("Information Type"),
            "informativeness": col("Informativeness"),
            "information_source": col("Information Source"),
        }))

    result = pd.concat(frames, ignore_index=True)
    result["original_text"] = result["original_text"].fillna("").astype(str)

    # Same shared cleaning function as HumAID (D2).
    result["cleaned_text"] = result["original_text"].map(clean_tweet_for_tfidf)

    # ---- 11-class mapping (flagged, never treated as confirmed ground truth)
    mapped = result["information_type"].map(CRISISLEX_LABEL_MAP)
    result["mapping_quality"] = mapped.map(lambda x: x[1] if isinstance(x, tuple) else "unmapped")
    result["humaid_label"] = mapped.map(lambda x: x[0] if isinstance(x, tuple) else None)
    result["label_id"] = result["humaid_label"].map(label2id)   # saved HumAID IDs only
    invalid = result["humaid_label"].notna() & result["label_id"].isna()
    result.loc[invalid, "mapping_quality"] = "unmapped_label_not_in_humaid"

    # ---- binary relatedness label to evaluate the Kaggle-trained pre-filter
    result["prefilter_label"] = result["informativeness"].map(CRISISLEX_PREFILTER_MAP)

    # ---- duplicate flag (retweets / copies); rows are kept, never dropped
    result["is_duplicate_text"] = result["original_text"].map(normalize_for_dedup).duplicated()

    # ---- language: CrisisLexT26 is MULTILINGUAL (several events are mostly
    #      Spanish/Italian/Portuguese/French). The HumAID models are English-only,
    #      so evaluation flags below require an English guess.
    result["language_guess"] = result["original_text"].map(guess_language)
    result["is_english_guess"] = result["language_guess"].eq("en")

    result["use_for_class_eval"] = (result["label_id"].notna() & ~result["is_duplicate_text"]
                                    & result["is_english_guess"])
    result["use_for_prefilter_eval"] = (result["prefilter_label"].notna() & ~result["is_duplicate_text"]
                                        & result["is_english_guess"])

    # ---- OOV against the frozen BiLSTM vocabulary (D5)
    total_tokens = oov_tokens = 0
    per_tweet = []
    for text in result["cleaned_text"]:
        tokens = tokenize_for_vocabulary(text)
        oov = sum(token not in vocabulary for token in tokens)
        total_tokens += len(tokens)
        oov_tokens += oov
        per_tweet.append(oov / len(tokens) if tokens else 0.0)
    result["bilstm_oov_rate_per_tweet"] = per_tweet

    # ---- frozen HumAID transformations (transform only)
    tfidf = vectorizer.transform(result["cleaned_text"].tolist())
    sequences = [encode_and_pad(t, vocabulary) for t in result["cleaned_text"]]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf, OUTPUT_DIR / "crisislex26_humaid_tfidf_v1.joblib")
    joblib.dump(sequences, OUTPUT_DIR / "crisislex26_humaid_sequences_v1.joblib")
    result.to_csv(OUTPUT_DIR / "crisislex26_transformed_v1.csv", index=False)

    classes_with_support = sorted(result.loc[result["use_for_class_eval"], "humaid_label"].unique())
    report = {
        "source_event_files": len(files),
        "rows": len(result),
        "events": int(result["event_name"].nunique()),
        "tfidf_shape": list(tfidf.shape),
        "sequence_shape": [len(sequences), MAX_SEQUENCE_LENGTH],
        "humaid_tfidf_features": int(tfidf.shape[1]),
        "total_bilstm_tokens": total_tokens,
        "oov_bilstm_tokens": oov_tokens,
        "bilstm_oov_rate": oov_tokens / total_tokens if total_tokens else 0.0,
        "mapping_quality_counts": result["mapping_quality"].value_counts(dropna=False).to_dict(),
        "information_type_counts": result["information_type"].value_counts(dropna=False).to_dict(),
        "label_id_missing_count": int(result["label_id"].isna().sum()),
        "duplicate_text_rows": int(result["is_duplicate_text"].sum()),
        "english_guess_rows": int(result["is_english_guess"].sum()),
        "english_share_pct": round(float(result["is_english_guess"].mean() * 100), 1),
        "language_guess_top": result["language_guess"].value_counts().head(8).to_dict(),
        "english_share_pct_by_event": (result.groupby("event_name")["is_english_guess"].mean() * 100).round(1).to_dict(),
        "class_eval_rows_before_filters": int(result["label_id"].notna().sum()),
        "class_eval_rows": int(result["use_for_class_eval"].sum()),
        "class_eval_classes_with_support": classes_with_support,
        "humaid_classes_without_crisislex_counterpart": sorted(set(label2id) - set(classes_with_support)),
        "prefilter_label_counts": result["prefilter_label"].value_counts(dropna=False).to_dict(),
        "prefilter_eval_rows_before_filters": int(result["prefilter_label"].notna().sum()),
        "prefilter_eval_rows": int(result["use_for_prefilter_eval"].sum()),
        "artifacts": [
            "crisislex26_humaid_tfidf_v1.joblib",
            "crisislex26_humaid_sequences_v1.joblib",
            "crisislex26_transformed_v1.csv",
        ],
        "fitting_performed_on_crisislex26": False,
    }
    (OUTPUT_DIR / "crisislex26_preprocessing_report_v1.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
