
"""Transform CrisisLexT26 using frozen HumAID preprocessing artifacts.

No vectorizer or vocabulary is fitted on CrisisLexT26.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing.text_cleaning import clean_tweet_for_tfidf
from src.preprocessing.vocabulary import (
    encode_and_pad,
    tokenize_for_vocabulary,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "crisislext26"
OUTPUT_DIR = ROOT / "data" / "processed" / "crisislext26"
ARTIFACT_DIR = ROOT / "models" / "preprocessing"

# EDA-proposed mapping. Rough mappings are flagged, not treated as
# confirmed ground truth. Unmapped categories remain without a HumAID label.
LABEL_MAP = {
    "Affected individuals": (
        "injured_or_dead_people", "rough"
    ),
    "Caution and advice": (
        "caution_and_advice", "good"
    ),
    "Donations and volunteering": (
        "rescue_volunteering_or_donation_effort", "good"
    ),
    "Infrastructure and utilities": (
        "infrastructure_and_utility_damage", "good"
    ),
    "Not applicable": ("not_humanitarian", "rough"),
    "Other Useful Information": (
        "other_relevant_information", "rough"
    ),
    "Sympathy and support": ("sympathy_and_support", "good"),
    "Not labeled": (None, "unmapped"),
}

def main():
    vectorizer_path = ARTIFACT_DIR / "tfidf_vectorizer.joblib"
    vocabulary_path = ARTIFACT_DIR / "bilstm_vocabulary.joblib"
    mapping_path = ARTIFACT_DIR / "label_mapping.json"

    for path in (vectorizer_path, vocabulary_path, mapping_path):
        if not path.exists():
            raise FileNotFoundError(f"Required HumAID artifact missing: {path}")

    # Load frozen HumAID artifacts. Never fit on CrisisLexT26.
    vectorizer = joblib.load(vectorizer_path)
    vocabulary = joblib.load(vocabulary_path)
    label_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    label2id = label_mapping["label2id"]

    files = sorted(RAW_DIR.rglob("*-tweets_labeled.csv"))
    if not files:
        raise FileNotFoundError(f"No labeled event CSV files found under {RAW_DIR}")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]

        required = {"Tweet Text", "Information Type"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"{path} is missing columns: {sorted(required - set(df.columns))}"
            )

        # Preserve event identity from the source directory.
        frame = pd.DataFrame({
            "event_name": path.parent.name,
            "source_file": path.name,
            "tweet_id": (
                df["Tweet ID"] if "Tweet ID" in df.columns else pd.NA
            ),
            "original_text": df["Tweet Text"],
            "information_type": df["Information Type"].astype("string").str.strip(),
            "informativeness": (
                df["Informativeness"].astype("string").str.strip()
                if "Informativeness" in df.columns else pd.NA
            ),
        })
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    result["original_text"] = result["original_text"].fillna("").astype(str)
    result["cleaned_text"] = result["original_text"].map(clean_tweet_for_tfidf)

    mapped = result["information_type"].map(LABEL_MAP)
    result["mapping_quality"] = mapped.map(
        lambda x: x[1] if isinstance(x, tuple) else "unmapped"
    )
    result["humaid_label"] = mapped.map(
        lambda x: x[0] if isinstance(x, tuple) else None
    )

    # Only exact HumAID labels from the saved mapping receive IDs.
    result["label_id"] = result["humaid_label"].map(label2id)
    invalid = result["humaid_label"].notna() & result["label_id"].isna()
    result.loc[invalid, "mapping_quality"] = "unmapped_label_not_in_humaid"

    # OOV audit against the actual frozen BiLSTM vocabulary.
    total_tokens = 0
    oov_tokens = 0
    oov_rates = []
    for text in result["cleaned_text"]:
        tokens = tokenize_for_vocabulary(text)
        total = len(tokens)
        oov = sum(token not in vocabulary for token in tokens)
        total_tokens += total
        oov_tokens += oov
        oov_rates.append(oov / total if total else 0.0)
    result["bilstm_oov_rate_per_tweet"] = oov_rates

    # Frozen HumAID transformations; no fit/fit_transform calls here.
    tfidf = vectorizer.transform(result["cleaned_text"].tolist())
    sequences = [
        encode_and_pad(text, vocabulary)
        for text in result["cleaned_text"]
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf, OUTPUT_DIR / "crisislex26_humaid_tfidf_v1.joblib")
    joblib.dump(sequences, OUTPUT_DIR / "crisislex26_humaid_sequences_v1.joblib")
    result.to_csv(OUTPUT_DIR / "crisislex26_transformed_v1.csv", index=False)

    report = {
        "source_event_files": len(files),
        "rows": len(result),
        "events": int(result["event_name"].nunique()),
        "tfidf_shape": list(tfidf.shape),
        "sequence_shape": [len(sequences), 64],
        "humAID_tfidf_features": int(tfidf.shape[1]),
        "total_bilstm_tokens": total_tokens,
        "oov_bilstm_tokens": oov_tokens,
        "bilstm_oov_rate": (
            oov_tokens / total_tokens if total_tokens else 0.0
        ),
        "mapping_quality_counts": (
            result["mapping_quality"].value_counts(dropna=False).to_dict()
        ),
        "information_type_counts": (
            result["information_type"].value_counts(dropna=False).to_dict()
        ),
        "label_id_missing_count": int(result["label_id"].isna().sum()),
        "artifacts": [
            "crisislex26_humaid_tfidf_v1.joblib",
            "crisislex26_humaid_sequences_v1.joblib",
            "crisislex26_transformed_v1.csv",
        ],
        "fitting_performed_on_crisislex26": False,
    }
    (OUTPUT_DIR / "crisislex26_preprocessing_report_v1.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
