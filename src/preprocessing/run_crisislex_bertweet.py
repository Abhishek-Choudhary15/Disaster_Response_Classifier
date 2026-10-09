
"""Transform CrisisLexT26 with the existing HumAID BERTweet tokenizer."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = (
    ROOT / "data" / "processed" / "crisislext26"
    / "crisislex26_transformed_v1.csv"
)
TOKENIZER_DIR = ROOT / "models" / "preprocessing" / "bertweet_tokenizer"
OUTPUT_DIR = ROOT / "data" / "processed" / "crisislext26"

MAX_LENGTH = 256  # Match the existing HumAID BERTweet pipeline.


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_FILE}")
    if not TOKENIZER_DIR.exists():
        raise FileNotFoundError(f"Missing saved tokenizer: {TOKENIZER_DIR}")

    df = pd.read_csv(INPUT_FILE)
    if "original_text" not in df.columns:
        raise ValueError("Expected original_text in CrisisLexT26 output")

    # Load the existing tokenizer; do not fit or modify it.
    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_DIR),
        local_files_only=True,
    )

    # Match the HumAID transformer path: original text, not TF-IDF-cleaned text.
    texts = df["original_text"].fillna("").astype(str).tolist()
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_attention_mask=True,
        return_tensors="np",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(
        OUTPUT_DIR / "crisislex26_bertweet_input_ids_v1.npy",
        encoded["input_ids"].astype(np.int32),
    )
    np.save(
        OUTPUT_DIR / "crisislex26_bertweet_attention_mask_v1.npy",
        encoded["attention_mask"].astype(np.int8),
    )

    report = {
        "tokenizer_source": str(TOKENIZER_DIR.relative_to(ROOT)),
        "max_length": MAX_LENGTH,
        "padding": "max_length",
        "truncation": True,
        "rows": len(df),
        "input_ids_shape": list(encoded["input_ids"].shape),
        "attention_mask_shape": list(encoded["attention_mask"].shape),
        "fitting_performed_on_crisislex26": False,
    }
    (OUTPUT_DIR / "crisislex26_bertweet_report_v1.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
