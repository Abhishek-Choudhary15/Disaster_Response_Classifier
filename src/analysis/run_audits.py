"""Reproducible audits that back checklist items B5, D5 and the leakage checks.

Run from repo root (after the preprocessing scripts):
    python -m src.analysis.run_audits
Writes docs/oov_audit.json and docs/split_integrity_audit.json.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing.config import HUMAID_VOCAB_ARTIFACT
from src.preprocessing.text_cleaning import clean_tweet_for_tfidf, normalize_for_dedup
from src.preprocessing.vocabulary import tokenize_for_vocabulary

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
SPLITS = ["train", "validation", "test"]


def oov_stats(texts, vocab):
    tokens = [t for x in texts for t in tokenize_for_vocabulary(x)]
    n_oov = sum(t not in vocab for t in tokens)
    return {"tokens": len(tokens), "oov_tokens": n_oov,
            "oov_rate_pct": round(100 * n_oov / max(1, len(tokens)), 2)}


def overlap(a: pd.Series, b: pd.Series):
    """Exact and near-duplicate overlap of b against a."""
    exact_shared = set(a.str.strip()) & set(b.str.strip())
    ka, kb = a.map(normalize_for_dedup), b.map(normalize_for_dedup)
    near_shared = set(ka) & set(kb)
    return {
        "exact_shared_texts": len(exact_shared),
        "near_shared_texts": len(near_shared),
        "rows_in_second_with_near_twin": int(kb.isin(near_shared).sum()),
        "second_rows": len(b),
        "pct_of_second": round(100 * float(kb.isin(near_shared).mean()), 2),
    }


def main():
    DOCS.mkdir(exist_ok=True)
    vocab = joblib.load(ROOT / "models" / "preprocessing" / HUMAID_VOCAB_ARTIFACT)

    humaid = {s: pd.read_csv(ROOT / f"data/raw/humaid_{s}.csv")["tweet_text"] for s in SPLITS}
    kaggle = {s: pd.read_csv(ROOT / f"data/processed/kaggle/{s}_text.csv") for s in SPLITS}
    cl = pd.read_csv(ROOT / "data/processed/crisislext26/crisislex26_transformed_v1.csv")

    # ---- OOV against the FROZEN HumAID vocabulary (B5, D5)
    oov = {f"humaid_{s}": oov_stats(humaid[s].map(clean_tweet_for_tfidf), vocab) for s in SPLITS}
    oov["kaggle_all_splits"] = oov_stats(
        pd.concat(kaggle.values())["cleaned_text"].astype(str), vocab)
    oov["crisislex26_all"] = oov_stats(cl["cleaned_text"].astype(str), vocab)
    oov["crisislex26_excluding_duplicates"] = oov_stats(
        cl.loc[~cl["is_duplicate_text"], "cleaned_text"].astype(str), vocab)
    (DOCS / "oov_audit.json").write_text(json.dumps(oov, indent=2), encoding="utf-8")

    # ---- Split integrity: exact + near-duplicate overlap (leakage)
    integrity = {"humaid": {}, "kaggle": {}, "cross_dataset": {}}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        integrity["humaid"][f"{a}->{b}"] = overlap(humaid[a], humaid[b])
        integrity["kaggle"][f"{a}->{b}"] = overlap(kaggle[a]["text"], kaggle[b]["text"])
    humaid_all = pd.concat(humaid.values(), ignore_index=True)
    integrity["cross_dataset"]["humaid_all->crisislex26"] = overlap(
        humaid_all, cl["original_text"].astype(str))
    (DOCS / "split_integrity_audit.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    print("OOV vs frozen HumAID vocabulary")
    for k, v in oov.items():
        print(f"  {k:34} {v['oov_rate_pct']:6.2f}%  ({v['oov_tokens']:,}/{v['tokens']:,})")
    print("Near-duplicate rows with a twin in an earlier split")
    for ds in ("humaid", "kaggle", "cross_dataset"):
        for k, v in integrity[ds].items():
            print(f"  {ds:14} {k:28} {v['rows_in_second_with_near_twin']:5d} rows ({v['pct_of_second']}%)")


if __name__ == "__main__":
    main()
