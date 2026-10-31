"""Stable label mappings for HumAID and the CrisisLexT26 evaluation mapping."""

import json
from pathlib import Path

import pandas as pd

from src.preprocessing.config import HUMAID_CARD_LABELS

LABEL_COLUMN = "class_label"

# CrisisLexT26 'Information Type' -> HumAID class. 'rough' rows are flagged,
# never treated as confirmed ground truth. CrisisLexT26 has NO counterpart for
# displaced_people_and_evacuations, missing_or_found_people or
# requests_or_urgent_needs, so 11-class (10-class) evaluation can only cover the
# 7 classes below.
CRISISLEX_LABEL_MAP = {
    "Affected individuals": ("injured_or_dead_people", "rough"),
    "Caution and advice": ("caution_and_advice", "good"),
    "Donations and volunteering": ("rescue_volunteering_or_donation_effort", "good"),
    "Infrastructure and utilities": ("infrastructure_and_utility_damage", "good"),
    "Not applicable": ("not_humanitarian", "rough"),
    "Other Useful Information": ("other_relevant_information", "rough"),
    "Sympathy and support": ("sympathy_and_support", "good"),
    "Not labeled": (None, "unmapped"),
}

# CrisisLexT26 'Informativeness' -> binary relatedness, used to evaluate the
# Kaggle-trained pre-filter ("Not applicable" stays unlabeled).
CRISISLEX_PREFILTER_MAP = {
    "Not related": 0,
    "Related and informative": 1,
    "Related - but not informative": 1,
}


def build_label_mapping(labels):
    """Build a deterministic mapping from sorted class names to integer IDs."""
    unique_labels = sorted(set(labels))
    if not unique_labels:
        raise ValueError("Cannot build a label mapping from empty labels.")
    label2id = {label: index for index, label in enumerate(unique_labels)}
    id2label = {str(index): label for label, index in label2id.items()}
    return label2id, id2label


def describe_label_schema(labels):
    """Compare the labels actually present with the public dataset card."""
    present = sorted(set(labels))
    return {
        "num_classes_present": len(present),
        "classes_present": present,
        "card_labels_absent_from_data": sorted(set(HUMAID_CARD_LABELS) - set(present)),
        "labels_not_on_card": sorted(set(present) - set(HUMAID_CARD_LABELS)),
    }


def save_label_mapping(label2id, id2label, output_path):
    """Save label mappings as a JSON artifact."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"label2id": label2id, "id2label": id2label},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def encode_labels(labels, label2id):
    """Encode labels using a mapping created from training labels."""
    unknown = sorted(set(labels) - set(label2id))
    if unknown:
        raise ValueError(f"Unknown labels encountered: {unknown}")
    return pd.Series(labels).map(label2id).to_numpy()
