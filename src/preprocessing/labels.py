
"""Stable label mappings for the HumAID dataset."""

from pathlib import Path
import json

import pandas as pd

LABEL_COLUMN = "class_label"


def build_label_mapping(labels):
    """Build a deterministic mapping from sorted class names to integer IDs."""
    unique_labels = sorted(set(labels))

    if not unique_labels:
        raise ValueError("Cannot build a label mapping from empty labels.")

    label2id = {label: index for index, label in enumerate(unique_labels)}
    id2label = {str(index): label for label, index in label2id.items()}

    return label2id, id2label


def save_label_mapping(label2id, id2label, output_path):
    """Save label mappings as a JSON artifact."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "label2id": label2id,
        "id2label": id2label,
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def encode_labels(labels, label2id):
    """Encode labels using a mapping created from training labels."""
    unknown = sorted(set(labels) - set(label2id))

    if unknown:
        raise ValueError(f"Unknown labels encountered: {unknown}")

    return pd.Series(labels).map(label2id).to_numpy()
