from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from deception_detection.data.base import DialogueDataset
from deception_detection.utils import preprocess_dialogue


def _dialogue_to_transcript(dialogue: list[Any]) -> str:
    processed = preprocess_dialogue(dialogue, fold_system=False)
    return "\n".join(f"{m['role']}: {m['content']}" for m in processed)


def export_skipped_control_samples_to_csv(
    dataset: DialogueDataset,
    *,
    skipped_indices: list[int],
    out_csv_path: str | Path,
) -> Path:
    """
    Export skipped control samples to CSV for auditing.

    `skipped_indices` are indices into `dataset` (before any filtering).
    """
    out_csv_path = Path(out_csv_path)

    rows: list[dict[str, Any]] = []
    for i in skipped_indices:
        dialogue = dataset.dialogues[i]
        row: dict[str, Any] = {
            "idx": i,
            "dataset": dataset.name,
            "transcript": _dialogue_to_transcript(dialogue),
        }
        if dataset.metadata is not None:
            for k, v in dataset.metadata.items():
                if i < len(v):
                    row[f"meta_{k}"] = v[i]
        rows.append(row)

    df = pd.DataFrame(rows)
    # If we have a meta_id column, sort by it for easier reading.
    sort_cols = [c for c in ["meta_id", "idx"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    df.to_csv(out_csv_path, index=False)
    return out_csv_path
