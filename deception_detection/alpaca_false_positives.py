from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from deception_detection.experiment import Experiment
from deception_detection.log import logger
from deception_detection.prompt_scorer import PromptScorerMethod, create_prompt_scorer
from deception_detection.utils import preprocess_dialogue


def _dialogue_to_transcript(dialogue: list[Any]) -> str:
    """
    Render a Dialogue (list[Message]) into a readable multi-line transcript.
    Uses preprocess_dialogue to concatenate adjacent same-role messages.
    """
    processed = preprocess_dialogue(dialogue, fold_system=False)
    return "\n".join(f"{m['role']}: {m['content']}" for m in processed)


def export_control_false_positives_to_csv(
    experiment: Experiment,
    *,
    dataset_name: str = "alpaca__plain",
    fpr: float = 0.03,
    scoring_method: PromptScorerMethod = "mean",
    out_csv_path: str | Path | None = None,
) -> Path:
    """
    Export the highest-scoring control examples (i.e. false positives) to CSV.

    We treat "false positives" as the top-fpr fraction of the control distribution, where the
    threshold is the (1 - fpr) quantile of the control prompt-score distribution.
    """
    if not (0 <= fpr <= 1):
        raise ValueError(f"fpr must be in [0, 1], got {fpr}")

    # Ensure we have control datasets + scores computed.
    _ = experiment.get_control_scores(all_tokens=False)

    # Find the control dataset object so we can include dialogue text and metadata.
    dataset = next((d for d in experiment.control_datasets if d.name == dataset_name), None)
    if dataset is None:
        available = [d.name for d in experiment.control_datasets]
        raise ValueError(f"Control dataset {dataset_name} not found. Available: {available}")

    scorer = create_prompt_scorer(
        scoring_method,
        experiment=experiment,
        fprs=[fpr],
        default_fpr=fpr,
        control_dataset_name=dataset_name,
    )

    control_scores_by_prompt = scorer.get_control_scores_by_prompt().float()
    threshold = scorer.prompt_score_threshold(fpr)

    idx = (control_scores_by_prompt > threshold).nonzero().flatten().tolist()
    idx_sorted = sorted(idx, key=lambda i: float(control_scores_by_prompt[i]), reverse=True)

    if out_csv_path is None:
        pct = int(round(fpr * 100))
        out_csv_path = experiment.cfg.folder / f"{dataset_name}__false_positives_top_{pct}pct.csv"
    out_csv_path = Path(out_csv_path)

    rows: list[dict[str, Any]] = []
    for rank, i in enumerate(idx_sorted, start=1):
        dialogue = dataset.dialogues[i]
        row: dict[str, Any] = {
            "rank": rank,
            "score": float(control_scores_by_prompt[i].item()),
            "threshold": float(threshold),
            "fpr": float(fpr),
            "dataset": dataset_name,
            "transcript": _dialogue_to_transcript(dialogue),
        }
        if dataset.metadata is not None:
            for k, v in dataset.metadata.items():
                if i < len(v):
                    row[f"meta_{k}"] = v[i]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv(out_csv_path, index=False)

    logger.info(
        f"[debug] Exported {len(df)} false positives from {dataset_name} "
        f"(top {fpr:.2%}, threshold={threshold:.4f}) to {out_csv_path}"
    )
    print(
        f"[debug] Alpaca/control false positives: n={len(df)} "
        f"(top {fpr:.2%}, threshold={threshold:.4f}) -> {out_csv_path}"
    )

    return out_csv_path
