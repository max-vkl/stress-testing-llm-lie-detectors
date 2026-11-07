"""Freeze a DolusChat seed-0 detector pool or full corpus from pinned source data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, cast

import numpy as np

from deception_detection.utils import REPO_ROOT

DATASET = "AlignmentResearch/DolusChat"
REVISION = "830e454b54349f8c923a7377e28c17d1796892cb"
PARQUET = "data/train-00000-of-00001.parquet"
SOURCE_SHA256 = "343f53d05916caf006c4761b152fc22a0820f14887e243dfcc01c3e6fefb30e2"
CODE_REPO = "https://github.com/AlignmentResearch/deception-evasion-honesty"
CODE_REVISION = "ba93d3fe53fe334d87daa44d7e7eb3ddd11230d9"
SOURCE_ROWS = 64888
OUTPUT = REPO_ROOT / "data/dolus_chat/dataset.json"
OUTPUTS = {"seed0": OUTPUT, "full_seed0": OUTPUT.with_name("full_dataset.json")}


def reconstruct_pairs(rows: list[dict[str, Any]], variant: str = "seed0") -> list[dict[str, Any]]:
    """Match munge_data's seeded split and context order without global RNG changes.

    Source preprocessing transforms preference-training rows BEFORE train_lr;
    those shuffles must consume the same Python RNG, even though we discard them.
    The full variant also keeps preference and source-test rows, continuing the
    same RNG through all three partitions in the upstream processing order.
    """
    if variant not in OUTPUTS:
        raise ValueError(f"Unknown DolusChat variant: {variant}")
    non_test = len(rows) - int(len(rows) * 0.05)
    preference = non_test - int(non_test * 0.05)
    order = np.random.RandomState(0).permutation(len(rows))
    rng = random.Random(0)
    pairs: list[dict[str, Any]] = []
    selected_order = order if variant == "full_seed0" else order[:non_test]
    for position, index in enumerate(selected_order):
        row = rows[int(index)]
        contexts = list(row["context"]["non_sensitive_info"]) + [row["context"]["sensitive_info"]]
        rng.shuffle(contexts)
        if variant == "seed0" and position < preference:
            continue
        pairs.append(
            {
                "source_row": int(index),
                "id": row["id"],
                "system": row["system_message"] + "\nContext:\n" + "\n".join(contexts),
                "user": (
                    " User ID: "
                    + row["user_query"]["speaker_type"]
                    + "\n"
                    + row["user_query"]["content"]
                ),
                "responses": row["responses"],
                "source_system_message": row["system_message"],
                "source_context": row["context"],
                "source_user_query": row["user_query"],
            }
        )
    return pairs


def build_artifact(source: Path, variant: str = "seed0") -> dict[str, Any]:
    """Verify the exact parquet before importing; no source-side calibration split."""
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError(
            f"Source parquet SHA256 does not match pinned DolusChat revision: {source}"
        )
    import pyarrow.parquet as pq

    rows = cast(list[dict[str, Any]], pq.read_table(source).to_pylist())
    if len(rows) != SOURCE_ROWS:
        raise ValueError(f"Expected {SOURCE_ROWS} source rows, found {len(rows)}")
    pairs = reconstruct_pairs(rows, variant)
    expected_pairs = SOURCE_ROWS if variant == "full_seed0" else 3082
    if len(pairs) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} pairs, found {len(pairs)}")
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "dataset": DATASET,
            "revision": REVISION,
            "file": PARQUET,
            "sha256": SOURCE_SHA256,
            "rows": SOURCE_ROWS,
            "code_repository": CODE_REPO,
            "code_revision": CODE_REVISION,
            "authors": ["Chris Cundy", "Adam Gleave"],
            "license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
        },
        "selection": {
            "seed": 0,
            "test_fraction": 0.05,
            "detector_fraction_of_non_test": 0.05,
            "preference_pairs": 58562,
            "detector_pairs": 3082,
            "response_order": ["truthful", "deceptive"],
            "description": "Released non-debug seed-0 detector pool, including calibration rows; "
            "not an authenticated original paper run. Context order reconstructed with the "
            "source RNG procedure. Original texts retained; native chat normalization occurs "
            "only in the dataset adapter. No source detector split or length cap applied.",
        },
        "pairs": pairs,
    }
    if variant == "full_seed0":
        artifact["selection"] = {
            "variant": variant,
            "seed": 0,
            "selected_pairs": SOURCE_ROWS,
            "includes_source_test": True,
            "source_partition_pairs": {"preference": 58562, "detector": 3082, "test": 3244},
            "response_order": ["truthful", "deceptive"],
            "description": "All released source rows in seed-0 permutation order, including "
            "the source test partition. Context shuffles continue across preference, detector, "
            "and test partitions exactly as upstream. Existing detector-pool pairs are "
            "unchanged. Native chat normalization, seed-42 response shuffle, 20% validation "
            "split, and class balancing occur in the standard repository pipeline. No length "
            "filter or deduplication applied.",
        }
    return artifact


def publish_artifact(artifact: dict[str, Any], output: Path) -> bool:
    """Write once, accepting identical reruns and rejecting conflicting artifacts."""
    content = (json.dumps(artifact, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if output.read_bytes() != content:
            raise ValueError(f"Refusing to replace conflicting frozen artifact: {output}") from None
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path)
    parser.add_argument("--variant", choices=list(OUTPUTS), default="seed0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.source_parquet
    if source is None:
        from huggingface_hub import hf_hub_download

        source = Path(
            hf_hub_download(
                repo_id=DATASET,
                repo_type="dataset",
                revision=REVISION,
                filename=PARQUET,
                cache_dir=str(REPO_ROOT / ".data/dolus_chat"),
            )
        )
    artifact = build_artifact(source, args.variant)
    output = args.output or OUTPUTS[args.variant]
    created = publish_artifact(artifact, output)
    action = "Created" if created else "Verified identical"
    count = len(artifact["pairs"])
    print(f"{action} {output}: {count} pairs / {2 * count} responses")


if __name__ == "__main__":
    main()
