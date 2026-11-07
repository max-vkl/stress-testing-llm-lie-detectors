"""Opt-in, auditable holdouts over unchanged complete rollout files."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from deception_detection.data.base import DialogueDataset
from deception_detection.utils import REPO_ROOT


def apply_eval_exclusions(
    datasets: list[DialogueDataset],
    model: str,
    manifest_file: Path | None,
    run_folder: Path,
    *,
    registered_paths: dict[str, str] | None = None,
) -> list[DialogueDataset]:
    identity_file = run_folder / "eval_dataset_identity.json"
    if manifest_file is None:
        if identity_file.exists():
            raise ValueError(
                "This result folder already used exclusions; use a new run folder to disable them."
            )
        return datasets
    manifest_path = REPO_ROOT / manifest_file
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported evaluation-exclusion manifest version")
    identity = {"manifest_sha256": hashlib.sha256(raw).hexdigest(), "model": model, "datasets": {}}
    filtered = []
    for dataset in datasets:
        key = f"{dataset.base_name}__{dataset.variant}__{model}"
        entry = manifest["datasets"].get(key)
        if entry is None:
            raise ValueError(f"Manifest does not describe evaluation dataset {key}")
        path = REPO_ROOT / entry["rollout_file"]
        if registered_paths is not None:
            registered = registered_paths.get(key)
            if registered is None or (REPO_ROOT / registered).resolve() != path.resolve():
                raise ValueError(f"Registered rollout path differs from exclusion manifest: {key}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != entry["rollout_sha256"]:
            raise ValueError(f"Rollout content changed since exclusions were computed: {key}")
        ids = (dataset.metadata or {}).get("sample_id")
        if ids is None or len(ids) != len(set(ids)):
            raise ValueError(f"Missing or duplicate stable sample IDs: {key}")
        excluded = set(entry["excluded_ids"])
        if not excluded.issubset(set(ids)):
            raise ValueError(f"Exclusion IDs absent from complete dataset: {key}")
        keep = [sid not in excluded for sid in ids]
        # The legacy subset constructor reshuffles mixed-label datasets. Holdouts must leave
        # the current sample order and all already-applied preprocessing unchanged.
        selected = copy.copy(dataset)
        selected.dialogues, selected.labels, selected.metadata = dataset._slice_by_index(
            [i for i, retained in enumerate(keep) if retained]
        )
        filtered.append(selected)
        identity["datasets"][key] = {
            "rollout_sha256": actual_hash,
            "complete_count": len(ids),
            "excluded_count": len(excluded),
            "selected_count": sum(keep),
        }
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise ValueError(
            "Evaluation inputs changed; use a new result folder instead of reusing scores."
        )
    identity_file.write_text(json.dumps(identity, indent=2) + "\n")
    (run_folder / "eval_exclusions_snapshot.json").write_bytes(raw)
    return filtered
