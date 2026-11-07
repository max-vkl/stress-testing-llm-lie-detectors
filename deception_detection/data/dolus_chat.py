"""Frozen DolusChat seed-0 detector-pool and full-corpus variants."""

import json
from pathlib import Path
from typing import Any

from deception_detection.data.base import DialogueDataset
from deception_detection.types import Dialogue, DialogueDataType, Label, Message
from deception_detection.utils import REPO_ROOT

DATA_PATH = REPO_ROOT / "data/dolus_chat/dataset.json"
FULL_DATA_PATH = REPO_ROOT / "data/dolus_chat/full_dataset.json"


def dolus_chat_data_path(variant: str = "seed0") -> Path:
    if variant == "seed0":
        return DATA_PATH
    if variant == "full_seed0":
        return FULL_DATA_PATH
    raise ValueError(f"Unknown DolusChat variant: {variant}")


def load_dolus_chat_data(variant: str = "seed0") -> dict[str, Any]:
    return json.loads(dolus_chat_data_path(variant).read_text())


class DolusChatDataset(DialogueDataset):
    """Interleave truthful/deceptive answers before the standard dataset shuffle.

    The artifact retains source whitespace. Normalize messages as ordinary
    repository chats so native model templates and detection spans agree.
    """

    base_name = "dolus_chat"
    default_variant = "seed0"
    all_variants = ["seed0", "full_seed0"]
    padding = {model: {"left": 0, "right": 0} for model in ("llama", "gemma", "mistral", "qwen")}

    def _get_dialogues(self) -> DialogueDataType:
        artifact = load_dolus_chat_data(self.variant)
        if artifact["schema_version"] != 1 or artifact["selection"]["seed"] != 0:
            raise ValueError("Expected the frozen DolusChat seed-0 artifact")
        if artifact["selection"].get("variant", "seed0") != self.variant:
            raise ValueError("DolusChat artifact does not match the requested variant")
        dialogues: list[Dialogue] = []
        labels: list[Label] = []
        metadata: dict[str, list[Any]] = {
            key: [] for key in ("id", "source_id", "source_row_index", "pair_id", "response_kind")
        }
        for pair in artifact["pairs"]:
            row = pair["source_row"]
            for kind, label in (("truthful", Label.HONEST), ("deceptive", Label.DECEPTIVE)):
                answer = pair["responses"][kind].rstrip()
                if not answer:
                    raise ValueError(f"Empty DolusChat response at source row {row}: {kind}")
                dialogues.append(
                    [
                        Message("system", pair["system"].strip(), False),
                        Message("user", pair["user"].strip(), False),
                        Message("assistant", answer, True),
                    ]
                )
                labels.append(label)
                metadata["id"].append(f"dolus_chat:{self.variant}:{row}:{kind}")
                metadata["source_id"].append(pair["id"])
                metadata["source_row_index"].append(row)
                metadata["pair_id"].append(row)
                metadata["response_kind"].append(kind)
        metadata["source_revision"] = [artifact["source"]["revision"]] * len(dialogues)
        return dialogues, labels, metadata
