import json
from typing import Any

from deception_detection.data.base import DialogueDataset, DialogueDataType
from deception_detection.types import Label, Message
from deception_detection.utils import REPO_ROOT


class TIUDataset(DialogueDataset):
    """
    Prewritten dataset exported from the Truth_is_Universal (TIU) repository.

    Export format:
      lie-detection/data/tiu/<variant>.jsonl
    Load as:
      tiu__<variant>
    """

    base_name = "tiu"
    # Variants are dynamic (based on exported files); we bypass validation in __init__.
    all_variants: list[str] = ["exported"]
    default_variant = "exported"

    @property
    def bare_statements(self) -> bool:
        """Keep canonical train/validation views bare, including after slicing."""
        return self.variant.removesuffix("_val") in {"cannonical_ct", "cannonical_cte"}

    def __init__(self, *args: Any, **kwargs: Any):
        # Allow arbitrary variants (files define valid variants).
        kwargs["skip_variant_validation"] = True
        super().__init__(*args, **kwargs)

    def _get_dialogues(self) -> DialogueDataType:
        assert self.variant is not None
        path = REPO_ROOT / "data" / "tiu" / f"{self.variant}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"TIU export not found: {path}")

        dialogues = []
        labels: list[Label] = []
        metadata: dict[str, list[Any]] = {
            "source_file": [],
            "category": [],
            "language": [],
            "variant_type": [],
        }

        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                obj = json.loads(raw)

                msgs: list[dict[str, Any]] | None = obj.get("messages")
                if not isinstance(msgs, list):
                    raise ValueError(f"{path}:{line_no}: missing/invalid 'messages' list")
                dialogue = [
                    Message(
                        role=m["role"],
                        content=str(m["content"]),
                        detect=bool(m.get("detect", False)),
                    )
                    for m in msgs
                ]
                dialogues.append(dialogue)

                label_val = obj.get("label")
                if label_val in (1, "1", True, "honest"):
                    labels.append(Label.HONEST)
                elif label_val in (0, "0", False, "deceptive"):
                    labels.append(Label.DECEPTIVE)
                else:
                    raise ValueError(f"{path}:{line_no}: invalid label {label_val!r}")

                meta = obj.get("meta", {}) or {}
                metadata["source_file"].append(meta.get("source_file"))
                metadata["category"].append(meta.get("category"))
                metadata["language"].append(meta.get("language"))
                metadata["variant_type"].append(meta.get("variant_type"))
                # New mixed TIU datasets supply group IDs so all views of one fact can
                # stay on the same side of validation. Old exports retain their metadata.
                for key in ("fact_id", "example_kind", "question_kind"):
                    if key in meta or key in metadata:
                        metadata.setdefault(key, [None] * (len(labels) - 1)).append(meta.get(key))

        return dialogues, labels, metadata
