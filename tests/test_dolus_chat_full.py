"""Full DolusChat corpus preserves source construction and the existing probe protocol."""

# pyright: reportPrivateUsage=false

import hashlib
import json
import random
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import yaml

from deception_detection.data.dolus_chat import DATA_PATH, load_dolus_chat_data
from deception_detection.experiment import Experiment, ExperimentConfig
from deception_detection.repository import DatasetRepository
from deception_detection.scripts.import_dolus_chat import reconstruct_pairs
from deception_detection.types import Label
from deception_detection.utils import CONFIG_DIR


@pytest.fixture(scope="module")
def full_artifact() -> dict[str, Any]:
    return load_dolus_chat_data("full_seed0")




def test_full_reconstruction_consumes_all_partitions_without_rng_side_effects() -> None:
    rows = [
        {
            "id": str(i),
            "system_message": "System",
            "context": {"sensitive_info": f"secret{i}", "non_sensitive_info": ["a", "b", "c"]},
            "user_query": {"speaker_type": "guest", "content": "Question?"},
            "responses": {"truthful": "Truth.", "deceptive": "Lie."},
        }
        for i in range(100)
    ]
    original = json.dumps(rows)
    state = random.getstate()
    full = reconstruct_pairs(rows, "full_seed0")
    assert len(full) == 100
    assert full[91:95] == reconstruct_pairs(rows)
    # Independently follow upstream's continuous shuffle through the final test row.
    rng = random.Random(0)
    contexts: list[str] = []
    for index in np.random.RandomState(0).permutation(100):
        contexts = ["a", "b", "c", f"secret{index}"]
        rng.shuffle(contexts)
    assert full[-1]["system"] == "System\nContext:\n" + "\n".join(contexts)
    assert json.dumps(rows) == original
    assert random.getstate() == state
    with pytest.raises(ValueError, match="Unknown"):
        reconstruct_pairs(rows, "typo")


def test_full_repository_labels_normalization_and_exact_split(
    full_artifact: dict[str, Any],
) -> None:
    dataset = DatasetRepository().get("dolus_chat__full_seed0", shuffle_upon_init=False)
    assert len(dataset) == 129776
    assert dataset.labels == [Label.HONEST, Label.DECEPTIVE] * 64888
    assert dataset.metadata is not None
    assert len(set(dataset.metadata["id"])) == len(dataset)
    assert "_split" not in dataset.metadata
    for i, pair in enumerate(full_artifact["pairs"]):
        for offset, kind in enumerate(("truthful", "deceptive")):
            index = 2 * i + offset
            assert [message.content for message in dataset.dialogues[index]] == [
                pair["system"].strip(),
                pair["user"].strip(),
                pair["responses"][kind].rstrip(),
            ]
            assert dataset.metadata["id"][index] == (
                f"dolus_chat:full_seed0:{pair['source_row']}:{kind}"
            )
    original_ids = list(dataset.metadata["id"])
    dataset.randomly_shuffle_dialogues()
    assert dataset.metadata["id"] == [
        original_ids[i] for i in np.random.RandomState(42).permutation(len(dataset))
    ]
    experiment = object.__new__(Experiment)
    experiment.cfg = cast(
        ExperimentConfig,
        SimpleNamespace(val_fraction=0.2, val_split_group_keys=[]),
    )
    train, validation = experiment._split_train_val(dataset)
    assert validation is not None
    experiment.train_dataset = train
    honest, deceptive = experiment._train_class_splits()
    assert (len(train), len(validation), len(honest), len(deceptive)) == (
        103821,
        25955,
        51891,
        51891,
    )
    assert train.labels.count(Label.HONEST) == 51930
    assert validation.labels.count(Label.HONEST) == 12958
    assert validation.labels.count(Label.DECEPTIVE) == 12997
    assert train.metadata is not None and validation.metadata is not None
    assert set(train.metadata["id"]).isdisjoint(validation.metadata["id"])
    assert set(train.metadata["pair_id"]) & set(validation.metadata["pair_id"])
