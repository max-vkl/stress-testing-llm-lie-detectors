from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from deception_detection.data import RepeCanonicalDataset
from deception_detection.experiment import Experiment
from deception_detection.repository import DatasetRepository
from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.types import Label
from deception_detection.utils import REPO_ROOT


class WhitespaceTokenizer:
    """Small tokenizer double sufficient for testing canonical prefix construction."""

    name_or_path = "test-llama-tokenizer"

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return text.split()

    @staticmethod
    def convert_tokens_to_string(tokens: list[str]) -> str:
        return " ".join(tokens)


def test_repe_canonical_reconstructs_first_512_prefix_pairs() -> None:
    tokenizer = WhitespaceTokenizer()
    dataset = RepeCanonicalDataset(tokenizer=tokenizer, shuffle_upon_init=False)

    assert len(dataset) == 1024
    assert dataset.labels == [Label.HONEST, Label.DECEPTIVE] * 512
    assert dataset.metadata is not None
    assert dataset.metadata["pair_id"] == [pair_id for pair_id in range(512) for _ in range(2)]

    expected: list[tuple[int, int, str, str]] = []
    frame = pd.read_csv(REPO_ROOT / "data/repe/true_false_facts.csv")
    statements = frame.loc[frame["label"] == 1, "statement"].tolist()
    for source_fact_id, statement in enumerate(statements):
        tokens = tokenizer.tokenize(statement)
        for prefix_index in range(1, len(tokens) - 5):
            expected.append(
                (
                    source_fact_id,
                    prefix_index,
                    statement,
                    tokenizer.convert_tokens_to_string(tokens[:prefix_index]),
                )
            )
            if len(expected) == 512:
                break
        if len(expected) == 512:
            break

    for pair_id, (source_id, prefix_index, statement, prefix) in enumerate(expected):
        honest, deceptive = dataset.dialogues[2 * pair_id : 2 * pair_id + 2]
        assert honest[-1].content == deceptive[-1].content == prefix
        assert "honest person" in honest[0].content
        assert "untruthful person" in deceptive[0].content
        assert dataset.metadata["source_fact_id"][2 * pair_id] == source_id
        assert dataset.metadata["prefix_index"][2 * pair_id] == prefix_index
        assert dataset.metadata["source_statement"][2 * pair_id] == statement


def test_repe_canonical_requires_and_receives_target_tokenizer(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="requires the target model tokenizer"):
        RepeCanonicalDataset()

    repository = DatasetRepository(storage_file=str(tmp_path / "paths.json"))
    dataset = repository.get(
        "repe_canonical__plain",
        tokenizer=WhitespaceTokenizer(),
        shuffle_upon_init=False,
    )
    assert isinstance(dataset, RepeCanonicalDataset)
    assert len(dataset) == 1024


def test_repe_canonical_split_keeps_pairs_together() -> None:
    dataset = RepeCanonicalDataset(
        tokenizer=WhitespaceTokenizer(),
        shuffle_upon_init=False,
    )
    experiment = object.__new__(Experiment)
    experiment.cfg = SimpleNamespace(
        val_fraction=0.2,
        val_split_group_keys=["pair_id"],
        train_data="repe_canonical__plain",
    )

    train, val = experiment._split_train_val(dataset)
    assert val is not None
    assert len(train) == 820
    assert len(val) == 204
    assert train.metadata is not None and val.metadata is not None
    assert set(train.metadata["pair_id"]).isdisjoint(val.metadata["pair_id"])


def test_repe_canonical_last_token_policy_is_dataset_local() -> None:
    dataset = RepeCanonicalDataset(
        tokenizer=WhitespaceTokenizer(),
        shuffle_upon_init=False,
    )[:2]
    sentinel = object()
    with patch.object(TokenizedDataset, "from_dialogue_list", return_value=sentinel) as build:
        result = TokenizedDataset.from_dataset(dataset, WhitespaceTokenizer())

    assert result is sentinel
    assert build.call_args.kwargs["detect_only_last_token"] is True
    assert build.call_args.kwargs["preserve_trailing_whitespace"] is True
