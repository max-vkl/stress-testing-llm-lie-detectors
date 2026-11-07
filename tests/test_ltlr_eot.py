from types import SimpleNamespace

import pytest
import torch

from deception_detection.data.repe import RepeHonestDishonestFacts
from deception_detection.detectors import (
    LastTokenLogisticRegressionDetector,
    LastTokenMMSDetector,
    get_detector_class,
)
from deception_detection.experiment import Experiment, ExperimentConfig
from deception_detection.models import ModelName
from deception_detection.tokenized_data import TokenizedDataset


def test_detection_mask_at_eot_uses_or_inserts_one_eot_per_dialogue() -> None:
    eot = 9
    tokens = torch.tensor(
        [
            [0, 1, 2, 3, eot, 0],
            [1, 2, eot, 3, eot, 0],
            [0, 1, eot, 2, eot, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
        ]
    )
    detection_mask = torch.zeros_like(tokens, dtype=torch.bool)
    detection_mask[0, 2:4] = True  # EOT is not yet detected: insert the following one.
    detection_mask[1, 1:3] = True  # EOT is already detected: retain that EOT.
    # Row 2 deliberately has no detected content: use the terminal EOT.

    output = TokenizedDataset._detection_mask_at_eot(tokens, detection_mask, attention_mask, eot)

    assert output.sum(dim=1).tolist() == [1, 1, 1]
    assert torch.equal(tokens[output], torch.tensor([eot, eot, eot]))
    assert torch.nonzero(output[0], as_tuple=False).item() == 4
    assert torch.nonzero(output[1], as_tuple=False).item() == 2
    assert torch.nonzero(output[2], as_tuple=False).item() == 4


def test_detection_mask_at_eot_rejects_detected_content_without_following_eot() -> None:
    tokens = torch.tensor([[9, 1, 2]])
    detection_mask = torch.tensor([[False, True, True]])
    attention_mask = torch.ones_like(tokens)

    with pytest.raises(ValueError, match="no following EOT"):
        TokenizedDataset._detection_mask_at_eot(
            tokens, detection_mask, attention_mask, eot_token_id=9
        )


@pytest.mark.parametrize("method", ["ltlr", "ltmms"])
def test_last_token_detectors_use_eot_by_default_without_mutating_dataset_padding(
    method: str,
) -> None:
    experiment = object.__new__(Experiment)
    experiment.cfg = SimpleNamespace(
        method=method,
        detect_only_last_token=False,
        train_data_last_content=False,
        train_data="repe_honesty__plain",
        trim_reasoning=False,
        max_llama_token_length=None,
        use_followup_question=False,
    )

    dataset = experiment.get_dataset("repe_honesty__plain", model="prewritten")

    assert experiment.uses_ltlr_eot(dataset)
    assert dataset.padding["llama"]["right"] == 0
    assert RepeHonestDishonestFacts.padding["llama"]["right"] == 0


@pytest.mark.parametrize(
    ("method", "expected_class"),
    [
        ("ltlr_ct", LastTokenLogisticRegressionDetector),
        ("ltmms_ct", LastTokenMMSDetector),
    ],
)
def test_content_token_methods_select_last_content_and_reuse_last_token_detectors(
    method: str,
    expected_class: type,
) -> None:
    experiment = object.__new__(Experiment)
    experiment.cfg = SimpleNamespace(method=method, detect_only_last_token=False)
    dataset = SimpleNamespace(base_name="tiu", variant="cities_translation_neg")

    assert experiment.uses_last_content_token()
    assert not experiment.uses_ltlr_eot(dataset)
    assert get_detector_class(method) is expected_class


@pytest.mark.parametrize("method", ["ltlr", "ltmms"])
def test_eot_methods_do_not_implicitly_select_last_content(method: str) -> None:
    experiment = object.__new__(Experiment)
    experiment.cfg = SimpleNamespace(method=method, detect_only_last_token=False)

    assert not experiment.uses_last_content_token()


def test_train_last_compatibility_only_exempts_primary_train_and_validation() -> None:
    experiment = object.__new__(Experiment)
    experiment.cfg = SimpleNamespace(
        method="ltlr",
        detect_only_last_token=False,
        train_data_last_content=True,
        train_data="repe_honesty__plain",
    )

    primary = SimpleNamespace(base_name="repe_honesty", variant="plain")
    validation = SimpleNamespace(base_name="repe_honesty", variant="plain_val")
    evaluation = SimpleNamespace(base_name="personas", variant="plain")

    assert not experiment.uses_ltlr_eot(primary)
    assert not experiment.uses_ltlr_eot(validation)
    assert experiment.uses_ltlr_eot(evaluation)


@pytest.mark.parametrize(
    ("method", "detect_only_last_token", "extra_train_data", "match"),
    [
        ("lr", False, [], "only supported for ltlr"),
        ("ltlr", True, [], "mutually exclusive"),
        ("ltlr", False, ["roleplaying__plain"], "cannot be combined"),
    ],
)
def test_train_last_compatibility_rejects_incompatible_configs(
    tmp_path,
    method: str,
    detect_only_last_token: bool,
    extra_train_data: list[str],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        ExperimentConfig(
            method=method,
            model_name=ModelName.LLAMA_70B_3_3,
            train_data="repe_honesty__plain",
            extra_train_data=extra_train_data,
            no_control=True,
            detect_layers=[0],
            train_data_last_content=True,
            detect_only_last_token=detect_only_last_token,
            folder=tmp_path,
            name=f"bad-{method}-{detect_only_last_token}",
        )


class _FakeEOTTokenizer:
    def __init__(
        self,
        encodings: dict[str, list[int]],
        *,
        eot_token: str | None = None,
        eot_token_id: int | None = None,
    ) -> None:
        self.name_or_path = "fake-gemma-tokenizer"
        self.unk_token_id = 3
        self.encodings = encodings
        if eot_token is not None:
            self.eot_token = eot_token
        if eot_token_id is not None:
            self.eot_token_id = eot_token_id

    def __call__(self, text: str, *, add_special_tokens: bool):
        assert not add_special_tokens
        return {"input_ids": self.encodings.get(text, [self.unk_token_id])}


def test_eot_resolver_uses_gemma4_tokenizer_metadata() -> None:
    tokenizer = _FakeEOTTokenizer(
        {
            "<turn|>": [106],
            # This is the Gemma 3 fallback and is deliberately invalid under Gemma 4.
            "<end_of_turn>": [900, 901, 902],
        },
        eot_token="<turn|>",
        eot_token_id=106,
    )

    resolved = TokenizedDataset._resolve_eot_token(tokenizer, "gemma")  # type: ignore[arg-type]

    assert resolved.text == "<turn|>"
    assert resolved.token_id == 106


def test_eot_resolver_uses_validated_gemma3_fallback() -> None:
    tokenizer = _FakeEOTTokenizer({"<end_of_turn>": [106]})

    resolved = TokenizedDataset._resolve_eot_token(tokenizer, "gemma")  # type: ignore[arg-type]

    assert resolved.text == "<end_of_turn>"
    assert resolved.token_id == 106


def test_eot_resolver_never_takes_first_id_of_multi_token_marker() -> None:
    tokenizer = _FakeEOTTokenizer({"<end_of_turn>": [700, 701]})

    with pytest.raises(ValueError, match="must encode to exactly one token"):
        TokenizedDataset._resolve_eot_token(tokenizer, "gemma")  # type: ignore[arg-type]


def test_eot_resolver_rejects_inconsistent_metadata() -> None:
    tokenizer = _FakeEOTTokenizer({"<turn|>": [106]}, eot_token="<turn|>", eot_token_id=999)

    with pytest.raises(ValueError, match="inconsistent EOT metadata"):
        TokenizedDataset._resolve_eot_token(tokenizer, "gemma")  # type: ignore[arg-type]


def test_terminal_eot_removal_is_strict() -> None:
    tokenizer = _FakeEOTTokenizer({"<turn|>": [106]}, eot_token="<turn|>", eot_token_id=106)
    eot = TokenizedDataset._resolve_eot_token(tokenizer, "gemma")  # type: ignore[arg-type]

    assert TokenizedDataset._strip_terminal_eot("model answer<turn|>\n", eot, 0) == "model answer"
    with pytest.raises(ValueError, match="does not end with the resolved EOT marker"):
        TokenizedDataset._strip_terminal_eot("model answer<end_of_turn>\n", eot, 0)


def test_last_content_selection_never_wraps_or_falls_back() -> None:
    eot = 106
    tokens = torch.tensor([[eot, 10, 11], [1, 20, eot]])
    detection_mask = torch.tensor([[False, False, True], [False, True, False]])
    attention_mask = torch.ones_like(tokens)

    with pytest.raises(ValueError, match="refusing to fall back"):
        TokenizedDataset._detection_mask_at_last_content(
            tokens, detection_mask, attention_mask, eot
        )
