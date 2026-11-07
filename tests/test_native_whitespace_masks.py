"""Readout masks follow native whitespace handling without rewriting model inputs."""

# pyright: reportPrivateUsage=false

from types import SimpleNamespace

import pytest
import torch

from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.types import Message


def mask_text(messages: list[Message], answer: str) -> str:
    formatted = (
        "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n" + answer + "<|eot_id|>"
    )
    tokens = torch.arange(len(formatted)).unsqueeze(0)
    # One character per token makes span boundaries independently observable.
    def char_to_token(_batch: int, position: int) -> int:
        return position

    offsets = SimpleNamespace(char_to_token=char_to_token)
    mask = TokenizedDataset._get_detection_mask(
        tokens,
        offsets,
        [messages],
        [formatted],
        False,
        False,
        {"left": 0, "right": 0},
        "llama",
    )
    assert mask is not None
    return "".join(formatted[index] for index in tokens[mask].tolist())


@pytest.mark.parametrize("trim_native", [False, True])
def test_leading_whitespace_matches_rendered_content(trim_native: bool) -> None:
    message = Message("assistant", "  Answer with  interior spaces.", True)
    answer = message.content.lstrip() if trim_native else message.content
    assert mask_text([message], answer) == answer
    assert message.content == "  Answer with  interior spaces."


def test_adjacent_assistant_suffix_keeps_its_leading_space() -> None:
    messages = [Message("assistant", "Prefix", False), Message("assistant", " suffix.", True)]
    assert mask_text(messages, "Prefix suffix.") == " suffix."


def test_substantive_message_mismatch_is_still_rejected() -> None:
    with pytest.raises(AssertionError):
        mask_text([Message("assistant", "  Truth.", True)], "Lie.")
