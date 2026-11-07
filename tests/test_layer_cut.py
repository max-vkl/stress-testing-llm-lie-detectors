"""Regression tests for the model layer-cut used during activation extraction.

Guards the off-by-one fix in ``Experiment.model`` / ``cut_at_layer_for_detect_layers``: the model
is truncated to ``max(detect_layers) + 1`` blocks, NOT ``max(detect_layers)``. transformers applies
the model's final RMSNorm only to the terminal ``hidden_states`` entry, so cutting exactly at the
top detection layer would make that layer read its post-final-norm output instead of the raw
residual stream (and, because the activation cache key is layer-independent, silently contaminate
that layer's blocks across runs with different cuts).

These tests use a tiny toy model that reproduces the transformers ``hidden_states`` contract, so
they are fast, deterministic and need no real weights or GPU.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from deception_detection import models as models_module
from deception_detection.experiment import cut_at_layer_for_detect_layers
from deception_detection.models import ModelName, get_model_and_tokenizer

N_LAYERS = 8
NORM_SCALE = 0.5  # toy final-norm: non-identity so normed output != raw residual


class _ToyBlock(nn.Module):
    """A decoder block that adds a fixed per-depth constant and returns a tuple (like HF layers)."""

    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta

    def forward(self, hidden_states: torch.Tensor, **_: object) -> tuple[torch.Tensor]:
        return (hidden_states + self.delta,)


class _ToyNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * NORM_SCALE


class _ToyInner(nn.Module):
    """Mirrors ``transformers`` ``LlamaModel.forward``: append each pre-block residual, then apply
    the final norm to the terminal entry."""

    def __init__(self, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_ToyBlock(float(i + 1)) for i in range(n_layers)])
        self.norm = _ToyNorm()

    def forward(self, inputs_embeds: torch.Tensor, output_hidden_states: bool = True):
        all_hidden: tuple[torch.Tensor, ...] = ()
        hidden = inputs_embeds
        for layer in self.layers:
            all_hidden += (hidden,)  # pre-block residual (raw)
            hidden = layer(hidden)[0]
        hidden = self.norm(hidden)  # final RMSNorm on the last state only
        all_hidden += (hidden,)  # terminal entry (normed)
        return SimpleNamespace(hidden_states=all_hidden, last_hidden_state=hidden)


class _ToyModel(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.model = _ToyInner(n_layers)

    def forward(self, inputs_embeds: torch.Tensor, output_hidden_states: bool = True):
        return self.model(inputs_embeds, output_hidden_states=output_hidden_states)


_EMB = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])


def _hidden_states_for_cut(monkeypatch: pytest.MonkeyPatch, cut_at_layer: int | None):
    """Load a fresh toy model through the REAL ``get_model_and_tokenizer`` cut branch and return its
    ``hidden_states`` tuple for ``_EMB``."""

    def fake_loader(model_name, omit_model=False, lora_path=None):  # noqa: ANN001, ANN202
        return _ToyModel(N_LAYERS), object()

    monkeypatch.setattr(models_module, "model_and_tokenizer", fake_loader)
    model, _ = get_model_and_tokenizer(ModelName.LLAMA_70B_3_3, cut_at_layer=cut_at_layer)
    return model(_EMB, output_hidden_states=True).hidden_states


def test_cut_at_layer_for_detect_layers() -> None:
    # One above the max, regardless of ordering; None when empty.
    assert cut_at_layer_for_detect_layers([10, 12]) == 13
    assert cut_at_layer_for_detect_layers([0]) == 1
    assert cut_at_layer_for_detect_layers([5, 41, 20]) == 42  # unsorted -> max + 1
    # top layer -> keep all blocks
    assert cut_at_layer_for_detect_layers([N_LAYERS - 1]) == N_LAYERS
    assert cut_at_layer_for_detect_layers([]) is None


def test_top_detect_layer_stays_raw_not_normed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixed cut (max+1) reads the top detection layer as the RAW residual; the old cut (==max)
    would instead read the post-final-norm terminal entry."""
    top_layer = 5  # an interior layer in the untruncated toy model

    uncut = _ToyModel(N_LAYERS)(_EMB, output_hidden_states=True).hidden_states
    raw = uncut[top_layer]

    fixed_cut = cut_at_layer_for_detect_layers([top_layer])  # == top_layer + 1
    fixed = _hidden_states_for_cut(monkeypatch, fixed_cut)
    buggy = _hidden_states_for_cut(monkeypatch, top_layer)  # the old off-by-one

    # Fixed cut: top detection layer is an interior entry == the untruncated raw residual.
    assert torch.allclose(fixed[top_layer], raw)
    # Old cut: top detection layer became the terminal entry -> post-final-norm, a DIFFERENT rep.
    assert not torch.allclose(buggy[top_layer], raw)
    assert torch.allclose(buggy[top_layer], raw * NORM_SCALE)


def test_cut_is_cache_safe_across_different_cuts(monkeypatch: pytest.MonkeyPatch) -> None:
    """With max+1, a shared layer's activations are identical no matter how deep the cut was — so
    reusing another run's cached block for that layer is safe (the core cross-run cache hazard)."""
    shared_layer = 5
    # Two runs whose top detection layer differs (5 vs 6) => cuts 6 vs 7.
    cut_a = _hidden_states_for_cut(monkeypatch, cut_at_layer_for_detect_layers([shared_layer]))
    cut_b = _hidden_states_for_cut(monkeypatch, cut_at_layer_for_detect_layers([shared_layer + 1]))
    assert torch.allclose(cut_a[shared_layer], cut_b[shared_layer])
