"""Training compaction must preserve probe values while releasing full-sequence storage."""

# White-box tests exercise the exact assembly and detector indexing path.
# pyright: reportPrivateUsage=false

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from deception_detection.activations import Activations
from deception_detection.detectors import LastTokenLogisticRegressionDetector
from deception_detection.experiment import Experiment
from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.types import Message


def make_acts(length: int, seed: int, offset: int = 1) -> Activations:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn((4, length, 2, 6), generator=generator).to(torch.bfloat16)
    tokens = torch.arange(4 * length).reshape(4, length)
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    # Different readout positions within a shard, including nonterminal content tokens.
    positions = torch.tensor([length - offset, length - offset - 1, 2, 3])
    mask[torch.arange(4), positions] = True
    td = TokenizedDataset(
        dialogues=[[Message("assistant", f"example {i}", True)] for i in range(4)],
        formatted_dialogues=[f"example {i}" for i in range(4)],
        tokens=tokens,
        str_tokens=[[str(t) for t in row] for row in cast(list[list[int]], tokens.tolist())],
        detection_mask=mask,
        attention_mask=torch.ones_like(tokens),
        dataset_partial_id=f"fixture-{seed}",
    )
    return Activations(values, td, logits=values[:, :, 0], layers=[0, 2])


def test_compaction_copies_values_and_metadata_without_retaining_backing_storage() -> None:
    full = make_acts(500, 42, offset=2)
    # Exercise the layer views returned by get_acts_by_layer, not just standalone tensors.
    view = Activations(full.all_acts[:, :, 1:2], full.tokenized_dataset, full.logits, [2])
    compact = view.compact_single_detection_token()
    mask = full.tokenized_dataset.detection_mask
    assert mask is not None
    assert torch.equal(compact.all_acts[:, 0], view.all_acts[mask])
    assert compact.all_acts.dtype == view.all_acts.dtype
    assert compact.all_acts.shape == (4, 1, 1, 6)
    assert compact.all_acts.untyped_storage().nbytes() == 4 * 6 * 2
    assert (
        compact.all_acts.untyped_storage().data_ptr() != full.all_acts.untyped_storage().data_ptr()
    )
    td = compact.tokenized_dataset
    assert td.dialogues is full.tokenized_dataset.dialogues
    assert td.dataset_partial_id == full.tokenized_dataset.dataset_partial_id
    assert torch.equal(td.tokens[:, 0], full.tokenized_dataset.tokens[mask])
    assert td.str_tokens is full.tokenized_dataset.str_tokens
    assert td.detection_mask is not None and td.detection_mask.all()
    assert td.attention_mask.all()
    assert view.logits is not None and compact.logits is not None
    assert torch.equal(compact.logits[:, 0], view.logits[mask])
    assert compact.logits.untyped_storage().nbytes() == compact.logits.numel() * 2
    assert full.all_acts.shape[1] == 500  # The extraction/cache object is untouched.


@pytest.mark.parametrize("case", ["missing", "empty_row", "multiple", "sparse"])
def test_non_single_token_inputs_keep_the_original_representation(case: str) -> None:
    acts = make_acts(8, 42)
    if case == "missing":
        acts.tokenized_dataset.detection_mask = None
    elif case == "sparse":
        acts.all_acts = acts.all_acts.to_sparse()
    else:
        mask = acts.tokenized_dataset.detection_mask
        assert mask is not None
        if case == "empty_row":
            mask[0] = False
        else:
            mask[0, 0] = True
    assert acts.compact_single_detection_token() is acts


@pytest.mark.parametrize("method", ["ltlr", "ltlr_ct"])
@pytest.mark.parametrize("weighted", [False, True])
def test_sharded_assembly_preserves_fitted_probes_and_scores(
    monkeypatch: pytest.MonkeyPatch, method: str, weighted: bool
) -> None:
    sources = [
        make_acts(length, length, 2 if method == "ltlr_ct" else 1) for length in (8, 11, 9, 13)
    ]
    shards = [
        (side, i, acts.tokenized_dataset)
        for i, (side, acts) in enumerate(
            zip(["honest", "deceptive", "honest", "deceptive"], sources, strict=True)
        )
    ]
    experiment = object.__new__(Experiment)
    experiment.cfg = SimpleNamespace(method=method)  # type: ignore[assignment]

    def train_shards(_size: int) -> Iterator[tuple[str, int, TokenizedDataset]]:
        return iter(shards)

    monkeypatch.setattr(experiment, "_train_shards", train_shards)

    def extract(toks: TokenizedDataset, layers: list[int], stage: str) -> dict[int, Activations]:
        source = next(a for a in sources if a.tokenized_dataset is toks)
        return {
            layer: Activations(source.all_acts[:, :, i : i + 1], toks, layers=[layer])
            for i, layer in enumerate(layers)
        }

    monkeypatch.setattr(experiment, "get_acts_by_layer", extract)
    concat = Activations.concat_padded
    compact_parts: list[Activations] = []

    def checked_concat(parts: list[Activations]) -> Activations:
        for part in parts:
            assert part.all_acts.shape[1] == 1  # Compact BEFORE accumulating/padding shards.
            assert part.all_acts.untyped_storage().nbytes() == part.all_acts.numel() * 2
        compact_parts.extend(parts)
        return concat(parts)

    monkeypatch.setattr(Activations, "concat_padded", checked_concat)
    honest, deceptive = experiment._train_paired_acts_by_layer([0, 2], shard_size=4)
    assert len(compact_parts) == 8
    weights: dict[str, Any] = (
        {"positive_weights": np.linspace(0.5, 1.5, 8), "negative_weights": np.linspace(1.5, 0.5, 8)}
        if weighted
        else {}
    )
    for i, layer in enumerate([0, 2]):
        full = [
            concat(
                [
                    Activations(a.all_acts[:, :, i : i + 1], a.tokenized_dataset, layers=[layer])
                    for a in sources[side::2]
                ]
            )
            for side in (0, 1)
        ]
        original = LastTokenLogisticRegressionDetector([layer], reg_coeff=10, max_iter=5000)
        compact = LastTokenLogisticRegressionDetector([layer], reg_coeff=10, max_iter=5000)
        original.fit(full[1], full[0], **weights)
        compact.fit(deceptive[layer], honest[layer], **weights)
        for name in ("directions", "scaler_mean", "scaler_scale"):
            assert torch.equal(getattr(original, name), getattr(compact, name))
        eval_acts = original._idx_detect_acts(full[0])
        assert torch.equal(
            original.get_score_tensor(eval_acts), compact.get_score_tensor(eval_acts)
        )
        assert honest[layer].all_acts.shape == (8, 1, 1, 6)
        assert deceptive[layer].tokenized_dataset.dialogues == full[1].tokenized_dataset.dialogues


def test_other_training_methods_keep_full_sequence_activations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_acts(8, 1)
    experiment = object.__new__(Experiment)
    experiment.cfg = SimpleNamespace(method="lr")  # type: ignore[assignment]

    def train_shards(_size: int) -> Iterator[tuple[str, int, TokenizedDataset]]:
        return iter(
            [("honest", 0, source.tokenized_dataset), ("deceptive", 0, source.tokenized_dataset)]
        )

    def extract(_toks: TokenizedDataset, _layers: list[int], stage: str) -> dict[int, Activations]:
        return {0: source}

    monkeypatch.setattr(experiment, "_train_shards", train_shards)
    monkeypatch.setattr(experiment, "get_acts_by_layer", extract)
    honest, deceptive = experiment._train_paired_acts_by_layer([0])
    assert torch.equal(honest[0].all_acts, source.all_acts)
    assert torch.equal(deceptive[0].all_acts, source.all_acts)


def test_padded_tensor_positions_do_not_index_unpadded_display_strings() -> None:
    acts = make_acts(100, 20, offset=2)
    td = acts.tokenized_dataset
    td.str_tokens = [["BOS", "content", "EOT"] for _ in td.dialogues]
    # Display strings intentionally omit left padding; Gemma can add a display-only BOS.
    td.attention_mask[0, :95] = 0
    compact = acts.compact_single_detection_token()
    assert td.detection_mask is not None
    assert torch.equal(compact.all_acts[:, 0], acts.all_acts[td.detection_mask])
    assert compact.tokenized_dataset.str_tokens is td.str_tokens


@pytest.mark.parametrize("case", ["single", "missing", "multiple", "empty_row", "logits"])
def test_compact_extraction_preserves_full_context_and_readout_values(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    td = make_acts(17, 23, offset=2).tokenized_dataset
    if case == "missing":
        td.detection_mask = None
    elif case == "multiple":
        assert td.detection_mask is not None
        td.detection_mask[0, 0] = True
    elif case == "empty_row":
        assert td.detection_mask is not None
        td.detection_mask[0] = False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    forwarded: list[torch.Tensor] = []

    class ContextModel:
        name_or_path = "fixture"

        def __call__(self, tokens: torch.Tensor, **kwargs: Any) -> SimpleNamespace:
            forwarded.append(tokens.clone())
            assert kwargs["use_cache"] is False
            # Each output depends on the entire preceding context, so truncating the
            # input to its readout token would break the equivalence assertion.
            values = tokens.cumsum(dim=1).to(torch.bfloat16).unsqueeze(-1)
            return SimpleNamespace(
                hidden_states=tuple(values + i for i in range(4)), logits=values
            )

    original = Activations.from_model(
        cast(Any, ContextModel()), td, batch_size=3, layers=[3, 0, 2],
        save_logits=case == "logits",
    )
    compact = Activations.from_model(
        cast(Any, ContextModel()), td, compact_single_token=True, batch_size=3,
        layers=[3, 0, 2], save_logits=case == "logits",
    )
    expected = original.compact_single_detection_token() if case == "single" else original
    assert torch.equal(compact.all_acts, expected.all_acts)
    assert compact.layers == expected.layers
    assert torch.equal(compact.tokenized_dataset.tokens, expected.tokenized_dataset.tokens)
    assert torch.equal(torch.cat(forwarded[:2]), td.tokens)
    assert torch.equal(torch.cat(forwarded[2:]), td.tokens)
    assert compact.tokenized_dataset.str_tokens is td.str_tokens
    if case == "single":
        assert compact.all_acts.shape == (4, 1, 3, 1)
    else:
        assert compact.all_acts.shape[1] == 17
    if case == "logits":
        assert original.logits is not None and compact.logits is not None
        assert torch.equal(compact.logits, original.logits)


@pytest.mark.parametrize(
    "stage,method,enabled,expected",
    [("train", "ltlr_ct", "1", True), ("eval", "ltlr_ct", "1", False),
     ("train", "lr", "1", False), ("train", "ltlr_ct", "0", False)],
)
@pytest.mark.parametrize("single_layer", [False, True])
def test_gpu_compaction_is_opt_in_last_token_training_only(
    monkeypatch: pytest.MonkeyPatch, stage: str, method: str, enabled: str, expected: bool,
    single_layer: bool,
) -> None:
    source = make_acts(17, 23)
    ex = object.__new__(Experiment)
    ex.cfg = SimpleNamespace(  # type: ignore[assignment]
        method=method, use_goodfire_sae_acts=False, use_local_sae_acts=False,
        inefficient_activation_caching=False,
    )
    monkeypatch.setenv("DD_GPU_COMPACT_TRAIN", enabled)
    monkeypatch.setattr(Experiment, "model", property(lambda _self: None))

    def extract(*_args: Any, **kwargs: Any) -> Activations:
        assert kwargs["compact_single_token"] is expected
        return source

    monkeypatch.setattr(Activations, "from_model", extract)
    if single_layer:
        assert ex.get_acts_for_layers(source.tokenized_dataset, [0], stage) is source
    else:
        assert set(ex.get_acts_by_layer(source.tokenized_dataset, [0, 2], stage)) == {0, 2}
