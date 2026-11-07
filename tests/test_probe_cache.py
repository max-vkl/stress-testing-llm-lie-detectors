"""CPU-only coverage for the probe-only cache policy."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from torch import nn

import deception_detection.activation_cache as cache_module
import deception_detection.experiment as experiment_module
from deception_detection.activation_cache import (
    ACTIVATION_CACHE_DISABLED_MESSAGE,
    PROBE_CACHE_DIRNAME,
    build_probe_cache_key,
    get_cache_root,
    get_probe_cache_dir,
    load_activation_blocks,
    load_cached_probe,
    save_activation_blocks,
    save_probe_atomically,
)
from deception_detection.activations import Activations
from deception_detection.detectors import MLPDetector
from deception_detection.experiment import Experiment, ExperimentConfig
from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.utils import REPO_ROOT


class _TextProbe:
    def __init__(self, value: str):
        self.value = value

    def save(self, file_path: str | Path) -> None:
        Path(file_path).write_text(self.value)

    @classmethod
    def load(cls, file_path: str | Path) -> _TextProbe:
        value = Path(file_path).read_text()
        if not value.startswith("valid:"):
            raise ValueError("corrupt probe")
        return cls(value)


def _toks(
    *,
    token_delta: int = 0,
    attention_last: int = 1,
    detection_last: bool = True,
) -> TokenizedDataset:
    tokens = torch.arange(6, dtype=torch.int64).reshape(1, 6)
    tokens[0, -1] += token_delta
    attention = torch.ones(1, 6, dtype=torch.int64)
    attention[0, -1] = attention_last
    detection = torch.ones(1, 6, dtype=torch.bool)
    detection[0, -1] = detection_last
    return TokenizedDataset(
        dialogues=[[]],
        formatted_dialogues=["synthetic"],
        tokens=tokens,
        str_tokens=[[""] * 6],
        detection_mask=detection,
        attention_mask=attention,
        dataset_partial_id="synthetic__probe_key",
    )


def _cfg(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "method": "ltlr",
        "model_name": "llama-70b-3.3",
        "reg_coeff": 10.0,
        "normalize_acts": True,
        "max_iter": 1000,
        "use_class_weights": False,
        "equal_dataset_weights": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_fixed_single_cache_root_ignores_legacy_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = tmp_path / "lie_detec_cache"
    monkeypatch.setattr(cache_module, "DEFAULT_CACHE_DIR", expected)
    monkeypatch.setenv("DD_CACHE_DIR", str(tmp_path / "must_not_be_used"))

    assert get_cache_root() == expected
    assert get_probe_cache_dir() == expected / PROBE_CACHE_DIRNAME
    assert expected.is_dir()
    assert not (tmp_path / "must_not_be_used").exists()


def test_atomic_probe_publish_and_corrupt_entry_self_heals(tmp_path: Path) -> None:
    path = tmp_path / "probes_v2" / "probe.pt"
    save_probe_atomically(_TextProbe("valid:first"), path)
    loaded = load_cached_probe(path, _TextProbe.load)
    assert loaded is not None and loaded.value == "valid:first"
    assert not list(path.parent.glob(".*.tmp.*"))

    path.write_text("partial")
    assert load_cached_probe(path, _TextProbe.load) is None
    assert not path.exists()


def test_concurrent_probe_publish_never_exposes_partial_file(tmp_path: Path) -> None:
    path = tmp_path / "probes_v2" / "same-key.pt"
    probes = [_TextProbe("valid:first"), _TextProbe("valid:second")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda probe: save_probe_atomically(probe, path), probes))
    assert path.read_text() in {"valid:first", "valid:second"}
    assert not list(path.parent.glob(".*.tmp.*"))


def test_activation_cache_apis_are_hard_disabled(tmp_path: Path) -> None:
    match = "Activation caching is disabled and must not be used"
    with pytest.raises(RuntimeError, match=match):
        load_activation_blocks(tmp_path, "forbidden", [0])
    with pytest.raises(RuntimeError, match=match):
        save_activation_blocks(tmp_path, "forbidden", [0], torch.zeros(1, 1, 1, 1))
    assert ACTIVATION_CACHE_DISABLED_MESSAGE.startswith(match)
    assert not list(tmp_path.iterdir())


def test_probe_key_tracks_inputs_masks_layers_and_effective_fit_weights() -> None:
    base = build_probe_cache_key(_cfg(), _toks(), _toks(), [20])
    assert build_probe_cache_key(_cfg(), _toks(token_delta=1), _toks(), [20]) != base
    assert build_probe_cache_key(_cfg(), _toks(attention_last=0), _toks(), [20]) != base
    assert build_probe_cache_key(_cfg(), _toks(detection_last=False), _toks(), [20]) != base
    assert build_probe_cache_key(_cfg(), _toks(), _toks(), [21]) != base
    assert build_probe_cache_key(_cfg(reg_coeff=1.0), _toks(), _toks(), [20]) != base

    weights_a = np.array([0.5, 1.5], dtype=np.float64)
    weights_b = np.array([1.5, 0.5], dtype=np.float64)
    weighted_a = build_probe_cache_key(
        _cfg(equal_dataset_weights=True),
        _toks(),
        _toks(),
        [20],
        positive_fit_weights=weights_a,
        negative_fit_weights=weights_a,
    )
    weighted_b = build_probe_cache_key(
        _cfg(equal_dataset_weights=True),
        _toks(),
        _toks(),
        [20],
        positive_fit_weights=weights_b,
        negative_fit_weights=weights_a,
    )
    # This is the metadata-only edge case: tokenized inputs can stay byte-identical while
    # train_source metadata changes the exact sample weights passed to fit().
    assert weighted_a != weighted_b


def test_config_defaults_to_probe_only_and_rejects_legacy_options(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        method="ltlr",
        model_name="llama-70b-3.3",
        detect_layers=[20],
        reg_coeff=10,
        no_control=True,
        folder=tmp_path,
        name="probe_default",
    )
    assert cfg.caching_off is False
    assert cfg.inefficient_activation_caching is False
    saved = yaml.safe_load((cfg.folder / "cfg.yaml").read_text())
    assert "caching_off" not in saved
    assert "inefficient_activation_caching" not in saved

    common = {
        "method": "ltlr",
        "model_name": "llama-70b-3.3",
        "detect_layers": [20],
        "reg_coeff": 10,
        "no_control": True,
        "folder": tmp_path,
    }
    with pytest.raises(ValueError, match="Activation caching is disabled and must not be used"):
        ExperimentConfig(
            **common,
            name="activation_cache_forbidden",
            inefficient_activation_caching=True,
        )
    with pytest.raises(ValueError, match="Deprecated cache option"):
        ExperimentConfig(**common, name="legacy", deactivate_cache=True)




def test_default_activation_path_does_not_look_up_activation_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = object.__new__(Experiment)
    experiment.cfg = SimpleNamespace(
        use_goodfire_sae_acts=False,
        use_local_sae_acts=False,
        sae_latent_whitelist=None,
        inefficient_activation_caching=False,
    )
    experiment._model = object()
    toks = _toks()
    calls: list[list[int]] = []

    def fake_from_model(
        _cls,
        _model,
        tokenized_dataset,
        *,
        batch_size,
        layers,
        verbose,
        compact_single_token=False,
    ):  # noqa: ANN001, ANN202
        del batch_size, verbose
        assert not compact_single_token
        calls.append(layers)
        return Activations(
            all_acts=torch.zeros(1, 6, len(layers), 3),
            tokenized_dataset=tokenized_dataset,
            layers=layers,
        )

    def unexpected_cache_lookup() -> Path:
        raise AssertionError("default activation path touched the activation cache")

    monkeypatch.setattr(Activations, "from_model", classmethod(fake_from_model))
    monkeypatch.setattr(experiment_module, "get_cache_root", unexpected_cache_lookup)
    by_layer = experiment.get_acts_by_layer(toks, [20, 21], stage="eval")

    assert calls == [[20, 21]]
    assert sorted(by_layer) == [20, 21]


def test_all_experiment_yamls_use_flag_free_cache_default() -> None:
    config_dir = REPO_ROOT / "deception_detection/scripts/configs"
    old_or_explicit = {
        "deactivate_cache",
        "cache_dir",
        "caching_off",
        "inefficient_activation_caching",
    }
    for path in config_dir.rglob("*.yaml"):
        data = yaml.safe_load(path.read_text())
        assert old_or_explicit.isdisjoint(data), f"cache flag in {path}"


def test_mlp_probe_lookup_deserializes_on_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = MLPDetector([20], hidden_dim=4)
    source.input_dim = 3
    source.mlp = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1), nn.Sigmoid())
    path = tmp_path / "mlp.pt"
    source.save(path)

    # Even when a GPU is reported available, lookup itself leaves parameters on CPU. The first
    # score() call moves them lazily to the scoring device.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    loaded = MLPDetector.load(path)
    assert loaded.mlp is not None
    assert {parameter.device.type for parameter in loaded.mlp.parameters()} == {"cpu"}


def test_global_score_cache_api_is_removed() -> None:
    assert not hasattr(cache_module, "load_cached_scores")
    assert not hasattr(cache_module, "save_cached_scores")


def test_chat_rendering_revision_only_invalidates_gemma4(monkeypatch):
    honest, deceptive = _toks(), _toks(token_delta=1)
    models = ["llama-70b-3.3", "gemma-3-27b-it", "gemma-4-31b-it"]
    before = {m: build_probe_cache_key(_cfg(model_name=m), honest, deceptive, [3]) for m in models}
    monkeypatch.setattr(cache_module, "GEMMA4_CHAT_RENDERING_VERSION", 999)
    after = {m: build_probe_cache_key(_cfg(model_name=m), honest, deceptive, [3]) for m in models}
    assert before[models[0]] == after[models[0]]
    assert before[models[1]] == after[models[1]]
    assert before[models[2]] != after[models[2]]
