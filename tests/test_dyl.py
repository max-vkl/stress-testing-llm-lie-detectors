import copy
import json
from types import SimpleNamespace

import pytest
import yaml

import deception_detection.dyl as dyl_module
from deception_detection.data.base import DialogueDataset
from deception_detection.detectors import (
    LastTokenLogisticRegressionDetector,
    LastTokenMMSDetector,
    LogisticRegressionDetector,
    MeanLogisticRegressionDetector,
    MMSDetector,
    get_detector_class,
)
from deception_detection.dyl import (
    DYL_METHODS,
    DYL_QUESTION,
    DYL_TRAIN_DATASET,
    augment_dataset_with_dyl_cache,
    dyl_dataset_fingerprint,
    dyl_generation_settings,
    extract_visible_dyl_generation,
    get_cached_dataset_entry,
    get_dyl_cache_path,
    load_dyl_cache,
    make_cache_record,
    new_dyl_cache,
    normalize_dyl_answer,
    save_dyl_cache,
)
from deception_detection.experiment import Experiment, ExperimentConfig
from deception_detection.models import ModelName
from deception_detection.scripts.prepare_dyl_followups import (
    _format_prompt,
    _valid_resume_records,
    prepare_dataset,
)
from deception_detection.types import Label, Message
from deception_detection.utils import CONFIG_DIR


def _dataset() -> DialogueDataset:
    dialogues = [
        [Message("user", f"question {i}", False), Message("assistant", f"answer {i}", True)]
        for i in range(3)
    ]
    dataset = DialogueDataset(
        "plain",
        dialogues,
        [Label.DECEPTIVE, Label.HONEST, Label.DECEPTIVE],
        {"id": [10, 11, 12]},
        skip_variant_validation=True,
        shuffle_upon_init=False,
        model="llama-70b-3.3",
    )
    dataset.base_name = "fixture"
    return dataset


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Yes", "yes"),
        (" yes. ", "yes"),
        ("NO", "no"),
        ("No.", "no"),
        ("Yes, I did", None),
        ("Yes\nNo", None),
        ("", None),
    ],
)
def test_normalize_dyl_answer_is_strict(answer: str, expected: str | None) -> None:
    assert normalize_dyl_answer(answer) == expected


class _FakeChannelTokenizer:
    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        assert token_ids == [1]
        return "final\nYes" if skip_special_tokens else "<|channel>final\nYes<channel|><turn|>"


def test_gemma4_visible_generation_removes_final_channel_envelope() -> None:
    answer, reason, raw = extract_visible_dyl_generation(
        [1], _FakeChannelTokenizer(), "gemma-4-31b-it"
    )

    assert answer == "Yes"
    assert reason is None
    assert raw == "<|channel>final\nYes<channel|><turn|>"


class _FakeThinkingTokenizer:
    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        return "thought\nsecret" if skip_special_tokens else "<|channel>thought\nsecret<channel|>"


def test_gemma4_generated_reasoning_is_invalid() -> None:
    answer, reason, _ = extract_visible_dyl_generation(
        [1], _FakeThinkingTokenizer(), "gemma-4-31b-it"
    )

    assert answer == "<|channel>thought\nsecret<channel|>"
    assert reason == "unexpected_reasoning_channel"






def test_cache_augmentation_excludes_invalids_and_never_relabels() -> None:
    dataset = _dataset()
    records = [
        make_cache_record(dataset, 0, "No."),
        make_cache_record(dataset, 1, "Yes"),
        make_cache_record(dataset, 2, "Yes, because..."),
    ]
    cache = {
        "version": 1,
        "model_name": dataset.model,
        "generation": dyl_generation_settings(),
        "datasets": {
            dataset.name: {
                "source_fingerprint": dyl_dataset_fingerprint(dataset),
                "records": records,
            }
        },
    }

    augmented = augment_dataset_with_dyl_cache(dataset, dataset.model, cache)

    assert len(augmented) == 2
    assert augmented.labels == [Label.DECEPTIVE, Label.HONEST]
    assert [dialogue[-1].content for dialogue in augmented.dialogues] == ["No.", "Yes"]
    for dialogue in augmented.dialogues:
        assert dialogue[-2] == Message("user", DYL_QUESTION, False)
        assert [message.detect for message in dialogue] == [False, False, False, True]


@pytest.mark.parametrize("model", ["llama-70b-3.3", "gemma-3-27b-it", "gemma-4-31b-it"])
def test_complete_cache_entry_is_reused_without_generation(model: str) -> None:
    dataset = _dataset()
    dataset.model = model
    records = [make_cache_record(dataset, index, "Yes") for index in range(len(dataset))]
    entry = {
        "source_fingerprint": dyl_dataset_fingerprint(dataset),
        "records": records,
    }
    cache = new_dyl_cache(dataset.model, dyl_generation_settings())
    cache["datasets"][dataset.name] = entry

    result = prepare_dataset(cache, dataset, object(), object(), 2, 1)

    assert result is entry


@pytest.mark.parametrize(
    ("method", "detector_class", "uses_eot"),
    [
        ("dyl_ltlr", LastTokenLogisticRegressionDetector, True),
        ("dyl_ltmms", LastTokenMMSDetector, True),
        ("dyl_ltlr_ct", LastTokenLogisticRegressionDetector, False),
        ("dyl_ltmms_ct", LastTokenMMSDetector, False),
        ("dyl_lr", LogisticRegressionDetector, False),
        ("dyl_mms", MMSDetector, False),
        ("dyl_mlr", MeanLogisticRegressionDetector, False),
    ],
)
def test_dyl_method_aliases_preserve_base_token_semantics(
    method: str, detector_class: type, uses_eot: bool
) -> None:
    experiment = object.__new__(Experiment)
    experiment.cfg = SimpleNamespace(
        method=method,
        detect_only_last_token=False,
        train_data_last_content=False,
        train_data="dyl_city_countries__train",
    )
    dataset = SimpleNamespace(base_name="fixture", variant="plain")

    assert get_detector_class(method) is detector_class
    assert experiment.uses_ltlr_eot(dataset) is uses_eot
    assert experiment.uses_last_content_token() is method.endswith("_ct")


def test_dyl_and_legacy_followup_are_mutually_exclusive(tmp_path) -> None:
    with pytest.raises(ValueError, match="already appends the DYL question"):
        ExperimentConfig(
            method="dyl_lr",
            model_name=ModelName.LLAMA_70B_3_3,
            train_data="dyl_city_countries__train",
            train_on_policy=True,
            no_control=True,
            detect_layers=[0],
            use_followup_question=True,
            folder=tmp_path,
            name="bad-dyl-followup",
        )


@pytest.mark.parametrize("method", DYL_METHODS)
def test_dyl_methods_require_dyl_training_dataset(method: str, tmp_path) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{method} requires train_data='{DYL_TRAIN_DATASET}'",
    ):
        ExperimentConfig(
            method=method,
            model_name=ModelName.LLAMA_70B_3_3,
            train_data="repe_honesty__plain",
            train_on_policy=True,
            no_control=True,
            detect_layers=[0],
            folder=tmp_path,
            name=f"bad-{method}-train-data",
        )




@pytest.fixture
def archived_persona_cache():
    dataset = _dataset()
    dataset.base_name = "personas"
    records = [
        make_cache_record(dataset, 0, "No."),
        make_cache_record(dataset, 1, "Yes"),
        make_cache_record(dataset, 2, "Yes, because..."),
    ]
    cache = {
        "model_name": dataset.model,
        "datasets": {
            dataset.name: {
                "source_fingerprint": dyl_dataset_fingerprint(dataset),
                "records": records,
            }
        },
    }
    expected = augment_dataset_with_dyl_cache(dataset, dataset.model, cache)
    dataset.variant = "old_" + dataset.variant
    return dataset, cache, expected


@pytest.mark.parametrize("legacy", [True, False])
def test_archived_persona_reuses_identical_answers_without_generation(
    archived_persona_cache, legacy: bool
) -> None:
    dataset, cache, expected = archived_persona_cache
    if not legacy:
        entry = cache["datasets"].pop("personas__plain")
        entry["source_fingerprint"] = dyl_dataset_fingerprint(dataset)
        cache["datasets"][dataset.name] = entry
    before = copy.deepcopy(cache)

    actual = augment_dataset_with_dyl_cache(dataset, dataset.model, cache)
    prepared = prepare_dataset(cache, dataset, object(), object(), 2, 1)

    assert actual.name == "personas__old_plain"
    assert actual.dialogues == expected.dialogues
    assert actual.labels == expected.labels
    assert actual.metadata == expected.metadata
    assert prepared["records"] == next(iter(cache["datasets"].values()))["records"]
    assert cache == before


@pytest.mark.parametrize("change", ["dialogue", "label", "order", "model"])
def test_legacy_persona_cache_rejects_changed_sources(archived_persona_cache, change: str) -> None:
    dataset, cache, _ = archived_persona_cache
    if change == "dialogue":
        dataset.dialogues[0][0].content += " changed"
    elif change == "label":
        dataset.labels[0] = Label.HONEST
    elif change == "order":
        dataset = dataset[::-1]
    else:
        dataset.model = "gemma-3-27b-it"

    with pytest.raises(ValueError, match="legacy source fingerprint changed"):
        augment_dataset_with_dyl_cache(dataset, dataset.model, cache)
    with pytest.raises(ValueError, match="legacy source fingerprint changed"):
        prepare_dataset(cache, dataset, object(), object(), 2, 1)


def test_legacy_persona_cache_still_validates_individual_records(archived_persona_cache) -> None:
    dataset, cache, _ = archived_persona_cache
    cache["datasets"]["personas__plain"]["records"][0]["source_fingerprint"] = "corrupted"
    with pytest.raises(ValueError, match="source row changed"):
        augment_dataset_with_dyl_cache(dataset, dataset.model, cache)


def test_current_persona_cache_takes_precedence_and_never_falls_back_on_corruption(
    archived_persona_cache,
) -> None:
    dataset, cache, _ = archived_persona_cache
    current = copy.deepcopy(cache["datasets"]["personas__plain"])
    current["source_fingerprint"] = dyl_dataset_fingerprint(dataset)
    current["records"][0] = make_cache_record(dataset, 0, "Yes.")
    cache["datasets"][dataset.name] = current

    actual = augment_dataset_with_dyl_cache(dataset, dataset.model, cache)
    assert actual.dialogues[0][-1].content == "Yes."
    current["source_fingerprint"] = "corrupted"
    with pytest.raises(ValueError, match="source fingerprint changed"):
        augment_dataset_with_dyl_cache(dataset, dataset.model, cache)


def test_new_unprefixed_persona_does_not_use_archived_cache(archived_persona_cache) -> None:
    dataset, cache, _ = archived_persona_cache
    entry = get_cached_dataset_entry(dataset, cache)
    cache["datasets"] = {dataset.name: entry}
    dataset.variant = "plain"
    with pytest.raises(KeyError, match="has no entry for personas__plain"):
        augment_dataset_with_dyl_cache(dataset, dataset.model, cache)


def test_partial_legacy_persona_followups_can_resume(archived_persona_cache) -> None:
    dataset, cache, _ = archived_persona_cache
    legacy = cache["datasets"]["personas__plain"]
    legacy["records"] = legacy["records"][:1]
    resolved = get_cached_dataset_entry(dataset, cache)
    assert resolved is not None
    resumed = _valid_resume_records(dataset, resolved)
    assert resumed == legacy["records"]
    resumed.append(make_cache_record(dataset, 1, "Yes"))
    assert len(legacy["records"]) == 1
    assert resolved["source_fingerprint"] == dyl_dataset_fingerprint(dataset)
    assert resolved["source_fingerprint"] != legacy["source_fingerprint"]


@pytest.mark.parametrize("fold_system", [False, True])
def test_non_gemma4_dyl_prompt_keeps_existing_system_handling(fold_system: bool) -> None:
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return json.dumps(messages)

    dialogue = [
        Message("system", "Persona.", False),
        Message("user", "Question?", False),
        Message("assistant", "Answer.", True),
    ]
    actual = json.loads(_format_prompt(dialogue, Tokenizer(), fold_system))
    expected = (
        [{"role": "user", "content": "Persona.\n\nQuestion?"}]
        if fold_system
        else [
            {"role": "system", "content": "Persona."},
            {"role": "user", "content": "Question?"},
        ]
    ) + [
        {"role": "assistant", "content": "Answer."},
        {"role": "user", "content": DYL_QUESTION},
    ]
    assert actual == expected


@pytest.mark.parametrize("model", ["llama-70b-3.3", "gemma-3-27b-it", "gemma-4-31b-it"])
def test_rendering_revision_separates_only_gemma4_cache(tmp_path, monkeypatch, model) -> None:
    monkeypatch.setenv("DD_DYL_CACHE_DIR", str(tmp_path))
    settings = dyl_generation_settings()
    cache = new_dyl_cache(model, settings)
    original = save_dyl_cache(cache)
    before = original.read_bytes()
    assert load_dyl_cache(model, settings) == cache

    monkeypatch.setattr(dyl_module, "GEMMA4_CHAT_RENDERING_VERSION", 999)
    if model == "gemma-4-31b-it":
        assert get_dyl_cache_path(model) != original
        with pytest.raises(FileNotFoundError, match="prepare_dyl_followups"):
            load_dyl_cache(model, settings)
    else:
        assert get_dyl_cache_path(model) == original == tmp_path / f"{model}.json"
        assert load_dyl_cache(model, settings) == cache
    assert original.read_bytes() == before


@pytest.mark.parametrize("revision", [None, 0])
def test_legacy_gemma4_followups_cannot_be_loaded_or_reused(tmp_path, monkeypatch, revision):
    monkeypatch.setenv("DD_DYL_CACHE_DIR", str(tmp_path))
    dataset = _dataset()
    dataset.model = "gemma-4-31b-it"
    cache = new_dyl_cache(dataset.model, dyl_generation_settings())
    cache["datasets"][dataset.name] = {
        "source_fingerprint": dyl_dataset_fingerprint(dataset),
        "records": [make_cache_record(dataset, i, "No") for i in range(len(dataset))],
    }
    if revision is None:
        cache.pop("chat_rendering_version")
    else:
        cache["chat_rendering_version"] = revision
    legacy_path = tmp_path / f"{dataset.model}.json"
    legacy_path.write_text(json.dumps(cache))
    before = legacy_path.read_bytes()
    with pytest.raises(FileNotFoundError, match="prepare_dyl_followups"):
        load_dyl_cache(dataset.model)
    # Even copying an old cache to the new path cannot bypass the rendering guard.
    get_dyl_cache_path(dataset.model).write_bytes(before)
    with pytest.raises(ValueError, match="Stale Gemma 4 DYL cache"):
        load_dyl_cache(dataset.model)
    with pytest.raises(ValueError, match="Stale Gemma 4 DYL cache"):
        augment_dataset_with_dyl_cache(dataset, dataset.model, cache)
    with pytest.raises(ValueError, match="Stale Gemma 4 DYL cache"):
        prepare_dataset(cache, dataset, object(), object(), 2, 1)
    assert legacy_path.read_bytes() == before
