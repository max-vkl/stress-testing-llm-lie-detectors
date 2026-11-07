"""Greedily generate and cache DYL follow-ups for an experiment's eval datasets.

One invocation prepares one model.  The resulting cache is shared by all five
DYL probe methods, so probe comparisons cannot differ because of generation.
Invalid native generations are retained in the manifest for auditing but are
excluded when an experiment loads the augmented dataset.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

from deception_detection.data.base import DialogueDataset
from deception_detection.dyl import (
    DYL_CACHE_VERSION,
    DYL_QUESTION,
    dyl_dataset_fingerprint,
    dyl_generation_settings,
    dyl_row_fingerprint,
    extract_visible_dyl_generation,
    get_cached_dataset_entry,
    get_dyl_cache_path,
    label_counts,
    load_dyl_cache,
    make_cache_record,
    new_dyl_cache,
    save_dyl_cache,
    validate_cached_dataset_entry,
)
from deception_detection.gemma4_chat import is_gemma4_tokenizer, render_gemma4_dialogue
from deception_detection.models import ModelName, get_model_and_tokenizer
from deception_detection.repository import DatasetRepository
from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.types import Message
from deception_detection.utils import preprocess_dialogue, resolve_config_path

MAX_NEW_TOKENS = 256
MAX_CONTEXT_TOKENS = 8192
SEED = 0


def _load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = resolve_config_path(path)
    with open(resolved) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Experiment config is not a mapping: {resolved}")
    return resolved, cfg


def _format_prompt(dialogue, tokenizer, fold_system: bool) -> str:
    messages = [Message(m.role, m.content, False) for m in dialogue]
    messages.append(Message("user", DYL_QUESTION, False))
    native_gemma4 = is_gemma4_tokenizer(tokenizer)
    processed = preprocess_dialogue(messages, fold_system=fold_system and not native_gemma4)
    if native_gemma4:
        return render_gemma4_dialogue(tokenizer, processed, add_generation_prompt=True)
    return tokenizer.apply_chat_template(
        processed,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _load_source_dataset(
    repository: DatasetRepository,
    dataset_name: str,
    model_name: str,
    cfg: dict[str, Any],
    tokenizer,
):
    return repository.get(
        dataset_name,
        model=model_name,
        trim_reasoning=cfg.get("trim_reasoning", True),
        shuffle_upon_init=not dataset_name.startswith("repe"),
        max_llama_token_length=cfg.get("max_llama_token_length"),
        tokenizer=tokenizer,
    )


def _valid_resume_records(dataset, entry: dict[str, Any]) -> list[dict[str, Any]]:
    if entry.get("source_fingerprint") != dyl_dataset_fingerprint(dataset):
        return []
    records = entry.get("records", [])
    if not isinstance(records, list) or len(records) > len(dataset):
        return []
    for index, record in enumerate(records):
        if record.get("source_fingerprint") != dyl_row_fingerprint(dataset, index):
            return []
        answer = record.get("answer")
        if not isinstance(answer, str):
            return []
        # make_cache_record is the single source of truth for validity/normalization.
        expected = make_cache_record(
            dataset,
            index,
            answer,
            record.get("invalid_reason"),
            record.get("raw_generation"),
        )
        if record != expected:
            return []
    return records


def _update_entry_summary(entry: dict[str, Any], dataset) -> None:
    records = entry["records"]
    valid_mask = [bool(record["valid"]) for record in records]
    complete = len(records) == len(dataset)
    valid_count = sum(valid_mask)
    entry.update(
        {
            "complete": complete,
            "source_size": len(dataset),
            "processed": len(records),
            "valid": valid_count,
            "invalid": len(records) - valid_count,
            "invalid_rate": (len(records) - valid_count) / len(records) if records else 0.0,
            "source_label_counts": label_counts(dataset),
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )
    if complete:
        retained = dataset.subset_where_true(valid_mask)
        entry["retained_label_counts"] = label_counts(retained)


@torch.inference_mode()
def prepare_dataset(
    cache: dict[str, Any],
    dataset: DialogueDataset,
    model,
    tokenizer,
    batch_size: int,
    checkpoint_batches: int,
) -> dict[str, Any]:
    prior = get_cached_dataset_entry(dataset, cache) or {}
    records = _valid_resume_records(dataset, prior)
    if len(records) == len(dataset):
        validate_cached_dataset_entry(dataset, prior)
        print(f"Reusing complete DYL cache entry: {dataset.name} ({len(dataset)} rows)")
        return prior

    entry = {
        "source_fingerprint": dyl_dataset_fingerprint(dataset),
        "records": records,
    }
    cache["datasets"][dataset.name] = entry
    fold_system = ModelName(cache["model_name"]).type == "gemma"
    tokenizer_type = ModelName(cache["model_name"]).type
    eot_token_id = TokenizedDataset._resolve_eot_token(tokenizer, tokenizer_type).token_id
    stop_token_ids = list(dict.fromkeys([int(tokenizer.eos_token_id), eot_token_id]))
    device = next(model.parameters()).device
    start = len(records)
    batches_since_checkpoint = 0

    progress = tqdm(desc=dataset.name, initial=start, total=len(dataset))
    for batch_start in range(start, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        indices = list(range(batch_start, batch_end))
        prompts = [
            _format_prompt(dataset.dialogues[index], tokenizer, fold_system) for index in indices
        ]
        prompt_token_counts = [
            len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts
        ]
        answers: dict[int, tuple[str, str | None, str]] = {}
        generation_indices = [
            index
            for index, token_count in zip(indices, prompt_token_counts, strict=True)
            if token_count + MAX_NEW_TOKENS <= MAX_CONTEXT_TOKENS
        ]
        for index, token_count in zip(indices, prompt_token_counts, strict=True):
            if index not in generation_indices:
                answers[index] = (
                    "",
                    f"context_too_long:{token_count}+{MAX_NEW_TOKENS}>{MAX_CONTEXT_TOKENS}",
                    "",
                )

        if generation_indices:
            generation_prompts = [prompts[indices.index(index)] for index in generation_indices]
            model_inputs = tokenizer(
                generation_prompts,
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
            generated = model.generate(
                **model_inputs,
                do_sample=False,
                disable_compile=True,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
            )
            input_width = model_inputs["input_ids"].shape[1]
            for index, generated_ids in zip(
                generation_indices, generated[:, input_width:], strict=True
            ):
                answers[index] = extract_visible_dyl_generation(
                    generated_ids, tokenizer, cache["model_name"]
                )

        for index in indices:
            answer, invalid_reason, raw_generation = answers[index]
            entry["records"].append(
                make_cache_record(
                    dataset,
                    index,
                    answer,
                    invalid_reason,
                    raw_generation,
                )
            )
        _update_entry_summary(entry, dataset)
        progress.update(len(indices))
        progress.set_postfix(invalid=entry["invalid"], valid=entry["valid"])
        batches_since_checkpoint += 1
        if batches_since_checkpoint >= checkpoint_batches:
            save_dyl_cache(cache)
            batches_since_checkpoint = 0
    progress.close()

    _update_entry_summary(entry, dataset)
    save_dyl_cache(cache)
    validate_cached_dataset_entry(dataset, entry)
    if not entry["retained_label_counts"].get("honest") or not entry["retained_label_counts"].get(
        "deceptive"
    ):
        raise ValueError(f"Invalid generations removed an entire class from {dataset.name}")
    marker = " EXCEEDS_1_PERCENT" if entry["invalid_rate"] > 0.01 else ""
    print(
        f"DYL invalid rate {dataset.name}: {entry['invalid']}/{len(dataset)} "
        f"({100 * entry['invalid_rate']:.3f}%){marker}"
    )
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=[model.value for model in ModelName])
    parser.add_argument("--config", required=True, help="A DYL 1_ config for this model")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-batches", type=int, default=25)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.checkpoint_batches <= 0:
        raise ValueError("batch-size and checkpoint-batches must be positive")

    config_path, cfg = _load_config(args.config)
    if cfg.get("model_name") != args.model:
        raise ValueError(f"Config {config_path} uses {cfg.get('model_name')!r}, not {args.model!r}")
    eval_data: list[str] | None = cfg.get("eval_data")
    if not isinstance(eval_data, list) or not eval_data:
        raise ValueError(f"Config {config_path} has no eval_data list")
    if len(eval_data) != len(set(eval_data)):
        raise ValueError(f"Config {config_path} contains duplicate eval datasets")

    settings = dyl_generation_settings(MAX_NEW_TOKENS, MAX_CONTEXT_TOKENS)
    if settings["seed"] != SEED:
        raise RuntimeError("DYL seed constants are inconsistent")
    cache_path = get_dyl_cache_path(args.model)
    cache = (
        load_dyl_cache(args.model, settings)
        if cache_path.exists()
        else new_dyl_cache(args.model, settings)
    )
    cache["source_configs"] = sorted(set(cache.get("source_configs", [])) | {str(config_path)})
    cache["version"] = DYL_CACHE_VERSION

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model_name = ModelName(args.model)
    model, tokenizer = get_model_and_tokenizer(model_name)
    assert model is not None
    tokenizer.padding_side = "left"
    repository = DatasetRepository()

    entries: list[dict[str, Any]] = []
    for dataset_name in eval_data:
        dataset = _load_source_dataset(repository, dataset_name, args.model, cfg, tokenizer)
        entries.append(
            prepare_dataset(
                cache,
                dataset,
                model,
                tokenizer,
                args.batch_size,
                args.checkpoint_batches,
            )
        )

    total = sum(entry["source_size"] for entry in entries)
    invalid = sum(entry["invalid"] for entry in entries)
    above_one = [
        name for name, entry in zip(eval_data, entries, strict=True) if entry["invalid_rate"] > 0.01
    ]
    print(f"DYL cache ready: {cache_path}")
    print(f"Overall invalid rate: {invalid}/{total} ({100 * invalid / total:.3f}%)")
    print(f"Datasets above 1% invalid: {above_one}")


if __name__ == "__main__":
    main()
