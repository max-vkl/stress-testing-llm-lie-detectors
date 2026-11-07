from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

import numpy as np
import torch

from deception_detection.gemma4_chat import GEMMA4_CHAT_RENDERING_VERSION
from deception_detection.log import logger
from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.utils import REPO_ROOT

# Keep trained-probe caches inside this checkout, separate from other repositories.
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "probes"
# Cache layout version. It is folded into every base key and block file for change detection, so
# keep V2_DIRNAME and V2_VERSION in lockstep.
V2_DIRNAME = "inefficient_activations_v4"
V2_VERSION = 4
DEFAULT_BLOCK_SIZE = 8
PROBE_CACHE_DIRNAME = "probes_v2"
ACTIVATION_CACHE_DISABLED_MESSAGE = (
    "Activation caching is disabled and must not be used. It is slower than recomputing "
    "activations directly. Run experiments through "
    "deception_detection/scripts/experiment.py without --inefficient_activation_caching."
)


def _mkdir_shared(path: Path) -> None:
    """Create a directory robustly when several processes or hosts race to create it.

    Some shared filesystems can report EEXIST before a concurrently created directory is
    visible to a follow-up stat. Path.mkdir(exist_ok=True) re-raises in that window. Retry
    the visibility check instead of turning a harmless cache race into a failed experiment.
    """
    last_error: FileExistsError | None = None
    for attempt in range(50):
        try:
            path.mkdir(parents=True, exist_ok=True)
            return
        except FileExistsError as err:
            last_error = err
            if path.is_dir():
                return
            time.sleep(min(0.02 * (attempt + 1), 0.2))
    if last_error is not None:
        raise last_error


def get_cache_root() -> Path:
    _mkdir_shared(DEFAULT_CACHE_DIR)
    return DEFAULT_CACHE_DIR


def _hash_tensor(tensor: torch.Tensor) -> str:
    tensor_cpu = tensor.detach().cpu().contiguous()
    if tensor_cpu.dtype == torch.bool:
        tensor_cpu = tensor_cpu.to(torch.uint8)
    return hashlib.sha256(tensor_cpu.numpy().tobytes()).hexdigest()


def _normalize_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _normalize_list(value: list[int] | None) -> list[int] | None:
    return None if value is None else list(value)


def _model_name_to_str(model_name: Any) -> str | None:
    if model_name is None:
        return None
    return getattr(model_name, "value", str(model_name))


# The forward-pass numerics depend on the model dtype and attention implementation, which are
# selected at runtime via env vars (see models.py). They are NOT stored on the cfg, yet activations
# computed under bf16 vs fp16 (or different attention impls) differ, so they MUST be part of every
# cache key or a bf16 run would silently reuse/overwrite fp16-cached activations. Keep these
# defaults identical to the ones resolved in deception_detection/models.py.
def runtime_dtype_str() -> str:
    return os.getenv("DD_DTYPE", "bf16").strip().lower()


def runtime_attn_impl() -> str:
    return os.getenv("DD_ATTN_IMPL", "sdpa").strip().lower()


# -------------------------
# Optional activation blocks
# -------------------------


def _get_or_compute_hash(
    tokenized_dataset: TokenizedDataset, attr: str, tensor: torch.Tensor
) -> str:
    cached = getattr(tokenized_dataset, attr, None)
    if isinstance(cached, str) and cached:
        return cached
    h = _hash_tensor(tensor)
    with suppress(Exception):
        setattr(tokenized_dataset, attr, h)
    return h


def _tokens_hash(tokenized_dataset: TokenizedDataset) -> str:
    return _get_or_compute_hash(tokenized_dataset, "_cache_tokens_hash", tokenized_dataset.tokens)


def _attention_mask_hash(tokenized_dataset: TokenizedDataset) -> str:
    return _get_or_compute_hash(
        tokenized_dataset, "_cache_attention_mask_hash", tokenized_dataset.attention_mask
    )


def build_cache_base_key(cfg: Any, tokenized_dataset: TokenizedDataset, stage: str) -> str:
    """
    Build a base cache key that is independent of layers.
    Includes strong change detection: tokens+mask hashes, shapes, dtypes, and relevant cfg flags.
    """
    key_data = {
        "version": V2_VERSION,
        "stage": stage,
        "model_name": _model_name_to_str(getattr(cfg, "model_name", None)),
        "model_dtype": runtime_dtype_str(),
        "attn_implementation": runtime_attn_impl(),
        "lora_path": _normalize_path(getattr(cfg, "lora_path", None)),
        "pw_locked": getattr(cfg, "pw_locked", None),
        "detect_only_start_of_turn": getattr(cfg, "detect_only_start_of_turn", None),
        "detect_only_last_token": getattr(cfg, "detect_only_last_token", None),
        "detect_all_tokens": getattr(cfg, "detect_all_tokens", None),
        "trim_reasoning": getattr(cfg, "trim_reasoning", None),
        "use_followup_question": getattr(cfg, "use_followup_question", None),
        "max_llama_token_length": getattr(cfg, "max_llama_token_length", None),
        "use_local_sae_acts": getattr(cfg, "use_local_sae_acts", None),
        "use_goodfire_sae_acts": getattr(cfg, "use_goodfire_sae_acts", None),
        "sae_latent_whitelist": _normalize_list(getattr(cfg, "sae_latent_whitelist", None)),
        "dataset_partial_id": tokenized_dataset.dataset_partial_id,
        "tokens_shape": list(tokenized_dataset.tokens.shape),
        "tokens_dtype": str(tokenized_dataset.tokens.dtype),
        "attention_mask_shape": list(tokenized_dataset.attention_mask.shape),
        "attention_mask_dtype": str(tokenized_dataset.attention_mask.dtype),
        "tokens_hash": _tokens_hash(tokenized_dataset),
        "attention_mask_hash": _attention_mask_hash(tokenized_dataset),
    }
    if _model_name_to_str(getattr(cfg, "model_name", None)) == "gemma-4-31b-it":
        key_data["chat_rendering_version"] = GEMMA4_CHAT_RENDERING_VERSION
    encoded = json.dumps(key_data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _v2_base_dir(cache_dir: Path, base_key: str) -> Path:
    return cache_dir / V2_DIRNAME / base_key


def _v2_manifest_path(cache_dir: Path, base_key: str) -> Path:
    return _v2_base_dir(cache_dir, base_key) / "manifest.json"


def read_manifest(cache_dir: Path, base_key: str) -> dict[str, Any]:
    raise RuntimeError(ACTIVATION_CACHE_DISABLED_MESSAGE)
    path = _v2_manifest_path(cache_dir, base_key)
    if not path.exists():
        return {"version": V2_VERSION, "blocks": []}
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("version") != V2_VERSION:
            logger.warning("Unexpected cache manifest version at %s", path)
        if "blocks" not in data or not isinstance(data["blocks"], list):
            return {"version": V2_VERSION, "blocks": []}
        return data
    except Exception as err:
        logger.warning("Failed reading cache manifest %s: %s", path, err)
        return {"version": V2_VERSION, "blocks": []}


def write_manifest_atomically(cache_dir: Path, base_key: str, manifest: dict[str, Any]) -> None:
    raise RuntimeError(ACTIVATION_CACHE_DISABLED_MESSAGE)
    base = _v2_base_dir(cache_dir, base_key)
    _mkdir_shared(base)
    path = _v2_manifest_path(cache_dir, base_key)
    # PIDs are node-local, so include a UUID to avoid cross-node temporary-file collisions.
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with open(tmp, "w") as f:
            json.dump(manifest, f)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _block_filename(layers: list[int]) -> str:
    lo, hi = min(layers), max(layers)
    sig = hashlib.sha256(json.dumps(layers).encode("utf-8")).hexdigest()[:10]
    return f"block_{lo}-{hi}__{sig}.pt"


def _existing_layers_from_manifest(manifest: dict[str, Any]) -> set[int]:
    existing: set[int] = set()
    for blk in manifest.get("blocks", []):
        try:
            for layer in blk.get("layers", []):
                existing.add(int(layer))
        except Exception:
            continue
    return existing


def load_activation_blocks(
    cache_dir: Path, base_key: str, requested_layers: list[int]
) -> dict[int, torch.Tensor]:
    """
    Load cached activations for requested layers.

    This feature is intentionally disabled; direct recomputation is the supported path.
    """
    raise RuntimeError(ACTIVATION_CACHE_DISABLED_MESSAGE)
    manifest = read_manifest(cache_dir, base_key)
    requested_set = set(requested_layers)
    out: dict[int, torch.Tensor] = {}

    blocks = list(manifest.get("blocks", []))
    # A manifest is only an index, not the source of truth. Concurrent writers can both publish
    # valid, distinct blocks before atomically replacing the manifest; whichever manifest lands
    # last may not mention the other writer's block. Discover published block files as well so no
    # cache entry is lost even in that race.
    seen_files = {block.get("file") for block in blocks if isinstance(block, dict)}
    base = _v2_base_dir(cache_dir, base_key)
    if base.is_dir():
        for path in sorted(base.glob("block_*.pt")):
            if path.name not in seen_files:
                blocks.append({"file": path.name})
                seen_files.add(path.name)

    for blk in blocks:
        blk_layers = blk.get("layers", [])
        if not isinstance(blk_layers, list):
            continue
        blk_layer_ints = [int(item) for item in cast(list[int | str], blk_layers)]
        # Files discovered outside the manifest do not have their layer list until loaded.
        if blk_layer_ints and not (requested_set & set(blk_layer_ints)):
            continue
        file_rel = blk.get("file")
        if not isinstance(file_rel, str):
            continue
        path = _v2_base_dir(cache_dir, base_key) / file_rel
        if not path.exists():
            continue
        try:
            payload = torch.load(path, map_location="cpu")
            acts = payload["acts"]
            layers = [int(x) for x in payload["layers"]]
            if blk_layer_ints and layers != blk_layer_ints:
                raise ValueError("Layer list mismatch")
            # acts: [batch, seq, n_layers, emb]
            for i, layer in enumerate(layers):
                if layer in requested_set and layer not in out:
                    out[layer] = acts[:, :, i : i + 1, :]
        except Exception as err:
            logger.warning("Failed loading cache block %s: %s", path, err)

    return out


def save_activation_blocks(
    cache_dir: Path,
    base_key: str,
    layers: list[int],
    acts_tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> list[int]:
    """
    Save activations for the provided layers into block files and update manifest.

    This feature is intentionally disabled; direct recomputation is the supported path.
    """
    raise RuntimeError(ACTIVATION_CACHE_DISABLED_MESSAGE)
    if acts_tensor.ndim != 4:
        raise ValueError(f"acts_tensor must be 4D [batch,seq,layer,emb], got {acts_tensor.shape}")
    if acts_tensor.shape[2] != len(layers):
        raise ValueError("acts_tensor layer dim does not match layers list length")

    # The caller computed precisely the layers that were missing at lookup time. Another writer
    # may publish the same layers meanwhile; publishing the same content-addressed filename again
    # is safe and avoids trusting a possibly stale concurrent manifest.
    manifest = read_manifest(cache_dir, base_key)
    to_save_pairs = [(i, int(layer)) for i, layer in enumerate(layers)]
    if not to_save_pairs:
        return []

    base = _v2_base_dir(cache_dir, base_key)
    _mkdir_shared(base)

    saved_layers: list[int] = []
    # Chunk in order, but into fixed-size blocks.
    idx_layers = [p[1] for p in to_save_pairs]
    idx_map = {layer: i for i, layer in enumerate([int(item) for item in layers])}
    for start in range(0, len(idx_layers), block_size):
        chunk_layers = idx_layers[start : start + block_size]
        chunk_idxs = [idx_map[layer] for layer in chunk_layers]
        chunk_acts = acts_tensor[:, :, chunk_idxs, :].contiguous()

        filename = _block_filename(chunk_layers)
        path = base / filename
        tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
        try:
            torch.save({"version": V2_VERSION, "layers": chunk_layers, "acts": chunk_acts}, tmp)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        block = {"layers": chunk_layers, "file": filename}
        manifest_blocks = manifest.setdefault("blocks", [])
        if block not in manifest_blocks:
            manifest_blocks.append(block)
        saved_layers.extend(chunk_layers)

    write_manifest_atomically(cache_dir, base_key, manifest)
    return saved_layers


# -------------------------
# Probe cache (global, content-addressed)
# -------------------------

ProbeT = TypeVar("ProbeT")


class SaveableProbe(Protocol):
    def save(self, file_path: str | Path) -> Any: ...


def _detection_mask_hash(tokenized_dataset: TokenizedDataset) -> str | None:
    mask = getattr(tokenized_dataset, "detection_mask", None)
    if mask is None:
        return None
    return _get_or_compute_hash(tokenized_dataset, "_cache_detection_mask_hash", mask)


def get_probe_cache_dir() -> Path:
    d = get_cache_root() / PROBE_CACHE_DIRNAME
    _mkdir_shared(d)
    return d


def load_cached_probe(path: Path, loader: Callable[[Path], ProbeT]) -> ProbeT | None:
    """Load a probe defensively. Corrupt/incomplete entries self-heal as cache misses."""
    if not path.exists():
        return None
    try:
        return loader(path)
    except Exception as err:
        logger.warning("Failed to load cached probe at %s: %s; retraining", path, err)
        with suppress(Exception):
            path.unlink(missing_ok=True)
        return None


def save_probe_atomically(probe: SaveableProbe, path: Path) -> None:
    """Publish a fully-written probe so concurrent readers never observe a partial file."""
    _mkdir_shared(path.parent)
    # PIDs are only node-local; UUIDs also avoid collisions between hosts sharing a filesystem.
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        probe.save(tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _array_identity(values: np.ndarray[Any, Any] | None) -> dict[str, Any] | None:
    if values is None:
        return None
    contiguous = np.ascontiguousarray(values)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "hash": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def build_probe_cache_key(
    cfg: Any,
    honest_toks: TokenizedDataset,
    deceptive_toks: TokenizedDataset,
    layers: list[int],
    *,
    honest_base_key: str | None = None,
    deceptive_base_key: str | None = None,
    positive_fit_weights: np.ndarray[Any, Any] | None = None,
    negative_fit_weights: np.ndarray[Any, Any] | None = None,
) -> str:
    """
    Build a global probe cache key.

    Key is stable only when:
    - probe hyperparams and effective fit weights are identical
    - the paired tokenized datasets are identical (tokens, attention mask, detection mask)
    - model/lora/pw_locked and other cfg fields that affect tokenization/acts are identical
    """
    layers = list(layers)
    honest_base = honest_base_key or build_cache_base_key(
        cfg, honest_toks, stage="detector_train_honest"
    )
    deceptive_base = deceptive_base_key or build_cache_base_key(
        cfg, deceptive_toks, stage="detector_train_deceptive"
    )
    key_data = {
        "version": 2,
        "kind": "probe",
        "method": getattr(cfg, "method", None),
        "model_name": _model_name_to_str(getattr(cfg, "model_name", None)),
        "lora_path": _normalize_path(getattr(cfg, "lora_path", None)),
        "pw_locked": getattr(cfg, "pw_locked", None),
        "layers": layers,
        # hyperparams / knobs
        "reg_coeff": getattr(cfg, "reg_coeff", None),
        "normalize_acts": getattr(cfg, "normalize_acts", None),
        "max_iter": getattr(cfg, "max_iter", None),
        "use_class_weights": getattr(cfg, "use_class_weights", None),
        "detect_num_latents": getattr(cfg, "detect_num_latents", None),
        # dataset identity
        "honest_base_key": honest_base,
        "deceptive_base_key": deceptive_base,
        "honest_detection_mask_hash": _detection_mask_hash(honest_toks),
        "deceptive_detection_mask_hash": _detection_mask_hash(deceptive_toks),
        # Hash the values actually passed to fit(). Metadata-only train_source changes can
        # change these weights without changing any tokens, and must invalidate the cached probe.
        "positive_fit_weights": _array_identity(positive_fit_weights),
        "negative_fit_weights": _array_identity(negative_fit_weights),
        # include dataset partial ids for easier debugging (hash still derived from bases)
        "honest_partial_id": honest_toks.dataset_partial_id,
        "deceptive_partial_id": deceptive_toks.dataset_partial_id,
    }
    if getattr(cfg, "equal_dataset_weights", False):
        key_data["equal_dataset_weights"] = True
    if _model_name_to_str(getattr(cfg, "model_name", None)) == "gemma-4-31b-it":
        key_data["chat_rendering_version"] = GEMMA4_CHAT_RENDERING_VERSION
    encoded = json.dumps(key_data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
