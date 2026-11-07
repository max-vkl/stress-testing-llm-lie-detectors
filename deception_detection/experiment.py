import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import torch
import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from tqdm import tqdm, trange
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from deception_detection.activation_cache import (
    ACTIVATION_CACHE_DISABLED_MESSAGE,
    build_cache_base_key,
    build_probe_cache_key,
    get_cache_root,
    get_probe_cache_dir,
    load_activation_blocks,
    load_cached_probe,
    save_activation_blocks,
    save_probe_atomically,
)
from deception_detection.activations import Activations
from deception_detection.control_data_filters import export_skipped_control_samples_to_csv
from deception_detection.data import DialogueDataset
from deception_detection.detectors import (
    Detector,
    get_detector_class,
)
from deception_detection.dyl import (
    DYL_TRAIN_BASE_NAME,
    DYL_TRAIN_DATASET,
    augment_dataset_with_dyl_cache,
    base_probe_method,
    is_dyl_method,
)
from deception_detection.gemma4_chat import GEMMA4_CHAT_RENDERING_VERSION
from deception_detection.log import logger
from deception_detection.models import (
    ModelName,
    get_model_and_tokenizer,
    get_model_name,
    get_num_layers_for_model,
)
from deception_detection.repository import DatasetRepository
from deception_detection.sae import get_sae_acts
from deception_detection.scores import Scores
from deception_detection.tokenized_data import TokenizedDataset, get_tokenized_dataset_pair
from deception_detection.types import Message
from deception_detection.utils import REPO_ROOT

REPOSITORY = DatasetRepository()

Reduction = Literal["mean", "max", "relu"]


def cut_at_layer_for_detect_layers(detect_layers: list[int]) -> int | None:
    """Number of decoder blocks to keep (``model.layers[:cut_at_layer]``) when truncating the model
    for activation extraction over ``detect_layers``.

    One block ABOVE the top detection layer, not AT it. transformers applies the model's final
    RMSNorm to the LAST ``hidden_states`` entry (the loop appends each pre-block residual, then does
    ``hidden_states = self.norm(hidden_states)`` and appends that terminal entry). Cutting at
    exactly ``max(detect_layers)`` would make the top detection layer that terminal entry, so we
    read its post-final-norm output instead of the raw residual stream — a different representation,
    (because the layer-independent activation cache key does not record the cut) a silent cross-run
    cache hazard where a narrow-range run's normed boundary layer collides with an all-layers run's
    raw interior layer. Keeping one extra block makes every requested layer an interior (raw) entry,
    identical regardless of the cut. Slicing past the layer count is a safe Python no-op.

    Returns ``None`` (no truncation) when ``detect_layers`` is empty.
    """
    return max(detect_layers) + 1 if detect_layers else None


# we wrap in a lambda to avoid avoid passing model info as kwarg to check_dataset_name
DatasetName = Annotated[str, AfterValidator(lambda name: REPOSITORY.check_dataset_name(name))]
ControlDatasetName = Annotated[
    str, AfterValidator(lambda name: REPOSITORY.check_dataset_name(name))
]

# Forward-pass batch size for activation extraction. Override with DD_ACTS_BATCH; lower it if
# activation extraction runs out of GPU memory.
BATCH_SIZE_ACTS = int(os.getenv("DD_ACTS_BATCH", "32"))
BATCH_PAIRED_ACTS = 1  # 1

# Default shard size for the (now shardable) train activation computation. Kept equal to the eval
# default (12) so a data-parallel pre-pass and the main run tokenize identical shard units and
# therefore hit identical activation-cache keys.
TRAIN_SHARD_SIZE = int(os.getenv("DD_TRAIN_SHARD_SIZE", "12"))

# Per-dataset eval shard sizes. Centralized so the data-parallel activation pre-pass
# (precompute_acts.py) tokenizes each eval dataset in the exact same shard units the scoring path
# uses, guaranteeing identical activation-cache keys. Datasets not listed default to 12.
EVAL_SHARD_SIZES: dict[str, int] = {
    "lmsys": 2,
    "sandbagging_v2__wmdp_mmlu": 2,
    "ai_audit__reasoning": 1,
    "ai_audit__no_reasoning": 1,
    "roleplaying__plain": 1,
    "harm_refusal_choice__plain": 1,
    "harm_refusal_knowledge_report__plain": 1,
    "insider_trading_cadenza__plain": 1,
    "instruct_dishonesty_default__plain": 1,
    "instruct_dishonesty_donotadmit__plain": 1,
    "mask__plain": 1,
    "convince_me__plain": 1,
    "sleeper_agents__plain": 1,
    "insider_trading__onpolicy": 1,
    "insider_trading_doubledown__onpolicy": 1,
    "insider_trading__upscale": 2,
    "insider_trading_doubledown__upscale": 2,
    "alpaca__plain": 4,
}


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    created: int = 0

    def reset(self) -> None:
        self.hits = 0
        self.misses = 0
        self.created = 0


class ExperimentConfig(BaseModel):
    """
    Pydantic model for storing the configuration of an experiment.
    Saved to file alongside the experiment, except for the folder and name, which are automatically
    created at runtime.

    Attributes:
        # Core experiment configuration
        method: The method used to train the detector
        model_name: Which family of model to use, e.g. gemma, mistral, llama - we always use
            instruction tuned models

        # Dataset configuration
        train_data: The dataset used to train the detector in "base_name__variant" format
        eval_data: The datasets used to evaluate the detector in "base_name__variant" format
        control_data: The control datasets used in "base_name__variant" format
        trim_reasoning: Whether to trim reasoning from the datasets. If True, removes any reasoning
            text from the responses.

        # Training and evaluation settings
        train_on_policy: Whether to train on on policy data
        eval_on_policy: Whether to evaluate on on policy data
        control_on_policy: Whether to use on-policy data for control datasets
        detect_only_start_of_turn: Whether to only detect at the start of turn tokens
        detect_only_last_token: Whether to only detect at the last token before EOT
        train_data_last_content: Whether LTLR should retain the historical last-content behavior
            for the primary training dataset and its validation split. LTLR otherwise uses EOT.
        val_fraction: The fraction of training data to use for validation
        control_dataset_size: The number of dialogues in the control dataset

        # Other
        detect_layers: The layers to use for the detector. By default, the middle half of the layers
            are used.
        use_local_sae_acts: Whether to use the SAE acts for the detector
        use_goodfire_sae_acts: Whether to use the Goodfire SAE acts for the detector
        sae_latent_whitelist: List of the SAE features to use. By default, all features are used
        pw_locked: Whether to use a pw_locked version of the model
        lora_path: The path to a LORA finetune to attach to the model
        reg_coeff: The regularization coefficient for logistic regression methods (lr and mlr)
        use_class_weights: Whether to use class weights for logistic regression methods
            (lr, mlr, ltlr)

        # Experiment identification and storage
        id: Optional id prepended to the experiment name
        timestamp: The timestamp of the experiment, automatically generated
        name: The name of the experiment, automatically generated
        folder: The folder where the experiment is saved, automatically generated
    """

    # Core experiment configuration
    method: Literal[
        "mms",
        "ltmms",
        "ltmms_ct",
        "lat",
        "pca_repe",
        "lr",
        "ltlr",
        "ltlr_ct",
        "mlp",
        "cmms",
        "mlr",
        "best",
        "dyl_ltlr",
        "dyl_ltmms",
        "dyl_ltlr_ct",
        "dyl_ltmms_ct",
        "dyl_lr",
        "dyl_mms",
        "dyl_mlr",
    ]
    model_name: ModelName

    # Dataset configuration
    train_data: DatasetName = "repe_honesty"
    # Additional training datasets concatenated onto `train_data` (e.g. a confounder set the probe
    # should also learn from). Each is loaded on-policy (per-model rollouts) and appended AFTER the
    # primary train/val split, so the validation split is still drawn purely from `train_data`.
    # Empty by default => no behaviour change for existing configs.
    extra_train_data: list[DatasetName] = Field(default_factory=list)
    eval_data: list[DatasetName] = Field(default_factory=list)
    # Opt-in holdouts over complete rollout datasets. None preserves existing runs.
    eval_exclusions_file: Path | None = None
    control_data: list[DatasetName] = Field(default_factory=list)
    # If True, do not use/require any control datasets for this run.
    # Control-calibrated metrics/plots should be suppressed by postprocessing.
    no_control: bool = False
    trim_reasoning: bool = True

    # Training and evaluation settings
    train_on_policy: bool = False
    eval_on_policy: bool = True
    control_on_policy: bool = False
    detect_only_start_of_turn: bool = False
    detect_only_last_token: bool = False
    # LTLR uses the EOT activation by default. This narrow compatibility flag retains the old raw
    # last-detected-content activation for the primary training source and its validation split,
    # while evaluation datasets continue to use EOT.
    train_data_last_content: bool = False
    val_fraction: float = Field(default=0.2, ge=0, le=1)
    control_dataset_size: int = 1000

    # Detector configuration
    detect_layers: list[int] = Field(default_factory=list)  # if empty, will be set to middle half
    merge_layers: bool = False
    detect_num_latents: int | None = None

    # Other
    use_local_sae_acts: bool = False
    use_goodfire_sae_acts: bool = False
    sae_latent_whitelist: list[int] | None = None
    pw_locked: bool = False
    lora_path: str | Path | None = None
    reg_coeff: float | None = None
    max_iter: int | None = None  # Maximum iterations for LogisticRegression convergence
    normalize_acts: bool = True
    max_llama_token_length: int | None = None
    use_class_weights: bool = False
    # If True, reweight the probe fit so each train dataset (train_data + each extra_train_data
    # entry) contributes equal total weight WITHIN each class split (honest / deceptive), computed
    # after the balanced truncation. Weights are normalized to mean 1 per split so the effective
    # regularization scale is unchanged. No-op with a single train source. Only for lr/mlr/ltlr.
    equal_dataset_weights: bool = False
    # If set, the train/val split is GROUP-aware instead of an item-level tail slice: a group is
    # the tuple of these metadata keys (e.g. ["persona", "source_id"] = one source question), and
    # the val split consists of WHOLE groups collected from the shuffled tail until at least
    # val_fraction of items is reached. Prevents train/val question contamination when several
    # items share a source question.
    val_split_group_keys: list[str] = Field(default_factory=list)

    use_followup_question: bool = False
    detect_all_tokens: bool = False  # If True, detect on all tokens rather than just marked ones
    # Cross-experiment probe caching is the default. This master switch disables every cache.
    caching_off: bool = False
    # Compatibility-only input. True is rejected: persisted activation caching is disabled.
    inefficient_activation_caching: bool = False
    enable_rerun: bool = False

    # Postprocessing / plotting
    # If None: default to False for per-layer runs, True otherwise (handled in postprocess).
    include_violin: bool | None = None
    # If set, downsample control points for violin plots (plot appearance only).
    violin_control_sample_size: int | None = None

    # Debugging / analysis helpers
    debug_alpaca_false_positives: bool = False
    debug_alpaca_false_positive_fpr: float = Field(default=0.03, ge=0, le=1)
    # An unused debug default must not require datasets omitted from this reproduction.
    # Validate it only when that optional workflow is explicitly enabled.
    debug_alpaca_false_positive_dataset: str = "alpaca__plain"

    # Control dataset filtering
    # If set, drop any control examples whose metadata["id"] is in this list (e.g. Alpaca meta ids).
    control_data_skip_ids: list[int] = Field(default_factory=list)

    # Experiment identification and storage
    id: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    name: str = ""
    folder: Path = Field(default_factory=lambda: REPO_ROOT / "results")

    @model_validator(mode="before")
    @classmethod
    def _validate_cache_options(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        legacy = sorted({"deactivate_cache", "cache_dir"} & data.keys())
        if legacy:
            names = ", ".join(legacy)
            raise ValueError(
                f"Deprecated cache option(s): {names}. Remove them to use the probe-only "
                "default, or set caching_off=True to disable probe caching."
            )
        if data.get("inefficient_activation_caching"):
            raise ValueError(ACTIVATION_CACHE_DISABLED_MESSAGE)
        return data

    @field_validator("model_name", mode="before")
    @classmethod
    def validate_model_name(cls, v: str | ModelName) -> ModelName:
        if isinstance(v, ModelName):
            return v
        try:
            return ModelName(v)
        except ValueError as e:
            raise ValueError(
                f"Invalid model name: {v} not in {[m.value for m in ModelName]}"
            ) from e

    @field_validator("detect_layers", mode="before")
    @classmethod
    def validate_detect_layers(cls, v: Any, info):  # pydantic v2
        """
        Allow:
        - detect_layers: all
        - detect_layers: "20-25"
        - detect_layers: [20-25]
        - detect_layers: ["20-25", 31]
        Normalizes to list[int].
        """
        if v is None:
            return []

        # Normalize input to a list of tokens
        if isinstance(v, str | int):
            tokens = [v]
        elif isinstance(v, list):
            tokens = v
        else:
            raise TypeError(f"detect_layers must be a list/int/str, got {type(v)}")

        # Resolve model_name (needed for 'all' and bounds checking)
        model_name = info.data.get("model_name") if hasattr(info, "data") else None
        if isinstance(model_name, str):
            model_name = ModelName(model_name)
        n_layers = get_num_layers_for_model(model_name) if model_name is not None else None

        def parse_token(tok: Any) -> list[int]:
            if isinstance(tok, int):
                return [tok]
            if not isinstance(tok, str):
                raise TypeError(f"detect_layers entries must be int/str, got {type(tok)}")
            s = tok.strip().lower()
            if s == "all":
                if n_layers is None:
                    raise ValueError("detect_layers: all requires model_name to be set")
                return list(range(n_layers))
            m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", s)
            if m:
                start = int(m.group(1))
                end = int(m.group(2))
                if end < start:
                    raise ValueError(f"Invalid detect_layers range {tok!r}: end < start")
                return list(range(start, end + 1))
            if s.isdigit():
                return [int(s)]
            raise ValueError(
                f"Invalid detect_layers entry {tok!r}. Use ints, 'all', or 'x-y' ranges."
            )

        expanded: list[int] = []
        saw_all = False
        for tok in tokens:
            parsed = parse_token(tok)
            if n_layers is not None and parsed == list(range(n_layers)):
                saw_all = True
            expanded.extend(parsed)

        # If 'all' appeared alongside other tokens, treat as an error to avoid surprises.
        if saw_all and (len(tokens) > 1):
            raise ValueError("detect_layers: 'all' cannot be combined with other entries")

        # Deduplicate while preserving order
        expanded = list(dict.fromkeys(expanded))

        # Bounds check when possible
        if n_layers is not None:
            bad = [x for x in expanded if x < 0 or x >= n_layers]
            if bad:
                raise ValueError(
                    f"detect_layers out of range for {model_name} (n_layers={n_layers}): {bad}"
                )

        return expanded

    def model_post_init(self, _context: Any):
        if self.debug_alpaca_false_positives:
            REPOSITORY.check_dataset_name(self.debug_alpaca_false_positive_dataset)
        self._set_default_name()
        self._set_default_detect_layers()
        self._validate_reg_coeff()
        self._validate_equal_dataset_weights()
        self._validate_local_sae_acts_config()
        self._validate_goodfire_sae_acts_config()
        self._validate_detector_config()
        self._validate_control_config()
        self._setup_folder_and_config()

    @classmethod
    def from_path(cls, folder: Path):
        """Load config file within the folder (either json or yaml)"""
        if (folder / "cfg.yaml").exists():
            with open(folder / "cfg.yaml") as f:
                config_data = yaml.safe_load(f)
        else:
            assert (folder / "cfg.json").exists(), f"No config file found in folder {folder}"
            with open(folder / "cfg.json") as f:
                config_data = json.load(f)

        config_data["folder"] = folder.parent
        config_data["name"] = folder.name

        # Handle old config format
        if "type_model" in config_data:
            # Convert old format to new ModelName
            type_model = config_data.pop("type_model")
            size_model = config_data.pop("size_model")
            use_llama_3_3 = config_data.pop("use_llama_3_3", False)

            # Construct model name string
            model_str = f"{type_model}-{size_model}b"
            if use_llama_3_3 and type_model == "llama" and size_model == 70:
                model_str += "-3.3"

            # Convert to ModelName enum
            config_data["model_name"] = model_str

        return cls(**config_data)

    def _get_default_detect_layers(self) -> list[int]:
        n_layers = get_num_layers_for_model(self.model_name)
        return list(range(int(0.25 * n_layers), int(0.75 * n_layers)))

    def _get_default_name(self):
        # Put timestamp first so result folders sort chronologically.
        name = f"{self.timestamp}__{self.method}__{self.train_data}__{self.model_name.value}"
        if self.id:
            name = (
                f"{self.timestamp}__{self.id}__{self.method}__{self.train_data}__"
                f"{self.model_name.value}"
            )
        name = name.replace("/", "_")
        return name

    def _set_default_name(self) -> None:
        if not self.name:
            self.name = self._get_default_name()

    def _check_train_data_not_in_eval_data(self) -> None:
        if self.train_data in self.eval_data:
            raise ValueError(f"Train data {self.train_data} is also in eval data")

    def _set_default_detect_layers(self) -> None:
        if not self.detect_layers:
            self.detect_layers = self._get_default_detect_layers()

    def _validate_local_sae_acts_config(self) -> None:
        if self.use_local_sae_acts:
            if not self.model_name.type == "gemma":
                raise ValueError("SAE acts not supported for non-gemma models")
            if not self.model_name.size == 9:
                raise ValueError("SAEs only trained for gemma 9b-it")
            if not all([layer in [9, 20, 31] for layer in self.detect_layers]):
                raise ValueError("SAEs only trained at layers [9, 20, 31]")

    def _validate_reg_coeff(self) -> None:
        """Validate that reg_coeff is only set for logistic-regression methods."""
        if self.reg_coeff is not None and base_probe_method(self.method) not in [
            "lr",
            "mlr",
            "ltlr",
            "ltlr_ct",
        ]:
            raise ValueError("reg_coeff can only be set for lr, mlr, ltlr, and ltlr_ct methods")

    def _validate_equal_dataset_weights(self) -> None:
        """Per-dataset fit weighting is implemented only for the sklearn LR-family detectors."""
        if self.equal_dataset_weights and base_probe_method(self.method) not in [
            "lr",
            "mlr",
            "ltlr",
            "ltlr_ct",
        ]:
            raise ValueError(
                "equal_dataset_weights is only supported for lr, mlr, ltlr, and ltlr_ct"
            )

    def _validate_goodfire_sae_acts_config(self) -> None:
        if self.use_goodfire_sae_acts:
            if not self.model_name.type == "llama":
                raise ValueError("Goodfire SAE acts only supported for llama models")
            if not self.model_name.size == 70:
                raise ValueError("Goodfire SAE acts only trained for llama 70b")
            if not self.detect_layers == [0]:
                raise ValueError("Detect layers must be set to [0] for goodfire SAEs.")
        if self.max_llama_token_length is not None:
            if self.model_name.type != "llama":
                raise ValueError("max_llama_token_length only supported for llama models")
            if not self.use_goodfire_sae_acts:
                raise ValueError("max_llama_token_length only supported when using goodfire SAEs")

    def _setup_folder_and_config(self) -> None:
        self.folder = self.folder / self.name
        self.folder.mkdir(parents=True, exist_ok=True)
        if not (self.folder / "cfg.yaml").exists():
            self._save_config()

    @field_serializer("model_name")
    def serialize_model_name(self, model_name: ModelName) -> str:
        return model_name.value

    def _save_config(self):
        with open(self.folder / "cfg.yaml", "w") as f:
            config_dict = self.model_dump(exclude={"name", "folder"})
            if config_dict.get("eval_exclusions_file") is None:
                config_dict.pop("eval_exclusions_file", None)
            else:
                config_dict["eval_exclusions_file"] = str(config_dict["eval_exclusions_file"])
            # Keep legacy cfg.yaml files byte-identical: only persist these when set.
            if not config_dict.get("equal_dataset_weights"):
                config_dict.pop("equal_dataset_weights", None)
            if not config_dict.get("val_split_group_keys"):
                config_dict.pop("val_split_group_keys", None)
            if not config_dict.get("caching_off"):
                config_dict.pop("caching_off", None)
            if not config_dict.get("inefficient_activation_caching"):
                config_dict.pop("inefficient_activation_caching", None)
            if not config_dict.get("train_data_last_content"):
                config_dict.pop("train_data_last_content", None)
            yaml.safe_dump(config_dict, f, indent=2)

    def _validate_detector_config(self) -> None:
        if self.method == "best" and self.detect_num_latents is None:
            raise ValueError("detect_num_latents must be set for best latents detector")
        if is_dyl_method(self.method) and self.train_data != DYL_TRAIN_DATASET:
            raise ValueError(
                f"{self.method} requires train_data={DYL_TRAIN_DATASET!r}; "
                f"got {self.train_data!r}"
            )
        if self.train_data_last_content and base_probe_method(self.method) != "ltlr":
            raise ValueError("train_data_last_content is only supported for ltlr")
        if self.train_data_last_content and self.detect_only_last_token:
            raise ValueError(
                "train_data_last_content and detect_only_last_token are mutually exclusive: "
                "the latter already selects last-content tokens for every dataset"
            )
        if self.train_data_last_content and self.extra_train_data:
            raise ValueError(
                "train_data_last_content cannot be combined with extra_train_data because the "
                "combined training rows cannot use two token-selection policies"
            )
        if is_dyl_method(self.method) and self.use_followup_question:
            raise ValueError(
                f"{self.method} already appends the DYL question; "
                "use_followup_question must be false"
            )

    def _validate_control_config(self) -> None:
        """
        Validate control dataset configuration.

        - By default (no_control=False), experiments require at least one control dataset
          because postprocessing and several scoring utilities calibrate thresholds on it.
        - If no_control=True, control datasets are optional and control-dependent debugging
          helpers are disabled.
        """
        if self.no_control:
            if self.debug_alpaca_false_positives:
                raise ValueError(
                    "debug_alpaca_false_positives requires a control dataset, but no_control=True."
                )
            return

        if not self.control_data:
            raise ValueError(
                "No control dataset specified. Set control_data (e.g. ['alpaca__plain']) "
                "or set no_control: true to run without a control dataset."
            )


class Experiment:
    cfg: ExperimentConfig
    tokenizer: PreTrainedTokenizerBase
    _model: PreTrainedModel | None

    def __init__(self, cfg: ExperimentConfig):
        """Initializes the experiment.

        Args:
            cfg (ExperimentConfig): Configuration for the experiment.
        """
        self.cfg = cfg
        self._model = None  # Store actual model in _model attribute
        _, tokenizer = get_model_and_tokenizer(
            cfg.model_name, omit_model=True, lora_path=cfg.lora_path
        )
        self.tokenizer = tokenizer
        self.train_dataset = self.get_train_dataset()
        self.eval_datasets = self.get_eval_datasets()
        self.control_datasets = self.get_control_datasets()
        assert self.cfg.folder.exists(), f"Folder {self.cfg.folder} does not exist"
        self._cache_stats: dict[str, CacheStats] = {
            "train": CacheStats(),
            "eval": CacheStats(),
            "control": CacheStats(),
            "unspecified": CacheStats(),
        }

    # ----- DETECTOR ----- #

    def _cache_stats_for(self, stage: str | None) -> CacheStats:
        key = stage or "unspecified"
        return self._cache_stats.setdefault(key, CacheStats())

    def _reset_cache_stats(self, stage: str | None) -> None:
        self._cache_stats_for(stage).reset()

    def _print_cache_stats(self, stage: str | None) -> None:
        stats = self._cache_stats_for(stage)
        key = stage or "unspecified"
        print(
            f"Cache summary ({key}): hits={stats.hits}, misses={stats.misses}, "
            f"created={stats.created}"
        )

    def _merge_layers_enabled(self) -> bool:
        return bool(self.cfg.merge_layers) or len(self.cfg.detect_layers) <= 1

    def _train_class_splits(self) -> tuple[DialogueDataset, DialogueDataset]:
        """Honest/deceptive splits of the combined train set, both truncated to the shorter
        side (balanced classes)."""
        honest = self.train_dataset.get_honest()
        deceptive = self.train_dataset.get_deceptive()
        min_length = min(len(honest), len(deceptive))
        return honest[:min_length], deceptive[:min_length]

    def _per_dataset_fit_weights(self, split: DialogueDataset) -> np.ndarray | None:
        """Per-dialogue fit weights making each train_source contribute equal total weight within
        this class split (computed AFTER the balanced truncation); normalized to mean 1 so the
        effective regularization scale is unchanged. None (= uniform) unless
        cfg.equal_dataset_weights is set and provenance tags with >=2 sources are present."""
        if not self.cfg.equal_dataset_weights:
            return None
        md = split.metadata
        if md is None or "train_source" not in md:
            return None
        sources = md["train_source"]
        counts = Counter(sources)
        if len(counts) <= 1:
            return None
        weights = np.array([1.0 / counts[s] for s in sources], dtype=np.float64)
        weights *= len(weights) / weights.sum()
        logger.info("equal_dataset_weights: sources=%s (n=%d)", dict(counts), len(weights))
        return weights

    def _fit_weight_kwargs(
        self, honest: DialogueDataset, deceptive: DialogueDataset
    ) -> dict[str, Any]:
        """fit() kwargs for per-dataset weighting; empty when weighting is off/uniform so the
        unweighted call (and non-LR detector signatures) stay untouched."""
        honest_weights = self._per_dataset_fit_weights(honest)
        deceptive_weights = self._per_dataset_fit_weights(deceptive)
        if honest_weights is None and deceptive_weights is None:
            return {}
        return {"positive_weights": deceptive_weights, "negative_weights": honest_weights}

    def _ensure_chat_rendering_current(self) -> None:
        """Never reuse an experiment-local Gemma 4 probe with the legacy chat."""
        if self.cfg.model_name != ModelName.GEMMA_4_31B_IT:
            return
        marker = self.cfg.folder / "gemma4_chat_rendering.json"
        expected = {"version": GEMMA4_CHAT_RENDERING_VERSION}
        if marker.exists() and json.loads(marker.read_text()) == expected:
            return
        if list(self.cfg.folder.glob("detector*.pt")) or list(self.cfg.folder.glob("*scores*.json")):
            raise ValueError(
                "Legacy Gemma 4 probes/scores predate the native-chat correction. "
                "Use a new result directory or remove the affected cached artifacts first."
            )
        marker.write_text(json.dumps(expected) + "\n")

    def get_detector(self) -> Detector:
        """Load a local/global probe when available, otherwise train and cache it."""

        self._ensure_chat_rendering_current()
        path = self.cfg.folder / "detector.pt"
        detector_class = get_detector_class(self.cfg.method)

        if path.exists():
            return detector_class.load(path)

        if not self._merge_layers_enabled():
            raise ValueError(
                "Multiple detect_layers provided with merge_layers=False. "
                "Use get_detectors() (one detector per layer)."
            )

        honest, deceptive = self._train_class_splits()
        honest_toks, deceptive_toks = get_tokenized_dataset_pair(
            honest,
            deceptive,
            self.tokenizer,
            enforce_equal_detect_lengths=False,
            detect_all=self.cfg.detect_all_tokens,
            detect_only_start_of_turn=self.cfg.detect_only_start_of_turn,
            detect_only_last_token=self.uses_last_content_token(),
            detect_only_eot=self.uses_ltlr_eot(honest),
        )
        fit_weight_kwargs = self._fit_weight_kwargs(honest, deceptive)

        global_path: Path | None = None
        if not self.cfg.caching_off:
            honest_base_key = build_cache_base_key(
                self.cfg, honest_toks, stage="probe_train_honest"
            )
            deceptive_base_key = build_cache_base_key(
                self.cfg, deceptive_toks, stage="probe_train_deceptive"
            )
            probe_cache_key = build_probe_cache_key(
                self.cfg,
                honest_toks,
                deceptive_toks,
                self.cfg.detect_layers,
                honest_base_key=honest_base_key,
                deceptive_base_key=deceptive_base_key,
                positive_fit_weights=fit_weight_kwargs.get("positive_weights"),
                negative_fit_weights=fit_weight_kwargs.get("negative_weights"),
            )
            global_path = get_probe_cache_dir() / f"{probe_cache_key}.pt"
            detector = load_cached_probe(global_path, detector_class.load)
            if detector is not None:
                logger.info("Loading probe from global cache: %s", global_path)
                logger.info("Saving cached probe to %s", path)
                detector.save(path)
                return detector

        logger.info("Training detector")
        self._reset_cache_stats("train")
        acts_bar = tqdm(
            [("honest", honest_toks), ("deceptive", deceptive_toks)],
            desc="Train activations",
        )
        acts_by_name: dict[str, Activations] = {}
        for name, toks in acts_bar:
            acts_by_name[name] = self.get_acts_for_layers(
                toks, self.cfg.detect_layers, stage="train"
            )
        honest_fit_acts = acts_by_name["honest"]
        deceptive_fit_acts = acts_by_name["deceptive"]

        detector_kwargs: dict[str, Any] = {"layers": self.cfg.detect_layers}
        if self.cfg.method == "best":
            detector_kwargs["detect_num_latents"] = self.cfg.detect_num_latents
            detector_kwargs["experiment_path"] = self.cfg.folder
        if self.cfg.reg_coeff is not None:
            detector_kwargs["reg_coeff"] = self.cfg.reg_coeff
        if base_probe_method(self.cfg.method) in ["lr", "ltlr", "ltlr_ct", "mlr"]:
            detector_kwargs["normalize"] = self.cfg.normalize_acts
            if self.cfg.max_iter is not None:
                detector_kwargs["max_iter"] = self.cfg.max_iter
            detector_kwargs["use_class_weights"] = self.cfg.use_class_weights
        detector = detector_class(**detector_kwargs)
        detector.fit(deceptive_fit_acts, honest_fit_acts, **fit_weight_kwargs)
        logger.info("Saving detector to %s", path)
        detector.save(path)
        if global_path is not None:
            try:
                save_probe_atomically(detector, global_path)
                logger.info("Saved probe to global cache: %s", global_path)
            except Exception as err:
                logger.warning("Failed saving probe to global cache %s: %s", global_path, err)
        if self.cfg.inefficient_activation_caching:
            self._print_cache_stats("train")
        return detector

    def _get_layer_detector_path(self, layer: int) -> Path:
        return self.cfg.folder / f"detector_layer_{layer}.pt"

    def get_detectors(self) -> dict[int, Detector]:
        """
        Return one detector per layer when merge_layers is disabled.
        Trains missing detectors efficiently by computing activations once per toks.
        """
        self._ensure_chat_rendering_current()
        if self._merge_layers_enabled():
            return {self.cfg.detect_layers[0]: self.get_detector()}

        layers = list(dict.fromkeys(self.cfg.detect_layers))
        detector_class = get_detector_class(self.cfg.method)

        detectors: dict[int, Detector] = {}
        missing_layers: list[int] = []
        for layer in layers:
            path = self._get_layer_detector_path(layer)
            if path.exists():
                detectors[layer] = detector_class.load(path)
            else:
                missing_layers.append(layer)
        if not missing_layers:
            return detectors

        # Prepare paired tokenized datasets (same logic as PairedActivations.from_dataset)
        honest, deceptive = self._train_class_splits()
        honest_toks, deceptive_toks = get_tokenized_dataset_pair(
            honest,
            deceptive,
            self.tokenizer,
            enforce_equal_detect_lengths=False,
            detect_all=self.cfg.detect_all_tokens,
            detect_only_start_of_turn=self.cfg.detect_only_start_of_turn,
            detect_only_last_token=self.uses_last_content_token(),
            detect_only_eot=self.uses_ltlr_eot(honest),
        )

        fit_weight_kwargs = self._fit_weight_kwargs(honest, deceptive)
        probe_paths: dict[int, Path] = {}
        if not self.cfg.caching_off:
            probe_cache_dir = get_probe_cache_dir()
            honest_base_key = build_cache_base_key(
                self.cfg, honest_toks, stage="probe_train_honest"
            )
            deceptive_base_key = build_cache_base_key(
                self.cfg, deceptive_toks, stage="probe_train_deceptive"
            )
            still_missing: list[int] = []
            for layer in missing_layers:
                probe_cache_key = build_probe_cache_key(
                    self.cfg,
                    honest_toks,
                    deceptive_toks,
                    [layer],
                    honest_base_key=honest_base_key,
                    deceptive_base_key=deceptive_base_key,
                    positive_fit_weights=fit_weight_kwargs.get("positive_weights"),
                    negative_fit_weights=fit_weight_kwargs.get("negative_weights"),
                )
                global_path = probe_cache_dir / f"{probe_cache_key}.pt"
                probe_paths[layer] = global_path
                detector = load_cached_probe(global_path, detector_class.load)
                if detector is None:
                    still_missing.append(layer)
                    continue
                logger.info("Loading probe for layer %d from global cache", layer)
                detector.save(self._get_layer_detector_path(layer))
                detectors[layer] = detector
            missing_layers = still_missing
        if not missing_layers:
            return detectors

        logger.info("Training %d per-layer detector(s)", len(missing_layers))
        self._reset_cache_stats("train")

        # Compute activations only for layers we still need to train, in CHUNKS of layers. The
        # last-token path compacts each shard to its selected readouts before accumulation.
        # Other methods may still retain full sequences. Per-layer fits are independent, so
        # chunking bounds memory; with cache-free activations, each chunk is a new forward pass.
        layer_chunk = int(os.getenv("DD_ASSEMBLE_LAYER_CHUNK", "16"))
        for chunk_start in range(0, len(missing_layers), layer_chunk):
            chunk = missing_layers[chunk_start : chunk_start + layer_chunk]
            logger.info(
                "Training detectors for layer chunk %s..%s (%d of %d remaining layers)",
                chunk[0],
                chunk[-1],
                len(chunk),
                len(missing_layers) - chunk_start,
            )
            honest_acts_by_layer, deceptive_acts_by_layer = self._train_paired_acts_by_layer(
                chunk, shard_size=TRAIN_SHARD_SIZE, stage="train"
            )

            for layer in chunk:
                detector_kwargs: dict[str, Any] = {"layers": [layer]}
                if self.cfg.method == "best":
                    detector_kwargs["detect_num_latents"] = self.cfg.detect_num_latents
                    detector_kwargs["experiment_path"] = self.cfg.folder
                if self.cfg.reg_coeff is not None:
                    detector_kwargs["reg_coeff"] = self.cfg.reg_coeff
                if base_probe_method(self.cfg.method) in ["lr", "ltlr", "ltlr_ct", "mlr"]:
                    detector_kwargs["normalize"] = self.cfg.normalize_acts
                if base_probe_method(self.cfg.method) in ["lr", "ltlr", "ltlr_ct", "mlr"]:
                    if self.cfg.max_iter is not None:
                        detector_kwargs["max_iter"] = self.cfg.max_iter
                    detector_kwargs["use_class_weights"] = self.cfg.use_class_weights

                detector = detector_class(**detector_kwargs)
                detector.fit(
                    deceptive_acts_by_layer[layer], honest_acts_by_layer[layer], **fit_weight_kwargs
                )
                out_file = self._get_layer_detector_path(layer)
                logger.info("Saving detector for layer %d to %s", layer, out_file)
                detector.save(out_file)
                global_path = probe_paths.get(layer)
                if global_path is not None:
                    try:
                        save_probe_atomically(detector, global_path)
                    except Exception as err:
                        logger.warning(
                            "Failed saving probe for layer %d to global cache %s: %s",
                            layer,
                            global_path,
                            err,
                        )
                detectors[layer] = detector
                # Free this layer's activations as soon as its detector is fit.
                del honest_acts_by_layer[layer], deceptive_acts_by_layer[layer]
            del honest_acts_by_layer, deceptive_acts_by_layer

        if self.cfg.inefficient_activation_caching:
            self._print_cache_stats("train")
        return detectors

    # ----- SHARDED ACTIVATIONS (data-parallel friendly) ----- #

    def _eval_shard_size(self, dataset: DialogueDataset) -> int | None:
        """Shard size used for a dataset's activation/score computation.

        Mirrors the logic in compute_scores*_by_layer so the pre-pass and the scoring path
        tokenize identical shard units (and therefore hit identical activation-cache keys).
        """
        if self.cfg.use_goodfire_sae_acts:
            return None
        return EVAL_SHARD_SIZES.get(dataset.name, 12)

    def _train_shards(self, shard_size: int):
        """Yield (side, start, toks) for fixed-size shards of each train class split.

        Tokenization is byte-identical to the monolithic path except for per-shard (vs whole-set)
        left-padding, which is invariant for real-token activations. This makes the train
        activation computation shardable and data-parallel-friendly.

        Iteration is start-major (honest then deceptive per start), reproducing the historical
        pair order — and therefore worker assignment and every activation cache key — exactly.
        Per-side tokenization matches get_tokenized_dataset_pair (independent from_dataset calls
        with additional_id=side), so cache keys are pairing-independent.
        """
        honest, deceptive = self._train_class_splits()
        for start in range(0, max(len(honest), len(deceptive)), shard_size):
            for side, split in (("honest", honest), ("deceptive", deceptive)):
                if start >= len(split):
                    continue
                toks = TokenizedDataset.from_dataset(
                    split[start : start + shard_size],
                    self.tokenizer,
                    detect_all=self.cfg.detect_all_tokens,
                    detect_only_start_of_turn=self.cfg.detect_only_start_of_turn,
                    detect_only_last_token=self.uses_last_content_token(),
                    detect_only_eot=self.uses_ltlr_eot(split),
                    additional_id=side,
                )
                yield side, start, toks

    def _train_paired_acts_by_layer(
        self, layers: list[int], shard_size: int = TRAIN_SHARD_SIZE, stage: str = "train"
    ) -> tuple[dict[int, Activations], dict[int, Activations]]:
        """Compute per-layer honest/deceptive train activations via the shardable, cached path.

        Returns ({layer: honest_acts}, {layer: deceptive_acts}), each combining all shards. Uses
        get_acts_by_layer per shard (per-shard cache keys), so a data-parallel pre-pass that warms
        the same shards produces full cache hits here. Last-token training retains only the
        selected token when each dialogue has exactly one readout. Full-context forwards and
        cache contents are unchanged; only the in-memory assembly representation is compacted.
        """
        layers = list(dict.fromkeys(layers))
        honest_by_layer: dict[int, list[Activations]] = {layer: [] for layer in layers}
        deceptive_by_layer: dict[int, list[Activations]] = {layer: [] for layer in layers}
        compact_readouts = base_probe_method(self.cfg.method) in (
            "ltlr",
            "ltlr_ct",
            "ltmms",
            "ltmms_ct",
        )
        shards = list(self._train_shards(shard_size))
        for side, _start, toks in tqdm(shards, desc="Train activations (sharded)"):
            acts = self.get_acts_by_layer(toks, layers, stage=stage)
            by_layer = honest_by_layer if side == "honest" else deceptive_by_layer
            for layer in layers:
                by_layer[layer].append(
                    acts[layer].compact_single_detection_token()
                    if compact_readouts
                    else acts[layer]
                )
            del acts
        honest_combined = {
            layer: Activations.concat_padded(honest_by_layer.pop(layer)) for layer in layers
        }
        deceptive_combined = {
            layer: Activations.concat_padded(deceptive_by_layer.pop(layer)) for layer in layers
        }
        return honest_combined, deceptive_combined

    def iter_activation_work(
        self, layers: list[int] | None = None, train_shard_size: int | None = None
    ):
        """Yield (label, thunk) for every activation shard this experiment needs (train + eval).

        Each thunk computes and caches one shard's activations (and returns None). The order is
        deterministic, so a data-parallel launcher can assign shard i to worker (i % num_shards)
        and every shard is covered exactly once. Building the Experiment with omit_model=True lets
        `enumerate`-style callers count the work without a GPU; the thunks lazily trigger the model
        via the `model` property only when actually run.
        """
        if layers is None:
            layers = list(dict.fromkeys(self.cfg.detect_layers))
        else:
            layers = list(dict.fromkeys(layers))
        train_shard_size = train_shard_size or TRAIN_SHARD_SIZE

        # Train shards (honest + deceptive), stage="train". Each thunk returns the number of
        # dialogues in the shard (ignored by the precompute worker; used by the benchmark harness).
        for side, start, toks in self._train_shards(train_shard_size):

            def _train_thunk(toks=toks) -> int:
                self.get_acts_by_layer(toks, layers, stage="train")
                return int(toks.tokens.shape[0])

            yield f"train:{side}:{start}", _train_thunk

        # Eval + control dataset shards (self.eval_datasets includes the injected *_val split).
        for stage, datasets in (
            ("eval", self.eval_datasets),
            ("control", self.control_datasets),
        ):
            for dataset in datasets:
                shard_size = self._eval_shard_size(dataset) or len(dataset)
                for start in range(0, len(dataset), shard_size):
                    shard = dataset[start : start + shard_size]

                    def _eval_thunk(shard=shard, stage=stage) -> int:
                        toks = TokenizedDataset.from_dataset(
                            shard,
                            self.tokenizer,
                            detect_all=self.cfg.detect_all_tokens,
                            detect_only_start_of_turn=self.cfg.detect_only_start_of_turn,
                            detect_only_last_token=self.uses_last_content_token(),
                            detect_only_eot=self.uses_ltlr_eot(shard),
                        )
                        self.get_acts_by_layer(toks, layers, stage=stage)
                        return int(toks.tokens.shape[0])

                    yield f"{stage}:{dataset.name}:{start}", _eval_thunk

    # ----- DATASETS ----- #

    def _split_train_val(
        self, dataset: DialogueDataset
    ) -> tuple[DialogueDataset, DialogueDataset | None]:
        """Split the loaded primary train dataset into (train, val).

        Default: the historical item-level tail slice. With cfg.val_split_group_keys set, the
        split is GROUP-aware: whole groups (tuples of the given metadata keys, e.g. one source
        question) are collected from the shuffled tail until at least val_fraction of items is
        reached, so no group appears on both sides. Deterministic given the seeded shuffle; used
        identically by get_train_dataset and get_eval_datasets.

        A dataset can define an explicit partition through metadata["_split"]. This takes precedence
        over val_fraction and is used for tiny custom train sets whose validation set is the
        remainder of a larger source dataset.
        """
        metadata = dataset.metadata
        if metadata is not None and "_split" in metadata:
            split_values = metadata["_split"]
            invalid = sorted(set(split_values) - {"train", "val"})
            if invalid:
                raise ValueError(f"Invalid predefined split values: {invalid}")
            train_mask = [value == "train" for value in split_values]
            val_mask = [value == "val" for value in split_values]
            if not any(train_mask) or not any(val_mask):
                raise ValueError("A predefined split must contain both train and val samples")
            return dataset.subset_where_true(train_mask), dataset.subset_where_true(val_mask)

        num_val = int(len(dataset) * self.cfg.val_fraction)
        if num_val <= 0:
            return dataset, None
        keys = self.cfg.val_split_group_keys
        if not keys:
            return dataset[:-num_val], dataset[-num_val:]
        md = dataset.metadata
        missing = [k for k in keys if md is None or k not in md]
        if missing:
            raise ValueError(
                f"val_split_group_keys {missing} not present in the metadata of "
                f"{self.cfg.train_data}"
            )
        groups = [tuple(md[k][i] for k in keys) for i in range(len(dataset))]
        group_sizes = Counter(groups)
        val_groups: set[tuple[Any, ...]] = set()
        collected = 0
        for group in reversed(groups):
            if group in val_groups:
                continue
            val_groups.add(group)
            collected += group_sizes[group]
            if collected >= num_val:
                break
        val_mask = [group in val_groups for group in groups]
        train = dataset.subset_where_true([not v for v in val_mask])
        val = dataset.subset_where_true(val_mask)
        return train, val

    def get_train_dataset(self) -> DialogueDataset:
        train_dataset = self.get_dataset(self.cfg.train_data, model=self.train_policy_model)
        train_dataset, _ = self._split_train_val(train_dataset)
        if not self.cfg.extra_train_data:
            return train_dataset
        # Append extra training datasets AFTER the val split so validation stays drawn purely from
        # `train_data`. These are loaded on-policy (they exist only as per-model rollouts).
        # Reduce metadata on both sides to a shared provenance tag {"train_source"}: the primary
        # and extra sets carry different metadata schemas (which DialogueDataset.__add__ refuses
        # to merge), LTLR training only needs dialogues + labels, and the tag enables the
        # per-dataset fit weighting (cfg.equal_dataset_weights). Slicing/shuffle keep metadata
        # index-aligned.
        train_dataset.metadata = {"train_source": [self.cfg.train_data] * len(train_dataset)}
        for name in self.cfg.extra_train_data:
            extra = self.get_dataset(name, model=self.cfg.model_name.value)
            extra.metadata = {"train_source": [name] * len(extra)}
            train_dataset = train_dataset + extra
        return train_dataset

    def get_eval_datasets(self) -> list[DialogueDataset]:
        eval_datasets = [
            self.get_dataset(name, model=self.eval_policy_model) for name in self.cfg.eval_data
        ]

        from deception_detection.eval_exclusions import apply_eval_exclusions

        eval_datasets = apply_eval_exclusions(
            eval_datasets, self.eval_policy_model, self.cfg.eval_exclusions_file, self.cfg.folder,
            registered_paths=REPOSITORY.dataset_paths,
        )

        # Add validation split if val_fraction > 0 (same helper as get_train_dataset, so the
        # two sides of the split always agree — including the group-aware mode).
        train_dataset = self.get_dataset(self.cfg.train_data, model=self.train_policy_model)
        _, val_dataset = self._split_train_val(train_dataset)
        if val_dataset:
            assert val_dataset.variant is not None
            val_dataset.variant = val_dataset.variant + "_val"
            eval_datasets = [val_dataset] + eval_datasets

        return eval_datasets

    def get_control_datasets(self) -> list[DialogueDataset]:
        """Get control datasets using either on-policy or pre-written data based on config."""
        model = get_model_name(self.cfg.model_name) if self.cfg.control_on_policy else "prewritten"

        control_datasets: list[DialogueDataset] = []
        for name in self.cfg.control_data:
            dataset = self.get_dataset(name, model=model)[: self.cfg.control_dataset_size]

            # Optional: filter control dataset by metadata id
            if self.cfg.control_data_skip_ids:
                if dataset.metadata is None or "id" not in dataset.metadata:
                    logger.warning(
                        f"control_data_skip_ids set, but dataset {dataset.name} has no "
                        "metadata['id']; "
                        "skipping filtering."
                    )
                else:
                    ids = dataset.metadata["id"]
                    skip_set = set(self.cfg.control_data_skip_ids)
                    # Indices *within this dataset slice* that should be dropped.
                    skipped_indices = [i for i, meta_id in enumerate(ids) if meta_id in skip_set]
                    if skipped_indices:
                        out_csv = self.cfg.folder / f"{dataset.name}__skipped_control_samples.csv"
                        export_skipped_control_samples_to_csv(
                            dataset, skipped_indices=skipped_indices, out_csv_path=out_csv
                        )
                        keep_mask = [i not in skip_set for i in ids]
                        dataset = dataset.subset_where_true(keep_mask)

            control_datasets.append(dataset)

        return control_datasets

    @property
    def train_policy_model(self) -> str:
        """
        Get the policy model used for training the detector. Either the same model or prewritten.
        """
        if self.cfg.train_on_policy:
            return self.cfg.model_name.value
        else:
            return "prewritten"

    @property
    def eval_policy_model(self) -> str:
        """
        Get the policy model used for evaluation. Either the same model or prewritten.
        """
        if self.cfg.eval_on_policy:
            return self.cfg.model_name.value
        else:
            return "prewritten"

    def uses_ltlr_eot(self, dataset: DialogueDataset) -> bool:
        """Whether this dataset should be tokenized at EOT for a last-token detector.

        EOT is the LTLR/LTMMS default. The only exceptions are the global last-content override and
        the LTLR compatibility mode used to preserve historical primary-train/validation runs.
        """
        if (
            base_probe_method(self.cfg.method) not in ("ltlr", "ltmms")
            or self.cfg.detect_only_last_token
        ):
            return False
        if not self.cfg.train_data_last_content:
            return True

        train_base_name, train_variant = self.cfg.train_data.split("__", maxsplit=1)
        dataset_variant = dataset.variant.removesuffix("_val")
        is_primary_train_source = (
            dataset.base_name == train_base_name and dataset_variant == train_variant
        )
        return not is_primary_train_source

    def uses_last_content_token(self) -> bool:
        """Whether tokenization should select content immediately before EOT.

        The explicit *_ct methods always use last-content activations. The legacy global
        override remains supported so existing configurations retain their behavior.
        """
        return (
            base_probe_method(self.cfg.method)
            in (
                "ltlr_ct",
                "ltmms_ct",
            )
            or self.cfg.detect_only_last_token
        )

    def get_dataset(self, name: str, model: str) -> DialogueDataset:
        dataset = REPOSITORY.get(
            name,
            model=model,
            trim_reasoning=self.cfg.trim_reasoning,
            shuffle_upon_init=not name.startswith("repe"),
            max_llama_token_length=self.cfg.max_llama_token_length,
            tokenizer=getattr(self, "tokenizer", None),
        )

        if is_dyl_method(self.cfg.method) and dataset.base_name != DYL_TRAIN_BASE_NAME:
            dataset = augment_dataset_with_dyl_cache(dataset, model)

        if self.cfg.use_followup_question:
            followup_question = "Was your last response deceptive? (Yes/No)"
            for dialogue in dataset.dialogues:
                for message in dialogue:
                    message.detect = False
                dialogue.extend(
                    [
                        Message(role="user", content=followup_question, detect=False),
                        Message(role="assistant", content="No", detect=True),
                    ]
                )
        return dataset

    # ---- SCORES ----- #

    def get_control_scores(self, all_tokens: bool = False) -> dict[str, Scores]:
        """
        Returns scores for each control dataset.

        Args:
            all_tokens: If True, returns scores for each prompt in the control dataset.
                        If False, returns aggregated scores for each dataset.

        Returns:
            A dict of (dataset name, scores)
        """
        if not self._merge_layers_enabled():
            raise ValueError(
                "merge_layers=False with multiple detect_layers. Use get_control_scores_by_layer()."
            )
        file_name = (
            "control_scores_all_tokens.json" if all_tokens else "control_scores.json"
        )
        path = self.cfg.folder / file_name

        if not path.exists():
            logger.info("Computing control scores")
            self.compute_control_scores()

        with open(path) as f:
            raw_scores = json.load(f)

        return {dataset: Scores.from_dict(scores) for dataset, scores in raw_scores.items()}

    def get_train_scores(self, all_tokens: bool = False) -> Scores:
        """
        Get the scores on the training dataset.
        If all_tokens is True, returns scores for every token.
        """
        if not self._merge_layers_enabled():
            raise ValueError(
                "merge_layers=False with multiple detect_layers. Use get_train_scores_by_layer()."
            )
        file_name = (
            "train_scores_all_tokens.json" if all_tokens else "train_scores.json"
        )
        path = self.cfg.folder / file_name

        if not path.exists():
            logger.info("Computing train scores")
            self.compute_train_scores()

        with open(path) as f:
            raw_scores = json.load(f)

        assert len(raw_scores["scores"]) == len(raw_scores["labels"]), (
            f"Mismatch between labels (len {len(raw_scores['labels'])}) and scores"
            f"(len {len(raw_scores['scores'])}) for train dataset"
        )
        return Scores.from_dict(raw_scores)

    def get_eval_scores(self, all_tokens: bool = False) -> dict[str, Scores]:
        """
        Get the scores on each dataset. Returns a dict of (dataset name, PairedScores).
        If all_tokens is True, returns scores for every token.
        """
        if not self._merge_layers_enabled():
            raise ValueError(
                "merge_layers=False with multiple detect_layers. Use get_eval_scores_by_layer()."
            )
        file_name = "scores_all_tokens.json" if all_tokens else "scores.json"
        path = self.cfg.folder / file_name
        if not path.exists():
            logger.info("Computing scores")
            self.compute_eval_scores()
        with open(path) as f:
            raw_scores = json.load(f)
        for name, scores in raw_scores.items():
            assert len(scores["scores"]) == len(scores["labels"]), (
                f"Mismatch between labels (len {len(scores['labels'])}) and scores"
                f"(len {len(scores['scores'])}) for dataset {name}"
            )
        return {name: Scores.from_dict(scores) for name, scores in raw_scores.items()}

    def get_control_scores_by_layer(self, all_tokens: bool = False) -> dict[int, dict[str, Scores]]:
        """
        Per-layer control scores.

        Returns: {layer: {dataset_name: Scores}}
        """
        file_name = (
            "control_scores_all_tokens_by_layer.json"
            if all_tokens
            else "control_scores_by_layer.json"
        )
        path = self.cfg.folder / file_name

        if not path.exists():
            logger.info("Computing control scores by layer")
            self.compute_control_scores()

        with open(path) as f:
            raw = json.load(f)

        out: dict[int, dict[str, Scores]] = {}
        for layer_str, dataset_map in raw.items():
            layer = int(layer_str)
            out[layer] = {
                dataset: Scores.from_dict(scores) for dataset, scores in dataset_map.items()
            }
        return out

    def get_eval_scores_by_layer(self, all_tokens: bool = False) -> dict[int, dict[str, Scores]]:
        """
        Per-layer eval scores.

        Returns: {layer: {dataset_name: Scores}}
        """
        file_name = (
            "scores_all_tokens_by_layer.json" if all_tokens else "scores_by_layer.json"
        )
        path = self.cfg.folder / file_name
        if not path.exists():
            logger.info("Computing eval scores by layer")
            self.compute_eval_scores()
        with open(path) as f:
            raw = json.load(f)
        out: dict[int, dict[str, Scores]] = {}
        for layer_str, dataset_map in raw.items():
            layer = int(layer_str)
            out[layer] = {
                dataset: Scores.from_dict(scores) for dataset, scores in dataset_map.items()
            }
        return out

    def get_train_scores_by_layer(self, all_tokens: bool = False) -> dict[int, Scores]:
        """
        Per-layer train scores.

        Returns: {layer: Scores}
        """
        file_name = (
            "train_scores_all_tokens_by_layer.json" if all_tokens else "train_scores_by_layer.json"
        )
        path = self.cfg.folder / file_name
        if not path.exists():
            logger.info("Computing train scores by layer")
            self.compute_train_scores()
        with open(path) as f:
            raw = json.load(f)
        return {int(layer_str): Scores.from_dict(scores) for layer_str, scores in raw.items()}

    def compute_control_scores(self) -> None:
        """Compute scores for control datasets"""
        self.compute_scores(is_control=True)

    def compute_eval_scores(self) -> None:
        """Compute scores for eval datasets"""
        self.compute_scores(is_control=False)

    def compute_train_scores(self) -> None:
        """Compute scores for training dataset"""
        logger.info("Computing train scores")

        if not self._merge_layers_enabled():
            self.compute_train_scores_by_layer()
            return
        self._reset_cache_stats("train")

        # Shard sizes for common training datasets (based on by_reg_coeff.py)
        shard_sizes = {
            "repe_honesty__plain": 20,
        }

        shard_size = (
            None if self.cfg.use_goodfire_sae_acts else shard_sizes.get(self.train_dataset.name, 12)
        )
        scores, scores_all_tokens = self.compute_scores_for_dataset(
            self.train_dataset, shard_size, stage="train"
        )

        # Add labels to scores
        assert len(self.train_dataset.labels) == len(scores) == len(scores_all_tokens), (
            f"Mismatch labels (len {len(self.train_dataset.labels)}) and scores (len {len(scores)})"
            f" and scores_all_tokens (len {len(scores_all_tokens)}) for train dataset"
        )
        scores.labels = self.train_dataset.labels
        scores_all_tokens.labels = self.train_dataset.labels

        # Save scores
        with open(self.cfg.folder / "train_scores.json", "w") as f:
            json.dump(scores, f, default=lambda scores: scores.to_dict())
            logger.info(f"Saved train_scores to {f.name}")

        with open(self.cfg.folder / "train_scores_all_tokens.json", "w") as f:
            json.dump(scores_all_tokens, f, default=lambda scores: scores.to_dict())
            logger.info(f"Saved train_scores_all_tokens to {f.name}")
        if self.cfg.inefficient_activation_caching:
            self._print_cache_stats("train")

    def compute_train_scores_by_layer(self) -> None:
        """Compute per-layer scores for training dataset."""
        logger.info("Computing train scores by layer")
        self._reset_cache_stats("train")

        shard_sizes = {
            "repe_honesty__plain": 20,
        }
        shard_size = (
            None if self.cfg.use_goodfire_sae_acts else shard_sizes.get(self.train_dataset.name, 12)
        )

        scores_by_layer, scores_all_tokens_by_layer = self.compute_scores_for_dataset_by_layer(
            self.train_dataset, shard_size, stage="train"
        )

        # Add labels and save
        out_scores: dict[int, Scores] = {}
        out_scores_all_tokens: dict[int, Scores] = {}
        for layer, scores in scores_by_layer.items():
            scores.labels = self.train_dataset.labels
            out_scores[layer] = scores
        for layer, scores in scores_all_tokens_by_layer.items():
            scores.labels = self.train_dataset.labels
            out_scores_all_tokens[layer] = scores

        with open(self.cfg.folder / "train_scores_by_layer.json", "w") as f:
            json.dump(out_scores, f, default=lambda scores: scores.to_dict())
            logger.info("Saved train_scores_by_layer to %s", f.name)
        with open(self.cfg.folder / "train_scores_all_tokens_by_layer.json", "w") as f:
            json.dump(out_scores_all_tokens, f, default=lambda scores: scores.to_dict())
            logger.info("Saved train_scores_all_tokens_by_layer to %s", f.name)
        if self.cfg.inefficient_activation_caching:
            self._print_cache_stats("train")

    def compute_scores(self, is_control: bool = False) -> None:
        """
        Compute the scores for datasets both on the detection mask and on all tokens and save
        them to file.

        If is_control is True, compute for control datasets, otherwise for eval datasets.
        """
        if not self._merge_layers_enabled():
            self.compute_scores_by_layer(is_control=is_control)
            return
        datasets = self.control_datasets if is_control else self.eval_datasets
        prefix = "control_" if is_control else ""
        stage = "control" if is_control else "eval"
        desc = "Control" if is_control else "Eval"
        self._compute_and_save_scores(datasets, prefix=prefix, stage=stage, desc=desc)

    def _compute_and_save_scores(
        self, datasets: list[DialogueDataset], prefix: str, stage: str, desc: str
    ) -> None:
        shard_sizes = EVAL_SHARD_SIZES

        all_scores = {}
        all_scores_all_tokens = {}
        self._reset_cache_stats(stage)
        for dataset in tqdm(datasets, desc=f"{desc} Datasets"):
            shard_size = (
                None if self.cfg.use_goodfire_sae_acts else shard_sizes.get(dataset.name, 12)
            )
            scores, scores_all_tokens = self.compute_scores_for_dataset(
                dataset, shard_size, stage=stage
            )
            all_scores[dataset.name] = scores
            all_scores_all_tokens[dataset.name] = scores_all_tokens
            # add is_deceptive to scores
            assert len(dataset.labels) == len(scores) == len(scores_all_tokens), (
                f"Mismatch labels (len {len(dataset.labels)}) and scores (len {len(scores)})"
                f" and scores_all_tokens (len {len(scores_all_tokens)}) for dataset {dataset.name}"
            )
            all_scores[dataset.name].labels = dataset.labels
            all_scores_all_tokens[dataset.name].labels = dataset.labels

        with open(self.cfg.folder / f"{prefix}scores.json", "w") as f:
            json.dump(all_scores, f, default=lambda scores: scores.to_dict())
            logger.info(f"Saved {prefix}scores to {f.name}")

        with open(self.cfg.folder / f"{prefix}scores_all_tokens.json", "w") as f:
            json.dump(all_scores_all_tokens, f, default=lambda scores: scores.to_dict())
            logger.info(f"Saved {prefix}scores_all_tokens to {f.name}")
        if self.cfg.inefficient_activation_caching:
            self._print_cache_stats(stage)

    def compute_scores_by_layer(self, is_control: bool = False) -> None:
        """
        Per-layer version of compute_scores. Saves:
        - scores_by_layer.json / scores_all_tokens_by_layer.json
        - control_scores_by_layer.json / control_scores_all_tokens_by_layer.json
        """
        datasets = self.control_datasets if is_control else self.eval_datasets
        prefix = "control_" if is_control else ""
        stage = "control" if is_control else "eval"
        desc = "Control" if is_control else "Eval"
        self._compute_and_save_scores_by_layer(datasets, prefix=prefix, stage=stage, desc=desc)

    def _compute_and_save_scores_by_layer(
        self, datasets: list[DialogueDataset], prefix: str, stage: str, desc: str
    ) -> None:
        layers = list(dict.fromkeys(self.cfg.detect_layers))
        self._reset_cache_stats(stage)

        # Ensure detectors exist
        detectors = self.get_detectors()
        for layer in layers:
            assert layer in detectors, f"Missing detector for layer {layer}"

        shard_sizes = EVAL_SHARD_SIZES

        all_scores: dict[int, dict[str, Scores]] = {layer: {} for layer in layers}
        all_scores_all_tokens: dict[int, dict[str, Scores]] = {layer: {} for layer in layers}

        for dataset in tqdm(datasets, desc=f"{desc} Datasets"):
            shard_size = (
                None if self.cfg.use_goodfire_sae_acts else shard_sizes.get(dataset.name, 12)
            )
            scores_by_layer, scores_all_tokens_by_layer = self.compute_scores_for_dataset_by_layer(
                dataset, shard_size, stage=stage, detectors=detectors
            )

            for layer in layers:
                scores = scores_by_layer[layer]
                scores_all = scores_all_tokens_by_layer[layer]
                assert len(dataset.labels) == len(scores) == len(scores_all), (
                    f"Mismatch labels (len {len(dataset.labels)}) and scores (len {len(scores)})"
                    f" and scores_all_tokens (len {len(scores_all)}) for dataset {dataset.name} "
                    f"layer {layer}"
                )
                scores.labels = dataset.labels
                scores_all.labels = dataset.labels
                all_scores[layer][dataset.name] = scores
                all_scores_all_tokens[layer][dataset.name] = scores_all

        with open(self.cfg.folder / f"{prefix}scores_by_layer.json", "w") as f:
            json.dump(all_scores, f, default=lambda scores: scores.to_dict())
            logger.info("Saved %sscores_by_layer to %s", prefix, f.name)

        with open(
            self.cfg.folder / f"{prefix}scores_all_tokens_by_layer.json", "w"
        ) as f:
            json.dump(all_scores_all_tokens, f, default=lambda scores: scores.to_dict())
            logger.info("Saved %sscores_all_tokens_by_layer to %s", prefix, f.name)
        if self.cfg.inefficient_activation_caching:
            self._print_cache_stats(stage)

    def compute_scores_for_dataset(
        self,
        dataset: DialogueDataset,
        shard_size: int | None,
        stage: str | None = None,
    ) -> tuple[Scores, Scores]:
        """
        Compute the scores for a dataset, return scores on the detection mask and on all tokens.

        A shard size of None means computing all acts in one shard
        """
        assert len(dataset) > 0, f"Dataset {dataset.name} is empty"
        detector = self.get_detector()

        def get_scores_for_shard(
            dataset: DialogueDataset,
        ) -> tuple[Scores, Scores]:
            toks = TokenizedDataset.from_dataset(
                dataset,
                self.tokenizer,
                detect_all=self.cfg.detect_all_tokens,
                detect_only_start_of_turn=self.cfg.detect_only_start_of_turn,
                detect_only_last_token=self.uses_last_content_token(),
                detect_only_eot=self.uses_ltlr_eot(dataset),
            )
            acts = self.get_acts(toks, stage=stage)
            scores = detector.score(acts)
            scores_all_tokens = detector.score(acts, all_acts=True)
            scores.labels = dataset.labels
            scores_all_tokens.labels = dataset.labels
            assert len(scores) == len(scores_all_tokens) == len(dataset) == len(toks), (
                f"Len mismatch: scores={len(scores)}, scores_all_tokens={len(scores_all_tokens)},"
                f"dataset={len(dataset)}, toks={len(toks)}"
            )
            return scores, scores_all_tokens

        all_scores: list[Scores] = []
        all_scores_all_tokens: list[Scores] = []
        shard_size = shard_size or len(dataset)
        for start_idx in trange(0, len(dataset), shard_size, desc=dataset.name):
            shard_dataset = dataset[start_idx : start_idx + shard_size]
            scores, scores_all_tokens = get_scores_for_shard(shard_dataset)
            all_scores.append(scores)
            all_scores_all_tokens.append(scores_all_tokens)

        scores = Scores.from_list(all_scores)
        scores_all_tokens = Scores.from_list(all_scores_all_tokens)
        assert len(scores) == len(scores_all_tokens) == len(dataset), (
            f"Len mismatch: scores={len(scores)}, scores_all_tokens={len(scores_all_tokens)}, "
            f"dataset={len(dataset)}"
        )
        return scores, scores_all_tokens

    def compute_scores_for_dataset_by_layer(
        self,
        dataset: DialogueDataset,
        shard_size: int | None,
        stage: str | None = None,
        detectors: dict[int, Detector] | None = None,
    ) -> tuple[dict[int, Scores], dict[int, Scores]]:
        """
        Compute per-layer scores for a dataset. Returns two dicts:
        - {layer: scores_on_detect_mask}
        - {layer: scores_on_all_tokens}
        """
        assert len(dataset) > 0, f"Dataset {dataset.name} is empty"
        layers = list(dict.fromkeys(self.cfg.detect_layers))

        if detectors is None:
            detectors = self.get_detectors()

        def get_scores_for_shard(
            shard_dataset: DialogueDataset,
        ) -> tuple[dict[int, Scores], dict[int, Scores]]:
            toks = TokenizedDataset.from_dataset(
                shard_dataset,
                self.tokenizer,
                detect_all=self.cfg.detect_all_tokens,
                detect_only_start_of_turn=self.cfg.detect_only_start_of_turn,
                detect_only_last_token=self.uses_last_content_token(),
                detect_only_eot=self.uses_ltlr_eot(shard_dataset),
            )
            out_scores: dict[int, Scores] = {}
            out_scores_all: dict[int, Scores] = {}
            acts_by_layer = self.get_acts_by_layer(toks, layers, stage=stage)
            for layer in layers:
                detector = detectors[layer]
                acts = acts_by_layer[layer]
                out_scores[layer] = detector.score(acts)
                out_scores_all[layer] = detector.score(acts, all_acts=True)
                out_scores[layer].labels = shard_dataset.labels
                out_scores_all[layer].labels = shard_dataset.labels

            for layer in layers:
                scores = out_scores[layer]
                scores_all_tokens = out_scores_all[layer]
                assert len(scores) == len(scores_all_tokens) == len(shard_dataset) == len(toks), (
                    f"Len mismatch: scores={len(scores)}, "
                    f"scores_all_tokens={len(scores_all_tokens)}, "
                    f"dataset={len(shard_dataset)}, toks={len(toks)} layer={layer}"
                )
            return out_scores, out_scores_all

        shard_size = shard_size or len(dataset)
        scores_accum: dict[int, list[Scores]] = {layer: [] for layer in layers}
        scores_all_accum: dict[int, list[Scores]] = {layer: [] for layer in layers}
        for start_idx in trange(0, len(dataset), shard_size, desc=dataset.name):
            shard_dataset = dataset[start_idx : start_idx + shard_size]
            shard_scores, shard_scores_all = get_scores_for_shard(shard_dataset)
            for layer in layers:
                scores_accum[layer].append(shard_scores[layer])
                scores_all_accum[layer].append(shard_scores_all[layer])

        merged_scores = {layer: Scores.from_list(scores_accum[layer]) for layer in layers}
        merged_scores_all = {layer: Scores.from_list(scores_all_accum[layer]) for layer in layers}
        for layer in layers:
            assert len(merged_scores[layer]) == len(merged_scores_all[layer]) == len(dataset), (
                f"Len mismatch: scores={len(merged_scores[layer])}, "
                f"scores_all_tokens={len(merged_scores_all[layer])}, "
                f"dataset={len(dataset)} layer={layer}"
            )
        return merged_scores, merged_scores_all

    def get_acts(self, toks: TokenizedDataset, stage: str | None = None) -> Activations:
        # In per-layer mode, callers should use get_acts_by_layer / get_acts_for_layers.
        return self.get_acts_for_layers(toks, self.cfg.detect_layers, stage=stage)

    def get_acts_for_layers(
        self, toks: TokenizedDataset, layers: list[int], stage: str | None = None
    ) -> Activations:
        """
        Get activations for a specific set of layers. Intended for merge_layers=True.
        """
        if self.cfg.use_goodfire_sae_acts:
            acts = Activations.from_goodfire(toks)
            if self.cfg.sae_latent_whitelist is not None:
                acts.apply_sae_latent_mask(self.cfg.sae_latent_whitelist)
            return acts

        layers = list(dict.fromkeys(layers))
        verbose = stage == "train"
        if not self.cfg.inefficient_activation_caching:
            acts = Activations.from_model(
                self.model,
                toks,
                batch_size=BATCH_SIZE_ACTS,
                layers=layers,
                verbose=verbose,
                # A fixed single-layer run uses this path rather than get_acts_by_layer.
                # Honor the same opt-in compaction so full-corpus last-token training
                # does not retain every context token in host memory.
                compact_single_token=(
                    os.getenv("DD_GPU_COMPACT_TRAIN", "0") == "1"
                    and stage == "train"
                    and not self.cfg.use_local_sae_acts
                    and base_probe_method(self.cfg.method)
                    in ("ltlr", "ltlr_ct", "ltmms", "ltmms_ct")
                ),
            )
            return self.to_local_sae_acts(acts)

        # This opt-in cache can be slower than recomputation on shared filesystems. Avoid it for
        # ordinary experiments; it exists only for explicit precompute workflows.
        cache_dir = get_cache_root()
        cache_stage = stage or "unspecified"
        stats = self._cache_stats_for(cache_stage)
        base_key = build_cache_base_key(self.cfg, toks, cache_stage)

        cached_by_layer = load_activation_blocks(cache_dir, base_key, layers)
        have = set(cached_by_layer.keys())
        missing = [layer for layer in layers if layer not in have]
        stats.hits += len(have)
        stats.misses += len(missing)

        if missing:
            computed = Activations.from_model(
                self.model,
                toks,
                batch_size=BATCH_SIZE_ACTS,
                layers=missing,
                verbose=verbose,
            )
            computed = self.to_local_sae_acts(computed)
            # Save blocks (only the layers we computed)
            saved = save_activation_blocks(cache_dir, base_key, missing, computed.all_acts)
            stats.created += len(saved)
            for i, layer in enumerate(missing):
                cached_by_layer[layer] = computed.all_acts[:, :, i : i + 1, :]

        # Assemble requested layers in order
        all_acts = torch.cat([cached_by_layer[layer] for layer in layers], dim=2)
        return Activations(all_acts=all_acts, tokenized_dataset=toks, logits=None, layers=layers)

    def _single_layer_from_multi(
        self, acts: Activations, layer_idx: int, layer_value: int
    ) -> Activations:
        return Activations(
            all_acts=acts.all_acts[:, :, layer_idx : layer_idx + 1, :],
            tokenized_dataset=acts.tokenized_dataset,
            logits=acts.logits,
            layers=[layer_value],
        )

    def get_acts_by_layer(
        self, toks: TokenizedDataset, layers: list[int] | None = None, stage: str | None = None
    ) -> dict[int, Activations]:
        """
        Get activations for each requested layer separately, computing layers in one forward pass
        unless inefficient activation caching was explicitly enabled.
        """
        if layers is None:
            layers = self.cfg.detect_layers
        layers = list(dict.fromkeys(layers))
        verbose = stage == "train"

        if self.cfg.use_goodfire_sae_acts:
            # Goodfire mode is validated elsewhere to use a single layer only.
            acts = Activations.from_goodfire(toks)
            if self.cfg.sae_latent_whitelist is not None:
                acts.apply_sae_latent_mask(self.cfg.sae_latent_whitelist)
            assert layers, "No layers provided"
            return {layers[0]: acts}

        if not self.cfg.inefficient_activation_caching:
            multi = Activations.from_model(
                self.model,
                toks,
                batch_size=BATCH_SIZE_ACTS,
                layers=layers,
                verbose=verbose,
                compact_single_token=(
                    os.getenv("DD_GPU_COMPACT_TRAIN", "0") == "1"
                    and stage == "train"
                    and not self.cfg.use_local_sae_acts
                    and base_probe_method(self.cfg.method)
                    in ("ltlr", "ltlr_ct", "ltmms", "ltmms_ct")
                ),
            )
            by_layer = {
                layer: self.to_local_sae_acts(self._single_layer_from_multi(multi, i, layer))
                for i, layer in enumerate(layers)
            }
            return by_layer

        # This opt-in cache can be slower than recomputation on shared filesystems. Avoid it for
        # ordinary experiments; it exists only for explicit precompute workflows.
        cache_dir = get_cache_root()
        cache_stage = stage or "unspecified"
        stats = self._cache_stats_for(cache_stage)
        base_key = build_cache_base_key(self.cfg, toks, cache_stage)

        cached_by_layer_tensor = load_activation_blocks(cache_dir, base_key, layers)
        have = set(cached_by_layer_tensor.keys())
        missing = [layer for layer in layers if layer not in have]
        stats.hits += len(have)
        stats.misses += len(missing)

        if missing:
            multi_missing = Activations.from_model(
                self.model,
                toks,
                batch_size=BATCH_SIZE_ACTS,
                layers=missing,
                verbose=verbose,
            )
            multi_missing = self.to_local_sae_acts(multi_missing)
            saved = save_activation_blocks(cache_dir, base_key, missing, multi_missing.all_acts)
            stats.created += len(saved)
            for i, layer in enumerate(missing):
                cached_by_layer_tensor[layer] = multi_missing.all_acts[:, :, i : i + 1, :]

        # Always show progress even on full hit (cheap now)
        if verbose:
            _ = tqdm(layers, desc=f"Cache coverage ({cache_stage})", leave=False)
            _.close()

        return {
            layer: Activations(
                all_acts=cached_by_layer_tensor[layer],
                tokenized_dataset=toks,
                logits=None,
                layers=[layer],
            )
            for layer in layers
        }

    def recompute_scores_on_dataset(self, dataset_name: str, shard_size: int = 12) -> None:
        """Recompute scores on a dataset (or add new dataset)"""
        dataset = self.get_dataset(dataset_name, model=self.eval_policy_model)
        if self._merge_layers_enabled():
            scores = self.get_eval_scores(all_tokens=False)
            scores_all_tokens = self.get_eval_scores(all_tokens=True)

            dataset_scores, dataset_scores_all_tokens = self.compute_scores_for_dataset(
                dataset, shard_size=shard_size, stage="eval"
            )
            scores[dataset_name] = dataset_scores
            scores_all_tokens[dataset_name] = dataset_scores_all_tokens
            scores[dataset.name].labels = dataset.labels
            scores_all_tokens[dataset.name].labels = dataset.labels

            with open(self.cfg.folder / "scores.json", "w") as f:
                json.dump(scores, f, default=lambda scores: scores.to_dict())
            with open(self.cfg.folder / "scores_all_tokens.json", "w") as f:
                json.dump(scores_all_tokens, f, default=lambda scores: scores.to_dict())
        else:
            scores_by_layer = self.get_eval_scores_by_layer(all_tokens=False)
            scores_all_tokens_by_layer = self.get_eval_scores_by_layer(all_tokens=True)
            dataset_scores_by_layer, dataset_scores_all_tokens_by_layer = (
                self.compute_scores_for_dataset_by_layer(
                    dataset, shard_size=shard_size, stage="eval"
                )
            )

            for layer, ds_scores in dataset_scores_by_layer.items():
                ds_scores.labels = dataset.labels
                scores_by_layer.setdefault(layer, {})[dataset_name] = ds_scores
            for layer, ds_scores_all in dataset_scores_all_tokens_by_layer.items():
                ds_scores_all.labels = dataset.labels
                scores_all_tokens_by_layer.setdefault(layer, {})[dataset_name] = ds_scores_all

            with open(self.cfg.folder / "scores_by_layer.json", "w") as f:
                json.dump(scores_by_layer, f, default=lambda scores: scores.to_dict())
            with open(self.cfg.folder / "scores_all_tokens_by_layer.json", "w") as f:
                json.dump(scores_all_tokens_by_layer, f, default=lambda scores: scores.to_dict())

        # update the config
        if dataset_name not in self.cfg.eval_data:
            self.cfg.eval_data.append(dataset_name)
            self.cfg._save_config()  # type: ignore

    # ----- BLACK BOX BASELINE ----- #

    def get_black_box_scores(self, mock: bool = False) -> dict[str, Scores]:
        raise NotImplementedError('API baselines are outside the main-table reproduction.')

    def get_black_box_control_scores(self) -> dict[str, Scores]:
        raise NotImplementedError('API baselines are outside the main-table reproduction.')

    def get_black_box_reasonings(self, dataset_name: str) -> list[str]:
        scores_file = self.cfg.folder / "black_box_reasonings.json"
        with open(scores_file) as f:
            data = json.load(f)
            return data[dataset_name]

    def get_black_box_control_reasonings(self, dataset_name: str = "alpaca__plain") -> list[str]:
        scores_file = self.cfg.folder / "black_box_control_reasonings.json"
        with open(scores_file) as f:
            data = json.load(f)
            return data[dataset_name]

    # ----- OTHER ----- #

    def __str__(self) -> str:
        return f"Experiment({self.cfg.name})"

    def to_local_sae_acts(self, acts: Activations) -> Activations:
        if self.cfg.use_local_sae_acts:
            return get_sae_acts(acts, self.cfg.sae_latent_whitelist)
        return acts

    @property
    def model(self) -> PreTrainedModel:
        """Lazily load the model when first accessed."""
        if self._model is None:
            # Safety net for the cache-only assembly step: if DD_FORBID_MODEL_LOAD=1, loading the
            # model here means the data-parallel activation pre-pass did NOT fully populate the
            # cache. Fail loudly instead of silently spending ~2 min loading a 70B on one GPU.
            if os.getenv("DD_FORBID_MODEL_LOAD", "0").strip() == "1":
                raise RuntimeError(
                    "Model load requested while DD_FORBID_MODEL_LOAD=1: the activation cache is "
                    "incomplete for this config (a shard was missed by the pre-pass). Re-run the "
                    "precompute workers, or unset DD_FORBID_MODEL_LOAD to allow on-demand compute."
                )
            cut_at_layer = cut_at_layer_for_detect_layers(self.cfg.detect_layers)
            model, _ = get_model_and_tokenizer(
                self.cfg.model_name,
                pw_locked=self.cfg.pw_locked,
                cut_at_layer=cut_at_layer,
                lora_path=self.cfg.lora_path,
            )
            self._model = model
        return self._model
