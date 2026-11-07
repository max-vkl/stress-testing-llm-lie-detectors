import os
from enum import Enum
from functools import cache
from pathlib import Path
from typing import Literal, cast, overload

import torch
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from deception_detection.log import logger

# We only use the instruct variations of the models. Note not all sizes are available for all model
# families.
ModelType = Literal["gemma", "mistral", "llama", "qwen"]
ModelSize = Literal[1, 2, 3, 7, 8, 9, 24, 27, 31, 70, 72]


class ModelName(str, Enum):
    MISTRAL_3_1_24B = "mistral-small-3.1-24b"
    LLAMA_70B_3_3 = "llama-70b-3.3"
    LLAMA_70B_3_1 = "llama-70b-3.1"
    QWEN_72B = "qwen-72b"
    GEMMA_2_9B_IT = "gemma-2-9b-it"
    GEMMA_3_27B_IT = "gemma-3-27b-it"
    GEMMA_4_31B_IT = "gemma-4-31b-it"

    @property
    def type(self) -> ModelType:
        return cast(ModelType, self.value.split("-")[0])

    @property
    def size(self) -> ModelSize:
        if self == ModelName.MISTRAL_3_1_24B:
            return cast(ModelSize, 24)
        elif self == ModelName.QWEN_72B:
            return cast(ModelSize, 72)
        else:
            # Most model names are like "<type>-<size>b-...", e.g. llama-70b-3.3.
            # Gemma 3 is like "gemma-3-27b-it" where the size is the third field.
            parts = self.value.split("-")
            if self.type == "gemma":
                size_str = parts[2].replace("b", "")
            else:
                size_str = parts[1].replace("b", "")
            return cast(ModelSize, int(size_str))

    @property
    def is_llama_3_3(self) -> bool:
        return self == ModelName.LLAMA_70B_3_3

    @property
    def is_llama_3_1(self) -> bool:
        return self == ModelName.LLAMA_70B_3_1

    @property
    def is_mistral(self) -> bool:
        return self == ModelName.MISTRAL_3_1_24B

    @property
    def is_qwen(self) -> bool:
        return self == ModelName.QWEN_72B

def get_model_name(model_name: ModelName) -> str:
    return model_name.value


# ----- Runtime load knobs (env-driven, backward-compatible defaults) ----- #
# These control the numerics/behaviour of the forward pass. DD_DTYPE and DD_ATTN_IMPL must be
# kept in sync with deception_detection/activation_cache.py, which folds them into the activation
# cache key so bf16 and fp16 activations never collide.


def resolve_torch_dtype() -> torch.dtype:
    """Model dtype from DD_DTYPE (default bf16)."""
    name = os.getenv("DD_DTYPE", "bf16").strip().lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported DD_DTYPE={name!r}; use bf16, fp16, or fp32")


def resolve_attn_implementation() -> str:
    """Attention impl from DD_ATTN_IMPL (default sdpa; use eager/flash_attention_2 to override)."""
    return os.getenv("DD_ATTN_IMPL", "sdpa").strip().lower()


def resolve_local_files_only() -> bool:
    """Whether from_pretrained is restricted to local files (default True, preserving old behaviour).

    Set DD_LOCAL_FILES_ONLY=0 to allow downloading weights on a fresh machine.
    """
    return os.getenv("DD_LOCAL_FILES_ONLY", "1").strip() == "1"


def get_num_layers_for_model(model_name: ModelName) -> int:
    if model_name.type == "llama":
        n_layers = {70: 80}
    elif model_name.type == "mistral":
        n_layers = {24: 56}  # Mistral Small has 56 layers
    elif model_name.type == "qwen":
        n_layers = {72: 80}
    elif model_name.type == "gemma":
        # Avoid hardcoding Gemma sizes; read from config. Gemma 3/4 are conditional-generation
        # (multimodal) configs where num_hidden_layers lives under `text_config`, so fall back to it.
        # Note: Gemma models on HF can be gated and may require authentication + license acceptance.
        base_model_path = {
            ModelName.GEMMA_2_9B_IT: "google/gemma-2-9b-it",
            ModelName.GEMMA_3_27B_IT: "google/gemma-3-27b-it",
            ModelName.GEMMA_4_31B_IT: "google/gemma-4-31B-it",
        }[model_name]
        cfg = AutoConfig.from_pretrained(base_model_path)
        n_layers_cfg = getattr(cfg, "num_hidden_layers", None)
        if n_layers_cfg is None and hasattr(cfg, "text_config"):
            n_layers_cfg = getattr(cfg.text_config, "num_hidden_layers", None)
        if n_layers_cfg is None:
            raise ValueError(f"Config for {base_model_path} missing num_hidden_layers")
        return int(n_layers_cfg)
    else:
        raise NotImplementedError(f"No number of layers for model {model_name}")
    return n_layers[model_name.size]


@overload
def get_model_and_tokenizer(
    model_name: ModelName,
    omit_model: Literal[True],
    pw_locked: bool = False,
    lora_path: str | Path | None = None,
    cut_at_layer: int | None = None,
) -> tuple[None, PreTrainedTokenizerBase]: ...


@overload
def get_model_and_tokenizer(
    model_name: ModelName,
    omit_model: Literal[False] = False,
    pw_locked: bool = False,
    lora_path: str | Path | None = None,
    cut_at_layer: int | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]: ...


@overload
def get_model_and_tokenizer(
    model_name: ModelName,
    omit_model: bool = False,
    pw_locked: bool = False,
    lora_path: str | Path | None = None,
    cut_at_layer: int | None = None,
) -> tuple[PreTrainedModel | None, PreTrainedTokenizerBase]: ...


def get_model_and_tokenizer(
    model_name: ModelName,
    omit_model: bool = False,
    pw_locked: bool = False,
    lora_path: str | Path | None = None,
    cut_at_layer: int | None = None,
) -> tuple[PreTrainedModel | None, PreTrainedTokenizerBase]:
    """
    Load the model and tokenizer based on the given parameters.
    """
    model, tokenizer = model_and_tokenizer(model_name, omit_model, lora_path)

    if cut_at_layer is not None and model is not None:
        # Keeps decoder blocks [0, cut_at_layer). The model's final norm is a separate module and is
        # NOT removed, so transformers still applies it to the terminal hidden_states entry. A
        # caller that reads hidden_states[L] for a raw (pre-final-norm) residual must therefore pass
        # cut_at_layer = L + 1, so block L stays interior rather than becoming the normed terminal.
        if hasattr(model.model, "layers"):                  # Llama-2/3, Gemma
            model.model.layers = model.model.layers[:cut_at_layer]
        elif hasattr(model.model, "language_model"):          # Mistral-3 family
            model.model.language_model.layers = model.model.language_model.layers[:cut_at_layer]
        else:
            raise ValueError(f"Model {model_name} has no layers to cut at layer {cut_at_layer}")

    return model, tokenizer


@cache
def get_llama3_tokenizer(tokenizer_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct") -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
    tokenizer.pad_token_id = tokenizer.bos_token_id
    return tokenizer


def model_and_tokenizer(
    model_name: ModelName, omit_model: bool = False, lora_path: str | Path | None = None
) -> tuple[PreTrainedModel | None, PreTrainedTokenizerBase]:
    base_model_path = {
        ModelName.GEMMA_2_9B_IT: "google/gemma-2-9b-it",
        ModelName.LLAMA_70B_3_3: "meta-llama/Llama-3.3-70B-Instruct",
        ModelName.LLAMA_70B_3_1: "meta-llama/Meta-Llama-3.1-70B-Instruct",
        ModelName.MISTRAL_3_1_24B: "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        ModelName.QWEN_72B: "Qwen/Qwen2.5-72B-Instruct",
        ModelName.GEMMA_3_27B_IT: "google/gemma-3-27b-it",
        ModelName.GEMMA_4_31B_IT: "google/gemma-4-31B-it",
    }[model_name]

    if model_name == ModelName.LLAMA_70B_3_3:
        from huggingface_hub import snapshot_download
        base_model_path = snapshot_download(
            "meta-llama/Llama-3.3-70B-Instruct",
            revision="6f6073b423013f6a7d4d9f39144961bfbfbc386b",
            local_files_only=resolve_local_files_only(),
        )

    # Determine final model path to load from
    if lora_path is not None:
        # Create merged model path based on model name and LoRA path
        lora_name = Path(lora_path).name
        merged_model_path = Path("models") / "peft" / f"{model_name.value}_merged_{lora_name}"

        # Check if merged model already exists
        if merged_model_path.exists() and (merged_model_path / "config.json").exists():
            logger.info(f">>> Found existing merged model at: {merged_model_path}")
            print(f">>> Found existing merged model at: {merged_model_path}")
            final_model_path = str(merged_model_path)
        elif omit_model:
            # When we only need the tokenizer, avoid an expensive LoRA merge and model load.
            # Tokenization is identical to the base model in this setup.
            final_model_path = base_model_path
        else:
            logger.info(f">>> Merging LoRA adapter: {lora_path}")
            print(f">>> Merging LoRA adapter: {lora_path}")

            # Load base model first
            logger.info(f"Loading base model for LoRA merge: {base_model_path}")
            base_processor = None  # Initialize for type safety

            if model_name != ModelName.GEMMA_2_9B_IT and (
                "mistral" in base_model_path.lower() or "gemma" in base_model_path.lower()
            ):
                base_model = AutoModelForImageTextToText.from_pretrained(
                    base_model_path,
                    torch_dtype=resolve_torch_dtype(),
                    device_map="auto",
                    attn_implementation=resolve_attn_implementation(),
                    local_files_only=resolve_local_files_only(),
                )
                base_processor = AutoProcessor.from_pretrained(base_model_path, padding_side="left")
                base_processor.tokenizer.chat_template = base_processor.chat_template
                base_tokenizer = base_processor.tokenizer
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=resolve_torch_dtype(),
                    device_map="auto",
                    attn_implementation=resolve_attn_implementation(),
                    local_files_only=resolve_local_files_only(),
                )
                base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side="left")

            if "qwen" in base_model_path.lower():
                base_tokenizer.bos_token_id = base_tokenizer.eos_token_id

            base_tokenizer.pad_token_id = base_tokenizer.bos_token_id

            # Now create PEFT model from loaded base model
            print("merging...")
            # Clear cache before merging
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            peft_model = PeftModel.from_pretrained(base_model, lora_path)  # type: ignore
            merged_model = peft_model.merge_and_unload()  # type: ignore

            # Clear intermediate model from memory
            del peft_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("done merging.")

            # Save merged model to disk
            print("saving merged model...")
            merged_model_path.mkdir(parents=True, exist_ok=True)

            # Move model to CPU before saving to avoid CUDA memory issues
            print("Moving model to CPU for saving...")
            merged_model = merged_model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            merged_model.save_pretrained(merged_model_path)

            # For image-text-to-text models, save the processor (which includes tokenizer)
            if (
                ("mistral" in base_model_path.lower() or "gemma" in base_model_path.lower())
                and base_processor is not None
            ):
                base_processor.save_pretrained(merged_model_path)
            else:
                base_tokenizer.save_pretrained(merged_model_path)
            print("done saving merged model")

            # Free memory after saving
            del merged_model
            del base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Additional cleanup handled above

            logger.info(f">>> Saved merged model to: {merged_model_path}")
            print(f">>> Saved merged model to: {merged_model_path}")
            final_model_path = str(merged_model_path)
    else:
        final_model_path = base_model_path

    # Fast path: tokenizer only (skip model load entirely)
    if omit_model:
        if model_name != ModelName.GEMMA_2_9B_IT and (
            "mistral" in final_model_path.lower()
            or "mistral" in base_model_path.lower()
            or "gemma" in final_model_path.lower()
            or "gemma" in base_model_path.lower()
        ):
            processor = AutoProcessor.from_pretrained(final_model_path, padding_side="left")
            processor.tokenizer.chat_template = processor.chat_template
            tokenizer = processor.tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained(final_model_path, padding_side="left")

        if "qwen" in final_model_path.lower():
            tokenizer.bos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.bos_token_id
        return None, tokenizer

    # Clean up any existing GPU memory before loading new model
    if torch.cuda.is_available():
        cleanup_model_memory()

    # Load the final model (either base model or merged model)
    if model_name != ModelName.GEMMA_2_9B_IT and (
        "mistral" in final_model_path.lower()
        or "mistral" in base_model_path.lower()
        or "gemma" in final_model_path.lower()
        or "gemma" in base_model_path.lower()
    ):
        model = AutoModelForImageTextToText.from_pretrained(
            final_model_path,
            torch_dtype=resolve_torch_dtype(),
            device_map="auto",
            attn_implementation=resolve_attn_implementation(),
            local_files_only=resolve_local_files_only(),
        )
        processor = AutoProcessor.from_pretrained(final_model_path, padding_side="left")
        processor.tokenizer.chat_template = processor.chat_template
        tokenizer = processor.tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(
            final_model_path,
            torch_dtype=resolve_torch_dtype(),
            device_map="auto",
            attn_implementation=resolve_attn_implementation(),
            local_files_only=resolve_local_files_only(),
        )
        tokenizer = AutoTokenizer.from_pretrained(final_model_path, padding_side="left")

    if "qwen" in final_model_path.lower():
        tokenizer.bos_token_id = tokenizer.eos_token_id

    tokenizer.pad_token_id = tokenizer.bos_token_id

    logger.info(f"Loaded model {model_name} from {final_model_path}")

    if model.device == torch.device("cpu"):
        logger.warning("Model is on CPU")

    model.eval()

    return model, tokenizer


def cleanup_model_memory():
    """Clean up GPU memory by clearing cache and garbage collection."""
    import gc
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
