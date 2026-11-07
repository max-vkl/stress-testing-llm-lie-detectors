"""
Creates and handles inputs to white box deception detectors. Run's the model on tokenized input data
and returns activations at the detection positions.
"""

import os
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import trange
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from deception_detection.data.base import DialogueDataset
from deception_detection.log import logger
from deception_detection.tokenized_data import TokenizedDataset, get_tokenized_dataset_pair
from deception_detection.utils import REPO_ROOT, mask_sparse_activations

# Consolidate the per-layer device->host activation copy into a single large transfer. On by
# default (numerically identical, strictly better bandwidth). Set DD_CONSOLIDATE_D2H=0 to restore
# the old per-layer .cpu() path (used only for A/B benchmarking). NOT part of the cache key: the
# stored values are byte-identical either way, so cached activations remain interchangeable.
_CONSOLIDATE_D2H = os.getenv("DD_CONSOLIDATE_D2H", "1").strip() == "1"


def _single_token_metadata(td: TokenizedDataset) -> TokenizedDataset:
    """Keep display metadata while selecting one tensor position per dialogue."""
    mask = td.detection_mask
    assert mask is not None and bool((mask.sum(dim=1) == 1).all())
    rows = torch.arange(len(td), device=td.tokens.device)
    positions = mask.long().argmax(dim=1)
    return replace(
        td,
        tokens=td.tokens[rows, positions].unsqueeze(1),
        attention_mask=td.attention_mask[rows, positions].unsqueeze(1),
        detection_mask=mask[rows, positions].unsqueeze(1),
    )


@dataclass
class Activations:
    """
    Responsible for running the model and grabbing activations on the "detect" tokens.

    Goodfire activations are sometimes represented sparsely to save memory.
    """

    all_acts: Float[Tensor, "batch seqpos layer emb"]
    tokenized_dataset: TokenizedDataset
    logits: None | Float[Tensor, "batch seqpos vocab"] = None
    layers: list[int] | None = None

    @classmethod
    @torch.inference_mode()
    def from_model(
        cls,
        model: PreTrainedModel,
        tokenized_dataset: TokenizedDataset,
        save_logits: bool = False,
        batch_size: int = 20,
        layers: list[int] | None = None,
        verbose: bool = False,
        compact_single_token: bool = False,
    ) -> "Activations":
        """Run full-context forwards, optionally copying only sole readouts to CPU.

        Compaction is opt-in and falls back to full sequences for non-single masks
        or requested logits. It changes storage and transfer only, never model inputs.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mask = tokenized_dataset.detection_mask
        compact_single_token = (
            compact_single_token
            and not save_logits
            and len(tokenized_dataset) > 0
            and mask is not None
            and bool((mask.sum(dim=1) == 1).all())
        )

        activations_by_batch: list[Float[Tensor, "batch seqpos layer emb"]] = []
        logits_by_batch: list[Float[Tensor, "batch seqpos vocab"]] = []

        for i in trange(
            0, len(tokenized_dataset), batch_size, desc="Computing activations", disable=not verbose
        ):
            batch: TokenizedDataset = tokenized_dataset[i : i + batch_size]

            # Enable mixed precision for modern large models.
            # IMPORTANT: pass the resolved dtype explicitly. A dtype-less autocast('cuda') defaults
            # to float16, which silently overrides bf16/fp32 weights for every matmul/attention op.
            # For Gemma-3, whose activations reach ~2.5e5 (>> fp16's 65504 max), that overflows to
            # inf -> NaN and cascades through the residual stream. Use DD_DTYPE (bf16 by default).
            from deception_detection.models import resolve_torch_dtype
            model_supports_amp = any(name in model.name_or_path.lower()
                                   for name in ["llama", "mistral", "gemma", "qwen"])
            with torch.amp.autocast('cuda', enabled=model_supports_amp, dtype=resolve_torch_dtype()):
                fwd_out = model(
                    batch.tokens.to(device),
                    output_hidden_states=True,
                    attention_mask=batch.attention_mask.to(device),
                    # only compute logits on last token if we don't care about logits.
                    # NB: transformers>=4.49 renamed `num_logits_to_keep` -> `logits_to_keep`;
                    # the old name is silently swallowed into **kwargs (computing ALL logits every
                    # forward), so we use the current name to actually save that memory.
                    logits_to_keep=0 if save_logits else 1,
                    use_cache=False,
                )

                layers_to_process = list(
                    layers if layers is not None else range(len(fwd_out.hidden_states))
                )
                if compact_single_token:
                    assert batch.detection_mask is not None
                    positions = batch.detection_mask.long().argmax(dim=1).to(device)
                    rows = torch.arange(len(batch), device=device)
                    # Index after the unchanged full-context forward, before stacking
                    # or transferring. This avoids copying hundreds of unused tokens.
                    selected = [
                        fwd_out.hidden_states[layer_idx][rows, positions].unsqueeze(1)
                        for layer_idx in layers_to_process
                    ]
                    batch_acts = torch.stack(selected, dim=2).cpu()
                    del selected
                elif _CONSOLIDATE_D2H:
                    # Stack the requested layers on-GPU, then do ONE large device->host copy
                    # instead of a per-layer .cpu() (which is 80 tiny, individually-synchronizing
                    # transfers). This is numerically IDENTICAL — torch.stack is pure rearrangement,
                    # the copy preserves values — but it uses PCIe/host-memory bandwidth far better,
                    # which is the bottleneck for this all-layers/all-tokens extraction.
                    stacked_gpu = torch.stack(
                        [fwd_out.hidden_states[layer_idx] for layer_idx in layers_to_process],
                        dim=2,
                    )
                    batch_acts = stacked_gpu.to("cpu")  # [batch, seqpos, layer, emb]
                    del stacked_gpu
                else:
                    all_activations: list[Float[Tensor, "batch seqpos emb"]] = [
                        fwd_out.hidden_states[layer_idx].cpu() for layer_idx in layers_to_process
                    ]
                    batch_acts = torch.stack(all_activations, dim=2)

                if save_logits:
                    logits_by_batch.append(fwd_out.logits.cpu())

                del fwd_out

            activations_by_batch.append(batch_acts)

        all_acts = torch.cat(activations_by_batch)
        logits = torch.cat(logits_by_batch) if save_logits else None

        if compact_single_token:
            tokenized_dataset = _single_token_metadata(tokenized_dataset)
        return cls(all_acts, tokenized_dataset, logits, layers)

    def compact_single_detection_token(self) -> "Activations":
        """Copy the sole detected token per dialogue into independent, compact storage.

        Intended for training assembly after the full-context model forward pass. Values,
        dialogue order, and per-dialogue fit weights are unchanged. Advanced indexing is
        essential: a slice/view would keep the entire full-sequence backing tensor alive.
        Unreduced or missing masks and sparse activations retain their existing representation.
        """
        td = self.tokenized_dataset
        mask = td.detection_mask
        if (
            self.all_acts.is_sparse
            or mask is None
            or len(td) == 0
            or not bool((mask.sum(dim=1) == 1).all())
        ):
            return self
        rows = torch.arange(len(td), device=td.tokens.device)
        positions = mask.long().argmax(dim=1)
        return Activations(
            all_acts=self.all_acts[rows, positions].unsqueeze(1),
            tokenized_dataset=_single_token_metadata(td),
            logits=None if self.logits is None else self.logits[rows, positions].unsqueeze(1),
            layers=self.layers,
        )

    @classmethod
    def concat_padded(cls, acts_list: list["Activations"]) -> "Activations":
        """Concatenate several Activations (e.g. per-shard) along the batch dim.

        Each input may have a different sequence length (independent left-padding per shard).
        We left-pad every shard's activations/tokens/masks to the common max sequence length before
        concatenating. Left-padding is inert for detectors, which read only masked / last-token
        positions (padded positions carry attention_mask=0 and detection_mask=False), so the
        combined object yields identical fit/score results to computing over one monolithic batch.

        All inputs must share the same layer list and embedding dim, and be dense (not sparse).
        """
        parts = [a for a in acts_list if a is not None and a.all_acts.shape[0] > 0]
        if not parts:
            raise ValueError("concat_padded received no non-empty Activations")

        layers = parts[0].layers
        n_layer = parts[0].all_acts.shape[2]
        emb = parts[0].all_acts.shape[3]
        for a in parts:
            assert not a.all_acts.is_sparse, "concat_padded does not support sparse activations"
            assert a.layers == layers, f"layer mismatch: {a.layers} != {layers}"
            assert a.all_acts.shape[2] == n_layer and a.all_acts.shape[3] == emb, "shape mismatch"

        max_seq = max(a.all_acts.shape[1] for a in parts)

        def _lpad_2d(t: Tensor, pad: int, value: float) -> Tensor:
            if pad == 0:
                return t
            block = torch.full(
                (t.shape[0], pad), value, dtype=t.dtype, device=t.device
            )
            return torch.cat([block, t], dim=1)

        acts_out: list[Tensor] = []
        tokens_out: list[Tensor] = []
        attn_out: list[Tensor] = []
        detect_out: list[Tensor | None] = []
        dialogues: list = []
        formatted: list = []
        str_tokens: list = []
        for a in parts:
            td = a.tokenized_dataset
            pad = max_seq - a.all_acts.shape[1]
            aa = a.all_acts
            if pad > 0:
                # pad dims order (last first): emb, layer, seqpos-left, seqpos-right
                aa = torch.nn.functional.pad(aa, (0, 0, 0, 0, pad, 0))
            acts_out.append(aa)
            tokens_out.append(_lpad_2d(td.tokens, pad, 0))
            attn_out.append(_lpad_2d(td.attention_mask, pad, 0))
            if td.detection_mask is None:
                detect_out.append(None)
            else:
                dm = td.detection_mask
                if pad > 0:
                    block = torch.zeros((dm.shape[0], pad), dtype=torch.bool)
                    dm = torch.cat([block, dm], dim=1)
                detect_out.append(dm)
            dialogues.extend(td.dialogues)
            formatted.extend(td.formatted_dialogues)
            str_tokens.extend(td.str_tokens)

        detection_mask = (
            None if any(d is None for d in detect_out) else torch.cat(detect_out, dim=0)  # type: ignore[arg-type]
        )
        combined_toks = TokenizedDataset(
            dialogues=dialogues,
            formatted_dialogues=formatted,
            tokens=torch.cat(tokens_out, dim=0),
            str_tokens=str_tokens,
            detection_mask=detection_mask,
            attention_mask=torch.cat(attn_out, dim=0),
            dataset_partial_id=parts[0].tokenized_dataset.dataset_partial_id,
        )
        return cls(
            all_acts=torch.cat(acts_out, dim=0),
            tokenized_dataset=combined_toks,
            logits=None,
            layers=layers,
        )

    @classmethod
    def from_goodfire(cls, tokenized_dataset: TokenizedDataset) -> "Activations":
        raise NotImplementedError('Hosted SAE services are outside the main-table reproduction.')

    def get_masked_activations(self) -> Float[Tensor, "toks layer emb"]:
        """Get activations at the detection positions.

        This flattens the batch and seq dimensions into a single 'toks' dimension."""
        assert self.tokenized_dataset.detection_mask is not None
        if self.all_acts.is_sparse:
            return mask_sparse_activations(self.all_acts, self.tokenized_dataset.detection_mask)
        else:
            return self.all_acts[self.tokenized_dataset.detection_mask]

    def get_all_activations(self) -> Float[Tensor, "toks layer emb"]:
        """Get all activations.

        This flattens the batch and seq dimensions into a single 'toks' dimension."""
        assert self.all_acts is not None
        # converting to bool is important for indexing reasons
        if self.all_acts.is_sparse:
            return mask_sparse_activations(
                self.all_acts, self.tokenized_dataset.attention_mask.bool()
            )
        else:
            return self.all_acts[self.tokenized_dataset.attention_mask.bool()]

    def get_masked_logits(self) -> Float[Tensor, "toks vocab"]:
        """Get logits at the detection positions.

        This flattens the batch and seq dimensions into a single 'toks' dimension."""
        assert self.logits is not None
        return self.logits[self.tokenized_dataset.detection_mask]

    def save(self, file_path: str | Path, only_at_mask: bool = False, sparse: bool = False) -> None:
        """Save the Activations object to a file.

        If only_at_mask is True, only the activations at the detection positions are saved. The
        activations at the other positions will be loaded as NaNs.
        """
        logits_to_save = (
            self.get_masked_logits() if (only_at_mask and self.logits is not None) else self.logits
        )

        if self.all_acts.is_sparse:
            if only_at_mask:
                raise NotImplementedError("Sparse activations not supported for only_at_mask")

            coalesced_acts = self.all_acts.coalesce()
            acts_to_save = {
                "indices": coalesced_acts.indices().T,
                "values": coalesced_acts.values(),
                "shape": coalesced_acts.shape,
                "sparse": True,
            }
        else:
            acts_to_save = self.get_masked_activations() if only_at_mask else self.all_acts

        with open(file_path, "wb") as f:
            pickle.dump(
                {
                    "acts": acts_to_save,
                    "logits": logits_to_save,
                    "tokenized_dataset": self.tokenized_dataset,
                    "only_at_mask": only_at_mask,
                    "layers": self.layers,
                },
                f,
            )

    @classmethod
    def load(cls, file_path: str | Path) -> "Activations":
        """Load an Activations object from a file."""
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        activations = cls.__new__(cls)
        activations.tokenized_dataset = data["tokenized_dataset"]
        activations.layers = data["layers"]

        if data["only_at_mask"]:
            mask = activations.tokenized_dataset.detection_mask
            assert mask
            if isinstance(data["acts"], dict) and data["acts"].get("sparse"):
                raise NotImplementedError("Sparse activations not supported for only_at_mask")

            else:
                act_shape = mask.shape + data["acts"].shape[-2:]  # [batch, seqpos] + [layer, emb]
                activations.all_acts = torch.full(act_shape, float("nan"), dtype=data["acts"].dtype)
                activations.all_acts[mask] = data["acts"]

            if data["logits"] is not None:
                logits_shape = mask.shape + (data["logits"].shape[-1],)
                activations.logits = torch.full(logits_shape, float("nan"))
                activations.logits[mask] = data["logits"]
            else:
                activations.logits = None
        else:
            if isinstance(data["acts"], dict) and data["acts"].get("sparse"):
                # Handle sparse format
                activations.all_acts = torch.sparse_coo_tensor(
                    indices=data["acts"]["indices"].long().t(),  # Convert to long and transpose
                    values=data["acts"]["values"],
                    size=data["acts"]["shape"],
                )
            else:
                activations.all_acts = data["acts"]
            activations.logits = data["logits"]

        return activations

    def __getitem__(self, idx: int | slice) -> "Activations":
        """
        Implement slicing functionality for the Activations class.
        This allows users to slice the dataset along the batch dimension.
        """
        sliced_activations = Activations.__new__(Activations)
        sliced_activations.all_acts = self.all_acts[idx]
        sliced_activations.logits = self.logits[idx] if self.logits is not None else None
        sliced_activations.tokenized_dataset = self.tokenized_dataset[idx]
        sliced_activations.layers = self.layers
        return sliced_activations

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.tokenized_dataset)

    def to(self, dtype: torch.dtype | None = None, device: torch.device | str | None = None):
        self.all_acts = self.all_acts.to(dtype=dtype, device=device)
        if self.logits is not None:
            self.logits = self.logits.to(dtype=dtype, device=device)

    def to_dense(self):
        """Converts sparse activations to dense in place."""
        if self.all_acts.is_sparse:
            self.all_acts = self.all_acts.to_dense()

    def apply_sae_latent_mask(self, whitelist: list[int]):
        """Zero out all activations except for the whitelisted latents."""
        # activations are a sparse tensor with shape [batch, seq, layer, feature]
        # mask out all but the whitelisted latents
        mask = torch.zeros(self.all_acts.shape[-1], dtype=self.all_acts.dtype)
        mask[whitelist] = 1.0
        self.all_acts = self.all_acts * mask


@dataclass
class PairedActivations:
    """
    Holds honest and deceptive activations for a DialogueDataset.

    This class is intended for training where it is useful to enforce that the honest and
    deceptive dialogues are of the same length as a variance reduction technique.
    """

    dataset: DialogueDataset
    honest_acts: Activations
    deceptive_acts: Activations

    def get_all_acts(self):
        return self.honest_acts, self.deceptive_acts

    @classmethod
    def from_dataset(
        cls,
        dataset: DialogueDataset,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel | None,
        batch_size: int = 12,
        detect_only_start_of_turn: bool = False,
        detect_only_last_token: bool = False,
        detect_only_eot: bool = False,
        layers: list[int] | None = None,
        save_logits: bool = False,
        verbose: bool = False,
        enforce_equal_detect_lengths: bool = False,
        use_goodfire_sae_acts: bool = False,
        detect_all: bool = False,
        cache_loader: Callable[[TokenizedDataset], Activations | None] | None = None,
        cache_saver: Callable[[TokenizedDataset, Activations], None] | None = None,
        postprocess: Callable[[Activations], Activations] | None = None,
    ) -> "PairedActivations":
        """
        Create SplitPairedActivations from a DialogueDataset.

        Args:
            dataset (DialogueDataset): Dataset to process
            tokenizer (PreTrainedTokenizerBase): Tokenizer to use
            model (PreTrainedModel): Model to get activations from
            batch_size (int): Batch size for processing
            detect_only_start_of_turn (bool): Whether to only detect at start of turn tokens
            detect_only_last_token (bool): Whether to only detect at last token before EOT
            detect_only_eot (bool): Whether to detect only the EOT associated with the final
                detected content
            layers (list[int] | None): Which layers to get activations from
            save_logits (bool): Whether to save logits
            verbose (bool): Whether to show progress bars
            enforce_equal_detect_lengths (bool): Whether to enforce equal detection lengths
            use_goodfire_sae_acts (bool): Whether to use Goodfire SAE activations
            cache_loader (callable | None): Optional cache loader for tokenized datasets
            cache_saver (callable | None): Optional cache saver for tokenized datasets
            postprocess (callable | None): Optional postprocess for computed activations
        """
        honest = dataset.get_honest()
        deceptive = dataset.get_deceptive()

        # make sure they are the same length by splicing to the min length
        min_length = min(len(honest), len(deceptive))
        honest = honest[:min_length]
        deceptive = deceptive[:min_length]

        honest_toks, deceptive_toks = get_tokenized_dataset_pair(
            honest,
            deceptive,
            tokenizer,
            enforce_equal_detect_lengths=enforce_equal_detect_lengths,
            detect_all=detect_all,
            detect_only_start_of_turn=detect_only_start_of_turn,
            detect_only_last_token=detect_only_last_token,
            detect_only_eot=detect_only_eot,
        )

        if not use_goodfire_sae_acts:
            def get_or_compute_acts(toks: TokenizedDataset) -> Activations:
                cached = cache_loader(toks) if cache_loader is not None else None
                if cached is not None:
                    return cached

                assert model is not None, "Model must be loaded to get activations."
                acts = Activations.from_model(
                    model,
                    toks,
                    layers=layers,
                    save_logits=save_logits,
                    verbose=verbose,
                    batch_size=batch_size,
                )
                if postprocess is not None:
                    acts = postprocess(acts)
                if cache_saver is not None:
                    cache_saver(toks, acts)
                return acts

            honest_acts = get_or_compute_acts(honest_toks)
            deceptive_acts = get_or_compute_acts(deceptive_toks)
        else:
            honest_acts = Activations.from_goodfire(honest_toks)
            deceptive_acts = Activations.from_goodfire(deceptive_toks)
            honest_acts.to_dense()
            deceptive_acts.to_dense()

        return cls(
            dataset=dataset,
            honest_acts=honest_acts,
            deceptive_acts=deceptive_acts,
        )
