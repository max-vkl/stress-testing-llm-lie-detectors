"""
Activation based deception detectors.
"""

import pickle
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self, cast

import einops
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from deception_detection.activations import Activations
from deception_detection.dyl import base_probe_method
from deception_detection.log import logger
from deception_detection.scores import Scores
from deception_detection.utils import mask_sparse_activations, squeeze_sparse_tensor


class Detector(ABC):
    """
    Abstract base class for deception detectors.
    """

    @abstractmethod
    def score(self, acts: Activations, all_acts: bool = False) -> Scores:
        """Returns score object for the dialogue.

        Uses attention mask if all_acts is True, otherwise uses detection mask.
        """
        ...

    @abstractmethod
    def fit(self, positive_acts: Activations, negative_acts: Activations) -> None:
        """Find a direction representing the difference between positive and negative inputs.

        Inputs should be _paired_ such that the same batch index in the positive/negative inputs
        correspond meaningfully. It is okay if the inputs are offset from one another in sequence
        position, as long as the output_mask is set correctly.
        """
        ...

    @abstractmethod
    def save(self, file_path: str | Path):
        """
        Save the detector to a file.

        Args:
            file_path (str | Path): The path to save the detector.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, file_path: str | Path) -> Self:
        """
        Load a detector from a file.

        Args:
            file_path (str | Path): The path to load the detector from.
        """
        ...


class DirectionDetector(Detector):
    """
    A base class for direction-based deception detectors.

    Each detector chooses a single linear direction per layer to score the activations. The
    scores are averaged across layers.

    Attributes:
        layer (int | list[int]): What layer(s) to detect deception at? If multiple layers will
            take average across layers.
        directions (None | Float[Tensor, "layer emb"]): Learned directions for detection.
    """

    layers: list[int]
    directions: None | Float[Tensor, "layer emb"] = None

    def __init__(self, layers: list[int]):
        """
        Initialize the DirectionDetector.

        Args:
            layers (int | list[int]): List of model layers to use for detection.
        """
        self.layers = layers

    def save(self, file_path: str | Path):
        """
        Save the detector to a file.

        Args:
            file_path (str | Path): The path to save the detector.
        """
        data = {
            "layers": self.layers,
            "directions": self.directions.cpu() if self.directions is not None else None,
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, file_path: str | Path) -> Self:
        """
        Load a detector from a file.

        Args:
            file_path (str | Path): The path to load the detector from.

        Returns:
            DirectionDetector: The loaded detector.
        """
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        detector = cls(data["layers"])
        detector.directions = data["directions"]
        return detector

    @abstractmethod
    def fit(self, positive_acts: Activations, negative_acts: Activations) -> None:
        """Find a direction representing the difference between positive and negative inputs.

        Inputs should be _paired_ such that the same batch index in the positive/negative inputs
        correspond meaningfully. It is okay if the inputs are offset from one another in sequence
        position, as long as the output_mask is set correctly.


        Args:
            positive_inputs (Inputs): Inputs corresponding to deceptive statements.
            negative_inputs (Inputs): Inputs corresponding to honest statements.
        """
        ...

    @torch.no_grad()
    def get_score_tensor(
        self, acts_tensor: Float[Tensor, "toks layer emb"]
    ) -> Float[Tensor, " toks"]:
        """Multiply the activations by the directions"""
        assert self.directions is not None
        if acts_tensor.is_sparse:
            assert acts_tensor.shape[1] == 1, "Sparse activations only supported for single layer"
            acts_tensor = squeeze_sparse_tensor(acts_tensor, 1)
            return acts_tensor @ self.directions[0]
        else:
            acts_tensor = acts_tensor.to(self.directions)
            by_layer = einsum(
                acts_tensor, self.directions, "toks layer emb, layer emb -> toks layer"
            )
            return by_layer.mean(dim=-1)

    def score(self, acts: Activations, all_acts: bool = False) -> Scores:
        """Compute Score object for activations.

        If all_acts is true, will base it on the attention mask, otherwise will use the
        detection mask.
        """
        assert acts.tokenized_dataset.detection_mask is not None
        acts_tensor = acts.get_all_activations() if all_acts else acts.get_masked_activations()
        acts_tensor = self._index_layers(acts, acts_tensor)
        scores_where_mask_is_true = self.get_score_tensor(acts_tensor)
        mask = (
            acts.tokenized_dataset.attention_mask
            if all_acts
            else acts.tokenized_dataset.detection_mask
        )
        assert not scores_where_mask_is_true.isnan().any()
        n_masked_by_dialogue: list[int] = mask.sum(dim=1).tolist()
        scores_by_dialogue = torch.split(scores_where_mask_is_true, n_masked_by_dialogue)
        return Scores(scores_by_dialogue)

    def _index_layers(
        self, acts: Activations, act_tensor: Float[Tensor, "toks layer emb"]
    ) -> Float[Tensor, "toks layer emb"]:
        if acts.layers == self.layers:
            return act_tensor  # avoid copying tensor
        if act_tensor.is_sparse:
            if acts.layers is None:
                return act_tensor
            raise NotImplementedError(
                f"Sparse activations not supported for indexing layers {self.layers}!={acts.layers}"
            )
        if acts.layers is None:
            return act_tensor[:, torch.tensor(self.layers)]
        else:
            assert all([detector_layer in acts.layers for detector_layer in self.layers]), (
                f"Detector layers {self.layers} must be subset of act layers {acts.layers}"
            )
            layer_idx = torch.tensor(
                [acts.layers.index(detector_layer) for detector_layer in self.layers]
            )
            return act_tensor[:, layer_idx]

    def _idx_detect_acts(self, acts: Activations) -> Float[Tensor, "toks layer emb"]:
        return self._index_layers(acts, acts.get_masked_activations())


class LastTokenDetectorMixin:
    """Restrict detector fitting and masked scoring to the end of each detected span.

    Experiment tokenization normally reduces last-token detector masks to one EOT position per
    dialogue. Keeping the span-end selection here as well makes the detector itself safe to use
    with an unreduced detection mask and supports inputs containing multiple detected spans.
    """

    def _last_token_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Return a mask containing only the final token of each contiguous True segment."""
        if mask.numel() == 0:
            return mask
        assert mask.dtype == torch.bool
        next_mask = torch.zeros_like(mask)
        if mask.shape[1] > 1:
            next_mask[:, :-1] = mask[:, 1:]
        return mask & (~next_mask)

    def _masked_acts(self, acts: Activations, mask: torch.Tensor) -> torch.Tensor:
        """Return ``acts.all_acts[mask]`` with sparse-activation support."""
        if acts.all_acts.is_sparse:
            return mask_sparse_activations(acts.all_acts, mask)
        return acts.all_acts[mask]

    def _idx_detect_acts(self, acts: Activations) -> Float[Tensor, "toks layer emb"]:
        """Use only the final position of each detected span when fitting."""
        det_mask = acts.tokenized_dataset.detection_mask
        assert det_mask is not None, "Detection mask required for a last-token detector"
        last_acts = self._masked_acts(acts, self._last_token_mask(det_mask.bool()))
        return self._index_layers(acts, last_acts)  # type: ignore[attr-defined,no-any-return]

    def _fit_row_counts(self, acts: Activations) -> list[int]:
        """Rows contributed per dialogue: one per contiguous detected span."""
        det_mask = acts.tokenized_dataset.detection_mask
        assert det_mask is not None, "Detection mask required for a last-token detector"
        return self._last_token_mask(det_mask.bool()).sum(dim=1).tolist()

    def score(self, acts: Activations, all_acts: bool = False) -> Scores:
        """Score detected span ends, or retain ordinary all-token scoring for visualizations."""
        if all_acts:
            return super().score(acts, all_acts=True)  # type: ignore[misc]

        det_mask = acts.tokenized_dataset.detection_mask
        assert det_mask is not None, "Detection mask required for a last-token detector"
        last_mask = self._last_token_mask(det_mask.bool())
        acts_tensor = self._index_layers(  # type: ignore[attr-defined]
            acts, self._masked_acts(acts, last_mask)
        )
        scores_where_mask_is_true = self.get_score_tensor(acts_tensor)  # type: ignore[attr-defined]
        assert not scores_where_mask_is_true.isnan().any()

        n_masked_by_dialogue: list[int] = last_mask.sum(dim=1).tolist()
        return Scores(torch.split(scores_where_mask_is_true, n_masked_by_dialogue))


class MMSDetector(DirectionDetector):
    """
    Detector using to the difference between mean activations on positive and negative activations.
    """

    @torch.no_grad()
    def fit(self, positive_acts: Activations, negative_acts: Activations) -> None:
        pos_mean = self._idx_detect_acts(positive_acts).mean(dim=0)
        neg_mean = self._idx_detect_acts(negative_acts).mean(dim=0)
        self.directions = pos_mean - neg_mean  # [layer, emb]


class LastTokenMMSDetector(LastTokenDetectorMixin, MMSDetector):
    """Difference-of-means detector trained and scored on one last token per dialogue."""


class CovarianceAdjustedMMSDetector(DirectionDetector):
    """
    Detector using the covariance-adjusted difference between mean activations on positive
    and negative activations.
    """

    @torch.no_grad()
    def fit(self, positive_acts: Activations, negative_acts: Activations) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        pos_acts = self._idx_detect_acts(positive_acts).to(device).to(dtype)
        neg_acts = self._idx_detect_acts(negative_acts).to(device).to(dtype)

        # Compute mean diff
        pos_mean = pos_acts.mean(dim=0)
        neg_mean = neg_acts.mean(dim=0)
        assert not torch.isnan(pos_mean).any(), pos_acts.shape
        assert not torch.isnan(neg_mean).any(), neg_acts.shape
        diff = pos_mean - neg_mean

        # Compute covariance matrices for each layer
        # we compute the covariance separately for the positive and negative activations
        # and then average them
        n_layers = pos_acts.size(1)
        pos_cov = torch.stack(
            [torch.cov(pos_acts[:, layer_idx, :].T, correction=0) for layer_idx in range(n_layers)]
        )
        neg_cov = torch.stack(
            [torch.cov(neg_acts[:, layer_idx, :].T, correction=0) for layer_idx in range(n_layers)]
        )
        cov_matrix: Float[Tensor, "layer emb emb"] = (pos_cov + neg_cov) / 2
        cov_inv = cast(Float[Tensor, "layer emb emb2"], torch.linalg.pinv(cov_matrix))
        self.directions = einops.einsum(diff, cov_inv, "layer emb, layer emb emb2 -> layer emb2")


class LATDetector(DirectionDetector):
    """
    Detector using Linear Artificial Tomography (LAT) from the Representation Engineering paper.

    To compute the direction we:
    * collect the positive and negative activations on the paired inputs
    * compute the difference vectors between the activations
    * flip the sign of half of the difference vectors
    * mean-center this set of difference vectors (mean should be ≈0 anyhow)
    * take the principal PCA component of these difference vectors
    """

    def fit(self, positive_acts: Activations, negative_acts: Activations) -> None:
        idxed_pos_acts = self._idx_detect_acts(positive_acts)
        idxed_neg_acts = self._idx_detect_acts(negative_acts)
        diffs = idxed_pos_acts - idxed_neg_acts  # [toks, layer, emb]
        # we flip the signs randomly
        if torch.cuda.is_available():
            diffs = diffs.cuda()
        diffs *= torch.where(torch.randn(len(diffs)) > 0, 1, -1).to(diffs)[:, None, None]

        # we keep diffs in float32 as PCA requires it and the precision is nice
        diffs = diffs.to(torch.float32)

        # find the 0th pca direction for each layer
        diffs = rearrange(diffs, "toks layer emb -> layer toks emb")
        _, _, V = torch.pca_lowrank(diffs, q=1, center=True)
        self.directions = V[:, :, 0].cpu()

        # above direction finding method is unsupervised and may find reversed directions.
        # we ensure that the directions are pointing in the positive direction below
        # first we compute the scores by layer, taking the mean across all positions in the dataset
        pos_by_layer = einsum(
            idxed_pos_acts.to(self.directions),
            self.directions,
            "toks layer emb, layer emb -> layer",
        )
        neg_by_layer = einsum(
            idxed_neg_acts.to(self.directions),
            self.directions,
            "toks layer emb, layer emb -> layer",
        )
        # then flip the needed directions
        self.directions *= torch.where(pos_by_layer > neg_by_layer, 1, -1).unsqueeze(1)


class PCARepeDetector(DirectionDetector):
    """RepE-style PCA direction with the repository's deceptive-positive score convention.

    Fit reproduces the original honesty notebook's deterministic within-pair shuffle, PCA over
    pair differences, stored difference-vector mean, and pairwise-majority sign selection. The
    final direction is flipped when needed so larger scores mean deceptive, as they do elsewhere
    in this repository.
    """

    centers: None | Float[Tensor, "layer emb"] = None

    def save(self, file_path: str | Path):
        data = {
            "layers": self.layers,
            "directions": self.directions.cpu() if self.directions is not None else None,
            "centers": self.centers.cpu() if self.centers is not None else None,
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, file_path: str | Path) -> Self:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        detector = cls(data["layers"])
        detector.directions = data["directions"]
        detector.centers = data["centers"]
        return detector

    @staticmethod
    def _paper_shuffle_signs(n_pairs: int) -> torch.Tensor:
        """Signs induced by random.seed(0) and shuffling [honest, untruthful] per pair."""
        rng = random.Random(0)
        signs: list[float] = []
        for _ in range(n_pairs):
            # Relative rows are first - second. Relative to (deceptive - honest), honest-first
            # has sign -1 and deceptive-first has sign +1.
            pair_signs = [-1.0, 1.0]
            rng.shuffle(pair_signs)
            signs.append(pair_signs[0])
        return torch.tensor(signs, dtype=torch.float32)

    @torch.no_grad()
    def fit(self, positive_acts: Activations, negative_acts: Activations) -> None:
        deceptive = self._idx_detect_acts(positive_acts).to(dtype=torch.float32, device="cpu")
        honest = self._idx_detect_acts(negative_acts).to(dtype=torch.float32, device="cpu")
        if deceptive.shape != honest.shape:
            raise ValueError(
                "pca_repe requires one-to-one paired deceptive/honest activations; got "
                f"{tuple(deceptive.shape)} and {tuple(honest.shape)}"
            )
        positive_mask = positive_acts.tokenized_dataset.detection_mask
        negative_mask = negative_acts.tokenized_dataset.detection_mask
        if positive_mask is None or negative_mask is None:
            raise ValueError("pca_repe requires detection masks on both paired datasets")
        positive_counts = positive_mask.sum(dim=1).tolist()
        negative_counts = negative_mask.sum(dim=1).tolist()
        if positive_counts != negative_counts:
            raise ValueError(
                "pca_repe requires the same number of detected tokens for every paired "
                "deceptive/honest dialogue"
            )

        signs = self._paper_shuffle_signs(len(deceptive))
        signed_differences = (deceptive - honest) * signs[:, None, None]

        directions: list[torch.Tensor] = []
        centers: list[torch.Tensor] = []
        for layer_index in range(signed_differences.shape[1]):
            layer_differences = signed_differences[:, layer_index, :].numpy()
            # RepE uses sklearn PCA(n_components=1, whiten=False). Randomized SVD is the solver its
            # high-dimensional matrices select; fixing random_state makes reruns deterministic.
            pca = PCA(
                n_components=1,
                whiten=False,
                svd_solver="randomized",
                random_state=0,
            ).fit(layer_differences)
            center = torch.from_numpy(pca.mean_.astype(np.float32, copy=False))
            direction = torch.from_numpy(pca.components_[0].astype(np.float32, copy=False))

            honest_scores = (honest[:, layer_index, :] - center) @ direction
            deceptive_scores = (deceptive[:, layer_index, :] - center) @ direction
            honest_is_min = (honest_scores <= deceptive_scores).float().mean()
            honest_is_max = (honest_scores >= deceptive_scores).float().mean()
            honesty_sign = -1.0 if honest_is_min > honest_is_max else 1.0
            # The paper orients toward honesty. Fold in the opposite sign so this repository's
            # standard positive/deceptive labels and metrics remain unchanged.
            directions.append(direction * -honesty_sign)
            centers.append(center)

        self.directions = torch.stack(directions)
        self.centers = torch.stack(centers)

    @torch.no_grad()
    def get_score_tensor(
        self, acts_tensor: Float[Tensor, "toks layer emb"]
    ) -> Float[Tensor, " toks"]:
        assert self.directions is not None and self.centers is not None
        if acts_tensor.is_sparse:
            raise NotImplementedError("pca_repe does not support sparse SAE activations")
        acts_tensor = acts_tensor.to(self.directions)
        centered = acts_tensor - self.centers
        by_layer = einsum(centered, self.directions, "toks layer emb, layer emb -> toks layer")
        return by_layer.mean(dim=-1)


class LogisticRegressionDetector(DirectionDetector):
    """
    Detector using sklearn's LogisticRegression with strong regularization. Uses a Scaler to
    standardize the activations before fitting.

    Defaults to torch.float16 for activations and directions.
    """

    scaler_mean: None | torch.Tensor = None
    scaler_scale: None | torch.Tensor = None

    def __init__(
        self,
        layers: list[int],
        dtype: torch.dtype = torch.float32,
        reg_coeff: float = 1e3,
        normalize: bool = True,
        use_class_weights: bool = False,
        max_iter: int = 1000,
    ):
        super().__init__(layers)
        self.dtype = dtype
        self.reg_coeff = reg_coeff
        self.normalize = normalize
        self.use_class_weights = use_class_weights
        self.max_iter = max_iter

    def _fit_row_counts(self, acts: Activations) -> list[int]:
        """Rows contributed per dialogue by _idx_detect_acts (batch-major flattening order).
        Used to expand per-dialogue fit weights to per-row sample weights."""
        det_mask = acts.tokenized_dataset.detection_mask
        assert det_mask is not None, "Detection mask required for weighted fit"
        return det_mask.bool().sum(dim=1).tolist()

    def _expand_fit_weights(
        self,
        positive_acts: Activations,
        negative_acts: Activations,
        positive_weights: "np.ndarray | None",
        negative_weights: "np.ndarray | None",
    ) -> "np.ndarray | None":
        """Per-row sample weights ([positive rows..., negative rows...], matching the vstack order
        in fit) from optional per-dialogue weights. None when both weight vectors are None."""
        if positive_weights is None and negative_weights is None:
            return None
        pos_rows = self._fit_row_counts(positive_acts)
        neg_rows = self._fit_row_counts(negative_acts)
        pos_w = (
            np.ones(len(pos_rows))
            if positive_weights is None
            else np.asarray(positive_weights, dtype=np.float64)
        )
        neg_w = (
            np.ones(len(neg_rows))
            if negative_weights is None
            else np.asarray(negative_weights, dtype=np.float64)
        )
        assert len(pos_w) == len(pos_rows), (len(pos_w), len(pos_rows))
        assert len(neg_w) == len(neg_rows), (len(neg_w), len(neg_rows))
        return np.concatenate([np.repeat(pos_w, pos_rows), np.repeat(neg_w, neg_rows)])

    def fit(
        self,
        positive_acts: Activations,
        negative_acts: Activations,
        positive_weights: "np.ndarray | None" = None,
        negative_weights: "np.ndarray | None" = None,
    ) -> None:
        # Prepare the data
        X_pos = self._idx_detect_acts(positive_acts).to(self.dtype).cpu().numpy()
        X_neg = self._idx_detect_acts(negative_acts).to(self.dtype).cpu().numpy()
        X = np.vstack([X_pos, X_neg])
        y = np.hstack([np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])])

        # Optional per-dialogue fit weights (e.g. equal per-dataset weighting), expanded to rows.
        sample_weight = self._expand_fit_weights(
            positive_acts, negative_acts, positive_weights, negative_weights
        )
        assert sample_weight is None or len(sample_weight) == X.shape[0], (
            len(sample_weight),  # type: ignore[arg-type]
            X.shape[0],
        )

        # Compute class weights for balanced training (if enabled)
        class_weights = None
        if self.use_class_weights:
            # Weight_honest = #D / #D(honest)
            # Weight_dishonest = #D / #D(dishonest)
            total_samples = len(y)
            n_honest = X_neg.shape[0]  # negative_acts = honest samples (y=0)
            n_dishonest = X_pos.shape[0]  # positive_acts = dishonest samples (y=1)

            weight_honest = total_samples / n_honest
            weight_dishonest = total_samples / n_dishonest

            class_weights = {0: weight_honest, 1: weight_dishonest}

            print(f"Class weights - Honest: {weight_honest:.3f}, Dishonest: {weight_dishonest:.3f}")
            print(
                f"Sample counts - Honest: {n_honest}, Dishonest: {n_dishonest}, "
                f"Total: {total_samples}"
            )

        # Reshape the data to 2D for sklearn
        n_samples, n_layers, n_features = X.shape
        X_reshaped = X.reshape(n_samples, n_layers * n_features)
        # Standardize the features
        if self.normalize:
            scaler = StandardScaler()
            if sample_weight is not None:
                # Weighted normalization statistics, so the fit isn't dominated by large sources.
                X_scaled = scaler.fit_transform(X_reshaped, sample_weight=sample_weight)  # type: ignore
            else:
                X_scaled = scaler.fit_transform(X_reshaped)  # type: ignore

            # Save scaler normalization information in torch
            scaler_mean = scaler.mean_.reshape(n_layers, n_features)  # type: ignore
            scaler_scale = scaler.scale_.reshape(n_layers, n_features)  # type: ignore
            self.scaler_mean = torch.tensor(scaler_mean, dtype=self.dtype)
            self.scaler_scale = torch.tensor(scaler_scale, dtype=self.dtype)
        else:
            X_scaled = X_reshaped

        # Train the logistic regression model with optional class weights
        model = LogisticRegression(
            C=1 / self.reg_coeff,
            fit_intercept=False,
            class_weight=class_weights,
            max_iter=self.max_iter,
        )
        # sklearn multiplies class_weight and sample_weight; None == uniform.
        model.fit(X_scaled, y, sample_weight=sample_weight)  # type: ignore

        # Extract the coefficients and reshape them back to the original shape
        coef = model.coef_.reshape(n_layers, n_features)
        self.directions = torch.tensor(coef, dtype=self.dtype)

    def get_score_tensor(
        self, acts_tensor: Float[Tensor, "toks layer emb"]
    ) -> Float[Tensor, " toks"]:
        assert self.directions is not None
        acts_tensor = acts_tensor.to(self.directions)
        orig_dtype = acts_tensor.dtype
        if self.normalize:
            assert self.scaler_mean is not None
            assert self.scaler_scale is not None
            if acts_tensor.is_sparse:
                raise NotImplementedError("Sparse activations not supported for normalization")
            acts_tensor = (acts_tensor - self.scaler_mean) / self.scaler_scale

        return super().get_score_tensor(acts_tensor).to(orig_dtype)

    def save(self, file_path: str | Path):
        """
        Save the detector to a file.

        Args:
            file_path (str | Path): The path to save the detector.
        """
        data = {
            "layers": self.layers,
            "directions": self.directions.cpu() if self.directions is not None else None,
            "scaler_mean": self.scaler_mean.cpu() if self.scaler_mean is not None else None,
            "scaler_scale": self.scaler_scale.cpu() if self.scaler_scale is not None else None,
            "normalize": self.normalize,
            "reg_coeff": self.reg_coeff,
            "use_class_weights": self.use_class_weights,
            "max_iter": self.max_iter,
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, file_path: str | Path) -> "LogisticRegressionDetector":
        """
        Load a detector from a file.

        Args:
            file_path (str | Path): The path to load the detector from.

        Returns:
            LogisticRegressionDetector: The loaded detector.
        """
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        detector = cls(data["layers"])
        detector.directions = data["directions"]
        detector.scaler_mean = data["scaler_mean"]
        detector.scaler_scale = data["scaler_scale"]
        detector.normalize = data["normalize"]
        detector.reg_coeff = data["reg_coeff"]
        # Handle backwards compatibility
        detector.use_class_weights = data.get("use_class_weights", False)
        detector.max_iter = data.get("max_iter", 1000)
        return detector


class LastTokenLogisticRegressionDetector(LastTokenDetectorMixin, LogisticRegressionDetector):
    """
    Logistic Regression trained/scored on the *last token* of each detected message.

    Concretely, we take the `detection_mask` (which marks tokens belonging to detect=True messages,
    potentially expanded by padding), and then keep only the final token of each contiguous True
    segment. This typically corresponds to the last token in the assistant's detected message,
    and supports datasets with multiple detected message spans.

    Notes:
    - `score(all_acts=True)` is unchanged (scores every token under the attention mask), so token
      visualizations still work.
    - `score(all_acts=False)` returns per-dialogue sequences whose lengths equal the number of
      detected message segments in that dialogue (often 1).
    """


class MeanLogisticRegressionDetector(LogisticRegressionDetector):
    """
    Logistic Regression but you take the mean of the activations across seqpos before fitting.
    """

    def _mean_idx_detect_acts(self, acts: Activations) -> Float[Tensor, "batch layer emb"]:
        det_mask = acts.tokenized_dataset.detection_mask
        assert det_mask is not None, "Detection mask required for mlr"
        valid_dialogues = det_mask.bool().any(dim=1)
        if not valid_dialogues.any():
            raise ValueError("MLR training requires at least one dialogue with detected tokens")

        nan_outside_mask: Float[Tensor, "batch seqpos layer emb"] = torch.full_like(
            acts.all_acts, float("nan")
        )
        nan_outside_mask[det_mask] = acts.get_masked_activations()
        mean_by_pos = nan_outside_mask.nanmean(dim=1)
        return self._index_layers(acts, mean_by_pos[valid_dialogues])

    def _fit_row_counts(self, acts: Activations) -> list[int]:
        """One row for each dialogue with detected tokens (matching _mean_idx_detect_acts)."""
        det_mask = acts.tokenized_dataset.detection_mask
        assert det_mask is not None, "Detection mask required for mlr"
        return det_mask.bool().any(dim=1).to(torch.int64).tolist()

    def fit(
        self,
        positive_acts: Activations,
        negative_acts: Activations,
        positive_weights: "np.ndarray | None" = None,
        negative_weights: "np.ndarray | None" = None,
    ) -> None:
        # Prepare the data

        X_pos = self._mean_idx_detect_acts(positive_acts).to(self.dtype).cpu().numpy()
        X_neg = self._mean_idx_detect_acts(negative_acts).to(self.dtype).cpu().numpy()
        X = np.vstack([X_pos, X_neg])
        y = np.hstack([np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])])

        # Optional per-dialogue fit weights (one row per dialogue here).
        sample_weight = self._expand_fit_weights(
            positive_acts, negative_acts, positive_weights, negative_weights
        )
        assert sample_weight is None or len(sample_weight) == X.shape[0], (
            len(sample_weight),  # type: ignore[arg-type]
            X.shape[0],
        )

        # Compute class weights for balanced training (if enabled)
        class_weights = None
        if self.use_class_weights:
            total_samples = len(y)
            n_honest = X_neg.shape[0]  # negative_acts = honest samples (y=0)
            n_dishonest = X_pos.shape[0]  # positive_acts = dishonest samples (y=1)

            weight_honest = total_samples / n_honest
            weight_dishonest = total_samples / n_dishonest
            class_weights = {0: weight_honest, 1: weight_dishonest}

            print(f"Class weights - Honest: {weight_honest:.3f}, Dishonest: {weight_dishonest:.3f}")
            print(
                f"Sample counts - Honest: {n_honest}, Dishonest: {n_dishonest}, "
                f"Total: {total_samples}"
            )

        # Reshape the data to 2D for sklearn
        n_samples, n_layers, n_features = X.shape
        X_reshaped = X.reshape(n_samples, n_layers * n_features)
        # Standardize the features
        if self.normalize:
            scaler = StandardScaler()
            if sample_weight is not None:
                X_scaled = scaler.fit_transform(X_reshaped, sample_weight=sample_weight)  # type: ignore
            else:
                X_scaled = scaler.fit_transform(X_reshaped)  # type: ignore

            # Save scaler normalization information in torch
            scaler_mean = scaler.mean_.reshape(n_layers, n_features)  # type: ignore
            scaler_scale = scaler.scale_.reshape(n_layers, n_features)  # type: ignore
            self.scaler_mean = torch.tensor(scaler_mean, dtype=self.dtype)
            self.scaler_scale = torch.tensor(scaler_scale, dtype=self.dtype)
        else:
            X_scaled = X_reshaped

        # Train the logistic regression model
        model = LogisticRegression(
            C=1 / self.reg_coeff,
            random_state=42,
            fit_intercept=False,
            class_weight=class_weights,
            max_iter=self.max_iter,
        )
        model.fit(X_scaled, y, sample_weight=sample_weight)  # type: ignore

        # Extract the coefficients and reshape them back to the original shape
        coef = model.coef_.reshape(n_layers, n_features)
        self.directions = torch.tensor(coef, dtype=self.dtype)


class MLPDetector(DirectionDetector):
    """
    Detector using a Multi-Layer Perceptron (MLP) with the architecture:
    sigmoid(w_2^T ReLU (w_1^Tx+b_1)+b_2)
    """

    def __init__(self, layers: list[int], hidden_dim: int = 64):
        super().__init__(layers)
        self.hidden_dim = hidden_dim
        self.mlp = None
        self.dtype = torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, positive_acts: Activations, negative_acts: Activations) -> None:
        # Prepare the data
        X_pos = self._idx_detect_acts(positive_acts)
        X_neg = self._idx_detect_acts(negative_acts)
        X = torch.cat([X_pos, X_neg], dim=0).to(self.device, self.dtype).detach()
        y = torch.cat([torch.ones(X_pos.shape[0]), torch.zeros(X_neg.shape[0])]).to(
            self.device, self.dtype
        )

        # flatten the activations over all the layers
        X_flat = einops.rearrange(X, "batch layer emb -> batch (layer emb)")

        # split into train and val sets
        n = X_flat.shape[0]
        n_train = int(0.8 * n)
        train_idx = torch.randperm(n)[:n_train]
        val_idx = torch.arange(n)[~train_idx]
        X_flat_train = X_flat[train_idx]
        y_train = y[train_idx]
        X_flat_val = X_flat[val_idx]
        y_val = y[val_idx]

        # Initialize the MLP
        self.input_dim = X_flat.shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        ).to(self.device, self.dtype)

        # Train the MLP
        optimizer = torch.optim.AdamW(self.mlp.parameters(), lr=0.0001)
        criterion = nn.BCELoss()

        best_loss = float("inf")
        for epoch in range(10000):
            optimizer.zero_grad()
            outputs = self.mlp(X_flat_train).squeeze()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            if epoch % 500 == 0:
                with torch.no_grad():
                    outputs = self.mlp(X_flat_val).squeeze()
                    loss = criterion(outputs, y_val)
                    if loss < best_loss:
                        best_loss = loss
                    if loss > best_loss:
                        logger.info(f"Stopping, best loss: {best_loss}")
                        break
                logger.info(f"Epoch {epoch}, Val Loss: {loss.item()}")

    def get_score_tensor(
        self, acts_tensor: Float[Tensor, "toks layer emb"]
    ) -> Float[Tensor, " toks"]:
        assert self.mlp is not None, "MLP must be fitted before scoring"
        parameter = next(self.mlp.parameters())
        if parameter.device != self.device or parameter.dtype != self.dtype:
            self.mlp.to(self.device, self.dtype)
        in_acts = einops.rearrange(acts_tensor, "batch layer emb -> batch (layer emb)")
        in_acts = in_acts.to(self.device, self.dtype)
        return self.mlp(in_acts).squeeze().cpu()

    def save(self, file_path: str | Path):
        """
        Save the detector to a file.
        """
        data = {
            "layers": self.layers,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "mlp_state_dict": self.mlp.state_dict() if self.mlp is not None else None,
        }
        torch.save(data, file_path)

    @classmethod
    def load(cls, file_path: str | Path) -> "MLPDetector":
        """
        Load a detector from a file.
        """
        data = torch.load(file_path, map_location="cpu")
        detector = cls(data["layers"], data["hidden_dim"])
        detector.mlp = nn.Sequential(
            nn.Linear(data["input_dim"], data["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(data["hidden_dim"], 1),
            nn.Sigmoid(),
        )
        detector.mlp.load_state_dict(data["mlp_state_dict"])
        return detector


class BestLatentsDetector(Detector):
    """
    Detector using the top k best latents. Should be used with SAE acts.
    """

    def __init__(self, layers: list[int], detect_num_latents: int, experiment_path: str | Path):
        assert len(layers) == 1, "BestLatentsDetector should only be used with a single layer"
        self.top_k = detect_num_latents
        self.experiment_path = Path(experiment_path)

    def score(self, acts: Activations, all_acts: bool = False) -> Scores:
        assert acts.tokenized_dataset.detection_mask is not None
        acts_tensor = acts.get_all_activations() if all_acts else acts.get_masked_activations()
        acts_tensor = acts_tensor[:, 0, self.best_latent_indices].mean(dim=-1)
        mask = (
            acts.tokenized_dataset.attention_mask
            if all_acts
            else acts.tokenized_dataset.detection_mask
        )
        n_masked_by_dialogue: list[int] = mask.sum(dim=1).tolist()
        scores_by_dialogue = torch.split(acts_tensor, n_masked_by_dialogue)
        return Scores(scores_by_dialogue)

    def fit(self, positive_acts: Activations, negative_acts: Activations) -> None:
        positive_acts_detect_mask: Float[Tensor, " toks d_sae"] = (
            positive_acts.get_masked_activations()[:, 0]
        )
        negative_acts_detect_mask: Float[Tensor, " toks d_sae"] = (
            negative_acts.get_masked_activations()[:, 0]
        )
        # get a score for each latent by summing over toks
        pos_by_latent = positive_acts_detect_mask.sum(dim=0)
        neg_by_latent = negative_acts_detect_mask.sum(dim=0)
        self.top_latent_plot(pos_by_latent, neg_by_latent)
        diff = pos_by_latent - neg_by_latent
        self.best_latent_indices: list[int] = diff.topk(k=self.top_k).indices.tolist()
        # self.best_latent_descriptions = [
        #     get_feature_description(idx) for idx in self.best_latent_indices
        # ]
        # print(f"Best {self.top_k} latent descriptions: {self.best_latent_descriptions}")
        print(f"Best {self.top_k} latent indices: {self.best_latent_indices}")

    def save(self, file_path: str | Path):
        """Save the detector to a file."""
        data = {
            "best_latent_indices": self.best_latent_indices,
            # "best_latent_descriptions": self.best_latent_descriptions,
            "top_k": self.top_k,
            "experiment_path": self.experiment_path,
        }
        torch.save(data, file_path)

    @classmethod
    def load(cls, file_path: str | Path) -> "BestLatentsDetector":
        data = torch.load(file_path, map_location="cpu")
        detector = cls(
            layers=[0], detect_num_latents=data["top_k"], experiment_path=data["experiment_path"]
        )
        detector.best_latent_indices = data["best_latent_indices"]
        # detector.best_latent_descriptions = data["best_latent_descriptions"]
        detector.top_k = data["top_k"]
        return detector

    def top_latent_plot(
        self, pos_by_latent: Float[Tensor, " d_sae"], neg_by_latent: Float[Tensor, " d_sae"]
    ):
        diff = pos_by_latent - neg_by_latent
        top_k_indices = diff.topk(k=self.top_k).indices.tolist()
        bottom_k_indices = diff.topk(k=self.top_k, largest=False).indices.tolist()
        top_k_pos = pos_by_latent[top_k_indices]
        top_k_neg = neg_by_latent[top_k_indices]
        bottom_k_pos = pos_by_latent[bottom_k_indices]
        bottom_k_neg = neg_by_latent[bottom_k_indices]
        pos_to_plot = torch.cat([top_k_pos, bottom_k_pos], dim=0)
        neg_to_plot = torch.cat([top_k_neg, bottom_k_neg], dim=0)
        labels: list[int] = top_k_indices + bottom_k_indices
        # descriptions: list[str] = [get_feature_description(idx) for idx in labels]
        fig = go.Figure()

        # Add scatter points
        fig.add_trace(
            go.Scatter(
                x=neg_to_plot.tolist(),
                y=pos_to_plot.tolist(),
                mode="markers",
                marker=dict(color="blue", size=8),
                text=labels,
                hovertemplate=(
                    "Latent %{text}<br>Positive: %{y:.2f}<br>Negative: %{x:.2f}<extra></extra>"
                ),
            )
        )

        # Add diagonal line y=x
        max_val = max(neg_to_plot.max(), pos_to_plot.max())  # type: ignore
        min_val = min(neg_to_plot.min(), pos_to_plot.min())  # type: ignore
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="gray", dash="dash"),
                showlegend=False,
            )
        )

        # Update layout
        fig.update_layout(
            title="Positive vs Negative Activations by Latent",
            xaxis_title="Negative Activation Sum",
            yaxis_title="Positive Activation Sum",
            width=800,
            height=800,
            showlegend=False,
        )
        # Save the figure to HTML file
        fig.write_html(self.experiment_path / "best_latent_plot.html")

        return fig


def get_detector_class(method: str) -> type[Detector]:
    detector_classes = {
        "lr": LogisticRegressionDetector,
        "ltlr": LastTokenLogisticRegressionDetector,
        "ltlr_ct": LastTokenLogisticRegressionDetector,
        "mms": MMSDetector,
        "ltmms": LastTokenMMSDetector,
        "ltmms_ct": LastTokenMMSDetector,
        "lat": LATDetector,
        "pca_repe": PCARepeDetector,
        "mlp": MLPDetector,
        "cmms": CovarianceAdjustedMMSDetector,
        "mlr": MeanLogisticRegressionDetector,
        "best": BestLatentsDetector,
    }
    resolved_method = base_probe_method(method)
    if resolved_method not in detector_classes:
        raise ValueError(f"Invalid method: {method}")
    return detector_classes[resolved_method]
