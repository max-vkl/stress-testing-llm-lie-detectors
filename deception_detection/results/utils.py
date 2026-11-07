from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from deception_detection.data.base import DialogueDataset
from deception_detection.experiment import Experiment
from deception_detection.log import logger
from deception_detection.prompt_scorer import PromptScorer
from deception_detection.scores import Scores
from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.utils import REPO_ROOT


_ORIG_RC_FONT_SIZE = float(plt.rcParams.get("font.size", 10))


def apply_plot_font_scale(scale: float = 0.5) -> None:
    """
    Apply a consistent matplotlib font scaling for pipeline plots.

    Uses the rcParams font size at import time as the baseline so calling this
    multiple times is idempotent (it won't keep shrinking).
    """
    scaled = max(1.0, _ORIG_RC_FONT_SIZE * float(scale))
    # Keep everything readable but smaller; many plots have dense legends/ticks.
    plt.rcParams.update(
        {
            "font.size": scaled,
            "axes.titlesize": scaled,
            "axes.labelsize": scaled,
            "xtick.labelsize": scaled,
            "ytick.labelsize": scaled,
            "legend.fontsize": scaled,
            "legend.title_fontsize": scaled,
        }
    )


def save_plot(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=400)
    try:
        logger.info(f"plot saved to {path.relative_to(REPO_ROOT)}")
    except Exception:
        logger.info(f"plot saved to {path}")


def save_plot_exp(experiment: Experiment, plot_name: str, outfile: Path | None = None):
    save_plot(outfile or REPO_ROOT / experiment.cfg.folder / "plots" / plot_name)


def get_dataset_from_name(experiment: Experiment, dataset_name: str) -> DialogueDataset:
    all_datasets = experiment.control_datasets + experiment.eval_datasets
    try:
        return [d for d in all_datasets if d.name == dataset_name][0]
    except IndexError as err:
        raise ValueError(f"Dataset {dataset_name} not found in experiment") from err


def get_scores_from_name(
    experiment: Experiment, dataset_name: str, all_tokens: bool = False
) -> Scores:
    all_scores = experiment.get_control_scores(all_tokens=all_tokens) | experiment.get_eval_scores(
        all_tokens=all_tokens
    )
    try:
        return all_scores[dataset_name]
    except KeyError as err:
        raise ValueError(f"Scores for dataset {dataset_name} not found in experiment") from err


def get_dataframe_of_scores_on_dataset(
    experiment: Experiment,
    dataset_name: str,
    scorer: PromptScorer,
    normalize_scores: bool = True,
    include_black_box: bool = False,
) -> pd.DataFrame:
    """Returns a dataframe of metadata and scores for each prompt in a dataset."""
    dataset = get_dataset_from_name(experiment, dataset_name)
    dataset_scores = get_scores_from_name(experiment, dataset_name)

    data = {} if dataset.metadata is None else dataset.metadata.copy()
    data["score"] = scorer.process(dataset_scores, normalize=normalize_scores).tolist()
    data["label"] = [label.value for label in dataset_scores.labels]
    assert len(dataset) == len(
        dataset_scores
    ), f"mismatch between len of {dataset.name}({len(dataset)}) and scores ({len(dataset_scores)})"

    if include_black_box:
        all_black_box_scores = experiment.get_black_box_scores()
        if dataset_name in all_black_box_scores:
            black_box_scores = all_black_box_scores[dataset_name]
            data["black_box_score"] = [response[0].item() for response in black_box_scores.scores]
            data["black_box_reasoning"] = experiment.get_black_box_reasonings(dataset_name)
        else:
            logger.warning(f"No black box scores found for dataset {dataset_name}")

    return pd.DataFrame(data)


def to_readable_dataset_name(dataset_name: str, separate_val: bool = False) -> str:
    dataset_name_map = {
        "repe_honesty__plain": "RepE",
        "repe_honesty__plain_val": "RepE\n(Validation)",
        "roleplaying__plain": "Roleplaying",
        "roleplaying__plain_val": "Roleplaying\n(Validation)" if separate_val else "Roleplaying",
        "harm_refusal_choice__plain": "Harm Refusal\nChoice",
        "harm_refusal_choice__plain_val": "Harm Refusal Choice\n(Validation)" if separate_val else "Harm Refusal Choice",
        "harm_refusal_knowledge_report__plain": "Harm Refusal\nKnowledge Report",
        "harm_refusal_knowledge_report__plain_val": "Harm Refusal Knowledge Report\n(Validation)" if separate_val else "Harm Refusal Knowledge Report",
        "insider_trading_cadenza__plain": "Insider Trading\nCadenza",
        "insider_trading_cadenza__plain_val": "Insider Trading Cadenza\n(Validation)" if separate_val else "Insider Trading Cadenza",
        "instruct_dishonesty_default__plain": "Instruct Dishonesty\nDefault",
        "instruct_dishonesty_default__plain_val": "Instruct Dishonesty Default\n(Validation)" if separate_val else "Instruct Dishonesty Default",
        "instruct_dishonesty_donotadmit__plain": "Instruct Dishonesty\nDo Not Admit",
        "instruct_dishonesty_donotadmit__plain_val": "Instruct Dishonesty Do Not Admit\n(Validation)" if separate_val else "Instruct Dishonesty Do Not Admit",
        "mask__plain": "Mask",
        "mask__plain_val": "Mask\n(Validation)" if separate_val else "Mask",
        "sleeper_agents__plain": "Sleeper Agents",
        "sleeper_agents__plain_val": "Sleeper Agents\n(Validation)" if separate_val else "Sleeper Agents",
        "convince_me__plain": "Convince Me",
        "convince_me__plain_val": "Convince Me\n(Validation)" if separate_val else "Convince Me",
        "insider_trading__onpolicy": "Insider Trading\nReport",
        "insider_trading__upscale": "Insider Trading\nReport",
        "insider_trading_doubledown__onpolicy": "Insider Trading\nConfirmation",
        "insider_trading_doubledown__upscale": "Insider Trading\nConfirmation",
        "alpaca__plain": "Alpaca",
        "sandbagging_v2__wmdp_mmlu": "Sandbagging",
        "ai_liar__original_without_answers": "AI Liar",
        "ai_audit__no_reasoning": "AI Audit\n(No Reasoning)",
        "ai_audit__with_reasoning": "AI Audit\n(Reasoning)",
        "ai_audit__a1_2_reasoning": "AI Audit",
    }
    return dataset_name_map.get(dataset_name, dataset_name)


def get_control_df(exp: Experiment, scorer: PromptScorer, normalize: bool = True):
    """Helper for violin plots involving alpaca scores"""
    control_dfs: list[pd.DataFrame] = []
    for control_dataset, control_scores in exp.get_control_scores().items():
        normed_scores = scorer.process(control_scores, normalize=normalize)
        df = pd.DataFrame(
            {
                "score": normed_scores,
                "dataset": to_readable_dataset_name(control_dataset),
                "label": "control",
            }
        )
        control_dfs.append(df)
    control_df = pd.concat(control_dfs)
    return control_df


Color = tuple[float, float, float]


def get_dataset_color(dataset_name: str) -> Color:
    dark2_colors: list[Color] = plt.get_cmap("Dark2").colors  # type: ignore

    color_map = {
        "roleplaying__plain": dark2_colors[4],
        "roleplaying__plain_val": dark2_colors[4],
        "harm_refusal_choice__plain": dark2_colors[5],
        "harm_refusal_choice__plain_val": dark2_colors[5],
        "harm_refusal_knowledge_report__plain": dark2_colors[6],
        "harm_refusal_knowledge_report__plain_val": dark2_colors[6],
        "insider_trading_cadenza__plain": dark2_colors[7],
        "insider_trading_cadenza__plain_val": dark2_colors[7],
        "instruct_dishonesty_default__plain": dark2_colors[0],
        "instruct_dishonesty_default__plain_val": dark2_colors[0],
        "instruct_dishonesty_donotadmit__plain": dark2_colors[1],
        "instruct_dishonesty_donotadmit__plain_val": dark2_colors[1],
        "mask__plain": dark2_colors[2],
        "mask__plain_val": dark2_colors[2],
        "convince_me__plain": dark2_colors[2],
        "convince_me__plain_val": dark2_colors[2],
        "sleeper_agents__plain": dark2_colors[3],
        "sleeper_agents__plain_val": dark2_colors[3],
        "ai_liar__original_without_answers": dark2_colors[1],
        "insider_trading__onpolicy": dark2_colors[2],
        "insider_trading__upscale": dark2_colors[2],
        "insider_trading_doubledown__onpolicy": dark2_colors[3],
        "insider_trading_doubledown__upscale": dark2_colors[3],
        "sandbagging_v2__wmdp_mmlu": dark2_colors[6],
    }
    return color_map.get(dataset_name, dark2_colors[-1])


HONEST_COLOR = sns.color_palette("deep")[0]  # Blue
DECEPTIVE_COLOR = sns.color_palette("deep")[1] # Red for deceptive samples
NEUTRAL_COLOR = sns.color_palette("deep")[7]
AMBIGUOUS_COLOR = sns.color_palette("deep")[8]
