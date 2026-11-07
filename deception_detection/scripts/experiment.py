from pathlib import Path
from time import time
from typing import Any, Literal

import fire
import torch
import yaml

from deception_detection.alpaca_false_positives import export_control_false_positives_to_csv
from deception_detection.experiment import Experiment, ExperimentConfig
from deception_detection.log import logger
from deception_detection.postprocess import ExperimentPostprocessor
from deception_detection.prompt_scorer import PromptScorerMethod
from deception_detection.results.detection_masks import save_detection_masks
from deception_detection.utils import resolve_config_path


def _delete_score_jsons(exp_folder: Path) -> None:
    """
    Delete large score JSON artifacts. Safe to call even if files don't exist.
    """
    candidates = [
        # merged mode
        "scores.json",
        "scores_all_tokens.json",
        "control_scores.json",
        "control_scores_all_tokens.json",
        "train_scores.json",
        "train_scores_all_tokens.json",
        # per-layer mode
        "scores_by_layer.json",
        "scores_all_tokens_by_layer.json",
        "control_scores_by_layer.json",
        "control_scores_all_tokens_by_layer.json",
        "train_scores_by_layer.json",
        "train_scores_all_tokens_by_layer.json",
    ]
    deleted = 0
    for name in candidates:
        path = exp_folder / name
        try:
            if path.exists():
                path.unlink()
                deleted += 1
        except Exception as err:
            logger.warning("Failed to delete %s: %s", path, err)
    if deleted:
        logger.info("Deleted %d score JSON file(s) (enable_rerun=false)", deleted)


def _delete_detectors(exp_folder: Path) -> None:
    """
    Delete detector artifacts from the experiment folder.
    Safe to call even if files don't exist.
    """
    deleted = 0
    # merged mode detector
    det = exp_folder / "detector.pt"
    try:
        if det.exists():
            det.unlink()
            deleted += 1
    except Exception as err:
        logger.warning("Failed to delete %s: %s", det, err)

    # per-layer detectors
    try:
        for path in exp_folder.glob("detector_layer_*.pt"):
            try:
                path.unlink()
                deleted += 1
            except Exception as err:
                logger.warning("Failed to delete %s: %s", path, err)
    except Exception as err:
        logger.warning("Failed listing per-layer detectors in %s: %s", exp_folder, err)

    if deleted:
        logger.info("Deleted %d detector file(s) (enable_rerun=false)", deleted)


def _format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in compact form (d h m s), showing only needed units."""
    days, remainder = divmod(int(seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def run(
    config_file: str | Path = "0repe",
    **kwargs: dict[str, Any],
):
    start_time = time()

    config_file = resolve_config_path(config_file)
    with open(config_file) as file:
        config_dict = yaml.safe_load(file)

    config_dict |= kwargs
    config = ExperimentConfig(**config_dict)  # type: ignore[reportGeneralTypeIssues]
    try:
        experiment = Experiment(config)
        save_detection_masks(experiment)
        if config.merge_layers or len(config.detect_layers) <= 1:
            experiment.get_detector()
        else:
            experiment.get_detectors()
    except Exception as err:
        # cleanup experiment folder and any files inside it
        for f in config.folder.iterdir():
            f.unlink()
        config.folder.rmdir()
        raise err
    if config.merge_layers or len(config.detect_layers) <= 1:
        experiment.get_eval_scores()
        if not experiment.cfg.no_control:
            experiment.get_control_scores()
    else:
        experiment.get_eval_scores_by_layer()
        if not experiment.cfg.no_control:
            experiment.get_control_scores_by_layer()
    if experiment.cfg.debug_alpaca_false_positives:
        export_control_false_positives_to_csv(
            experiment,
            dataset_name=experiment.cfg.debug_alpaca_false_positive_dataset,
            fpr=experiment.cfg.debug_alpaca_false_positive_fpr,
            scoring_method="mean",
        )
    # Print max memory usage
    max_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
    logger.info(f"Max GPU memory usage: {max_memory:.2f} GB")
    results(str(experiment.cfg.folder), scoring_method="mean")
    if not experiment.cfg.enable_rerun:
        _delete_score_jsons(experiment.cfg.folder)
        _delete_detectors(experiment.cfg.folder)

    elapsed = time() - start_time
    print(f"Experiment run time: {_format_elapsed_time(elapsed)}")


def rerun(folder: str, scoring_method: Literal["mean", "max", "relu"] = "mean"):
    """Runs experiment from existing folder, reusing saved results when possible."""
    start_time = time()

    cfg = ExperimentConfig.from_path(Path(folder))
    experiment = Experiment(cfg)
    if cfg.merge_layers or len(cfg.detect_layers) <= 1:
        experiment.get_detector()
        experiment.get_eval_scores()
        if not experiment.cfg.no_control:
            experiment.get_control_scores()
    else:
        experiment.get_detectors()
        experiment.get_eval_scores_by_layer()
        if not experiment.cfg.no_control:
            experiment.get_control_scores_by_layer()
    if experiment.cfg.debug_alpaca_false_positives:
        export_control_false_positives_to_csv(
            experiment,
            dataset_name=experiment.cfg.debug_alpaca_false_positive_dataset,
            fpr=experiment.cfg.debug_alpaca_false_positive_fpr,
            scoring_method="mean",
        )
    max_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
    logger.info(f"Max GPU memory usage: {max_memory:.2f} GB")
    results(str(experiment.cfg.folder), scoring_method=scoring_method)
    if not experiment.cfg.enable_rerun:
        _delete_score_jsons(experiment.cfg.folder)
        _delete_detectors(experiment.cfg.folder)

    elapsed = time() - start_time
    print(f"Experiment run time: {_format_elapsed_time(elapsed)}")


def recompute_dataset(folder: str, dataset_name: str, shard_size: int = 12):
    cfg = ExperimentConfig.from_path(Path(folder))
    experiment = Experiment(cfg)
    experiment.recompute_scores_on_dataset(dataset_name, shard_size=shard_size)
    results(str(experiment.cfg.folder), scoring_method="mean")
    if not experiment.cfg.enable_rerun:
        _delete_score_jsons(experiment.cfg.folder)
        _delete_detectors(experiment.cfg.folder)


def results(
    folder: str, scoring_method: PromptScorerMethod = "mean", include_black_box: bool = False
):
    cfg = ExperimentConfig.from_path(Path(folder))
    experiment = Experiment(cfg)
    if cfg.merge_layers or len(cfg.detect_layers) <= 1:
        experiment.get_eval_scores(all_tokens=False)
    else:
        experiment.get_eval_scores_by_layer(all_tokens=False)
    control_dataset_name = cfg.control_data[0] if cfg.control_data else "alpaca__plain"
    postprocessor = ExperimentPostprocessor(
        experiment,
        scoring_method=scoring_method,
        include_black_box=include_black_box,
        control_dataset_name=control_dataset_name,
    )
    postprocessor.run()


def just_black_box_scores(folder: str):
    cfg = ExperimentConfig.from_path(Path(folder))
    experiment = Experiment(cfg)
    experiment.get_black_box_scores()


if __name__ == "__main__":
    fire.Fire(
        {
            "run": run,
            "results": results,
            "rerun": rerun,
            "recompute": recompute_dataset,
            "black_box": just_black_box_scores,
        }
    )
