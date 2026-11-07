import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deception_detection.experiment import Experiment
from deception_detection.prompt_scorer import PromptScorerMethod, create_prompt_scorer
from deception_detection.repository import DatasetRepository
from deception_detection.results import token_html
from deception_detection.results.plot import (
    make_dataset_specific_plots,
    plot_roc_comparison,
    plot_roc_curve_with_auroc,
    violin_plot,
)
from deception_detection.results.utils import apply_plot_font_scale, save_plot
from deception_detection.types import Label

REPOSITORY = DatasetRepository()
FPRS = [0.01, 0.001]

# Dataset-name suffixes marking system-prompt swap conditions. Whenever any eval dataset carries
# one of these tags, the by_metric plots are additionally emitted once per present tag (only the
# tagged datasets) and once without any tagged datasets, next to the standard all-datasets plot.
SYS_PROMPT_TAGS = ("assistantOnlySysPrompt", "deceptOnlySysPrompt")


def _safe_plot_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\\-]+", "_", name).strip("_")


def _sys_prompt_tag(dataset_name: str) -> str | None:
    return next((tag for tag in SYS_PROMPT_TAGS if dataset_name.endswith(f"_{tag}")), None)


def _plot_per_layer_lines(
    lines: dict[str, dict[int, float]],
    layers: list[int],
    title: str,
    ylabel: str,
    outfile: Path,
) -> None:
    """Plot one line per entry in `lines` ({label: {layer: value}}) against layer and save."""
    # Keep a tick + vertical guide line for every layer. Scale figure width with the
    # number of layers to reduce tick overlap (capped to avoid giant images).
    width = min(30.0, max(10.0, 0.18 * len(layers)))

    # Use the default matplotlib color cycle (usually 10 colors). When we run out of
    # colors, keep reusing them but switch linestyles to stay differentiable.
    prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
    base_colors = []
    if prop_cycle is not None and hasattr(prop_cycle, "by_key"):
        base_colors = list(prop_cycle.by_key().get("color", []))
    if not base_colors:
        base_colors = list(plt.get_cmap("tab10").colors)  # type: ignore[list-item]
    linestyles = ["-", "--", ":", "-."]

    fig, ax = plt.subplots(figsize=(width, 4))
    for i, (label, values_by_layer) in enumerate(lines.items()):
        y = np.array([values_by_layer.get(layer, np.nan) for layer in layers], dtype=float)
        color = base_colors[i % len(base_colors)]
        linestyle = linestyles[(i // len(base_colors)) % len(linestyles)]
        ax.plot(layers, y, linewidth=2, color=color, linestyle=linestyle, label=label)

    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_xlim(min(layers), max(layers))
    ax.set_axisbelow(True)
    # Grey vertical line for every layer.
    ax.grid(True, axis="x", color="0.85", linewidth=0.6)
    ax.grid(True, axis="y", alpha=0.3)
    # Keep legend outside on the left so it doesn't block the plot.
    ax.legend(loc="upper right", bbox_to_anchor=(-0.02, 1), borderaxespad=0)
    ax.set_xticks(layers)
    ax.tick_params(axis="x", labelrotation=90)
    save_plot(outfile)
    plt.close(fig)


def make_by_metric_plots(
    plot_data: dict[str, dict[str, dict[int, float]]],
    layers: list[int],
    by_metric_dir: Path,
) -> None:
    """Write per_layer_metric__<metric>.png plots (lines are datasets) into `by_metric_dir`.

    `plot_data` maps dataset -> metric -> {layer: value}. If any dataset carries a system-prompt
    tag (see SYS_PROMPT_TAGS), additionally writes, per metric, one plot per present tag (only
    the datasets with that tag) and one plot excluding all tagged datasets, next to the standard
    all-datasets plot. Standalone (no Experiment needed) so existing result folders can be
    regenerated from results_table.csv alone.
    """
    by_metric_dir.mkdir(parents=True, exist_ok=True)

    # Invert plot_data: metric -> dataset -> {layer: value}
    metric_to_dataset: dict[str, dict[str, dict[int, float]]] = {}
    for dataset_name, metric_map in plot_data.items():
        for metric, values_by_layer in metric_map.items():
            metric_to_dataset.setdefault(metric, {})[dataset_name] = values_by_layer

    present_tags = [
        tag for tag in SYS_PROMPT_TAGS if any(_sys_prompt_tag(name) == tag for name in plot_data)
    ]

    for metric, dataset_map in metric_to_dataset.items():
        metric_fname = _safe_plot_name(metric) or "metric"

        # (filename, title suffix, dataset subset)
        variants: list[tuple[str, str, dict[str, dict[int, float]]]] = [
            (f"per_layer_metric__{metric_fname}.png", "", dataset_map)
        ]
        for tag in present_tags:
            variants.append(
                (
                    f"per_layer_metric__{metric_fname}__{tag}.png",
                    f" ({tag} only)",
                    {n: v for n, v in dataset_map.items() if _sys_prompt_tag(n) == tag},
                )
            )
        if present_tags:
            variants.append(
                (
                    f"per_layer_metric__{metric_fname}__noSysPromptTag.png",
                    " (without sys-prompt tags)",
                    {n: v for n, v in dataset_map.items() if _sys_prompt_tag(n) is None},
                )
            )

        for plot_name, title_suffix, subset in variants:
            if not subset:
                continue
            _plot_per_layer_lines(
                subset,
                layers,
                title=f"Per-layer performance: {metric}{title_suffix}",
                ylabel=metric,
                outfile=by_metric_dir / plot_name,
            )


def _sys_prompt_category_rank(dataset_name: str) -> int:
    """Sort rank for category grouping: untagged first, then tags in SYS_PROMPT_TAGS order."""
    tag = _sys_prompt_tag(dataset_name)
    return 0 if tag is None else 1 + SYS_PROMPT_TAGS.index(tag)


def _floatfmt_for_columns(columns: list[str]) -> tuple[str, ...]:
    """Per-column float formats for markdown tables (first entry formats the index column).

    Matches ExperimentPostprocessor's table formatting: counts as integers, recall/fpr columns
    as percentages, everything else as 3-decimal floats.
    """
    fmts: list[str] = [""]
    for col in columns:
        if isinstance(col, str) and col.startswith("n_"):
            fmts.append(".0f")
        elif isinstance(col, str) and (col.startswith("recall ") or col.startswith("fpr ")):
            fmts.append(".1%")
        else:
            fmts.append(".3f")
    return tuple(fmts)


def _best_layers_summary_lines(
    dfs_by_layer: dict[int, pd.DataFrame], dataset_order: list[str]
) -> list[str]:
    """Build the "## Best layers" markdown block (highest AUROC per dataset, best mean AUROC
    overall, and — when sys-prompt-tagged datasets are present — per-category overall lines)."""
    lines: list[str] = []
    try:
        # Expect AUROC to be present in per-layer results.
        auroc_by_dataset: dict[str, dict[int, float]] = {}
        for layer, df in dfs_by_layer.items():
            if "auroc" not in df.columns:
                continue
            for name in df.index:
                auroc_by_dataset.setdefault(str(name), {})[int(layer)] = float(
                    df.loc[name, "auroc"]
                )
        if not auroc_by_dataset:
            return []

        def _best_layers(values_by_layer: dict[int, float]) -> list[int]:
            items = [(layer, val) for layer, val in values_by_layer.items() if not np.isnan(val)]
            if not items:
                return []
            max_val = max(val for _, val in items)
            # Use isclose for robustness across floating computations.
            return sorted([layer for layer, val in items if np.isclose(val, max_val)])

        def _mean_best_layers(names: list[str]) -> list[int]:
            # Best layer(s) by mean AUROC across the given datasets.
            layer_to_vals: dict[int, list[float]] = {}
            for name in names:
                for layer, val in auroc_by_dataset.get(name, {}).items():
                    if np.isnan(val):
                        continue
                    layer_to_vals.setdefault(layer, []).append(val)
            layer_mean = {
                layer: float(np.mean(vals)) for layer, vals in layer_to_vals.items() if vals
            }
            return _best_layers(layer_mean) if layer_mean else []

        def _fmt_layers(layers: list[int]) -> str:
            return ", ".join(str(x) for x in layers) if layers else "(none)"

        all_names = list(auroc_by_dataset)
        lines.append("## Best layers")
        lines.append("")
        lines.append(f"overall: {_fmt_layers(_mean_best_layers(all_names))}")

        # Per-sys-prompt-category overalls, only when tagged datasets are present.
        if any(_sys_prompt_tag(name) for name in all_names):
            for tag in SYS_PROMPT_TAGS:
                tagged = [name for name in all_names if _sys_prompt_tag(name) == tag]
                if tagged:
                    lines.append(f"overall__{tag}: {_fmt_layers(_mean_best_layers(tagged))}")
            untagged = [name for name in all_names if _sys_prompt_tag(name) is None]
            if untagged:
                lines.append(f"overall__noSysPromptTag: {_fmt_layers(_mean_best_layers(untagged))}")

        # Preserve the same dataset ordering as the tables.
        for name in dataset_order:
            if name in auroc_by_dataset:
                lines.append(f"{name}: {_fmt_layers(_best_layers(auroc_by_dataset[name]))}")
        lines.append("")
        return lines
    except Exception:
        # Never fail postprocessing due to an optional markdown summary.
        return []


def write_results_tables_by_layer(
    folder: Path,
    header_lines: list[str],
    dfs_by_layer: dict[int, pd.DataFrame],
) -> None:
    """Write results_tables_by_layer.md and, when sys-prompt-tagged datasets are present,
    results_tables_by_layer_byCategories.md.

    `dfs_by_layer` maps layer -> results dataframe (index: dataset names). The main file keeps
    the dataframes' row order; the byCategories file contains the same tables with rows grouped
    by sys-prompt category (untagged first, then each tag in SYS_PROMPT_TAGS order) and sorted
    by name within each group. Standalone (no Experiment needed) so existing result folders can
    be regenerated from results_table.csv alone.
    """
    dataset_names: list[str] = []
    for df in dfs_by_layer.values():
        for name in df.index:
            if str(name) not in dataset_names:
                dataset_names.append(str(name))

    def _write(md_path: Path, order: list[str], extra_header_lines: list[str]) -> None:
        sections: list[str] = []
        for layer, df in dfs_by_layer.items():
            df = df.loc[[name for name in order if name in df.index]]
            sections.append(f"## Layer {layer}\n")
            sections.append(df.to_markdown(floatfmt=_floatfmt_for_columns(list(df.columns))) + "\n")
        header = "\n".join(header_lines + extra_header_lines) + "\n\n"
        best_layers_lines = _best_layers_summary_lines(dfs_by_layer, order)
        if best_layers_lines:
            header += "\n".join(best_layers_lines) + "\n"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write("\n".join(sections).strip() + "\n")

    _write(folder / "results_tables_by_layer.md", dataset_names, [])

    if any(_sys_prompt_tag(name) for name in dataset_names):
        by_category_order = sorted(
            dataset_names, key=lambda name: (_sys_prompt_category_rank(name), name)
        )
        _write(
            folder / "results_tables_by_layer_byCategories.md",
            by_category_order,
            [
                "- Ordering: grouped by sys-prompt category (no tag, then "
                + ", ".join(SYS_PROMPT_TAGS)
                + "), sorted by name within each group"
            ],
        )


class ExperimentPostprocessor:
    def __init__(
        self,
        experiment: Experiment,
        scoring_method: PromptScorerMethod = "mean",
        fprs: list[float] = FPRS,
        default_fpr: float = 0.01,
        include_black_box: bool = False,
        control_dataset_name: str = "alpaca__plain",
    ):
        """
        Postprocess an experiment. Produce the standard plots, etc.
        Add things here once they become useful.

        Args:
            experiment: The experiment to postprocess.
            scoring_method: The method to use to score the experiment.
            fprs: The false positive rates used to produce plots that require multiple FPRs.
            default_fpr: The default false positive rate used to produce plots that require a
            single FPR.
            include_black_box: Whether to include black box scores in the results table.
        """
        self.experiment = experiment
        self.cfg = experiment.cfg
        self.plots_dir = self.cfg.folder / "plots"
        self.has_control = not bool(getattr(self.cfg, "no_control", False))
        self.scoring_method = scoring_method
        self.fprs = fprs
        self.default_fpr = default_fpr
        self.per_layer = (not self.cfg.merge_layers) and len(self.cfg.detect_layers) > 1
        scorer_experiment = experiment if self.has_control else None
        self.scorer = (
            None
            if self.per_layer
            else create_prompt_scorer(
                scoring_method,
                scorer_experiment,
                fprs,
                default_fpr,
                control_dataset_name=control_dataset_name,
            )
        )
        self.include_black_box = include_black_box
        self.control_dataset_name = control_dataset_name

    def _include_violin_plots(self) -> bool:
        """
        Whether to generate violin plots during postprocessing.

        Default behavior (when cfg.include_violin is unset):
        - per-layer mode: skip (very slow across many layers)
        - non per-layer mode: include (keeps legacy behavior)
        """
        include_violin = getattr(self.cfg, "include_violin", None)
        if include_violin is None:
            return not self.per_layer
        return bool(include_violin)

    def run(self) -> None:
        """
        Run the postprocessing.
        """
        # Reduce plot font sizes globally to avoid legends/ticks crowding plots.
        apply_plot_font_scale(0.5)
        self.make_results_table()
        if not self.per_layer:
            self.make_all_plots()
        else:
            self.make_all_plots_by_layer()

    def make_all_plots(self) -> None:
        """
        Make all the standard plots.
        """
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        assert self.scorer is not None

        # Always safe: deceptive-vs-honest ROC does not require control calibration.
        plot_roc_curve_with_auroc(
            self.experiment,
            self.scorer,
            compare_to="honest",
            control_dataset_name=self.control_dataset_name,
            outfile=self.plots_dir / f"roc_with_auroc_honest_{self.scorer.method}.png",
        )

        if not self.has_control:
            return

        token_html.plot_control_token_scores(self.experiment, self.scorer, num_samples=20)
        plot_roc_curve_with_auroc(
            self.experiment,
            self.scorer,
            compare_to="control",
            x_max=0.03,
            control_dataset_name=self.control_dataset_name,
        )
        if self._include_violin_plots():
            violin_plot(
                self.experiment,
                self.scorer,
                control_sample_size=getattr(self.cfg, "violin_control_sample_size", None),
            )
        plot_roc_comparison(
            self.experiment, self.scorer, control_dataset_name=self.control_dataset_name
        )
        make_dataset_specific_plots(self.experiment, self.scorer)

    def make_all_plots_by_layer(self) -> None:
        """
        In per-layer mode, generate the standard single-layer plots inside:
        results/<experiment>/plots/by_layer/layer_<layer>/
        """

        # Preload per-layer scores once. Plotting functions call get_eval_scores()/get_control_scores()
        # multiple times; if those methods hit disk (json.load) each time, postprocessing becomes
        # extremely slow.
        eval_scores_by_layer = self.experiment.get_eval_scores_by_layer(all_tokens=False)
        control_scores_by_layer = (
            self.experiment.get_control_scores_by_layer(all_tokens=False)
            if self.has_control
            else None
        )

        class _LayerExperimentView:
            def __init__(
                self,
                base: Experiment,
                layer: int,
                eval_scores_by_layer: dict[int, dict[str, "Scores"]],
                control_scores_by_layer: dict[int, dict[str, "Scores"]] | None,
            ):
                self._base = base
                self._layer = layer
                self._eval_scores_by_layer = eval_scores_by_layer
                self._control_scores_by_layer = control_scores_by_layer
                self.cfg = base.cfg
                self.tokenizer = base.tokenizer
                self.eval_datasets = base.eval_datasets
                self.control_datasets = base.control_datasets

            def get_eval_scores(self, all_tokens: bool = False):
                if all_tokens:
                    # Not preloaded in plotting path; defer to base (may hit disk).
                    return self._base.get_eval_scores_by_layer(all_tokens=True)[self._layer]
                return self._eval_scores_by_layer[self._layer]

            def get_control_scores(self, all_tokens: bool = False):
                if self._control_scores_by_layer is None:
                    raise ValueError(
                        "Control scores requested but this run has no control dataset (no_control=true)."
                    )
                if all_tokens:
                    # Not preloaded in plotting path; defer to base (may hit disk).
                    return self._base.get_control_scores_by_layer(all_tokens=True)[self._layer]
                return self._control_scores_by_layer[self._layer]

            def get_black_box_scores(self, mock: bool = False):
                return self._base.get_black_box_scores(mock=mock)

        plots_root = self.plots_dir
        plots_root.mkdir(parents=True, exist_ok=True)

        for layer in self.cfg.detect_layers:
            layer_dir = plots_root / "by_layer" / f"layer_{layer}"
            layer_dir.mkdir(parents=True, exist_ok=True)

            layer_exp = _LayerExperimentView(
                self.experiment, layer, eval_scores_by_layer, control_scores_by_layer
            )
            if self.has_control:
                assert control_scores_by_layer is not None
                layer_control = control_scores_by_layer[layer][self.control_dataset_name]
                scorer = create_prompt_scorer(
                    self.scoring_method,
                    self.experiment,
                    self.fprs,
                    self.default_fpr,
                    control_dataset_name=self.control_dataset_name,
                    control_scores=layer_control,
                )
            else:
                scorer = create_prompt_scorer(
                    self.scoring_method,
                    None,
                    self.fprs,
                    self.default_fpr,
                    control_dataset_name=self.control_dataset_name,
                )

            # Match the usual output filenames, but inside layer_<N>/.
            plot_roc_curve_with_auroc(
                layer_exp,  # type: ignore[arg-type]
                scorer,
                compare_to="honest",
                control_dataset_name=self.control_dataset_name,
                outfile=layer_dir / f"roc_with_auroc_honest_{scorer.method}.png",
            )
            if self.has_control:
                plot_roc_curve_with_auroc(
                    layer_exp,  # type: ignore[arg-type]
                    scorer,
                    compare_to="control",
                    x_max=0.03,
                    control_dataset_name=self.control_dataset_name,
                    outfile=layer_dir / f"roc_with_auroc_control_{scorer.method}.png",
                )
                if self._include_violin_plots():
                    violin_plot(
                        layer_exp,  # type: ignore[arg-type]
                        scorer,
                        outfile=layer_dir / f"violin_plot_{scorer.method}.png",
                        control_sample_size=getattr(
                            self.cfg, "violin_control_sample_size", None
                        ),
                    )
                plot_roc_comparison(
                    layer_exp,  # type: ignore[arg-type]
                    scorer,
                    control_dataset_name=self.control_dataset_name,
                    outfile=layer_dir / f"all_roc_plots_{scorer.method}.png",
                )

    def make_results_table(self) -> None:
        """
        Make a results table of summary statistics
        - AUROCs between honest and deceptive dialogues
        - TPRs and FPRs of datasets at various FPRs on control datasets.
        """
        def _floatfmt_for_df(df: pd.DataFrame) -> tuple[str, ...]:
            """
            Build a per-column float formatting tuple for pandas' markdown rendering.

            The first entry formats the index column.
            """
            if self.has_control:
                # Legacy formatting (keeps index blank, counts as integers, summary as floats,
                # then percent-format for recall/fpr columns).
                assert self.scorer is not None
                return (
                    "",
                    ".0f",
                    ".0f",
                    ".0f",
                    ".3f",
                    ".3f",
                    ".3f",
                    *[".1%" for _ in self.scorer.fprs * 2],
                )

            fmts: list[str] = [""]
            for col in df.columns:
                if isinstance(col, str) and col.startswith("n_"):
                    fmts.append(".0f")
                else:
                    # Keep metrics compact and consistent.
                    fmts.append(".3f")
            return tuple(fmts)

        if not self.per_layer:
            assert self.scorer is not None
            df = self.get_results_dataframe()
            new_columns = {
                f"TPR_{fpr:.4f}": f"TPR @ {fpr:.2%}" for fpr in self.scorer.fprs
            }
            new_columns.update(
                {f"FPR_{fpr:.4f}": f"FPR @ {fpr:.2%}" for fpr in self.scorer.fprs}
            )
            new_dataset_names: dict[str, str] = dict()  # {"geometry_of_truth_cities": "cities"}
            df.rename(index=new_dataset_names, columns=new_columns, inplace=True)
            print(
                df.to_markdown(
                    floatfmt=_floatfmt_for_df(df)
                )
            )
            csv_file = self.experiment.cfg.folder / "results_table.csv"
            df.to_csv(csv_file)
            return

        # Per-layer mode: print one table per layer, and write a combined CSV with `layer` column.
        eval_scores_by_layer = self.experiment.get_eval_scores_by_layer()
        control_scores_by_layer = (
            self.experiment.get_control_scores_by_layer() if self.has_control else None
        )

        all_rows: list[dict] = []
        # For plotting: dataset -> metric -> {layer: value}
        plot_data: dict[str, dict[str, dict[int, float]]] = {}
        # For the markdown files: layer -> results dataframe (index: dataset names)
        dfs_by_layer: dict[int, pd.DataFrame] = {}

        def _metric_columns(df: pd.DataFrame) -> list[str]:
            # Keep the most relevant 0..1 summary metrics.
            cols: list[str] = []
            for c in df.columns:
                if c in {"auroc", "black_box_auroc"} or self.has_control and c in {"auroc_vs_alpaca", "balanced_accuracy_at_fpr"} or self.has_control and c.startswith("recall "):
                    cols.append(c)
            return cols

        for layer in self.cfg.detect_layers:
            print(f"\nLayer {layer}:\n")

            if self.has_control:
                assert control_scores_by_layer is not None
                layer_control = control_scores_by_layer[layer][self.control_dataset_name]
                scorer = create_prompt_scorer(
                    self.scoring_method,
                    self.experiment,
                    self.fprs,
                    self.default_fpr,
                    control_dataset_name=self.control_dataset_name,
                    control_scores=layer_control,
                )
            else:
                scorer = create_prompt_scorer(
                    self.scoring_method,
                    None,
                    self.fprs,
                    self.default_fpr,
                    control_dataset_name=self.control_dataset_name,
                )

            df_layer = self.get_results_dataframe_for_scores(
                eval_scores_by_layer[layer], scorer
            )

            # Collect per-dataset values for plots
            metric_cols = _metric_columns(df_layer)
            for dataset_name in df_layer.index.tolist():
                plot_data.setdefault(dataset_name, {})
                for metric in metric_cols:
                    val = float(df_layer.loc[dataset_name, metric])
                    plot_data[dataset_name].setdefault(metric, {})
                    plot_data[dataset_name][metric][layer] = val

            layer_md = df_layer.to_markdown(
                floatfmt=_floatfmt_for_df(df_layer)
            )
            print(layer_md)
            dfs_by_layer[layer] = df_layer

            # Collect rows for CSV
            df_layer = df_layer.reset_index().rename(columns={"index": "eval_dataset"})
            df_layer.insert(0, "layer", layer)
            all_rows.append(df_layer)

        csv_df = pd.concat(all_rows, ignore_index=True)
        csv_file = self.experiment.cfg.folder / "results_table.csv"
        csv_df.to_csv(csv_file, index=False)

        # Write the per-layer tables (as printed) into markdown file(s), including the
        # "best layers" summary and (when sys-prompt-tagged datasets are present) the
        # category-grouped variant. See write_results_tables_by_layer.
        header_lines = [
            "# Results tables (by layer)",
            "",
            f"- Experiment: `{self.experiment.cfg.name}`",
            f"- Scoring method: `{self.scoring_method}`",
        ]
        if self.has_control:
            header_lines.append(f"- Control dataset: `{self.control_dataset_name}`")
        else:
            header_lines.append("- Control dataset: (none; `no_control=true`)")
        write_results_tables_by_layer(self.experiment.cfg.folder, header_lines, dfs_by_layer)

        # Create metric-vs-layer plots
        # - by_dataset/: one plot per eval dataset, lines are metrics
        # - by_metric/: one plot per metric, lines are datasets (plus, when sys-prompt-tagged
        #   datasets are present, per-tag and no-tag subset plots; see make_by_metric_plots)
        plots_root = self.plots_dir
        by_dataset_dir = plots_root / "by_dataset"
        by_dataset_dir.mkdir(parents=True, exist_ok=True)

        if self.cfg.detect_layers:
            layers = sorted({int(x) for x in self.cfg.detect_layers})

            # ---- by_dataset ----
            for dataset_name, metric_map in plot_data.items():
                plot_name = f"per_layer_metrics__{_safe_plot_name(dataset_name)}.png"
                _plot_per_layer_lines(
                    metric_map,
                    layers,
                    title=f"Per-layer metrics: {dataset_name}",
                    ylabel="Metric value",
                    outfile=by_dataset_dir / plot_name,
                )

            # ---- by_metric ----
            make_by_metric_plots(plot_data, layers, plots_root / "by_metric")

    def get_results_dataframe(self) -> pd.DataFrame:
        assert self.scorer is not None, "Scorer must be initialized before getting results"
        eval_scores = self.experiment.get_eval_scores()

        table_contents = []
        for dataset_name, scores in eval_scores.items():
            split_scores = scores.get_scores_by_label()

            row = {
                "n_honest": len(split_scores[Label.HONEST]),
                "n_ambiguous": len(split_scores[Label.AMBIGUOUS]),
                "n_deceptive": len(split_scores[Label.DECEPTIVE]),
                # "n_skipped": len(split_scores[Label.SKIP]),
                "auroc": self.scorer.auroc(scores),
                # "balanced_accuracy": self.scorer.balanced_accuracy(scores)

            }
            if self.has_control:
                row["auroc_vs_alpaca"] = self.scorer.auroc_alpaca(scores)
                row["balanced_accuracy_at_fpr"] = self.scorer.balanced_accuracy_at_fpr(scores)
                for threshold in self.scorer.fprs:
                    fpr = self.scorer.positive_rate(split_scores[Label.HONEST].scores, threshold)
                    fnr = 1 - self.scorer.positive_rate(
                        split_scores[Label.DECEPTIVE].scores, threshold
                    )
                    row[f"recall {threshold:.1%}"] = 1 - fnr
                    row[f"fpr {threshold:.1%}"] = fpr

            if self.include_black_box:
                bb_scores = self.experiment.get_black_box_scores()[dataset_name]
                row["black_box_auroc"] = self.scorer.auroc(bb_scores)
                if self.has_control:
                    row["black_box_auroc_vs_alpaca"] = float("nan")

            table_contents.append(row)

        return pd.DataFrame(table_contents, index=list(eval_scores.keys()))  # type: ignore

    def get_results_dataframe_for_scores(
        self, eval_scores: dict[str, "Scores"], scorer
    ) -> pd.DataFrame:
        """
        Build a results dataframe for a provided (dataset -> Scores) dict and a scorer.
        Used for per-layer mode.
        """
        table_contents = []
        for dataset_name, scores in eval_scores.items():
            split_scores = scores.get_scores_by_label()
            row = {
                "n_honest": len(split_scores[Label.HONEST]),
                "n_ambiguous": len(split_scores[Label.AMBIGUOUS]),
                "n_deceptive": len(split_scores[Label.DECEPTIVE]),
                "auroc": scorer.auroc(scores),
            }
            if self.has_control:
                row["auroc_vs_alpaca"] = scorer.auroc_alpaca(scores)
                row["balanced_accuracy_at_fpr"] = scorer.balanced_accuracy_at_fpr(scores)
                for threshold in scorer.fprs:
                    fpr = scorer.positive_rate(split_scores[Label.HONEST].scores, threshold)
                    fnr = 1 - scorer.positive_rate(
                        split_scores[Label.DECEPTIVE].scores, threshold
                    )
                    row[f"recall {threshold:.1%}"] = 1 - fnr
                    row[f"fpr {threshold:.1%}"] = fpr

            if self.include_black_box:
                bb_scores = self.experiment.get_black_box_scores()[dataset_name]
                row["black_box_auroc"] = scorer.auroc(bb_scores)
                if self.has_control:
                    row["black_box_auroc_vs_alpaca"] = float("nan")

            table_contents.append(row)

        return pd.DataFrame(table_contents, index=list(eval_scores.keys()))  # type: ignore
