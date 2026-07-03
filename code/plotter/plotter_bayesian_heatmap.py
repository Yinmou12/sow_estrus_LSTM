from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sow_estrus_LSTM_Info import result_save_path

ALL_EXPERIMENTS_ROOT = Path(result_save_path) / "all_experiments"
BO_EXPERIMENT_PREFIX = "07_bilstm_ast_bayesian_tuning"
FIGURE_DPI = 1000

HIDDEN_SIZE_ORDER = [
    "32_32",
    "64_32",
    "64_64",
    "64_64_32",
    "128_64_32",
    "64_64_64_64",
    "128_128_64_32",
]


def find_latest_experiment_dir(prefix=BO_EXPERIMENT_PREFIX, root=ALL_EXPERIMENTS_ROOT):
    root = Path(root)
    candidates = [
        path
        for path in root.glob(f"{prefix}_*")
        if path.is_dir() and (path / "final_summary.xlsx").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No experiment directory matching '{prefix}_*' was found under {root}."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _prepare_output_dir(experiment_dir):
    figures_dir = Path(experiment_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _save_figure(fig, figures_dir, file_stem):
    png_path = Path(figures_dir) / f"{file_stem}.png"
    pdf_path = Path(figures_dir) / f"{file_stem}.pdf"
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _require_columns(df, columns, source_path):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source_path} is missing required columns: {missing}")


def _normalize_hidden_sizes(value):
    if isinstance(value, str):
        return (
            value.strip()
            .replace("[", "")
            .replace("]", "")
            .replace(", ", "_")
            .replace(",", "_")
        )
    if isinstance(value, (list, tuple)):
        return "_".join(str(int(item)) for item in value)
    return str(value)


def _format_hidden_sizes(value):
    return "[" + ",".join(str(part) for part in str(value).split("_")) + "]"


def _format_learning_rate(value):
    return f"{float(value):g}"


def _load_validation_summary(summary_path):
    df = pd.read_excel(summary_path)
    _require_columns(
        df,
        ["Dataset", "param_hidden_sizes", "param_learning_rate", "F1-Score"],
        summary_path,
    )
    val_df = df[df["Dataset"] == "Validation"].copy()
    if val_df.empty:
        raise ValueError(f"No Validation rows were found in {summary_path}.")

    val_df["param_hidden_sizes"] = val_df["param_hidden_sizes"].map(
        _normalize_hidden_sizes
    )
    val_df["param_learning_rate"] = pd.to_numeric(
        val_df["param_learning_rate"], errors="coerce"
    )
    val_df["F1-Score"] = pd.to_numeric(val_df["F1-Score"], errors="coerce")
    val_df = val_df.dropna(subset=["param_learning_rate", "F1-Score"])
    if val_df.empty:
        raise ValueError(f"No usable Validation F1 rows were found in {summary_path}.")
    return val_df


def _build_mean_f1_heatmap(val_df):
    mean_df = (
        val_df.groupby(["param_hidden_sizes", "param_learning_rate"], as_index=False)[
            "F1-Score"
        ]
        .mean()
        .rename(columns={"F1-Score": "mean_f1"})
    )
    heatmap_df = mean_df.pivot(
        index="param_hidden_sizes", columns="param_learning_rate", values="mean_f1"
    )

    row_order = [item for item in HIDDEN_SIZE_ORDER if item in heatmap_df.index]
    extra_rows = [item for item in heatmap_df.index if item not in row_order]
    row_order.extend(sorted(extra_rows))
    col_order = sorted(heatmap_df.columns)
    return heatmap_df.reindex(index=row_order, columns=col_order)


def _load_best_validation_point(summary_path):
    df = pd.read_excel(summary_path)
    _require_columns(
        df,
        [
            "Dataset",
            "Experiment",
            "param_hidden_sizes",
            "param_learning_rate",
            "F1-Score",
        ],
        summary_path,
    )
    val_df = df[df["Dataset"] == "Validation"].copy()
    val_df["F1-Score"] = pd.to_numeric(val_df["F1-Score"], errors="coerce")
    val_df["param_learning_rate"] = pd.to_numeric(
        val_df["param_learning_rate"], errors="coerce"
    )
    val_df = val_df.dropna(subset=["F1-Score", "param_learning_rate"])
    if val_df.empty:
        raise ValueError(f"No usable Validation rows were found in {summary_path}.")
    best_row = val_df.loc[val_df["F1-Score"].idxmax()]
    return {
        "experiment": best_row["Experiment"],
        "hidden_sizes": _normalize_hidden_sizes(best_row["param_hidden_sizes"]),
        "learning_rate": float(best_row["param_learning_rate"]),
        "f1": float(best_row["F1-Score"]),
    }


def plot_hidden_sizes_learning_rate_f1_heatmap(
    experiment_dir=None,
    mark_best=False,
    file_stem="bayesian_hidden_sizes_learning_rate_f1_heatmap",
):
    """Plot mean validation F1 by hidden_sizes and learning_rate.

    The heatmap uses Validation rows from final_summary.xlsx. Each row is the
    five-fold mean validation F1 for one Bayesian-search trial. Each heatmap
    cell is the average of those trial-level mean F1 scores for a
    hidden_sizes + learning_rate pair.
    """
    if experiment_dir is None:
        experiment_dir = find_latest_experiment_dir()
    experiment_dir = Path(experiment_dir)
    summary_path = experiment_dir / "final_summary.xlsx"
    figures_dir = _prepare_output_dir(experiment_dir)

    val_df = _load_validation_summary(summary_path)
    heatmap_df = _build_mean_f1_heatmap(val_df)

    masked_values = np.ma.masked_invalid(heatmap_df.to_numpy(dtype=float))
    cmap = plt.colormaps["YlGnBu"].copy()
    cmap.set_bad(color="#F2F2F2")

    fig_width = max(7.2, 0.9 * len(heatmap_df.columns) + 2.5)
    fig_height = max(5.2, 0.55 * len(heatmap_df.index) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(masked_values, cmap=cmap, aspect="auto", vmin=0.70, vmax=0.80)

    ax.set_xticks(np.arange(len(heatmap_df.columns)))
    ax.set_xticklabels([_format_learning_rate(col) for col in heatmap_df.columns])
    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_yticklabels([_format_hidden_sizes(row) for row in heatmap_df.index])
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Hidden sizes")
    ax.set_title("Mean Validation F1-score by Hidden Sizes and Learning Rate")

    for row_idx, row_name in enumerate(heatmap_df.index):
        for col_idx, col_name in enumerate(heatmap_df.columns):
            value = heatmap_df.loc[row_name, col_name]
            if pd.isna(value):
                continue
            text_color = "white" if value >= 0.765 else "#222222"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    if mark_best:
        best = _load_best_validation_point(summary_path)
        if (
            best["hidden_sizes"] in heatmap_df.index
            and best["learning_rate"] in heatmap_df.columns
        ):
            row_idx = list(heatmap_df.index).index(best["hidden_sizes"])
            col_idx = list(heatmap_df.columns).index(best["learning_rate"])
            ax.scatter(
                col_idx,
                row_idx,
                marker="*",
                s=260,
                color="#D62728",
                edgecolors="white",
                linewidths=0.8,
                zorder=4,
                label=f"Best validation model ({best['experiment']})",
            )
            ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, -0.12))

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean validation F1-score")
    ax.tick_params(axis="x", rotation=35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save_figure(fig, figures_dir, file_stem)
