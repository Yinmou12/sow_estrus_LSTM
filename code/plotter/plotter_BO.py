import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sow_estrus_LSTM_Info import result_save_path

ALL_EXPERIMENTS_ROOT = Path(result_save_path) / "all_experiments"
BO_EXPERIMENT_PREFIX = "07_bilstm_ast_bayesian_tuning"

FIGURE_DPI = 1000
LINE_WIDTH = 2.2
MARKER_SIZE = 5


def find_latest_experiment_dir(prefix=BO_EXPERIMENT_PREFIX, root=ALL_EXPERIMENTS_ROOT):
    """Return the newest experiment result directory matching the given prefix."""
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


def _validation_summary_from_final_summary(summary_path):
    df = pd.read_excel(summary_path)
    _require_columns(
        df,
        ["Dataset", "param_bayes_iteration", "F1-Score"],
        summary_path,
    )
    val_df = df[df["Dataset"] == "Validation"].copy()
    val_df["param_bayes_iteration"] = pd.to_numeric(
        val_df["param_bayes_iteration"], errors="coerce"
    )
    val_df = val_df.dropna(subset=["param_bayes_iteration", "F1-Score"])
    val_df["param_bayes_iteration"] = val_df["param_bayes_iteration"].astype(int)
    val_df = val_df.sort_values("param_bayes_iteration")
    if val_df.empty:
        raise ValueError(f"No Validation rows were found in {summary_path}.")
    return val_df


def _validation_fold_summary(details_path):
    df = pd.read_excel(details_path)
    _require_columns(
        df,
        ["Dataset", "param_bayes_iteration", "Fold", "F1-Score"],
        details_path,
    )
    val_df = df[df["Dataset"] == "Validation"].copy()
    val_df["param_bayes_iteration"] = pd.to_numeric(
        val_df["param_bayes_iteration"], errors="coerce"
    )
    val_df = val_df.dropna(subset=["param_bayes_iteration", "F1-Score"])
    val_df["param_bayes_iteration"] = val_df["param_bayes_iteration"].astype(int)
    if val_df.empty:
        raise ValueError(f"No Validation fold rows were found in {details_path}.")

    summary_df = (
        val_df.groupby("param_bayes_iteration", as_index=False)["F1-Score"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_df["std"] = summary_df["std"].fillna(0.0)
    return summary_df.sort_values("param_bayes_iteration")


def _style_axis(ax):
    ax.set_xlabel("Bayesian optimization iteration")
    ax.set_ylabel("Validation F1-Score")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)


def plot_bayesian_convergence_basic(experiment_dir=None):
    """Plot iteration F1 and best-so-far F1 from final_summary.xlsx."""
    if experiment_dir is None:
        experiment_dir = find_latest_experiment_dir()
    experiment_dir = Path(experiment_dir)
    summary_path = experiment_dir / "final_summary.xlsx"
    figures_dir = _prepare_output_dir(experiment_dir)

    val_df = _validation_summary_from_final_summary(summary_path)
    x = val_df["param_bayes_iteration"]
    y = val_df["F1-Score"]
    best_so_far = y.cummax()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        x,
        y,
        marker="o",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        color="#1F77B4",
        label="Current iteration F1",
    )
    ax.plot(
        x,
        best_so_far,
        marker="s",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        color="#D62728",
        label="Best-so-far F1",
    )
    ax.set_title("Bayesian Optimization Convergence")
    _style_axis(ax)
    return _save_figure(fig, figures_dir, "bayesian_convergence_f1_basic")


def plot_bayesian_convergence_with_std(experiment_dir=None):
    """Plot mean +/- std F1 and best-so-far F1 from fold-level details."""
    if experiment_dir is None:
        experiment_dir = find_latest_experiment_dir()
    experiment_dir = Path(experiment_dir)
    details_path = experiment_dir / "all_experiments_cv_details.xlsx"
    figures_dir = _prepare_output_dir(experiment_dir)

    summary_df = _validation_fold_summary(details_path)
    x = summary_df["param_bayes_iteration"]
    mean_f1 = summary_df["mean"]
    std_f1 = summary_df["std"]
    best_so_far = mean_f1.cummax()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        x,
        mean_f1,
        marker="o",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        color="#2CA02C",
        label="Mean F1 across folds",
    )
    ax.fill_between(
        x,
        mean_f1 - std_f1,
        mean_f1 + std_f1,
        color="#2CA02C",
        alpha=0.16,
        linewidth=0,
        label="Mean +/- std",
    )
    ax.plot(
        x,
        best_so_far,
        marker="s",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        color="#9467BD",
        label="Best-so-far mean F1",
    )
    ax.set_title("Bayesian Optimization Convergence with Fold Variability")
    _style_axis(ax)
    return _save_figure(fig, figures_dir, "bayesian_convergence_f1_std")


def plot_bayesian_convergence_figures(experiment_dir=None):
    """Create both Bayesian optimization convergence figures."""
    if experiment_dir is None:
        experiment_dir = find_latest_experiment_dir()
    experiment_dir = Path(experiment_dir)
    outputs = []
    outputs.extend(plot_bayesian_convergence_basic(experiment_dir))
    outputs.extend(plot_bayesian_convergence_with_std(experiment_dir))
    return outputs
