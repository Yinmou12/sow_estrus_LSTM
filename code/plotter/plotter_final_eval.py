from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve

from sow_estrus_LSTM_Info import result_save_path

ALL_EXPERIMENTS_ROOT = Path(result_save_path) / "all_experiments"
FINAL_EVAL_PREFIX = "09_final_model_test_evaluation"
FIGURE_DPI = 1000


def find_latest_final_eval_dir(root=ALL_EXPERIMENTS_ROOT):
    """Return the newest final evaluation subdirectory with predictions."""
    root = Path(root)
    candidates = [
        path
        for path in root.glob(f"{FINAL_EVAL_PREFIX}_*/*")
        if path.is_dir() and (path / "final_test_predictions.xlsx").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No final_test_predictions.xlsx found under {root}/{FINAL_EVAL_PREFIX}_*."
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


def _load_predictions(experiment_dir):
    prediction_path = Path(experiment_dir) / "final_test_predictions.xlsx"
    df = pd.read_excel(prediction_path)
    required_cols = {"y_true", "y_prob", "y_pred"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"{prediction_path} is missing columns: {sorted(missing)}")
    return df, prediction_path


def plot_final_test_roc_curve(experiment_dir=None):
    if experiment_dir is None:
        experiment_dir = find_latest_final_eval_dir()
    experiment_dir = Path(experiment_dir)
    figures_dir = _prepare_output_dir(experiment_dir)
    df, _ = _load_predictions(experiment_dir)

    y_true = df["y_true"].astype(int).to_numpy()
    y_prob = df["y_prob"].astype(float).to_numpy()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(fpr, tpr, color="#1F77B4", linewidth=2.2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#777777", linewidth=1.2, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve on Independent Test Set")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return _save_figure(fig, figures_dir, "final_test_roc_curve")


def plot_final_test_confusion_matrix(experiment_dir=None, normalize=False):
    if experiment_dir is None:
        experiment_dir = find_latest_final_eval_dir()
    experiment_dir = Path(experiment_dir)
    figures_dir = _prepare_output_dir(experiment_dir)
    df, _ = _load_predictions(experiment_dir)

    y_true = df["y_true"].astype(int).to_numpy()
    y_pred = df["y_pred"].astype(int).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display_cm = cm.astype(float)
    if normalize:
        row_sums = display_cm.sum(axis=1, keepdims=True)
        display_cm = np.divide(
            display_cm,
            row_sums,
            out=np.zeros_like(display_cm),
            where=row_sums != 0,
        )

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    image = ax.imshow(display_cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix on Independent Test Set")

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            value_text = (
                f"{display_cm[row, col]:.2f}" if normalize else str(cm[row, col])
            )
            ax.text(
                col,
                row,
                value_text,
                ha="center",
                va="center",
                color=(
                    "white" if display_cm[row, col] > display_cm.max() / 2 else "black"
                ),
            )

    file_stem = (
        "final_test_confusion_matrix_normalized"
        if normalize
        else "final_test_confusion_matrix"
    )
    return _save_figure(fig, figures_dir, file_stem)


def plot_final_test_evaluation_figures(experiment_dir=None):
    if experiment_dir is None:
        experiment_dir = find_latest_final_eval_dir()
    outputs = []
    outputs.extend(plot_final_test_roc_curve(experiment_dir))
    outputs.extend(plot_final_test_confusion_matrix(experiment_dir, normalize=False))
    return outputs


def _infer_curve_label(experiment_dir):
    """Infer a compact curve label from a final-evaluation directory name."""
    name = Path(experiment_dir).name
    label = name
    for prefix in ("Final_", "final_"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
    for suffix in ("_AST", "_test", "_eval", "_evaluation"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]

    aliases = {
        "EstrusLSTM": "LSTM",
        "EstrusGRU": "GRU",
        "EstrusRNN": "RNN",
        "EstrusRNN_sample": "RNN_sample",
    }
    return aliases.get(label, label)


def _find_latest_prediction_dirs_by_label(root=ALL_EXPERIMENTS_ROOT, max_curves=4):
    """Find latest prediction directories and keep one newest directory per label."""
    prediction_paths = sorted(
        Path(root).glob("**/final_test_predictions.xlsx"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    selected = {}
    for prediction_path in prediction_paths:
        experiment_dir = prediction_path.parent
        label = _infer_curve_label(experiment_dir)
        if label not in selected:
            selected[label] = experiment_dir
        if len(selected) >= max_curves:
            break

    preferred_order = ["BiLSTM", "LSTM", "GRU", "RNN", "RNN_sample"]
    ordered = [
        (label, selected[label]) for label in preferred_order if label in selected
    ]
    ordered.extend(
        (label, path)
        for label, path in selected.items()
        if label not in preferred_order
    )
    return ordered[:max_curves]


def plot_combined_final_test_roc_curves(
    experiment_dirs=None,
    labels=None,
    output_dir=None,
    file_stem="combined_final_test_roc_curves",
):
    """
    Plot four final-test ROC curves in one figure.

    Parameters
    ----------
    experiment_dirs : list[str | Path] or dict[str, str | Path], optional
        Directories that contain final_test_predictions.xlsx. If omitted, the
        function searches all_experiments and uses the newest four labels.
    labels : list[str], optional
        Curve labels used when experiment_dirs is a list.
    output_dir : str | Path, optional
        Directory used to save the combined figure. Defaults to
        all_experiments/combined_figures.
    file_stem : str
        Output filename without extension.
    """
    if experiment_dirs is None:
        curve_items = _find_latest_prediction_dirs_by_label(max_curves=4)
    elif isinstance(experiment_dirs, dict):
        curve_items = [
            (str(label), Path(experiment_dir))
            for label, experiment_dir in experiment_dirs.items()
        ]
    else:
        curve_dirs = [Path(experiment_dir) for experiment_dir in experiment_dirs]
        if labels is None:
            labels = [
                _infer_curve_label(experiment_dir) for experiment_dir in curve_dirs
            ]
        if len(labels) != len(curve_dirs):
            raise ValueError("labels and experiment_dirs must have the same length.")
        curve_items = list(zip(labels, curve_dirs))

    if len(curve_items) != 4:
        raise ValueError(f"Expected 4 ROC curves, got {len(curve_items)}.")

    figures_dir = (
        Path(output_dir)
        if output_dir is not None
        else ALL_EXPERIMENTS_ROOT / "combined_figures"
    )
    figures_dir.mkdir(parents=True, exist_ok=True)

    colors = ["#5F7F95", "#B66A5A", "#7E9A6D", "#8B6F9E"]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))

    for (label, experiment_dir), color in zip(curve_items, colors):
        df, _ = _load_predictions(experiment_dir)
        y_true = df["y_true"].astype(int).to_numpy()
        y_prob = df["y_prob"].astype(float).to_numpy()
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color=color,
            linewidth=1.5,
            label=f"{label} (AUC = {roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], color="#777777", linewidth=1.2, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves on Independent Test Set")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return _save_figure(fig, figures_dir, file_stem)
