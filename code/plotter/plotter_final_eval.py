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
            value_text = f"{display_cm[row, col]:.2f}" if normalize else str(cm[row, col])
            ax.text(
                col,
                row,
                value_text,
                ha="center",
                va="center",
                color="white" if display_cm[row, col] > display_cm.max() / 2 else "black",
            )

    file_stem = "final_test_confusion_matrix_normalized" if normalize else "final_test_confusion_matrix"
    return _save_figure(fig, figures_dir, file_stem)


def plot_final_test_evaluation_figures(experiment_dir=None):
    if experiment_dir is None:
        experiment_dir = find_latest_final_eval_dir()
    outputs = []
    outputs.extend(plot_final_test_roc_curve(experiment_dir))
    outputs.extend(plot_final_test_confusion_matrix(experiment_dir, normalize=False))
    return outputs
