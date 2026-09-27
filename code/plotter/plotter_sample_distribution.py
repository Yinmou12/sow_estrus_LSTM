import os
import sys

import numpy as np
import pandas as pd

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CODE_DIR)

import sow_estrus_LSTM_Function as myFunction
from sow_estrus_LSTM_Info import info_FINAL_SAVE_PATH

pd.set_option("future.no_silent_downcasting", True)

FIXED_SPLIT_DIR = os.path.join(info_FINAL_SAVE_PATH, "cross_validation")
FIXED_TRAIN_VAL_PATH = os.path.join(FIXED_SPLIT_DIR, "train_val_df.xlsx")
PICTURE_DIR = os.path.join(CODE_DIR, "pictures")
FIGURE_DPI = 600


def load_fixed_train_val_df():
    if not os.path.exists(FIXED_TRAIN_VAL_PATH):
        raise FileNotFoundError(
            f"Fixed train_val file not found: {FIXED_TRAIN_VAL_PATH}"
        )
    return pd.read_excel(FIXED_TRAIN_VAL_PATH, index_col=False)


def add_rate_features(flat_df):
    temp_feats = flat_df.iloc[:, 1:-1].copy()
    rate_feats = temp_feats.diff(axis=1).fillna(0)
    rate_feats.columns = [f"rate_{i}" for i in range(1, rate_feats.shape[1] + 1)]
    return pd.concat([flat_df.iloc[:, :-1], rate_feats, flat_df.iloc[:, -1]], axis=1)


def build_distribution_datasets(add_temp_rate=False):
    # Use the full fixed train_val split, not one CV fold.
    train_val_df = load_fixed_train_val_df()
    train_df = myFunction.fill_data(train_val_df)
    train_df_flat = myFunction.convert_features(train_df)

    df_min = train_df_flat[train_df_flat["isEstrus"] == 1]
    df_maj = train_df_flat[train_df_flat["isEstrus"] == 0]
    train_df_processed = myFunction.ADASYN(
        threshold=0.9, gamma=1, df_min=df_min, df_maj=df_maj
    )
    train_df_processed = myFunction.SMOTE(
        train_df_processed, amount_oversampling=800, k=7
    )
    train_df_processed = myFunction.TomekLinked(train_df_processed, k=1)

    if add_temp_rate:
        train_df_flat = add_rate_features(train_df_flat)
        train_df_processed = add_rate_features(train_df_processed)

    return train_df_flat, train_df_processed


def _plot_embedding(embedding, labels, title, xlabel, ylabel, file_stem):
    import matplotlib.pyplot as plt

    os.makedirs(PICTURE_DIR, exist_ok=True)

    plt.figure(figsize=(10, 7))
    plt.scatter(
        embedding[labels == 0, 0],
        embedding[labels == 0, 1],
        label="non-estrus",
        alpha=0.5,
        c="blue",
        s=10,
    )
    plt.scatter(
        embedding[labels == 1, 0],
        embedding[labels == 1, 1],
        label="estrus",
        alpha=0.5,
        c="red",
        s=10,
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    output_path = os.path.join(PICTURE_DIR, f"{file_stem}_{FIGURE_DPI}dpi.png")
    plt.savefig(output_path, format="png", dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    plt.close()


def analysis_PCA_old(df=None, add_temp_rate=False):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    train_df_flat, train_df_processed = build_distribution_datasets(add_temp_rate)

    def perform_pca_analysis(data, title, file_stem):
        X = data.iloc[:, 1:-1].values
        y = data.iloc[:, -1].values

        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        explained_var = pca.explained_variance_ratio_
        print(
            f"\n{title} - PCA explained variance: "
            f"PC1={explained_var[0]:.4f}, PC2={explained_var[1]:.4f}"
        )
        print(f"Cumulative explained variance: {np.sum(explained_var):.4f}")

        _plot_embedding(
            X_pca,
            y,
            f"PCA Visualization: {title}",
            "First Principal Component",
            "Second Principal Component",
            file_stem,
        )

    perform_pca_analysis(train_df_flat, "Original Data (train dataset)", "pca_original")
    perform_pca_analysis(train_df_processed, "ASHS (ADASYN+SMOTE+Tomek)", "pca_ashs")


def analysis_PCA_shared_projection(df=None, add_temp_rate=False):
    """Use one scaler and PCA fitted on original data for both datasets."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    train_df_flat, train_df_processed = build_distribution_datasets(add_temp_rate)

    X_original = train_df_flat.iloc[:, 1:-1].values
    y_original = train_df_flat.iloc[:, -1].values
    X_ashs = train_df_processed.iloc[:, 1:-1].values
    y_ashs = train_df_processed.iloc[:, -1].values

    scaler = StandardScaler().fit(X_original)
    X_original_scaled = scaler.transform(X_original)
    X_ashs_scaled = scaler.transform(X_ashs)

    pca = PCA(n_components=2).fit(X_original_scaled)
    X_original_pca = pca.transform(X_original_scaled)
    X_ashs_pca = pca.transform(X_ashs_scaled)

    explained_var = pca.explained_variance_ratio_
    print(
        "\nShared PCA fitted on original data - explained variance: "
        f"PC1={explained_var[0]:.4f}, PC2={explained_var[1]:.4f}"
    )
    print(f"Cumulative explained variance: {np.sum(explained_var):.4f}")

    _plot_embedding(
        X_original_pca,
        y_original,
        "PCA Visualization: Original Data (train dataset)",
        "First Principal Component",
        "Second Principal Component",
        "pca_original",
    )
    _plot_embedding(
        X_ashs_pca,
        y_ashs,
        "PCA Visualization: ASHS (ADASYN+SMOTE+Tomek)",
        "First Principal Component",
        "Second Principal Component",
        "pca_ashs",
    )


def analysis_tsne_old(df=None, add_temp_rate=False):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    train_df_flat, train_df_processed = build_distribution_datasets(add_temp_rate)

    def perform_tsne_analysis(data, title, file_stem):
        X = data.iloc[:, 1:-1].values
        y = data.iloc[:, -1].values

        X_scaled = StandardScaler().fit_transform(X)
        tsne = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            random_state=123,
        )
        X_tsne = tsne.fit_transform(X_scaled)

        _plot_embedding(
            X_tsne,
            y,
            f"t-SNE Visualization: {title}",
            "t-SNE Dimension 1",
            "t-SNE Dimension 2",
            file_stem,
        )

    perform_tsne_analysis(
        train_df_flat, "Original Data (train dataset)", "tsne_original"
    )
    perform_tsne_analysis(train_df_processed, "ASHS (ADASYN+SMOTE+Tomek)", "tsne_ashs")


def analysis_tsne_shared_embedding(df=None, add_temp_rate=False):
    """Embed original and ASHS data together in one t-SNE coordinate system."""
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    train_df_flat, train_df_processed = build_distribution_datasets(add_temp_rate)

    X_original = train_df_flat.iloc[:, 1:-1].values
    y_original = train_df_flat.iloc[:, -1].values
    X_ashs = train_df_processed.iloc[:, 1:-1].values
    y_ashs = train_df_processed.iloc[:, -1].values

    scaler = StandardScaler().fit(X_original)
    X_original_scaled = scaler.transform(X_original)
    X_ashs_scaled = scaler.transform(X_ashs)

    X_combined = np.vstack([X_original_scaled, X_ashs_scaled])
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        random_state=123,
    )
    X_combined_tsne = tsne.fit_transform(X_combined)

    original_count = len(X_original_scaled)
    X_original_tsne = X_combined_tsne[:original_count]
    X_ashs_tsne = X_combined_tsne[original_count:]

    _plot_embedding(
        X_original_tsne,
        y_original,
        "t-SNE Visualization: Original Data (train dataset)",
        "t-SNE Dimension 1",
        "t-SNE Dimension 2",
        "tsne_original",
    )
    _plot_embedding(
        X_ashs_tsne,
        y_ashs,
        "t-SNE Visualization: ASHS (ADASYN+SMOTE+Tomek)",
        "t-SNE Dimension 1",
        "t-SNE Dimension 2",
        "tsne_ashs",
    )


if __name__ == "__main__":
    # analysis_PCA_old(add_temp_rate=True)
    # analysis_tsne_old(add_temp_rate=True)
    # analysis_PCA_shared_projection(add_temp_rate=True)
    analysis_tsne_shared_embedding(add_temp_rate=True)
