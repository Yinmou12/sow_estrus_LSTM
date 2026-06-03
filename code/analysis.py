import sys
import os

# 将code目录添加到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sow_estrus_LSTM_Function as myFunction

from sow_estrus_LSTM_Info import *

import joblib

import pandas as pd
import numpy as np

from datetime import datetime

pd.set_option("future.no_silent_downcasting", True)


def analysis_PCA(df, add_temp_rate=False):
    independent_test_df, folds = myFunction.stratified_group_kfold(df, n_splits=5)
    train_df_raw, val_df = folds[0]
    train_df = myFunction.fill_data(train_df_raw)
    train_df_flat = myFunction.convert_features(train_df)

    train_df_processed = train_df_flat.copy()
    df_min = train_df_flat[train_df_flat["isEstrus"] == 1]
    df_maj = train_df_flat[train_df_flat["isEstrus"] == 0]
    train_df_processed = myFunction.ADASYN(
        threshold=0.5, gamma=1, df_min=df_min, df_maj=df_maj
    )
    train_df_processed = myFunction.SMOTE(
        train_df_processed, amount_oversampling=800, k=7
    )
    train_df_processed = myFunction.TomekLinked(train_df_processed, k=1)

    if add_temp_rate:
        temp_feats = train_df_processed.iloc[:, 1:-1].copy()
        rate_feats = temp_feats.diff(axis=1).fillna(0)
        rate_feats.columns = [f"rate_{i}" for i in range(1, 49)]
        train_df_processed = pd.concat(
            [
                train_df_processed.iloc[:, :-1],
                rate_feats,
                train_df_processed.iloc[:, -1],
            ],
            axis=1,
        )

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    def perform_pca_analysis(data, title):
        X = data.iloc[:, 1:-1].values
        y = data.iloc[:, -1].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 拟合 PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # 打印方差解释率
        explained_var = pca.explained_variance_ratio_
        print(
            f"\n{title} - PCA 解释方差比例: PC1={explained_var[0]:.4f}, PC2={explained_var[1]:.4f}"
        )
        print(f"累计解释方差: {np.sum(explained_var):.4f}")

        # 可视化
        plt.figure(figsize=(10, 7))
        plt.scatter(
            X_pca[y == 0, 0],
            X_pca[y == 0, 1],
            label="Not Estrus",
            alpha=0.5,
            c="blue",
            s=10,
        )
        plt.scatter(
            X_pca[y == 1, 0], X_pca[y == 1, 1], label="Estrus", alpha=0.5, c="red", s=10
        )
        plt.title(f"PCA Visualization: {title}")
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.legend()
        plt.show()

    # 执行分析
    perform_pca_analysis(train_df_flat, "Original Data (train_df_flat)")
    perform_pca_analysis(train_df_processed, "Augmented Data (train_df_processed)")

    return None


def analysis_tsne(df, add_temp_rate=False):
    independent_test_df, folds = myFunction.stratified_group_kfold(df, n_splits=5)
    train_df_raw, val_df = folds[0]
    train_df = myFunction.fill_data(train_df_raw)
    train_df_flat = myFunction.convert_features(train_df)

    train_df_processed = train_df_flat.copy()
    df_min = train_df_flat[train_df_flat["isEstrus"] == 1]
    df_maj = train_df_flat[train_df_flat["isEstrus"] == 0]
    train_df_processed = myFunction.ADASYN(
        threshold=0.5, gamma=1, df_min=df_min, df_maj=df_maj
    )
    train_df_processed = myFunction.SMOTE(
        train_df_processed, amount_oversampling=800, k=7
    )
    train_df_processed = myFunction.TomekLinked(train_df_processed, k=1)

    if add_temp_rate:
        temp_feats = train_df_processed.iloc[:, 1:-1].copy()
        rate_feats = temp_feats.diff(axis=1).fillna(0)
        rate_feats.columns = [f"rate_{i}" for i in range(1, 49)]
        train_df_processed = pd.concat(
            [
                train_df_processed.iloc[:, :-1],
                rate_feats,
                train_df_processed.iloc[:, -1],
            ],
            axis=1,
        )

    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    def perform_tsne_analysis(data, title):
        X = data.iloc[:, 1:-1].values
        y = data.iloc[:, -1].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 拟合 t-SNE
        # 注意：t-SNE 计算量较大。设置 init='pca' 可以提高稳定性。
        tsne = TSNE(
            n_components=2,
            random_state=42,
            init="pca",
            learning_rate="auto",
            random_state=123,
        )
        X_tsne = tsne.fit_transform(X_scaled)

        # 可视化
        plt.figure(figsize=(10, 7))
        plt.scatter(
            X_tsne[y == 0, 0],
            X_tsne[y == 0, 1],
            label="Not Estrus",
            alpha=0.5,
            c="blue",
            s=10,
        )
        plt.scatter(
            X_tsne[y == 1, 0],
            X_tsne[y == 1, 1],
            label="Estrus",
            alpha=0.5,
            c="red",
            s=10,
        )
        plt.title(f"t-SNE Visualization: {title}")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend()
        plt.show()

    # 执行分析
    perform_tsne_analysis(train_df_flat, "Original Data (train_df_flat)")
    perform_tsne_analysis(train_df_processed, "Augmented Data (train_df_processed)")

    return None


if __name__ == "__main__":
    # 加载数据的示例
    df = pd.read_excel(
        os.path.join(
            experimentRecord_data_path,
            "splited_dataset",
            "splited_dataset_2026_0406_1140.xlsx",
        ),
        index_col=False,
    )

    # analysis_PCA(df, add_temp_rate=True)
    analysis_tsne(df, add_temp_rate=True)
