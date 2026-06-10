"""
交叉验证示例
"""

import sys
import os

# 将code目录添加到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sow_estrus_LSTM_Function as myFunction
from lstm_model import EstrusLSTM, EarlyStopping
from sow_estrus_LSTM_Info import *
from sow_estrus_LSTM_train import __train_info as TrainInfo, get_model
from sow_estrus_LSTM_train import EstrusDataset, evaluate_model
from torch.utils.data import DataLoader
import joblib

import pandas as pd
import numpy as np
import random

from datetime import datetime

import torch
import torch.nn as nn

pd.set_option("future.no_silent_downcasting", True)

# 消融实验默认配置
""" ABLATION_CONFIGS = [
    {"name": "Baseline", "A": False, "S": False, "T": False},
    {"name": "ADASYN", "A": True, "S": False, "T": False},
    {"name": "SMOTE", "A": False, "S": True, "T": False},
    {"name": "TomekLinks", "A": False, "S": False, "T": True},
    {"name": "ADASYN+SMOTE", "A": True, "S": True, "T": False},
    {"name": "ADASYN+TomekLinks", "A": True, "S": False, "T": True},
    {"name": "SMOTE+TomekLinks", "A": False, "S": True, "T": True},
    {"name": "Full_Aug", "A": True, "S": True, "T": True},
] """

""" ABLATION_CONFIGS = [
    {"name": "Baseline", "S": False, "T": False},
    {"name": "SMOTE_Tomek", "S": True, "T": True},
] """

ABLATION_CONFIGS = [
    {"name": "AST", "T": True, "S": True, "T": True},
]


def run_5fold_cv(df):
    """
    运行5折交叉验证
    """
    # 使用新添加的分层分组K折划分函数
    independent_test_df, folds = myFunction.stratified_group_kfold(df, n_splits=5)

    timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
    saved_result_path = os.path.join(result_save_path, "cv", f"{timestamp}")
    os.makedirs(saved_result_path, exist_ok=True)

    independent_test_df.to_excel(
        os.path.join(saved_result_path, "independent_test_df.xlsx"), index=False
    )

    results = {}

    for fold_idx, (train_df_raw, val_df_raw) in enumerate(folds):
        print(f"\n{'='*20} Fold {fold_idx + 1} {'='*20}")

        # 数据预处理
        train_df = myFunction.fill_data(train_df_raw)
        val_df = myFunction.fill_data(val_df_raw)

        # 准备数据格式
        X_train, y_train, scaler = myFunction.prepare_lstm_data(train_df)
        # 保存缩放器以便后续评估独立测试集
        scaler_path = os.path.join(
            saved_result_path, f"scaler_fold{fold_idx + 1}.joblib"
        )
        joblib.dump(scaler, scaler_path)
        print(f"训练集 X_train 形状: {X_train.shape}, y_train 形状: {y_train.shape}")
        X_val, y_val, _ = myFunction.prepare_lstm_data(val_df, scaler=scaler)
        print(f"验证集 X_val 形状: {X_val.shape}, y_val 形状: {y_val.shape}")

        # 模型训练
        train_info = TrainInfo()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(train_info, device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=train_info.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=train_info.lr_patience
        )
        batch_size = train_info.batch_size
        train_loader = DataLoader(
            EstrusDataset(X_train, y_train), batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(EstrusDataset(X_val, y_val), batch_size)

        num_epochs = train_info.num_epochs
        early_patience = train_info.early_patience

        best_model_path = os.path.join(
            saved_result_path, f"best_model_fold{fold_idx + 1}.pth"
        )
        early_stopping = EarlyStopping(
            patience=early_patience,
            verbose=True,
            path=best_model_path,
        )

        for epoch in range(num_epochs):
            model.train()
            # epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                # epoch_loss += loss.item() * batch_X.size(0)

            # epoch_loss /= len(train_loader.dataset)
            # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
            val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)
            epoch_loss = val_metrics["avg_loss"]
            scheduler.step(epoch_loss)
            early_stopping(epoch_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered. Ending training.")
                break

        # 训练结束后，加载该折保存的最佳模型权重进行评估
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        print(f"已加载最佳模型权重: {best_model_path}")

        record_dict, _, _ = evaluate_model(
            model,
            DataLoader(
                EstrusDataset(X_val, y_val),
                batch_size,
                shuffle=False,
                drop_last=False,
            ),
            criterion,
            device,
        )
        record_dict.pop("avg_loss", None)  # 移除平均损失
        record_dict.pop("MCC", None)  # 移除MCC指标
        results[f"Fold_{fold_idx + 1}"] = record_dict
        print(f"Fold {fold_idx + 1} 完成")

    # 计算平均指标
    avg_results = {}
    for key in next(iter(results.values())).keys():
        avg_results[key] = np.mean(
            [fold_result[key] for fold_result in results.values()]
        )
    print(f"\nCV 平均结果: {avg_results}")

    # 保存平均结果到Excel
    summary_df = pd.DataFrame(results).T
    summary_df.loc["Average"] = avg_results
    summary_df.to_excel(os.path.join(saved_result_path, "cv_summary.xlsx"))

    return saved_result_path


class DataAugInfo:
    use_adasyn: bool = False
    ada_threshold: float = 0.5
    ada_gamma: float = 1
    ada_k: int = 5

    use_smote: bool = True
    smote_amount_oversampling: int = 800
    smote_k_neighbors: int = 7

    use_tomek: bool = True
    tomek_k: int = 1


# 数据增强
# 单耳温特征
def run_5fold_cv_with_aug_temp(df):
    """
    运行5折交叉验证，包含8种数据增强组合的消融实验
    """
    independent_test_df, folds = myFunction.stratified_group_kfold(df, n_splits=5)

    configs = ABLATION_CONFIGS
    train_info = TrainInfo()
    data_aug_info = DataAugInfo()

    timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
    saved_result_path = os.path.join(result_save_path, "cv_ablation", f"{timestamp}")
    os.makedirs(saved_result_path, exist_ok=True)

    independent_test_df.to_excel(
        os.path.join(saved_result_path, "independent_test_df.xlsx"), index=False
    )

    cv_all_exp_results = {}
    all_detailed_results = []

    for config in configs:
        exp_name = config["name"]
        print(f"\n开始消融实验: {exp_name} " + "*" * 30)

        cv_ablation_result_path = os.path.join(saved_result_path, exp_name)
        os.makedirs(cv_ablation_result_path, exist_ok=True)

        fold_metrics = []
        fold_results_dict = {}

        for fold_idx, (train_df_raw, val_df_raw) in enumerate(folds):
            print(f"\n--- {exp_name} | Fold {fold_idx + 1} ---")

            # 基础预处理
            train_df = myFunction.fill_data(train_df_raw, balanced_data=True, stride=6)
            val_df = myFunction.fill_data(val_df_raw)

            # 数据增强 (仅针对训练集)
            # 先转换为特征矩阵格式以便增强函数处理
            train_df_flat = myFunction.convert_features(train_df)

            if config["A"]:
                df_min = train_df_flat[train_df_flat["isEstrus"] == 1]
                df_maj = train_df_flat[train_df_flat["isEstrus"] == 0]
                train_df_flat = myFunction.ADASYN(
                    threshold=data_aug_info.ada_threshold,
                    gamma=data_aug_info.ada_gamma,
                    df_min=df_min,
                    df_maj=df_maj,
                )

            if config["S"]:
                train_df_flat = myFunction.SMOTE(
                    train_df_flat,
                    amount_oversampling=data_aug_info.smote_amount_oversampling,
                    k=data_aug_info.smote_k_neighbors,
                )

            if config["T"]:
                train_df_flat = myFunction.TomekLinked(
                    train_df_flat, k=data_aug_info.tomek_k
                )

            # 准备训练和验证数据 (仅使用耳温单特征)
            X_train, y_train, scaler = myFunction.prepare_univariate_lstm_data(
                train_df_flat
            )
            # 保存缩放器
            scaler_path = os.path.join(
                cv_ablation_result_path, f"scaler_fold{fold_idx + 1}.joblib"
            )
            joblib.dump(scaler, scaler_path)
            X_val, y_val, _ = myFunction.prepare_univariate_lstm_data(
                val_df, scaler=scaler
            )

            # 模型训练
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = get_model(train_info, device, input_size=1)

            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=train_info.learning_rate, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=train_info.lr_patience
            )

            train_loader = DataLoader(
                EstrusDataset(X_train, y_train),
                train_info.batch_size,
                shuffle=True,
                drop_last=True,
            )
            val_loader = DataLoader(EstrusDataset(X_val, y_val), train_info.batch_size)

            best_model_path = os.path.join(
                cv_ablation_result_path, f"best_model_fold{fold_idx + 1}.pth"
            )
            early_stopping = EarlyStopping(
                patience=train_info.early_patience, verbose=False, path=best_model_path
            )

            for epoch in range(train_info.num_epochs):
                model.train()
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(batch_X), batch_y)
                    loss.backward()
                    optimizer.step()

                # 评估
                val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)
                scheduler.step(val_metrics["avg_loss"])
                early_stopping(val_metrics["avg_loss"], model)
                if early_stopping.early_stop:
                    break

            # 6. 加载最佳并记录
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
            final_rec, _, _ = evaluate_model(model, val_loader, criterion, device)
            final_rec.pop("avg_loss", None)
            final_rec.pop("MCC", None)
            fold_metrics.append(final_rec)
            fold_results_dict[f"Fold_{fold_idx + 1}"] = final_rec
            all_detailed_results.append(
                {"Experiment": exp_name, "Fold": f"Fold_{fold_idx + 1}", **final_rec}
            )
            print(f"Fold {fold_idx + 1} 结果: {final_rec}")

        # 计算该实验的5折平均
        avg_exp_metrics = {}
        for m_key in fold_metrics[0].keys():
            avg_exp_metrics[m_key] = np.mean([f[m_key] for f in fold_metrics])

        # 参考 run_5fold_cv 的保存方式，为当前增强方式保存详细折结果汇总
        exp_summary_df = pd.DataFrame(fold_results_dict).T
        exp_summary_df.loc["Average"] = avg_exp_metrics
        exp_summary_df.to_excel(
            os.path.join(cv_ablation_result_path, f"{exp_name}_cv_summary.xlsx")
        )

        all_detailed_results.append(
            {"Experiment": exp_name, "Fold": "Average", **avg_exp_metrics}
        )
        cv_all_exp_results[exp_name] = avg_exp_metrics
        print(f"\n>>> 实验 {exp_name} 平均结果: {avg_exp_metrics}")

    # 汇总打印所有实验
    print("\n" + "=" * 50)
    print("消融实验 5-Fold CV 最终汇总结果")
    print("=" * 50)
    summary_df = pd.DataFrame(cv_all_exp_results).T
    print(summary_df)

    # 保存到Excel
    summary_df.to_excel(os.path.join(saved_result_path, "final_summary.xlsx"))

    # 将所有数据增强方式的验证集结果进行汇总保存在一张 Excel 表格中
    all_detailed_df = pd.DataFrame(all_detailed_results)
    all_detailed_df.to_excel(
        os.path.join(saved_result_path, "all_experiments_cv_details.xlsx"), index=False
    )

    return saved_result_path, configs


# 数据增强
def run_5fold_cv_with_aug(df):
    """
    运行5折交叉验证，包含8种数据增强组合的消融实验
    """
    independent_test_df, folds = myFunction.stratified_group_kfold(df, n_splits=5)

    configs = ABLATION_CONFIGS
    data_aug_info = DataAugInfo()

    timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
    saved_result_path = os.path.join(
        result_save_path, "cv_ablation", "ST_temp_rate", f"{timestamp}"
    )
    os.makedirs(saved_result_path, exist_ok=True)

    independent_test_df.to_excel(
        os.path.join(saved_result_path, "independent_test_df.xlsx"), index=False
    )

    cv_all_exp_results = {}
    all_detailed_results = []

    for config in configs:
        exp_name = config["name"]
        print(f"\n开始消融实验: {exp_name} " + "*" * 30)

        cv_ablation_result_path = os.path.join(saved_result_path, exp_name)
        os.makedirs(cv_ablation_result_path, exist_ok=True)

        fold_metrics = []
        fold_results_dict = {}

        for fold_idx, (train_df_raw, val_df_raw) in enumerate(folds):
            print(f"\n--- {exp_name} | Fold {fold_idx + 1} ---")

            # 基础预处理
            train_df = myFunction.fill_data(train_df_raw)
            print("+" * 60)
            val_df = myFunction.fill_data(val_df_raw)

            # 数据增强 (仅针对训练集)
            # 先转换为特征矩阵格式以便增强函数处理
            train_df_flat = myFunction.convert_features(train_df)

            if config["A"]:
                df_min = train_df_flat[train_df_flat["isEstrus"] == 1]
                df_maj = train_df_flat[train_df_flat["isEstrus"] == 0]
                train_df_flat = myFunction.ADASYN(
                    threshold=data_aug_info.ada_threshold,
                    gamma=1,
                    df_min=df_min,
                    df_maj=df_maj,
                )
                print(
                    f"ADASYN后样本总数:{train_df_flat['isEstrus'].shape[0]}, 正样本数:{train_df_flat[train_df_flat['isEstrus'] == 1].shape[0]}, 负样本数:{train_df_flat[train_df_flat['isEstrus'] == 0].shape[0]}"
                )

            if config["S"]:
                train_df_flat = myFunction.SMOTE(
                    train_df_flat, amount_oversampling=800, k=7
                )
                print(
                    f"SMOTE后样本总数:{train_df_flat['isEstrus'].shape[0]}, 正样本数:{train_df_flat[train_df_flat['isEstrus'] == 1].shape[0]}, 负样本数:{train_df_flat[train_df_flat['isEstrus'] == 0].shape[0]}"
                )

            if config["T"]:
                train_df_flat = myFunction.TomekLinked(train_df_flat, k=1)

            # 增加温度变化率特征
            temp_feats = train_df_flat.iloc[:, 1:-1].copy()
            rate_feats = temp_feats.diff(axis=1).fillna(0)
            rate_feats.columns = [f"rate_{i}" for i in range(1, 49)]
            # 合并：ID + 48温度 + 48变化率 + Label
            train_final_df = pd.concat(
                [train_df_flat.iloc[:, :-1], rate_feats, train_df_flat.iloc[:, -1]],
                axis=1,
            )

            # 准备训练和测试数据
            X_train, y_train, scaler = myFunction.prepare_lstm_data(train_final_df)
            # 保存缩放器
            scaler_path = os.path.join(
                cv_ablation_result_path, f"scaler_fold{fold_idx + 1}.joblib"
            )
            joblib.dump(scaler, scaler_path)

            X_val, y_val, _ = myFunction.prepare_lstm_data(val_df, scaler=scaler)

            # 模型训练
            train_info = TrainInfo()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = get_model(train_info, device)

            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=train_info.learning_rate, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=train_info.lr_patience
            )

            train_loader = DataLoader(
                EstrusDataset(X_train, y_train),
                train_info.batch_size,
                shuffle=True,
                drop_last=True,
            )
            val_loader = DataLoader(EstrusDataset(X_val, y_val), train_info.batch_size)

            best_model_path = os.path.join(
                cv_ablation_result_path, f"best_model_fold{fold_idx + 1}.pth"
            )
            early_stopping = EarlyStopping(
                patience=train_info.early_patience, verbose=False, path=best_model_path
            )

            for epoch in range(train_info.num_epochs):
                model.train()
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(batch_X), batch_y)
                    loss.backward()
                    optimizer.step()

                # 评估
                val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)
                scheduler.step(val_metrics["avg_loss"])
                early_stopping(val_metrics["avg_loss"], model)
                if early_stopping.early_stop:
                    break

            # 6. 加载最佳并记录
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
            final_rec, _, _ = evaluate_model(model, val_loader, criterion, device)
            final_rec.pop("avg_loss", None)
            # final_rec.pop("MCC", None)
            fold_metrics.append(final_rec)
            fold_results_dict[f"Fold_{fold_idx + 1}"] = final_rec
            all_detailed_results.append(
                {"Experiment": exp_name, "Fold": f"Fold_{fold_idx + 1}", **final_rec}
            )
            print(f"Fold {fold_idx + 1} 结果: {final_rec}")

        # 计算该实验的5折平均
        avg_exp_metrics = {}
        for m_key in fold_metrics[0].keys():
            avg_exp_metrics[m_key] = np.mean([f[m_key] for f in fold_metrics])

        # 参考 run_5fold_cv 的保存方式，为当前增强方式保存详细折结果汇总
        exp_summary_df = pd.DataFrame(fold_results_dict).T
        exp_summary_df.loc["Average"] = avg_exp_metrics
        exp_summary_df.to_excel(
            os.path.join(cv_ablation_result_path, f"{exp_name}_cv_summary.xlsx")
        )

        all_detailed_results.append(
            {"Experiment": exp_name, "Fold": "Average", **avg_exp_metrics}
        )
        cv_all_exp_results[exp_name] = avg_exp_metrics
        print(f"\n>>> 实验 {exp_name} 平均结果: {avg_exp_metrics}")

    # 汇总打印所有实验
    print("\n" + "=" * 50)
    print("消融实验 5-Fold CV 最终汇总结果")
    print("=" * 50)
    summary_df = pd.DataFrame(cv_all_exp_results).T
    print(summary_df)

    # 保存到Excel
    summary_df.to_excel(os.path.join(saved_result_path, "final_summary.xlsx"))

    # 将所有数据增强方式的验证集结果进行汇总保存在一张 Excel 表格中
    all_detailed_df = pd.DataFrame(all_detailed_results)
    all_detailed_df.to_excel(
        os.path.join(saved_result_path, "all_experiments_cv_details.xlsx"), index=False
    )

    return saved_result_path, configs


def evaluate_independent_test_set(base_path, configs=None, input_size=1):
    """
    加载交叉验证中保存的最佳模型和缩放器，并在独立测试集上进行最终评估。
    """
    print("\n" + "#" * 30)
    print("开始在独立测试集上进行最终评估")
    print("#" * 30)

    # 加载独立测试集
    df_path = os.path.join(base_path, "independent_test_df.xlsx")
    if not os.path.exists(df_path):
        print(f"错误: 在路径 {base_path} 中找不到 independent_test_df.xlsx")
        return

    independent_test_df = pd.read_excel(df_path)
    all_sows = independent_test_df["sSowsNo"].unique()
    estrus_sows = independent_test_df[independent_test_df["isEstrus"] == 1][
        "sSowsNo"
    ].unique()
    not_estrus_sows = [s for s in all_sows if s not in estrus_sows]
    print(f"发情母猪个数: {len(estrus_sows)}, 非发情母猪个数: {len(not_estrus_sows)}")

    # 基础预处理
    """ test_df_filled = myFunction.fill_data(
        independent_test_df, balanced_data=False, stride=6
    ) """
    test_df_filled = myFunction.fill_data(
        independent_test_df,
    )

    final_test_results = {}
    train_info = TrainInfo()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()

    # 确定评估任务列表
    # 如果没有传入 configs，则视为普通单次 5 折 CV 评估 (模型直接位于 base_path)
    if configs is None:
        eval_list = [{"name": "Standard_CV", "is_ablation": False}]
    else:
        eval_list = configs

    for config in eval_list:
        exp_name = config["name"]
        is_ablation = config.get("is_ablation", True)  # 默认假设是消融实验

        # 确定存放该组模型文件的根路径
        exp_root = os.path.join(base_path, exp_name) if is_ablation else base_path

        if not os.path.exists(exp_root):
            print(f"跳过: 路径不存在 -> {exp_root}")
            continue

        fold_metrics_list = []
        for fold_idx in range(1, 6):
            model_path = os.path.join(exp_root, f"best_model_fold{fold_idx}.pth")
            scaler_path = os.path.join(exp_root, f"scaler_fold{fold_idx}.joblib")

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                continue

            # 加载模型和缩放器
            model = get_model(train_info, device, input_size=input_size)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            scaler = joblib.load(scaler_path)

            # 准备数据并评估
            if input_size == 1:
                # 单特征评估使用 univariate 准备函数
                X_test, y_test, _ = myFunction.prepare_univariate_lstm_data(
                    test_df_filled, scaler=scaler
                )
            else:
                X_test, y_test, _ = myFunction.prepare_lstm_data(
                    test_df_filled, scaler=scaler
                )

            print(
                f"独立测试集 X_test 形状: {X_test.shape}, y_test 形状: {y_test.shape}"
            )
            print(f"正样本数: {(y_test == 1).sum()}, 负样本数: {(y_test == 0).sum()}")
            loader = DataLoader(
                EstrusDataset(X_test, y_test), train_info.batch_size, shuffle=False
            )

            metrics, _, _ = evaluate_model(model, loader, criterion, device)
            metrics.pop("avg_loss", None)
            # metrics.pop("MCC", None)
            fold_metrics_list.append(metrics)

        if fold_metrics_list:
            avg_metrics = {
                k: np.mean([m[k] for m in fold_metrics_list])
                for k in fold_metrics_list[0].keys()
            }
            key_name = "Independent_Test" if not is_ablation else exp_name
            final_test_results[key_name] = avg_metrics
            print(f"实验 {key_name} 独立测试集平均结果: {avg_metrics}")

    if final_test_results:
        summary_df = pd.DataFrame(final_test_results).T
        save_path = os.path.join(f"{base_path}_evaluate", "temp.xlsx")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        summary_df.to_excel(save_path)
        print(f"\n独立测试集评估完成，汇总已保存至: {save_path}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_n_times_cv_with_aug_and_eval(df, n=10):
    """
    执行n次run_5fold_cv_with_aug和evaluate_independent_test_set
    保留这两个函数的主要逻辑，并在统一目录下通过1, 2, ..., n保存文件，最终统一测试并输出结果。
    """
    timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
    base_save_path = os.path.join(
        result_save_path, "cv_ablation", "ST_temp_rate_n_times", f"{timestamp}"
    )
    os.makedirs(base_save_path, exist_ok=True)

    configs = ABLATION_CONFIGS
    data_aug_info = DataAugInfo()
    train_info = TrainInfo()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()

    random_seed = random.sample(range(1, 1000), n)
    # ==========================
    # 训练和验证阶段
    # ==========================
    for run_idx in range(1, n + 1):
        current_seed = random_seed[run_idx - 1]
        set_seed(current_seed)
        print(
            f"\n{'='*50}\nStarting Run {run_idx}/{n} (Seed: {current_seed})\n{'='*50}"
        )

        run_save_path = os.path.join(base_save_path, str(run_idx))
        os.makedirs(run_save_path, exist_ok=True)

        # 使用不同的随机数种子以确保每次运行的数据划分和模型初始化不同
        independent_test_df, folds = myFunction.stratified_group_kfold(
            df, n_splits=5, random_state=current_seed
        )

        independent_test_df.to_excel(
            os.path.join(run_save_path, "independent_test_df.xlsx"), index=False
        )

        cv_all_exp_results = {}
        all_detailed_results = []

        for config in configs:
            exp_name = config["name"]
            print(f"\n开始消融实验: {exp_name} " + "*" * 30)

            cv_ablation_result_path = os.path.join(run_save_path, exp_name)
            os.makedirs(cv_ablation_result_path, exist_ok=True)

            fold_metrics = []
            fold_results_dict = {}

            for fold_idx, (train_df_raw, val_df_raw) in enumerate(folds):
                print(f"\n--- {exp_name} | Fold {fold_idx + 1} ---")

                # 基础预处理
                train_df = myFunction.fill_data(train_df_raw)
                print("+" * 60)
                val_df = myFunction.fill_data(val_df_raw)

                # 数据增强 (仅针对训练集)
                train_df_flat = myFunction.convert_features(train_df)

                if config.get("A", False):
                    df_min = train_df_flat[train_df_flat["isEstrus"] == 1]
                    df_maj = train_df_flat[train_df_flat["isEstrus"] == 0]
                    train_df_flat = myFunction.ADASYN(
                        threshold=data_aug_info.ada_threshold,
                        gamma=1,
                        df_min=df_min,
                        df_maj=df_maj,
                    )
                    print(
                        f"ADASYN后样本总数:{train_df_flat['isEstrus'].shape[0]}, 正样本数:{train_df_flat[train_df_flat['isEstrus'] == 1].shape[0]}, 负样本数:{train_df_flat[train_df_flat['isEstrus'] == 0].shape[0]}"
                    )

                if config.get("S", False):
                    train_df_flat = myFunction.SMOTE(
                        train_df_flat, amount_oversampling=800, k=7
                    )
                    print(
                        f"SMOTE后样本总数:{train_df_flat['isEstrus'].shape[0]}, 正样本数:{train_df_flat[train_df_flat['isEstrus'] == 1].shape[0]}, 负样本数:{train_df_flat[train_df_flat['isEstrus'] == 0].shape[0]}"
                    )

                if config.get("T", False):
                    train_df_flat = myFunction.TomekLinked(train_df_flat, k=1)

                # 增加温度变化率特征
                temp_feats = train_df_flat.iloc[:, 1:-1].copy()
                rate_feats = temp_feats.diff(axis=1).fillna(0)
                rate_feats.columns = [f"rate_{i}" for i in range(1, 49)]
                train_final_df = pd.concat(
                    [train_df_flat.iloc[:, :-1], rate_feats, train_df_flat.iloc[:, -1]],
                    axis=1,
                )

                # 准备训练和测试数据
                X_train, y_train, scaler = myFunction.prepare_lstm_data(train_final_df)
                scaler_path = os.path.join(
                    cv_ablation_result_path, f"scaler_fold{fold_idx + 1}.joblib"
                )
                joblib.dump(scaler, scaler_path)

                X_val, y_val, _ = myFunction.prepare_lstm_data(val_df, scaler=scaler)

                # 模型训练
                model = get_model(train_info, device)

                optimizer = torch.optim.Adam(
                    model.parameters(), lr=train_info.learning_rate, weight_decay=1e-4
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=train_info.lr_patience
                )

                train_loader = DataLoader(
                    EstrusDataset(X_train, y_train),
                    train_info.batch_size,
                    shuffle=True,
                    drop_last=True,
                )
                val_loader = DataLoader(
                    EstrusDataset(X_val, y_val), train_info.batch_size
                )

                best_model_path = os.path.join(
                    cv_ablation_result_path, f"best_model_fold{fold_idx + 1}.pth"
                )
                early_stopping = EarlyStopping(
                    patience=train_info.early_patience,
                    verbose=False,
                    path=best_model_path,
                )

                for epoch in range(train_info.num_epochs):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        optimizer.zero_grad()
                        loss = criterion(model(batch_X), batch_y)
                        loss.backward()
                        optimizer.step()

                    # 评估
                    val_metrics, _, _ = evaluate_model(
                        model, val_loader, criterion, device
                    )
                    scheduler.step(val_metrics["avg_loss"])
                    early_stopping(val_metrics["avg_loss"], model)
                    if early_stopping.early_stop:
                        break

                # 加载最佳并记录
                model.load_state_dict(torch.load(best_model_path, weights_only=True))
                final_rec, _, _ = evaluate_model(model, val_loader, criterion, device)
                fold_metrics.append(final_rec)
                fold_results_dict[f"Fold_{fold_idx + 1}"] = final_rec
                all_detailed_results.append(
                    {
                        "Seed": current_seed,
                        "Experiment": exp_name,
                        "Fold": f"Fold_{fold_idx + 1}",
                        **final_rec,
                    }
                )
                print(f"Fold {fold_idx + 1} 结果: {final_rec}")

            avg_exp_metrics = {}
            for m_key in fold_metrics[0].keys():
                avg_exp_metrics[m_key] = np.mean([f[m_key] for f in fold_metrics])

            exp_summary_df = pd.DataFrame(fold_results_dict).T
            exp_summary_df.loc["Average"] = avg_exp_metrics
            exp_summary_df.to_excel(
                os.path.join(cv_ablation_result_path, f"{exp_name}_cv_summary.xlsx")
            )

            all_detailed_results.append(
                {
                    "Seed": current_seed,
                    "Experiment": exp_name,
                    "Fold": "Average",
                    **avg_exp_metrics,
                }
            )
            cv_all_exp_results[exp_name] = {"Seed": current_seed, **avg_exp_metrics}
            print(f"\n>>> 实验 {exp_name} 平均结果: {avg_exp_metrics}")

        summary_df = pd.DataFrame(cv_all_exp_results).T
        summary_df.to_excel(os.path.join(run_save_path, "final_summary.xlsx"))

        all_detailed_df = pd.DataFrame(all_detailed_results)
        all_detailed_df.to_excel(
            os.path.join(run_save_path, "all_experiments_cv_details.xlsx"), index=False
        )

    # ==========================
    # 统一测试操作
    # ==========================
    print("\n" + "#" * 30)
    print("开始在独立测试集上进行最终统一评估")
    print("#" * 30)

    input_size = 2  # 基于包含温度和温度变化率
    all_test_results = []

    for run_idx in range(1, n + 1):
        current_seed = random_seed[run_idx - 1]
        set_seed(current_seed)

        run_save_path = os.path.join(base_save_path, str(run_idx))

        df_path = os.path.join(run_save_path, "independent_test_df.xlsx")
        if not os.path.exists(df_path):
            print(f"错误: 在路径 {run_save_path} 中找不到 independent_test_df.xlsx")
            continue

        independent_test_df = pd.read_excel(df_path)
        test_df_filled = myFunction.fill_data(independent_test_df)

        for config in configs:
            exp_name = config["name"]
            exp_root = os.path.join(run_save_path, exp_name)

            if not os.path.exists(exp_root):
                print(f"跳过: 路径不存在 -> {exp_root}")
                continue

            fold_metrics_list = []
            for fold_idx in range(1, 6):
                model_path = os.path.join(exp_root, f"best_model_fold{fold_idx}.pth")
                scaler_path = os.path.join(exp_root, f"scaler_fold{fold_idx}.joblib")

                if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                    continue

                model = get_model(train_info, device, input_size=input_size)
                model.load_state_dict(torch.load(model_path, weights_only=True))
                scaler = joblib.load(scaler_path)

                X_test, y_test, _ = myFunction.prepare_lstm_data(
                    test_df_filled, scaler=scaler
                )
                loader = DataLoader(
                    EstrusDataset(X_test, y_test), train_info.batch_size, shuffle=False
                )

                metrics, _, _ = evaluate_model(model, loader, criterion, device)
                metrics.pop("avg_loss", None)
                fold_metrics_list.append(metrics)

            if fold_metrics_list:
                avg_metrics = {
                    k: np.mean([m[k] for m in fold_metrics_list])
                    for k in fold_metrics_list[0].keys()
                }
                all_test_results.append(
                    {
                        "Run": run_idx,
                        "Seed": current_seed,
                        "Experiment": exp_name,
                        **avg_metrics,
                    }
                )
                print(
                    f"Run {run_idx} (Seed: {current_seed}) 实验 {exp_name} 独立测试集平均结果: {avg_metrics}"
                )

    if all_test_results:
        summary_df = pd.DataFrame(all_test_results)
        save_path = os.path.join(base_save_path, "test_result.xlsx")
        summary_df.to_excel(save_path, index=False)
        print(f"\n所有Run的独立测试集评估完成，最终汇总已保存至: {save_path}")


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

    # 测试集评估
    # 普通 5 折 CV
    # cv_path = run_5fold_cv(df)
    # cv_path = os.path.join(result_save_path, "cv", "2026_0515_1517")
    # evaluate_independent_test_set(cv_path)

    """
    带有 8 种数据增强组合的消融实验
    """
    # 单体温
    # ablation_path, configs = run_5fold_cv_with_aug_temp(df)
    # ablation_path = os.path.join(result_save_path, "cv_ablation", "2026_0522_0737")
    # configs = ABLATION_CONFIGS
    # evaluate_independent_test_set(ablation_path, configs, input_size=1)

    # 体温 + 体温变化率
    # ablation_path, configs = run_5fold_cv_with_aug(df)
    """ ablation_path = os.path.join(
        result_save_path, "cv_ablation", "ST_temp_rate_attn", "2026_0607_1813"
    )
    configs = ABLATION_CONFIGS """
    # evaluate_independent_test_set(ablation_path, configs, input_size=2)

    """
        执行n次交叉验证和评估 观察结果的稳定性
    """
    run_n_times_cv_with_aug_and_eval(df, n=10)
