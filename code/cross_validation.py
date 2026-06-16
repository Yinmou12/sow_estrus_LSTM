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
import itertools

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
    {"name": "Baseline", "A": False, "S": False, "T": False},
    {"name": "SMOTE", "A": False, "S": True, "T": False},
    {"name": "SMOTE_Tomek", "S": True, "T": True},
] """

ABLATION_CONFIGS = [
    {"name": "AST", "A": True, "S": True, "T": True},
]


FIXED_SPLIT_DIR = os.path.join(info_FINAL_SAVE_PATH, "cross_validation")
FIXED_TRAIN_VAL_PATH = os.path.join(FIXED_SPLIT_DIR, "train_val_df.xlsx")
FIXED_TEST_PATH = os.path.join(FIXED_SPLIT_DIR, "independent_test_df.xlsx")


def load_or_create_fixed_split(df=None, random_state=123):
    """Load the fixed train/validation pool and independent test set."""
    if os.path.exists(FIXED_TRAIN_VAL_PATH) and os.path.exists(FIXED_TEST_PATH):
        train_val_df = pd.read_excel(FIXED_TRAIN_VAL_PATH, index_col=False)
        independent_test_df = pd.read_excel(FIXED_TEST_PATH, index_col=False)
    else:
        if df is None:
            raise FileNotFoundError(
                "Fixed split files were not found. Provide df once to create them."
            )
        os.makedirs(FIXED_SPLIT_DIR, exist_ok=True)
        train_val_df, independent_test_df = myFunction.split_dataset_train_val_test(
            df, random_state=random_state
        )
        train_val_df.to_excel(FIXED_TRAIN_VAL_PATH, index=False)
        independent_test_df.to_excel(FIXED_TEST_PATH, index=False)

    return train_val_df, independent_test_df


def get_fixed_cv_data(df=None, n_splits=5, random_state=123):
    train_val_df, independent_test_df = load_or_create_fixed_split(
        df, random_state=random_state
    )
    folds = myFunction.stratified_group_kfold_only(
        train_val_df, n_splits=n_splits, random_state=random_state
    )
    return train_val_df, independent_test_df, folds


def save_cv_split_snapshot(saved_result_path, train_val_df, independent_test_df):
    train_val_df.to_excel(
        os.path.join(saved_result_path, "train_val_df.xlsx"), index=False
    )
    independent_test_df.to_excel(
        os.path.join(saved_result_path, "independent_test_df.xlsx"), index=False
    )


def save_fold_threshold(exp_root, fold_idx, metrics):
    threshold = metrics.get("Threshold", 0.5)
    threshold_path = os.path.join(exp_root, f"threshold_fold{fold_idx}.txt")
    with open(threshold_path, "w", encoding="utf-8") as f:
        f.write(str(float(threshold)))


def load_fold_threshold(exp_root, fold_idx, default=0.5):
    threshold_path = os.path.join(exp_root, f"threshold_fold{fold_idx}.txt")
    if not os.path.exists(threshold_path):
        return default
    with open(threshold_path, "r", encoding="utf-8") as f:
        return float(f.read().strip())


def should_optimize_threshold(config_or_train_info):
    model_name = getattr(config_or_train_info, "model_name", None)
    if model_name is None and isinstance(config_or_train_info, dict):
        model_name = config_or_train_info.get("model_name", "EstrusLSTM")
    return model_name == "EstrusLSTM"


def run_5fold_cv(df):
    train_val_df, independent_test_df, folds = get_fixed_cv_data(df, n_splits=5)

    timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
    saved_result_path = os.path.join(result_save_path, "cv", f"{timestamp}")
    os.makedirs(saved_result_path, exist_ok=True)

    save_cv_split_snapshot(saved_result_path, train_val_df, independent_test_df)

    results = {}

    for fold_idx, (train_df_raw, val_df_raw) in enumerate(folds):
        print(f"\n{'='*20} Fold {fold_idx + 1} {'='*20}")

        train_df = myFunction.fill_data(train_df_raw)
        val_df = myFunction.fill_data(val_df_raw)

        # 准备数据格式
        X_train, y_train, scaler = myFunction.prepare_lstm_data(train_df)
        # 保存缩放器以便后续评估独立测试集
        scaler_path = os.path.join(
            saved_result_path, f"scaler_fold{fold_idx + 1}.joblib"
        )
        joblib.dump(scaler, scaler_path)
        X_val, y_val, _ = myFunction.prepare_lstm_data(val_df, scaler=scaler)

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

        model.load_state_dict(torch.load(best_model_path, weights_only=True))

        optimize_threshold = should_optimize_threshold(train_info)
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
            optimize_threshold=optimize_threshold,
            threshold_metric="f1",
        )
        if optimize_threshold:
            save_fold_threshold(saved_result_path, fold_idx + 1, record_dict)
        record_dict.pop("avg_loss", None)
        record_dict.pop("MCC", None)
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
    ada_threshold: float = 0.9
    ada_gamma: float = 1
    ada_k: int = 5

    use_smote: bool = True
    smote_amount_oversampling: int = 800
    smote_k_neighbors: int = 7

    use_tomek: bool = True
    tomek_k: int = 1


# 数据增强
# 鍗曡€虫俯鐗瑰緛
TRAIN_INFO_GRID_KEYS = {
    "model_name",
    "hidden_sizes",
    "learning_rate",
    "weight_decay",
    "dropout_rate",
    "batch_size",
    "use_cell_state",
    "bidirectional",
    "num_heads",
    "num_epochs",
    "early_patience",
    "lr_patience",
}


def apply_train_info_config(train_info, config):
    """Apply tunable experiment config values to TrainInfo."""
    for key in TRAIN_INFO_GRID_KEYS:
        if key in config:
            setattr(train_info, key, config[key])

    if getattr(train_info, "hidden_sizes", None):
        train_info.num_layers = len(train_info.hidden_sizes)

    return train_info


def _format_grid_value(value):
    if isinstance(value, (list, tuple)):
        return "_".join(map(str, value))
    return str(value).replace(".", "p")


def build_grid_search_configs(base_config, param_grid):
    target_configs = []
    grid_keys = list(param_grid.keys())
    grid_values = [
        values if isinstance(values, (list, tuple)) else [values]
        for values in param_grid.values()
    ]

    for exp_idx, values in enumerate(itertools.product(*grid_values), start=1):
        new_config = base_config.copy()
        for key, value in zip(grid_keys, values):
            new_config[key] = value
        new_config["grid_search_id"] = f"GS{exp_idx:03d}"
        new_config["name"] = f"{base_config['name']}_{new_config['grid_search_id']}"
        target_configs.append(new_config)

    return target_configs


def get_result_config_fields(config, include_experiment=True):
    fields = {}
    if include_experiment:
        fields["Experiment"] = config["name"]

    config_keys = [
        "grid_search_id",
        "A",
        "S",
        "T",
        "smote_amount",
        *sorted(TRAIN_INFO_GRID_KEYS),
    ]
    for key in config_keys:
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, (list, tuple)):
            value = "_".join(map(str, value))
        fields[f"param_{key}"] = value

    if "hidden_sizes" in config:
        fields["param_num_layers"] = len(config["hidden_sizes"])

    return fields


def run_5fold_cv_with_aug_temp(df, configs=None):
    """
    运行5折交叉验证，包含8种数据增强组合的消融实验
    """
    train_val_df, independent_test_df, folds = get_fixed_cv_data(df, n_splits=5)

    if configs is None:
        configs = ABLATION_CONFIGS
    data_aug_info = DataAugInfo()

    timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
    saved_result_path = os.path.join(
        result_save_path, "cv_ablation", "AST_temp", f"{timestamp}_EstrusLSTM"
    )
    os.makedirs(saved_result_path, exist_ok=True)

    save_cv_split_snapshot(saved_result_path, train_val_df, independent_test_df)

    cv_all_exp_results = {}
    all_detailed_results = []

    for config in configs:
        exp_name = config["name"]
        print(f"\n寮€濮嬫秷铻嶅疄楠? {exp_name} " + "*" * 30)

        cv_ablation_result_path = os.path.join(saved_result_path, exp_name)
        os.makedirs(cv_ablation_result_path, exist_ok=True)

        fold_metrics = []
        fold_results_dict = {}

        for fold_idx, (train_df_raw, val_df_raw) in enumerate(folds):
            print(f"\n--- {exp_name} | Fold {fold_idx + 1} ---")

            # 鍩虹棰勫鐞?
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
                smote_amount = config.get(
                    "smote_amount", data_aug_info.smote_amount_oversampling
                )
                train_df_flat = myFunction.SMOTE(
                    train_df_flat,
                    amount_oversampling=smote_amount,
                    k=data_aug_info.smote_k_neighbors,
                )

            if config["T"]:
                train_df_flat = myFunction.TomekLinked(
                    train_df_flat, k=data_aug_info.tomek_k
                )

            # 鍑嗗璁粌鍜岄獙璇佹暟鎹?(浠呬娇鐢ㄨ€虫俯鍗曠壒寰?
            X_train, y_train, scaler = myFunction.prepare_univariate_lstm_data(
                train_df_flat
            )
            # 淇濆瓨缂╂斁鍣?
            scaler_path = os.path.join(
                cv_ablation_result_path, f"scaler_fold{fold_idx + 1}.joblib"
            )
            joblib.dump(scaler, scaler_path)
            X_val, y_val, _ = myFunction.prepare_univariate_lstm_data(
                val_df, scaler=scaler
            )

            # 模型训练
            train_info = apply_train_info_config(TrainInfo(), config)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = get_model(train_info, device, input_size=1)

            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=train_info.learning_rate,
                weight_decay=getattr(train_info, "weight_decay", 1e-4),
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

            # 6. 鍔犺浇鏈€浣冲苟璁板綍
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
            optimize_threshold = should_optimize_threshold(config)
            final_rec, _, _ = evaluate_model(
                model,
                val_loader,
                criterion,
                device,
                optimize_threshold=optimize_threshold,
                threshold_metric="f1",
            )
            if optimize_threshold:
                save_fold_threshold(cv_ablation_result_path, fold_idx + 1, final_rec)
            final_rec.pop("avg_loss", None)
            final_rec.pop("MCC", None)
            fold_metrics.append(final_rec)
            fold_results_dict[f"Fold_{fold_idx + 1}"] = {
                **get_result_config_fields(config, include_experiment=False),
                **final_rec,
            }
            all_detailed_results.append(
                {
                    **get_result_config_fields(config),
                    "Fold": f"Fold_{fold_idx + 1}",
                    **final_rec,
                }
            )
            print(f"Fold {fold_idx + 1} 结果: {final_rec}")

        # 璁＄畻璇ュ疄楠岀殑5鎶樺钩鍧?
        avg_exp_metrics = {}
        for m_key in fold_metrics[0].keys():
            avg_exp_metrics[m_key] = np.mean([f[m_key] for f in fold_metrics])

        # 鍙傝€?run_5fold_cv 鐨勪繚瀛樻柟寮忥紝涓哄綋鍓嶅寮烘柟寮忎繚瀛樿缁嗘姌缁撴灉姹囨€?
        exp_summary_df = pd.DataFrame(fold_results_dict).T
        exp_summary_df.loc["Average"] = {
            **get_result_config_fields(config, include_experiment=False),
            **avg_exp_metrics,
        }
        exp_summary_df.to_excel(
            os.path.join(cv_ablation_result_path, f"{exp_name}_cv_summary.xlsx")
        )

        all_detailed_results.append(
            {
                **get_result_config_fields(config),
                "Fold": "Average",
                **avg_exp_metrics,
            }
        )
        cv_all_exp_results[exp_name] = {
            **get_result_config_fields(config, include_experiment=False),
            **avg_exp_metrics,
        }
        print(f"\n>>> 实验 {exp_name} 平均结果: {avg_exp_metrics}")

    # 姹囨€绘墦鍗版墍鏈夊疄楠?
    print("\n" + "=" * 50)
    print("5-Fold CV final summary")
    print("=" * 50)
    summary_df = pd.DataFrame(cv_all_exp_results).T
    print(summary_df)

    # 保存到Excel
    summary_df.to_excel(os.path.join(saved_result_path, "final_summary.xlsx"))

    # 灏嗘墍鏈夋暟鎹寮烘柟寮忕殑楠岃瘉闆嗙粨鏋滆繘琛屾眹鎬讳繚瀛樺湪涓€寮?Excel 琛ㄦ牸涓?
    all_detailed_df = pd.DataFrame(all_detailed_results)
    all_detailed_df.to_excel(
        os.path.join(saved_result_path, "all_experiments_cv_details.xlsx"), index=False
    )

    return saved_result_path, configs


# 数据增强
def run_5fold_cv_with_aug(df, configs=None):
    """
    运行5折交叉验证，包含8种数据增强组合的消融实验
    """
    train_val_df, independent_test_df, folds = get_fixed_cv_data(df, n_splits=5)

    if configs is None:
        configs = ABLATION_CONFIGS
    data_aug_info = DataAugInfo()

    timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
    saved_result_path = os.path.join(
        result_save_path, "cv_ablation", "AST_temp_rate", f"{timestamp}_ModelComparison"
    )
    os.makedirs(saved_result_path, exist_ok=True)

    save_cv_split_snapshot(saved_result_path, train_val_df, independent_test_df)

    cv_all_exp_results = {}
    all_detailed_results = []

    for config in configs:
        exp_name = config["name"]

        cv_ablation_result_path = os.path.join(saved_result_path, exp_name)
        os.makedirs(cv_ablation_result_path, exist_ok=True)

        fold_metrics = []
        fold_results_dict = {}

        for fold_idx, (train_df_raw, val_df_raw) in enumerate(folds):
            print(f"\n--- {exp_name} | Fold {fold_idx + 1} ---")

            # 鍩虹棰勫鐞?
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

            if config["S"]:
                smote_amount = config.get("smote_amount", 800)
                train_df_flat = myFunction.SMOTE(
                    train_df_flat, amount_oversampling=smote_amount, k=7
                )

            if config["T"]:
                train_df_flat = myFunction.TomekLinked(train_df_flat, k=1)

            temp_feats = train_df_flat.iloc[:, 1:-1].copy()
            rate_feats = temp_feats.diff(axis=1).fillna(0)
            rate_feats.columns = [f"rate_{i}" for i in range(1, 49)]

            train_final_df = pd.concat(
                [train_df_flat.iloc[:, :-1], rate_feats, train_df_flat.iloc[:, -1]],
                axis=1,
            )

            X_train, y_train, scaler = myFunction.prepare_lstm_data(train_final_df)

            scaler_path = os.path.join(
                cv_ablation_result_path, f"scaler_fold{fold_idx + 1}.joblib"
            )
            joblib.dump(scaler, scaler_path)

            X_val, y_val, _ = myFunction.prepare_lstm_data(val_df, scaler=scaler)

            # 模型训练
            train_info = apply_train_info_config(TrainInfo(), config)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = get_model(train_info, device)

            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=train_info.learning_rate,
                weight_decay=getattr(train_info, "weight_decay", 1e-4),
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

            model.load_state_dict(torch.load(best_model_path, weights_only=True))
            optimize_threshold = should_optimize_threshold(config)
            final_rec, _, _ = evaluate_model(
                model,
                val_loader,
                criterion,
                device,
                optimize_threshold=optimize_threshold,
                threshold_metric="f1",
            )
            if optimize_threshold:
                save_fold_threshold(cv_ablation_result_path, fold_idx + 1, final_rec)
            final_rec.pop("avg_loss", None)
            # final_rec.pop("MCC", None)
            fold_metrics.append(final_rec)
            fold_results_dict[f"Fold_{fold_idx + 1}"] = {
                **get_result_config_fields(config, include_experiment=False),
                **final_rec,
            }
            all_detailed_results.append(
                {
                    **get_result_config_fields(config),
                    "Fold": f"Fold_{fold_idx + 1}",
                    **final_rec,
                }
            )
            print(f"Fold {fold_idx + 1} 结果: {final_rec}")

        avg_exp_metrics = {}
        for m_key in fold_metrics[0].keys():
            avg_exp_metrics[m_key] = np.mean([f[m_key] for f in fold_metrics])

        exp_summary_df = pd.DataFrame(fold_results_dict).T
        exp_summary_df.loc["Average"] = {
            **get_result_config_fields(config, include_experiment=False),
            **avg_exp_metrics,
        }
        exp_summary_df.to_excel(
            os.path.join(cv_ablation_result_path, f"{exp_name}_cv_summary.xlsx")
        )

        all_detailed_results.append(
            {
                **get_result_config_fields(config),
                "Fold": "Average",
                **avg_exp_metrics,
            }
        )
        cv_all_exp_results[exp_name] = {
            **get_result_config_fields(config, include_experiment=False),
            **avg_exp_metrics,
        }
        print(f"\n>>> 实验 {exp_name} 平均结果: {avg_exp_metrics}")

    print("\n" + "=" * 50)
    print("5-Fold CV final summary")
    print("=" * 50)
    summary_df = pd.DataFrame(cv_all_exp_results).T
    print(summary_df)

    # 保存到Excel
    summary_df.to_excel(os.path.join(saved_result_path, "final_summary.xlsx"))

    all_detailed_df = pd.DataFrame(all_detailed_results)
    all_detailed_df.to_excel(
        os.path.join(saved_result_path, "all_experiments_cv_details.xlsx"), index=False
    )

    return saved_result_path, configs


def evaluate_independent_test_set(base_path, configs=None, input_size=1):
    print("\n" + "#" * 30)
    print("Starting independent test evaluation")
    print("#" * 30)

    df_path = os.path.join(base_path, "independent_test_df.xlsx")
    if not os.path.exists(df_path):
        if not os.path.exists(FIXED_TEST_PATH):
            return
        print(f"Using fixed independent test set: {FIXED_TEST_PATH}")
        df_path = FIXED_TEST_PATH

    independent_test_df = pd.read_excel(df_path)
    all_sows = independent_test_df["sSowsNo"].unique()
    estrus_sows = independent_test_df[independent_test_df["isEstrus"] == 1][
        "sSowsNo"
    ].unique()
    not_estrus_sows = [s for s in all_sows if s not in estrus_sows]

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

    if configs is None:
        eval_list = [{"name": "Standard_CV", "is_ablation": False}]
    else:
        eval_list = configs

    for config in eval_list:
        exp_name = config["name"]
        is_ablation = config.get("is_ablation", True)
        current_train_info = apply_train_info_config(
            TrainInfo(), config
        )  # 榛樿鍋囪鏄秷铻嶅疄楠?

        # 确定存放该组模型文件的根路径
        exp_root = os.path.join(base_path, exp_name) if is_ablation else base_path

        if not os.path.exists(exp_root):
            print(f"璺宠繃: 璺緞涓嶅瓨鍦?-> {exp_root}")
            continue

        fold_metrics_list = []
        for fold_idx in range(1, 6):
            model_path = os.path.join(exp_root, f"best_model_fold{fold_idx}.pth")
            scaler_path = os.path.join(exp_root, f"scaler_fold{fold_idx}.joblib")

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                continue

            # 加载模型和缩放器
            model = get_model(current_train_info, device, input_size=input_size)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            scaler = joblib.load(scaler_path)

            if input_size == 1:
                X_test, y_test, _ = myFunction.prepare_univariate_lstm_data(
                    test_df_filled, scaler=scaler
                )
            else:
                X_test, y_test, _ = myFunction.prepare_lstm_data(
                    test_df_filled, scaler=scaler
                )

            print(f"正样本数: {(y_test == 1).sum()}, 负样本数: {(y_test == 0).sum()}")
            if should_optimize_threshold(config):
                threshold = load_fold_threshold(exp_root, fold_idx, default=0.5)
            else:
                threshold = 0.5
            print(f"Fold {fold_idx} threshold: {threshold:.2f}")
            loader = DataLoader(
                EstrusDataset(X_test, y_test),
                current_train_info.batch_size,
                shuffle=False,
            )

            metrics, _, _ = evaluate_model(
                model, loader, criterion, device, threshold=threshold
            )
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
            print(f"瀹為獙 {key_name} 鐙珛娴嬭瘯闆嗗钩鍧囩粨鏋? {avg_metrics}")

    if final_test_results:
        summary_df = pd.DataFrame(final_test_results).T
        save_path = os.path.join(f"{base_path}_evaluate", "temp.xlsx")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        summary_df.to_excel(save_path)


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
    timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
    base_save_path = os.path.join(
        result_save_path,
        "cv_ablation",
        "ST_temp_rate_n_times",
        "all_experiments",
        f"{timestamp}",
    )
    os.makedirs(base_save_path, exist_ok=True)

    configs = ABLATION_CONFIGS
    data_aug_info = DataAugInfo()
    train_info = TrainInfo()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()

    random_seed = random.sample(range(1, 1000), n)

    for run_idx in range(1, n + 1):
        current_seed = random_seed[run_idx - 1]
        set_seed(current_seed)
        print(
            f"\n{'='*50}\nStarting Run {run_idx}/{n} (Seed: {current_seed})\n{'='*50}"
        )

        run_save_path = os.path.join(base_save_path, str(run_idx))
        os.makedirs(run_save_path, exist_ok=True)

        # 使用不同的随机数种子以确保每次运行的数据划分和模型初始化不同
        train_val_df, independent_test_df = load_or_create_fixed_split(df)
        folds = myFunction.stratified_group_kfold_only(
            train_val_df, n_splits=5, random_state=current_seed
        )

        save_cv_split_snapshot(run_save_path, train_val_df, independent_test_df)

        cv_all_exp_results = {}
        all_detailed_results = []

        for config in configs:
            exp_name = config["name"]

            cv_ablation_result_path = os.path.join(run_save_path, exp_name)
            os.makedirs(cv_ablation_result_path, exist_ok=True)

            fold_metrics = []
            fold_results_dict = {}

            for fold_idx, (train_df_raw, val_df_raw) in enumerate(folds):
                print(f"\n--- {exp_name} | Fold {fold_idx + 1} ---")

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

                if config.get("S", False):
                    smote_amount = config.get("smote_amount", 800)
                    train_df_flat = myFunction.SMOTE(
                        train_df_flat, amount_oversampling=smote_amount, k=7
                    )

                if config.get("T", False):
                    train_df_flat = myFunction.TomekLinked(train_df_flat, k=1)

                temp_feats = train_df_flat.iloc[:, 1:-1].copy()
                rate_feats = temp_feats.diff(axis=1).fillna(0)
                rate_feats.columns = [f"rate_{i}" for i in range(1, 49)]
                train_final_df = pd.concat(
                    [train_df_flat.iloc[:, :-1], rate_feats, train_df_flat.iloc[:, -1]],
                    axis=1,
                )

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

                model.load_state_dict(torch.load(best_model_path, weights_only=True))
                optimize_threshold = should_optimize_threshold(config)
                final_rec, _, _ = evaluate_model(
                    model,
                    val_loader,
                    criterion,
                    device,
                    optimize_threshold=optimize_threshold,
                    threshold_metric="f1",
                )
                if optimize_threshold:
                    save_fold_threshold(cv_ablation_result_path, fold_idx + 1, final_rec)
                fold_metrics.append(final_rec)
                fold_results_dict[f"Fold_{fold_idx + 1}"] = {
                    **get_result_config_fields(config, include_experiment=False),
                    **final_rec,
                }
                all_detailed_results.append(
                    {
                        "Seed": current_seed,
                        **get_result_config_fields(config),
                        "Fold": f"Fold_{fold_idx + 1}",
                        **final_rec,
                    }
                )
                print(f"Fold {fold_idx + 1} 结果: {final_rec}")

            avg_exp_metrics = {}
            for m_key in fold_metrics[0].keys():
                avg_exp_metrics[m_key] = np.mean([f[m_key] for f in fold_metrics])

            exp_summary_df = pd.DataFrame(fold_results_dict).T
            exp_summary_df.loc["Average"] = {
                **get_result_config_fields(config, include_experiment=False),
                **avg_exp_metrics,
            }
            exp_summary_df.to_excel(
                os.path.join(cv_ablation_result_path, f"{exp_name}_cv_summary.xlsx")
            )

            all_detailed_results.append(
                {
                    "Seed": current_seed,
                    **get_result_config_fields(config),
                    "Fold": "Average",
                    **avg_exp_metrics,
                }
            )
            cv_all_exp_results[exp_name] = {
                "Seed": current_seed,
                **get_result_config_fields(config, include_experiment=False),
                **avg_exp_metrics,
            }
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
    input_size = 2  # 基于包含温度和温度变化率
    all_test_results = []

    for run_idx in range(1, n + 1):
        current_seed = random_seed[run_idx - 1]
        set_seed(current_seed)

        run_save_path = os.path.join(base_save_path, str(run_idx))

        df_path = os.path.join(run_save_path, "independent_test_df.xlsx")
        if not os.path.exists(df_path):
            if not os.path.exists(FIXED_TEST_PATH):
                continue
            print(f"Using fixed independent test set: {FIXED_TEST_PATH}")
            df_path = FIXED_TEST_PATH

        independent_test_df = pd.read_excel(df_path)
        test_df_filled = myFunction.fill_data(independent_test_df)

        for config in configs:
            exp_name = config["name"]
            current_train_info = apply_train_info_config(TrainInfo(), config)
            exp_root = os.path.join(run_save_path, exp_name)

            if not os.path.exists(exp_root):
                continue

            fold_metrics_list = []
            for fold_idx in range(1, 6):
                model_path = os.path.join(exp_root, f"best_model_fold{fold_idx}.pth")
                scaler_path = os.path.join(exp_root, f"scaler_fold{fold_idx}.joblib")

                if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                    continue

                model = get_model(current_train_info, device, input_size=input_size)
                model.load_state_dict(torch.load(model_path, weights_only=True))
                scaler = joblib.load(scaler_path)

                X_test, y_test, _ = myFunction.prepare_lstm_data(
                    test_df_filled, scaler=scaler
                )
                if should_optimize_threshold(config):
                    threshold = load_fold_threshold(exp_root, fold_idx, default=0.5)
                else:
                    threshold = 0.5
                print(f"Fold {fold_idx} threshold: {threshold:.2f}")
                loader = DataLoader(
                    EstrusDataset(X_test, y_test),
                    current_train_info.batch_size,
                    shuffle=False,
                )

                metrics, _, _ = evaluate_model(
                    model, loader, criterion, device, threshold=threshold
                )
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
                    f"Run {run_idx} (Seed: {current_seed}) 瀹為獙 {exp_name} 鐙珛娴嬭瘯闆嗗钩鍧囩粨鏋? {avg_metrics}"
                )

    if all_test_results:
        summary_df = pd.DataFrame(all_test_results)
        save_path = os.path.join(base_save_path, "test_result.xlsx")
        summary_df.to_excel(save_path, index=False)


def run_grid_search_tuning(
    df,
    base_config,
    hidden_sizes_list=None,
    learning_rates=None,
    use_temp_only=False,
    param_grid=None,
):
    """
    使用网格搜索调整模型的隐藏单元数量和不同学习率的组合
    """
    if param_grid is None:
        if hidden_sizes_list is None or learning_rates is None:
            raise ValueError(
                "Provide either param_grid or both hidden_sizes_list and learning_rates."
            )
        param_grid = {
            "hidden_sizes": hidden_sizes_list,
            "learning_rate": learning_rates,
        }

    target_configs = build_grid_search_configs(base_config, param_grid)

    print("\n" + "=" * 50)
    print("Starting grid search for model structure and training hyperparameters ...")
    print(f"Total experiments: {len(target_configs)}")
    print(f"Search keys: {list(param_grid.keys())}")
    print("=" * 50)

    if use_temp_only:
        return run_5fold_cv_with_aug_temp(df, configs=target_configs)
    else:
        return run_5fold_cv_with_aug(df, configs=target_configs)


def run_smote_ratio_tuning(df, amounts=[300, 400, 500, 600, 700, 800, 900, 1000]):
    base_config = {
        "A": True,
        "S": True,
        "T": True,
        "model_name": "EstrusLSTM",
        "hidden_sizes": [64, 64, 64, 64],
        "learning_rate": 1e-3,
    }

    target_configs = []
    for amount in amounts:
        config = base_config.copy()
        config["name"] = f"AST_SMOTE_{amount}"
        config["smote_amount"] = amount
        target_configs.append(config)

    print("\n" + "=" * 50)
    print(f"Total experiments: {len(target_configs)}")
    print(f"SMOTE 扩充比例列表: {amounts}")
    print("=" * 50)

    return run_5fold_cv_with_aug(df, configs=target_configs)


def run_model_comparison(df):
    base_config = {
        "A": True,
        "S": True,
        "T": True,
        "hidden_sizes": [64, 64, 64, 64],
        "learning_rate": 5e-4,
    }

    target_configs = [
        {
            **base_config,
            "name": "RNN",
            "model_name": "EstrusRNN_sample",
            "bidirectional": True,
        },
        {
            **base_config,
            "name": "LSTM",
            "model_name": "EstrusLSTM",
            "bidirectional": False,
        },
        {
            **base_config,
            "name": "Bi-LSTM",
            "model_name": "EstrusLSTM",
            "bidirectional": True,
        },
        {
            **base_config,
            "name": "GRU",
            "model_name": "EstrusGRU",
            "bidirectional": True,
        },
    ]

    print("\n" + "=" * 50)
    print("Starting model comparison...")
    print(f"Total experiments: {len(target_configs)}")
    print("=" * 50)

    return run_5fold_cv_with_aug(df, configs=target_configs)


if __name__ == "__main__":
    df = pd.read_excel(
        os.path.join(
            experimentRecord_data_path,
            "splited_dataset",
            "splited_dataset_2026_0406_1140.xlsx",
        ),
        index_col=False,
    )

    train_val_df, independent_test_df = load_or_create_fixed_split(df)

    """
    带有 8 种数据增强组合的消融实验
    """
    # ablation_path, configs = run_5fold_cv_with_aug_temp(df)
    # ablation_path = os.path.join(result_save_path, "cv_ablation", "2026_0522_0737")
    # configs = ABLATION_CONFIGS
    # evaluate_independent_test_set(ablation_path, configs, input_size=1)

    # ablation_path, configs = run_5fold_cv_with_aug(df)
    """ ablation_path = os.path.join(
        result_save_path, "cv_ablation", "AST_temp_rate", "EstrusLSTM"
    )
    configs = ABLATION_CONFIGS """
    # evaluate_independent_test_set(ablation_path, configs, input_size=2)

    """
        多次交叉验证
    """
    # run_n_times_cv_with_aug_and_eval(df, n=10)

    """
        网格搜索
    """
    base_config = {"name": "Full_Aug", "A": True, "S": True, "T": True}
    # 定义你要搜索的超参数空间
    param_grid = {
        "model_name": ["EstrusLSTM"],
        "hidden_sizes": [
            [32, 32],
            [64, 32],
            [64, 64, 32],
            [128, 64, 32],
            [64, 64, 64, 64],
            [128, 128, 64, 32],
        ],
        "learning_rate": [5e-4, 1e-3],
        # "weight_decay": [1e-4],
        # "dropout_rate": [0.1, 0.2],
        # "batch_size": [32],
        # "use_cell_state": [False],
    }
    # run_grid_search_tuning(df, base_config, use_temp_only=False, param_grid=param_grid)

    # run_smote_ratio_tuning(df, amounts=[300, 400, 500, 600, 700, 800, 900, 1000])

    run_model_comparison(df)
