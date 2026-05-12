# -*- coding: utf-8 -*-
"""
消融实验核心模块

提供灵活的数据增强pipeline和实验执行功能。
"""

import json
import os
import sys
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
)

# 添加父目录到路径以导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lstm_model import (
    EarlyStopping,
    EstrusLSTM,
    EstrusLSTM_Attn,
    EstrusGRU,
    EstrusLSTM_MultiHeadAttn,
)
import sow_estrus_LSTM_Function as myFunction

from configs.ablation_configs import (
    ExperimentConfig,
    DataAugmentationConfig,
    ModelConfig,
)

# ============================================================================
# 数据集类
# ============================================================================


class EstrusDataset(Dataset):
    """发情预测数据集"""

    def __init__(self, X, y):
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# 数据增强应用
# ============================================================================


def apply_data_augmentation(
    train_df: pd.DataFrame,
    config: DataAugmentationConfig,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    根据配置应用数据增强策略

    Args:
        train_df: 训练数据DataFrame
        config: 数据增强配置
        verbose: 是否打印详细信息

    Returns:
        增强后的DataFrame
    """
    augmented_df = train_df.copy()

    if verbose:
        print(f"原始训练集大小: {len(augmented_df)}")
        print(
            f"类别分布: 正类={sum(augmented_df['isEstrus'] == 1)}, "
            f"负类={sum(augmented_df['isEstrus'] == 0)}"
        )

    # 临时添加ID列
    # augmented_df.insert(0, "sSowsNo", [f"orig_{i}" for i in range(len(augmented_df))])

    # Step 1: ADASYN过采样
    if config.use_adasyn:
        df_min = augmented_df[augmented_df["isEstrus"] == 1].copy()
        df_maj = augmented_df[augmented_df["isEstrus"] == 0].copy()

        if len(df_min) < len(df_maj):
            augmented_df = myFunction.ADASYN(
                threshold=config.adasyn_threshold,
                gamma=config.adasyn_gamma,
                df_min=df_min,
                df_maj=df_maj,
                k=config.adasyn_k,
            )
            if verbose:
                print(f"ADASYN后训练集大小: {len(augmented_df)}")

    # Step 2: SMOTE过采样
    if config.use_smote:
        augmented_df = myFunction.SMOTE(
            data=augmented_df,
            amount_oversampling=config.smote_amount,
            k=config.smote_k,
        )
        if verbose:
            print(f"SMOTE后训练集大小: {len(augmented_df)}")

    # Step 3: TomekLinks欠采样
    if config.use_tomek_links:
        augmented_df = myFunction.TomekLinked(data=augmented_df, k=config.tomek_k)
        if verbose:
            print(f"TomekLinks后训练集大小: {len(augmented_df)}")

    if verbose:
        print(
            f"最终训练集大小: {len(augmented_df)}, "
            f"正类={sum(augmented_df['isEstrus'] == 1)}, "
            f"负类={sum(augmented_df['isEstrus'] == 0)}"
        )

    return augmented_df


# ============================================================================
# 数据加载
# ============================================================================


def load_data_from_excel(
    file_path: str, num_features: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """从Excel文件加载数据"""
    df = pd.read_excel(file_path, index_col=False)
    y = df["label_isEstrus"].values
    X_raw = df.drop(columns=["label_isEstrus"]).values
    seq_len = X_raw.shape[1] // num_features
    X_3d = X_raw.reshape(-1, seq_len, num_features)
    return X_3d, y


def load_raw_train_data(
    data_path: str, num_features: int = 2
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载原始训练数据和验证/测试数据

    Args:
        data_path: 数据目录路径
        num_features: 特征数量

    Returns:
        train_df: 原始训练数据DataFrame (用于数据增强)
        val_df: 验证集
        test_df: 测试集
    """
    # 加载原始数据
    splited_dataset = pd.read_excel(
        os.path.join(data_path, "splited_dataset_2026_0406_1140.xlsx"), index_col=False
    )

    # 分层分组划分数据集，确保同一母猪的数据不会同时出现在训练集、验证集和测试集中
    train_df_raw, val_df_raw, test_df_raw = myFunction.stratified_group_split(
        splited_dataset, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2
    )

    # 数据填补
    train_df = myFunction.fill_data(train_df_raw, balanced_data=False, stride=6)
    val_df = myFunction.fill_data(val_df_raw)
    test_df = myFunction.fill_data(test_df_raw)

    copy_train_df = train_df.copy()
    train_df = myFunction.convert_features(copy_train_df)

    return train_df, val_df, test_df


def prepare_augmented_data(
    train_df: pd.DataFrame,
    num_features: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将增强后的DataFrame转换为训练用的numpy数组

    Args:
        train_df: 增强后的训练数据
        num_features: 特征数量

    Returns:
        X, y: 训练数据
    """
    y = train_df.iloc[:, -1].values

    # 获取特征列
    feature_cols = train_df.columns[1:-1]
    X_raw = train_df[feature_cols].values.astype(float)

    seq_len = X_raw.shape[1] // num_features
    X_3d = X_raw.reshape(-1, seq_len, num_features)

    return X_3d, y


# ============================================================================
# 模型构建
# ============================================================================


def build_model(config: ModelConfig, device: torch.device) -> nn.Module:
    """根据配置构建模型"""
    model_map = {
        "EstrusLSTM": EstrusLSTM,
        "EstrusLSTM_Attn": EstrusLSTM_Attn,
        "EstrusGRU": EstrusGRU,
        "EstrusLSTM_MultiHeadAttn": EstrusLSTM_MultiHeadAttn,
    }

    if config.model_type not in model_map:
        raise ValueError(f"未知的模型类型: {config.model_type}")

    model_class = model_map[config.model_type]
    model = model_class(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout_rate=config.dropout_rate,
    ).to(device)

    return model


def build_criterion(config: ModelConfig, device: torch.device) -> nn.Module:
    """根据配置构建损失函数"""
    if config.loss_type == "BCELoss":
        return nn.BCELoss()
    elif config.loss_type == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    elif config.loss_type == "FocalLoss":
        # Focal Loss实现
        class FocalLoss(nn.Module):
            def __init__(self, alpha=0.25, gamma=2):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, inputs, targets):
                BCE_loss = nn.functional.binary_cross_entropy(
                    inputs, targets, reduction="none"
                )
                pt = torch.exp(-BCE_loss)
                F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
                return F_loss.mean()

        return FocalLoss()
    else:
        raise ValueError(f"未知的损失函数类型: {config.loss_type}")


def build_optimizer(model: nn.Module, config: ModelConfig) -> optim.Optimizer:
    """根据配置构建优化器"""
    if config.optimizer_type == "Adam":
        return optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer_type == "AdamW":
        return optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer_type == "SGD":
        return optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"未知的优化器类型: {config.optimizer_type}")


def build_scheduler(
    optimizer: optim.Optimizer, config: ModelConfig
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """根据配置构建学习率调度器"""
    if config.scheduler_type == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )
    elif config.scheduler_type == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    else:
        return None


# ============================================================================
# 训练和评估
# ============================================================================


def evaluate_model(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float, float, float, float, float]:
    """评估模型性能"""
    model.eval()
    total_loss = 0
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)

            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            probs = outputs.cpu().numpy().flatten()
            all_probs.extend(probs)

            preds = (probs >= 0.5).astype(float)
            all_preds.extend(preds)

            all_labels.extend(batch_y.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)
    mcc = matthews_corrcoef(all_labels, all_preds)

    return avg_loss, accuracy, precision, recall, f1, roc_auc, mcc


def train_single_run(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: ModelConfig,
    seed: int,
    save_dir: str,
    run_id: int,
) -> Dict:
    """
    单次训练运行

    Args:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        config: 模型配置
        seed: 随机种子
        save_dir: 保存目录
        run_id: 运行编号

    Returns:
        训练结果字典
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建数据加载器
    train_loader = DataLoader(
        EstrusDataset(X_train, y_train),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        EstrusDataset(X_val, y_val),
        batch_size=config.batch_size,
    )

    # 构建模型和训练组件
    model = build_model(config, device)
    criterion = build_criterion(config, device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # 早停
    best_model_path = os.path.join(save_dir, f"best_model_run{run_id}.pth")
    early_stopping = EarlyStopping(
        patience=config.early_patience, verbose=False, path=best_model_path
    )

    # 训练历史
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_auc": [],
        "val_mcc": [],
    }

    # 训练循环
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        # 验证
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, mcc = evaluate_model(
            model, val_loader, criterion, device
        )

        # 记录历史
        history["train_loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)
        history["val_mcc"].append(mcc)

        # 学习率调度
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # 早停检查
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            break

    # 加载最佳模型并返回最佳验证结果
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_mcc = evaluate_model(
        model, val_loader, criterion, device
    )

    return {
        "history": history,
        "best_val_metrics": {
            "loss": val_loss,
            "accuracy": val_acc,
            "precision": val_prec,
            "recall": val_rec,
            "f1": val_f1,
            "auc": val_auc,
            "mcc": val_mcc,
        },
        "early_stop_epoch": len(history["train_loss"]),
    }


def run_experiment(
    config: ExperimentConfig,
    data_path: str,
    result_dir: str,
    num_features: int = 2,
    verbose: bool = True,
) -> Dict:
    """
    运行单个消融实验

    Args:
        config: 实验配置
        data_path: 数据路径
        result_dir: 结果保存目录
        num_features: 特征数量
        verbose: 是否打印详细信息

    Returns:
        实验结果字典
    """
    print(f"\n{'=' * 60}")
    print(f"实验 {config.experiment_id}: {config.experiment_name}")
    print(f"描述: {config.description}")
    print(f"{'=' * 60}")

    # 创建结果目录
    exp_dir = os.path.join(
        result_dir, f"{config.experiment_id}_{config.experiment_name.replace(' ', '_')}"
    )
    os.makedirs(exp_dir, exist_ok=True)

    # 加载数据
    if verbose:
        print("加载数据...")

    train_df, val_df, test_df = load_raw_train_data(data_path, num_features)

    # 多次运行结果存储
    all_run_results = []
    best_val_metrics_list = []
    test_metrics_list = []  # 新增：用于存储每次运行的测试集指标

    augmented_df = apply_data_augmentation(
        train_df.copy(), config.data_augmentation, verbose=verbose
    )

    # 增加耳温变化率
    temperatures_features = augmented_df.iloc[:, 1:-1].copy()
    rate_features = temperatures_features.diff(axis=1).fillna(0)
    rate_features.columns = [f"features{i}" for i in range(49, 97)]
    df_final = pd.concat(
        [augmented_df.iloc[:, :-1], rate_features, augmented_df.iloc[:, -1]],
        axis=1,
        ignore_index=True,
    )

    # 准备数据
    X_train, y_train, train_scaler = myFunction.prepare_lstm_data(df_final)
    X_val, y_val, _ = myFunction.prepare_lstm_data(val_df, scaler=train_scaler)
    X_test, y_test, _ = myFunction.prepare_lstm_data(test_df, scaler=train_scaler)

    for run_id in range(1, config.num_runs + 1):
        print(f"\n--- 运行 {run_id}/{config.num_runs} ---")

        # 应用数据增强 (每次运行都要重新应用以获得不同的随机结果)
        seed = config.random_seed + run_id
        random.seed(seed)
        np.random.seed(seed)

        if verbose and run_id == 1:  # 只在第一次运行打印数据大小
            print(f"训练集大小: {len(y_train)}")

        # 训练
        run_result = train_single_run(
            X_train,
            y_train,
            X_val,
            y_val,
            config.model,
            seed,
            exp_dir,
            run_id,
        )

        all_run_results.append(run_result)
        best_val_metrics_list.append(run_result["best_val_metrics"])

        if verbose:
            metrics = run_result["best_val_metrics"]
            print(
                f"最佳验证指标 - Acc: {metrics['accuracy']:.4f}, "
                f"Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, "
                f"F1: {metrics['f1']:.4f}, "
                f"AUC: {metrics['auc']:.4f}"
                f"MCC: {metrics['mcc']:.4f}"
            )

        # 在测试集上评估最佳模型
        # 注意：这里的 model 变量只是一个空壳，需要重新构建并加载权重
        # 重新构建模型以确保其在同一设备上
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_for_test = build_model(config.model, device)
        best_model_path = os.path.join(exp_dir, f"best_model_run{run_id}.pth")
        model_for_test.load_state_dict(torch.load(best_model_path, map_location=device))

        test_loader = DataLoader(
            EstrusDataset(X_test, y_test), batch_size=config.model.batch_size
        )
        criterion = build_criterion(config.model, device)
        test_loss, test_acc, test_prec, test_rec, test_f1, test_auc, test_mcc = (
            evaluate_model(model_for_test, test_loader, criterion, device)
        )
        test_metrics_list.append(
            {
                "loss": test_loss,
                "accuracy": test_acc,
                "precision": test_prec,
                "recall": test_rec,
                "f1": test_f1,
                "auc": test_auc,
                "mcc": test_mcc,
            }
        )
        if verbose:
            print(
                f"测试集指标 - Acc: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}, MCC: {test_mcc:.4f}"
            )

    # 计算平均指标
    avg_metrics = {}
    std_metrics = {}
    for key in best_val_metrics_list[0].keys():
        values = [m[key] for m in best_val_metrics_list]
        avg_metrics[key] = np.mean(values)
        std_metrics[key] = np.std(values)

    print(f"\n--- 平均验证指标 (n={config.num_runs}) ---")
    print(f"  Accuracy:  {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
    print(
        f"  Precision: {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}"
    )
    print(f"  Recall:    {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"  F1:        {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
    print(f"  AUC:       {avg_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}")
    print(f"  MCC:       {avg_metrics['mcc']:.4f} ± {std_metrics['mcc']:.4f}")

    # 计算平均测试指标
    avg_test_metrics = {}
    std_test_metrics = {}
    for key in test_metrics_list[0].keys():
        values = [m[key] for m in test_metrics_list]
        avg_test_metrics[key] = np.mean(values)
        std_test_metrics[key] = np.std(values)

    print(f"\n--- 平均测试指标 (n={config.num_runs}) ---")
    print(
        f"  Accuracy:  {avg_test_metrics['accuracy']:.4f} ± {std_test_metrics['accuracy']:.4f}"
    )
    print(
        f"  Precision: {avg_test_metrics['precision']:.4f} ± {std_test_metrics['precision']:.4f}"
    )
    print(
        f"  Recall:    {avg_test_metrics['recall']:.4f} ± {std_test_metrics['recall']:.4f}"
    )
    print(f"  F1:        {avg_test_metrics['f1']:.4f} ± {std_test_metrics['f1']:.4f}")
    print(f"  AUC:       {avg_test_metrics['auc']:.4f} ± {std_test_metrics['auc']:.4f}")
    print(f"  MCC:       {avg_test_metrics['mcc']:.4f} ± {std_test_metrics['mcc']:.4f}")

    # 保存结果
    result = {
        "config": {
            "experiment_id": config.experiment_id,
            "experiment_name": config.experiment_name,
            "description": config.description,
            "data_augmentation": {
                "use_adasyn": config.data_augmentation.use_adasyn,
                "use_smote": config.data_augmentation.use_smote,
                "use_tomek_links": config.data_augmentation.use_tomek_links,
            },
            "model": {
                "model_type": config.model.model_type,
                "hidden_size": config.model.hidden_size,
                "num_layers": config.model.num_layers,
                "dropout_rate": config.model.dropout_rate,
            },
            "num_runs": config.num_runs,
        },
        "best_val_metrics_per_run": best_val_metrics_list,
        "test_metrics_per_run": test_metrics_list,  # 新增测试集指标
        "avg_test_metrics": avg_test_metrics,  # 平均测试集指标
        "std_test_metrics": std_test_metrics,  # 测试集指标标准差
        "avg_metrics": avg_metrics,
        "std_metrics": std_metrics,
        "all_histories": [r["history"] for r in all_run_results],
    }

    # 保存JSON
    result_path = os.path.join(exp_dir, "experiment_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存至: {exp_dir}")

    return result


if __name__ == "__main__":
    # 测试配置
    from configs.ablation_configs import get_experiment_config

    config = get_experiment_config("Exp0_Baseline")

    # 测试路径
    data_path = r"D:\_Software_Projects\VSCode\scientific_research\sow_estrus\my_code\data\splited_dataset"
    result_dir = r"D:\_Software_Projects\VSCode\scientific_research\sow_estrus\my_code\result\ablation"

    # 仅运行一次测试
    config.num_runs = 1
    run_experiment(config, data_path, result_dir)
