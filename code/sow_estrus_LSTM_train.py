from sow_estrus_LSTM_Info import *
from lstm_model import (
    EarlyStopping,
    EstrusLSTM,
    EstrusLSTM_Attn,
    EstrusGRU,
    EstrusLSTM_MultiHeadAttn,
)
import sow_estrus_LSTM_Function as myFunction

import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class __train_info:
    """
    DATA_NotCor_AddTempRate_2026_0406_1213 : 耳温+变化率 未进行数据增强
    DATA_AST_AddTempRate_2026_0416_1754 : 耳温+变化率 数据增强
    """

    saved_file_path = os.path.join(
        info_FINAL_SAVE_PATH, "DATA_AST_AddTempRate_2026_0416_1754"
    )
    """
        模型参数
    """
    model_name: str = "EstrusLSTM"
    num_feature: int = 2
    input_size: int = 2
    layer_hidden_size: int = 64
    num_layers: int = 4
    hidden_sizes: list = (
        None  # 支持每层不同单元数，如 [128, 64, 32]。一旦设置将优先使用它
    )
    learning_rate = 0.0005
    use_cell_state: bool = False
    dropout_rate = 0.2
    bidirectional: bool = True
    num_heads: int = 4
    """
        训练控制参数
    """
    batch_size: int = 32
    num_epochs: int = 100
    early_patience: int = 7
    lr_patience: int = 5

    # VERSION_TRAIN = "BiLSTM"
    VERSION_TRAIN = "BiLSTM_DATA_AST_AddTempRate"


# 数据准备类
class EstrusDataset(Dataset):
    def __init__(self, X, y):
        if X.ndim == 2:
            X = X[:, :, np.newaxis]  # 添加特征维度，变为 (samples, seq_len, features)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def evaluate_model(model, loader, criterion, device):
    model.eval()  # 设置模型为评估模式，关闭 dropout 和 batch normalization
    total_loss = 0
    all_raw_probs = []  # 搜集概率以计算 ROC_AUC
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 评估时不计算梯度，节省内存和计算资源
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)

            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            probs = outputs.cpu().numpy().flatten()
            all_raw_probs.extend(probs)

            preds = (probs >= 0.5).astype(float)
            all_preds.extend(preds)

            all_labels.extend(batch_y.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    f1 = f1_score(all_labels, all_preds, zero_division=0)

    roc_auc = roc_auc_score(all_labels, all_raw_probs)
    mcc = matthews_corrcoef(all_labels, all_preds)

    metrics_dict = {
        "avg_loss": avg_loss,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Specificity": specificity,
        "AUC": roc_auc,
        "MCC": mcc,
    }

    return metrics_dict, all_labels, all_preds


def load_combined_dataset(file_name, num_features=2):
    df = pd.read_excel(file_name, index_col=False)
    y = df["label_isEstrus"].values

    X_raw = df.drop(columns=["label_isEstrus"]).values

    # X_3d = np.expand_dims(X_raw, axis=-1)
    seq_len = X_raw.shape[1] // num_features
    X_3d = X_raw.reshape(-1, seq_len, num_features)

    return X_3d, y


def get_model(train_info, device, input_size=None):
    from lstm_model import (
        EstrusLSTM,
        EstrusLSTM_Attn,
        EstrusLSTM_DotProductAttn,
        EstrusLSTM_ScaledDotProductAttn,
        EstrusLSTM_MultiHeadAttn,
        EstrusGRU,
    )

    actual_input_size = input_size if input_size is not None else train_info.input_size

    if train_info.model_name == "EstrusLSTM":
        model = EstrusLSTM(
            input_size=actual_input_size,
            hidden_size=train_info.layer_hidden_size,
            num_layers=train_info.num_layers,
            hidden_sizes=getattr(train_info, "hidden_sizes", None),
            use_cell_state=train_info.use_cell_state,
            dropout_rate=train_info.dropout_rate,
            bidirectional=train_info.bidirectional,
        )
    elif train_info.model_name == "EstrusLSTM_Attn":
        model = EstrusLSTM_Attn(
            input_size=actual_input_size,
            hidden_size=train_info.layer_hidden_size,
            num_layers=train_info.num_layers,
            hidden_sizes=getattr(train_info, "hidden_sizes", None),
            dropout_rate=train_info.dropout_rate,
            bidirectional=train_info.bidirectional,
        )
    elif train_info.model_name == "EstrusLSTM_DotProductAttn":
        model = EstrusLSTM_DotProductAttn(
            input_size=actual_input_size,
            hidden_size=train_info.layer_hidden_size,
            num_layers=train_info.num_layers,
            hidden_sizes=getattr(train_info, "hidden_sizes", None),
            dropout_rate=train_info.dropout_rate,
            bidirectional=train_info.bidirectional,
        )
    elif train_info.model_name == "EstrusLSTM_ScaledDotProductAttn":
        model = EstrusLSTM_ScaledDotProductAttn(
            input_size=actual_input_size,
            hidden_size=train_info.layer_hidden_size,
            num_layers=train_info.num_layers,
            hidden_sizes=getattr(train_info, "hidden_sizes", None),
            dropout_rate=train_info.dropout_rate,
            bidirectional=train_info.bidirectional,
        )
    elif train_info.model_name == "EstrusLSTM_MultiHeadAttn":
        model = EstrusLSTM_MultiHeadAttn(
            input_size=actual_input_size,
            hidden_size=train_info.layer_hidden_size,
            num_layers=train_info.num_layers,
            hidden_sizes=getattr(train_info, "hidden_sizes", None),
            num_heads=train_info.num_heads,
            dropout_rate=train_info.dropout_rate,
            bidirectional=train_info.bidirectional,
        )
    elif train_info.model_name == "EstrusGRU":
        model = EstrusGRU(
            input_size=actual_input_size,
            hidden_size=train_info.layer_hidden_size,
            num_layers=train_info.num_layers,
            dropout=train_info.dropout_rate,
        )
    else:
        raise ValueError(f"Unknown model_name: {train_info.model_name}")

    return model.to(device)


def main():
    train_info = __train_info()
    saved_file_path = train_info.saved_file_path
    num_feature = train_info.num_feature
    input_size = train_info.input_size
    layer_hidden_size = train_info.layer_hidden_size
    learning_rate = train_info.learning_rate
    use_cell_state = train_info.use_cell_state
    dropout_rate = train_info.dropout_rate
    batch_size = train_info.batch_size
    VERSION_TRAIN = train_info.VERSION_TRAIN

    print(f"本次实验数据读取于 {saved_file_path}")
    X_train, y_train = load_combined_dataset(
        os.path.join(saved_file_path, "train.xlsx"), num_features=num_feature
    )
    X_val, y_val = load_combined_dataset(
        os.path.join(saved_file_path, "val.xlsx"), num_features=num_feature
    )
    print(f"X_train 原始形状: {X_train.shape}")

    # 结果保存
    timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
    saved_result_path = os.path.join(
        result_save_path, f"LSTM", f"{VERSION_TRAIN}_{timestamp}"
    )
    os.makedirs(saved_result_path, exist_ok=True)

    # 模型
    best_model_path = os.path.join(saved_result_path, "best_model.pth")
    # 信息保存
    info_log_path = os.path.join(saved_result_path, "info_log.txt")

    # 验证集历史指标
    history_json_path = os.path.join(saved_result_path, "val_history.json")
    history_excel_path = os.path.join(saved_result_path, "val_history.xlsx")

    # 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = get_model(train_info, device)

    # 使用二元交叉熵损失函数
    criterion = nn.BCELoss()
    """num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    pos_weight_val = torch.tensor([num_neg / num_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)"""

    # 优化器 新增权重衰减(L2正则化)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # 新增：学习率调度器(动态调整学习率)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )

    # 训练指标记录
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_specificity": [],
        "val_f1": [],
        "val_auc": [],
        "val_mcc": [],
    }

    # 创建数据集和数据加载器
    train_loader = DataLoader(
        EstrusDataset(X_train, y_train), batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        EstrusDataset(X_val, y_val),
        batch_size,
    )

    num_epochs = train_info.num_epochs
    early_patience = train_info.early_patience

    early_stopping = EarlyStopping(
        patience=early_patience, verbose=True, path=best_model_path
    )
    early_stopping_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            # batch_y 在 Dataset 中已经经历了 unsqueeze(1)，形状就是 (batch, 1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        record_dict, _, _ = evaluate_model(model, val_loader, criterion, device)
        # 存放指标记录
        history["train_loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(record_dict["avg_loss"])
        history["val_accuracy"].append(record_dict["Accuracy"])
        history["val_precision"].append(record_dict["Precision"])
        history["val_recall"].append(record_dict["Recall"])
        history["val_specificity"].append(record_dict["Specificity"])
        history["val_f1"].append(record_dict["F1-Score"])
        history["val_auc"].append(record_dict["AUC"])
        history["val_mcc"].append(record_dict["MCC"])

        # 学习率调度
        scheduler.step(record_dict["avg_loss"])

        # early_stopping会决定是否保存当前 model 至 best_model_path
        early_stopping(record_dict["avg_loss"], model)

        if early_stopping.early_stop:
            print("Early stopping triggered. Ending training.")
            early_stopping_epoch = epoch
            break

    # 绘图
    saved_pictures_path = os.path.join(saved_result_path, "pictures")
    os.makedirs(saved_pictures_path, exist_ok=True)

    # 绘制训练和验证损失曲线
    myFunction.plot_training_history(history, saved_pictures_path)

    # 信息记录
    with open(info_log_path, "w", encoding="utf-8") as f:
        f.write("-" * 45 + " 信息记录 " + "-" * 45 + "\n")
        f.write("\n")
        f.write(f"记录生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"实验版本 (VERSION): {train_info.VERSION_TRAIN}\n")
        f.write(f"数据读取路径 (saved_file_path): {train_info.saved_file_path}\n")
        f.write(f"结果保存路径 (saved_result_path): {saved_result_path}\n")
        f.write("\n")
        f.write("+" * 30 + " 模型参数 " + "+" * 30 + "\n")
        f.write(f"model_name: {train_info.model_name}\n")
        f.write(f"num_features & input_size : {train_info.num_feature}\n")
        f.write(f"hidden_size: {train_info.layer_hidden_size}\n")
        f.write(f"num_layers: {train_info.num_layers}\n")
        if getattr(train_info, "hidden_sizes", None):
            f.write(f"hidden_sizes (逐层单元数): {train_info.hidden_sizes}\n")
        f.write(f"use_cell_state: {train_info.use_cell_state}\n")
        f.write(f"dropout_rate: {train_info.dropout_rate}\n")
        f.write(f"bidirectional: {train_info.bidirectional}\n")
        if train_info.model_name == "EstrusLSTM_MultiHeadAttn":
            f.write(f"num_heads: {train_info.num_heads}\n")
        f.write(f"batch_size: {train_info.batch_size}\n")
        f.write(f"num_epochs: {train_info.num_epochs}\n")
        f.write(f"early_patience: {train_info.early_patience}\n")
        f.write(f"early_stopping_epoch: {early_stopping_epoch}\n")
        f.write("\n")
        f.write("+" * 30 + " 其它信息 " + "+" * 30 + "\n")
    print(f"数据来源记录已保存至: {info_log_path}")

    # 验证集历史指标
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f)
    print(f"验证集历史指标已保存至: {history_json_path}")
    history_df = pd.DataFrame(history)
    history_df.to_excel(history_excel_path, index_label="epoch")
    print(f"验证集历史指标已保存至: {history_excel_path}")


if __name__ == "__main__":
    main()
