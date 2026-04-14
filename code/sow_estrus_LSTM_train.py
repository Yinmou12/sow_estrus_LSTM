# 在提交前，养成先运行 status 的习惯，看看哪些文件被动过了：git status
# 如果你修改了多个文件，想全部提交，运行：git add .
# 最后完成本地记录并上传到服务器：
# git commit -m "这里写你的修改说明，例如：优化了模型参数"
# git push origin main

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
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


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

            probs = (outputs >= 0.5).cpu().numpy().flatten()
            all_raw_probs.extend(probs)

            preds = (probs >= 0.5).astype(float)
            all_preds.extend(preds)

            all_labels.extend(batch_y.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_raw_probs)

    return avg_loss, accuracy, precision, recall, f1, roc_auc, all_preds, all_labels


def load_combined_dataset(file_name, num_features=2):
    df = pd.read_excel(file_name, index_col=False)
    y = df["label_isEstrus"].values

    X_raw = df.drop(columns=["label_isEstrus"]).values

    # X_3d = np.expand_dims(X_raw, axis=-1)
    seq_len = X_raw.shape[1] // num_features
    X_3d = X_raw.reshape(-1, seq_len, num_features)

    return X_3d, y


# 修改第二个参数以获取存放路径
saved_file_path = os.path.join(
    info_FINAL_SAVE_PATH, "DATA_NotCor_AddTempRate_2026_0406_1213"
)
layer_hidden_size = 64

VERSION_TRAIN = "BiLSTM_MultiHeadAttn"


def main():
    print(f"本次实验数据读取于 {saved_file_path}")
    X_train, y_train = load_combined_dataset(
        os.path.join(saved_file_path, "train.xlsx")
    )
    X_val, y_val = load_combined_dataset(os.path.join(saved_file_path, "val.xlsx"))
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
    # model = EstrusLSTM(input_size=2, hidden_size=layer_hidden_size).to(device)
    model = EstrusLSTM_MultiHeadAttn(input_size=2, hidden_size=layer_hidden_size).to(
        device
    )

    # 使用二元交叉熵损失函数
    criterion = nn.BCELoss()
    """num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    pos_weight_val = torch.tensor([num_neg / num_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)"""

    # 优化器 新增权重衰减(L2正则化)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
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
        "val_f1": [],
        "val_auc": [],
    }

    # 创建数据集和数据加载器
    batch_size = 32
    train_loader = DataLoader(
        EstrusDataset(X_train, y_train), batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        EstrusDataset(X_val, y_val),
        batch_size,
    )

    num_epochs = 100
    early_patience = 7

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
        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc, _, _ = (
            evaluate_model(model, val_loader, criterion, device)
        )
        # 存放指标记录
        history["train_loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)

        # 学习率调度
        scheduler.step(val_loss)

        # early_stopping会决定是否保存当前 model 至 best_model_path
        early_stopping(val_loss, model)

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
        f.write(f"实验版本 (VERSION): {VERSION_TRAIN}\n")
        f.write(f"数据读取路径 (saved_file_path): {saved_file_path}\n")
        f.write(f"结果保存路径 (saved_result_path): {saved_result_path}\n")
        f.write("\n")
        f.write("+" * 45 + "\n")
        f.write(f"hidden_size: {layer_hidden_size}\t")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"early_patience: {early_patience}\t")
        f.write(f"early_stopping_epoch: {early_stopping_epoch}\n")
        f.write(f"dropout_rate=0.5\t")
        f.write("\n")
        f.write("+" * 20 + " 其它信息 " + "+" * 20 + "\n")
        f.write(f"新增温度变化率特征\n")
        f.write(f"新增多头注意力机制\n")
        # f.write(f"池化层采用torch.max捕捉最显著的温升特征\n")
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
