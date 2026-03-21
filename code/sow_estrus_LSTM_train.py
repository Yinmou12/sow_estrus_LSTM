# 在提交前，养成先运行 status 的习惯，看看哪些文件被动过了：git status
# 如果你修改了多个文件，想全部提交，运行：git add .
# 最后完成本地记录并上传到服务器：
# git commit -m "这里写你的修改说明，例如：优化了模型参数"
# git push origin main

from sow_estrus_LSTM_Info import *
from lstm_model import EstrusLSTM, EarlyStopping
import sow_estrus_LSTM_Function as myFunction

import joblib
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


def evaluate_model(model, loader, criterion, device, save_path=None):
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

    return avg_loss, accuracy, precision, recall, f1, roc_auc


def main():
    # ------ 设置结果保存路径，避免覆盖之前的结果 ------
    version = "SLIDING_WINDOW_SIZE(2)_v1"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = os.path.join(result_save_path, f"{version}_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    print(f"当前结果将保存到: {save_path}")

    # 读取预处理后的数据集
    data = pd.read_excel(
        experimentRecord_data_path + "filled_dataset_SLIDING_WINDOW_SIZE(2)_1.xlsx",
        index_col=False,
    )

    # 分层分组划分数据集，确保同一母猪的数据不会同时出现在训练集、验证集和测试集中
    train_df_raw, val_df_raw, test_df_raw = myFunction.stratified_group_split(data)

    train_df = myFunction.fill_data(train_df_raw)
    val_df = myFunction.fill_data(val_df_raw)
    test_df = myFunction.fill_data(test_df_raw)

    # 划分数据集并进行标准化处理，准备输入 LSTM 模型的格式
    X_train, y_train, train_scaler = myFunction.prepare_univariate_lstm_data(train_df,stride=1)
    print(f"训练集 X_train 形状: {X_train.shape}, y_train 形状: {y_train.shape}")
    X_val, y_val, _ = myFunction.prepare_univariate_lstm_data(val_df, train_scaler,stride=24)
    print(f"验证集 X_val 形状: {X_val.shape}, y_val 形状: {y_val.shape}")
    X_test, y_test, _ = myFunction.prepare_univariate_lstm_data(test_df, train_scaler,stride=24)
    print(f"测试集 X_test 形状: {X_test.shape}, y_test 形状: {y_test.shape}")

    os.makedirs(save_path, exist_ok=True)
    joblib.dump(train_scaler, save_path + "\\scaler.joblib")
    print(f"Scaler saved to {save_path}\\scaler.joblib")

    print(f"X_train 原始形状: {X_train.shape}")
    # 创建数据集和数据加载器
    batch_size = 32
    train_loader = DataLoader(
        EstrusDataset(X_train, y_train), batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        EstrusDataset(X_val, y_val),
        batch_size,
    )
    test_loader = DataLoader(
        EstrusDataset(X_test, y_test),
        batch_size,
    )

    # 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = EstrusLSTM(input_size=1, hidden_size=16).to(device)

    # 使用二元交叉熵损失函数
    criterion = nn.BCELoss()
    """num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    pos_weight_val = torch.tensor([num_neg / num_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)"""
    optimizer = optim.Adam(model.parameters(), lr=0.01)

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

    # 训练模型
    num_epochs = 50

    early_stopping = EarlyStopping(
        patience=5, verbose=True, path=os.path.join(save_path, "best_model.pth")
    )
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc = (
            evaluate_model(model, val_loader, criterion, device)
        )
        history["train_loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Ending training.")
            break

    # 测试集评估
    best_model_path = os.path.join(save_path, "best_model.pth")
    final_model = EstrusLSTM(input_size=1, hidden_size=16).to(device)
    final_model.load_state_dict(torch.load(best_model_path))
    _, t_acc, t_precision, t_recall, t_f1, t_auc = evaluate_model(
        final_model,
        DataLoader(EstrusDataset(X_test, y_test), batch_size),
        criterion,
        device,
        save_path,
    )
    print(f"测试集准确率 (Accuracy): {t_acc:.4f}")
    print(f"测试集精确率 (Precision): {t_precision:.4f}")
    print(f"测试集召回率 (Recall): {t_recall:.4f}")
    print(f"测试集 F1 分数: {t_f1:.4f}")
    print(f"测试集 AUC 指标: {t_auc:.4f}")

    # 绘制训练和验证损失曲线
    myFunction.plot_training_history(history, save_path + "\\pictures")

    # 混淆矩阵
    final_model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = final_model(batch_X)
            preds = (outputs >= 0.5).float().cpu().numpy().flatten()
            test_preds.extend(preds)
            test_labels.extend(batch_y.cpu().numpy().flatten())

    myFunction.plot_matrix(
        test_labels, test_preds, save_dir=save_path
    )


if __name__ == "__main__":
    main()
