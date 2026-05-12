from sow_estrus_LSTM_Info import *
from lstm_model import (
    EarlyStopping,
    EstrusLSTM,
    EstrusLSTM_Attn,
    EstrusGRU,
    EstrusLSTM_MultiHeadAttn,
)
import sow_estrus_LSTM_Function as myFunction
from sow_estrus_LSTM_train import EstrusDataset, evaluate_model

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

    metrics_dict = {}
    metrics_names = [
        "avg_loss",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "AUC",
        "MCC",
    ]

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
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_raw_probs)

    return avg_loss, accuracy, precision, recall, f1, roc_auc, all_labels, all_preds


def load_combined_dataset(file_name, num_features=2, add_rate: bool = False):
    df = pd.read_excel(file_name, index_col=False)
    y = df["isEstrus"].values

    X_raw = df.drop(columns=["isEstrus"]).values

    # X_3d = np.expand_dims(X_raw, axis=-1)
    seq_len = X_raw.shape[1] // num_features
    X_3d = X_raw.reshape(-1, seq_len, num_features)

    return X_3d, y


class ablation_info:
    saved_file_path = os.path.join(
        info_FINAL_SAVE_PATH, "ablation", "Data_combination_1"
    )

    layer_hidden_size: int = 64
    num_feature = 2
    input_size = 2

    batch_size: int = 32

    exp_names = {
        "exp0": "Exp0_Baseline",
        "exp1": "Exp1_ADASYN",
        "exp2": "Exp2_SMOTE",
        "exp3": "Exp3_TomekLinked",
        "exp4": "Exp4_ADASYN_SMOTE",
        "exp5": "Exp5_ADASYN_TomekLinked",
        "exp6": "Exp6_SMOTE_TomekLinked",
        "exp7": "Exp7_Full_Pipeline",
    }

    num_runs: int = 5


def ablation_train(info: ablation_info):
    saved_file_path = info.saved_file_path
    layer_hidden_size = info.layer_hidden_size
    num_feature = info.num_feature
    input_size = info.input_size
    batch_size = info.batch_size
    exp_names = info.exp_names
    num_runs = info.num_runs

    for key, exp_name in exp_names.items():
        print(f"本次实验数据读取于 {saved_file_path}\\{exp_name}")
        X_train, y_train = load_combined_dataset(
            os.path.join(saved_file_path, f"{exp_name}.xlsx"), num_features=num_feature
        )
        X_val, y_val = load_combined_dataset(
            os.path.join(saved_file_path, "val.xlsx"), num_features=num_feature
        )

        train_loader = DataLoader(
            EstrusDataset(X_train, y_train),
            batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            EstrusDataset(X_val, y_val),
            batch_size,
        )

        # 结果保存
        saved_result_path = os.path.join(
            result_save_path, "ablation", "Data_combination_1", f"{exp_name}"
        )
        os.makedirs(saved_result_path, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i in range(num_runs):
            # 模型
            best_model_path = os.path.join(saved_result_path, f"best_model_{i+1}.pth")
            # 验证集历史指标
            history_json_path = os.path.join(
                saved_result_path, f"val_history_{i+1}.json"
            )

            model = EstrusLSTM(input_size=input_size, hidden_size=layer_hidden_size).to(
                device
            )

            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", patience=5, factor=0.5
            )

            history = {
                "train_loss": [],
                "val_loss": [],
                "val_accuracy": [],
                "val_precision": [],
                "val_recall": [],
                "val_f1": [],
                "val_auc": [],
            }

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
                (
                    val_loss,
                    val_accuracy,
                    val_precision,
                    val_recall,
                    val_f1,
                    val_auc,
                    _,
                    _,
                ) = evaluate_model(model, val_loader, criterion, device)
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

            # 验证集历史指标
            with open(history_json_path, "w", encoding="utf-8") as f:
                json.dump(history, f)
            print(f"验证集历史指标已保存至: {history_json_path}")


def ablation_predict(info: ablation_info):
    saved_file_path = info.saved_file_path
    num_feature = info.num_feature
    input_size = info.input_size
    layer_hidden_size = info.layer_hidden_size
    batch_size = info.batch_size
    exp_names = info.exp_names
    num_runs = info.num_runs

    metric_names = ["Accuracy", "Precision", "Recall", "F1", "AUC"]

    file_name = "ablation\\Data_combination_1"

    history = {}
    avg_test_metrics = {}
    std_test_metrics = {}

    for key, exp_name in exp_names.items():
        save_path = os.path.join(result_save_path, file_name, exp_name)

        X_test, y_test = load_combined_dataset(
            os.path.join(saved_file_path, "test.xlsx"), num_features=num_feature
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        test_metrics = []
        for i in range(num_runs):
            model_saved_path = os.path.join(save_path, f"best_model_{i+1}.pth")

            model = EstrusLSTM(input_size=input_size, hidden_size=layer_hidden_size).to(
                device
            )
            model.load_state_dict(torch.load(model_saved_path))

            criterion = nn.BCELoss()
            _, t_acc, t_precision, t_recall, t_f1, t_auc, test_labels, test_preds = (
                evaluate_model(
                    model,
                    DataLoader(EstrusDataset(X_test, y_test), batch_size),
                    criterion,
                    device,
                )
            )
            run_key = f"{key}_{i+1}"
            history[run_key] = [t_acc, t_precision, t_recall, t_f1, t_auc]
            test_metrics.append([t_acc, t_precision, t_recall, t_f1, t_auc])
        test_metrics = np.array(test_metrics)
        test_metrics = test_metrics.T

        avg_metrics = np.mean(test_metrics, axis=1)
        std_metrics = np.std(test_metrics, axis=1)

        avg_test_metrics[key] = avg_metrics
        std_test_metrics[key] = std_metrics

        for name, avg, std in zip(metric_names, avg_metrics, std_metrics):
            print(f"{name}: 平均值±标准差 : {avg:.4f} ± {std:.4f}")

    history_json_path = os.path.join(result_save_path, file_name, "test_history.json")
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f)


def main(_train: bool = True, _predict: bool = True):
    info = ablation_info()
    if _train:
        ablation_train(info)

    if _predict:
        ablation_predict(info)
    return None


if __name__ == "__main__":
    main(_train=False, _predict=True)
