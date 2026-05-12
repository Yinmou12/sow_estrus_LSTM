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
from sow_estrus_LSTM_train import __train_info as TrainInfo

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


def load_combined_dataset(file_name, num_features=2, add_rate: bool = False):
    df = pd.read_excel(file_name, index_col=False)
    y = df["isEstrus"].values
    X_df = df.drop(columns=["isEstrus"])

    if add_rate:
        # 计算温度变化率 (T_t - T_t-1)，第一小时变化率设为0
        temp_values = X_df.values  # 假设输入包含 48 小时耳温
        rate_values = np.diff(temp_values, axis=1)
        rate_values = np.hstack([np.zeros((rate_values.shape[0], 1)), rate_values])

        # 设置列名为 features49 到 features96
        rate_cols = [f"features{i}" for i in range(49, 97)]
        rate_df = pd.DataFrame(rate_values, columns=rate_cols, index=X_df.index)

        # 将变化率特征拼接在原始耳温特征之后
        X_df = pd.concat([X_df, rate_df], axis=1)
        num_features = 2  # 特征数量更新为 2 (耳温 + 变化率)

    X_raw = X_df.values
    seq_len = X_raw.shape[1] // num_features
    X_3d = X_raw.reshape(-1, seq_len, num_features)

    return X_3d, y


class ablation_info:
    saved_file_path = os.path.join(
        info_FINAL_SAVE_PATH, "ablation", "Data_combination_1"
    )

    add_rate: bool = False

    train_info = TrainInfo()
    layer_hidden_size: int = train_info.layer_hidden_size
    num_feature = 2 if add_rate else train_info.num_feature
    input_size = 2 if add_rate else train_info.input_size
    batch_size: int = train_info.batch_size

    use_cell_state: bool = True

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
            os.path.join(saved_file_path, f"{exp_name}.xlsx"),
            num_features=num_feature,
            add_rate=info.add_rate,
        )
        X_val, y_val = load_combined_dataset(
            os.path.join(saved_file_path, "val.xlsx"),
            num_features=num_feature,
            add_rate=info.add_rate,
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

        # 保存路径
        saved_result_path = os.path.join(
            result_save_path, "ablation", "Data_combination_1_addRate", f"{exp_name}"
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

            model = EstrusLSTM(
                input_size=input_size,
                hidden_size=layer_hidden_size,
                use_cell_state=info.use_cell_state,
            ).to(device)

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
                record_dict, _, _ = evaluate_model(model, val_loader, criterion, device)
                val_loss = record_dict["avg_loss"]
                val_accuracy = record_dict["Accuracy"]
                val_precision = record_dict["Precision"]
                val_recall = record_dict["Recall"]
                val_f1 = record_dict["F1-Score"]
                val_auc = record_dict["AUC"]
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

    metric_names = ["Accuracy", "Precision", "Recall", "F1", "AUC", "MCC"]

    file_name = "ablation\\Data_combination_1_addRate"

    history = {}
    avg_test_metrics = {}
    std_test_metrics = {}

    for key, exp_name in exp_names.items():
        save_path = os.path.join(result_save_path, file_name, exp_name)

        X_test, y_test = load_combined_dataset(
            os.path.join(saved_file_path, "test.xlsx"),
            num_features=num_feature,
            add_rate=info.add_rate,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        test_metrics = []
        for i in range(num_runs):
            model_saved_path = os.path.join(save_path, f"best_model_{i+1}.pth")

            model = EstrusLSTM(
                input_size=input_size,
                hidden_size=layer_hidden_size,
                use_cell_state=info.use_cell_state,
            ).to(device)
            model.load_state_dict(torch.load(model_saved_path))

            criterion = nn.BCELoss()
            record_dict, test_labels, test_preds = evaluate_model(
                model,
                DataLoader(EstrusDataset(X_test, y_test), batch_size),
                criterion,
                device,
            )

            avg_los, t_acc, t_precision, t_recall, t_f1, t_auc, t_mcc = (
                record_dict["avg_loss"],
                record_dict["Accuracy"],
                record_dict["Precision"],
                record_dict["Recall"],
                record_dict["F1-Score"],
                record_dict["AUC"],
                record_dict["MCC"],
            )

            run_key = f"{key}_{i+1}"
            history[run_key] = [t_acc, t_precision, t_recall, t_f1, t_auc, t_mcc]
            test_metrics.append([t_acc, t_precision, t_recall, t_f1, t_auc, t_mcc])
        test_metrics = np.array(test_metrics)
        test_metrics = test_metrics.T

        avg_metrics = np.mean(test_metrics, axis=1)
        std_metrics = np.std(test_metrics, axis=1)

        avg_test_metrics[key] = avg_metrics
        std_test_metrics[key] = std_metrics

        for name, avg, std in zip(metric_names, avg_metrics, std_metrics):
            print(f"{name}: 平均值±标准差 : {avg:.4f} ± {std:.4f}")

    # 保存路径准备
    output_root = os.path.join(result_save_path, file_name)
    os.makedirs(output_root, exist_ok=True)

    # 1. 保存详细的历史记录 (每一轮跑的结果)
    history_json_path = os.path.join(output_root, "test_history.json")
    history_excel_path = os.path.join(output_root, "test_history.xlsx")

    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

    history_df = pd.DataFrame.from_dict(history, orient="index", columns=metric_names)
    history_df.to_excel(history_excel_path, index_label="Run_ID")

    # 2. 保存汇总结果 (平均值 ± 标准差)
    summary_data = []
    for key in avg_test_metrics:
        row = {"Experiment": exp_names[key]}
        for i, name in enumerate(metric_names):
            avg = avg_test_metrics[key][i]
            std = std_test_metrics[key][i]
            row[name] = f"{avg:.4f} ± {std:.4f}"
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_excel_path = os.path.join(output_root, "test_summary.xlsx")
    summary_df.to_excel(summary_excel_path, index=False)

    print("-" * 30)
    print(f"预测历史详细记录已保存至: {history_excel_path}")
    print(f"预测汇总结果已保存至: {summary_excel_path}")
    print(f"JSON 格式记录已保存至: {history_json_path}")


def main(_train: bool = True, _predict: bool = True):
    info = ablation_info()
    if _train:
        ablation_train(info)

    if _predict:
        ablation_predict(info)
    return None


if __name__ == "__main__":
    # main(_train=True, _predict=False)
    main(_train=False, _predict=True)
