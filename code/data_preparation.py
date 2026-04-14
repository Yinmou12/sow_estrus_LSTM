from sow_estrus_LSTM_Info import *
import sow_estrus_LSTM_Function as myFunction

import joblib
import os
import pandas as pd
from datetime import datetime

pd.set_option("future.no_silent_downcasting", True)

VERSION_DP = "DATA_NotCor_AddTempRate"
timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
SAVE_PATH_DP = os.path.join(info_FINAL_SAVE_PATH, f"{VERSION_DP}_{timestamp}")

if __name__ == "__main__":
    # 原始数据读取
    """dataset = pd.read_excel(summary_data_path + "add_2025-07-08.xlsx", sheet_name=None)
    data = pd.DataFrame(columns=columns_name)
    for sheet_name, sheet_data in dataset.items():
        # 确保 sheet_data 是 DataFrame 类型
        if isinstance(sheet_data, pd.DataFrame):
            data = pd.concat([data, sheet_data], axis=0, ignore_index=False)"""

    # 预处理
    """
            异常体温处理
            更新小时数据
            计算温度变化率
            设置标签
    """
    """processed_dataset = myFunction.data_processing(
        data, TIME_CHOICE, False, "mean", DEL_CODE
    )"""

    # 拆分多次发情的数据并更新发情编号
    """splited_dataset = myFunction.split_estrusData(processed_dataset, TIME_CHOICE, 7)
    splited_dataset.to_excel(
        experimentRecord_data_path
        + f"splited_dataset\\splited_dataset_{timestamp}.xlsx",
        index=False,
    )"""
    splited_dataset = pd.read_excel(
        experimentRecord_data_path
        + "splited_dataset\\splited_dataset_2026_0406_1140.xlsx",
        index_col=False,
    )

    # 分层分组划分数据集，确保同一母猪的数据不会同时出现在训练集、验证集和测试集中
    train_df_raw, val_df_raw, test_df_raw = myFunction.stratified_group_split(
        splited_dataset, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2
    )

    # 数据填补
    train_df = myFunction.fill_data(train_df_raw)
    val_df = myFunction.fill_data(val_df_raw)
    test_df = myFunction.fill_data(test_df_raw)

    # ++++++++++ 最终实验数据 ++++++++++
    # 单温度变量
    """X_train, y_train, train_scaler = myFunction.prepare_univariate_lstm_data(train_df)
    print(f"训练集 X_train 形状: {X_train.shape}, y_train 形状: {y_train.shape}")
    X_val, y_val, _ = myFunction.prepare_univariate_lstm_data(val_df, train_scaler)
    print(f"验证集 X_val 形状: {X_val.shape}, y_val 形状: {y_val.shape}")
    X_test, y_test, _ = myFunction.prepare_univariate_lstm_data(test_df, train_scaler)
    print(f"测试集 X_test 形状: {X_test.shape}, y_test 形状: {y_test.shape}")"""
    # 增加温度变化率
    X_train, y_train, train_scaler = myFunction.prepare_lstm_data(train_df)
    print(f"训练集 X_train 形状: {X_train.shape}, y_train 形状: {y_train.shape}")
    X_val, y_val, _ = myFunction.prepare_lstm_data(val_df, train_scaler)
    print(f"验证集 X_val 形状: {X_val.shape}, y_val 形状: {y_val.shape}")
    X_test, y_test, _ = myFunction.prepare_lstm_data(test_df, train_scaler)
    print(f"测试集 X_test 形状: {X_test.shape}, y_test 形状: {y_test.shape}")

    # 保存
    os.makedirs(SAVE_PATH_DP, exist_ok=True)
    print(f"数据将保存到: {SAVE_PATH_DP}")

    sacler_path = os.path.join(SAVE_PATH_DP, "train_scaler.joblib")
    joblib.dump(train_scaler, sacler_path)

    def save_combined_dataset(X, y, name, save_path):
        X_2d = X.reshape(X.shape[0], -1)
        # columns_name = [f"temperature_{i+1}" for i in range(X.shape[1])]
        columns_name = [f"feature{i+1}" for i in range(X_2d.shape[1])]
        combined_df = pd.DataFrame(X_2d, columns=columns_name)
        combined_df["label_isEstrus"] = y
        file_name = f"{name}.xlsx"
        full_path = os.path.join(save_path, file_name)
        combined_df.to_excel(full_path, index=False)

    save_combined_dataset(X_train, y_train, "train", SAVE_PATH_DP)
    save_combined_dataset(X_val, y_val, "val", SAVE_PATH_DP)
    save_combined_dataset(X_test, y_test, "test", SAVE_PATH_DP)
