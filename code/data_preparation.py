from sow_estrus_LSTM_Info import *
import sow_estrus_LSTM_Function as myFunction

import joblib
import os
import pandas as pd
from datetime import datetime

pd.set_option("future.no_silent_downcasting", True)

VERSION_DP = "DATA_AST_AddTempRate"
timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
SAVE_PATH_DP = os.path.join(info_FINAL_SAVE_PATH, f"{VERSION_DP}_{timestamp}")


def sample_choice(processed_dataset: pd.DataFrame = None):
    """
    从已完成 data_processing 的小时数据中选择非发情母猪样本。

    不传入 DataFrame 时，读取 processed_dataset_2026_0913_1609.xlsx。
    同一 sSowsNo 只要出现过 isEstrus == 1，就保留该母猪的全部记录。
    非发情母猪分别从首个可用的 09:00 / 16:00 开始，每次推进 48 小时，
    生成以 08:00 / 15:00 结束的窗口（首尾均包含，共 48 个小时位置）。
    超出该母猪记录起止范围的窗口舍弃；窗口内部缺失的小时保留原样，
    不在此处补齐；完全没有记录的窗口跳过。

    返回含 sSowsNo_split 的 DataFrame，不修改输入或原始 sSowsNo。
    非发情分段命名为 母猪编号_M_1 / 母猪编号_A_1，各自按保留顺序编号；
    发情母猪的 sSowsNo_split 使用原编号。后续补齐应按分段编号分组。
    """
    if processed_dataset is None:
        processed_dataset = pd.read_excel(
            os.path.join(
                experimentRecord_data_path,
                "processed_dataset",
                "processed_dataset_2026_0913_1609.xlsx",
            ),
            dtype={"sSowsNo": str},
        )

    data = processed_dataset.copy()
    data["sSowsNo"] = data["sSowsNo"].astype(str)
    data["tLastUploadTime"] = pd.to_datetime(data["tLastUploadTime"])
    data = data.sort_values(["sSowsNo", "tLastUploadTime"])
    data["sSowsNo_split"] = data["sSowsNo"]

    estrus_sows = set(data.loc[data["isEstrus"] == 1, "sSowsNo"])
    selected_windows = []
    window_step = pd.Timedelta(hours=48)
    window_span = pd.Timedelta(hours=47)

    for sow_no, sow_data in data.groupby("sSowsNo", sort=False):
        if sow_no in estrus_sows:
            selected_windows.append(sow_data)
            continue

        first_time = sow_data["tLastUploadTime"].min()
        last_time = sow_data["tLastUploadTime"].max()

        # M / A 是两套独立分段，同一套内部不重复，两套之间允许重叠。
        for period, start_hour in (("M", 9), ("A", 16)):
            window_start = first_time.normalize() + pd.Timedelta(hours=start_hour)
            if window_start < first_time:
                window_start += pd.Timedelta(days=1)

            sample_number = 1
            while window_start + window_span <= last_time:
                window_end = window_start + window_span
                window = sow_data.loc[
                    sow_data["tLastUploadTime"].between(window_start, window_end)
                ].copy()
                if not window.empty:
                    window["sSowsNo_split"] = f"{sow_no}_{period}_{sample_number}"
                    selected_windows.append(window)
                    sample_number += 1
                window_start += window_step

    if not selected_windows:
        return data.iloc[:0].reset_index(drop=True)
    return pd.concat(selected_windows, ignore_index=True)


def sample_split():
    # 对已处理的小时数据选择样本，结果另存，不覆盖预处理数据。
    """sampled_save_dir = os.path.join(experimentRecord_data_path, "sampled_dataset")
    os.makedirs(sampled_save_dir, exist_ok=True)
    sampled_save_path = os.path.join(
        sampled_save_dir, f"sampled_dataset_{timestamp}.xlsx"
    )"""

    # 样本选择
    """ sampled_dataset = sample_choice()
    sampled_dataset.to_excel(sampled_save_path, index=False)
    print(f"样本选择完成，共 {len(sampled_dataset)} 行，已保存到: {sampled_save_path}") """

    # 统计样本数量
    """ sampled_input_path = os.path.join(
        sampled_save_dir, "sampled_dataset_2026_0914_1900.xlsx"
    )
    sampled_dataset = pd.read_excel(
        sampled_input_path, dtype={"sSowsNo": str, "sSowsNo_split": str}
    )
    # 按分段编号统计样本数；同一母猪出现过发情标签，其所属样本归为发情。
    estrus_sows = sampled_dataset.loc[
        sampled_dataset["isEstrus"] == 1, "sSowsNo"
    ].unique()
    estrus_mask = sampled_dataset["sSowsNo"].isin(estrus_sows)
    estrus_sample_count = sampled_dataset.loc[estrus_mask, "sSowsNo_split"].nunique()
    not_estrus_sample_count = sampled_dataset.loc[
        ~estrus_mask, "sSowsNo_split"
    ].nunique()
    print(f"发情样本数量: {estrus_sample_count}")
    print(f"非发情样本数量: {not_estrus_sample_count}")
    print(f"样本总数量: {estrus_sample_count + not_estrus_sample_count}") """

    """
        划分数据
    """
    splited_dataset = pd.read_excel(
        os.path.join(
            experimentRecord_data_path,
            "splited_dataset",
            "splited_dataset_2026_0914_1930.xlsx",
        ),
        index_col=False,
    )
    train_df_raw, val_df_raw, test_df_raw = myFunction.stratified_group_split(
        splited_dataset,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        random_count=263,
    )

    # 在各自集合内逐样本筛选、补齐，保持划分时的母猪归属
    print("------ 训练集补齐 ------")
    train_df = myFunction.fill_data(train_df_raw)
    print("------ 验证集补齐 ------")
    val_df = myFunction.fill_data(val_df_raw)
    print("------ 测试集补齐 ------")
    test_df = myFunction.fill_data(test_df_raw)

    """
        交叉验证数据准备
    """
    # 合并补齐后的训练集和验证集，后续在其中按 sSowsNo 进行交叉验证
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    # 沿用划分前的母猪类别，避免质量筛选移除发情标签后影响样本统计。
    estrus_sows = splited_dataset.loc[
        splited_dataset["isEstrus"] == 1, "sSowsNo"
    ].unique()
    sample_count = train_val_df["sSowsNo_split"].nunique()
    positive_count = train_val_df.loc[
        train_val_df["sSowsNo"].isin(estrus_sows), "sSowsNo_split"
    ].nunique()
    print("------ 合并后的训练集（用于交叉验证）------")
    print(
        f"母猪 {train_val_df['sSowsNo'].nunique()} 头，样本总数 {sample_count}，"
        f"正样本（发情）{positive_count}，负样本（非发情）{sample_count - positive_count}"
    )
    save_index = 10
    train_val_df.to_excel(
        os.path.join(
            experimentRecord_data_path,
            "cross_validation_dataset",
            str(save_index),
            "train_val_df.xlsx",
        ),
        index=False,
    )
    test_df.to_excel(
        os.path.join(
            experimentRecord_data_path,
            "cross_validation_dataset",
            str(save_index),
            "test.xlsx",
        ),
        index_col=False,
    )


def processing_feature(data: pd.DataFrame = None, save_index=1, estrus_sows=None):
    """将每个 sSowsNo_split 的48小时记录按时间升序展开为一行。

    返回102列：5列基本信息、48列温度、48列温度变化率、isEstrus。
    编号及品牌保持样本内一致，tLastUploadTime 使用样本结束时间；
    特征值直接展开，不重新计算。输入数据不修改，结果不自动写入文件。

    未传入 data 时读取 cross_validation_dataset/{save_index}/train_val_df.xlsx，
    并从划分前的 splited_dataset_2026_0914_1930.xlsx 恢复母猪类别。
    传入 data 时，默认按其中出现过 isEstrus == 1 的母猪确定正样本；
    若此前筛选可能移除该母猪的所有发情标签，应通过 estrus_sows 传入
    筛选前的发情母猪编号集合，其编号类型应与 data 中的 sSowsNo 一致。
    """
    if data is None:
        data = pd.read_excel(
            os.path.join(
                experimentRecord_data_path,
                "cross_validation_dataset",
                str(save_index),
                "train_val_df.xlsx",
            ),
            dtype={"sEarTagCode": str, "sSowsNo": str, "sSowsNo_split": str},
        )
        if estrus_sows is None:
            label_source = pd.read_excel(
                os.path.join(
                    experimentRecord_data_path,
                    "splited_dataset",
                    "splited_dataset_2026_0914_1930.xlsx",
                ),
                usecols=["sSowsNo", "isEstrus"],
                dtype={"sSowsNo": str},
            )
            unknown_sows = set(data["sSowsNo"]) - set(label_source["sSowsNo"])
            if unknown_sows:
                raise ValueError(f"原始标签数据中缺少母猪编号: {sorted(unknown_sows)}")
            estrus_sows = label_source.loc[
                label_source["isEstrus"] == 1, "sSowsNo"
            ].unique()

    metadata_cols = ["sEarTagCode", "sSowsNo", "sSowsNo_split", "sBrand"]
    output_cols = (
        metadata_cols
        + ["tLastUploadTime"]
        + [f"iTemperature_{i}" for i in range(1, 49)]
        + [f"temperatureRate_{i}" for i in range(1, 49)]
        + ["isEstrus"]
    )
    required = set(
        metadata_cols
        + ["tLastUploadTime", "iTemperature", "temperatureRate", "isEstrus"]
    )
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"缺少必要列: {sorted(missing)}")
    if data[["sSowsNo", "sSowsNo_split"]].isna().any().any():
        raise ValueError("母猪编号和样本编号不能缺失")

    record_dataset = data.copy()
    record_dataset["tLastUploadTime"] = pd.to_datetime(
        record_dataset["tLastUploadTime"]
    )
    if estrus_sows is None:
        estrus_sows = record_dataset.loc[
            record_dataset["isEstrus"] == 1, "sSowsNo"
        ].unique()
    positive_sows = set(estrus_sows)

    rows = []
    for split_id, group in record_dataset.groupby("sSowsNo_split", sort=False):
        group = group.sort_values("tLastUploadTime")
        if len(group) != 48:
            raise ValueError(f"样本 {split_id} 应有48行，实际为 {len(group)} 行")
        times = group["tLastUploadTime"]
        if (
            times.isna().any()
            or times.ne(times.dt.floor("h")).any()
            or not times.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
        ):
            raise ValueError(f"样本 {split_id} 的时间必须为48个连续且不重复的整点小时")
        if group[metadata_cols].nunique(dropna=False).gt(1).any():
            raise ValueError(f"样本 {split_id} 的编号或品牌信息不一致")
        if group[["iTemperature", "temperatureRate"]].isna().any().any():
            raise ValueError(f"样本 {split_id} 存在缺失特征，请先完成补齐")
        if not group["isEstrus"].isin([0, 1]).all():
            raise ValueError(f"样本 {split_id} 的 isEstrus 必须为0或1")

        last = group.iloc[-1]
        rows.append(
            last[metadata_cols + ["tLastUploadTime"]].tolist()
            + group["iTemperature"].tolist()
            + group["temperatureRate"].tolist()
            + [int(last["sSowsNo"] in positive_sows)]
        )

    return pd.DataFrame(rows, columns=output_cols)


def convert_feature(path_index=1):
    _path = os.path.join(
        experimentRecord_data_path, "cross_validation_dataset", str(path_index)
    )
    train_val_data_path = os.path.join(_path, "train_val_df.xlsx")
    test_data_path = os.path.join(_path, "test.xlsx")

    train_val_data = pd.read_excel(
        train_val_data_path,
        dtype={"sEarTagCode": str, "sSowsNo": str, "sSowsNo_split": str},
    )
    test_data = pd.read_excel(
        test_data_path, dtype={"sEarTagCode": str, "sSowsNo": str, "sSowsNo_split": str}
    )
    train_val_df = processing_feature(train_val_data)
    test_df = processing_feature(test_data)
    train_val_df.to_excel(os.path.join(_path, "train_val_features.xlsx"), index=False)
    test_df.to_excel(os.path.join(_path, "test_features.xlsx"), index=False)
    return None


if __name__ == "__main__":
    # 样本选择与划分
    # sample_split()

    # 特征维度转换
    """for i in range(1, 11):
    convert_feature(path_index=i)"""

    # ++++++++++ 最终实验数据 ++++++++++
    # 单温度变量
    """ X_train, y_train, train_scaler = myFunction.prepare_univariate_lstm_data(df_tomek)
    print(f"训练集 X_train 形状: {X_train.shape}, y_train 形状: {y_train.shape}")
    X_val, y_val, _ = myFunction.prepare_univariate_lstm_data(val_df, train_scaler)
    print(f"验证集 X_val 形状: {X_val.shape}, y_val 形状: {y_val.shape}")
    X_test, y_test, _ = myFunction.prepare_univariate_lstm_data(test_df, train_scaler)
    print(f"测试集 X_test 形状: {X_test.shape}, y_test 形状: {y_test.shape}") """
    # 增加温度变化率
    """ X_train, y_train, train_scaler = myFunction.prepare_lstm_data(df_final)
    print(f"训练集 X_train 形状: {X_train.shape}, y_train 形状: {y_train.shape}")
    X_val, y_val, _ = myFunction.prepare_lstm_data(val_df, train_scaler)
    print(f"验证集 X_val 形状: {X_val.shape}, y_val 形状: {y_val.shape}")
    X_test, y_test, _ = myFunction.prepare_lstm_data(test_df, train_scaler)
    print(f"测试集 X_test 形状: {X_test.shape}, y_test 形状: {y_test.shape}") """

    # 保存
    """ os.makedirs(SAVE_PATH_DP, exist_ok=True)
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
    save_combined_dataset(X_test, y_test, "test", SAVE_PATH_DP) """
