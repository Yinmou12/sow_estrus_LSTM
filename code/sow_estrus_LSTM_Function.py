from sow_estrus_LSTM_Info import *
from correct_abnormal_temperatures import (
    correct_abnormal_temperatures_linear,
    correct_abnormal_temperatures_moving_avg,
    correct_abnormal_temperatures_spline,
    correct_abnormal_temperatures_circadian,
    correct_abnormal_temperatures_ensemble,
)
import sow_estrus_LSTM_Function as myFunction

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os as os
import random
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# 数据提取
def intercept_data(
    data: pd.DataFrame,  # 数据集
    sowNo,  # 标号
    first_time: str,  # 起始时间
    last_time: str,  # 结束时间
    output_file_path=None,  # 输出路径
    is_save=False,  # 是否保存
):
    data["sSowsNo"] = data["sSowsNo"].astype(str)
    # 转换日期列为datetime格式
    data["tLastUploadTime"] = pd.to_datetime(data["tLastUploadTime"])

    # 将sSowsNo列转换为字符串类型
    data["sSowsNo"] = data["sSowsNo"].astype(str)
    # 将sowNo转换为字符串类型
    sowNo = str(sowNo)

    # 截取数据
    filtered_data = data[
        (data["tLastUploadTime"] >= first_time)
        & (data["tLastUploadTime"] <= last_time)
        & (data["sSowsNo"] == sowNo)
    ]

    # 将结果以`.xlsx`文件保存到指定输出路径
    # **使用openpyxl会出现保存后的文件无法打开或者打开后文件中没有数据的情况**
    if is_save:
        if output_file_path:
            filtered_data.to_excel(output_file_path, index=True)
        else:
            # 使用os模块创建文件
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, "w") as file:
                pass

    return filtered_data


# 数据选择
def estrusSows_data_choice(
    data: pd.DataFrame,
    time_choice,
):
    data["sSowsNo"] = data["sSowsNo"].astype(str)
    # 所有发情母猪耳标号
    all_estrusSows_ear_codes = []

    # 记录发情母猪数据
    estrusSows_dataset = pd.DataFrame(columns=columns_name)
    estrusSows_list = []

    # 提取信息
    for info in time_choice:
        # 耳号
        estrus_ear_tag_codes = []
        str_ear_tag_codes = info.split(sep="_")[-1]
        for code in str_ear_tag_codes.split(sep=","):
            estrus_ear_tag_codes.append(str(code))
            all_estrusSows_ear_codes.append(str(code))

        # 日期
        date = pd.to_datetime(info.split(sep="_")[0])

        # 早上或下午
        AorM = info.split(sep="_")[1]
        time = None
        if AorM == "A":
            time = pd.to_datetime("15:00:00")
        elif AorM == "M":
            time = pd.to_datetime("08:00:00")

        # 时间段
        prev_time = 0
        # last_time = 0
        if time != None:
            date_time = pd.to_datetime(f"{date.date()} {time.time()}")
            prev_time = date_time - pd.Timedelta(hours=WINDOW_SIZE - 1)
            # last_time = date_time + pd.Timedelta(hours=SLIDING_WINDOW_SIZE)
            last_time = date_time + pd.Timedelta(hours=1)

            # 提取发情母猪数据
            for code in estrus_ear_tag_codes:
                code = str(code)
                temp_data = intercept_data(
                    data[data["sSowsNo"] == code], code, prev_time, last_time
                )
                if not temp_data.empty:
                    estrusSows_list.append(temp_data)
    estrusSows_dataset = pd.concat(estrusSows_list, axis=0, ignore_index=True)

    # 未发情母猪的数据
    """notEstrusSows_dataset = pd.DataFrame(columns=columns_name)
    for code in data["sSowsNo"].drop_duplicates(keep="first").to_numpy():
        if code in all_estrusSows_ear_codes:
            continue
        else:
            notEstrusSows_dataset = pd.concat(
                [
                    notEstrusSows_dataset,
                    data[data["sSowsNo"] == code],
                ],
                axis=0,
                ignore_index=True,
            )"""
    not_estrus_mask = ~data["sSowsNo"].isin(all_estrusSows_ear_codes)
    notEstrusSows_dataset = data[not_estrus_mask].copy()

    # 合并
    final_dataset = pd.concat(
        [estrusSows_dataset, notEstrusSows_dataset], axis=0, ignore_index=True
    ).sort_values(by=["sSowsNo", "tLastUploadTime"])

    return final_dataset


# 更新小时数据
def update_hourly_temperature_step(
    dataset: pd.DataFrame,
    start_time,
    end_time,
    operator_temperature="mean",
    time_split: str = "1h",
):
    """
    参数：
        dataset 要处理的数据集
        operator_temperature 对每小时体温的操作，默认为取均值

    修改:解决因时间跨度导致的数据缺失问题,参考耳标号4577
    """
    dataset["sSowsNo"] = dataset["sSowsNo"].astype(str)
    ear_tag_codes = dataset["sSowsNo"].drop_duplicates(keep="first").to_numpy()

    data = dataset.copy()
    data["tLastUploadTime"] = pd.to_datetime(data["tLastUploadTime"])
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    final_hourly_data = pd.DataFrame(columns=columns_name)

    # 将 time_split 转为 Timedelta，用于判断断点
    split_td = pd.to_timedelta(time_split)
    gap_threshold = split_td * 2  # 相邻记录间隔大于此阈值则视为不同连续段（可调整）

    for ear_code in ear_tag_codes:
        ear_code = str(ear_code)
        tempDataframe = intercept_data(data, ear_code, start_time, end_time)
        if tempDataframe.empty:
            continue

        # 创建新的列存储调整后的步数
        tempDataframe.loc[:, "iStep_diff"] = tempDataframe.loc[:, "iStep"].diff()

        # 保存其他需要列的信息
        others_columns = tempDataframe.drop(
            columns=["tLastUploadTime", "iTemperature", "iStep", "iStep_diff"]
        ).reset_index(drop=True)

        # 设置时间列为索引，便于分段与 resample
        temp = tempDataframe.set_index("tLastUploadTime", drop=False)

        # 找出断点：相邻时间差大于 gap_threshold 的地方就是新段开始
        time_diffs = temp.index.to_series().diff()
        new_segment = (time_diffs > gap_threshold).fillna(False)
        segment_id = new_segment.cumsum()

        # 对每个连续 segment 单独 resample 并合并
        segment_results = []
        for seg in segment_id.unique():
            seg_idx = segment_id[segment_id == seg].index
            seg_df = temp.loc[seg_idx]

            if seg_df.empty:
                continue

            # 对该段进行 resample 聚合
            resampled = seg_df.resample(time_split).agg(
                {
                    "iTemperature": operator_temperature,
                    "iStep_diff": "sum",
                }
            )
            if resampled.empty:
                continue

            resampled = resampled.reset_index()

            resampled.loc[:, "iStep"] = resampled.loc[:, "iStep_diff"]
            resampled = resampled.drop(columns="iStep_diff")

            # 取该耳号该段的其他列值（通常相同），用第一行广播
            if not others_columns.empty:
                # 找到原段在 others_columns 中对应的行索引范围
                # 因为 others_columns 与 tempDataframe 行一一对应，取该段第一个原行的 others 列
                first_row_idx = seg_df.index[0]
                # 在原 tempDataframe 中找到位置
                orig_pos = tempDataframe.index.get_loc(
                    tempDataframe.index[
                        tempDataframe["tLastUploadTime"]
                        == seg_df["tLastUploadTime"].iloc[0]
                    ][0]
                )
                base_other = others_columns.iloc[orig_pos : orig_pos + 1].reset_index(
                    drop=True
                )
                # 重复以匹配 resampled 的行数
                other_repeated = pd.concat(
                    [base_other] * len(resampled), ignore_index=True
                )
                concat_data = pd.concat(
                    [other_repeated, resampled.reset_index(drop=True)],
                    axis=1,
                    ignore_index=False,
                )
            else:
                # 若没有其他列，则直接使用 resampled
                concat_data = resampled

            # 丢弃全为空的行（与原逻辑一致）
            concat_data = concat_data.dropna(
                subset=["tLastUploadTime", "iTemperature", "iStep"], how="all"
            )
            segment_results.append(concat_data)

        if segment_results:
            ear_hourly = pd.concat(segment_results, axis=0, ignore_index=True)
            final_hourly_data = pd.concat(
                [final_hourly_data, ear_hourly], axis=0, ignore_index=True
            )

    # 排序并返回
    final_hourly_data = final_hourly_data.sort_values(
        by=["sSowsNo", "tLastUploadTime"]
    ).reset_index(drop=True)
    return final_hourly_data


# 计算温度变化率
def calculate_temperatureRate(
    data: pd.DataFrame,  # 处理过的以小时为单位的数据集
    start_time: str,  # 起始时间
    end_time: str,  # 结束时间
):
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    # data["tLastUploadTime"] = pd.to_datetime(data["tLastUploadTime"])

    # 用于记录处理过程中的数据
    final_data_with_temperatureRate = pd.DataFrame(columns=columns_name)

    sowNos = data["sSowsNo"].drop_duplicates(keep="first").to_numpy()
    # 按标号处理
    for sowNo in sowNos:
        sow_data = data[
            (data["sSowsNo"] == sowNo)
            & (data["tLastUploadTime"] >= start_time)
            & (data["tLastUploadTime"] <= end_time)
        ].copy()

        try:
            # 计算相邻时间点的温度变化率
            sow_data.loc[:, "temperatureRate"] = sow_data.loc[:, "iTemperature"].diff()
        except ZeroDivisionError:
            sow_data["temperatureRate"] = 0
        except Exception as e:
            print(f"Error occurred when processing sowNo {sowNo}: {e}")

        # 将结果添加到 final_data_with_temperatureRate 中
        if not sow_data.empty:
            sow_data = sow_data.dropna(axis=1, how="all")
            final_data_with_temperatureRate = pd.concat(
                [final_data_with_temperatureRate, sow_data], axis=0, ignore_index=True
            )

    return final_data_with_temperatureRate


# 数据处理
def data_processing(
    source_dataset: pd.DataFrame,  # 数据集
    time_choice,  # 发情数据的时间选择
    correct_temperature=False,  # 是否执行异常体温处理
    resampling_method="mean",
    del_code=[],  # 要删除数据的编号
):
    record_dataset = source_dataset.copy()
    # record_dataset = record_dataset[record_dataset["iTemperature"] > 25]
    record_dataset["sSowsNo"] = record_dataset["sSowsNo"].astype(str)
    # 删除一些异常数据
    if len(del_code) > 0:
        for code in del_code:
            code_str = str(code)
            record_dataset = record_dataset[record_dataset["sSowsNo"] != code_str]

    # -------------------- 获取发情时间，编号 --------------------
    estrus_time = []
    estrus_ear_tag_codes = []
    unkonwnTime_ear_tag_codes = []
    all_estrus_time = []
    for info in time_choice:
        # 日期
        date = pd.to_datetime(info.split(sep="_")[0])
        # 早上或下午
        AorM = info.split(sep="_")[1]

        time = None
        if AorM == "A":
            time = pd.to_datetime("15:00:00")
        elif AorM == "M":
            time = pd.to_datetime("08:00:00")

        # 耳号
        str_ear_tag_codes = info.split(sep="_")[-1]

        # 时间段
        if time != None:
            date_time = pd.to_datetime(f"{date.date()} {time.time()}")
            prev_time = date_time - pd.Timedelta(hours=WINDOW_SIZE - 1)
            # last_time = date_time + pd.Timedelta(hours=SLIDING_WINDOW_SIZE)
            last_time = date_time
            all_estrus_time.append(str(prev_time) + "~" + str(last_time))

            temp_list = []
            for code in str_ear_tag_codes.split(sep=","):
                temp_list.append(code)
            estrus_ear_tag_codes.append(temp_list)
            estrus_time.append(str(prev_time) + "~" + str(last_time))
        else:
            for code in str_ear_tag_codes.split(sep=","):
                unkonwnTime_ear_tag_codes.append(code)

    # -------------------- 异常体温处理 --------------------
    correct_dataset = pd.DataFrame()
    if correct_temperature:
        # correct_dataset = correct_abnormal_temperatures_spline(record_dataset, 34, 50)
        correct_dataset = correct_abnormal_temperatures_moving_avg(
            record_dataset, 34, 50
        )
        # print(f"异常体温处理:{type(record_dataset)}")
    else:
        correct_dataset = record_dataset.copy()

    # -------------------- 数据选择 --------------------
    # 选择发情母猪48小时数据，对非发情母猪数据复制
    choose_data = estrusSows_data_choice(correct_dataset, time_choice)
    # choose_data.to_excel(test_data_path +"loss_of_time\\choose_data.xlsx",index=False)

    # -------------------- 计算小时数据 --------------------
    # 小时平均体温
    hourly_dataset = update_hourly_temperature_step(
        choose_data, START_TIME, END_TIME, resampling_method
    )
    # hourly_dataset.to_excel(test_data_path +"loss_of_time\\hourly_dataset.xlsx",index=False)
    # 小时最高体温
    """hourly_dataset = update_hourly_temperature_step(
        choose_data, start_time, end_time, "max"
    )"""
    hourly_dataset = hourly_dataset[hourly_dataset["iTemperature"].notna()]
    # print(hourly_dataset.shape)

    # -------------------- 计算体温变化率 --------------------
    cal_tempRate = calculate_temperatureRate(hourly_dataset, START_TIME, END_TIME)
    cal_tempRate["temperatureRate"] = cal_tempRate["temperatureRate"].fillna(0)
    # cal_tempRate.to_excel(test_data_path + "loss_of_time\\cal_tempRate.xlsx", index=False)

    # -------------------- 标签设置 --------------------
    setLabels_dataset = cal_tempRate.copy()
    setLabels_dataset["sSowsNo"] = setLabels_dataset["sSowsNo"].astype(str)
    setLabels_dataset["isEstrus"] = 0
    # estrus_time 的时间范围是 [发情时刻-48, 发情时刻+SLIDING_WINDOW_SIZE]
    all_estrus_time_and_earCode = zip(estrus_time, estrus_ear_tag_codes)
    for row in all_estrus_time_and_earCode:
        # 发情时间段
        """estrus_end_time = pd.to_datetime(row[0].split(sep="~")[-1]) - pd.Timedelta(
            hours=1
        )
        estrus_start_time = estrus_end_time - pd.Timedelta(
            hours=SLIDING_WINDOW_SIZE - 1
        )
        # 根据耳号和发情时间段设置标签为1
        for code in row[1]:
            code_str = str(code)
            condition = (
                (setLabels_dataset["sSowsNo"] == code_str)
                & (setLabels_dataset["tLastUploadTime"] >= estrus_start_time)
                & (setLabels_dataset["tLastUploadTime"] <= estrus_end_time)
            )
            setLabels_dataset.loc[condition, "isEstrus"] = 1"""

        estrus_end_time = pd.to_datetime(row[0].split(sep="~")[-1])
        for code in row[1]:
            code_str = str(code)
            condition = (setLabels_dataset["sSowsNo"] == code_str) & (
                setLabels_dataset["tLastUploadTime"] == estrus_end_time
            )
            setLabels_dataset.loc[condition, "isEstrus"] = 1
    setLabels_dataset.dropna(subset=["iTemperature"], inplace=True)

    return setLabels_dataset


# 获取发情编号(根据给定的 TIME_CHOICE)
def get_estrus_earCode(time_choice):
    estrus_earCode = []
    for info in time_choice:
        str_ear_tag_codes = info.split(sep="_")[-1]
        for code in str_ear_tag_codes.split(sep=","):
            estrus_earCode.append(code)
    return estrus_earCode


# 获取非发请编号
def get_notEstrus_earCode(data: pd.DataFrame, estrus_earCode):
    data["sSowsNo"] = data["sSowsNo"].astype(str)
    all_sows_earCode = data["sSowsNo"].drop_duplicates(keep="first").to_numpy()
    notEstrus_earCode = []
    for code in all_sows_earCode:
        if code in estrus_earCode:
            continue
        else:
            notEstrus_earCode.append(code)
    return notEstrus_earCode


# 拆分多次发情的数据并更新发情编号
def split_estrusData(data: pd.DataFrame, estrusTime, threshold: int):
    data["sSowsNo"] = data["sSowsNo"].astype("str")
    # 确保确保时间列是 datetime 类型并排序
    data["tLastUploadTime"] = pd.to_datetime(data["tLastUploadTime"])
    data = data.sort_values(by=["sSowsNo", "tLastUploadTime"])
    record_dataset = data.copy()

    splited_dataset = pd.DataFrame()
    final_dataset = pd.DataFrame()

    # 获取多次发情的耳标号
    several_estrus_ear_tag_codes = set()
    estrus_ear_tag_codes = []
    for info in estrusTime:
        str_ear_tag_codees = info.split(sep="_")[-1]
        for each_code in str_ear_tag_codees.split(sep=","):
            each_code = str(each_code)
            if each_code in estrus_ear_tag_codes:
                several_estrus_ear_tag_codes.add(each_code)
            else:
                estrus_ear_tag_codes.append(each_code)
    for del_earCode in DEL_CODE:
        del_earCode = str(del_earCode)
        several_estrus_ear_tag_codes.discard(del_earCode)
    print(sorted(several_estrus_ear_tag_codes))
    print(len(several_estrus_ear_tag_codes))

    # 分离多次发情的母猪的数据
    for earCode in several_estrus_ear_tag_codes:
        earCode = str(earCode)
        subset = data[data["sSowsNo"] == earCode].copy()

        delta_limit = pd.Timedelta(days=threshold)
        is_new_period = subset["tLastUploadTime"].diff() > delta_limit
        subset["preiod_group"] = is_new_period.cumsum()

        for group_id, group_df in subset.groupby("preiod_group"):
            group_df = group_df.copy()
            group_df["sSowsNo"] = group_df["sSowsNo"] + "_" + str(group_id + 1)
            splited_dataset = pd.concat(
                [splited_dataset, group_df], ignore_index=True
            ).sort_values(by=["sSowsNo", "tLastUploadTime"])

        # 删掉多次发情的耳标对应的数据
        record_dataset = record_dataset[record_dataset["sSowsNo"] != earCode]

    splited_dataset = splited_dataset.drop(columns="preiod_group")
    final_dataset = pd.concat(
        [record_dataset, splited_dataset], ignore_index=True
    ).sort_values(by=["sSowsNo", "tLastUploadTime"])

    return final_dataset


# 更新发情编号
def update_estrus_earCode(data: pd.DataFrame):
    data["sSowsNo"] = data["sSowsNo"].astype(str)
    record_dataset = data.copy()
    estrus_df = record_dataset[record_dataset["isEstrus"] == 1]
    estrus_earCode = estrus_df["sSowsNo"].drop_duplicates(keep="first").tolist()
    return estrus_earCode


# 分层分组划分数据集，确保同一母猪的数据不会同时出现在训练集、验证集和测试集中
def stratified_group_split(df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    estrus_sows = df[df["isEstrus"] == 1]["sSowsNo"].unique()
    all_rows = df["sSowsNo"].unique()
    not_estrus_sows = np.array([sow for sow in all_rows if sow not in estrus_sows])

    # 对发情组进行划分
    e_train, e_temp = train_test_split(
        estrus_sows, test_size=1 - train_ratio, random_state=123
    )
    # 计算验证集和测试集的相对比例
    val_size_relative = val_ratio / (val_ratio + test_ratio)
    e_val, e_test = train_test_split(
        e_temp, test_size=1 - val_size_relative, random_state=123
    )

    # 对非发情组进行划分
    n_train, n_temp = train_test_split(
        not_estrus_sows, test_size=1 - train_ratio, random_state=123
    )
    n_val, n_test = train_test_split(
        n_temp, test_size=1 - val_size_relative, random_state=123
    )

    # 合并列表
    final_train_ids = np.concatenate([e_train, n_train])
    final_val_ids = np.concatenate([e_val, n_val])
    final_test_ids = np.concatenate([e_test, n_test])

    # 根据划分的ID创建训练集、验证集和测试集
    train_df = df[df["sSowsNo"].isin(final_train_ids)].copy()
    val_df = df[df["sSowsNo"].isin(final_val_ids)].copy()
    test_df = df[df["sSowsNo"].isin(final_test_ids)].copy()

    print(f"------ 划分结果 ------")
    print(
        f"训练集：总数 {len(final_train_ids)},其中发情猪 {len(e_train)},非发情猪 {len(n_train)}"
    )
    print(
        f"验证集：总数 {len(final_val_ids)},其中发情猪 {len(e_val)},非发情猪 {len(n_val)}"
    )
    print(
        f"测试集：总数 {len(final_test_ids)},其中发情猪 {len(e_test)},非发情猪 {len(n_test)}"
    )

    return train_df, val_df, test_df


# 填补
def function_filled(data: pd.DataFrame):
    if data.empty:
        return data

    record_dataset = data.copy()
    # 避免出现重复时间点
    record_dataset = record_dataset.drop_duplicates(subset=["tLastUploadTime"])
    first_time = record_dataset["tLastUploadTime"].min()
    last_time = record_dataset["tLastUploadTime"].max()
    full_time_range = pd.date_range(start=first_time, end=last_time, freq="1h")
    # 以完整时间序列为索引重新索引数据
    record_dataset = (
        record_dataset.set_index("tLastUploadTime")
        .reindex(full_time_range)
        .reset_index()
        .rename(columns={"index": "tLastUploadTime"})
    )

    # 填充其他列
    fill_cols = [
        "sEarTagCode",
        "sSowsNo",
        "sBrand",
        "dBreedDate",
        "dWeanDate",
        "iTemperature",
        "isEstrus",
    ]
    for col in fill_cols:
        if col in record_dataset.columns:
            record_dataset[col] = record_dataset[col].ffill()

    # 填充数值列
    if "iStep" in record_dataset.columns:
        mean_iStep = record_dataset["iStep"].mean()
        record_dataset["iStep"] = record_dataset["iStep"].fillna(mean_iStep)

    if "temperatureRate" in record_dataset.columns:
        record_dataset["temperatureRate"] = record_dataset["temperatureRate"].fillna(0)

    return record_dataset


def fill_data(data: pd.DataFrame, balanced_data=True, stride=12):
    final_dataset = pd.DataFrame()

    estrus_sows = data[data["isEstrus"] == 1]["sSowsNo"].unique()
    all_rows = data["sSowsNo"].unique()
    not_estrus_sows = np.array([sow for sow in all_rows if sow not in estrus_sows])

    estrus_dataset = pd.DataFrame()
    notEstrus_dataset = pd.DataFrame()

    for earCode in estrus_sows:
        sub_df = data[data["sSowsNo"] == earCode].sort_values("tLastUploadTime")
        if len(sub_df) >= 44:
            filled_df = function_filled(sub_df)
            estrus_dataset = pd.concat(
                [estrus_dataset, filled_df], axis=0, ignore_index=True
            )

    skipped_constant_windows = []
    # 发情母猪个数与非发情母猪个数的比例
    ratio = max(1, int(len(estrus_sows) / len(not_estrus_sows)) + 1)
    for earCode in not_estrus_sows:
        sub_df = data[data["sSowsNo"] == earCode].sort_values("tLastUploadTime")
        total_len = len(sub_df)

        if total_len < WINDOW_SIZE:
            continue

        # 平衡数据
        if balanced_data:
            sample_count = 0
            for start_idx in range(0, total_len - WINDOW_SIZE + 1, stride):
                if sample_count >= ratio:
                    continue

                window_df = sub_df.iloc[start_idx : start_idx + WINDOW_SIZE].copy()
                if window_df["iTemperature"].nunique() == 1:
                    skipped_constant_windows.append(f"{earCode}_window_at_{start_idx}")
                    continue

                sprev_time = window_df["tLastUploadTime"].min()
                last_time = window_df["tLastUploadTime"].max()
                actual_span_hours = (last_time - sprev_time).total_seconds() / 3600
                missing_hours = actual_span_hours - (WINDOW_SIZE - 1)
                if missing_hours < 5:
                    filled_df = function_filled(window_df)
                    filled_df["sSowsNo"] = f"{earCode}_neg_{sample_count}"
                    notEstrus_dataset = pd.concat(
                        [notEstrus_dataset, filled_df], axis=0, ignore_index=True
                    )
                    sample_count += 1
        elif balanced_data == False:
            # 每头非发情母猪取多个数据 -- 每间隔stride时间步取一段数据
            attempts = 0
            for start_idx in range(0, total_len - WINDOW_SIZE + 1, stride):
                window_df = sub_df.iloc[start_idx : start_idx + WINDOW_SIZE].copy()
                # 判断是否已经全部是同样的耳温值
                if window_df["iTemperature"].nunique() == 1:
                    skipped_constant_windows.append(f"{earCode}_neg_{attempts}")
                else:
                    span = (
                        window_df["tLastUploadTime"].max()
                        - window_df["tLastUploadTime"].min()
                    ).total_seconds() / 3600
                    if (span - (WINDOW_SIZE - 1)) < 5:
                        filled_df = function_filled(window_df)
                        filled_df["sSowsNo"] = f"{earCode}_neg_{attempts}"
                        notEstrus_dataset = pd.concat(
                            [notEstrus_dataset, filled_df], axis=0, ignore_index=True
                        )
                attempts += 1

    final_dataset = pd.concat(
        [estrus_dataset, notEstrus_dataset], axis=0, ignore_index=True
    ).sort_values(by=["sSowsNo", "tLastUploadTime"])

    if skipped_constant_windows:
        print("-" * 30)
        print(
            f"以下非发情窗口由于耳温值无变化被跳过 (共 {len(skipped_constant_windows)} 个):"
        )
        print(skipped_constant_windows)

    discarded_sows = []
    valid_groups = []
    for sow_id, group in final_dataset.groupby("sSowsNo"):
        if group["iTemperature"].nunique() <= 1:
            discarded_sows.append(sow_id)
        else:
            valid_groups.append(group)

    if valid_groups:
        final_dataset = pd.concat(valid_groups, axis=0, ignore_index=True)
    else:
        final_dataset = pd.DataFrame(columns=final_dataset.columns)

    if discarded_sows:
        print("-" * 30)
        print(f"以下样本由于48小时耳温完全一致被丢弃 (共 {len(discarded_sows)} 个):")
        print(discarded_sows)

    final_all_sows = final_dataset["sSowsNo"].unique()
    final_estrus_sows = final_dataset[final_dataset["isEstrus"] == 1][
        "sSowsNo"
    ].unique()
    finbal_notEstrus_sows = np.array(
        [sow for sow in final_all_sows if sow not in final_estrus_sows]
    )
    print("-" * 30)
    print(
        f"总数 {len(final_all_sows)},其中发情样本 {len(final_estrus_sows)},非发情样本 {len(finbal_notEstrus_sows)}"
    )
    return final_dataset


# 仅考虑温度特征的单变量LSTM数据准备
def prepare_univariate_lstm_data(data: pd.DataFrame, scaler=None):
    df_copy = data.copy()

    if len(df_copy.columns) >= WINDOW_SIZE:
        X_raw = df_copy.iloc[:, 1:-1].values
        y = df_copy.iloc[:, -1].values

        temp_values = X_raw.reshape(-1, 1)
        if scaler is None:
            scaler = StandardScaler()
            temp_scaled = scaler.fit_transform(temp_values)
        else:
            temp_scaled = scaler.transform(temp_values)
        X = temp_scaled.reshape(-1, WINDOW_SIZE, 1)
        return X, y, scaler

    feature_col = "iTemperature"
    temp_values = df_copy[feature_col].values.reshape(-1, 1)
    if scaler is None:
        scaler = StandardScaler()
        df_copy[feature_col] = scaler.fit_transform(temp_values)
    else:
        df_copy[feature_col] = scaler.transform(temp_values)

    X_list, y_list = [], []
    for earCode, group in df_copy.groupby("sSowsNo"):
        group = group.sort_values("tLastUploadTime")
        vals = group[feature_col].values
        labels = group["isEstrus"].values

        if len(vals) < WINDOW_SIZE:
            continue

        if (labels == 1).any():
            estrus_indices = np.where(labels == 1)[0]
            for end_idx in estrus_indices:
                start_idx = end_idx - WINDOW_SIZE + 1
                if start_idx >= 0:
                    X_list.append(vals[start_idx : end_idx + 1])
                    y_list.append(1)
        else:
            X_list.append(vals[:WINDOW_SIZE])
            y_list.append(0)

    X = np.array(X_list)
    X = np.expand_dims(X, axis=-1)
    y = np.array(y_list)

    return X, y, scaler


def convert_features(data: pd.DataFrame):
    df_copy = data.copy()
    cols = ["sSowsNo"] + [f"feature{i}" for i in range(1, 49)] + ["isEstrus"]
    rows_list = []
    drop_list = []

    for earCode, group in df_copy.groupby("sSowsNo"):
        # 确保每个分组都包含完整的48个时间步长数据
        if len(group) >= WINDOW_SIZE:
            # 提取48个iTemperature值
            vals = group["iTemperature"].values[len(group) - WINDOW_SIZE :]

            # 如果isEstrus列中存在1，则该样本标签为1，否则为0
            label = group["isEstrus"].max()

            # 构建新行: [编号, feature_1, ..., feature_48, 标签]
            new_row = [earCode] + vals.tolist() + [label]
            rows_list.append(new_row)
        else:
            drop_list.append(earCode)

    if drop_list:
        print("-" * 30)
        print(f"以下样本由于时间步长不足被丢弃 (共 {len(drop_list)} 个):")
        print(drop_list)

    final_dataset = pd.DataFrame(rows_list, columns=cols)
    return final_dataset


# 增加 temperatureRate 特征
def prepare_lstm_data(data: pd.DataFrame, scaler=None):
    df_copy = data.copy()

    if len(df_copy.columns) >= WINDOW_SIZE:
        X_raw = df_copy.iloc[:, 1:-1].values
        y = df_copy.iloc[:, -1].values

        num_features = X_raw.shape[1] // WINDOW_SIZE
        temp_values = (
            X_raw.reshape(-1, num_features, WINDOW_SIZE)
            .transpose(0, 2, 1)
            .reshape(-1, num_features)
        )
        if scaler is None:
            scaler = StandardScaler()
            temp_scaled = scaler.fit_transform(temp_values)
        else:
            temp_scaled = scaler.transform(temp_values)
        X = temp_scaled.reshape(-1, WINDOW_SIZE, num_features)
        return X, y, scaler

    feature_col = ["iTemperature", "temperatureRate"]
    temp_values = df_copy[feature_col].values
    if scaler is None:
        scaler = StandardScaler()
        df_copy[feature_col] = scaler.fit_transform(temp_values)
    else:
        df_copy[feature_col] = scaler.transform(temp_values)

    X_list, y_list = [], []
    for earCode, group in df_copy.groupby("sSowsNo"):
        group = group.sort_values("tLastUploadTime")
        vals = group[feature_col].values
        labels = group["isEstrus"].values

        if len(vals) < WINDOW_SIZE:
            continue

        if (labels == 1).any():
            estrus_indices = np.where(labels == 1)[0]
            for end_idx in estrus_indices:
                start_idx = end_idx - WINDOW_SIZE + 1
                if start_idx >= 0:
                    X_list.append(vals[start_idx : end_idx + 1])
                    y_list.append(1)
        else:
            X_list.append(vals[:WINDOW_SIZE])
            y_list.append(0)

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y, scaler


# 绘制训练历史的函数，单独保存每个指标的图
def plot_training_history(history, save_path=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.style.use("seaborn-v0_8-whitegrid")

    plot_configs = [
        (
            "loss",
            ["train_loss", "val_loss"],
            ["b", "r"],
            "Training and Validation Loss",
        ),
        ("accuracy", ["val_accuracy"], ["g"], "Validation Accuracy"),
        ("precision", ["val_precision"], ["c"], "Validation Precision"),
        ("recall", ["val_recall"], ["y"], "Validation Recall"),
        ("f1_score", ["val_f1"], ["m"], "Validation F1 Score"),
        ("auc", ["val_auc"], ["k"], "Validation AUC"),
    ]

    for filename, keys, colors, title in plot_configs:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()

        for key, color in zip(keys, colors):
            # 兼容处理：如果 history 中没有该 key 则跳过
            if key in history:
                label = "Train" if "train" in key else "Val"
                plt.plot(
                    epochs,
                    history[key],
                    color=color,
                    label=label if len(keys) > 1 else None,
                )
        # 强制横坐标显示为整数
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.title(title, fontsize=14)
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        if len(keys) > 1:
            plt.legend()
        plt.grid(True)

        # 保存图片
        file_save_path = os.path.join(save_path, f"val_{filename}.png")
        plt.savefig(file_save_path, dpi=330, bbox_inches="tight")
        plt.close()  # 必须关闭，否则多图运行时会占用大量内存
        print(f"已保存: {file_save_path}")

    print("--- 所有指标图表已单独保存完成 ---")


# 绘制混淆矩阵热力图和测试集预测柱状图
def plot_matrix(y_true, y_pred, save_dir=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))

    # 使用百分比和原始数值同时展示
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Estrus (0)", "Estrus (1)"],
        yticklabels=["Not Estrus (0)", "Estrus (1)"],
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    save_count = 0
    if save_dir:
        conf_matrix_path = os.path.join(save_dir, "confusion_matrix.png")
        plt.savefig(conf_matrix_path, dpi=330, bbox_inches="tight")
        save_count += 1
    plt.close()

    """
        绘制测试结果的其他指标 -- 柱状图, 保留两位小数
    """
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred) * 100
    recall = recall_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred) * 100
    auc = roc_auc_score(y_true, y_pred) * 100
    mcc = matthews_corrcoef(y_true, y_pred) * 100

    metrics_name = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
    metrics_values = [accuracy, precision, recall, f1, auc]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        metrics_name,
        metrics_values,
        color=["#3498db", "#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6"],
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    plt.ylim(0, 110)
    plt.ylabel("Values (%)")
    plt.title("")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    if save_dir:
        metrics_path = os.path.join(save_dir, "test_metrics.png")
        plt.savefig(metrics_path, dpi=330, bbox_inches="tight")
        save_count += 1
    plt.close()

    if save_count >= 2:
        print(f"保存至: {os.path.join(save_dir)}")
    # 额外打印详细数值供分析
    tn, fp, fn, tp = cm.ravel()

    print("-" * 30)
    print(f"测试集准确率 (Accuracy): {accuracy:.4f}")
    print(f"测试集精确率 (Precision): {precision:.4f}")
    print(f"测试集召回率 (Recall): {recall:.4f}")
    print(f"测试集 F1 分数: {f1:.4f}")
    print(f"测试集 AUC 指标: {auc:.4f}")
    print(f"测试集 MCC : {mcc}:.4f")

    print(f"\n--- 混淆矩阵详细分析 ---")
    print(f"真负类 (TN): {tn} | 伪正类 (FP): {fp} (误报)")
    print(f"伪负类 (FN): {fn} (漏报) | 真正类 (TP): {tp}")


def ADASYN(threshold, gamma, df_min: pd.DataFrame, df_maj: pd.DataFrame, k=7):
    """
    threshold: 过采样的阈值
    gamma: 过采样的强度
    df_min: 少数类样本
    df_maj: 多数类样本
    k: k近邻的数量
    """
    d = len(df_min) / len(df_maj)
    if d >= threshold:
        return pd.concat([df_min, df_maj], axis=0).reset_index(drop=True)

    # 特征矩阵
    X_min = df_min.iloc[:, 1:49].values.astype(float)
    X_maj = df_maj.iloc[:, 1:49].values.astype(float)
    X_all = np.vstack((X_min, X_maj))

    # 标签数组用于统计近邻类别
    labels_all = np.array([1] * len(df_min) + [0] * len(df_maj))

    # 合成样本数
    G = int(gamma * (len(df_maj) - len(df_min)))
    if G <= 0:
        return pd.concat([df_min, df_maj], axis=0)

    # 寻找K个近邻并计算权重
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_all)
    _, indices = nn.kneighbors(X_min)

    r = []
    for i in range(len(df_min)):
        # 计算近邻中多数类的比例
        count_maj = np.sum(labels_all[indices[i][1:]] == 0)
        r.append(count_maj / k)

    r = np.array(r)
    r_hat = r / r.sum() if r.sum() > 0 else np.ones(len(df_min)) / len(df_min)
    g = np.round(r_hat * G).astype(int)

    # 生成合成样本
    X_syn = []
    nn_min = NearestNeighbors(n_neighbors=min(k + 1, len(df_min))).fit(X_min)
    _, min_indices = nn_min.kneighbors(X_min)

    for i in range(len(df_min)):
        for _ in range(g[i]):
            zi_idx = np.random.choice(min_indices[i][1:])
            x_zi = X_min[zi_idx]

            # 线性内插
            lambd = np.random.uniform(0, 1)
            s_features = X_min[i] + lambd * (x_zi - X_min[i])
            X_syn.append(s_features)

    # 构造新的DataFrame
    if len(X_syn) > 0:
        # 先用纯 float 特征构造 DataFrame
        feat_cols = df_min.columns[1:49]
        df_syn = pd.DataFrame(X_syn, columns=feat_cols)

        # 单独插入 ID 列和标签列，不影响特征列的 dtype
        df_syn.insert(0, df_min.columns[0], [f"adasyn_{i}" for i in range(len(X_syn))])
        df_syn[df_min.columns[-1]] = 1

        # 合并
        df_augmented = pd.concat([df_min, df_maj, df_syn], axis=0).reset_index(
            drop=True
        )

        # 确保类型转换
        df_augmented[df_min.columns[0]] = df_augmented[df_min.columns[0]].astype(str)
        df_augmented[df_min.columns[-1]] = df_augmented[df_min.columns[-1]].astype(int)
        return df_augmented

    return pd.concat([df_min, df_maj], axis=0)


def SMOTE(data: pd.DataFrame, amount_oversampling=400, k=5):
    """ "
    data: 训练集数据
    amount_oversampling: 过采样比例
    k: K近邻的数量
    """

    all_smote_dfs = []

    col_id = data.columns[0]
    col_label = data.columns[-1]
    col_features = data.columns[1:-1]

    # 便利类别进行独立扩充
    for label in data["isEstrus"].unique():
        df_label = data[data["isEstrus"] == label].copy()
        X_class = df_label.iloc[:, 1:49].values.astype(float)
        n_samples = len(X_class)

        if n_samples <= 1:
            continue

        N = int(amount_oversampling / 100)
        if N < 1:
            # 比例小于100%时,随机选择部分样本进行过采样
            sample_size = int(n_samples * amount_oversampling / 100)
            indices = np.random.choice(n_samples, size=sample_size, replace=False)
            X_subset = X_class[indices]
            n_gen_loop = len(X_subset)
            N_to_gen = 1
        else:
            # 比例大于等于100%时,对全部样本进行过采样
            X_subset = X_class
            n_gen_loop = n_samples
            N_to_gen = N

        nn = NearestNeighbors(n_neighbors=min(k + 1, n_samples)).fit(X_class)
        _, indices = nn.kneighbors(X_subset)

        X_smote_class = []
        for i in range(n_gen_loop):
            neighbor_indices = indices[i][1:]
            for _ in range(N_to_gen):
                nn_idx = np.random.choice(neighbor_indices)
                diff = X_class[nn_idx] - X_subset[i]
                gap = np.random.random()
                s_features = X_subset[i] + gap * diff
                X_smote_class.append(s_features)

        if len(X_smote_class) > 0:
            df_smote_class = pd.DataFrame(X_smote_class, columns=col_features)
            df_smote_class.insert(
                0, col_id, [f"smote_{label}_{i}" for i in range(len(df_smote_class))]
            )
            df_smote_class[col_label] = label
            all_smote_dfs.append(df_smote_class)

    df_augmented = pd.concat([data] + all_smote_dfs, axis=0).reset_index(drop=True)
    df_augmented["sSowsNo"] = df_augmented["sSowsNo"].astype(str)
    df_augmented["isEstrus"] = df_augmented["isEstrus"].astype(int)

    print(f"原始样本总数: {len(data)}")
    print(f"过采样后总数: {len(df_augmented)}")
    return df_augmented


def TomekLinked(data: pd.DataFrame, k):
    """
    data: 包含特征和标签的 DataFrame
    k: 近邻数量
    """
    df_min = data[data["isEstrus"] == 1].copy()
    df_maj = data[data["isEstrus"] == 0].copy()

    X_min = df_min.iloc[:, 1:49].values
    X_maj = df_maj.iloc[:, 1:49].values

    nn_maj = NearestNeighbors(n_neighbors=k).fit(X_maj)
    _, min_to_maj_idx = nn_maj.kneighbors(X_min)
    nn_min = NearestNeighbors(n_neighbors=k).fit(X_min)
    _, maj_to_min_idx = nn_min.kneighbors(X_maj)

    removed_maj_indices = []

    for i in range(len(df_min)):
        # j 是少数类样本 i 在多数类中最相似样本的索引
        j = min_to_maj_idx[i, 0]

        # 如果多数类样本 j 在少数类中最相似的样本正好也是 i
        # 则 (i, j) 互为对方在异类中的“最相似者”，构成 Tomek Link
        if maj_to_min_idx[j, 0] == i:
            removed_maj_indices.append(df_maj.index[j])

    final_df_maj = df_maj.drop(index=list(set(removed_maj_indices)))

    cleaned_data = pd.concat([df_min, final_df_maj], axis=0).reset_index(drop=True)
    print(f"识别并删除的多数类样本数: {len(set(removed_maj_indices))}")

    return cleaned_data
