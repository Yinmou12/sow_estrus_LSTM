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
from sklearn.model_selection import train_test_split, StratifiedKFold
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
    """
    发情母猪：
        根据 time_choice 保留对应时间段的 48h 数据

    非发情母猪：
        调整数据选择的时间段，也以 08:00:00 或 15:00:00 作为样本时间段的结尾
    """
    record_dataset = source_dataset.copy()
    # record_dataset = record_dataset[record_dataset["iTemperature"] > 25]
    record_dataset["sSowsNo"] = record_dataset["sSowsNo"].astype(str)
    # -------------------- 删除一些异常数据 --------------------
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
    # 选择母猪48小时数据
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


# 拆分多次发情的数据，将发情次数写入 sSowsNo_split
def split_estrusData(data: pd.DataFrame, estrusTime, threshold: int):
    """
    保留原始 sSowsNo，按时间顺序将多次发情样本编号为 编号_1、编号_2。

    沿用 estrusTime 中的重复发情记录及 DEL_CODE 排除规则；对数据中至少
    有一次 isEstrus == 1 的母猪，相邻记录间隔超过 threshold 天时分段。
    有多个时间段时，更新每段全部记录的 sSowsNo_split；仅有一个时间段
    或属于非发情母猪时，保留已有分段编号（包括 M / A 编号）。
    缺失小时和缺失的发情终点标签不在此处补齐，不修改输入 DataFrame。
    """
    record_dataset = data.copy()
    record_dataset["sSowsNo"] = record_dataset["sSowsNo"].astype(str)
    record_dataset["tLastUploadTime"] = pd.to_datetime(
        record_dataset["tLastUploadTime"]
    )
    record_dataset = record_dataset.sort_values(
        by=["sSowsNo", "tLastUploadTime"]
    ).reset_index(drop=True)
    if "sSowsNo_split" not in record_dataset.columns:
        record_dataset["sSowsNo_split"] = record_dataset["sSowsNo"]

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

    # 仅处理发情母猪，避免覆盖上一步生成的非发情 M / A 分段。
    estrus_sows = record_dataset.loc[record_dataset["isEstrus"] == 1, "sSowsNo"]
    several_estrus_ear_tag_codes.intersection_update(estrus_sows)
    print(sorted(several_estrus_ear_tag_codes))
    print(len(several_estrus_ear_tag_codes))

    delta_limit = pd.Timedelta(days=threshold)
    for earCode in sorted(several_estrus_ear_tag_codes):
        subset = record_dataset.loc[record_dataset["sSowsNo"] == earCode]
        is_new_period = subset["tLastUploadTime"].diff() > delta_limit
        period_group = is_new_period.cumsum()
        if period_group.nunique() <= 1:
            continue

        record_dataset.loc[subset.index, "sSowsNo_split"] = (
            earCode + "_" + (period_group + 1).astype(str)
        )

    return record_dataset


# 更新发情编号
def update_estrus_earCode(data: pd.DataFrame):
    data["sSowsNo"] = data["sSowsNo"].astype(str)
    record_dataset = data.copy()
    estrus_df = record_dataset[record_dataset["isEstrus"] == 1]
    estrus_earCode = estrus_df["sSowsNo"].drop_duplicates(keep="first").tolist()
    return estrus_earCode


# 分层分组划分数据集，确保同一母猪的数据不会同时出现在训练集、验证集和测试集中
def stratified_group_split(
    df,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    random_count=123,
):
    """
    按 sSowsNo 分层划分母猪，按 sSowsNo_split 统计各集合的样本数量。

    同一母猪的全部发情次数和 M / A 窗口归入同一个集合。某头母猪只要
    出现过 isEstrus == 1，其所属样本均按发情样本统计，原始标签不变。
    比例针对母猪数量，样本数不保证严格符合比例；随机种子沿用 123。
    返回 train_df、val_df、test_df，不修改输入数据。
    """
    required_columns = {"sSowsNo", "sSowsNo_split", "isEstrus"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"缺少必要列: {sorted(missing_columns)}")
    for column in ("sSowsNo", "sSowsNo_split"):
        if df[column].isna().any():
            raise ValueError(f"{column} 不能包含缺失编号")
    if (df.groupby("sSowsNo_split")["sSowsNo"].nunique() > 1).any():
        raise ValueError("同一个 sSowsNo_split 不能对应多头母猪")

    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=float)
    if (
        not np.isfinite(ratios).all()
        or (ratios <= 0).any()
        or not np.isclose(ratios.sum(), 1.0)
    ):
        raise ValueError("训练集、验证集、测试集的比例必须均大于 0，且总和为 1")
    train_ratio, val_ratio, test_ratio = ratios

    estrus_sows = df[df["isEstrus"] == 1]["sSowsNo"].unique()
    all_rows = df["sSowsNo"].unique()
    not_estrus_sows = np.array([sow for sow in all_rows if sow not in estrus_sows])

    # 对发情组进行划分
    e_train, e_temp = train_test_split(
        estrus_sows, test_size=1 - train_ratio, random_state=random_count
    )
    # 计算验证集和测试集的相对比例
    val_size_relative = val_ratio / (val_ratio + test_ratio)
    e_val, e_test = train_test_split(
        e_temp, test_size=1 - val_size_relative, random_state=random_count
    )

    # 对非发情组进行划分
    n_train, n_temp = train_test_split(
        not_estrus_sows, test_size=1 - train_ratio, random_state=random_count
    )
    n_val, n_test = train_test_split(
        n_temp, test_size=1 - val_size_relative, random_state=random_count
    )

    # 合并列表
    final_train_ids = np.concatenate([e_train, n_train])
    final_val_ids = np.concatenate([e_val, n_val])
    final_test_ids = np.concatenate([e_test, n_test])

    # 根据划分的ID创建训练集、验证集和测试集
    train_df = df[df["sSowsNo"].isin(final_train_ids)].copy()
    val_df = df[df["sSowsNo"].isin(final_val_ids)].copy()
    test_df = df[df["sSowsNo"].isin(final_test_ids)].copy()

    datasets = (("训练集", train_df), ("验证集", val_df), ("测试集", test_df))
    for column in ("sSowsNo", "sSowsNo_split"):
        id_sets = [set(part[column]) for _, part in datasets]
        for i in range(len(id_sets)):
            for j in range(i + 1, len(id_sets)):
                if id_sets[i].intersection(id_sets[j]):
                    raise RuntimeError(f"划分结果中 {column} 存在跨集合交叉")
        if set.union(*id_sets) != set(df[column]):
            raise RuntimeError(f"划分前后的 {column} 编号不完整")
    if sum(len(part) for _, part in datasets) != len(df):
        raise RuntimeError("划分前后的数据总行数不一致")

    print("------ 划分结果（按 sSowsNo_split 统计样本）------")
    for name, part in datasets:
        sow_count = part["sSowsNo"].nunique()
        sample_count = part["sSowsNo_split"].nunique()
        estrus_mask = part["sSowsNo"].isin(estrus_sows)
        positive_count = part.loc[estrus_mask, "sSowsNo_split"].nunique()
        negative_count = part.loc[~estrus_mask, "sSowsNo_split"].nunique()
        print(
            f"{name}：母猪 {sow_count} 头，样本总数 {sample_count}，"
            f"正样本（发情）{positive_count}，负样本（非发情）{negative_count}"
        )

    return train_df, val_df, test_df


def split_dataset_train_val_test(df, test_ratio=0.2, random_state=123):
    """
    新增：将数据集先划分为训练用（训练集+验证集）和独立测试用。
    保持发情与非发情猪比例。
    """
    estrus_sows = df[df["isEstrus"] == 1]["sSowsNo"].unique()
    all_sows = df["sSowsNo"].unique()

    sow_labels = np.array([1 if sow in estrus_sows else 0 for sow in all_sows])

    internal_sows, test_sows, internal_labels, test_labels = train_test_split(
        all_sows,
        sow_labels,
        test_size=test_ratio,
        stratify=sow_labels,
        random_state=random_state,
    )

    train_val_df = df[df["sSowsNo"].isin(internal_sows)].copy()
    test_df = df[df["sSowsNo"].isin(test_sows)].copy()

    print(
        f"独立测试集划分完成: 总母猪数 {len(test_sows)}, 其中发情 {int(test_labels.sum())}, 非发情 {len(test_sows) - int(test_labels.sum())}"
    )

    return train_val_df, test_df


def stratified_group_kfold_only(train_val_df, n_splits=5, random_state=123):
    """
    新增：仅对传入的训练验证集进行分层 K 折交叉验证划分，不再额外划分测试集。
    """
    estrus_sows = train_val_df[train_val_df["isEstrus"] == 1]["sSowsNo"].unique()
    all_sows = train_val_df["sSowsNo"].unique()

    sow_labels = np.array([1 if sow in estrus_sows else 0 for sow in all_sows])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []
    for train_idx, val_idx in skf.split(all_sows, sow_labels):
        train_sows_fold = all_sows[train_idx]
        val_sows_fold = all_sows[val_idx]

        train_df = train_val_df[train_val_df["sSowsNo"].isin(train_sows_fold)].copy()
        val_df = train_val_df[train_val_df["sSowsNo"].isin(val_sows_fold)].copy()

        folds.append((train_df, val_df))

        e_train_count = np.sum(sow_labels[train_idx] == 1)
        n_train_count = np.sum(sow_labels[train_idx] == 0)
        e_val_count = np.sum(sow_labels[val_idx] == 1)
        n_val_count = np.sum(sow_labels[val_idx] == 0)

        print(f"--- Fold {len(folds)} ---")
        print(
            f"训练集：母猪总数 {len(train_sows_fold)}, 其中发情猪 {e_train_count}, 非发情猪 {n_train_count}"
        )
        print(
            f"验证集：母猪总数 {len(val_sows_fold)}, 其中发情猪 {e_val_count}, 非发情猪 {n_val_count}"
        )

    return folds


# 5折分层分组交叉验证
def stratified_group_kfold(df, n_splits=5, test_ratio=0.2, random_state=123):
    """
    先划分出独立测试集，再对剩余数据进行5折分层分组交叉验证。
    确保同一母猪的数据不会同时出现在训练集、验证集或独立测试集中，并保持发情与非发情猪的比例。
    """
    train_val_df, independent_test_df = split_dataset_train_val_test(
        df, test_ratio, random_state
    )
    folds = stratified_group_kfold_only(train_val_df, n_splits, random_state)

    return independent_test_df, folds


# 填补
def function_filled(data: pd.DataFrame, start_time=None, end_time=None):
    """补齐一个样本；可指定包含缺失首尾小时的完整窗口。"""
    if data.empty:
        return data

    record_dataset = data.copy()
    # 避免出现重复时间点
    record_dataset = record_dataset.drop_duplicates(subset=["tLastUploadTime"])
    first_time = (
        pd.to_datetime(start_time)
        if start_time is not None
        else record_dataset["tLastUploadTime"].min()
    )
    last_time = (
        pd.to_datetime(end_time)
        if end_time is not None
        else record_dataset["tLastUploadTime"].max()
    )
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
        "sSowsNo_split",
        "sBrand",
        "dBreedDate",
        "dWeanDate",
        "iTemperature",
        "isEstrus",
    ]
    for col in fill_cols:
        if col in record_dataset.columns:
            # 中间和末尾沿用前值；缺失的开头使用本样本首个已知值。
            record_dataset[col] = record_dataset[col].ffill().bfill()

    # 填充数值列
    if "iStep" in record_dataset.columns:
        mean_iStep = record_dataset["iStep"].mean()
        record_dataset["iStep"] = record_dataset["iStep"].fillna(mean_iStep)

    if "temperatureRate" in record_dataset.columns:
        record_dataset["temperatureRate"] = record_dataset["temperatureRate"].fillna(0)

    return record_dataset


def fill_data(data: pd.DataFrame, balanced_data=True, stride=12):
    """
    逐个 sSowsNo_split 筛选并补齐已有样本，不再抽样、滑窗或修改编号。

    每个窗口至少保留 44 个不同小时的记录，且温度不能恒定。优先使用
    isEstrus == 1 的时刻作为终点；M / A 样本分别以 08:00 / 15:00 结束。
    缺少发情终点标签时，根据现有记录确定唯一能容纳它们的 08:00 / 15:00
    窗口。筛选后补足 WINDOW_SIZE 个小时，包括缺失首尾。
    balanced_data、stride 仅为兼容旧调用保留，不参与处理。
    """
    if data.empty:
        return data.copy()
    required = {
        "sSowsNo",
        "sSowsNo_split",
        "tLastUploadTime",
        "iTemperature",
        "isEstrus",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"缺少必要列: {sorted(missing)}")
    record_dataset = data.copy()
    record_dataset["tLastUploadTime"] = pd.to_datetime(
        record_dataset["tLastUploadTime"]
    )
    if (
        record_dataset[["sSowsNo", "sSowsNo_split", "tLastUploadTime"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("样本编号、母猪编号和时间不能缺失")
    if (
        record_dataset["tLastUploadTime"]
        .ne(record_dataset["tLastUploadTime"].dt.floor("h"))
        .any()
    ):
        raise ValueError("fill_data 需要整点的小时数据")
    if record_dataset.groupby("sSowsNo_split")["sSowsNo"].nunique().gt(1).any():
        raise ValueError("一个 sSowsNo_split 只能对应一头母猪")

    estrus_sows = set(record_dataset.loc[record_dataset["isEstrus"] == 1, "sSowsNo"])
    window_span = pd.Timedelta(hours=WINDOW_SIZE - 1)
    filled_groups = []
    insufficient_samples = []
    constant_samples = []
    added_hours = 0
    trimmed_rows = 0

    for split_id, group in record_dataset.groupby("sSowsNo_split", sort=False):
        sub_df = group.sort_values("tLastUploadTime").drop_duplicates("tLastUploadTime")
        if len(sub_df) < 44:
            insufficient_samples.append(split_id)
            continue

        estrus_times = sub_df.loc[sub_df["isEstrus"] == 1, "tLastUploadTime"]
        if len(estrus_times) == 1:
            window_end = estrus_times.iloc[0]
            if window_end.hour not in (8, 15):
                raise ValueError(f"样本 {split_id} 的发情终点不是 08:00 或 15:00")
        elif len(estrus_times) > 1:
            raise ValueError(f"样本 {split_id} 包含多个发情终点，请先完成样本拆分")
        else:
            first_time = sub_df["tLastUploadTime"].min()
            last_time = sub_df["tLastUploadTime"].max()
            suffix = str(split_id).rsplit("_", 2)
            period = suffix[-2] if len(suffix) == 3 else None
            end_hours = (8,) if period == "M" else (15,) if period == "A" else (8, 15)
            candidates = []
            for hour in end_hours:
                candidate = last_time.normalize() + pd.Timedelta(hours=hour)
                if candidate < last_time:
                    candidate += pd.Timedelta(days=1)
                if candidate - window_span <= first_time:
                    candidates.append(candidate)
            if len(candidates) != 1:
                raise ValueError(
                    f"无法唯一确定样本 {split_id} 的48小时窗口，请检查分段时间"
                )
            window_end = candidates[0]

        window_start = window_end - window_span
        window = sub_df.loc[
            sub_df["tLastUploadTime"].between(window_start, window_end)
        ].copy()
        trimmed_rows += len(sub_df) - len(window)
        if len(window) < 44:
            insufficient_samples.append(split_id)
            continue
        if window["iTemperature"].nunique() <= 1:
            constant_samples.append(split_id)
            continue

        filled_df = function_filled(window, window_start, window_end)
        filled_df["sSowsNo_split"] = split_id
        if len(filled_df) != WINDOW_SIZE:
            raise RuntimeError(f"样本 {split_id} 补齐后长度异常")
        added_hours += len(filled_df) - len(window)
        filled_groups.append(filled_df)

    final_dataset = (
        pd.concat(filled_groups, ignore_index=True).loc[:, data.columns]
        if filled_groups
        else data.iloc[:0].copy()
    )
    sample_count = final_dataset["sSowsNo_split"].nunique()
    positive_count = final_dataset.loc[
        final_dataset["sSowsNo"].isin(estrus_sows), "sSowsNo_split"
    ].nunique()
    print(
        f"输入样本 {record_dataset['sSowsNo_split'].nunique()} 个；"
        f"不足44条记录剔除 {len(insufficient_samples)} 个，温度恒定剔除 {len(constant_samples)} 个"
    )
    print(
        f"补齐后：样本总数 {sample_count}，正样本（发情）{positive_count}，"
        f"负样本（非发情）{sample_count - positive_count}，每个样本 {WINDOW_SIZE} 小时"
    )
    print(f"新增小时记录 {added_hours} 条，移除窗口外记录 {trimmed_rows} 条")
    return final_dataset


# 仅考虑温度特征的单变量LSTM数据准备
def prepare_univariate_lstm_data(data: pd.DataFrame, scaler=None):
    """按独立48小时窗口准备temp_only输入；返回X、标签和训练标准化器，不写文件。"""
    from experiment_data import legacy_tensors
    return legacy_tensors(data, "temp_only", scaler)


def convert_features(data: pd.DataFrame):
    """逐sSowsNo_split展开48个温度值；旧首列存样本编号，避免合并同一母猪多个窗口。"""
    from experiment_data import build_windows, flat_windows
    frame = data.copy()
    if "sSowsNo_split" not in frame:
        frame["sSowsNo_split"] = frame["sSowsNo"].astype(str)
    return flat_windows(build_windows(frame))


# 增加 temperatureRate 特征
def prepare_lstm_data(data: pd.DataFrame, scaler=None):
    """按独立48小时窗口准备temp_rate输入；返回X、标签和训练标准化器，不写文件。"""
    from experiment_data import legacy_tensors
    return legacy_tensors(data, "temp_rate", scaler)


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
        ("specificity", ["val_specificity"], ["orange"], "Validation Specificity"),
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
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred) * 100
    auc = roc_auc_score(y_true, y_pred) * 100
    mcc = matthews_corrcoef(y_true, y_pred) * 100

    metrics_name = ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score", "AUC"]
    metrics_values = [accuracy, precision, recall, specificity, f1, auc]
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
    print(f"测试集特异度 (Specificity): {specificity:.4f}")
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
    print(f"{"-"*60} ADASYN 过采样 {"-"*60}")
    d = len(df_min) / len(df_maj)
    print(
        f"当前少数类样本数: {len(df_min)}, 多数类样本数: {len(df_maj)}, 比例: {d:.4f}"
    )
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
    print(f"需要生成的合成样本数: {G}")
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

    count_min = len(df_min)
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

        print(f"{"-"*60} ADASYN 过采样个数: {len(X_syn)} {"-"*60}")
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
            if len(neighbor_indices) == 0:
                continue  # 如果没有可用的邻居，则跳过此样本

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
