from sow_estrus_LSTM_Info import *
import sow_estrus_LSTM_Function as myFunction

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os as os


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


# 异常体温处理
def correct_abnormal_temperatures_linear(
    data: pd.DataFrame,  # 数据集
    threshold: float = 34.0,  # 异常体温阈值
):
    tempDataframe = data.copy()
    ear_tag_codes = tempDataframe.drop_duplicates(subset="sSowsNo", keep="first")[
        "sSowsNo"
    ].to_numpy()

    final_data = pd.DataFrame(columns=data.columns)
    for ear_code in ear_tag_codes:
        part_data = data[data["sSowsNo"] == ear_code]
        sow_temperatures = part_data["iTemperature"].to_numpy()
        prev_valid_index = -1
        next_valid_index = 0

        for index, temperature in enumerate(sow_temperatures):
            if temperature <= threshold:
                # 找到前一个正常体温值的索引
                while prev_valid_index < index and (
                    prev_valid_index == -1
                    or sow_temperatures[prev_valid_index] <= threshold
                ):
                    prev_valid_index += 1
                # 找到下一个正常体温值的索引
                while next_valid_index < len(sow_temperatures) and (
                    next_valid_index <= index
                    or sow_temperatures[next_valid_index] <= threshold
                ):
                    next_valid_index += 1

                if prev_valid_index >= 0 and next_valid_index < len(sow_temperatures):
                    # 线性插值
                    slope = (
                        sow_temperatures[next_valid_index]
                        - sow_temperatures[prev_valid_index]
                    ) / (next_valid_index - prev_valid_index)
                    interpolated_value = sow_temperatures[prev_valid_index] + slope * (
                        index - prev_valid_index
                    )
                    sow_temperatures[index] = interpolated_value

        part_data["iTemperature"] = sow_temperatures
        final_data = pd.concat([final_data, part_data], axis=0, ignore_index=True)

    return final_data


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
            time = pd.to_datetime("16:00:00")
        elif AorM == "M":
            time = pd.to_datetime("09:00:00")

        # 时间段
        prev_time = 0
        # last_time = 0
        if time != None:
            date_time = pd.to_datetime(f"{date.date()} {time.time()}")
            prev_time = date_time - pd.Timedelta(hours=WINDOW_SIZE)
            last_time = date_time + pd.Timedelta(hours=SLIDING_WINDOW_SIZE)

            # 提取发情母猪数据
            for code in estrus_ear_tag_codes:
                code = str(code)
                estrusSows_dataset = pd.concat(
                    [
                        estrusSows_dataset,
                        intercept_data(data, code, prev_time, last_time),
                    ],
                    axis=0,
                    ignore_index=True,
                )

    # 未发情母猪的数据
    notEstrusSows_dataset = pd.DataFrame(columns=columns_name)
    for code in data["sSowsNo"].drop_duplicates(keep="first").to_numpy():
        if code in all_estrusSows_ear_codes:
            continue
        else:
            notEstrusSows_dataset = pd.concat(
                [
                    notEstrusSows_dataset,
                    intercept_data(data, code, START_TIME, END_TIME),
                ],
                axis=0,
                ignore_index=True,
            )

    final_dataset = pd.DataFrame(columns=columns_name)
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
    print(type(data))
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
        record_dataset["sSowsNo"] = record_dataset["sSowsNo"].astype(str)
        for code in del_code:
            code_str = str(code)
            record_dataset = record_dataset[record_dataset["sSowsNo"] != code_str]
    # print(record_dataset.shape)

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
            time = pd.to_datetime("16:00:00")
        elif AorM == "M":
            time = pd.to_datetime("09:00:00")

        # 耳号
        str_ear_tag_codes = info.split(sep="_")[-1]

        # 时间段
        if time != None:
            date_time = pd.to_datetime(f"{date.date()} {time.time()}")
            prev_time = date_time - pd.Timedelta(hours=WINDOW_SIZE + 1)
            last_time = date_time + pd.Timedelta(hours=SLIDING_WINDOW_SIZE + 1)
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
    if correct_temperature:
        record_dataset = correct_abnormal_temperatures_linear(record_dataset, 34, 5)
        # record_dataset = correct_abnormal_temperatures_linear(record_dataset, 34)
        # print(f"异常体温处理:{type(record_dataset)}")

    # -------------------- 数据选择 --------------------
    # 选择发情母猪48小时数据，对非发情母猪数据复制
    choose_data = estrusSows_data_choice(record_dataset, time_choice)
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
        estrus_end_time = pd.to_datetime(row[0].split(sep="~")[-1])
        estrus_start_time = estrus_end_time - pd.Timedelta(
            hours=SLIDING_WINDOW_SIZE + 1
        )
        # 根据耳号和发情时间段设置标签为1
        for code in row[1]:
            code_str = str(code)
            condition = (
                (setLabels_dataset["sSowsNo"] == code_str)
                & (setLabels_dataset["tLastUploadTime"] >= estrus_start_time)
                & (setLabels_dataset["tLastUploadTime"] <= estrus_end_time)
            )
            setLabels_dataset.loc[condition, "isEstrus"] = 1
    setLabels_dataset.dropna(subset=["iTemperature"], inplace=True)

    # 由于一些数据中可能不存在有2*count个正常体温值，故在此直接舍弃一些数据
    # if correct_temperature:
    # setLabels_dataset = setLabels_dataset[setLabels_dataset["iTemperature"] >= 34]

    return setLabels_dataset


# 获取发情编号
def get_estrus_earCode(time_choice):
    estrus_earCode = []
    for info in time_choice:
        str_ear_tag_codes = info.split(sep="_")[-1]
        for code in str_ear_tag_codes.split(sep=","):
            estrus_earCode.append(code)
    return estrus_earCode


# 获取非发请编号
def get_notEstrus_earCode(data: pd.DataFrame, estrus_earCode):
    all_sows_earCode = data["sSowsNo"].drop_duplicates(keep="first").to_numpy()
    notEstrus_earCode = []
    for code in all_sows_earCode:
        if code in estrus_earCode:
            continue
        else:
            notEstrus_earCode.append(code)
    return notEstrus_earCode


# 数据填补
def fill_data(
    data: pd.DataFrame,
    estrus_earCode,
    several_eastrus_earCode,
    notEStrus_earCode,
):
    filled_dataset = data.copy()
    drop_earCode = []

    # 发情母猪的数据填补
    for index in estrus_earCode:
        # 未再次发情 数据行数应为53
        if index not in several_eastrus_earCode:
            df = data[data["sSowsNo"] == index].copy()
            if len(df) < 49:  #  缺失 > 4 丢弃
                drop_earCode.append(index)
                continue
            else:  # 填补
                sub_df = df.copy()
                # 获取完整时间序列
                first_time = sub_df["tLastUploadTime"].min()
                last_time = sub_df["tLastUploadTime"].max()
                full_time_index = pd.date_range(
                    start=first_time, end=last_time, freq="h"
                )
                # 以完整时间序列为索引
                sub_df = (
                    sub_df.set_index("tLastUploadTime")
                    .reindex(full_time_index)
                    .reset_index()
                    .rename(columns={"index": "tLastUploadTime"})
                )
                # 填充其它列
                for col in [
                    "sEarTagCode",
                    "sSowsNo",
                    "sBrand",
                    "dBreedDate",
                    "dWeanDate",
                    "iTemperature",
                    "isEstrus",
                ]:
                    sub_df[col] = sub_df[col].ffill()

                # 步数采用均值填补
                mean_iStep = sub_df["iStep"].mean()
                sub_df["iStep"] = sub_df["iStep"].fillna(mean_iStep)

                # 填充temperatureRate为0
                sub_df["temperatureRate"] = sub_df["temperatureRate"].fillna(0)

                filled_dataset = filled_dataset[filled_dataset["sSowsNo"] != index]
                filled_dataset = pd.concat([filled_dataset, sub_df], ignore_index=True)
        # else: # 再次发情

    return None


# 特征构建
def feature_construction(
    data: pd.DataFrame,  # 函数data_processing处理之后的数据
):
    # 发情：检测到标签为1就获取该行及前47行的耳温数据

    # 非发请：暂定生成1个随机数获取连续的57行数据并拆分为10段连续的48小时数据
    return None
