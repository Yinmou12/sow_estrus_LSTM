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

# 数据选择
def data_choice(
    data:pd.DataFrame, # 数据
    time_choice, # 发情时间记录
    window_size=6, # 滑动窗口大小
):
    for info in time_choice: # 逐天获取发情母猪的信息
        estrus_ear_tag_codes = [] # 所有发情母猪耳标代码
        str_ear_tag_codes = info.split(sep="_")[-1] # 当天发情母猪耳标代码
        for code in str_ear_tag_codes.split(sep=","):
            estrus_ear_tag_codes.append(str(code))
    
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
        prev_time = date_time - pd.Timedelta(hours=48)
        # last_time = date_time + pd.Timedelta(days=1)

        # 提取发情母猪数据
        for code in estrus_ear_tag_codes:
            code = str(code)
            estrusSows_dataset = pd.concat(
                [
                    estrusSows_dataset,
                    intercept_data(data, code, prev_time, date_time)
                ],
                axis=0,
                ignore_index=True,
            )