"""
母猪体温异常值处理模块
提供多种异常值处理方法，用于提升数据质量
"""

import numpy as np
import pandas as pd
from scipy import interpolate


def correct_abnormal_temperatures_linear(
    data: pd.DataFrame,  # 数据集
    threshold_low: float = 34.0,  # 低体温阈值
    threshold_high: float = 40.0,  # 高体温阈值
):
    """
    基于线性插值的异常值处理方法
    修复过低或过高的体温值，使用相邻正常值进行线性插值

    Args:
        data: 包含体温数据的DataFrame
        threshold_low: 低体温阈值，低于此值视为异常
        threshold_high: 高体温阈值，高于此值视为异常

    Returns:
        处理后的DataFrame
    """
    tempDataframe = data.copy()
    ear_tag_codes = tempDataframe.drop_duplicates(subset="sSowsNo", keep="first")[
        "sSowsNo"
    ].to_numpy()

    final_data = pd.DataFrame(columns=data.columns)
    processed_data_list = []

    for ear_code in ear_tag_codes:
        part_data = data[data["sSowsNo"] == ear_code].copy()
        part_data = part_data.sort_values("tLastUploadTime").reset_index(drop=True)

        # 获取时间序列
        time_series = part_data["tLastUploadTime"]
        temperatures = part_data["iTemperature"].values

        # 识别异常值
        abnormal_mask = (temperatures <= threshold_low) | (
            temperatures >= threshold_high
        )

        # 线性插值修复
        for idx in np.where(abnormal_mask)[0]:
            if idx == 0:
                # 如果是第一个点，使用下一个正常值
                next_valid = np.where(~abnormal_mask)[0]
                if len(next_valid) > 0:
                    temperatures[idx] = temperatures[next_valid[0]]
            elif idx == len(temperatures) - 1:
                # 如果是最后一个点，使用前一个正常值
                prev_valid = np.where(~abnormal_mask)[0]
                if len(prev_valid) > 0:
                    temperatures[idx] = temperatures[prev_valid[-1]]
            else:
                # 找到前一个和后一个正常值
                prev_valid = np.where(~abnormal_mask)[0]
                prev_idx = (
                    prev_valid[prev_valid < idx][-1]
                    if len(prev_valid[prev_valid < idx]) > 0
                    else -1
                )
                next_idx = (
                    prev_valid[prev_valid > idx][0]
                    if len(prev_valid[prev_valid > idx]) > 0
                    else len(temperatures)
                )

                if prev_idx != -1 and next_idx < len(temperatures):
                    # 线性插值
                    ratio = (idx - prev_idx) / (next_idx - prev_idx)
                    temperatures[idx] = temperatures[prev_idx] + ratio * (
                        temperatures[next_idx] - temperatures[prev_idx]
                    )

        part_data["iTemperature"] = temperatures
        processed_data_list.append(part_data)

    final_data = pd.concat(processed_data_list, axis=0, ignore_index=True)

    return final_data


# 基于移动平均的异常值处理
def correct_abnormal_temperatures_moving_avg(
    data: pd.DataFrame,
    window_size: int = 3,
    threshold_low: float = 34.0,
    threshold_high: float = 40.0,
    sigma: float = 2.0,
):
    """
    基于移动平均和统计方法的异常值处理

    使用移动平均和标准差来识别和处理异常值，考虑时间序列的局部特征

    Args:
        data: 包含体温数据的DataFrame
        window_size: 移动平均窗口大小
        threshold_low: 低体温阈值
        threshold_high: 高体温阈值
        sigma: 统计倍数，用于动态阈值计算

    Returns:
        处理后的DataFrame
    """
    tempDataframe = data.copy()
    ear_tag_codes = tempDataframe.drop_duplicates(subset="sSowsNo", keep="first")[
        "sSowsNo"
    ].to_numpy()

    final_data = pd.DataFrame(columns=data.columns)
    processed_data_list = []

    for ear_code in ear_tag_codes:
        part_data = data[data["sSowsNo"] == ear_code].copy()
        part_data = part_data.sort_values("tLastUploadTime").reset_index(drop=True)

        temperatures = part_data["iTemperature"].values

        # 计算移动平均和标准差
        moving_avg = (
            pd.Series(temperatures)
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )

        moving_std = (
            pd.Series(temperatures)
            .rolling(window=window_size, center=True, min_periods=1)
            .std()
        )

        # 动态阈值：移动平均 ± sigma * 标准差
        upper_threshold = moving_avg + sigma * moving_std
        lower_threshold = moving_avg - sigma * moving_std

        # 识别异常值
        abnormal_mask = (
            (temperatures <= threshold_low)
            | (temperatures >= threshold_high)
            | (temperatures > upper_threshold)
            | (temperatures < lower_threshold)
        )

        # 使用移动平均修复异常值
        temperatures[abnormal_mask] = moving_avg[abnormal_mask]

        part_data["iTemperature"] = temperatures
        processed_data_list.append(part_data)

    final_data = pd.concat(processed_data_list, axis=0, ignore_index=True)

    return final_data


# 基于样条插值的异常值处理
def correct_abnormal_temperatures_spline(
    data: pd.DataFrame,
    threshold_low: float = 34.0,
    threshold_high: float = 40.0,
    smoothing: float = 0.5,
):
    """
    使用三次样条插值处理异常值

    利用样条插值函数来平滑异常值，更适合复杂的时间序列模式

    Args:
        data: 包含体温数据的DataFrame
        threshold_low: 低体温阈值
        threshold_high: 高体温阈值
        smoothing: 平滑参数，0为插值，越大越平滑

    Returns:
        处理后的DataFrame
    """
    tempDataframe = data.copy()
    ear_tag_codes = tempDataframe.drop_duplicates(subset="sSowsNo", keep="first")[
        "sSowsNo"
    ].to_numpy()

    final_data = pd.DataFrame(columns=data.columns)
    processed_data_list = []

    for ear_code in ear_tag_codes:
        part_data = data[data["sSowsNo"] == ear_code].copy()
        part_data = part_data.sort_values("tLastUploadTime").reset_index(drop=True)

        # 转换为时间数值（小时）
        times = pd.to_datetime(part_data["tLastUploadTime"])
        time_numeric = (times - times.min()).dt.total_seconds() / 3600
        temperatures = part_data["iTemperature"].values

        # 识别异常值
        abnormal_mask = (temperatures <= threshold_low) | (
            temperatures >= threshold_high
        )

        # 获取正常值用于插值
        normal_mask = ~abnormal_mask
        normal_times = time_numeric[normal_mask]
        normal_temps = temperatures[normal_mask]

        # 如果有足够的正常值，进行样条插值
        if len(normal_times) >= 4:  # 样条插值至少需要4个点
            # 创建样条插值函数（使用UnivariateSpline支持平滑）
            spline_func = interpolate.UnivariateSpline(
                normal_times, normal_temps, s=smoothing
            )

            # 修复异常值
            abnormal_times = time_numeric[abnormal_mask]
            temperatures[abnormal_mask] = spline_func(abnormal_times)

        part_data["iTemperature"] = temperatures
        processed_data_list.append(part_data)

    final_data = pd.concat(processed_data_list, axis=0, ignore_index=True)

    return final_data


# 基于生理节律的异常值处理
def correct_abnormal_temperatures_circadian(
    data: pd.DataFrame,
    threshold_low: float = 34.0,
    threshold_high: float = 40.0,
    time_window: int = 6,
):
    """
    考虑昼夜节律的异常值处理

    基于母猪的生理节律，在不同时间段使用不同的基准体温进行异常值修复

    Args:
        data: 包含体温数据的DataFrame
        threshold_low: 低体温阈值
        threshold_high: 高体温阈值
        time_window: 时间窗口大小（小时），用于计算局部基准体温

    Returns:
        处理后的DataFrame
    """
    tempDataframe = data.copy()
    ear_tag_codes = tempDataframe.drop_duplicates(subset="sSowsNo", keep="first")[
        "sSowsNo"
    ].to_numpy()

    final_data = pd.DataFrame(columns=data.columns)
    processed_data_list = []

    for ear_code in ear_tag_codes:
        part_data = data[data["sSowsNo"] == ear_code].copy()
        part_data = part_data.sort_values("tLastUploadTime").reset_index(drop=True)

        # 提取小时信息
        times = pd.to_datetime(part_data["tLastUploadTime"])
        hours = times.dt.hour
        temperatures = part_data["iTemperature"].values

        # 计算每个时间段的基准体温（基于历史数据的同时间段）
        baseline_temps = {}
        for hour in range(24):
            hour_mask = hours == hour
            if hour_mask.sum() > 0:
                baseline_temps[hour] = np.median(temperatures[hour_mask])
            else:
                baseline_temps[hour] = 38.5  # 默认体温

        # 识别异常值
        abnormal_mask = (temperatures <= threshold_low) | (
            temperatures >= threshold_high
        )

        # 基于节律修正异常值
        for idx in np.where(abnormal_mask)[0]:
            current_hour = hours.iloc[idx] if hasattr(hours, "iloc") else hours[idx]

            # 找到前后时间窗口内的正常值
            start_hour = max(0, current_hour - time_window)
            end_hour = min(23, current_hour + time_window)

            window_temps = []
            for h in range(start_hour, end_hour + 1):
                if h in baseline_temps:
                    window_temps.append(baseline_temps[h])

            if window_temps:
                # 使用窗口内基准体温的中位数
                temperatures[idx] = np.median(window_temps)
            else:
                # 使用全局中位数
                normal_temps = temperatures[~abnormal_mask]
                if len(normal_temps) > 0:
                    temperatures[idx] = np.median(normal_temps)
                else:
                    temperatures[idx] = 38.5  # 默认体温

        part_data["iTemperature"] = temperatures
        processed_data_list.append(part_data)

    final_data = pd.concat(processed_data_list, axis=0, ignore_index=True)

    return final_data


# 多重方法组合异常值处理
def correct_abnormal_temperatures_ensemble(
    data: pd.DataFrame, methods: list = None, weights: list = None, **kwargs
):
    """
    多重方法组合异常值处理

    组合多种异常值处理方法的结果，通过加权平均获得最终结果

    Args:
        data: 包含体温数据的DataFrame
        methods: 使用的处理方法列表，可选: 'linear', 'moving_avg', 'spline', 'circadian'
        weights: 各方法的权重列表
        **kwargs: 传递给各方法的额外参数

    Returns:
        处理后的DataFrame
    """
    if methods is None:
        methods = ["linear", "moving_avg", "circadian"]

    if weights is None:
        weights = [0.3, 0.4, 0.3]  # 默认权重

    # 确保权重和方法数量匹配
    if len(methods) != len(weights):
        weights = [1.0 / len(methods)] * len(methods)

    # 存储每种方法的结果
    results = []

    # 执行每种方法
    for method in methods:
        if method == "linear":
            result = correct_abnormal_temperatures_linear(data, **kwargs)
        elif method == "moving_avg":
            result = correct_abnormal_temperatures_moving_avg(data, **kwargs)
        elif method == "spline":
            result = correct_abnormal_temperatures_spline(data, **kwargs)
        elif method == "circadian":
            result = correct_abnormal_temperatures_circadian(data, **kwargs)
        else:
            print(f"未知方法: {method}")
            continue

        results.append(result)

    # 如果只有一个方法，直接返回
    if len(results) == 1:
        return results[0]

    # 加权合并结果
    final_data = pd.DataFrame(columns=data.columns)
    processed_data_list = []
    ear_tag_codes = data["sSowsNo"].unique()

    for ear_code in ear_tag_codes:
        combined_temperatures = None

        for i, result in enumerate(results):
            part_result = result[result["sSowsNo"] == ear_code]
            part_result = part_result.sort_values("tLastUploadTime").reset_index(
                drop=True
            )

            if combined_temperatures is None:
                combined_temperatures = part_result["iTemperature"].values * weights[i]
            else:
                combined_temperatures += part_result["iTemperature"].values * weights[i]

        # 获取基础数据结构
        part_data = data[data["sSowsNo"] == ear_code].copy()
        part_data = part_data.sort_values("tLastUploadTime").reset_index(drop=True)
        part_data["iTemperature"] = combined_temperatures
        processed_data_list.append(part_data)

    final_data = pd.concat(processed_data_list, axis=0, ignore_index=True)

    return final_data
