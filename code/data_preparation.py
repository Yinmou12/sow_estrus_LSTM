from sow_estrus_LSTM_Info import *
import sow_estrus_LSTM_Function as myFunction

import pandas as pd

# 原始数据读取
dataset = pd.read_excel(summary_data_path + "add_2025-07-08.xlsx", sheet_name=None)
data = pd.DataFrame(columns=columns_name)
for sheet_name, sheet_data in dataset.items():
    # 确保 sheet_data 是 DataFrame 类型
    if isinstance(sheet_data, pd.DataFrame):
        data = pd.concat([data, sheet_data], axis=0, ignore_index=False)

"""
    预处理:
        异常体温处理
        更新小时数据
        计算温度变化率
        设置标签
"""
temp_dataset = myFunction.data_processing(data, TIME_CHOICE, False, "mean", DEL_CODE)
temp_dataset.to_excel(test_data_path + "测试data_precessing函数.xlsx", index=False)

"""
    特征构建：
    
"""
