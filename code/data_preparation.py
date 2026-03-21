from sow_estrus_LSTM_Info import *
import sow_estrus_LSTM_Function as myFunction

import pandas as pd

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

"""processed_dataset = pd.read_excel(
    experimentRecord_data_path + "data_processed\\data_processed.xlsx", index_col=False
)"""


# 拆分多次发情的数据并更新发情编号
# splited_dataset = myFunction.split_estrusData(processed_dataset, TIME_CHOICE, 7)
splited_dataset = pd.read_excel(
    experimentRecord_data_path + "data_processed\\splited_dataset.xlsx", index_col=False
)


# 数据填补
filled_dataset, drop_estrus_earCode, drop_notEStrus_earCode = myFunction.fill_data(
    splited_dataset
)

filled_dataset.to_excel(
    experimentRecord_data_path + "filled_dataset_SLIDING_WINDOW_SIZE(2)_1.xlsx",
    index=False,
)
