# 在提交前，养成先运行 status 的习惯，看看哪些文件被动过了：git status
# 如果你修改了多个文件，想全部提交，运行：git add .
# 最后完成本地记录并上传到服务器：
# git commit -m "这里写你的修改说明，例如：优化了模型参数"
# git push origin main

from sow_estrus_LSTM_Info import *
import sow_estrus_LSTM_Function as myFunction

import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

save_path = result_save_path + "Filename"

data = pd.read_excel(
    experimentRecord_data_path + "data_precessed\\filled_dataset.xlsx", index_col=False
)

# 分层分组划分数据集，确保同一母猪的数据不会同时出现在训练集、验证集和测试集中
train_df, val_df, test_df = myFunction.stratified_group_split(data)

# 划分数据集并进行标准化处理，准备输入 LSTM 模型的格式
X_train, y_train, train_scaler = myFunction.prepare_univariate_lstm_data(train_df)
X_val, y_val, _ = myFunction.prepare_univariate_lstm_data(val_df, train_scaler)
X_test, y_test, _ = myFunction.prepare_univariate_lstm_data(test_df, train_scaler)

os.makedirs(save_path, exist_ok=True)
joblib.dump(train_scaler, save_path)
print(f"Scaler saved to {save_path}")
