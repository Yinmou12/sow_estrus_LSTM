# 在提交前，养成先运行 status 的习惯，看看哪些文件被动过了：git status
# 如果你修改了多个文件，想全部提交，运行：git add .
# 最后完成本地记录并上传到服务器：
# git commit -m "这里写你的修改说明，例如：优化了模型参数"
# git push origin master

import sow_estrus_LSTM_Info
import sow_estrus_LSTM_Function as myFunction

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    log_loss,
    roc_auc_score,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

source_data_path = sow_estrus_LSTM_Info.source_data_path
summary_data_path = sow_estrus_LSTM_Info.summary_data_path
experimentRecord_data_path = sow_estrus_LSTM_Info.experimentRecord_data_path
test_data_path = sow_estrus_LSTM_Info.test_data_path
save_modeel_path = sow_estrus_LSTM_Info.save_model_path

time_choice = sow_estrus_LSTM_Info.time_choice

columns_name = sow_estrus_LSTM_Info.columns_name
columns_temperatures = [f"temperature_{i}" for i in range(1, 49)]

start_time = sow_estrus_LSTM_Info.start_time
end_time = sow_estrus_LSTM_Info.end_time
