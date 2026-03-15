import torch
import torch.nn as nn


class EstrusLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout=0.2):
        super(EstrusLSTM, self).__init__()

        # LSTM核心层
        # batch_first=True 适配 (batch, seq_len, input_size) 的数据格式
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # 批归一化层，稳定梯度
        self.bn = nn.BatchNorm1d(hidden_size)

        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 全连接层，进行分类映射
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        # Sigmoid激活函数，输出概率值
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # X 形状: (batch, 48, 1)

        # LSTM 的输出: out 是所有时间步的隐藏状态, (h_n, c_n) 是最后一步的状态
        out, (h_n, c_n) = self.lstm(x)

        # 只取序列中最后一个时间步的特征进行分类
        # out[:, -1, :] 形状为 (batch, hidden_size)
        x = out[:, -1, :]

        # 进入全连接网络
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return self.sigmoid(x)
