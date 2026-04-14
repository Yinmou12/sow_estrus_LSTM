import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


"""class EstrusLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(EstrusLSTM, self).__init__()

        # LSTM核心层
        # batch_first=True 适配 (batch, seq_len, input_size) 的数据格式
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # 批归一化层，稳定梯度
        self.bn = nn.LayerNorm(hidden_size)
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(dropout)
        # 全连接层，进行分类映射
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        # Sigmoid激活函数，输出概率值
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # X 形状: (batch, 48, 1)

        # LSTM 的输出: out 是所有时间步的隐藏状态, (h_n, c_n) 是最后一步的状态
        out, (h_n, c_n) = self.lstm(x)

        # 只取序列中最后一个时间步的特征进行分类
        # out[:, -1, :] 形状为 (batch, hidden_size)
        feature = out[:, -1, :]

        # 进入全连接网络
        x = self.bn(feature)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return self.sigmoid(x)"""


"""class EstrusLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, dropout=0.2):
        super(EstrusLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        feature = out[:, -1, :]
        x = self.relu(self.fc1(feature))
        x = torch.sigmoid(self.fc2(x))
        return x"""


class EstrusLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, dropout_rate=0.2):
        super(EstrusLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True,  # 双向LSTM
        )
        # 双向LSTM的输出维度是 2*hidden_size
        self.fc1 = nn.Linear(2 * hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.LayerNorm(2 * hidden_size)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        # 提取最后两层（即第3层的正向和反向）的隐藏状态
        # h_n[-2,:,:] 是最后一层的正向状态
        # h_n[-1,:,:] 是最后一层的反向状态
        feature = torch.cat(
            (h_n[-2, :, :], h_n[-1, :, :]), dim=1
        )  # 形状: (batch, 2*hidden_size)
        out = self.batch_norm(feature)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))
        return out


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # 线性层用于计算注意力得分
        self.attn = nn.Linear(hidden_size, hidden_size)
        # 上下文变量，用于衡量每个时间步的重要性
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output 形状: (batch, seq_len, hidden_size)

        # 计算能量值 (Energy)
        energy = torch.tanh(self.attn(lstm_output))
        # 计算权重得分
        weights = F.softmax(self.v(energy), dim=1)

        # 计算加权后的上下文向量 (Context Vector)
        # (batch, seq_len, 1) * (batch, seq_len, hidden_size) -> 并在 seq_len 维度求和
        context_vector = torch.sum(weights * lstm_output, dim=1)
        return context_vector, weights


class EstrusLSTM_Attn(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, dropout_rate=0.2):
        super(EstrusLSTM_Attn, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True,  # 双向LSTM
        )
        # 注意力层
        self.attention = Attention(hidden_size * 2)
        # 双向LSTM的输出维度是 2*hidden_size
        self.fc1 = nn.Linear(2 * hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.LayerNorm(2 * hidden_size)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        # 提取最后两层（即第3层的正向和反向）的隐藏状态
        # h_n[-2,:,:] 是最后一层的正向状态
        # h_n[-1,:,:] 是最后一层的反向状态
        """feature = torch.cat(
            (h_n[-2, :, :], h_n[-1, :, :]), dim=1
        )  # 形状: (batch, 2*hidden_size)
        out = self.batch_norm(feature)"""

        attn_output, attn_weights = self.attention(out)
        out = self.batch_norm(attn_output)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))
        return out


class EstrusLSTM_MultiHeadAttn(nn.Module):
    def __init__(
        self, input_size=2, hidden_size=128, num_layers=2, num_heads=4, dropout_rate=0.2
    ):
        super(EstrusLSTM_MultiHeadAttn, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True,  # 双向LSTM
        )

        # 多头注意力层，embed_dim 为双向LSTM的输出维度 (2 * hidden_size)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=2 * hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,  # 确保 batch 在第 0 维
        )

        self.batch_norm = nn.LayerNorm(2 * hidden_size)
        self.fc1 = nn.Linear(2 * hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # lstm_out 形状: (batch, seq_len, 2 * hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 自注意力机制：Query, Key, Value 均使用 lstm_out
        attn_output, attn_weights = self.multihead_attn(lstm_out, lstm_out, lstm_out)

        # 对序列维度求平均，得到单个上下文向量作为分类特征
        context_vector = torch.mean(
            attn_output, dim=1
        )  # 形状: (batch, 2 * hidden_size)

        out = self.batch_norm(context_vector)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))

        return out


class EstrusGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, dropout=0.2):
        super(EstrusGRU, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        out, h_n = self.gru(x)
        feature = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        out = self.batch_norm(feature)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))
        return out


class EarlyStopping:
    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        monitor="val_loss",
        path="best_model.pth",
    ):
        """
        Args:
            patience (int): 忍受多少个 epoch 没有进步
            verbose (bool): 是否打印信息
            delta (float): 指标改善的最小阈值
            path (str): 最佳模型保存路径
            monitor (str): 监控的指标 ('val_loss' 或 'val_f1')
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.monitor = monitor
        self.path = path

    def __call__(self, current_metric, model):
        score = -current_metric if self.monitor == "val_loss" else current_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(current_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(current_metric, model)
            self.counter = 0

    def save_checkpoint(self, current_metric, model):
        if self.verbose:
            print(f"Saving checkpoint to {self.path}")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = current_metric
