import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_hidden_sizes(hidden_size=128, num_layers=4, hidden_sizes=None):
    if hidden_sizes is None and isinstance(hidden_size, (list, tuple)):
        hidden_sizes = hidden_size

    if hidden_sizes is None:
        return [int(hidden_size)] * int(num_layers)

    resolved = [int(size) for size in hidden_sizes]
    if not resolved:
        raise ValueError("hidden_sizes must contain at least one layer size")
    return resolved


class LayerWiseBiLSTM(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=128,
        num_layers=4,
        dropout_rate=0.2,
        bidirectional=True,
        hidden_sizes=None,
    ):
        super().__init__()
        self.hidden_sizes = _resolve_hidden_sizes(hidden_size, num_layers, hidden_sizes)
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        layers = []
        layer_input_size = input_size
        for layer_hidden_size in self.hidden_sizes:
            layers.append(
                nn.LSTM(
                    input_size=layer_input_size,
                    hidden_size=layer_hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
            )
            layer_input_size = layer_hidden_size * self.num_directions

        self.layers = nn.ModuleList(layers)
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(max(0, len(layers) - 1))]
        )
        self.output_size = self.hidden_sizes[-1] * self.num_directions

    def forward(self, x):
        out = x
        last_state = None
        for layer_idx, lstm in enumerate(self.layers):
            out, last_state = lstm(out)
            if layer_idx < len(self.dropouts):
                out = self.dropouts[layer_idx](out)
        return out, last_state


class LayerWiseBiRNN(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=128,
        num_layers=4,
        dropout_rate=0.2,
        bidirectional=True,
        hidden_sizes=None,
        nonlinearity="tanh",
    ):
        super().__init__()
        self.hidden_sizes = _resolve_hidden_sizes(hidden_size, num_layers, hidden_sizes)
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        layers = []
        layer_input_size = input_size
        for layer_hidden_size in self.hidden_sizes:
            layers.append(
                nn.RNN(
                    input_size=layer_input_size,
                    hidden_size=layer_hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                    nonlinearity=nonlinearity,
                )
            )
            layer_input_size = layer_hidden_size * self.num_directions

        self.layers = nn.ModuleList(layers)
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(max(0, len(layers) - 1))]
        )
        self.output_size = self.hidden_sizes[-1] * self.num_directions

    def forward(self, x):
        out = x
        last_state = None
        for layer_idx, rnn in enumerate(self.layers):
            out, last_state = rnn(out)
            if layer_idx < len(self.dropouts):
                out = self.dropouts[layer_idx](out)
        return out, last_state


class EstrusLSTM(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=128,
        num_layers=4,
        hidden_sizes=None,
        dropout_rate=0.2,
        dropout=None,
        use_cell_state=True,
        feature_mode="h_n_state",
        bidirectional=True,
    ):
        super().__init__()

        if dropout is not None:
            dropout_rate = dropout

        self.use_cell_state = use_cell_state
        self.feature_mode = feature_mode
        self.bidirectional = bidirectional

        self.encoder = LayerWiseBiLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
        )
        self.lstm_layers = self.encoder.layers
        self.hidden_sizes = self.encoder.hidden_sizes
        feature_size = self.encoder.output_size

        if self.feature_mode == "h_n_state":
            feat_dim = feature_size * 2 if self.use_cell_state else feature_size
        elif self.feature_mode in ["last_out", "mean_out"]:
            feat_dim = feature_size
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

        self.layer_norm = nn.LayerNorm(feat_dim)
        self.fc1 = nn.Linear(feat_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def _last_state_feature(self, h_n, c_n):
        if self.bidirectional:
            h_last = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1)
            if not self.use_cell_state:
                return h_last
            c_last = torch.cat([c_n[-2, :, :], c_n[-1, :, :]], dim=1)
            return torch.cat([h_last, c_last], dim=1)

        h_last = h_n[-1, :, :]
        if not self.use_cell_state:
            return h_last
        c_last = c_n[-1, :, :]
        return torch.cat([h_last, c_last], dim=1)

    def forward(self, x):
        out, (h_n, c_n) = self.encoder(x)

        if self.feature_mode == "h_n_state":
            feature = self._last_state_feature(h_n, c_n)
        elif self.feature_mode == "last_out":
            feature = out[:, -1, :]
        elif self.feature_mode == "mean_out":
            feature = torch.mean(out, dim=1)
        else:
            raise ValueError(f"Unknown feature_mode: {self.feature_mode}")

        out = self.layer_norm(feature)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return torch.sigmoid(self.fc2(out))


class DotProductAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # A learnable query vector that represents the "optimal" feature for estrus detection
        self.query = nn.Parameter(torch.randn(input_dim))

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, input_dim)
        # scores: (batch, seq_len) - measure similarity between query and each time step
        scores = torch.matmul(lstm_output, self.query)
        weights = F.softmax(scores, dim=1)
        # context_vector: (batch, input_dim)
        context_vector = torch.sum(weights.unsqueeze(-1) * lstm_output, dim=1)
        return context_vector, weights


class ScaledDotProductAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(input_dim))
        self.scale = np.sqrt(input_dim)

    def forward(self, lstm_output):
        # scores: (batch, seq_len)
        scores = torch.matmul(lstm_output, self.query) / self.scale
        weights = F.softmax(scores, dim=1)
        context_vector = torch.sum(weights.unsqueeze(-1) * lstm_output, dim=1)
        return context_vector, weights


class EstrusLSTM_DotProductAttn(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=128,
        num_layers=3,
        hidden_sizes=None,
        dropout_rate=0.2,
        bidirectional=True,
    ):
        super().__init__()
        self.encoder = LayerWiseBiLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
        )
        feature_size = self.encoder.output_size
        self.attention = DotProductAttention(feature_size)
        self.layer_norm = nn.LayerNorm(feature_size)
        self.fc1 = nn.Linear(feature_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        lstm_out, _ = self.encoder(x)
        attn_output, _ = self.attention(lstm_out)
        out = self.layer_norm(attn_output)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return torch.sigmoid(self.fc2(out))


class EstrusLSTM_ScaledDotProductAttn(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=128,
        num_layers=3,
        hidden_sizes=None,
        dropout_rate=0.2,
        bidirectional=True,
    ):
        super().__init__()
        self.encoder = LayerWiseBiLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
        )
        feature_size = self.encoder.output_size
        self.attention = ScaledDotProductAttention(feature_size)
        self.layer_norm = nn.LayerNorm(feature_size)
        self.fc1 = nn.Linear(feature_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        lstm_out, _ = self.encoder(x)
        attn_output, _ = self.attention(lstm_out)
        out = self.layer_norm(attn_output)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return torch.sigmoid(self.fc2(out))


class TemporalAttention(nn.Module):
    def __init__(self, input_dim, attention_dim=None):
        super().__init__()
        attention_dim = attention_dim or input_dim
        self.attn = nn.Linear(input_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, lstm_output):
        energy = torch.tanh(self.attn(lstm_output))
        weights = F.softmax(self.v(energy), dim=1)
        context_vector = torch.sum(weights * lstm_output, dim=1)
        return context_vector, weights


class Attention(TemporalAttention):
    def __init__(self, hidden_size):
        super().__init__(hidden_size)


class EstrusLSTM_Attn(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=128,
        num_layers=3,
        hidden_sizes=None,
        dropout_rate=0.2,
        dropout=None,
        bidirectional=True,
    ):
        super().__init__()

        if dropout is not None:
            dropout_rate = dropout

        self.encoder = LayerWiseBiLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
        )
        self.lstm_layers = self.encoder.layers
        self.hidden_sizes = self.encoder.hidden_sizes
        feature_size = self.encoder.output_size

        self.attention = TemporalAttention(feature_size)
        self.layer_norm = nn.LayerNorm(feature_size)
        self.fc1 = nn.Linear(feature_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        lstm_out, _ = self.encoder(x)
        attn_output, _ = self.attention(lstm_out)
        out = self.layer_norm(attn_output)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return torch.sigmoid(self.fc2(out))


class EstrusLSTM_MultiHeadAttn(nn.Module):
    def __init__(
        self,
        input_size=2,
        hidden_size=128,
        num_layers=2,
        hidden_sizes=None,
        num_heads=4,
        dropout_rate=0.2,
        dropout=None,
        bidirectional=True,
    ):
        super().__init__()

        if dropout is not None:
            dropout_rate = dropout

        self.encoder = LayerWiseBiLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
        )
        self.lstm_layers = self.encoder.layers
        self.hidden_sizes = self.encoder.hidden_sizes
        feature_size = self.encoder.output_size

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(feature_size)
        self.fc1 = nn.Linear(feature_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        lstm_out, _ = self.encoder(x)
        attn_output, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)
        context_vector = torch.mean(attn_output, dim=1)
        out = self.layer_norm(context_vector)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return torch.sigmoid(self.fc2(out))


class EstrusGRU(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=128,
        num_layers=3,
        dropout=0.2,
        bidirectional=True,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.num_directions = 2 if bidirectional else 1
        self.fc1 = nn.Linear(hidden_size * self.num_directions, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

    def forward(self, x):
        _, h_n = self.gru(x)
        if self.gru.bidirectional:
            feature = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            feature = h_n[-1, :, :]
        out = self.layer_norm(feature)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return torch.sigmoid(self.fc2(out))


class EstrusRNN_sample(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=128,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
    ):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * self.num_directions, 1)

    def forward(self, x):
        _, h_n = self.rnn(x)
        if self.rnn.bidirectional:
            feature = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            feature = h_n[-1]
        return torch.sigmoid(self.fc(feature))


class EstrusRNN(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=128,
        num_layers=4,
        hidden_sizes=None,
        dropout_rate=0.2,
        dropout=None,
        feature_mode="h_n_state",
        bidirectional=False,
        nonlinearity="tanh",
    ):
        super().__init__()

        if dropout is not None:
            dropout_rate = dropout

        self.feature_mode = feature_mode
        self.bidirectional = bidirectional

        self.encoder = LayerWiseBiRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
            nonlinearity=nonlinearity,
        )
        self.rnn_layers = self.encoder.layers
        self.hidden_sizes = self.encoder.hidden_sizes
        feature_size = self.encoder.output_size

        if self.feature_mode in ["h_n_state", "last_out", "mean_out"]:
            feat_dim = feature_size
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

        self.layer_norm = nn.LayerNorm(feat_dim)
        self.fc1 = nn.Linear(feat_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def _last_state_feature(self, h_n):
        if self.bidirectional:
            return torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1)
        return h_n[-1, :, :]

    def forward(self, x):
        out, h_n = self.encoder(x)

        if self.feature_mode == "h_n_state":
            feature = self._last_state_feature(h_n)
        elif self.feature_mode == "last_out":
            feature = out[:, -1, :]
        elif self.feature_mode == "mean_out":
            feature = torch.mean(out, dim=1)
        else:
            raise ValueError(f"Unknown feature_mode: {self.feature_mode}")

        out = self.layer_norm(feature)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return torch.sigmoid(self.fc2(out))


class EarlyStopping:
    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        monitor="val_loss",
        path="best_model.pth",
    ):
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
