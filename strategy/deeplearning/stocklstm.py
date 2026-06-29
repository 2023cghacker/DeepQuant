import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from data_processor.DataPlot import read_stock_data
import numpy as np
from strategy.deeplearning.StockDataset import StockDataset_ret, StockDataset_close
import torch
import torch.nn as nn

# from tests.SyntheticDataset import generate_sine_data, SyntheticDataset


class StockLSTM(nn.Module):
    def __init__(
        self,
        input_size=5,
        hidden_size=128,
        num_layers=3,
        output_size=1,
        dropout=0.3,
        bidirectional=True,
    ):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,  # 在 LSTM 层之间加 Dropout，防止过拟合
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out_size = hidden_size * (2 if bidirectional else 1)

        # 对 LSTM 输出做归一化
        self.norm1 = nn.LayerNorm(lstm_out_size)

        # 全连接层 + ReLU + 归一化
        self.fc1 = nn.Linear(lstm_out_size, lstm_out_size // 2)
        self.relu = nn.ReLU()
        self.norm2 = nn.LayerNorm(lstm_out_size // 2)

        self.fc2 = nn.Linear(lstm_out_size // 2, output_size)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, _ = self.lstm(x)  # [batch, seq_len, hidden_size*2]
        out = out[:, -1, :]  # 取最后时间步的输出
        out = self.norm1(out)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.norm2(out)

        out = self.fc2(out)  # [batch, 1]
        return out.squeeze(-1)  # [batch]
