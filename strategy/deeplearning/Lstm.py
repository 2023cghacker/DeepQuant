from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from data_processor.DataPlot import read_stock_data
import numpy as np

from strategy.StockDataset import StockDataset_earning, StockDataset_val

# =====================
# 模型定义
# =====================
import torch
import torch.nn as nn


import torch
import torch.nn as nn

from tests.SyntheticDataset import generate_sine_data, SyntheticDataset


class StockLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=3, output_size=1, dropout=0.3, bidirectional=True):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,  # 在 LSTM 层之间加 Dropout，防止过拟合
            batch_first=True,
            bidirectional=bidirectional
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

# =====================
# 训练逻辑
# =====================
def train_model(dataset, batch_size=32, epochs=20, lr=1e-3):
    # 划分训练/验证
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            # print(f"X.shape={X.shape}, y.shape={y.shape}")
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                # print(f"preds={preds}")
                loss = criterion(preds, y)
                val_loss += loss.item() * X.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # return model


def plot_predictions(model, dataset, device, num_points=200):
    """
    绘制真实收盘价 vs 预测收盘价

    num_points: 画多少个点（从最后开始取）
    """
    model.eval()
    preds = []
    reals = []

    with torch.no_grad():
        for i in range(len(dataset)):
            X, y = dataset[i]
            X = X.unsqueeze(0).to(device)  # [1, seq_len, 5]
            pred = model(X).item()
            preds.append(pred)
            reals.append(y.item())

    preds = np.array(preds)
    reals = np.array(reals)

    # 取最后 num_points 个点绘制
    preds = preds[-num_points:]
    reals = reals[-num_points:]
    # print(f"reals={reals},preds={preds}")

    plt.figure(figsize=(12, 6))
    plt.plot(reals, label="真实收盘价", color='black')
    plt.plot(preds, label="预测收盘价", color='red', linestyle='--')
    plt.title("真实 vs 预测 收盘价")
    plt.xlabel("时间步")
    plt.ylabel("收盘价")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = read_stock_data('D:\lc\githubCode\DeepQuant\data\pingan_bank_201807.csv')  # 替换为你的CSV文件路径

    # 数据集类
    # X, y = generate_sine_data(num_samples=2000, seq_len=10, num_features=5)
    # dataset = SyntheticDataset(X, y)
    dataset = StockDataset_earning(df, seq_len=10, target='Close', pred_horizon=1)
    # dataset = StockDataset_val(df, seq_len=60, target='Close', pred_horizon=1)
    print(f"len(dataset)={len(dataset)}")

    # 训练模型
    model = StockLSTM(input_size=5).to(device)
    train_model(dataset, batch_size=8, epochs=50, lr=1e-4)

    # 绘制真实 vs 预测
    plot_predictions(model, dataset, device, num_points=200)

    # 预测例子（取最后一段数据预测下一天）
    model.eval()
    last_seq = torch.tensor(df[['Open', 'High', 'Low', 'Close', 'Volume']].values[-30:], dtype=torch.float32)
    pred = model(last_seq.unsqueeze(0).to(device))

    print("Predicted next close:", pred.item())
