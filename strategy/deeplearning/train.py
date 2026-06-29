from stocklstm import StockLSTM
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from data_processor.DataPlot import read_stock_data
import numpy as np
from strategy.deeplearning.StockDataset import (
    StockDataset_ret,
    StockDataset_close,
    StockDataset_binary,
)
import torch
import torch.nn as nn


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

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

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
    plt.plot(reals, label="真实收盘价", color="black")
    plt.plot(preds, label="预测收盘价", color="red", linestyle="--")
    plt.title("真实 vs 预测 收盘价")
    plt.xlabel("时间步")
    plt.ylabel("收盘价")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    print(f"\n[-] loading dataset...")
    df = read_stock_data(r"data\raw\000001_SZ_20250701_20260601_ind.csv")
    # df = read_stock_data(r"data\raw\000001_SZ_20250701_20260601_ohlcv.csv")
    seq_len = 20  # 时间序列长度
    # dataset = StockDataset_ret(df, seq_len=seq_len, target="close", pred_horizon=1)
    dataset = StockDataset_binary(df, seq_len=seq_len, target="close", pred_horizon=1)
    dim = dataset[0][0].shape[1]  # 单个时间步特征维度数
    print(f"[√] dataset loaded: dataset.len={len(dataset)} x.shape={seq_len}*{dim}")

    # 加载模型
    model = StockLSTM(input_size=dim).to(device)
    print(f"\n[√] model = {model}")

    # 训练模型
    print(f"\n[-] training...")
    train_model(dataset, batch_size=8, epochs=100, lr=1e-4)
    print(f"[√] training finished ")

    # 绘制真实 vs 预测
    plot_predictions(model, dataset, device, num_points=50)
