from torch.utils.data import Dataset
import torch
import numpy as np


class StockDataset_val(Dataset):
    def __init__(self, df, seq_len=30, target='Close', pred_horizon=1):
        """
        df: pandas DataFrame, 带有Open, High, Low, Close, Volume等列
        seq_len: 输入的时间窗口长度 (比如 30天)
        target: 预测的目标列名
        pred_horizon: 预测多少天后的收盘价 (1 表示预测下一天)
        """
        self.df = df
        self.seq_len = seq_len
        self.target = target
        self.pred_horizon = pred_horizon
        # self.features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
        self.features = df[["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]].values
        self.targets = df[target].values

    def __len__(self):
        return len(self.df) - self.seq_len - self.pred_horizon + 1

    def __getitem__(self, idx):
        # 取 seq_len 天的特征
        X = self.features[idx: idx + self.seq_len]
        # 预测 pred_horizon 天后的目标
        y = self.targets[idx + self.seq_len + self.pred_horizon - 1]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class StockDataset_earning(Dataset):
    def __init__(self, df, seq_len=30, target='Close', pred_horizon=1):
        """
        df: pandas DataFrame, 带有Open, High, Low, Close, Volume等列
        seq_len: 输入的时间窗口长度 (比如 30天)
        target: 预测的目标列名
        pred_horizon: 预测多少天后的涨跌幅 (1 表示预测下一天)
        """
        self.df = df
        self.seq_len = seq_len
        self.target = target
        self.pred_horizon = pred_horizon
        # self.features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
        self.features = df[["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]].values

        # 计算每日收益率 (Close[t+1] - Close[t]) / Close[t]
        prices = df[target].values
        returns = (prices[self.pred_horizon:] - prices[:-self.pred_horizon]) / prices[:-self.pred_horizon]
        # 对齐特征长度
        self.targets = returns[self.seq_len - 1:]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        X = self.features[idx: idx + self.seq_len]
        y = self.targets[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class StockDataset_binary(Dataset):
    """
    输入：seq_len 天的特征（Open, High, Low, Close, Volume）
    输出：二分类标签，预测下一天是涨（1）还是跌（0）
    """

    def __init__(self, df, seq_len=30, target='close', pred_horizon=1):
        self.df = df
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon

        # self.features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
        self.features = df[["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]].values

        # 计算涨跌标签
        prices = df[target].values
        returns = prices[self.pred_horizon:] - prices[:-self.pred_horizon]
        labels = (returns > 0).astype(np.float32)  # 涨为1，跌为0
        self.targets = labels[self.seq_len - 1:]  # 对齐特征长度

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        X = self.features[idx: idx + self.seq_len]
        y = self.targets[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
