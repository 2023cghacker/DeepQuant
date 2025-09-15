import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# =====================
# 生成有规律的时间序列数据
# =====================
def generate_sine_data(num_samples=2000, seq_len=30, num_features=1, noise_std=0.1):
    """
    生成正弦波加噪声的时间序列
    """
    X_all = []
    y_all = []

    # 构造连续时间点
    t = np.linspace(0, 100, num_samples + seq_len)

    # 多特征：每个特征都是不同频率的正弦波
    data = []
    for i in range(num_features):
        freq = 0.1 + 0.05 * i
        amp = 0.5 + 0.5 * i
        series = amp * np.sin(2 * np.pi * freq * t) + np.random.normal(0, noise_std, size=t.shape)
        data.append(series)
    data = np.stack(data, axis=-1)  # [num_samples+seq_len, num_features]

    # 构建样本
    for i in range(num_samples):
        X_all.append(data[i:i + seq_len])
        y_all.append(data[i + seq_len, 0])  # 预测第一个特征的下一个值

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32)
    return X_all, y_all


# =====================
# Dataset类
# =====================
class SyntheticDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


# =====================
# 测试
# =====================
if __name__ == "__main__":
    # 生成数据
    X, y = generate_sine_data(num_samples=2000, seq_len=30, num_features=3)
    dataset = SyntheticDataset(X, y)

    print("X.shape:", X.shape)  # [2000, 30, 3]
    print("y.shape:", y.shape)  # [2000]

    # 用DataLoader查看
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for xb, yb in loader:
        print("Batch X:", xb.shape)  # [batch, seq_len, num_features]
        print("Batch y:", yb.shape)  # [batch]
        break
