from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from data_processor.DataPlot import read_stock_data
import numpy as np
from strategy.StockDataset import StockDataset_earning, StockDataset_val, StockDataset_binary
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr  # 用于计算皮尔逊相关系数及显著性


class StockMLP(nn.Module):
    def __init__(self, seq_len, num_features, hidden_sizes=[64, 256, 64], dropout=0.3):
        """
        二分类MLP，用于预测涨跌
        seq_len: 时间序列长度
        num_features: 每天的特征数，比如 Open, High, Low, Close, Volume
        hidden_sizes: 隐藏层大小列表
        dropout: Dropout比率
        """
        super(StockMLP, self).__init__()
        self.input_size = seq_len * num_features  # 展平后的输入维度

        layers = []
        prev_size = self.input_size
        for hs in hidden_sizes:
            layers.append(nn.BatchNorm1d(prev_size))
            layers.append(nn.Linear(prev_size, hs))
            layers.append(nn.BatchNorm1d(hs))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hs

        layers.append(nn.Linear(prev_size, 1))  # 输出单个logit
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch, seq_len, num_features]
        x = x.reshape(x.size(0), -1)  # 展平为 [batch, seq_len*num_features]
        logits = self.model(x)  # [batch, 1]
        probs = torch.sigmoid(logits)  # 二分类概率
        return probs.squeeze(-1)  # [batch]


# =====================
# 训练逻辑
# =====================
import datetime  # 用于生成时间戳


def train_model(model, batch_size=32, epochs=20, lr=1e-3, device='cpu', save_interval=10, save_dir=""):
    """
    训练模型并定期保存

    参数:
        model: 要训练的模型
        batch_size: 批次大小
        epochs: 训练轮次
        lr: 学习率
        device: 训练设备
        save_interval: 保存间隔（每多少个epoch保存一次）
    """
    # 初始化时间戳（训练开始时的时间，确保所有保存的模型使用同一时间戳）
    timestamp = datetime.datetime.now().strftime("%m%d%H%M")

    criterion = nn.BCELoss()  # 模型已应用sigmoid，使用BCELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).float()  # 确保y是float类型
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
        correct = 0
        total = 0
        threshold = 0.5  # 概率阈值，用于分类

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).float()
                preds = model(X)
                loss = criterion(preds, y)
                val_loss += loss.item() * X.size(0)

                # 计算准确率
                pred_classes = (preds >= threshold).float()  # 转换为0或1
                correct += (pred_classes == y).sum().item()
                total += y.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = (correct / total) * 100  # 转换为百分比
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_acc:.2f}%")

        # 模型保存逻辑：每save_interval个epoch保存一次，且最后一个epoch必须保存
        current_epoch = epoch + 1  # 转换为1-based索引
        if current_epoch % save_interval == 0 or current_epoch == epochs:
            # 构建保存路径和文件名：mlp_时间戳_epX.pth
            save_path = f"/mlp_{timestamp}_ep{current_epoch}_{val_acc:.2f}.pth"
            # 保存模型状态字典（推荐方式，更灵活）
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, save_dir + save_path)
            print(f"模型已保存至: {save_path}")

    return train_loss_list, val_loss_list, val_acc_list


def plot_predictions(model, dataset, device, num_points=200, threshold=0.5):
    """
    绘制真实涨跌标签 vs 预测涨跌类别

    model: 二分类模型，输出概率
    dataset: StockDatasetBinary 类实例
    device: 训练/推理设备
    num_points: 绘制最后多少个点
    threshold: 概率阈值，大于等于该值判为1，否则为0
    """
    model.eval()
    preds = []
    reals = []

    with torch.no_grad():
        for i in range(len(dataset)):
            X, y = dataset[i]
            X = X.unsqueeze(0).to(device)  # [1, seq_len, num_features]
            prob = model(X).item()  # 概率
            pred_class = 1 if prob >= threshold else 0  # 转换为类别
            preds.append(pred_class)
            reals.append(int(y.item()))  # 确保真实标签是整数0或1

    preds = np.array(preds)
    reals = np.array(reals)

    # 取最后 num_points 个点绘制
    preds = preds[-num_points:]
    reals = reals[-num_points:]

    plt.figure(figsize=(12, 6))
    plt.plot(reals, label="真实涨跌", color='black')
    plt.plot(preds, label="预测涨跌", color='red', linestyle='--')
    plt.title("真实 vs 预测 涨跌类别")
    plt.xlabel("时间步")
    plt.ylabel("涨跌类别 (0=跌, 1=涨)")
    plt.legend()
    plt.grid(True)
    plt.show()


def analyze_loss_acc_correlation(train_loss_list, val_loss_list, val_acc_list, epoch_num=None):
    """
    分析val_acc与train_loss/val_loss的相关性（loss下降时acc是否上升）

    参数：
        train_loss_list: list，训练损失列表（长度=训练轮次epochs）
        val_loss_list: list，验证损失列表（长度=训练轮次epochs）
        val_acc_list: list，验证准确率列表（长度=训练轮次epochs）
        epoch_num: list/int，可选， epoch编号（若未传入则自动生成1,2,...len(list)）
    """
    # 1. 数据预处理：确保三个列表长度一致
    assert len(train_loss_list) == len(val_loss_list) == len(val_acc_list), \
        "三个指标列表的长度必须一致（需对应相同的训练轮次）"
    n_epochs = len(train_loss_list)
    if epoch_num is None:
        epoch_num = list(range(1, n_epochs + 1))  # 自动生成epoch编号（1,2,...n）

    # 2. 计算皮尔逊相关系数（含显著性检验p值）
    # 皮尔逊相关系数r：衡量线性相关性（-1=完全负相关，1=完全正相关）
    # p值：显著性（p<0.05说明相关性在统计上显著，非偶然）
    val_loss_acc_r, val_loss_acc_p = pearsonr(val_loss_list, val_acc_list)  # val_loss vs val_acc
    train_loss_acc_r, train_loss_acc_p = pearsonr(train_loss_list, val_acc_list)  # train_loss vs val_acc

    # 3. 打印量化分析结果
    print("=" * 80)
    print("【Loss与Acc相关性量化分析结果】")
    print("=" * 80)
    print(f"1. val_loss 与 val_acc 的相关系数：{val_loss_acc_r:.4f}")
    print(f"   显著性p值：{val_loss_acc_p:.4f} → {'相关性显著' if val_loss_acc_p < 0.05 else '相关性不显著'}")
    print(f"2. train_loss 与 val_acc 的相关系数：{train_loss_acc_r:.4f}")
    print(f"   显著性p值：{train_loss_acc_p:.4f} → {'相关性显著' if train_loss_acc_p < 0.05 else '相关性不显著'}")
    print("\n【结论解读】")
    if val_loss_acc_r < -0.7:
        print("✅ val_loss与val_acc呈强负相关：val_loss下降时val_acc明显上升，模型训练正常")
    elif -0.7 <= val_loss_acc_r < -0.3:
        print("⚠️ val_loss与val_acc呈弱负相关：val_loss下降时val_acc有上升趋势，但波动较大（可能存在过拟合风险）")
    elif -0.3 <= val_loss_acc_r < 0:
        print("❌ val_loss与val_acc呈极弱负相关：val_loss下降时val_acc几乎无同步上升（模型可能欠拟合或特征无效）")
    else:
        print("❌ 异常！val_loss与val_acc呈正相关：val_loss下降时val_acc反而下降（需检查数据标签/模型结构）")
    print("=" * 80)

    # 4. 可视化分析（3个子图：趋势图、val_loss-acc散点图、相关性热力图）
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1行3列子图

    # 子图1：双轴趋势图（train_loss/val_loss 左轴；val_acc 右轴）
    ax1 = axes[0]
    # 绘制loss趋势（左轴）
    ax1.plot(epoch_num, train_loss_list, label='train_loss', color='blue', linewidth=2, marker='o', markersize=4)
    ax1.plot(epoch_num, val_loss_list, label='val_loss', color='orange', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('训练轮次（Epoch）', fontsize=12)
    ax1.set_ylabel('损失值（Loss）', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    # 绘制acc趋势（右轴）
    ax1_twin = ax1.twinx()  # 创建共享x轴的右轴
    ax1_twin.plot(epoch_num, val_acc_list, label='val_acc', color='red', linewidth=2, marker='^', markersize=4)
    ax1_twin.set_ylabel('验证准确率（%）', fontsize=12, color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    ax1_twin.legend(loc='lower right', fontsize=10)
    ax1.set_title('Loss与Acc趋势图', fontsize=14, fontweight='bold')

    # 子图2：val_loss vs val_acc 散点图（带线性拟合线）
    ax2 = axes[1]
    ax2.scatter(val_loss_list, val_acc_list, color='green', alpha=0.6, s=50)
    # 添加线性拟合线（直观展示相关性趋势）
    z = np.polyfit(val_loss_list, val_acc_list, 1)  # 1次多项式拟合（直线）
    p = np.poly1d(z)
    x_fit = np.linspace(min(val_loss_list), max(val_loss_list), 100)
    ax2.plot(x_fit, p(x_fit), color='red', linestyle='--', linewidth=2,
             label=f'拟合线（r={val_loss_acc_r:.4f}）')
    ax2.set_xlabel('验证损失（val_loss）', fontsize=12)
    ax2.set_ylabel('验证准确率（val_acc, %）', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('val_loss与val_acc散点图', fontsize=14, fontweight='bold')

    # 子图3：三者相关性热力图
    ax3 = axes[2]
    # 构建相关系数矩阵
    correlation_matrix = np.corrcoef([train_loss_list, val_loss_list, val_acc_list])
    # 标签列表
    labels = ['train_loss', 'val_loss', 'val_acc']
    # 绘制热力图
    im = ax3.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)  # RdBu_r：红（负）蓝（正）
    # 添加数值标注
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax3.text(j, i, f'{correlation_matrix[i, j]:.4f}',
                            ha="center", va="center", color="black", fontweight='bold')
    # 设置坐标轴标签
    ax3.set_xticks(range(len(labels)))
    ax3.set_yticks(range(len(labels)))
    ax3.set_xticklabels(labels, fontsize=12)
    ax3.set_yticklabels(labels, fontsize=12)
    # 添加颜色条（解释数值含义）
    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('皮尔逊相关系数', fontsize=10)
    ax3.set_title('三者相关性热力图', fontsize=14, fontweight='bold')

    # 调整子图间距
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = read_stock_data('D:\lc\githubCode\DeepQuant\data\\000001_20180701_20250701.csv')  # 替换为你的CSV文件路径

    # 数据集类
    seq_length = 20
    # dataset = StockDataset_earning(df, seq_len=10, target='close', pred_horizon=1)
    # dataset = StockDataset_val(df, seq_len=60, target='close', pred_horizon=1)
    dataset = StockDataset_binary(df, seq_len=seq_length, target='close', pred_horizon=1)

    # 划分训练/验证
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"len(dataset)={len(dataset)}, len(train)={len(train_ds)}, len(val)={len(val_ds)}")

    # 训练模型
    model = StockMLP(seq_len=seq_length, num_features=9).to(device)
    epoch_num = 500
    train_loss, val_loss, val_acc = train_model(
        model,
        batch_size=8, epochs=epoch_num, lr=1e-4,
        device=device,
        save_interval=10,
        save_dir="D:\lc\githubCode\DeepQuant\output"
    )
    analyze_loss_acc_correlation(
        train_loss_list=train_loss,
        val_loss_list=val_loss,
        val_acc_list=val_acc,
        epoch_num=list(range(1, epoch_num + 1))  # Epoch编号（1~10）
    )
    # 绘制真实 vs 预测
    plot_predictions(model, dataset, device, num_points=200)

    # 预测例子（取最后一段数据预测下一天）
    # model.eval()
    # last_seq = torch.tensor(df[['Open', 'High', 'Low', 'Close', 'Volume']].values[-seq_length:], dtype=torch.float32)
    # pred_prob = model(last_seq.unsqueeze(0).to(device))
    # pred_class = 1 if pred_prob.item() >= 0.5 else 0
    #
    # print(f"预测上涨概率: {pred_prob.item():.4f}")
    # print(f"预测结果: {'上涨' if pred_class == 1 else '下跌'}")
