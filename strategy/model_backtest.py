import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # 多指标评估
from data_processor.DataPlot import read_stock_data  # 复用你现有的数据读取函数
from strategy.StockDataset import StockDataset_binary
from strategy.Mlp import StockMLP


# ----------------------
# 核心回测函数
# ----------------------
def load_pretrained_model(model_path, seq_len, num_features, device):
    """
    加载预训练模型权重
    参数：
    model_path: 模型文件路径（如"D:\lc\githubCode\DeepQuant\output\mlp_10251430_ep500_82.50.pth"）
    seq_len: 输入序列长度（需与训练时一致）
    num_features: 特征数（固定为5，与训练一致）
    device: 运行设备（cpu/cuda）
    返回：加载权重后的模型
    """
    # 初始化模型（结构与训练一致）
    model = StockMLP(seq_len=seq_len, num_features=num_features).to(device)
    # 加载模型字典（处理CPU/GPU不匹配问题）
    checkpoint = torch.load(model_path, map_location=device)
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    # 切换为评估模式（禁用Dropout/BatchNorm更新）
    model.eval()
    print(f"✅ 成功加载模型：{model_path}")
    print(f"   模型训练时的Epoch：{checkpoint['epoch']}")
    print(f"   模型训练时的Val Acc：{checkpoint['val_acc']:.2f}%")
    return model


def backtest_model(model, test_dataset, device, threshold=0.5):
    """
    回测模型在新数据上的表现
    参数：
    model: 加载好的预训练模型
    test_dataset: 新数据的Dataset实例
    device: 运行设备
    threshold: 涨跌判断阈值（默认0.5）
    返回：预测概率、预测标签、真实标签、评估指标字典
    """
    model.eval()  # 评估模式
    pred_probs = []  # 预测概率
    pred_labels = []  # 预测标签（0/1）
    true_labels = []  # 真实标签（0/1）

    # 无梯度推理（加速且避免修改模型）
    with torch.no_grad():
        for x, y in test_dataset:
            # 增加batch维度（模型输入需为[batch, seq_len, num_features]）
            x = x.unsqueeze(0).to(device)
            # 预测概率
            prob = model(x).item()
            # 转换为标签
            pred_label = 1 if prob >= threshold else 0
            # 收集结果
            pred_probs.append(prob)
            pred_labels.append(pred_label)
            true_labels.append(int(y.item()))  # 真实标签转整数

    # 计算多维度评估指标（sklearn）
    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels) * 100,  # 准确率（%）
        "precision": precision_score(true_labels, pred_labels, zero_division=0) * 100,  # 精确率（%）
        "recall": recall_score(true_labels, pred_labels, zero_division=0) * 100,  # 召回率（%）
        "f1_score": f1_score(true_labels, pred_labels, zero_division=0) * 100  # F1值（%）
    }

    return np.array(pred_probs), np.array(pred_labels), np.array(true_labels), metrics


def plot_backtest_result(true_labels, pred_labels, num_points=300, return_fig=False):
    """
    绘制回测结果对比图（真实涨跌 vs 预测涨跌）
    参数：
    true_labels: 真实标签数组
    pred_labels: 预测标签数组
    num_points: 绘制最后N个点（避免图过于拥挤）
    return_fig: 为 True 时返回 Figure 对象（供 GUI 嵌入），否则弹窗显示
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
    plt.rcParams['axes.unicode_minus'] = False  # 负号显示

    # 取最后num_points个点（若数据不足则取全部）
    plot_true = true_labels[-num_points:] if len(true_labels) > num_points else true_labels
    plot_pred = pred_labels[-num_points:] if len(pred_labels) > num_points else pred_labels
    time_steps = range(len(plot_true))  # 时间步

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 子图1：真实涨跌 vs 预测涨跌（时序对比）
    ax1.plot(time_steps, plot_true, label='真实涨跌', color='black', linewidth=2)
    ax1.plot(time_steps, plot_pred, label='预测涨跌', color='red', linestyle='--', linewidth=2)
    ax1.set_ylabel('涨跌标签（0=跌，1=涨）', fontsize=12)
    ax1.set_title('回测：真实涨跌 vs 预测涨跌', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.2, 1.2)  # 固定y轴范围，避免波动过大

    # 子图2：预测正确性标记（绿色=正确，红色=错误）
    correct = (plot_true == plot_pred).astype(int)
    incorrect = (plot_true != plot_pred).astype(int)
    # 正确预测：用绿色点标记
    ax2.scatter(
        [t for t, c in zip(time_steps, correct) if c == 1],
        [1] * sum(correct),
        color='green', label='预测正确', s=20, alpha=0.7
    )
    # 错误预测：用红色点标记
    ax2.scatter(
        [t for t, i in zip(time_steps, incorrect) if i == 1],
        [0] * sum(incorrect),
        color='red', label='预测错误', s=20, alpha=0.7
    )
    ax2.set_xlabel('时间步', fontsize=12)
    ax2.set_ylabel('预测结果', fontsize=12)
    ax2.set_title(f'预测正确性（正确数：{sum(correct)}，错误数：{sum(incorrect)}）', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['错误', '正确'])

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()


# ----------------------
# 3. 回测主逻辑（可直接运行）
# ----------------------
if __name__ == "__main__":
    # ----------------------
    # 配置回测参数（需根据你的实际情况修改）
    # ----------------------
    NEW_CSV_PATH = "/data/000001_20250719_20250912.csv"  # 新数据CSV路径（如8月数据）
    MODEL_PATH = "/output/09151514/mlp_09151514_ep260_55.95.pth"  # 预训练模型路径
    SEQ_LEN = 20  # 输入序列长度（必须与训练时一致！）
    THRESHOLD = 0.5  # 涨跌判断阈值（默认0.5，可根据需求调整）

    # ----------------------
    # 步骤1：读取并预处理新数据
    # ----------------------
    print(f"📊 读取新数据：{NEW_CSV_PATH}")
    df_new = read_stock_data(NEW_CSV_PATH)
    print(f"新数据总行数：{len(df_new)}")

    # ----------------------
    # 步骤2：创建新数据的Dataset
    # ----------------------
    print(f"\n📦 创建新数据数据集（序列长度：{SEQ_LEN}）")
    test_dataset = StockDataset_binary(
        df=df_new,
        seq_len=SEQ_LEN,
        target='close',
        pred_horizon=1
    )
    print(f"测试集样本数：{len(test_dataset)}")

    # ----------------------
    # 步骤3：初始化设备并加载模型
    # ----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💻 使用设备：{device}")
    model = load_pretrained_model(
        model_path=MODEL_PATH,
        seq_len=SEQ_LEN,
        num_features=9,  # 固定为5个特征（Open/High/Low/Close/Volume）
        device=device
    )

    # ----------------------
    # 步骤4：执行回测并计算指标
    # ----------------------
    print(f"\n🚀 开始回测（阈值：{THRESHOLD}）")
    pred_probs, pred_labels, true_labels, metrics = backtest_model(
        model=model,
        test_dataset=test_dataset,
        device=device,
        threshold=THRESHOLD
    )

    # 打印回测结果
    print("\n" + "=" * 60)
    print("📈 回测结果汇总")
    print("=" * 60)
    print(f"准确率（Accuracy）：{metrics['accuracy']:.2f}%")
    print(f"精确率（Precision）：{metrics['precision']:.2f}%")  # 预测为涨时，实际涨的概率
    print(f"召回率（Recall）：{metrics['recall']:.2f}%")  # 实际为涨时，被预测对的概率
    print(f"F1值（F1-Score）：{metrics['f1_score']:.2f}%")  # 精确率和召回率的调和平均
    print(f"预测样本总数：{len(true_labels)}")
    print(f"正确预测数：{sum(pred_labels == true_labels)}")
    print(f"错误预测数：{sum(pred_labels != true_labels)}")
    print("=" * 60)

    # ----------------------
    # 步骤5：绘制回测可视化图
    # ----------------------
    print(f"\n📊 绘制回测结果图（显示最后300个点）")
    plot_backtest_result(
        true_labels=true_labels,
        pred_labels=pred_labels,
        num_points=300  # 可调整显示的点数
    )

    # ----------------------
    # 步骤6：（可选）保存回测结果到CSV
    # ----------------------
    save_result = input("\n是否保存回测结果到CSV？（y/n）：")
    if save_result.lower() == 'y':
        result_df = pd.DataFrame({
            "时间步": range(len(true_labels)),
            "真实标签": true_labels,
            "预测标签": pred_labels,
            "预测上涨概率": pred_probs.round(4),
            "预测是否正确": (pred_labels == true_labels).astype(int)
        })
        # 保存路径（与模型同目录）
        save_path = MODEL_PATH.replace(".pth", "_backtest_result.csv")
        result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"✅ 回测结果已保存到：{save_path}")
