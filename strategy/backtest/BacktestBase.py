import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.dates as mdates

from data_processor.DataPlot import read_stock_data
from strategy.deeplearning.StockDataset import StockDataset_binary
from strategy.deeplearning.Mlp import StockMLP


class BaseBacktester:
    """股票策略回测器：使用前N天数据预测次日涨跌，在次日开盘时进行交易"""

    def __init__(self,
                 seq_len=20, threshold=0.5,
                 initial_capital=100000.00, trade_amount_ratio=0.5):
        """
        初始化回测器
        :param seq_len: 输入序列长度（使用前N天数据）
        :param threshold: 预测阈值
        :param initial_capital: 初始资金
        :param trade_amount_ratio: 每次交易的资金比例（0-1之间）
        """
        self.seq_len = seq_len
        self.threshold = threshold
        self.initial_capital = initial_capital
        self.trade_amount_ratio = trade_amount_ratio  # 每次交易使用的资金比例
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化变量
        self.model_path = None
        self.model = None
        self.data = None
        self.test_dataset = None
        self.predictions = None  # 存储预测结果的DataFrame
        self.portfolio = None  # 存储交易记录和资产变化

    def load_data(self, data_path):
        """加载并预处理数据"""
        print(f"📊 读取数据：{data_path}")
        self.data = read_stock_data(data_path)
        print("数据列名:", self.data.columns.tolist())
        print(f"数据总行数：{len(self.data)}")

        # 确保数据包含必要的列
        required_columns = ['open', 'close', 'trade_date']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"数据中缺少必要的列：{col}")

        return self

    def load_model(self, model_path, num_features=9):
        """加载预训练模型"""
        print(f"\n💻 使用设备：{self.device}")
        # 初始化模型
        self.model_path = model_path
        self.model = StockMLP(seq_len=self.seq_len, num_features=num_features).to(self.device)

        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # 设置为评估模式

        print(f"✅ 成功加载模型：{model_path}")
        print(f"   模型训练时的Epoch：{checkpoint['epoch']}")
        print(f"   模型训练时的Val Acc：{checkpoint['val_acc']:.2f}%")

        return self

    def generate_predictions(self):
        """生成每日预测：使用当天及之前共seq_len天的数据预测次日涨跌"""
        if self.model is None or self.data is None:
            raise ValueError("请先加载模型和数据")

        print(f"\n🚀 生成每日预测（使用前{self.seq_len}天数据预测次日涨跌）")

        # 创建测试数据集
        self.test_dataset = StockDataset_binary(
            df=self.data,
            seq_len=self.seq_len,
            target='close',
            pred_horizon=1
        )
        print(f"测试集样本数：{len(self.test_dataset)}")

        # 存储预测结果
        pred_probs = []
        pred_labels = []
        true_labels = []
        prediction_dates = []  # 预测日期（基于该日数据进行预测）
        target_dates = []  # 目标日期（预测的是该日的涨跌）

        # 无梯度推理
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_dataset):
                # 增加batch维度
                x = x.unsqueeze(0).to(self.device)
                # 预测概率
                prob = self.model(x).item()
                # 转换为标签
                pred_label = 1 if prob >= self.threshold else 0

                # 收集结果
                pred_probs.append(prob)
                pred_labels.append(pred_label)
                true_labels.append(int(y.item()))

                # 记录日期：预测基于第i+seq_len-1天的数据，预测的是第i+seq_len天的涨跌
                prediction_date_idx = i + self.seq_len - 1
                target_date_idx = i + self.seq_len

                prediction_dates.append(self.data.iloc[prediction_date_idx]['trade_date'])
                target_dates.append(self.data.iloc[target_date_idx]['trade_date'])

        # 创建预测结果DataFrame
        self.predictions = pd.DataFrame({
            'prediction_date': prediction_dates,  # 基于该日数据进行预测
            'target_date': target_dates,  # 预测的是该日的涨跌
            'pred_prob': pred_probs,
            'pred_label': pred_labels,
            'true_label': true_labels
        })

        return self

    def calculate_prediction_metrics(self):
        """计算预测性能指标"""
        if self.predictions is None:
            raise ValueError("请先生成预测结果")

        metrics = {
            "accuracy": accuracy_score(
                self.predictions['true_label'],
                self.predictions['pred_label']
            ) * 100,
            "precision": precision_score(
                self.predictions['true_label'],
                self.predictions['pred_label'],
                zero_division=0
            ) * 100,
            "recall": recall_score(
                self.predictions['true_label'],
                self.predictions['pred_label'],
                zero_division=0
            ) * 100,
            "f1_score": f1_score(
                self.predictions['true_label'],
                self.predictions['pred_label'],
                zero_division=0
            ) * 100
        }
        return metrics

    def run_backtest(self):
        """执行回测：根据每日预测结果在次日开盘时进行交易"""
        if self.predictions is None:
            raise ValueError("请先生成预测结果")

        print("\n📈 执行回测...")

        # 初始化交易记录
        portfolio = []
        cash = self.initial_capital  # 现金
        shares = 0  # 持股数量
        total_assets = self.initial_capital  # 总资产 = 现金 + 持股价值

        # 按目标日期排序
        sorted_predictions = self.predictions.sort_values('target_date').reset_index(drop=True)

        # 遍历每个预测结果，执行交易
        for i, row in sorted_predictions.iterrows():
            target_date = row['target_date']
            pred_label = row['pred_label']

            # 找到目标日期的开盘价（用于交易）和收盘价（用于计算资产）
            try:
                date_data = self.data[self.data['trade_date'] == target_date].iloc[0]
                open_price = date_data['open']
                close_price = date_data['close']
                low_price = date_data['low']
                high_price = date_data['high']
            except IndexError:
                print(f"警告：无法找到日期 {target_date} 的交易数据，跳过该日交易")
                continue

            # 根据预测结果执行交易
            if pred_label == 1:
                # 预测上涨：买入
                # 计算可买入金额（总资产的一定比例）
                buy_amount = total_assets * self.trade_amount_ratio
                # 计算可买入的股数（向下取整）
                buy_shares = int(buy_amount / open_price)

                if buy_shares > 0 and cash >= buy_shares * open_price:
                    # 执行买入
                    cash -= buy_shares * open_price
                    shares += buy_shares
                    action = f"买入 {buy_shares} 股"
                else:
                    action = "资金不足，未买入"
            else:
                # 预测下跌：卖出
                # 计算可卖出的股数（总资产的一定比例）
                sell_shares = int(shares * self.trade_amount_ratio)

                if sell_shares > 0:
                    # 执行卖出
                    cash += sell_shares * open_price
                    shares -= sell_shares
                    action = f"卖出 {sell_shares} 股"
                else:
                    action = "无持股，未卖出"

            # 计算当日总资产（按收盘价计算持股价值）
            total_assets = cash + shares * close_price

            # 记录交易
            portfolio.append({
                'date': target_date,
                'open_price': open_price,
                'close_price': close_price,
                'low_price': low_price,
                'high_price': high_price,
                'pred_label': pred_label,
                'true_label': row['true_label'],
                'cash': cash,
                'shares': shares,
                'asset_value': shares * close_price,
                'total_assets': total_assets,
                'action': action
            })

        # 创建portfolio DataFrame
        self.portfolio = pd.DataFrame(portfolio)

        # 计算每日收益率
        if not self.portfolio.empty:
            self.portfolio['daily_return'] = self.portfolio['total_assets'].pct_change() * 100
            self.portfolio['cumulative_return'] = (1 + self.portfolio['daily_return'] / 100).cumprod() * 100 - 100

        return self

    def print_performance(self):
        """打印回测性能指标"""
        if self.portfolio is None or self.portfolio.empty:
            raise ValueError("请先执行回测")

        # 计算预测指标
        pred_metrics = self.calculate_prediction_metrics()

        print("\n" + "=" * 60)
        print("📊 预测性能指标")
        print("=" * 60)
        print(f"准确率（Accuracy）：{pred_metrics['accuracy']:.2f}%")
        print(f"精确率（Precision）：{pred_metrics['precision']:.2f}%")
        print(f"召回率（Recall）：{pred_metrics['recall']:.2f}%")
        print(f"F1值（F1-Score）：{pred_metrics['f1_score']:.2f}%")
        print(f"预测样本总数：{len(self.predictions)}")
        print(f"正确预测数：{sum(self.predictions['true_label'] == self.predictions['pred_label'])}")
        print(f"错误预测数：{sum(self.predictions['true_label'] != self.predictions['pred_label'])}")

        print("\n" + "=" * 60)
        print("💹 策略收益表现")
        print("=" * 60)
        print(f"初始资金：{self.initial_capital:.2f}元, 单次交易资金比例：{self.trade_amount_ratio:.0%}")
        # 计算交易次数
        buy_count = sum(1 for action in self.portfolio['action'] if '买入' in action and '未买入' not in action)
        sell_count = sum(1 for action in self.portfolio['action'] if '卖出' in action and '未卖出' not in action)
        print(f"总买入次数：{buy_count}, 总卖出次数：{sell_count}")

        final_assets = self.portfolio['total_assets'].iloc[-1]
        print(f"最终总资产：{final_assets:.2f}元")

        total_return = ((final_assets / self.initial_capital) - 1) * 100
        print(f"总收益率：{total_return:.2f}%")

        # 计算年化收益率
        # 假设portfolio有'date'列存储交易日期
        if 'date' in self.portfolio.columns:
            # 转换为日期类型
            start_date = pd.to_datetime(self.portfolio['date'].iloc[0])
            end_date = pd.to_datetime(self.portfolio['date'].iloc[-1])

            # 计算回测天数
            days = (end_date - start_date).days

            if days > 0:
                # 计算年化收益率 (复利计算)
                total_return_rate = final_assets / self.initial_capital
                annualized_return = (total_return_rate ** (365 / days) - 1) * 100
                print(f"回测周期：{days}天 ({start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')})")
                print(f"年化收益率：{annualized_return:.2f}%")
            else:
                print("回测周期不足一天，无法计算年化收益率")
        else:
            print("portfolio中缺少'date'列，无法计算年化收益率")

        print("=" * 60)

    def plot_results(self, num_points=None, return_fig=False):
        """绘制回测结果可视化图表，其中价格部分使用K线图展示"""
        if self.portfolio is None or self.portfolio.empty:
            raise ValueError("请先执行回测")

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
        plt.rcParams['axes.unicode_minus'] = False  # 负号显示

        # 准备绘图数据
        plot_data = self.portfolio.copy()
        if num_points and len(plot_data) > num_points:
            plot_data = plot_data.iloc[-num_points:]

        plot_data['date'] = pd.to_datetime(plot_data['date'])

        # 创建一个包含3个子图的图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
        for ax in [ax1, ax2, ax3]:
            ax.minorticks_on()  # 开启次刻度
            ax.grid(which='both', linestyle='--', linewidth=0.5)  # 主/次刻度都画

        # 子图1：K线图和交易信号
        # 上涨时用红色，下跌时用绿色
        up = plot_data[plot_data.close_price >= plot_data.open_price]
        down = plot_data[plot_data.close_price < plot_data.open_price]

        # 绘制K线实体
        col1 = 'red'  # 上涨颜色
        col2 = 'green'  # 下跌颜色

        # 绘制蜡烛实体
        ax1.bar(up['date'], up['close_price'] - up['open_price'],
                bottom=up['open_price'], width=0.6, color=col1)
        ax1.bar(down['date'], down['close_price'] - down['open_price'],
                bottom=down['open_price'], width=0.6, color=col2)

        # 绘制高低点连线
        ax1.vlines(up['date'], up['low_price'], up['high_price'], color=col1, linewidth=1)
        ax1.vlines(down['date'], down['low_price'], down['high_price'], color=col2, linewidth=1)

        # 标记买入信号
        buy_signals = plot_data[plot_data['action'].str.contains('买入') & ~plot_data['action'].str.contains('未买入')]
        ax1.scatter(buy_signals['date'], buy_signals['open_price'],
                    marker='^', color='purple', label='买入', s=100, zorder=3)

        # 标记卖出信号
        sell_signals = plot_data[plot_data['action'].str.contains('卖出') & ~plot_data['action'].str.contains('未卖出')]
        ax1.scatter(sell_signals['date'], sell_signals['open_price'],
                    marker='v', color='orange', label='卖出', s=100, zorder=3)

        ax1.set_ylabel('价格', fontsize=12)
        ax1.set_title('K线图与交易信号', fontsize=14, fontweight='bold')
        ax1.legend()

        # 子图2：预测与实际对比
        ax2.plot(plot_data['date'], plot_data['true_label'], label='实际涨跌',
                 color='black', linewidth=2)
        ax2.plot(plot_data['date'], plot_data['pred_label'], label='预测涨跌',
                 color='red', linestyle='--', linewidth=2)
        ax2.set_ylabel('涨跌标签（0=跌，1=涨）', fontsize=12)
        ax2.set_title('实际涨跌与预测涨跌对比', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.set_ylim(-0.2, 1.2)

        # 子图3：资产变化
        ax3.plot(plot_data['date'], plot_data['total_assets'],
                 label='总资产', color='green', linewidth=2)
        ax3.axhline(y=self.initial_capital, color='gray', linestyle='--',
                    label=f'初始资金 ({self.initial_capital:.2f}元)')
        ax3.set_xlabel('日期', fontsize=12)
        ax3.set_ylabel('资产价值 (元)', fontsize=12)
        ax3.set_title('资产变化曲线', fontsize=14, fontweight='bold')
        ax3.legend()

        # 设置x轴日期格式
        plt.gcf().autofmt_xdate()
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        plt.tight_layout()
        if return_fig:
            return fig
        plt.show()

    def save_results(self):
        """保存回测结果到CSV文件"""
        if self.portfolio is None or self.portfolio.empty:
            raise ValueError("请先执行回测")

        # 生成保存路径
        model_dir = os.path.dirname(self.model_path)
        model_name = os.path.splitext(os.path.basename(self.model_path))[0]
        save_path = os.path.join(model_dir, f"{model_name}_strategy_results.csv")

        # 保存结果
        self.portfolio.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"✅ 回测结果已保存到：{save_path}")

        return save_path


if __name__ == "__main__":
    # 配置参数
    DATA_PATH = "D:\\lc\\githubCode\\DeepQuant\\data\\000001_20250719_20250912.csv"  # 数据路径
    MODEL_PATH = "D:\\lc\\githubCode\\DeepQuant\\output\\09151514\\mlp_09151514_ep260_55.95.pth"  # 模型路径
    SEQ_LEN = 20  # 使用前20天数据进行预测
    THRESHOLD = 0.5  # 预测阈值
    INITIAL_CAPITAL = 10000.00  # 初始资金
    TRADE_AMOUNT_RATIO = 0.7  # 每次交易使用多少可用资金/持股

    # 创建回测器实例并执行回测流程
    backtester = BaseBacktester(
        seq_len=SEQ_LEN,
        threshold=THRESHOLD,
        initial_capital=INITIAL_CAPITAL,
        trade_amount_ratio=TRADE_AMOUNT_RATIO
    )

    # 执行完整回测流程
    backtester.load_data(DATA_PATH) \
        .load_model(MODEL_PATH, num_features=9) \
        .generate_predictions() \
        .run_backtest()

    # 输出性能指标
    backtester.print_performance()

    # 绘制结果图表
    print("\n📊 绘制回测结果可视化图表")
    backtester.plot_results()  # 可指定num_points参数限制显示点数，如num_points=300

    # 询问是否保存结果
    save_result = input("\n是否保存回测结果到CSV？（y/n）：")
    if save_result.lower() == 'y':
        backtester.save_results()
