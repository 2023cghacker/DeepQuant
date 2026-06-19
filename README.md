# DeepQuant

基于 PyTorch 的 A 股量化研究项目：数据获取 → 深度学习建模 → 预测回测 → 策略模拟。

## 功能

- **数据**：通过 [Tushare](https://tushare.pro/) 拉取日线行情，支持 K 线可视化
- **模型**：
  - `StockLSTM`：双向 LSTM，预测收盘价或收益率
  - `StockMLP`：MLP 二分类，预测次日涨跌
- **回测**：分类指标（Accuracy / Precision / Recall / F1）+ 模拟交易收益分析

## 项目结构

```
DeepQuant/
├── data_processor/
│   ├── DataDownload.py    # Tushare 数据下载
│   └── DataPlot.py          # K 线图绘制
├── strategy/
│   ├── StockDataset.py      # 数据集（价格 / 收益 / 涨跌分类）
│   ├── Lstm.py              # LSTM 训练与预测
│   ├── Mlp.py               # MLP 训练与评估
│   └── model_backtest.py    # 模型在新数据上的回测
├── tests/
│   ├── BacktestBase.py      # 策略回测（买卖信号 + 资产曲线）
│   └── SyntheticDataset.py  # 合成时序数据（调试用）
├── data/                    # 行情 CSV（git 忽略）
├── output/                  # 模型权重（git 忽略）
└── config/token.txt         # Tushare Token（需自行创建）
```

## 快速开始

### 依赖

```bash
pip install torch pandas numpy matplotlib mplfinance tushare scikit-learn seaborn scipy
```

### 1. 配置 Token

在 `config/token.txt` 中写入 Tushare API Token，然后下载数据：

```bash
python data_processor/DataDownload.py
```

### 2. 训练模型

```bash
# LSTM：预测收益率
python strategy/Lstm.py

# MLP：预测涨跌（默认）
python strategy/Mlp.py
```

### 3. 回测

```bash
# 模型指标回测
python strategy/model_backtest.py

# 策略交易回测（含收益曲线）
python tests/BacktestBase.py
```

运行前请修改各脚本中的数据路径、模型路径及 `seq_len` 等超参数，需与训练时保持一致。

## 数据格式

CSV 需包含 Tushare 日线字段：`trade_date`, `open`, `high`, `low`, `close`, `pre_close`, `change`, `pct_chg`, `vol`, `amount`。

## License

Apache License 2.0

## 免责声明

本项目仅供学习与研究，不构成任何投资建议。股市有风险，入市需谨慎。
