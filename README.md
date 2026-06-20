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
├── visualization/
│   └── view.py              # Tkinter 图形界面（薄 UI 层，复用下方模块）
├── data/                    # 行情 CSV（git 忽略）
├── output/                  # 模型权重（git 忽略）
└── config/token.txt         # Tushare Token（需自行创建）
```

## 快速开始

### 环境（推荐 Conda）

```bash
conda env create -f environment.yml
conda activate deepquant
```

依赖说明见 `environment.yml` 与 `enviornment.txt`。

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

### 4. 可视化界面

```bash
python visualization/view.py
```

`view.py` 是 **Tkinter 薄 UI 层**：只负责表单、线程调度与图表嵌入，业务逻辑和绘图均复用已有模块，不在界面里重复实现。

#### 设计理念

```
view.py（UI）
  ├── ChartPanel          # matplotlib Figure → Tk 画布
  ├── DeepQuantApp        # 三个 Tab + 参数校验 + 后台线程
  └── run_predict / run_backtest / K 线回调
        ↓ 调用已有代码
DataPlot.plot_kline              ← K 线
model_backtest.backtest_model      ← 模型推理
model_backtest.plot_backtest_result← 预测对比图
BaseBacktester + plot_results      ← 策略回测与三图展示
```

各绘图函数支持 `return_fig=True`，返回 `matplotlib.figure.Figure` 供 GUI 嵌入；默认 `return_fig=False` 时仍 `plt.show()` 弹窗，命令行脚本无需改动。

| Tab | 复用的模块 | 展示内容 |
|-----|-----------|---------|
| **K 线** | `DataPlot.read_stock_data` + `plot_kline` | 蜡烛图 + 均线 + 成交量 |
| **模型预测** | `model_backtest.backtest_model` + `plot_backtest_result` | 涨跌对比、正确性标记、Accuracy / Precision / Recall / F1 |
| **策略回测** | `BacktestBase.BaseBacktester` + `plot_results` | K 线与买卖信号、预测对比、资产曲线、收益指标 |

界面自动扫描 `data/*.csv` 与 `output/**/*.pth`；K 线 Tab 会从文件名解析日期范围（如 `000001_20180701_20250701.csv`），缺失时回退读取 CSV 实际区间。

预测与回测在后台线程执行，避免阻塞界面。`seq_len`、`threshold` 等超参数需与训练时一致。

#### 命令行可视化（无需 GUI）

各模块也可独立运行，效果与 GUI 内调用的绘图逻辑相同：

```bash
python data_processor/DataPlot.py      # K 线
python strategy/model_backtest.py      # 模型预测回测
python tests/BacktestBase.py           # 策略交易回测
```

运行前请修改各脚本中的数据路径、模型路径及 `seq_len` 等超参数。

## 数据格式

CSV 需包含 Tushare 日线字段：`trade_date`, `open`, `high`, `low`, `close`, `pre_close`, `change`, `pct_chg`, `vol`, `amount`。

## License

Apache License 2.0

## 免责声明

本项目仅供学习与研究，不构成任何投资建议。股市有风险，入市需谨慎。
