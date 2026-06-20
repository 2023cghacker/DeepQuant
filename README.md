# 📈 DeepQuant

> 基于 PyTorch 的 A 股量化研究：**数据获取 → 深度学习建模 → 预测回测 → 策略模拟**

```
  Tushare 行情          训练 MLP / LSTM         View 可视化
      │                      │                      │
      ▼                      ▼                      ▼
  data/*.csv  ──────►  output/*.pth  ──────►  K线 / 预测 / 回测
```

---

## ✨ 功能一览

| 模块 | 说明 |
|------|------|
| 📥 **数据** | [Tushare](https://tushare.pro/) 拉取 A 股日线，K 线可视化 |
| 🧠 **模型** | `StockLSTM` 预测收益 · `StockMLP` 二分类预测涨跌 |
| 📊 **回测** | Accuracy / Precision / Recall / F1 + 模拟交易收益 |

---

## 🚀 快速开始

> 推荐路径：**装好环境 → 准备数据与模型 → 打开 View 界面**，无需记命令行。

### ① 安装环境

```bash
conda env create -f environment.yml
conda activate deepquant
```

### ② 准备数据

1. 在 `config/token.txt` 写入 [Tushare Token](https://tushare.pro/)
2. 下载行情：

```bash
python data_processor/DataDownload.py
```

> CSV 会保存到 `data/` 目录。

### ③ 准备模型（首次使用）

```bash
python strategy/Mlp.py    # 训练 MLP，权重输出到 output/
```

> 若 `output/` 里已有 `.pth` 文件，可跳过此步。

### ④ 启动可视化界面 🖥️

```bash
python visualization/view.py
```

| Tab | 做什么 |
|-----|--------|
| 📉 **K 线** | 选 CSV → 设日期 → 看蜡烛图 |
| 🔮 **模型预测** | 选数据 + 模型 → 看涨跌对比与指标 |
| 💰 **策略回测** | 设资金与仓位 → 看资产曲线与收益 |

界面自动扫描 `data/*.csv` 和 `output/**/*.pth`，选文件点按钮即可。

---

## 📖 使用教程

### 📁 目录说明

```
DeepQuant/
├── 📂 data_processor/     数据下载 & K 线绘图
├── 📂 strategy/           数据集、LSTM/MLP 训练、模型回测
├── 📂 tests/              策略交易回测（买卖信号 + 资产曲线）
├── 📂 visualization/      View 图形界面（推荐入口 ⭐）
├── 📂 data/               行情 CSV（git 忽略）
├── 📂 output/             模型权重 .pth（git 忽略）
└── 📂 config/             Tushare Token
```

### 🔧 命令行用法

各脚本可独立运行，适合改参数、批量实验：

```bash
# 📥 下载数据
python data_processor/DataDownload.py

# 🧠 训练模型
python strategy/Lstm.py          # LSTM：预测收益率
python strategy/Mlp.py           # MLP：预测涨跌（默认）

# 📊 回测
python strategy/model_backtest.py   # 模型指标回测
python tests/BacktestBase.py        # 策略交易回测

# 📉 单独画 K 线（无需 GUI）
python data_processor/DataPlot.py
```

> ⚠️ 运行前请修改脚本内的数据路径、模型路径，`seq_len` 等超参数需与训练时一致。

### 🖼️ View 界面说明

`view.py` 是薄 UI 层，绘图与回测逻辑复用 `DataPlot`、`model_backtest`、`BacktestBase`，GUI 与命令行看到的是同一套结果。

---

## 📋 数据格式

CSV 需包含 Tushare 日线字段：

`trade_date` · `open` · `high` · `low` · `close` · `pre_close` · `change` · `pct_chg` · `vol` · `amount`

---

## 📄 License

Apache License 2.0

## ⚠️ 免责声明

本项目仅供学习与研究，不构成任何投资建议。股市有风险，入市需谨慎。
