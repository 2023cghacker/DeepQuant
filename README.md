# 📈 DeepQuant

> 基于 PyTorch 的 A 股量化研究：**数据获取 → 深度学习建模 → 预测评估 → 策略模拟**


---

## ✨ 功能一览


| 模块        | 说明                                              |
| --------- | ----------------------------------------------- |
| 📥 **数据** | [Tushare](https://tushare.pro/) 拉取 A 股日线，K 线可视化 |
| 🧠 **模型** | 深度学习建模，预测价格走势与涨跌方向                              |
| 📊 **回测** | 预测指标（Accuracy / F1 等）+ 模拟交易收益                   |


---

## 🚀 快速开始

> 推荐路径：**装好环境 → 准备数据与模型 → 打开 View 界面**

### ① 安装环境

```bash
conda env create -f environment.yml
conda activate deepquant
```

### ② 准备数据

1. 复制 `config/token.txt.example` 为 `config/token.txt`，填入 [Tushare Token](https://tushare.pro/)
2. 下载行情：

```bash
python data_processor/DataDownload.py
```

> CSV 保存至 `data/` 目录。

### ③ 准备模型（首次使用）

```bash
python strategy/deeplearning/Mlp.py    # 训练模型，产物写入 output/models/
```

> 若 `output/models/` 下已有 `.pth` 权重，可跳过此步。

### ④ 启动可视化界面 🖥️

```bash
python visualization/view.py
```


| Tab         | 做什么                 |
| ----------- | ------------------- |
| 📉 **K 线**  | 选 CSV → 设日期 → 看蜡烛图  |
| 🔮 **模型预测** | 选数据 + 模型 → 看涨跌对比与指标 |
| 💰 **策略回测** | 设资金与仓位 → 看资产曲线与收益   |


界面自动扫描 `data/*.csv` 和 `output/**/*.pth`，选文件点按钮即可。

---

## 📖 使用教程

### 📁 目录说明

```
DeepQuant/
├── 📂 data/                      原始行情 CSV
│
├── 📂 data_processor/            数据下载 & K 线绘图
│
├── 📂 strategy/
│   ├── 📂 deeplearning/         建模与训练：数据集、网络、训练脚本、模型评估
│   └── 📂 backtest/             策略实操：买卖信号、资金曲线、交易回测
│
├── 📂 output/                    实验产物
│   ├── 📂 models/               训练相关：权重 .pth、超参、loss 曲线等
│   ├── 📂 backtests/            交易相关：持仓记录、成交明细、资产曲线、收益指标
│   └── 📂 reports/              其他：跨实验对比、导出图表、分析笔记等
│
├── 📂 visualization/             View 图形界面（推荐入口 ⭐）
│
└── 📂 config/                    Tushare Token
```

#### `strategy/deeplearning/` — 建模 & 训练


| 文件                     | 作用                 |
| ---------------------- | ------------------ |
| `StockDataset.py`      | 构造时序样本与标签          |
| `Mlp.py` / `Lstm.py` … | 各模型定义与训练入口         |
| `model_backtest.py`    | 加载权重，评估预测指标（非交易回测） |


#### `strategy/backtest/` — 策略 & 交易模拟


| 文件                    | 作用                 |
| --------------------- | ------------------ |
| `BacktestBase.py`     | 预测信号 → 模拟买卖 → 资产曲线 |
| `SyntheticDataset.py` | 合成数据（调试辅助）         |


#### `output/` 三层分工


| 目录               | 典型内容                                           |
| ---------------- | ---------------------------------------------- |
| `**models/**`    | checkpoint（`.pth`）、训练 config、loss/acc 曲线、验证集指标 |
| `**backtests/**` | 每日资产、买卖记录、预测 vs 实际、夏普/回撤/总收益等                  |
| `**reports/**`   | 多模型对比表、手动导出的图表、实验日志等杂项                         |


> 💡 建议每次实验在对应目录下建子文件夹（如 `models/20250620_run1/`），便于追溯。

### 🔧 命令行用法

```bash
# 📥 下载数据
python data_processor/DataDownload.py

# 📉 单独画 K 线
python data_processor/DataPlot.py

# 🧠 训练模型（deeplearning/ 下各脚本对应不同模型）
python strategy/deeplearning/Lstm.py
python strategy/deeplearning/Mlp.py

# 📊 评估 & 回测
python strategy/deeplearning/model_backtest.py   # 模型预测指标
python strategy/backtest/BacktestBase.py         # 策略交易模拟
```

> ⚠️ 运行前请修改脚本内的数据路径、模型路径等；`seq_len` 等超参数需与训练时一致。

### 🖼️ View 界面说明

`view.py` 是可视化 GUI，绘图与回测逻辑复用 `data_processor`、`strategy/deeplearning`、`strategy/backtest`，与命令行看到的是同一套结果。

---

## 📋 数据格式

CSV 需包含 Tushare 日线字段：

`trade_date` · `open` · `high` · `low` · `close` · `pre_close` · `change` · `pct_chg` · `vol` · `amount`

---

## 📄 License

Apache License 2.0

## ⚠️ 免责声明

本项目仅供学习与研究，不构成任何投资建议。股市有风险，入市需谨慎。