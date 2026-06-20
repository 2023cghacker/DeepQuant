"""
DeepQuant 可视化界面

直接运行:
  python visualization/view.py
"""

import re
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"


def list_csv_files():
    if not DATA_DIR.exists():
        return []
    return sorted(f.name for f in DATA_DIR.glob("*.csv"))


def _format_yyyymmdd(s: str) -> str:
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def parse_dates_from_filename(filename: str) -> tuple[str | None, str | None]:
    """从文件名解析起止日期，如 000001_20180701_20250701.csv"""
    stem = Path(filename).stem
    dates_8 = re.findall(r"(\d{8})", stem)
    if len(dates_8) >= 2:
        return _format_yyyymmdd(dates_8[0]), _format_yyyymmdd(dates_8[-1])
    if len(dates_8) == 1:
        return _format_yyyymmdd(dates_8[0]), None
    dates_6 = re.findall(r"(?<!\d)(\d{6})(?!\d)", stem)
    if dates_6:
        d = dates_6[0]
        return f"{d[:4]}-{d[4:6]}-01", None
    return None, None


def resolve_csv_date_range(csv_name: str) -> tuple[str, str]:
    """优先从文件名取日期，缺失时读 CSV 实际范围"""
    start, end = parse_dates_from_filename(csv_name)
    if start and end:
        return start, end
    from data_processor.DataPlot import read_stock_data

    df = read_stock_data(str(DATA_DIR / csv_name))
    return start or df.index[0].strftime("%Y-%m-%d"), end or df.index[-1].strftime(
        "%Y-%m-%d"
    )


def list_model_files():
    if not OUTPUT_DIR.exists():
        return []
    return sorted(
        str(f.relative_to(OUTPUT_DIR)).replace("\\", "/")
        for f in OUTPUT_DIR.rglob("*.pth")
    )


def run_predict(csv_name, model_name, seq_len, threshold):
    import torch
    from data_processor.DataPlot import read_stock_data
    from strategy.deeplearning.StockDataset import StockDataset_binary
    from strategy.backtest.model_backtest import (
        backtest_model,
        load_pretrained_model,
        plot_backtest_result,
    )

    csv_path = str(DATA_DIR / csv_name)
    model_path = str(OUTPUT_DIR / model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = read_stock_data(csv_path)
    dataset = StockDataset_binary(df, seq_len=seq_len, target="close", pred_horizon=1)
    model = load_pretrained_model(model_path, seq_len, 9, device)
    _, pred_labels, true_labels, metrics = backtest_model(
        model, dataset, device, threshold
    )

    fig = plot_backtest_result(
        true_labels,
        pred_labels,
        num_points=200,
        return_fig=True,
    )

    text = (
        f"准确率: {metrics['accuracy']:.2f}%   "
        f"精确率: {metrics['precision']:.2f}%   "
        f"召回率: {metrics['recall']:.2f}%   "
        f"F1: {metrics['f1_score']:.2f}%"
    )
    return fig, text


def run_backtest(csv_name, model_name, seq_len, threshold, capital, ratio):
    from strategy.backtest.BacktestBase import BaseBacktester

    bt = BaseBacktester(
        seq_len=seq_len,
        threshold=threshold,
        initial_capital=capital,
        trade_amount_ratio=ratio,
    )
    bt.load_data(str(DATA_DIR / csv_name))
    bt.load_model(str(OUTPUT_DIR / model_name), num_features=9)
    bt.generate_predictions().run_backtest()

    portfolio = bt.portfolio
    if portfolio is None or portfolio.empty:
        raise ValueError("回测未产生有效记录")

    pred = bt.calculate_prediction_metrics()
    final_assets = float(portfolio["total_assets"].iloc[-1])
    total_return = (final_assets / capital - 1) * 100

    fig = bt.plot_results(return_fig=True)

    text = (
        f"准确率: {pred['accuracy']:.2f}%   F1: {pred['f1_score']:.2f}%   "
        f"总收益: {total_return:.2f}%   最终资产: {final_assets:.2f}"
    )
    return fig, text


class PlaceholderEntry(tk.Entry):
    """带灰色占位提示的输入框（获焦清空，失焦且为空时恢复）"""

    def __init__(self, master, placeholder, **kwargs):
        super().__init__(master, **kwargs)
        self.placeholder = placeholder
        self._is_placeholder = False
        self._default_fg = kwargs.get("fg", "black")
        self._ph_fg = "grey"
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)
        self._show_placeholder()

    def _show_placeholder(self):
        self.delete(0, tk.END)
        self.insert(0, self.placeholder)
        self.config(fg=self._ph_fg)
        self._is_placeholder = True

    def _on_focus_in(self, _event=None):
        if self._is_placeholder:
            self.delete(0, tk.END)
            self.config(fg=self._default_fg)
            self._is_placeholder = False

    def _on_focus_out(self, _event=None):
        if not self.get().strip():
            self._show_placeholder()

    def get_value(self) -> str:
        if self._is_placeholder:
            return ""
        return self.get().strip()

    def set_value(self, value: str):
        self.delete(0, tk.END)
        if value:
            self.config(fg=self._default_fg)
            self._is_placeholder = False
            self.insert(0, value)
        else:
            self._show_placeholder()


class ChartPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self._fig = None

    def show_figure(self, fig):
        if self._fig is not None:
            plt.close(self._fig)
        self._fig = fig
        for w in self.canvas_frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


class DeepQuantApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DeepQuant")
        self.geometry("1000x700")
        self.minsize(800, 600)

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._build_kline_tab(notebook)
        self._build_predict_tab(notebook)
        self._build_backtest_tab(notebook)

        self.status = ttk.Label(self, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    def _row(self, parent, label, widget, col=0):
        ttk.Label(parent, text=label).grid(
            row=0, column=col * 2, sticky=tk.W, padx=(0, 4)
        )
        widget.grid(row=0, column=col * 2 + 1, sticky=tk.W, padx=(0, 12))

    def _fill_combo(self, combo, items):
        combo["values"] = items
        if items:
            combo.current(0)

    def _build_kline_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="K 线")

        bar = ttk.Frame(tab)
        bar.pack(fill=tk.X, pady=(0, 8))

        self.kline_csv = ttk.Combobox(bar, width=36, state="readonly")
        self.kline_start = PlaceholderEntry(bar, width=12, placeholder="2018-07-01")
        self.kline_end = PlaceholderEntry(bar, width=12, placeholder="2018-07-01")
        self.kline_btn = ttk.Button(bar, text="绘制 K 线", command=self._on_kline)

        self._row(bar, "数据文件", self.kline_csv, 0)
        self._row(bar, "开始", self.kline_start, 1)
        self._row(bar, "结束", self.kline_end, 2)
        self.kline_btn.grid(row=0, column=6, padx=4)

        self._fill_combo(self.kline_csv, list_csv_files())
        self.kline_csv.bind("<<ComboboxSelected>>", self._on_kline_csv_changed)
        if self.kline_csv.get():
            self._fill_kline_dates(self.kline_csv.get())
        self.kline_chart = ChartPanel(tab)
        self.kline_chart.pack(fill=tk.BOTH, expand=True)

    def _build_predict_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="模型预测")

        bar = ttk.Frame(tab)
        bar.pack(fill=tk.X, pady=(0, 4))

        self.pred_csv = ttk.Combobox(bar, width=30, state="readonly")
        self.pred_model = ttk.Combobox(bar, width=30, state="readonly")
        self.pred_seq = ttk.Entry(bar, width=6)
        self.pred_seq.insert(0, "20")
        self.pred_th = ttk.Entry(bar, width=6)
        self.pred_th.insert(0, "0.5")
        self.pred_btn = ttk.Button(bar, text="运行预测", command=self._on_predict)

        self._row(bar, "数据", self.pred_csv, 0)
        self._row(bar, "模型", self.pred_model, 1)
        self._row(bar, "序列长", self.pred_seq, 2)
        self._row(bar, "阈值", self.pred_th, 3)
        self.pred_btn.grid(row=0, column=8, padx=4)

        self._fill_combo(self.pred_csv, list_csv_files())
        self._fill_combo(self.pred_model, list_model_files())

        self.pred_info = ttk.Label(tab, text="")
        self.pred_info.pack(fill=tk.X, pady=4)
        self.pred_chart = ChartPanel(tab)
        self.pred_chart.pack(fill=tk.BOTH, expand=True)

    def _build_backtest_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="策略回测")

        bar = ttk.Frame(tab)
        bar.pack(fill=tk.X, pady=(0, 4))

        self.bt_csv = ttk.Combobox(bar, width=28, state="readonly")
        self.bt_model = ttk.Combobox(bar, width=28, state="readonly")
        self._row(bar, "数据", self.bt_csv, 0)
        self._row(bar, "模型", self.bt_model, 1)

        bar2 = ttk.Frame(tab)
        bar2.pack(fill=tk.X, pady=(0, 4))

        self.bt_seq = ttk.Entry(bar2, width=5)
        self.bt_seq.insert(0, "20")
        self.bt_th = ttk.Entry(bar2, width=5)
        self.bt_th.insert(0, "0.5")
        self.bt_capital = ttk.Entry(bar2, width=8)
        self.bt_capital.insert(0, "100000")
        self.bt_ratio = ttk.Entry(bar2, width=5)
        self.bt_ratio.insert(0, "0.5")
        self.bt_btn = ttk.Button(bar2, text="运行回测", command=self._on_backtest)

        self._row(bar2, "序列长", self.bt_seq, 0)
        self._row(bar2, "阈值", self.bt_th, 1)
        self._row(bar2, "初始资金", self.bt_capital, 2)
        self._row(bar2, "仓位比", self.bt_ratio, 3)
        self.bt_btn.grid(row=0, column=8, padx=4)

        self._fill_combo(self.bt_csv, list_csv_files())
        self._fill_combo(self.bt_model, list_model_files())

        self.bt_info = ttk.Label(tab, text="")
        self.bt_info.pack(fill=tk.X, pady=4)
        self.bt_chart = ChartPanel(tab)
        self.bt_chart.pack(fill=tk.BOTH, expand=True)

    def _set_busy(self, busy, msg):
        self.status.config(text=msg)
        state = tk.DISABLED if busy else tk.NORMAL
        self.kline_btn.config(state=state)
        self.pred_btn.config(state=state)
        self.bt_btn.config(state=state)

    def _fill_kline_dates(self, csv_name: str):
        try:
            start, end = resolve_csv_date_range(csv_name)
        except Exception:
            return
        self.kline_start.set_value(start)
        self.kline_end.set_value(end)

    def _on_kline_csv_changed(self, _event=None):
        csv_name = self.kline_csv.get()
        if csv_name:
            self._fill_kline_dates(csv_name)

    def _on_kline(self):
        csv_name = self.kline_csv.get()
        if not csv_name:
            messagebox.showwarning("提示", "请选择数据文件")
            return
        try:
            from data_processor.DataPlot import read_stock_data, plot_kline

            df = read_stock_data(str(DATA_DIR / csv_name))
            start = self.kline_start.get_value() or df.index[0].strftime("%Y-%m-%d")
            end = self.kline_end.get_value() or df.index[-1].strftime("%Y-%m-%d")
            datetime.strptime(start, "%Y-%m-%d")
            datetime.strptime(end, "%Y-%m-%d")
            fig = plot_kline(df, start, end, csv_name, return_fig=True)
            self.kline_chart.show_figure(fig)
            self.status.config(text=f"K 线已绘制: {start} ~ {end}")
        except ValueError as e:
            if "does not match format" in str(e) or "unconverted data" in str(e):
                messagebox.showerror(
                    "错误", "日期格式错误，请使用如 '2018-07-01' 的格式"
                )
            else:
                messagebox.showerror("错误", str(e))
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def _on_predict(self):
        csv_name = self.pred_csv.get()
        model_name = self.pred_model.get()
        if not csv_name or not model_name:
            messagebox.showwarning("提示", "请选择数据和模型文件")
            return
        try:
            seq_len = int(self.pred_seq.get())
            threshold = float(self.pred_th.get())
        except ValueError:
            messagebox.showwarning("提示", "序列长和阈值必须是数字")
            return

        self._set_busy(True, "预测运行中...")
        threading.Thread(
            target=self._predict_worker,
            args=(csv_name, model_name, seq_len, threshold),
            daemon=True,
        ).start()

    def _predict_worker(self, csv_name, model_name, seq_len, threshold):
        try:
            fig, text = run_predict(csv_name, model_name, seq_len, threshold)
            self.after(0, lambda: self._predict_done(fig, text))
        except Exception as e:
            self.after(0, lambda: self._on_error("预测失败", e))

    def _predict_done(self, fig, text):
        self.pred_chart.show_figure(fig)
        self.pred_info.config(text=text)
        self._set_busy(False, "预测完成")

    def _on_backtest(self):
        csv_name = self.bt_csv.get()
        model_name = self.bt_model.get()
        if not csv_name or not model_name:
            messagebox.showwarning("提示", "请选择数据和模型文件")
            return
        try:
            seq_len = int(self.bt_seq.get())
            threshold = float(self.bt_th.get())
            capital = float(self.bt_capital.get())
            ratio = float(self.bt_ratio.get())
        except ValueError:
            messagebox.showwarning("提示", "参数格式不正确")
            return

        self._set_busy(True, "回测运行中...")
        threading.Thread(
            target=self._backtest_worker,
            args=(csv_name, model_name, seq_len, threshold, capital, ratio),
            daemon=True,
        ).start()

    def _backtest_worker(
        self, csv_name, model_name, seq_len, threshold, capital, ratio
    ):
        try:
            fig, text = run_backtest(
                csv_name, model_name, seq_len, threshold, capital, ratio
            )
            self.after(0, lambda: self._backtest_done(fig, text))
        except Exception as e:
            self.after(0, lambda: self._on_error("回测失败", e))

    def _backtest_done(self, fig, text):
        self.bt_chart.show_figure(fig)
        self.bt_info.config(text=text)
        self._set_busy(False, "回测完成")

    def _on_error(self, title, err):
        self._set_busy(False, "就绪")
        messagebox.showerror(title, str(err))


def main():
    app = DeepQuantApp()
    app.mainloop()


if __name__ == "__main__":
    main()
