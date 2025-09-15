import pandas as pd
import mplfinance as mpf
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')  # 切换到TkAgg后端
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def read_stock_data(csv_file):
    """读取CSV格式的股票数据"""
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 将trade_date转换为datetime格式并设置为索引
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df['date'] = df['trade_date']
    df = df.set_index('date')

    # 确保数据按日期排序
    df = df.sort_index()

    # # 重命名列以符合mplfinance的要求
    # df = df.rename(columns={
    #     'open': 'Open',
    #     'high': 'High',
    #     'low': 'Low',
    #     'close': 'Close',
    #     'vol': 'Volume'
    # })

    return df


def plot_kline(df, start_date, end_date, title="股票K线图"):
    """
    绘制指定日期范围内的K线图

    参数:
    df: 包含股票数据的DataFrame
    start_date: 开始日期，格式如'2018-07-01'
    end_date: 结束日期，格式如'2018-07-31'
    title: 图表标题
    """
    # 转换日期格式
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    # 筛选指定日期范围内的数据
    mask = (df.index >= start) & (df.index <= end)
    df_filtered = df.loc[mask]

    if df_filtered.empty:
        print("指定日期范围内没有数据！")
        return

    # 设置K线图样式
    plt.style.use('seaborn-v0_8-whitegrid')
    my_color = mpf.make_marketcolors(
        up='red',  # 上涨K线颜色
        down='green',  # 下跌K线颜色
        edge='i',  # 边缘颜色与实体颜色一致
        wick='i',  # 上下影线颜色与实体颜色一致
        volume='in'  # 成交量颜色与K线颜色一致
    )
    my_style = mpf.make_mpf_style(
        base_mpl_style='seaborn-v0_8',
        marketcolors=my_color,
        figcolor='white'
    )

    # 绘制K线图
    # fig, ax = plt.subplots(figsize=(12, 8))
    mpf.plot(
        df_filtered,
        type='candle',  # 蜡烛图类型
        mav=(5, 10, 20),  # 5日、10日、20日均线
        volume=True,  # 显示成交量
        title=title,
        style=my_style,
        show_nontrading=False,  # 不显示非交易日
        figratio=(12, 8),
        tight_layout=True
    )

    plt.show()


# 示例用法
if __name__ == "__main__":
    # 读取股票数据
    stock_data = read_stock_data('D:\lc\githubCode\DeepQuant\data\\000001_20250719_20250912.csv')

    # 绘制2023年1月至2023年6月的K线图
    plot_kline(
        stock_data,
        start_date='2025-07-19',
        end_date='2025-09-12',
        title="000001.SZ"
    )
