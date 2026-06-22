import tushare as ts

txt_file_path = "config/token.txt"  # 读取tushare网址账号的token
with open(txt_file_path, "r", encoding="utf-8") as file:
    token = file.read().strip()

# tushare设置
ts.set_token(token)
pro = ts.pro_api()

# 获取股票数据
start_date = "20250701"
end_date = "20260601"
ts_code = "000001.SZ"
df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

# 保存为CSV文件（后缀 _ohlcv 表示日K线）
save_dir = "data/raw/"
csv_filename = f"{ts_code.replace('.', '_')}_{start_date}_{end_date}_ohlcv.csv"
df.to_csv(save_dir + csv_filename, index=False, encoding="utf-8")
print(f"数据已保存为{csv_filename}")
print(f"列名: {list(df.columns)}")
