import tushare as ts

txt_file_path = "config/token.txt"
with open(txt_file_path, "r", encoding="utf-8") as file:
    token = file.read().strip()

ts.set_token(token)
pro = ts.pro_api()

# 每日指标：换手率、市盈率、市净率、股本、市值等（与 daily 的 OHLCV 互补）
start_date = "20250701"
end_date = "20260601"
ts_code = "000001.SZ"
df = pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date)

# 后缀 _ind 表示每日指标（indicators）
save_dir = "data/raw/"
csv_filename = f"{ts_code.replace('.', '_')}_{start_date}_{end_date}_ind.csv"
df.to_csv(save_dir + csv_filename, index=False, encoding="utf-8")
print(f"数据已保存为{csv_filename}")
print(f"列名: {list(df.columns)}")
