import tushare as ts

# 设置token并初始化接口
token = ""
ts.set_token(token)
pro = ts.pro_api()

# 获取数据
start_date = '20180701'
end_date = '20250701'
ts_code = '000001'
df = pro.daily(ts_code=ts_code + '.SZ', start_date=start_date, end_date=end_date)

# 保存为CSV文件（推荐格式）
save_dir = "D:\lc\githubCode\DeepQuant\data\\"
csv_filename = f"{ts_code}_{start_date}_{end_date}.csv"
df.to_csv(save_dir + csv_filename, index=False, encoding='utf-8')
print(f"数据已保存为{csv_filename}")
