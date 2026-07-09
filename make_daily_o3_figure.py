import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("figures").mkdir(exist_ok=True)

# 读取处理后的 O3 数据和时间索引
data = np.load("matrix_N95/data.npy")
time_index = pd.to_datetime(
    np.load("matrix_N95/time_index.npy", allow_pickle=True)
)

# data 形状为 (95, 8717)
# 每个小时对 95 个站点求平均
hourly_mean_o3 = np.nanmean(data, axis=0)

# 将逐小时数据按日期聚合为日均值
df = pd.DataFrame({
    "time": time_index,
    "mean_o3": hourly_mean_o3
})

df["date"] = df["time"].dt.date

daily_o3 = (
    df.groupby("date", as_index=False)["mean_o3"]
    .mean()
)

daily_o3["date"] = pd.to_datetime(daily_o3["date"])

# 保存日均数据，后续报告或作图时可直接使用
daily_o3.to_csv(
    "daily_mean_o3.csv",
    index=False,
    encoding="utf-8-sig"
)

# 绘制日均 O3 曲线
plt.figure(figsize=(12, 4))
plt.plot(daily_o3["date"], daily_o3["mean_o3"])
plt.xlabel("Date")
plt.ylabel("Daily Mean O3")
plt.title("Daily Mean O3 Time Series of 95 Stations")
plt.tight_layout()

plt.savefig(
    "figures/o3_daily_mean_timeseries.png",
    dpi=200
)

plt.close()

print("已生成 daily_mean_o3.csv")
print("已生成 figures/o3_daily_mean_timeseries.png")