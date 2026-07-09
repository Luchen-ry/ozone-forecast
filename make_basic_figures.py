import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("figures").mkdir(exist_ok=True)

# 1. 95 个站点每小时平均 O3 的全年变化
data = np.load("matrix_N95/data.npy")
time_index = np.load("matrix_N95/time_index.npy", allow_pickle=True)

o3_mean = np.nanmean(data, axis=0)

plt.figure(figsize=(12, 4))
plt.plot(time_index, o3_mean)
plt.xlabel("Time")
plt.ylabel("Mean O3")
plt.title("Mean O3 Time Series of 95 Stations")
plt.tight_layout()
plt.savefig("figures/o3_mean_timeseries.png", dpi=200)
plt.close()

# 2. 95 个站点经纬度分布
station = pd.read_excel("xlsx_N95/station_loc1.xlsx")

plt.figure(figsize=(6, 5))
plt.scatter(station["经度"], station["纬度"])
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Location Distribution of 95 Air Quality Stations")
plt.tight_layout()
plt.savefig("figures/station_location.png", dpi=200)
plt.close()

print("已生成 figures/o3_mean_timeseries.png")
print("已生成 figures/station_location.png")