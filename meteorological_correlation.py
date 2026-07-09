import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("figures").mkdir(exist_ok=True)

# O3: (95, 8717)，转置后变为 (8717, 95)，与气象数据对齐
o3 = np.load("matrix_N95/data.npy").T

# 读取 14 个气象变量
met = np.load("matrix_N95/met_raw_aligned_cache.npz")

# 变量中文说明
name_map = {
    "blh": "边界层高度",
    "d2m": "2 米露点温度",
    "fsr": "地表相关辐射变量",
    "kx": "K 指数",
    "sp": "地表气压",
    "ssr": "地表太阳辐射",
    "ssrd": "向下短波辐射",
    "t2m": "2 米气温",
    "tcc": "总云量",
    "tcwv": "整层可降水量",
    "tp": "总降水量",
    "u10": "10 米 U 向风速",
    "v10": "10 米 V 向风速",
    "zust": "摩擦速度",
}

rows = []

# 展平后，每个位置对应“某一小时、某一站点”的数据
o3_flat = o3.reshape(-1)

for key in met.files:
    met_flat = met[key].reshape(-1)

    # 去掉 O3 或气象变量中无效的数据
    valid = np.isfinite(o3_flat) & np.isfinite(met_flat)

    x = o3_flat[valid]
    y = met_flat[valid]

    if len(x) == 0 or np.std(x) == 0 or np.std(y) == 0:
        corr = np.nan
    else:
        corr = np.corrcoef(x, y)[0, 1]

    rows.append({
        "variable": key,
        "meaning": name_map.get(key, key),
        "correlation_with_o3": corr,
        "sample_count": len(x)
    })

result = pd.DataFrame(rows)
result = result.sort_values("correlation_with_o3", ascending=False)

print(result.to_string(index=False))

# 保存相关系数表
result.to_csv(
    "o3_meteorological_correlation.csv",
    index=False,
    encoding="utf-8-sig"
)

# 画相关系数柱状图
plot_df = result.sort_values("correlation_with_o3")

plt.figure(figsize=(9, 6))
plt.barh(plot_df["variable"], plot_df["correlation_with_o3"])
plt.axvline(0, linewidth=0.8)
plt.xlabel("Pearson Correlation with O3")
plt.ylabel("Meteorological Variable")
plt.title("Correlation Between O3 and Meteorological Variables")
plt.tight_layout()
plt.savefig("figures/o3_meteorological_correlation.png", dpi=200)
plt.close()

print("\n已生成：")
print("o3_meteorological_correlation.csv")
print("figures/o3_meteorological_correlation.png")