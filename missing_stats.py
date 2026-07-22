import pandas as pd
from pathlib import Path


# 数据目录
data_dir = Path("data_N95")

# 需要统计的污染物
pollutants = ["O3", "PM2.5", "PM10"]

station_file = Path("xlsx_N95/station_loc1.xlsx")

stations = pd.read_excel(station_file)

# 查看站点文件结构
print("station_loc1.xlsx columns:")
print(stations.columns.tolist())

# 默认取第一列作为站点编号
# 如果你的文件中有明确列名，可以改成对应列
site_col = stations.columns[0]

AIR_N95 = (
    stations[site_col]
    .astype(str)
    .tolist()
)

results = []

for pollutant in pollutants:
    print("\n")
    print("正在统计:", pollutant)

    total_count = 0
    missing_count = 0
    files = sorted(data_dir.glob("china_sites_*.csv"))
    print("文件数量:", len(files))

    for idx, file in enumerate(files):

        # 每50个文件提示一次进度
        if (idx + 1) % 50 == 0 or idx == 0 or idx == len(files) - 1:
            print(
                f"[{idx+1}/{len(files)}] 已处理"
            )

        df = pd.read_csv(file)

        # 找对应污染物
        sub = df[df["type"] == pollutant]
        if sub.empty:
            continue

        # 只保留95个站点
        available_sites = [
            s for s in AIR_N95
            if s in sub.columns
        ]
        values = sub[available_sites]
        total_count += values.size
        missing_count += (
            values.isna()
            .sum()
            .sum()
        )

    ratio = (
        missing_count / total_count
        if total_count > 0
        else 0
    )
    results.append(
        {
            "pollutant": pollutant,
            "total_count": total_count,
            "missing_count": missing_count,
            "missing_ratio": ratio
        }
    )
result_df = pd.DataFrame(results)


print("\n缺失统计结果")
print(result_df)
result_df.to_csv(
    "pollutant_missing_summary.csv",
    index=False,
    encoding="utf-8-sig"
)
print("\n已保存:")
print("pollutant_missing_summary.csv")