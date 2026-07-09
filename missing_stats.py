import pandas as pd
from pathlib import Path

data_dir = Path("data_N95")
pollutants = ["O3", "PM2.5", "PM10"]

results = []

for pollutant in pollutants:
    total_count = 0
    missing_count = 0

    for file in sorted(data_dir.glob("china_sites_*.csv")):
        df = pd.read_csv(file)

        sub = df[df["type"] == pollutant]

        if sub.empty:
            continue

        values = sub.drop(columns=["date", "hour", "type"], errors="ignore")

        total_count += values.size
        missing_count += values.isna().sum().sum()

    missing_ratio = missing_count / total_count if total_count else 0

    results.append({
        "pollutant": pollutant,
        "total_count": total_count,
        "missing_count": missing_count,
        "missing_ratio": missing_ratio
    })

out = pd.DataFrame(results)
print(out)

out.to_csv("pollutant_missing_summary.csv", index=False, encoding="utf-8-sig")
print("已生成 pollutant_missing_summary.csv")