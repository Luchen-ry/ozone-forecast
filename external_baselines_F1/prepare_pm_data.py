"""
Prepare PM2.5 and PM10 data for DiffSTG baseline.
===================================================
从 CSV 提取 PM2.5/PM10 → DiffSTG 的 flow.npy + adj.npy

Usage:
  conda activate pedw
  python external_baselines_F1/prepare_pm_data.py
"""
import numpy as np
import pandas as pd
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- 读取站点列表 ----
stations_df = pd.read_excel(ROOT / 'xlsx_N95' / 'station_loc1.xlsx')
stations_df.columns = ['code', 'name', 'city', 'lon', 'lat']
station_codes = stations_df['code'].astype(str).tolist()
print(f'站点数: {len(station_codes)}')

# ---- 匹配 CSV 列名 ----
sample_csv = sorted((ROOT / 'data_N95').glob('china_sites_*.csv'))[0]
csv_header = pd.read_csv(sample_csv, nrows=0).columns.tolist()
csv_station_cols = [c for c in csv_header if c not in ('date', 'hour', 'type')]

code_map = {}
for sc in station_codes:
    for cc in csv_station_cols:
        if str(sc) == str(cc) or str(sc).rstrip('A') == str(cc).rstrip('A'):
            code_map[sc] = cc
            break
print(f'站点匹配: {len(code_map)}/{len(station_codes)}')

# ---- 分别处理 PM2.5 和 PM10 ----
for pollutant, dst_name in [('PM2.5', 'AIR_N95_PM25'), ('PM10', 'AIR_N95_PM10')]:
    print(f'\n处理 {pollutant} ...')
    frames = []
    for csv_file in sorted((ROOT / 'data_N95').glob('china_sites_*.csv')):
        df = pd.read_csv(csv_file)
        sub = df[df['type'] == pollutant]
        if sub.empty:
            continue
        for _, row in sub.sort_values('hour').iterrows():
            values = []
            for sc in station_codes:
                try:
                    values.append(float(row.get(code_map[sc], np.nan)))
                except (ValueError, TypeError):
                    values.append(np.nan)
            frames.append(values)

    data = np.array(frames, dtype=np.float32).T  # (95, T)
    print(f'  shape: {data.shape}, 缺失: {np.isnan(data).sum()/data.size:.2%}')

    # 缺失值填补
    df_fill = pd.DataFrame(data).ffill(axis=1).bfill(axis=1).fillna(0)
    data = df_fill.values.astype(np.float32)

    # flow.npy: (T, 95, 1)
    flow = data.T[:, :, np.newaxis].astype(np.float32)
    dst_dir = ROOT / 'external_baselines_F1' / 'DiffSTG' / 'data' / 'dataset' / dst_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(dst_dir / 'flow.npy'), flow)
    print(f'  [OK] flow.npy: {flow.shape}, range=[{flow.min():.0f}, {flow.max():.0f}]')

    # 复用 O3 的距离矩阵
    shutil.copy(
        str(ROOT / 'external_baselines_F1/DiffSTG/data/dataset/AIR_N95/adj.npy'),
        str(dst_dir / 'adj.npy'))
    print(f'  [OK] adj.npy 已复制')

print('\n完成。运行:')
print('  python train.py --data AIR_N95_PM25 --T_h 24 --T_p 6 --seed 42 --N 50 --sample_steps 50 --hidden_size 32 --batch_size 4 --n_samples 1')
print('  python train.py --data AIR_N95_PM10 --T_h 24 --T_p 6 --seed 42 --N 50 --sample_steps 50 --hidden_size 32 --batch_size 4 --n_samples 1')
