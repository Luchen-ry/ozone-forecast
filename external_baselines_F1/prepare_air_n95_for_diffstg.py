"""
Prepare AIR_N95 data for DiffSTG baseline.
============================================
Input:
  - matrix_N95/data.npy          (95, 8717)   O3 concentration
  - xlsx_N95/station_loc1.xlsx   95 station metadata (lon, lat)

Output (written to DiffSTG data directory):
  - data/dataset/AIR_N95/flow.npy   (8717, 95, 1)   DiffSTG format
  - data/dataset/AIR_N95/adj.npy    (95, 95)        distance-based adjacency

Usage:
  cd "/d/生产实习-时空数据/臭氧预测资料"
  python external_baselines_F1/prepare_air_n95_for_diffstg.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from geopy.distance import geodesic

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / 'external_baselines_F1' / 'DiffSTG' / 'data' / 'dataset' / 'AIR_N95'
DST.mkdir(parents=True, exist_ok=True)

# ---- 1. flow.npy: (T, N, F) = (8717, 95, 1) ----
data = np.load(ROOT / 'matrix_N95' / 'data.npy')          # (95, 8717)
flow = data.T[:, :, np.newaxis].astype(np.float32)         # (8717, 95, 1)
np.save(str(DST / 'flow.npy'), flow)
print(f'[OK] flow.npy saved: {flow.shape}, range=[{flow.min():.1f}, {flow.max():.1f}]')

# ---- 2. adj.npy: (95, 95) distance-based ----
stations = pd.read_excel(ROOT / 'xlsx_N95' / 'station_loc1.xlsx')
stations.columns = ['code', 'name', 'city', 'lon', 'lat']
coords = stations[['lat', 'lon']].values

n = len(stations)
adj = np.zeros((n, n), dtype=np.float32)
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        d = geodesic(coords[i], coords[j]).km
        if d < 100:                                     # 100km cutoff
            adj[i, j] = np.exp(-d**2 / (2 * 50**2))     # Gaussian kernel, sigma=50km

np.save(str(DST / 'adj.npy'), adj)
sparsity = (adj != 0).sum() / adj.size
print(f'[OK] adj.npy saved: {adj.shape}, non-zero={sparsity:.2%}')

print('\nData ready. Now run:')
print(f'  cd external_baselines_F1/DiffSTG')
print(f'  python train.py --data AIR_N95 --T_h 24 --T_p 6 --N 50 --sample_steps 50 --hidden_size 32 --batch_size 8 --lr 7e-4 --n_samples 3')
