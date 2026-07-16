"""
Prepare AIR_N95 data for DiffSTG baseline.
============================================
Input:
  - matrix_N95/data.npy                    (95, 8717)   O3 concentration (raw, ug/m^3)
  - matrix_N95/met_raw_aligned_cache.npz   14 met vars  (8717, 95) each, raw scale
  - xlsx_N95/station_loc1.xlsx             95 station metadata (lon, lat)

Output (written to DiffSTG data directory):
  - data/dataset/AIR_N95/flow.npy   (8717, 95, 15)  DiffSTG format, raw scale
  - data/dataset/AIR_N95/adj.npy    (95, 95)         distance-based adjacency

Channel order (15):
  0: O3 (ug/m^3)
  1-14: blh, d2m, fsr, kx, sp, ssr, ssrd, t2m, tcc, tcwv, tp, u10, v10, zust

Usage:
  cd "/d/生产实习-时空数据/臭氧预测资料"
  python external_baselines/prepare_air_n95_for_diffstg.py            # F=15 (default)
  python external_baselines/prepare_air_n95_for_diffstg.py --F 1      # F=1 (O3 only, legacy)
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from geopy.distance import geodesic

# 14 meteorological variables from met_raw_aligned_cache.npz (consistent order with PE-DiffWaveNet)
MET_KEYS = ['blh', 'd2m', 'fsr', 'kx', 'sp', 'ssr', 'ssrd', 't2m', 'tcc', 'tcwv', 'tp', 'u10', 'v10', 'zust']

# ROOT = project root (contains matrix_N95/, xlsx_N95/, external_baselines/)
# script lives at external_baselines/external_baselines/prepare_air_n95_for_diffstg.py
ROOT = Path(__file__).resolve().parent.parent.parent
DST = ROOT / 'external_baselines' / 'external_baselines' / 'DiffSTG' / 'data' / 'dataset' / 'AIR_N95'
DST.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--F', type=int, default=15, choices=[1, 15],
                    help='number of feature channels: 1=O3 only, 15=O3+14met (default 15)')
args = parser.parse_args()

# ---- 1. flow.npy: (T, N, F) = (8717, 95, F) ----
o3 = np.load(ROOT / 'matrix_N95' / 'data.npy')             # (95, 8717) raw ug/m^3
o3 = o3.T.astype(np.float32)                                # (8717, 95)

if args.F == 1:
    flow = o3[:, :, np.newaxis]                             # (8717, 95, 1)
    print(f'[F=1] O3 only')
else:
    met = np.load(ROOT / 'matrix_N95' / 'met_raw_aligned_cache.npz')
    met_arr = np.stack([met[k].astype(np.float32) for k in MET_KEYS], axis=-1)  # (8717, 95, 14)
    met_arr = np.nan_to_num(met_arr, nan=0.0, posinf=0.0, neginf=0.0)
    flow = np.concatenate([o3[:, :, np.newaxis], met_arr], axis=-1)  # (8717, 95, 15)
    print(f'[F=15] O3 + 14 met vars: {MET_KEYS}')

np.save(str(DST / 'flow.npy'), flow)
print(f'[OK] flow.npy saved: {flow.shape}, dtype={flow.dtype}')
print(f'     O3  range: [{flow[:,:,0].min():.2f}, {flow[:,:,0].max():.2f}]')
if args.F == 15:
    for i, k in enumerate(MET_KEYS, start=1):
        print(f'     {k:>6} range: [{flow[:,:,i].min():.2f}, {flow[:,:,i].max():.2f}]')

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

print(f'\nData ready (F={args.F}). Now run:')
print(f'  cd {ROOT / "external_baselines" / "external_baselines" / "DiffSTG"}')
if args.F == 15:
    print(f'  python train.py --data AIR_N95 --T_h 24 --T_p 6 --seed 42 \\')
    print(f'    --N 50 --sample_steps 50 --hidden_size 32 --batch_size 4 --n_samples 1 \\')
    print(f'    --num_features 15 --split_mode noleak')
else:
    print(f'  python train.py --data AIR_N95 --T_h 24 --T_p 6 --N 50 --sample_steps 50 --hidden_size 32 --batch_size 8 --n_samples 3')
