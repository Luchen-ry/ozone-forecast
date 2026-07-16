"""
Generate alternative adjacency matrices for DiffSTG baseline.
==============================================================
读取已有的 AIR_N95 flow.npy，生成：
  - output/corr_adj.npy  Pearson 相关图
  - output/pe_adj.npy    排列熵相似度图

Usage:
  conda activate pedw
  python external_baselines_F1/prepare_alt_adj.py

生成后手动替换 data/dataset/AIR_N95/adj.npy 即可跑实验。
"""
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- 加载 O3 flow.npy (T, 95, 1) → 转为 (95, T) ----
flow_path = ROOT / 'external_baselines_F1/DiffSTG/data/dataset/AIR_N95/flow.npy'
flow = np.load(flow_path)  # (8717, 95, 1)
data = flow.squeeze(-1).T  # (95, 8717)
n_stations, n_times = data.shape
print(f'数据: {n_stations} 站点 × {n_times} 时间步')
print(f'缺失值: {np.isnan(data).sum()}/{data.size}')

# ---- 1. Pearson 相关图 ----
print('\n计算 Pearson 相关矩阵...')
corr = np.zeros((n_stations, n_stations), dtype=np.float32)
for i in range(n_stations):
    for j in range(i + 1, n_stations):
        r, _ = pearsonr(data[i], data[j])
        corr[i, j] = r
        corr[j, i] = r
    corr[i, i] = 1.0

corr_adj = np.abs(corr)
threshold = np.percentile(corr_adj[corr_adj < 1.0], 70)  # top 30%
corr_adj[corr_adj < threshold] = 0
print(f'  阈值: {threshold:.3f}, 非零边: {(corr_adj!=0).sum()/corr_adj.size:.2%}')

np.save(str(OUTPUT_DIR / 'corr_adj.npy'), corr_adj)
print(f'  [OK] {OUTPUT_DIR / "corr_adj.npy"}')

# ---- 2. 排列熵相似度图 ----
print('\n计算排列熵相似度矩阵...')

def permutation_entropy(ts, order=3, delay=1):
    n = len(ts)
    if n < order * delay:
        return np.nan
    from collections import Counter
    patterns = Counter()
    for i in range(n - (order - 1) * delay):
        pattern = tuple(np.argsort(ts[i:i + order * delay:delay]))
        patterns[pattern] += 1
    total = sum(patterns.values())
    pe = sum(-c/total * np.log(c/total) for c in patterns.values())
    return pe / np.log(np.math.factorial(order))

pe_vals = np.array([permutation_entropy(data[i]) for i in range(n_stations)], dtype=np.float32)
print(f'  PE 范围: [{pe_vals.min():.4f}, {pe_vals.max():.4f}]')

pe_diff = np.abs(pe_vals[:, None] - pe_vals[None, :])
sigma = np.std(pe_vals) * 0.5
pe_adj = np.exp(-pe_diff**2 / (2 * sigma**2)).astype(np.float32)
pe_threshold = np.percentile(pe_adj[pe_adj < 1.0], 50)
pe_adj[pe_adj < pe_threshold] = 0
print(f'  非零边: {(pe_adj!=0).sum()/pe_adj.size:.2%}')

np.save(str(OUTPUT_DIR / 'pe_adj.npy'), pe_adj)
print(f'  [OK] {OUTPUT_DIR / "pe_adj.npy"}')

print(f'\n完成。使用方法:')
print('  1. cp output/corr_adj.npy data/dataset/AIR_N95/adj.npy → 跑相关图实验')
print('  2. cp output/pe_adj.npy   data/dataset/AIR_N95/adj.npy → 跑 PE 图实验')
