"""
DiffSTG — 邻接矩阵对比图（距离图 / 相关图 / PE 图）
=====================================================
pre_len=6, F=1, seed=42
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

adj_types = ['Distance', 'Correlation', 'PE']
mae  = [34.50, 35.86, 37.97]
rmse = [41.10, 42.11, 44.62]
colors = ['#2563EB', '#DC2626', '#CA8A04']

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(adj_types))
w = 0.32
ax.bar(x - w/2, mae, w, color=colors, alpha=0.85, edgecolor='white', label='MAE')
ax.bar(x + w/2, rmse, w, color=colors, alpha=0.30, edgecolor='white', label='RMSE', hatch='//')

for i, (m, r) in enumerate(zip(mae, rmse)):
    ax.text(i - w/2, m + 1, f'{m:.1f}', ha='center', fontsize=11, fontweight='bold')
    ax.text(i + w/2, r + 1, f'{r:.1f}', ha='center', fontsize=11)

ax.set_xticks(x)
ax.set_xticklabels(adj_types, fontweight='bold')
ax.set_ylabel('Error')
ax.set_title('DiffSTG: Adjacency Matrix Comparison\n(pre_len=6, F=1, seed=42)')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 55)

plt.tight_layout()
fig.savefig(FIG_DIR / 'adjacency_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('[OK] adjacency_comparison.png')
