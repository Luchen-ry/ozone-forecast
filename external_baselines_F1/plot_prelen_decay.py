"""
DiffSTG vs PE-DiffWaveNet — 多步长衰减曲线
==============================================
生成两张图：
  1. RMSE 衰减曲线（双模型对比）
  2. MAE 衰减曲线（双模型对比）

输出到 external_baselines_F1/figures/
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- 数据 ----
pre_lens = [1, 3, 6, 12, 24]

# DiffSTG (F=1, O3 only)
diffstg_rmse = [13.08, 32.42, 41.10, 45.70, 48.65]
diffstg_mae  = [9.61,  26.60, 34.50, 39.10, 41.89]

# PE-DiffWaveNet backbone (F=15, O3+14 meteorology)
# Extracted from logs/test_rmse, test_mae fields; p3 uses "fix" version
pe_rmse = [6.878, 9.047, 11.240, 14.239, 16.789]
pe_mae  = [4.996, 6.365, 7.909, 10.144, 12.468]

# ---- 风格设置 ----
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

color_diff = '#DC2626'  # red for DiffSTG
color_pe   = '#2563EB'  # blue for PE

# ---- 图 1: RMSE 衰减 ----
fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(pre_lens, diffstg_rmse, 'o-', color=color_diff, linewidth=2, markersize=8,
        label='DiffSTG (F=1, O3 only)')
ax.plot(pre_lens, pe_rmse, 's--', color=color_pe, linewidth=2, markersize=8,
        label='PE-DiffWaveNet (F=15, O3+meteorology)')

# 标注数值
for x, y in zip(pre_lens, diffstg_rmse):
    ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, 10),
                fontsize=9, ha='center', color=color_diff)
for x, y in zip(pre_lens, pe_rmse):
    ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, -15),
                fontsize=9, ha='center', color=color_pe)

ax.set_xlabel('Prediction Horizon (hours)')
ax.set_ylabel('RMSE')
ax.set_title('RMSE vs Prediction Horizon\nDiffSTG vs PE-DiffWaveNet')
ax.set_xticks(pre_lens)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 25)

plt.tight_layout()
fig.savefig(FIG_DIR / 'rmse_decay_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('[OK] rmse_decay_comparison.png')

# ---- 图 2: MAE 衰减 ----
fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(pre_lens, diffstg_mae, 'o-', color=color_diff, linewidth=2, markersize=8,
        label='DiffSTG (F=1, O3 only)')
ax.plot(pre_lens, pe_mae, 's--', color=color_pe, linewidth=2, markersize=8,
        label='PE-DiffWaveNet (F=15, O3+meteorology)')

for x, y in zip(pre_lens, diffstg_mae):
    ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, 10),
                fontsize=9, ha='center', color=color_diff)
for x, y in zip(pre_lens, pe_mae):
    ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, -15),
                fontsize=9, ha='center', color=color_pe)

ax.set_xlabel('Prediction Horizon (hours)')
ax.set_ylabel('MAE')
ax.set_title('MAE vs Prediction Horizon\nDiffSTG vs PE-DiffWaveNet')
ax.set_xticks(pre_lens)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 25)

plt.tight_layout()
fig.savefig(FIG_DIR / 'mae_decay_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('[OK] mae_decay_comparison.png')

# ---- 图 3: RMSE 比率 (DiffSTG / PE-DiffWaveNet) ----
ratios = [d / p for d, p in zip(diffstg_rmse, pe_rmse)]

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(range(len(pre_lens)), ratios, color='#F59E0B', alpha=0.85, edgecolor='white')
for bar, r in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{r:.1f}×', ha='center', fontsize=10, fontweight='bold')

ax.set_xticks(range(len(pre_lens)))
ax.set_xticklabels([f'{h}h' for h in pre_lens])
ax.set_ylabel('RMSE Ratio (DiffSTG / PE-DiffWaveNet)')
ax.set_title('Performance Gap vs Prediction Horizon')
ax.axhline(y=1.0, color='#6B7280', linestyle='--', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(FIG_DIR / 'rmse_ratio_bars.png', dpi=150, bbox_inches='tight')
plt.close()
print('[OK] rmse_ratio_bars.png')

print('\nDone! All figures saved to:', FIG_DIR)
