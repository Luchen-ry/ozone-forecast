"""
DiffSTG — 单目标预测对比图（O3 vs PM2.5 vs PM10）
=====================================================
pre_len=6, F=1, seed=42

输出到 external_baselines_F1/figures/
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- 数据 ----
pollutants = ['O3', 'PM2.5', 'PM10']
mae  = [34.50, 25.05, 38.81]
rmse = [41.10, 38.96, 83.59]
mape = [557.66, 124.28, 83.89]
data_range = [410, 543, 2759]

colors = ['#2563EB', '#DC2626', '#CA8A04']  # blue, red, gold

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
})

# ---- 图 1: MAE + RMSE 分组柱状图 ----
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(pollutants))
width = 0.32
bars_mae = ax.bar(x - width/2, mae, width, color=colors, alpha=0.85, edgecolor='white', label='MAE')
bars_rmse = ax.bar(x + width/2, rmse, width, color=colors, alpha=0.35, edgecolor='white', label='RMSE', hatch='//')

for bar in bars_mae:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{bar.get_height():.1f}', ha='center', fontsize=10, fontweight='bold')
for bar in bars_rmse:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{bar.get_height():.1f}', ha='center', fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(pollutants, fontsize=13, fontweight='bold')
ax.set_ylabel('Error')
ax.set_title('DiffSTG Single-Target Prediction (pre_len=6, F=1)\nMAE & RMSE Comparison')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(FIG_DIR / 'pollutant_mae_rmse.png', dpi=150, bbox_inches='tight')
plt.close()
print('[OK] pollutant_mae_rmse.png')

# ---- 图 2: MAPE 对比 ----
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(pollutants, mape, color=colors, alpha=0.85, edgecolor='white', width=0.5)
for bar, v in zip(bars, mape):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
            f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('MAPE (%)')
ax.set_title('DiffSTG MAPE by Pollutant (pre_len=6, F=1)')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(FIG_DIR / 'pollutant_mape.png', dpi=150, bbox_inches='tight')
plt.close()
print('[OK] pollutant_mape.png')

# ---- 图 3: RMSE / 数据范围归一化 + 原始对比 ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左: raw RMSE
ax = axes[0]
raw_bars = ax.bar(pollutants, rmse, color=colors, alpha=0.85, edgecolor='white', width=0.5)
for bar, v in zip(raw_bars, rmse):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{v:.1f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('RMSE')
ax.set_title('Raw RMSE')
ax.grid(True, alpha=0.3, axis='y')

# 右: RMSE / Data Range %
ax = axes[1]
rmse_pct = [r / rng * 100 for r, rng in zip(rmse, data_range)]
pct_bars = ax.bar(pollutants, rmse_pct, color=colors, alpha=0.85, edgecolor='white', width=0.5)
for bar, v in zip(pct_bars, rmse_pct):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('RMSE / Data Range (%)')
ax.set_title('Normalized RMSE (% of Data Range)')
ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('DiffSTG — RMSE: Raw vs Normalized by Data Range', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(FIG_DIR / 'pollutant_rmse_normalized.png', dpi=150, bbox_inches='tight')
plt.close()
print('[OK] pollutant_rmse_normalized.png')

# ---- 图 4: 汇总信息表 ----
fig, ax = plt.subplots(figsize=(9, 4))
ax.axis('off')
table_data = [
    ['O3',   '34.50',  '41.10',  '557.66%', '[1, 410]',  '17'],
    ['PM2.5','25.05',  '38.96',  '124.28%', '[1, 543]',  '22'],
    ['PM10', '38.81',  '83.59',  '83.89%',  '[1, 2759]', '17'],
]
col_labels = ['Pollutant', 'MAE', 'RMSE', 'MAPE', 'Data Range', 'Best Epoch']
table = ax.table(cellText=table_data, colLabels=col_labels,
                 loc='center', cellLoc='center',
                 colColours=['#E5E7EB']*6)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.8)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(fontweight='bold')
    cell.set_edgecolor('white')
ax.set_title('DiffSTG Single-Target Prediction Results (pre_len=6, F=1, seed=42)',
             fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
fig.savefig(FIG_DIR / 'pollutant_summary_table.png', dpi=150, bbox_inches='tight')
plt.close()
print('[OK] pollutant_summary_table.png')

print('\nDone!')
