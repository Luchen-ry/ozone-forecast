"""
DiffSTG 预测 vs 真实值可视化
=============================
Input:  forecast pkl + O3 raw data (for reverse normalization)
Output: figures/diffstg_pred_vs_true.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / 'external_baselines_F1' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- 1. Load forecast pkl ----
pkl_path = ROOT / 'external_baselines_F1' / 'DiffSTG' / 'output' / 'forecast' / \
    'UGnet+32+50+quad+0.1+50+ddpm+24+6+42+1+True+AIR_N95+0.0+False+False+0.002+4.pkl'
with open(pkl_path, 'rb') as f:
    samples, targets, observed_flag, evaluate_flag = pickle.load(f)

# samples: (50, 1, 30, 95, 1)  = (B, n_samples, T, V, F)
# targets: (50, 30, 95, 1)     = (B, T, V, F)
# T = 30 = 24 (T_h) + 6 (T_p)

samples = samples.squeeze(1).numpy()  # (50, 30, 95, 1)
targets = targets.numpy()

# ---- 2. Reverse normalization ----
raw = np.load(ROOT / 'matrix_N95' / 'data.npy')  # (95, 8717)
raw_flat = raw.flatten()
mean = raw_flat.mean()
std = raw_flat.std()
print(f'Data: mean={mean:.2f}, std={std:.2f}')

samples_raw = samples * std + mean  # denormalize
targets_raw = targets * std + mean

# ---- 3. Extract future prediction (last 6 steps) ----
T_h = 24
T_p = 6
# targets: (B, 30, 95, 1)
target_future = targets_raw[:, -T_p:, :, 0]    # (50, 6, 95)  真实值
pred_future = samples_raw[:, -T_p:, :, 0]       # (50, 6, 95)  预测值

# Flatten all stations & samples
y_true_all = target_future.flatten()
y_pred_all = pred_future.flatten()

# ---- 4. Figure 1: Scatter plot (predicted vs true, all points) ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter
ax = axes[0]
ax.scatter(y_true_all, y_pred_all, alpha=0.15, s=1, color='#2563EB')
ax.plot([0, 400], [0, 400], '--', color='#DC2626', linewidth=1, label='Perfect')
ax.set_xlabel('True O3', fontsize=11)
ax.set_ylabel('Predicted O3', fontsize=11)
ax.set_title('DiffSTG: Predicted vs True (All 95 stations, 6h forecast)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 300)
ax.set_ylim(0, 300)

# Histogram of errors
ax = axes[1]
errors = y_pred_all - y_true_all
ax.hist(errors, bins=80, color='#2563EB', alpha=0.7, edgecolor='white')
ax.axvline(0, color='#DC2626', linewidth=1.5, linestyle='--')
ax.axvline(errors.mean(), color='#CA8A04', linewidth=1.5, linestyle='-', label=f'Mean: {errors.mean():.1f}')
ax.set_xlabel('Prediction Error (Pred - True)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Error Distribution (MAE={:.1f}, RMSE={:.1f})'.format(
    np.abs(errors).mean(), np.sqrt((errors**2).mean())), fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(FIG_DIR / 'diffstg_scatter_error.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: diffstg_scatter_error.png')

# ---- 5. Figure 2: Time series comparison (3 stations, first 50 hours) ----
station_picks = [0, 30, 60]  # pick 3 stations
n_hours = 50

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

for idx, st in enumerate(station_picks):
    ax = axes[idx]
    true_series = target_future[:n_hours, :, st].T.flatten()  # (6 * 50,)
    pred_series = pred_future[:n_hours, :, st].T.flatten()

    hours = np.arange(len(true_series))
    ax.plot(hours, true_series, linewidth=1.2, color='#2563EB', label='True', alpha=0.8)
    ax.plot(hours, pred_series, linewidth=1.2, color='#DC2626', label='Predicted', alpha=0.8)

    st_mae = np.abs(pred_series - true_series).mean()
    ax.set_title(f'Station #{st}: Prediction vs True (6h ahead, MAE={st_mae:.1f})',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('O3', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time step (6h blocks concatenated)', fontsize=11)
plt.tight_layout()
plt.savefig(str(FIG_DIR / 'diffstg_ts_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: diffstg_ts_comparison.png')

# ---- 6. Figure 3: Per-step RMSE (step 1 to step 6) ----
per_step_rmse = []
for step in range(T_p):
    err = (pred_future[:, step, :] - target_future[:, step, :]).flatten()
    per_step_rmse.append(np.sqrt((err**2).mean()))

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(1, T_p+1), per_step_rmse, color='#2563EB', alpha=0.7)
ax.set_xlabel('Prediction Horizon (hours ahead)', fontsize=11)
ax.set_ylabel('RMSE', fontsize=11)
ax.set_title('DiffSTG: Per-step RMSE (1h to 6h ahead)', fontsize=12, fontweight='bold')
ax.set_xticks(range(1, T_p+1))
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(str(FIG_DIR / 'diffstg_per_step_rmse.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: diffstg_per_step_rmse.png')

print('\nPer-step RMSE:', [f'{v:.1f}' for v in per_step_rmse])
print('Overall MAE: {:.2f}'.format(np.abs(errors).mean()))
print('Overall RMSE: {:.2f}'.format(np.sqrt((errors**2).mean())))
print('Done!')
