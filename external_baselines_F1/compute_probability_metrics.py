"""
DiffSTG — 概率预测指标（CRPS / MIS）& 置信区间图
===================================================
加载已保存模型，用 n_samples=50 做 multi-sample 推理，
计算 CRPS、MIS(80%/90%/95%)，画置信区间阴影图。

Usage:
  conda activate pedw
  cd external_baselines_F1
  python compute_probability_metrics.py
"""
import sys, os, torch, pickle, argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'DiffSTG'))

from utils.eval import Metric
from utils.common_utils import dir_check, to_device
from algorithm.dataset import CleanDataset, TrafficDataset
from algorithm.diffstg.model import DiffSTG

def crps_gaussian(y_true, y_pred_samples):
    """Gaussian CRPS"""
    from scipy.stats import norm
    mu = y_pred_samples.mean(axis=0)
    sigma = y_pred_samples.std(axis=0) + 1e-8
    z = (mu - y_true) / sigma
    return float((sigma * (z * (2*norm.cdf(z)-1) + 2*norm.pdf(z) - 1/np.sqrt(np.pi))).mean())

def interval_score(y_true, y_pred_samples, alpha=0.1):
    """MIS for (1-alpha)*100% interval"""
    lo = np.percentile(y_pred_samples, alpha/2*100, axis=0)
    hi = np.percentile(y_pred_samples, (1-alpha/2)*100, axis=0)
    w = hi - lo
    p_lo = 2/alpha * (lo - y_true) * (y_true < lo)
    p_hi = 2/alpha * (y_true - hi) * (y_true > hi)
    return float((w + p_lo + p_hi).mean())

# ---- 配置 ----
DATA_NAME = 'AIR_N95'       # 改成 AIR_N95_PM25 等跑其他污染物
T_H, T_P = 24, 6
N_SAMPLES = 50
SEED = 42

# ---- 匹配 model 文件名 ----
pattern = f'N-50+T_h-{T_H}+T_p-{T_P}+epsilon_theta-UGnet.dm4stg'
forecast_dir = ROOT / 'DiffSTG' / 'output' / 'forecast'
model_dir = ROOT / 'DiffSTG' / 'output' / 'model'

# 找到对应 model 文件
model_files = sorted(model_dir.glob(f'*{DATA_NAME}*{pattern}'))
if not model_files:
    print(f'未找到模型: *{DATA_NAME}*{pattern}')
    sys.exit(1)
model_path = model_files[0]
print(f'Model: {model_path.name}')

# ---- 构建 config（复用 train.py 逻辑） ----
from easydict import EasyDict as edict

ws = str(ROOT / 'DiffSTG')
config = edict()
config.PATH_MOD = ws + '/output/model/'
config.PATH_LOG = ws + '/output/log/'
config.PATH_FORECAST = ws + '/output/forecast/'
config.data = edict()
config.data.name = DATA_NAME
config.data.path = ws + '/data/dataset/'
config.data.feature_file = config.data.path + DATA_NAME + '/flow.npy'
config.data.spatial = config.data.path + DATA_NAME + '/adj.npy'
config.data.num_recent = 1
config.data.num_features = 1
config.data.num_vertices = 95
config.data.points_per_hour = 1

total_len = np.load(config.data.feature_file).shape[0]
config.data.val_start_idx = int(total_len * 0.6)
config.data.test_start_idx = int(total_len * 0.8)

config.model = edict()
config.model.T_p = T_P
config.model.T_h = T_H
config.model.V = 95
config.model.F = 1
config.model.week_len = 7
config.model.day_len = 24
config.model.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.model.d_h = 32
config.model.N = 50
config.model.sample_steps = 50
config.model.epsilon_theta = 'UGnet'
config.model.is_label_condition = True
config.model.beta_end = 0.1
config.model.beta_schedule = 'quad'
config.model.sample_strategy = 'ddim_multi'
config.model.channel_multipliers = [1, 2]
config.model.supports_len = 2
config.model.C = 32
config.model.n_channels = 32
config.n_samples = N_SAMPLES
config.batch_size = 4
config.is_test = False
config.device = config.model.device

print(f'Device: {config.device}')

# ---- 加载数据 ----
clean_data = CleanDataset(config)
config.model.A = clean_data.adj

# ---- 加载模型 ----
model = torch.load(str(model_path), map_location=config.device, weights_only=False)
model.set_ddim_sample_steps(40)
model.set_sample_strategy('ddim_multi')
model = model.to(config.device)
model.eval()
print('Model loaded.')

# ---- 构造 test_loader ----
test_dataset = TrafficDataset(clean_data, (config.data.test_start_idx + T_P, -1), config)
test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False)

# ---- 多采样推理 ----
print(f'\nRunning {N_SAMPLES}-sample inference on {len(test_loader)} test batches...')
# 只取前 N 个 batch 避免太久
max_batches = min(300, len(test_loader))  # 加大覆盖
y_pred_all, y_true_all = [], []

for i, batch in enumerate(test_loader):
    if i >= max_batches:
        break
    if i % 20 == 0:
        print(f'  {i}/{max_batches}')

    future, history, pos_w, pos_d = to_device(batch, config.device)
    x = torch.cat((history, future), dim=1).to(config.device)
    x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(config.device)
    x = x.transpose(1, 3)
    x_masked = x_masked.transpose(1, 3)

    with torch.no_grad():
        x_hat = model((x_masked, pos_w, pos_d), N_SAMPLES)  # (B, n_samples, F, V, T)

    # 反归一化
    x = clean_data.reverse_normalization(x).detach()
    x_hat = clean_data.reverse_normalization(x_hat).detach()

    f_true = x[:, :, :, -T_P:].cpu().numpy()        # (B, F, V, T_p)
    f_pred = x_hat[:, :, :, :, -T_P:].cpu().numpy()  # (B, n_samples, F, V, T_p)

    y_true_all.append(f_true[0, 0, :, :])     # (V, T_p)
    y_pred_all.append(f_pred[0, :, 0, :, :])  # (n_samples, V, T_p)

y_true_all = np.stack(y_true_all)   # (B_eval, V, T_p)
y_pred_all = np.stack(y_pred_all)   # (B_eval, n_samples, V, T_p)

print(f'Evaluated {len(y_true_all)} batches.')
print(f'y_true: {y_true_all.shape}, y_pred: {y_pred_all.shape}')

# ---- 计算 CRPS ----
crps_vals = []
for b in range(len(y_true_all)):
    crps_vals.append(crps_gaussian(y_true_all[b], y_pred_all[b]))
crps_mean = np.mean(crps_vals)
print(f'\nCRPS = {crps_mean:.4f}')

# ---- 计算 MIS ----
for alpha, label in [(0.2, '80%'), (0.1, '90%'), (0.05, '95%')]:
    vals = [interval_score(y_true_all[b], y_pred_all[b], alpha) for b in range(len(y_true_all))]
    print(f'MIS {label} = {np.mean(vals):.4f}')

# ---- 置信区间图 ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 选第一个 batch、第一个站点
y_samples = y_pred_all[0, :, 0, :]   # (n_samples, T_p)
y_true = y_true_all[0, 0, :]           # (T_p,)
mu = y_samples.mean(axis=0)
sigma = y_samples.std(axis=0)
t = np.arange(1, T_P + 1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, y_true, 'k-o', lw=2, ms=4, label='Ground Truth')
ax.plot(t, mu, 'b-', lw=2, label='Predicted Mean')
ax.fill_between(t, mu-sigma, mu+sigma, alpha=0.3, color='blue', label='68% CI (±1σ)')
ax.fill_between(t, mu-2*sigma, mu+2*sigma, alpha=0.15, color='blue', label='95% CI (±2σ)')
ax.set_xlabel('Forecast Horizon (h)')
ax.set_ylabel(f'{DATA_NAME} Concentration')
ax.set_title(f'DiffSTG Probabilistic Forecast\nCRPS={crps_mean:.3f}  N_samples={N_SAMPLES}  Station#0')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / f'confidence_{DATA_NAME}_p{T_P}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\n[OK] {FIG_DIR}/confidence_{DATA_NAME}_p{T_P}.png')

# ---- 保存指标 ----
result_csv = ROOT / 'probability_metrics.csv'
with open(result_csv, 'w') as f:
    f.write('Experiment,CRPS,MIS_80,MIS_90,MIS_95\n')
    mis_80 = np.mean([interval_score(y_true_all[b], y_pred_all[b], 0.2) for b in range(len(y_true_all))])
    mis_90 = np.mean([interval_score(y_true_all[b], y_pred_all[b], 0.1) for b in range(len(y_true_all))])
    mis_95 = np.mean([interval_score(y_true_all[b], y_pred_all[b], 0.05) for b in range(len(y_true_all))])
    f.write(f'{DATA_NAME}_p{T_P},{crps_mean:.4f},{mis_80:.4f},{mis_90:.4f},{mis_95:.4f}\n')
print(f'[OK] {result_csv}')

print('\nDone!')
