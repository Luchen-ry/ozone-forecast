#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DiffATGCN-PE2.3 双卡训练脚本 (PyTorch 版本)
架构: 粗预测 + 扩散精炼 (Coarse-to-Fine Diffusion)
  1) Encoder (GCN+GRU) 编码时空特征 -> coarse_fc 输出粗预测 y_coarse
  2) 训练: 对真值加噪, Decoder 以 y_coarse 为条件去噪预测 x0
  3) 推理: 从 y_coarse 加噪启动 DDIM 去噪, 扩散模型输出即为最终预测
扩散模型是预测核心主体, 粗预测仅提供初始条件.
邻接矩阵: S + T + PE = 3，PE 自适应门控 (高值区增强)
启动方式: torchrun --nproc_per_node=2 train_atgcn_pe2.3.py [args]
"""

import os
import glob
import argparse
import math
import time
import numpy as np
import pandas as pd
import random



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler 

from geopy.distance import distance as geo_distance
from scipy.spatial.distance import cdist
"""
def weighted_peak_mse_loss(preds, targets, alpha=2.0, peak_thr=0.5):
    err2 = (preds - targets) ** 2
    weight = 1.0 + alpha * (targets >= peak_thr).float()
    return (err2 * weight).mean(dim=-1).sum()
    """

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# ===================== 参数 =====================
def parse_args():
    p = argparse.ArgumentParser(description='ATGCN-PE3 Multi-GPU Training (PyTorch)')
    p.add_argument('--train_rate', type=float, default=0.8465)
    p.add_argument('--seq_len', type=int, default=12)
    p.add_argument('--pre_len', type=int, default=6)
    p.add_argument('--batch_size', type=int, default=16,
                   help='每张卡的 batch_size')
    p.add_argument('--lr', type=float, default=7e-4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--hidden_size', type=int, default=64)
    p.add_argument('--N_node', type=int, default=95)
    p.add_argument('--m', type=int, default=15,
                   help='特征维度: O3 + 14 气象变量 = 15')
    p.add_argument('--adj_units', type=int, default=3,
                   help='邻接矩阵数量: S + T + PE = 3')
    p.add_argument('--lambda_reg', type=float, default=1e-4)
    p.add_argument('--pe_threshold', type=float, default=0.75,
                   help='PE 相似图的相似度阈值')
    p.add_argument('--pe_sigma', type=float, default=0.3,
                   help='PE 相似图的高斯核 sigma 参数')
    p.add_argument('--data_dir', type=str,
                   default='/home/chenxudong/graduate/代码 2/代码/代码')
    p.add_argument('--resume_epoch', type=int, default=-1,
                   help='从第几个 epoch 继续训练，-1 表示从头开始')
    p.add_argument('--diff_steps', type=int, default=50,
                   help='扩散过程总步数 T (cosine schedule)')
    p.add_argument('--inference_steps', type=int, default=50,
                   help='DDIM 推理采样步数')
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--peak_weight', type=float, default=0.0,
                   help='高值区(峰值)样本损失权重系数，>0 时减轻峰谷被压平')
    p.add_argument('--peak_thr', type=float, default=0.4,
                   help='归一化后 O3 峰值阈值，超过则加权 (0~1)')
    p.add_argument('--lambda_temporal', type=float, default=0.0,
                   help='时间梯度损失系数，>0 时鼓励预测变化与真值同步，减轻相位滞后')
    p.add_argument('--num_samples', type=int, default=3,
                   help='推理时 DDIM 采样次数，多次取均值降低方差')
    p.add_argument('--coarse_weight', type=float, default=0.1,
                   help='粗预测分支辅助损失权重 (仅训练时引导编码器)')
    p.add_argument('--t_start_ratio', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42,
                   help='随机种子，用于多种子复现实验')
    p.add_argument('--coarse_only', type=int, default=0,
                   help='1=验证/评估时仅用粗预测(不走扩散)，用于对照实验')
    p.add_argument('--disable_pe', type=int, default=0,
                   help='1=禁用PE邻接，将PE矩阵置零，用于消融实验')
    p.add_argument('--exp_name', type=str, default='',
                   help='实验名称后缀，用于区分不同组的权重和曲线目录')
    p.add_argument('--patience', type=int, default=30,
                   help='早停耐心值: 验证RMSE多少个epoch不下降则停止训练')
    p.add_argument('--min_delta', type=float, default=0.001,
                   help='早停判定阈值: RMSE下降小于此值视为无改善')
    return p.parse_args()


MET_VARS = ['blh', 'd2m', 'fsr', 'kx', 'sp', 'ssr', 'ssrd',
            't2m', 'tcc', 'tcwv', 'tp', 'u10', 'v10', 'zust']

"""""
def weighted_peak_loss(preds, targets, alpha=3.0, peak_thr=0.75):
    err = torch.abs(preds - targets)
    weight = 1.0 + alpha * (targets >= peak_thr).float()
    return (err * weight).mean()
"""
def fill_met_missing(df: pd.DataFrame) -> pd.DataFrame:
    """对齐到 time_index 后的气象数据缺失值填充（不丢时间点）"""
    # 确保数值
    df = df.apply(pd.to_numeric, errors='coerce')

    # 按时间插值（index 必须是 DatetimeIndex）
    try:
        df = df.interpolate(method='time', limit_direction='both')
    except Exception:
        # 若 index 不是 DatetimeIndex，则退化为线性插值
        df = df.interpolate(method='linear', limit_direction='both')

    # 前后向填充兜底
    df = df.ffill().bfill()

    # 仍缺失（整列全缺等极端情况），用列中位数/0 兜底
    for c in df.columns:
        if df[c].isna().any():
            med = df[c].median()
            if not np.isfinite(med):
                med = 0.0
            df[c] = df[c].fillna(med)

    return df.fillna(0.0)

# ===================== 工具函数 =====================
def make_weight(*shape):
    p = nn.Parameter(torch.empty(*shape))
    if p.dim() >= 2:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.zeros_(p)
    return p


def init_linear(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def get_sampling_prob(initial, final, step, decay_steps):
    ratio = min(step / max(decay_steps, 1), 1.0)
    return initial * (1.0 - ratio) + final * ratio


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.clone().detach()
                       for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


def compute_metrics(preds_np, targets_np, max_value):
    preds = preds_np * max_value
    tgts = targets_np * max_value
    rmse = np.sqrt(np.mean((preds - tgts) ** 2))
    mae = np.mean(np.abs(preds - tgts))
    mask = np.abs(tgts) > 5.0
    mape = np.mean(np.abs((preds[mask] - tgts[mask]) / tgts[mask])) * 100 \
        if mask.sum() > 0 else float('inf')
    return rmse, mae, mape


# ===================== 数据加载 =====================
def load_raw_data(data_dir):
    point = pd.read_excel(os.path.join(data_dir, 'xlsx_N95/station_loc1.xlsx'))
    target_numbers = point.iloc[:, 0]
    data_folder = os.path.join(data_dir, 'data_N95')
    all_data = [pd.read_csv(os.path.join(data_folder, f))
                for f in os.listdir(data_folder)]
    O3 = np.vstack([d.loc[d['type'] == 'O3', target_numbers].values
                     for d in all_data]).T.astype(np.float32)
    cols_ok = [i for i in range(O3.shape[1])
               if not np.any(np.isnan(O3[:, i]))]
    return O3[:, cols_ok]



def get_sites(data_dir):
    station_file = os.path.join(data_dir, 'xlsx_N95/station_loc1.xlsx')
    df = pd.read_excel(station_file)
    for col in ['监测点编码', '站点编码', 'site', 'code']:
        if col in df.columns:
            return df[col].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()


def load_met_data(data_dir, sites, time_index):
    """加载气象数据并严格对齐到 O3 的真实 time_index（不因缺失丢小时）"""
    met_dir = os.path.join(data_dir, 'Var_Values_Hourly_2022')
    met_data = {}

    # time_index 需要是 DatetimeIndex 才能用 method='time' 插值
    if not isinstance(time_index, pd.DatetimeIndex):
        time_index = pd.DatetimeIndex(time_index)

    for idx, var in enumerate(MET_VARS):
        print(f"  [{idx+1}/{len(MET_VARS)}] 加载 {var} ...", end=' ', flush=True)
        pattern = os.path.join(met_dir, f"{var}*.csv")
        files = glob.glob(pattern)
        if not files:
            print("未找到")
            continue

        path = files[0]
        df = pd.read_csv(path, dtype=str)

        # 自动识别列名
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'site' in col_lower or 'code' in col_lower:
                col_map['site'] = col
            elif col_lower == 'year':
                col_map['year'] = col
            elif col_lower == 'month':
                col_map['month'] = col
            elif col_lower == 'day':
                col_map['day'] = col
            elif col_lower == 'hour':
                col_map['hour'] = col
            elif col_lower in ('value', 'val'):
                col_map['value'] = col

        missing = [k for k in ['site','year','month','day','hour','value'] if k not in col_map]
        if missing:
            print(f"列缺失{missing}，跳过")
            continue

        df['datetime'] = pd.to_datetime(
            df[col_map['year']] + '-' +
            df[col_map['month']] + '-' +
            df[col_map['day']] + '-' +
            df[col_map['hour']],
            format='%Y-%m-%d-%H',
            errors='coerce'
        )
        df[col_map['value']] = pd.to_numeric(df[col_map['value']], errors='coerce')

        mat = df.pivot_table(
            index='datetime', columns=col_map['site'],
            values=col_map['value'], aggfunc='mean'
        )

        # 先把站点列对齐，再把时间对齐到 O3 的 time_index
        mat = mat.reindex(columns=sites)
        mat = mat.reindex(time_index)

        # 关键：补齐缺失，不丢小时
        mat = fill_met_missing(mat)

        met_data[var] = mat
        print("完成")

    print(f"  [INFO] 加载了 {len(met_data)} 个气象变量")
    return met_data

def build_multi_feature_data(data_dir, data_o3, N_node):
    """
    构建多特征数据: O3 + 14 气象变量 -> (T, N, 15)
    返回: data_combined, met_stats
    """
    import os, json
    import numpy as np
    import pandas as pd
    import json as _json
    cache_path = os.path.join(data_dir, 'matrix_N95/data_combined_m15.npy')
    stats_path = os.path.join(data_dir, 'matrix_N95/met_stats.json')
    expected_shape = (data_o3.shape[1], N_node, len(MET_VARS) + 1)

    if os.path.exists(cache_path) and os.path.exists(stats_path):
        print(f'[INFO] 从缓存加载: {cache_path}')
        data_combined = np.load(cache_path)
        if data_combined.shape == expected_shape:
            with open(stats_path, 'r') as f:
                met_stats = _json.load(f)
            return data_combined, met_stats
        print('[WARN] 缓存形状不匹配，重新生成...')


    # 读取站点列表（你脚本里一般已有 get_sites）
    sites = get_sites(data_dir)
    assert len(sites) == N_node, f"sites({len(sites)}) != N_node({N_node})"

    # data_o3: (N, T) -> (T, N)
    o3 = data_o3.T.astype(np.float32)
    T = o3.shape[0]

    # ✅ 用真实 time_index（如果存在）；否则退回到 date_range
    time_index_path = os.path.join(data_dir, 'matrix_N95/time_index.npy')
    if os.path.exists(time_index_path):
        time_index = np.load(time_index_path)
        time_index = pd.to_datetime(time_index)
        assert len(time_index) == T, f"time_index({len(time_index)}) != O3_T({T})"
    else:
        time_index = pd.date_range(start='2022-01-01', periods=T, freq='h')

    # 读气象（你脚本里已有 load_met_data 和 MET_VARS）
    met_data = load_met_data(data_dir, sites, time_index)

    # --- 缺失填充：不要 drop 小时 ---
    def _fill_df(df):
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.interpolate(method='time', limit_direction='both')
        df = df.ffill().bfill()
        for c in df.columns:
            if df[c].isna().any():
                med = df[c].median()
                if not np.isfinite(med):
                    med = 0.0
                df[c] = df[c].fillna(med)
        return df.fillna(0.0)

    # 归一化统计（min-max），并保存到 met_stats
    met_stats = {}
    met_norm = {}
    for var in MET_VARS:
        df = met_data[var].copy()
        df = _fill_df(df)  # ✅ 关键：填充缺失，不丢小时

        v = df.values.astype(np.float32)
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        rng = vmax - vmin
        if rng == 0:
            rng = 1.0
        met_stats[var] = {'min': vmin, 'max': vmax}
        met_norm[var] = (v - vmin) / rng

    # 拼接 (T, N, 15)
    feats = [o3]  # (T,N)
    for var in MET_VARS:
        feats.append(met_norm[var])  # (T,N)
    data_combined = np.stack(feats, axis=-1).astype(np.float32)  # (T,N,15)

    # 保证无 NaN
    data_combined = np.nan_to_num(data_combined, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"[INFO] 合并后数据形状: {data_combined.shape}")
    print(f"[INFO] NaN 总数: {np.isnan(data_combined).sum()}")

    # 可选：缓存（避免每次都重算气象）
    cache_dir = os.path.join(data_dir, 'matrix_N95')
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, 'data_combined_m15.npy'), data_combined)
    with open(os.path.join(cache_dir, 'met_stats.json'), 'w') as f:
        json.dump(met_stats, f, ensure_ascii=False, indent=2)

    return data_combined, met_stats

def preprocess_data_multi_feature(data, time_len, train_rate, seq_len, pre_len, target_col=0):
    """
    与 main 的调用保持一致：
      preprocess_data_multi_feature(data_combined, time_len, args.train_rate, args.seq_len, args.pre_len)

    输入:
      data: (T, N, F)  F=15, 第0维为 O3
      time_len: T（你传进来的，函数内部也会校验）
      train_rate: 训练集比例（例如 0.7）
      seq_len: 历史长度（例如 12）
      pre_len: 预测步长（例如 6）
      target_col: 预测目标列索引（默认 0=O3）

    输出:
      trainX: (train_samples, seq_len, N, F)
      trainY: (train_samples, pre_len, N)
      testX : (test_samples,  seq_len, N, F)
      testY : (test_samples,  pre_len, N)
    """
    import numpy as np

    assert data.ndim == 3, f"data must be (T,N,F), got {data.shape}"
    T, N, F = data.shape
    if time_len is not None:
        assert time_len == T, f"time_len({time_len}) != data_T({T})"
    assert 0 <= target_col < F

    total = T - seq_len - pre_len + 1
    if total <= 0:
        raise ValueError(f"T={T} too small for seq_len={seq_len}, pre_len={pre_len}")

    X = np.zeros((total, seq_len, N, F), dtype=np.float32)
    Y = np.zeros((total, pre_len, N), dtype=np.float32)

    for i in range(total):
        X[i] = data[i:i + seq_len]
        Y[i] = data[i + seq_len:i + seq_len + pre_len, :, target_col]

    n_train = int(total * train_rate)
    if n_train <= 0 or n_train >= total:
        raise ValueError(f"train_rate={train_rate} makes n_train={n_train}, total={total}")

    trainX, trainY = X[:n_train], Y[:n_train]
    testX, testY = X[n_train:], Y[n_train:]

    print(f"[INFO] trainX={trainX.shape}, testX={testX.shape}")
    return trainX, trainY, testX, testY

# ===================== 邻接矩阵 =====================
def S_adjacency_matrix(data_dir, N):
    df = pd.read_excel(os.path.join(data_dir, 'xlsx_N95/station_loc1.xlsx'))
    df['loc'] = list(zip(df['纬度'], df['经度']))
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i, N):
            d = geo_distance(df['loc'][i], df['loc'][j]).km / 20
            if d < 2.5:
                v = np.exp(-d ** 2)
                A[i, j] = A[j, i] = v
    return A


def T_adjacency_matrix(data_dir, N, k=12):
    data = np.load(os.path.join(data_dir, 'matrix_N95/data.npy')).astype(np.float32)

    # 全序列相关 + top-k 稀疏化
    x = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-6)
    corr = np.corrcoef(x)

    # 去自环
    np.fill_diagonal(corr, -np.inf)

    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        idx = np.argpartition(-corr[i], k)[:k]
        A[i, idx] = corr[i, idx]

    # 对称化 + 去负相关
    A = np.maximum(A, A.T)
    A[A < 0] = 0
    return A

def PE_adjacency_matrix(data_dir, N, threshold_similarity, sigma):
    station_file = os.path.join(data_dir, 'xlsx_N95/station_loc1.xlsx')
    pe_file = os.path.join(data_dir, 'O3/PE_average_O3_results.csv')

    station_df = pd.read_excel(station_file)
    station_ids = station_df.iloc[:, 0].astype(str).tolist()

    pe_df = pd.read_csv(pe_file, index_col=0)
    pe_df.index = pe_df.index.astype(str)

    selected_scales = ['PE_scale_6h', 'PE_scale_9h', 'PE_scale_12h',
                       'PE_scale_24h', 'PE_scale_48h', 'PE_scale_72h']

    pe_features = []
    for sid in station_ids:
        if sid in pe_df.index:
            pe_features.append(pe_df.loc[sid, selected_scales].values)
        else:
            pe_features.append(np.full(len(selected_scales), np.nan))
    pe_features = np.array(pe_features, dtype=np.float32)

    for i in range(pe_features.shape[1]):
        col = pe_features[:, i]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            mean_val = np.nanmean(col)
            if np.isnan(mean_val):
                mean_val = 0.5
            pe_features[nan_mask, i] = mean_val

    pe_distances = cdist(pe_features, pe_features, metric='euclidean')
    pe_similarity = np.exp(-pe_distances ** 2 / (sigma ** 2))

    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i, N):
            if i == j:
                A[i, j] = 1.0
            elif pe_similarity[i, j] > threshold_similarity:
                A[i, j] = A[j, i] = pe_similarity[i, j]
    return A

    


# ===================== Dataset =====================
class ATGCNDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ===================== 模型组件 =====================


class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, d_model=64, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.wq = nn.Linear(in_dim, d_model)
        self.wk = nn.Linear(in_dim, d_model)
        self.wv = nn.Linear(in_dim, d_model)
        self.apply(init_linear)

    def forward(self, x):
        bs, n, _ = x.shape
        q = self.wq(x).view(bs, n, self.num_heads, self.depth).permute(0, 2, 1, 3)
        k = self.wk(x).view(bs, n, self.num_heads, self.depth).permute(0, 2, 1, 3)
        scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)
        return F.softmax(scaled, dim=-1).mean(dim=1)


class AGCRUCell(nn.Module):
    def __init__(self, in_dim, hidden_size, k=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.k = k
        self.attn = MultiHeadAttention(in_dim, d_model=64, num_heads=8)
        cat_dim = in_dim + hidden_size
        self.W_u = nn.ParameterList([make_weight(cat_dim, hidden_size) for _ in range(k)])
        self.W_r = nn.ParameterList([make_weight(cat_dim, hidden_size) for _ in range(k)])
        self.W_h = nn.ParameterList([make_weight(cat_dim, hidden_size) for _ in range(k)])

    def forward(self, xt, ht, adj):
        alpha = self.attn(xt)
        A = adj * alpha
        deg = A.sum(dim=-1).clamp(min=1e-8)
        D_inv = torch.diag_embed(deg.pow(-0.5))
        A_norm = D_inv @ A @ D_inv

        Z = torch.sigmoid(self._gc(xt, ht, A_norm, self.W_u))
        R = torch.sigmoid(self._gc(xt, ht, A_norm, self.W_r))
        h_hat = torch.tanh(self._gc(xt, R * ht, A_norm, self.W_h))
        return (1 - Z) * h_hat + Z * ht

    def _gc(self, xt, ht, A, Ws):
        xh = torch.cat([xt, ht], dim=-1)
        out = torch.zeros_like(ht)
        Ak = A.clone()
        for i in range(self.k):
            if i > 0:
                Ak = Ak @ A
            out = out + Ak @ xh @ Ws[i]
        return out


class Encoder(nn.Module):
    def __init__(self, hidden_size, adj_units, N_node, m, d_met=8):
        super().__init__()
        self.adj_units = adj_units
        self.N_node = N_node
        self.hidden_size = hidden_size
        self.d_met = d_met
        self.in_dim = 1 + d_met
        self.met_proj = nn.Sequential(
            nn.Linear(m - 1, d_met),
            nn.LayerNorm(d_met),
        )
        self.met_proj.apply(init_linear)
        self.h0 = nn.Parameter(torch.zeros(1, N_node, hidden_size))
        self.agcru1 = AGCRUCell(self.in_dim, hidden_size, k=2)
        self.agcru2 = AGCRUCell(self.in_dim, hidden_size, k=2)

    def forward(self, x, adj):
        # x: (bs, T, N, m) — channel 0 = O3, channel 1..14 = met
        bs, T, N, m = x.shape
        o3 = x[:, :, :, 0:1]                          # (bs, T, N, 1)
        met = x[:, :, :, 1:].reshape(bs * T, N, m - 1)
        met_p = self.met_proj(met).reshape(bs, T, N, self.d_met)
        x_in = torch.cat([o3, met_p], dim=-1)          # (bs, T, N, 1+d_met)

        H_all = []
        for i in range(self.adj_units):
            states = []
            h = self.h0.expand(bs, -1, -1)
            for t in range(T):
                xt = x_in[:, t, :, :]
                h = self.agcru1(xt, h, adj[i])
                h = self.agcru2(xt, h, adj[i])
                states.append(h)
            H_all.append(states)
        return H_all


class MultiModuleAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        H = hidden_size
        self.W_M = make_weight(H, H)
        self.Wp_M = make_weight(H, H)
        self.v_M = make_weight(H, 1)
        self.b_M = nn.Parameter(torch.zeros(1, H))

    def forward(self, h_list, dt):
        T = len(h_list[0])
        dt_exp = dt.unsqueeze(0).expand(T, -1, -1, -1)
        e = 0
        for h_seq in h_list:
            h_stack = torch.stack(h_seq, dim=0)
            score = torch.tanh(h_stack @ self.W_M + dt_exp @ self.Wp_M + self.b_M)
            beta = F.softmax(score @ self.v_M, dim=-1)
            e = e + beta * h_stack
        return e


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        H = hidden_size
        self.W_T = make_weight(H, H)
        self.Wp_T = make_weight(H, H)
        self.v_T = make_weight(H, 1)
        self.b_T = nn.Parameter(torch.zeros(1, H))

    def forward(self, e, dt):
        T = e.size(0)
        dt_exp = dt.unsqueeze(0).expand(T, -1, -1, -1)
        score = torch.tanh(e @ self.W_T + dt_exp @ self.Wp_T + self.b_T)
        alpha = F.softmax(score @ self.v_T, dim=-1)
        c = (alpha * e).sum(dim=0)
        return c


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class GaussianDiffusion(nn.Module):
    """Cosine schedule (Nichol & Dhariwal, 2021) — 更适合低维回归数据"""
    def __init__(self, num_steps=50):
        super().__init__()
        self.num_steps = num_steps

        s = 0.008
        steps = torch.arange(num_steps + 1, dtype=torch.float64)
        f = torch.cos((steps / num_steps + s) / (1 + s) * math.pi * 0.5) ** 2
        ab = f / f[0]
        betas = torch.clamp(1 - ab[1:] / ab[:-1], min=1e-5, max=0.999)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas', alphas.float())
        self.register_buffer('alpha_bars', alpha_bars.float())
        self.register_buffer('sqrt_ab', torch.sqrt(alpha_bars).float())
        self.register_buffer('sqrt_1m_ab', torch.sqrt(1.0 - alpha_bars).float())

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        s_ab = self.sqrt_ab[t]
        s_1m = self.sqrt_1m_ab[t]
        while s_ab.dim() < x_0.dim():
            s_ab = s_ab.unsqueeze(-1)
            s_1m = s_1m.unsqueeze(-1)
        return s_ab * x_0 + s_1m * noise, noise



class DilatedTemporalBlock(nn.Module):
    """Causal gated temporal convolution with zero-init residual output."""
    def __init__(self, hidden_size, kernel_size=2, dilation=1, dropout=0.1):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        padding = (self.kernel_size - 1) * self.dilation
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size * 2,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=padding,
        )
        self.out_proj = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout = nn.Dropout(float(dropout))
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        # x: (bs*N, H, T)
        z = self.conv(x)
        trim = (self.kernel_size - 1) * self.dilation
        if trim > 0:
            z = z[:, :, :-trim]
        a, b = z.chunk(2, dim=1)
        z = torch.tanh(a) * torch.sigmoid(b)
        z = self.dropout(self.out_proj(z))
        return x + z


class PEAwareDTR(nn.Module):
    """PE-aware dilated temporal refiner for encoder hidden sequences."""
    def __init__(self, hidden_size, adj_units, layers=3, kernel_size=2,
                 dropout=0.1, pe_fusion=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.adj_units = adj_units
        self.pe_fusion = bool(pe_fusion)
        self.blocks = nn.ModuleList([
            DilatedTemporalBlock(
                hidden_size,
                kernel_size=kernel_size,
                dilation=2 ** i,
                dropout=dropout,
            )
            for i in range(int(layers))
        ])
        if self.pe_fusion:
            self.pe_gate = nn.Sequential(
                nn.LayerNorm(hidden_size * 2),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
            )
            self.pe_proj = nn.Linear(hidden_size, hidden_size)
            self.pe_scale = nn.Parameter(torch.tensor(0.0))
            self.pe_gate.apply(init_linear)
            self.pe_proj.apply(init_linear)

    def _refine_one(self, seq):
        h = torch.stack(seq, dim=1)  # (bs, T, N, H)
        bs, steps, n_node, hidden = h.shape
        z = h.permute(0, 2, 3, 1).reshape(bs * n_node, hidden, steps)
        for block in self.blocks:
            z = block(z)
        z = z.reshape(bs, n_node, hidden, steps).permute(0, 3, 1, 2)
        return [z[:, t] for t in range(steps)]

    def forward(self, h_enc):
        refined = [self._refine_one(seq) for seq in h_enc]
        if not self.pe_fusion or len(refined) < 3:
            return refined
        pe_ctx = refined[2][-1]
        pe_ctx_proj = self.pe_proj(pe_ctx)
        fused = []
        for seq in refined:
            out_seq = []
            for h in seq:
                gate = self.pe_gate(torch.cat([h, pe_ctx], dim=-1))
                out_seq.append(h + self.pe_scale * gate * pe_ctx_proj)
            fused.append(out_seq)
        return fused


class DiffusionDecoder(nn.Module):
    def __init__(self, hidden_size, adj_units, pre_len, N_node, m, d_met=8):
        super().__init__()
        self.adj_units = adj_units
        self.pre_len = pre_len
        self.N_node = N_node
        self.hidden_size = hidden_size
        self.in_dim = 1 + d_met

        self.agcru = AGCRUCell(self.in_dim, hidden_size, k=2)
        self.mma = MultiModuleAttention(hidden_size)
        self.ta = TemporalAttention(hidden_size)

        self.W_s = make_weight(hidden_size, 1)
        self.W_t = make_weight(hidden_size, 1)
        self.W_pe = make_weight(hidden_size, 1)
        self.b_fc = nn.Parameter(torch.zeros(1))
        self.h0 = nn.Parameter(torch.zeros(1, N_node, hidden_size))
        self.W_l = make_weight(hidden_size, self.in_dim)

        self.pe_low = nn.Parameter(torch.tensor(0.1))
        self.pe_high = nn.Parameter(torch.tensor(0.5))
        self.pe_k = nn.Parameter(torch.tensor(3.0))
        self.pe_center = nn.Parameter(torch.tensor(0.4))

        self.noise_proj = nn.Sequential(
            nn.Linear(1, self.in_dim),
            nn.LayerNorm(self.in_dim),
            nn.SiLU(),
        )
        self.noise_proj.apply(init_linear)

        self.step_emb = nn.Embedding(pre_len, self.in_dim)

        self.self_cond_proj = nn.Sequential(
            nn.Linear(1, self.in_dim),
            nn.SiLU(),
        )
        self.self_cond_proj.apply(init_linear)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.h_norm = nn.LayerNorm(hidden_size)
        self.film_scale = nn.Linear(hidden_size, hidden_size)
        self.film_shift = nn.Linear(hidden_size, hidden_size)

    def forward(self, pre_len, h_enc, adj, x_noisy, t, o3_hist,
                self_cond=None):
        bs = x_noisy.size(0)
        device = x_noisy.device
        t_emb = self.time_mlp(t)                            # (bs, H)
        t_s = self.film_scale(t_emb).unsqueeze(1)           # (bs, 1, H)
        t_b = self.film_shift(t_emb).unsqueeze(1)           # (bs, 1, H)

        D = []
        h_sliced = [seq[-pre_len:] for seq in h_enc]
        for i in range(self.adj_units):
            states = []
            h = self.h0.expand(bs, -1, -1)
            for step in range(pre_len):
                ct_noisy = self.noise_proj(
                    x_noisy[:, step, :].unsqueeze(-1))      # (bs, N, in_dim)

                step_idx = torch.tensor(step, device=device, dtype=torch.long)
                ct_step = self.step_emb(step_idx).unsqueeze(0).expand(
                    bs, self.N_node, -1)                    # (bs, N, in_dim)

                if self_cond is not None:
                    ct_sc = self.self_cond_proj(
                        self_cond[:, step, :].unsqueeze(-1))
                else:
                    ct_sc = 0

                e = self.mma(h_sliced, h)
                ct_attn = (self.ta(e, h)) @ self.W_l        # (bs, N, in_dim)

                ct = ct_noisy + ct_attn + ct_step + ct_sc
                h = self.agcru(ct, h, adj[i])
                h = (1 + t_s) * self.h_norm(h) + t_b        # FiLM
                states.append(h)
            D.append(torch.stack(states, dim=0))

        y_base = D[0] @ self.W_s + D[1] @ self.W_t

        hist_sig = o3_hist.unsqueeze(0).unsqueeze(-1).expand(pre_len, -1, -1, -1)
        pe_alpha = torch.sigmoid(self.pe_k * (hist_sig - self.pe_center))
        pe_w = self.pe_low + (self.pe_high - self.pe_low) * pe_alpha
        out = y_base + pe_w * (D[2] @ self.W_pe) + self.b_fc
        return out.squeeze(-1).permute(1, 0, 2)


class DiffusionATGCN(nn.Module):
    """
    粗预测 + 扩散精炼架构。

    coarse_mode:
      final      : final encoder state -> MLP -> all horizons
      horizon    : last pre_len encoder states -> node-wise GRU
      pe_horizon : S/T/PE tails + PE context + horizon embedding -> node-wise GRU
    """
    def __init__(self, hidden_size, adj_units, pre_len, N_node, m=15, d_met=8,
                 diff_steps=50, predict_residual=False, coarse_mode='final',
                 pe_refine_gate=False, pe_gate_min=0.15, pe_gate_max=0.85,
                 pe_delta_adapter=False, pe_delta_max=0.03,
                 pe_delta_start_step=4, coarse_ms_residual=False,
                 coarse_ms_delta_max=0.03, coarse_ms_start_step=3,
                 temporal_refiner='none', dtr_layers=3,
                 dtr_kernel_size=2, dtr_dropout=0.1,
                 dtr_pe_fusion=True):
        super().__init__()
        self.pre_len = pre_len
        self.N_node = N_node
        self.adj_units = adj_units
        self.hidden_size = hidden_size
        self.predict_residual = bool(predict_residual)
        self.coarse_mode = coarse_mode
        self.pe_refine_gate = bool(pe_refine_gate)
        self.pe_gate_min = float(pe_gate_min)
        self.pe_gate_max = float(pe_gate_max)
        self.pe_delta_adapter = bool(pe_delta_adapter)
        self.pe_delta_max = float(pe_delta_max)
        self.pe_delta_start_step = int(pe_delta_start_step)
        self.coarse_ms_residual = bool(coarse_ms_residual)
        self.coarse_ms_delta_max = float(coarse_ms_delta_max)
        self.coarse_ms_start_step = int(coarse_ms_start_step)
        self.temporal_refiner = str(temporal_refiner).lower()
        self.dtr_layers = int(dtr_layers)
        self.dtr_kernel_size = int(dtr_kernel_size)
        self.dtr_dropout = float(dtr_dropout)
        self.dtr_pe_fusion = bool(dtr_pe_fusion)
        if self.pe_refine_gate and self.predict_residual:
            raise ValueError('pe_refine_gate=1 only supports predict_residual=0 in v1')
        if self.pe_delta_adapter and self.coarse_mode != 'final':
            raise ValueError('pe_delta_adapter=1 is intentionally limited to coarse_mode=final')
        if self.coarse_ms_residual and self.coarse_mode != 'final':
            raise ValueError('coarse_ms_residual=1 is intentionally limited to coarse_mode=final')
        if self.pe_delta_max < 0:
            raise ValueError(f'bad pe_delta_max={self.pe_delta_max}')
        if self.coarse_ms_delta_max < 0:
            raise ValueError(f'bad coarse_ms_delta_max={self.coarse_ms_delta_max}')
        if self.temporal_refiner not in ('none', 'dtr'):
            raise ValueError(f'unsupported temporal_refiner={temporal_refiner}')
        if self.dtr_layers < 1:
            raise ValueError(f'bad dtr_layers={self.dtr_layers}')
        if self.dtr_kernel_size < 2:
            raise ValueError(f'bad dtr_kernel_size={self.dtr_kernel_size}')
        if not (0.0 <= self.pe_gate_min <= self.pe_gate_max <= 1.0):
            raise ValueError(
                f'bad PE gate range: min={self.pe_gate_min}, max={self.pe_gate_max}'
            )

        self.encoder = Encoder(hidden_size, adj_units, N_node, m, d_met=d_met)
        self.h_refiner = None
        if self.temporal_refiner == 'dtr':
            self.h_refiner = PEAwareDTR(
                hidden_size,
                adj_units,
                layers=self.dtr_layers,
                kernel_size=self.dtr_kernel_size,
                dropout=self.dtr_dropout,
                pe_fusion=self.dtr_pe_fusion,
            )
        self.decoder = DiffusionDecoder(hidden_size, adj_units, pre_len,
                                        N_node, m, d_met=d_met)
        self.diffusion = GaussianDiffusion(diff_steps)

        if coarse_mode == 'final':
            self.coarse_fc = nn.Sequential(
                nn.Linear(hidden_size * adj_units, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, pre_len),
            )
            self.coarse_fc.apply(init_linear)
        elif coarse_mode == 'horizon':
            self.coarse_gru = nn.GRU(
                hidden_size * adj_units,
                hidden_size,
                batch_first=True,
            )
            self.coarse_step_emb = nn.Parameter(torch.zeros(pre_len, hidden_size))
            self.coarse_out = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )
            self.coarse_out.apply(init_linear)
            self._init_gru(self.coarse_gru)
        elif coarse_mode == 'pe_horizon':
            # S/T/PE horizon states + PE branch context + explicit horizon embedding.
            pe_horizon_in = hidden_size * adj_units + hidden_size * 2
            self.coarse_pe_gru = nn.GRU(
                pe_horizon_in,
                hidden_size,
                batch_first=True,
            )
            self.coarse_pe_step_emb = nn.Parameter(torch.zeros(pre_len, hidden_size))
            self.coarse_pe_out = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )
            self.coarse_pe_out.apply(init_linear)
            self._init_gru(self.coarse_pe_gru)
        else:
            raise ValueError(f"unsupported coarse_mode={coarse_mode}")

        if self.pe_refine_gate:
            self.pe_gate_step_emb = nn.Parameter(torch.zeros(pre_len, hidden_size))
            self.pe_gate_net = nn.Sequential(
                nn.LayerNorm(hidden_size * 2),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )
            self.pe_gate_net.apply(init_linear)

        if self.pe_delta_adapter:
            self.pe_delta_step_emb = nn.Parameter(torch.zeros(pre_len, hidden_size))
            self.pe_delta_hist_proj = nn.Linear(2, hidden_size)
            self.pe_delta_net = nn.Sequential(
                nn.LayerNorm(hidden_size * 3),
                nn.Linear(hidden_size * 3, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )
            self.pe_delta_hist_proj.apply(init_linear)
            self.pe_delta_net.apply(init_linear)
            last_linear = self.pe_delta_net[-1]
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

        if self.coarse_ms_residual:
            ms_in_dim = hidden_size * adj_units * 4
            self.coarse_ms_ctx_proj = nn.Sequential(
                nn.LayerNorm(ms_in_dim),
                nn.Linear(ms_in_dim, hidden_size),
                nn.GELU(),
            )
            self.coarse_ms_hist_proj = nn.Linear(3, hidden_size)
            self.coarse_ms_step_emb = nn.Parameter(torch.zeros(pre_len, hidden_size))
            self.coarse_ms_delta_net = nn.Sequential(
                nn.LayerNorm(hidden_size * 3),
                nn.Linear(hidden_size * 3, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )
            self.coarse_ms_ctx_proj.apply(init_linear)
            self.coarse_ms_hist_proj.apply(init_linear)
            self.coarse_ms_delta_net.apply(init_linear)
            last_linear = self.coarse_ms_delta_net[-1]
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    @staticmethod
    def _init_gru(gru):
        for name, param in gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _refine_h_enc(self, h_enc):
        if self.h_refiner is None:
            return h_enc
        return self.h_refiner(h_enc)

    def _stack_horizon_states(self, h_enc):
        parts = []
        for seq in h_enc:
            tail = seq[-self.pre_len:]
            if len(tail) < self.pre_len:
                tail = [seq[0]] * (self.pre_len - len(tail)) + tail
            parts.append(torch.stack(tail, dim=1))  # (bs, pre_len, N, H)
        return parts

    def _coarse_predict(self, h_enc):
        """编码器粗预测: 为扩散模型提供条件, 不作为最终输出。"""
        if self.coarse_mode == 'horizon':
            return self._coarse_predict_horizon(h_enc)
        if self.coarse_mode == 'pe_horizon':
            return self._coarse_predict_pe_horizon(h_enc)

        h_parts = [seq[-1] for seq in h_enc]
        h_cat = torch.cat(h_parts, dim=-1)       # (bs, N, H*adj_units)
        y = self.coarse_fc(h_cat)                 # (bs, N, pre_len)
        return y.permute(0, 2, 1)                 # (bs, pre_len, N)

    def _coarse_predict_horizon(self, h_enc):
        parts = self._stack_horizon_states(h_enc)
        h_seq = torch.cat(parts, dim=-1)            # (bs, pre_len, N, H*adj_units)
        bs, horizon, n_node, feat_dim = h_seq.shape
        z = h_seq.permute(0, 2, 1, 3).reshape(bs * n_node, horizon, feat_dim)
        z, _ = self.coarse_gru(z)
        z = z + self.coarse_step_emb.unsqueeze(0)
        y = self.coarse_out(z).squeeze(-1)
        return y.reshape(bs, n_node, horizon).permute(0, 2, 1)

    def _coarse_predict_pe_horizon(self, h_enc):
        parts = self._stack_horizon_states(h_enc)
        h_seq = torch.cat(parts, dim=-1)            # (bs, pre_len, N, H*adj_units)
        bs, horizon, n_node, _ = h_seq.shape

        # The third encoder branch is PE. If PE is disabled its graph is zero, so this
        # path gracefully degenerates to a horizon decoder with a near-zero PE branch.
        pe_idx = 2 if len(parts) > 2 else len(parts) - 1
        pe_context = h_enc[pe_idx][-1].unsqueeze(1).expand(-1, horizon, -1, -1)
        step_emb = self.coarse_pe_step_emb.view(1, horizon, 1, -1).expand(
            bs, horizon, n_node, -1
        )
        h_seq = torch.cat([h_seq, pe_context, step_emb], dim=-1)
        feat_dim = h_seq.shape[-1]
        z = h_seq.permute(0, 2, 1, 3).reshape(bs * n_node, horizon, feat_dim)
        z, _ = self.coarse_pe_gru(z)
        y = self.coarse_pe_out(z).squeeze(-1)
        return y.reshape(bs, n_node, horizon).permute(0, 2, 1)

    def _pe_refinement_gate(self, h_enc):
        if not self.pe_refine_gate:
            return None
        parts = self._stack_horizon_states(h_enc)
        bs, horizon, n_node, _ = parts[0].shape
        pe_idx = 2 if len(parts) > 2 else len(parts) - 1
        pe_context = h_enc[pe_idx][-1].unsqueeze(1).expand(-1, horizon, -1, -1)
        step_emb = self.pe_gate_step_emb.view(1, horizon, 1, -1).expand(
            bs, horizon, n_node, -1
        )
        gate_in = torch.cat([pe_context, step_emb], dim=-1)
        raw_gate = torch.sigmoid(self.pe_gate_net(gate_in)).squeeze(-1)
        return self.pe_gate_min + (self.pe_gate_max - self.pe_gate_min) * raw_gate

    def _apply_pe_refine_gate(self, h_enc, diff_out, y_coarse):
        if not self.pe_refine_gate:
            return diff_out
        gate = self._pe_refinement_gate(h_enc)
        return y_coarse + gate * (diff_out - y_coarse)

    def _horizon_ramp(self, start_step, device, dtype):
        steps = torch.arange(1, self.pre_len + 1, device=device, dtype=dtype)
        start_step = max(1, int(start_step))
        denom = max(float(self.pre_len - start_step + 2), 1.0)
        return ((steps - float(start_step - 2)) / denom).clamp(0.0, 1.0)

    def _coarse_ms_pool(self, h_enc, width):
        pooled = []
        for seq in h_enc:
            tail = seq[-min(width, len(seq)):]
            pooled.append(torch.stack(tail, dim=0).mean(dim=0))
        return torch.cat(pooled, dim=-1)

    def _coarse_ms_residual_correction(self, h_enc, x):
        if not self.coarse_ms_residual:
            return None
        multi_scale = torch.cat([
            self._coarse_ms_pool(h_enc, 1),
            self._coarse_ms_pool(h_enc, 3),
            self._coarse_ms_pool(h_enc, 6),
            self._coarse_ms_pool(h_enc, len(h_enc[0])),
        ], dim=-1)
        ctx_emb = self.coarse_ms_ctx_proj(multi_scale)

        last_o3 = x[:, -1, :, 0]
        trend1 = x[:, -1, :, 0] - x[:, -2, :, 0]
        trend6 = x[:, -1, :, 0] - x[:, max(0, x.shape[1] - 7), :, 0]
        hist_feat = torch.stack([last_o3, trend1, trend6], dim=-1).to(dtype=ctx_emb.dtype)
        hist_emb = self.coarse_ms_hist_proj(hist_feat)

        bs, n_node, _ = ctx_emb.shape
        ctx_seq = ctx_emb.unsqueeze(1).expand(-1, self.pre_len, -1, -1)
        hist_seq = hist_emb.unsqueeze(1).expand(-1, self.pre_len, -1, -1)
        step_seq = self.coarse_ms_step_emb.view(1, self.pre_len, 1, -1).expand(
            bs, self.pre_len, n_node, -1
        )
        delta_in = torch.cat([ctx_seq, hist_seq, step_seq], dim=-1)
        raw_delta = self.coarse_ms_delta_net(delta_in).squeeze(-1)
        ramp = self._horizon_ramp(
            self.coarse_ms_start_step, ctx_emb.device, ctx_emb.dtype
        ).view(1, self.pre_len, 1)
        return float(self.coarse_ms_delta_max) * ramp * torch.tanh(raw_delta)

    def _apply_coarse_ms_residual(self, h_enc, y_coarse, x):
        if not self.coarse_ms_residual:
            return y_coarse
        delta = self._coarse_ms_residual_correction(h_enc, x)
        return y_coarse + delta

    def _pe_delta_horizon_ramp(self, device, dtype):
        return self._horizon_ramp(self.pe_delta_start_step, device, dtype)

    def _pe_delta_adapter_correction(self, h_enc, x):
        if not self.pe_delta_adapter:
            return None
        pe_idx = 2 if len(h_enc) > 2 else len(h_enc) - 1
        pe_context = h_enc[pe_idx][-1]  # (bs, N, H)
        bs, n_node, _ = pe_context.shape
        device = pe_context.device
        dtype = pe_context.dtype

        last_o3 = x[:, -1, :, 0]
        trend1 = x[:, -1, :, 0] - x[:, -2, :, 0]
        hist_feat = torch.stack([last_o3, trend1], dim=-1).to(dtype=dtype)
        hist_emb = self.pe_delta_hist_proj(hist_feat)

        pe_seq = pe_context.unsqueeze(1).expand(-1, self.pre_len, -1, -1)
        hist_seq = hist_emb.unsqueeze(1).expand(-1, self.pre_len, -1, -1)
        step_seq = self.pe_delta_step_emb.view(1, self.pre_len, 1, -1).expand(
            bs, self.pre_len, n_node, -1
        )
        delta_in = torch.cat([pe_seq, hist_seq, step_seq], dim=-1)
        raw_delta = self.pe_delta_net(delta_in).squeeze(-1)
        ramp = self._pe_delta_horizon_ramp(device, dtype).view(1, self.pre_len, 1)
        return float(self.pe_delta_max) * ramp * torch.tanh(raw_delta)

    def _apply_pe_delta_adapter(self, h_enc, y_coarse, x):
        if not self.pe_delta_adapter:
            return y_coarse
        delta = self._pe_delta_adapter_correction(h_enc, x)
        return y_coarse + delta

    def forward(self, x, adj, y_noisy=None, t=None, self_cond=None,
                return_coarse=False, y_target=None, noise=None,
                return_diffusion=False):
        """训练: 扩散 Decoder 以粗预测为条件, 去噪预测 x0。"""
        h_enc = self.encoder(x, adj)
        h_enc = self._refine_h_enc(h_enc)
        o3_hist = x[:, -3:, :, 0].mean(dim=1)
        y_coarse = self._coarse_predict(h_enc)
        y_coarse = self._apply_coarse_ms_residual(h_enc, y_coarse, x)
        y_coarse = self._apply_pe_delta_adapter(h_enc, y_coarse, x)

        diff_target = None
        if y_target is not None:
            if self.predict_residual:
                diff_target = y_target - y_coarse.detach()
            else:
                diff_target = y_target

        if y_noisy is None:
            if diff_target is None or t is None:
                raise ValueError("y_noisy or (y_target and t) must be provided")
            y_noisy, _ = self.diffusion.q_sample(diff_target, t, noise)

        if self_cond is None:
            self_cond = y_coarse if self.pe_delta_adapter else y_coarse.detach()

        diff_out = self.decoder(self.pre_len, h_enc, adj, y_noisy, t, o3_hist,
                                self_cond=self_cond)
        if self.predict_residual:
            pred = y_coarse + diff_out
            diffusion_pred = diff_out
        else:
            pred = self._apply_pe_refine_gate(h_enc, diff_out, y_coarse)
            diffusion_pred = pred if self.pe_refine_gate else diff_out

        if return_diffusion and return_coarse:
            return pred, y_coarse, diffusion_pred, diff_target
        if return_diffusion:
            return pred, diffusion_pred, diff_target
        if return_coarse:
            return pred, y_coarse
        return pred

    @torch.no_grad()
    def sample(self, x, adj, num_steps=50, num_samples=1, t_start_ratio=0.5,
               coarse_only=False, self_condition_mode='prev_pred',
               self_condition_mix=0.5):
        """DDIM 采样: 从粗预测加噪启动, 扩散去噪输出最终预测。"""
        h_enc = self.encoder(x, adj)
        h_enc = self._refine_h_enc(h_enc)
        o3_hist = x[:, -3:, :, 0].mean(dim=1)
        bs, _, _, _ = x.shape
        device = x.device

        y_coarse = self._coarse_predict(h_enc)
        y_coarse = self._apply_coarse_ms_residual(h_enc, y_coarse, x)
        y_coarse = self._apply_pe_delta_adapter(h_enc, y_coarse, x).clamp(0, 1)

        if coarse_only:
            return y_coarse

        schedule = torch.linspace(
            self.diffusion.num_steps - 1, 0, num_steps, dtype=torch.long,
            device=device,
        )

        t_start_idx = max(0, min(len(schedule)-1, int(len(schedule) * t_start_ratio)))
        schedule_refine = schedule[t_start_idx:]
        mode = str(self_condition_mode).lower()
        if mode not in ('prev_pred', 'coarse', 'mix'):
            raise ValueError(f"unsupported self_condition_mode={self_condition_mode}")
        mix = float(max(0.0, min(1.0, self_condition_mix)))

        all_samples = []
        for _ in range(num_samples):
            t_init = schedule_refine[0].item()
            ab_init = self.diffusion.alpha_bars[t_init].to(device=device)
            noise = torch.randn_like(y_coarse)
            if self.predict_residual:
                x_t = torch.sqrt(1 - ab_init) * noise
            else:
                x_t = torch.sqrt(ab_init) * y_coarse + \
                      torch.sqrt(1 - ab_init) * noise

            self_cond = y_coarse

            for i in range(len(schedule_refine)):
                t_now = schedule_refine[i].item()
                t_batch = torch.full((bs,), t_now, device=device,
                                     dtype=torch.long)
                raw_pred = self.decoder(self.pre_len, h_enc, adj,
                                        x_t, t_batch, o3_hist,
                                        self_cond=self_cond)
                if self.predict_residual:
                    final_pred = (y_coarse + raw_pred).clamp(0, 1)
                    x0_pred = final_pred - y_coarse
                else:
                    final_pred = self._apply_pe_refine_gate(h_enc, raw_pred, y_coarse).clamp(0, 1)
                    x0_pred = final_pred

                if mode == 'coarse':
                    self_cond = y_coarse
                elif mode == 'mix':
                    self_cond = mix * y_coarse + (1.0 - mix) * final_pred
                else:
                    self_cond = final_pred

                if i + 1 < len(schedule_refine):
                    ab_now = self.diffusion.alpha_bars[t_now].to(device=device)
                    ab_next = self.diffusion.alpha_bars[
                        schedule_refine[i + 1].item()]
                    ab_next = ab_next.to(device=device)
                    eps = (x_t - torch.sqrt(ab_now) * x0_pred) / \
                          torch.sqrt(1 - ab_now).clamp(min=1e-8)
                    x_t = torch.sqrt(ab_next) * x0_pred + \
                          torch.sqrt(1 - ab_next) * eps
                else:
                    x_t = x0_pred

            if self.predict_residual:
                all_samples.append((y_coarse + x_t).clamp(0, 1))
            else:
                all_samples.append(x_t)

        if num_samples > 1:
            return torch.stack(all_samples).mean(0)
        return all_samples[0]


# ===================== 主训练流程 =====================
def main():
    # ---- 早停计数器 (需要在循环前初始化) ----
    main.patience_counter = 0

    args = parse_args()
    data_dir = args.data_dir

    # ---- DDP 初始化 ----
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    is_main = (rank == 0)

    if is_main:
        print(f'[INFO] {world_size} 个 GPU，使用 DDP 训练')
        print(f'[INFO] 每卡 batch_size={args.batch_size}, '
              f'全局 batch_size={args.batch_size * world_size}')
        print(f'[INFO] m={args.m} (O3 + 14 气象变量)')

    # ---- 加载 O3 数据 ----
    data = np.load(os.path.join(data_dir, 'matrix_N95/data.npy'))
    max_value = float(np.max(data))
    data_norm = data / max_value  # (N, T)
    num_nodes, time_len = data.shape

    if is_main:
        print(f'[INFO] N={num_nodes}, T={time_len}, max={max_value}')

    # ---- 构建多特征数据 ----
    if is_main:
        print('[INFO] 加载气象数据...')
    data_combined, met_stats = build_multi_feature_data(
        data_dir, data_norm, args.N_node)
    if is_main:
        print(f'[INFO] 合并后数据形状: {data_combined.shape}')

    # ---- 预处理 ----
    trainX, trainY, testX, testY = preprocess_data_multi_feature(
        data_combined, time_len, args.train_rate, args.seq_len, args.pre_len)
    mid = len(testX) // 2
    validX, validY = testX[:mid], testY[:mid]
    testX, testY = testX[mid:], testY[mid:]

    if is_main:
        print(f'[INFO] trainX={trainX.shape}, trainY={trainY.shape}')
        print(f'[INFO] validX={validX.shape}, testX={testX.shape}')
        save_dir = os.path.join(data_dir, 'matrix_N95_PE3')
        os.makedirs(save_dir, exist_ok=True)
        for name, arr in [('trainX', trainX), ('trainY', trainY),
                          ('validX', validX), ('validY', validY),
                          ('testX', testX), ('testY', testY)]:
            np.save(os.path.join(save_dir, f'{name}.npy'), arr)

    # ---- 邻接矩阵 (S + T + PE) ----
    S = S_adjacency_matrix(data_dir, args.N_node)
    T_mat = T_adjacency_matrix(data_dir, args.N_node)
    PE_mat = PE_adjacency_matrix(data_dir, args.N_node,
                                 threshold_similarity=args.pe_threshold,
                                 sigma=args.pe_sigma)
    if args.disable_pe:
        PE_mat = np.zeros_like(PE_mat, dtype=np.float32)
        if is_main:
            print('[INFO] PE ablation enabled: PE_matrix is set to zero.')

    if is_main:
        save_dir_m = os.path.join(data_dir, 'matrix_N95_PE3')
        np.save(os.path.join(save_dir_m, 'S_matrix.npy'), S)
        np.save(os.path.join(save_dir_m, 'T_matrix_1.npy'), T_mat)
        np.save(os.path.join(save_dir_m, 'PE_matrix.npy'), PE_mat)
        print(f'[INFO] S 非零: {np.sum(S > 0)}, T 非零: {np.sum(T_mat > 0)}, '
              f'PE 非零: {np.sum(PE_mat > 0)}')

    adj = torch.tensor(np.stack([S, T_mat, PE_mat], axis=0),
                       dtype=torch.float32).to(device)

    # ---- 数据加载器 ----
    train_ds = ATGCNDataset(trainX, trainY)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler, drop_last=True,
                              num_workers=2, pin_memory=True)

    # ---- 模型 (扩散) ----
    model = DiffusionATGCN(
        hidden_size=args.hidden_size,
        adj_units=args.adj_units,
        pre_len=args.pre_len,
        N_node=args.N_node,
        m=args.m,
        d_met=8,
        diff_steps=args.diff_steps,
    ).to(device)

    if is_main:
        print(f'[INFO] 架构: 粗预测+扩散精炼 (Coarse-to-Fine)')
        print(f'[INFO] 扩散步数 T={args.diff_steps} (cosine), '
              f'DDIM 推理步数={args.inference_steps}, '
              f'预测目标=x0, EMA={args.ema_decay}, '
              f'coarse_weight={args.coarse_weight}')

    model = DDP(model, device_ids=[local_rank], static_graph=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    warmup_epochs = 5
    def _lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return max(1e-6 / args.lr, 0.5 * (1 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    ema = EMA(model.module, decay=args.ema_decay) if is_main else None

    # ---- 恢复训练 ----
    suffix = f'_{args.exp_name}' if args.exp_name else ''
    weight_dir = os.path.join(data_dir, f'weights_N95/weights_pe3_multi_gpu{suffix}')
    if is_main:
        os.makedirs(weight_dir, exist_ok=True)

    start_epoch = 0
    total_batches = len(train_loader)

    if args.resume_epoch >= 0:
        ckpt = os.path.join(weight_dir, f'epoch_{args.resume_epoch}.pt')
        if is_main:
            print(f'[INFO] 恢复: {ckpt}')
        map_loc = {'cuda:0': f'cuda:{local_rank}'}
        model.module.load_state_dict(torch.load(ckpt, map_location=map_loc))
        start_epoch = args.resume_epoch + 1

    # ---- 训练循环 (粗预测 + 扩散精炼) ----
    epoch_losses, rmse_list, mae_list = [], [], []
    diff = model.module.diffusion

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0

        for step, (bx, by) in enumerate(train_loader):
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            bs = bx.size(0)

            t_diff = torch.randint(0, diff.num_steps, (bs,), device=device)
            noise = torch.randn_like(by)
            y_noisy, _ = diff.q_sample(by, t_diff, noise)

            x0_pred, y_coarse = model(bx, adj, y_noisy, t_diff,
                                      self_cond=None, return_coarse=True)

            if random.random() < 0.5:
                with torch.no_grad():
                    sc = model(bx, adj, y_noisy, t_diff).detach()
                x0_pred = model(bx, adj, y_noisy, t_diff, self_cond=sc)

            err2 = (x0_pred - by) ** 2
            if args.peak_weight > 0:
                pw = 1.0 + args.peak_weight * (by >= args.peak_thr).float()
                err2 = err2 * pw
            diff_loss = err2.mean(dim=-1).sum()

            coarse_loss = ((y_coarse - by) ** 2).mean(dim=-1).sum()
            loss = diff_loss + args.coarse_weight * coarse_loss

            if args.lambda_temporal > 0:
                grad_pred = x0_pred[:, 1:, :] - x0_pred[:, :-1, :]
                grad_true = by[:, 1:, :] - by[:, :-1, :]
                temporal_loss = ((grad_pred - grad_true) ** 2).mean(dim=-1).sum()
                loss = loss + args.lambda_temporal * temporal_loss

            l2_loss = args.lambda_reg * sum(
                p.pow(2).sum() for p in model.parameters() if p.requires_grad)
            loss = loss + l2_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if ema is not None:
                ema.update(model.module)

            running_loss += loss.item()
            if is_main and (step + 1) % 20 == 0:
                print(f'  Epoch {epoch + 1} Batch {step + 1}/{total_batches}, '
                      f'Loss={loss.item():.6f} '
                      f'(diff={diff_loss.item():.4f}, '
                      f'coarse={coarse_loss.item():.4f})')

        scheduler.step()
        avg_loss = running_loss / total_batches
        epoch_losses.append(avg_loss)

        # ---- 验证 (EMA 权重 + DDIM 采样) ----
        rmse = mae = mape = float('nan')
        if is_main:
            ema.apply(model.module)
            model.eval()
            with torch.no_grad():
                vx = torch.from_numpy(validX).float().to(device)
                vy_np = validY
                vpreds = model.module.sample(
                    vx, adj, num_steps=args.inference_steps,
                    num_samples=args.num_samples,
                    t_start_ratio=args.t_start_ratio,
                    coarse_only=bool(args.coarse_only))
                rmse, mae, mape = compute_metrics(
                    vpreds.cpu().numpy(), vy_np, max_value)
            ema.restore(model.module)

            rmse_list.append(rmse)
            mae_list.append(mae)

            elapsed = time.time() - t0
            print(f'Epoch {epoch + 1}/{args.epochs} | '
                  f'Loss={avg_loss:.6f} | RMSE={rmse:.4f} | '
                  f'MAE={mae:.4f} | MAPE={mape:.2f}% | {elapsed:.1f}s')

            # ---- 早停机制 ----
            best_rmse = float('inf')
            patience_counter = 0
            if epoch > 0 and len(rmse_list) > 1:
                best_rmse = min(rmse_list[:-1])
                if rmse < best_rmse - args.min_delta:
                    patience_counter = 0
                    print(f'  *** New best RMSE: {rmse:.4f} ***')
                else:
                    patience_counter = main.patience_counter + 1
                    if patience_counter >= args.patience:
                        print(f'\n[EARLY STOP] Patience={args.patience} reached, '
                              f'RMSE not improved. Best RMSE was {best_rmse:.4f}')
                        break
            main.patience_counter = patience_counter

            # 保存 EMA 权重 (推理用)
            ema.apply(model.module)
            torch.save(model.module.state_dict(),
                       os.path.join(weight_dir, f'epoch_{epoch}_ema.pt'))
            ema.restore(model.module)
            # 保存原始权重 (续训用)
            torch.save(model.module.state_dict(),
                       os.path.join(weight_dir, f'epoch_{epoch}.pt'))

        dist.barrier()

    # ---- 保存训练曲线 ----
    if is_main:
        curve_dir = os.path.join(data_dir, f'matrix_N95_PE3{suffix}')
        os.makedirs(curve_dir, exist_ok=True)
        np.save(os.path.join(curve_dir, 'epoch_loss_multi.npy'),
                np.array(epoch_losses))
        np.save(os.path.join(curve_dir, 'rmse_val.npy'), np.array(rmse_list))
        np.save(os.path.join(curve_dir, 'mae_val.npy'), np.array(mae_list))
        print('\n[INFO] 训练完成!')
        print(f'[INFO] 权重: {weight_dir}')
        print(f'[INFO] 曲线: {curve_dir}')

    dist.destroy_process_group()


if __name__ == '__main__':
    main()