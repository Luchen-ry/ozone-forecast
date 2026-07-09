#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Leakage-safe ATGCN-PE3 training.

Key differences from train_atgcn_pe3.py:
1. Split raw time first, then create sliding windows inside each split.
2. Fit O3 scaling and meteorological min/max on the training slice only.
3. Build T/PE graphs from the training slice only.
4. Save outputs to dedicated noleak directories to avoid mixing with old runs.
"""

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from contextlib import nullcontext
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from train_atgcn_pe3 import (
    MET_VARS,
    ATGCNDataset,
    DiffusionATGCN,
    EMA,
    S_adjacency_matrix,
    compute_metrics,
    fill_met_missing,
    get_sites,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="Leakage-safe ATGCN-PE3 training")
    p.add_argument("--train_rate", type=float, default=0.8465)
    p.add_argument("--seq_len", type=int, default=12)
    p.add_argument("--pre_len", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--N_node", type=int, default=95)
    p.add_argument("--m", type=int, default=15)
    p.add_argument("--adj_units", type=int, default=3)
    p.add_argument("--lambda_reg", type=float, default=1e-4)
    p.add_argument("--pe_threshold", type=float, default=0.75)
    p.add_argument("--pe_sigma", type=float, default=0.3)
    p.add_argument(
        "--data_dir",
        type=str,
        default="/home/chenxudong/graduate/代码 2/代码/代码",
    )
    p.add_argument("--diff_steps", type=int, default=50)
    p.add_argument("--inference_steps", type=int, default=50)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--peak_weight", type=float, default=0.0)
    p.add_argument("--peak_thr", type=float, default=0.4)
    p.add_argument("--lambda_temporal", type=float, default=0.0)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument(
        "--horizon_weights",
        type=str,
        default="",
        help="Comma-separated per-horizon loss weights, e.g. 1,1,1.1,1.2,1.35,1.5",
    )
    p.add_argument(
        "--coarse_horizon_weights",
        type=str,
        default="",
        help="Optional separate coarse-loss horizon weights. Empty means reuse --horizon_weights.",
    )
    p.add_argument("--predict_residual", type=int, default=0)
    p.add_argument("--coarse_mode", type=str, default="final", choices=("final", "horizon", "pe_horizon"))
    p.add_argument("--pe_refine_gate", type=int, default=0)
    p.add_argument("--pe_gate_min", type=float, default=0.15)
    p.add_argument("--pe_gate_max", type=float, default=0.85)
    p.add_argument("--pe_adaptive_loss", type=int, default=0)
    p.add_argument("--pe_loss_weight", type=float, default=0.20)
    p.add_argument("--pe_loss_start_step", type=int, default=4)
    p.add_argument("--pe_delta_adapter", type=int, default=0)
    p.add_argument("--pe_delta_max", type=float, default=0.03)
    p.add_argument("--pe_delta_start_step", type=int, default=4)
    p.add_argument("--coarse_ms_residual", type=int, default=0)
    p.add_argument("--coarse_ms_delta_max", type=float, default=0.03)
    p.add_argument("--coarse_ms_start_step", type=int, default=3)
    p.add_argument("--temporal_refiner", type=str, default="none", choices=("none", "dtr"))
    p.add_argument("--dtr_layers", type=int, default=3)
    p.add_argument("--dtr_kernel_size", type=int, default=2)
    p.add_argument("--dtr_dropout", type=float, default=0.1)
    p.add_argument("--dtr_pe_fusion", type=int, default=1)
    p.add_argument("--warm_start_ckpt", type=str, default="")
    p.add_argument("--freeze_backbone", type=int, default=0)
    p.add_argument("--skip_nonfinite_loss", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--self_condition_mode", type=str, default="prev_pred", choices=("prev_pred", "coarse", "mix"))
    p.add_argument("--self_condition_mix", type=float, default=0.5)
    p.add_argument("--coarse_weight", type=float, default=0.1)
    p.add_argument("--coarse_weight_end", type=float, default=-1.0)
    p.add_argument("--coarse_decay_epochs", type=int, default=0)
    p.add_argument("--t_start_ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--coarse_only", type=int, default=0)
    p.add_argument("--disable_pe", type=int, default=0)
    p.add_argument("--exp_name", type=str, default="")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--min_delta", type=float, default=0.001)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--temporal_topk", type=int, default=12)
    p.add_argument("--temporal_stride", type=int, default=1)
    p.add_argument("--pe_dim", type=int, default=3)
    p.add_argument("--pe_delay", type=int, default=1)
    p.add_argument("--pe_scales", type=str, default="6,9,12,24,48,72")
    p.add_argument("--pe_window_step", type=int, default=1)
    p.add_argument("--max_train_windows", type=int, default=0)
    p.add_argument("--max_valid_windows", type=int, default=0)
    p.add_argument("--max_test_windows", type=int, default=0)
    p.add_argument("--save_predictions", type=int, default=1)
    p.add_argument("--save_train_arrays", type=int, default=0)
    p.add_argument("--use_met_cache", type=int, default=1)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--resume_epoch", type=int, default=-1)
    p.add_argument("--ddp_timeout_minutes", type=int, default=180)
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--eval_seed", type=int, default=-1)
    p.add_argument("--eval_inference_steps", type=int, default=0)
    p.add_argument("--eval_num_samples", type=int, default=0)
    p.add_argument("--eval_t_start_ratio", type=float, default=-1.0)
    return p.parse_args()


def resolve_device(device_arg):
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def parse_pe_scales(scales_str):
    scales = []
    for item in scales_str.split(","):
        item = item.strip()
        if item:
            scales.append(int(item))
    if not scales:
        raise ValueError("pe_scales cannot be empty")
    return scales


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_time_index(data_dir, time_len):
    path = os.path.join(data_dir, "matrix_N95", "time_index.npy")
    if os.path.exists(path):
        time_index = pd.to_datetime(np.load(path))
        if len(time_index) != time_len:
            raise ValueError(
                f"time_index length mismatch: {len(time_index)} vs {time_len}"
            )
        return time_index
    return pd.date_range(start="2022-01-01", periods=time_len, freq="h")


def load_met_data_raw(data_dir, sites, time_index, use_cache=True, verbose=True):
    met_dir = os.path.join(data_dir, "Var_Values_Hourly_2022")
    cache_path = os.path.join(data_dir, "matrix_N95", "met_raw_aligned_cache.npz")
    if not isinstance(time_index, pd.DatetimeIndex):
        time_index = pd.DatetimeIndex(time_index)

    if use_cache and os.path.exists(cache_path):
        cache = np.load(cache_path)
        if all(var in cache for var in MET_VARS):
            if verbose:
                print(f"[INFO] loading met cache: {cache_path}", flush=True)
            return {
                var: pd.DataFrame(cache[var], index=time_index, columns=sites)
                for var in MET_VARS
            }

    met_data = {}
    for idx, var in enumerate(MET_VARS, start=1):
        if verbose:
            print(f"[INFO] loading met {idx}/{len(MET_VARS)}: {var}", flush=True)
        matches = sorted(
            [p for p in os.listdir(met_dir) if p.startswith(var) and p.endswith(".csv")]
        )
        if not matches:
            raise FileNotFoundError(f"missing meteorological file for {var}")

        path = os.path.join(met_dir, matches[0])
        df = pd.read_csv(path, dtype=str)

        col_map = {}
        for col in df.columns:
            name = col.lower()
            if "site" in name or "code" in name:
                col_map["site"] = col
            elif name == "year":
                col_map["year"] = col
            elif name == "month":
                col_map["month"] = col
            elif name == "day":
                col_map["day"] = col
            elif name == "hour":
                col_map["hour"] = col
            elif name in ("value", "val"):
                col_map["value"] = col

        required = ("site", "year", "month", "day", "hour", "value")
        missing = [k for k in required if k not in col_map]
        if missing:
            raise ValueError(f"{var} missing columns: {missing}")

        df["datetime"] = pd.to_datetime(
            df[col_map["year"]]
            + "-"
            + df[col_map["month"]]
            + "-"
            + df[col_map["day"]]
            + "-"
            + df[col_map["hour"]],
            format="%Y-%m-%d-%H",
            errors="coerce",
        )
        df[col_map["value"]] = pd.to_numeric(df[col_map["value"]], errors="coerce")

        mat = df.pivot_table(
            index="datetime",
            columns=col_map["site"],
            values=col_map["value"],
            aggfunc="mean",
        )
        mat = mat.reindex(columns=sites)
        mat = mat.reindex(time_index)
        mat = mat.apply(pd.to_numeric, errors="coerce")
        met_data[var] = mat

    if use_cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path,
            **{var: met_data[var].values.astype(np.float32) for var in MET_VARS},
        )
        if verbose:
            print(f"[INFO] saved met cache: {cache_path}", flush=True)

    return met_data


def minmax_fit(arr):
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    rng = vmax - vmin
    if not np.isfinite(rng) or rng == 0:
        rng = 1.0
    return vmin, vmax, rng


def normalize_with_stats(arr, vmin, rng):
    out = (arr - vmin) / rng
    return out.astype(np.float32)


def make_windows(data, seq_len, pre_len, target_col=0):
    if data.ndim != 3:
        raise ValueError(f"expected (T,N,F), got {data.shape}")
    total = data.shape[0] - seq_len - pre_len + 1
    if total <= 0:
        raise ValueError(
            f"not enough timesteps: T={data.shape[0]}, seq_len={seq_len}, pre_len={pre_len}"
        )

    X = np.zeros((total, seq_len, data.shape[1], data.shape[2]), dtype=np.float32)
    Y = np.zeros((total, pre_len, data.shape[1]), dtype=np.float32)
    for i in range(total):
        X[i] = data[i : i + seq_len]
        Y[i] = data[i + seq_len : i + seq_len + pre_len, :, target_col]
    return X, Y


def subsample_windows(X, Y, max_windows, split_name, verbose=True):
    if max_windows <= 0 or len(X) <= max_windows:
        return X, Y
    idx = np.linspace(0, len(X) - 1, max_windows, dtype=int)
    if verbose:
        print(f"[INFO] {split_name}: downsample windows {len(X)} -> {len(idx)}", flush=True)
    return X[idx], Y[idx]


def build_raw_time_slices(time_len, train_rate):
    train_end = int(time_len * train_rate)
    if train_end <= 0 or train_end >= time_len:
        raise ValueError(f"bad train_rate={train_rate} for time_len={time_len}")
    remain = time_len - train_end
    valid_len = remain // 2
    test_len = remain - valid_len
    if valid_len <= 0 or test_len <= 0:
        raise ValueError("validation/test time ranges are empty")
    return {
        "train": slice(0, train_end),
        "valid": slice(train_end, train_end + valid_len),
        "test": slice(train_end + valid_len, time_len),
    }


def describe_time_slices(time_index, slices):
    desc = {}
    for name, sl in slices.items():
        desc[name] = {
            "start_idx": int(sl.start),
            "end_idx": int(sl.stop - 1),
            "start_time": str(time_index[sl.start]),
            "end_time": str(time_index[sl.stop - 1]),
            "length": int(sl.stop - sl.start),
        }
    return desc


def compute_permutation_entropy(series, dim=3, delay=1):
    series = np.asarray(series, dtype=float)
    if np.isnan(series).any():
        return np.nan
    if len(series) < (dim - 1) * delay + 1:
        return np.nan

    embedded = np.array(
        [series[i : i + dim * delay : delay] for i in range(len(series) - (dim - 1) * delay)]
    )
    patterns = [tuple(np.argsort(x)) for x in embedded]
    counts = Counter(patterns)
    total = sum(counts.values())
    if total == 0:
        return np.nan

    probs = np.array([value / total for value in counts.values()], dtype=float)
    pe = -np.sum(probs * np.log2(probs))
    pe /= math.log2(math.factorial(dim))
    return pe


def interpolate_series(series):
    series = pd.Series(np.asarray(series, dtype=float))
    series = series.interpolate(limit_direction="both")
    series = series.ffill().bfill()
    if series.isna().any():
        med = series.median()
        if not np.isfinite(med):
            med = 0.0
        series = series.fillna(med)
    return series.to_numpy(dtype=np.float32)


def average_sliding_pe(series, window_size, dim=3, delay=1, step=1):
    values = []
    step = max(1, int(step))
    for end in range(window_size - 1, len(series), step):
        start = end - window_size + 1
        pe = compute_permutation_entropy(series[start : end + 1], dim=dim, delay=delay)
        if np.isfinite(pe):
            values.append(pe)
    if not values:
        return np.nan
    return float(np.mean(values))


def build_pe_feature_matrix(o3_raw, scales, dim=3, delay=1, step=1):
    num_nodes = o3_raw.shape[0]
    features = np.zeros((num_nodes, len(scales)), dtype=np.float32)
    print(
        f"[INFO] building PE features: nodes={num_nodes}, scales={scales}, step={step}",
        flush=True,
    )
    for node in range(num_nodes):
        series = interpolate_series(o3_raw[node])
        for col, scale in enumerate(scales):
            avg_pe = average_sliding_pe(
                series,
                window_size=scale,
                dim=dim,
                delay=delay,
                step=step,
            )
            if not np.isfinite(avg_pe):
                avg_pe = 0.5
            features[node, col] = avg_pe
        if (node + 1) % 20 == 0 or node + 1 == num_nodes:
            print(f"[INFO] PE feature progress: {node + 1}/{num_nodes}", flush=True)
    return features


def T_adjacency_from_o3(o3_raw, N, k=12, time_stride=1):
    time_stride = max(1, int(time_stride))
    if time_stride > 1:
        print(f"[INFO] temporal graph stride={time_stride}", flush=True)
        o3_raw = o3_raw[:, ::time_stride]
    x = (o3_raw - o3_raw.mean(axis=1, keepdims=True)) / (
        o3_raw.std(axis=1, keepdims=True) + 1e-6
    )
    corr = np.corrcoef(x)
    np.fill_diagonal(corr, -np.inf)

    topk = min(k, N - 1)
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        idx = np.argpartition(-corr[i], topk)[:topk]
        A[i, idx] = corr[i, idx]
    A = np.maximum(A, A.T)
    A[A < 0] = 0
    return A


def PE_adjacency_from_o3(
    o3_raw,
    N,
    threshold_similarity,
    sigma,
    scales,
    dim=3,
    delay=1,
    step=1,
):
    pe_features = build_pe_feature_matrix(
        o3_raw,
        scales,
        dim=dim,
        delay=delay,
        step=step,
    )
    pe_distances = cdist(pe_features, pe_features, metric="euclidean")
    pe_similarity = np.exp(-(pe_distances ** 2) / (sigma ** 2))

    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i, N):
            if i == j:
                A[i, j] = 1.0
            elif pe_similarity[i, j] > threshold_similarity:
                A[i, j] = A[j, i] = pe_similarity[i, j]
    return A


def prepare_noleak_data(args, raw_o3, time_index, met_raw):
    time_slices = build_raw_time_slices(raw_o3.shape[1], args.train_rate)
    split_desc = {
        "mode": "split_raw_time_first",
        "time_slices": describe_time_slices(time_index, time_slices),
    }

    train_raw = raw_o3[:, time_slices["train"]]
    max_value = float(np.max(train_raw))
    if not np.isfinite(max_value) or max_value <= 0:
        max_value = 1.0

    o3_splits = {}
    for name, sl in time_slices.items():
        o3_splits[name] = (raw_o3[:, sl].T / max_value).astype(np.float32)

    met_norm_splits = {name: [] for name in time_slices}
    met_stats = {}
    for var in MET_VARS:
        train_df = fill_met_missing(met_raw[var].iloc[time_slices["train"]].copy())
        train_values = train_df.values.astype(np.float32)
        vmin, vmax, rng = minmax_fit(train_values)
        met_stats[var] = {"min": vmin, "max": vmax}

        for name, sl in time_slices.items():
            split_df = fill_met_missing(met_raw[var].iloc[sl].copy())
            split_values = split_df.values.astype(np.float32)
            met_norm_splits[name].append(normalize_with_stats(split_values, vmin, rng))

    split_sets = {}
    for name in ("train", "valid", "test"):
        feats = [o3_splits[name]] + met_norm_splits[name]
        split_data = np.stack(feats, axis=-1).astype(np.float32)
        split_data = np.nan_to_num(split_data, nan=0.0, posinf=0.0, neginf=0.0)
        split_sets[name] = make_windows(split_data, args.seq_len, args.pre_len)

    scale_stats = {"o3_train_max": float(max_value)}
    return split_sets, max_value, met_stats, split_desc, scale_stats, time_slices


def maybe_limit_split_windows(split_sets, args, verbose=True):
    limits = {
        "train": args.max_train_windows,
        "valid": args.max_valid_windows,
        "test": args.max_test_windows,
    }
    out = {}
    for name, (X, Y) in split_sets.items():
        out[name] = subsample_windows(X, Y, limits[name], name, verbose=verbose)
    return out


def resolve_coarse_weight(args, epoch):
    end = args.coarse_weight if args.coarse_weight_end < 0 else args.coarse_weight_end
    if args.coarse_decay_epochs <= 0:
        return float(args.coarse_weight)
    ratio = min(max(epoch, 0) / float(max(args.coarse_decay_epochs, 1)), 1.0)
    return float(args.coarse_weight + (end - args.coarse_weight) * ratio)


def parse_horizon_weight_list(value, pre_len, name):
    value = str(value or "").strip()
    if not value:
        return None
    try:
        weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated floats: {value}") from exc
    if len(weights) != pre_len:
        raise ValueError(f"{name} length must equal pre_len={pre_len}, got {len(weights)}")
    if any((not np.isfinite(w)) or w <= 0 for w in weights):
        raise ValueError(f"{name} values must be finite and > 0: {weights}")
    return weights


def resolve_horizon_weights(args):
    horizon_weights = parse_horizon_weight_list(
        args.horizon_weights,
        args.pre_len,
        "horizon_weights",
    )
    if str(args.coarse_horizon_weights or "").strip():
        coarse_horizon_weights = parse_horizon_weight_list(
            args.coarse_horizon_weights,
            args.pre_len,
            "coarse_horizon_weights",
        )
    else:
        coarse_horizon_weights = horizon_weights
    return horizon_weights, coarse_horizon_weights


def weighted_horizon_loss(err2, horizon_weights=None, extra_weights=None):
    if extra_weights is not None:
        extra_weights = extra_weights.to(device=err2.device, dtype=err2.dtype)
        err2 = err2 * extra_weights.view(1, extra_weights.shape[0], extra_weights.shape[1])
    step_loss = err2.mean(dim=-1)
    if horizon_weights is not None:
        step_loss = step_loss * horizon_weights.view(1, -1)
    return step_loss.sum()


def load_pe_node_weights(data_dir, sites, scales):
    pe_path = os.path.join(data_dir, "O3", "PE_average_O3_results.csv")
    if not os.path.exists(pe_path):
        raise FileNotFoundError(f"PE adaptive loss needs {pe_path}")
    pe_df = pd.read_csv(pe_path, index_col=0)
    pe_df.index = pe_df.index.astype(str)
    cols = [f"PE_scale_{int(scale)}h" for scale in scales]
    cols = [col for col in cols if col in pe_df.columns]
    if not cols:
        raise ValueError(
            f"No PE columns found in {pe_path} for scales={scales}"
        )

    scores = []
    for site in sites:
        site_key = str(site)
        if site_key in pe_df.index:
            values = pe_df.loc[site_key, cols]
            if isinstance(values, pd.DataFrame):
                values = values.mean(axis=0)
            score = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float32)
            scores.append(float(np.nanmean(score)))
        else:
            scores.append(np.nan)

    arr = np.asarray(scores, dtype=np.float32)
    if not np.isfinite(arr).any():
        raise ValueError("PE adaptive loss could not match any site PE values")
    fill_value = float(np.nanmean(arr[np.isfinite(arr)]))
    arr = np.where(np.isfinite(arr), arr, fill_value).astype(np.float32)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def build_pe_adaptive_loss_weights(args, pe_node_weights, device):
    node_weight = torch.tensor(pe_node_weights, dtype=torch.float32, device=device)
    steps = torch.arange(1, args.pre_len + 1, dtype=torch.float32, device=device)
    start_step = max(1, int(args.pe_loss_start_step))
    denom = max(float(args.pre_len - start_step + 2), 1.0)
    ramp = ((steps - float(start_step - 2)) / denom).clamp(0.0, 1.0)
    return 1.0 + float(args.pe_loss_weight) * ramp.view(-1, 1) * node_weight.view(1, -1)


def build_graphs(data_dir, args, raw_o3_train):
    S = S_adjacency_matrix(data_dir, args.N_node)
    T_mat = T_adjacency_from_o3(
        raw_o3_train,
        args.N_node,
        k=args.temporal_topk,
        time_stride=args.temporal_stride,
    )
    if args.disable_pe:
        PE_mat = np.zeros((args.N_node, args.N_node), dtype=np.float32)
    else:
        PE_mat = PE_adjacency_from_o3(
            raw_o3_train,
            args.N_node,
            threshold_similarity=args.pe_threshold,
            sigma=args.pe_sigma,
            scales=parse_pe_scales(args.pe_scales),
            dim=args.pe_dim,
            delay=args.pe_delay,
            step=args.pe_window_step,
        )
    return S, T_mat, PE_mat


def sample_in_batches(
    model,
    X,
    adj,
    device,
    args,
    use_amp=False,
    inference_steps=None,
    num_samples=None,
    t_start_ratio=None,
):
    inference_steps = args.inference_steps if inference_steps is None else inference_steps
    num_samples = args.num_samples if num_samples is None else num_samples
    t_start_ratio = args.t_start_ratio if t_start_ratio is None else t_start_ratio
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), args.eval_batch_size):
            end = min(start + args.eval_batch_size, len(X))
            batch_x = torch.from_numpy(X[start:end]).float().to(device)
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp and device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                batch_pred = model.sample(
                    batch_x,
                    adj,
                    num_steps=inference_steps,
                    num_samples=num_samples,
                    t_start_ratio=t_start_ratio,
                    coarse_only=bool(args.coarse_only),
                    self_condition_mode=getattr(args, "self_condition_mode", "prev_pred"),
                    self_condition_mix=getattr(args, "self_condition_mix", 0.5),
                )
            preds.append(batch_pred.cpu())
    return torch.cat(preds, dim=0).numpy()


def shard_bounds(total, rank, world_size):
    start = (total * rank) // world_size
    end = (total * (rank + 1)) // world_size
    return start, end


def metric_sums(preds_np, targets_np, max_value):
    preds = preds_np.astype(np.float64) * float(max_value)
    tgts = targets_np.astype(np.float64) * float(max_value)
    diff = preds - tgts
    sse = float(np.sum(diff ** 2, dtype=np.float64))
    sae = float(np.sum(np.abs(diff), dtype=np.float64))
    count = float(diff.size)
    mask = np.abs(tgts) > 5.0
    if np.any(mask):
        mape_sum = float(np.sum(np.abs(diff[mask] / tgts[mask]), dtype=np.float64))
        mape_count = float(np.sum(mask))
    else:
        mape_sum = 0.0
        mape_count = 0.0
    return sse, sae, count, mape_sum, mape_count


def compute_per_step_metrics_np(preds_np, targets_np, max_value):
    preds = preds_np.astype(np.float64) * float(max_value)
    tgts = targets_np.astype(np.float64) * float(max_value)
    mse_per_step = np.mean(np.mean(np.square(preds - tgts), axis=-1), axis=0)
    rmse_per_step = np.sqrt(mse_per_step)
    mae_per_step = np.mean(np.mean(np.abs(preds - tgts), axis=-1), axis=0)
    return rmse_per_step, mae_per_step


def compute_peak_metrics_np(preds_np, targets_np, max_value,
                            peak_mode="percentile", percentile=90.0,
                            peak_thr=0.2):
    preds = preds_np.astype(np.float64) * float(max_value)
    tgts = targets_np.astype(np.float64) * float(max_value)
    if peak_mode == "fixed":
        threshold = float(peak_thr) * float(max_value)
    else:
        threshold = float(np.percentile(tgts, percentile))
    mask = tgts >= threshold
    peak_count = int(np.sum(mask))
    peak_ratio = float(peak_count / max(mask.size, 1))
    if peak_count == 0:
        return {
            "peak_mode": peak_mode,
            "peak_threshold": threshold,
            "peak_count": 0,
            "peak_ratio": 0.0,
            "rmse_peak": None,
            "mae_peak": None,
        }
    rmse_peak = float(np.sqrt(np.mean((preds[mask] - tgts[mask]) ** 2)))
    mae_peak = float(np.mean(np.abs(preds[mask] - tgts[mask])))
    return {
        "peak_mode": peak_mode,
        "peak_threshold": threshold,
        "peak_count": peak_count,
        "peak_ratio": peak_ratio,
        "rmse_peak": rmse_peak,
        "mae_peak": mae_peak,
    }


def distributed_eval(
    model,
    X,
    Y,
    adj,
    device,
    args,
    max_value,
    use_amp=False,
    rank=0,
    world_size=1,
    gather_outputs=False,
    inference_steps=None,
    num_samples=None,
    t_start_ratio=None,
):
    start, end = shard_bounds(len(X), rank, world_size)
    local_X = X[start:end]
    local_Y = Y[start:end]

    if len(local_X) > 0:
        local_preds = sample_in_batches(
            model,
            local_X,
            adj,
            device,
            args,
            use_amp=use_amp,
            inference_steps=inference_steps,
            num_samples=num_samples,
            t_start_ratio=t_start_ratio,
        )
    else:
        local_preds = np.zeros((0, args.pre_len, args.N_node), dtype=np.float32)

    stats = torch.tensor(
        metric_sums(local_preds, local_Y, max_value),
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    rmse = math.sqrt(stats[0].item() / max(stats[2].item(), 1.0))
    mae = stats[1].item() / max(stats[2].item(), 1.0)
    mape = (
        stats[3].item() / stats[4].item() * 100.0
        if stats[4].item() > 0
        else float("inf")
    )

    gathered_preds = None
    gathered_targets = None
    if gather_outputs:
        if dist.is_available() and dist.is_initialized():
            pred_parts = [None for _ in range(world_size)]
            tgt_parts = [None for _ in range(world_size)]
            dist.all_gather_object(pred_parts, local_preds)
            dist.all_gather_object(tgt_parts, local_Y)
            if rank == 0:
                gathered_preds = np.concatenate(pred_parts, axis=0)
                gathered_targets = np.concatenate(tgt_parts, axis=0)
        else:
            gathered_preds = local_preds
            gathered_targets = local_Y

    return rmse, mae, mape, gathered_preds, gathered_targets


def resolve_eval_sampling(args):
    inference_steps = args.inference_steps
    if args.eval_inference_steps > 0:
        inference_steps = args.eval_inference_steps

    num_samples = args.num_samples
    if args.eval_num_samples > 0:
        num_samples = args.eval_num_samples

    t_start_ratio = args.t_start_ratio
    if args.eval_t_start_ratio >= 0:
        t_start_ratio = args.eval_t_start_ratio

    return inference_steps, num_samples, t_start_ratio


def resolve_eval_seed(args, offset=0):
    base = args.eval_seed if args.eval_seed >= 0 else args.seed + 2048
    return int(base + offset)


def clone_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def is_distributed_launch():
    return all(key in os.environ for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE"))


def init_runtime(args):
    use_ddp = is_distributed_launch()
    rank = 0
    local_rank = 0
    world_size = 1

    if use_ddp:
        backend = "nccl" if torch.cuda.is_available() and args.device.startswith("cuda") else "gloo"
        timeout_minutes = max(1, int(args.ddp_timeout_minutes))
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=timeout_minutes),
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        if backend == "nccl":
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = resolve_device(args.device)
    else:
        device = resolve_device(args.device)

    is_main = rank == 0
    return use_ddp, rank, local_rank, world_size, device, is_main


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_sum(value, device):
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def extract_model_state(payload):
    if isinstance(payload, dict):
        if "model_state" in payload:
            return payload["model_state"]
        if "best_state" in payload and payload["best_state"] is not None:
            return payload["best_state"]
        if all(isinstance(k, str) for k in payload.keys()):
            return payload
    raise ValueError("checkpoint does not contain a recognizable model state dict")


def load_warm_start(model, ckpt_path, device):
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    source_state = extract_model_state(payload)
    target_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in source_state.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if clean_key in target_state and target_state[clean_key].shape == value.shape:
            filtered[clean_key] = value
        else:
            skipped.append(clean_key)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return {
        "loaded": len(filtered),
        "skipped": len(skipped),
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def set_backbone_trainable_for_adapter(model, freeze_backbone):
    if not freeze_backbone:
        return
    prefixes = ("pe_delta_", "coarse_ms_", "h_refiner")
    has_adapter = any(name.startswith(prefixes) for name, _ in model.named_parameters())
    if not has_adapter:
        raise ValueError("freeze_backbone=1 requires pe_delta_adapter=1, coarse_ms_residual=1, or temporal_refiner=dtr")
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(prefixes)


def get_resume_state_path(weight_dir, epoch):
    return os.path.join(weight_dir, f"resume_epoch_{epoch}.pt")


def get_latest_resume_state_path(weight_dir):
    return os.path.join(weight_dir, "resume_latest.pt")


def capture_rng_state(device):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available() and device.type == "cuda":
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda_all"] = None
    return state


def _as_cpu_rng_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value, dtype=torch.uint8, device="cpu").contiguous()
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=torch.uint8, device="cpu")
    return None


def restore_rng_state(state, device, warn_fn=None):
    if not state:
        return
    warn = warn_fn or (lambda msg: None)
    if "python" in state:
        try:
            random.setstate(state["python"])
        except Exception as exc:
            warn(f"[WARN] failed to restore python RNG state: {exc}")
    if "numpy" in state:
        try:
            np.random.set_state(state["numpy"])
        except Exception as exc:
            warn(f"[WARN] failed to restore numpy RNG state: {exc}")
    if "torch_cpu" in state:
        try:
            torch_cpu_state = _as_cpu_rng_tensor(state["torch_cpu"])
            if torch_cpu_state is None:
                raise TypeError(f"unsupported torch_cpu RNG state type: {type(state['torch_cpu'])}")
            torch.random.set_rng_state(torch_cpu_state)
        except Exception as exc:
            warn(f"[WARN] failed to restore torch CPU RNG state: {exc}")
    cuda_state = state.get("torch_cuda_all")
    if cuda_state is not None and torch.cuda.is_available() and device.type == "cuda":
        try:
            cuda_states = [_as_cpu_rng_tensor(item) for item in cuda_state]
            cuda_states = [item for item in cuda_states if item is not None]
            if cuda_states:
                torch.cuda.set_rng_state_all(cuda_states)
        except Exception as exc:
            warn(f"[WARN] failed to restore torch CUDA RNG state: {exc}")


def build_resume_payload(
    epoch,
    base_model,
    optimizer,
    scheduler,
    scaler,
    ema,
    history,
    best_rmse,
    best_epoch,
    patience_counter,
    best_state,
    args,
    device,
):
    return {
        "epoch": int(epoch),
        "model_state": base_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "ema_shadow": ema.shadow if ema is not None else None,
        "history": history,
        "best_rmse": float(best_rmse),
        "best_epoch": int(best_epoch),
        "patience_counter": int(patience_counter),
        "best_state": best_state,
        "args": vars(args),
        "rng_state": capture_rng_state(device),
    }


def main():
    args = parse_args()
    use_ddp, rank, local_rank, world_size, device, is_main = init_runtime(args)
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.pe_refine_gate and args.predict_residual:
        cleanup_distributed()
        raise ValueError("pe_refine_gate=1 only supports predict_residual=0 in v1")
    if args.pe_delta_adapter and args.coarse_mode != "final":
        cleanup_distributed()
        raise ValueError("pe_delta_adapter=1 is intentionally limited to coarse_mode=final")
    if args.coarse_ms_residual and args.coarse_mode != "final":
        cleanup_distributed()
        raise ValueError("coarse_ms_residual=1 is intentionally limited to coarse_mode=final")
    if args.freeze_backbone and not (args.pe_delta_adapter or args.coarse_ms_residual or args.temporal_refiner == "dtr"):
        cleanup_distributed()
        raise ValueError("freeze_backbone=1 requires pe_delta_adapter=1, coarse_ms_residual=1, or temporal_refiner=dtr")

    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    suffix = f"_{args.exp_name}" if args.exp_name else ""
    out_dir = os.path.join(args.data_dir, f"matrix_N95_PE3_noleak{suffix}")
    weight_dir = os.path.join(args.data_dir, f"weights_N95/weights_pe3_noleak{suffix}")
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(weight_dir, exist_ok=True)
    barrier()

    if is_main:
        if use_ddp:
            unit = "GPUs" if device.type == "cuda" else "processes"
            print(f"[INFO] {world_size} {unit} detected, using DDP", flush=True)
            print(f"[INFO] ddp_timeout_minutes={args.ddp_timeout_minutes}", flush=True)
            print(
                f"[INFO] per_gpu_batch={args.batch_size}, "
                f"global_batch={args.batch_size * world_size * max(1, args.grad_accum_steps)}",
                flush=True,
            )
        print(f"[INFO] device={device}", flush=True)
        print("[INFO] split_mode=noleak", flush=True)
        if args.coarse_mode == "pe_horizon" and args.disable_pe:
            print(
                "[WARN] coarse_mode=pe_horizon but disable_pe=1; "
                "PE branch is zeroed and decoder degenerates toward horizon mode",
                flush=True,
            )
        print(f"[INFO] amp={use_amp}", flush=True)
        print(f"[INFO] grad_accum_steps={args.grad_accum_steps}", flush=True)
        print(
            f"[INFO] drift_flags: horizon_weights={args.horizon_weights or 'uniform'}, "
            f"coarse_horizon_weights={args.coarse_horizon_weights or args.horizon_weights or 'uniform'}, "
            f"predict_residual={args.predict_residual}, coarse_mode={args.coarse_mode}, "
            f"pe_refine_gate={args.pe_refine_gate}, pe_gate=[{args.pe_gate_min}, {args.pe_gate_max}], "
            f"pe_adaptive_loss={args.pe_adaptive_loss}, pe_loss_weight={args.pe_loss_weight}, "
            f"pe_loss_start_step={args.pe_loss_start_step}, "
            f"pe_delta_adapter={args.pe_delta_adapter}, pe_delta_max={args.pe_delta_max}, "
            f"pe_delta_start_step={args.pe_delta_start_step}, coarse_ms_residual={args.coarse_ms_residual}, "
            f"coarse_ms_delta_max={args.coarse_ms_delta_max}, coarse_ms_start_step={args.coarse_ms_start_step}, "
            f"temporal_refiner={args.temporal_refiner}, dtr_layers={args.dtr_layers}, "
            f"dtr_kernel_size={args.dtr_kernel_size}, dtr_dropout={args.dtr_dropout}, "
            f"dtr_pe_fusion={args.dtr_pe_fusion}, freeze_backbone={args.freeze_backbone}, "
            f"warm_start_ckpt={args.warm_start_ckpt or 'none'}, grad_clip={args.grad_clip}, "
            f"self_condition_mode={args.self_condition_mode}, self_condition_mix={args.self_condition_mix}",
            flush=True,
        )

    raw_o3 = np.load(os.path.join(args.data_dir, "matrix_N95", "data.npy")).astype(np.float32)
    if raw_o3.shape[0] != args.N_node:
        cleanup_distributed()
        raise ValueError(f"N_node mismatch: {raw_o3.shape[0]} vs {args.N_node}")

    time_index = load_time_index(args.data_dir, raw_o3.shape[1])
    sites = get_sites(args.data_dir)
    if len(sites) != args.N_node:
        cleanup_distributed()
        raise ValueError(f"site count mismatch: {len(sites)} vs {args.N_node}")

    if is_main:
        print("[INFO] loading meteorological data...", flush=True)
    met_cache_path = os.path.join(args.data_dir, "matrix_N95", "met_raw_aligned_cache.npz")
    if use_ddp and bool(args.use_met_cache) and not os.path.exists(met_cache_path):
        if is_main:
            met_raw = load_met_data_raw(
                args.data_dir,
                sites,
                time_index,
                use_cache=True,
                verbose=True,
            )
        barrier()
        if not is_main:
            met_raw = load_met_data_raw(
                args.data_dir,
                sites,
                time_index,
                use_cache=True,
                verbose=False,
            )
    else:
        met_raw = load_met_data_raw(
            args.data_dir,
            sites,
            time_index,
            use_cache=bool(args.use_met_cache),
            verbose=is_main,
        )
    split_sets, max_value, met_stats, split_desc, scale_stats, time_slices = prepare_noleak_data(
        args, raw_o3, time_index, met_raw
    )
    split_sets = maybe_limit_split_windows(split_sets, args, verbose=is_main)
    horizon_weights_list, coarse_horizon_weights_list = resolve_horizon_weights(args)
    args.horizon_weights_resolved = horizon_weights_list
    args.coarse_horizon_weights_resolved = coarse_horizon_weights_list

    trainX, trainY = split_sets["train"]
    validX, validY = split_sets["valid"]
    testX, testY = split_sets["test"]

    raw_o3_train = raw_o3[:, time_slices["train"]]

    if is_main:
        save_json(os.path.join(out_dir, "config.json"), vars(args))
        save_json(os.path.join(out_dir, "split_summary.json"), split_desc)
        save_json(os.path.join(out_dir, "met_stats.json"), met_stats)
        save_json(os.path.join(out_dir, "scale_stats.json"), scale_stats)

        if args.save_train_arrays:
            np.save(os.path.join(out_dir, "trainX.npy"), trainX)
            np.save(os.path.join(out_dir, "trainY.npy"), trainY)
        np.save(os.path.join(out_dir, "validX.npy"), validX)
        np.save(os.path.join(out_dir, "validY.npy"), validY)
        np.save(os.path.join(out_dir, "testX.npy"), testX)
        np.save(os.path.join(out_dir, "testY.npy"), testY)

        print(f"[INFO] trainX={trainX.shape}, trainY={trainY.shape}", flush=True)
        print(f"[INFO] validX={validX.shape}, validY={validY.shape}", flush=True)
        print(f"[INFO] testX={testX.shape}, testY={testY.shape}", flush=True)

        print("[INFO] building graphs...", flush=True)
        S, T_mat, PE_mat = build_graphs(args.data_dir, args, raw_o3_train)
        np.save(os.path.join(out_dir, "S_matrix.npy"), S)
        np.save(os.path.join(out_dir, "T_matrix.npy"), T_mat)
        np.save(os.path.join(out_dir, "PE_matrix.npy"), PE_mat)
        save_json(
            os.path.join(out_dir, "graph_summary.json"),
            {
                "S_nnz": int(np.sum(S > 0)),
                "T_nnz": int(np.sum(T_mat > 0)),
                "PE_nnz": int(np.sum(PE_mat > 0)),
                "temporal_topk": int(args.temporal_topk),
                "temporal_stride": int(args.temporal_stride),
                "disable_pe": int(args.disable_pe),
                "pe_scales": parse_pe_scales(args.pe_scales),
                "pe_window_step": int(args.pe_window_step),
            },
        )
        print(
            f"[INFO] graph nnz: S={int(np.sum(S > 0))}, "
            f"T={int(np.sum(T_mat > 0))}, PE={int(np.sum(PE_mat > 0))}",
            flush=True,
        )

    barrier()
    S = np.load(os.path.join(out_dir, "S_matrix.npy")).astype(np.float32)
    T_mat = np.load(os.path.join(out_dir, "T_matrix.npy")).astype(np.float32)
    PE_mat = np.load(os.path.join(out_dir, "PE_matrix.npy")).astype(np.float32)

    adj = torch.tensor(np.stack([S, T_mat, PE_mat], axis=0), dtype=torch.float32, device=device)

    train_ds = ATGCNDataset(trainX, trainY)
    train_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    if is_main:
        print(
            f"[INFO] train batches/epoch={len(train_loader)}, "
            f"effective_batch={args.batch_size * max(1, args.grad_accum_steps) * world_size}",
            flush=True,
        )

    model = DiffusionATGCN(
        hidden_size=args.hidden_size,
        adj_units=args.adj_units,
        pre_len=args.pre_len,
        N_node=args.N_node,
        m=args.m,
        d_met=8,
        diff_steps=args.diff_steps,
        predict_residual=bool(args.predict_residual),
        coarse_mode=args.coarse_mode,
        pe_refine_gate=bool(args.pe_refine_gate),
        pe_gate_min=args.pe_gate_min,
        pe_gate_max=args.pe_gate_max,
        pe_delta_adapter=bool(args.pe_delta_adapter),
        pe_delta_max=args.pe_delta_max,
        pe_delta_start_step=args.pe_delta_start_step,
        coarse_ms_residual=bool(args.coarse_ms_residual),
        coarse_ms_delta_max=args.coarse_ms_delta_max,
        coarse_ms_start_step=args.coarse_ms_start_step,
        temporal_refiner=args.temporal_refiner,
        dtr_layers=args.dtr_layers,
        dtr_kernel_size=args.dtr_kernel_size,
        dtr_dropout=args.dtr_dropout,
        dtr_pe_fusion=bool(args.dtr_pe_fusion),
    ).to(device)
    if args.warm_start_ckpt:
        if not os.path.exists(args.warm_start_ckpt):
            cleanup_distributed()
            raise FileNotFoundError(f"warm_start_ckpt not found: {args.warm_start_ckpt}")
        summary = load_warm_start(model, args.warm_start_ckpt, device)
        if is_main:
            print(
                f"[INFO] warm-start loaded from {args.warm_start_ckpt}: "
                f"loaded={summary['loaded']}, skipped={summary['skipped']}, "
                f"missing={len(summary['missing'])}, unexpected={len(summary['unexpected'])}",
                flush=True,
            )
    set_backbone_trainable_for_adapter(model, bool(args.freeze_backbone))
    if is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[INFO] trainable_params={trainable}/{total}", flush=True)
    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            static_graph=True,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        cleanup_distributed()
        raise ValueError("no trainable parameters; check freeze_backbone / adapter flags")
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    warmup_epochs = min(5, max(1, args.epochs))

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return max(args.lr_min / max(args.lr, 1e-12), 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    base_model = unwrap_model(model)
    ema = EMA(base_model, decay=args.ema_decay)
    diff = base_model.diffusion
    horizon_weights = (
        torch.tensor(horizon_weights_list, dtype=torch.float32, device=device)
        if horizon_weights_list is not None else None
    )
    coarse_horizon_weights = (
        torch.tensor(coarse_horizon_weights_list, dtype=torch.float32, device=device)
        if coarse_horizon_weights_list is not None else None
    )
    pe_loss_weights = None
    pe_node_weights_np = None
    if args.pe_adaptive_loss:
        pe_node_weights_np = load_pe_node_weights(
            args.data_dir,
            sites,
            parse_pe_scales(args.pe_scales),
        )
        pe_loss_weights = build_pe_adaptive_loss_weights(
            args,
            pe_node_weights_np,
            device,
        )
        if is_main:
            np.save(os.path.join(out_dir, "pe_node_loss_weights.npy"), pe_node_weights_np)
            save_json(
                os.path.join(out_dir, "pe_adaptive_loss_summary.json"),
                {
                    "pe_adaptive_loss": int(args.pe_adaptive_loss),
                    "pe_loss_weight": float(args.pe_loss_weight),
                    "pe_loss_start_step": int(args.pe_loss_start_step),
                    "pe_node_weight_min": float(np.min(pe_node_weights_np)),
                    "pe_node_weight_max": float(np.max(pe_node_weights_np)),
                    "pe_node_weight_mean": float(np.mean(pe_node_weights_np)),
                },
            )
            print(
                f"[INFO] PE-adaptive loss enabled: node_weight mean="
                f"{float(np.mean(pe_node_weights_np)):.4f}, max={float(np.max(pe_node_weights_np)):.4f}",
                flush=True,
            )

    history = {
        "train_loss": [],
        "valid_rmse": [],
        "valid_mae": [],
        "valid_mape": [],
    }
    best_rmse = float("inf")
    best_epoch = -1
    patience_counter = 0
    best_state = None

    if args.resume_epoch >= 0:
        ckpt_path = get_resume_state_path(weight_dir, args.resume_epoch)
        map_location = device
        if not os.path.exists(ckpt_path):
            legacy_ckpt_path = os.path.join(weight_dir, f"epoch_{args.resume_epoch}.pt")
            if not os.path.exists(legacy_ckpt_path):
                cleanup_distributed()
                raise FileNotFoundError(
                    f"resume checkpoint not found: {ckpt_path} or {legacy_ckpt_path}"
                )
            if is_main:
                print(
                    f"[WARN] full resume state missing, falling back to weight-only resume: "
                    f"{legacy_ckpt_path}",
                    flush=True,
                )
            base_model.load_state_dict(torch.load(legacy_ckpt_path, map_location=map_location))
            start_epoch = args.resume_epoch + 1
        else:
            if is_main:
                print(f"[INFO] resuming full state from {ckpt_path}", flush=True)
            state = torch.load(ckpt_path, map_location=map_location, weights_only=False)
            base_model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            scheduler.load_state_dict(state["scheduler_state"])
            scaler_state = state.get("scaler_state")
            if scaler_state:
                scaler.load_state_dict(scaler_state)
            if ema is not None and state.get("ema_shadow") is not None:
                ema.shadow = state["ema_shadow"]
            history = state.get("history", history)
            best_rmse = float(state.get("best_rmse", best_rmse))
            best_epoch = int(state.get("best_epoch", best_epoch))
            patience_counter = int(state.get("patience_counter", patience_counter))
            best_state = state.get("best_state", best_state)
            restore_rng_state(state.get("rng_state"), device, print if is_main else None)
            start_epoch = int(state.get("epoch", args.resume_epoch)) + 1
            if is_main:
                print(
                    f"[INFO] resumed epoch={start_epoch}, best_epoch={best_epoch + 1}, "
                    f"best_rmse={best_rmse:.4f}, patience_counter={patience_counter}",
                    flush=True,
                )
    else:
        start_epoch = 0
    last_completed_epoch = start_epoch - 1
    eval_inference_steps, eval_num_samples, eval_t_start_ratio = resolve_eval_sampling(args)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        current_coarse_weight = resolve_coarse_weight(args, epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main:
            print(f"[INFO] start epoch {epoch + 1}/{args.epochs}", flush=True)

        for batch_idx, (bx, by) in enumerate(train_loader):
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            bs = bx.size(0)
            should_step = (
                (batch_idx + 1) % max(1, args.grad_accum_steps) == 0
                or (batch_idx + 1) == len(train_loader)
            )
            sync_ctx = model.no_sync() if use_ddp and not should_step else nullcontext()

            with sync_ctx:
                t_diff = torch.randint(0, diff.num_steps, (bs,), device=device)
                noise = torch.randn_like(by)
                y_noisy = None if args.predict_residual else diff.q_sample(by, t_diff, noise)[0]

                amp_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if use_amp and device.type == "cuda"
                    else nullcontext()
                )
                with amp_ctx:
                    x0_pred, y_coarse, diff_pred, diff_target = model(
                        bx,
                        adj,
                        y_noisy,
                        t_diff,
                        self_cond=None,
                        return_coarse=True,
                        y_target=by,
                        noise=noise,
                        return_diffusion=True,
                    )

                    if random.random() < 0.5:
                        with torch.no_grad():
                            sc = model(
                                bx,
                                adj,
                                y_noisy,
                                t_diff,
                                y_target=by,
                                noise=noise,
                            ).detach()
                        x0_pred, y_coarse, diff_pred, diff_target = model(
                            bx,
                            adj,
                            y_noisy,
                            t_diff,
                            self_cond=sc,
                            return_coarse=True,
                            y_target=by,
                            noise=noise,
                            return_diffusion=True,
                        )

                    err2 = (diff_pred - diff_target) ** 2
                    if args.peak_weight > 0:
                        pw = 1.0 + args.peak_weight * (by >= args.peak_thr).float()
                        err2 = err2 * pw
                    diff_loss = weighted_horizon_loss(err2, horizon_weights, pe_loss_weights)

                    coarse_err2 = (y_coarse - by) ** 2
                    coarse_loss = weighted_horizon_loss(coarse_err2, coarse_horizon_weights, pe_loss_weights)
                    loss = diff_loss + current_coarse_weight * coarse_loss

                    if args.lambda_temporal > 0:
                        grad_pred = x0_pred[:, 1:, :] - x0_pred[:, :-1, :]
                        grad_true = by[:, 1:, :] - by[:, :-1, :]
                        temporal_loss = weighted_horizon_loss(
                            (grad_pred - grad_true) ** 2,
                            horizon_weights[1:] if horizon_weights is not None else None,
                            pe_loss_weights[1:] if pe_loss_weights is not None else None,
                        )
                        loss = loss + args.lambda_temporal * temporal_loss

                    l2_loss = args.lambda_reg * sum(
                        p.pow(2).sum() for p in model.parameters() if p.requires_grad
                    )
                    loss = loss + l2_loss

                if args.skip_nonfinite_loss:
                    finite_tensor = torch.tensor(
                        int(torch.isfinite(loss.detach()).all().item()),
                        dtype=torch.int64,
                        device=device,
                    )
                    if use_ddp:
                        dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
                    if finite_tensor.item() == 0:
                        if is_main:
                            print(
                                f"  [WARN] skip non-finite loss at epoch {epoch + 1} "
                                f"batch {batch_idx + 1}",
                                flush=True,
                            )
                        optimizer.zero_grad(set_to_none=True)
                        continue

                loss_value = float(loss.detach().item())
                loss_for_backward = loss / max(1, args.grad_accum_steps)
                scaler.scale(loss_for_backward).backward()

            if should_step:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float(args.grad_clip),
                )
                if args.skip_nonfinite_loss:
                    grad_ok = torch.tensor(
                        int(torch.isfinite(torch.as_tensor(grad_norm, device=device)).item()),
                        dtype=torch.int64,
                        device=device,
                    )
                    if use_ddp:
                        dist.all_reduce(grad_ok, op=dist.ReduceOp.MIN)
                    if grad_ok.item() == 0:
                        if is_main:
                            print(
                                f"  [WARN] skip non-finite grad at epoch {epoch + 1} "
                                f"batch {batch_idx + 1}",
                                flush=True,
                            )
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        continue
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(base_model)

            running_loss += loss_value
            if is_main and args.log_interval > 0 and (
                (batch_idx + 1) % args.log_interval == 0
                or (batch_idx + 1) == len(train_loader)
            ):
                print(
                    f"  Epoch {epoch + 1} Batch {batch_idx + 1}/{len(train_loader)} "
                    f"| Loss={loss.item():.6f} "
                    f"| diff={diff_loss.item():.4f} "
                    f"| coarse={coarse_loss.item():.4f} "
                    f"| {time.time() - t0:.1f}s",
                    flush=True,
                )

        scheduler.step()
        avg_loss = reduce_sum(running_loss, device) / max(len(train_loader) * world_size, 1)
        stop_training = False

        if device.type == "cuda":
            torch.cuda.empty_cache()
        train_rng_state = capture_rng_state(device)
        set_seed(resolve_eval_seed(args))
        ema.apply(base_model)
        valid_rmse, valid_mae, valid_mape, _, _ = distributed_eval(
            base_model,
            validX,
            validY,
            adj,
            device,
            args,
            max_value,
            use_amp=use_amp,
            rank=rank,
            world_size=world_size,
            gather_outputs=False,
            inference_steps=eval_inference_steps,
            num_samples=eval_num_samples,
            t_start_ratio=eval_t_start_ratio,
        )
        ema.restore(base_model)
        restore_rng_state(train_rng_state, device)

        if is_main:
            history["train_loss"].append(avg_loss)
            history["valid_rmse"].append(valid_rmse)
            history["valid_mae"].append(valid_mae)
            history["valid_mape"].append(valid_mape)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Loss={avg_loss:.6f} | "
                f"Val RMSE={valid_rmse:.4f} | "
                f"Val MAE={valid_mae:.4f} | "
                f"Val MAPE={valid_mape:.2f}% | "
                f"LR={optimizer.param_groups[0]['lr']:.2e} | "
                f"CW={current_coarse_weight:.4f} | "
                f"{elapsed:.1f}s",
                flush=True,
            )

            improved = valid_rmse < best_rmse - args.min_delta
            if improved:
                ema.apply(base_model)
                best_state = clone_state_dict(base_model)
                ema.restore(base_model)
                best_rmse = valid_rmse
                best_epoch = epoch
                patience_counter = 0
                torch.save(best_state, os.path.join(weight_dir, "best_ema.pt"))
                print(f"  [BEST] epoch={epoch + 1}, val_rmse={valid_rmse:.4f}", flush=True)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(
                        f"[EARLY STOP] patience={args.patience}, "
                        f"best_val_rmse={best_rmse:.4f} at epoch {best_epoch + 1}",
                        flush=True,
                    )
                    stop_training = True

            ema.apply(base_model)
            torch.save(base_model.state_dict(), os.path.join(weight_dir, f"epoch_{epoch}_ema.pt"))
            ema.restore(base_model)
            torch.save(base_model.state_dict(), os.path.join(weight_dir, f"epoch_{epoch}.pt"))
            resume_payload = build_resume_payload(
                epoch=epoch,
                base_model=base_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                history=history,
                best_rmse=best_rmse,
                best_epoch=best_epoch,
                patience_counter=patience_counter,
                best_state=best_state,
                args=args,
                device=device,
            )
            torch.save(resume_payload, get_resume_state_path(weight_dir, epoch))
            torch.save(resume_payload, get_latest_resume_state_path(weight_dir))
            last_completed_epoch = epoch

        if use_ddp:
            barrier()
            stop_tensor = torch.tensor(int(stop_training), dtype=torch.int64, device=device)
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())
            barrier()

        if stop_training:
            break

    if is_main:
        if best_state is None:
            ema.apply(base_model)
            best_state = clone_state_dict(base_model)
            ema.restore(base_model)
            torch.save(best_state, os.path.join(weight_dir, "best_ema.pt"))

        torch.save(base_model.state_dict(), os.path.join(weight_dir, "last.pt"))
        final_resume_payload = build_resume_payload(
            epoch=last_completed_epoch,
            base_model=base_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            history=history,
            best_rmse=best_rmse,
            best_epoch=best_epoch,
            patience_counter=patience_counter,
            best_state=best_state,
            args=args,
            device=device,
        )
        torch.save(final_resume_payload, get_latest_resume_state_path(weight_dir))

    if use_ddp:
        barrier()

    best_ckpt_path = os.path.join(weight_dir, "best_ema.pt")
    base_model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    test_rng_state = capture_rng_state(device)
    set_seed(resolve_eval_seed(args, offset=1))
    test_rmse, test_mae, test_mape, test_preds, test_targets = distributed_eval(
        base_model,
        testX,
        testY,
        adj,
        device,
        args,
        max_value,
        use_amp=use_amp,
        rank=rank,
        world_size=world_size,
        gather_outputs=bool(args.save_predictions),
        inference_steps=eval_inference_steps,
        num_samples=eval_num_samples,
        t_start_ratio=eval_t_start_ratio,
    )
    restore_rng_state(test_rng_state, device)

    if is_main:
        metrics = {
            "split_mode": "noleak",
            "best_epoch": int(best_epoch + 1),
            "best_valid_rmse": float(best_rmse),
            "best_valid_mae": float(history["valid_mae"][best_epoch]) if best_epoch >= 0 else None,
            "best_valid_mape": float(history["valid_mape"][best_epoch]) if best_epoch >= 0 else None,
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
            "test_mape": float(test_mape),
            "relative_rmse": float(test_rmse / max_value * 100.0),
            "relative_mae": float(test_mae / max_value * 100.0),
            "train_scale_max": float(max_value),
            "world_size": int(world_size),
            "eval_inference_steps": int(eval_inference_steps),
            "eval_num_samples": int(eval_num_samples),
            "eval_t_start_ratio": float(eval_t_start_ratio),
        }

        np.save(os.path.join(out_dir, "train_loss.npy"), np.array(history["train_loss"]))
        np.save(os.path.join(out_dir, "valid_rmse.npy"), np.array(history["valid_rmse"]))
        np.save(os.path.join(out_dir, "valid_mae.npy"), np.array(history["valid_mae"]))
        np.save(os.path.join(out_dir, "valid_mape.npy"), np.array(history["valid_mape"]))

        if test_preds is not None and test_targets is not None:
            rmse_per_step, mae_per_step = compute_per_step_metrics_np(
                test_preds,
                test_targets,
                max_value,
            )
            peak_info = compute_peak_metrics_np(test_preds, test_targets, max_value)
            metrics.update(
                {
                    "per_step_rmse": [float(x) for x in rmse_per_step],
                    "per_step_mae": [float(x) for x in mae_per_step],
                    "peak_mode": peak_info["peak_mode"],
                    "peak_threshold": float(peak_info["peak_threshold"]),
                    "peak_count": int(peak_info["peak_count"]),
                    "peak_ratio": float(peak_info["peak_ratio"]),
                    "rmse_peak": None if peak_info["rmse_peak"] is None else float(peak_info["rmse_peak"]),
                    "mae_peak": None if peak_info["mae_peak"] is None else float(peak_info["mae_peak"]),
                }
            )

        if args.save_predictions and test_preds is not None and test_targets is not None:
            np.save(os.path.join(out_dir, "test_predictions.npy"), test_preds)
            np.save(os.path.join(out_dir, "test_targets.npy"), test_targets)

        save_json(os.path.join(out_dir, "metrics_summary.json"), metrics)

        print("[INFO] noleak training finished", flush=True)
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        if "peak_mode" in metrics:
            print(f"  Peak_mode      : {metrics['peak_mode']}", flush=True)
            print(f"  Peak_threshold : {metrics['peak_threshold']:.2f} μg/m³", flush=True)
            print(f"  Peak_count     : {metrics['peak_count']}", flush=True)
            print(f"  Peak_ratio     : {metrics['peak_ratio'] * 100:.2f}%", flush=True)
            if metrics["rmse_peak"] is not None:
                print(f"  RMSE_peak      : {metrics['rmse_peak']:.4f}", flush=True)
                print(f"  MAE_peak       : {metrics['mae_peak']:.4f}", flush=True)
            print("逐步预测指标:", flush=True)
            print(f"  {'Step':>4}  {'RMSE':>10}  {'MAE':>10}", flush=True)
            for i, (rmse_step, mae_step) in enumerate(
                zip(metrics["per_step_rmse"], metrics["per_step_mae"]),
                start=1,
            ):
                print(f"  {i:>4}  {rmse_step:>10.4f}  {mae_step:>10.4f}", flush=True)
        print(f"[INFO] outputs: {out_dir}", flush=True)
        print(f"[INFO] weights: {weight_dir}", flush=True)

    barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
