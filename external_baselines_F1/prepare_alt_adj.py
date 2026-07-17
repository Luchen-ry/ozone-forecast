"""
Generate alternative adjacency matrices for DiffSTG AIR_N95 experiments.
======================================================================
Input:
  DiffSTG/data/dataset/AIR_N95/flow.npy   (T, N, 1)

Output:
  output/corr_adj.npy   Pearson top-k correlation graph
  output/pe_adj.npy     Permutation-entropy similarity graph

Usage:
  python external_baselines_F1/prepare_alt_adj.py
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent
FLOW_PATH = ROOT / "DiffSTG" / "data" / "dataset" / "AIR_N95" / "flow.npy"
OUT_DIR = ROOT / "output"

# Defaults aligned with code/train_atgcn_pe3_noleak.py
CORR_TOPK = 12
PE_SCALES = (6, 9, 12, 24, 48, 72)
PE_THRESHOLD = 0.75
PE_SIGMA = 0.3
PE_DIM = 3
PE_DELAY = 1
PE_STEP = 1


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
        f"[INFO] building PE features: nodes={num_nodes}, scales={list(scales)}, step={step}",
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


def corr_adjacency_from_o3(o3_raw, k=CORR_TOPK):
    """Pearson top-k correlation graph (diagonal = 0)."""
    n = o3_raw.shape[0]
    x = (o3_raw - o3_raw.mean(axis=1, keepdims=True)) / (
        o3_raw.std(axis=1, keepdims=True) + 1e-6
    )
    corr = np.corrcoef(x)
    np.fill_diagonal(corr, -np.inf)

    topk = min(k, n - 1)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        idx = np.argpartition(-corr[i], topk)[:topk]
        adj[i, idx] = corr[i, idx]
    adj = np.maximum(adj, adj.T)
    adj[adj < 0] = 0
    np.fill_diagonal(adj, 0.0)
    return adj


def pe_adjacency_from_o3(
    o3_raw,
    threshold_similarity=PE_THRESHOLD,
    sigma=PE_SIGMA,
    scales=PE_SCALES,
    dim=PE_DIM,
    delay=PE_DELAY,
    step=PE_STEP,
):
    """PE similarity graph (diagonal = 0, matching distance adj convention)."""
    n = o3_raw.shape[0]
    pe_features = build_pe_feature_matrix(
        o3_raw, scales, dim=dim, delay=delay, step=step
    )
    pe_distances = cdist(pe_features, pe_features, metric="euclidean")
    pe_similarity = np.exp(-(pe_distances ** 2) / (sigma ** 2))

    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if pe_similarity[i, j] > threshold_similarity:
                adj[i, j] = adj[j, i] = pe_similarity[i, j]
    return adj


def _sparsity(adj: np.ndarray) -> float:
    return float((adj != 0).sum() / adj.size)


def main():
    if not FLOW_PATH.is_file():
        raise FileNotFoundError(f"Missing flow.npy: {FLOW_PATH}")

    flow = np.load(FLOW_PATH)  # (T, N, 1)
    if flow.ndim != 3:
        raise ValueError(f"Expected flow shape (T, N, 1), got {flow.shape}")
    o3 = flow[:, :, 0].T.astype(np.float32)  # (N, T)
    print(f"[INFO] loaded flow {flow.shape} -> o3 {o3.shape}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    corr_adj = corr_adjacency_from_o3(o3, k=CORR_TOPK)
    corr_path = OUT_DIR / "corr_adj.npy"
    np.save(corr_path, corr_adj)
    print(
        f"[OK] {corr_path}  shape={corr_adj.shape}, non-zero={_sparsity(corr_adj):.2%}",
        flush=True,
    )

    pe_adj = pe_adjacency_from_o3(o3)
    pe_path = OUT_DIR / "pe_adj.npy"
    np.save(pe_path, pe_adj)
    print(
        f"[OK] {pe_path}  shape={pe_adj.shape}, non-zero={_sparsity(pe_adj):.2%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
