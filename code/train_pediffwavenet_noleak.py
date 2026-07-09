#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Leakage-safe PE-DiffWaveNet training."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pediffwavenet_model import PEDiffWaveNet
from train_atgcn_pe3 import ATGCNDataset, EMA, compute_metrics, get_sites, set_seed
from train_atgcn_pe3_noleak import (
    barrier,
    build_graphs,
    build_pe_feature_matrix,
    cleanup_distributed,
    clone_state_dict,
    compute_peak_metrics_np,
    compute_per_step_metrics_np,
    distributed_eval,
    init_runtime,
    load_met_data_raw,
    load_time_index,
    maybe_limit_split_windows,
    parse_pe_scales,
    prepare_noleak_data,
    reduce_sum,
    resolve_eval_sampling,
    resolve_eval_seed,
    save_json,
    unwrap_model,
    weighted_horizon_loss,
)


def parse_args():
    p = argparse.ArgumentParser(description="Leakage-safe PE-DiffWaveNet training")
    p.add_argument("--train_rate", type=float, default=0.8465)
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--pre_len", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min_delta", type=float, default=0.001)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--N_node", type=int, default=95)
    p.add_argument("--m", type=int, default=15)
    p.add_argument("--data_dir", type=str, default="/home/chenxudong/graduate/代码 2/代码/代码")
    p.add_argument("--exp_name", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ddp_timeout_minutes", type=int, default=180)
    p.add_argument("--use_met_cache", type=int, default=1)
    p.add_argument("--max_train_windows", type=int, default=0)
    p.add_argument("--max_valid_windows", type=int, default=0)
    p.add_argument("--max_test_windows", type=int, default=0)
    p.add_argument("--save_predictions", type=int, default=1)
    p.add_argument("--save_train_arrays", type=int, default=0)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--ema_decay", type=float, default=0.999)

    p.add_argument("--temporal_topk", type=int, default=12)
    p.add_argument("--temporal_stride", type=int, default=1)
    p.add_argument("--pe_source", type=str, default="train", choices=("train", "precomputed"))
    p.add_argument("--pe_threshold", type=float, default=0.9)
    p.add_argument("--pe_sigma", type=float, default=0.1)
    p.add_argument("--pe_dim", type=int, default=3)
    p.add_argument("--pe_delay", type=int, default=1)
    p.add_argument("--pe_scales", type=str, default="6,9,12,24,48,72")
    p.add_argument("--pe_window_step", type=int, default=1)
    p.add_argument(
        "--pe_shuffle_seed",
        type=int,
        default=-1,
        help="Ablation only: shuffle PE node features and PE graph labels with this seed; -1 disables.",
    )
    p.add_argument("--use_pe_graph", type=int, default=1)
    p.add_argument("--use_pe_film", type=int, default=1)
    p.add_argument("--use_adaptive_adj", type=int, default=1)
    p.add_argument("--pe_graph_alpha", type=float, default=1.0)
    p.add_argument("--pe_film_scale", type=float, default=1.0)
    p.add_argument("--pe_film_zero_init", type=int, default=0)
    p.add_argument("--normalize_pe_features", type=int, default=1)
    p.add_argument("--disable_pe", type=int, default=0, help="Compatibility flag; prefer --use_pe_graph 0.")

    p.add_argument("--use_diffusion", type=int, default=1)
    p.add_argument("--diff_steps", type=int, default=50)
    p.add_argument("--inference_steps", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--t_start_ratio", type=float, default=0.25)
    p.add_argument("--eval_inference_steps", type=int, default=0)
    p.add_argument("--eval_num_samples", type=int, default=0)
    p.add_argument("--eval_t_start_ratio", type=float, default=-1.0)
    p.add_argument("--eval_seed", type=int, default=-1)
    p.add_argument("--coarse_only", type=int, default=0)
    p.add_argument("--self_condition_mode", type=str, default="prev_pred")
    p.add_argument("--self_condition_mix", type=float, default=0.5)

    p.add_argument("--coarse_weight", type=float, default=0.08)
    p.add_argument("--horizon_weights", type=str, default="1.0,1.0,1.1,1.2,1.35,1.5")
    p.add_argument("--lambda_temporal", type=float, default=0.0)
    p.add_argument("--pe_adaptive_loss", type=int, default=0)
    p.add_argument("--pe_loss_weight", type=float, default=0.15)
    p.add_argument("--pe_loss_start_step", type=int, default=4)
    p.add_argument("--pe_loss_normalize", type=int, default=1)
    return p.parse_args()


def parse_horizon_weights(value, pre_len):
    weights = [float(x.strip()) for x in str(value or "").split(",") if x.strip()]
    if not weights:
        return None
    if len(weights) != pre_len:
        raise ValueError(f"horizon_weights length must equal pre_len={pre_len}")
    return weights


def load_precomputed_pe_features(data_dir, sites, scales):
    pe_path = os.path.join(data_dir, "O3", "PE_average_O3_results.csv")
    if not os.path.exists(pe_path):
        raise FileNotFoundError(f"precomputed PE file not found: {pe_path}")
    df = pd.read_csv(pe_path, index_col=0)
    df.index = df.index.astype(str)
    cols = [f"PE_scale_{int(scale)}h" for scale in scales]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"missing PE columns in {pe_path}: {missing}")
    rows = []
    for site in sites:
        key = str(site)
        if key in df.index:
            values = pd.to_numeric(df.loc[key, cols], errors="coerce").to_numpy(dtype=np.float32)
        else:
            values = np.full(len(cols), np.nan, dtype=np.float32)
        rows.append(values)
    arr = np.asarray(rows, dtype=np.float32)
    fill = np.nanmean(arr, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.5).astype(np.float32)
    bad = ~np.isfinite(arr)
    arr[bad] = np.take(fill, np.where(bad)[1])
    return arr.astype(np.float32)


def build_pe_features(args, raw_o3_train, sites, is_main=True):
    scales = parse_pe_scales(args.pe_scales)
    needs_pe_features = bool(args.use_pe_film) or bool(args.pe_adaptive_loss)
    if not needs_pe_features:
        return np.zeros((args.N_node, len(scales)), dtype=np.float32)
    if args.pe_source == "train":
        return build_pe_feature_matrix(
            raw_o3_train,
            scales,
            dim=args.pe_dim,
            delay=args.pe_delay,
            step=args.pe_window_step,
        )
    if is_main:
        print("[WARN] pe_source=precomputed is for ablation only; do not use as main paper result", flush=True)
    return load_precomputed_pe_features(args.data_dir, sites, scales)


def apply_pe_shuffle(pe_features, pe_matrix, shuffle_seed):
    """Shuffle PE semantics across nodes while preserving PE value distribution."""
    if int(shuffle_seed) < 0:
        return pe_features, pe_matrix, None
    rng = np.random.default_rng(int(shuffle_seed))
    perm = rng.permutation(pe_features.shape[0])
    shuffled_features = pe_features[perm].astype(np.float32)
    shuffled_matrix = pe_matrix[np.ix_(perm, perm)].astype(np.float32)
    return shuffled_features, shuffled_matrix, perm.astype(np.int64)


def build_pe_loss_extra_weights(pe_features, pre_len, loss_weight, start_step=4, normalize=True):
    if float(loss_weight) <= 0:
        return None
    pe_arr = np.asarray(pe_features, dtype=np.float32)
    if pe_arr.ndim != 2 or pe_arr.shape[0] == 0:
        return None
    node_score = np.nanmean(pe_arr, axis=1).astype(np.float32)
    node_score = np.where(np.isfinite(node_score), node_score, np.nanmean(node_score))
    lo, hi = float(np.min(node_score)), float(np.max(node_score))
    if hi - lo < 1e-6:
        node_score = np.zeros_like(node_score, dtype=np.float32)
    else:
        node_score = (node_score - lo) / (hi - lo)
    ramp = np.zeros(int(pre_len), dtype=np.float32)
    start = max(1, int(start_step)) - 1
    if start < int(pre_len):
        denom = max(int(pre_len) - start, 1)
        ramp[start:] = np.arange(1, int(pre_len) - start + 1, dtype=np.float32) / float(denom)
    weights = 1.0 + float(loss_weight) * ramp[:, None] * node_score[None, :]
    if normalize:
        weights = weights / max(float(np.mean(weights)), 1e-6)
    return weights.astype(np.float32)


def make_dirs(args):
    suffix = f"_{args.exp_name}" if args.exp_name else ""
    out_dir = os.path.join(args.data_dir, f"matrix_N95_PEDiffWaveNet_noleak{suffix}")
    weight_dir = os.path.join(args.data_dir, f"weights_N95/weights_pediffwavenet_noleak{suffix}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    return out_dir, weight_dir


def main():
    args = parse_args()
    use_ddp, rank, local_rank, world_size, device, is_main = init_runtime(args)
    set_seed(args.seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    use_amp = bool(args.amp) and device.type == "cuda"

    out_dir, weight_dir = make_dirs(args)
    if is_main:
        print("[INFO] PE-DiffWaveNet noleak training", flush=True)
        print(f"[INFO] device={device}, world_size={world_size}", flush=True)
        print(f"[INFO] output_dir={out_dir}", flush=True)
        print(
            f"[INFO] clean_room=1, pe_source={args.pe_source}, use_pe_graph={args.use_pe_graph}, "
            f"use_pe_film={args.use_pe_film}, use_diffusion={args.use_diffusion}, "
            f"pe_graph_alpha={args.pe_graph_alpha}, pe_film_scale={args.pe_film_scale}, "
            f"pe_film_zero_init={args.pe_film_zero_init}, pe_shuffle_seed={args.pe_shuffle_seed}, "
            f"pe_adaptive_loss={args.pe_adaptive_loss}, pe_loss_weight={args.pe_loss_weight}",
            flush=True,
        )

    raw_o3 = np.load(os.path.join(args.data_dir, "matrix_N95", "data.npy")).astype(np.float32)
    if raw_o3.shape[0] != args.N_node:
        cleanup_distributed()
        raise ValueError(f"N_node mismatch: data has {raw_o3.shape[0]}, args={args.N_node}")
    time_index = load_time_index(args.data_dir, raw_o3.shape[1])
    sites = get_sites(args.data_dir)
    if len(sites) != args.N_node:
        cleanup_distributed()
        raise ValueError(f"site count mismatch: {len(sites)} vs {args.N_node}")

    if is_main:
        print("[INFO] loading meteorological data...", flush=True)
    met_raw = load_met_data_raw(args.data_dir, sites, time_index, use_cache=bool(args.use_met_cache), verbose=is_main)
    split_sets, max_value, met_stats, split_desc, scale_stats, time_slices = prepare_noleak_data(
        args,
        raw_o3,
        time_index,
        met_raw,
    )
    split_sets = maybe_limit_split_windows(split_sets, args, verbose=is_main)
    trainX, trainY = split_sets["train"]
    validX, validY = split_sets["valid"]
    testX, testY = split_sets["test"]
    raw_o3_train = raw_o3[:, time_slices["train"]]

    if is_main:
        print(f"[INFO] trainX={trainX.shape}, trainY={trainY.shape}", flush=True)
        print(f"[INFO] validX={validX.shape}, validY={validY.shape}", flush=True)
        print(f"[INFO] testX={testX.shape}, testY={testY.shape}", flush=True)
        print(f"[INFO] train-only PE time range: {split_desc['time_slices']['train']}", flush=True)
        S, T_mat, PE_mat = build_graphs(args.data_dir, args, raw_o3_train)
        if not args.use_pe_graph:
            PE_mat = np.zeros_like(PE_mat, dtype=np.float32)
        pe_features = build_pe_features(args, raw_o3_train, sites, is_main=is_main)
        pe_features, PE_mat, pe_perm = apply_pe_shuffle(pe_features, PE_mat, args.pe_shuffle_seed)
        if pe_perm is not None:
            np.save(os.path.join(out_dir, "pe_shuffle_perm.npy"), pe_perm)
            print(
                f"[WARN] shuffled PE ablation enabled: pe_shuffle_seed={args.pe_shuffle_seed}; "
                "do not report this as the main PE model",
                flush=True,
            )
        np.save(os.path.join(out_dir, "S_matrix.npy"), S.astype(np.float32))
        np.save(os.path.join(out_dir, "T_matrix.npy"), T_mat.astype(np.float32))
        np.save(os.path.join(out_dir, "PE_matrix.npy"), PE_mat.astype(np.float32))
        np.save(os.path.join(out_dir, "pe_node_features.npy"), pe_features.astype(np.float32))
        np.save(os.path.join(out_dir, "validX.npy"), validX)
        np.save(os.path.join(out_dir, "validY.npy"), validY)
        np.save(os.path.join(out_dir, "testX.npy"), testX)
        np.save(os.path.join(out_dir, "testY.npy"), testY)
        if args.save_train_arrays:
            np.save(os.path.join(out_dir, "trainX.npy"), trainX)
            np.save(os.path.join(out_dir, "trainY.npy"), trainY)
        save_json(os.path.join(out_dir, "split_summary.json"), split_desc)
        save_json(os.path.join(out_dir, "scale_stats.json"), scale_stats)
        save_json(os.path.join(out_dir, "met_stats.json"), met_stats)
        save_json(
            os.path.join(out_dir, "graph_summary.json"),
            {
                "pe_source": args.pe_source,
                "pe_main_result_safe": args.pe_source == "train",
                "S_nnz": int(np.sum(S > 0)),
                "T_nnz": int(np.sum(T_mat > 0)),
                "PE_nnz": int(np.sum(PE_mat > 0)),
                "use_pe_graph": int(args.use_pe_graph),
                "use_pe_film": int(args.use_pe_film),
                "pe_graph_alpha": float(args.pe_graph_alpha),
                "pe_film_scale": float(args.pe_film_scale),
                "pe_film_zero_init": int(args.pe_film_zero_init),
                "normalize_pe_features": int(args.normalize_pe_features),
                "pe_shuffle_seed": int(args.pe_shuffle_seed),
                "pe_shuffle_enabled": int(args.pe_shuffle_seed >= 0),
                "pe_adaptive_loss": int(args.pe_adaptive_loss),
                "pe_loss_weight": float(args.pe_loss_weight),
                "pe_loss_start_step": int(args.pe_loss_start_step),
                "pe_loss_normalize": int(args.pe_loss_normalize),
                "pe_scales": parse_pe_scales(args.pe_scales),
                "pe_graph_threshold": float(args.pe_threshold),
                "pe_graph_sigma": float(args.pe_sigma),
            },
        )
        config = vars(args).copy()
        config.update(
            {
                "split_mode": "noleak",
                "model_name": "PE-DiffWaveNet",
                "clean_room_implementation": 1,
                "train_scale_max": float(max_value),
                "pe_feature_dim": int(pe_features.shape[1]),
                "academic_integrity_note": (
                    "Clean-room implementation; Graph WaveNet/MTGNN are cited baselines and not imported."
                ),
            }
        )
        save_json(os.path.join(out_dir, "config.json"), config)
        print(
            f"[INFO] graph nnz: S={int(np.sum(S > 0))}, T={int(np.sum(T_mat > 0))}, "
            f"PE={int(np.sum(PE_mat > 0))}",
            flush=True,
        )

    barrier()
    S = np.load(os.path.join(out_dir, "S_matrix.npy")).astype(np.float32)
    T_mat = np.load(os.path.join(out_dir, "T_matrix.npy")).astype(np.float32)
    PE_mat = np.load(os.path.join(out_dir, "PE_matrix.npy")).astype(np.float32)
    pe_features = np.load(os.path.join(out_dir, "pe_node_features.npy")).astype(np.float32)
    adj = torch.tensor(np.stack([S, T_mat, PE_mat], axis=0), dtype=torch.float32, device=device)

    train_ds = ATGCNDataset(trainX, trainY)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    if is_main:
        print(
            f"[INFO] train batches/epoch={len(train_loader)}, "
            f"effective_batch={args.batch_size * world_size}",
            flush=True,
        )

    model = PEDiffWaveNet(
        num_nodes=args.N_node,
        input_dim=args.m,
        hidden_size=args.hidden_size,
        pre_len=args.pre_len,
        diff_steps=args.diff_steps,
        pe_features=pe_features,
        use_pe_film=bool(args.use_pe_film),
        use_diffusion=bool(args.use_diffusion),
        use_adaptive_adj=bool(args.use_adaptive_adj),
        pe_graph_alpha=float(args.pe_graph_alpha),
        pe_film_scale=float(args.pe_film_scale),
        pe_film_zero_init=bool(args.pe_film_zero_init),
        normalize_pe_features=bool(args.normalize_pe_features),
    ).to(device)
    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    base_model = unwrap_model(model)
    if is_main:
        total = sum(p.numel() for p in base_model.parameters())
        print(f"[INFO] model_params={total}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = min(5, max(1, args.epochs))

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return max(args.lr_min / max(args.lr, 1e-12), 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = EMA(base_model, decay=args.ema_decay)
    horizon_weights_list = parse_horizon_weights(args.horizon_weights, args.pre_len)
    horizon_weights = (
        torch.tensor(horizon_weights_list, dtype=torch.float32, device=device)
        if horizon_weights_list is not None
        else None
    )
    pe_extra_weights_np = (
        build_pe_loss_extra_weights(
            pe_features,
            args.pre_len,
            args.pe_loss_weight,
            start_step=args.pe_loss_start_step,
            normalize=bool(args.pe_loss_normalize),
        )
        if args.pe_adaptive_loss
        else None
    )
    pe_extra_weights = (
        torch.tensor(pe_extra_weights_np, dtype=torch.float32, device=device)
        if pe_extra_weights_np is not None
        else None
    )
    if is_main and pe_extra_weights_np is not None:
        print(
            f"[INFO] PE-adaptive loss enabled: weight={args.pe_loss_weight}, "
            f"start_step={args.pe_loss_start_step}, range=[{pe_extra_weights_np.min():.4f}, {pe_extra_weights_np.max():.4f}]",
            flush=True,
        )
    history = {"train_loss": [], "valid_rmse": [], "valid_mae": [], "valid_mape": []}
    best_rmse = float("inf")
    best_epoch = -1
    best_state = None
    patience_counter = 0
    eval_inference_steps, eval_num_samples, eval_t_start_ratio = resolve_eval_sampling(args)

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        running_loss = 0.0
        if is_main:
            print(f"[INFO] start epoch {epoch + 1}/{args.epochs}", flush=True)
        for batch_idx, (bx, by) in enumerate(train_loader):
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            bs = bx.shape[0]
            optimizer.zero_grad(set_to_none=True)
            t_diff = torch.randint(0, base_model.diffusion.num_steps, (bs,), device=device)
            noise = torch.randn_like(by)
            y_noisy = base_model.diffusion.q_sample(by, t_diff, noise)[0] if args.use_diffusion else None
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            with amp_ctx:
                x0_pred, y_coarse, diff_pred, diff_target = model(
                    bx,
                    adj,
                    y_noisy,
                    t_diff,
                    return_coarse=True,
                    y_target=by,
                    noise=noise,
                    return_diffusion=True,
                )
                coarse_loss = weighted_horizon_loss((y_coarse - by) ** 2, horizon_weights, pe_extra_weights)
                if args.use_diffusion:
                    diff_loss = weighted_horizon_loss((diff_pred - diff_target) ** 2, horizon_weights, pe_extra_weights)
                    loss = diff_loss + float(args.coarse_weight) * coarse_loss
                else:
                    diff_loss = coarse_loss
                    loss = coarse_loss
                if args.lambda_temporal > 0:
                    grad_pred = x0_pred[:, 1:, :] - x0_pred[:, :-1, :]
                    grad_true = by[:, 1:, :] - by[:, :-1, :]
                    hw = horizon_weights[1:] if horizon_weights is not None else None
                    loss = loss + float(args.lambda_temporal) * weighted_horizon_loss((grad_pred - grad_true) ** 2, hw)

            if not torch.isfinite(loss.detach()).all():
                if is_main:
                    print(f"  [WARN] skip non-finite loss at epoch {epoch + 1} batch {batch_idx + 1}", flush=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            if not torch.isfinite(torch.as_tensor(grad_norm, device=device)):
                if is_main:
                    print(f"  [WARN] skip non-finite grad at epoch {epoch + 1} batch {batch_idx + 1}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
            ema.update(base_model)
            running_loss += float(loss.detach().item())
            if is_main and args.log_interval > 0 and (
                (batch_idx + 1) % args.log_interval == 0 or (batch_idx + 1) == len(train_loader)
            ):
                print(
                    f"  Epoch {epoch + 1} Batch {batch_idx + 1}/{len(train_loader)} "
                    f"| Loss={loss.item():.6f} | diff={diff_loss.item():.4f} "
                    f"| coarse={coarse_loss.item():.4f} | {time.time() - t0:.1f}s",
                    flush=True,
                )

        scheduler.step()
        avg_loss = reduce_sum(running_loss, device) / max(len(train_loader) * world_size, 1)
        if device.type == "cuda":
            torch.cuda.empty_cache()
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
            use_amp=False,
            rank=rank,
            world_size=world_size,
            gather_outputs=False,
            inference_steps=eval_inference_steps,
            num_samples=eval_num_samples,
            t_start_ratio=eval_t_start_ratio,
        )
        ema.restore(base_model)

        stop_training = False
        if is_main:
            history["train_loss"].append(avg_loss)
            history["valid_rmse"].append(valid_rmse)
            history["valid_mae"].append(valid_mae)
            history["valid_mape"].append(valid_mape)
            print(
                f"Epoch {epoch + 1}/{args.epochs} | Loss={avg_loss:.6f} | "
                f"Val RMSE={valid_rmse:.4f} | Val MAE={valid_mae:.4f} | "
                f"Val MAPE={valid_mape:.2f}% | LR={optimizer.param_groups[0]['lr']:.2e} | "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )
            if valid_rmse < best_rmse - args.min_delta:
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
                        f"[EARLY STOP] patience={args.patience}, best_val_rmse={best_rmse:.4f} "
                        f"at epoch {best_epoch + 1}",
                        flush=True,
                    )
                    stop_training = True
            ema.apply(base_model)
            torch.save(base_model.state_dict(), os.path.join(weight_dir, f"epoch_{epoch}_ema.pt"))
            ema.restore(base_model)
            torch.save(base_model.state_dict(), os.path.join(weight_dir, "last.pt"))
            np.save(os.path.join(out_dir, "train_loss.npy"), np.array(history["train_loss"], dtype=np.float32))
            np.save(os.path.join(out_dir, "valid_rmse.npy"), np.array(history["valid_rmse"], dtype=np.float32))
            np.save(os.path.join(out_dir, "valid_mae.npy"), np.array(history["valid_mae"], dtype=np.float32))
            np.save(os.path.join(out_dir, "valid_mape.npy"), np.array(history["valid_mape"], dtype=np.float32))

        if use_ddp:
            barrier()
            stop_tensor = torch.tensor(int(stop_training), dtype=torch.int64, device=device)
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())
            barrier()
        if stop_training:
            break

    if is_main and best_state is None:
        ema.apply(base_model)
        best_state = clone_state_dict(base_model)
        ema.restore(base_model)
        torch.save(best_state, os.path.join(weight_dir, "best_ema.pt"))
    barrier()

    base_model.load_state_dict(torch.load(os.path.join(weight_dir, "best_ema.pt"), map_location=device))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    set_seed(resolve_eval_seed(args, offset=1))
    test_rmse, test_mae, test_mape, test_preds, test_targets = distributed_eval(
        base_model,
        testX,
        testY,
        adj,
        device,
        args,
        max_value,
        use_amp=False,
        rank=rank,
        world_size=world_size,
        gather_outputs=bool(args.save_predictions),
        inference_steps=eval_inference_steps,
        num_samples=eval_num_samples,
        t_start_ratio=eval_t_start_ratio,
    )

    if is_main:
        metrics = {
            "split_mode": "noleak",
            "model_name": "PE-DiffWaveNet",
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
            "use_diffusion": int(args.use_diffusion),
            "use_pe_graph": int(args.use_pe_graph),
            "use_pe_film": int(args.use_pe_film),
            "pe_source": args.pe_source,
            "pe_graph_alpha": float(args.pe_graph_alpha),
            "pe_film_scale": float(args.pe_film_scale),
            "pe_film_zero_init": int(args.pe_film_zero_init),
            "normalize_pe_features": int(args.normalize_pe_features),
            "pe_shuffle_seed": int(args.pe_shuffle_seed),
            "pe_adaptive_loss": int(args.pe_adaptive_loss),
            "pe_loss_weight": float(args.pe_loss_weight),
            "pe_loss_start_step": int(args.pe_loss_start_step),
            "pe_loss_normalize": int(args.pe_loss_normalize),
        }
        if test_preds is not None and test_targets is not None:
            rmse_per_step, mae_per_step = compute_per_step_metrics_np(test_preds, test_targets, max_value)
            peak_info = compute_peak_metrics_np(test_preds, test_targets, max_value)
            metrics.update(
                {
                    "per_step_rmse": [float(x) for x in rmse_per_step],
                    "per_step_mae": [float(x) for x in mae_per_step],
                    "step4_6_rmse_avg": float(np.mean(rmse_per_step[3:])) if len(rmse_per_step) >= 6 else None,
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
        print("[INFO] PE-DiffWaveNet noleak training finished", flush=True)
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        if "peak_mode" in metrics:
            print(f"  Peak_mode      : {metrics['peak_mode']}", flush=True)
            print(f"  Peak_threshold : {metrics['peak_threshold']:.2f} μg/m³", flush=True)
            print(f"  Peak_count     : {metrics['peak_count']}", flush=True)
            print(f"  Peak_ratio     : {metrics['peak_ratio'] * 100:.2f}%", flush=True)
            if metrics["rmse_peak"] is not None:
                print(f"  RMSE_peak      : {metrics['rmse_peak']:.4f}", flush=True)
                print(f"  MAE_peak       : {metrics['mae_peak']:.4f}", flush=True)
            print("Per-step metrics:", flush=True)
            print(f"  {'Step':>4}  {'RMSE':>10}  {'MAE':>10}", flush=True)
            for i, (rmse_step, mae_step) in enumerate(zip(metrics["per_step_rmse"], metrics["per_step_mae"]), start=1):
                print(f"  {i:>4}  {rmse_step:>10.4f}  {mae_step:>10.4f}", flush=True)
        print(f"[INFO] outputs: {out_dir}", flush=True)
        print(f"[INFO] weights: {weight_dir}", flush=True)

    barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
