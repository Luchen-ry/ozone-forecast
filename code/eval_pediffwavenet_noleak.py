#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Leakage-safe PE-DiffWaveNet evaluation."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from eval_atgcn_pe3_noleak import (
    apply_residual_calibrator,
    compute_peak_metrics,
    compute_per_step_metrics,
    fit_residual_calibrator,
    format_metrics_text,
    load_valid_arrays,
    parse_node_indices,
    plot_per_step_metrics,
    plot_predictions,
    plot_training_curves,
    summarize_calibration,
)
from pediffwavenet_model import PEDiffWaveNet
from train_atgcn_pe3 import compute_metrics, set_seed
from train_atgcn_pe3_noleak import resolve_device, sample_in_batches


def parse_args():
    p = argparse.ArgumentParser(description="Leakage-safe PE-DiffWaveNet evaluation")
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--pre_len", type=int, default=6)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--N_node", type=int, default=95)
    p.add_argument("--m", type=int, default=15)
    p.add_argument("--data_dir", type=str, default="/home/chenxudong/graduate/代码 2/代码/代码")
    p.add_argument("--exp_name", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--diff_steps", type=int, default=None)
    p.add_argument("--inference_steps", type=int, default=None)
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--t_start_ratio", type=float, default=None)
    p.add_argument("--coarse_only", type=int, default=None)
    p.add_argument("--use_diffusion", type=int, default=None)
    p.add_argument("--use_pe_film", type=int, default=None)
    p.add_argument("--use_adaptive_adj", type=int, default=None)
    p.add_argument("--pe_graph_alpha", type=float, default=None)
    p.add_argument("--pe_film_scale", type=float, default=None)
    p.add_argument("--pe_film_zero_init", type=int, default=None)
    p.add_argument("--normalize_pe_features", type=int, default=None)
    p.add_argument("--self_condition_mode", type=str, default="prev_pred")
    p.add_argument("--self_condition_mix", type=float, default=0.5)
    p.add_argument("--save_predictions", type=int, default=1)
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--use_saved_config", type=int, default=1)
    p.add_argument("--peak_mode", type=str, default="percentile", choices=("percentile", "fixed"))
    p.add_argument("--peak_percentile", type=float, default=90.0)
    p.add_argument("--peak_thr", type=float, default=0.2)
    p.add_argument("--plot_nodes", type=str, default="0,10,30,50,70,90")
    p.add_argument("--plot_max_samples", type=int, default=200)
    p.add_argument("--calibrate_residual", type=int, default=0)
    p.add_argument(
        "--calibration_mode",
        type=str,
        default="bias_horizon_linear",
        choices=("horizon_bias", "horizon_node_bias", "horizon_linear", "bias_horizon_linear"),
    )
    p.add_argument("--calibration_features", type=str, default="pred,last,trend1")
    p.add_argument("--calibration_ridge", type=float, default=1.0)
    p.add_argument("--calibration_shrinkage", type=float, default=256.0)
    p.add_argument("--calibration_clip_min", type=float, default=0.0)
    p.add_argument("--calibration_clip_max", type=float, default=-1.0)
    p.add_argument(
        "--calibration_guard_mode",
        type=str,
        default="none",
        choices=("none", "pred_peak", "hist_peak", "hist_or_pred_peak"),
    )
    p.add_argument("--calibration_guard_thr", type=float, default=0.16)
    p.add_argument("--calibration_guard_margin", type=float, default=0.0)
    p.add_argument("--calibration_guard_negative_only", type=int, default=1)
    return p.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def apply_config(args, out_dir):
    config_path = os.path.join(out_dir, "config.json")
    if not args.use_saved_config or not os.path.exists(config_path):
        return args
    saved = load_json(config_path)
    for key in (
        "seq_len",
        "pre_len",
        "hidden_size",
        "N_node",
        "m",
        "diff_steps",
        "inference_steps",
        "num_samples",
        "t_start_ratio",
        "coarse_only",
        "use_diffusion",
        "use_pe_film",
        "use_adaptive_adj",
        "pe_graph_alpha",
        "pe_film_scale",
        "pe_film_zero_init",
        "normalize_pe_features",
    ):
        saved_structural = (
            "seq_len",
            "pre_len",
            "hidden_size",
            "N_node",
            "m",
            "diff_steps",
            "use_diffusion",
            "use_pe_film",
            "use_adaptive_adj",
            "pe_graph_alpha",
            "pe_film_scale",
            "pe_film_zero_init",
            "normalize_pe_features",
        )
        if key in saved and (getattr(args, key, None) is None or key in saved_structural):
            setattr(args, key, saved[key])
    print(f"[INFO] loaded structural config from {config_path}", flush=True)
    return args


def load_saved_arrays(out_dir):
    required = [
        "testX.npy",
        "testY.npy",
        "S_matrix.npy",
        "T_matrix.npy",
        "PE_matrix.npy",
        "pe_node_features.npy",
        "scale_stats.json",
    ]
    missing = [name for name in required if not os.path.exists(os.path.join(out_dir, name))]
    if missing:
        raise FileNotFoundError(f"missing saved PE-DiffWaveNet artifacts under {out_dir}: {missing}")
    testX = np.load(os.path.join(out_dir, "testX.npy")).astype(np.float32)
    testY = np.load(os.path.join(out_dir, "testY.npy")).astype(np.float32)
    S = np.load(os.path.join(out_dir, "S_matrix.npy")).astype(np.float32)
    T = np.load(os.path.join(out_dir, "T_matrix.npy")).astype(np.float32)
    PE = np.load(os.path.join(out_dir, "PE_matrix.npy")).astype(np.float32)
    pe_features = np.load(os.path.join(out_dir, "pe_node_features.npy")).astype(np.float32)
    max_value = float(load_json(os.path.join(out_dir, "scale_stats.json"))["o3_train_max"])
    return testX, testY, S, T, PE, pe_features, max_value


def load_best_epoch(out_dir):
    path = os.path.join(out_dir, "metrics_summary.json")
    if not os.path.exists(path):
        return -1
    try:
        best_epoch = int(load_json(path).get("best_epoch", 0))
    except Exception:
        return -1
    return best_epoch - 1 if best_epoch > 0 else -1


def main():
    args = parse_args()
    suffix = f"_{args.exp_name}" if args.exp_name else ""
    out_dir = os.path.join(args.data_dir, f"matrix_N95_PEDiffWaveNet_noleak{suffix}")
    weight_dir = os.path.join(args.data_dir, f"weights_N95/weights_pediffwavenet_noleak{suffix}")
    eval_dir = os.path.join(args.data_dir, f"eval_results_pediffwavenet_noleak{suffix}")
    os.makedirs(eval_dir, exist_ok=True)
    args = apply_config(args, out_dir)
    saved_config_path = os.path.join(out_dir, "config.json")
    saved_config = load_json(saved_config_path) if os.path.exists(saved_config_path) else {}
    if args.diff_steps is None:
        args.diff_steps = 50
    if args.inference_steps is None:
        args.inference_steps = 50
    if args.num_samples is None:
        args.num_samples = 3
    if args.t_start_ratio is None:
        args.t_start_ratio = 0.25
    if args.coarse_only is None:
        args.coarse_only = 0
    if args.use_diffusion is None:
        args.use_diffusion = 1
    if args.use_pe_film is None:
        args.use_pe_film = 1
    if args.use_adaptive_adj is None:
        args.use_adaptive_adj = 1
    if args.pe_graph_alpha is None:
        args.pe_graph_alpha = 1.0
    if args.pe_film_scale is None:
        args.pe_film_scale = 1.0
    if args.pe_film_zero_init is None:
        args.pe_film_zero_init = 0
    if args.normalize_pe_features is None:
        args.normalize_pe_features = 1

    device = resolve_device(args.device)
    set_seed(args.seed)
    print(f"[INFO] device={device}", flush=True)
    print(f"[INFO] output_dir={out_dir}", flush=True)
    testX, testY, S, T, PE, pe_features, max_value = load_saved_arrays(out_dir)
    adj = torch.tensor(np.stack([S, T, PE], axis=0), dtype=torch.float32, device=device)
    ckpt_path = args.ckpt_path or os.path.join(weight_dir, "best_ema.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

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
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    valid_preds = None
    validY = None
    calibrator = None
    if args.calibrate_residual:
        validX, validY = load_valid_arrays(out_dir)
        if validX is None:
            raise FileNotFoundError("validX.npy / validY.npy are required for DriftCalib")
        print(
            f"[INFO] fitting noleak residual calibrator on valid split: mode={args.calibration_mode}, "
            f"features={args.calibration_features}, ridge={args.calibration_ridge}, "
            f"shrinkage={args.calibration_shrinkage}",
            flush=True,
        )
        set_seed(args.seed + 1500)
        valid_preds = sample_in_batches(model, validX, adj, device, args, use_amp=False)
        calibrator = fit_residual_calibrator(
            valid_preds,
            validY,
            mode=args.calibration_mode,
            shrinkage=args.calibration_shrinkage,
            validX=validX,
            features=args.calibration_features,
            ridge=args.calibration_ridge,
        )

    set_seed(args.seed + 2000)
    preds = sample_in_batches(model, testX, adj, device, args, use_amp=False)
    test_rmse, test_mae, test_mape = compute_metrics(preds, testY, max_value)
    rmse_per_step, mae_per_step = compute_per_step_metrics(preds, testY, max_value)
    peak_info = compute_peak_metrics(
        preds,
        testY,
        max_value,
        peak_mode=args.peak_mode,
        peak_thr=args.peak_thr,
        percentile=args.peak_percentile,
    )
    metrics = {
        "split_mode": "noleak",
        "model_name": "PE-DiffWaveNet",
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "test_mape": float(test_mape),
        "relative_rmse": float(test_rmse / max_value * 100.0),
        "relative_mae": float(test_mae / max_value * 100.0),
        "per_step_rmse": [float(x) for x in rmse_per_step],
        "per_step_mae": [float(x) for x in mae_per_step],
        "step4_6_rmse_avg": float(np.mean(rmse_per_step[3:])) if len(rmse_per_step) >= 6 else None,
        "peak_mode": peak_info["peak_mode"],
        "peak_threshold": float(peak_info["peak_threshold"]),
        "peak_count": int(peak_info["peak_count"]),
        "peak_ratio": float(peak_info["peak_ratio"]),
        "rmse_peak": None if peak_info["rmse_peak"] is None else float(peak_info["rmse_peak"]),
        "mae_peak": None if peak_info["mae_peak"] is None else float(peak_info["mae_peak"]),
        "train_scale_max": float(max_value),
        "checkpoint": ckpt_path,
        "use_diffusion": int(args.use_diffusion),
        "use_pe_graph": int(saved_config.get("use_pe_graph", int(np.sum(PE > 0) > 0))),
        "use_pe_film": int(args.use_pe_film),
        "pe_source": saved_config.get("pe_source", "unknown"),
        "pe_shuffle_seed": int(saved_config.get("pe_shuffle_seed", -1)),
        "use_adaptive_adj": int(args.use_adaptive_adj),
        "pe_graph_alpha": float(args.pe_graph_alpha),
        "pe_film_scale": float(args.pe_film_scale),
        "pe_film_zero_init": int(args.pe_film_zero_init),
        "normalize_pe_features": int(args.normalize_pe_features),
    }

    calibrated_outputs = None
    if calibrator is not None:
        reference_history = testX[:, -1, :, 0].astype(np.float32)
        cal_preds = apply_residual_calibrator(
            preds,
            calibrator,
            clip_min=args.calibration_clip_min,
            clip_max=args.calibration_clip_max,
            reference=reference_history,
            guard_mode=args.calibration_guard_mode,
            guard_thr=args.calibration_guard_thr,
            guard_margin=args.calibration_guard_margin,
            guard_negative_only=bool(args.calibration_guard_negative_only),
            X=testX,
        )
        cal_rmse, cal_mae, cal_mape = compute_metrics(cal_preds, testY, max_value)
        cal_rmse_per_step, cal_mae_per_step = compute_per_step_metrics(cal_preds, testY, max_value)
        cal_peak_info = compute_peak_metrics(
            cal_preds,
            testY,
            max_value,
            peak_mode=args.peak_mode,
            peak_thr=args.peak_thr,
            percentile=args.peak_percentile,
        )
        cal_metrics = {
            **metrics,
            "test_rmse": float(cal_rmse),
            "test_mae": float(cal_mae),
            "test_mape": float(cal_mape),
            "relative_rmse": float(cal_rmse / max_value * 100.0),
            "relative_mae": float(cal_mae / max_value * 100.0),
            "per_step_rmse": [float(x) for x in cal_rmse_per_step],
            "per_step_mae": [float(x) for x in cal_mae_per_step],
            "step4_6_rmse_avg": float(np.mean(cal_rmse_per_step[3:])) if len(cal_rmse_per_step) >= 6 else None,
            "peak_mode": cal_peak_info["peak_mode"],
            "peak_threshold": float(cal_peak_info["peak_threshold"]),
            "peak_count": int(cal_peak_info["peak_count"]),
            "peak_ratio": float(cal_peak_info["peak_ratio"]),
            "rmse_peak": None if cal_peak_info["rmse_peak"] is None else float(cal_peak_info["rmse_peak"]),
            "mae_peak": None if cal_peak_info["mae_peak"] is None else float(cal_peak_info["mae_peak"]),
            "raw_test_rmse": float(test_rmse),
            "raw_test_mae": float(test_mae),
            "raw_test_mape": float(test_mape),
            "calibration": summarize_calibration(calibrator, max_value),
            "calibration_guard_mode": args.calibration_guard_mode,
            "calibration_guard_thr": float(args.calibration_guard_thr),
            "calibration_guard_margin": float(args.calibration_guard_margin),
            "calibration_guard_negative_only": int(args.calibration_guard_negative_only),
        }
        metrics["calibrated"] = {
            "test_rmse": float(cal_rmse),
            "test_mae": float(cal_mae),
            "test_mape": float(cal_mape),
            "relative_rmse": float(cal_rmse / max_value * 100.0),
            "relative_mae": float(cal_mae / max_value * 100.0),
            "rmse_peak": None if cal_peak_info["rmse_peak"] is None else float(cal_peak_info["rmse_peak"]),
            "mae_peak": None if cal_peak_info["mae_peak"] is None else float(cal_peak_info["mae_peak"]),
            "per_step_rmse": [float(x) for x in cal_rmse_per_step],
            "per_step_mae": [float(x) for x in cal_mae_per_step],
            "step4_6_rmse_avg": float(np.mean(cal_rmse_per_step[3:])) if len(cal_rmse_per_step) >= 6 else None,
            "calibration": summarize_calibration(calibrator, max_value),
        }
        calibrated_outputs = (cal_preds, cal_metrics, cal_rmse_per_step, cal_mae_per_step, cal_peak_info)

    if args.save_predictions:
        np.save(os.path.join(out_dir, "eval_test_predictions.npy"), preds)
        np.save(os.path.join(out_dir, "eval_test_targets.npy"), testY)
        np.save(os.path.join(eval_dir, "test_predictions.npy"), preds * float(max_value))
        np.save(os.path.join(eval_dir, "test_targets.npy"), testY * float(max_value))
        if calibrator is not None:
            np.save(os.path.join(out_dir, "eval_valid_predictions.npy"), valid_preds)
            np.save(os.path.join(out_dir, "eval_valid_targets.npy"), validY)
            np.save(os.path.join(eval_dir, "valid_predictions.npy"), valid_preds * float(max_value))
            np.save(os.path.join(eval_dir, "valid_targets.npy"), validY * float(max_value))
            np.save(os.path.join(out_dir, "residual_calibrator_bias.npy"), calibrator["bias"])
            np.save(os.path.join(out_dir, "eval_test_predictions_calibrated.npy"), calibrated_outputs[0])
            np.save(os.path.join(eval_dir, "test_predictions_calibrated.npy"), calibrated_outputs[0] * float(max_value))

    best_epoch = load_best_epoch(out_dir)
    plot_training_curves(out_dir, best_epoch, eval_dir)
    plot_per_step_metrics(rmse_per_step, mae_per_step, eval_dir)
    node_indices = parse_node_indices(args.plot_nodes, args.N_node)
    plot_predictions(preds, testY, max_value, eval_dir, node_indices, max_samples=args.plot_max_samples)
    metrics_text = format_metrics_text(metrics, rmse_per_step, mae_per_step, peak_info)
    save_json(os.path.join(out_dir, "eval_metrics.json"), metrics)
    save_json(os.path.join(eval_dir, "eval_metrics.json"), metrics)
    with open(os.path.join(eval_dir, "eval_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(metrics_text + "\n")

    calibrated_text = None
    if calibrated_outputs is not None:
        cal_preds, cal_metrics, cal_rmse_per_step, cal_mae_per_step, cal_peak_info = calibrated_outputs
        cal_dir = os.path.join(eval_dir, "calibrated")
        os.makedirs(cal_dir, exist_ok=True)
        calibrated_text = format_metrics_text(cal_metrics, cal_rmse_per_step, cal_mae_per_step, cal_peak_info)
        save_json(os.path.join(out_dir, "eval_metrics_calibrated.json"), cal_metrics)
        save_json(os.path.join(eval_dir, "eval_metrics_calibrated.json"), cal_metrics)
        with open(os.path.join(eval_dir, "eval_metrics_calibrated.txt"), "w", encoding="utf-8") as f:
            f.write(calibrated_text + "\n")
        save_json(os.path.join(eval_dir, "calibration_summary.json"), cal_metrics["calibration"])
        plot_per_step_metrics(cal_rmse_per_step, cal_mae_per_step, cal_dir)
        plot_predictions(cal_preds, testY, max_value, cal_dir, node_indices, max_samples=args.plot_max_samples)

    print("[INFO] PE-DiffWaveNet noleak evaluation finished", flush=True)
    print(metrics_text, flush=True)
    if calibrated_text is not None:
        print("\n[INFO] calibrated noleak evaluation", flush=True)
        print(calibrated_text, flush=True)
    print(f"[INFO] eval artifacts: {eval_dir}", flush=True)


if __name__ == "__main__":
    main()
