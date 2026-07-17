#!/bin/bash
# ============================================================
# 生产实习  - 实际运行命令
# 包含: PE-DiffWaveNet 主实验/消融/多步长, DiffSTG F=1 baseline
# 不含: F=15 重跑(进行中), PM2.5
# ============================================================

set -e
cd /home/cl/ozone-forecast

# ============================================================
# 0. 环境准备
# ============================================================
# conda activate atgcn
# PyTorch 2.7+cu128 (RTX 5080 sm_120 兼容)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# ============================================================
# 1. PE-DiffWaveNet 主实验 (seq_len=24, pre_len=6, seed=42)
# ============================================================
DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_p6_l24_s42 \
  bash scripts/run_train_pediffwavenet.sh 6 24 42

# ============================================================
# 2. PE-DiffWaveNet 多 seed
# ============================================================
DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_p6_l24_s52 \
  bash scripts/run_train_pediffwavenet.sh 6 24 52

DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_p6_l24_s62 \
  bash scripts/run_train_pediffwavenet.sh 6 24 62

# ============================================================
# 3. PE-DiffWaveNet 输入窗口实验
# ============================================================
DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_p6_l12_s42 \
  bash scripts/run_train_pediffwavenet.sh 6 12 42

DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_p6_l48_s42 \
  bash scripts/run_train_pediffwavenet.sh 6 48 42

# ============================================================
# 4. PE-DiffWaveNet 预测步长实验
# ============================================================
DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_p1_l24_s42 \
  bash scripts/run_train_pediffwavenet.sh 1 24 42

# pre_len=3 需要降低学习率+关闭AMP+增加patience (修复训练不稳定)
DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_p3_l24_s42 LR=3e-4 AMP=0 PATIENCE=25 \
  bash scripts/run_train_pediffwavenet.sh 3 24 42

DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_p12_l24_s42 \
  bash scripts/run_train_pediffwavenet.sh 12 24 42

DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_p24_l24_s42 \
  bash scripts/run_train_pediffwavenet.sh 24 24 42

# ============================================================
# 5. PE-DiffWaveNet 消融实验
# ============================================================
# 无扩散
DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_no_diff_p6_l24_s42 USE_DIFFUSION=0 \
  bash scripts/run_train_pediffwavenet.sh 6 24 42

# 无 PE 图
DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_no_pe_graph_p6_l24_s42 USE_PE_GRAPH=0 \
  bash scripts/run_train_pediffwavenet.sh 6 24 42

# 无 PE FiLM
DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_no_pe_film_p6_l24_s42 USE_PE_FILM=0 \
  bash scripts/run_train_pediffwavenet.sh 6 24 42

# PE shuffle (打乱 PE 站点对应关系)
DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_pe_shuffle52_p6_l24_s42 PE_SHUFFLE_SEED=52 \
  bash scripts/run_train_pediffwavenet.sh 6 24 42

# ============================================================
# 6. DiffSTG Baseline (F=1: 仅 O3 单通道)
# 数据切分: default 0.6/0.8
# ============================================================
cd /home/cl/ozone-forecast/external_baselines_F1/DiffSTG

# F=1, pre_len=1 ( 101 epochs)
python train.py --data AIR_N95 --T_h 24 --T_p 1 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 \
  --num_features 1 --split_mode default --is_test False

# F=1, pre_len=3 ( 33 epochs)
python train.py --data AIR_N95 --T_h 24 --T_p 3 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 \
  --num_features 1 --split_mode default --is_test False

# F=1, pre_len=6 ( 194 epochs, best of both)
python train.py --data AIR_N95 --T_h 24 --T_p 6 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 \
  --num_features 1 --split_mode default --is_test False

# F=1, pre_len=12 ( 18 epochs early stop)
python train.py --data AIR_N95 --T_h 24 --T_p 12 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 \
  --num_features 1 --split_mode default --is_test False

# F=1, pre_len=24 ( 23 epochs early stop)
python train.py --data AIR_N95 --T_h 24 --T_p 24 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 \
  --num_features 1 --split_mode default --is_test False

# ============================================================
# 7. DiffSTG Baseline (F=15: O3 + 14气象, noleak 切分) [进行中]
# 调整: early_stop 10->30, lr 0.002->0.001 (减少验证MAE震荡)
# ============================================================
# cd /home/cl/ozone-forecast
# nohup bash scripts/rerun_diffstg_F15_noleak.sh > logs/rerun_F15_noleak.log 2>&1 &
# 完成后日志: logs/diffstg_air_n95_F15_noleak_p{1,3,6,12}_rerun.log
# 完成后将结果追加到 results.csv
