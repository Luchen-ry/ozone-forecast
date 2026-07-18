#!/bin/bash
# ============================================================
# 生产实习 G3 组 - 实际运行命令
# 完整覆盖：PE 主实验/消融/多步长, DiffSTG F=1, F=15, PM2.5/PM10, 置信区间
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

# F=1, pre_len=1 ( 101 epochs, MAE=9.61)
python train.py --data AIR_N95 --T_h 24 --T_p 1 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 \
  --num_features 1 --split_mode default --is_test False

# F=1, pre_len=3 (33 epochs, MAE=26.60)
python train.py --data AIR_N95 --T_h 24 --T_p 3 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 \
  --num_features 1 --split_mode default --is_test False

# F=1, pre_len=6 ( 194 epochs, MAE=29.20, best of both)
python train.py --data AIR_N95 --T_h 24 --T_p 6 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 \
  --num_features 1 --split_mode default --is_test False

# F=1, pre_len=12 ( 18 epochs early stop, MAE=37.59)
python train.py --data AIR_N95 --T_h 24 --T_p 12 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 \
  --num_features 1 --split_mode default --is_test False

# F=1, pre_len=24 ( 23 epochs early stop, MAE=40.61)
python train.py --data AIR_N95 --T_h 24 --T_p 24 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 \
  --num_features 1 --split_mode default --is_test False

# ============================================================
# 7. DiffSTG Baseline (F=15: O3 + 14气象, noleak 切分)
# 调整: early_stop 10->30, lr 0.002->0.001 (减少验证MAE震荡)
# p1/p3/p6/p12 重跑，p24 保留原结果
# ============================================================
cd /home/cl/ozone-forecast

# F=15, pre_len=1 (rerun, 51 epochs, MAE=10.46)
cd external_baselines_F1/DiffSTG
python train.py --data AIR_N95 --T_h 24 --T_p 1 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 \
  --num_features 15 --split_mode noleak --is_test False \
  --lr 0.001 --early_stop 30

# F=15, pre_len=3 (rerun, 237 epochs, MAE=17.86)
python train.py --data AIR_N95 --T_h 24 --T_p 3 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 \
  --num_features 15 --split_mode noleak --is_test False \
  --lr 0.001 --early_stop 30

# F=15, pre_len=6 (rerun, 196 epochs, MAE=24.83)
python train.py --data AIR_N95 --T_h 24 --T_p 6 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 \
  --num_features 15 --split_mode noleak --is_test False \
  --lr 0.001 --early_stop 30

# F=15, pre_len=12 (rerun, 33 epochs, MAE=33.30)
python train.py --data AIR_N95 --T_h 24 --T_p 12 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 \
  --num_features 15 --split_mode noleak --is_test False \
  --lr 0.001 --early_stop 30

# F=15, pre_len=24 (original kept, 46 epochs, MAE=27.11)
python train.py --data AIR_N95 --T_h 24 --T_p 24 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 \
  --num_features 15 --split_mode noleak --is_test False \
  --lr 0.002 --early_stop 10

# ============================================================
# 8. 进阶实验
# ============================================================
cd /home/cl/ozone-forecast/external_baselines_F1

# 8.1 三污染物对比 (O3 / PM2.5 / PM10, F=1, pre_len=6)
# 数据准备：从 N95 原始数据抽取各污染物单通道
python prepare_air_n95_for_diffstg.py --pollutant PM25
python prepare_air_n95_for_diffstg.py --pollutant PM10

# 训练（DiffSTG F=1, pre_len=6, 各污染物）
cd DiffSTG
python train.py --data AIR_N95_PM25 --T_h 24 --T_p 6 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 --num_features 1 --is_test False

python train.py --data AIR_N95_PM10 --T_h 24 --T_p 6 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 --num_features 1 --is_test False

# 8.2 邻接矩阵对比 (Distance / Correlation / PE, F=1, pre_len=6)
cd /home/cl/ozone-forecast/external_baselines_F1
python prepare_alt_adj.py  # 生成 corr_adj.npy 和 pe_adj.npy
bash run_alt_adj_exps.sh   # 跑三种邻接矩阵对比实验

# 8.3 概率预测指标 (CRPS / MIS, 50 次采样推理)
python compute_probability_metrics.py --data AIR_N95 --pre_len 6 --n_samples 50

# 8.4 可视化（生成所有进阶实验图表）
cd plot_scripts
python plot_prelen_decay.py        # 多步长衰减对比
python plot_pollutant_comparison.py # 三污染物对比
python plot_adjacency_comparison.py # 邻接矩阵对比
python plot_diffstg_forecast.py     # O3 单步预测可视化
