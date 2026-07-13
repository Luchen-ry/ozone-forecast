#!/bin/bash
# ============================================================
# PE-DiffWaveNet 批量实验运行命令
# 环境: atgcn conda, PyTorch 2.7+cu128, RTX 5080 (sm_120)
# ============================================================

cd /home/cl/ozone-forecast

# ============================================================
# 1. 环境准备 (PyTorch升级支持RTX 5080 sm_120)
# ============================================================
# pip uninstall torch torchvision torchaudio -y
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())"

# ============================================================
# 2. 主实验 (seq_len=24, pre_len=6, seed=42)
# ============================================================
DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_p6_l24_s42 \
  nohup bash scripts/run_train_pediffwavenet.sh 6 24 42 > logs/g3_pedw_p6_l24_s42.log 2>&1 &

# ============================================================
# 3. 多seed实验
# ============================================================
DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_p6_l24_s52 \
  nohup bash scripts/run_train_pediffwavenet.sh 6 24 52 > logs/g3_pedw_p6_l24_s52.log 2>&1 &

DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_p6_l24_s62 \
  nohup bash scripts/run_train_pediffwavenet.sh 6 24 62 > logs/g3_pedw_p6_l24_s62.log 2>&1 &

# ============================================================
# 4. 输入窗口实验
# ============================================================
DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_p6_l12_s42 \
  nohup bash scripts/run_train_pediffwavenet.sh 6 12 42 > logs/g3_pedw_p6_l12_s42.log 2>&1 &

DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_p6_l48_s42 \
  nohup bash scripts/run_train_pediffwavenet.sh 6 48 42 > logs/g3_pedw_p6_l48_s42.log 2>&1 &

# ============================================================
# 5. 预测步长实验
# ============================================================
DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_p1_l24_s42 \
  nohup bash scripts/run_train_pediffwavenet.sh 1 24 42 > logs/g3_pedw_p1_l24_s42.log 2>&1 &

# pre_len=3 需要降低学习率+关闭AMP+增加patience (修复训练不稳定)
DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_p3_l24_s42 LR=3e-4 AMP=0 PATIENCE=25 \
  nohup bash scripts/run_train_pediffwavenet.sh 3 24 42 > logs/g3_pedw_p3_l24_s42_fix.log 2>&1 &

DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_p12_l24_s42 \
  nohup bash scripts/run_train_pediffwavenet.sh 12 24 42 > logs/g3_pedw_p12_l24_s42.log 2>&1 &

DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_p24_l24_s42 \
  nohup bash scripts/run_train_pediffwavenet.sh 24 24 42 > logs/g3_pedw_p24_l24_s42.log 2>&1 &

# ============================================================
# 6. 消融实验
# ============================================================
# 无扩散
DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_no_diff_p6_l24_s42 USE_DIFFUSION=0 \
  nohup bash scripts/run_train_pediffwavenet.sh 6 24 42 > logs/g3_pedw_no_diff_p6_l24_s42.log 2>&1 &

# 无PE图
DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_no_pe_graph_p6_l24_s42 USE_PE_GRAPH=0 \
  nohup bash scripts/run_train_pediffwavenet.sh 6 24 42 > logs/g3_pedw_no_pe_graph_p6_l24_s42.log 2>&1 &

# 无PE FiLM
DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_no_pe_film_p6_l24_s42 USE_PE_FILM=0 \
  nohup bash scripts/run_train_pediffwavenet.sh 6 24 42 > logs/g3_pedw_no_pe_film_p6_l24_s42.log 2>&1 &

# PE shuffle
DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_pe_shuffle52_p6_l24_s42 PE_SHUFFLE_SEED=52 \
  nohup bash scripts/run_train_pediffwavenet.sh 6 24 42 > logs/g3_pedw_pe_shuffle52_p6_l24_s42.log 2>&1 &

# ============================================================
# 7. 批量运行全部实验 (可选)
# ============================================================
# nohup bash scripts/run_all_experiments.sh > logs/all_experiments.log 2>&1 &
# tail -f logs/all_experiments.log
