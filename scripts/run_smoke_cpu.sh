#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/code:${PYTHONPATH:-}"

python -u "${ROOT}/code/train_pediffwavenet_noleak.py" \
  --data_dir "${ROOT}" \
  --device cpu \
  --exp_name student_smoke_cpu \
  --pre_len 6 \
  --seq_len 24 \
  --seed 42 \
  --N_node 95 \
  --m 15 \
  --hidden_size 16 \
  --batch_size 2 \
  --eval_batch_size 2 \
  --lr 7e-4 \
  --epochs 1 \
  --patience 1 \
  --diff_steps 3 \
  --inference_steps 2 \
  --num_samples 1 \
  --eval_inference_steps 2 \
  --eval_num_samples 1 \
  --use_diffusion 1 \
  --use_pe_graph 1 \
  --use_pe_film 1 \
  --pe_window_step 168 \
  --max_train_windows 8 \
  --max_valid_windows 4 \
  --max_test_windows 4 \
  --save_predictions 0 \
  --save_train_arrays 0 \
  --use_met_cache 1 \
  --amp 0 \
  --log_interval 1
