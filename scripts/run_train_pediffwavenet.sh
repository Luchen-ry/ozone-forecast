#!/usr/bin/env bash
set -euo pipefail

PRE_LEN="${1:-6}"
SEQ_LEN="${2:-24}"
SEED="${3:-42}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/code:${PYTHONPATH:-}"

DEVICE="${DEVICE:-cuda}"
EXP_NAME="${EXP_NAME:-student_pedw_p${PRE_LEN}_l${SEQ_LEN}_s${SEED}}"
HORIZON_WEIGHTS_VALUE="${HORIZON_WEIGHTS:-}"
if [[ -z "${HORIZON_WEIGHTS_VALUE}" ]]; then
  HORIZON_WEIGHTS_VALUE="1.0"
  for ((i = 2; i <= PRE_LEN; i++)); do
    HORIZON_WEIGHTS_VALUE="${HORIZON_WEIGHTS_VALUE},1.0"
  done
fi

python -u "${ROOT}/code/train_pediffwavenet_noleak.py" \
  --data_dir "${ROOT}" \
  --device "${DEVICE}" \
  --exp_name "${EXP_NAME}" \
  --pre_len "${PRE_LEN}" \
  --seq_len "${SEQ_LEN}" \
  --seed "${SEED}" \
  --N_node "${N_NODE:-95}" \
  --m "${INPUT_DIM:-15}" \
  --hidden_size "${HIDDEN_SIZE:-64}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-16}" \
  --lr "${LR:-7e-4}" \
  --lr_min "${LR_MIN:-1e-5}" \
  --epochs "${EPOCHS:-120}" \
  --patience "${PATIENCE:-15}" \
  --min_delta "${MIN_DELTA:-0.001}" \
  --diff_steps "${DIFF_STEPS:-50}" \
  --inference_steps "${INFERENCE_STEPS:-50}" \
  --num_samples "${NUM_SAMPLES:-3}" \
  --t_start_ratio "${T_START_RATIO:-0.25}" \
  --eval_inference_steps "${EVAL_INFERENCE_STEPS:-0}" \
  --eval_num_samples "${EVAL_NUM_SAMPLES:-0}" \
  --eval_t_start_ratio "${EVAL_T_START_RATIO:--1}" \
  --use_diffusion "${USE_DIFFUSION:-1}" \
  --coarse_only "${COARSE_ONLY:-0}" \
  --coarse_weight "${COARSE_WEIGHT:-0.08}" \
  --horizon_weights "${HORIZON_WEIGHTS_VALUE}" \
  --lambda_temporal "${LAMBDA_TEMPORAL:-0.0}" \
  --pe_adaptive_loss "${PE_ADAPTIVE_LOSS:-0}" \
  --pe_loss_weight "${PE_LOSS_WEIGHT:-0.15}" \
  --pe_loss_start_step "${PE_LOSS_START_STEP:-4}" \
  --pe_loss_normalize "${PE_LOSS_NORMALIZE:-1}" \
  --pe_source "${PE_SOURCE:-train}" \
  --pe_scales "${PE_SCALES:-6,9,12,24,48,72}" \
  --pe_dim "${PE_DIM:-3}" \
  --pe_delay "${PE_DELAY:-1}" \
  --pe_window_step "${PE_WINDOW_STEP:-1}" \
  --pe_shuffle_seed "${PE_SHUFFLE_SEED:--1}" \
  --pe_threshold "${PE_THRESHOLD:-0.9}" \
  --pe_sigma "${PE_SIGMA:-0.1}" \
  --use_pe_graph "${USE_PE_GRAPH:-1}" \
  --use_pe_film "${USE_PE_FILM:-1}" \
  --use_adaptive_adj "${USE_ADAPTIVE_ADJ:-1}" \
  --pe_graph_alpha "${PE_GRAPH_ALPHA:-1.0}" \
  --pe_film_scale "${PE_FILM_SCALE:-1.0}" \
  --pe_film_zero_init "${PE_FILM_ZERO_INIT:-0}" \
  --normalize_pe_features "${NORMALIZE_PE_FEATURES:-1}" \
  --temporal_topk "${TEMPORAL_TOPK:-12}" \
  --temporal_stride "${TEMPORAL_STRIDE:-1}" \
  --max_train_windows "${MAX_TRAIN_WINDOWS:-0}" \
  --max_valid_windows "${MAX_VALID_WINDOWS:-0}" \
  --max_test_windows "${MAX_TEST_WINDOWS:-0}" \
  --save_predictions "${SAVE_PREDICTIONS:-1}" \
  --save_train_arrays "${SAVE_TRAIN_ARRAYS:-0}" \
  --use_met_cache "${USE_MET_CACHE:-1}" \
  --amp "${AMP:-1}" \
  --grad_clip "${GRAD_CLIP:-1.0}" \
  --weight_decay "${WEIGHT_DECAY:-1e-4}" \
  --log_interval "${LOG_INTERVAL:-50}" \
  --ddp_timeout_minutes "${DDP_TIMEOUT_MINUTES:-180}"
