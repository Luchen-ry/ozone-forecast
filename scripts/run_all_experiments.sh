#!/usr/bin/env bash
# ============================================================
# PE-DiffWaveNet 批量正式实验
# 用法（服务器后台）:
#   cd /home/cl/ozone-forecast
#   nohup bash scripts/run_all_experiments.sh > logs/all_experiments.log 2>&1 &
#   tail -f logs/all_experiments.log
#
# 可选: 指定 GPU
#   CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_all_experiments.sh > logs/all_experiments.log 2>&1 &
# ============================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/logs"
RESULTS_CSV="${ROOT}/experiment_results.csv"
mkdir -p "${LOG_DIR}"

# 结果记录表（字段向 paper_assets/table1 对齐）
echo "experiment_id,model,seq_len,pre_len,seed,use_diffusion,use_pe_graph,use_pe_film,pe_shuffle_seed,rmse,mae,mape,peak_rmse,step6_rmse,best_epoch,output_dir,log_file,command,status" > "${RESULTS_CSV}"

# ---- 单个实验函数 ----
run_exp() {
  local EXP_NAME="$1"
  local PRE_LEN="$2"
  local SEQ_LEN="$3"
  local SEED="$4"
  shift 4

  local USE_DIFFUSION=1
  local USE_PE_GRAPH=1
  local USE_PE_FILM=1
  local PE_SHUFFLE_SEED=-1
  local EXTRA_ENV=""

  for arg in "$@"; do
    EXTRA_ENV="${EXTRA_ENV} ${arg}"
    case "$arg" in
      USE_DIFFUSION=*)   USE_DIFFUSION="${arg#USE_DIFFUSION=}";;
      USE_PE_GRAPH=*)    USE_PE_GRAPH="${arg#USE_PE_GRAPH=}";;
      USE_PE_FILM=*)     USE_PE_FILM="${arg#USE_PE_FILM=}";;
      PE_SHUFFLE_SEED=*) PE_SHUFFLE_SEED="${arg#PE_SHUFFLE_SEED=}";;
    esac
  done

  local LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
  local OUT_DIR="${ROOT}/matrix_N95_PEDiffWaveNet_noleak_${EXP_NAME}"
  # DEVICE=cuda EPOCHS=120 为正式配置（run_train_pediffwavenet.sh 默认 hidden=64 diff_steps=50）
  local CMD="cd '${ROOT}' && DEVICE=cuda EPOCHS=120 EXP_NAME=${EXP_NAME}${EXTRA_ENV} bash scripts/run_train_pediffwavenet.sh ${PRE_LEN} ${SEQ_LEN} ${SEED}"

  echo ""
  echo "============================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: ${EXP_NAME}  (seq=${SEQ_LEN} pre=${PRE_LEN} seed=${SEED} diff=${USE_DIFFUSION} pe_graph=${USE_PE_GRAPH} pe_film=${USE_PE_FILM} pe_shuffle=${PE_SHUFFLE_SEED})"
  echo "============================================================"

  eval "${CMD}" > "${LOG_FILE}" 2>&1
  local EXIT_CODE=$?

  # 提取指标
  local STATUS="failed(exit=${EXIT_CODE})"
  local RMSE="" MAE="" MAPE="" PEAK_RMSE="" STEP6_RMSE="" BEST_EPOCH=""
  local METRICS_FILE="${OUT_DIR}/metrics_summary.json"

  if [ ${EXIT_CODE} -eq 0 ] && [ -f "${METRICS_FILE}" ]; then
    STATUS="success"
    eval "$(python3 - <<PYEOF 2>/dev/null || true
import json
with open("${METRICS_FILE}") as f:
    d = json.load(f)
print(f"RMSE='{d.get('test_rmse','')}'")
print(f"MAE='{d.get('test_mae','')}'")
print(f"MAPE='{d.get('test_mape','')}'")
p = d.get('rmse_peak')
print(f"PEAK_RMSE='{p if p is not None else ''}'")
ps = d.get('per_step_rmse', [])
print(f"STEP6_RMSE='{ps[5] if len(ps) >= 6 else ''}'")
print(f"BEST_EPOCH='{d.get('best_epoch','')}'")
PYEOF
)" || true
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] END: ${EXP_NAME} -> ${STATUS}  RMSE=${RMSE} MAE=${MAE} MAPE=${MAPE}"

  echo "${EXP_NAME},PE-DiffWaveNet,${SEQ_LEN},${PRE_LEN},${SEED},${USE_DIFFUSION},${USE_PE_GRAPH},${USE_PE_FILM},${PE_SHUFFLE_SEED},${RMSE},${MAE},${MAPE},${PEAK_RMSE},${STEP6_RMSE},${BEST_EPOCH},${OUT_DIR},${LOG_FILE},\"${CMD}\",${STATUS}" >> "${RESULTS_CSV}"
}

# ============================================================
# 实验矩阵（共 13 个实验）
# ============================================================

# --- 1. 主实验 + 多 seed（3个）---
run_exp g3_pedw_p6_l24_s42  6 24 42
run_exp g3_pedw_p6_l24_s52  6 24 52
run_exp g3_pedw_p6_l24_s62  6 24 62

# --- 2. 输入窗口消融（2个）---
run_exp g3_pedw_p6_l12_s42  6 12 42
run_exp g3_pedw_p6_l48_s42  6 48 42

# --- 3. 预测步长（4个）---
run_exp g3_pedw_p1_l24_s42   1 24 42
run_exp g3_pedw_p3_l24_s42   3 24 42
run_exp g3_pedw_p12_l24_s42 12 24 42
run_exp g3_pedw_p24_l24_s42 24 24 42

# --- 4. 模块消融（4个）---
run_exp g3_pedw_no_diff_p6_l24_s42      6 24 42 USE_DIFFUSION=0
run_exp g3_pedw_no_pe_graph_p6_l24_s42  6 24 42 USE_PE_GRAPH=0
run_exp g3_pedw_no_pe_film_p6_l24_s42   6 24 42 USE_PE_FILM=0
run_exp g3_pedw_pe_shuffle52_p6_l24_s42 6 24 42 PE_SHUFFLE_SEED=52

# ============================================================
echo ""
echo "============================================================"
echo "全部实验完成！结果汇总: ${RESULTS_CSV}"
echo "各实验日志: ${LOG_DIR}/<exp_name>.log"
echo "============================================================"
cat "${RESULTS_CSV}"
