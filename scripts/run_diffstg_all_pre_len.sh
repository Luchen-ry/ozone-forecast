#!/bin/bash
# DiffSTG 多 pre_len 批量实验脚本
# 对齐 PE-DiffWaveNet 的 pre_len=1,3,6,12,24
# 配置: F=15 (O3+14气象变量), noleak切分, is_test=False, seed=42, T_h=24
#
# 用法:
#   nohup bash scripts/run_diffstg_all_pre_len.sh > logs/diffstg_all_pre_len.log 2>&1 &
#   tail -f logs/diffstg_all_pre_len.log

set -e

cd /home/cl/ozone-forecast/external_baselines/external_baselines/DiffSTG

LOG_DIR=/home/cl/ozone-forecast/logs
mkdir -p "$LOG_DIR"

# 公共参数
COMMON_ARGS="--data AIR_N95 --T_h 24 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 \
  --num_features 15 --split_mode noleak \
  --is_test False"

# pre_len 列表 (跳过已完成的 6)
PRE_LENS="1 3 12 24"

for P in $PRE_LENS; do
  LOG_FILE="$LOG_DIR/diffstg_air_n95_F15_noleak_p${P}.log"
  echo "================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running DiffSTG pre_len=$P"
  echo "  log: $LOG_FILE"
  echo "================================================"

  python train.py $COMMON_ARGS --T_p $P > "$LOG_FILE" 2>&1
  STATUS=$?

  if [ $STATUS -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] pre_len=$P DONE (exit 0)"
    # 提取 Final results
    grep -E "Final results|Final result" "$LOG_FILE" | tail -2
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] pre_len=$P FAILED (exit $STATUS)"
    tail -20 "$LOG_FILE"
  fi
  echo ""
done

echo "================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All DiffSTG experiments finished."
echo "================================================"
echo ""
echo "Results summary:"
for P in $PRE_LENS; do
  LOG_FILE="$LOG_DIR/diffstg_air_n95_F15_noleak_p${P}.log"
  echo "--- pre_len=$P ---"
  grep -E "Final results" "$LOG_FILE" 2>/dev/null || echo "  (no final results)"
done
