#!/bin/bash
# DiffSTG F=15 重跑 p1/p3/p6/p12 (early_stop=30, lr=0.001 减少震荡)
# p24 保留(已充分收敛 46 epoch)
#
# 配置: F=15 (O3+14气象), noleak切分, is_test=False, seed=42, T_h=24, bs=4
# 调整: early_stop 10->30, lr 0.002->0.001 (降低学习率减少验证MAE震荡)
#
# 用法:
#   nohup bash scripts/rerun_diffstg_F15_noleak.sh > logs/rerun_F15_noleak.log 2>&1 &
#   tail -f logs/rerun_F15_noleak.log

set -e

cd /home/cl/ozone-forecast/external_baselines/external_baselines/DiffSTG

LOG_DIR=/home/cl/ozone-forecast/logs
mkdir -p "$LOG_DIR"

# 公共参数 (early_stop=30, lr=0.001)
COMMON_ARGS="--data AIR_N95 --T_h 24 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 \
  --num_features 15 --split_mode noleak \
  --is_test False --lr 0.001 --early_stop 30"

# 重跑 p1/p3/p6/p12 (p24 已充分收敛, 保留)
PRE_LENS="1 3 6 12"

for P in $PRE_LENS; do
  LOG_FILE="$LOG_DIR/diffstg_air_n95_F15_noleak_p${P}_rerun.log"
  echo "================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rerun DiffSTG F=15 pre_len=$P (early_stop=30, lr=0.001)"
  echo "  log: $LOG_FILE"
  echo "================================================"

  python train.py $COMMON_ARGS --T_p $P > "$LOG_FILE" 2>&1
  STATUS=$?

  if [ $STATUS -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] pre_len=$P DONE (exit 0)"
    grep -E "Final results" "$LOG_FILE" | tail -1
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] pre_len=$P FAILED (exit $STATUS)"
    tail -20 "$LOG_FILE"
  fi
  echo ""
done

echo "================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] F=15 rerun finished."
echo "================================================"
echo ""
echo "Results summary (rerun):"
for P in $PRE_LENS; do
  LOG_FILE="$LOG_DIR/diffstg_air_n95_F15_noleak_p${P}_rerun.log"
  echo "--- pre_len=$P ---"
  grep -E "Final results" "$LOG_FILE" 2>/dev/null || echo "  (no final results)"
done
echo ""
echo "p24 (保留原结果):"
grep -E "Final results" "$LOG_DIR/diffstg_air_n95_F15_noleak_p24.log" 2>/dev/null || echo "  (no original results)"
