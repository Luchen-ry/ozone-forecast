#!/bin/bash
# DiffSTG F=1 (O3 only) 多 pre_len 批量实验脚本
# 重跑所有 pre_len=1,3,6,12,24 (使用修复后的 str2bool 参数解析)
#
# 配置: F=1 (仅O3), 默认切分(0.6/0.8), is_test=False, seed=42, T_h=24
# F=1 数据: flow.npy (8717, 95, 1), adj.npy (95, 95)
#
# 用法:
#   nohup bash scripts/run_diffstg_F1_all_pre_len.sh > logs/diffstg_F1_all_pre_len.log 2>&1 &
#   tail -f logs/diffstg_F1_all_pre_len.log

set -e

cd /home/cl/ozone-forecast/external_baselines_F1/DiffSTG

LOG_DIR=/home/cl/ozone-forecast/logs
mkdir -p "$LOG_DIR"

# 公共参数 (is_test=False 完整训练 300 epoch, early_stop=10)
COMMON_ARGS="--data AIR_N95 --T_h 24 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --n_samples 1 --is_test False"

# pre_len 列表: 全部重跑
PRE_LENS="1 3 6 12 24"

for P in $PRE_LENS; do
  LOG_FILE="$LOG_DIR/diffstg_F1_air_n95_p${P}.log"
  echo "================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running DiffSTG F=1 pre_len=$P"
  echo "  log: $LOG_FILE"
  echo "================================================"

  # batch_size=4 对齐组员配置 (组员 p1 用 bs=4 跑了 101 epoch, MAE=9.61)
  # bs=4 每epoch梯度更新多4倍, 模型学习更充分, 收敛更好
  BS=4

  python train.py $COMMON_ARGS --T_p $P --batch_size $BS > "$LOG_FILE" 2>&1
  STATUS=$?

  if [ $STATUS -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] pre_len=$P DONE (exit 0)"
    grep -E "Final results|Final result" "$LOG_FILE" | tail -2
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] pre_len=$P FAILED (exit $STATUS)"
    tail -20 "$LOG_FILE"
  fi
  echo ""
done

echo "================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All DiffSTG F=1 experiments finished."
echo "================================================"
echo ""
echo "Results summary:"
for P in $PRE_LENS; do
  LOG_FILE="$LOG_DIR/diffstg_F1_air_n95_p${P}.log"
  echo "--- pre_len=$P ---"
  grep -E "Final results" "$LOG_FILE" 2>/dev/null || echo "  (no final results)"
done
