#!/usr/bin/env bash
# DiffSTG adjacency ablation: corr graph + PE graph (serial), then restore distance adj.
# Designed for: nohup bash run_alt_adj_exps.sh > logs/nohup_alt_adj.out 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFFSTG_DIR="${SCRIPT_DIR}/DiffSTG"
DATA_DIR="${DIFFSTG_DIR}/data/dataset/AIR_N95"
OUT_DIR="${SCRIPT_DIR}/output"
LOG_DIR="${SCRIPT_DIR}/logs"
RESULTS_CSV="${SCRIPT_DIR}/results.csv"
METRICS_CSV="${DIFFSTG_DIR}/output/metrics/DiffSTG.csv"

ADJ="${DATA_DIR}/adj.npy"
ADJ_DIST="${DATA_DIR}/adj_distance.npy"
CORR_ADJ="${OUT_DIR}/corr_adj.npy"
PE_ADJ="${OUT_DIR}/pe_adj.npy"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/alt_adj_exps_${STAMP}.log"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

# Tee all subsequent output into the run log as well.
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "============================================================"
echo "[START] DiffSTG alt-adj experiments  ${STAMP}"
echo "  SCRIPT_DIR=${SCRIPT_DIR}"
echo "  RUN_LOG=${RUN_LOG}"
echo "============================================================"

restore_adj() {
  if [[ -f "${ADJ_DIST}" ]]; then
    cp -f "${ADJ_DIST}" "${ADJ}"
    echo "[RESTORE] adj.npy <- adj_distance.npy"
  else
    echo "[WARN] adj_distance.npy missing; cannot restore adj.npy" >&2
  fi
}
trap restore_adj EXIT

append_results() {
  local method="$1"
  local source="$2"
  python - "$RESULTS_CSV" "$METRICS_CSV" "$method" "$source" <<'PY'
import csv
import sys
from pathlib import Path

results_csv = Path(sys.argv[1])
metrics_csv = Path(sys.argv[2])
method = sys.argv[3]
source = sys.argv[4]

if not metrics_csv.is_file():
    print(f"[WARN] metrics csv missing: {metrics_csv}", flush=True)
    sys.exit(0)

import pandas as pd

df = pd.read_csv(metrics_csv)
# Prefer formal runs (is_test False) with T_p=6
if "model.T_p" in df.columns:
    sub = df[df["model.T_p"] == 6]
else:
    sub = df
if "is_test" in sub.columns:
    formal = sub[sub["is_test"].astype(str).str.lower().isin(["false", "0"])]
    if len(formal):
        sub = formal
if len(sub) == 0:
    print("[WARN] no T_p=6 rows in metrics csv", flush=True)
    sys.exit(0)

row = sub.iloc[-1]
mae = float(row["mae"])
rmse = float(row["rmse"])
mape = float(row["mape"])
best_epoch = int(row["best_epoch"]) if "best_epoch" in row and pd.notna(row["best_epoch"]) else -1

header = ["Method", "RMSE", "MAE", "MAPE", "Peak_RMSE", "Step6_RMSE", "Source"]
write_header = not results_csv.is_file() or results_csv.stat().st_size == 0
with results_csv.open("a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(header)
    w.writerow([
        method,
        f"{rmse:.2f}",
        f"{mae:.2f}",
        f"{mape:.2f}",
        "N/A",
        "N/A",
        f"{source} best_epoch={best_epoch}",
    ])
print(f"[OK] appended results: {method}  MAE={mae:.2f} RMSE={rmse:.2f} MAPE={mape:.2f}", flush=True)
PY
}

run_train() {
  local tag="$1"
  local adj_src="$2"
  echo "------------------------------------------------------------"
  echo "[TRAIN] ${tag}  using $(basename "${adj_src}")"
  echo "------------------------------------------------------------"
  cp -f "${adj_src}" "${ADJ}"
  (
    cd "${DIFFSTG_DIR}"
    python train.py \
      --data AIR_N95 --T_h 24 --T_p 6 --seed 42 \
      --N 50 --sample_steps 50 --hidden_size 32 --batch_size 4 --n_samples 1
  )
  append_results \
    "DiffSTG (pre_len=6, F=1, ${tag})" \
    "alt-adj seed42 (${tag})"
}

# 1) Backup distance adjacency once
if [[ ! -f "${ADJ_DIST}" ]]; then
  if [[ ! -f "${ADJ}" ]]; then
    echo "[ERR] missing ${ADJ}" >&2
    exit 1
  fi
  cp -f "${ADJ}" "${ADJ_DIST}"
  echo "[OK] backed up adj.npy -> adj_distance.npy"
else
  echo "[OK] adj_distance.npy already exists (skip backup)"
fi

# 2) Generate alt matrices if needed
if [[ ! -f "${CORR_ADJ}" || ! -f "${PE_ADJ}" ]]; then
  echo "[INFO] generating corr/pe adjacency matrices..."
  python "${SCRIPT_DIR}/prepare_alt_adj.py"
else
  echo "[OK] corr_adj.npy / pe_adj.npy already exist (skip prepare)"
fi

# 3) Corr graph experiment
run_train "corr_adj" "${CORR_ADJ}"

# 4) PE graph experiment
run_train "pe_adj" "${PE_ADJ}"

# 5) Explicit restore (also done by EXIT trap)
restore_adj

echo "============================================================"
echo "[DONE] alt-adj experiments finished"
echo "  log: ${RUN_LOG}"
echo "  results: ${RESULTS_CSV}"
echo "  adj restored from: ${ADJ_DIST}"
echo "============================================================"
