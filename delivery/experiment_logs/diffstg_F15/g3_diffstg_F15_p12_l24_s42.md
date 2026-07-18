# 实验记录

## 基本信息

- 组别：g3
- 学生：陈璐
- 日期：2026-07-17
- 实验编号：g3_diffstg_F15_p12_l24_s42
- 模型：DiffSTG（外部基线）
- 数据目录：/home/cl/ozone-forecast/matrix_N95 (AIR_N95, F=15: O3 + 14 气象变量)

## 运行命令

```bash
cd /home/cl/ozone-forecast/external_baselines_F1/DiffSTG
python train.py --data AIR_N95 --T_h 24 --T_p 12 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 \
  --num_features 15 --split_mode noleak --is_test False \
  --lr 0.001 --early_stop 30
```

## 关键配置

- seq_len: 24
- pre_len: 12
- seed: 42
- device: cuda
- epochs: 300 (early_stop=30)
- batch_size: 4
- use_diffusion: 1 (DDPM, 50 steps)
- hidden_size: 32
- sample_strategy: ddim_multi, sample_steps: 40
- num_features: 15 (F=15, O3 + 14 气象)
- split_mode: noleak (防数据泄漏，对齐 PE-DiffWaveNet)
- lr: 0.001 (从 0.002 降低)
- early_stop: 30 (从 10 增加)

## 输出位置

- 输出目录：external_baselines_F1/DiffSTG
- 日志文件：logs/diffstg_F15/diffstg_air_n95_F15_noleak_p12_rerun.log

## 指标

| RMSE | MAE | MAPE | Peak RMSE | Step6 RMSE |
| --- | --- | --- | --- | --- |
| 39.82 | 33.3 | 473.76 |  | 39.82 |

best_epoch: 33

## 现象和结论

重跑版本：33 epochs（仍早停，但比旧版 12 epochs 改善）。相比旧版（MAE=36.15）有改善，验证 MAE 仍有震荡但已收敛。

本实验为 F=15 baseline，使用 noleak 切分对齐 PE-DiffWaveNet，用于验证：
1. 气象因子（F=15 vs F=1）对 DiffSTG baseline 的提升
2. PE-DiffWaveNet 相对 DiffSTG 在相同输入特征下的优势（PE 图 + PE-FiLM 的贡献）

相比 PE-DiffWaveNet pre_len=12（见 pedw_F15 目录），DiffSTG F=15 误差仍显著更高，
说明 PE-DiffWaveNet 的架构改进（PE 图 + PE-FiLM）对性能提升贡献显著。

## 问题

- 旧版（lr=0.002, es=10）因验证 MAE 震荡严重，多数 pre_len 仅训练 12-15 epochs 就早停，训练不充分
- 重跑调整 lr=0.001 + es=30 后，训练更充分，p1/p3/p6 收敛 epoch 数大幅增加
- p12 仍仅 33 epochs 早停，但比旧版 12 epochs 改善明显
- p24 未重跑（原版已充分收敛）
