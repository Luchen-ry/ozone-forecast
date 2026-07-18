# 实验记录

## 基本信息

- 组别：g3
- 学生：吴蕊
- 日期：2026-07-16
- 实验编号：g3_diffstg_F1_o3_p6_l24_s42_corr
- 模型：DiffSTG（外部基线）
- 数据目录：/home/cl/ozone-forecast/external_baselines_F1/DiffSTG (AIR_N95, F=1 单通道 O3)

## 运行命令

```bash
cd /home/cl/ozone-forecast/external_baselines_F1/DiffSTG
# 步骤 1：生成相关图邻接矩阵
python ../prepare_alt_adj.py   # 输出 output/corr_adj.npy
# 步骤 2：备份距离矩阵并替换为相关图
cp data/dataset/AIR_N95/adj.npy data/dataset/AIR_N95/adj_distance.npy
cp /home/cl/ozone-forecast/output/corr_adj.npy data/dataset/AIR_N95/adj.npy
# 步骤 3：训练
python train.py --data AIR_N95 --T_h 24 --T_p 6 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1
# 步骤 4：恢复距离矩阵
cp data/dataset/AIR_N95/adj_distance.npy data/dataset/AIR_N95/adj.npy
```

## 关键配置

- seq_len: 24
- pre_len: 6
- seed: 42
- device: cuda
- epochs: 300 (early_stop=10)
- batch_size: 4
- use_diffusion: 1 (DDPM, 50 steps)
- hidden_size: 32
- num_features: 1 (F=1, 仅 O3)
- split_mode: default (0.6/0.8)
- adjacency: Correlation (Pearson 相关图，替代默认距离图)

## 输出位置

- 输出目录：external_baselines_F1/DiffSTG/output
- 日志文件：delivery/logs/diffstg_advanced/diffstg_F1_corr_adj_p6_l24_s42.log
- 邻接矩阵：share_alt_adj_exps_20260717/corr_adj.npy

## 指标

| RMSE | MAE | MAPE | Peak RMSE | Step6 RMSE |
| --- | --- | --- | --- | --- |
| 42.11 | 35.86 | 567.52 |  |  |

best_epoch: 2

## 现象和结论

进阶实验：邻接矩阵对比之一——Pearson 相关图（站点 O3 序列同步性越强连接越强）。
相比距离图基准（MAE=29.20），相关图 MAE 升至 35.86（+6.66），RMSE 升至 42.11（+6.37）。
三种邻接矩阵（距离/相关/PE）差距（RMSE 最大差 8.9）远小于 DiffSTG 与 PE-DiffWaveNet 差距（24），
说明 F=1 下瓶颈不在空间建模策略，而在输入特征丰富度。换更聪明的邻接矩阵无法弥补缺少气象因子的信息损失。

## 问题

- best_epoch=2，early_stop 在第 12 epoch 触发，验证 MAE 震荡（1.81-19.28），训练不稳定
- 相关图预测文件被随后的 PE 图实验覆盖，本实验结果以 train_log 和 results_alt_adj.csv 为准
