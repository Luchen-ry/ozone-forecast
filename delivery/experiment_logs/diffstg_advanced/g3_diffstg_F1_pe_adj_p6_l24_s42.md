# 实验记录

## 基本信息

- 组别：g3
- 学生：teammate
- 日期：2026-07-16
- 实验编号：g3_diffstg_F1_o3_p6_l24_s42_pe
- 模型：DiffSTG（外部基线）
- 数据目录：/home/cl/ozone-forecast/external_baselines_F1/DiffSTG (AIR_N95, F=1 单通道 O3)

## 运行命令

```bash
cd /home/cl/ozone-forecast/external_baselines_F1/DiffSTG
# 步骤 1：生成 PE 图邻接矩阵
python ../prepare_alt_adj.py   # 输出 output/pe_adj.npy
# 步骤 2：替换为 PE 图
cp /home/cl/ozone-forecast/output/pe_adj.npy data/dataset/AIR_N95/adj.npy
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
- adjacency: PE (排列熵相似度图，序列复杂度越接近连接越强)

## 输出位置

- 输出目录：external_baselines_F1/DiffSTG/output
- 日志文件：delivery/logs/diffstg_advanced/diffstg_F1_pe_adj_p6_l24_s42.log
- 邻接矩阵：share_alt_adj_exps_20260717/pe_adj.npy
- 预测文件：share_alt_adj_exps_20260717/pe_adj/forecast_T_p6_seed42.pkl

## 指标

| RMSE | MAE | MAPE | Peak RMSE | Step6 RMSE |
| --- | --- | --- | --- | --- |
| 44.62 | 37.97 | 605.83 |  |  |

best_epoch: 5

## 现象和结论

进阶实验：邻接矩阵对比之二——PE 图（排列熵相似度图）。
相比距离图基准（MAE=34.50），PE 图 MAE 升至 37.97（+3.47），RMSE 升至 44.62（+3.52），是三种邻接矩阵中误差最高的。
三种邻接矩阵差距极小（RMSE 最大差 3.5），而 DiffSTG 与 PE-DiffWaveNet 差距达 30，
说明 F=1 下瓶颈不在空间建模策略，而在输入特征丰富度。这从空间维度反向验证了 PE-DiffWaveNet 引入气象因子的设计合理性。

## 问题

- best_epoch=5，early_stop 在第 17 epoch 触发，验证 MAE 震荡（4.06-33.75），训练不稳定
- DiffSTG 默认把 forecast.pkl 写到同一路径，相关图预测文件被本实验覆盖，仅保留 PE 图预测
