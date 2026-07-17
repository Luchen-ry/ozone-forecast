# 实验记录

## 基本信息

- 组别：g3
- 学生：teammate
- 日期：2026-07-12
- 实验编号：g3_diffstg_F1_pm10_p6_l24_s42
- 模型：DiffSTG（外部基线）
- 数据目录：/home/cl/ozone-forecast/external_baselines_F1/DiffSTG (AIR_N95_PM10, F=1 单通道 PM10)

## 运行命令

```bash
cd /home/cl/ozone-forecast
# 生成 PM10 单通道数据
python external_baselines_F1/prepare_pm_data.py
# 训练
cd external_baselines_F1/DiffSTG
python train.py --data AIR_N95_PM10 --T_h 24 --T_p 6 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1
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
- num_features: 1 (F=1, 仅 PM10)
- split_mode: default (0.6/0.8)
- target_pollutant: PM10 (数据范围 [1, 2759])

## 输出位置

- 结果 CSV：delivery/logs/diffstg_advanced/single_target_results.csv
- 运行记录：external_baselines_F1/run_command.sh（第九节）
- 训练日志：本实验在队友本地 Windows 11 + RTX 3050 环境运行，原始 stdout 日志未同步至本机；结果以 single_target_results.csv 为准（标注 best_epoch=17）

## 指标

| RMSE | MAE | MAPE | Peak RMSE | Step6 RMSE |
| --- | --- | --- | --- | --- |
| 83.59 | 38.81 | 83.89 |  |  |

best_epoch: 17

## 现象和结论

进阶实验：三污染物对比之二——PM10。
PM10 绝对误差最高（RMSE=83.59，MAE=38.81），但因数据范围极大 [1, 2759]，归一化 RMSE/Range 仅 3.03%，是三种污染物中最易预测的。
MAPE 仅 83.89%，远低于 O3（557%）和 PM2.5（124%），因 PM10 数值普遍较大，分母效应使 MAPE 更稳定。
三污染物对比结论：O3 预测难度最大（归一化 RMSE 10.02%），因光化学反应对气象条件敏感；
PM2.5/PM10 主要受排放和传输影响，历史值信息量更大。这证实了为 O3 引入气象因子（F=15）的必要性。

## 问题

- 原始训练日志在队友本地环境，未同步至交付机；结果以 single_target_results.csv 为准
- PM10 数据范围 [1, 2759] 跨度大，沙尘暴等极端高值难捕捉
