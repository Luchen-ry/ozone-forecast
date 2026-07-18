# 实验记录

## 基本信息

- 组别：g3
- 学生：熊佳惠
- 日期：2026-07-12
- 实验编号：g3_diffstg_F1_pm25_p6_l24_s42
- 模型：DiffSTG（外部基线）
- 数据目录：/home/cl/ozone-forecast/external_baselines_F1/DiffSTG (AIR_N95_PM25, F=1 单通道 PM2.5)

## 运行命令

```bash
cd /home/cl/ozone-forecast
# 生成 PM2.5 单通道数据
python external_baselines_F1/prepare_pm_data.py
# 训练
cd external_baselines_F1/DiffSTG
python train.py --data AIR_N95_PM25 --T_h 24 --T_p 6 --seed 42 \
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
- num_features: 1 (F=1, 仅 PM2.5)
- split_mode: default (0.6/0.8)
- target_pollutant: PM2.5 (数据范围 [1, 543])

## 输出位置

- 结果 CSV：delivery/logs/diffstg_advanced/single_target_results.csv
- 运行记录：external_baselines_F1/run_command.sh（第九节）
- 训练日志：本实验在队友本地 Windows 11 + RTX 3050 环境运行，原始 stdout 日志未同步至本机；结果以 single_target_results.csv 为准（标注 best_epoch=22）

## 指标

| RMSE | MAE | MAPE | Peak RMSE | Step6 RMSE |
| --- | --- | --- | --- | --- |
| 38.96 | 25.05 | 124.28 |  |  |

best_epoch: 22

## 现象和结论

进阶实验：三污染物对比之一——PM2.5。
PM2.5 绝对误差最低（MAE=25.05），归一化 RMSE/Range=7.18%，介于 O3（10.02%）和 PM10（3.03%）之间。
相比 O3（MAE=29.20），PM2.5 更易预测，因为 PM2.5 主要受排放和传输影响，历史值信息量较大；
而 O3 是光化学反应产物，强依赖辐射和温度等气象条件，仅靠历史值不足以预测光化学拐点。
这证实了为 O3 引入气象因子（F=15）的必要性。

## 问题

- 原始训练日志在队友本地环境，未同步至交付机；结果以 single_target_results.csv 为准
- 数据范围 [1, 543]，极端高值（沙尘暴等）可能影响 RMSE
