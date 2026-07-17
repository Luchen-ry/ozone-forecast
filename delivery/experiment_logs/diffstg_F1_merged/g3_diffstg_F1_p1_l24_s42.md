# 实验记录

## 基本信息

- 组别：g3
- 学生：teammate
- 日期：2026-07-10
- 实验编号：g3_diffstg_F1_p1_l24_s42
- 模型：DiffSTG（外部基线）
- 数据目录：/home/cl/ozone-forecast/external_baselines_F1/DiffSTG (AIR_N95, F=1 单通道 O3)

## 运行命令

```bash
cd /home/cl/ozone-forecast/external_baselines_F1/DiffSTG
python train.py --data AIR_N95 --T_h 24 --T_p 1 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 \
  --num_features 1 --split_mode default --is_test False
```

## 关键配置

- seq_len: 24
- pre_len: 1
- seed: 42
- device: cuda
- epochs: 200 (early_stop=10)
- batch_size: 4
- use_diffusion: 1 (DDPM, 50 steps)
- hidden_size: 32
- sample_strategy: ddim_multi, sample_steps: 40
- num_features: 1 (F=1, 仅 O3)
- split_mode: default (0.6/0.8 切分，非 noleak)

## 输出位置

- 输出目录：external_baselines_F1/DiffSTG
- 日志文件：external_baselines_F1/DiffSTG/output/log/[AIR_N95]mae_9.61.log

## 指标

| RMSE | MAE | MAPE | Peak RMSE | Step6 RMSE |
| --- | --- | --- | --- | --- |
| 13.08 | 9.61 | 122.21 |  |  |

best_epoch: 101

## 现象和结论

两人均跑了 pre_len=1：队友 101 epochs MAE=9.61（日志 `[AIR_N95]mae   9.61+...+24+1+...log`），我 15 epochs 早停 MAE=12.68（日志 `[AIR_N95]mae  12.68+...+24+1+...log`）。队友训练更充分且 MAE 更低，取队友结果。本实验为 F=1 单通道 O3 基线，用于和 PE-DiffWaveNet（F=15）对比，验证气象因子的必要性。
相比 PE-DiffWaveNet pre_len=1（见 pedw_F15 目录），DiffSTG F=1 误差显著更高，
说明仅靠 O3 历史值不足以支撑准确预测，需引入气象因子。

## 问题

- F=1 数据切分使用默认 0.6/0.8（非 noleak），与 PE-DiffWaveNet 的 noleak 切分不同，对比时需注意
- 部分实验因 early_stop=10 触发过早停止（如 pre_len=12/24），训练轮数偏少
- DiffSTG train.py 已做 3 处 bug 修复（str2bool、硬编码 is_test、nni 导入），详见 README 代码修改说明
