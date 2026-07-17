# 实验记录

## 基本信息

- 组别：g3
- 学生：cl
- 日期：2026-07-15
- 实验编号：g3_pedw_p1_l24_s42
- 模型：PE-DiffWaveNet
- 数据目录：/home/cl/ozone-forecast/matrix_N95 (AIR_N95, 95 站点 2022 年 O3 + 14 气象变量)

## 运行命令

```bash
cd '/home/cl/ozone-forecast' && DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_p1_l24_s42 bash scripts/run_train_pediffwavenet.sh 1 24 42
```

## 关键配置

- seq_len: 24
- pre_len: 1
- seed: 42
- device: cuda (RTX 5080, PyTorch 2.7+cu128)
- epochs: 120 (early stop patience=15)
- batch_size: 4
- use_diffusion: 1
- use_pe_graph: 1
- use_pe_film: 1
- pe_adaptive_loss: 1
- pe_shuffle_seed: -1
- split_mode: noleak (防数据泄漏)
- num_features: 15 (F=15, O3 + 14 气象)

## 输出位置

- 输出目录：/home/cl/ozone-forecast/matrix_N95_PEDiffWaveNet_noleak_g3_pedw_p1_l24_s42
- 权重目录：weights_N95/
- 日志文件：/home/cl/ozone-forecast/logs/g3_pedw_p1_l24_s42.log

## 指标

| RMSE | MAE | MAPE | Peak RMSE | Step6 RMSE |
| --- | --- | --- | --- | --- |
| 6.88 | 5.00 | 21.02 | 5.68 |  |

best_epoch: 46

## 现象和结论

预测步长扫描：pre_len=1（单步预测）。RMSE=6.88，MAE=5.00，误差最低。短期预测相对简单，模型表现最佳。

## 问题

- pre_len=3 在默认配置下训练不稳定（验证 MAE 震荡），通过 LR=3e-4 + AMP=0 + PATIENCE=25 解决
- 所有实验在 RTX 5080 上运行，需 PyTorch 2.7+cu128 兼容 sm_120 架构
