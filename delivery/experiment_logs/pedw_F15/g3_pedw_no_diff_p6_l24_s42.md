# 实验记录

## 基本信息

- 组别：g3
- 学生：cl
- 日期：2026-07-15
- 实验编号：g3_pedw_no_diff_p6_l24_s42
- 模型：PE-DiffWaveNet
- 数据目录：/home/cl/ozone-forecast/matrix_N95 (AIR_N95, 95 站点 2022 年 O3 + 14 气象变量)

## 运行命令

```bash
cd '/home/cl/ozone-forecast' && DEVICE=cuda EPOCHS=120 EXP_NAME=g3_pedw_no_diff_p6_l24_s42 USE_DIFFUSION=0 bash scripts/run_train_pediffwavenet.sh 6 24 42
```

## 关键配置

- seq_len: 24
- pre_len: 6
- seed: 42
- device: cuda (RTX 5080, PyTorch 2.7+cu128)
- epochs: 120 (early stop patience=15)
- batch_size: 4
- use_diffusion: 0
- use_pe_graph: 1
- use_pe_film: 1
- pe_adaptive_loss: 1
- pe_shuffle_seed: -1
- split_mode: noleak (防数据泄漏)
- num_features: 15 (F=15, O3 + 14 气象)

## 输出位置

- 输出目录：/home/cl/ozone-forecast/matrix_N95_PEDiffWaveNet_noleak_g3_pedw_no_diff_p6_l24_s42
- 权重目录：weights_N95/
- 日志文件：/home/cl/ozone-forecast/logs/g3_pedw_no_diff_p6_l24_s42.log

## 指标

| RMSE | MAE | MAPE | Peak RMSE | Step6 RMSE |
| --- | --- | --- | --- | --- |
| 10.93 | 7.58 | 29.90 | 14.97 | 13.612256792334898 |

best_epoch: 19

## 现象和结论

去掉扩散模块（USE_DIFFUSION=0），仅保留 PE 图与 PE-FiLM。RMSE=10.93 相对主实验 11.24 略降 2.8%，MAE=7.58。扩散模块在本任务上去除反而略有提升，可能因 O3 时序较平滑，扩散噪声对短期预测（pre_len=6）非必需；但 Peak_RMSE 上升（14.15→14.97），说明扩散对极端值建模有正向作用。

## 问题

- pre_len=3 在默认配置下训练不稳定（验证 MAE 震荡），通过 LR=3e-4 + AMP=0 + PATIENCE=25 解决
- 所有实验在 RTX 5080 上运行，需 PyTorch 2.7+cu128 兼容 sm_120 架构
