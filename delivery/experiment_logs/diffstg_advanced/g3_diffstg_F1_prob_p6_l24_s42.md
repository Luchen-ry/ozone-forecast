# 实验记录

## 基本信息

- 组别：g3
- 学生：teammate
- 日期：2026-07-17
- 实验编号：g3_diffstg_F1_o3_p6_l24_s42_prob
- 模型：DiffSTG（外部基线，概率预测评估）
- 数据目录：/home/cl/ozone-forecast/external_baselines_F1/DiffSTG (AIR_N95, F=1 单通道 O3)

## 运行命令

```bash
cd /home/cl/ozone-forecast/external_baselines_F1
# 加载已训练的 O3 pre_len=6 模型，做 50 次采样推理
python compute_probability_metrics.py
# 脚本内 DATA_NAME='AIR_N95', pre_len=6, n_samples=50
# 约 300 batch × 50 samples ≈ 5 分钟
```

## 关键配置

- seq_len: 24
- pre_len: 6
- seed: 42
- device: cuda
- n_samples: 50 (DDPM 50 次采样)
- sample_strategy: ddim_multi, sample_steps: 40
- num_features: 1 (F=1, 仅 O3)
- target_pollutant: O3
- 加载模型：DiffSTG F=1 O3 pre_len=6 best 模型（MAE=34.50 那次）

## 输出位置

- 结果 CSV：delivery/logs/diffstg_advanced/probability_metrics.csv
- 置信区间图：delivery/figures/diffstg_advanced/confidence_AIR_N95_p6.png
- 脚本：external_baselines_F1/compute_probability_metrics.py
- 运行记录：external_baselines_F1/run_command.sh（第十节）

## 指标

| CRPS | MIS 80% | MIS 90% | MIS 95% |
| --- | --- | --- | --- |
| 26.71 | 219.82 | 352.78 | 579.32 |

## 现象和结论

进阶实验：概率预测指标与置信区间——展示扩散模型的不确定性量化能力。
- CRPS=26.71：与 MAE=34.50 同量级，概率预测未系统偏离真实分布，估计合理但不够精准
- MIS 较高（MIS 95%=579.32）：F=1 下扩散模型不确定性极大，预测区间宽到几乎无参考价值
- 叙事：扩散模型的天然优势是不确定性量化。F=1 时 MIS 极高（模型"知道自己不准"，诚实表达信息不足），
  F=15 时预计降至合理范围——从概率维度再次证实气象因子的必要性
- 可视化：confidence_AIR_N95_p6.png 展示 50 次采样的预测均值 ± σ 阴影带

## 问题

- 本实验为推理评估（非训练），无独立训练日志；运行输出记录在 run_command.sh 第十节注释中
- MIS 95% 高达 579.32，预测区间过宽，实用性低；需引入气象因子（F=15）缩小不确定性
