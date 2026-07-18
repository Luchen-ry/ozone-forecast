# DiffSTG Baseline 结果汇总

> 配置：L=24, hidden_size=32, N=50, batch_size=4, seed=42, lr=0.002
> 数据：单特征（F=1），不含气象因子
> 指标详见 delivery/results.csv

## 文件说明

### baseline_results.csv — 主对比（pre_len=6）

与 `paper_assets_pediffwavenet/table1_main_raw_comparison.csv` 格式对齐，可直接合并。

| Method | RMSE | MAE | MAPE |
|------|:---:|:---:|:---:|
| DiffSTG (F=1, O3 only) | 35.74 | 29.20 | 469.99 |

### results.csv — 多步长衰减

| pre_len | RMSE | MAE |
|:---:|:---:|:---:|
| 1 | 13.08 | 9.61 |
| 3 | 32.42 | 26.60 |
| 6 | 35.74 | 29.20 |
| 12 | 43.97 | 37.59 |
| 24 | 47.22 | 40.61 |

### single_target_results.csv — 三污染物对比

| Pollutant | MAE | RMSE | MAPE |
|------|:---:|:---:|:---:|
| O3 | 29.20 | 35.74 | 469.99% |
| PM2.5 | 25.05 | 38.96 | 124.28% |
| PM10 | 38.81 | 83.59 | 83.89% |

### adjacency_results.csv — 邻接矩阵对比

| Adjacency | MAE | RMSE |
|------|:---:|:---:|
| Distance | 29.20 | 35.74 |
| Correlation | 35.86 | 42.11 |
| PE | 37.97 | 44.62 |

### probability_metrics.csv — 概率预测指标

| Experiment | CRPS | MIS 80% | MIS 90% | MIS 95% |
|------|:---:|:---:|:---:|:---:|
| O3 (p6) | 26.71 | 219.82 | 352.78 | 579.32 |

## 对应图表

| 图表 | 脚本 |
|------|------|
| 多步长衰减曲线 | `plot_scripts/plot_prelen_decay.py` |
| 三污染物对比 | `plot_scripts/plot_pollutant_comparison.py` |
| 邻接矩阵对比 | `plot_scripts/plot_adjacency_comparison.py` |
| 概率指标+置信区间 | `compute_probability_metrics.py` |
| O3 单步可视化 | `plot_scripts/plot_diffstg_forecast.py` |
