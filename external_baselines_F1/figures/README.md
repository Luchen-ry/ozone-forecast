# DiffSTG Baseline — 可视化及结果汇总

> 配置：L=24, hidden_size=32, N=50, batch_size=4, seed=42
> 数据：O3 单特征（F=1），不含气象因子

---

## 图表说明

### `rmse_decay_comparison.png`
### `mae_decay_comparison.png`
### `rmse_ratio_bars.png`

**DiffSTG vs PE-DiffWaveNet 多步长衰减对比**（pre_len = 1, 3, 6, 12, 24）

- 蓝色虚线：PE-DiffWaveNet（F=15，O3 + 14 气象因子）
- 红色实线：DiffSTG（F=1，仅 O3）
- 柱状图：RMSE 差距倍数（DiffSTG / PE-DiffWaveNet）

核心结论：气象因子对中长期预测（3h+）尤为重要，DiffSTG 在 6h 后 RMSE 趋于饱和（~41→49），而 PE-DiffWaveNet 保持线性退化，差距从 1h 的 1.9× 扩大到 6h 的 3.7×。

---

### `diffstg_scatter_error.png`

**左侧**：预测 vs 真实散点图（95 站点 × 50 时间窗口）。每个点代表某一小时某站点的预测-真实对。对角线为完美预测。点越靠近对角线说明预测越准。

**右侧**：预测误差分布直方图。偏右说明模型有正偏（高估），偏左说明低估。当前以 0 为中心但分布较宽。

### `diffstg_ts_comparison.png`

3 个代表站点的预测值（红色）vs 真实值（蓝色）时序对比。X 轴为连续时间步，每个步长为 1 小时。可直观看到模型是否学到了日周期。

### `diffstg_per_step_rmse.png`

1 小时到 6 小时的逐步 RMSE。预测步长越长，RMSE 越高。

| 预测步长 | RMSE |
|---|---|
| 1h | 25.0 |
| 2h | 34.1 |
| 3h | 39.9 |
| 4h | 43.7 |
| 5h | 42.6 |
| 6h | 43.1 |

---

## 汇总指标（参见 results.csv）

| pre_len | RMSE | MAE | MAPE |
|:---:|:---:|:---:|:---:|
| 1 | 13.08 | 9.61 | 122.21 |
| 3 | 32.42 | 26.60 | 426.64 |
| 6 | 41.10 | 34.50 | 557.66 |
| 12 | 45.70 | 39.10 | 636.65 |
| 24 | 48.65 | 41.89 | 676.79 |

> **结论**：仅凭 O3 历史值（无气象因子）的扩散模型在 1-2 小时内有一定预测能力，但 4 小时后误差趋于饱和。与 PE-DiffWaveNet（MAE 7.56 / RMSE 10.94）的差距主要来自缺少气象因子（温度、辐射、气压等 14 个变量）。
