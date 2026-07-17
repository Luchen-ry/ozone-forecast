DiffSTG 邻接矩阵对比实验交付包（日志 + 预测）
================================================
配置: AIR_N95, T_h=24, T_p=6, seed=42, N=50, sample_steps=50, hidden=32, batch_size=4, n_samples=1

结果
----
corr_adj (相关图): MAE=35.86  RMSE=42.11  MAPE=567.52  best_epoch=2
pe_adj   (PE图):   MAE=37.97  RMSE=44.62  MAPE=605.83  best_epoch=5

文件说明
--------
corr_adj/train_log_mae35.86.log     相关图训练日志
pe_adj/train_log_mae37.97.log       PE图训练日志
pe_adj/forecast_T_p6_seed42.pkl     PE图预测文件（见下）
corr_adj.npy / pe_adj.npy           两次实验使用的邻接矩阵
results_alt_adj.csv                 汇总指标

重要说明
--------
两次训练超参相同，DiffSTG 默认把 forecast.pkl 写到同一路径，
因此相关图(corr) 的预测文件在随后的 PE 训练中被覆盖，本包仅保留 PE 图预测。
相关图结果请以 train_log_mae35.86.log / results_alt_adj.csv 为准。
本包不含权重文件。
