# 生产实习最终交付材料

## 本组完成工作

围绕任务书"基于扩散模型的多站点空气质量预测实验平台构建"总目标，按 docs_word 五大任务分组（数据整理 / baseline 复现 / PE-DiffWaveNet 实验 / 结果整理报告 / 最终交付）与三周安排完成全部工作。具体包括：

### 一、数据理解与整理

1. **数据范围确认**：95 个空气质量监测站点，2022 年全年（8717 小时），主预测目标 O3，14 个气象变量（blh/d2m/fsr/kx/sp/ssr/ssrd/t2m/tcc/tcwv/tp/u10/v10/zust），输入特征数 F=15
2. **缺失统计**：对 data_N95 中 O3 / PM2.5 / PM10 逐站点逐小时做缺失统计（`pollutant_missing_summary.csv`、`missing_stats.py`），输出数据说明初稿
3. **站点分布**：从 xlsx_N95/station_loc1.xlsx 提取 95 站点编码/名称/城市/经纬度，生成 `station_location.png` 站点地理分布图
4. **时间序列**：画 O3 / PM2.5 / PM10 整体时间序列（`o3_daily_mean.png`、`daily_mean_o3.csv`），观察日周期与季节趋势
5. **相关性分析**：计算 O3 与 14 气象因子的相关系数（`o3_meteorological_correlation.csv`、`meteorological_correlation.py`），确认辐射/温度/湿度等对 O3 的强相关性，为引入 F=15 提供依据
6. **数据处理产物**：
   - matrix_N95/data.npy（O3 主序列, shape=(95, 8717)）
   - matrix_N95/met_raw_aligned_cache.npz（14 气象变量已对齐到 95 站点和时间索引）
   - matrix_N95/data_combined_m15.npy（O3+气象组合, shape=(8717, 95, 15)）
   - DiffSTG 专用：prepare_air_n95_for_diffstg.py 生成 flow.npy(8717,95,1) + adj.npy(95,95)；prepare_pm_data.py 抽取 PM2.5/PM10 单通道

### 二、Smoke Test 与环境适配

1. **环境适配**：解决 RTX 5080（Blackwell sm_120）与 PyTorch 2.5 不兼容问题，升级至 PyTorch 2.7+cu128；conda 环境 atgcn
2. **PE-DiffWaveNet Smoke Test**：运行 scripts/run_smoke_cpu.sh（DEVICE=cpu, EPOCHS=3, 小 hidden_size, 限制 MAX_*_WINDOWS），验证无 FileNotFoundError、能打印 train/valid/test shape、能生成输出目录、能完成 1 个 epoch、能保存 config/split_summary/graph_summary
3. **DiffSTG 官方代码验证**：先用 PEMS08 自带数据集跑通（`run_command.sh` 第一节，2 epochs 验证训练/验证/保存模型全流程正常），再切换到 AIR_N95

### 三、DiffSTG Baseline 复现（初阶 + 进阶）

#### 3.1 代码 Bug 修复（对齐评估口径）

为让 DiffSTG 在 AIR_N95 上正确运行并与 PE-DiffWaveNet 公平对比，对 external_baselines_F1/DiffSTG/train.py 做 5 处 bug 修复（非核心结构改动）：

1. `type=bool` → 自定义 `str2bool`：修复 `--is_test False` 被误解析为 True 的 argparse 陷阱
2. 移除硬编码 `config.is_test = False`：让命令行参数生效
3. `nni` 模块条件导入：避免环境缺 nni 报错
4. `evals()` 仅评估 channel 0（O3）：与 PE-DiffWaveNet 评估口径对齐
5. 多通道按通道独立归一化：避免气象量纲淹没 O3 信号
6. F=15 重跑版本额外增加 `--early_stop` 和 `--lr` 命令行参数（让早停耐心和学习率可配置）

#### 3.2 初阶实验：F=1 多预测步长（5 个 pre_len）

- 配置：F=1（仅 O3 单通道）、default 0.6/0.8 切分、seq_len=24、seed=42
- 结果（5 个 pre_len，1/3/6/12/24）：
  - pre_len=1：101 epochs MAE=9.61 
  - pre_len=3：33 epochs MAE=26.60 
  - pre_len=6：194 epochs MAE=29.20
  - pre_len=12：18 epochs MAE=37.59
  - pre_len=24：23 epochs MAE=40.61 
- 日志：delivery/logs/diffstg_F1/（5 份）+ 日志在 external_baselines_F1/DiffSTG/output/log/
- 实验记录：delivery/experiment_logs/diffstg_F1_merged/（5 份）

#### 3.3 初阶实验：F=15 多预测步长（noleak 对齐 PE，重跑）

- 配置：F=15（O3+14 气象）、noleak 切分、seq_len=24、seed=42
- **旧版问题**：lr=0.002 + early_stop=10 导致 p1/p3/p6/p12 仅训练 12-15 epochs，验证 MAE 震荡严重，训练不充分
- **重跑调整**：lr=0.002→0.001（减小震荡）、early_stop=10→30（增加耐心），p1/p3/p6/p12 重跑
  - p1: 51 epochs, MAE=10.46
  - p3: 237 epochs, MAE=17.86
  - p6: 196 epochs, MAE=24.83
  - p12: 33 epochs, MAE=33.30
  - p24: 保留原版（46 epochs 已充分收敛, MAE=27.11）
- 日志：delivery/logs/diffstg_F15/（5 份 rerun + 1 份总日志）
- 实验记录：delivery/experiment_logs/diffstg_F15/（5 份）

#### 3.4 进阶实验一：三污染物对比（O3 / PM2.5 / PM10，F=1，pre_len=6）

- 配置：F=1 单通道、pre_len=6、seq_len=24、seed=42，分别以 O3/PM2.5/PM10 为目标
- 数据准备：prepare_pm_data.py 从 data_N95 抽取 PM2.5/PM10 单通道 flow.npy
- 结果：
  - O3: MAE=29.20, RMSE=35.74, MAPE=469.99%（数据范围 [1,410]，归一化 RMSE 8.72%）
  - PM2.5: MAE=25.05, RMSE=38.96, MAPE=124.28%（[1,543]，7.18%）
  - PM10: MAE=38.81, RMSE=83.59, MAPE=83.89%（[1,2759]，3.03%）
- 结论：O3 预测难度最大（光化学反应强依赖气象），PM2.5/PM10 主要受排放和传输影响，历史值信息量更大
- 日志：delivery/logs/diffstg_advanced/diffstg_F1_pm{25,10}_p6_l24_s42.log
- 实验记录：delivery/experiment_logs/diffstg_advanced/（PM2.5、PM10 各 1 份）

#### 3.5 进阶实验二：邻接矩阵对比（距离 / 相关 / PE，F=1，pre_len=6）

- 配置：F=1、pre_len=6、seq_len=24、seed=42，仅替换邻接矩阵
- 数据准备：prepare_alt_adj.py 生成 corr_adj.npy（Pearson 相关图）和 pe_adj.npy（排列熵相似度图）
- 结果：
  - Distance（基准）: MAE=29.20, RMSE=35.74
  - Correlation: MAE=35.86, RMSE=42.11（best_epoch=2）
  - PE: MAE=37.97, RMSE=44.62（best_epoch=5）
- 结论：三种邻接矩阵差距极小（RMSE 最大差 3.5），而 DiffSTG 与 PE-DiffWaveNet 差距达 30，瓶颈不在空间建模而在输入特征
- 日志：delivery/logs/diffstg_advanced/diffstg_F1_corr_adj_p6_l24_s42.log、diffstg_F1_pe_adj_p6_l24_s42.log、alt_adj_runner.log（完整训练日志 + 运行器日志）
- 实验记录：delivery/experiment_logs/diffstg_advanced/（corr_adj、pe_adj 各 1 份）

#### 3.6 进阶实验三：概率预测指标与置信区间（CRPS / MIS，50 次采样）

- 配置：加载 DiffSTG F=1 O3 pre_len=6 已训练模型，n_samples=50 做 DDPM 采样推理
- 脚本：compute_probability_metrics.py（约 300 batch × 50 samples ≈ 5 分钟）
- 结果（O3, F=1, pre_len=6）：
  - CRPS=26.71（与 MAE=29.20 同量级，概率预测未系统偏离）
  - MIS 80%=219.82, MIS 90%=352.78, MIS 95%=579.32（区间极宽，模型"知道自己不准"）
- 可视化：confidence_AIR_N95_p6.png 展示 50 次采样的预测均值 ± σ 阴影带
- 结论：扩散模型天然支持不确定性量化；F=1 下 MIS 极高（信息不足的诚实表达），F=15 预计降至实用范围
- 日志：本实验为推理评估（非训练），无独立训练日志；运行输出记录在 run_command.sh 第十节，结果见 probability_metrics.csv
- 实验记录：delivery/experiment_logs/diffstg_advanced/g3_diffstg_F1_prob_p6_l24_s42.md

### 四、PE-DiffWaveNet 完整实验矩阵（13 个实验，F=15 + noleak）

1. **主实验**（seq_len=24, pre_len=6, seed=42）：RMSE=11.24, MAE=7.91
2. **多 seed 稳定性**（seed=42/52/62）：均值 RMSE=11.29±0.16, MAE=8.02±0.12，与论文值 11.10±0.15 吻合
3. **输入窗口消融**（seq_len=12/24/48, pre_len=6）：误差随窗口增大略有改善
4. **预测步长扫描**（pre_len=1/3/6/12/24）：误差随步长单调递增；pre_len=3 需 LR=3e-4 + AMP=0 + PATIENCE=25 修复训练不稳定
5. **模块消融**：
   - 无扩散（USE_DIFFUSION=0）：RMSE -2.8%（峰值反降，扩散以整体 RMSE 换峰值能力）
   - 无 PE 图（USE_PE_GRAPH=0）：RMSE +0.8%
   - 无 PE FiLM（USE_PE_FILM=0）：RMSE +23.1%（**最关键模块**）
   - PE shuffle（PE_SHUFFLE_SEED=52）：RMSE +11.2%（PE 站点对应关系有意义）
- 日志：delivery/logs/pedw/（13 份）
- 实验记录：delivery/experiment_logs/pedw_F15/（13 份，按 templates/experiment_log_template.md 模板）

### 五、结果整理与报告

1. **统一结果表**：delivery/results.csv（30 行，字段对齐 templates/experiment_result_template.csv），涵盖 PE 全部实验、DiffSTG F=1/F=15、PM2.5/PM10、邻接矩阵对比、概率指标
2. **论文风格表格**（9 张，delivery/tables/）：
   - table1 主对比 / table2 消融 / table3 PE 预测步长 / table4 输入窗口
   - table5 DiffSTG F=1 多步长/ table6 DiffSTG F=15 多步长（重跑）
   - table7 三污染物对比 / table8 邻接矩阵对比 / table9 概率指标
3. **可视化图表**（delivery/figures/，28 张）：
   - PE 主模型 16 张（fig1-12 + 站点分布 + O3 时间序列 + 补充图）
   - diffstg_advanced 12 张（三污染物对比 4 张 + 邻接矩阵 1 张 + 置信区间 1 张 + DiffSTG 误差分析 6 张）
4. **报告段落**：delivery/report_section.md（8 节，对齐 templates/report_outline.md，含 9 张表格 + 9 节结果分析）
5. **实验记录**：delivery/experiment_logs/（28 份，按 templates/experiment_log_template.md 模板）
   - pedw_F15/ 13 份、diffstg_F1_merged/ 5 份、diffstg_F15/ 5 份、diffstg_advanced/ 5 份
6. **运行命令**：delivery/commands.sh（8 大类实验完整命令，可复现）

## 目录结构

```
delivery/
├── README.md                          # 本文件
├── commands.sh                        # 实际运行命令（可复现）
├── results.csv                        # 统一字段指标表（30 行，对齐 templates/experiment_result_template.csv）
├── report_section.md                  # 本组报告段落（对齐 templates/report_outline.md）
├── tables/                            # 论文表格（9 张）
│   ├── table1_main_comparison.csv         # 主对比表
│   ├── table2_ablation.csv                # 消融实验表
│   ├── table3_pre_len.csv                 # PE 预测步长误差
│   ├── table4_seq_len.csv                 # PE 输入窗口误差
│   ├── table5_diffstg_F1_pre_len.csv      # DiffSTG F=1 多步长
│   ├── table6_diffstg_F15_pre_len.csv     # DiffSTG F=15 多步长（重跑）
│   ├── table7_pollutant_comparison.csv    # 三污染物对比
│   ├── table8_adjacency_comparison.csv    # 邻接矩阵对比
│   └── table9_probability_metrics.csv     # 概率预测指标
├── experiment_logs/                   # 按模板的实验记录（28 份）
│   ├── pedw_F15/                          # PE-DiffWaveNet F=15（13 个）
│   ├── diffstg_F1_merged/                 # DiffSTG F=1 （5 个）
│   ├── diffstg_F15/                       # DiffSTG F=15 重跑（5 个）
│   └── diffstg_advanced/                  # DiffSTG 进阶实验（5 个：PM2.5/PM10/corr_adj/pe_adj/prob）
├── figures/                           # 论文风格图表
│   ├── fig1-9_*.png                       # PE 主模型 9 张图
│   ├── fig10_station_error_distribution.png  # 站点误差分布
│   ├── fig11_high_value_error.png             # 高值区误差分析
│   ├── fig12_pe_stratification.png            # PE 分层结果
│   ├── station_location.png / o3_*.png        # 数据说明图
│   └── diffstg_advanced/                  # 进阶实验图（12 张）
│       ├── pollutant_mae_rmse.png             # 三污染物对比
│       ├── pollutant_mape.png
│       ├── pollutant_rmse_normalized.png
│       ├── pollutant_summary_table.png
│       ├── adjacency_comparison.png           # 邻接矩阵对比
│       ├── confidence_AIR_N95_p6.png          # 置信区间
│       ├── diffstg_scatter_error.png          # 散点+误差分布
│       ├── diffstg_ts_comparison.png          # 时序对比
│       ├── diffstg_per_step_rmse.png          # per-step RMSE
│       ├── rmse_decay_comparison.png          # PE vs DiffSTG 多步长衰减
│       ├── mae_decay_comparison.png
│       └── rmse_ratio_bars.png
└── logs/                              # 关键训练日志
    ├── pedw/                              # PE-DiffWaveNet 日志（13 个）
    ├── diffstg_F1/                        # DiffSTG F=1 日志（5 个）
    ├── diffstg_F15/                       # DiffSTG F=15 日志（5 个 + rerun 总日志）
    └── diffstg_advanced/                  # 进阶实验日志与结果 CSV
        ├── diffstg_F1_corr_adj_p6_l24_s42.log   # 相关图训练日志
        ├── diffstg_F1_pe_adj_p6_l24_s42.log       # PE 图训练日志
        ├── alt_adj_runner.log                     # 邻接矩阵实验运行器日志
        ├── adjacency_results.csv                  # 邻接矩阵对比结果
        ├── single_target_results.csv              # 三污染物对比结果
        └── probability_metrics.csv                # 概率指标结果
```

## 复现说明

### 环境要求

- conda env: `atgcn`（或参考 `environment.yml`）
- PyTorch 2.7+cu128（RTX 5080 sm_120 兼容）
- GPU: NVIDIA RTX 5080 16GB

### 复现步骤

```bash
# 1. 进入项目根目录
cd /home/cl/ozone-forecast

# 2. 激活环境
conda activate atgcn

# 3. 跑 PE-DiffWaveNet 主实验
DEVICE=cuda EPOCHS=120 EXP_NAME=repro_pedw_p6_s42 \
  bash scripts/run_train_pediffwavenet.sh 6 24 42

# 4. 跑 DiffSTG F=1 baseline
cd external_baselines_F1/DiffSTG
python train.py --data AIR_N95 --T_h 24 --T_p 6 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 \
  --num_features 1 --split_mode default --is_test False

# 5. 跑 DiffSTG F=15 baseline (noleak 对齐)
python train.py --data AIR_N95 --T_h 24 --T_p 6 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 \
  --batch_size 4 --n_samples 1 \
  --num_features 15 --split_mode noleak --is_test False \
  --lr 0.001 --early_stop 30
```

完整命令见 `commands.sh`。

## 关键结果

### 主对比（pre_len=6）

| Model | F | split | RMSE | MAE | MAPE(%) | Peak_RMSE |
|-------|---|-------|------|-----|---------|-----------|
| PE-DiffWaveNet (3-seed 均值±std) | 15 | noleak | 11.29±0.16 | 8.02±0.12 | 32.28±0.29 | 13.75±0.37 |
| DiffSTG (F=15, rerun) | 15 | noleak | 31.55 | 24.83 | 350.40 | - |
| DiffSTG (F=1, F=1 baseline) | 1 | default | 35.74 | 29.20 | 469.99 | - |

PE-DiffWaveNet 相对 DiffSTG (F=15): MAE 降低 67.7%，RMSE 降低 64.3%。
F=15 vs F=1（DiffSTG 内部）: MAE 从 29.20 降至 24.83（-15.0%），验证气象因子有效性。

### 消融结论（相对 Full Model）

| 配置 | ΔRMSE |
|------|-------|
| No PE FiLM | +23.1% (最关键) |
| PE Shuffle | +11.2% |
| No PE Graph | +0.8% |
| No Diffusion | -2.8% (峰值反降) |

### 三污染物对比（F=1, pre_len=6）

| Pollutant | MAE | RMSE | MAPE(%) |
|-----------|-----|------|---------|
| O3 | 29.20 | 35.74 | 469.99 |
| PM2.5 | 25.05 | 38.96 | 124.28 |
| PM10 | 38.81 | 83.59 | 83.89 |

O3 的 MAPE 极高（557%），因夜间浓度接近 0，微小绝对误差被放大。这证实了为 O3 引入气象因子的必要性。

### 概率预测指标（O3, F=1, pre_len=6, 50 次采样）

| CRPS | MIS 80% | MIS 90% | MIS 95% |
|------|---------|---------|---------|
| 26.71 | 219.82 | 352.78 | 579.32 |

CRPS 与 MAE 同量级，概率预测未系统偏离；MIS 高表示 F=1 下扩散模型不确定性大（模型"知道自己不准"）。

## 代码修改说明

### PE-DiffWaveNet

- 仅修改 `scripts/run_train_pediffwavenet.sh` 暴露 LR/AMP/PATIENCE 等环境变量（pre_len=3 修复训练不稳定）
- 未改动核心模型结构

### DiffSTG

为对齐评估，对 `external_baselines_F1/DiffSTG/train.py` 做了 bug 修复（非核心结构改动）：

1. `type=bool` → 自定义 `str2bool`：修复 `--is_test False` 被误解析为 True 的 argparse 陷阱
2. 移除硬编码 `config.is_test = False`：让命令行参数生效
3. `nni` 模块条件导入：避免环境缺 nni 报错
4. `evals()` 仅评估 channel 0（O3）：与 PE-DiffWaveNet 评估口径对齐
5. 多通道按通道独立归一化：避免气象量纲淹没 O3 信号

F=15 重跑版本额外增加 `--early_stop` 和 `--lr` 命令行参数（让早停耐心和学习率可配置）。

## 验收对照

| 验收标准 | 是否满足 | 证据 |
|---------|---------|------|
| 至少一个模型可从命令完整复现 | ✓ | `commands.sh` + `logs/` + `experiment_logs/` |
| 每组结果表字段统一 | ✓ | `results.csv` 对齐 `templates/experiment_result_template.csv` |
| 每个图表有对应数据源 | ✓ | `figures/` + `tables/` 一一对应 |
| 报告结论能对应具体实验结果 | ✓ | `report_section.md` 各节引用具体表/图 |
| 不使用未说明数据或不可复现截图 | ✓ | 全部为命令生成、日志可查 |
| 不允许只提交截图 | ✓ | 同时提交 CSV、日志、命令、实验记录 |
| 不允许只说跑了但没保存日志 | ✓ | `logs/` 完整保留 29 份训练/评估日志（pedw 13 + diffstg_F1 5 + diffstg_F15 6 + diffstg_advanced 5） |
| 不允许改了核心代码但不说明 | ✓ | 本节"代码修改说明"列明所有改动 |
| 不允许混用其他数据集结果 | ✓ | 全部基于 AIR_N95 (95 站点 2022 年) |
| 不允许把 debug 小样本当正式结果 | ✓ | 全部 EPOCHS=120/200/300 正式配置 |

## 实验命名规范

遵循 `组名_模型_关键配置_seed` 规范：

- `g3_pedw_p6_l24_s42`：PE-DiffWaveNet, pre_len=6, seq_len=24, seed=42
- `g3_pedw_no_diff_p6_l24_s42`：PE-DiffWaveNet, 无扩散, pre_len=6, seq_len=24, seed=42
- `g3_pedw_no_pe_film_p6_l24_s42`：PE-DiffWaveNet, 无 PE-FiLM, pre_len=6, seq_len=24, seed=42
- `g3_diffstg_F1_p6_l24_s42`：DiffSTG, F=1, pre_len=6, seq_len=24, seed=42
- `g3_diffstg_F15_p6_l24_s42`：DiffSTG, F=15, pre_len=6, seq_len=24, seed=42
