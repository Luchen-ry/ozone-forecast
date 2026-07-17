# 生产实习交付材料

## 本组完成工作

本组围绕**基于扩散模型的多站点空气质量预测**完成任务，重点在 PE-DiffWaveNet 主模型实验与 DiffSTG baseline 对齐。具体包括：

1. **环境适配**：解决 RTX 5080 (Blackwell sm_120) 与 PyTorch 2.5 不兼容问题，升级至 PyTorch 2.7+cu128
2. **PE-DiffWaveNet 完整实验矩阵（13 个实验）**：
   - 主实验（seq_len=24, pre_len=6, seed=42）
   - 多 seed 稳定性（seed=42/52/62）
   - 输入窗口消融（seq_len=12/24/48）
   - 预测步长扫描（pre_len=1/3/6/12/24）
   - 模块消融（无扩散 / 无 PE 图 / 无 PE FiLM / PE shuffle）
3. **DiffSTG Baseline 对齐**：
   - 修复 DiffSTG 3 个关键 bug（argparse type=bool 陷阱、多通道量纲爆炸、评估通道不一致）
   - F=1（单通道 O3）5 个预测步长实验，pre_len=6 取队友与我结果的合并最优
4. **结果整理**：统一字段结果表、9 张论文风格图表、报告段落初稿

## 目录结构

```
delivery/
├── README.md                # 本文件
├── commands.sh              # 实际运行命令 (可复现)
├── results.csv              # 统一字段指标表 (对齐 templates/experiment_result_template.csv)
├── report_section.md        # 本组报告段落 (对齐 templates/report_outline.md)
├── figures/                 # 论文风格图表
│   ├── fig1_multi_seed.png            # 多 seed 对比
│   ├── fig2_seq_len.png               # 输入窗口对比
│   ├── fig3_pre_len.png               # 预测步长误差曲线
│   ├── fig4_ablation.png              # 消融实验柱状图
│   ├── fig5_per_step_rmse.png         # per-step RMSE 曲线
│   ├── fig6_training_curve.png        # 训练/验证曲线
│   ├── fig7_pe_vs_diffstg.png         # PE vs DiffSTG (pre_len=6)
│   ├── fig8_predictions.png           # 真实值 vs 预测值曲线
│   ├── fig9_pe_vs_diffstg_prelen.png  # PE vs DiffSTG 多步长误差曲线
│   ├── station_location.png           # 95 站点地理分布
│   ├── o3_mean_timeseries.png         # O3 全年时序
│   ├── o3_daily_mean_timeseries.png   # O3 日均时序
│   └── o3_meteorological_correlation.png  # O3 与气象因子相关性
└── logs/                   # 关键训练日志
    ├── pedw/               # PE-DiffWaveNet 日志 (13 个)
    └── diffstg_F1/         # DiffSTG F=1 日志 (5 个 pre_len)
```

## 待补充（后续添加）

- **F=15 DiffSTG baseline**（O3 + 14 气象，noleak 切分）：4 个 pre_len（1/3/6/12）正在重跑，完成后追加到 `results.csv` 和 `figures/fig9_pe_vs_diffstg_prelen.png`
- **PM2.5 / PM10 实验**：队友正在跑
- **多目标扩展实验**：待 PM 数据完成后

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
```

完整命令见 `commands.sh`。

## 关键结果

### 主对比（pre_len=6）

| Model | RMSE | MAE | MAPE(%) | Peak_RMSE |
|-------|------|-----|---------|-----------|
| PE-DiffWaveNet (3-seed 均值±std) | 11.29±0.16 | 8.02±0.12 | 32.28±0.29 | 13.75±0.37 |
| DiffSTG (F=1) | 35.74 | 29.20 | 469.99 | - |

PE-DiffWaveNet 相对 DiffSTG：MAE 降低 72.6%，RMSE 降低 68.4%。

### 消融结论（相对 Full Model）

| 配置 | ΔRMSE |
|------|-------|
| No PE FiLM | +23.1% (最关键) |
| PE Shuffle | +11.1% |
| No PE Graph | +0.8% |
| No Diffusion | -2.8% (峰值反降) |

## 代码修改说明

### PE-DiffWaveNet

- 仅修改 `scripts/run_train_pediffwavenet.sh` 暴露 LR/AMP/PATIENCE 等环境变量（pre_len=3 修复训练不稳定）
- 未改动核心模型结构

### DiffSTG

为对齐评估，对 `external_baselines_F1/DiffSTG/train.py` 做了 3 处 bug 修复（非核心结构改动）：

1. `type=bool` → 自定义 `str2bool`：修复 `--is_test False` 被误解析为 True 的 argparse 陷阱
2. 移除硬编码 `config.is_test = False`：让命令行参数生效
3. `nni` 模块条件导入：避免环境缺 nni 报错
4. `evals()` 仅评估 channel 0（O3）：与 PE-DiffWaveNet 评估口径对齐
5. 多通道按通道独立归一化：避免气象量纲淹没 O3 信号

F=15 重跑版本额外增加 `--early_stop` 命令行参数（让早停耐心可配置）。

## 验收对照

| 验收标准 | 是否满足 | 证据 |
|---------|---------|------|
| 至少一个模型可从命令完整复现 | ✓ | `commands.sh` + `logs/` |
| 每组结果表字段统一 | ✓ | `results.csv` 对齐 `templates/experiment_result_template.csv` |
| 每个图表有对应数据源 | ✓ | `figures/` + `results.csv` |
| 报告结论能对应具体实验结果 | ✓ | `report_section.md` 第 5、6 节 |
| 不使用未说明数据或不可复现截图 | ✓ | 全部为命令生成、日志可查 |
| 不允许只提交截图 | ✓ | 同时提交 CSV、日志、命令 |
| 不允许只说跑了但没保存日志 | ✓ | `logs/` 完整保留 |
| 不允许改了核心代码但不说明 | ✓ | 本节"代码修改说明"列明所有改动 |
| 不允许混用其他数据集结果 | ✓ | 全部基于 AIR_N95 (95 站点 2022 年) |
| 不允许把 debug 小样本当正式结果 | ✓ | 全部 EPOCHS=120 正式配置 |
