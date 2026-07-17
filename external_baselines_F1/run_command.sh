# ============================================================
# DiffSTG Baseline — 运行记录
# 日期：2026-07-09 ~ 2026-07-12
# 环境：Windows 11, pedw (Python 3.11, PyTorch 2.7.1+cu118, RTX 3050 4GB)
# ============================================================

# ============================================================
# 一、验证官方代码可运行（PEMS08 自带数据集）
# ============================================================

cd external_baselines_F1/DiffSTG

python train.py \
  --data PEMS08 \
  --T_h 12 --T_p 12 \
  --N 10 --sample_steps 10 \
  --batch_size 4 --n_samples 1

# 终端关键输出：
# ┌─────────────────────────────────────────────────────────────┐
# │ GPU: 0                                                     │  GPU 可用
# │ sample num: 10690                                          │  数据加载
# │ Num_of_parameters: 996,793                                 │  模型大小
# │ Epoch 0: Loss 13.92 | Val MAE 167.43, Val RMSE 208.75     │  训练正常
# │ Epoch 1: Loss 10.31 | Val MAE 168.63, Val RMSE 210.79     │  loss 下降
# │ best model loaded from: ...UGnet...dm4stg                  │  模型保存/加载 OK
# └─────────────────────────────────────────────────────────────┘
# 结论：DiffSTG 官方代码训练、验证、保存模型全流程正常。验证通过。


# ============================================================
# 二、生成本项目数据（95 站点 O3）
# ============================================================

cd "$PROJECT_ROOT"
python external_baselines_F1/prepare_air_n95_for_diffstg.py

# 终端输出：
# ┌─────────────────────────────────────────────────────────────┐
# │ [OK] flow.npy saved: (8717, 95, 1), range=[1.0, 410.0]    │
# │ [OK] adj.npy saved: (95, 95), non-zero=15.05%             │
# └─────────────────────────────────────────────────────────────┘


# ============================================================
# 三、AIR_N95 烟雾测试（极简配置，验证流程）
# ============================================================

cd external_baselines_F1/DiffSTG

python train.py \
  --data AIR_N95 --T_h 24 --T_p 6 --seed 42 \
  --N 20 --sample_steps 10 --hidden_size 16 --batch_size 2 --n_samples 1 --is_test True

# 终端关键输出：
# ┌─────────────────────────────────────────────────────────────┐
# │ sample num: 5205 / 1742 / 1739                             │  训练/验证/测试样本数
# │ Num_of_parameters: 257,271                                 │  极小模型
# │ Epoch 0: Loss 11.46 | Val MAE 67.42, Val RMSE 85.63       │
# │ Epoch 1: Loss 10.04 | Val MAE 67.26, Val RMSE 85.38       │  loss 下降，流程正常
# └─────────────────────────────────────────────────────────────┘
# 结论：AIR_N95 数据正确加载，全流程（数据→训练→验证）无报错。


# ============================================================
# 四、AIR_N95 正式训练（seq_len=24, pre_len=6, seed=42，F=1）
# ============================================================

cd external_baselines_F1/DiffSTG

python train.py --data AIR_N95 --T_h 24 --T_p 6 --seed 42 --N 50 --sample_steps 50 --hidden_size 32 --batch_size 4 --n_samples 1

# 终端关键输出：
# ┌─────────────────────────────────────────────────────────────┐
# │ Num_of_parameters: 911,047                                 │  最终模型
# │                                                             │
# │ best model loaded from: ...AIR_N95...False+False...dm4stg  │
# │  |[34.50  41.10  ] |                                       │  最佳验证 MAE/RMSE
# │                                                             │
# │ Final results in test:                                     │
# │   sample_strategy:ddim_multi, sample_steps:40              │
# │   MAE  = 34.50                                             │
# │   RMSE = 41.10                                             │
# │   MAPE = 557.66%                                           │
# │   best_epoch = 17 (early stop)                             │
# │                                                             │
# │ Per-step RMSE (from forecast pkl):                         │
# │   1h=25.0  2h=34.1  3h=39.9  4h=43.7  5h=42.6  6h=43.1    │
# └─────────────────────────────────────────────────────────────┘


# ============================================================
# 五、结果汇总
# ============================================================

# 最终指标（extracted to baseline_results.csv）：
#   model       = DiffSTG (F=1, O3 only, no meteorological vars)
#   seq_len     = 24
#   pre_len     = 6
#   seed        = 42
#   MAE         = 34.50
#   RMSE        = 41.10
#   MAPE        = 557.66%
#   best_epoch  = 17 (early stop)
#   parameters  = 911,047
#
# 对比 PE-DiffWaveNet（论文，F=15, O3+14 meteorology）：
#   MAE  = 7.56   (DiffSTG 差 4.6x)
#   RMSE = 10.94  (DiffSTG 差 3.8x)
#   MAPE = 30.79% (DiffSTG 差 18x)
#
# 结论：缺少气象因子是 DiffSTG 误差大的主因。
#       O3 预测必须依赖温度、辐射、气压等气象条件。
#       该差距验证了 PE-DiffWaveNet 中气象因子输入的必要性。


# ============================================================
# 六、对官方代码的修改（5 处）
# ============================================================
# 1. 添加 --T_p 和 --seed 命令行参数
# 2. 添加 AIR_N95 数据配置（95 站点，动态读取 total_len）
# 3. 修复 torch.load 兼容性（weights_only=False）
# 4. 将 --is_test 默认值从 True 改为 False
# 5. setup_seed 改为使用命令行 --seed 参数（默认 42）


# ============================================================
# 七、进阶实验 — pre_len 扫描（预测步长 1/3/6/12/24）
#    用途：观察 RMSE 随预测步长的衰减曲线
# ============================================================

cd external_baselines_F1/DiffSTG

# pre_len=1（已有结果 per-step RMSE 第一步=25.0）
python train.py --data AIR_N95 --T_h 24 --T_p 1 --seed 42 --N 50 --sample_steps 50 --hidden_size 32 --batch_size 4 --n_samples 1

# pre_len=3
python train.py --data AIR_N95 --T_h 24 --T_p 3 --seed 42 --N 50 --sample_steps 50 --hidden_size 32 --batch_size 4 --n_samples 1

# pre_len=6（已完成，RMSE=41.10）
# 使用第四节结果

# pre_len=12
python train.py --data AIR_N95 --T_h 24 --T_p 12 --seed 42 --N 50 --sample_steps 50 --hidden_size 32 --batch_size 4 --n_samples 1

# pre_len=24
python train.py --data AIR_N95 --T_h 24 --T_p 24 --seed 42 --N 50 --sample_steps 50 --hidden_size 32 --batch_size 4 --n_samples 1


# ============================================================
# 八、进阶实验 — 邻接矩阵对比
#    用途：对比三种空间建模策略：距离图 / 相关图 / PE 图
# ============================================================
#
# 【前置条件】确保以下文件存在，不在 git 里需单独传输：
#   - matrix_N95/data.npy             O3 原始数据 (95, 8717)
#   - DiffSTG/data/dataset/AIR_N95/flow.npy  已有，用于生成矩阵
#   - DiffSTG/data/dataset/AIR_N95/adj.npy   距离矩阵（原版）
#   - pip install scipy               prepare_alt_adj.py 的依赖
#
# 【操作步骤】按顺序执行，共 5 步

# ---- 步骤 1：生成两种新邻接矩阵 ----
cd "$PROJECT_ROOT"
python external_baselines_F1/prepare_alt_adj.py
# 输出：output/corr_adj.npy  ← Pearson 相关图
#       output/pe_adj.npy    ← 排列熵相似度图

cd external_baselines_F1/DiffSTG

# ---- 步骤 2：备份原始距离矩阵 ----
cp data/dataset/AIR_N95/adj.npy data/dataset/AIR_N95/adj_distance.npy
# 这一步很重要！跑完才能恢复

# ---- 步骤 3：相关图实验 ----
cp "$PROJECT_ROOT/output/corr_adj.npy" data/dataset/AIR_N95/adj.npy
# ↑ 替换为相关图
python train.py --data AIR_N95 --T_h 24 --T_p 6 --seed 42 --N 50 --sample_steps 50 --hidden_size 32 --batch_size 4 --n_samples 1
# 记录日志文件名，提取 test MAE/RMSE

# ---- 步骤 4：PE 图实验 ----
cp "$PROJECT_ROOT/output/pe_adj.npy" data/dataset/AIR_N95/adj.npy
# ↑ 替换为 PE 图
python train.py --data AIR_N95 --T_h 24 --T_p 6 --seed 42 --N 50 --sample_steps 50 --hidden_size 32 --batch_size 4 --n_samples 1

# ---- 步骤 5：恢复原始矩阵 ----
cp data/dataset/AIR_N95/adj_distance.npy data/dataset/AIR_N95/adj.npy

# 【实验结果】
# ┌─────────────────────────────────────────────────────────────┐
# │ Adjacency     MAE      RMSE                                 │
# │ Distance      34.50    41.10    ← 基准                      │
# │ Correlation   35.86    42.11    ← 差 +1.36 / +1.01          │
# │ PE            37.97    44.62    ← 差 +3.47 / +3.52          │
# │                                                             │
# │ 结论：三种矩阵差距极小（RMSE 最多差 3.5），瓶颈不在空间建模。│
# │ 详见 results/adjacency_results.csv                          │
# └─────────────────────────────────────────────────────────────┘


# ============================================================
# 九、进阶实验 — PM2.5 / PM10 单目标预测
#    用途：验证 DiffSTG 在不同污染物上的通用性
# ============================================================

# 先运行数据提取脚本生成 PM flow.npy
cd "$PROJECT_ROOT"
python external_baselines_F1/prepare_pm_data.py

# 9a. PM2.5
cd external_baselines_F1/DiffSTG
python train.py --data AIR_N95_PM25 --T_h 24 --T_p 6 --seed 42 --N 50 --sample_steps 50 --hidden_size 32 --batch_size 4 --n_samples 1

# 9b. PM10
python train.py --data AIR_N95_PM10 --T_h 24 --T_p 6 --seed 42 --N 50 --sample_steps 50 --hidden_size 32 --batch_size 4 --n_samples 1

# 【实验结果】
# ┌─────────────────────────────────────────────────────────────┐
# │ Pollutant  MAE      RMSE     MAPE       Data Range          │
# │ O3         34.50    41.10    557.66%    [1, 410]            │
# │ PM2.5      25.05    38.96    124.28%    [1, 543]            │
# │ PM10       38.81    83.59     83.89%    [1, 2759]           │
# │                                                             │
# │ 结论：O3 最难（光化学反应），PM2.5 最易（更持久），         │
# │ 三者在 F=1 下均显著弱于 PE-DiffWaveNet F=15。               │
# │ 详见 results/single_target_results.csv                      │
# └─────────────────────────────────────────────────────────────┘


# ============================================================
# 十、进阶实验 — 概率预测指标 & 置信区间图
#    用途：展示扩散模型的不确定性量化能力
# ============================================================

# 加载已保存模型做 multi-sample (n=50) 推理。
# 约 300 batch × 50 samples ≈ 5 分钟。
cd external_baselines_F1

#O3 (pre_len=6)
python compute_probability_metrics.py
# 也可修改脚本内 DATA_NAME 跑其他污染物
#例如：10b. PM2.5（改 DATA_NAME='AIR_N95_PM25' 后运行）
# python compute_probability_metrics.py
# 10c. PM10（改 DATA_NAME='AIR_N95_PM10' 后运行）
# python compute_probability_metrics.py

# 输出：
#   probability_metrics.csv     CRPS + MIS(80%/90%/95%)
#   figures/confidence_*.png    置信区间图

# O3 结果（300 batch, n_samples=50）：
# ┌─────────────────────────────────────────────────────────────┐
# │ Experiment       CRPS    MIS_80   MIS_90   MIS_95           │
# │ AIR_N95_p6      26.71   219.82   352.78   579.32            │
# │                                                             │
# │ CRPS ≈ 26.7 与 MAE=34.5 同一量级，概率估计合理但不够精准。   │
# │ MIS 较高说明 F=1 下扩散模型不确定性很大——模型"知道自己不准"。 │
# │ 此为扩散生成式模型的诚实不确定量化，论文章正面论述。         │
# └─────────────────────────────────────────────────────────────┘


# ============================================================
# 十一、实验结果文件索引
# ============================================================
# results/baseline_results.csv         主对比（pre_len=6, 对齐 table1）
# results/results.csv                  多步长衰减（1/3/6/12/24）
# results/single_target_results.csv    三污染物对比
# results/adjacency_results.csv        邻接矩阵对比
# results/probability_metrics.csv      概率预测指标
# results/README.md                    结果文件说明
#
# figures/README.md                    完整图表说明 + 分析结论
# plot_scripts/                        所有绘图脚本
# compute_probability_metrics.py       概率指标计算脚本
