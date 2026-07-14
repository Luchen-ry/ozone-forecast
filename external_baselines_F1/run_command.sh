# ============================================================
# DiffSTG Baseline — 运行记录
# 日期：2026-07-09 ~ 2026-07-12
# 环境：Windows 11, pedw (Python 3.11, PyTorch 2.7.1+cu118, RTX 3050 4GB)
# ============================================================

# ============================================================
# 一、验证官方代码可运行（PEMS08 自带数据集）
# ============================================================

cd external_baselines/DiffSTG

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

cd "d:/生产实习-时空数据/臭氧预测资料"
python external_baselines/prepare_air_n95_for_diffstg.py

# 终端输出：
# ┌─────────────────────────────────────────────────────────────┐
# │ [OK] flow.npy saved: (8717, 95, 1), range=[1.0, 410.0]    │
# │ [OK] adj.npy saved: (95, 95), non-zero=15.05%             │
# └─────────────────────────────────────────────────────────────┘


# ============================================================
# 三、AIR_N95 烟雾测试（极简配置，验证流程）
# ============================================================

cd external_baselines/DiffSTG

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

cd external_baselines/DiffSTG

python train.py \
  --data AIR_N95 --T_h 24 --T_p 6 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 --batch_size 4 --n_samples 1

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

cd external_baselines/DiffSTG

# pre_len=1（已有结果 per-step RMSE 第一步=25.0）
python train.py --data AIR_N95 --T_h 24 --T_p 1 --seed 42 --N 50 --sample_steps 50 --hidden_size 32 --batch_size 16 --n_samples 1

# pre_len=3
python train.py --data AIR_N95 --T_h 24 --T_p 3 --seed 42 --N 50 --sample_steps 50 --hidden_size 32 --batch_size 16 --n_samples 1

# pre_len=6（已完成，RMSE=41.10）
# 使用第四节结果

# pre_len=12
python train.py --data AIR_N95 --T_h 24 --T_p 12 --seed 42 --N 50 --sample_steps 50 --hidden_size 32 --batch_size 8 --n_samples 1

# pre_len=24
python train.py --data AIR_N95 --T_h 24 --T_p 24 --seed 42 --N 50 --sample_steps 50 --hidden_size 32 --batch_size 8 --n_samples 1


# ============================================================
# 八、进阶实验 — 邻接矩阵对比
#    用途：验证不同空间建模策略对预测的影响
# ============================================================

# 先运行脚本生成两种新邻接矩阵
cd "d:/生产实习-时空数据/臭氧预测资料"
python external_baselines/prepare_alt_adj.py

# 8a. 相关图（Pearson 相关矩阵）
#     手动将 output/corr_adj.npy 复制替换 data/dataset/AIR_N95/adj.npy
python train.py \
  --data AIR_N95 --T_h 24 --T_p 6 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 --batch_size 16 --n_samples 1

# 8b. PE 图（排列熵相似度矩阵）
#     手动将 output/pe_adj.npy 复制替换 data/dataset/AIR_N95/adj.npy
python train.py \
  --data AIR_N95 --T_h 24 --T_p 6 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 --batch_size 16 --n_samples 1


# ============================================================
# 九、进阶实验 — PM2.5 / PM10 单目标预测
#    用途：验证 DiffSTG 在不同污染物上的通用性
# ============================================================

# 先运行数据提取脚本生成 PM flow.npy
cd "d:/生产实习-时空数据/臭氧预测资料"
python external_baselines/prepare_pm_data.py

# 9a. PM2.5
cd external_baselines/DiffSTG
python train.py \
  --data AIR_N95_PM25 --T_h 24 --T_p 6 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 --batch_size 16 --n_samples 1

# 9b. PM10
python train.py \
  --data AIR_N95_PM10 --T_h 24 --T_p 6 --seed 42 \
  --N 50 --sample_steps 50 --hidden_size 32 --batch_size 16 --n_samples 1


# ============================================================
# 十、进阶实验 — 概率预测指标 & 置信区间图
#    用途：展示 DiffSTG 的不确定性量化能力
# ============================================================

# 不跑训练，从已有 forecast.pkl 提取：
#   CRPS  = Continuous Ranked Probability Score（越低越好）
#   MIS   = Mean Interval Score（越低越好）
#   画置信区间阴影带（预测均值 ± 标准差）
#
# 每个实验的输出目录中包含 forecast.pkl，运行：
cd "d:/生产实习-时空数据/臭氧预测资料"
python external_baselines/plot_probability_metrics.py
#   → 输出各实验的 CRPS/MIS + 置信区间图到 external_baselines/figures/


# ============================================================
# 十一、服务器运行注意事项
# ============================================================
# 1. 上传整个项目到服务器，保持目录结构不变
# 2. 服务器环境：pip install torch easydict nni numpy pandas
# 3. 首次运行：先跑 PEMS08 验证 → 生成 AIR_N95 数据 → 烟雾测试
# 4. 服务器 GPU 显存充足时可加大 batch_size（16/32/64），加快训练
# 5. 多条命令可用 && 串联或写 shell 脚本批量跑
# 6. 每个实验结束后记录指标到 baseline_results.csv
