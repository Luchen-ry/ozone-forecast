# 生产实习材料包：PE-DiffWaveNet 空气质量预测

本目录用于“生产实习”课程，目标是让学生基于现有臭氧、PM 和气象因子数据，围绕扩散模型空气质量预测完成可复现实验、结果整理和报告初稿。

核心要求很简单：不重新找数据，不随意改核心模型，重点帮助完成数据说明、baseline 对比、消融实验、图表和报告材料。

## 目录结构

```text
production_internship_pediffwavenet/
  README.md
  environment.yml
  code/                         # 当前模型核心代码副本
  data_N95/                     # 原始污染物逐日 CSV，含 O3/PM 等
  matrix_N95/                   # 当前模型直接使用的处理后数据和气象缓存
  xlsx_N95/                     # 95 个站点信息和辅助表
  paper_assets_pediffwavenet/   # 已有论文表格，学生结果需向这些字段对齐
  docs/                         # 任务书、三周安排、数据说明、提交规范
  templates/                    # 实验记录、结果汇总、报告提纲模板
  scripts/                      # 学生运行脚本
```

## 快速检查

先跑一个 CPU smoke test，确认环境和数据没有问题：

```bash
cd "/home/chenxudong/graduate/代码 2/代码/代码/production_internship_pediffwavenet"
bash scripts/run_smoke_cpu.sh
```

这只跑极少窗口和 1 个 epoch，用于验证代码、数据路径、气象缓存和输出目录是否正常。

## 正式训练示例

默认任务是历史 24 小时预测未来 6 小时，seed=42：

```bash
cd "/home/chenxudong/graduate/代码 2/代码/代码/production_internship_pediffwavenet"
DEVICE=cuda EPOCHS=120 EXP_NAME=student_pedw_p6_s42 bash scripts/run_train_pediffwavenet.sh 6 24 42
```

如果 GPU 不够，可以先用更小配置：

```bash
DEVICE=cpu EPOCHS=3 HIDDEN_SIZE=16 MAX_TRAIN_WINDOWS=64 MAX_VALID_WINDOWS=32 MAX_TEST_WINDOWS=32 \
  EXP_NAME=student_debug_cpu bash scripts/run_train_pediffwavenet.sh 6 24 42
```

## 学生最终交付

每个实验必须提交：

- 运行命令或配置；
- 日志文件；
- 输出目录；
- 指标结果 CSV；
- 1 页以内实验结论；
- 可放入报告的图表。

统一结果模板见 `templates/experiment_result_template.csv`。

## 教师侧建议

把学生分成四类任务最省事：

- 数据整理组：补数据说明、缺失统计、相关性图、站点图；
- baseline 组：跑通已有 baseline 或开源论文代码；
- 本模型实验组：跑 PE-DiffWaveNet 的主实验、多 seed、预测步长、消融；
- 结果整理组：统一表格、画图、整理报告初稿。

详细安排见 `docs/02_三周安排与分工.md`。

有代码论文推荐见 `docs/05_有代码论文推荐_DiffSTG.md`。首选 DiffSTG 作为扩散类时空图 baseline，能较自然地迁移到 95 站点空气质量预测。

学生实际执行顺序见 `docs/06_学生执行顺序.md`。
