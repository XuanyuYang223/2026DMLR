# 2026DMLR

最小 data-centric 迁移试点：固定四个预训练源任务，比较源任务组成对未见目标任务样本效率的影响。

- [中文试点结果](PILOT_REPORT.md)
- [任务、配置、数据隔离和复现命令](experiments/permworld_pilot/README.md)
- [准确率曲线](experiments/permworld_pilot/runs/pilot_v1/sample_efficiency.png)
- [全部原始结果](experiments/permworld_pilot/runs/pilot_v1/results_all.csv)

实验分支：`pilot/permworld-composition`。参考项目克隆在本地 `reference/`，不修改原 alignment 实验。模型 checkpoint 保留在本地实验目录，因体积未纳入 Git；数据、配置、源码、CSV、图和审计信息保留在仓库中。

第二轮固定256样本诊断位于 `experiments/pilot_v2_diagnostics/`：

- [第二轮中文报告](experiments/pilot_v2_diagnostics/DIAGNOSTIC_REPORT.md)
- [第二轮配置、隔离规则与复现命令](experiments/pilot_v2_diagnostics/README.md)

第二轮分支：`pilot/permworld-v2-diagnostics`。原 `pilot_v1` 全部结果与checkpoint保留，并以文件哈希核验。

第三轮固定256样本、四目标、三个配对种子实验位于 `experiments/pilot_v3_multitarget/`：

- [第三轮任务目录](experiments/pilot_v3_multitarget/TASK_CATALOG.csv)
- [第三轮中文报告](experiments/pilot_v3_multitarget/PILOT_V3_REPORT.md)
- [第三轮配置与复现命令](experiments/pilot_v3_multitarget/README.md)

第三轮分支：`pilot/permworld-v3-multitarget`。前两轮文件与结果逐字节保留；跨目标结果只解释固定256样本下的迁移，不构成样本效率曲线或scaling law。

第四轮背景组合研究位于 `experiments/pilot_v4_backgrounds/`，分支为 `pilot/permworld-v4-backgrounds`：

- [第四轮中文报告](experiments/pilot_v4_backgrounds/PILOT_V4_REPORT.md)
- [配置、严格配对与复现命令](experiments/pilot_v4_backgrounds/README.md)
- [可复用数据平台文档](experiments/pilot_v4_backgrounds/DATA_PLATFORM.md)、[任务目录](experiments/pilot_v4_backgrounds/TASK_CATALOG.csv)与[联合数学关系补充](experiments/pilot_v4_backgrounds/catalog_addenda/README.md)

第四轮固定两个探索性目标、三个背景、三个配对种子和256个目标样本，复用兼容权重，在独立新测试上比较同一源任务替换的背景依赖性。

第五轮固定假设验证位于 `experiments/pilot_v5_confirmation/`，分支为 `pilot/permworld-v5-confirmation`：

- [第五轮中文报告](experiments/pilot_v5_confirmation/PILOT_V5_REPORT.md)
- [配置、种子与复现命令](experiments/pilot_v5_confirmation/README.md)
- [旧结果的逐种子配对重算](experiments/pilot_v5_confirmation/exploratory/EXPLORATORY_ANALYSIS_ZH.md)
- [数学关系与联合可推导性](experiments/pilot_v5_confirmation/mathematics/README.md)、[下一阶段任务族方案](experiments/pilot_v5_confirmation/TASK_FAMILY_PLAN.md)

第五轮只验证由第四轮探索选出的两个背景对比，使用五个新数据/初始化配对重复；主统计量为逐重复的背景间 B−C 交互差。旧探索与新验证分别报告，保留前四轮全部文件，不自动追加背景变体。
