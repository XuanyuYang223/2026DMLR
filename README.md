# 2026DMLR

最小 data-centric 迁移试点：固定四个预训练源任务，比较源任务组成对未见目标任务样本效率的影响。

- [中文试点结果](PILOT_REPORT.md)
- [任务、配置、数据隔离和复现命令](experiments/permworld_pilot/README.md)
- [准确率曲线](experiments/permworld_pilot/runs/pilot_v1/sample_efficiency.png)
- [全部原始结果](experiments/permworld_pilot/runs/pilot_v1/results_all.csv)

实验分支：`pilot/permworld-composition`。参考项目克隆在本地 `reference/`，不修改原 alignment 实验。模型 checkpoint 保留在本地实验目录，因体积未纳入 Git；数据、配置、源码、CSV、图和审计信息保留在仓库中。
