# PermWorld 第二轮诊断

本轮固定 n=10、目标 fixed_points、B/C 四任务组成、one-line 编码和 v1 Transformer，仅分析256个目标训练样本、配对种子17/42。v1的所有文件及checkpoint保持原字节；新增数据、模型、曲线、审计、CSV放在本目录 `artifacts/`。本轮完成后暂停扩展。

## 冻结的比较

每个种子共15个适配候选：A随机初始化3个；B/C各在2400和4800源更新预算下各3个。所有候选学习率集合均为 `3e-4, 1e-4, 3e-5`，最多800更新、批量64，训练池为v1目标池前256条，开发集为原1024条；原数据、任务、模型、优化器类型、warmup100和余弦末端比例0.1不变。新增完整train/dev的accuracy、answer CE、EOS CE、二token CE每100步记录。训练batch loss另列，不能与完整数据eval loss混淆。

候选内与跨候选共同规则：开发答案accuracy最大，其次开发answer CE最小，其次更早更新，其次声明的学习率顺序。step0仅记录，不参选。固定最终800步仍保存并报告；不提前停止任何候选。A没有源预算，在2400/4800两组比较中复用同一组A结果，不重复算成独立实验。

v1缺少逐任务历史源损失、完整目标训练曲线和中间checkpoint。原源终点损失从已保存checkpoint补算；中间轨迹不伪造，重新适配的曲线明确为v2 replay。高学习率终点与v1原终点逐参数比较，以检查诊断插入是否改变原运行。历史v1测试结果保持原有探索性质，不混入v2测试均值。

## 预训练预算扩展

v1源任务开发结果支持尚未充分学习。因此B/C均从原2400步checkpoint继承模型、AdamW状态、NumPy采样和torch/CUDA随机状态，再训练2400步。仍是同一个8192输入池，四任务各占25%、每批各16条。扩展段固定原余弦末端学习率 `3e-5`，不重启warmup，也不重写前2400步。**这是真实原训练轨迹的继续，不是从头使用4800步cosine日程的等价实验。**

每个预算总输入tokens分别为4,300,800和8,601,600（含答案/EOS），监督tokens为307,200和614,400。每个源任务样本曝光38400→76800；“可用8192个输入”与“每任务实际访问的独特输入数”分别报告。源训练/开发全量指标在原终点和新增3000/3600/4200/4800步计算。

## 初始化、保存与隔离

每次适配先严格加载source/step0 state_dict并核对全部参数哈希；恢复 `<FIXED_POINTS>` 对应的token_embedding/lm_head绑定行至同一种子的step0，记录行哈希。共享数字输出参数沿用预训练值，没有新增独立输出头。新建AdamW、确认初始状态为空；所有候选配对采样schedule hash一致。源续训则明确继承优化器，不与适配混淆。

`FROZEN_PROTOCOL.json` 在训练前记录配置与训练代码哈希。`SELECTIONS_FROZEN.json` 在全部候选结束后记录每组选择及checkpoint哈希，然后才生成新测试。每种子2048个新的n=10排列；排除v1所有数据池（包含独立开发、两个旧测试池）和v2另一个种子的完整 inverse/reverse/complement 轨道。新测试只用于统一最终评估。所有候选best/endpoint都保留原始测试读数，主表只引用冻结选择，不再按测试挑候选。

多数基线由相应训练标签众数确定，按长度报告；本轮只有长度10，不声称跨长度诊断。answer accuracy采用与v1相同的prompt-only预测、全188词表argmax；EOS准确率条件于gold answer，不是自由生成整段准确率。

## 运行和复现

当前环境沿用v1的Python3.13 / torch2.11.0+cu128 / NumPy2.5.4，完整依赖见上一级 `permworld_pilot/requirements-lock.txt`。需要v1的原数据与step0/pretrain/256-adapt checkpoint；大体积checkpoint保留本地并由清单哈希验证，未提交Git。主训练代码直接import已冻结的v1模型/任务实现，未复制或更改数学定义。

在仓库根目录运行：

```powershell
.venv/Scripts/python.exe -u experiments/pilot_v2_diagnostics/run_all.py
.venv/Scripts/python.exe experiments/pilot_v2_diagnostics/audit_final.py
.venv/Scripts/python.exe experiments/pilot_v2_diagnostics/summarize.py
.venv/Scripts/python.exe experiments/pilot_v2_diagnostics/audit_final.py
```

`run_all.py` 串行执行 inventory→smoke→freeze→train→select→test，跳过有完成标记的阶段。单个完整候选可复用；未完成候选从其原始起点重做，不宣称任意中间步自动恢复。smoke实际检验源续训连续8步与4步保存加载再4步的参数逐项一致，并核验评估不消耗dropout RNG。

首次 `audit_final.py` 核验训练与测试产物，随后 `summarize.py` 生成报告、汇总CSV和曲线并引用审计结果；最后再次审计，更新包含这些生成产物的 `artifact_manifest.json`。汇总与最终审计均不运行模型forward。

独立重跑使用新目录，保留本次结果：

```powershell
.venv/Scripts/python.exe -u experiments/pilot_v2_diagnostics/run_all.py --out experiments/pilot_v2_diagnostics/replication
.venv/Scripts/python.exe experiments/pilot_v2_diagnostics/audit_final.py --out experiments/pilot_v2_diagnostics/replication
.venv/Scripts/python.exe experiments/pilot_v2_diagnostics/summarize.py --out experiments/pilot_v2_diagnostics/replication
.venv/Scripts/python.exe experiments/pilot_v2_diagnostics/audit_final.py --out experiments/pilot_v2_diagnostics/replication
```

如需从v1起始重建，先按 `../permworld_pilot/README.md` 在新checkout完整复现v1。数值核验记录设备和运行环境；不同CUDA/硬件环境不保证位级重现。后续实验不能用本轮新测试选择方案。
