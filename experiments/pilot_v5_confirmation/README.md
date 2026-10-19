# pilot_v5_confirmation

本目录将 v4 的两项背景观察固定为验证对比，使用五个全新数据/初始化配对重复。旧 v1–v4 代码、配置、checkpoint 与结果保持只读；旧 seed 17/42/73 为探索，新 seed 101/211/307/401/503 为验证，主统计不合并两批数据。

主对比：

| 目标 | 固定背景差 | 候选 B / C | 预指定交互方向 |
| --- | --- | --- | --- |
| fixed_points | g2_deficiencies − g0_v3 | anti_fixed_points / successions | Δ(B−C) > 0 |
| longest_decreasing_run | g1_recoils − g0_v3 | longest_increasing_run / left_to_right_maxima | Δ(B−C) < 0 |

方向来自 v4 探索。主量先在每重复内计算 `(B−C)新−(B−C)原`；B−A/C−A及源学习是辅助结果，不依据两个背景各自SD判断差异可靠性。

冻结前完成的材料：

- [探索逐seed重算](exploratory/EXPLORATORY_ANALYSIS_ZH.md)：两个目标的 g1−g0/g2−g0、三种配对差，共12条完整原始差值和逐seed图，无新训练。
- [统计方案](stats_plan.json)与[纯CSV统计模块](paired_statistics.py)：五重复均值、样本SD、范围、95%配对t区间、两个主要对比的Bonferroni同时区间；双侧exact sign/sign-flip仅作假设敏感诊断。n=5最小双侧p为0.0625，不保证显著。
- [数学说明](mathematics/README.md)、[公式和证据分类](mathematics/audit/RELATION_INVENTORY.csv)、[联合可推导性](mathematics/audit/TARGET_JOINT_DERIVABILITY.csv)、[反例](mathematics/audit/COUNTEREXAMPLES.csv)。合法任务关系与数据泄漏分开检查。
- [下一阶段任务族方案](TASK_FAMILY_PLAN.md)：新族、单/联合关系超边、可复用生成器职责；本轮不执行扩展。

完成后的主要结果为 [PILOT_V5_REPORT.md](PILOT_V5_REPORT.md)、`artifacts/confirmation_paired_differences.csv`、`confirmation_summary.csv`、`results_selected.csv` 及 `figures/confirmation_interactions_by_repeat.png`。全部五重复合计300条候选best/endpoint记录，分为 `results_seed101.csv`、`results_seed211.csv`、`results_seed307.csv`、`results_seed401.csv`、`results_seed503.csv`，每份60条，其中10条是该重复的主结果。

## 固定预算与配对

模型、编码与 v3/v4 相同：PermWorld one-line，n=10，4层Transformer、width256、8 heads、dropout .1、词表188、绑定embedding/output，参数3,240,448。source pool8192、source dev1024、target train256、target dev1024、新test2048。每重复内各目标/背景/条件共享对应输入池和初始化；A在两个背景间只训练和选择一次。

各B/C预训练4800步、batch64、精确每任务16条；候选slot0，公共slots1–3，B/C仅候选任务定义/标签不同。源LR沿用100步warmup至3e−4、2400步余弦至3e−5后常数；不延长。适配沿用3e−4/1e−4/3e−5三档，每档800步，每100步全量train/dev，依次按最高dev答案准确率、最低dev答案CE、最早step、固定LR顺序选择，step0不参与。优化器重置；目标绑定token/output行恢复同块初值；无新输出头。

共40个新增源run、150个新增适配LR候选、50个唯一选中模型、300条全部候选best/endpoint测试记录、60条按背景展开的A/B/C展示行。A重复展示不增加独立重复数。预训练192000更新/344064000输入tokens；适配120000更新/215040000输入tokens。全部训练token合计559104000，评估tokens另计。运行时间见 `artifacts/compute_summary.csv` 和 `stage_timings.csv`，不将训练timer与包含它的阶段墙钟相加。

数据seed为2026101901…1905，初始化seed为101/211/307/401/503；新测试seed为2026101951…1955。源采样/dropout seed=初始化seed+1000，适配=+2000。所有旧输入、新角色、新重复按完整inverse/reverse/complement轨道隔离；跨块不放回排除产生有限总体依赖，因此独立RNG流不等于严格IID池，t与符号推断有额外假设。测试生成只能发生在全部50项开发选择冻结之后。

## 复现命令

在项目根目录 `F:/2026DMLR` 的 PowerShell 中，使用项目已有 `.venv`。环境版本与GPU由每次运行的 `artifacts/environment.json` 记录；不需要 scipy，统计模块对df=4使用解析t CDF求临界值。

当前已完成目录的纯结果重建，不重新训练或评估模型：

```powershell
.venv/Scripts/python.exe experiments/pilot_v5_confirmation/paired_statistics.py self-check
.venv/Scripts/python.exe experiments/pilot_v5_confirmation/summarize.py confirmation
.venv/Scripts/python.exe experiments/pilot_v5_confirmation/audit_final.py
```

在新的artifact目录完整重跑固定实验，保留交付目录：

```powershell
.venv/Scripts/python.exe -u experiments/pilot_v5_confirmation/pilot_v5.py all --out experiments/pilot_v5_confirmation/artifacts_reproduction
.venv/Scripts/python.exe experiments/pilot_v5_confirmation/audit_final.py --out experiments/pilot_v5_confirmation/artifacts_reproduction
.venv/Scripts/python.exe experiments/pilot_v5_confirmation/summarize.py confirmation --out experiments/pilot_v5_confirmation/artifacts_reproduction
.venv/Scripts/python.exe experiments/pilot_v5_confirmation/audit_final.py --out experiments/pilot_v5_confirmation/artifacts_reproduction
```

默认交付报告保存在本目录的 `PILOT_V5_REPORT.md`；使用自定义 `--out` 时，报告写入该输出目录，图表和文档链接按报告实际位置重新计算，分别指向该次输出和本目录的冻结说明。

`all` 依次执行prepare、smoke、freeze、pretrain、adapt、select、test，已有完成标记时跳过，允许中断后用同一命令继续。训练按完整run恢复：已有completed标记的run跳过，未完成的run从头运行；它不是自动从最近step checkpoint续训。smoke中的保存恢复测试只核对底层状态恢复能力。错误/失败日志保留；不得因为某个重复结果不好而更换seed或延长训练。以结果文件为依据重建报告不会重新选择模型。

冻结之后不要在原路径重跑数学或探索输出构建器：其文件hash属于冻结证据，图的metadata时间也可能改变文件字节。独立数学复核使用新输出目录，详见数学README。冻结的 `paired_statistics.py` 不要改名为 `statistics.py`，避免遮蔽Python标准库同名模块。

## 文件职责与限制

`design.py`固定四个目标×背景组、所有seed、预算和原v4配置；`pilot_v5.py`负责生成、配对训练、开发选择、延后测试；`paired_statistics.py`是训练前冻结的纯数据统计；`summarize.py`只读取已保存CSV/JSON绘图报告，不接触模型；`audit_final.py`独立核对冻结、数据、checkpoint、采样流、选择及旧文件保留。

`source_learning_table.csv`包含160条源任务的暴露、按长度准确率、CE与多数基线；`source_curves.csv`保留完整训练/开发曲线。旧单任务引用单独保存为 `historical_single_references.csv`，明确不同输入/初始化，不能当作新严格配对的单任务对照。源状态与迁移变化只支持关联描述，不证明机制。

本轮还在新测试生成前完成了[源学习独立描述](SOURCE_LEARNING_NOTES.md)，`source_analysis/`保存全部候选与未换公共源的逐重复背景差，审计明确没有读取新目标测试。其生成器为 `source_learning_analysis.py`，只读源CSV/config、无需GPU；该说明的时间点与最终关联分析分开保留，不用后来目标结果改写测试前观察。

本轮仅检验固定256样本下的两个案例，不宣称完整样本效率曲线、scaling law或所有任务的一般性。完成后停止，不自动追加这两个目标的背景变体。
