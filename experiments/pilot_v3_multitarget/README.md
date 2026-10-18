# Pilot v3：固定256样本的多目标组成诊断

本实验在独立目录中保留四目标、三个配对种子的 PermWorld 比较；不修改 pilot_v1/pilot_v2 的代码、配置、数据、结果或 checkpoint。本轮结束后停止扩展。结果生成后阅读 [PILOT_V3_REPORT.md](PILOT_V3_REPORT.md)；若引擎未完成，汇总脚本拒绝生成“已完成”报告。

## 范围与设计

目标是 `fixed_points`、`exceedances`、`left_to_right_minima`、`longest_decreasing_run`，使用现有库的数学定义；相关源分别是 `anti_fixed_points`、`deficiencies`、`right_to_left_minima`、`longest_increasing_run`。目标与相关源满足 `target(p) = source(T(p))`，变换依次为补值、反演、反序、补值，答案映射为恒等。全部四目标从所有预训练集合中排除。fixed_points 与 exceedances 同属位置比较类；四个目标不是四个独立数学领域。

这四条关系都使用恒等答案映射，没有覆盖非平凡答案映射或多任务加和恒等式；任务目录保存了这些其他关系，但本轮在可学习性和规模约束下未纳入。

每个目标的 A 为随机初始化，B 为四源任务预训练，C 替换B的相关源并保留另三个公共源。fixed_points 保留 v1/v2 源组合和 successions 替换；其他候选集合在 `DESIGN_FROZEN.json` 中声明，最终组成在 `FROZEN_PROTOCOL.json` 中保存。C不代表数学上完全无关。

例如exceedances的C以recoils替换deficiencies时，公共源descents与recoils满足 `recoils(I(p)) = descents(p)`（目录中的 `descent_inverse`）。替换还改变源集合内部的已知关系结构，不能将B/C差值只归因于指定目标—源关系。

目标门槛使用seed17单任务4800步的源开发准确率：至少90%，且高于训练众数开发基线至少15个百分点。替换选择分数为源训练答案分布TV + 单任务4800步开发准确率差绝对值 + 600/1200/2400/3600/4800步开发准确率差绝对值的均值；取最小值，并列按候选顺序。没有使用迁移测试挑目标或替换任务。候选分数、剩余难度/分布差异全部保留，单一候选不声称通过搜索改善了匹配。

目标资格门槛使用8192条单任务训练样本；通过该门槛不意味着A在本轮256条目标样本、800步预算下也能学会。报告分别解释目标资格与小样本条件下接近多数基线的结果。

`TASK_CATALOG.csv` 区分数学任务、参数变体、编码/输出接口变体。目录覆盖当前PermWorld及项目已有其他world接口，本轮只运行现有PermWorld标量任务。`catalog/RELATION_AUDIT.csv` 和 `catalog/REPLACEMENT_RELATION_COUNTEREXAMPLES.csv` 记录关系检查；有限反例只能否定检查的恒等式，不能证明数学无关。

## 数据、模型与预算

- 输入均为长度10的均匀随机置换，one-line encoding。源训练池8192、源开发1024、目标训练256、目标开发1024、新测试2048，种子17/42/73。
- 同种子内所有目标与A/B/C共享置换输入，按不同任务定义重新生成精确标签。seed17/42复用原输入池；seed73输入新生成。目标间共享输入、公共源任务乃至完整源组合，不能将跨目标结果当独立样本。
- 新数据与新测试以 inverse/reverse/complement 生成的完整变换轨道隔离。新测试在36项开发选择冻结后生成，排除v1/v2所有输入和v3训练/开发；不同新测试种子也隔离。旧测试仅用于排除，不参与选择。
- 原Transformer：4层、宽256、8头、dropout0.1、188词表、绑定token embedding/output权重。FP32权重与评估、BF16训练、TF32关闭。精确模型配置见冻结JSON。
- 源预训练4800步、batch64，四任务每步各16个样本（25%）。LR前100步warmup到3e-4，至2400步cosine到3e-5，随后常数3e-5至4800；本轮不自动延长。每个组合前向训练输入8,601,600 tokens，监督614,400 tokens；每任务76,800次样本暴露。
- 所有实际源任务有三个种子的单任务参照，复用相同初始化和源输入池、相同4800步LR轨迹；保存1200/2400/4800 checkpoint及0/600/1200/2400/3600/4800全量train/dev曲线。单任务4800步每任务暴露是组合的4倍；单任务1200步仅等每任务暴露，LR轨迹不同，不能据此单独识别因果干扰。
- A/B/C各有三档适配LR：3e-4、1e-4、3e-5；每个候选800步、batch64、warmup100后cosine到初始LR的0.1，输入1,433,600 tokens、监督102,400 tokens。相同目标/种子使用相同样本顺序；不同条件共享相同最大选择预算。
- AdamW：weight decay=.01，betas=(.9,.999)，eps=1e-8；gradient clip=1。训练loss为答案与EOS两位置CE均值。prompt-only全词表argmax计算答案准确率，gold-conditioned EOS单独计量。多数基线使用训练众数，分别计算各split命中率。
- 适配加载源权重后，只将目标已有task token的绑定embedding/output行恢复为同种子step0值；未新增输出头。优化器清空。每100步评估完整训练/开发；最高开发答案准确率、最低开发answer CE、最早步数、声明LR顺序依次破同分，step0不参与选择。

fixed_points的seed17/42精确复用v2两倍预算的4个预训练模型与18个适配候选，核对checkpoint hash和配置/数据后使用v3新测试重新评估。这些历史权重不是新独立训练重复。其他20个组合预训练和90个适配候选实际新增；单任务运行数取决于冻结的源组合，见计算成本表。逻辑源组可能重复；`source_composition_reuse.csv` 记录有序/无序组合与模型state hash重复，重复计算按实际执行计成本。

这些复用的fixed_points源模型缺少2400步以前的全量train/dev曲线。图表仅保留2400步原预算终点以及v2续训3000/3600/4200/4800步记录，B/C曲线从2400步开始；不补造或插值到0/600步。本轮新训练的单任务和其他多任务曲线才有统一0/600/1200/2400/3600/4800评估。未选替换候选的单任务曲线与4800步终点同样保留，即使开发准确率未达90%；未选且未用作其他源任务的候选通常只有设计seed17参照。

## 复现

在仓库根目录 `F:\2026DMLR` 使用已有CUDA PyTorch环境（环境版本最终见 `artifacts/environment.json`）：

```powershell
.venv/Scripts/python.exe -u experiments/pilot_v3_multitarget/run_pipeline.py
.venv/Scripts/python.exe experiments/pilot_v3_multitarget/audit_final.py
.venv/Scripts/python.exe experiments/pilot_v3_multitarget/summarize.py
.venv/Scripts/python.exe experiments/pilot_v3_multitarget/audit_final.py
```

包装入口依次执行 `pilot_v3.py` 的 prepare、smoke、design_references、freeze，在freeze之后执行 `freeze_evidence.py` 保存冻结证据，然后完成single references、pretrain、adapt、select、test。完成标记存在时跳过该阶段；源码和配置hash检查阻止已冻结实验静默变更。原始底层入口 `pilot_v3.py all` 可执行引擎全部阶段，但完整审计流程应使用包装入口，确保额外冻结证据保存于后续计算之前。

在新目录重跑（仍按设计复用明确兼容的历史 fixed_points 权重，不覆盖现有输出）：

```powershell
.venv/Scripts/python.exe -u experiments/pilot_v3_multitarget/run_pipeline.py --out experiments/pilot_v3_multitarget/replication
.venv/Scripts/python.exe experiments/pilot_v3_multitarget/audit_final.py --out experiments/pilot_v3_multitarget/replication
.venv/Scripts/python.exe experiments/pilot_v3_multitarget/summarize.py --out experiments/pilot_v3_multitarget/replication
.venv/Scripts/python.exe experiments/pilot_v3_multitarget/audit_final.py --out experiments/pilot_v3_multitarget/replication
```

`summarize.py` 只读取保存的CSV/JSON，不import模型，不训练、推理或依据测试作新选择。先audit、后summary、再audit使报告能引用审计状态，最终manifest包含报告、表格和图。checkpoint保留本地并由hash追踪，通常不进入Git；复用路径指向v1/v2，迁移实验目录时须保留这些引用或完整复制仓库。

报告仅在 `FINAL_AUDIT.json` 的 `status` 为 `passed` 且 `old_files_byte_preserved` 为正时写历史字节保留检查通过；缺失或其他状态明确标为待审计，不把实验完成标记当作最终审计通过。

## 输出索引

| 文件 | 内容 |
| --- | --- |
| `TASK_CATALOG.csv` | 任务定义、输入/输出范围、生成、关系、变换和答案映射；区分参数/编码变体 |
| `PILOT_V3_REPORT.md` | 中文观察、限制与下一阶段建议 |
| `artifacts/results_seed17.csv` / `results_seed42.csv` / `results_seed73.csv` | 每种子72条全部候选best/endpoint新测试原始结果 |
| `artifacts/results_all_candidates.csv` / `results_selected.csv` | 216条披露全集 / 36条开发选择主结果 |
| `artifacts/results_summary.csv` | 每目标条件三个种子的均值、样本SD和范围 |
| `artifacts/paired_deltas.csv` / `paired_deltas_summary.csv` | B−A、C−A、B−C逐种子与描述性汇总 |
| `artifacts/selected_vs_endpoint.csv` | 所选LR的最佳开发checkpoint与800步终点 |
| `artifacts/target_curves.csv` / `late_training_diagnostics.csv` | 全部候选目标train/dev曲线 / 固定400→800窗口变化 |
| `artifacts/single_task_references.csv` | 单任务1200/4800步完整train/dev指标、长度及多数基线 |
| `artifacts/source_learning_all.csv` / `single_vs_multi_sources.csv` | 所有源学习曲线 / 同种子单多任务对照 |
| `artifacts/source_classification_summary.csv` | 预定义描述性分类计数，不将重复源出现数当独立任务 |
| `artifacts/source_learning_effect_counts.csv` / `source_learning_effects_unique.csv` | 源组合相对单任务的改善/下降计数；按seed×任务×source state去重后的完整正负差值 |
| `artifacts/common_source_differences.csv` | 公共源B/C开发准确率、CE及其差异 |
| `artifacts/source_composition_reuse.csv` | 源任务组合重复和相同模型state hash |
| `artifacts/replacement_matching.csv` | 全部预设替换候选及剩余匹配差异 |
| `artifacts/compute_by_run.csv` / `compute_summary.csv` / `stage_timings.csv` | 新增与历史复用预算、时间，分别计成本 |
| `artifacts/FROZEN_PROTOCOL.json` / `SELECTIONS_FROZEN.json` | 完整冻结配置、组成和开发选择 |
| `artifacts/FINAL_AUDIT.json` / `artifact_manifest.json` | 协议/泄漏/权重/保存核验及文件hash |
| `artifacts/figures/*.png` | 每目标逐种子准确率、配对差值、单多任务曲线与目标所选曲线 |

单/多任务分类门槛事先固定：单任务4800步开发准确率<90%为“本配置单任务未充分学会”；≥90%且组合低至少5个百分点为“单任务能学会、组合训练下降”；其余“不足以认定组合训练下降”。这受样本暴露和LR轨迹混杂影响，不是任务固有难度或纯多任务干扰的结论。

三类学习状态不会单独标记组合改善，因此另外披露组合4800步减单任务4800步开发准确率≥+5pp和≤−5pp的出现次数、每方向最多三个最强例子及全部差值。统计同时提供原始任务出现次数和seed×任务×source state hash去重版本；去重仍不意味着独立样本。单任务4800与组合4800的总优化器更新数和全局LR轨迹相同，但每任务暴露相差四倍并存在任务交替，纯任务干扰仍被混杂；单任务1200只匹配每任务暴露，LR历史与总更新数不同。改善与下降也受seed相关优化过程影响，不直接解释为因果协同或干扰。

主图均保留三个种子。跨目标共享输入与来源模型，只有三个配对种子，均值±SD仅作描述。本轮固定256样本，不声称建立完整样本效率曲线或scaling law；保留全部负迁移、无收益和未充分学会结果。
