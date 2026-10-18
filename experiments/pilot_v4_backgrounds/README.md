# Pilot v4：有限背景组成交互实验

本轮只比较两个由v3结果选定的探索性案例：`fixed_points` 与 `longest_decreasing_run`。每个目标保留v3的B/C候选对，分别放入三个预先冻结的公共任务背景，研究组成差值B−C怎样随背景改变。保留v1/v2/v3全部代码、配置、数据与结果；本轮完成后停止，等待后续指导。

最终报告为 [PILOT_V4_REPORT.md](PILOT_V4_REPORT.md)。`summarize.py` 仅读取保存的CSV/JSON及v3单任务参照，不import模型、不训练或推理；缺少完整 `TEST_COMPLETE.json` 时拒绝生成完成报告。

## 实验组成

| 目标 | B候选slot0 | C候选slot0 | g0公共slot1/2/3 |
| --- | --- | --- | --- |
| fixed_points | anti_fixed_points | successions | descents / peaks / left_to_right_maxima |
| longest_decreasing_run | longest_increasing_run | left_to_right_maxima | descents / peaks / successions |

三个背景固定如下：`g0_v3` 保留原组合；`g1_recoils` 仅将公共slot1的descents换为recoils；`g2_deficiencies` 仅将公共slot2的peaks换为deficiencies。每背景内部B/C仅slot0不同。新增背景依据数学定义及已有单任务记录在新迁移结果之前冻结，不声称替换任务难度、答案分布或集合内部关系完全匹配。C不表示数学上无关。候选/公共任务、已知关系、数据平台和精确复用依据见 [TASK_CATALOG.csv](TASK_CATALOG.csv)、[DATA_PLATFORM.md](DATA_PLATFORM.md)、[REUSE_AUDIT.md](references/REUSE_AUDIT.md)。

冻结的 [BACKGROUND_MATCHING.csv](catalog/BACKGROUND_MATCHING.csv) 保存实际差异：g1源训练答案TV为0.00989–0.01428，但1200步recoils开发准确率62.79%/74.02%/93.07%，低于对应descents的99.90%/99.71%/99.80%；g2的TV为0.76013–0.76526，peaks与deficiencies答案分布差异很大，早期学习曲线也不同。总体等分布、有限池直方图接近、学习过程匹配是不同要求；背景预先决定，没有以这些记录搜索迁移最优方案。

沿用的B/C候选也并非完全匹配。正式汇总生成 `artifacts/candidate_matching.csv`：从冻结的 `catalog/SOURCE_ANSWER_DISTRIBUTIONS.csv` 和 `catalog/SOURCE_REFERENCE_CURVES.csv` 计算两个目标各三个块的候选答案TV、单任务1200/4800步开发准确率及早期曲线差异，不使用本轮新增训练结果。原选择来源为 [v3 replacement_matching.csv](../pilot_v3_multitarget/artifacts/replacement_matching.csv)：fixed_points保留历史锚点，longest_decreasing_run沿用v3的seed17开发选择；此表只披露剩余混杂，不重新挑组合。

两个目标与各自相关源均满足 `target(p)=related(C(p))`，其中 `C(p)_i=n+1-p_i`；答案映射为恒等。这两个经v3观察选择的案例不代表完整任务库，也不覆盖全部关系类型。

定义级科学复核在组合冻结后、查看本轮新迁移结果之前另存为 [catalog_addenda/README.md](catalog_addenda/README.md)，未修改冻结组合或协议。它证明 `fixed_points(p)=n−D(p)−D(I(p))`、`anti_fixed_points(p)=n−D(C(p))−D(R(I(p)))`（D为deficiencies）。因此fixed_points的g2 B/C都包含与目标具有多输入联合关系的背景任务；一个输入上的单个D标签仍不足以确定目标，正式训练不提供这些配对查询或联合约束，不能推断模型利用了该关系。

g1移除descents而不是新增共存的descents/recoils对。LDR/LIR/descents/peaks共享局部比较符号序列，但不由彼此标量答案互定；g1移除descents，g2移除peaks，对局部计算结构与源集合内部关系的改变不同。原图对已声明关系取闭包不等于完整数学关系分类；详细定义、证明、上下界和反例见补充说明。

## 配对、预算与选择

- PermWorld长度10、one-line encoding，原Transformer 4层/宽256/8头/dropout0.1、188词表及绑定embedding/output。FP32权重与评估、BF16训练、TF32关闭。
- 配对块17/42/73沿用相同v3输入池和step0权重。每块源训练8192、源开发1024、目标训练256、目标开发1024；最终新测试2048。条件、背景和两个目标在同块共享输入，根据数学任务生成各自标签。
- 块标签不等同于数据生成RNG seed：s17/s42的输入由RNG20261018顺序生成；s73由RNG2026101803生成；初始化seed为17/42/73。预训练采样/dropout分别1017/1042/1073，适配2017/2042/2073。配对块同时改变输入池和初始化，不能单独解释为初始化效应。详见 `seed_provenance.json`。
- B/C预训练均4800更新、batch64，四slot每步各16条（25%）。同块所有背景/条件共享输入索引及slot赋值；每任务累计76800次暴露，真实不同输入数与三类流hash保存于 `source_exposures.csv` 和 `sampling_slots.csv`。重复暴露不等于不同样本数。
- 源LR前100步warmup至3e−4，2400步cosine到3e−5，再恒定至4800；AdamW weight decay .01、betas .9/.999、eps 1e−8、clip 1。本轮不继续延长预算。
- 适配各条件同三档LR：3e−4/1e−4/3e−5，每档800步、batch64，warmup100后cosine到起始LR的0.1。严格加载源权重后，恢复目标已有task token绑定行至该块step0值并清空优化器；不新增输出头。
- 每100步评估完整train/dev，按最高dev答案准确率、最低dev answer CE、最早步数、声明LR顺序选择，step0排除。所有候选完整训练800步，保留最佳dev checkpoint及终点。
- 全词表prompt-only argmax计算答案准确率；答案/EOS各占训练loss一半，EOS单独记录。多数基线用256条目标训练数据众数在各split计算，不使用测试众数。

A每目标/块只训练和选择一次，`background=shared`，展示时加入三个背景对照；没有给A额外选择预算。唯一适配候选126个（6个A单元×3 LR + 36个B/C单元×3 LR），唯一所选checkpoint42个。54行A/B/C展示表含重复显示的A，不能作为54次独立训练。全部候选best/endpoint共252条新测试结果披露，每块84条。

主交互为 `(B−C)_后背景 − (B−C)_前背景`，比较g1−g0、g2−g0、g2−g1；每项按同块相减。保留B−A、C−A、B−C逐块差值及均值/样本SD，SD不解释为置信区间。背景、目标及配对块共享结构，不能把全部记录当独立样本。

## 复用、源任务参照和测试隔离

36个逻辑预训练run中12个g0精确复用，24个新增；126适配候选中18个共享A和36个g0 B/C复用，72个新增。精确复用核对有序任务tuple、数据池、初始化、采样、预算及权重hash；不因无序源任务集合相同就宣称等价。历史权重在本轮新测试重评估，不是新增训练重复。

两个目标在每个背景下的C实际具有相同无序任务集合，successions与left_to_right_maxima在slot0/slot3互换，从而改变任务到输入的分配。目标未出现在预训练输入中，跨目标C权重差异不能解释成目标名称影响了预训练；两个案例也并非独立抽取的源集合。

全部实际源任务复用v3对应块的单任务1200/4800步曲线，不新增单任务训练。单任务4800与组合4800总更新数和全局LR轨迹相同，但每任务暴露是组合4倍且组合引入任务交替；单任务1200只匹配每任务暴露，LR历史和总更新不同。单/多任务差异为描述证据，不能单独识别干扰或协同。

g0中fixed_points块17/42复用源曲线缺少2400步以前的全量train/dev指标，仅有2400、3000、3600、4200、4800步；图从2400开始，不补造早期点。其他兼容历史曲线及本轮新训练曲线记录0/600/1200/2400/3600/4800步。目标训练/dev曲线完整保留0…800每100步记录。

42项dev选择与checkpoint hash冻结后才用RNG2026101805生成新测试；统一排除v1/v2/v3所有输入及v4训练/dev输入在inverse/reverse/complement完整轨道下的重叠，三个新测试池也互相隔离。同块目标、背景、条件共享测试输入。全部252次候选测试在冻结后披露，不按测试重新选择。

## 复现命令

仓库根目录 `F:\2026DMLR` 使用既有CUDA PyTorch环境，具体版本见最终 `artifacts/environment.json`：

```powershell
.venv/Scripts/python.exe -u experiments/pilot_v4_backgrounds/pilot_v4.py all
.venv/Scripts/python.exe experiments/pilot_v4_backgrounds/audit_final.py
.venv/Scripts/python.exe experiments/pilot_v4_backgrounds/summarize.py
.venv/Scripts/python.exe experiments/pilot_v4_backgrounds/audit_final.py
```

已完成阶段按标记跳过；冻结源码/配置/数据/证据hash防止静默改变。最后重跑audit使manifest包含报告、汇总和图。报告仅在最终审计status为passed且记录已保留历史文件数时写字节保留通过，缺失时标记待审计。

在独立新输出目录重跑：

```powershell
.venv/Scripts/python.exe -u experiments/pilot_v4_backgrounds/pilot_v4.py all --out experiments/pilot_v4_backgrounds/replication
.venv/Scripts/python.exe experiments/pilot_v4_backgrounds/audit_final.py --out experiments/pilot_v4_backgrounds/replication
.venv/Scripts/python.exe experiments/pilot_v4_backgrounds/summarize.py --out experiments/pilot_v4_backgrounds/replication
.venv/Scripts/python.exe experiments/pilot_v4_backgrounds/audit_final.py --out experiments/pilot_v4_backgrounds/replication
```

新输出运行仍按预设协议复用已冻结目录/参考证据与兼容v3权重，不覆盖原输出。checkpoint保留本地并由hash追踪，通常不进入Git；搬迁时需保留v1/v2/v3的引用。`catalog/build_catalog.py` 和 `references/reuse_inventory.py` 会写入被冻结的来源证据，只能在freeze之前或新实验版本中执行；冻结后不要在原地重新生成这些文件。

## 输出

| 文件（未标前缀均在artifacts/） | 内容 |
| --- | --- |
| `PILOT_V4_REPORT.md`（实验根目录） | 中文观察、协议、解释边界、成本与下一步 |
| `results_seed17.csv` / `results_seed42.csv` / `results_seed73.csv` | 每块84条候选best/endpoint原始结果 |
| `results_all_candidates.csv` / `results_selected.csv` | 252条全集 / 42唯一dev选择主结果 |
| `results_display_ABC.csv` / `results_summary.csv` | 54行背景展示（A重复显示）/三个块描述性汇总 |
| `paired_deltas.csv` / `paired_deltas_summary.csv` | B−A、C−A、B−C逐块及均值/SD |
| `background_interactions.csv` / `background_interactions_summary.csv` | 18条逐块Δ(B−C)及六项交互汇总 |
| `selected_vs_endpoint.csv` | 每项所选LR的best-dev与800步终点train/dev/test，区分早期选择与后期拟合 |
| `target_curves.csv` / `late_training_diagnostics.csv` | 全部126候选1134行train/dev曲线 / 固定400→800变化 |
| `source_curves.csv` / `source_learning_table.csv` | 源全量曲线 / 144行终点及单任务参照 |
| `source_exposures.csv` / `sampling_slots.csv` | 144行实际任务暴露及每块每slot索引/输入流核验 |
| `common_source_BC.csv` | 同一背景内相同公共数学任务的B/C差异 |
| `source_background_slot_changes.csv` | 背景间同task源学习变化；换task的slot不计算差值 |
| `interaction_source_context.csv` | 目标交互与候选/相同公共源状态变化并列，不作因果相关推断 |
| `source_effect_counts.csv` / `source_effects_unique.csv` | 单多任务正负变化，含seed/task/state去重计数 |
| `candidate_matching.csv` | 沿用B/C候选对的逐块答案TV与单任务1200/4800开发表现；仅使用冻结前已有证据 |
| `source_state_reuse.csv` | 逻辑源run有序任务及模型state hash重复 |
| `compute_by_run.csv` / `compute_summary.csv` / `stage_timings.csv` | 新增/复用成本分开，以及引擎阶段墙钟 |
| `FROZEN_PROTOCOL.json` / `SELECTIONS_FROZEN.json` / `FINAL_AUDIT.json` | 冻结配置、dev选择与最终审计 |
| `STATISTICS_AUDIT.json` | 独立重算配对差值/交互/均值/SD，并绑定对应CSV hash |
| `figures/*.png` | 逐块准确率、组成差值、背景交互、源/目标学习曲线 |

新增预训练训练输入tokens为206,438,400（24×4800×64×28），新增适配为103,219,200（72×800×64×28）；监督token计答案/EOS两位置。计数不含评估tokens；训练timer含全量评估、保存和部分记账，历史复用timer不计本轮新增，阶段墙钟不能与训练timer相加。既有单任务参照无新增训练成本。

本轮固定256样本和现有预算，不建立样本效率曲线或scaling law。完整保留负迁移、无收益、符号翻转和训练不足，不追加背景、任务、LR或预算。

最终报告基于实测幅度和反例讨论后续任务库覆盖：可逆输入变换及恒等/仿射答案映射、同分布不同计算、同输入多源/多输入联合关系、共享局部结构但标量不互定、单向信息关系与其他计算族。它是后续设计建议，不代表本轮增加了这些任务或实验。
