# pilot_v5：固定假设的背景交互验证

本轮完成两个由v4探索结果提出的固定对比、五个新数据/初始化配对重复。主量是在每个重复内先计算的 `Δ(B−C)=(B−C)新背景−(B−C)原背景`，不是两个背景均值各自标准差的比较。

- **fixed_points**：预指定均值方向未复现。交互均值-3.02pp，2/5个重复符合预指定方向；95%配对t区间[-22.57, +16.54]pp，两项Bonferroni同时区间[-27.63, +21.60]pp。
- **longest_decreasing_run**：验证点估计方向与探索一致，但单项和同时区间仍包含0，尚未取得稳定的背景效应证据。交互均值-13.15pp，4/5个重复符合预指定方向；95%配对t区间[-28.89, +2.58]pp，两项Bonferroni同时区间[-32.97, +6.66]pp。

以上只验证这两个预选案例、n=10、256个目标样本和固定预算。旧三个seed属于探索，新五个重复属于验证；没有将它们合并为八次重复。训练及开发选择已完成后才生成统一新测试，保留所有负收益、未学会和不一致结果。

## 先重算探索证据，再冻结验证

从v4的42条唯一selected原始行重算所有目标的g1−g0、g2−g0，12条逐seed背景差和图见[探索说明](exploratory/EXPLORATORY_ANALYSIS_ZH.md)。FP g2−g0的Δ(B−A)/Δ(C−A)/Δ(B−C)探索均值为+16.59/+12.22/+4.36pp，主交互逐seed为+1.42/+0.34/+11.33pp；LDR g1−g0为−11.56/+0.44/−12.00pp，主交互逐seed为−20.07/+2.00/−17.92pp。共享A在背景差中严格消去。

预先固定H1：fixed_points，原背景→deficiencies背景，期望主交互为正；H2：longest_decreasing_run，原背景→recoils背景，期望主交互为负。方向来自v4探索，不是独立提出的理论预测。此次不再验证另外两个背景对比，也不依据新结果更换方向、改用单侧检验或新增重复。

| 目标 | 候选B / 候选C | 原三个公共源 | 新三个公共源 |
| --- | --- | --- | --- |
| fixed_points | anti_fixed_points / successions | descents, peaks, left_to_right_maxima | descents, deficiencies, left_to_right_maxima |
| longest_decreasing_run | longest_increasing_run / left_to_right_maxima | descents, peaks, successions | recoils, peaks, successions |

候选固定slot0，公共源slots1–3。两个目标均不进入任一预训练组；数学任务关系与实例隔离是不同层面的检查。组定义和统计方法在新正式训练之前按hash冻结，详见[冻结配置](artifacts/FROZEN_PROTOCOL.json)、[stats_plan.json](stats_plan.json)。

## 五个新重复的验证结果

| 目标 | v4探索均值 pp | v5验证均值±样本SD pp | 逐重复范围 pp | 95%配对t区间 pp | 两项同时区间 pp |
| --- | --- | --- | --- | --- | --- |
| fixed_points | +4.36 | -3.02 ± 15.75 | [-22.80, +20.70] | [-22.57, +16.54] | [-27.63, +21.60] |
| longest_decreasing_run | -12.00 | -13.15 ± 12.67 | [-24.46, +1.66] | [-28.89, +2.58] | [-32.97, +6.66] |

| 目标 | 新重复/初始化seed | Δ(B−A) pp | Δ(C−A) pp | 主Δ(B−C) pp |
| --- | --- | --- | --- | --- |
| fixed_points | 101 | +32.96 | +12.26 | +20.70 |
| fixed_points | 211 | +26.27 | +26.07 | +0.20 |
| fixed_points | 307 | +23.19 | +28.81 | -5.62 |
| fixed_points | 401 | +25.20 | +32.76 | -7.57 |
| fixed_points | 503 | +5.47 | +28.27 | -22.80 |
| longest_decreasing_run | 101 | -26.27 | -1.81 | -24.46 |
| longest_decreasing_run | 211 | -22.80 | -0.54 | -22.27 |
| longest_decreasing_run | 307 | -0.29 | +0.15 | -0.44 |
| longest_decreasing_run | 401 | -8.20 | -9.86 | +1.66 |
| longest_decreasing_run | 503 | -33.30 | -13.04 | -20.26 |

![逐重复主交互及区间](artifacts/figures/confirmation_interactions_by_repeat.png)

![主交互的B−A/C−A分量](artifacts/figures/confirmation_background_components.png)

下表为五重复的准确率均值，仅帮助理解水平与多数基线；A每目标/重复只训练和选择一次，在背景展示表中重复显示。60条展示行对应50个唯一选中模型，并非60次独立选择。

| 目标 | 背景 | A准确率 | B准确率 | C准确率 | 目标多数基线 |
| --- | --- | --- | --- | --- | --- |
| fixed_points | g0_v3 | 80.96% | 67.10% | 62.83% | 36.30% |
| fixed_points | g2_deficiencies | 80.96% | 89.72% | 88.47% | 36.30% |
| longest_decreasing_run | g0_v3 | 53.56% | 71.67% | 57.40% | 53.67% |
| longest_decreasing_run | g1_recoils | 53.56% | 53.50% | 52.38% | 53.67% |

LDR的随机初始化A均值53.56%，与多数基线53.67%接近；recoils背景的B/C也接近该基线。这说明固定256样本和当前适配预算下，目标学习仍有明显限制。A在主要交互中严格消去，因此不改变主统计量的定义，但收益水平不能脱离这项训练限制来解释。

区间直接基于五个配对交互差，SD使用n−1分母。单项区间采用df=4、t=.975分位2.776445；两个主要对比的Bonferroni同时区间，每项边际覆盖97.5%，临界值3.495406。后者不要求两个目标对比之间独立，但t覆盖仍依赖重复层面的近似独立正态假设。五次重复不足以检验正态性或尾部；不同RNG/初始化流也不等于严格IID数据池，因为跨块orbit排除构成有限总体不放回依赖。区间包含有限测试池的随机变化，不能当作所有任务族的区间。

| 目标 | 双侧符号翻转p | 双侧符号p | 符合方向/非零n | 留一均值范围 pp |
| --- | --- | --- | --- | --- |
| fixed_points | 0.6250 | 1.0000 | 2/5 | [-8.95, +1.93] |
| longest_decreasing_run | 0.1875 | 0.3750 | 4/5 | [-16.86, -10.33] |

符号翻转诊断穷举32种符号，要求零中心下独立符号对称，不是由本实验设计保证的处理随机化检验；符号检验另要求独立重复下正负概率各1/2，针对符号/中位数倾向而非均值。五个非零重复时两者双侧p最小都为0.0625，所以不能承诺常规显著性。留一均值只描述对单个重复的敏感性，不替代主分析。辅助Δ(B−A)/Δ(C−A)及源学习只报告描述结果，不增加检验家族或事后筛选显著结果。

## 联合数学关系与数据隔离

实际任务实现与独立核验见[数学关系说明](mathematics/README.md)、[核验结果](mathematics/MATH_AUDIT.json)。清单明确区分有公式/证明、数据观察和未检查；穷举与随机大n核验是实现检查，不能将有限核验代替一般证明。尤其要区分同一输入的源答案能否联合恢复目标，和允许对变换输入另做查询时能否恢复目标。

对于每个位置i，p(i)与i恰好满足等于、大于、小于之一，所以 `fixed_points(p)+exceedances(p)+deficiencies(p)=n`。若同时提供同输入的exceedances和deficiencies答案，则可求fixed_points；本轮FP新背景只有deficiencies，没有exceedances，因此不能仅凭该等式声称当前组同输入联合给出了fixed_points。又有 `exceedances(p)=deficiencies(inverse(p))`，故允许额外逆排列查询时，`fixed_points(p)=n−deficiencies(p)−deficiencies(inverse(p))`；本轮没有给模型显式成对查询或联合推理监督。

**FP的deficiencies背景中，B和C都具有上述严格的跨输入联合关系，不能把C解释为不含目标数学关系的对照。** 数学核验已为v4全部12组提供n=10的反例：相同n及同一输入的四源标量答案不能唯一恢复目标。限定到四源×I/R/C八变换的32个答案时，六个B都可从候选变换查询恢复目标，FP g2 C可用两次deficiencies查询恢复；其余五个C各有n=10不充分反例。这只排除指定答案查询模型，不排除从raw input计算目标或其他未检查关系。清单见[TARGET_JOINT_DERIVABILITY.csv](mathematics/audit/TARGET_JOINT_DERIVABILITY.csv)和[显式反例](mathematics/audit/COUNTEREXAMPLES.csv)。

核验实际完成S1…S7穷举5913个输入与4096个n=10 fixture；20个定义在八变换上完成1,601,440次独立标签比对，28条公式/界各验证10,009次。fixture仅为数学实现核对，没有送入训练或用于选择迁移配置。仅由数据观察的关联与未检查机制分别见[OBSERVED_ASSOCIATIONS.csv](mathematics/audit/OBSERVED_ASSOCIATIONS.csv)、[UNCHECKED_CLAIMS.csv](mathematics/audit/UNCHECKED_CLAIMS.csv)。

相关候选anti_fixed_points与fixed_points由补值/反序输入变换联系，LIR/LDR由补值或反序变换联系。descents、runs、peaks/valleys等还共享局部比较结构。它们是合法数学关系，不能自动标为泄漏；本轮泄漏检查针对训练、开发、测试的实际输入及inverse/reverse/complement完整轨道重叠。是否能从某组全部源标量答案恢复目标，以及是否需要变换查询，以数学清单逐组条目和反例为准；没有检查的关系不写成不存在。

## 源学习辅助结果及机制边界

每个四源run4800步，batch64，每步每任务16条，精确25%采样。各任务76800次暴露、2150400输入tokens、153600监督tokens；实际不同输入数、流hash、每长度(n=10)训练/开发准确率、答案CE、EOS CE与多数基线逐条保存于[source_learning_table.csv](artifacts/source_learning_table.csv)与[source_exposures.csv](artifacts/source_exposures.csv)。

| 目标 | 背景 | 条件 | 候选源 | train均值 | dev均值 | dev范围 | dev多数基线 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_points | g0_v3 | B | anti_fixed_points | 99.19% | 99.08% | [98.05%, 99.41%] | 37.03% |
| fixed_points | g0_v3 | C | successions | 49.26% | 46.66% | [41.80%, 51.27%] | 39.00% |
| fixed_points | g2_deficiencies | B | anti_fixed_points | 98.51% | 98.50% | [97.66%, 99.32%] | 37.03% |
| fixed_points | g2_deficiencies | C | successions | 51.44% | 48.95% | [44.82%, 56.05%] | 39.00% |
| longest_decreasing_run | g0_v3 | B | longest_increasing_run | 76.78% | 74.04% | [56.54%, 96.29%] | 54.00% |
| longest_decreasing_run | g0_v3 | C | left_to_right_maxima | 98.21% | 97.73% | [96.58%, 98.93%] | 31.35% |
| longest_decreasing_run | g1_recoils | B | longest_increasing_run | 57.02% | 57.01% | [53.22%, 62.11%] | 54.00% |
| longest_decreasing_run | g1_recoils | C | left_to_right_maxima | 97.06% | 96.27% | [93.16%, 98.05%] | 31.35% |

| 目标 | 重复 | 目标交互 pp | B候选源dev背景差 pp | C候选源dev背景差 pp | 未换公共源最大绝对dev差 pp |
| --- | --- | --- | --- | --- | --- |
| fixed_points | 101 | +20.70 | -0.49 | -2.25 | 21.88 |
| fixed_points | 211 | +0.20 | -1.56 | +12.11 | 11.13 |
| fixed_points | 307 | -5.62 | +1.27 | -2.15 | 4.39 |
| fixed_points | 401 | -7.57 | -1.17 | +3.52 | 7.91 |
| fixed_points | 503 | -22.80 | -0.98 | +0.20 | 8.30 |
| longest_decreasing_run | 101 | -24.46 | -23.05 | +1.46 | 33.11 |
| longest_decreasing_run | 211 | -22.27 | -15.14 | -0.78 | 9.47 |
| longest_decreasing_run | 307 | -0.44 | -2.25 | -5.76 | 13.28 |
| longest_decreasing_run | 401 | +1.66 | -10.55 | -1.27 | 8.20 |
| longest_decreasing_run | 503 | -20.26 | -34.18 | -0.98 | 35.06 |

上述10条为全部重复，而不是按迁移幅度挑出的案例。候选源两列比较的是同一任务跨背景的dev准确率；公共源最大值仅覆盖没有更换定义的任务。更换的peaks/deficiencies或descents/recoils只并列其水平与基线，不把不同任务的准确率相减当作学习改变。相同背景内公共任务B/C dev差的绝对值最大为45.70pp，完整记录见[common_source_BC.csv](artifacts/common_source_BC.csv)。

源记录能支持“组合改变与源学习状态改变同时出现”及“某些源在当前预算仍未学会”的描述，不能证明源准确率变化导致迁移交互。源与目标可能共同受优化、共享计算、答案分布、表示竞争和有限样本影响；本轮没有对这些中介量做独立干预。数学上可推导也不等于Transformer实际学会或利用了公式。若某个重复源与迁移变化方向不一致，仍完整保留，不能事后只解释符合直觉的重复。

源学习的测试前独立描述已保存在[SOURCE_LEARNING_NOTES.md](SOURCE_LEARNING_NOTES.md)，其审计记录当时TEST_COMPLETE尚不存在。以下才将源记录与冻结后的目标结果并列，区分证据和解释。

**FP：B/C共同改善再次出现，预指定的正替换交互未复现。** anti_fixed_points的源dev均值由99.08%变为98.50%，两边都已接近99%；4/5个重复中该源dev下降但B的目标准确率提高。因此“原背景没有学会相关候选”不能作为这一目标的统一解释，也不能用候选源准确率直接代理目标收益。B−A与C−A的背景改变均值分别为+22.62/+25.63pp，五个重复的两项都提高，但后者平均更大；主交互为-3.02pp。deficiencies在B/C新背景中都学得很高，也都具有跨输入联合恢复FP的公式，这是重要的设计事实；本轮并未证明模型利用该公式，不能把共同收益直接归因于这条路径。

**LDR：源学习下降与B目标收益下降共同出现，但不能据此推出交互机制。** LIR源dev均值由74.04%变为57.01%，五次均下降；新背景train均值57.02%、dev多数基线均值54.00%。训练和开发都低，支持当前预算下源学习不足，而不是仅有训练拟合后的泛化落差。五次B目标准确率也均下降，B−A背景改变均值-18.17pp；C−A同时改变-5.02pp，所以主交互均值为-13.15pp。保留关键反例：seed401的LIR源dev变化-10.55pp，目标B变化-8.20pp、C变化-9.86pp，主交互却为+1.66pp。交互取决于B和C两边，单看B源下降会漏掉它。

**公共源并非一致受损。** LDR的peaks跨两个条件共10条同任务背景差，8条降低、2条提高；正向项为C/seed211: +2.34pp; C/seed307: +2.93pp。这也说明v4“peaks全部下降”的源观察在新重复中不逐项成立。公共源以及successions的变化存在seed/条件差异，不能把组合效应描述为所有源同步变差。

已有记录支持上述并列变化、低训练学习水平和反例。它们仍无法区分共享计算结构、优化速度/梯度竞争、答案分布、联合关系利用或适配阶段可塑性等解释；本轮没有中介量干预、梯度实验或显式变换查询，也没有证明任何机制。

旧v3单任务记录作为历史参照复制到[historical_single_references.csv](artifacts/historical_single_references.csv)，本轮没有重训单任务。它们使用旧seed/输入池，与新多任务不构成严格配对对照；单1200步只匹配每任务暴露，学习率轨迹与总更新仍不同，单4800步暴露为多任务每任务的四倍。因此不能把新多任务−旧单任务差直接解释为组合干扰效应。完整新源曲线与目标曲线均保留，选中LR的train/dev图位于[figures目录](artifacts/figures)。

## 配置、实际预算与复现

| 重复 | 初始化seed | 训练/开发生成seed | 测试生成seed |
| --- | --- | --- | --- |
| R1 | 101 | 2026101901 | 2026101951 |
| R2 | 211 | 2026101902 | 2026101952 |
| R3 | 307 | 2026101903 | 2026101953 |
| R4 | 401 | 2026101904 | 2026101954 |
| R5 | 503 | 2026101905 | 2026101955 |

各重复重新初始化模型并生成独立seed流的数据，没有用旧权重充当新重复。五个新重复之间及与v1–v4所有旧输入，按统一的完整变换轨道规则隔离。每块source_train8192/source_dev1024/target_train256/target_dev1024，测试2048；同一块各目标、背景、条件共享相应输入，目标标签由定义确定。source采样/dropout seed为初始化seed+1000，adapt为+2000；完整生成顺序与数据hash见seed_provenance、DATA_READY及test_data_audit。

模型保持4层Transformer，width256、8 heads、MLP4、dropout.1、绑定输入/output embedding、词表188，PermWorld one-line，n=10；参数3,240,448。每训练前向28 tokens，其中答案/EOS两位置监督，前缀不计loss。B/C各4800步，LR先100步warmup到3e−4，2400步余弦到3e−5后保持到4800。每适配条件沿用三档LR 3e−4/1e−4/3e−5，各800步，100步warmup和余弦到初值0.1。没有新增超参数搜索。

每100步全量评估train/dev，排除step0，依次按最高dev答案准确率、最低dev答案CE、最早step、固定LR顺序选择。50个唯一模型选择LR计数：{'0.0003': 28, '0.0001': 5, '3e-05': 17}。适配严格加载源参数、将目标token的绑定embedding/output行重置为同块step0值、使用全新AdamW状态。AdamW wd=.01、betas=.9/.999、eps=1e−8、clip1；BF16训练、FP32权重/评估、TF32关闭。

先冻结全部50项开发选择，再生成五个新测试池；300条全部LR的best/endpoint测试结果保留为诊断，不据测试重选模型。主分析只使用50条selected；训练终点对照保留于[selected_vs_endpoint.csv](artifacts/selected_vs_endpoint.csv)。[最终独立CPU审计](artifacts/FINAL_AUDIT.json)已通过：1,801个历史文件字节不变，25个输入池独立重放，425个checkpoint、参数加载、目标行重置、优化器计数、50项开发选择及全部配对统计核验通过。核心审计耗时63.35秒；它检查保存的证据和权重，没有再次运行神经网络推理。实际命令、文件职责与继续运行方式见[README](README.md)。

| 阶段 | run数 | 更新数 | 输入tokens | 监督tokens | 运行计时分钟 |
| --- | --- | --- | --- | --- | --- |
| pretrain | 40 | 192000 | 344064000 | 24576000 | 91.79 |
| adapt | 150 | 120000 | 215040000 | 15360000 | 28.66 |

新增训练计时合计120.45分钟；引擎各阶段墙钟合计122.79分钟。训练计时包括训练、全量train/dev评估、checkpoint保存和记账，不等同于纯GPU计算时间；二者不相加，末尾旧文件保留校验、独立审计与绘图另计。40个预训练run和150个适配候选全部新增，合计312000次更新、559104000输入tokens；没有延长预算，也没有新增单任务训练。checkpoint和配置保留，便于后续CKA接口使用。

## 下一阶段任务族扩展方案与收口

详见冻结在新迁移结果之前的[TASK_FAMILY_PLAN.md](TASK_FAMILY_PLAN.md)。优先复用现有循环分解、全局有序子序列、置换分块族，使已研究的局部比较和位置不等计数获得不同计算结构的参照；其次加入位移/成对次序与模式计数，参数实例不重复计成独立族。后续真正需要新增的族包括无向图连通/循环秩、含奇异矩阵的有限域rank/kernel、奇特征域二次型/双线性形式；这些新生成器本轮尚未实现或认证。各族先完成独立oracle核验与单任务开发，再冻结组合，保留不能学会的候选；不由两个预选案例外推任务族的普遍排序。

平台应把数学task_id、参数、编码、输入变换分开；单任务关系记录域条件、方向、输入作用与答案映射，联合关系用多输入/多源超边记录需要的源集合、查询变换、公式/证明与测试范围，另存观察关联及未检查标记。生成器复用base-instance→变换→标签oracle→统一split/orbit去重→编码，并存版本、seed、实际pool hash、采样slot与暴露量。关系导出的合法任务不自动称为泄漏，实例隔离和关系类别分别审计。

本轮只完成两个预指定对比的五次新重复。对效应方向、幅度不确定性和机制猜想分别报告，不追加这两个目标的背景变体、额外重复、训练预算或学习率，等待后续指导。

## 主要交付

- [逐重复配对差](artifacts/confirmation_paired_differences.csv)、[主/辅助汇总](artifacts/confirmation_summary.csv)、[探索与验证分表](artifacts/confirmation_vs_exploration.csv)。
- [50条唯一主结果](artifacts/results_selected.csv)、[300条全部候选best/endpoint](artifacts/results_all_candidates.csv)；每重复原始CSV：[101](artifacts/results_seed101.csv)、[211](artifacts/results_seed211.csv)、[307](artifacts/results_seed307.csv)、[401](artifacts/results_seed401.csv)、[503](artifacts/results_seed503.csv)。
- [源学习](artifacts/source_learning_table.csv)、[实际暴露](artifacts/source_exposures.csv)、[背景间同源变化](artifacts/source_background_changes.csv)、[源与目标交互并列](artifacts/source_interaction_context.csv)。
- [统计审计](artifacts/confirmation_statistics_audit.json)、[最终独立审计](artifacts/FINAL_AUDIT.json)、[数学核验](mathematics/MATH_AUDIT.json)、[冻结统计方案](stats_plan.json)。
