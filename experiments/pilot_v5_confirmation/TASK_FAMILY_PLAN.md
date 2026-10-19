# 下一阶段任务族与可复用生成器方案

本文件只给出后续方案，不新增训练、目标、背景或领域实验。两个验证案例和所有验证选择先冻结，不根据此方案自动追加。方案依据现有 [57 行任务目录](../pilot_v4_backgrounds/TASK_CATALOG.csv) 和本轮定义审计制定，不依赖 v5 新迁移结果。

## 优先增加结构差异，不以 token 个数充当任务族数量

当前目录含 PermWorld 的 48 个端点，其中 32 个是现有 scalar pilot 注册端点；其余包括编码输出、需要参数/第二输入、输出结构不同的任务。端点、独立数学任务、参数化任务族和编码变体须分别计数。one-line/cycle/Lehmer 是表示变体，不能按新数学任务计数；two_cycle_count/three_cycle_count 是 cycle_length_count 的 k=2/k=3 参数实例；固定 n、不同随机 seed、任务 token 名称也不新增任务族。

下一阶段宜先加入当前库中尚未覆盖的**计算结构族**，使已研究的“局部相邻比较/位置不等计数”有真正不同的参照；每个目标保留固定四源预算与少量事前组合，先不扩大源任务数量。下表明确区分复用已有实现和需要新增实现。

| 优先级 | 任务族与候选 | 现有状态 | 与本轮的区别 | 关键关系及进入实验前的工作 |
| --- | --- | --- | --- | --- |
| 1 | 循环分解：cycle_count、nontrivial_cycle_count、even/odd_cycle_count、cycle_length_count(k) | 现有 scalar 实现；k 变体归为一个参数族 | 需跟随置换作用遍历，区别于 one-line 邻域或对角线逐点比较 | C=F+NT、C=Even+Odd；parity=(n−C) mod2 为单向信息映射。先完成匹配预算的单任务可学习性开发，既有 cycle_count 学习不足不能忽略 |
| 1 | 全局有序子序列与 shape：lis_length、lds_length、rsk_shape | 现有实现；shape 是结构输出，不能直接混入现有单 token 协议 | 全局子序列/插入结构，区别于连续 run | LIS(p)=LDS(Cp)，shape 的第一行/列给 LIS/LDS。先复用标量 LIS/LDS；结构输出另建 token/loss 合约；一般关系须给实现证书 |
| 1 | 置换分块：global_descents、components | 现有 scalar 实现 | 前缀与后缀的全局分隔，区别于相邻下降 | components(Cp)=global_descents(p)+1；含非恒等答案映射。先检查集中答案分布与多数基线，避免多数类别掩盖学习不足 |
| 2 | 位移统计：max_displacement、displacement_one_count | 现有 scalar 实现 | 同一位移向量的极值与计数，提供共享中间结构但不等价的统计 | 反演下不变；二者不预设一一映射。为单输入/多输入联合关系保存反例，不将“没登记公式”写成无关 |
| 2 | inversion/全局成对次序：length、parity；可新增 major_index=Σᵢ i·1[pᵢ>pᵢ₊₁] | length/parity 现有，major_index 尚未实现 | 二位置比较总量与位置加权邻接统计；答案支持更宽 | parity=length mod2；length(Cp)=n(n−1)/2−length(p)；major_index(Cp)=n(n−1)/2−major_index(p)。新端点需独立核验；n=10 支持仍在既有数值 token 范围，但 n 扩大时需重新检查 |
| 2 | 模式结构：已有 pattern_avoidance，补 pattern_count(p,pattern) 与固定窗口模式计数 | avoidance 已有，计数端点需新增；不同 pattern 是参数，不当多个独立族 | 子序列模式与连续窗口模式应分开，能构造存在性、计数与联合分割 | avoidance=1[count=0] 是非可逆映射；固定 k 的全部相对次序模式计数之和等于 C(n,k)，窗口模式之和等于 max(n−k+1,0)。标注这些联合关系需要全部哪些成员；检查标签平衡采样改变了何种输入分布 |

优先级不代表立即选定新的实验目标；先做定义、生成器和预设单任务开发，再冻结目标。若某族在兼容预算下不能学会，保留失败，按既定门槛报告，不根据迁移测试替换它。

## 与当前目录有区别的后续新任务族

当前目录已有 GraphWorld 的出入度极值、MatrixGroupWorld 的四个行列/XOR Hamming 统计，以及 FiniteFieldLegacy 的线性形式模板。这些不能被当作“新发现的领域”。未来若需要验证跨计算结构的外推，可在单独阶段加入以下新增任务族；不能混入本轮两案例的结论。

| 新任务族 | 与现有目录的差别 | 可审计的关系例子 | 生成器/标签约束 |
| --- | --- | --- | --- |
| 图连通与循环秩 | 当前 GraphWorld 只有有向简单图的入/出度极值；新增无向图 components、edges、cycle_rank 等拓扑统计 | 无向图 β₁=m−n+c，因此已知边数和连通分量可联合确定循环秩；connected=1[c=1] 为单向映射 | 新增明确的无向、无自环对象类型，边提议分布和密度参数单独记录；不能复用有向图标签而省略语义变化。小 n 全图枚举与独立 BFS/并查集双实现 |
| 有限域矩阵 rank/kernel | 当前 MatrixGroupWorld 采样 GL(n,F₂)，rank 恒为 n，不适合作 rank 学习任务；需要包含奇异矩阵的新分布 | rank(A)+nullity(A)=列数；|ker A|=q^nullity(A)；rank(A)=rank(Aᵀ)；矩阵作用或多列信息可形成联合问题 | 用 F_q 上任意 m×n 矩阵，精确模运算；记录尺寸和秩分层提议是否改变自然分布。kernel-size 输出过大时使用 log_q 值并登记为输出映射，不能静默截断 |
| 二次型及双线性形式 | 现有 FiniteFieldLegacy 为线性形式；二次型需要乘法交叉项，计算结构不同 | 奇特征域上 Q(x+y)−Q(x)−Q(y)=B(x,y)，其中 Q(x)=xᵀMx，B(x,y)=xᵀ(M+Mᵀ)y；联合关系同时涉及多输入和新函数 | 先限定奇素数域，避免把特征2边界混入；矩阵/系数作为参数或世界参数需事先固定语义。小域穷举；x,y,x+y 必须按关系参与的对象组统一划分，防止拆分后间接重用 |

这些新增任务族的公式可从相应定义推导，但**本轮没有实现或认证这些新生成器**。进入后续训练前，须为实际实现添加机器证书、边界约定、单任务开发和新冻结协议。对已有域扩充数学族，与真正新增域，成本和解释范围应分别统计。

## 关系登记应为带输入查询的超边

仅用“任务 A 与任务 B 有边”不够表达本轮 F=n−D−DI 的情况。建议每条关系记录：

| 字段 | 内容 |
| --- | --- |
| relation_id / version | 稳定编号及定义版本；旧证书保留 |
| evidence_status | proved、empirically_observed、counterexample、unchecked；不能把空字段解释为无关 |
| domain / conditions | 对象空间、n/参数范围、长度/端点约定、分布假设 |
| queries | task_id、参数、所需原始输入编号及变换；同任务不同变换仍是两次查询 |
| answer_map | 可执行的答案表达式、是否使用 n/其他原始特征、精确/不等式/概率关系 |
| direction / invertibility | 单向或双向；是否信息丢失；所需成员最小集合 |
| proof / certificate | 定义证明、实现路径/hash、小规模穷举、独立实现、反例 |
| query_model | 同输入多源、跨输入同源、跨输入多源；最大允许查询集合 |
| observations | 经验相关使用的数据版本和分布、样本数、不确定性；与公式单独存表 |
| training_exposure | 实际是否联合展示答案/配对输入，不能从关系存在推断模型看到了它 |

例如 fixed_points 的重建应登记为同任务的两查询超边，而不是误写成 D 单标签足够：

```json
{
  "relation_id": "fixed_from_two_deficiency_queries",
  "evidence_status": "proved",
  "domain": "S_n, n>=1",
  "queries": [
    {"task_id": "deficiencies", "input_id": "p", "transform": "identity"},
    {"task_id": "deficiencies", "input_id": "p", "transform": "inverse"}
  ],
  "target": {"task_id": "fixed_points", "input_id": "p"},
  "answer_map": "n - y0 - y1",
  "query_model": "multiple_transformed_inputs",
  "actually_jointly_presented_in_v5_training": false
}
```

组合分析另维护每组的 task set、有序 slot、每条超边是否满足查询条件、源内部关系、源到目标关系及未检查项。两个无序 task set 一样而 slot 分配不同的组不能自动视为相同训练。反例必须明确反驳的是哪个查询模型和定义域，不能扩写为“数学上无关”。

## 可复用生成器组织

建议新平台版本按职责拆分，沿用已有 CPU 平台的接口思想，保持旧目录只读：

```text
platform_v_next/
  objects/             permutation, directed_graph, undirected_graph, finite_field_matrix
  tasks/               versioned pure label functions and parameter schemas
  relations/           unary edges, joint-query hyperedges, proofs, counterexamples
  encodings/           object/answer serialization separate from mathematical identity
  generators/          proposal distributions and explicit rejection/stratification policies
  splits/              canonical input IDs, relation-aware equivalence/group assignment
  certificates/        exhaustive fixtures, independent implementations, boundary cases
  manifests/           immutable task order, vocabulary, object/label hashes, RNG streams
```

纯函数接口可统一为 `label(task_spec, object)`、`transform(action_spec, object)`、`encode(encoding_spec, object)`；生成器只产生对象，标签由同一个已版本化 registry 计算。构造平衡标签的提议策略要登记在 sampling policy 中，不藏在标签函数内。参数化任务需要 typed parameter schema；任务数学身份与输入参数、编码、世界初始化 seed 分开。

`generate_objects(spec, data_seed, role, exclusion_manifest)` 返回对象、输入 hash、提议/接受数、长度/参数直方图；`build_labels(objects, task_registry_version)` 单独返回标签 hash。记录 data/model/sampler/dropout/test RNG seed，避免以一个 block 名字替代所有 seed。模型可对所有条件共享对象、slot 输入流和初始化，从而保留严格配对。

划分规则在抽样前声明：单输入任务按对象及登记变换的等价类隔离；涉及多个输入的联合关系需按整个查询组分配角色，不能把 x 与 x+y 的监督关系随意拆到训练和测试后再声称独立。并非每条合法恒等式都要求全宇宙关系闭包去重——先明确目标泛化问题和隔离策略，再保存可审核的碰撞证据，避免使可用输入空间坍缩或把数学关系本身当成泄漏。

每个新增族在训练前应有：精确定义与范围、非法输入拒绝、编码往返、穷举/独立标签核验、关系证明与反例、固定划分统计、单任务学习曲线/按长度多数基线、冻结的组合与选择预算。单任务与多任务参照须标注每任务暴露、总更新和学习率历史的不同。预设少量目标后，测试只做冻结后的统一评价；没有“最佳测试背景”的自动回搜。

## 下一阶段的收口原则

优先覆盖“局部比较—全局子序列—循环遍历—全局分块”的差异，再决定是否值得加入图拓扑或有限域代数族。关系类型至少分层为：可逆输入变换/恒等答案、非恒等答案映射、同分布但计算不同、同输入联合恒等式、跨输入联合恒等式、单向信息映射，以及共享结构但指定标签查询不足。每层保留分布与学习难度匹配情况，不靠任务名判断。

验证结果若只支持一个案例，应报告其条件性，不自动加两个现有目标的更多背景；若两者复现，也只是这两个预选案例的验证。下一阶段应另行冻结跨任务族的目标样本，而不是继续围绕它们寻找更大的效果。本文件不触发任何新实验。
