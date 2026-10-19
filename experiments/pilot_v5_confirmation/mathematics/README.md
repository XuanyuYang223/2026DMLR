# 当前背景的数学关系与联合可推导性

本核对只读取实际数学实现、v4 冻结组合、旧关系目录及旧源任务学习表，不读取 v5 新训练数据、checkpoint 或迁移结果。它在验证实验前完成，不改变已指定的两个验证对比。旧文件的 SHA256 前后相同。

**fixed_points 的 deficiencies 背景在 B、C 中都具有严格的跨变换输入联合关系；不能把 C 叫作不含目标数学关系的控制组。** 另一方面，当前全部 12 组 v4 组合都不能仅由同一输入上的四个源标量答案确定目标。本轮为每组给出了实际 n=10 的反例。这两个结论并不矛盾，区别在于允许查询哪些输入。

## 1. 定义、符号与问题边界

定义来源为 [vendored math_ops.py](../../permworld_pilot/vendor/neurips_permutations/math_ops.py)，与 [v4 任务目录](../../pilot_v4_backgrounds/TASK_CATALOG.csv)逐项对应。置换为一行形式 p=(p₁,…,pₙ)，值域及位置均为 1…n。

| 记号 | 实际定义 |
| --- | --- |
| F | fixed_points = Σᵢ1[pᵢ=i] |
| AF | anti_fixed_points = Σᵢ1[pᵢ=n+1−i] |
| E | exceedances = Σᵢ1[pᵢ>i] |
| D | deficiencies = Σᵢ1[pᵢ<i] |
| d | descents = Σᵢ₌₁ⁿ⁻¹1[pᵢ>pᵢ₊₁] |
| recoils | d(p⁻¹)，即相邻值在逆位置序列中的下降数 |
| LIR / LDR | 最长连续严格递增 / 递减区间的长度；不是最长子序列 |
| peaks | 内部位置 i 满足 pᵢ₋₁<pᵢ>pᵢ₊₁ 的个数，不包括端点 |
| successions | 相邻差 pᵢ₊₁−pᵢ=1 的个数，不含差为 −1 的项 |
| left_to_right_maxima | 左至右严格新最大值个数，包括第一个位置 |

I 为反演，R 为反序，C 为补值 pᵢ↦n+1−pᵢ。位串沿用 v4：依次应用 I、R、C，所以 100=I，001=C，110=R∘I=I∘C，111=C∘R∘I。

本文件的“答案可推导”特指：**已知 n，仅从所列源任务的标量答案，是否存在对所有 p∈Sₙ 都正确的确定性答案映射**。这与“已知完整原始输入 p 能否计算目标”不同；原始 p 本来就确定全部数学标签。这里也不把神经网络掌握源任务等同于它获得了一个精确的外部查询器。

分别审计两种查询模型：

1. 同输入查询：给四个源任务在同一个 p 上的答案，共四个数。
2. 八变换查询：给四个源任务在 I/R/C 生成的八个变换输入上的答案，按动作和任务顺序组织成 32 个数。这是严格限定的假想查询条件，不是本轮实际训练提示格式。

## 2. 明确公式与证明

完整机器清单为 [RELATION_INVENTORY.csv](audit/RELATION_INVENTORY.csv)：28 条恒等式/上下界，外加 1 条带有分布假设的等分布命题。以下列出与当前组合最相关的部分。

**位置划分与多输入重建。** 每个位置恰好满足等于、高于、低于对角线之一，因此对所有 n≥1：

```text
F(p)+E(p)+D(p)=n.
```

令 j=pᵢ。条件 pᵢ>i 等价于 p⁻¹ⱼ<j；这是逐个图点的双射，所以 E(p)=D(Ip)。代入即得：

```text
F(p)=n−D(p)−D(Ip).
AF(p)=F(Cp)=n−D(Cp)−D(ICp).
```

由此可区分三个问题：若同一输入上同时有 E、D 的标签，F 可由两源直接计算；当前组只有 D 没有 E，不能照搬这种同输入结论；若允许同一 D 任务在 p 与 Ip 的两个精确查询，则 F 又可以计算。g2 的 B/C 均含 D，故均满足第三种条件。

**B 候选的直接变换关系。** F(p)=AF(Cp)=AF(Rp)，由对角线/反对角线的映射直接得到；LDR(p)=LIR(Cp)=LIR(Rp)，由连续区间的一一对应得到。因此六个 B 组合都在八变换查询条件下可恢复目标。这些关系在各背景间没有消失。

**局部比较结构。** 设 sᵢ=1[pᵢ>pᵢ₊₁]。d 是 s 中 1 的总数；LDR 是最长连续 1 段加一；LIR 是最长连续 0 段加一；peaks 是 0→1 的转变次数；valleys 是 1→0 的转变次数。因此：

```text
peaks+valleys+double_ascents+double_descents=max(n−2,0)
peaks−valleys=s_last−s_first      （n=1 时右端定义为0）
ceil(n/(n−d)) ≤ LDR ≤ d+1
ceil(n/(d+1)) ≤ LIR ≤ n−d.
```

前一式按相邻比较位的四种状态分割内部位置；第二式是转变数的望远镜求和，但需要额外的两端比较位；两个界由最大 run 至少为平均 run 长度，以及可用的下降/上升边数得到。**这些共享计算结构与上下界不意味着相应标量标签互相确定。** v5 的 LDR 对比将 d 换成 recoils，移除的是同输入的 d；不能说新增了一对同时在场的 descents/recoils。

**successions 与公共源。** 每个 succession 都是一个 ascent，因此 successions(p)+d(p)≤n−1。将置换图的边 (i,a)→(i+1,a+1) 转置，可得 successions(Ip)=successions(p)；把前一个界应用于 Ip，得到 successions(p)+recoils(p)≤n−1。所以替换 descents 不等于删除所有涉及 successions 的数学联系。另有 adjacencies(p)=successions(p)+successions(Cp)，它是向上/向下相邻差为一的分割。

**其他已核对恒等式。** cycle_count=F+nontrivial_cycle_count=even_cycle_count+odd_cycle_count，以及 parity=(n−cycle_count) mod 2，分别来自循环的分类和每个 k-cycle 的符号。目前没有任何组合包含这些恢复 F 所需的 cycle 源对，故不能用这些恒等式说当前组合已有相应同输入恢复途径。peaks(Rp)=peaks(p)、successions(CRp)=successions(p)、D(CRIp)=D(p) 等自对称也已独立核对。

descents 与 recoils 在**全部 Sₙ 上的均匀分布**下等分布，因为 I 是保均匀测度的双射；这不保证经过轨道排除的有限训练池直方图完全相同，更不保证学习速度相同。

## 3. 每个背景的联合可推导性

源任务按 slot0 到 slot3 排序。FP 的候选 B/C 为 anti_fixed_points/successions，LDR 的候选 B/C 为 longest_increasing_run/left_to_right_maxima。

| 目标 | 背景公共任务 | 同输入四标量（B / C） | 八变换查询 B | 八变换查询 C | 本轮验证 |
| --- | --- | --- | --- | --- | --- |
| fixed_points | g0: descents, peaks, left_to_right_maxima | 均不充分，有 n=10 反例 | AF(Cp) | 不充分，有 n=10 反例 | 是 |
| fixed_points | g1: recoils, peaks, left_to_right_maxima | 均不充分，有 n=10 反例 | AF(Cp) | 不充分，有 n=10 反例 | 否，仅旧探索审计 |
| fixed_points | g2: descents, deficiencies, left_to_right_maxima | 均不充分，有 n=10 反例 | AF(Cp)；亦可 n−D(p)−D(Ip) | n−D(p)−D(Ip) | 是 |
| longest_decreasing_run | g0: descents, peaks, successions | 均不充分，有 n=10 反例 | LIR(Cp) | 不充分，有 n=10 反例 | 是 |
| longest_decreasing_run | g1: recoils, peaks, successions | 均不充分，有 n=10 反例 | LIR(Cp) | 不充分，有 n=10 反例 | 是 |
| longest_decreasing_run | g2: descents, deficiencies, successions | 均不充分，有 n=10 反例 | LIR(Cp) | 不充分，有 n=10 反例 | 否，仅旧探索审计 |

逐条件记录见 [TARGET_JOINT_DERIVABILITY.csv](audit/TARGET_JOINT_DERIVABILITY.csv)。所有断言“不充分”均有显式有限反例，不是从未找到公式推断不存在。对另外的输入变换、更多原始输入特征或任意算法关系，本次未作穷尽分类。

例如 n=10 的两个排列：

```text
p = (4,10,5,7,8,2,9,6,3,1)
q = (9,4,3,7,2,8,5,6,10,1)
```

FP g2 B 的 (AF,d,D,LTRmax) 都是 (2,5,4,2)，但 F(p)=0、F(q)=1；g2 C 的 (successions,d,D,LTRmax) 都是 (1,5,4,2)，故也不能区分。这里 F 不同、D 相同意味着 E 必然不同，也说明为什么只检查同一个 D 标签不够。

对六个 B 与 FP g2 C 之外的五个 C 组合，甚至将四个源任务在八变换输入上的全部 32 个标签给出，也存在目标不同的 n=10 配对。17 条完整反例（12 条同输入、5 条八变换）保留在 [COUNTEREXAMPLES.csv](audit/COUNTEREXAMPLES.csv)，逐行由独立实现与原 vendor 双重核对。这只排除这两种指定答案查询模型下的普适映射，不等于数学上无关。

## 4. 公式、经验关联与未检查解释分开保存

| 证据类别 | 本轮材料 | 可以说什么 |
| --- | --- | --- |
| 有定义级证明 | RELATION_INVENTORY、显式反例及证明 | 给出允许查询条件下的恢复公式、界或不充分性；没有模型机制结论 |
| 仅由旧数据观察 | [OBSERVED_ASSOCIATIONS.csv](audit/OBSERVED_ASSOCIATIONS.csv) | v4 的 g1−g0 中 peaks 的 12 个配对源开发变化均为负；LDR-B 的 LIR 三块也均下降；这是有限运行的共同变化 |
| 尚未检查 | [UNCHECKED_CLAIMS.csv](audit/UNCHECKED_CLAIMS.csv) | 模型是否实际计算 F=n−D−DI、是否存在梯度冲突或中间表征中介、其他长度上是否复现、未列出的联合关系 |

这里只重算旧源学习表中的 15 条差值，不对其与目标迁移拟合相关系数或解释因果；v5 结果必须另由冻结的验证统计汇总。证明一条任务恒等式、观察源准确率共同改变、确认背景配对交互复现，是三种不同证据。

## 5. 数学关系不等于数据泄漏

当前预训练逐个提示指定一个源任务，不提供目标答案、四源联合答案或配对变换查询。源训练池可在不同时刻用不同源任务标注同一底层输入，这是多任务训练的设计；输入轨道隔离禁止同一输入或 I/R/C 等价输入跨训练、开发和测试角色复用。池内也只保留每个轨道的一个代表，未显式配对提供 p 与 Ip。自对称排列可能满足 Ip=p，不改变上述原则。

合法任务关系是研究对象的一部分。只有违反冻结角色隔离、混入目标监督/测试标签、依据测试选择模型或组合等问题才应另据证据标记协议违规。不能因为一个源函数在数学上可以帮助算目标，就自动标为泄漏。反过来，本轮保证输入隔离也不能证明模型未利用任何数学关系。

## 6. 核验范围与复现

[audit_relations.py](audit_relations.py) 不调用 Torch、模型或训练。它从实际 vendor 加载标签函数，另写一个参考实现：逆排列按图点排序、run 通过穷举连续区间、循环通过独立遍历；参考实现不调用 vendor 标签函数。

实际完成：穷举 S₁…S₇ 共 5913 个输入；另用 Python Random seed=2026101811 生成 4096 个 n=10 定义核验 fixture。20 个定义在每个输入的八变换上比较，共 **1,601,440 次标签一致性核验**；28 条公式/界各检查 10,009 个输入；descents/recoils 直方图在每个穷举 Sₙ 上完全相同。全部通过，CPU 约 5.14 秒。有限检查验证实现，普遍性依赖所列证明，而不是由随机检验推出。

fixture 只用于数学实现核对，不传给模型，不作为正式测试，不用于决定目标、组合或超参数。全部审计输出拒绝覆盖；若复核请给新目录：

```powershell
.venv/Scripts/python.exe -B experiments/pilot_v5_confirmation/mathematics/audit_relations.py --out experiments/pilot_v5_confirmation_recheck/mathematics_audit
```

冻结标记 [MATH_AUDIT.json](MATH_AUDIT.json) 绑定本说明、脚本、审计表和下一阶段方案；详细旧文件 hash 和核验计数见 [MATHEMATICS_AUDIT.json](audit/MATHEMATICS_AUDIT.json)。本目录完成后不追加这两个目标的背景变体。
