# PermWorld v0.1：数学任务、关系与有限实现证书

本目录完成任务定义级核验，不运行训练，不读取校准开发结果来决定任务或公式。v5 的未复现及不确定结果保持原样；这里没有追加两个旧目标的背景变体。

机器入口是 [RELATIONS.json](RELATIONS.json)；核验入口是 [CERTIFICATE.json](CERTIFICATE.json)。新旧任务共用 `permworld_platform.label(task_id, p, params=...)`，置换采用 `p=(p₁,…,pₙ)`、位置和值均为 `1…n`。平台合法长度为 `1…30`，参数计数任务允许 `k=1…30`；当 `k>n` 时精确答案为零。

## 模板、实例和等价类分别计数

三个工程任务族合计 **16 个模板、24 个已声明参数实例**。下表的固定长度常数/非恒定计数针对默认参数 `k=1,2,3` 与本轮数据长度 **n=8,10,12**，不声称在所有合法长度上相同。本版本实际输入编码为 `one_line_v1` 一种，编码不增加数学任务数。新族包含对既有标签的整理和一般化，不能把 16 个模板都声称为新数学任务。

| 任务族 | 模板 | 参数实例 | 固定长度常数实例 |
| --- | --- | ---: | ---: |
| 循环分解 | cycle_count、nontrivial_cycle_count、even_cycle_count、odd_cycle_count、cycle_length_count(k)、longest_cycle | 8 | 0 |
| 全局子序列 | lis_length、lds_length、increasing_subsequence_count(k)、decreasing_subsequence_count(k)、rsk_shape | 9 | 2 |
| 排列块 | components、global_descents、largest_component、smallest_component、component_length_count(k) | 7 | 0 |

四个参数模板各声明 `k=1,2,3`；这些是实例，不是四倍的新模板。已有 `shortest_cycle` 继续保留并核验，但不为凑数加入六模板的循环族工程范围。旧 `two_cycle_count`、`three_cycle_count` 端点仍可调用，归入 `cycle_length_count(k)` 的已有参数别名。

[EQUIVALENCE_AUDIT.json](EQUIVALENCE_AUDIT.json) 明确区分以下口径：

- 24 个焦点实例中，`I₁(p)=D₁(p)=n`，两者是一个相同输入函数；因此在完整声明定义域上是 **23 个不同的相同输入函数**。针对本轮默认长度，每个长度下有 **22 个非恒定的函数类**。核验给其他 275 个实例对保存了答案不同的显式输入，排除完整定义域上相同输入函数的重复；唯一合并的二者有定义证明。
- 若按**已登记的保持答案值的输入变换关系**进一步合并 LIS/LDS、I₂/D₂、I₃/D₃，得到 **20 类，去掉长度常数后 19 类**。这描述该登记关系生成的商，不声称穷尽所有可能变换或答案映射。
- 如果还允许 `components(p)=global_descents(Cp)+1` 的可逆仿射答案映射，则上述口径进一步变为 **19 类，去掉常数后 18 类**。这不能混写成 18 个独立信息源；联合恒等式仍然存在。
- 与旧目录比较，`cycle_length_count(k=1,2,3)` 分别等于 fixed_points、two_cycle_count、three_cycle_count；`D₂=length`，其中旧 `length` 指逆序对数；`I₂=binom(n,2)−length`。这些不能算新增独立的数学能力。

小长度会有更多退化：例如 n=1 只有一个合法输入，所有任务都在该长度上恒定；k>n 的计数也恒为零。因而上述“22”不适用于所有 n=1…30 与所有运行时参数，更不能据此推断任意有限数据池一定观察到所有答案。逐长度的有限分布/常数观察见 [DEFINITION_DISTRIBUTIONS.csv](DEFINITION_DISTRIBUTIONS.csv) 及平台数据质量表；一般输出边界与非法参数约束由任务注册表单独记录。

## 精确数学定义

令不相交循环长度为 `ℓ₁,…,ℓ_c`。循环族的答案依次为 `c`、`Σ1[ℓᵢ>1]`、`Σ1[ℓᵢ偶数]`、`Σ1[ℓᵢ奇数]`、`Σ1[ℓᵢ=k]` 和 `max ℓᵢ`。包含长度为一的循环；最短循环为 `min ℓᵢ`。这些量从置换作用的轨道计算，不能用一行表示的连续 run 替代。

全局递增子序列是满足 `i₁<…<i_k` 且 `pᵢ₁<…<pᵢₖ` 的位置子集；递减定义反转值比较。`I_k,D_k` 是满足条件的位置子集数量，LIS/LDS 是存在这类子序列的最大长度。它们允许不相邻位置，区别于 pilot 的连续 run。`rsk_shape` 是逐值 RSK 行插入所得的分拆，完整输出行长序列，而非仅第一行或分拆的哈希。

直接和块采用唯一的最细不可分解分解。在每个满足 `{p₁,…,pⱼ}={1,…,j}` 的位置 `j` 切开（包括末端 `j=n`），相邻切点之差为块长度。components 为块数，largest/smallest_component 为块长度极值，component_length_count(k) 为长度 k 的块数。global_descents 单独计数所有内部切点 `j<n`，要求左侧每个值都大于右侧每个值；它不等于普通相邻下降数，也不等于直接和块数。

## 关系的三类记录

每个执行关系记录原输入编号 `p`、每个查询的任务/参数、依次施加的输入变换，以及安全的 JSON 答案表达式。`inverse`、`reverse`、`complement` 分别为逆置换、位置反序和补值 `n+1−pᵢ`；列表按左到右执行。单任务变换边与多查询联合超边分别标记；变换后输入在对称置换上可能与原输入数值相同，不要求查询参数一定不同。

| 类型 | 例子 | 所需查询 |
| --- | --- | --- |
| 单任务/输入变换 | `LIS(p)=LDS(Cp)`；`I_k(p)=D_k(Cp)` | 另一任务在补值输入上的一个答案 |
| 单任务/非恒等答案映射 | `components(p)=global_descents(Cp)+1` | 一个变换输入答案，再加一 |
| 同输入联合关系 | `cycle_count=even_cycle_count+odd_cycle_count` | 同一个 p 的两个任务答案 |
| 同输入完整参数族 | `Σ_k k·cycle_length_count(p,k)=n`；块长度同理 | `k=1…n` 全部答案，不能只给 k=1,2,3 就声称成立的重建条件已满足 |
| 跨输入联合关系 | `fixed_points(p)=n−deficiencies(p)−deficiencies(p⁻¹)` | 同一个 deficiencies 任务的两个输入查询，外加已知 n |
| 单向结构输出映射 | `LIS=shape[0]`，`LDS=len(shape)` | 完整 shape 可给两个标量，反向一般不成立 |

循环分区、奇偶性、长度直方图公式直接来自循环分解；子序列的补值/反序关系来自位置子集的双射，逆置换关系来自置换图点的转置。递增/递减二元组恰好分割所有位置对，因此 `I₂+D₂=binom(n,2)`；**不能把这个二项分割推广到 k≥3**，因为还有混合升降模式。

块与 global_descents 的关系有直接定义证明：直接和内部切点等价于所有左侧值小于所有右侧值；补值交换大小，且块数等于内部切点数加一。反演保留不变初始区间；反序再补值会倒转块的次序而保持长度多重集，所以所列块统计对这些变换不变。

RSK 第一行/第一列结论采用行插入定理，而不是从少量枚举概括。原论文给出了行插入、第一行最长递增子序列及相应的递减结论：[Schensted, *Longest Increasing and Decreasing Subsequences*, 1961](https://doi.org/10.4153/CJM-1961-015-3)。本平台另外用不同算法验证实际 shape 实现。关系表将这两条标为 `proved_rsk_insertion_theorem`，其他新恒等式为 `proved_from_definitions`；有限实现检查另列状态和范围。

证据类别保持分离：

- **证明/定理**：RELATIONS 中的公式、适用条件和证明；包括 v5 已证明关系的原证书引用。
- **有限范围核验**：NEW_RELATION_CHECKS 和 LABEL_CHECKS，验证实际实现与公式一致；不把有限通过当作普遍证明。
- **经验关联**：复用 v5 数学目录中记录的 v4 源学习共同变化；保留原数据来源，不加入恒等式清单，不据此作机制结论。
- **尚未检查**：任意未登记的联合重建、模型是否计算某公式、以及不同族是否独立等问题。没有登记关系只表示尚未分类，不能写成“无关”。

合法公式是任务关系；输入/数据角色违规是另一问题。平台按显式输入轨道规则隔离数据，不把“源函数数学上可帮助推导目标”自动标成泄漏。登记联合超边也不意味着训练提示已经提供了联合答案或配对变换查询。

## 实际完成的核验

最终规范证书使用固定 fixture seed `2026101907`：穷举 `S₁…S₇` 的 **5,913** 个输入，另取 `n=8,10,12` 各 **64** 个固定随机输入，共 **6,105** 个。计数任务逐个检查 `k=1…n+1`，包含不可能长度返回零的边界。

| 标签/关系 | 独立核验方法 |
| --- | --- |
| 循环统计 | 把边 i—pᵢ 当作无向图，用并查集求连通分量大小；不调用生产循环遍历 |
| LIS/LDS 与 I_k/D_k | 穷举所有非空位置子集；不调用生产 tails/动态规划算法 |
| rsk_shape | 置换矩阵增长图局部规则；不调用生产行插入/替换代码 |
| 排列块 | 逐切点检验集合相等及跨切点比较；不调用生产前缀最大值算法 |
| 新公式 | 对查询指定的实际标签与安全表达式重建答案逐个比较 |

结果：**17 个实现**（16 个焦点模板加保留的 shortest_cycle）、**272,741 次独立标签对照**、**45 条新登记关系的 445,581 次逐点核验**，全部通过。最终规范运行 CPU 约 **10.71 秒**。定义分布表保留按长度直方图、多数比例和常数标记；其中随机 fixture 是数学核验样本，不能充当正式训练或测试分布估计。

**29 条 v5 历史关系未重新穷举。** 只核对原关系清单、原证书、原 math_ops 实现以及所引用经验/未检查表的 SHA256；见 [HISTORICAL_REUSE.json](HISTORICAL_REUSE.json)。旧证书仍保留原 1,601,440 次独立标签检查的历史计数，不计入本次新增工作量。两个初版证书分别保留在子目录，规范入口始终为本目录 CERTIFICATE.json；初版与最终版标签结果相同，差别仅为历史来源绑定和查询字段澄清。

## 可执行接口与复现

```python
from permworld_platform.relations import list_relations, evaluate_relation

edge = next(r for r in list_relations()
            if r['relation_id'] == 'fixed_from_two_deficiency_queries')
result = evaluate_relation(edge, (2, 1, 3, 5, 4))
assert result['passed']
```

答案映射采用 allowlist 的 JSON AST（query/context、add/sub/mod、sum/len、直方图极值等）；没有 `eval` 或任意代码执行。部分继承的历史不等式/分布命题仅保留声明和证书，`answer_map=null`，执行接口会明确拒绝；它们没有被伪装成单点重建函数。

重新核验时指定**新的目录**，命令拒绝覆盖已有证书：

```powershell
.venv/Scripts/python.exe -m permworld_platform.certificates --out experiments/platform_v01_recheck/mathematics
```

本目录所有 fixture 仅用于标签和公式核验，从未传入模型。有限单任务标定、开发曲线、数据划分与编码支持范围见平台主报告；没有由本目录发起的迁移搜索。
