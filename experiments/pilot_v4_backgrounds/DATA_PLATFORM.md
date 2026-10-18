# PermWorld 数据与关系平台

本目录复用当前任务库的精确定义，供固定规模背景组成实验及后续显式授权的实验使用。它不添加数学任务，不改写 v1/v2/v3，也不自动启动训练。本轮仅使用 PermWorld、one-line 编码和原 Transformer。其他 world 的条目仅作现有接口盘点。

## 定义与身份

`TASK_CATALOG.csv` 有57条记录：48个PermWorld注册端点、4个GraphWorld任务、4个MatrixGroupWorld任务和1个历史有限域线性函数模板。`scalar_pilot_eligible` 表示已有标量接口兼容性，`pilot_v4_roles` 表示本轮实际用途；两者不是同一概念。

每个条目记录数学定义、输入/输出范围、参数、编码、生成方法、已知关系、原始源码位置与关系图引用。以下区别不得折叠为独立任务数：

- n、循环长度k、幂指数和simple index是参数或输入，不是新增任务族。`two_cycle_count`/`three_cycle_count`来自同一 `cycle_length_count(p,k)`，k=1即fixed_points。
- one_line、cycle、Lehmer、inversion_vector是相同排列对象的输入表示；四个 `to_*` 注册端点是输出表示转换。
- 32个scalar property存在确切对应、同分布或联合恒等式；它们不是32个独立统计重复。
- 当前库没有独立ascents任务。double_ascents数递增的连续三元组，不能当作ascents。

数学输入为一行表示的排列 `p=(p_1,...,p_n)`，恰好包含1至n。正式实验固定n=10。标量答案使用已有数字token；one-line prefix长度为 `2n+4`（n=10时24），追加任务token、等号、答案、EOS后训练输入28 token，只有答案与EOS两个位置计入监督损失。

## 可调用接口

`catalog/data_platform.py` 不导入Torch、不读取checkpoint或实验结果，提供：

| 接口 | 契约 |
| --- | --- |
| `TASK_ORDER` / `labels(p,tasks)` | vendored `PROPERTY_FUNCTIONS`的固定32任务顺序；未知任务立即报错 |
| `encode_one_line(p)` / `decode_one_line(ids)` | 无损、规范化、n=1…30的已有token编码；检查边界、长度、分隔符和排列有效性 |
| `transform(p,action)` | 下表八种输入动作，不更改任务定义 |
| `orbit(p)` / `orbit_key(p)` | 完整八动作轨道及字典序最小代表；可同时用于精确重复和变换等价泄漏检查 |
| `sample_pool(n,count,seed,excluded_orbit_keys)` | Python `Random.sample`的均匀排列提议，拒绝已占用轨道，返回对象和明确的生成元数据；超过容量/拒绝上限报错 |
| `validate_pools({role:permutations})` | 独立展开全部轨道，检查池内重复与池间交集；共享同一个角色供多个任务/条件使用是有意设计 |
| `action_composition_table()` / `unit_checks()` | 八动作复合表、编码往返、群作用、确定性生成和拒绝泄漏检查 |

实际试验引擎为保持历史可复现性，可以继续使用原NumPy `Generator.permutation`提议流。本模块的 `sample_pool`明确是另一条Python RNG实现，**不声称同一个seed会产生相同输入**；生成器、版本、seed、输入hash与角色必须共同记录。两者都在均匀排列提议后进行轨道拒绝，不声称接受的轨道类别等概率。变量长度实验还须声明长度采样规则；本轮只有n=10。

## 输入变换、答案映射与证明范围

action的三个bit依次表示是否执行 **I、R、C**；从左到右按这个执行顺序组合，不能交换I与R/C的次序。I为反演、R为一行序列反序、C为补值 `C(p)_i=n+1−p_i`。

| action | 数学函数写法 |
| --- | --- |
| 000 | identity |
| 100 | I |
| 010 | R |
| 001 | C |
| 110 | R∘I |
| 101 | C∘I |
| 011 | C∘R |
| 111 | C∘R∘I |

`DIRECT_RELATION_DEFINITIONS.csv` 保存35条定义关系及证明草图。`RELATION_GRAPH.csv` 将这些关系及其反向关系，对输入动作复合和仿射答案映射复合取闭包，得到132条边（包括恒等和自不变边）。一般形式是：

```text
right(T_action(p)) = scale * left(p) + n_coefficient * n + constant
```

`proof_basis` 给出组成该边的直接关系路径。这个闭包只对**已声明关系**完备，绝不是所有数学关系的完备分类。原来的恒等答案映射之外，图包含 `descents(Cp)=n−1−descents(p)` 等非平凡映射与由此导出的复合动作关系。

`AGGREGATE_RELATIONS.csv` 单独保存联合公式，不能伪装成单源任务双射。例如：

```text
fixed_points(p) + exceedances(p) + deficiencies(p) = n
cycle_count(p) = fixed_points(p) + nontrivial_cycle_count(p)
cycle_count(p) = even_cycle_count(p) + odd_cycle_count(p)
peaks + valleys + double_ascents + double_descents = max(n-2,0)
adjacencies(p) = successions(p) + successions(Cp)
```

直接关系、复合关系的数学依据为定义上的计数集合双射、分割或可逆映射；有限检查验证当前实现。关系图的每条边在S1…S7全部5913个排列及额外2048个随机n=10排列上通过检查；这些检查不构成对任意新数学命题的自动证明。聚合公式沿用另一个固定随机验证流，范围和seed写在 `RELATION_AUDIT.json`。

`REPLACEMENT_RELATION_COUNTEREXAMPLES.csv` 保留v3已验证的六对候选（包括本轮两对），每对在八动作下都有源标量答案相同、变换后目标答案不同的两个n=7输入。这排除对应动作下仅依赖该源标量的确定性答案映射；不意味着数学上无关，也不排除其他变换、多源关系或算法联系。

这些I/R/C关系用于声明数学联系和隔离等价输入。**正式预训练不按关系构造配对输入，不自动施加变换增强，也不输入联合答案。** 每个原始排列独立按被抽到的任务标注；关系存在不等于训练过程显式看到该关系。

## 本轮背景组成

两目标固定为fixed_points与longest_decreasing_run；前者B/C候选槽位固定为anti_fixed_points/successions，后者固定为longest_increasing_run/left_to_right_maxima。候选在槽位0，三个公共背景在槽位1…3。两个目标从所有源组全局排除。

| target | g0_v3公共背景 | g1_recoils公共背景 | g2_deficiencies公共背景 |
| --- | --- | --- | --- |
| fixed_points | descents, peaks, left_to_right_maxima | recoils, peaks, left_to_right_maxima | descents, deficiencies, left_to_right_maxima |
| longest_decreasing_run | descents, peaks, successions | recoils, peaks, successions | descents, deficiencies, successions |

g1只将槽位1的descents换成其inverse counterpart recoils。两者在均匀排列下具有相同Eulerian答案分布，但one-line下的计算结构及已有学习速度不同。descents已经被移除，不能说这个操作“新增了同时存在的descents↔recoils源对”。

g2只将槽位2的peaks换成deficiencies；它把局部三位置极值计数换成位置不等计数，**也大幅改变答案分布**。源train经验TV在seed17/42/73分别约0.760/0.761/0.765；g1的TV约0.014/0.010/0.013。背景是事前指定的系统变化，不是匹配搜索。不能将g2效果归因于单一“局部/位置”因素。

既有单任务1200步开发准确率（seed17/42/73）：descents约99.90/99.71/99.80%，recoils约62.79/74.02/93.07%；peaks约94.82/96.09/96.78%，deficiencies约99.90/99.22/99.12%。4800步四任务都接近100%。终点相近不表示学习轨迹或多任务难度匹配。`SOURCE_REFERENCE_CURVES.csv` 和 `SOURCE_ANSWER_DISTRIBUTIONS.csv` 保留三个种子的完整证据，`BACKGROUND_MATCHING.csv`只量化差异，不用于事后挑背景。

`BACKGROUND_RELATION_GRAPH.csv` 将完整关系闭包限制到每个目标/背景/条件的实际节点，区分涉及目标和源内部的关系。`BACKGROUND_AGGREGATE_RELATIONS.csv` 列出聚合公式的已在场和缺失成员。例如fixed_points的g2含deficiencies，但缺少exceedances，因此 `fixed+exceed+def=n` 并不使目标由当前这一个背景源答案决定。它仍是需要披露的数学联系。

## 数据角色与冻结契约

数据对象、数学标签、编码、实验角色应分别记录：

1. 固定任务顺序和定义版本，构造/复用source_train、source_dev、target_train、target_dev；记录输入对象及label数组的hash。旧兼容池可复用，新角色必须对全部已知历史池作统一轨道排除。
2. 在单任务开发或背景选择前冻结候选集合、选择原则、模型与预算。已有单任务记录必须核对模型/编码/长度/数据/优化轨迹兼容性，不能把高低预算混成同一参照。
3. 适配条件共享目标输入、开发集和选择预算；优化器重置，目标专有参数同初始化。源模型恢复和目标行重置需另有训练审计，本模块不声称替代它们。
4. 在所有模型/LR/checkpoint选择冻结后生成独立测试；加入跨新测试种子的完整轨道排除。测试不用于选择背景、任务、超参或追加预算。
5. 保存数据角色、生成器、种子、n、任务顺序、对象hash、标签hash、轨道审计和phase时间。多数基线由训练标签定义，并按长度分别报告。本轮不能从单一n=10推断长度泛化。

未来增加任务时先登记数学身份和参数/编码归属，补充精确定义、输入/输出范围、生成及边界约定，再给出关系的变换/答案映射/证明范围和反例或单元检查。创建新版本目录并冻结后再训练；不要为扩任务静默改旧运行的任务顺序、词表或数据hash。

## 复现与保留

项目根目录执行以下CPU命令可重新生成目录与审计；它只写本v4目录，不改旧文件，不运行模型，也不生成正式v4测试输入：

该命令用于正式冻结之前的目录构建。完成冻结后不要原地重跑，因为审计时间和文件hash会改变；复现实验应使用实验README中的新输出目录命令，扩充目录则创建新的版本目录。已经封存的v4目录与审计文件应原样保留。

```powershell
.venv/Scripts/python.exe experiments/pilot_v4_backgrounds/catalog/build_catalog.py
```

`CATALOG_AUDIT.json` 保存脚本、vendored数学源、旧v3目录代码以及所有新目录/关系/参照表的hash。`SOURCE_REFERENCE_AUDIT.json`记录被读取的v3源train及单任务曲线hash，且显式说明未读取迁移结果。`unit_checks`包含规范编码往返、64种动作复合、轨道key不变、确定性小池生成、池间泄漏拒绝和非法编码拒绝。小n校验fixture与正式实验数据不同，不影响测试冻结。

本轮结束后停止扩展；本平台接口不构成继续新增领域、架构、任务数量或实验的授权。
