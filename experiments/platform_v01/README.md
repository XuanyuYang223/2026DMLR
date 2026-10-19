# PermWorld 数据平台 v0.1

本版本落实 [TASK_FAMILY_PLAN.md](../pilot_v5_confirmation/TASK_FAMILY_PLAN.md)，以统一任务接口、精确标签、版本化编码、可复现数据与有限单任务标定为交付。pilot_v1–v5 保持只读；v5 的未复现及不确定结果不改写。本轮没有两个旧目标的新背景变体、迁移组合、学习率搜索或预算延长。

源码在 [`permworld_platform/`](../../permworld_platform)，版本 `0.1.0`。从仓库根目录使用已有 `.venv`，无需安装新依赖或下载远端代码。当前是依托仓库内已归档 vendor 的研究平台，不声称已打包成独立 wheel。

## 任务数量与工程范围

| 重点任务族 | 规范模板 | 默认声明实例 | 固定长度非恒定实例 | 实际输入编码 |
| --- | ---: | ---: | ---: | ---: |
| 循环分解 | 6 | 8 | 8 | 1 |
| 全局子序列 | 5 | 9 | 7 | 1 |
| 排列块 | 5 | 7 | 7 | 1 |
| 合计 | **16** | **24** | **22** | **1 种共享编码** |

- 循环：`cycle_count`、`nontrivial_cycle_count`、`even_cycle_count`、`odd_cycle_count`、`cycle_length_count(k)`、`longest_cycle`。
- 子序列：`lis_length`、`lds_length`、`increasing_subsequence_count(k)`、`decreasing_subsequence_count(k)`、`rsk_shape`。
- 排列块：`components`、`global_descents`、`largest_component`、`smallest_component`、`component_length_count(k)`。

四个参数模板默认声明 k=1,2,3。计数 API 允许 k=1…30；额外参数值不会变成新模板。CPU 示例另外展示 k=4，不把该示例扩充成新的核心任务模板。保留的 shortest_cycle、cycle_type 等旧端点仍可调用，不混入六模板的循环族工程目标。

所有 **48 个旧端点**均保留，新增 **6 个注册端点**，合计 **54**。其中 cycle_length_count 是旧计数能力的参数化组织，不能把六个端点都叫全新数学能力。去掉两个参数别名及四个表示转换端点，目录含 **48 个规范数学模板**，这不是数学独立性计数。全目录默认声明 **62 个端点实例**（含别名、二元任务），默认数据物化 **56 个单输入非别名实例**。

24 个重点实例中，I₁=D₁=n，因此有 23 个相同输入函数类、22 个固定长度非恒定类。进一步按已登记的保持答案值的输入变换合并，为 20 类/19 个非恒定类；加入 components/global_descents 的加一映射后为 19/18。后一口径仅针对登记的关系，不能称为独立任务数。cycle_length_count(k=1,2,3) 与 fixed_points/two_cycle_count/three_cycle_count 重复；D₂ 等于旧逆序对数 length，I₂ 是其补值。详见 [TASK_COUNTS.json](registry/TASK_COUNTS.json) 与 [等价性审计](mathematics/EQUIVALENCE_AUDIT.json)。

上述22个非恒定实例针对本版n=8/10/12、k=1/2/3的标定/数据设置；小n或k>n还会产生额外退化，例如S₁只有一个输入。不能将计数外推为每个合法长度都非恒定，逐长度分布和边界条件另行记录。

## 统一接口与注册表

```python
from permworld_platform import label
from permworld_platform.objects import transform
from permworld_platform.registry import task_spec, task_output_bounds

p = (3, 1, 4, 2)
assert label('cycle_count', p) == 1
assert label('increasing_subsequence_count', p, {'k': 2}) == 3
assert label('largest_component', p) == 4
q = transform('inverse', p)
assert label('cycle_count', q) == 1
print(task_spec('component_length_count'))
print(task_output_bounds('increasing_subsequence_count', 30, {'k': 15}))
```

输入为一基置换，合法长度 1…30；right_multiply_simple 还要求 n≥2、index<n。函数拒绝重复值、布尔输入、隐式类型强转、缺失/额外参数与非法额外操作数。所有标签均为精确 Python 整数、布尔或整数结构；参数 k>n 的计数精确为零。

[TASK_REGISTRY.json](registry/TASK_REGISTRY.json) 和 [CSV](registry/TASK_REGISTRY.csv) 逐任务记录定义、参数、输入条件、精确算法、输出类型/范围、编码、族、复杂度、计时和关系证据。[TASK_INSTANCES.csv](registry/TASK_INSTANCES.csv) 单独列具体参数配置。[TASK_BENCHMARK.csv](registry/TASK_BENCHMARK.csv) 的计时包含参数/输入校验和标签生成，区别于数据集生成与训练耗时。

旧二元任务用 `operands={'q': q}`，含 compose、conjugate、commutator、bruhat_leq；输入与答案的完整序列编码可执行。本版未物化二元数据池，也未认证二元查询组的数据划分。不要把单输入池的隔离结论推广到二元任务。

## 明确版本化的模型接口

输入编码为 `one_line_v1`，输出协议为 **`permworld_sequence_v1`**。实际暴露一种输入编码；旧上游已有 cycle/Lehmer/inversion-vector 表示仍归档保留，本版未把它们伪计为已接入的编码变体。

词表从旧 188 项追加到 **203 项**，旧 ID 不变。小于 100 的整数仍为一个数值 token；大整数使用原有 `<NUM_START>` / base-100 数位 / `<NUM_END>`，结构答案使用新增的嵌套列表标记。例：n=30、k=15 的递增子序列计数最大值 155117520 能精确往返，答案含 EOS 共 8 token；不截断、不取模。rsk_shape 输出完整分拆。

训练在 prompt 后对所有答案 token 和 EOS 监督；评估从 prompt 独立自回归生成并比较完整答案，不能继续调用旧 pilot 的单答案 token 准确率函数。`encode_example` 返回 `prompt_ids`、`prompt_length`、`answer_ids`（含 EOS）、`input_ids` 及 `answer_value`。下一 token 监督从 logits 的 `prompt_length-1` 位置开始。

[MODEL_INTERFACE_COMPATIBILITY.csv](registry/MODEL_INTERFACE_COMPATIBILITY.csv) 列出输出范围和上下文支持。三个重点族在合法 n≤30 下的所列实例可用 128 token；旧 to_cycle、to_reduced_word、部分二元/参数化排列输出在大 n 时可能超长。它们仍可生成精确标签和完整序列；向固定 128 上下文读取时明确报错。需要另行配置模型上下文，禁止静默截断。新词表意味着旧 checkpoint 不能未经显式迁移就直接加载；本轮标定全部重新初始化。

## 固定数据与完整读取示例

规范数据在 [dataset/manifest.json](dataset/manifest.json)。两个数据/初始化单元为 2026102001/617、2026102002/829。每个单元各有 n=8,10,12；每长度 train=1024、dev=256、test=256。合计 18 个池、9,216 个排列、516,096 个物化标签。

生成器先生成对象，再调用统一标签函数。每个池的 RNG 为 `PCG64(SeedSequence([data_seed,role_index,n]))`。按固定种子、train/dev/test、长度顺序做均匀置换提议，拒绝与历史 v1–v5、其他角色及重复单元的完整 inverse/reverse/complement 轨道相交。提议数、拒绝数、输入内容哈希、文件哈希、实例清单均写入 manifest。均匀提议加顺序轨道排除不等于均匀轨道采样，也使不同单元存在有限总体依赖；本版不声称严格 IID 输入池。

实际核验见 [quality_full.json](dataset/quality_full.json)：随机流重放、全部标签复算、重复输入及跨角色/单元/历史轨道检查通过。[answer_distributions.csv](dataset/answer_distributions.csv) 保留逐长度答案直方图、观察常数标记及训练集多数基线。常数实例及集中分布不删除；小样本观察到常数不能替代数学上的恒定证明。

CPU 端到端示例已实际运行：

```powershell
.venv/Scripts/python.exe experiments/platform_v01/examples/demo.py
.venv/Scripts/python.exe -m permworld_platform sample --count 2
```

第一条生成一个单独的小型示例池，物化标签并读回模型样本；示例显式关闭历史排除，仅用于接口演示，未进入标定。第二条读取规范数据。Python 中可使用：

```python
from permworld_platform.data import read_samples

sample = next(read_samples('experiments/platform_v01/dataset', 617, 'train',
                           'increasing_subsequence_count', {'k': 3}))
assert sample['input_ids'][sample['prompt_length']:] == sample['answer_ids']
```

`read_samples` 核对对象/标签文件哈希、逐行对象 ID 和精确答案。`load_objects` 是低层便利读取器，不单独执行完整认证。规范训练另外冻结词表、代码及数据；独立使用者应保存对应 vocabulary.json，并运行 `verify-data` 检查它，而不能让旧模型自动使用更新后的词表。

默认生成器物化全部 56 个单输入实例；自定义配置可用 `task_instances` 指定任务及参数。配置必须满足所有实例的长度条件；过小空间、超过提议上限或需要二元数据时明确报错，不悄悄重复样本。

## 关系与正确性

[RELATIONS.json](mathematics/RELATIONS.json) 含 45 条新登记关系和 29 条 v5 证书引用。单任务变换关系、联合查询关系、每个查询的输入编号和变换分别记录；安全 JSON 表达式执行答案映射，不使用 eval。公式证明、有限实现核验、经验关联和未检查项分开保存。没有登记关系表示尚未分类，不能解释为“无关”。

数学核验穷举 S₁…S₇，加 n=8/10/12 各 64 个固定随机输入，共 6,105 个 fixture：272,741 次独立标签检查及 445,581 次新关系逐点检查通过。使用并查集、子集枚举、RSK 增长图、集合切点作为独立算法。历史 v5 关系只核对证书和定义来源哈希，未重复枚举。详见 [数学说明](mathematics/README.md) 和 [CERTIFICATE.json](mathematics/CERTIFICATE.json)。

## 有限单任务标定

训练前固定两个代表/族：cycle_count、longest_cycle；lis_length、increasing_subsequence_count(k=3)；components、largest_component。分别覆盖计数/极值、最长存在性/定长数量、块数/最大块。k=3 避开 k=1 的常数与 k=2 的旧逆序对互补关系；任务选择不依赖本次结果。

每任务两个种子，共 12 run；每 run 2400 更新、batch64，每长度精确 800 更新；单档 LR3e-4、warmup100、余弦降到3e-5。沿用4层、宽256、8头的Transformer结构，BF16训练/FP32评估、TF32关闭。每200步保存全量 train/dev 曲线和按长度指标。固定终点是主要标定参照；开发集最佳 checkpoint 只作描述，按准确率、答案CE、最早step选定。没有新增学习率搜索。开发选择不是独立测试成绩。

每个种子共享初始权重和对象池，各任务新建 AdamW；样本与长度调度哈希记录用于核对。多 token 答案长度不同会改变实际 tokens 和 dropout 随机数消费，所以相同更新/样本预算不声称相同 tokens 或完全相同 dropout 掩码。

主数据的 test 仅为平台保留划分，未用于标定模型评估或代表选择。数据质量程序会核对 test 对象/标签；这与查看模型测试表现不同。数学 fixture 及 CPU 示例从未用于训练。

标定结果入口：[CALIBRATION_REPORT.md](calibration/CALIBRATION_REPORT.md)，原始逐种子 CSV、按长度 CSV、曲线和成本表均在同目录。多数基线根据各长度训练标签确定。数据分布表对并列类别采用规范 JSON 字符串顺序，标定采用最小数值；本次六个代表没有因此产生差异，少量非代表结构任务的合法平局选择不同，不能混用而声称模型成绩变化。

## 安全复现：新目录、同配置

现有规范文件属于完成证据，不要在原路径重新导出带计时的注册表或覆盖数学证书。读取或验证：

```powershell
.venv/Scripts/python.exe -m permworld_platform sample
.venv/Scripts/python.exe -m permworld_platform.calibration verify
.venv/Scripts/python.exe experiments/platform_v01/calibration/audit.py --forwards
```

从头生成同配置数据并重跑 12 项标定，使用新的目录：

```powershell
.venv/Scripts/python.exe -m permworld_platform generate --out experiments/platform_v01_reproduction/dataset
.venv/Scripts/python.exe -m permworld_platform verify-data --out experiments/platform_v01_reproduction/dataset
.venv/Scripts/python.exe -m permworld_platform.calibration smoke --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data
.venv/Scripts/python.exe -m permworld_platform.calibration freeze --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data
.venv/Scripts/python.exe -m permworld_platform.calibration train --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data
```

复现沿用本版本的规范注册表和数学证书；重新导出/核验应另指定新目录。数据池已有 manifest 时只验证相同配置，拒绝覆盖不同配置或未完成的池；训练仅复用完整且哈希匹配的已完成 run，未完成训练不承诺逐步断点自动恢复。checkpoint 保存在本地，按项目惯例不提交大体积 .pt；源码、配置、数据、CSV、图和审计提交 Git。

## 本轮边界

平台提供正确生成和读取接口，不保证所有任务在当前模型/预算下可学。两种子、三个训练内长度只作有限标定，不能当成跨长度外推或完整样本效率规律。保留困难任务、常数实例、未核验关系和编码/上下文限制。后续是否改训练协议或选组合，需要另行冻结方案；本版不自动追加迁移组合。
