# 现有任务目录与数学关系审计

`../TASK_CATALOG.csv` 只登记当前工作树已经存在的任务定义，没有新建任务或读取迁移结果。它有 57 行：PermWorld 48 个注册端点、GraphWorld 4 个任务、MatrixGroupWorld 4 个任务，以及旧有限域线性函数的一个参数化模板。后面三个领域仅供盘点，本轮训练仍仅使用 PermWorld。

48 个 PermWorld 注册端点并不等于 48 个独立数学任务族。`to_cycle`、`to_lehmer`、`to_inversion_vector`、`to_reduced_word` 是同一排列对象的输出编码转换；`two_cycle_count` 与 `three_cycle_count` 是 `cycle_length_count(p,k)` 的两个参数实例，且 k=1 等于 fixed_points。32 个 scalar property 任务也存在确切对应与加和依赖，不能当作 32 个独立统计复制。目录中的 `identity_kind`、`mathematical_family`、`parameter_variants` 和 `input_encoding_variants` 用于保留这些区分。不同 n、随机种子、架构和输入编码都不另算数学任务。

上游共有 one_line、cycle、lehmer、inversion_vector 四种输入表示；2026TMLR 的匹配长度编码模块使用 one_line 和 canonical_cycle。它们是同一数学对象的表示变体。本轮保持 one_line。数学范围按 n≥1 记录；部分区间是外界而非完整可达支持，历史生成器以及本轮固定 n=10 会进一步限制输入。v3 通用生成器对 pattern_avoidance 与 bruhat_leq 有专门平衡构造，不能把所有历史任务都说成统一随机排列标注。

`RELATION_AUDIT.csv` 包含原注册的 8 个跨任务关系、由现有定义直接得到的其他 9 个关系、13 个 inverse 自不变关系。脚本对 S1 至 S7 全部 5,913 个排列和额外 2,048 个随机 n=10 排列检查了这些关系，并检查了 7 组加和或编码恒等式。有限穷举是实现验证；恒等式的数学依据来自定义。所有公式均采用 `right(transform(p)) = left(p) + offset` 的方向。

四目标候选及其关系：

| 目标 | B 中指定相关任务 | 输入变换 | 答案映射 | 统计内容 |
|---|---|---|---|---|
| fixed_points | anti_fixed_points | C(p)_i=n+1−p_i | 恒等 | 位置等号计数 |
| exceedances | deficiencies | I(p)=p⁻¹ | 恒等 | 位置不等计数 |
| left_to_right_minima | right_to_left_minima | R(p)_i=p_(n+1−i) | 恒等 | 单侧记录计数 |
| longest_decreasing_run | longest_increasing_run | C | 恒等 | 最长连续单调段 |

这些覆盖三种动作 C、I、R 和四种具体统计结构；fixed_points 与 exceedances 仍同属宽泛 positional 家族，不宣称是四个独立数学任务族。已有库没有独立 ascents 注册任务；double_ascents 指严格递增连续三元组，不能混为 ascents。循环类任务有 `cycle_count=fixed_points+nontrivial_cycle_count=even_cycle_count+odd_cycle_count`，但这需要多个答案，不能伪称单源任务的恒等映射关系。

`REPLACEMENT_RELATION_COUNTEREXAMPLES.csv` 对本轮提出的六对替换候选逐一提供反例：fixed_points/successions、exceedances/recoils、left_to_right_minima/{cycle_count,longest_increasing_run}、longest_decreasing_run/{left_to_right_maxima,cycle_count}。在 n=7 的完整排列集合，对由 I、R、C 生成的八种输入动作，每个候选均存在两个源答案相同、变换后目标答案不同的输入。因此这些动作下不存在仅依赖该源标量答案的确定性映射。这不排除其他数学关系、联合任务关系或更复杂输入变换，也不意味着 C 数学上无关。特别是 exceedances 与 recoils 同为 Eulerian 边缘分布，但同分布不构成指定 inverse 的点对点恒等式。

脚本不导入 Torch，不运行模型，不读取任何神经网络评估或测试集。它导入 v1 已使用的 vendored 数学定义；`CATALOG_AUDIT.json` 保留源文件哈希、计数与运行耗时，并验证与当前上游快照可对照。现有工作树未发现 AGENTS.md。可复现命令（项目根目录）：

```powershell
.venv/Scripts/python.exe experiments/pilot_v3_multitarget/catalog/build_catalog.py
```

任务可学习性、最终四目标和替换任务的选择，须另外依据单任务训练与开发集记录冻结；本目录只证明定义及所声明关系。
