# 冻结后的定义级关系复核补充

本补充在 v4 组合、配置和原始目录冻结后、正式预训练期间完成。只读取任务定义、原关系图与冻结协议，不读取迁移结果、checkpoint或训练表现，不更改组合、数据、预算、选择规则或被冻结的目录。原目录明确只对已声明关系取闭包，本补充说明其未覆盖的联合关系、自对称和算法结构；不能将原图的无边解释为数学上无关。

## g2 中需要披露的多输入联合关系

记 `F=fixed_points`、`AF=anti_fixed_points`、`D=deficiencies`、`E=exceedances`。位置划分给出 `F(p)+D(p)+E(p)=n`。反演将每个 `p_i>i` 对应到逆排列的低于对角线位置，因此 `E(p)=D(I(p))`。两式合并：

```text
F(p) = n - D(p) - D(I(p)).
```

又因 `AF(p)=F(C(p))`，代入上式得到：

```text
AF(p) = n - D(C(p)) - D(I(C(p)))
      = n - D(C(p)) - D(R(I(p))).
```

I 表示反演，R 表示一行序列反序，C 表示补值。沿用冻结平台的 bit 顺序，先 I、再 R、再 C；因此 `100=I`、`001=C`、`110=R∘I=I∘C`、`111=C∘R∘I`。公式对所有 n≥1 的排列成立。

这意味着 fixed_points 的 g2 中，背景 deficiencies 与目标及 B 候选都存在可由两个变换输入联合使用的确切关系。原表“缺少 exceedances”仍正确描述同一输入上的已命名任务集合，却不能据此排除跨变换输入的联合重建。g2 的 C 同样含 deficiencies，因此不能解释为完全不含目标关系的控制组。

**一个输入上的单个 deficiencies 标签不能确定目标。** n=10 时，`p=(1,3,2,4,5,6,7,8,9,10)` 与 `q=(2,3,1,4,5,6,7,8,9,10)` 都有 D=1，但 F 分别为8和7。正式训练不提供上述配对查询、联合答案或关系约束；池内轨道去重也不会同时保留两个不同的变换等价输入（自身对称的排列可能使变换输入与原输入相同）。这是一项隐含数学联系，不是“单标签泄漏”或已证明模型利用该联系。

## 六组背景怎样改变计算结构

两个目标的 B 候选在全部背景中保留 complement/reverse 的恒等答案关系。g1 将 descents 移除并换成 recoils；不能称为添加一对共存的 descents/recoils 源任务。其总体答案分布相同，但 one-line 下求邻接位置下降与求相邻值的逆位置下降是不同计算。既有单任务早期学习差异已在冻结 `BACKGROUND_MATCHING.csv` 中记录。

对 LDR 目标，令 `s_i=1[p_i>p_(i+1)]`、`d=sum(s_i)`。descents 是负向比较总数；LDR 是最长连续1段长度加1；LIR 是最长连续0段长度加1；peaks 是局部从0转1的次数。这些任务共享同一个局部比较序列，但不是由一个标量答案互相确定。将序列按上升边或下降边切开，分别有 n−d 个下降run和 d+1 个上升run，由最长段至少为平均段长得到：

```text
ceil(n/(n-d)) <= LDR(p) <= d+1
ceil(n/(d+1)) <= LIR(p) <= n-d.
```

g1 移除了这种同输入 descents 统计，仍保留 peaks；g2 保留 descents，却移除 peaks，换入位置不等计数。因此两种背景更改对 LDR 的局部计算结构影响不同，不能只用关系图边数概括。

successions 是局部上升边的子集，所以 `successions(p)+descents(p)<=n−1`。又因 successions 在反演下不变，`successions(p)+recoils(p)<=n−1` 也成立。此公共源联系在 g0/g2 与 g1 间有对应形式；替换不等于删除全部局部结构联系。对 fixed_points，g2 的位置比较结构及上述多输入联合恒等式尤其需要披露。

## 原关系图未列出的三个自对称

以下关系可直接由定义证明，并经过同一 CPU 校验；没有原地扩写冻结的 `RELATION_GRAPH.csv`：

| 关系 | 定义上的理由 |
| --- | --- |
| `peaks(R(p))=peaks(p)` | 反序交换峰的左右邻点，中心仍大于两邻点 |
| `successions(C(R(p)))=successions(p)` | 反序与补值合用，把每个相邻 `(a,a+1)` 再映为同向差1相邻对 |
| `deficiencies(C(R(I(p))))=deficiencies(p)` | RC把低于对角线与高于对角线互换，再用反演的 D/E 双射 |

这些边主要补全当前节点的自对称，不能当作新增独立任务或新增训练证据。它们也表明“已声明关系的完整闭包”与“所有数学关系的完整分类”是不同命题。

## 可支持的解释和限制

本设计可检验：在两个事后选定的探索性目标、三种冻结背景和有限配对块上，B−C 的差值是否随一个公共任务替换而改变；结合源任务曲线，可描述哪些公共任务共同变好或变差。

它不能单独识别“数学关系本身”的因果效应，也不能从相关性证明梯度冲突、表征共享或模型使用了某条恒等式。g1 保持总体边缘分布，但学习速度和计算结构不同；g2 同时改变局部/位置结构、答案分布、学习动态及 fixed_points 的多输入关系。两个目标共享 complement 类型的候选关系、数据/初始化块和大量源任务，不是独立抽取的任务族样本。无论结果正负，都不能将此轮当作样本效率曲线或任务数量 scaling 证据。

后续任务库宜优先标注关系类别而非直接扩大任务数：可逆输入变换加恒等/仿射答案映射；同分布但缺少当前直接变换对应；多源/多输入联合恒等式；单向可恢复关系与不等式；共享局部计算结构。已有 cycle_count/奇偶/非平凡cycle、位置划分和局部run等族可提供候选，但新增目标、配对输入干预或训练均需另行冻结并获后续授权。本补充不触发任何新实验。

## 校验与复现

审计穷举 S1…S7 共5913个排列，并用固定 seed=2026101806 生成2048个 n=10 排列作为定义校验 fixture；不是正式实验测试输入，也不写入实验输入池。对9条恒等式或上下界各检查7961个输入，合计71649项，结果见 `RELATION_ADDENDUM_AUDIT.json`。一般性依据是上述定义证明，有限检查用于验证实现。

```powershell
.venv/Scripts/python.exe -B experiments/pilot_v4_backgrounds/catalog_addenda/audit_postfreeze_relations.py
# 已有审计结果不可覆盖；复查时给一个新文件：
.venv/Scripts/python.exe -B experiments/pilot_v4_backgrounds/catalog_addenda/audit_postfreeze_relations.py --out experiments/pilot_v4_backgrounds/catalog_addenda/RECHECK.json
```

审计记录原 catalog 全部文件、TASK_CATALOG、DATA_PLATFORM、design 与 FROZEN_PROTOCOL 的前后 SHA256，断言这些文件未改；另记录本补充代码/说明及原数学实现hash。脚本不导入 Torch，不加载模型，不读迁移结果。
