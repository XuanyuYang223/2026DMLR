已有单任务参照核对

`candidates_existing.csv` 保留逐模型开发集证据和预算，明确区分单任务、四任务及适配记录。复现：

```powershell
.venv/Scripts/python.exe experiments/pilot_v3_multitarget/references/inventory_existing.py
```

本地 `reference` 检出内容没有 `.pt`、`.pth` 或 `.safetensors`。NeurIPS specialist 和 TMLR 的 one-line Transformer 在宽度256、4层、8头、188词表及大多数优化参数上兼容；NeurIPS 的 `math_ops.py`、`models.py`、`passage.py` 与 pilot vendored 文件逐字节一致。但这些历史模型在长度2..30或10..30、20,000更新、不同训练池上学习，不能作为本轮固定n10、4,800更新的配对单任务控制。

可直接引用的本地兼容单任务记录是 v1 fixed_points 开发模型（8192例、2400更新，开发准确率100%，多数基线34.67%）和 v2 A 条件（256例、800更新）。前者支持保留原目标，后者支持历史小样本参照。它们不能替代新的源任务单任务基线；本轮新数据也不应直接复用这些旧适配权重。

TMLR eight-relations 表中，n10、identity、validation、全词表回答准确率的候选证据：

| 任务 | 准确率 | 训练定义的多数基线 | 已知配对 |
|---|---:|---:|---|
| exceedances | 100% | 38.28% | deficiencies，inverse |
| left_to_right_minima | 100% | 33.98% | left_to_right_maxima，complement；right_to_left_minima，reverse |
| longest_decreasing_run | 100% | 52.73% | longest_increasing_run，complement |
| descents | 100% | 35.94% | recoils，inverse |
| recoils | 98.83% | 38.28% | descents，inverse |
| peaks | 98.83% | 51.95% | valleys，complement |
| valleys | 100% | 42.97% | peaks，complement |

以上每行仅256个开发例，来自单个初始化95100001；只能用于提出候选，必须经本轮 n10 固定配置开发门槛再冻结。优先推荐 exceedances、left_to_right_minima、longest_decreasing_run：分别涉及位置比较、记录、连续子串统计。采用 deficiencies、right_to_left_minima、longest_increasing_run 作为相关源，则覆盖 inverse、reverse、complement。这三个对称操作也并非统计独立的任务族。若希望保留局部峰谷任务，可用 valleys 替换其中一项，但仍应先检查本轮可学习性。

为保留原目标锚点并节省计算，`reusable_local_checkpoints.csv` 单列42个已实际在CPU加载核对的本地文件：两个种子的原始step0，四个B/C的4800步续训，以及A/B/C三学习率的best和endpoint。加载核对验证188×256 embedding、tied output数值以及保存的优化器步数；文件与数据SHA逐项记录。具体来源是 `experiments/pilot_v2_diagnostics/artifacts/extended/seed{17,42}_{B,C}_4800.pt` 和 `adapt/s{seed}_{condition}_p{budget}_lr{lr}/`。预训练前2400步由v1的源checkpoint提供，后2400步沿用优化器与RNG且学习率固定3e-5；适配为新的AdamW、800更新。**仅在原任务顺序、原数据和适配选择规则完全不变的fixed_points锚点上复用**。独立新测试仍应在本轮冻结后生成，之前查看过的v1/v2测试结果不能构成本轮独立确认。

历史结果不能给任务贴上固定难度标签：TMLR 另一个8k、lr1e-3、dropout0、长度10..12的开发设置中，Transformer descents平均准确率56.67%，recoils为99.87%；与原20k设置的顺序不同，多因素同时改变，不能推断某单一原因。

最小必要新增训练集合应为本轮最终 B/C 源任务的去重并集，外加尚未经过相同配置开发门槛的新增目标；不需要重训整个32性质库。所有候选共用固定训练和开发输入，保留 n10 学习曲线、训练与开发损失、全词表准确率及训练定义的多数基线。若单任务800或4800步也未学会，表述为“在本配置与预算下未充分学会”；若单任务能学会、组合下降，仍需说明单任务与四任务的每任务曝光差异，不把它直接等同于组合干扰的因果证明。更严格的每任务曝光匹配控制可以后续再做。
