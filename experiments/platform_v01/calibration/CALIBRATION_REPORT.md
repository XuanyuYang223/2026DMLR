# PermWorld v0.1 单任务标定

本轮预先固定六个任务、两个种子和统一 2400 次更新；不进行迁移组合搜索，也不按表现删除任务。终点是主要固定预算参照，开发集最佳 checkpoint 仅作描述性结果；平台测试集没有用于标定。

模型沿用 4 层、宽度 256、8 个注意力头的 Transformer 结构。为显式版本化的新任务与结构化答案接口，词表从历史 188 扩展到 203，参数量为 3,244,288；它不是与历史 checkpoint 完全相同的参数空间。本轮每个种子从同一份新初始化开始，六个任务分别建立空状态 AdamW，没有加载旧迁移权重。

训练使用长度 8、10、12，各长度 1024 个训练输入、256 个开发输入；每三次更新按固定种子打乱长度顺序，因此每长度恰好 800 次更新。所有任务共用每个重复的数据与模型初始权重。三个长度都出现在训练中，按长度结果不构成未见长度的外推测试。

准确率要求从提示独立生成全部答案及 EOS 完全正确；交叉熵使用 teacher forcing，两者不同。输出编码已显式版本化，可容纳 increasing_subsequence_count(k=3) 超过原单 token 范围的答案。这六个代表均输出标量；平台支持结构化答案的正确编码，但本轮没有标定所有结构化输出任务的可学习性。

| 任务 | 终点训练准确率均值 | 终点开发准确率均值 | 两种子开发范围 | 开发多数基线 | 超多数基线 |
|---|---:|---:|---:|---:|---:|
| cycle_count | 73.88% | 42.84% | 41.93–43.75% | 32.68% | +10.16 pp |
| longest_cycle | 57.41% | 22.53% | 20.57–24.48% | 16.99% | +5.53 pp |
| lis_length | 97.62% | 86.26% | 85.03–87.50% | 41.80% | +44.47 pp |
| increasing_subsequence_count{"k": 3} | 46.61% | 25.26% | 24.48–26.04% | 4.36% | +20.90 pp |
| components | 99.97% | 99.74% | 99.74–99.74% | 77.08% | +22.66 pp |
| largest_component | 99.93% | 99.02% | 98.96–99.09% | 77.08% | +21.94 pp |

循环任务的困难没有被总体均值掩盖：cycle_count 在长度 12 的平均开发准确率只有 35.74%，低于该长度多数基线 36.72%。longest_cycle 的平均训练准确率为 57.41%，开发准确率为 22.53%，同时存在未充分拟合与泛化落差；这两者不能归结为单一机制。

increasing_subsequence_count(k=3) 的开发准确率为 25.26%，虽然比多数基线高 20.90 个百分点，训练准确率也只有 46.61%，仍保留为当前配置下未充分学会的任务。相比之下，两个排列块代表的开发准确率都接近 99%，且明确高于约 77% 的多数基线。上述是有限预算下的开发观察，不是对整个任务族的保证。

多数基线按任务及长度只从训练集选择，再应用于开发集；高原始准确率可能源于答案不平衡，需同时读取超多数基线差值和按长度曲线。仅两个种子，开发集参与 checkpoint 选择，不能据此估计稳定迁移能力或形成测试性能结论。未学会的任务全部保留，预算不足与编码/优化难度仍不能分离。本轮的输入长度、训练样本量及编码接口已有显式变化，准确率不能与 v5 直接作为控制对照。

共 12 次训练，28,800 次更新，1,843,200 次样本暴露，52,841,514 个实际输入 tokens；右侧 padding 计算量另列。累计训练计时 14.23 分钟，包含完整训练/开发评估和保存。

- [逐种子与按长度结果](results_by_seed.csv)
- [训练/开发完整曲线](learning_curves.csv)
- [每个输入实际暴露量](input_exposures.csv)
- [按长度预算](length_exposures.csv)
- [运行成本](compute.csv)
- [按长度图](figures/endpoint_by_length.png)
- [完整开发前向与 checkpoint 审计](CALIBRATION_AUDIT.json)

复现命令从仓库根目录运行，先按[平台复现说明](../README.md)在 `experiments/platform_v01_reproduction/dataset` 生成并核验新数据目录。下面所有输出写入新目录，freeze 绑定其中的数据及当时的代码哈希。已有完整运行按哈希检查后跳过；不完整运行从相同初始模型重新开始，不承诺逐更新断点恢复。归档目录的 SMOKE 也属于冻结证据，重跑 smoke 应使用新目录。

```powershell
.venv/Scripts/python.exe -m permworld_platform.calibration smoke --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data
.venv/Scripts/python.exe -m permworld_platform.calibration freeze --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data
.venv/Scripts/python.exe -m permworld_platform.calibration train --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data
.venv/Scripts/python.exe experiments/platform_v01/calibration/summarize.py --out experiments/platform_v01_reproduction/calibration
.venv/Scripts/python.exe experiments/platform_v01/calibration/audit.py --forwards --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data
```
