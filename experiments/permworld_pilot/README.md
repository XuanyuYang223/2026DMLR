# PermWorld data-centric pilot

这是独立的最小迁移试点；不修改或重新解释 alignment 结果。代码直接复用固定上游的任务函数、one-line 词表、CausalTransformer 与 TMLR 的 SDPA 等价实现。独立运行器负责配对任务采样、数据隔离和适配。

## 来源

- `https://github.com/XuanyuYang223/2026TMLR`，提交 `d9a113d20d34db3c21cf69cf546cff4a0ab4904a`。参考 `experiments/permworld_encoding/core.py`、`experiments/permworld_eight_relations_20261010/train.py`、`experiments/longrun_attention.py` 和 source_learning.csv。
- `https://github.com/XuanyuYang223/neurips`，提交 `74f0de2017115f06f11b7285366b66e7e8431d30`。参考 `training.py`、`property_fewshot.py` 与 `configs/property_task_geometry.toml`；vendor 中四个 Python 文件逐字复制自其 `src/neurips_permutations/`。
- `vendor/sdpa.py` 逐字复制自上述 TMLR 的 `experiments/longrun_attention.py`。没有重许可上游文件。检查的上游没有提供 LICENSE 文件。
- 用户提到的 `XuanyuYang223/2026neurips` 地址此次访问返回 repository not found；采用 TMLR 自身记录的 neurips 上游。

## 任务与冻结

固定 n=10、均匀随机排列。目标 `fixed_points` 为 #{i:p(i)=i}。候选在迁移前唯一确定：数学关系明确，TMLR one-line Transformer 的 n=10 单任务开发准确率为 1.0；本轮仍以独立单任务开发实验的最终准确率 ≥0.90 为门槛。开发训练8192、开发评估1024，完全独立于正式试点。失败则保留并停止正式阶段，不换任务搜索收益。

- A：随机初始化，直接适配目标。
- B：`anti_fixed_points`, `descents`, `peaks`, `left_to_right_maxima`。
- C：`successions`, `descents`, `peaks`, `left_to_right_maxima`。

`anti_fixed_points(p)=#{i:p(i)=n+1-i}`；`successions(p)=#{i:p(i+1)=p(i)+1}`；descents 计数相邻下降；peaks 计数严格内部峰；left_to_right_maxima 计数从左扫描的新纪录。上游全部使用1-based排列值。

已知关系为 `fixed_points(p)=anti_fixed_points(c(p))`，其中 `c(p)(i)=n+1-p(i)`。这是输入变换下的恒等式，不是同一输入上的相等标签。successions 不在上游已登记的 fixed_points 直接关系对中；没有断言它数学上无关。均匀排列下 anti_fixed_points 期望1，successions期望(n−1)/n=0.9，皆近似 Poisson(1)，但非完全同分布。记录经验直方图、熵、总变差以及源任务训练/开发表现；难度不预设相同。

## 数据与比较口径

每个种子独立生成：预训练8192个不同排列（B/C共用），源开发1024，目标训练1024、目标开发1024、最终测试2048。训练子集取固定随机生成顺序的前64/256/1024，严格嵌套。三个条件同种子使用完全一致的目标数据、任务采样槽位、随机初始模型和适配随机序列。跨种子数据也隔离。

所有池（含独立开发池）按 identity、reverse、complement、inverse 生成的完整至多8元素轨道统一排斥；独立审计逐项检查输入重复和跨池轨道交集。B/C共用预训练输入属于有意配对，不计为泄漏。没有加载旧 alignment 数据/权重，未声称穷尽排除旧项目所有历史输入。最终测试只在全部训练完成且方案冻结后用于一次性最终评估；不选任务、超参或checkpoint。

## 模型和预算

沿用宽256、4层、8头、MLP ratio4、dropout0.1、词表188、最大长度128、tied embeddings 的3,240,448参数因果Transformer。为压缩试点只取 n=10。输入 `<BOS> <SIZE> 10 <ONE_START> ... <ONE_END> <TASK> =`；训练附单数字答案和EOS，只计算这两个token的平均交叉熵。评估只给prompt，答案在全188词表argmax；EOS也记录。准确率主结果是答案token准确率。

B/C各2400更新，每批64、四任务各16条（逐批打乱），相同输入索引和槽位；总输入tokens严格一致。适配每个条件、每个样本量均800更新×64曝光；按不同样本数比较，允许重复使用同一训练样本。AdamW lr3e-4, betas(0.9,0.999), eps1e-8, weight_decay0.01，100步warmup，余弦衰减至0.1，梯度裁剪1，FP32参数、CUDA BF16、TF32关闭。相较原20k更新明显缩短；结果不保证所有源任务充分学会。

每次适配均新建优化器，不继承动量。所有任务token自始存在；固定目标task token对应的tied embedding/output行在A/B/C适配前均恢复为同一个seed的step0行。目标数值输出沿用共享语言模型输出头，这是预训练迁移的一部分。每个样本量均从step0或对应预训练endpoint独立开始，不串行继续训练。最终checkpoint固定，不按开发集挑选。

## 复现（PowerShell，仓库根目录）

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe numpy==2.5.4 matplotlib==3.11.2
.venv/Scripts/python.exe experiments/permworld_pilot/pilot.py prepare
.venv/Scripts/python.exe experiments/permworld_pilot/pilot.py smoke
.venv/Scripts/python.exe experiments/permworld_pilot/pilot.py develop
.venv/Scripts/python.exe experiments/permworld_pilot/pilot.py freeze
.venv/Scripts/python.exe experiments/permworld_pilot/pilot.py formal
.venv/Scripts/python.exe experiments/permworld_pilot/pilot.py results
```

完整环境见 requirements-lock.txt。重跑请在每条命令添加同一个新 `--out experiments/permworld_pilot/runs/replication`，避免覆盖已冻结数据。formal 可跳过已保存endpoint，但不支持从未保存的中途步数恢复；被中断的单次训练将重做，已完成结果保留。CUDA不支持时脚本可在CPU运行，但会慢很多。

也可以运行 `.venv/Scripts/python.exe -u experiments/permworld_pilot/run_all.py`，按上述顺序执行并跳过已完成阶段。正式结果完成后的审计命令为 `.venv/Scripts/python.exe experiments/permworld_pilot/check_engineering.py` 和 `.venv/Scripts/python.exe experiments/permworld_pilot/audit_results.py`；两者只使用保存产物和开发数据，不运行测试集forward。

smoke 穷举 n=2..7 检查6种独立标签定义及complement恒等式，检查因果无答案泄漏、参数更新、保存加载精确性、真实多任务训练→重置优化器→适配→prompt评估完整路径。它不作为学习成功门槛。

## 输出与CKA接口

`runs/pilot_v1/` 包含冻结协议/源码哈希、data_audit、原始npz、smoke和独立开发记录、step0/预训练/适配checkpoint、全部loss/dev曲线、source_learning.csv、replacement_matching.json、每种子CSV和准确率曲线。每个checkpoint的`model`是上游CausalTransformer的兼容state_dict，包含config和vocabulary的固定上游来源；可用原工具挂钩`<ONE_END>`位置（n=10时索引23）的block输出和final_norm，后续接入CKA。本轮不扫描几何指标。

最小特征提取示例（仅开发输入，16条）:

```powershell
.venv/Scripts/python.exe experiments/permworld_pilot/extract_features.py --checkpoint experiments/permworld_pilot/runs/pilot_v1/seed17/adapt_B_256.pt --data experiments/permworld_pilot/runs/pilot_v1/data/s17_target_dev.npz --output experiments/permworld_pilot/runs/pilot_v1/cka_interface_smoke.npz --limit 16
```
