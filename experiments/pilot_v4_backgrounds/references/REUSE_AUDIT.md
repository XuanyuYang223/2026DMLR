本轮只使用 `fixed_points` 和 `longest_decreasing_run`，三个配对模型种子17、42、73；只读盘点 v3 的训练/开发记录与 checkpoint。脚本不导入训练引擎，不运行模型，不读取任何新旧测试输入、预测或准确率。所有 checkpoint 以 `map_location='cpu'` 加载。

```powershell
.venv/Scripts/python.exe experiments/pilot_v4_backgrounds/references/reuse_inventory.py
```

精确复用方案

| 内容 | 可复用数量 | 条件 |
|---|---:|---|
| A 的三学习率候选 | 18，形成6个选择组 | 相同模型初始化、目标、256例训练、开发集、采样及选择规则；各背景共享同一个A |
| g0 的 B/C 预训练 | 12 | 逐项保持源任务有序tuple、源训练池、模型初始状态、采样及4800步学习率轨迹 |
| g0 的 B/C 适配候选 | 36 | 相同目标及源状态、目标行重置、数据、800步预算和三学习率候选 |
| g1/g2 预训练 | 无现成精确匹配；新训24 | g1只将公共slot1的descents改为recoils；g2只将公共slot2的peaks改为deficiencies |
| g1/g2 适配候选 | 新训72 | 每个新源状态下重新适配，优化器状态重置 |
| 已有单任务参照 | 24 | 八种实际使用的源任务×三个种子；可引用1200和4800步记录，注明曝光/优化步数差异 |

总共126个唯一适配候选、42个唯一开发集选择组。按三个背景展示A/B/C时有54个逻辑结果行，其中A重复展示不是新增运行或独立观察。

g0 有序组合如下；两个干预只改变指定公共slot，slot0的相关/替换任务不动。

| 目标 | B | C |
|---|---|---|
| fixed_points | anti_fixed_points, descents, peaks, left_to_right_maxima | successions, descents, peaks, left_to_right_maxima |
| longest_decreasing_run | longest_increasing_run, descents, peaks, successions | left_to_right_maxima, descents, peaks, successions |

完整 v3 有24个源训练记录、21个不同的有序组成×种子单位、18个无序组成×种子单位。v3 的 `left_to_right_minima/C` 与 `longest_decreasing_run/B` 源任务有序列表相同，同种子实际参数哈希相同，二者可作为源 checkpoint 的别名。不能跨目标复用已经适配的权重。`fixed_points/C` 与 `longest_decreasing_run/C` 仅无序集合相同，slot0和slot3交换会改变每个输入上的任务标签分配，因此不是精确复用。

种子与数据溯源

| 随机量 | seed17 | seed42 | seed73 |
|---|---:|---:|---:|
| 模型参数初始化 | 17 | 42 | 73 |
| 训练/开发输入生成流 | 20261018 | 20261018 | 2026101803 |
| 源任务采样及Torch/CUDA dropout | 1017 | 1042 | 1073 |
| 目标适配采样及Torch/CUDA dropout | 2017 | 2042 | 2073 |

模型种子不是数据生成种子。v1用一个连续的NumPy生成器20261018，依次生成development_train、development_dev，然后seed17的pretrain/source_dev/target_train/target_dev/test，再生成seed42的对应池。v3复用seed17/42相应输入并重算32列标签；目标训练取原1024池的前256例。seed73由v3的2026101803流依次生成source_train8192、source_dev1024、target_train256、target_dev1024，排除已有完整inverse/reverse/complement轨道。所有目标及条件共享同一种子的输入池。

`seed(s)`同时设定Python random、NumPy全局、Torch CPU及所有CUDA随机种子；训练再创建独立的NumPy `default_rng(s+1000)` 或 `default_rng(s+2000)` 用于批采样。所有条件和学习率候选使用相同采样/dropout种子。每步64个不重复索引，源任务slot按四项各16个打乱。更换任务有序列表会改变slot与任务标签的绑定；不要只按集合匹配。

配置与重置

- 模型为原256宽度、4层、8头、FFN倍率4、dropout0.1、上下文128、188词表、tied embedding/output的Transformer；one-line，固定n10。
- 源训练4800更新：前100步warmup至3e-4，余下至2400步cosine降到3e-5，再保持3e-5到4800。不是4800步重新拉长的cosine调度。
- 适配每候选800更新，学习率3e-4、1e-4、3e-5；warmup100后cosine至原学习率的0.1。每100步评估完整训练256和开发1024，排除step0后按开发准确率最高、开发answer CE最低、较早checkpoint、预声明LR顺序选择。
- 适配只恢复目标query token对应的既有embedding行；output与embedding共享参数，因此该output行同步恢复。没有新head；数字答案行和其他源权重转移。AdamW重新建立，状态项为0，终点所有优化器step为800。
- 注意mask语义：attention为严格因果与valid-token mask；loss只在answer和EOS两处计算各0.5权重，分类始终覆盖188词表，没有仅数字类别mask。测试回答只读prompt，EOS单独用gold answer条件化评估。

`reuse_inventory.py` 会从实际源参数和初始目标行重新计算适配初始模型hash，验证优化器计数、采样随机流终态、tied行数值、模型形状和参数有限性，并从开发曲线重算选择。任何改变输入、seed、目标、任务slot顺序、三学习率网格或选择规则都会破坏相应层次的精确复用前提。

可供引擎依赖的原v3元数据schema

`multi/s{seed}_{target}_{B|C}/completed.json`：

- `seed`, `tasks`（有序list）, `kind='multi'`, `updates=4800`, `input_tokens=8601600`, `supervised_tokens=614400`, `sample_exposures=307200`, `task_exposures=76800`, `task_fraction=.25`。
- `checkpoint`、`checkpoint_sha256`、`model_state_hash`、`initial_state_hash`、`schedule_sha256`、`train_sha256`、`dev_sha256`，以及`actual_unique_per_task`。
- `checkpoint`相对 **v3 artifacts** 解析，不相对该completion目录。fixed17/42真实文件指向v2；`new_compute`描述当时v3是否新算，不是v4计算成本。
- `curve.csv`在completion相邻目录。新v3源有0/600/1200/2400/3600/4800全训练/开发测量；历史fixed17/42源的早期曲线缺失，只有2400和续训点，不能补造。

`adapt/s{seed}_{target}_{A|B|C}_lr{lr:.0e}/completed.json`：

- `seed`, `target`, `condition`, `lr`, `best`、`endpoint`（含step、完整train/dev accuracy/CE/count），`updates=800`及预算。
- `best_path`, `endpoint_path`, `best_sha256`, `endpoint_sha256`, `final_model_state_hash`, `schedule_sha256`。路径同样相对 **v3 artifacts**，必须跟随，不能假定权重位于completion目录。
- `v3_train_sha256`、`v3_dev_sha256`统一绑定当前32列标签的v3数据；建议v4使用这两个顶层字段，不直接比较嵌套legacy字段。
- `initialization`中`source_state_hash`、`loaded_state_hash`、`adapt_initial_state_hash`、`target_row_hash`、`reset_row_ids`、`optimizer_entries=0`、`optimizer_state_reset=true`、`new_output_head=false`可核对实际加载与重置。
- **legacy例外**：fixed_points seed17/42复用记录的`initialization.source_path`相对旧的`experiments/`目录，其他新v3记录则相对v3 artifacts；其嵌套train/dev哈希绑定旧6列标签npz，且`train_slice=[0,256]`。使用实际原来源路径验证，再以顶层v3数据hash判断新实验输入一致性。
- best.pt仅含model、config、seed、step、lr、condition和metrics，通常没有optimizer；endpoint.pt另有optimizer、NumPy/Torch/CUDA RNG。继承自v2的checkpoint内部可能缺少`target`，目标应由已核对的v3completion及任务token绑定。

v4应建立自己的逻辑run记录并指向旧文件，保留`reused_from`、来源completion SHA、权重SHA、曲线SHA和新逻辑background，而非改写旧checkpoint或旧记录。新测试需在v4配置冻结后独立生成，排除v1/v2/v3所有旧输入轨道；本审计没有读取这些测试内容。
