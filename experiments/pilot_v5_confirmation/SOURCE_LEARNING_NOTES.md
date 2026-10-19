# v5 源任务学习记录：在验证测试之前的描述

本说明只读 `source_curves.csv`、`source_exposures.csv`、冻结config和预训练完成标记。没有导入模型、没有新增训练/forward，没有读取任何新目标测试结果。分析时TEST_COMPLETE存在：False。**尚未将这些源变化与新验证交互联系起来**；下面只描述在固定4800步预算下能直接看到的源学习现象。

40个新预训练run全部完成，160条任务×run记录均为n=10、每任务76800次暴露。每个源run均按原配置完成4800步，源训练不足不触发延长。结果由各任务的8192条train和1024条dev计算；多数类别由该任务训练池决定，并在相应dev上评估。学习程度须结合任务自己的基线，而不是跨任务直接比较准确率。

## 候选源的全部五重复配对差

| 目标 | seed | B dev原→新 | B dev变化pp | C dev原→新 | C dev变化pp |
| --- | --- | --- | --- | --- | --- |
| fixed_points | 101 | 99.41%→98.93% | -0.49 | 49.32%→47.07% | -2.25 |
| fixed_points | 211 | 99.22%→97.66% | -1.56 | 43.95%→56.05% | +12.11 |
| fixed_points | 307 | 98.05%→99.32% | +1.27 | 46.97%→44.82% | -2.15 |
| fixed_points | 401 | 99.41%→98.24% | -1.17 | 41.80%→45.31% | +3.52 |
| fixed_points | 503 | 99.32%→98.34% | -0.98 | 51.27%→51.46% | +0.20 |
| longest_decreasing_run | 101 | 82.52%→59.47% | -23.05 | 96.58%→98.05% | +1.46 |
| longest_decreasing_run | 211 | 68.36%→53.22% | -15.14 | 98.44%→97.66% | -0.78 |
| longest_decreasing_run | 307 | 56.54%→54.30% | -2.25 | 98.93%→93.16% | -5.76 |
| longest_decreasing_run | 401 | 66.50%→55.96% | -10.55 | 97.85%→96.58% | -1.27 |
| longest_decreasing_run | 503 | 96.29%→62.11% | -34.18 | 96.88%→95.90% | -0.98 |

FP相关候选anti_fixed_points的dev均值由99.08%变为98.50%，平均变化-0.59pp；五次中4次降低、1次提高。两个背景下均接近99%，因此不能把FP背景差泛化解释为“原背景完全没学会相关候选”。这一观察尚不说明目标迁移如何变化。

LDR相关候选longest_increasing_run的dev均值由74.04%变为57.01%，五次均下降：-23.05, -15.14, -2.25, -10.55, -34.18pp。新背景dev多数基线均值为54.00%，平均超基线仅+3.01pp，其中0/5次低于各自基线，seed211恰好等于自己的基线。其新背景train均值也仅57.02%，所以这种源学习不足并非只表现为训练集学会后的开发集落差；优化速度与表示限制仍不能据此区分。原背景dev从56.54%到96.29%，仍有明显初始化/数据重复差异。这支持“recoils背景伴随LIR源学习下降”的有限重复描述，尚不证明源下降导致目标收益消失。

## 未换公共任务及不一致现象

| 目标 | seed | 条件 | 未换公共源 | dev背景差pp |
| --- | --- | --- | --- | --- |
| fixed_points | 101 | B | descents | -9.67 |
| fixed_points | 101 | B | left_to_right_maxima | +3.42 |
| fixed_points | 101 | C | descents | -21.88 |
| fixed_points | 101 | C | left_to_right_maxima | +4.49 |
| fixed_points | 211 | B | descents | -7.42 |
| fixed_points | 211 | B | left_to_right_maxima | +0.39 |
| fixed_points | 211 | C | descents | +11.13 |
| fixed_points | 211 | C | left_to_right_maxima | +3.03 |
| fixed_points | 307 | B | descents | -1.95 |
| fixed_points | 307 | B | left_to_right_maxima | +0.49 |
| fixed_points | 307 | C | descents | -4.39 |
| fixed_points | 307 | C | left_to_right_maxima | +1.27 |
| fixed_points | 401 | B | descents | +0.59 |
| fixed_points | 401 | B | left_to_right_maxima | +1.17 |
| fixed_points | 401 | C | descents | +7.91 |
| fixed_points | 401 | C | left_to_right_maxima | -0.10 |
| fixed_points | 503 | B | descents | +2.83 |
| fixed_points | 503 | B | left_to_right_maxima | +4.39 |
| fixed_points | 503 | C | descents | +8.30 |
| fixed_points | 503 | C | left_to_right_maxima | +5.08 |
| longest_decreasing_run | 101 | B | peaks | -17.48 |
| longest_decreasing_run | 101 | B | successions | -33.11 |
| longest_decreasing_run | 101 | C | peaks | -4.69 |
| longest_decreasing_run | 101 | C | successions | -0.20 |
| longest_decreasing_run | 211 | B | peaks | -2.34 |
| longest_decreasing_run | 211 | B | successions | -9.47 |
| longest_decreasing_run | 211 | C | peaks | +2.34 |
| longest_decreasing_run | 211 | C | successions | -1.17 |
| longest_decreasing_run | 307 | B | peaks | -13.28 |
| longest_decreasing_run | 307 | B | successions | -0.78 |
| longest_decreasing_run | 307 | C | peaks | +2.93 |
| longest_decreasing_run | 307 | C | successions | -5.47 |
| longest_decreasing_run | 401 | B | peaks | -4.79 |
| longest_decreasing_run | 401 | B | successions | +0.98 |
| longest_decreasing_run | 401 | C | peaks | -3.52 |
| longest_decreasing_run | 401 | C | successions | -8.20 |
| longest_decreasing_run | 503 | B | peaks | -35.06 |
| longest_decreasing_run | 503 | B | successions | -32.62 |
| longest_decreasing_run | 503 | C | peaks | -12.99 |
| longest_decreasing_run | 503 | C | successions | -33.01 |

LDR两个条件的peaks共有10条同任务跨背景差，其中8条为负、2条为正。因此旧v4中peaks各块均下降的描述在新源记录上并非逐项完全复现；必须保留这些正向变化，不能写成所有公共源一致受损。successions同样具有seed/条件差异，完整40条未换公共源对照保留；20条被换slot只列不同任务的原/新水平，不计算其“同任务学习差”。

同一背景内B/C公共源差另见 `source_analysis/common_source_BC.csv`。这些task定义及输入流相同而组合不同，所以它们的学习改变是组合研究的实际结果；仅凭准确率表仍不能判断是优化速度、梯度竞争、共享结构还是其他原因。候选与公共源可能不同步改善，也不能用一个候选准确率作所有目标效果的代理。

## 所有源的训练、开发与后半预算变化

| 目标 | 背景 | 条件 | 任务 | train均值 | dev均值 | dev范围 | 多数基线均值 | 2400→4800 dev均值变化pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_points | g0_v3 | B | anti_fixed_points | 99.19% | 99.08% | [98.05%, 99.41%] | 37.03% | +0.70 |
| fixed_points | g0_v3 | B | descents | 51.47% | 50.43% | [45.70%, 58.98%] | 36.23% | +4.18 |
| fixed_points | g0_v3 | B | peaks | 62.89% | 61.64% | [60.35%, 66.21%] | 51.15% | +1.91 |
| fixed_points | g0_v3 | B | left_to_right_maxima | 95.67% | 95.21% | [93.07%, 96.88%] | 31.35% | +2.34 |
| fixed_points | g0_v3 | C | successions | 49.26% | 46.66% | [41.80%, 51.27%] | 39.00% | +3.18 |
| fixed_points | g0_v3 | C | descents | 62.82% | 62.29% | [49.32%, 87.01%] | 36.23% | +10.43 |
| fixed_points | g0_v3 | C | peaks | 66.67% | 65.29% | [59.38%, 71.00%] | 51.15% | +2.23 |
| fixed_points | g0_v3 | C | left_to_right_maxima | 96.34% | 95.88% | [93.95%, 97.46%] | 31.35% | +1.02 |
| fixed_points | g2_deficiencies | B | anti_fixed_points | 98.51% | 98.50% | [97.66%, 99.32%] | 37.03% | +2.73 |
| fixed_points | g2_deficiencies | B | descents | 49.43% | 47.30% | [42.77%, 52.15%] | 36.23% | +2.29 |
| fixed_points | g2_deficiencies | B | deficiencies | 99.62% | 99.55% | [99.12%, 99.90%] | 36.56% | +1.76 |
| fixed_points | g2_deficiencies | B | left_to_right_maxima | 97.68% | 97.19% | [95.12%, 98.73%] | 31.35% | +1.39 |
| fixed_points | g2_deficiencies | C | successions | 51.44% | 48.95% | [44.82%, 56.05%] | 39.00% | +5.66 |
| fixed_points | g2_deficiencies | C | descents | 64.19% | 62.50% | [48.05%, 72.17%] | 36.23% | +11.05 |
| fixed_points | g2_deficiencies | C | deficiencies | 99.80% | 99.80% | [99.61%, 99.90%] | 36.56% | +0.86 |
| fixed_points | g2_deficiencies | C | left_to_right_maxima | 98.98% | 98.63% | [97.36%, 99.71%] | 31.35% | +1.37 |
| longest_decreasing_run | g0_v3 | B | longest_increasing_run | 76.78% | 74.04% | [56.54%, 96.29%] | 54.00% | +11.50 |
| longest_decreasing_run | g0_v3 | B | descents | 85.94% | 85.27% | [55.37%, 99.80%] | 36.23% | +13.30 |
| longest_decreasing_run | g0_v3 | B | peaks | 79.82% | 78.34% | [67.58%, 96.58%] | 51.15% | +10.27 |
| longest_decreasing_run | g0_v3 | B | successions | 67.05% | 66.43% | [48.44%, 95.41%] | 39.00% | +9.26 |
| longest_decreasing_run | g0_v3 | C | left_to_right_maxima | 98.21% | 97.73% | [96.58%, 98.93%] | 31.35% | +0.39 |
| longest_decreasing_run | g0_v3 | C | descents | 68.03% | 67.07% | [49.12%, 97.36%] | 36.23% | +9.79 |
| longest_decreasing_run | g0_v3 | C | peaks | 68.35% | 66.11% | [61.33%, 74.32%] | 51.15% | +3.75 |
| longest_decreasing_run | g0_v3 | C | successions | 57.70% | 55.02% | [44.53%, 83.01%] | 39.00% | +7.58 |
| longest_decreasing_run | g1_recoils | B | longest_increasing_run | 57.02% | 57.01% | [53.22%, 62.11%] | 54.00% | +2.62 |
| longest_decreasing_run | g1_recoils | B | recoils | 57.25% | 55.88% | [50.29%, 68.36%] | 35.51% | +4.59 |
| longest_decreasing_run | g1_recoils | B | peaks | 64.64% | 63.75% | [59.57%, 68.95%] | 51.15% | +1.02 |
| longest_decreasing_run | g1_recoils | B | successions | 52.93% | 51.43% | [43.75%, 62.79%] | 39.00% | +6.07 |
| longest_decreasing_run | g1_recoils | C | left_to_right_maxima | 97.06% | 96.27% | [93.16%, 98.05%] | 31.35% | +1.56 |
| longest_decreasing_run | g1_recoils | C | recoils | 55.52% | 53.50% | [47.27%, 56.25%] | 35.51% | +3.67 |
| longest_decreasing_run | g1_recoils | C | peaks | 63.72% | 62.93% | [59.96%, 66.21%] | 51.15% | +1.80 |
| longest_decreasing_run | g1_recoils | C | successions | 48.94% | 45.41% | [43.85%, 50.00%] | 39.00% | +2.27 |

`source_final_by_repeat.csv`逐条保留train/dev CE、与多数基线之差、泛化差、2400→4800变化、暴露量及输入流hash。许多descents/recoils/peaks/successions条件在4800步仍远未接近满分，且部分仍有后半程改善；这说明当前预算的源学习状态不同，不能视为所有源充分学会后的纯任务组成比较。这里不设新“学会”门槛，也不据曲线延长预算；已有历史单任务数据使用不同seed和池，不能替代同预算新重复的单任务因果对照。

至此能支持的是源学习存在背景及重复依赖，以及LDR相关候选在recoils背景下的五次一致下降。是否复现预指定的正/负目标交互，必须等待全部新适配、开发选择和独立测试完成，再按冻结统计方案回答。本说明不读取新目标曲线或测试，也不改变任何组合、训练预算、选择规则或统计方法。
