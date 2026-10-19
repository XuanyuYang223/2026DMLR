# v5 source-task learning records: description before confirmation testing

This note reads only `source_curves.csv`, `source_exposures.csv`, the frozen configuration, and pretraining completion markers. It imports no model, performs no new training/forward passes, and reads no new target-test results. TEST_COMPLETE exists at analysis time: False. **These source changes have not yet been linked to the new confirmation interactions.** The following describes only directly observed source learning under the fixed4800-step budget.

All 40 new pretraining runs are complete. The 160 task×run records all use n=10 and 76800 exposures per task. Every source run completed 4800 steps under the original configuration; insufficient source learning does not trigger extensions. Metrics use 8192 training and 1024 development examples per task. The majority class is determined from that task's training pool and evaluated on its development pool. Learning must be interpreted relative to each task's own baseline, not by directly comparing accuracies across tasks.

## Paired candidate-source differences for all five repeats

| Target | seed | B dev original→new | B dev change pp | C dev original→new | C dev change pp |
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

For the FP-related candidate anti_fixed_points, mean development accuracy changes from 99.08% to 98.50%, a mean change of -0.59pp; 4 of five repeats decrease and 1 increase. Both backgrounds are near 99%, so FP background differences cannot generally be explained by complete failure to learn the related candidate in the original background. This observation does not yet establish how target transfer changes.

For the LDR-related candidate longest_increasing_run, mean development accuracy changes from 74.04% to 57.01%, declining in all five repeats: -23.05, -15.14, -2.25, -10.55, -34.18pp. The new-background mean development majority baseline is 54.00%; mean accuracy above baseline is only +3.01pp, with 0/5 below their own baseline and seed211 exactly at its baseline. New-background mean training accuracy is also only 57.02%; insufficient source learning is therefore not limited to a development gap after successful training fit. Optimization speed and representational limitations remain indistinguishable. Original-background development accuracy ranges from 56.54% to 96.29%, with substantial initialization/data-repeat variation. This supports the limited-repeat observation that the recoils background accompanies lower LIR source learning, not a causal claim that this decline removes target benefits.

## Unchanged common tasks and inconsistent changes

| Target | seed | Condition | Unchanged common source | Dev background difference pp |
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

For peaks across the two LDR conditions, the 10 same-task background differences include 8 negative and 2 positive changes. The v4 observation that peaks decreased in every block therefore does not reproduce entry by entry. These positive changes must be retained, rather than claiming that all common sources are consistently harmed. successions also varies by seed/condition. All 40 unchanged-common-source comparisons are retained;20 replaced-slot entries show original/new task levels without computing a same-task learning difference.

Within-background B/C common-source differences are in `source_analysis/common_source_BC.csv`. These tasks have identical definitions and input streams but different compositions, so learning changes are substantive composition results. Accuracy tables alone cannot determine whether optimization speed, gradient competition, shared structure, or other causes explain them. Candidate and common sources need not improve together; one candidate accuracy cannot proxy all target outcomes.

## Training, development, and second-half budget changes for all sources

| Target | Background | Condition | Task | Mean train | Mean dev | Dev range | Mean majority baseline | 2400→4800 mean dev change pp |
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

`source_final_by_repeat.csv` retains train/dev CE, differences from majority baselines, generalization gaps,2400→4800 changes, exposures, and input-stream hashes for every entry. Many descents/recoils/peaks/successions conditions remain far below perfect accuracy at 4800, with some still improving in the second half. Source learning therefore differs under this budget; this is not a pure composition comparison after all sources have been fully learned. No new learning threshold or curve-based budget extension is introduced. Historical single-task records use different seeds/pools and cannot replace same-budget single-task causal controls for the new repeats.

The current evidence supports background- and repeat-dependent source learning and five consistent decreases for the LDR-related candidate in the recoils background. Whether the prespecified positive/negative target interactions reproduce must be answered under the frozen statistical plan after all new adaptation, development selection, and independent testing finish. This note reads no new target curves or tests and changes no compositions, budgets, selection rules, or statistical methods.
