# PermWorld second-round diagnostic report (2026-10-18)

Completed two paired seeds [17, 42], 30 learning-rate candidate adaptations of 800 updates each, 4 pretraining extensions of 2400 additional updates each, and evaluation on a new isolated test set generated after selections were frozen. The target is fixed_points with 256 distinct training examples; task composition, one-line encoding, and architecture follow pilot_v1. No tasks or targets were added.

**2400 pretraining steps and development selection: B−A=-17.26, C−A=-8.74, B−C=-8.52 percentage points. 4800 pretraining steps and development selection: B−A=-10.55, C−A=-12.87, B−C=+2.32 percentage points.**

Some development-selected means remain below A; negative transfer is retained in the results. After doubling the budget, 11/16 source-task × seed × condition combinations still have development accuracy below 90%. The 90% threshold describes incomplete mastery diagnostically, not a universal difficulty standard across tasks. These results cannot attribute transfer mechanisms to the known mathematical relation.

## Tasks, data, and fair comparison

The target is fixed_points(p)=#{i:p(i)=i}, at fixed n=10. B pretrains on anti_fixed_points, descents, peaks, left_to_right_maxima; C replaces only anti_fixed_points with successions. The relation fixed_points(p)=anti_fixed_points(c(p)), c(p)(i)=11−p(i), is an identity under input complement. Successions is not registered as having this direct relation; mathematical unrelatedness is not asserted.

Both seeds retain the first 256 v1 target training inputs and 1024 development inputs. B/C share 8192 pretraining inputs and 1024 source development inputs, with 16/64 examples per task in every batch. All adaptation candidates share same-seed data, sampling sequences, and a maximum 800 updates. No parameters are added: target-specific initialization restores only the existing target-task token's tied embedding/output row to the shared same-seed step0 value; numeric output parameters continue to transfer. Each adaptation creates AdamW afresh and verifies an empty state.

The new test set was generated after all 10 development selections were frozen, with 2048 inputs per seed. Complete identity/inverse/reverse/complement orbits exclude every v1 pool, including historical tests, and cross-seed new test inputs. Explicit orbit overlap is 0. Within each seed, all conditions, candidates, and protocols share the same new test set. All tables below use this test set; old v1 scores are not pooled into averages. Original v1 endpoints are reevaluated on the new test. The old development set is reused and the question arose from historical exploration, so this remains diagnostic exploration.

## Historical records and supplementary diagnostics

Historical records retain sparse last-batch two-token pretraining losses, source endpoint accuracies, target development accuracy/answer CE every 400 updates (only steps 400 and 800), and last-batch training loss. They lack per-source historical train/dev loss trajectories, full-target-training trajectories, and intermediate checkpoints. This round recomputes full source train/dev accuracy, answer CE, two-token CE, EOS accuracy, and majority baselines by length from original checkpoints. Historical intermediate models cannot be recovered from endpoints; new trajectories are explicitly labeled replay rather than presented as historical measurements.

The length-stratified baseline has only n=10 in this round. The majority label comes from the corresponding training pool, then its accuracy is measured on train/dev/test. Source-table answer CE uses the full 188-token vocabulary; training loss averages answer and EOS CE and must not be conflated with answer CE.

Strict weight-load hash checks passed=True; same-seed target-specific rows match=True; target data hashes match=True; adaptation sampling hashes match=True; save/resume 4+4 updates are bitwise equal=True. Original pretraining optimizer steps are 2400 and saved adaptation steps 800; new adaptation optimizer states start at 0.

V1 did not save adaptation-start parameter fingerprints. Historical load/reset conclusions rely on code, saved optimizer steps, and reconstruction checks, not direct historical start-state hashes. Every v2 candidate saves initialization.json before running, recording pre/post-load hashes, target-row hashes, and empty optimizer state.

Across the 6 historical 256-example adaptations from step400 to 800, development answer CE rises in 6 groups and development accuracy declines in 1. Worse CE may reflect confidence/calibration deterioration and does not always mean accuracy overfitting. Historical last-batch mixed two-token loss is not a substitute for full-training-set loss.

High-learning-rate protocol replays are bitwise identical to original v1 endpoints in 6/6 units. Maximum absolute parameter differences are in the [replay table](artifacts/replay_v1_weight_differences.csv). The replayed full training curves can be associated with identical endpoint weights; sparse historical records remain separately available.

| Seed | Condition | Pretraining updates | Bitwise equal | Maximum parameter difference |
| --- | --- | --- | --- | --- |
| 17 | A | 0 | True | 0 |
| 17 | B | 2400 | True | 0 |
| 17 | C | 2400 | True | 0 |
| 42 | A | 0 | True | 0 |
| 42 | B | 2400 | True | 0 |
| 42 | C | 2400 | True | 0 |

## Source learning and replacement difficulty

The table gives original- and double-budget endpoints for each seed. Accuracy and CE use complete pools. Full two-token/EOS metrics and training majority baselines are in the [source-learning table](artifacts/source_learning_table.csv); extension measurements every 600 updates are in the [complete source-curve CSV](artifacts/source_learning_all.csv).

| seed | Condition/updates | Task | Train acc | Dev acc | Train CE | Dev CE | Dev majority |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | B/2400 | anti_fixed_points | 79.42% | 79.20% | 0.4954 | 0.4998 | 36.82% |
| 17 | B/2400 | descents | 46.63% | 44.63% | 1.1989 | 1.2233 | 36.82% |
| 17 | B/2400 | left_to_right_maxima | 94.92% | 95.80% | 0.1475 | 0.1345 | 31.15% |
| 17 | B/2400 | peaks | 59.24% | 58.98% | 0.9093 | 0.8860 | 50.00% |
| 17 | B/4800 | anti_fixed_points | 88.48% | 88.57% | 0.2780 | 0.2763 | 36.82% |
| 17 | B/4800 | descents | 47.84% | 47.56% | 1.1646 | 1.1920 | 36.82% |
| 17 | B/4800 | left_to_right_maxima | 95.80% | 96.09% | 0.1227 | 0.1200 | 31.15% |
| 17 | B/4800 | peaks | 61.07% | 60.94% | 0.8556 | 0.8500 | 50.00% |
| 17 | C/2400 | descents | 52.86% | 49.80% | 1.0512 | 1.1173 | 36.82% |
| 17 | C/2400 | left_to_right_maxima | 96.00% | 96.58% | 0.1098 | 0.1135 | 31.15% |
| 17 | C/2400 | peaks | 63.81% | 63.77% | 0.7987 | 0.7919 | 50.00% |
| 17 | C/2400 | successions | 45.47% | 42.48% | 1.1768 | 1.2053 | 41.41% |
| 17 | C/4800 | descents | 58.30% | 55.08% | 0.9373 | 0.9996 | 36.82% |
| 17 | C/4800 | left_to_right_maxima | 97.91% | 98.54% | 0.0682 | 0.0633 | 31.15% |
| 17 | C/4800 | peaks | 68.58% | 66.41% | 0.7144 | 0.7286 | 50.00% |
| 17 | C/4800 | successions | 48.93% | 48.34% | 1.1254 | 1.1550 | 41.41% |
| 42 | B/2400 | anti_fixed_points | 79.28% | 76.76% | 0.5672 | 0.6235 | 36.91% |
| 42 | B/2400 | descents | 52.42% | 49.51% | 1.0613 | 1.1281 | 37.70% |
| 42 | B/2400 | left_to_right_maxima | 97.72% | 96.68% | 0.0813 | 0.0985 | 34.57% |
| 42 | B/2400 | peaks | 63.50% | 62.70% | 0.8000 | 0.8054 | 52.64% |
| 42 | B/4800 | anti_fixed_points | 92.03% | 90.04% | 0.2479 | 0.2838 | 36.91% |
| 42 | B/4800 | descents | 55.38% | 53.22% | 0.9793 | 1.0308 | 37.70% |
| 42 | B/4800 | left_to_right_maxima | 97.94% | 97.07% | 0.0637 | 0.0772 | 34.57% |
| 42 | B/4800 | peaks | 66.22% | 65.23% | 0.7494 | 0.7645 | 52.64% |
| 42 | C/2400 | descents | 53.64% | 49.51% | 1.0430 | 1.0859 | 37.70% |
| 42 | C/2400 | left_to_right_maxima | 98.00% | 97.36% | 0.0763 | 0.0870 | 34.57% |
| 42 | C/2400 | peaks | 66.08% | 65.82% | 0.7621 | 0.7639 | 52.64% |
| 42 | C/2400 | successions | 47.57% | 47.75% | 1.1557 | 1.1336 | 41.31% |
| 42 | C/4800 | descents | 66.99% | 65.23% | 0.7698 | 0.8065 | 37.70% |
| 42 | C/4800 | left_to_right_maxima | 98.46% | 97.75% | 0.0538 | 0.0752 | 34.57% |
| 42 | C/4800 | peaks | 70.32% | 71.19% | 0.6617 | 0.6622 | 52.64% |
| 42 | C/4800 | successions | 56.41% | 55.37% | 0.9968 | 0.9891 | 41.31% |

Answer distributions use the same 8192 source training inputs; extending training does not change the input distribution:

| seed | Task | Label mean | Entropy (bits) | B/C total variation |
| --- | --- | --- | --- | --- |
| 17 | anti_fixed_points | 0.9982 | 1.8757 | 0.0443 |
| 17 | successions | 0.8905 | 1.7832 | 0.0443 |
| 42 | anti_fixed_points | 1.0160 | 1.9031 | 0.0520 |
| 42 | successions | 0.8829 | 1.7777 | 0.0520 |

2400 updates, mean development accuracy for the replaced tasks is anti_fixed_points=77.98%, successions=45.12%; their mean development answer CE is 0.5617/1.1695. This is a learning difference under the current multitask optimization conditions, not an independent estimate of intrinsic task difficulty.

4800 updates, mean development accuracy for the replaced tasks is anti_fixed_points=89.31%, successions=51.86%; their mean development answer CE is 0.2801/1.0721. This is a learning difference under the current multitask optimization conditions, not an independent estimate of intrinsic task difficulty.

Among 16 paired source units from the original to doubled budget, development accuracy improves in 16/16 and development answer CE decreases in 16/16. Per-unit train/dev changes are in [source_budget_changes.csv](artifacts/source_budget_changes.csv). This records the actual impact of continued training while retaining units without mastery; improvement is not equivalent to having learned a task.

The three common B/C sources also reach different learning levels in the two compositions. The following development results average both seeds; B/C use identical common-task inputs, labels, sampling ratios, and total budgets:

| Pretraining updates | Common task | B dev acc | C dev acc | B−C pp | B dev CE | C dev CE |
| --- | --- | --- | --- | --- | --- | --- |
| 2400 | descents | 47.07% | 49.66% | -2.59 | 1.1757 | 1.1016 |
| 2400 | left_to_right_maxima | 96.24% | 96.97% | -0.73 | 0.1165 | 0.1002 |
| 2400 | peaks | 60.84% | 64.79% | -3.96 | 0.8457 | 0.7779 |
| 4800 | descents | 50.39% | 60.16% | -9.77 | 1.1114 | 0.9031 |
| 4800 | left_to_right_maxima | 96.58% | 98.14% | -1.56 | 0.0986 | 0.0693 |
| 4800 | peaks | 63.09% | 68.80% | -5.71 | 0.8073 | 0.6954 |

Task replacement accompanies source-capability differences across the whole multitask model, including common tasks. Later replacement matching should therefore examine both candidate and common-task learning curves and final capability. This observation identifies neither gradient conflict, mathematical relations, nor another specific mechanism, and does not prove intrinsic difficulty of any single task. Full train/dev means are in [common_source_tasks_comparison.csv](artifacts/common_source_tasks_comparison.csv).

## Adaptation protocol and new-test comparison

Every condition/seed/pretraining budget has the same three learning rates 3e−4,1e−4,3e−5 and 800 updates each, with 100 warmup steps followed by cosine decay to 0.1 of the initial rate. Full train/dev metrics are measured every 100 updates. Step0 is diagnostic only and ineligible for selection. Ties are resolved by highest development answer accuracy, lowest answer CE, earliest update, then declared learning-rate order. A checkpoint is selected within each candidate and then a learning rate using the same rule; test scores never participate.

The doubled budget continues from the original 2400-update checkpoint, preserving optimizer and random states, for another 2400 updates at the original endpoint learning rate 3e−5. It tests continuation at a fixed final rate, not a fresh 4800-step cosine plan. Both budgets use the same adaptation selection rule. A always has pretraining budget 0 and is reused in both panels, without extra training or counting as another seed.

| Comparison pretraining budget | Protocol | A | B | C |
| --- | --- | --- | --- | --- |
| 2400 | original_v1_endpoint_new_test | 84.35% | 69.68% | 77.05% |
| 2400 | replay_fixed_lr_endpoint | 84.35% | 69.68% | 77.05% |
| 2400 | dev_selected | 84.64% | 67.38% | 75.90% |
| 2400 | selected_lr_endpoint | 84.35% | 67.16% | 77.05% |
| 4800 | replay_fixed_lr_endpoint | 84.35% | 74.56% | 72.39% |
| 4800 | dev_selected | 84.64% | 74.10% | 71.78% |
| 4800 | selected_lr_endpoint | 84.35% | 74.56% | 72.39% |

`original_v1_endpoint_new_test` evaluates original 800-step weights at the original learning rate on the new test; `replay_fixed_lr_endpoint` is the retrained endpoint at that rate; `dev_selected` is the development-selected learning rate and checkpoint; `selected_lr_endpoint` is the 800-step endpoint at the same selected rate. Tables show arithmetic means across two seeds. Per-seed values and descriptive standard deviations are in the [protocol means CSV](artifacts/protocol_comparison_mean.csv) and [paired comparison CSV](artifacts/protocol_comparison_paired.csv); no significance inference is made.

![Protocol comparison](artifacts/figures/protocol_comparison.png)

| seed | Condition/pretraining | Selected LR | Selected step | Selected dev acc | Endpoint dev acc | Selected test acc | Endpoint test acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | A/0 | 0.0003 | 700 | 81.54% | 81.05% | 82.81% | 82.23% |
| 17 | B/2400 | 0.0003 | 700 | 83.79% | 82.71% | 84.47% | 83.89% |
| 17 | C/2400 | 0.0003 | 800 | 81.54% | 81.54% | 81.40% | 81.40% |
| 17 | B/4800 | 0.0003 | 800 | 76.56% | 76.56% | 76.32% | 76.32% |
| 17 | C/4800 | 0.0003 | 500 | 74.71% | 74.41% | 73.58% | 74.76% |
| 42 | A/0 | 0.0003 | 800 | 85.74% | 85.74% | 86.47% | 86.47% |
| 42 | B/2400 | 0.0001 | 500 | 55.76% | 53.91% | 50.29% | 50.44% |
| 42 | C/2400 | 0.0003 | 500 | 72.75% | 72.66% | 70.41% | 72.71% |
| 42 | B/4800 | 0.0003 | 500 | 74.22% | 73.24% | 71.88% | 72.80% |
| 42 | C/4800 | 0.0003 | 700 | 70.70% | 70.51% | 69.97% | 70.02% |

Development-selected learning-rate counts: 0.0003 selected in 9/10 groups, 0.0001 selected in 1/10 groups, 3e-05 selected in 0/10 groups; seed42 B/2400 selects 0.0001, 500 steps. Each seed's A0 is counted once;10 groups are not 10 independent seeds. With equally budgeted 800-step candidates, the results do not support uniformly lowering the rate as a universal repair. This does not exclude lower rates requiring more updates, different warmup/decay schedules, or better adaptation strategies outside the candidate set. See [selected_learning_rate_counts.csv](artifacts/selected_learning_rate_counts.csv).

At the fixed 2400-update pretraining budget, development selection changes new-test accuracy relative to the original v1 endpoint by A +0.29, B -2.29, C -1.15 percentage points. Both learning rate and stopping point change, so this evaluates their joint adaptation protocol; the difference cannot be attributed entirely to either factor.

Of 10 selections, 7 occur before 800 updates; in 6 of these, development accuracy decreases and development CE rises from the best development checkpoint to the endpoint. This count is affected by selection: the selected checkpoint having development accuracy at least as high as the endpoint is almost inherent in the rule, and alone does not prove overfitting. See the [curve CSV](artifacts/target_curves.csv) and [best-versus-endpoint differences](artifacts/best_vs_endpoint.csv). Corresponding new-test differences are frozen-protocol validation only, never used for reselection.

Among these 7 early selections, only 2 selected checkpoints have higher new-test accuracy than their same-rate 800-step endpoints, while 5 are lower. Development selection does not guarantee new-test benefits. Failed selections are fully retained under the frozen plan, without using the new test to choose endpoints or rates retrospectively or to retrain.

Separately, a uniform fixed 400→800-step comparison over all 30 candidates, independent of selected rates or checkpoints, finds training answer CE decreasing while development answer CE increases in 23/30; development accuracy also declines in 1/30. The former supports improving training fit with worsening development loss; only the latter additionally shows accuracy deterioration. Neither proves a sole cause of negative transfer. Sparse 100-step evaluation cannot equate this with numerical optimization instability or rule out transient instability at unsampled steps. This is an exploratory fixed-window diagnostic;30 candidates sharing data/initialization are not 30 independent seeds. Raw metrics and flags are in [late_training_diagnostics.csv](artifacts/late_training_diagnostics.csv).

| Condition/pretraining | Candidates | Train CE decreases | Dev CE increases | Dev acc decreases | Train CE↓ and dev CE↑ | Previous and dev acc↓ |
| --- | --- | --- | --- | --- | --- | --- |
| A/0 | 6 | 6 | 2 | 0 | 2 | 0 |
| B/2400 | 6 | 6 | 6 | 1 | 6 | 1 |
| C/2400 | 6 | 6 | 5 | 0 | 5 | 0 |
| B/4800 | 6 | 6 | 5 | 0 | 5 | 0 |
| C/4800 | 6 | 6 | 5 | 0 | 5 | 0 |

New-test differences after development selection for doubled versus original pretraining budget: B +6.71 percentage points; C -4.13 percentage points. Increased-budget results are separate from fixed-budget results; differences from twice the computation are not task-composition effects under the original fixed budget.

Per-seed budget effects follow (development-selected 4800−2400 test accuracy, percentage points). A mean benefit need not occur in every seed:

| seed | B budget effect | C budget effect |
| --- | --- | --- |
| 17 | -8.15 | -7.81 |
| 42 | +21.58 | -0.44 |

B budget effects have opposite signs across the two seeds, showing substantial heterogeneity; their means cannot be interpreted as stable benefits.Per-seed budget differences for all protocols are in [target_budget_deltas.csv](artifacts/target_budget_deltas.csv).

Differences by paired seed (percentage points):

| seed | Budget | Protocol | B−A | C−A | B−C |
| --- | --- | --- | --- | --- | --- |
| 17 | 2400 | dev_selected | +1.66 | -1.42 | +3.08 |
| 17 | 2400 | selected_lr_endpoint | +1.66 | -0.83 | +2.49 |
| 17 | 4800 | dev_selected | -6.49 | -9.23 | +2.73 |
| 17 | 4800 | selected_lr_endpoint | -5.91 | -7.47 | +1.56 |
| 42 | 2400 | dev_selected | -36.18 | -16.06 | -20.12 |
| 42 | 2400 | selected_lr_endpoint | -36.04 | -13.77 | -22.27 |
| 42 | 4800 | dev_selected | -14.60 | -16.50 | +1.90 |
| 42 | 4800 | selected_lr_endpoint | -13.67 | -16.46 | +2.78 |

## Compute budget and runtime

Each original B/C pretraining run uses 2400 updates×64=153600 sample exposures, 4300800 input tokens including answer/EOS, and 307200 supervised tokens. The doubled budget totals 307200 exposures, 8601600 input tokens, and 614400 supervised tokens. Each of four tasks receives exactly 25%. The pool has 8192 inputs; distinct inputs actually visited per task and sampling hashes are in [source_budgets.csv](artifacts/source_budgets.csv). Continuation uses the same pool, increasing exposures rather than distinct examples.

Each adaptation uses 800×64=51200 exposures, 1433600 input tokens, and 102400 supervised tokens; each of 256 target training examples is exposed 200 times on average. 3 candidates per group total 2400 updates. Reusing A gives 30 actual candidates, not 36.

| Stage | Runs | New compute | Total updates | Input tokens | Supervised tokens | Recorded loop seconds |
| --- | --- | --- | --- | --- | --- | --- |
| v1_pretraining_reused | 4 | False | 9,600 | 17,203,200 | 1,228,800 | 169.6 |
| v2_adapt_candidate | 30 | True | 24,000 | 43,008,000 | 3,072,000 | 866.9 |
| v2_pretrain_extension | 4 | True | 9,600 | 17,203,200 | 1,228,800 | 348.5 |

Recorded training and associated evaluation/saving timers total 1215.3 seconds, including 348.5 seconds and 17,203,200 additional input tokens for extended pretraining. Recorded-stage wall time totals 1275.7 seconds (21.3 minutes), coveringfreeze, select, smoke, test, train; it overlaps training timers and must not be added to them. Inventory ran through a separate CLI without a timing_inventory record, so its duration is excluded from the wall-time total above. This is not end-to-end duration for all diagnostic work. Adaptation timing includes step0 and later full train/dev evaluations, candidate best-checkpoint saves, and replay-weight comparisons where applicable; it excludes initialization and final endpoint saving. Source-extension timing includes training, source train/dev measurements, and CPU NumPy sampler replay checks, but excludes final saving. These are not pure GPU training times; exact diagnostic-forward tokens were not recorded. Model training tokens and evaluation costs are distinguished. Smoke-test updates are counted separately in smoke-stage wall time, not formal training tokens.

| Stage | Wall seconds |
| --- | --- |
| freeze | 0.01 |
| select | 0.02 |
| smoke | 2.41 |
| test | 40.88 |
| train | 1232.41 |

## Supported explanations and unresolved factors

1. **There is direct evidence of incomplete source mastery and mismatched learning levels under current multitask conditions.** Full source train/dev results and continuation curves show B/C capability differences for both replaced and common tasks. Similar answer distributions do not ensure matched learning conditions or final capability. Low training and development scores cannot be explained solely by source overfitting, and common-task differences do not isolate gradient conflict or a mathematical-relation mechanism.
2. **The fixed window shows some late-training generalization deterioration; same-data, same-budget candidates directly test protocol sensitivity.** From 400→800 steps, training answer CE decreases while development answer CE increases in 23/30 candidates, with development accuracy also declining in 1/30. This evidence does not depend on best-checkpoint selection. The development gap between best and endpoint is constrained by selection and does not independently prove overfitting. New-test differences between development selection and the original protocol are frozen-protocol validation. Joint learning-rate/checkpoint selection cannot alone identify the cause of negative transfer; failed selections are retained.
3. **Checks found no inconsistency in loading, target-specific rows, optimizer reset, or data/evaluation across conditions.** Hash/state audits, continuation smoke tests, and original-high-LR replay comparisons provide inspectable evidence. These engineering errors are not supported as explanations of the observed differences.
4. **Mechanisms remain unidentified.** Final replacement-task capability, gradient conflicts, representation changes, and complement-input inductive bias were not controlled. Longer pretraining changes both source capability and parameter location; it is not a mechanism-isolating intervention. No CKA sweep was conducted; compatible state_dicts and configurations remain available.
5. **Statistical and selection scope is limited.** Both initialization and data change across two random units, so their effects cannot be estimated separately. One length, target, and task replacement cannot generalize to the library. Repeated development-set reuse may introduce selection bias. All candidate new-test scores are disclosed only after freezing, without further rate selection, retraining, or configuration changes. The old v1 test retains its exploratory status.

## Next-step recommendation and stopping boundary

**Prioritize replacement-task matching, then add targets, and only later increase source-task count.** Source-development capability differences remain unresolved. Match replacements using single-task/source-development learning curves, label distributions, and convergence capability while target transfer tests remain unseen, reducing confounds in interpreting B/C mechanisms. Matching must include train/dev curves and final capability for the replaced task and all three common sources, because replacement also accompanies common-task changes here. V2 does not change tasks. If later work retains these replacements, first establish more adequate source learning under equal compute budgets. After matching, add a few prespecified targets with known relations to test reproducibility beyond one target. Increasing task count now would add both confounds and cost.

Expansion stops here pending further guidance. All completed results, including negative transfer and incompletely learned sources, are retained.

## Artifacts and reproduction

- All candidate tests by seed: [seed17](artifacts/results_seed17.csv), [seed42](artifacts/results_seed42.csv); [raw protocol comparison](artifacts/protocol_comparison.csv).
- [Full configuration and frozen code hashes](artifacts/FROZEN_PROTOCOL.json), [frozen selections](artifacts/SELECTIONS_FROZEN.json), [new-test isolation audit](artifacts/test_data_audit.json).
- [Source-learning table](artifacts/source_learning_table.csv), [actual budgets](artifacts/source_budgets.csv), [candidate development results](artifacts/candidate_development_results.csv), [complete target curves](artifacts/target_curves.csv), [actual cost](artifacts/compute_costs.csv).
- Selected-rate target train/dev curves by seed/condition/budget are `artifacts/figures/target_selected_*.png`; three-rate comparisons are `target_lr_comparison_*.png`; source-learning curves are `source_seed*.png`. Figures use answer accuracy and answer CE, without presenting two-token training loss as answer CE.
- [Reproduction instructions](README.md). This report and figures are generated from saved artifacts by `.venv/Scripts/python.exe experiments/pilot_v2_diagnostics/summarize.py`; the script has no model forwards and does not select or rerun training.

Final artifact audit status: passed; see [FINAL_AUDIT.json](artifacts/FINAL_AUDIT.json). The audit also verifies that selection freezing precedes new-test generation, test/selection hashes, frozen source hashes, and byte preservation of v1 files.
