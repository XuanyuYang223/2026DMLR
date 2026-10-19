# Pilot v3: negative transfer and task-composition diagnostics across multiple targets

This round completed a fixed-scale comparison of four targets, 256 training examples per target, and three paired seeds (17, 42, 73). Main results use 36 development-selected A/B/C checkpoints; all 216 test results from best and endpoint checkpoints of 108 learning-rate candidates are retained. All choices used mathematical definitions, training data, and development data only, followed by unified evaluation on an independent new test set generated after selections froze.

Mean B−A is negative for 2/4 targets, with all three seeds negative for 1/4; the corresponding C−A counts are 3/4 and 1/4. B−C changes sign across seeds for 4/4 targets. These are observed differences, not strong statistical inference from three seeds or a direct interpretation of B−C as the pure causal effect of a mathematical relation.

## Main results

Accuracies are means over three seeds. Differences are in percentage points; ± denotes the sample standard deviation of per-seed differences, not a confidence interval.

| Target | A | B | C | B−A mean±SD | C−A mean±SD | B−C mean±SD |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_points | 83.56% | 73.44% | 74.93% | -10.12 ± 3.45 | -8.63 ± 6.40 | -1.50 ± 6.07 |
| exceedances | 85.68% | 88.26% | 82.44% | +2.59 ± 3.60 | -3.24 ± 5.01 | +5.83 ± 8.37 |
| left_to_right_minima | 88.25% | 86.72% | 91.29% | -1.53 ± 8.80 | +3.04 ± 10.46 | -4.57 ± 10.72 |
| longest_decreasing_run | 53.89% | 65.09% | 53.11% | +11.20 ± 10.69 | -0.78 ± 1.11 | +11.98 ± 11.53 |

![Per-seed accuracy for each target](artifacts/figures/target_accuracy_by_seed.png)

![Paired differences and descriptive summaries](artifacts/figures/paired_deltas_summary.png)

Counts of negative / zero / positive seeds follow; no-benefit and opposite-direction results are retained.

| Target | B−A negative/zero/positive | C−A negative/zero/positive | B−C negative/zero/positive |
| --- | --- | --- | --- |
| fixed_points | 3/0/0 | 3/0/0 | 1/0/2 |
| exceedances | 1/0/2 | 2/0/1 | 1/0/2 |
| left_to_right_minima | 1/0/2 | 1/0/2 | 2/0/1 |
| longest_decreasing_run | 1/0/2 | 2/0/1 | 1/0/2 |

Targets with both B−A and C−A negative in all three seeds are fixed_points; these are observed directions across three seeds, not statistical significance or universal negative transfer on unseen tasks. For longest_decreasing_run, A/C test performance is close to the training-mode baseline: per-seed A minus baseline is +0.00, +0.00, -0.49, and C minus baseline is -1.95, +0.24, -1.12 percentage points (seed order 17/42/73). B improves over A for seeds 17/73, while seeds without improvement are 42; corresponding B−A differences are +16.21, -1.07, +18.46 percentage points. B's related source longest_increasing_run has development accuracies 72.17%, 63.18%, 86.62%, showing source learning and target benefit both varying with seed. Seeds change both input pools and initialization; three paired observations cannot establish a causal source. Target eligibility was established with 8192 single-task training examples; this does not imply A can learn with 256 examples and 800 adaptation steps. Target eligibility and small-sample learning failure therefore require separate interpretation.

Raw main results for each target and seed follow ([CSV](artifacts/results_selected.csv)). Test majority baselines use the most frequent answer among that seed's 256 target training examples, not the test-set mode.

| Target | seed | Condition | New-test accuracy | Dev accuracy | Selected LR | Selected step | Majority baseline | Training provenance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_points | 17 | A | 81.40% | 81.54% | 0.0003 | 700 | 36.23% | Historical weights reused |
| fixed_points | 17 | B | 74.95% | 76.56% | 0.0003 | 800 | 36.23% | Historical weights reused |
| fixed_points | 17 | C | 72.61% | 74.71% | 0.0003 | 500 | 36.23% | Historical weights reused |
| exceedances | 17 | A | 85.01% | 85.35% | 0.0001 | 800 | 34.18% | Trained this round |
| exceedances | 17 | B | 88.48% | 87.50% | 0.0003 | 500 | 34.18% | Trained this round |
| exceedances | 17 | C | 83.30% | 84.28% | 0.0003 | 800 | 34.18% | Trained this round |
| left_to_right_minima | 17 | A | 90.72% | 94.04% | 0.0003 | 800 | 32.81% | Trained this round |
| left_to_right_minima | 17 | B | 90.97% | 93.16% | 0.0003 | 700 | 32.81% | Trained this round |
| left_to_right_minima | 17 | C | 83.64% | 86.23% | 0.0003 | 800 | 32.81% | Trained this round |
| longest_decreasing_run | 17 | A | 53.52% | 56.93% | 0.0003 | 100 | 53.52% | Trained this round |
| longest_decreasing_run | 17 | B | 69.73% | 72.27% | 0.0001 | 800 | 53.52% | Trained this round |
| longest_decreasing_run | 17 | C | 51.56% | 56.25% | 3e-05 | 100 | 53.52% | Trained this round |
| fixed_points | 42 | A | 86.43% | 85.74% | 0.0003 | 800 | 35.64% | Historical weights reused |
| fixed_points | 42 | B | 73.14% | 74.22% | 0.0003 | 500 | 35.64% | Historical weights reused |
| fixed_points | 42 | C | 71.48% | 70.70% | 0.0003 | 700 | 35.64% | Historical weights reused |
| exceedances | 42 | A | 87.16% | 87.79% | 0.0001 | 800 | 35.84% | Trained this round |
| exceedances | 42 | B | 85.79% | 85.64% | 0.0003 | 700 | 35.84% | Trained this round |
| exceedances | 42 | C | 87.99% | 88.96% | 0.0003 | 600 | 35.84% | Trained this round |
| left_to_right_minima | 42 | A | 92.24% | 91.80% | 0.0003 | 400 | 27.98% | Trained this round |
| left_to_right_minima | 42 | B | 81.15% | 82.81% | 0.0001 | 800 | 27.98% | Trained this round |
| left_to_right_minima | 42 | C | 94.63% | 95.12% | 0.0001 | 800 | 27.98% | Trained this round |
| longest_decreasing_run | 42 | A | 53.47% | 55.86% | 0.0003 | 100 | 53.47% | Trained this round |
| longest_decreasing_run | 42 | B | 52.39% | 54.88% | 3e-05 | 100 | 53.47% | Trained this round |
| longest_decreasing_run | 42 | C | 53.71% | 56.05% | 3e-05 | 100 | 53.47% | Trained this round |
| fixed_points | 73 | A | 82.86% | 84.08% | 0.0001 | 400 | 36.67% | Trained this round |
| fixed_points | 73 | B | 72.22% | 72.56% | 0.0003 | 500 | 36.67% | Trained this round |
| fixed_points | 73 | C | 80.71% | 79.98% | 0.0003 | 800 | 36.67% | Trained this round |
| exceedances | 73 | A | 84.86% | 87.50% | 0.0001 | 800 | 35.99% | Trained this round |
| exceedances | 73 | B | 90.53% | 91.02% | 0.0003 | 400 | 35.99% | Trained this round |
| exceedances | 73 | C | 76.03% | 78.32% | 0.0003 | 700 | 35.99% | Trained this round |
| left_to_right_minima | 73 | A | 81.79% | 84.77% | 0.0001 | 700 | 31.25% | Trained this round |
| left_to_right_minima | 73 | B | 88.04% | 86.91% | 0.0003 | 700 | 31.25% | Trained this round |
| left_to_right_minima | 73 | C | 95.61% | 95.80% | 0.0003 | 800 | 31.25% | Trained this round |
| longest_decreasing_run | 73 | A | 54.69% | 55.47% | 3e-05 | 400 | 55.18% | Trained this round |
| longest_decreasing_run | 73 | B | 73.14% | 72.95% | 0.0003 | 800 | 55.18% | Trained this round |
| longest_decreasing_run | 73 | C | 54.05% | 55.47% | 0.0001 | 100 | 55.18% | Trained this round |

## Task catalog, target freezing, and composition

`TASK_CATALOG.csv` inventories 57 registered/historical interface records, including 48 PermWorld endpoints, 32 scalar property tasks, and 4 output-encoding conversion endpoints. Catalog rows are not independent mathematical task counts: input lengths and parameter values are parameter variants; one-line/cycle/Lehmer/inversion-vector are encoding variants of the same mathematical object, with conversions labeled separately. GraphWorld, MatrixGroupWorld, and historical finite-field tasks are inventoried only, not run. Relation checks cover all 5913 permutations of n=1…7 plus 2048 random n=10 permutations. Finite checks verify implementation; identities follow from definitions.

The four targets and compositions follow. In every row, target(p)=related_source(T(p)), with identity answer mapping; the three common B/C tasks remain fixed. Inverse, reverse, and complement cover three input transformations. Fixed points and exceedances share the broader positional-comparison family, so task diversity must not be overstated.

All four B relations have identity answer mappings. Nontrivial answer mappings and multitask additive identities are not covered. The catalog includes such other relations, but this round excludes them under learnability and limited-scale constraints.

| Target | Definition | B's related source | Input transform T | Replacement source in C | Three common sources |
| --- | --- | --- | --- | --- | --- |
| fixed_points | Fixed-point count # {i:p_i=i} | anti_fixed_points | C(p)_i=n+1-p_i | successions | descents, peaks, left_to_right_maxima |
| exceedances | Exceedance count # {i:p_i>i} | deficiencies | I(p)=p⁻¹ | recoils | descents, peaks, successions |
| left_to_right_minima | Number of strict new minima scanning left to right | right_to_left_minima | R(p)_i=p_{n+1-i} | longest_increasing_run | descents, peaks, successions |
| longest_decreasing_run | Length of the longest contiguous strictly decreasing run (at least 1) | longest_increasing_run | C(p)_i=n+1-p_i | left_to_right_maxima | descents, peaks, successions |

The plan froze before single-task development experiments. Target eligibility requires seed17 single-task 4800-step development accuracy≥90%, at least 15 percentage points above the training-mode development baseline. Transfer test results were not used to select targets.

| Target | Single-task dev accuracy | Training-mode dev baseline | Passed predefined gate |
| --- | --- | --- | --- |
| fixed_points | 99.71% | 36.43% | True |
| exceedances | 100.00% | 33.20% | True |
| left_to_right_minima | 99.90% | 31.45% | True |
| longest_decreasing_run | 99.90% | 55.76% | True |

Replacements minimize the sum of three terms within predefined candidate sets using seed17: source-training answer-histogram total variation TV, the absolute 4800-step single-task development-accuracy difference, and the mean absolute development-accuracy difference at steps 600/1200/2400/3600/4800. Ties follow candidate order. Fixed_points retains successions as a historical anchor, and exceedances also has only one predefined candidate; neither is improved through multiple-candidate matching. All lengths are 10, so length distributions match exactly. Remaining difficulty and answer-distribution differences are explicitly retained.

| Target | Related source | Replacement candidate | TV | Absolute endpoint accuracy gap | Mean absolute curve gap | Total score | Selected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_points | anti_fixed_points | successions | 0.0443 | 0.3125 | 0.4580 | 0.8148 | True |
| exceedances | deficiencies | recoils | 0.0109 | 0.0010 | 0.1727 | 0.1845 | True |
| left_to_right_minima | right_to_left_minima | cycle_count | 0.0239 | 0.5254 | 0.5111 | 1.0604 | False |
| left_to_right_minima | right_to_left_minima | longest_increasing_run | 0.2119 | 0.0010 | 0.0822 | 0.2951 | True |
| longest_decreasing_run | longest_increasing_run | left_to_right_maxima | 0.2006 | 0.0020 | 0.0902 | 0.2927 | True |
| longest_decreasing_run | longest_increasing_run | cycle_count | 0.2181 | 0.5264 | 0.4293 | 1.1738 | False |

Single-task records are retained for candidates outside the final source sets. The table uses actual 4800-step endpoints. Development accuracy<90% means incomplete learning under this configuration, not intrinsic task difficulty. Unselected candidates generally have only the design seed17 reference; three-seed coverage is not added.

| Unselected candidate | seed | Single-task train | Single-task dev | Dev majority baseline | 90% descriptive threshold |
| --- | --- | --- | --- | --- | --- |
| cycle_count | 17 | 53.45% | 47.36% | 33.50% | Not fully learned |

C means only that the designated direct input-transformation/answer-mapping relation is not annotated for this experiment; it never means complete mathematical unrelatedness. The catalog retains other known relations. Counterexamples refute only the tested direct identities, not all more complex relations. Similar answer distributions do not imply similar learning processes.

## Single-task and mixture learning references

Every actual source task has single-task references for three seeds, sharing each seed's initialization, 8192-example source training pool, and 1024-example development pool with mixture training. Length-stratified records cover only n=10; no length extrapolation is claimed. Full train/dev accuracy, answer CE, EOS CE, two-token CE, majority baselines, and learning curves are in the [single/multitask comparison](artifacts/single_vs_multi_sources.csv) and [complete source curves](artifacts/source_learning_all.csv).

New single- and multitask runs record full metrics at steps 0/600/1200/2400/3600/4800. Reused fixed_points source models for seeds17/42 lack full train/dev curves before 2400; only the original 2400-step endpoints and v2 continuation records at 3000/3600/4200/4800 remain. Their B/C curves start at 2400, without interpolated or fabricated earlier data. New single-task curves do not fill these historical multitask gaps.

Single-task 4800-step training exposes its task 307200 times, 4 times the 76800 per-task exposures in four-task 4800-step training. A1200-step single-task reference with 76800 exposures is also saved. The single-task 1200 learning-rate trajectory/final rate differs from mixture 4800, and distinct visited inputs may differ. Thus the following classes are descriptive diagnostics, not identification of pure task interference: single-task 4800-step development accuracy<90% is labeled "Single task not fully learned under this configuration"; ≥90% with mixture accuracy at least 5 percentage points lower is "Learned alone but degraded in multitask training"; all others are "Insufficient evidence of multitask degradation".

Across 96 target×condition×seed×source-task occurrences, the three class counts are 14, 65, 17. Sources repeat across compositions, so these are not 96 independent tasks; the first class also does not establish intrinsic learning difficulty.

These classes depend on single-task learning and do not separately identify mixture improvements: a task not mastered alone but improving by at least 5 percentage points in a mixture remains in the first class. We therefore separately count both directions of mixture 4800 minus single-task 4800 development accuracy. The±5-point threshold is descriptive only and changes neither the frozen protocol nor the classes above. Counts include all logical task occurrences and a version deduplicated by seed×task×source endpoint-state hash. Deduplicated records still share tasks, inputs, initializations, and compositions, so they are not independent samples.

| Counting scope | Records | Improvement≥5pp | Decline≥5pp | Absolute difference<5pp |
| --- | --- | --- | --- | --- |
| Raw logical task occurrences | 96 | 7 | 70 | 19 |
| Deduplicated by seed×task×source state | 84 | 6 | 60 | 18 |

The three largest improvements and three largest declines follow, or all entries if a direction has fewer than three. Logical runs sharing a state appear in the same row. See the full deduplicated table in [source_learning_effects_unique.csv](artifacts/source_learning_effects_unique.csv). These are descriptive source-development examples, not test-based selection.

| Direction | seed | Source task | Single 4800 dev | Multi 4800 dev | Difference pp | Logical runs containing this state |
| --- | --- | --- | --- | --- | --- | --- |
| Improvement | 73 | successions | 48.93% | 92.48% | +43.55 | s73_left_to_right_minima_C;s73_longest_decreasing_run_B |
| Improvement | 17 | successions | 68.65% | 86.62% | +17.97 | s17_exceedances_C |
| Improvement | 73 | successions | 48.93% | 61.23% | +12.30 | s73_exceedances_C |
| Decline | 17 | descents | 100.00% | 47.56% | -52.44 | s17_fixed_points_B |
| Decline | 17 | descents | 100.00% | 51.27% | -48.73 | s17_longest_decreasing_run_C |
| Decline | 73 | descents | 100.00% | 52.44% | -47.56 | s73_longest_decreasing_run_C |

Mixture improvements and declines may depend on seed and optimization. Single-task 4800 and mixture 4800 share total optimizer updates and global learning-rate trajectories, but single-task per-task exposure is 4 times higher and mixtures introduce task alternation. These differences prevent isolating pure interference. Single-task 1200 matches only per-task exposure, with different learning-rate history and total updates. Improvements do not directly prove causal synergy, nor do declines directly prove pure task interference. Both directions are retained so that failure to learn alone is not mistaken for failure in every mixture.

The following values average three seeds. Raw tables retain per-seed and training-set metrics.

| Target | Condition | Source task | Single 1200 dev | Single 4800 dev | Multi 4800 dev | Multi dev CE | Multi dev majority baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_points | B | anti_fixed_points | 99.32% | 99.93% | 90.79% | 0.246 | 36.56% |
| fixed_points | B | descents | 99.80% | 99.97% | 54.92% | 1.011 | 36.82% |
| fixed_points | B | peaks | 95.90% | 99.90% | 61.82% | 0.833 | 50.75% |
| fixed_points | B | left_to_right_maxima | 98.99% | 99.87% | 96.78% | 0.095 | 33.04% |
| fixed_points | C | successions | 47.17% | 69.56% | 53.61% | 1.048 | 40.33% |
| fixed_points | C | descents | 99.80% | 99.97% | 66.05% | 0.797 | 36.82% |
| fixed_points | C | peaks | 95.90% | 99.90% | 68.16% | 0.712 | 50.75% |
| fixed_points | C | left_to_right_maxima | 98.99% | 99.87% | 98.08% | 0.069 | 33.04% |
| exceedances | B | deficiencies | 99.41% | 99.87% | 99.71% | 0.030 | 34.47% |
| exceedances | B | descents | 99.80% | 99.97% | 83.07% | 0.409 | 36.82% |
| exceedances | B | peaks | 95.90% | 99.90% | 68.07% | 0.733 | 50.75% |
| exceedances | B | successions | 47.17% | 69.56% | 69.21% | 0.771 | 40.33% |
| exceedances | C | recoils | 76.63% | 99.90% | 67.97% | 0.757 | 36.98% |
| exceedances | C | descents | 99.80% | 99.97% | 80.31% | 0.486 | 36.82% |
| exceedances | C | peaks | 95.90% | 99.90% | 70.51% | 0.674 | 50.75% |
| exceedances | C | successions | 47.17% | 69.56% | 68.07% | 0.752 | 40.33% |
| left_to_right_minima | B | right_to_left_minima | 99.09% | 99.97% | 94.95% | 0.131 | 32.32% |
| left_to_right_minima | B | descents | 99.80% | 99.97% | 55.01% | 1.007 | 36.82% |
| left_to_right_minima | B | peaks | 95.90% | 99.90% | 70.05% | 0.665 | 50.75% |
| left_to_right_minima | B | successions | 47.17% | 69.56% | 48.60% | 1.139 | 40.33% |
| left_to_right_minima | C | longest_increasing_run | 90.04% | 99.71% | 73.99% | 0.637 | 52.28% |
| left_to_right_minima | C | descents | 99.80% | 99.97% | 78.09% | 0.519 | 36.82% |
| left_to_right_minima | C | peaks | 95.90% | 99.90% | 77.18% | 0.546 | 50.75% |
| left_to_right_minima | C | successions | 47.17% | 69.56% | 68.88% | 0.748 | 40.33% |
| longest_decreasing_run | B | longest_increasing_run | 90.04% | 99.71% | 73.99% | 0.637 | 52.28% |
| longest_decreasing_run | B | descents | 99.80% | 99.97% | 78.09% | 0.519 | 36.82% |
| longest_decreasing_run | B | peaks | 95.90% | 99.90% | 77.18% | 0.546 | 50.75% |
| longest_decreasing_run | B | successions | 47.17% | 69.56% | 68.88% | 0.748 | 40.33% |
| longest_decreasing_run | C | left_to_right_maxima | 98.99% | 99.87% | 97.49% | 0.076 | 33.04% |
| longest_decreasing_run | C | descents | 99.80% | 99.97% | 54.33% | 1.023 | 36.82% |
| longest_decreasing_run | C | peaks | 95.90% | 99.90% | 65.49% | 0.772 | 50.75% |
| longest_decreasing_run | C | successions | 47.17% | 69.56% | 47.92% | 1.140 | 40.33% |

The mean absolute B/C development-accuracy difference for common tasks is 14.00 percentage points, with maximum 48.24 percentage points across 36 task×seed×target occurrences. Common-task learning differences are retained as composition-training results; changing only the replaced task does not imply equal representations or learning states for other tasks.

![Common-source differences](artifacts/figures/common_source_differences.png)

The 24 logical pretraining runs correspond to only 7 distinct ordered source compositions and 6 unordered sets. Across three seeds there are 21 distinct endpoint-state hashes and 3 duplicate-hash groups (3 extra duplicate runs). Equal seeds, inputs, and ordered compositions can produce identical weights. Equal unordered sets with different slot orders need not be bitwise identical. The engine still executes every logical run, and new compute counts actual execution; duplicate states are not independent source-model evidence. See the [composition/state deduplication audit](artifacts/source_composition_reuse.csv).

## Training, adaptation, and testing protocol

The original PermWorld one-line tokenizer/vocabulary and Transformer are retained (4 layers,width 256,8 heads,dropout 0.1, tied embedding/output weights). All four targets are excluded from every pretraining source set. Source train/dev pools contain 8192/1024 inputs; target train/dev/new-test pools contain 256/1024/2048. Within each seed, targets and A/B/C reuse the same permutations with labels generated by their respective definitions. Seeds have distinct pools; shared inputs across targets mean cross-target results are not independent samples.

B/C each use four tasks,batch 64 with exactly 16 examples per task, and 4800 updates. Learning rate warms up for 100 steps to 3e−4, decays by cosine to 3e−5 at 2400, and stays 3e−5 until 4800; no automatic further extension is performed. AdamW(weight_decay=.01, betas=.9/.999, eps=1e−8), gradient clipping 1, FP32 weights/evaluation, BF16 training, TF32 disabled. Training loss weights answer-token and EOS cross-entropy equally. Accuracy is answer argmax over the full 188-token vocabulary given the input/task prompt; EOS is reported separately.

Each A/B/C has the same three rates (3e−4,1e−4,3e−5),800 steps each, and identical sampling order. After 100 warmup steps, cosine decay reaches 0.1 of the initial rate. Full train/dev metrics are computed every 100 steps. Selection maximizes development answer accuracy, then minimizes answer CE, favors earlier updates, and breaks remaining ties by declared rate order; step0 is ineligible. The existing target-specific tied token/output row is restored to same-seed step0, optimizer state is emptied, and source parameters are loaded strictly. Selected-rate counts are {0.0003: 23, 0.0001: 9, 3e-05: 4}. In the fixed 400→800 candidate window, development CE rises in 65/108 and development accuracy declines in 18/108. The endpoint gap after best-development selection is not treated as independent proof of overfitting.

`DESIGN_FROZEN.json` freezes design, code, and data; `FROZEN_PROTOCOL.json` freezes targets/compositions; `SELECTIONS_FROZEN.json` freezes 36 development selections and weight hashes. Only afterward is a new test set generated beyond round two: all v1/v2 inputs, v3 train/dev inputs, and their complete inverse/reverse/complement orbits are excluded, and new-test seeds are mutually orbit-isolated. Audited overlap is 0. The new test does not select targets, replacements, rates, or checkpoints. All 108 candidates' best/endpoints are evaluated together after freezing; disclosing 216 rows introduces no further selection.

For fixed_points seeds17/42,4 v2 double-budget pretrained weights and 18 adaptation candidates are reused exactly and reevaluated on this round's new test after file-hash and configuration/data compatibility checks. These retain historical exploratory status and are not independent retrained repetitions. Seed73 and the other three targets are newly trained. Final audit status=passed: byte hashes of 348 historical files, including v1/v2 code, configurations, data, results, and checkpoints, were checked; preservation passed. Full final audit: passed; see FINAL_AUDIT.json.

## Compute costs and limitations

Training timers in the table total 72.34 new minutes; stage wall times total 73.75 minutes. Training timers include full evaluations, saving, and some bookkeeping. Historical reused timers have slightly different scopes and are not counted as new compute. Stage times additionally include labeling/isolation checks, selection, and 216 test evaluations, and cannot be added to training timers. Summary plotting and final audits run separately and are excluded from engine-stage wall time. See per-row timing scopes in `compute_by_run.csv` and stage times in `stage_timings.csv`.

| Type | Cost provenance | Runs | Updates | Input tokens | Supervised tokens | Training timer minutes |
| --- | --- | --- | --- | --- | --- | --- |
| single | New | 32 | 153,600 | 275,251,200 | 19,660,800 | 31.83 |
| multi | New | 20 | 96,000 | 172,032,000 | 12,288,000 | 23.27 |
| multi | Historical reuse | 4 | 19,200 | 34,406,400 | 2,457,600 | 8.63 |
| adapt | New | 90 | 72,000 | 129,024,000 | 9,216,000 | 17.25 |
| adapt | Historical reuse | 18 | 14,400 | 25,804,800 | 1,843,200 | 8.67 |

Each new mixture run processes 8,601,600 input tokens and 614,400 supervised tokens, with 25% sampling per source. Each adaptation candidate processes 1,433,600 input tokens and 102,400 supervised tokens. Input tokens count the full 28-token forward training sequence, including answer/EOS; supervised tokens count the two loss positions. Timers include full evaluation and saving, but these training-token counts exclude evaluation tokens.

This round compares transfer only at 256 examples, 4800 pretraining steps, and 800 adaptation steps; it establishes no full sample-efficiency curve or scaling law. It has only three paired seeds, four learnability-screened PermWorld targets, and one length/model/encoding; design choices use seed17 development records. Best development values across rate candidates have selection bias. Test evaluation is frozen, but multiple-candidate results are disclosure only and must not select a new best main conclusion. Without a cross-budget sweep, effects are conditional on v2's specific extended-budget rate trajectory, not budget-independent. Answer distributions, known transformation relations, source learning, joint-training effects, and initialization may jointly explain B/C differences.

One concrete confound is exceedances C replacing deficiencies with recoils while descents remains common, introducing the known source–source relation recoils(I(p))=descents(p) (the catalog audit's descent_inverse relation). B/C change both internal source-set relations and the designated target–source relation; differences cannot be attributed solely to the latter.

## Next-stage recommendation

Prioritize deeper composition analysis: B−C changes direction with target or seed, common-source learning varies across B/C compositions, and some tasks are learned alone but degrade in mixtures.First hold task count, target sample size, and total budget fixed while controlling matching, per-task exposures, and optimization trajectories around common-source learning differences; then add a few more independent target families.These are research priorities based on this round's observations, not significance tests. All four targets remain in PermWorld, and fixed_points/exceedances both concern position comparisons.Increasing source-task count is not yet a priority, to avoid changing task count, exposure, and composition simultaneously.

Expansion stops here pending further guidance.

## Files and reproduction

- [All candidate raw results for three seeds](artifacts/results_seed17.csv), [seed42](artifacts/results_seed42.csv), [seed73](artifacts/results_seed73.csv); [all 216 rows](artifacts/results_all_candidates.csv), [36 main results](artifacts/results_selected.csv).
- [Paired differences](artifacts/paired_deltas.csv), [difference summaries](artifacts/paired_deltas_summary.csv), [endpoint versus development selection](artifacts/selected_vs_endpoint.csv); all means and sample SDs are in [results_summary.csv](artifacts/results_summary.csv).
- [Single/multitask comparison](artifacts/single_vs_multi_sources.csv), [common-source differences](artifacts/common_source_differences.csv), [candidate matching differences](artifacts/replacement_matching.csv).
- [All target train/dev curves](artifacts/target_curves.csv); figures are in `artifacts/figures/target_selected_*.png` and `source_learning_*.png`; original per-run curves/weight hashes are in `adapt/`, `single/`, and `multi/`.
- [Frozen configuration](artifacts/FROZEN_PROTOCOL.json), [compute costs](artifacts/compute_summary.csv), [stage timings](artifacts/stage_timings.csv), [audit](artifacts/FINAL_AUDIT.json); see the experiment directory's `README.md` for the task catalog and reproduction commands.
