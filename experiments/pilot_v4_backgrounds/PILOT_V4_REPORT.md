# Pilot v4: do background tasks change composition effects?

A limited comparison is complete for two exploratory targets, three prespecified backgrounds, three paired blocks, and 256 fixed target-training examples. There are 42 unique development-selected primary results. Shared A is trained/selected once; showing it in three background comparisons produces 54 display rows, not 54 independent training runs. Best and endpoint checkpoints for all 126 learning-rate candidates were evaluated under the same new-test protocol. All 252 rows are retained, with no test-based reselection.

The clearest observation is that fixed_points negative transfer in the original background can change with the background, while the original LDR benefit disappears in g1. Candidate-source endpoint accuracy alone cannot explain this. Effect sizes, block-level differences, training endpoints, and counterexamples are all retained below. With n=3 these are descriptive observations, not general causal conclusions about backgrounds, tasks, or initialization.

## Primary results and background interactions

Accuracies are means over three paired blocks; differences are percentage points, and ± denotes the sample SD of block-level differences, not a confidence interval. A uses the same weights and test predictions across backgrounds; its mean is repeated only for comparison.

| Target | Background | A | B | C | B−A mean±SD | C−A mean±SD | B−C mean±SD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_points | g0_v3 | 83.27% | 74.12% | 75.13% | -9.15 ± 2.83 | -8.14 ± 6.62 | -1.01 ± 6.34 |
| fixed_points | g1_recoils | 83.27% | 61.80% | 60.40% | -21.47 ± 7.75 | -22.87 ± 9.94 | +1.40 ± 3.74 |
| fixed_points | g2_deficiencies | 83.27% | 90.71% | 87.35% | +7.44 ± 6.29 | +4.09 ± 5.41 | +3.35 ± 1.46 |
| longest_decreasing_run | g0_v3 | 53.34% | 64.96% | 52.21% | +11.62 ± 10.91 | -1.12 ± 0.90 | +12.74 ± 11.68 |
| longest_decreasing_run | g1_recoils | 53.34% | 53.40% | 52.65% | +0.07 ± 0.10 | -0.68 ± 0.61 | +0.75 ± 0.70 |
| longest_decreasing_run | g2_deficiencies | 53.34% | 59.65% | 52.59% | +6.32 ± 3.84 | -0.75 ± 2.32 | +7.06 ± 6.02 |

The following sequences are ordered by paired block 17/42/73. For fixed_points, B−A and C−A are negative in all 3/3 blocks in g0/g1; in g2, B−A becomes +10.16/+0.24/+11.91pp and C−A is +5.22/-1.81/+8.84pp. The original fixed_points negative transfer depends on the background and cannot be generalized as an intrinsic property of the two candidate compositions. The g2 B gain in block42 is only about 0.24pp; three matching signs do not establish a universal benefit. g2 improves both B and C, and both contain the multi-input joint target relation involving deficiencies. The absolute transfer improvement cannot be attributed entirely to the directly related candidate in B.

For longest_decreasing_run, g0 B−A is +17.97/-0.98/+17.87pp; it contracts to -0.05/+0.10/+0.15pp in g1 and is +10.25/+2.59/+6.10pp in g2. Both B/C in g1 are near the test baseline corresponding to the training mode, and the original g0 gains in blocks 17/73 disappear. g2 retains some benefit, while block42 changes from no benefit to a small positive gain.

![Accuracy by block](artifacts/figures/accuracy_by_background_seed.png)

![Composition contrasts across backgrounds](artifacts/figures/contrasts_by_background_seed.png)

The interaction is Δ(B−C)=(B−C)_later_background−(B−C)_earlier_background, differenced within the same paired block on identical new-test inputs. Positive values mean the later background favors B relative to C, not necessarily that B itself improves; B−A and C−A provide additional context.

| Target | Later−earlier background | Block17 interaction pp | Block42 interaction pp | Block73 interaction pp | Interaction mean±SD (pp) | Negative/zero/positive blocks | Blocks with B−C sign reversal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_points | g1_recoils-g0_v3 | -5.76 | +3.52 | +9.47 | +2.41 ± 7.68 | 1/0/2 | 2 |
| fixed_points | g2_deficiencies-g0_v3 | +1.42 | +0.34 | +11.33 | +4.36 ± 6.06 | 0/0/3 | 1 |
| fixed_points | g2_deficiencies-g1_recoils | +7.18 | -3.17 | +1.86 | +1.95 ± 5.18 | 1/0/2 | 1 |
| longest_decreasing_run | g1_recoils-g0_v3 | -20.07 | +2.00 | -17.92 | -12.00 ± 12.17 | 2/0/1 | 2 |
| longest_decreasing_run | g2_deficiencies-g0_v3 | -7.37 | +1.42 | -11.08 | -5.68 ± 6.42 | 2/0/1 | 1 |
| longest_decreasing_run | g2_deficiencies-g1_recoils | +12.70 | -0.59 | +6.84 | +6.32 ± 6.66 | 1/0/2 | 1 |

fixed_points: paired blocks where B−C changes sign across backgrounds: [17, 73]; background interactions with the same direction in all three blocks: ['g2_deficiencies-g0_v3']. longest_decreasing_run: paired blocks where B−C changes sign across backgrounds: [17, 42]; background interactions with the same direction in all three blocks: none. Signs must be interpreted alongside magnitude: LDR g1 block17 B−C is only −0.0488pp, one of 2048 test examples; g2 block42 B−C is +0.6836pp, whereas development B−C is −0.3906pp. These near-zero differences do not establish stable superiority. Sign-flip counts cannot replace effect sizes.

![Background interaction](artifacts/figures/background_interactions.png)

The 42 unique primary results are below ([CSV](artifacts/results_selected.csv)); A has background=shared. Links at the end provide all 84 candidate rows per seed. The majority baseline predicts the mode of that block's 256 target-training labels and measures accuracy on the test set.

| Target | Background | Paired block | Condition | New-test accuracy | Majority baseline | Selected LR | Selected step | Training origin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_points | shared | 17 | A | 82.18% | 38.82% | 0.0003 | 700 | Reused |
| fixed_points | g0_v3 | 17 | B | 75.83% | 38.82% | 0.0003 | 800 | Reused |
| fixed_points | g0_v3 | 17 | C | 72.31% | 38.82% | 0.0003 | 500 | Reused |
| fixed_points | g1_recoils | 17 | B | 59.62% | 38.82% | 0.0003 | 700 | New |
| fixed_points | g1_recoils | 17 | C | 61.87% | 38.82% | 0.0003 | 700 | New |
| fixed_points | g2_deficiencies | 17 | B | 92.33% | 38.82% | 0.0003 | 200 | New |
| fixed_points | g2_deficiencies | 17 | C | 87.40% | 38.82% | 0.0003 | 400 | New |
| longest_decreasing_run | shared | 17 | A | 52.00% | 52.00% | 0.0003 | 100 | Reused |
| longest_decreasing_run | g0_v3 | 17 | B | 69.97% | 52.00% | 0.0001 | 800 | Reused |
| longest_decreasing_run | g0_v3 | 17 | C | 49.95% | 52.00% | 3e-05 | 100 | Reused |
| longest_decreasing_run | g1_recoils | 17 | B | 51.95% | 52.00% | 3e-05 | 100 | New |
| longest_decreasing_run | g1_recoils | 17 | C | 52.00% | 52.00% | 3e-05 | 100 | New |
| longest_decreasing_run | g2_deficiencies | 17 | B | 62.26% | 52.00% | 3e-05 | 400 | New |
| longest_decreasing_run | g2_deficiencies | 17 | C | 49.61% | 52.00% | 3e-05 | 100 | New |
| fixed_points | shared | 42 | A | 86.43% | 37.06% | 0.0003 | 800 | Reused |
| fixed_points | g0_v3 | 42 | B | 74.41% | 37.06% | 0.0003 | 500 | Reused |
| fixed_points | g0_v3 | 42 | C | 72.71% | 37.06% | 0.0003 | 700 | Reused |
| fixed_points | g1_recoils | 42 | B | 57.81% | 37.06% | 0.0003 | 800 | New |
| fixed_points | g1_recoils | 42 | C | 52.59% | 37.06% | 0.0003 | 600 | New |
| fixed_points | g2_deficiencies | 42 | B | 86.67% | 37.06% | 0.0001 | 500 | New |
| fixed_points | g2_deficiencies | 42 | C | 84.62% | 37.06% | 0.0003 | 700 | New |
| longest_decreasing_run | shared | 42 | A | 53.08% | 53.08% | 0.0003 | 100 | Reused |
| longest_decreasing_run | g0_v3 | 42 | B | 52.10% | 53.08% | 3e-05 | 100 | Reused |
| longest_decreasing_run | g0_v3 | 42 | C | 52.83% | 53.08% | 3e-05 | 100 | Reused |
| longest_decreasing_run | g1_recoils | 42 | B | 53.17% | 53.08% | 3e-05 | 100 | New |
| longest_decreasing_run | g1_recoils | 42 | C | 51.90% | 53.08% | 3e-05 | 100 | New |
| longest_decreasing_run | g2_deficiencies | 42 | B | 55.66% | 53.08% | 3e-05 | 500 | New |
| longest_decreasing_run | g2_deficiencies | 42 | C | 54.98% | 53.08% | 3e-05 | 100 | New |
| fixed_points | shared | 73 | A | 81.20% | 36.08% | 0.0001 | 400 | Reused |
| fixed_points | g0_v3 | 73 | B | 72.12% | 36.08% | 0.0003 | 500 | Reused |
| fixed_points | g0_v3 | 73 | C | 80.37% | 36.08% | 0.0003 | 800 | Reused |
| fixed_points | g1_recoils | 73 | B | 67.97% | 36.08% | 0.0003 | 800 | New |
| fixed_points | g1_recoils | 73 | C | 66.75% | 36.08% | 0.0003 | 800 | New |
| fixed_points | g2_deficiencies | 73 | B | 93.12% | 36.08% | 0.0003 | 600 | New |
| fixed_points | g2_deficiencies | 73 | C | 90.04% | 36.08% | 3e-05 | 600 | New |
| longest_decreasing_run | shared | 73 | A | 54.93% | 55.27% | 3e-05 | 400 | Reused |
| longest_decreasing_run | g0_v3 | 73 | B | 72.80% | 55.27% | 0.0003 | 800 | Reused |
| longest_decreasing_run | g0_v3 | 73 | C | 53.86% | 55.27% | 0.0001 | 100 | Reused |
| longest_decreasing_run | g1_recoils | 73 | B | 55.08% | 55.27% | 3e-05 | 100 | New |
| longest_decreasing_run | g1_recoils | 73 | C | 54.05% | 55.27% | 3e-05 | 100 | New |
| longest_decreasing_run | g2_deficiencies | 73 | B | 61.04% | 55.27% | 0.0001 | 600 | New |
| longest_decreasing_run | g2_deficiencies | 73 | C | 53.17% | 55.27% | 0.0001 | 200 | New |

LDR adaptation in g1 is not unable to fit throughout training. All six development selections return to early checkpoints at LR=3e−5, step100; selected training accuracy is 52.73%–56.64%, whereas at the same LR the step800 endpoint training accuracy rises to 81.25%–95.31%, but endpoint test accuracy falls to 43.51%–48.88%. This supports improved late training fit with worse generalization and selection returning to an early model near baseline. Selected training accuracy near the mode must not be described as failure to learn the training data throughout the run.

| Block | Condition | Selected LR | Selected step | Selected train | Endpoint train | Selected dev | Endpoint dev | Selected test | Endpoint test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | B | 3e-05 | 100 | 53.52% | 81.25% | 56.84% | 49.32% | 51.95% | 46.48% |
| 17 | C | 3e-05 | 100 | 53.12% | 85.16% | 56.35% | 46.19% | 52.00% | 43.51% |
| 42 | B | 3e-05 | 100 | 52.73% | 89.84% | 55.86% | 49.22% | 53.17% | 48.88% |
| 42 | C | 3e-05 | 100 | 56.64% | 95.31% | 54.88% | 48.05% | 51.90% | 46.68% |
| 73 | B | 3e-05 | 100 | 54.69% | 89.84% | 55.37% | 47.17% | 55.08% | 48.10% |
| 73 | C | 3e-05 | 100 | 55.86% | 89.45% | 53.61% | 43.65% | 54.05% | 44.19% |

## Limited design and existing data platform

The fixed_points and longest_decreasing_run cases were selected from v3 results and are exploratory. They do not represent the full task library, and target selection is not independent of v3. Related/replacement candidates remain unchanged: anti_fixed_points/successions for fixed_points, and longest_increasing_run/left_to_right_maxima for longest_decreasing_run. Both target–related-source relations use complement C(p)_i=n+1−p_i and an identity answer map: target(p)=related(C(p)). They do not cover all mathematical relations.

| Target | Background | B candidate slot0 | C candidate slot0 | Common slot1 | Common slot2 | Common slot3 |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_points | g0_v3 | anti_fixed_points | successions | descents | peaks | left_to_right_maxima |
| fixed_points | g1_recoils | anti_fixed_points | successions | recoils | peaks | left_to_right_maxima |
| fixed_points | g2_deficiencies | anti_fixed_points | successions | descents | deficiencies | left_to_right_maxima |
| longest_decreasing_run | g0_v3 | longest_increasing_run | left_to_right_maxima | descents | peaks | successions |
| longest_decreasing_run | g1_recoils | longest_increasing_run | left_to_right_maxima | recoils | peaks | successions |
| longest_decreasing_run | g2_deficiencies | longest_increasing_run | left_to_right_maxima | descents | deficiencies | successions |

g0_v3 preserves the original background; g1_recoils replaces only common slot1 descents with recoils; g2_deficiencies replaces only common slot2 peaks with deficiencies. New backgrounds were frozen from definitions and existing single-task references before this round's transfer training/testing, without claiming fully matched difficulty or answer distributions. Within each background only slot0 differs between B/C, and the three common slots match. Across backgrounds, task definitions and known relations within the set can both change. Effects cannot be attributed solely to the direct candidate–target relation, and C must not be called mathematically unrelated. The catalog, input transformations/answer maps, and reuse checks are in TASK_CATALOG.csv, DATA_PLATFORM.md, and references/REUSE_AUDIT.md in the experiment directory.

Frozen [BACKGROUND_MATCHING.csv](catalog/BACKGROUND_MATCHING.csv) shows source-training answer-distribution TV of 0.00989–0.01428 for the g1 replacement, but early single-task learning is much slower for recoils than descents. g2 TV is 0.76013–0.76526, a large distribution difference, with different early learning trajectories as well. Equal population distributions do not imply identical finite-pool histograms or matched optimization difficulty. The table describes existing records before freezing; backgrounds were not searched for an optimal score using these values.

| block | Background | Replacement | Answer TV | Original dev1200 | New dev1200 | Original dev4800 | New dev4800 | Mean absolute curve gap pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | g1_recoils | descents→recoils | 0.01428 | 99.90% | 62.79% | 100.00% | 99.80% | 16.56 |
| 17 | g2_deficiencies | peaks→deficiencies | 0.76013 | 94.82% | 99.90% | 100.00% | 99.90% | 3.93 |
| 42 | g1_recoils | descents→recoils | 0.00989 | 99.71% | 74.02% | 99.90% | 100.00% | 12.48 |
| 42 | g2_deficiencies | peaks→deficiencies | 0.76135 | 96.09% | 99.22% | 99.71% | 100.00% | 4.69 |
| 73 | g1_recoils | descents→recoils | 0.01294 | 99.80% | 93.07% | 100.00% | 99.90% | 8.79 |
| 73 | g2_deficiencies | peaks→deficiencies | 0.76526 | 96.78% | 99.12% | 100.00% | 99.71% | 4.75 |

Remaining differences within the B/C candidate pairs must also be retained, beyond checking common-background replacements. [candidate_matching.csv](artifacts/candidate_matching.csv) below uses frozen source-answer histograms and existing single-task reference curves; the same pair is used in all three backgrounds. This round's new pretraining, adaptation, and test results were not used to choose candidates. The original selection evidence is [v3 replacement_matching.csv](../pilot_v3_multitarget/artifacts/replacement_matching.csv): fixed_points retains the historical anti_fixed_points/successions anchor, and longest_decreasing_run retains the longest_increasing_run/left_to_right_maxima pair selected in v3 using seed17 development records. This round only describes matching across the three paired blocks.

| Target | block | B candidate | C candidate | Answer TV | B single dev1200 | C single dev1200 | B single dev4800 | C single dev4800 | Mean absolute curve gap pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_points | 17 | anti_fixed_points | successions | 0.04431 | 99.12% | 47.17% | 99.90% | 68.65% | 45.80 |
| fixed_points | 42 | anti_fixed_points | successions | 0.05200 | 99.80% | 55.47% | 99.90% | 91.11% | 31.91 |
| fixed_points | 73 | anti_fixed_points | successions | 0.03320 | 99.02% | 38.87% | 100.00% | 48.93% | 55.94 |
| longest_decreasing_run | 17 | longest_increasing_run | left_to_right_maxima | 0.20056 | 93.07% | 99.61% | 100.00% | 99.80% | 9.02 |
| longest_decreasing_run | 42 | longest_increasing_run | left_to_right_maxima | 0.21887 | 87.89% | 98.44% | 99.51% | 99.80% | 9.53 |
| longest_decreasing_run | 73 | longest_increasing_run | left_to_right_maxima | 0.20776 | 89.16% | 98.93% | 99.61% | 100.00% | 9.41 |

This table and the background-matching table describe two types of replacement. Similar TV does not guarantee equal learning speed; both tasks being learned by step4800 does not guarantee matching early trajectories. These are retained confounding differences, not grounds for reoptimizing composition in this round.

After composition freezing and before inspecting new transfer results, a definition-level scientific review added relations absent from the original catalog; compositions, data, budgets, and selection rules were unchanged. See [catalog_addenda/README.md](catalog_addenda/README.md). Let D=deficiencies, I=inverse, R=one-line reversal, and C=complement. Then fixed_points(p)=n−D(p)−D(I(p)), and anti_fixed_points(p)=n−D(C(p))−D(R(I(p))). Thus both B/C in fixed_points g2 contain a background task that jointly reconstructs the target across transformed inputs; C cannot be interpreted as lacking target relations. One D label on one input does not determine the target. Training supplies no such paired queries, joint answers, or relation loss, and within-pool orbit isolation does not retain distinct transformation-equivalent inputs. This is an implicit mathematical relation, not single-label leakage or evidence that the model used the identity.

g1 removes descents and replaces it with recoils; it does not add a coexisting descents/recoils pair. For longest_decreasing_run, LDR/LIR/descents/peaks share the local comparison-sign sequence: descents counts descending edges, LDR/LIR take the longest consecutive descending/ascending segment, and peaks counts local peaks. These scalar answers do not determine each other. g1 removes descents but retains peaks; g2 retains descents, removes peaks, and adds a position-inequality count. They alter computational structure differently. Marginal answer distributions, learning dynamics, source–source/multi-input relations, and local computational structure all change; a single relation-graph edge count cannot summarize this experiment.

Model/data/budgets remain v3-compatible: PermWorld n=10, one-line encoding, Transformer with 4 layers/width256/8 heads/dropout0.1, vocabulary188, and tied embedding/output; FP32 weights/evaluation, BF16 training, TF32 disabled. Source train/dev pools have 8192/1024 inputs, target train/dev have 256/1024, and the final new test has 2048 per block. Targets, backgrounds, and A/B/C share input pools within a block; this sharing does not make targets or backgrounds independent samples.

Paired-block labels 17/42/73 jointly index initialization and data pools; they are not data-generation RNG seeds. s17/s42 were generated sequentially with RNG 20261018, and s73 with RNG 2026101803; model initialization seeds are 17/42/73. Pretraining sampling/dropout seed=block label +1000, adaptation=+2000, and new-test RNG 2026101805. Training/development pools and step0 weights are reused byte-for-byte from v3; seed_provenance.json records their origins. Block differences combine input, initialization, and optimization randomness; these factors were not independently manipulated.

## Source learning, exposure, and limits of interpretation

Each pretraining run has 4800 steps. Each batch of 64 contains exactly 16 examples in each slot0…3, giving exactly 25% task sampling. Each slot receives 76800 exposures. Input/label/index-stream hashes are checked per slot; all backgrounds/conditions in a block share input indices and slot assignment. The same mathematical task in the same slot also has the same label stream. All 144 rows of [source_exposures.csv](artifacts/source_exposures.csv) retain unique-input counts, actual exposures, tokens, and hashes. 76800 exposures are not 76800 distinct inputs.

All source train/dev accuracies, answer/EOS/two-token CE, n=10 majority baselines, and curves are retained. [source_learning_table.csv](artifacts/source_learning_table.csv) additionally links v3 single-task step1200/4800 references using identical inputs/initialization. No new single-task computation was performed. Single-task4800 and multitask 4800 have the same total updates and global LR trajectory, but single-task per-task exposure is four times larger, with different task alternation. Single-task1200 matches only per-task exposure; LR history and total updates differ. These comparisons cannot isolate pure interference.

Both increases and decreases in source-development accuracy relative to single-task4800 are retained. ±5pp is only a descriptive threshold; deduplication uses seed×task×source-endpoint-state hash. Multiple appearances of a task in compositions are not independent samples. The 36 logical source runs have 36 distinct endpoint-state hashes.

| Scope | Records | Improvement≥5pp | Decline≥5pp | Absolute gap<5pp |
| --- | --- | --- | --- | --- |
| logical_occurrences | 144 | 4 | 92 | 48 |
| unique_seed_task_source_state | 144 | 4 | 92 | 48 |

Within-background B/C development differences for the same common task reach an absolute maximum of 48.24pp. Across backgrounds, [source_background_slot_changes.csv](artifacts/source_background_slot_changes.csv) computes changes only for the same mathematical task in each slot. When a slot's task is replaced, it shows both tasks and metrics but leaves the difference blank, avoiding interpretation of different-task accuracy differences as a change in learning the same task.

The following are up to four descriptive cases ranked by absolute target interaction. Candidate B/C source-development changes are later-minus-earlier background differences for the same task; common-source columns cover only unchanged slots. These tables juxtapose state changes without fitting correlations or making causal attributions, and were not used to add experiments or select primary results.

| Target | Block | Later−earlier background | Target interaction pp | B candidate source dev change pp | C candidate source dev change pp | Max absolute same-task common-source change pp |
| --- | --- | --- | --- | --- | --- | --- |
| longest_decreasing_run | 17 | g1_recoils-g0_v3 | -20.07 | -15.82 | +0.98 | 12.99 |
| longest_decreasing_run | 73 | g1_recoils-g0_v3 | -17.92 | -34.86 | -0.59 | 49.51 |
| longest_decreasing_run | 17 | g2_deficiencies-g1_recoils | +12.70 | +16.60 | -0.39 | 31.15 |
| fixed_points | 73 | g2_deficiencies-g0_v3 | +11.33 | +4.98 | -7.62 | 11.52 |

Source-learning states for the two B candidates are listed below in 17/42/73 order. Single-task1200 matches only per-task exposure; LR history and total updates still differ.

| Target | Background | B candidate source | Multitask train across 3 blocks | Multitask dev across 3 blocks | Single1200 dev across 3 blocks |
| --- | --- | --- | --- | --- | --- |
| fixed_points | g0_v3 | anti_fixed_points | 88.48%/92.03%/95.20% | 88.57%/90.04%/93.75% | 99.12%/99.80%/99.02% |
| fixed_points | g1_recoils | anti_fixed_points | 99.40%/88.12%/97.11% | 99.41%/85.74%/96.39% | 99.12%/99.80%/99.02% |
| fixed_points | g2_deficiencies | anti_fixed_points | 98.85%/77.56%/99.15% | 98.73%/76.07%/98.73% | 99.12%/99.80%/99.02% |
| longest_decreasing_run | g0_v3 | longest_increasing_run | 75.85%/61.21%/89.07% | 72.17%/63.18%/86.62% | 93.07%/87.89%/89.16% |
| longest_decreasing_run | g1_recoils | longest_increasing_run | 57.32%/54.92%/55.62% | 56.35%/53.61%/51.76% | 93.07%/87.89%/89.16% |
| longest_decreasing_run | g2_deficiencies | longest_increasing_run | 72.35%/72.47%/70.78% | 72.95%/69.63%/66.31% | 93.07%/87.89%/89.16% |

For peaks, g1−g0 lowers development accuracy in 12/12 entries across two targets × B/C × three blocks. The LDR-related source LIR has low train/dev accuracy in g1, below its single-task step1200 reference. This accompanies loss of target benefit in blocks 17/73, but block42 had no original benefit, so target performance does not decline in every block.

Candidate-source endpoint accuracy alone cannot explain transfer: fixed_points block17, g0 to g1_recoils: anti_fixed_points source dev 88.57%→99.41%, target B test 75.83%→59.62%; fixed_points block42, g0 to g2_deficiencies: anti_fixed_points source dev 90.04%→76.07%, target B test 74.41%→86.67%. For fixed_points g2 versus g0, C candidate successions source dev changes by -2.44/-11.33/-7.62pp while C target test changes by +15.09/+11.91/+9.67pp; in all three blocks, the candidate source worsens while the target improves.

LDR g2 common-source B−C development differences: descents: +40.53/+9.57/+29.69pp; successions: +32.91/+9.28/+18.75pp; deficiencies: -0.59/+0.39/-0.29pp. Target B/C differences thus accompany changes in common descents/successions learning, while deficiencies is almost unchanged; these cannot be decomposed into independent mechanisms. Compositions do not only harm sources: improvements in LDR-B successions over single-task step4800 include g2_deficiencies block17 +10.64pp; g0_v3 block73 +43.55pp; g2_deficiencies block73 +15.92pp. Unequal per-task exposure, task alternation, and optimization differences across three blocks still limit interference/synergy attribution.

Reused g0 source curves have historical gaps: fixed_points blocks 17/42 contain full train/dev records only at 2400 and 3000/3600/4200/4800. Their plots start at 2400; earlier points were not fabricated. Newly trained runs and other compatible v3 records contain metrics at 0/600/1200/2400/3600/4800. Length-specific references cover only n=10, with no length-extrapolation claim.

## Adaptation selection, isolation, and budget

B/C each receive 4800 steps. Source LR warms up for 100 steps to 3e−4, follows cosine decay to 3e−5 at 2400, then stays at 3e−5 through 4800; no extension was added. AdamW (weight_decay=.01, betas=.9/.999, eps=1e−8), gradient clipping 1. Every adaptation condition uses the same three LRs (3e−4/1e−4/3e−5),800 steps each,100-step warmup and cosine decay to 0.1 of initial LR. All candidates within a block share adaptation data and sampling order. The existing target token's tied embedding/output row is restored to the block's step0 value. Pretrained weights are strictly loaded, the optimizer is reset, and no new output head is added.

Full training/development evaluation occurs every 100 steps. Selection prioritizes highest development answer accuracy, lowest development answer CE, earliest step, then declared LR order; step0 is excluded. Answer accuracy uses prompt-only argmax over the full vocabulary; answer and EOS each contribute half of training loss. LR counts among 42 unique selections are {0.0003: 21, 0.0001: 6, 3e-05: 15}. Over the fixed400→800 window, 106/126 candidates have increased development CE and 32/126 decreased development accuracy. Best-development and endpoint comparisons are retained; post-selection gaps alone are not treated as independent evidence of overfitting.

FROZEN_PROTOCOL locks six target/background compositions, code, and prior evidence. Independent new tests are generated only after SELECTIONS_FROZEN locks 42 selections and checkpoint hashes. New tests exclude the full inverse/reverse/complement orbits of all v1/v2/v3 inputs and v4 train/dev inputs; the three new test pools are isolated from each other as well. Targets/backgrounds/conditions are evaluated uniformly within a block. Tests do not select backgrounds, LRs, or checkpoints. Final audit passed; byte hashes checked for 1134 historical files; see FINAL_AUDIT.json for complete evidence.

Of 36 logical pretraining runs,12 are reused and 24 new. Of 126 adaptation candidates,18 shared-A and 36 g0 B/C candidates are reused, and 72 are new. Exact reuse requires compatibility of ordered task tuples, input pools, initialization, sampling, and budgets, not merely unordered task sets. Historical weights are reevaluated on new tests, not retrained as independent repeats. All 252 best/endpoint candidate test records are retained. Test evaluation costs appear in stage timings, not training-token totals.

| Stage | Origin | Runs | Updates | Training input tokens | Supervised tokens | Training timer minutes |
| --- | --- | --- | --- | --- | --- | --- |
| pretrain | New | 24 | 115,200 | 206,438,400 | 14,745,600 | 20.94 |
| pretrain | Historical reuse | 12 | 57,600 | 103,219,200 | 7,372,800 | 17.92 |
| adapt | New | 72 | 57,600 | 103,219,200 | 7,372,800 | 13.52 |
| adapt | Historical reuse | 54 | 43,200 | 77,414,400 | 5,529,600 | 15.62 |

New training timers total 34.46 minutes; engine-stage wall time totals 36.03 minutes. Timers include training, full train/dev evaluation, saving, and some bookkeeping. Reuse timers come from historical stages with somewhat different scope and are not new compute. The two timing measures must not be added. Auditing and plotting run separately, outside engine-stage timing. Input tokens count the full 28-token training forward sequence; supervised tokens count answer/EOS positions; evaluation tokens are excluded. Single-task references are reused records with no additional training cost.

## Limits and next steps

This is a limited exploration of two cases selected from v3 results, three backgrounds, and three paired blocks. Backgrounds and tasks share data, initialization, and source weights. Target samples are fixed at 256, pretraining at 4800 steps, and adaptation at 800 per LR; this establishes neither a sample-efficiency curve nor a scaling law. Nonpositive gains, negative transfer, direction reversals, and insufficiently learned sources/targets are retained. Targets previously passed a learnability threshold with 8192 single-task examples; this does not ensure learning with 256. Small-sample outcomes require interpretation alongside majority baselines.

For each background, C has the same unordered task set for both targets, with successions and left_to_right_maxima swapped between slot0/slot3. Under identical index/slot streams, the specific inputs assigned to tasks therefore differ, so C weights/performance can differ. The target task never enters pretraining inputs; cross-target C differences cannot be interpreted as the target name changing pretraining. This further limits independence of the two cases.

The next priority is a task catalog and limited composition controls spanning relation type × background computational structure, rather than immediately increasing source-task count. Current evidence shows that fixed_points negative transfer can change in g2, LDR benefits disappear in g1, common-source learning changes concurrently, and candidate-source accuracy has counterexamples as a proxy for target gains. Candidate libraries should distinguish distribution matching, learning-curve matching, and relation categories:

1. **Direct invertible transformations and answer maps.** Retain identity-answer relations under complement/reverse/inverse as anchors; add nonidentity affine-map references, such as the same descents task on complemented inputs satisfying d(Cp)=n−1−d(p). These are transformation/map variants of one task, not new independent tasks.
2. **Equal marginal distributions with different computational structures.** Existing Eulerian statistics such as descents/recoils/exceedances can distinguish adjacent-position comparisons, inverse-position comparisons, and diagonal-position comparisons under one-line encoding. g1 shows that similar TV can coexist with different early learning and composition effects; matching answer histograms alone is insufficient.
3. **Multi-source or multi-input joint relations.** Register the fixed/exceedances/deficiencies position partition, joint recovery of fixed from D(p) and D(Ip), and cycle_count=fixed_points+nontrivial_cycle_count. Distinguish multiple source answers on one input from multiple queries on transformed inputs. Testing whether models use these relations requires separately frozen paired-input/relation-constraint interventions; implicit relations here are not evidence of an implemented mechanism.
4. **Shared local computational structure without mutually determining scalars.** Organize backgrounds using existing LIR/LDR/descents/peaks/valleys and local triple counts. Distinguish a shared intermediate comparison sequence, equalities, and bounds, rather than reducing retention/removal of shared computations to relation-edge counts.
5. **One-way information relations and other computational families.** The existing cycle_count-to-parity modulo 2 map and cycle-decomposition counts offer structures beyond local/position statistics, but require single-task learnability review first. The v3 cycle_count candidate failed its learning threshold; an unlearned task cannot simply be treated as a matched control.

At the composition level, retain four sources, fixed budgets, and ordered-slot pairing. Specify which groups change only marginal distributions/learning speed and which also change local structure or joint relations. Cross-target comparisons of identical C sets must separately address slot-to-input assignment. These are future recommendations: no new targets, paired queries, slot controls, or training were executed, and no search for the best background was performed using candidate test maxima.

This round stopped at the prespecified scale, with no additional learning rates, budgets, backgrounds, or targets, pending further guidance.

## Files

- [42 unique primary results](artifacts/results_selected.csv), [54 background-comparison display rows](artifacts/results_display_ABC.csv), [All 252 candidate results](artifacts/results_all_candidates.csv).
- Raw CSV per block: [17](artifacts/results_seed17.csv), [42](artifacts/results_seed42.csv), [73](artifacts/results_seed73.csv).
- [Paired composition differences](artifacts/paired_deltas.csv), [Background interactions by block](artifacts/background_interactions.csv), [Interaction summary](artifacts/background_interactions_summary.csv), [Endpoint comparison](artifacts/selected_vs_endpoint.csv).
- [Source learning/single-task references](artifacts/source_learning_table.csv), [Common-source B/C differences](artifacts/common_source_BC.csv), [Source and target interactions side by side](artifacts/interaction_source_context.csv).
- [Distribution and difficulty differences for retained B/C candidate pairs](artifacts/candidate_matching.csv); selection evidence: [v3 pre-freeze matching records](../pilot_v3_multitarget/artifacts/replacement_matching.csv).
- [Full target curves](artifacts/target_curves.csv), [Source curves](artifacts/source_curves.csv), [Frozen configuration](artifacts/FROZEN_PROTOCOL.json), [Cost](artifacts/compute_summary.csv), [Stage timings](artifacts/stage_timings.csv); see README.md for reproduction commands.
- [Independent statistical recomputation and CSV-binding audit](artifacts/STATISTICS_AUDIT.json); recorded separately from the final weights/protocol audit.
- [Post-freeze definition-level relation review addendum](catalog_addenda/README.md); completed before inspecting new transfer results, without triggering new experiments or rewriting the frozen catalog.
