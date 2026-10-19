# PermWorld Data Platform v0.1 Delivery Report

The unified interface, registry, exact generators, versioned sequence protocol, fixed data, correctness certificates, and two-seed single-task calibration of six preselected representatives are complete for three task families. Additional training for the two previous background-interaction comparisons has stopped. pilot_v5's non-replication and uncertain conclusions remain unchanged; this round performed no transfer-combination search.

## Platform deliverables and effective counts

| Family | Focus templates | Default instances | Nonconstant instances at fixed length |
| --- | ---: | ---: | ---: |
| Cycle decomposition | 6 | 8 | 8 |
| Global subsequences | 5 | 9 | 7 |
| Permutation blocks | 5 | 7 | 7 |

All 48 legacy endpoints are retained, with 6 additions, for 54 in total. Excluding two parameter aliases and four representation-output endpoints leaves 48 canonical mathematical templates, not the same number of independent mathematical capabilities. The full catalog declares 62 instances; canonical data materialize 56 unary non-alias instances. One input encoding, one_line_v1, is exposed.

The focus families contain 16 templates/24 declared instances. I₁=D₁=n; both are retained and marked as constants at fixed length, leaving 23 same-input function classes and 22 nonconstant classes. Merging under registered answer-preserving transformations gives 20 classes/19 nonconstant classes; additionally allowing the components/global_descents add-one mapping gives 19/18. These counting conventions use explicit relations and do not claim mathematical independence. Witness inputs with different answers are retained for the other 275 instance pairs; distinct same-input functions can still have joint relations.

cycle_length_count(k=1,2,3) already corresponds to fixed_points/two_cycle_count/three_cycle_count; D₂=length and I₂=binom(n,2)−length. New endpoints and genuinely new computational capabilities must be counted separately. Existing tasks such as shortest_cycle are fully retained. The CPU example with k=4 is a runtime configuration and does not increase the template count.

The 22 nonconstant instances apply to this release's n=8/10/12 and default k=1/2/3. Small n and k>n introduce further degeneracies, so nonconstancy is not claimed at every legal length.

The [task registry](registry/TASK_REGISTRY.json) contains definitions, legal lengths, parameters, exact algorithms, output ranges/types, families, complexity, measured label-generation costs, and relation evidence. The [instance table](registry/TASK_INSTANCES.csv) and [effective counts](registry/TASK_COUNTS.json) are saved separately.

## Data, output protocol, and correctness

Fixed data cover n=8/10/12 and two data units, with 9,216 distinct permutations and 516,096 exact labels. Each unit has 1024 training, 256 development, and 256 reserved test inputs per length. Full inverse/reverse/complement orbit overlaps with v1–v5 and across this round's roles/units are rejected consistently; RNG replay, duplicate, and leakage checks passed. Proposals are uniform permutations, but the pools after orbit rejection are not claimed to sample uniform orbit classes or be strictly IID. Generation took 8.74 seconds, excluding subsequent independent verification.

The explicitly versioned answer protocol is permworld_sequence_v1. All 188 legacy token IDs are preserved, with the vocabulary extended to 203 entries. Large integers use delimited base-100 and structured answers use nested lists, without modulo or truncation. The n=30,k=15 count 155117520 has passed round-trip verification. All focus instances fit a 128-token context; some legacy structured/binary tasks exceed it at large n and are explicitly rejected by the reader. Training six scalar representatives does not substitute for calibrating all structured-output tasks.

Data-quality tables retain actual answer distributions, observed constants, and majority baselines. The canonical data contain 32 increasing-subsequence counts at k=3 that are ≥100, exercising the multi-token path. The smoke test additionally uses the explicit answer 220 to check variable-length labels, causal masks, padding, full autoregressive evaluation, save/resume, and a fresh optimizer. Development/test majority baselines derive from training labels. The data platform and calibration use different valid tie-breaks for tied structured classes; the six calibration representatives are unaffected.

The independent mathematical certificate exhausts S₁…S₇ and adds 192 fixed large-n inputs, totaling 6,105 fixtures. All 272,741 label comparisons and 445,581 pointwise checks of new formulas passed, taking about 10.71 CPU seconds. Independent checks use union-find for cycles, subset enumeration for subsequences, growth diagrams for RSK, and set-based cuts for permutation blocks. Finite checks and general formula proofs are registered separately. The 29 v5 relations reuse definition/certificate hashes only; historical formulas were re-verified 0 times.

The [relation list](mathematics/RELATIONS.json) records single-task mappings, joint queries, and input transformations separately, explicitly stating whether answers on different inputs are needed. Proofs/theorems, finite checks, empirical associations, and unchecked items have separate statuses. An unregistered relation is not evidence of being unrelated, and valid relations are not automatically leakage. Legacy binary-task interfaces remain callable, but this release does not generate or certify binary data pools.

## Limited single-task calibration results

Representatives were frozen before formal training: cycle count/longest cycle, LIS/length-3 subsequence count, and block count/largest block. Each task uses seeds 617 and 829, corresponding to data seeds 2026102001 and 2026102002. Each run has 2400 updates, batch size 64, 800 updates at each of three lengths, and one learning rate of 3e-4. The fixed endpoint is the primary reference; the development-best checkpoint is saved separately and is not presented as independent test performance. No task, seed, or budget was changed in response to performance.

The table reports fixed-endpoint development accuracy for the complete answer plus EOS. Baselines apply each length's training-majority answer to the development set. Lengths are pooled by sample count before averaging over the two seeds.

| Task | seed617 | seed829 | Two-seed mean | Mean majority baseline | Mean training accuracy | Above baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cycle_count | 43.75% | 41.93% | 42.84% | 32.68% | 73.88% | +10.16 pp |
| longest_cycle | 20.57% | 24.48% | 22.53% | 16.99% | 57.41% | +5.53 pp |
| lis_length | 87.50% | 85.03% | 86.26% | 41.80% | 97.62% | +44.47 pp |
| increasing_subsequence_count(k=3) | 26.04% | 24.48% | 25.26% | 4.36% | 46.61% | +20.90 pp |
| components | 99.74% | 99.74% | 99.74% | 77.08% | 99.97% | +22.66 pp |
| largest_component | 98.96% | 99.09% | 99.02% | 77.08% | 99.93% | +21.94 pp |

See the [calibration report](calibration/CALIBRATION_REPORT.md), [raw per-seed CSV](calibration/results_by_seed.csv), and [curve CSV](calibration/learning_curves.csv) for per-length performance, full train/dev curves, and endpoint versus development-best comparisons. All representatives are retained. Block tasks have high majority baselines, so their gains above baseline must be considered. For other tasks, moderate accuracy does not establish that a task is fully learned, even when well above baseline. Low training performance supports insufficient learning under the current configuration; train/dev gaps may additionally reflect generalization issues. These results cannot separate algorithmic difficulty, optimization, representation, and budget effects.

Specific observations: cycle_count and longest_cycle have train/dev gaps of about 31.04 and 34.88 percentage points, respectively, which cannot be attributed solely to failure to fit the training data. At n=12, cycle_count's mean development accuracy is 35.74%, slightly below the 36.72% majority baseline. LIS reaches 98.44% at n=8 and 73.24% at n=12. The length-3 subsequence count reaches only 10.94% at n=12, with training accuracy of just 33.15%, indicating substantial remaining learning limitations. Both block representatives reach roughly 99% and exceed baseline by about 22 percentage points, showing that this configuration can learn these two specific statistics, not that the whole family is easy. All three lengths occur in training, so these differences do not establish unseen-length extrapolation.

Train/dev curve figures: [cycle decomposition](calibration/figures/cycle_decomposition_curves.png), [global subsequences](calibration/figures/global_subsequence_curves.png), and [permutation blocks](calibration/figures/permutation_blocks_curves.png). Raw per-seed tables: [617](calibration/results_seed617.csv) and [829](calibration/results_seed829.csv).

![Endpoint and majority baseline by length](calibration/figures/endpoint_by_length.png)

Structured rsk_shape and other tasks not selected as representatives have no learning calibration in this round. [LEARNABILITY_INDEX.csv](LEARNABILITY_INDEX.csv) distinguishes calibrated/not-run instances explicitly. Two seeds are insufficient to rank entire task families. Reserved test objects received quality checks, but no model test forwards were performed. Development curves therefore do not establish new test performance, unseen-length extrapolation, transfer, or scaling conclusions.

## Costs, preservation, and closure

All 12 runs completed: 28,800 updates and 52,841,514 nonpadding training-input tokens. Total run time was 14.23 minutes, including full train/dev evaluation and checkpoint saving every 200 steps. The RTX5070 ran a 4-layer/width 256/8-head model with 3,244,288 parameters under the 203-token vocabulary. The architecture matches the legacy 188-token model, but its parameter space is not identical. Equal update budgets do not imply equal token costs. Actual per-run values are in [compute.csv](calibration/compute.csv); software details are in [ENVIRONMENT.json](ENVIRONMENT.json).

Final audits passed: 2,740 historical v1–v5 files were byte-preserved; 26 calibration checkpoints, including 2 initializations, were loaded and checked; 24 best/endpoint development forwards reproduced saved results; and 1248 curve records and 36864 input-exposure records were verified. Forward replay uses the frozen model and evaluator; independent mathematical algorithms separately establish label correctness. Full evidence is retained in the [platform audit](PLATFORM_AUDIT.json), [calibration audit](calibration/CALIBRATION_AUDIT.json), and [integration review](INTEGRATION_REVIEW.json).

Historical-file auditing ran in parallel with training. The platform audit's seconds include wall-clock time waiting for calibration to finish and must not be added to training time as a total. No v5 result was retrained or rewritten.

See the [README](README.md) for complete runnable commands and examples. Source, data, configurations, CSVs, figures, certificates, and environment records are archived. By project convention, checkpoints stay local and are excluded from Git. Regenerate registries/certificates and repeat training in new directories to avoid overwriting frozen evidence.

The next step should use fixed calibration to distinguish specific templates that are usable from those requiring separately frozen training-protocol diagnostics. Family names alone do not determine difficulty or independence. Future combination selection should exclude count inflation from constants/aliases and inspect the input conditions of single-task and joint relations. This round stops at platform v0.1 and limited calibration, without automatically adding background variants for the two previous targets or new transfer searches.
