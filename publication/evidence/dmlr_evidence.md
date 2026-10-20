# DMLR evidence synthesis for the integrated PermWorld paper

This document synthesizes archived evidence across the entire DMLR line. It does not run models or pool stages into a larger statistical sample. The accompanying `dmlr_claims.csv` is the claim ledger: each row names its evidential status, unit of repetition, quantitative anchor, limitations, and exact current artifact hashes. `dmlr_recompute_checks.json` records independent arithmetic checks from raw selected results.

## Defensible contribution

PermWorld turns mathematical source-task composition into an auditable learning experiment: exact task definitions and answer generators, explicitly typed single-task and joint relations, transformation-orbit input isolation, paired source/target training, and source-learning diagnostics. The empirical contribution is conditional evidence that task composition and training budget affect transfer; the archive also establishes limits on that evidence. It is not a theorem that mathematical relatedness improves transfer, a general ranking of task families, or a scaling law.

The broad line matters. The original fixed-point pilot exposed negative means; protocol and source diagnostics did not remove them; four-target exploration showed heterogeneous effects; background exploration motivated two fixed hypotheses; the five-unit confirmation did not replicate the fixed-point direction and left the LDR effect uncertain. The platform then made richer task families and full-answer supervision usable, followed by four-target task-count curves and a new fixed-composition budget comparison. A paper centered only on the latest positive LIS interaction would omit both the resource contribution and important negative evidence.

## Evidence stages and units

| Stage | Status | Actual unit / scope | Critical qualification |
|---|---|---|---|
| pilot_v1 | Exploratory | 2 paired units; 1 target; 64/256/1024 examples | Fixed-endpoint, legacy single-answer-token accuracy |
| pilot_v2 | Diagnostic exploration | Same 2 units; 256 examples; adaptation selection and source continuation | Reused development data; not a new independent replication |
| pilot_v3 | Multi-target exploration | 3 paired units; 4 targets; 256 examples | Historical weights reused where compatible; targets are not extra units |
| pilot_v4 | Background exploration | Same 3 blocks; 2 selected targets; 3 backgrounds | Shared A and reused weights do not create extra repeats |
| pilot_v5 | Frozen confirmation | 5 new paired units; 2 fixed background contrasts | Report separately from old 3 units; retain non-replication and uncertainty |
| platform v0.1 | Resource and development calibration | Registry/certificates; 6 representatives x 2 units | Calibration has no model test forwards and no transfer experiment |
| task-count v1 | Frozen finite comparison motivated by exploration | 3 new units; 4 targets; 2 chains; k=2/4/8/16 | 24 source models reused for108 target fits; composition changes with k |
| task-count v2 | Prospectively frozen budget follow-up | 3 new units; primary LIS, auxiliary FP; 2 chains; k=2/16; 4M/32M | 24 source models,54 target fits; shared models and baselines are not repetitions |

Pilot stages use the legacy 188-token vocabulary and answer-token accuracy at n=10. The platform and task-count stages use a 203-token versioned sequence protocol, per-example answer-plus-EOS mean cross-entropy, and complete generated-answer-plus-EOS accuracy; the task-count lengths are10/12. These protocols and datasets are different. Do not concatenate their accuracy values into a single longitudinal performance curve, interpret rerun checkpoints on new tests as new trained repeats, or sum all stage counts as an independent sample size.

## Findings that must survive paper compression

1. **Original and multi-target exploration.** At256 examples, pilot_v1 averages A/B/C=84.01/68.26/75.51%. In v3, mean B-A is-10.12pp for fixed_points,+2.59 for exceedances,-1.53 for left_to_right_minima,+11.20 for longest_decreasing_run. Every target's B-C contrast changes sign across the three seeds. Source learning changes with composition; exposure and optimization differences prevent interpreting these as pure interference or synergy.

2. **Exploration did not become blanket confirmation.** V4's fixed-point deficiencies interaction was+4.36pp and its LDR recoils interaction-12.00pp. In five new v5 units, FP is-3.02pp with95% paired-t interval[-22.57,+16.54], so the prespecified positive mean direction did not replicate. LDR is-13.15pp with interval[-28.89,+2.58], matching direction but remaining uncertain. Two-contrast simultaneous intervals are wider. Both results must accompany any v4 background-effect figure; they are not evidence of absence of all background effects.

3. **Resource contribution has auditable counting and correctness.** The registry retains48 legacy endpoints and adds6, giving54 endpoints and48 canonical mathematical templates. It declares62 instances and materializes56 unary non-alias instances. The three focus families contain16 templates/24 declared instances/22 nonconstant instances at the default lengths and parameters, with one input encoding. These are not independent mathematical capabilities. The certificate records272,741 independent label comparisons and445,581 new relation checks on6,105 fixtures;29 historical relation certificates are reused without rerunning old formulas. Explicit formulas, finite verification, empirical association and unchecked claims remain distinct.

4. **Learnability is uneven and measurement-specific.** Fixed-budget two-seed endpoint development means are42.84% cycle_count,22.53% longest_cycle,86.26% LIS,25.26% length-three increasing-subsequence count,99.74% components and99.02% largest_component. Block majority is77.08%; cycle_count atn12 is35.74% against36.72% majority. All lengths occur in training; these are development calibration, not held-out-length generalization or new test results.

5. **The finite task-count curve depends on what is trained and for how long.** At4M tokens in task-countv1, LIS k16-k2 is negative in all three repeats of both chains, despite positive mean transfer over A. The four-target curves retain cycle-count nonbenefit, small component gains and unstable fixed-point gains. In the new task-countv2 units, the LIS interaction[(k16-k2)_32M-(k16-k2)_4M] is+17.94pp(alpha) and+6.54pp(beta), positive in all three repeats of each chain; only alpha establishes positive k16-k2 in every32M repeat. Auxiliary FP interactions have opposite signs,+41.41pp and-9.08pp. This is a conditional budget/composition finding, not monotonic universal benefit.

6. **A larger budget is not proportionally more independent data.** Both task-countv2 budgets use the same8192 source objects per repeat. More steps can increase a task's coverage of that fixed pool and repeat previously seen task-input pairs. Uniform sample shares are not equal token shares for different answer lengths. Both budgets start from paired random weights with fresh optimizers, but the absolute-token LR trajectories differ. Thus updates, learning-rate trajectory and fixed-pool exposure are not independently identified.

## Mathematical relations and mechanism boundaries

The exact position partition is fixed_points(p)+exceedances(p)+deficiencies(p)=n. Together with exceedances(p)=deficiencies(p^-1), it gives fixed_points(p)=n-deficiencies(p)-deficiencies(p^-1). The first form needs two same-input answers; the second needs two transformed-input queries. The v5 fixed-point deficiencies groups do not contain exceedances, and both B and C have the second relation. Declared relationships are legitimate properties of tasks, not automatically data leakage. No explicit transformed-query supervision or mediator intervention demonstrated that a Transformer implemented these formulas.

Single-task accuracy is not a sufficient proxy for target benefit. V5's related fixed-point candidate is near99% development in both backgrounds while its primary replacement interaction fails directional replication. In task-countv2, alpha-k2 LDS improves from84.57% to99.09% development while mean LIS target accuracy declines. Source-learning records constrain simple explanations, but do not distinguish formula use, optimization, representation changes, answer-distribution effects, or adaptation plasticity.

## Claim ledger

### D01 — The original held-out fixed-point pilot did not show uniformly beneficial related-source pretraining.

**Status:** exploratory. **Units:** Two paired units (17,42); sample sizes and A/B/C share their unit's inputs/initialization.

At 256 samples: A 84.01%, B 68.26%, C 75.51%; B-A=-15.75 pp, C-A=-8.50 pp. At 1024: A 98.85%, B 98.36%, C 97.92%.

**Supported:** The raw two-seed sample-size curves retain negative and near-zero average gains.

**Limit:** One target and two seeds; a negative average does not mean every seed is negative. Fixed update budget causes unequal reuse as sample count changes.

Evidence: [experiments/permworld_pilot/runs/pilot_v1/results_all.csv](../../experiments/permworld_pilot/runs/pilot_v1/results_all.csv); [experiments/permworld_pilot/runs/pilot_v1/summary.csv](../../experiments/permworld_pilot/runs/pilot_v1/summary.csv); [PILOT_REPORT.md](../../PILOT_REPORT.md). Exact SHA-256 values are in the CSV.

### D02 — Development selection and the historical doubled source budget did not remove mean negative transfer.

**Status:** exploratory_diagnostic. **Units:** Same two historical units 17/42; no new independent repeats.

Development-selected 2400: B-A=-17.26 pp, C-A=-8.74 pp, B-C=-8.52 pp; 4800: B-A=-10.55 pp, C-A=-12.87 pp, B-C=+2.32 pp.

**Supported:** Failure to load/reset weights is not supported by engineering audits; negative means persist under the selected protocol.

**Limit:** Historical continuation uses low terminal LR and is not the fresh relative-budget comparison in task-count v2; reused development inputs and hypothesis remain exploratory.

Evidence: [experiments/pilot_v2_diagnostics/artifacts/protocol_comparison.csv](../../experiments/pilot_v2_diagnostics/artifacts/protocol_comparison.csv); [experiments/pilot_v2_diagnostics/artifacts/protocol_comparison_mean.csv](../../experiments/pilot_v2_diagnostics/artifacts/protocol_comparison_mean.csv); [experiments/pilot_v2_diagnostics/DIAGNOSTIC_REPORT.md](../../experiments/pilot_v2_diagnostics/DIAGNOSTIC_REPORT.md). Exact SHA-256 values are in the CSV.

### D03 — The nominally matched candidate pair differed markedly in learnability under multitask training.

**Status:** exploratory_diagnostic. **Units:** Two paired source-model units; source records within units are not independent experiments.

Mean source development accuracy at 2400: 77.98% versus 45.12%; at 4800: 89.31% versus 51.86%; source-histogram TV 0.0443/0.0520.

**Supported:** Similar marginal answer distributions do not ensure similar learning curves.

**Limit:** These are mixture-learning differences, not intrinsic difficulty or a causal isolation of relation effects.

Evidence: [experiments/pilot_v2_diagnostics/artifacts/source_learning_table.csv](../../experiments/pilot_v2_diagnostics/artifacts/source_learning_table.csv); [experiments/pilot_v2_diagnostics/artifacts/replacement_matching_summary.csv](../../experiments/pilot_v2_diagnostics/artifacts/replacement_matching_summary.csv); [experiments/pilot_v2_diagnostics/DIAGNOSTIC_REPORT.md](../../experiments/pilot_v2_diagnostics/DIAGNOSTIC_REPORT.md). Exact SHA-256 values are in the CSV.

### D04 — Transfer varies across held-out targets, and the candidate replacement contrast changes sign across seeds for all four targets.

**Status:** exploratory. **Units:** Three paired units 17/42/73; first two fixed-point units reuse historical training; four targets are not 12 independent repeats.

Mean B-A: fixed_points -10.12 pp; exceedances +2.59 pp; left_to_right_minima -1.53 pp; longest_decreasing_run +11.20 pp. B-C signs are mixed for all four targets.

**Supported:** The fixed-point negative effect is not a universal outcome across the tested mathematical targets.

**Limit:** LDR A/C are near majority; eligibility at 8192 single-task examples does not establish learnability at 256. All tested direct candidate relations have identity answer maps.

Evidence: [experiments/pilot_v3_multitarget/artifacts/results_selected.csv](../../experiments/pilot_v3_multitarget/artifacts/results_selected.csv); [experiments/pilot_v3_multitarget/artifacts/paired_deltas.csv](../../experiments/pilot_v3_multitarget/artifacts/paired_deltas.csv); [experiments/pilot_v3_multitarget/PILOT_V3_REPORT.md](../../experiments/pilot_v3_multitarget/PILOT_V3_REPORT.md). Exact SHA-256 values are in the CSV.

### D05 — Single-task and mixture references show both degradation and improvement, with exposure confounding.

**Status:** exploratory_diagnostic. **Units:** Three units; 84 deduplicated seed-task-model-state records share inputs/tasks and are not 84 independent repeats.

84 deduplicated source records: 60 declines >=5 pp, 6 improvements >=5 pp, 18 smaller gaps; successions seed 73 improvement +43.55 pp retained.

**Supported:** Composition accompanies different source learning; low performance cannot be summarized only by relatedness labels.

**Limit:** The references do not isolate interference or synergy because exposure, alternation and LR histories differ.

Evidence: [experiments/pilot_v3_multitarget/artifacts/source_learning_effects_unique.csv](../../experiments/pilot_v3_multitarget/artifacts/source_learning_effects_unique.csv); [experiments/pilot_v3_multitarget/artifacts/single_vs_multi_sources.csv](../../experiments/pilot_v3_multitarget/artifacts/single_vs_multi_sources.csv); [experiments/pilot_v3_multitarget/PILOT_V3_REPORT.md](../../experiments/pilot_v3_multitarget/PILOT_V3_REPORT.md). Exact SHA-256 values are in the CSV.

### D06 — The same candidate-source replacement has different observed effects across background sets.

**Status:** exploratory. **Units:** Three paired blocks 17/42/73; reuse from v3; 42 unique selected models, not 54 independent display rows.

FP deficiencies-v3 interaction +4.36 pp (seeds +1.42,+0.34,+11.33); LDR recoils-v3 -12.00 pp (seeds -20.07,+2.00,-17.92).

**Supported:** Observed background-dependent variation motivated the two frozen v5 hypotheses.

**Limit:** Case/background selection is exploratory; v3/v4 reused units must not be counted as independent replications. Must present v5 follow-up beside these effects.

Evidence: [experiments/pilot_v4_backgrounds/artifacts/results_selected.csv](../../experiments/pilot_v4_backgrounds/artifacts/results_selected.csv); [experiments/pilot_v4_backgrounds/artifacts/background_interactions.csv](../../experiments/pilot_v4_backgrounds/artifacts/background_interactions.csv); [experiments/pilot_v5_confirmation/exploratory/exploratory_paired_differences.csv](../../experiments/pilot_v5_confirmation/exploratory/exploratory_paired_differences.csv). Exact SHA-256 values are in the CSV.

### D07 — The prespecified positive fixed-point background interaction did not reproduce in mean direction.

**Status:** confirmatory_nonreplication. **Units:** Five new paired units 101/211/307/401/503; old three exploratory units are not pooled.

Mean -3.02 pp, SD 15.75; 2/5 direction matches; 95% paired-t CI [-22.57,+16.54]; two-contrast simultaneous CI [-27.63,+21.60].

**Supported:** The exploration's positive replacement interaction is not corroborated by the new point estimate.

**Limit:** Wide intervals include both directions; failure of directional replication is not proof of zero background effect. Five units and finite-pool dependence limit interval interpretation.

Evidence: [experiments/pilot_v5_confirmation/artifacts/confirmation_paired_differences.csv](../../experiments/pilot_v5_confirmation/artifacts/confirmation_paired_differences.csv); [experiments/pilot_v5_confirmation/artifacts/confirmation_summary.csv](../../experiments/pilot_v5_confirmation/artifacts/confirmation_summary.csv); [experiments/pilot_v5_confirmation/stats_plan.json](../../experiments/pilot_v5_confirmation/stats_plan.json). Exact SHA-256 values are in the CSV.

### D08 — The LDR confirmation point estimate matches the exploratory direction but does not establish a stable interaction.

**Status:** confirmatory_uncertain. **Units:** Five new paired units, shared across targets; not 10 independent target replicates.

Mean -13.15 pp, SD 12.67; 4/5 direction matches; 95% CI [-28.89,+2.58]; simultaneous CI [-32.97,+6.66].

**Supported:** A large negative point estimate remains uncertain under the prespecified paired analysis.

**Limit:** Both interval types include zero; LDR A 53.56% and recoils B/C 53.50%/52.38% are near 53.67% majority. No claim of reliable universal background dependence.

Evidence: [experiments/pilot_v5_confirmation/artifacts/confirmation_paired_differences.csv](../../experiments/pilot_v5_confirmation/artifacts/confirmation_paired_differences.csv); [experiments/pilot_v5_confirmation/artifacts/confirmation_summary.csv](../../experiments/pilot_v5_confirmation/artifacts/confirmation_summary.csv); [experiments/pilot_v5_confirmation/PILOT_V5_REPORT.md](../../experiments/pilot_v5_confirmation/PILOT_V5_REPORT.md). Exact SHA-256 values are in the CSV.

### D09 — Same-input joint derivability and transformed-input joint derivability are distinct design facts.

**Status:** resource_verified_relations. **Units:** Deterministic mathematical audits, not learning replicates.

fixed_points+exceedances+deficiencies=n; exceedances(p)=deficiencies(p^-1). All 12 same-input groups have counterexamples; 7/12 groups recover targets under 32 transformed queries, and five have insufficiency witnesses.

**Supported:** Both B/C fixed-point deficiencies groups can recover FP from D(p),D(p^-1), but the position partition alone cannot recover FP from one D label.

**Limit:** Valid relations are not instance leakage; no explicit paired-query training or mechanism demonstration was performed. Counterexamples restrict only the declared query model.

Evidence: [experiments/pilot_v5_confirmation/mathematics/MATH_AUDIT.json](../../experiments/pilot_v5_confirmation/mathematics/MATH_AUDIT.json); [experiments/pilot_v5_confirmation/mathematics/audit/TARGET_JOINT_DERIVABILITY.csv](../../experiments/pilot_v5_confirmation/mathematics/audit/TARGET_JOINT_DERIVABILITY.csv); [experiments/pilot_v5_confirmation/mathematics/audit/RELATION_INVENTORY.csv](../../experiments/pilot_v5_confirmation/mathematics/audit/RELATION_INVENTORY.csv). Exact SHA-256 values are in the CSV.

### R01 — The unified platform distinguishes endpoints, canonical templates, instances and encoding variants.

**Status:** resource. **Units:** Deterministic registry inventory, not statistical replicates.

48 legacy + 6 new = 54 endpoints; 48 canonical mathematical templates; 62 declared instances; 56 materialized unary non-alias instances; one input encoding.

**Supported:** Machine-readable definitions, parameters, ranges, complexities, encoding and relation status are available under one interface.

**Limit:** 48 templates are not 48 mathematically independent capabilities; binary pools are not generated or certified by this release.

Evidence: [experiments/platform_v01/registry/TASK_COUNTS.json](../../experiments/platform_v01/registry/TASK_COUNTS.json); [experiments/platform_v01/registry/TASK_REGISTRY.json](../../experiments/platform_v01/registry/TASK_REGISTRY.json); [experiments/platform_v01/registry/TASK_INSTANCES.csv](../../experiments/platform_v01/registry/TASK_INSTANCES.csv). Exact SHA-256 values are in the CSV.

### R02 — The three focus families provide differentiated tasks while explicitly retaining constants and aliases.

**Status:** resource. **Units:** Deterministic function classes under named relations.

Focus: 16 templates / 24 instances; 22 nonconstant instances; 23 same-input classes; 20 registered answer-preserving transformation classes (19 nonconstant).

**Supported:** Cycle family: 6 templates / 8 instances; subsequences: 5/9; blocks: 5/7. I1=D1=n constants remain visible.

**Limit:** Other lengths/parameters can be degenerate; known transformation classes do not classify arbitrary answer maps or all joint relations.

Evidence: [experiments/platform_v01/registry/TASK_COUNTS.json](../../experiments/platform_v01/registry/TASK_COUNTS.json); [experiments/platform_v01/mathematics/EQUIVALENCE_AUDIT.json](../../experiments/platform_v01/mathematics/EQUIVALENCE_AUDIT.json). Exact SHA-256 values are in the CSV.

### R03 — Exact generators have independent algorithm checks and explicit relation evidence.

**Status:** resource_verified_correctness. **Units:** 6105 deterministic fixtures, not 6105 training or statistical units.

272741 independent label comparisons; 445581 new formula/bound pointwise checks; 45 new relations + 29 historical relations reused; zero historical formula rechecks.

**Supported:** Independent cycle, subsequence, RSK and block algorithms verify oracle implementation; proof and finite-check status remain separate.

**Limit:** Finite verification is not a general proof; mathematical correctness does not imply model learnability.

Evidence: [experiments/platform_v01/mathematics/CERTIFICATE.json](../../experiments/platform_v01/mathematics/CERTIFICATE.json); [experiments/platform_v01/mathematics/RELATIONS.json](../../experiments/platform_v01/mathematics/RELATIONS.json); [experiments/platform_v01/mathematics/HISTORICAL_REUSE.json](../../experiments/platform_v01/mathematics/HISTORICAL_REUSE.json). Exact SHA-256 values are in the CSV.

### R04 — The versioned sequence protocol supports large and structured answers without silent truncation.

**Status:** resource_protocol. **Units:** Engineering and data audits, not statistical replicates.

9216 fixed objects / 516096 exact labels; n=30,k=15 integer 155117520 round trip; 32 canonical I3 labels >=100; all focus instances fit a 128-token context.

**Supported:** Delimited base-100 integers and nested lists are explicitly encoded; oversized legacy examples are visibly rejected.

**Limit:** The six scalar representatives do not calibrate structured-output learning; legacy and platform accuracies use different protocols.

Evidence: [experiments/platform_v01/PLATFORM_REPORT.md](../../experiments/platform_v01/PLATFORM_REPORT.md); [experiments/platform_v01/dataset/manifest.json](../../experiments/platform_v01/dataset/manifest.json); [experiments/platform_v01/registry/MODEL_INTERFACE_COMPATIBILITY.csv](../../experiments/platform_v01/registry/MODEL_INTERFACE_COMPATIBILITY.csv). Exact SHA-256 values are in the CSV.

### R05 — Six frozen single-task representatives span sharply different attainable learning levels under one budget.

**Status:** resource_development_calibration. **Units:** Two paired seed/data units 617/829; six tasks sharing this protocol are not 12 independent replicates.

Mean endpoint development: cycle_count 42.84%; longest_cycle 22.53%; LIS 86.26%; I3 count 25.26%; components 99.74%; largest_component 99.02%.

**Supported:** All six retained; block majority is 77.08%, so high accuracy needs baseline context; cycle_count at n=12 has development accuracy 35.74% versus 36.72% majority.

**Limit:** Development only: no unseen-length extrapolation, transfer or entire-family ranking; 12 runs are six tasks x two units.

Evidence: [experiments/platform_v01/calibration/results_by_seed.csv](../../experiments/platform_v01/calibration/results_by_seed.csv); [experiments/platform_v01/calibration/task_summary.csv](../../experiments/platform_v01/calibration/task_summary.csv); [experiments/platform_v01/calibration/FROZEN_CALIBRATION.json](../../experiments/platform_v01/calibration/FROZEN_CALIBRATION.json). Exact SHA-256 values are in the CSV.

### T01 — At 4M tokens, the tested count curves do not show universally monotonic transfer improvement.

**Status:** exploratory_frozen_comparison. **Units:** Three new paired units 1103/2207/3301; chains, targets and shared baselines are not extra replicates.

24 pretrained models / 108 target fits; LIS k16-k2 mean -5.47 pp alpha / -6.02 pp beta, negative in all three repeats of each chain; all eight cycle-count means are at or below A.

**Supported:** LIS retains positive mean gains over A while losing benefit as k increases; modest component gains and fixed-point chain/seed variation are preserved.

**Limit:** Changing k changes source composition, relations and exposure; three paired units cannot establish a pure task-count causal effect or scaling law.

Evidence: [experiments/task_count_v1/raw_test_results.csv](../../experiments/task_count_v1/raw_test_results.csv); [experiments/task_count_v1/paired_k_contrasts.csv](../../experiments/task_count_v1/paired_k_contrasts.csv); [experiments/task_count_v1/REPORT.md](../../experiments/task_count_v1/REPORT.md). Exact SHA-256 values are in the CSV.

### T02 — The fixed-point average peak is unstable, and many source endpoints remain underlearned.

**Status:** exploratory_diagnostic. **Units:** Three paired units; 180 source-task/model records are not 180 independent replicates.

FP alpha-k8 gains +26.17,+12.01,-10.94 pp (mean +9.08, SD 18.73); 61/180 source endpoints are at or below their training-derived majority baseline.

**Supported:** The mean peak is not a stable optimum; limited source learning motivates a frozen budget follow-up.

**Limit:** Selecting the best k after inspecting outcomes would overstate this exploration; source-accuracy/transfer association does not establish a mechanism.

Evidence: [experiments/task_count_v1/paired_deltas.csv](../../experiments/task_count_v1/paired_deltas.csv); [experiments/task_count_v1/source_learning.csv](../../experiments/task_count_v1/source_learning.csv); [experiments/task_count_v1/REPORT.md](../../experiments/task_count_v1/REPORT.md). Exact SHA-256 values are in the CSV.

### T03 — A larger relative-progress budget reverses the LIS count disadvantage in alpha and reduces it in beta.

**Status:** prospectively_frozen_followup. **Units:** Three new paired units 4103/5207/6301; two chains share these units, not six independent interactions.

Alpha D4=-6.18 pp, D32=+11.75, interaction +17.94 (SD 7.89); beta D4=-5.79, D32=+0.75, interaction +6.54 (SD 5.96); both interactions are positive in all three repeats.

**Supported:** The direction of k16-k2 depends on budget in the prespecified alpha case; beta shrinks the disadvantage without positive D32 in all three repeats.

**Limit:** The hypothesis comes from v1, with limited new repeats and composition coverage; different absolute-token LR paths mean budget and trajectory are not separately identified.

Evidence: [experiments/task_count_v2/raw_test_results.csv](../../experiments/task_count_v2/raw_test_results.csv); [experiments/task_count_v2/paired_budget_interactions.csv](../../experiments/task_count_v2/paired_budget_interactions.csv); [experiments/task_count_v2/budget_interaction_summary.csv](../../experiments/task_count_v2/budget_interaction_summary.csv); [experiments/task_count_v2/FROZEN_PROTOCOL.json](../../experiments/task_count_v2/FROZEN_PROTOCOL.json). Exact SHA-256 values are in the CSV.

### T04 — The auxiliary fixed-point budget interaction has opposite signs across source chains.

**Status:** prospectively_frozen_auxiliary. **Units:** Three new paired units, shared with LIS; the auxiliary target does not double the independent repeat count.

Alpha interaction +41.41 pp (SD 3.27), beta -9.08 (SD 11.55); alpha-k16-32M test accuracy 88.44%, +23.63 pp over A; beta-k16-32M 43.39%, -21.42 pp over A.

**Supported:** A larger budget does not generally eliminate negative transfer; seven of eight fixed-point conditions retain negative mean gains.

**Limit:** Relation availability and improved source learning do not prove formula use; the auxiliary outcome must not replace the primary LIS question.

Evidence: [experiments/task_count_v2/raw_test_results.csv](../../experiments/task_count_v2/raw_test_results.csv); [experiments/task_count_v2/paired_budget_interactions.csv](../../experiments/task_count_v2/paired_budget_interactions.csv); [experiments/task_count_v2/budget_interaction_summary.csv](../../experiments/task_count_v2/budget_interaction_summary.csv). Exact SHA-256 values are in the CSV.

### T05 — The budget comparison increases repeated training on a fixed pool, with audited pairing and budget-specific LR paths.

**Status:** resource_execution_audit. **Units:** 24 source models and 54 target fits across three repeat units; not 78 independent repetitions.

431999672 actual pretraining tokens; 82944000 adaptation tokens; 12 pairs verified; 135 formal checkpoints checked; 54 development selections; 324 test CSV rows; zero new test forwards in the audit.

**Supported:** 32M extends the sampled-example prefix but starts from random weights, not 4M weights; per-task coverage and repeated exposure counts are explicit.

**Limit:** Eightfold total tokens are not eightfold independent data or strict per-task exposure matching; absolute learning rates differ from the first update.

Evidence: [experiments/task_count_v2/RUN_AUDIT.json](../../experiments/task_count_v2/RUN_AUDIT.json); [experiments/task_count_v2/audit_budget_pairing.csv](../../experiments/task_count_v2/audit_budget_pairing.csv); [experiments/task_count_v2/task_exposures.csv](../../experiments/task_count_v2/task_exposures.csv); [experiments/task_count_v2/learning_rate_schedule.csv](../../experiments/task_count_v2/learning_rate_schedule.csv). Exact SHA-256 values are in the CSV.

## Recompute

```powershell
.venv/Scripts/python.exe publication/evidence/build_dmlr_evidence.py
```

This rebuild performs 8 stage-level arithmetic checks using archived raw rows, writes only these publication evidence files, and performs no model inference, training, task selection, or historical-file updates.
