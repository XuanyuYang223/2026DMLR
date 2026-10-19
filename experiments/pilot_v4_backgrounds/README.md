# Pilot v4: a limited background-composition interaction experiment

This round compares only two exploratory cases selected from v3 results: `fixed_points` and `longest_decreasing_run`. Each retains its v3 B/C candidate pair in three prespecified common-task backgrounds, to study how B−C changes with background. All v1/v2/v3 code, configuration, data, and results were preserved during the experiment. Work stops after this round pending further guidance.

Final report: [PILOT_V4_REPORT.md](PILOT_V4_REPORT.md). `summarize.py` reads saved CSV/JSON and v3 single-task references only, without model imports, training, or inference. It refuses a completion report without a complete `TEST_COMPLETE.json`.

## Compositions

| Target | B candidate slot0 | C candidate slot0 | g0 common slots 1/2/3 |
| --- | --- | --- | --- |
| fixed_points | anti_fixed_points | successions | descents / peaks / left_to_right_maxima |
| longest_decreasing_run | longest_increasing_run | left_to_right_maxima | descents / peaks / successions |

Three backgrounds are fixed: `g0_v3` retains the original composition; `g1_recoils` replaces only common slot1 descents with recoils; `g2_deficiencies` replaces only common slot2 peaks with deficiencies. Within each background, B/C differ only in slot0. New backgrounds were frozen from definitions and existing single-task records before new transfer results. No claim is made of perfectly matched difficulty, answer distributions, or internal relations. C does not mean mathematically unrelated. Candidates/common tasks, known relations, the data platform, and exact-reuse evidence are in [TASK_CATALOG.csv](TASK_CATALOG.csv),[DATA_PLATFORM.md](DATA_PLATFORM.md), and [REUSE_AUDIT.md](references/REUSE_AUDIT.md).

Frozen [BACKGROUND_MATCHING.csv](catalog/BACKGROUND_MATCHING.csv) records actual differences. g1 source-training answer TV is 0.00989–0.01428, but step1200 recoils development accuracy 62.79%/74.02%/93.07% is below corresponding descents 99.90%/99.71%/99.80%. g2 TV is 0.76013–0.76526, with very different peaks/deficiencies answer distributions and early learning curves. Equal population distributions, similar finite-pool histograms, and matched learning processes are distinct requirements. Backgrounds were prespecified, not searched for optimal transfer using these records.

Retained B/C candidates are also imperfectly matched. Formal summarization creates `artifacts/candidate_matching.csv` from frozen `catalog/SOURCE_ANSWER_DISTRIBUTIONS.csv` and `catalog/SOURCE_REFERENCE_CURVES.csv`, covering candidate answer TV, single-task step1200/4800 development accuracy, and early-curve gaps for both targets and all three blocks, without new training results. Original selection evidence is [v3 replacement_matching.csv](../pilot_v3_multitarget/artifacts/replacement_matching.csv): fixed_points preserves its historical anchor; longest_decreasing_run preserves v3's seed17 development selection. This table discloses remaining confounding differences without reselecting compositions.

Both targets and their related sources satisfy `target(p)=related(C(p))`, where `C(p)_i=n+1-p_i`, with identity answer maps. These two cases selected from v3 observations do not represent the full task library or all relation types.

A definition-level scientific review was separately saved after composition freezing and before new transfer results in [catalog_addenda/README.md](catalog_addenda/README.md), without changing frozen compositions/protocol. It proves `fixed_points(p)=n−D(p)−D(I(p))` and `anti_fixed_points(p)=n−D(C(p))−D(R(I(p)))`, where D=deficiencies. Thus both fixed_points g2 B/C contain a background task jointly related to the target across multiple inputs. One D label on one input remains insufficient; formal training supplies no paired queries or joint constraints. This does not establish model use of the relation.

g1 removes descents rather than adding a coexisting descents/recoils pair. LDR/LIR/descents/peaks share the local comparison-sign sequence, but their scalar answers do not mutually determine each other. g1 removes descents and g2 removes peaks, changing local computational structure and source-internal relations differently. Closure of declared edges is not a complete mathematical classification; definitions, proofs, bounds, and counterexamples are in the addendum.

## Pairing, budgets, and selection

- PermWorld length 10, one-line encoding, original Transformer with 4 layers, width256,8 heads, dropout0.1, vocabulary188, and tied embedding/output. FP32 weights/evaluation, BF16 training, TF32 disabled.
- Paired blocks 17/42/73 reuse identical v3 input pools and step0 weights. Each block has source train8192, source dev1024, target train256, target dev1024, and final new test 2048. Conditions, backgrounds, and both targets share block inputs, with labels computed by each mathematical task.
- Block labels are not data-generation RNG seeds: s17/s42 inputs were generated sequentially using RNG 20261018; s73 used RNG 2026101803. Initialization seeds are 17/42/73. Pretraining sampling/dropout seeds are 1017/1042/1073; adaptation seeds 2017/2042/2073. Blocks jointly vary input pools and initialization, so differences are not initialization effects alone. See `seed_provenance.json`.
- B/C pretraining uses 4800 updates, batch64, and exactly 16 examples per slot(25%). All backgrounds/conditions in a block share input indices and slot assignments. Each task receives 76800 exposures; actual unique-input counts and three stream-hash types are in `source_exposures.csv` and `sampling_slots.csv`. Repeated exposure is not distinct sample count.
- Source LR warms up for 100 steps to 3e−4, cosine-decays to 3e−5 by 2400, and remains constant through 4800. AdamW weight decay .01, betas .9/.999, eps1e−8, clip1. No budget extension is added.
- Every adaptation condition uses LRs 3e−4/1e−4/3e−5,800 steps each, batch64,100-step warmup, and cosine decay to 0.1 of initial LR. After strict source loading, the target's existing tied task-token row is restored to that block's step0 value and optimizer state is cleared. No new output head is added.
- Full train/dev evaluation occurs every 100 steps. Selection prioritizes highest dev answer accuracy, lowest dev answer CE, earliest step, then declared LR order; step0 is excluded. Every candidate trains all 800 steps, retaining best-dev and endpoint checkpoints.
- Answer accuracy uses prompt-only argmax over the full vocabulary. Answer/EOS each contribute half the training loss; EOS is recorded separately. The majority baseline predicts the mode from 256 target-training labels on each split, not the test mode.

A is trained/selected once per target/block, with `background=shared`, then displayed across three backgrounds. It receives no extra selection budget. There are 126 unique adaptation candidates(6 A units×3 LRs +36 B/C units×3 LRs),42 unique selected checkpoints, and 54 A/B/C display rows that repeat A rather than represent 54 independent runs. All 252 candidate best/endpoint new-test results are disclosed,84 per block.

The primary interaction is `(B−C)_later_background − (B−C)_earlier_background`, comparing g1−g0, g2−g0, and g2−g1 within each block. B−A, C−A, B−C differences and means/sample SDs are retained; SD is not a confidence interval. Backgrounds, targets, and paired blocks share structure, and all records cannot be treated as independent samples.

## Reuse, source-task references, and test isolation

Of 36 logical pretraining runs,12 g0 runs are exactly reused and 24 are new. Of 126 adaptation candidates,18 shared-A and 36 g0 B/C candidates are reused and 72 are new. Exact reuse checks ordered task tuples, pools, initialization, sampling, budgets, and weight hashes; identical unordered source sets are insufficient. Historical weights are reevaluated on new tests, not counted as new training repeats.

For each background, C has the same unordered task set across the two targets, with successions and left_to_right_maxima swapping slot0/slot3 and thus task-to-input assignment. Targets are absent from pretraining inputs; cross-target C weight differences cannot be interpreted as the target name influencing pretraining. These cases are not independently sampled source sets.

All actual source tasks reuse v3 single-task step1200/4800 curves from corresponding blocks; no new single-task training is performed. Single4800 and multitask 4800 match total updates/global LR trajectory, but single-task per-task exposure is 4 times larger and multitask adds task alternation. Single1200 matches only per-task exposure, with different LR history and total updates. Single/multitask differences are descriptive, not isolated interference/synergy effects.

Reused fixed_points g0 blocks 17/42 lack full train/dev metrics before 2400; only 2400,3000,3600,4200,4800 are available. Plots start at 2400 without fabricated early points. Other compatible historical/new curves record 0/600/1200/2400/3600/4800. Complete target train/dev curves retain 0…800 every 100 steps.

Only after 42 development selections and checkpoint hashes freeze are new tests generated using RNG 2026101805. Full inverse/reverse/complement orbits exclude all v1/v2/v3 inputs and v4 train/dev inputs; the three new test pools are also mutually isolated. Targets/backgrounds/conditions share test inputs within a block. All 252 candidate evaluations are disclosed after freezing, without test-based reselection.

## Reproduction commands

From repository root `F:\2026DMLR`, use the existing CUDA PyTorch environment. Exact versions are in final `artifacts/environment.json`:

```powershell
.venv/Scripts/python.exe -u experiments/pilot_v4_backgrounds/pilot_v4.py all
.venv/Scripts/python.exe experiments/pilot_v4_backgrounds/audit_final.py
.venv/Scripts/python.exe experiments/pilot_v4_backgrounds/summarize.py
.venv/Scripts/python.exe experiments/pilot_v4_backgrounds/audit_final.py
```

Completed stages are skipped via markers; frozen source/configuration/data/evidence hashes prevent silent changes. The final audit refresh includes reports, summaries, and plots in the manifest. Reports state byte-preservation success only if the final audit status is passed and a historical-file count is present; otherwise they mark it pending.

To rerun in an independent new output directory:

```powershell
.venv/Scripts/python.exe -u experiments/pilot_v4_backgrounds/pilot_v4.py all --out experiments/pilot_v4_backgrounds/replication
.venv/Scripts/python.exe experiments/pilot_v4_backgrounds/audit_final.py --out experiments/pilot_v4_backgrounds/replication
.venv/Scripts/python.exe experiments/pilot_v4_backgrounds/summarize.py --out experiments/pilot_v4_backgrounds/replication
.venv/Scripts/python.exe experiments/pilot_v4_backgrounds/audit_final.py --out experiments/pilot_v4_backgrounds/replication
```

New-output runs still reuse frozen catalog/reference evidence and compatible v3 weights under the protocol, without overwriting original outputs. Checkpoints remain local and hash-tracked, usually outside Git; retain v1/v2/v3 references when moving them. `catalog/build_catalog.py` and `references/reuse_inventory.py` write frozen source evidence and must run only before freezing or in a new experiment version, not regenerate original frozen files in place.

## Outputs

| File (in artifacts/ unless noted) | Contents |
| --- | --- |
| `PILOT_V4_REPORT.md`(experiment root) | Observations, protocol, interpretation limits, costs, and next steps in English |
| `results_seed17.csv` / `results_seed42.csv` / `results_seed73.csv` | 84 raw candidate best/endpoint rows per block |
| `results_all_candidates.csv` / `results_selected.csv` | All 252 rows /42 unique development-selected primary results |
| `results_display_ABC.csv` / `results_summary.csv` | 54 background-display rows (A repeated)/descriptive summaries over 3 blocks |
| `paired_deltas.csv` / `paired_deltas_summary.csv` | B−A, C−A, B−C by block and mean/SD |
| `background_interactions.csv` / `background_interactions_summary.csv` | 18 block-level Δ(B−C) entries and 6 interaction summaries |
| `selected_vs_endpoint.csv` | Selected-LR best-dev versus step800 endpoint train/dev/test, distinguishing early selection from later fitting |
| `target_curves.csv` / `late_training_diagnostics.csv` | 1134 train/dev-curve rows for all 126 candidates/fixed400→800 changes |
| `source_curves.csv` / `source_learning_table.csv` | Full source curves/144 endpoint rows with single-task references |
| `source_exposures.csv` / `sampling_slots.csv` | 144 actual task-exposure rows and block/slot index/input-stream verification |
| `common_source_BC.csv` | B/C differences for identical common mathematical tasks within a background |
| `source_background_slot_changes.csv` | Same-task source-learning changes across backgrounds; replaced-task slots have no difference computed |
| `interaction_source_context.csv` | Target interactions juxtaposed with candidate/unchanged-common-source changes, without causal correlation claims |
| `source_effect_counts.csv` / `source_effects_unique.csv` | Positive/negative single-versus-multitask changes, including seed/task/state deduplication |
| `candidate_matching.csv` | Per-block answer TV and single-task step1200/4800 development metrics for retained B/C pairs, using pre-freeze evidence only |
| `source_state_reuse.csv` | Ordered task lists and repeated model-state hashes of logical source runs |
| `compute_by_run.csv` / `compute_summary.csv` / `stage_timings.csv` | Separate new/reused costs and engine-stage wall times |
| `FROZEN_PROTOCOL.json` / `SELECTIONS_FROZEN.json` / `FINAL_AUDIT.json` | Frozen configuration, development selections, and final audit |
| `STATISTICS_AUDIT.json` | Independent recomputation of paired differences/interactions/means/SDs, with CSV hashes |
| `figures/*.png` | Block-level accuracies, composition contrasts, background interactions, and source/target learning curves |

New pretraining input tokens total 206,438,400(24×4800×64×28); new adaptation totals 103,219,200(72×800×64×28). Supervised tokens count answer/EOS positions. Evaluation tokens are excluded; training timers include full evaluation, saving, and some bookkeeping. Reused historical timers are not new compute, and stage wall time must not be added to training timers. Existing single-task references incur no new training cost.

With 256 samples and existing budgets, this round establishes neither sample-efficiency curves nor a scaling law. Negative transfer, no benefit, sign reversals, and insufficient training are retained, with no additional backgrounds, tasks, LRs, or budgets.

The final report uses observed magnitudes/counterexamples to discuss future coverage: invertible input transforms with identity/affine answer maps, equal distributions with different computation, same-input multi-source/multi-input joint relations, shared local structure without mutually determining scalars, one-way information relations, and other computational families. These are future design recommendations, not tasks or experiments added in this round.
