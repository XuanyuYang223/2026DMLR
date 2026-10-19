# Pilot v3: multitarget composition diagnostics with 256 fixed samples

This independent directory retains a PermWorld comparison with four targets and three paired seeds. When conducted, this round left pilot_v1/pilot_v2 code, configurations, data, results, and checkpoints unchanged. Expansion stops after this round. Read [PILOT_V3_REPORT.md](PILOT_V3_REPORT.md) for results; if the engine has not completed, the summary script refuses to generate a completion report.

## Scope and design

Targets are `fixed_points`, `exceedances`, `left_to_right_minima`, and `longest_decreasing_run`, using existing mathematical definitions. Related sources are respectively `anti_fixed_points`, `deficiencies`, `right_to_left_minima`, and `longest_increasing_run`. They satisfy `target(p) = source(T(p))` under complement, inverse, reverse, and complement, respectively, with identity answer mappings. All four targets are excluded from every pretraining set. Fixed_points and exceedances share the positional-comparison family; the four targets are not four independent mathematical domains.

All four relations use identity answer mappings, without nontrivial mappings or multitask additive identities. The catalog retains those other relations, which this round excludes under learnability and scale constraints.

For every target, A starts randomly, B pretrains on four sources, and C replaces B's related source while keeping three common sources. Fixed_points retains the v1/v2 composition and successions replacement. Other candidate sets are declared in `DESIGN_FROZEN.json`, with final compositions in `FROZEN_PROTOCOL.json`. C does not mean complete mathematical unrelatedness.

For example, when exceedances C replaces deficiencies with recoils, common source descents satisfies `recoils(I(p)) = descents(p)` with recoils (the catalog's `descent_inverse`). Replacement also changes known relations within the source set; B/C differences cannot be attributed solely to the designated target–source relation.

Target eligibility uses seed17 single-task 4800-step source-development accuracy: at least 90%, and at least 15 percentage points above the training-mode development baseline. The replacement score is source-training answer-distribution TV + absolute 4800-step single-task development-accuracy difference + mean absolute development-accuracy difference at 600/1200/2400/3600/4800 steps. Choose the minimum, breaking ties by candidate order. Transfer tests select neither targets nor replacements. Candidate scores and residual difficulty/distribution differences are retained; a single candidate does not imply search improved matching.

The eligibility gate uses 8192 single-task training examples. Passing it does not mean A can learn with this round's 256 target examples and 800-step budget. The report distinguishes target eligibility from near-majority-baseline outcomes under small-sample conditions.

`TASK_CATALOG.csv` distinguishes mathematical tasks, parameter variants, and encoding/output-interface variants. It covers current PermWorld and existing interfaces from other worlds, while this round runs existing PermWorld scalar tasks only. `catalog/RELATION_AUDIT.csv` and `catalog/REPLACEMENT_RELATION_COUNTEREXAMPLES.csv` record relation checks; finite counterexamples reject only checked identities, not all mathematical relationships.

## Data, model, and budget

- Inputs are uniformly random length 10 permutations with one-line encoding. Source train/dev pools contain 8192/1024 examples; target train/dev/new-test pools contain 256/1024/2048. Seeds are 17/42/73.
- Within each seed, all targets and A/B/C share permutation inputs with exact labels regenerated for each definition. Seeds17/42 reuse original pools; seed73 has new inputs. Shared inputs, common sources, and even entire source compositions prevent treating cross-target outcomes as independent samples.
- New data and tests are isolated by full inverse/reverse/complement orbits. New tests are generated after 36 development selections freeze, excluding all v1/v2 inputs and v3 train/dev inputs. New test seeds are also isolated. Old tests are used only for exclusion, not selection.
- Original Transformer:4 layers,width 256,8 heads,dropout 0.1, vocabulary 188, tied token embedding/output weights. FP32 weights/evaluation, BF16 training, TF32 disabled. Exact configuration is in the frozen JSON.
- Source pretraining uses 4800 steps,batch 64, exactly 16 examples per task each step (25%). Learning rate warms up to 3e-4 over 100 steps, follows cosine to 3e-5 by 2400, then stays 3e-5 through 4800. No automatic extension follows. Each mixture processes 8,601,600 forward training input tokens and 614,400 supervised tokens; each source receives 76,800 exposures.
- Every actual source has three-seed single-task references using the same initialization, source pools, and 4800-step rate trajectory. Checkpoints at 1200/2400/4800 and full train/dev curves at 0/600/1200/2400/3600/4800 are saved. Single-task 4800 exposure is 4 times the mixture's per-task exposure; single-task 1200 matches exposure only, with a different rate trajectory, so it cannot alone identify causal interference.
- A/B/C each have three adaptation rates:3e-4,1e-4,3e-5. Every candidate runs 800 steps,batch 64, warmup 100, then cosine to 0.1 of the initial rate, processing 1,433,600 input and 102,400 supervised tokens. The same target/seed uses identical sample order; conditions share the maximum selection budget.
- AdamW:weight decay=.01, betas=(.9,.999),eps=1e-8; gradient clip=1. Training loss averages CE at answer/EOS positions. Answer accuracy is prompt-only full-vocabulary argmax; gold-conditioned EOS is separate. Majority baselines use the training mode, with accuracy measured separately per split.
- After loading source weights, adaptation restores only the target's existing tied task-token embedding/output row to same-seed step0. No head is added. Optimizer state is emptied. Full train/dev evaluation occurs every 100 steps. Ties follow highest development answer accuracy, lowest answer CE, earliest step, then declared rate order; step0 is ineligible.

Fixed_points seeds17/42 exactly reuse 4 v2 double-budget pretrained models and 18 adaptation candidates, verifying checkpoint hashes and configuration/data compatibility before reevaluation on v3's new test. These historical weights are not newly trained independent repetitions. The other 20 mixture pretrainings and 90 adaptation candidates are new. The single-task run count depends on frozen source sets; see the cost table. Logical source groups may repeat. `source_composition_reuse.csv` records ordered/unordered compositions and duplicate state hashes; compute counts actual execution.

These reused fixed_points source models lack full train/dev curves before 2400 steps. Figures retain only the original 2400-step endpoint and v2 continuation at 3000/3600/4200/4800. B/C curves begin at 2400, without fabrication or interpolation back to 0/600. Only new single-task and other multitask runs have unified 0/600/1200/2400/3600/4800 evaluations. Unselected replacement candidates retain single-task curves and 4800-step endpoints even below 90% development accuracy. Unselected candidates not used elsewhere generally have only the design seed17 reference.

## Reproduction

Use the existing CUDA PyTorch environment from repository root `F:\2026DMLR`; final environment versions are in `artifacts/environment.json`:

```powershell
.venv/Scripts/python.exe -u experiments/pilot_v3_multitarget/run_pipeline.py
.venv/Scripts/python.exe experiments/pilot_v3_multitarget/audit_final.py
.venv/Scripts/python.exe experiments/pilot_v3_multitarget/summarize.py
.venv/Scripts/python.exe experiments/pilot_v3_multitarget/audit_final.py
```

The wrapper runs `pilot_v3.py` prepare, smoke, design_references, and freeze, then `freeze_evidence.py` to save frozen evidence before single references, pretrain, adapt, select, and test. Completion markers skip finished stages; source/configuration hash checks prevent silent changes to frozen experiments. The lower-level `pilot_v3.py all` runs all engine stages, but the wrapper is required for the complete audit workflow so additional frozen evidence precedes later computation.

Rerun in a new directory, still reusing explicitly compatible historical fixed_points weights by design, without overwriting existing outputs:

```powershell
.venv/Scripts/python.exe -u experiments/pilot_v3_multitarget/run_pipeline.py --out experiments/pilot_v3_multitarget/replication
.venv/Scripts/python.exe experiments/pilot_v3_multitarget/audit_final.py --out experiments/pilot_v3_multitarget/replication
.venv/Scripts/python.exe experiments/pilot_v3_multitarget/summarize.py --out experiments/pilot_v3_multitarget/replication
.venv/Scripts/python.exe experiments/pilot_v3_multitarget/audit_final.py --out experiments/pilot_v3_multitarget/replication
```

`summarize.py` only reads saved CSV/JSON; it imports no model, runs no training or inference, and makes no new test-based selections. Audit→summary→audit lets the report cite audit status and includes reports, tables, and figures in the final manifest. Checkpoints remain local and hash-tracked, usually outside Git. Reuse paths point to v1/v2; preserve references or copy the full repository when relocating experiment directories.

The report claims historical byte preservation passed only when `FINAL_AUDIT.json` has `status=passed` and positive `old_files_byte_preserved`. Missing or other states are explicitly marked pending audit; experiment completion is not equated with final audit success.

## Output index

| File | Contents |
| --- | --- |
| `TASK_CATALOG.csv` | Definitions, input/output ranges, generation, relations, transformations, answer mappings; parameter/encoding variants distinguished |
| `PILOT_V3_REPORT.md` | Observations, limitations, and next-stage recommendations |
| `artifacts/results_seed17.csv` / `results_seed42.csv` / `results_seed73.csv` |72 raw new-test candidate best/endpoint results per seed |
| `artifacts/results_all_candidates.csv` / `results_selected.csv` |216 disclosed results /36 development-selected main results |
| `artifacts/results_summary.csv` | Three-seed means, sample SDs, and ranges per target/condition |
| `artifacts/paired_deltas.csv` / `paired_deltas_summary.csv` | Per-seed B−A,C−A,B−C and descriptive summaries |
| `artifacts/selected_vs_endpoint.csv` | Best development checkpoint versus 800-step endpoint at the selected LR |
| `artifacts/target_curves.csv` / `late_training_diagnostics.csv` | All candidate target train/dev curves / fixed 400→800-window changes |
| `artifacts/single_task_references.csv` | Full 1200/4800-step single-task train/dev metrics, length, and majority baseline |
| `artifacts/source_learning_all.csv` / `single_vs_multi_sources.csv` | All source curves / same-seed single/multitask comparisons |
| `artifacts/source_classification_summary.csv` | Predefined descriptive category counts; repeated source occurrences are not independent tasks |
| `artifacts/source_learning_effect_counts.csv` / `source_learning_effects_unique.csv` | Source-mixture improvement/decline counts relative to single-task training; complete signed differences deduplicated by seed×task×source state |
| `artifacts/common_source_differences.csv` | Common-source B/C development accuracy, CE, and differences |
| `artifacts/source_composition_reuse.csv` | Repeated source compositions and identical model-state hashes |
| `artifacts/replacement_matching.csv` | All predefined replacement candidates and residual matching differences |
| `artifacts/compute_by_run.csv` / `compute_summary.csv` / `stage_timings.csv` | New versus historical reused budgets and runtimes, with separate costs |
| `artifacts/FROZEN_PROTOCOL.json` / `SELECTIONS_FROZEN.json` | Full frozen configuration, compositions, and development selections |
| `artifacts/FINAL_AUDIT.json` / `artifact_manifest.json` | Protocol/leakage/weight/save checks and file hashes |
| `artifacts/figures/*.png` | Per-target/per-seed accuracy, paired differences, single/multitask curves, and selected target curves |

Single/multitask class thresholds were fixed in advance: single-task 4800-step development accuracy<90% means "Single task not fully learned under this configuration"; ≥90% with mixture accuracy at least 5 points lower means "Learned alone but degraded in multitask training"; otherwise, "Insufficient evidence of multitask degradation". Exposure and rate-trajectory confounds prevent conclusions about intrinsic task difficulty or pure multitask interference.

These classes do not separately mark mixture improvements. Additional disclosures therefore count mixture 4800 minus single-task 4800 development accuracy≥+5pp and≤−5pp, show up to three strongest examples per direction, and retain all differences. Counts are reported both by raw occurrence and after seed×task×source-state deduplication, which still does not imply independent samples. Single-task 4800 and mixture 4800 share total optimizer updates and global rates but differ fourfold in per-task exposure and introduce task alternation, leaving pure interference confounded. Single-task 1200 matches exposure only, not rate history or total updates. Both improvements and declines also depend on seed-related optimization; neither is directly interpreted as causal synergy or interference.

All main figures retain three seeds. Cross-target inputs and source models are shared, with only three paired seeds; mean±SD is descriptive. This fixed 256-example round establishes neither a full sample-efficiency curve nor a scaling law. All negative transfer, no-benefit, and incomplete-learning results are retained.
