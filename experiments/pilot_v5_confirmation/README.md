# pilot_v5_confirmation

This directory fixes two v4 background observations as confirmation contrasts, using five fresh paired data/initialization repeats. The v1–v4 code, configurations, checkpoints, and results were kept read-only during the experiment. Old seeds 17/42/73 are exploratory; new seeds 101/211/307/401/503 are confirmatory. Primary statistics do not pool the two batches.

Primary contrasts:

| Target | Fixed background difference | Candidate B / C | Prespecified interaction direction |
| --- | --- | --- | --- |
| fixed_points | g2_deficiencies − g0_v3 | anti_fixed_points / successions | Δ(B−C) > 0 |
| longest_decreasing_run | g1_recoils − g0_v3 | longest_increasing_run / left_to_right_maxima | Δ(B−C) < 0 |

Directions come from v4 exploration. The primary quantity is computed within each repeat as `(B−C)_new−(B−C)_original`; B−A/C−A and source learning are auxiliary. Separate background SDs are not used to judge difference reliability.

Materials completed before freezing:

- [Per-seed exploratory recomputation](exploratory/EXPLORATORY_ANALYSIS_ZH.md): g1−g0/g2−g0 for both targets and three paired contrasts, with all 12 raw differences and per-seed plots, without new training. The historical `_ZH` filename is retained for link compatibility; its content is English.
- [Statistical plan](stats_plan.json) and [CSV-only statistical module](paired_statistics.py): five-repeat means, sample SDs, ranges, 95% paired-t intervals, and Bonferroni simultaneous intervals for two primary contrasts. Two-sided exact sign/sign-flip diagnostics assess assumption sensitivity only. With n=5, the minimum two-sided p is 0.0625, so significance is not guaranteed.
- [Mathematical notes](mathematics/README.md), [formulas and evidence categories](mathematics/audit/RELATION_INVENTORY.csv), [joint derivability](mathematics/audit/TARGET_JOINT_DERIVABILITY.csv), and [counterexamples](mathematics/audit/COUNTEREXAMPLES.csv). Legitimate task relations and data leakage are checked separately.
- [Next-stage task-family plan](TASK_FAMILY_PLAN.md): new families, single/joint relation hyperedges, and reusable-generator responsibilities; no expansion is executed in this round.

Completed primary outputs are [PILOT_V5_REPORT.md](PILOT_V5_REPORT.md), `artifacts/confirmation_paired_differences.csv`, `confirmation_summary.csv`, `results_selected.csv`, and `figures/confirmation_interactions_by_repeat.png`. All five repeats contribute 300 candidate best/endpoint records, divided into `results_seed101.csv`, `results_seed211.csv`, `results_seed307.csv`, `results_seed401.csv`, and `results_seed503.csv`. Each has 60 rows, including 10 primary results for that repeat.

## Fixed budget and pairing

Model and encoding match v3/v4: PermWorld one-line, n=10, 4-layer Transformer, width256, 8 heads, dropout .1, vocabulary188, tied embedding/output, and 3,240,448 parameters. Pools contain source 8192, source dev1024, target train256, target dev1024, and new test 2048. Targets/backgrounds/conditions share corresponding pools and initialization within each repeat. A is trained and selected once across both backgrounds.

Each B/C pretraining run has 4800 steps, batch64, and exactly 16 examples per task. The candidate occupies slot0 and common sources slots 1–3; B/C differ only in the candidate definition/labels. Source LR retains 100-step warmup to 3e−4, cosine decay to 3e−5 at 2400, then a constant rate, without extension. Adaptation retains three LRs 3e−4/1e−4/3e−5,800 steps each, with full train/dev evaluation every 100 steps. Selection prioritizes highest dev answer accuracy, lowest dev answer CE, earliest step, then fixed LR order; step0 is excluded. Optimizer state resets, the target's tied token/output row returns to its block's initial value, and no new output head is added.

There are 40 new source runs,150 new adaptation LR candidates,50 unique selected models,300 candidate best/endpoint test records, and 60 A/B/C display rows expanded by background. Repeating A in displays does not add independent repeats. Pretraining totals 192000 updates/344064000 input tokens; adaptation totals 120000 updates/215040000 input tokens. Total training tokens are 559104000; evaluation tokens are separate. Timings are in `artifacts/compute_summary.csv` and `stage_timings.csv`; do not add training timers to enclosing stage wall time.

Data seeds are 2026101901…1905, initialization seeds 101/211/307/401/503, and new-test seeds 2026101951…1955. Source sampling/dropout seed=initialization seed+1000; adaptation=+2000. All old inputs, new roles, and new repeats are isolated by full inverse/reverse/complement orbits. Cross-block exclusion without replacement creates finite-population dependence, so independent RNG streams do not imply strictly IID pools; t and sign inference require additional assumptions. Tests can be generated only after all 50 development selections are frozen.

## Reproduction commands

Use the existing project `.venv` from PowerShell at repository root `F:/2026DMLR`. Each run's `artifacts/environment.json` records environment versions and GPU. scipy is unnecessary: the statistical module uses an analytic t CDF for df=4 critical values.

The following historical commands rebuild saved results without retraining or model evaluation:

```powershell
.venv/Scripts/python.exe experiments/pilot_v5_confirmation/paired_statistics.py self-check
.venv/Scripts/python.exe experiments/pilot_v5_confirmation/summarize.py confirmation
.venv/Scripts/python.exe experiments/pilot_v5_confirmation/audit_final.py
```

Rerun the fixed experiment in a new artifact directory to preserve delivered outputs:

```powershell
.venv/Scripts/python.exe -u experiments/pilot_v5_confirmation/pilot_v5.py all --out experiments/pilot_v5_confirmation/artifacts_reproduction
.venv/Scripts/python.exe experiments/pilot_v5_confirmation/audit_final.py --out experiments/pilot_v5_confirmation/artifacts_reproduction
.venv/Scripts/python.exe experiments/pilot_v5_confirmation/summarize.py confirmation --out experiments/pilot_v5_confirmation/artifacts_reproduction
.venv/Scripts/python.exe experiments/pilot_v5_confirmation/audit_final.py --out experiments/pilot_v5_confirmation/artifacts_reproduction
```

The default report is `PILOT_V5_REPORT.md` in this directory. With custom `--out`, it is written there, and links are recomputed from its location to that run's artifacts and this directory's frozen notes.

`all` executes prepare, smoke, freeze, pretrain, adapt, select, test in order, skipping stages with completion markers and allowing the same command after interruption. Training resumes at complete-run granularity: completed runs are skipped; incomplete runs restart from the beginning rather than automatically resuming the latest step checkpoint. Smoke save/restore checks validate only underlying state-restoration capability. Error/failure logs are retained; poor outcomes must not trigger seed replacement or budget extension. Rebuilding reports from results does not reselect models.

After freezing, do not rerun mathematics or exploratory-output builders in their original paths: their hashes are frozen evidence, and plot timestamps can change bytes. Use new directories for independent mathematical checks, as described in the mathematics README. Do not rename frozen `paired_statistics.py` to `statistics.py`, which would shadow the Python standard library.

## File responsibilities and limitations

`design.py` fixes four target×background groups, all seeds, budgets, and the original v4 configuration. `pilot_v5.py` handles generation, paired training, development selection, and delayed testing. `paired_statistics.py` contains pure statistics frozen before training. `summarize.py` reads only saved CSV/JSON for plots/reports, without models. `audit_final.py` independently checks freezing, data, checkpoints, sampling streams, selection, and historical preservation.

`source_learning_table.csv` contains exposure, length-specific accuracy, CE, and majority baselines for 160 source-task entries; `source_curves.csv` preserves full train/dev curves. Old single-task references are saved separately in `historical_single_references.csv`, explicitly using different inputs/initializations; they are not new strictly paired single-task controls. Source-state and transfer changes support association descriptions, not mechanisms.

An [independent source-learning description](SOURCE_LEARNING_NOTES.md) was also completed before new tests were generated. `source_analysis/` preserves per-repeat background differences for all candidates and unchanged common sources; its audit explicitly records no reading of new target tests. Its generator, `source_learning_analysis.py`, reads source CSV/config only and needs no GPU. This note's time point is preserved separately from the final association analysis; later target results do not rewrite the pretest observations.

This round tests only two cases at 256 samples. It establishes neither a full sample-efficiency curve, a scaling law, nor a general result for all tasks. Work stops after completion without automatically adding background variants for these targets.
