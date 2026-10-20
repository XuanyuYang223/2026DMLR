# PermWorld task-count v2: budget interaction

Completed scope: 24 pretrained models, 48 pretrained adaptations and six shared
scratch baselines. LIS length is primary; fixed points is auxiliary. Each unchanged
v1 chain contributes k2 and k16 at 4M and 32M sequence-token budgets, with three new
paired repeats. The [report](REPORT.md), its generator, and this README were revised
to English at the user's request; experimental results are unchanged.
V1 exploration is not pooled with these data.

Start with [paired interactions](paired_budget_interactions.csv),
[interaction summary](budget_interaction_summary.csv), [raw test results](raw_test_results.csv)
and the [frozen plan](EXPERIMENT_PLAN.md). Each interaction subtracts the within-repeat
k16-minus-k2 effect at 4M from that at 32M. Shared baselines cancel exactly.
Three repeats remain three statistical units, regardless of reused models or targets.

Both budgets begin at the same paired random initialization with empty optimizers;
32M does not continue a decayed 4M endpoint. Both use warmup over 1/24 of their
own budget and cosine decay in normalized token progress. The absolute-token LR
trajectories differ: at 4M tokens the short run has reached 3e-5 while the 32M run
is still near its peak. See [schedule CSV](learning_rate_schedule.csv) and
[schedule plot](figures/learning_rate_schedule.png). The same finite 8192-object
source pool is reused, so eightfold tokens do not mean eightfold independent data
or exact per-task exposure matching. A task may not have covered the full pool at
4M, especially at k16: additional tokens can expose previously unseen task-input
pairs within that fixed pool as well as repeat already seen pairs. The per-task
unique-input and repeated-exposure counts distinguish these cases; not every
additional training example is a repeated task-input pair. Input IDs and counts remain in each source
run's input_exposures.csv; task_exposures.csv records per-task/per-length samples,
tokens, unique inputs and repeated exposures.

## Reproduce the frozen scope

From the repository root, use the existing project Python environment and a fresh
output directory. CUDA is required for training/testing. Historical input pools,
including v1, are needed for the unchanged full inverse/reverse/complement orbit
exclusion. Do not overwrite the completed output or relax the data rules.

```powershell
.venv/Scripts/python.exe experiments/task_count_v2/design.py --out experiments/task_count_v2_reproduction/design
Copy-Item -LiteralPath experiments/task_count_v2/EXPERIMENT_PLAN.md -Destination experiments/task_count_v2_reproduction/EXPERIMENT_PLAN.md
.venv/Scripts/python.exe experiments/task_count_v2/data_pipeline.py prepare --out experiments/task_count_v2_reproduction/data
.venv/Scripts/python.exe experiments/task_count_v2/train.py smoke --out experiments/task_count_v2_reproduction
.venv/Scripts/python.exe experiments/task_count_v2/train.py freeze --out experiments/task_count_v2_reproduction
.venv/Scripts/python.exe experiments/task_count_v2/train.py train --out experiments/task_count_v2_reproduction
.venv/Scripts/python.exe experiments/task_count_v2/train.py select --out experiments/task_count_v2_reproduction
.venv/Scripts/python.exe experiments/task_count_v2/train.py test --out experiments/task_count_v2_reproduction
.venv/Scripts/python.exe experiments/task_count_v2/summarize.py --out experiments/task_count_v2_reproduction
```

The design command creates the fresh directory. Freeze binds the copied plan,
design, code, smoke evidence and training/development data. Test inputs are only
generated after all 54 development selections are frozen. Completed runs are
verified before reuse; partial runs fail visibly, without silent extension.
The reporting script never imports Torch or runs a model: it verifies identities,
hashes, development choices and paired accounting, then derives tables and plots.

For an exact UTF-8 training log instead of PowerShell's output-file defaults,
replace the training command with the following (an existing log is refused):

```powershell
.venv/Scripts/python.exe experiments/task_count_v2/run_logged.py --log experiments/task_count_v2_reproduction/execution.log -- .venv/Scripts/python.exe experiments/task_count_v2/train.py train --out experiments/task_count_v2_reproduction
```

The initial engineering smoke had a manifest setup mismatch before any model
training. Its evidence remains in smoke/failed_manifest_setup; the corrected
smoke copied exact manifest/archive bytes and passed. This is separate from
formal-run outcomes. SUMMARY_SELF_CHECK.json checks interaction arithmetic and
rejects duplicate/missing records using synthetic numbers without model calls.

For the archived run, verify frozen code/data without training:

```powershell
.venv/Scripts/python.exe experiments/task_count_v2/train.py verify
```

The completed [independent audit](RUN_AUDIT.json) passed all 24 source and 54 target
runs, 135 formal checkpoint files, 12 budget pairs and all 324 raw result rows.
It independently rebuilt all three initializations and replayed sampling, checked
fresh optimizers and target-row resets, and recomputed two development checkpoints.
It performed no new test neural forwards. See [budget pairing](audit_budget_pairing.csv),
[run accounting](audit_run_accounting.csv) and [development forwards](audit_development_forwards.csv).

After reproducing training/testing, reproduce the audit and update the report:

```powershell
.venv/Scripts/python.exe experiments/task_count_v2/audit_run.py --out experiments/task_count_v2_reproduction --distributions-only
.venv/Scripts/python.exe experiments/task_count_v2/audit_run.py --out experiments/task_count_v2_reproduction --forwards
.venv/Scripts/python.exe experiments/task_count_v2/summarize.py --out experiments/task_count_v2_reproduction
```

The `--forwards` option uses local checkpoints for development-only numerical
checks. To create a new release manifest, first place the canonical experiment
scripts alongside the new output, then inventory and verify the finalized bytes:

```powershell
Copy-Item -LiteralPath experiments/task_count_v2/train.py,experiments/task_count_v2/design.py,experiments/task_count_v2/data_pipeline.py,experiments/task_count_v2/summarize.py,experiments/task_count_v2/audit_run.py,experiments/task_count_v2/release.py,experiments/task_count_v2/run_logged.py,experiments/task_count_v2/SUMMARY_SELF_CHECK.json -Destination experiments/task_count_v2_reproduction
.venv/Scripts/python.exe experiments/task_count_v2/release.py --out experiments/task_count_v2_reproduction
.venv/Scripts/python.exe experiments/task_count_v2/release.py --out experiments/task_count_v2_reproduction --verify
```

The original RELEASE_MANIFEST.json remains the evidence of the pretranslation
archive, and frozen scientific records are unchanged. The user-requested English
revision is recorded in the [translation ledger documentation](../../maintenance/english_translation/task_count_v2/README.md).
Release verification resolves that explicit ledger rather than rewriting old
hashes. Do not rerun the release writer to erase the original archive's provenance.
Full verification includes local weights; a Git checkout without weights
explicitly uses source-only verification:

```powershell
.venv/Scripts/python.exe experiments/task_count_v2/release.py --verify
.venv/Scripts/python.exe experiments/task_count_v2/release.py --verify --source-only
```

All derived CSVs and figures can be regenerated from the delivered raw
tables without weights using `summarize.py`. Any regenerated file changes can
invalidate the old release inventory; verify the archive before regenerating,
or write derived output in a copied directory.

## Artifacts

- FROZEN_PROTOCOL.json: complete model, loss, optimizer, seeds, budgets and hashes.
- FROZEN_SELECTIONS.json: all development-only checkpoint decisions.
- raw_test_results.csv and results_seed*.csv: selected and endpoint tests at n10,
  n12 and their equally weighted aggregate; no condition is discarded.
- paired_budget_interactions.csv: D4, D32 and D32-minus-D4 for each repeat.
- v1_exploratory_reference.csv: the four old 4M k16-minus-k2 summaries and seed
  values, with a source hash; never pooled with v2 or used as new repeats.
- budget_interaction_summary.csv: three-repeat descriptive mean, sample SD and
  range of those paired differences, with individual values retained.
- paired_deltas.csv and summary.csv: condition minus the same repeat's shared A.
- source_learning.csv, source_summary.csv and source_focus.csv: all source
  start/mid/end learning, and LDS/I3/D3 focus; training is a fixed diagnostic probe.
- source_budget_comparisons.csv: matched 32M-minus-4M source development/training
  changes and actual exposure ratios; these are associations, not mechanisms.
- source_budget_summary.csv: descriptive paired source-change means, sample SD,
  ranges and per-repeat development changes; tasks are never pooled as repeats.
- task_exposures.csv and task_exposure_totals.csv: samples, total/supervised tokens,
  unique objects, repeat exposures and pool coverage, by length and summed.
- adaptation_curves.csv, checkpoint_diagnostics.csv and majority_per_length.csv:
  target curves, preselected-versus-endpoint diagnostics and train-derived baselines.
- compute.csv: actual tokens, updates, runtime, memory, per-run unique source inputs.
- figures/: budget interactions, accuracy/gains, source learning and LR trajectories.
- SUMMARY.json: completed counts, timings and evidence bindings.

Weights remain local and Git-ignored. Other artifacts are shareable on the dedicated
experiment branch. Archived Windows checkpoint paths may need separator
normalization for a POSIX consumer; fresh runs write native paths. Reading the CSVs
and reproducing plots needs no weights. Negative or undertrained results are retained.
The completed frozen scope does not trigger extra training or task replacement.
