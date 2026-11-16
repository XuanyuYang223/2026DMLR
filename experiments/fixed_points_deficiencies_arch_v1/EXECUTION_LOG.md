# Architecture-replication execution log

All commands run from the repository root in the existing `.venv` environment, with bytecode writing disabled. Historical experiment/publication files are read-only. This log distinguishes executed steps from later planned commands; it is not evidence that an unexecuted stage has completed.

## Pre-optimization preparation

Stage 2 completed and was committed as `58e485b` before this stage began. `START_STATE.json` records the full parent commit, branch, working-tree snapshot and new seed inventory. `LOCAL_STATE_SUMMARY.md` remains a pre-existing untracked file outside this work.

Read-only compatibility review checked the reference GRU, legacy 188-token tokenizer, scalar answer/EOS loss and evaluation, model factories, target-row reset and optimizer/state restoration. Static comparisons found matching reference GRU constructor/forward behavior after explicit vocabulary binding, unchanged source LR values at every prescribed step, and unchanged sampler/RNG consumption. No model or optimizer was executed for those reviews. Existing mathematics is reused in `RELATION_REUSE.json`, without new enumeration or model-mechanism claims.

The initially considered gate stream identifiers 2026111602/2026111603 were found in historical smoke code. Before registration or training, they were replaced by the new identifiers in `plan.py`. A fresh word-boundary scan of historical experiment Python/JSON/Markdown metadata found no selected seed; only the new plan and gate protocol matched. This was a pre-execution identity check, not seed selection by results.

Executed capacity preflight:

```powershell
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/data_pipeline.py capacity
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/record_provenance.py
```

The capacity command exited zero. The raw PowerShell console transcript is `artifacts/preflight/CAPACITY_PREFLIGHT.json` (UTF-16); `CAPACITY_AUDIT.json` is its UTF-8 machine-readable copy with source/transcript hashes. Its conservative n=10 bound is 185551 available historical-disjoint orbits against 115712 required for the gate and all formal pools, including deferred tests. There are 268049 historically occupied n=10 orbits. The scope includes 219 historical object archives and 395 classified non-object exposure archives. No overlap rule was relaxed.

A prefreeze hardening edit after that command added formal-protocol verification to the deferred-test release gate; it changed no capacity or generation calculation. Gate-data generation binds the resulting final data source hash explicitly.

Executed gate preparation and registration:

```powershell
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/data_pipeline.py prepare-gate
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/protocol.py freeze-gate
```

Both commands exited zero. `prepare-gate` created only 8192 training and 1024 development objects. `freeze-gate` performed exact RNG replay and full historical/between-role orbit checks before writing `GATE_FROZEN.json`. Its SHA256 is `807880cc18b61ebe1b5d1c468a00902b01896aaaf013ee1bd4a4fb685c2aac9c`. Console records are retained under `artifacts/preflight/`. No gate, formal-training or model-test optimizer/forward has run at this registration point.

## Planned gate commands

Run only after the frozen gate and all bound files are committed:

```powershell
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/gate_runner.py smoke
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/gate_runner.py run
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/gate_runner.py verify
```

Run the two gate tasks sequentially. Their outcome determines the formal architecture by the registered endpoint rule. Trained gate weights never initialize a formal run.

## Planned formal preparation and training

After the gate decision, generate only formal training/development data, check replay/isolation, construct the eight fresh step-zero states, register the complete formal protocol, and commit it before smoke optimization:

```powershell
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/data_pipeline.py prepare
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/data_pipeline.py verify
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/train.py prepare-initials
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/protocol.py freeze
```

After that commit, run formal smoke and then two workers with disjoint unit ownership. Wait for both source workers before adaptation, and both adaptation workers before post-training processing:

```powershell
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/train.py smoke
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/worker.py pretrain --units R1 R3 R5 R7
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/worker.py pretrain --units R2 R4 R6 R8
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/worker.py adapt --units R1 R3 R5 R7
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/worker.py adapt --units R2 R4 R6 R8
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/post_training.py
```

The final controller freezes every development selection before test generation, checks full test replay, performs the unified evaluation, independently reconstructs statistics and writes the result report. Completion and actual cost will be appended from execution records; this planned command list does not assert those outcomes.

## Gate completed and formal registration prepared

Gate smoke and both gate tasks ran after commit `27f88ad4fb413f788519006aa590aacb2bfc18ee`. The gate worker exited zero, and `gate_runner.py verify` exited zero. Both prescribed endpoint development accuracies were 1024/1024, so the deterministic rule selects the four-layer GRU. No fallback was used. The gate is separate from the eight formal units and its weights are not reused. Console records, smoke checks, curves, exposure vectors and checkpoint hashes are under `artifacts/gate/` and `artifacts/GATE_RESULT.json`.

The formal `prepare`, `verify`, `prepare-initials` and `freeze` commands above each exited zero. Full generation replay and isolation passed for 32 training/development pools and 90112 objects, with no within-pool, between-pool or historical orbit overlap. Eight distinct fresh step-zero model states were prepared. No formal optimizer update or test generation occurred before registration. The formal protocol hash is `c2ef7262ec63cc32d9b1895982ca3acbde1fed0f796fc1e0415aa78d3e3fa9ab`; it will be committed before formal smoke.

Before formal registration, one report-generator link was corrected from `PROTOCOL.md` to this execution log. No gate-bound source or scientific setting changed. A PowerShell read-only inspection initially failed because brace expansion is unsupported; the corrected inspection succeeded and changed no file.

## Formal smoke and source training completed

Formal registration commit `5fde30e` preceded the formal smoke. `protocol.py verify` and `train.py smoke` each exited zero. The smoke checked strict save/load, bitwise resumed updates, target-only tied-row restoration, fresh optimizer, causal evaluation, complete-answer prediction consistency, RNG preservation, balanced task slots and selection priorities. Its discarded engineering work is recorded separately in `artifacts/SMOKE.json`.

Both prescribed source workers completed and returned OS exit code zero. The odd-unit worker started at `2026-11-16T21:31:45.446856+00:00` and finished at `2026-11-16T22:04:06.205763+00:00`; the even-unit worker started at `2026-11-16T21:31:45.446917+00:00` and finished at `2026-11-16T22:04:06.205802+00:00`. All 32 source endpoints are present. Their overlapping worker timers and per-run timers are not additive GPU-hours. The source worker records and UTF-16 console transcripts are retained under `artifacts/`.

Both prescribed adaptation workers then started at `2026-11-16T22:04:34.257484+00:00` (odd units) and `2026-11-16T22:04:34.257437+00:00` (even units). No target test inputs exist at this point. Completion and test processing remain pending.

## Adaptation completed; unified post-training processing started

All 360 prescribed candidates completed. Both adaptation workers returned OS exit code zero. The odd-unit worker finished at `2026-11-16T23:10:14.280180+00:00` with recorded elapsed time `3940.0226952000003` seconds; the even-unit worker finished at `2026-11-16T23:10:01.944756+00:00` with `3927.6873115` seconds. All sampling seeds, budgets and candidate lists were unchanged, with no failure reruns.

Near the end, progress temporarily slowed while total GPU memory use was observed at 11764 MiB (73% utilization at the `2026-11-16T23:06:13.4288282Z` inspection). Intermediate checkpoints continued updating. A later inspection showed 1106 MiB and 21% utilization, and candidate completion resumed at the earlier pace. The cause was not established; no unrelated process was stopped and no scientific configuration was changed. These observations are context for the measured runtime, not a portable performance benchmark or causal diagnosis.

The prescribed `post_training.py` command started after both workers completed. It will record every stage and exit code under `artifacts/execution_logs/`; the top-level console transcript is `artifacts/post_training_console.log`. At this entry, unified evaluation and independent statistical verification remain pending.

## Unified testing and independent verification completed

The post-training controller returned OS exit code zero. Attempt `20261116T231037Z_a954fdfb` completed all twelve prescribed stages; every child command exited zero. All 120 development choices were fixed before test generation. Full historical/between-role/between-unit test isolation and exact generation replay passed. The unified evaluation produced 240 selected/endpoint records and 491520 per-example prediction rows. Independent reconstruction verified labels, complete-answer counts, majority classes, scores, paired differences and the primary t interval without model calls.

The canonical `RESULT_REPORT.md` reports primary D-U `+3.1235` pp with paired 95% t interval `[-1.2743, +7.5213]` pp: uncertainty remains. This is not pooled with the original eight units, and it triggers no further training. The canonical report, README, paper paragraph, raw records, figures and manifests are complete. Gate and smoke costs remain separate. No checkpoint is staged for publication.
