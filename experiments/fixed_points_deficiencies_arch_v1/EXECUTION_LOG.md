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
