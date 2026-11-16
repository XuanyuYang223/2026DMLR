# Execution log for author-approved shared-pool calibration

All commands run from the repository root. Python uses `-B` to avoid writing bytecode into historical source directories. Existing experiment and publication records are read-only. The original stopped-preflight files remain unchanged.

## Preparation and pretraining commitment

1. `.venv/Scripts/python.exe -B experiments/platform_calibration_v02/reuse_authorized_v1/verify_pool_reuse.py` passed. No model forward or reserved-test input access occurred.
2. `.venv/Scripts/python.exe -B experiments/platform_calibration_v02/reuse_authorized_v1/encoding_preflight.py` passed with one recorded whole-instance context rejection and explicit generation ceilings. No model was constructed.
3. `.venv/Scripts/python.exe -B experiments/platform_calibration_v02/reuse_authorized_v1/freeze.py` wrote the protocol and its byte-identical output-directory copy. Protocol SHA256: `70b46531be73c7dec3b3c817552a910ffc24b49f7d248f8f6b24197350dcdffd`.
4. Commit `b95a3f3` records the protocol, analysis plan, scope decision and bound implementation before any new optimizer update.

## Engineering and formal launch

```powershell
.venv/Scripts/python.exe -B experiments/platform_calibration_v02/reuse_authorized_v1/loader_runner.py smoke
.venv/Scripts/python.exe -B experiments/platform_calibration_v02/reuse_authorized_v1/loader_runner.py train --seeds 617
.venv/Scripts/python.exe -B experiments/platform_calibration_v02/reuse_authorized_v1/loader_runner.py train --seeds 829
```

The smoke completed successfully, including the inherited save/resume and causal checks and the new full-gold structured-answer mask/generation-cap check. It uses an engineering initialization and is not a formal calibration repeat. Its log and records are in `artifacts/smoke_console.log` and `artifacts/engineering/`.

The two formal commands run as separate concurrent workers with disjoint seed ownership. Standard output/error is retained in `artifacts/worker_617_console.log` and `artifacts/worker_829_console.log`. Per-seed exclusive ownership records prevent overlap. Completion records will supply actual worker wall time; summing concurrent per-run elapsed times does not produce wall time or dedicated GPU-hours. No old calibration writer is invoked on an old output path.

Environment metadata is recorded separately in `artifacts/ENVIRONMENT.json`. No existing or reserved test pool is opened by the calibration.

## Planned post-completion command

Only after both workers and all eligible formal runs finish:

```powershell
.venv/Scripts/python.exe -B experiments/platform_calibration_v02/reuse_authorized_v1/report_and_audit.py
```

This performs checkpoint/metadata/table validation and writes a new `artifacts/final/` report bundle. It constructs no model and performs no model forward. Its output directory must not already exist. Actual completion is recorded separately after the command succeeds; this section is not an assertion that it has already run.

## Initial completed-run review

A separate read-only review of the first two `to_cycle` completions checked initialization tensors against the archived v0.1 seed states, original length/index schedule hashes, every input exposure count, checkpoint hashes, curve inventory, and development-selection ordering. Independent cycle-encoding token accounting matched full gold-answer supervision, including answer tokens beyond the generation budget. The zero exact scores agree with the predeclared structural ceiling and do not identify failure to learn. This review performed no model forward and changed no file. The complete final audit remains required after all runs finish.

`BASELINE_AND_REPRODUCTION_NOTES.md` records two interpretation constraints identified before the remaining runs completed: majority answers are not generation-capped, and reproducing the hash-bound initializations requires the archived local initialization checkpoints.

## Completed execution and audit

Both workers wrote complete records covering their prescribed runs, and their exclusive ownership files were removed. Their start/finish timestamps and elapsed timers remain in `artifacts/WORKER_617.json` and `artifacts/WORKER_829.json`. The execution-session handles were no longer available at final inspection; no worker OS exit code is reconstructed from that fact.

The planned post-completion command was executed with console capture:

```powershell
.venv/Scripts/python.exe -B experiments/platform_calibration_v02/reuse_authorized_v1/report_and_audit.py 2>&1 | Tee-Object -FilePath experiments/platform_calibration_v02/reuse_authorized_v1/artifacts/final_audit_console.log
```

It exited zero and reported `status=passed`, 98 formal runs, 400 endpoint rows, 200 summary rows and 62 index rows. The source-only audit records no model construction/forward, no reserved test input access and no additional optimization. Its audit timer excludes interpreter startup and subsequent plotting; worker timings describe formal training/evaluation separately.

A separate read-only review matched all endpoint rows to source curves, summaries to their paired rows, index entries to the old scope/provenance, and every manifest-listed output hash. Historical tracked files and frozen bindings remained unchanged. A visual check found overlapping title/legend placement in the original derived curve pages; a separate presentation-only output path is used for a layout correction. The frozen reporter and its original result bundle are preserved.

The layout-only correction completed successfully with:

```powershell
.venv/Scripts/python.exe -B experiments/platform_calibration_v02/reuse_authorized_v1/plot_readable_curves.py
```

Its new `artifacts/presentation_v1/` pages retain the recorded curve coordinates and generation ceilings. The manifest binds the two source CSVs and the unchanged final-bundle manifest. No scientific statistic, model call or training run was added.
