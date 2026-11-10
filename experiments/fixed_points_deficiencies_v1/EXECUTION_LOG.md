# Execution record

All paths below are repository-relative. The working directory was `F:/2026DMLR` on Windows 11, with the repository's `.venv/Scripts/python.exe` (Python 3.13.14, PyTorch 2.11.0+cu128, NumPy 2.5.4, CUDA 12.8 runtime). The GPU was an NVIDIA GeForce RTX 5070. Exact environment fields and UTC timestamps are in `artifacts/SMOKE.json` and `artifacts/INITIALS.json`. Existing historical environments were reused for this experiment; this is not an isolated-install acceptance exercise.

## Completed setup

1. Read `AGENTS.md`, recorded clean starting commit `1ea08df6b867e719bdbc4110d971fa0161304a39`, and created branch `experiments/fixed-points-deficiencies-v1`.
2. Executed the mathematical preflight:

```powershell
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_v1/preflight.py --out experiments/fixed_points_deficiencies_v1/preflight_audit
```

3. After receiving the complete request, saved `design.py`, `stats_plan.json`, `PROTOCOL.md`, and `RELATION_TABLE.csv`. Read-only capacity auditing passed. The data agent called `prepare(artifacts/data)` and `verify(artifacts/data, include_test=False, replay=True)`, then checked rejection of incomplete selection manifests and all nested target prefixes. The corresponding CLI reproduction commands are:

```powershell
.venv/Scripts/python.exe experiments/fixed_points_deficiencies_v1/data_pipeline.py capacity
.venv/Scripts/python.exe experiments/fixed_points_deficiencies_v1/data_pipeline.py prepare
.venv/Scripts/python.exe experiments/fixed_points_deficiencies_v1/data_pipeline.py verify
```

The retained pretest verification reports 32 pools, 90,112 inputs, zero within/between/history orbit overlaps, and full RNG replay. Generation took 12.27 seconds; replay verification took 11.49 seconds. Test objects have not been generated at this stage.

4. The runner executed a GPU smoke, prepared all eight distinct initial models, and repeated the final smoke on 256 already-isolated R1 source-training inputs. The first discarded engineering fixture's record is retained separately. Smoke checkpoints are engineering artifacts and are never loaded as formal parents. Bitwise resume, causal masking, exact-match predictions, balanced slots, target-row-only restoration, fresh optimizers, evaluation RNG preservation, and selection priorities passed. No learning-rate, architecture, or budget choice was tuned on the smoke.

```powershell
.venv/Scripts/python.exe experiments/fixed_points_deficiencies_v1/train.py prepare-initials
.venv/Scripts/python.exe experiments/fixed_points_deficiencies_v1/train.py smoke
```

## Frozen execution commands

The command history and completion status for the following formal stages will be appended after execution. Source training and adaptation may be resumed with the same commands; completed bound runs are verified and skipped, while saved optimizer/RNG states resume interrupted runs. A failed run never receives a replacement seed.

```powershell
.venv/Scripts/python.exe experiments/fixed_points_deficiencies_v1/protocol.py freeze
.venv/Scripts/python.exe -u experiments/fixed_points_deficiencies_v1/train.py pretrain
.venv/Scripts/python.exe -u experiments/fixed_points_deficiencies_v1/train.py adapt
.venv/Scripts/python.exe experiments/fixed_points_deficiencies_v1/train.py verify
.venv/Scripts/python.exe experiments/fixed_points_deficiencies_v1/train.py select
.venv/Scripts/python.exe experiments/fixed_points_deficiencies_v1/data_pipeline.py generate-test --selection-manifest experiments/fixed_points_deficiencies_v1/artifacts/SELECTIONS_FROZEN.json
.venv/Scripts/python.exe experiments/fixed_points_deficiencies_v1/data_pipeline.py verify --include-test
.venv/Scripts/python.exe -u experiments/fixed_points_deficiencies_v1/train.py test
```

The test-generation command must follow the complete 120-selection freeze. No test outcome is available to any earlier choice. Local delivery only; no push or publication command is part of this protocol.

## Actual formal execution schedule

- Scientific freeze: `2026-11-10T19:36:00.792284+00:00`; SHA256 `f998a5e37cf324f0884d13b2e537b7b8801a7cb434cf4683107cada6b1566083`. Local freeze commit: `519133c` (the commit also captures the first in-progress source curve; the hash freeze itself precedes formal training).
- Source command: the listed `train.py pretrain`, with stdout/stderr retained in `artifacts/pretrain_console.log`. The log was created at 19:36:20 UTC. Each completed source record gives its measured elapsed time and finish timestamp.
- After all 32 source models finish, adaptation uses the following two concurrent processes. This replaces only the serial execution schedule of the equivalent `train.py adapt` command above; it does not alter the fixed candidates. See `ENGINEERING_LOG.md` for the independent scheduling review.

```powershell
.venv/Scripts/python.exe -u experiments/fixed_points_deficiencies_v1/adapt_worker.py --units R1 R3 R5 R7
.venv/Scripts/python.exe -u experiments/fixed_points_deficiencies_v1/adapt_worker.py --units R2 R4 R6 R8
```

The two stdout/stderr logs and `ADAPT_WORKER_*.json` summaries record actual execution. Wait for both successful completions before `train.py aggregate`, `verify`, `select`, the final training audit, or any test generation. Serial reproduction remains available through the unchanged frozen runner.

## Completed formal execution and acceptance

All 32 source models and 360 adaptation candidates completed without formal failures, optimizer-budget extensions, or resumptions. Both adaptation workers exited zero. All 12 post-training stages passed in 238.872 seconds.

Development selections froze at `2026-11-10T21:33:45.200629+00:00`; the test manifest was completed at `2026-11-10T21:34:21.405166+00:00`. The full test-input RNG replay passed before model evaluation and its exact record is preserved at `artifacts/data/quality_with_test_full_replay_before_evaluation.json`. The evaluation runner separately records its hash/isolation precheck in `quality_with_test.json`.

| Stage | Exit code | Measured seconds |
|---|---:|---:|
| aggregate | 0 | 1.493 |
| verify_training | 0 | 1.575 |
| freeze_selection | 0 | 7.979 |
| independent_training_audit | 0 | 12.233 |
| plot_losses | 0 | 11.206 |
| generate_test | 0 | 12.449 |
| verify_test_full_replay | 0 | 15.902 |
| preserve_test_replay_proof | 0 | 0.066 |
| evaluate_frozen_test_inventory | 0 | 166.164 |
| analyze | 0 | 3.931 |
| independent_prediction_statistics_audit | 0 | 5.359 |
| canonical_report | 0 | 0.460 |

The protocol-freeze-to-post-pipeline-completion wall-clock window was **121.543 minutes**. It includes orchestration gaps, CPU auditing, plotting and test inference; it excludes preparation before the freeze and editorial review afterward. It is not an isolated GPU-time measurement. Source-run times sum to 36.544 minutes; the concurrent adaptation worker envelope is 80.145 minutes. Do not add overlapping candidate timers as elapsed wall time.

The actual sequential driver was:

```powershell
.venv/Scripts/python.exe -u experiments/fixed_points_deficiencies_v1/post_training.py
```

The driver stored exact argument lists, current directory, UTC start/finish times, exit codes, stdout/stderr hashes and logs in `experiments/fixed_points_deficiencies_v1/artifacts/execution_logs/20261110T213334Z_8d9ef265_post_training_commands.jsonl`. `artifacts/POST_TRAINING_COMPLETE.json` retains this first successful stage-level acceptance record.

The raw prediction audit independently checked all 491,520 prediction records (240 evaluations) and the eight-unit statistics. Complete-answer and legacy answer-token correct counts happened to coincide for all 240 evaluated checkpoints in this study; their definitions and records remain separate, and no historical results are pooled.

The final editorial review added explicit scratch-baseline context and a canonical `PAPER_RESULT_DRAFT.md` through the unbound documentation generator:

```powershell
.venv/Scripts/python.exe experiments/fixed_points_deficiencies_v1/final_report.py
```

This updates only canonical documents and their amendment records, preserving the earlier stage-bound documents under `artifacts/report_revisions/`. The original frozen analysis, raw metrics, checkpoints and tests are unchanged. No training or model evaluation is repeated for this prose clarification.
