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
