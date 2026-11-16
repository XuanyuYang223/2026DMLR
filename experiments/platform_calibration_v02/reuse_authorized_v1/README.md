# Completed remaining-instance calibration

Stage 2 is complete. The [audited result report](artifacts/final/RESULT_REPORT.md) covers all 50 requested instances: 49 completed with both inherited units (98 runs), and `to_reduced_word` explicitly rejected because required examples exceed context 128. The fixed update-2400 endpoint remains the primary development reference. No new input pool or model test evaluation was produced.

These results share the canonical training/development inputs and initialization/data pairing of the original six-task calibration. They are directly comparable in those settings and are **not independent new data units**. The original capacity audit, all historical inputs and the original calibration remain unchanged.

## Results and interpretation

- [Complete endpoint table](artifacts/final/RESULT_REPORT.md), [per-unit CSV](artifacts/final/endpoint_by_unit.csv), and [summary CSV](artifacts/final/endpoint_summary.csv).
- [New learnability index](artifacts/final/LEARNABILITY_INDEX_v02.csv), preserving inherited results and all explicit exclusions.
- [Training/development curves](artifacts/final/learning_curves.csv) and [best-versus-endpoint diagnostics](artifacts/final/checkpoint_diagnostics.csv).
- [Actual accounting](artifacts/final/accounting.csv) and [input exposure summary](artifacts/final/exposure_summary.csv).
- [Baseline and reproduction notes](BASELINE_AND_REPRODUCTION_NOTES.md) and [reporting scope notes](REPORTING_SCOPE_NOTES.md).

Eight completed instances have a development generation ceiling below one under the inherited eight-token output budget. Six cannot emit any complete gold answer under that budget. Their full gold tokens remain supervised, and their zero exact accuracy is a protocol limit, not a demonstration of intrinsic unlearnability. The training-majority predictor is uncapped; gains for structured answers must be read with that asymmetry. Constants, equivalent functions and parameter variants do not become independent capabilities by appearing in separate rows.

## Verification and cost

The [source-only audit](artifacts/final/SOURCE_ONLY_AUDIT.json) passed, checking all 196 checkpoint file hashes, 98 endpoint checkpoint records, 10,192 curve rows and 301,056 input-exposure records. It constructed no model and made no forward call. A separate read-only review checked all endpoint/summary/index rows and the output manifest without finding a discrepancy.

The report records 235200 optimizer updates, 15052800 sample exposures, 483639226 nonpadding sequence tokens, 73220026 supervised tokens and 5591.73 seconds of worker-envelope wall time. Accumulated per-run elapsed time is 11131.37 seconds; concurrent work overlaps, so this is not elapsed wall time or dedicated GPU-hours. See the report and CSVs for exact definitions and unrounded records.

Both worker completion records are present and their ownership locks were removed normally. The original execution-session handles had expired by final inspection, so no worker operating-system exit code is inferred. The final audit/report command itself returned exit code zero. Completion is established by the worker records, full run inventory and passed audit.

## Protocol and reproduction

The [protocol](PROTOCOL.md), [analysis plan](ANALYSIS_PLAN.md), implementation and [frozen record](FROZEN_PROTOCOL.json) were committed as `b95a3f3f66dd5aa99db0397618b332e7b936e5d4` before smoke or formal optimization. [POOL_REUSE_AUDIT.json](POOL_REUSE_AUDIT.json) verifies actual historical use of the canonical pools. [AUTHOR_SCOPE_DECISION.md](AUTHOR_SCOPE_DECISION.md) records the explicit reuse exception for this development-only calibration. Stage 3 transfer retains the fresh-unit isolation requirement.

Exact commands appear in [EXECUTION_LOG.md](EXECUTION_LOG.md). Environment metadata is in [artifacts/ENVIRONMENT.json](artifacts/ENVIRONMENT.json). Rebuilding the full audit into a fresh directory requires the matching ignored local checkpoints:

```powershell
.venv/Scripts/python.exe -B experiments/platform_calibration_v02/reuse_authorized_v1/report_and_audit.py --out experiments/platform_calibration_v02/reuse_authorized_v1/artifacts/final_reproduction
```

The original frozen report bundle remains unchanged. The [readable curve pages](artifacts/presentation_v1/README.md) use the same saved tables and correct title/legend spacing without changing any result or manifest in `artifacts/final/`. Their separate manifest records the source hashes and layout-only scope.
