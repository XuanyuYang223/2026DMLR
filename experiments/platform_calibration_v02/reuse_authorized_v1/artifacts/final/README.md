# Completed remaining-instance calibration

Read [RESULT_REPORT.md](RESULT_REPORT.md) for the complete endpoint table, shared-unit interpretation, generation limits and cost. [SOURCE_ONLY_AUDIT.json](SOURCE_ONLY_AUDIT.json) records the passed no-forward audit. [endpoint_by_unit.csv](endpoint_by_unit.csv) and [endpoint_summary.csv](endpoint_summary.csv) are the numerical source tables. [LEARNABILITY_INDEX_v02.csv](LEARNABILITY_INDEX_v02.csv) updates status without changing v0.1 records.

Rebuild into a fresh directory from the repository root with `.venv/Scripts/python.exe -B experiments/platform_calibration_v02/reuse_authorized_v1/report_and_audit.py --out experiments/platform_calibration_v02/reuse_authorized_v1/artifacts/final_reproduction`. Local checkpoints are required for source-only checkpoint verification; no model forward or new training runs. The existing output is never overwritten.
