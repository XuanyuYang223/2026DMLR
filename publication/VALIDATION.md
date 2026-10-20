# DMLR release validation

The clean-installation acceptance passes. See the [full report](RELEASE_CHECK_REPORT.md)
for the two paper boundaries, failures/repairs, exact environment and limits.

- [Starting Git/worktree state](../release_checks/START_STATE.json).
- [Final 21-command clean run](../release_checks/logs/attempt03/RUN.json).
- [Accepted source hashes](../release_checks/ACCEPTED_SOURCE.json).
- [Installed interface and data checks](../release_checks/artifacts/attempt03/ACCEPTANCE.json).
- [Four budget-interaction groups](../release_checks/artifacts/attempt03/budget_interactions.json).
- [Default workflow with optional audits absent](../release_checks/artifacts/attempt03/OPTIONAL_PROVENANCE_ABSENT.json).
- [Publication manifest verification](VERIFICATION.json).

Validated: noneditable wheel installation on Windows/Python 3.14.6 with no system
third-party packages; 12 exact fixtures; parameter and complete-answer handling;
64 disposable objects and 384 exact labels; model-ready reading; deterministic
replay; DMLR raw-table arithmetic; two main figures and one supplement; source-only
verification of 650 task-count v2 deliverables; and English repository content.

All 208 DMLR plotted values and 72 descriptive summaries reproduce exactly. The
four v2 budget-interaction means and sample SDs agree within 1e-12. Eight optional
upstream records can be absent without breaking default DMLR figures or checks.

No training, neural forwards or checkpoint-inclusive audit was rerun. The old
integration validation remains in Git at 7ef277d; it is distinct from this
clean-environment acceptance. Historical scientific artifacts remain unchanged.
