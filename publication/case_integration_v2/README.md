# Integrated PermWorld case manuscript

This is the new publication layer. The earlier manuscript, publication evidence, historical experiments and release acceptance remain unchanged.

- [Manuscript](../PAPER_DRAFT_v2.md): the platform and conditional findings, with the full calibration appendix.
- [Numerical claim ledger](../evidence/dmlr_claims_v2.csv): exact source excerpts, locations and hashes.
- [Final main figure](../figures/case_integration_v2/final/worked_case.png) and [figure sources/caption](../figures/case_integration_v2/final/README.md).
- [Revision record](REVISION.md) and [author decisions](../AUTHOR_DECISIONS.md).
- [Extended calibration](../../experiments/platform_calibration_v02/reuse_authorized_v1/README.md).
- [Separate GRU replication](../../experiments/fixed_points_deficiencies_arch_v1/RESULT_REPORT.md).

The original Transformer confirmation's D-U advantage is reported with its smaller gain over scratch and negative comparison-route gains. The separate GRU primary interval includes zero; its uncertainty and incomplete source/target learning are retained. The earlier unsuccessful background confirmation is not replaced. External formula readout uses different models, units and information from the confirmation studies.

## Source-only verification

From the repository root, using the existing Python environment:

```powershell
.venv/Scripts/python.exe -B publication/case_integration_v2/verify_integration.py --output publication/case_integration_v2/audit_reproduction.json
```

Use a fresh output name. The verifier never overwrites an audit. It checks the abstract limit, numerical-line coverage, exact claim/source excerpts and locations, source hashes, the complete copied calibration table, figure source/output bindings, local figure link, preservation of historical tracked files, and exclusion of checkpoints from the branch changes. It does not rerun a scientific analysis, training, model evaluation or checkpoint loading. Literal coverage and hashes supplement the independent scientific review; they do not by themselves prove that a source supports every interpretation.

The initial `INTEGRATION_AUDIT.json` passed. A subsequent wording review clarified that scratch A receives a shared adaptation suite, rather than a single learning-rate candidate; `FINAL_INTEGRATION_AUDIT.json` binds the final manuscript. Both audit records are retained with their own hashes and timestamps.

## Figure reproduction

The final renderer reads only existing tables and documentation:

```powershell
.venv/Scripts/python.exe -B publication/case_integration_v2/build_worked_case_final.py --out publication/figures/case_integration_v2/final/reproduction
```

The output directory must be fresh. Unit points, means and primary interval endpoints are copied from saved fields; no statistical estimator is rerun. The first layout preview and its renderer remain unchanged in the parent figure directory. Only the `final/` bundle is embedded in the manuscript.

## Remaining publication metadata

The authors must supply author order, affiliations/contact, rights-holder-approved code/data licenses, the original workshop citation, checkpoint hosting, and an archival tag/DOI. No suitable local style/class file or TeX executable was found in the documented search, so this delivery is Markdown only. It is a branch update, not a merge, release, tag or submission. Checkpoints and the pre-existing untracked local-state summary are excluded.
