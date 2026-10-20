# PermWorld and DMLR: integrated publication package

This is an English, evidence-linked synthesis of the original PermWorld resource, representation-alignment work, and the DMLR transfer program. It contains **no new training or model test evaluation**. Scientific evidence remains in its original experiment directories. The agreed framing gives equal weight to the data platform and the scientific findings.

## Start here

| Purpose | Entry point |
| --- | --- |
| Understand the contributions and inherited work | [Contribution statement](CONTRIBUTIONS.md) |
| Read the integrated argument | [Paper draft](PAPER_DRAFT.md) |
| Inspect the main visuals | [Figure guide and captions](figures/README.md) |
| Decide what is actually missing | [Decision memo](DECISION_MEMO.md) |
| Use or reproduce the resource | [Artifact and release guide](RELEASE.md) |
| Audit original PermWorld/alignment evidence | [Pinned upstream audit](evidence/original_permworld_audit.md) and [source inventory](evidence/source_inventory.csv) |
| Audit DMLR claims | [Evidence synthesis](evidence/dmlr_evidence.md) and [claim ledger](evidence/dmlr_claims.csv) |

## Scope of the figures

The main figures cover the original alignment evidence, platform breadth and calibration, four-target transfer, exploratory-versus-confirmatory background interactions, and task-count/budget contrasts. A supplementary figure retains the complete four-target task-count curves. Each panel uses its own protocol and statistical unit; the panels are not pooled into a single estimate.

Figures are provided as PNG for browsing and SVG/PDF for publication. Their script and source bindings are in this directory. Figure regeneration uses archived CSVs and the compact upstream extracts; it requires neither model weights nor the ignored local reference clones.

## Rebuild and verify without training

Run from the repository root in the existing Python environment:

```powershell
.venv/Scripts/python.exe publication/build_figures.py
.venv/Scripts/python.exe publication/verify.py
.venv/Scripts/python.exe -m permworld_platform sample --count 1
.venv/Scripts/python.exe experiments/task_count_v2/release.py --verify --source-only
```

The default figure output is `publication/figures/`; use `--out` with a fresh directory to preserve released bytes. Regenerating plots with a different Matplotlib/font version can change bytes without changing numerical results; keep the checked-in figures when verifying the publication manifest. `verify.py` does not train or evaluate models and reports missing optional local upstream clones separately.

Completed checks are summarized in [VALIDATION.md](VALIDATION.md). The paper's external references are also provided as [BibTeX](references.bib).

The [release guide](RELEASE.md) distinguishes a browsable research artifact from a formal licensed, archived release. Checkpoints remain local and Git-ignored. No DOI, accepted paper, authorship list, or new software license is implied.
