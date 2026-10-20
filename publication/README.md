# PermWorld DMLR publication package

This English package presents a relation-annotated data platform, a composition-and-budget transfer workflow, and conditional findings from completed DMLR experiments. The current literature revision checks primary sources and sharpens contribution boundaries. The earlier release check verified CPU installation/interfaces and table/figure reconstruction. This revision adds **no training or model test evaluation** and preserves that acceptance record.

## Start here

| Purpose | Entry point |
| --- | --- |
| Understand the three contributions and paper boundary | [Contribution statement](CONTRIBUTIONS.md) |
| Read the DMLR argument | [Paper draft](PAPER_DRAFT.md) |
| Compare the nearest resources and methods | [Resource comparison and source evidence](RESOURCE_COMPARISON.md) |
| Review the literature changes and preservation checks | [Revision record](literature_review/REVISION.md) |
| Inspect the main visuals | [Figure guide and captions](figures/README.md) |
| Assess scientific and release gaps | [Decision memo](DECISION_MEMO.md) |
| Install and use the resource | [Artifact and release guide](RELEASE.md) |
| Inspect actual clean-environment acceptance | [Release-check report](RELEASE_CHECK_REPORT.md) |
| Audit DMLR claims | [Evidence synthesis](evidence/dmlr_evidence.md) and [claim ledger](evidence/dmlr_claims.csv) |

## Paper and figure scope

The main figures cover platform breadth, typed relations, archived data quality, development-only calibration, four-target transfer, exploratory-versus-confirmatory background interactions, and task-count/budget contrasts. The supplementary figure retains the complete four-target task-count curves. Each panel uses its own protocol and paired repeat units; panels are not pooled into a single estimate.

Original PermWorld task definitions, exact generators and mathematical transformations are inherited and credited. Original workshop alignment research is cited background only. The newer TMLR encoding/architecture/initialization/answer-control results are reserved for the separate representation paper; they are absent from the DMLR manuscript, main figures and default recomputation. The [upstream audit](evidence/original_permworld_audit.md) and [historical inventory](evidence/source_inventory.csv) remain optional provenance records.

## Rebuild without training or checkpoints

First follow the [new-environment installation instructions](RELEASE.md). From the source checkout, with the installed environment's Python:

```powershell
python -I publication/evidence/build_dmlr_evidence.py
python -I publication/build_figures.py --out ../permworld-rebuilt-figures
python -I publication/verify.py
python -I -m permworld_release.cli sample --dataset experiments/platform_v01/dataset --seed 617 --role train --task cycle_count --count 1
python -I experiments/task_count_v2/release.py --verify --source-only
```

The `python` executable above must be the newly installed environment; the release guide shows explicit PowerShell paths. All default numerical inputs are tracked DMLR-local files. These commands do not require ignored reference clones, untracked files or local model weights. The full automated acceptance creates a new clone/environment and also runs from outside the checkout to exercise the wheel install.

For standalone data generation, use the installed `permworld-data generate --out ... --config ...` interface. The [release guide](RELEASE.md) provides a complete generate/verify/read example using the acceptance fixture's saved configuration. This interface requires an explicit `historical_exclusion: false` policy: the fixture is isolated within its own splits, while historical research pools are absent from the wheel. The unchanged low-level research generator is not a standalone historical-isolation entry point.

Figures are supplied as PNG and SVG/PDF. Use `--out` with a fresh directory so rebuilding does not overwrite published bytes. Different plotting/font versions may change image bytes without changing plotted numbers. Verification checks the release's stored bytes and numerical source bindings. See [VALIDATION.md](VALIDATION.md) for verification scope and [references.bib](references.bib) for the paper's external references.

The accepted `MANIFEST.json` and `VERIFICATION.json` remain unchanged. The default
verifier additionally resolves only the exact editorial hash transitions in
[`literature_review/REVISION.json`](literature_review/REVISION.json); it keeps all
scientific and figure checks. `--record` is deliberately blocked while this
overlay is active. Current documentation-check output is separate from the
historical clean-installation evidence.

The package is a reviewable research artifact. Author approval, licensing and formal archival publication remain separate release decisions; no DOI or acceptance is implied.
