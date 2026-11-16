# Case-integration manuscript revision

This supplement records new publication files only. `publication/PAPER_DRAFT.md`, its earlier evidence ledger, figures, release acceptance, literature review and all historical experiments remain unchanged. The pre-existing untracked `LOCAL_STATE_SUMMARY.md` is outside this delivery.

## Scientific scope

The new manuscript brings the original fixed-points confirmation and external relation-readout study into the same account as the platform, task-count studies, extended calibration and separate GRU replication. It distinguishes mathematical answer sufficiency, learned source predictions and downstream adaptation. The original comparison's advantage over the specified source replacement is reported alongside the much smaller gain over scratch and negative transfer in the comparison conditions.

The GRU primary interval includes zero, and its incomplete source and target learning are retained. A successful single-task gate does not establish successful multitask learning under different per-task exposure. The original and new confirmation units are never pooled. The earlier background-interaction confirmation is still unsuccessful or uncertain; the new experiment does not retroactively confirm it. Task-count/readout measurements retain their different vocabulary, lengths, training units and inference information budgets.

The extended calibration shares canonical inputs and initialization pairing with its predecessor. It adds no independent data units or model test forwards. The full requested-instance table includes reader rejection, generation ceilings and weak results. The historical capacity audit is preserved, and the paper states the practical limitation on another fresh calibration at the shortest length.

## New artifacts

- `publication/PAPER_DRAFT_v2.md`: new reader-facing manuscript and calibration appendix.
- `publication/evidence/dmlr_claims_v2.csv`: numerical claims mapped to existing source records and exact excerpts.
- `publication/figures/case_integration_v2/final/`: final worked-case figure and source mappings. The parent directory retains the first layout preview unchanged; the paper uses only the final subdirectory.
- `publication/case_integration_v2/build_worked_case_final.py`: final table-only figure builder. `build_worked_case.py` preserves the initial preview renderer.
- `publication/AUTHOR_DECISIONS.md`: unresolved attribution, licensing, checkpoint and archival metadata.

Only the new figure builder is run for manuscript preparation. Historical analysis/report generators are not rerun. No additional model training or test evaluation follows the completed replication.

Visual inspection found a legend/title overlap in the first rendering. Its files and builder are retained in place. A separately named renderer and fresh final output directory correct only the layout, with unchanged source values and explicit non-overlap checks.

Manuscript construction first encountered a JavaScript quoting rejection and then the Windows command-length limit, both before filesystem writes. Short document writes and direct copies of the existing report tables resolved these construction issues. Neither failure executed an experiment or changed a source value.

## Typesetting scope

No DMLR/JMLR LaTeX style was found locally. The search used `rg --files --hidden --no-ignore` with `*.sty` and `*.cls` globs, excluding `.git`, `.venv` and `node_modules`, while including ignored reference directories. It returned no matching style/class files. `Get-Command` also found no `pdflatex`, `kpsewhich`, `latexmk` or `tectonic` executable on PATH. This delivery therefore contains Markdown, not a speculative LaTeX substitute. Author metadata and a submission-ready typeset layout remain explicit publication decisions.

## Preservation and verification

The final integration audit records source/output hashes, numerical-copy checks, abstract length, local figure-link checks, repository preservation and the absence of checkpoint uploads. It validates the new publication layer against existing records; it is not another scientific replication. The normal branch push is separate from any merge, public release or archival tag.
