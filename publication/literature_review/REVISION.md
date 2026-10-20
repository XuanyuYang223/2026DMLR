# Literature and positioning revision

This editorial revision starts from release-check commit
`5cb6b9f3898c4d8c2857b5f5530d9785c8124a38` on
`publication/permworld-dmlr-release-check`. HEAD matched the requested baseline
and the worktree was clean; [START_STATE.json](START_STATE.json) records the
inspection. Work is on `publication/permworld-dmlr-literature-positioning`.
The applicable `AGENTS.md` requires English repository content and separate
documentation provenance.

## What changed

- Added [RESOURCE_COMPARISON.md](../RESOURCE_COMPARISON.md), separating resources,
  evaluation platforms and selection/prediction methods. Primary paper sections,
  appendix locations and pinned official code substantiate the comparisons.
- Expanded the [paper](../PAPER_DRAFT.md)'s related work, introduction and
  contribution framing. The abstract and limitations now distinguish exact
  answer derivability, learned source predictions and measured transfer benefit.
- Retained Saxton, Lample and Charton, and DoReMi, and added CLRS, CLRS-Text,
  Reasoning Gym, Skill-It, DoGE and Data Mixing Laws to
  [references.bib](../references.bib). Lample and Charton now use ICLR 2020;
  CLRS-Text remains a verified 2024 preprint with formal venue unverified.
  DoReMi author order was confirmed, not changed.
- Updated the contribution statement and publication entry page for consistency.
  Experimental protocol/results sections, numerical figures and figure captions
  are unchanged. The independent TMLR representation-study boundary is retained.

The closest broad resource is Reasoning Gym; Skill-It is the closest dependency
and data-selection comparison. CLRS-Text is directly relevant to procedural
multi-task evaluation, and Data Mixing Laws already combines composition,
budget and model size. DoGE already has target adaptation and sample-count
logging. The supported difference concerns explicit answer/query semantics over
a shared permutation domain, used to interpret conditional transfer measurements.
Generation, task dependencies, mixtures, transfer, logging and reproducibility
are not claimed as firsts.

## Evidence and unresolved questions

Source notes cover [existing citations](EXISTING_REFERENCES.md),
[resources](RESOURCE_NOTES.md), [methods](METHOD_NOTES.md), and
[local implementation support](LOCAL_EVIDENCE_NOTES.md). External code was
inspected, not installed or executed. Formal conference metadata are used where
verified; code revisions are distinguished from paper experiments. For example,
current Reasoning Gym held-out evaluation configuration and the paper use
different example counts. Access restrictions and unverified capabilities are
explicitly recorded.

No equivalent systematic exact single/joint/query/evidence registry was verified
in the inspected external sources; this is not a proof of absence. There is no
controlled superiority comparison, isolated benefit of relation annotation,
demonstration of formula use, or general task-count law. Small repeat counts,
negative transfer, failed confirmation, source underlearning, limited task
families and query-support gaps remain. Missing author-approved workshop
citation metadata, authorship, licensing and archival decisions are unchanged.
No evidence gap triggers a new experiment in this revision.

## Preservation and verification

The original publication manifest, historical verification record, release-check
logs, scientific manifests, raw data, results, configurations, historical reports,
figures, package code and training code remain unchanged.
[REVISION.json](REVISION.json) separately pins baseline and new hashes for the
five changed existing files and for new review files.

The only operational change is a narrow documentation overlay in
`publication/verify.py`. Each overridden hash must match both the baseline Git
blob and the old manifest, then the recorded new bytes. All original scientific
and figure checks remain. The old `--record` path and report destinations that
could overwrite historical records are blocked while this overlay is active.
The new documentation verifier also requires every other baseline tracked
path to remain unchanged and checks the paper's experimental/results/compute
sections verbatim.

Commands run with the previously accepted CPU environment's Python (a fresh
installation is unnecessary for this editorial change):

```powershell
python -I publication/literature_review/verify_revision.py --record --report publication/literature_review/VERIFICATION.json
python -I publication/verify.py --report publication/literature_review/PUBLICATION_VERIFICATION.json
python -I maintenance/english_translation/task_count_v2/verify.py
git diff --check
```

`--record` in the first command writes only the new editorial ledger. Its reports
and the publication verifier output are stored in this directory, never over the
historical acceptance records. Rechecking after checkout needs no `--record`.
The [external link audit](LINK_CHECK_SUMMARY.json) covers 56 distinct URLs:
all return HTTP 200, with two OpenReview browser challenges explicitly marked;
accessible author PDFs and official repositories supply the corresponding
content evidence. HTTP reachability alone is not semantic verification.

Validation passed: nine bibliography entries, eleven Markdown table blocks,
unchanged experimental/results/compute text, fifty scientific source hashes,
eight saved arithmetic-audit stages, and all 208 existing plotted values.
The default publication link check found no broken local links. The repository
English check passed with no remaining Han text. Three
[expected-rejection checks](GUARD_CHECKS.json) confirmed that historical manifest
and verification writes are blocked without modifying their bytes.
See [editorial verification](VERIFICATION.json) and
[publication verification](PUBLICATION_VERIFICATION.json).

No training, model forward, checkpoint loading, model test evaluation, mathematical
certificate rerun or repeat of full installation acceptance is part of this work.
