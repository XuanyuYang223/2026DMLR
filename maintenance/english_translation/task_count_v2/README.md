# Task-count v2 English revision

The user requested English throughout the repository after task-count v2 completed.
This revision translates its report and report generator, updates the README, and
changes the release utility's language policy. The original completed study is Git
revision `d5f2b5c22e241322c619ada0e9ec7642b5b1eced`.

`TRANSLATION_MANIFEST.json` binds the four original files to their current English
versions. It also binds the unchanged original release manifest. That manifest's
language field describes the original publication; it is historical evidence and
has not been rewritten. No training, model evaluation, dataset generation, numerical
CSV modification, or checkpoint modification is part of this revision.

The summarizer's executable AST is preserved except for translated strings and the
README's language-policy explanation. Its English report is regenerated from the
same recorded results. Report tables retain their numeric cells and scientific
links. The release utility now requires an English report for future releases.
For the existing release it resolves only the four explicitly registered file
changes, reporting them separately from byte-identical archived files.

From the repository root:

```powershell
.venv/Scripts/python.exe maintenance/english_translation/task_count_v2/verify.py
.venv/Scripts/python.exe experiments/task_count_v2/release.py --verify
```

The first command checks original Git bytes, current English hashes, the generator's
AST, report tables and links, unchanged tracked study artifacts, and repository-wide
language coverage. It also checks the earlier translation ledger without rehashing
all older local archives. The second command verifies every task-count v2 artifact,
including the locally retained checkpoints, against the original release and this
explicit translation ledger. `VERIFICATION.json` records the language and structure
checks; `RELEASE_VERIFICATION.json` records the full archive check, including local
checkpoints and the separately identified documentation revisions.

A Git checkout without checkpoints uses the explicit source-only release check:

```powershell
.venv/Scripts/python.exe experiments/task_count_v2/release.py --verify --source-only
```

Git history containing the original revisions is required for translation checks.
Downloaded dependencies, ignored reference checkouts, and Git history itself are
outside the working repository's language revision. Earlier translation evidence
remains under the parent directory without modification.
