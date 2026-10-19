# English repository revision

The repository documentation, report-generation text, and textual result
classifications have been translated into English. The original revision is
`c4aeb499a0c7f2dc59b70e438f66c5b6dad4244a`. Original text remains available in Git
history; no extra copy of the untranslated documents is kept in the working tree.

Training code, mathematical task definitions, numerical results, datasets, model
weights, frozen configurations, and historical audit records are unchanged. CSV
classification text is translated, with all other cells preserved. Report
generators retain the same executable AST except for translated string literals.
No training or model evaluation is part of this revision.

`TRANSLATION_MANIFEST.json` binds each original file hash to its English version.
The original experiment manifests are intentionally retained: their byte hashes
describe the completed experiments before this documentation revision. A strict
historical audit will therefore reject translated documentation in the current
checkout. It must not be interpreted as a new experimental failure, or bypassed
by rewriting the frozen scientific evidence. Reports describing byte preservation
refer to their original experiment-completion audits.

Verify the current English checkout and its connection to the archived evidence
from the repository root:

```powershell
.venv/Scripts/python.exe maintenance/english_translation/verify.py
```

This checks original Git bytes and current English hashes for translated files,
Python AST structure, CSV cell preservation, and remaining Han characters in
tracked or non-ignored repository text. It also verifies the unchanged archived
files, including local checkpoints, against the platform release and preservation
manifests, and resolves v4/v5 frozen documentation references through the ledger.
Git history containing the original revision is required. Full archive checks
require the locally retained checkpoints, which are ignored by Git as before.

For a source-only checkout without those checkpoint files:

```powershell
.venv/Scripts/python.exe maintenance/english_translation/verify.py --skip-archived-bytes
```

This shorter check explicitly reports that unchanged archive byte verification
was skipped. Neither command runs a model or rewrites historical artifacts.
`VERIFICATION.json` records the full check performed for this revision.

The platform calibration's own frozen code, registry, and data are unchanged, so
its existing verification command remains valid:

```powershell
.venv/Scripts/python.exe -m permworld_platform.calibration verify
```

The historical filename `EXPLORATORY_ANALYSIS_ZH.md` and CSV column
`classification_zh` are retained for link/schema compatibility; their content is
now English. Downloaded dependencies, ignored reference checkouts,
and Git history are outside the working repository translation scope.
