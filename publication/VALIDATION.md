# Integration validation

The integration uses no new training, neural forwards, checkpoint selection or
mathematical re-enumeration. Validation reads existing source artifacts and
recomputes arithmetic and figures. All checks below completed in the existing
local environment; this is not a clean-machine installation audit.

| Check | Completed evidence |
| --- | --- |
| Original-source audit | 42 pinned files from the original repositories and DMLR; clean reference snapshots; original CKA exports reproduce saved summaries to 1e-14 |
| DMLR numerical synthesis | 19 claims bound to 49 source artifacts; eight stage-level arithmetic checks from raw CSVs pass |
| Figure generation | Two main figures and one supplement, each in PNG/SVG/PDF; 232 plotted repeat values and 80 descriptive summaries |
| Visual inspection | Main figures inspected for labeling, clipping, axes and protocol separation; full task-count supplement retains four targets and both chains |
| Publication verification | Package hashes, source bindings, local links, allowed file changes and CPU label/encoding/sample-reader smoke all pass; see VERIFICATION.json |
| Repository English | Existing English verifier passes with the explicit root-README revision bridge; zero remaining Han characters in scanned repository text or decoded JSON |
| task_count_v2 source release | 650 files verified: 646 exact archived bytes and four previously documented English revisions; 151 local checkpoints explicitly skipped |
| Historical preservation | No prior experiment path differs from the integration base; original scientific manifests and translation ledgers remain unchanged |

Reproduce the lightweight checks from the repository root:

```powershell
.venv/Scripts/python.exe publication/evidence/build_dmlr_evidence.py
.venv/Scripts/python.exe publication/verify.py
.venv/Scripts/python.exe maintenance/english_translation/task_count_v2/verify.py
.venv/Scripts/python.exe experiments/task_count_v2/release.py --verify --source-only
```

The first command rebuilds only the new evidence synthesis. For plot regeneration,
use `publication/build_figures.py --out` with a fresh directory. Optional original
source re-extraction requires the pinned local clones and is documented in the
upstream audit. The publication verifier checks the saved upstream audit and
exports; it does not independently download or verify public release payloads.

Full checkpoint-inclusive audits were not repeated for this documentation task.
Weights remain local and unchanged by the executed commands. The difference
between source-only verification and full model replication is intentional.
