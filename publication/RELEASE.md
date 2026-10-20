# Artifact and release guide

## Publication status

This branch provides a **reviewable integrated research artifact**, not an accepted paper or a new DOI release. The paper draft, figures, contribution statement and evidence ledgers are in [publication/](README.md). Existing experiments remain in their original directories. Checkpoints in this DMLR repository remain local and Git-ignored; no model weights are uploaded by this integration.

The root README is the public entry point. This guide links the original PermWorld and alignment artifacts as prior parts of the research program, with their own provenance and caveats. It does not merge incompatible training protocols or relicense inherited code.

## Original resources and alignment work

| Resource | Source snapshot | Existing release entry |
| --- | --- | --- |
| Original permutation environment, behavior and geometry | [neurips at 74f0de2](https://github.com/XuanyuYang223/neurips/tree/74f0de2017115f06f11b7285366b66e7e8431d30) | [Dataset release](https://github.com/XuanyuYang223/neurips/releases/tag/permutation-data-2026-09-04), [checkpoint release](https://github.com/XuanyuYang223/neurips/releases/tag/permutation-checkpoints-2026-09-04) |
| Representation pairing with encoding/architecture/answer controls | [2026TMLR at d9a113d](https://github.com/XuanyuYang223/2026TMLR/tree/d9a113d20d34db3c21cf69cf546cff4a0ab4904a) | [Artifact prerelease with retained caveats](https://github.com/XuanyuYang223/2026TMLR/releases/tag/representation-selectivity-2026-10-11-caveat) |

Release metadata was checked during the [upstream audit](evidence/original_permworld_audit.md). Asset payloads were not downloaded or rehashed in this integration. Use the original repositories' checksum and extraction instructions. Small audited CSV extracts are committed here so the integrated figures do not require those assets or local reference clones.

## DMLR resource entry points

| Artifact | Path / interface | What it supports |
| --- | --- | --- |
| Unified exact-label API | [`permworld_platform/`](../permworld_platform) | `label`, parameter validation, output bounds, encoding, data reading |
| Complete platform guide | [Platform README](../experiments/platform_v01/README.md) | Generation, splits, model interface and reproduction |
| Machine-readable task registry | [JSON](../experiments/platform_v01/registry/TASK_REGISTRY.json), [CSV](../experiments/platform_v01/registry/TASK_REGISTRY.csv), [instances](../experiments/platform_v01/registry/TASK_INSTANCES.csv) | Templates, instances, definitions, algorithms, constraints and costs |
| Count/equivalence accounting | [Task counts](../experiments/platform_v01/registry/TASK_COUNTS.json), [equivalence audit](../experiments/platform_v01/mathematics/EQUIVALENCE_AUDIT.json) | Explicit denominators and aliases |
| Mathematical relations | [Relations](../experiments/platform_v01/mathematics/RELATIONS.json), [certificate](../experiments/platform_v01/mathematics/CERTIFICATE.json), [guide](../experiments/platform_v01/mathematics/README.md) | Single/joint queries, transformations and evidence status |
| Canonical generated data | [Manifest](../experiments/platform_v01/dataset/manifest.json), [quality checks](../experiments/platform_v01/dataset/quality_full.json), [answer distributions](../experiments/platform_v01/dataset/answer_distributions.csv) | Reproducible pools, exact labels, isolation and baselines |
| Model-interface bounds | [Compatibility table](../experiments/platform_v01/registry/MODEL_INTERFACE_COMPATIBILITY.csv) | Multi-token answers and explicit context limitations |
| Learnability calibration | [Report](../experiments/platform_v01/calibration/CALIBRATION_REPORT.md) | Six preselected tasks, two seeds, development curves and per-length results |

## Experiment entry points

| Study | Report and reproduction | Raw evidence |
| --- | --- | --- |
| Pilot v1, sample-size exploration | [Report](../PILOT_REPORT.md), [commands](../experiments/permworld_pilot/README.md) | [Results](../experiments/permworld_pilot/runs/pilot_v1/results_all.csv) |
| Pilot v2, diagnostics | [Report](../experiments/pilot_v2_diagnostics/DIAGNOSTIC_REPORT.md), [commands](../experiments/pilot_v2_diagnostics/README.md) | Original adaptation/source records linked by the report |
| Pilot v3, four targets | [Report](../experiments/pilot_v3_multitarget/PILOT_V3_REPORT.md), [commands](../experiments/pilot_v3_multitarget/README.md) | [Selected results](../experiments/pilot_v3_multitarget/artifacts/results_selected.csv) |
| Pilot v4, background exploration | [Report](../experiments/pilot_v4_backgrounds/PILOT_V4_REPORT.md), [commands](../experiments/pilot_v4_backgrounds/README.md) | Paired background results linked by the report |
| Pilot v5, new-repeat confirmation | [Report](../experiments/pilot_v5_confirmation/PILOT_V5_REPORT.md), [commands](../experiments/pilot_v5_confirmation/README.md) | New-repeat and historical exploratory evidence remain separate |
| Task-count v1, four-target curves | [Report](../experiments/task_count_v1/REPORT.md), [commands](../experiments/task_count_v1/README.md) | [Raw tests](../experiments/task_count_v1/raw_test_results.csv), [paired deltas](../experiments/task_count_v1/paired_deltas.csv) |
| Task-count v2, budget interaction | [Report](../experiments/task_count_v2/REPORT.md), [commands](../experiments/task_count_v2/README.md) | [Raw tests](../experiments/task_count_v2/raw_test_results.csv), [paired interactions](../experiments/task_count_v2/paired_budget_interactions.csv) |

The [claim ledger](evidence/dmlr_claims.csv) gives exact source paths and hashes for integrated numerical statements. Original experiment reports contain their full configurations, costs, seed roles, source-learning records and frozen selection rules. No new training command is needed to read or reproduce the publication figures.

## Minimal local use

Use the recorded project Python environment from the repository root. For a clean machine, inspect the [task-count environment inventory](../experiments/task_count_v1/ENVIRONMENT.json) and the archived [requirements lock](../experiments/permworld_pilot/requirements-lock.txt). They document environments; this integration does not claim a newly validated cross-platform installer or standalone wheel.

```python
from permworld_platform import label
from permworld_platform.encoding import encode_example
from permworld_platform.data import read_samples

p = (3, 1, 4, 2)
assert label("cycle_count", p) == 1
assert label("increasing_subsequence_count", p, {"k": 2}) == 3
example = encode_example("increasing_subsequence_count", p, {"k": 2},
                         max_seq_len=128)
assert example["input_ids"][example["prompt_length"]:] == example["answer_ids"]
sample = next(read_samples("experiments/platform_v01/dataset", 617, "train",
                           "increasing_subsequence_count", {"k": 3}))
assert sample["input_ids"][sample["prompt_length"]:] == sample["answer_ids"]
```

For a data-reading smoke check and source-only archive verification:

```powershell
.venv/Scripts/python.exe -m permworld_platform sample --count 1
.venv/Scripts/python.exe publication/verify.py
.venv/Scripts/python.exe experiments/task_count_v2/release.py --verify --source-only
```

To regenerate figures, use `publication/build_figures.py`. To repeat model experiments, follow each study's commands and create fresh output directories. Historical input pools are required for the original orbit exclusions. Do not relax those rules or overwrite frozen outputs to make a reproduction run succeed.

## Preservation and publication checks

The integration starts from DMLR revision `95e84e6af5de2c0a692da2f64bf95745455f5c3c` on its own branch. It adds new publication files and revises the root entry page; the English audit checker learns to follow an explicit documentation revision bridge. Historical scientific manifests, data, raw results, configurations, reports and model weights are not changed.

The [documentation revision record](../maintenance/documentation_revisions/publication_integration.json) binds the prior English root README to the new entry page. It is separate from the old English translation ledger and from scientific preservation manifests. Publication verification checks the package's local links, source hashes, derived evidence checks and allowed change scope. It is not a fresh model replication or a replacement for the full checkpoint-inclusive release audits.

## Before a formal archived release

1. Confirm author and contributor credit across the original PermWorld, TMLR and DMLR work.
2. Choose code and data licenses with the relevant rights holders; no license is invented here.
3. Decide which local weights to archive, their access policy and storage location.
4. Validate installation and the minimal data-to-sample example in a clean environment.
5. Select a release tag and archival DOI, then cite those exact immutable artifacts in the submitted manuscript.

These are identifiable remaining release tasks. Existing upstream releases remain accessible; this integration does not replace or silently upgrade their scientific or numerical caveats.
