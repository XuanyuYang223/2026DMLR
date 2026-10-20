# Artifact and release guide

## Publication status and paper boundary

This branch provides a **DMLR research artifact under release review**, not an accepted paper or a new DOI release. The [paper draft](PAPER_DRAFT.md) has three contributions: a relation-annotated data platform, an auditable composition-and-budget transfer workflow, and conditional empirical findings. Existing experiments remain in their original directories. DMLR checkpoints remain local and Git-ignored.

Original PermWorld definitions, generators and mathematical transformations are inherited from the [pinned original repository](https://github.com/XuanyuYang223/neurips/tree/74f0de2017115f06f11b7285366b66e7e8431d30). Its [dataset release](https://github.com/XuanyuYang223/neurips/releases/tag/permutation-data-2026-09-04) is a separate historical resource, not an installation prerequisite. The workshop alignment study is background only. Newer TMLR representation results belong to the separate paper and are excluded from DMLR's figures and default numerical reproduction.

The [upstream audit](evidence/original_permworld_audit.md) and its source inventory remain optional archival provenance. Installing this package, reading samples and rebuilding DMLR tables/figures require no upstream clone, release asset, reference directory or checkpoint. Historical upstream assets were not downloaded or rehashed in this release check.

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

## Install in a new environment

The release acceptance uses Python 3.14 on Windows, a new virtual environment with system and user third-party packages excluded, and a **noneditable wheel install**. Git and access to the Python package index are required. Core installation provides the data interface with NumPy; the `publication` extra supplies the plotting dependencies. It does not install a GPU training runtime.

From a fresh clone, the PowerShell installation commands are:

```powershell
git clone --branch publication/permworld-dmlr-release-check https://github.com/XuanyuYang223/2026DMLR.git permworld-dmlr-release
Set-Location permworld-dmlr-release
py -3.14 -m venv ../permworld-release-venv
$releasePython = (Resolve-Path ../permworld-release-venv/Scripts/python.exe).Path
& $releasePython -I -m pip --isolated install ".[publication]"
& $releasePython -I -m pip check
$releaseData = (Resolve-Path ../permworld-release-venv/Scripts/permworld-data.exe).Path
```

Use new checkout and environment paths when repeating this procedure. The package metadata declares runtime dependencies; the actual resolved versions and command transcripts are recorded by the [release acceptance report](RELEASE_CHECK_REPORT.md). The tested configuration is a reproducible example, not a claim that every Python/operating-system combination has been validated.

The installed API can be imported outside the checkout:

```python
from permworld_platform import label
from permworld_platform.encoding import encode_example

p = (3, 1, 4, 2)
assert label("cycle_count", p) == 1
assert label("increasing_subsequence_count", p, {"k": 2}) == 3
example = encode_example("increasing_subsequence_count", p, {"k": 2},
                         max_seq_len=128)
assert example["input_ids"][example["prompt_length"]:] == example["answer_ids"]
```

For installed-package generation, use the `permworld-data` entry point or `permworld_release.generate_dataset(out, config)`. Both require an explicit output path and a configuration declaring `historical_exclusion: false`; they reject an absent or enabled historical-exclusion policy. The wheel includes exact oracles and certificates but does not include historical research input pools. The archived low-level generator under `permworld_platform` retains its original scientific source bytes and must not be used from the standalone wheel to claim historical isolation.

Released data and experiment tables remain in the source checkout; they are not copied into the wheel. The installed reader can read a released dataset when its path is supplied explicitly. From the clone root:

```powershell
& $releaseData sample --dataset experiments/platform_v01/dataset --seed 617 --role train --task cycle_count --count 1
& $releasePython -I publication/evidence/build_dmlr_evidence.py
& $releasePython -I publication/build_figures.py --out ../permworld-rebuilt-figures
& $releasePython -I experiments/task_count_v2/release.py --verify --source-only
& $releasePython -I publication/verify.py
```

The table recomputation reads released aggregate and raw records; it does not run inference. The figure output should be a fresh directory. Rendering with a different Matplotlib/font version can alter image bytes even when plot data are identical; publication verification checks the checked-in release bytes.

## Full clean-environment acceptance

The acceptance runner creates a second fresh Git clone, a new environment and a separate working directory outside that clone. It clears inherited Python-path environment variables, uses Python isolated mode (`-I`), checks that no `reference/`, prior environment or checkpoint is present, and verifies that imports resolve to the installed wheel. Run it with a standard Python interpreter; it uses the standard library before installation:

```powershell
py -3.14 release_checks/run_clean_check.py --source https://github.com/XuanyuYang223/2026DMLR.git --branch publication/permworld-dmlr-release-check --work ../permworld-clean-acceptance --logs ../permworld-clean-acceptance-logs
```

Both output directories must be new. Logs include the exact checkout revision, system and Python version, initial and installed packages, each command/exit code, timing and failure information. The runner exercises:

- Installed-package imports, representative exact labels and parameterized tasks.
- Multi-token answers, versioned encoding, bounds/errors and model-ready sample reading.
- Small reproducible generated pools, within-fixture orbit isolation, duplicates, answer distributions and corruption detection.
- DMLR result recomputation, figure rebuilding and all four v2 budget-interaction means/sample standard deviations against raw results.
- Source-only task-count release verification and DMLR publication verification.

The interface check writes its explicit small configuration to `../permworld-clean-acceptance/acceptance/demo_config.json`. It declares lengths 8 and 10, two repeats, 8/4/4 train/development/test objects per length and repeat, six task instances, and `historical_exclusion: false`. After that check has completed, the following commands exercise the installed CLI using the saved configuration and another fresh output directory. Run them from the same source-checkout root as the acceptance command above:

```powershell
$acceptedData = (Resolve-Path ../permworld-clean-acceptance/venv/Scripts/permworld-data.exe).Path
& $acceptedData generate --out ../permworld-demo-cli --config ../permworld-clean-acceptance/acceptance/demo_config.json
& $acceptedData verify-data --out ../permworld-demo-cli
& $acceptedData sample --dataset ../permworld-demo-cli --seed 19001 --role train --task cycle_count --count 2
```

This example reads a scalar task; the same fixture also contains parameterized length-3 subsequence counts and length-2 cycle counts. The automated interface check reads their full encoded answers, verifies multi-token handling and checks deterministic regeneration. A Python caller uses `from permworld_release import generate_dataset, verify_dataset` with the same JSON configuration and explicit paths.

The small generator fixture explicitly sets `historical_exclusion=False`: it tests installation and data interfaces, with isolation between its own roles. It is **not a research split** and does not claim isolation from historical experiments. Reproducing a frozen scientific dataset requires its recorded historical exclusion inputs and rules; those are never silently relaxed. The fixture output cannot be substituted for published research data.

The actual release-check outcome, including failures followed by repairs and clean reruns, is recorded in [RELEASE_CHECK_REPORT.md](RELEASE_CHECK_REPORT.md). No GPU training, checkpoint loading or model test-set evaluation is part of this acceptance. Use each original study's frozen protocol and a fresh output directory for such a later rerun.

## Preservation and provenance

The original integration was based on `95e84e6af5de2c0a692da2f64bf95745455f5c3c`; this release check starts from the latest integration commit recorded in [START_STATE.json](../release_checks/START_STATE.json). Starting worktree changes, if any, are recorded before edits. Raw data, results, frozen configurations and scientific manifests remain unchanged.

Publication figures, captions, source mappings and documentation are derived release material with a new manifest. Historical upstream audit files are retained as archival provenance. Documentation changes are recorded separately from original experiment and translation manifests. Source-only verification explicitly excludes missing local checkpoint payloads; it is not a substitute for a full checkpoint-inclusive archive audit.

## Before a formal archived release

1. Confirm author and contributor credit for original PermWorld and DMLR, and the boundary with the separate representation paper.
2. Choose code and data licenses with the relevant rights holders; no license is invented here.
3. Decide whether local weights will be archived, and specify their access policy and storage location.
4. Select an immutable release tag and archival DOI, then cite those exact artifacts in the submitted manuscript.
5. Review the clean-environment record and maintain supported dependency versions; GPU-training reproduction remains a separate validation scope.

These release decisions remain explicit. Successful CPU/interface/table acceptance does not imply formal licensing, archival publication or a new model replication.
