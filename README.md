# PermWorld and DMLR

**Exact mathematical tasks, explicit relations, and controlled studies of representation alignment and transfer.**

This repository extends the original PermWorld and representation-alignment work with a unified data platform and paired experiments on source-task composition, task count and training budget. The integrated account gives equal weight to the resource and the scientific findings, including negative transfer, underlearned tasks and unsuccessful confirmation.

## Paper, figures and release entry

- [Integrated paper draft](publication/PAPER_DRAFT.md)
- [Contribution statement and attribution boundaries](publication/CONTRIBUTIONS.md)
- [Main figures, captions and PDF/SVG downloads](publication/figures/README.md)
- [Evidence-based assessment of remaining gaps](publication/DECISION_MEMO.md)
- [Artifact, release and reproduction guide](publication/RELEASE.md)

![Resource, original alignment and platform calibration](publication/figures/figure1_permworld_resource.png)

![Cross-target transfer, confirmation, task count and budget evidence](publication/figures/figure2_evidence_overview.png)

Each panel retains its own protocol and statistical units. Alignment is not transfer accuracy; calibration is development-only; exploratory and new-repeat results remain separate. The [original source audit](publication/evidence/original_permworld_audit.md) connects the [neurips](https://github.com/XuanyuYang223/neurips) and [2026TMLR](https://github.com/XuanyuYang223/2026TMLR) contributions and existing artifact releases to this DMLR extension.

## Use the platform and latest experiment artifacts

Start with the [platform interface guide](experiments/platform_v01/README.md), [machine-readable registry](experiments/platform_v01/registry/TASK_REGISTRY.json), and [mathematical relations](experiments/platform_v01/mathematics/README.md). The platform exposes 54 endpoints, 48 canonical mathematical templates, 62 declared instances and 56 materialized unary instances. These are different denominators, not counts of independent mathematical capabilities. It currently integrates one input encoding.

```python
from permworld_platform import label

p = (3, 1, 4, 2)
assert label("cycle_count", p) == 1
assert label("increasing_subsequence_count", p, {"k": 2}) == 3
```

- Task-count v1: [four-target report](experiments/task_count_v1/REPORT.md), [raw results](experiments/task_count_v1/raw_test_results.csv), [reproduction](experiments/task_count_v1/README.md).
- Task-count v2: [budget-interaction report](experiments/task_count_v2/REPORT.md), [paired interactions](experiments/task_count_v2/paired_budget_interactions.csv), [reproduction](experiments/task_count_v2/README.md).
- [Publication rebuild and verification commands](publication/README.md) require no training or checkpoints.

The integrated paper is a draft. Authorship, licensing and a stable archival release remain to be finalized. Its central claims are bounded to the recorded tasks, compositions and budgets; neither a general scaling law nor a causal geometry-to-transfer mechanism is established.

## Preserved research history

Documentation and report templates are in English. Numerical results and frozen experiment records are unchanged. Historical byte-preservation claims refer to the original experiment revisions; see the [translation ledger](maintenance/english_translation/README.md) and the separate [publication entry-page revision](maintenance/documentation_revisions/publication_integration.json).

- [Pilot results](PILOT_REPORT.md)
- [Tasks, configuration, data isolation, and reproduction commands](experiments/permworld_pilot/README.md)
- [Accuracy curves](experiments/permworld_pilot/runs/pilot_v1/sample_efficiency.png)
- [All raw results](experiments/permworld_pilot/runs/pilot_v1/results_all.csv)

Original pilot branch: `pilot/permworld-composition`. Reference repositories are cloned locally under `reference/`; the original alignment experiments are unchanged. Model checkpoints remain in the local experiment directories and are excluded from Git because of their size. Data, configurations, source code, CSV files, figures, and audit records are retained in the repository.

The second-round diagnostics with 256 target samples are in `experiments/pilot_v2_diagnostics/`:

- [Second-round report](experiments/pilot_v2_diagnostics/DIAGNOSTIC_REPORT.md)
- [Second-round configuration, isolation rules, and reproduction commands](experiments/pilot_v2_diagnostics/README.md)

Second-round branch: `pilot/permworld-v2-diagnostics`. All original `pilot_v1` results and checkpoints were preserved and checked by file hashes.

The third-round experiment with 256 target samples, four targets, and three paired seeds is in `experiments/pilot_v3_multitarget/`:

- [Third-round task catalog](experiments/pilot_v3_multitarget/TASK_CATALOG.csv)
- [Third-round report](experiments/pilot_v3_multitarget/PILOT_V3_REPORT.md)
- [Third-round configuration and reproduction commands](experiments/pilot_v3_multitarget/README.md)

Third-round branch: `pilot/permworld-v3-multitarget`. Files and results from the first two rounds were preserved byte for byte when this round was conducted. Cross-target results concern transfer at a fixed sample size of 256; they do not establish a sample-efficiency curve or scaling law.

The fourth-round background-composition study is in `experiments/pilot_v4_backgrounds/`, on branch `pilot/permworld-v4-backgrounds`:

- [Fourth-round report](experiments/pilot_v4_backgrounds/PILOT_V4_REPORT.md)
- [Configuration, strict pairing, and reproduction commands](experiments/pilot_v4_backgrounds/README.md)
- [Reusable data-platform documentation](experiments/pilot_v4_backgrounds/DATA_PLATFORM.md), [task catalog](experiments/pilot_v4_backgrounds/TASK_CATALOG.csv), and [joint mathematical relation addendum](experiments/pilot_v4_backgrounds/catalog_addenda/README.md)

The fourth round fixes two exploratory targets, three backgrounds, three paired seeds, and 256 target samples. It reuses compatible weights and evaluates on an independent new test set to compare the background dependence of the same source-task replacement.

The fifth-round fixed-hypothesis validation is in `experiments/pilot_v5_confirmation/`, on branch `pilot/permworld-v5-confirmation`:

- [Fifth-round report](experiments/pilot_v5_confirmation/PILOT_V5_REPORT.md)
- [Configuration, seeds, and reproduction commands](experiments/pilot_v5_confirmation/README.md)
- [Paired per-seed reanalysis of earlier results](experiments/pilot_v5_confirmation/exploratory/EXPLORATORY_ANALYSIS_ZH.md)
- [Mathematical relations and joint derivability](experiments/pilot_v5_confirmation/mathematics/README.md), [next-stage task-family plan](experiments/pilot_v5_confirmation/TASK_FAMILY_PLAN.md)

The fifth round validates only two background contrasts selected from the fourth-round exploration, using five new repeats with paired data and initialization. The primary statistic is the per-repeat interaction difference in B−C between backgrounds. Earlier exploratory and new validation evidence are reported separately. All files from the first four rounds were preserved; no additional background variants are added automatically.

PermWorld data platform v0.1 is in `permworld_platform/` and `experiments/platform_v01/`, on branch `platform/permworld-v0.1`:

- [Platform interfaces, task counts, and reproduction commands](experiments/platform_v01/README.md)
- [Machine-readable task registry](experiments/platform_v01/registry/TASK_REGISTRY.json), [mathematical relations and certificates](experiments/platform_v01/mathematics/README.md)
- [Platform delivery report](experiments/platform_v01/PLATFORM_REPORT.md), [limited single-task calibration](experiments/platform_v01/calibration/CALIBRATION_REPORT.md)

This version retains all 48 legacy endpoints under a unified interface and implements three focus families: cycle decomposition, global subsequences, and permutation blocks. It distinguishes templates, parameter instances, encodings, and mathematical equivalence. The non-replication and uncertainty in pilot_v5 are retained. This round only calibrates six representatives fixed in advance, with two seeds each; it adds no background-interaction or transfer-composition training.
