# PermWorld for Data-Centric Transfer

**A relation-annotated data platform and auditable experiments on source-task composition, task count and training budget.**

This repository builds on the original PermWorld task definitions and mathematical transformations. Its DMLR contribution combines a reusable data platform with a controlled transfer workflow and conditional empirical findings, including negative transfer, underlearned tasks and unsuccessful confirmation.

## Paper, figures and release entry

- [DMLR paper draft](publication/PAPER_DRAFT.md)
- [Three contributions and attribution boundaries](publication/CONTRIBUTIONS.md)
- [Main figures, captions and PDF/SVG downloads](publication/figures/README.md)
- [Assessment of remaining scientific and release gaps](publication/DECISION_MEMO.md)
- [Installation, artifact and reproduction guide](publication/RELEASE.md)
- [Release acceptance evidence](publication/VALIDATION.md)

![Relation-annotated resource and development-only calibration](publication/figures/figure1_permworld_resource.png)

![Cross-target transfer, confirmation, task count and budget evidence](publication/figures/figure2_evidence_overview.png)

Each panel retains its own protocol and paired repeat units. Calibration is development-only, exploratory and new-repeat results remain separate, and shared models or baselines do not create extra independent repeats. The manuscript does not establish a universal benefit from related tasks, a task-count scaling law or formula use by a model.

## Use the platform

Follow the [clean installation and acceptance commands](publication/RELEASE.md) before running this example:

```python
from permworld_platform import label

p = (3, 1, 4, 2)
assert label("cycle_count", p) == 1
assert label("increasing_subsequence_count", p, {"k": 2}) == 3
```

The [platform guide](experiments/platform_v01/README.md), [machine-readable registry](experiments/platform_v01/registry/TASK_REGISTRY.json), and [relation certificates](experiments/platform_v01/mathematics/README.md) document the interface. The platform exposes 54 endpoints, 48 canonical mathematical templates, 62 declared instances and 56 materialized unary instances. These are distinct denominators, not counts of independent mathematical capabilities. One input encoding is integrated in platform v0.1.

The three focus families are cycle decomposition, global subsequences and permutation blocks. Definitions, parameters, output bounds, exact algorithms, alias accounting and proof/verification status are explicit. The sequence answer format supports large integers and structured answers without silent truncation or modular reduction.

## Results and reproduction

| Study | Main question | Entry points |
| --- | --- | --- |
| Pilot v1 | Fixed-points sample efficiency | [Report](PILOT_REPORT.md), [raw results](experiments/permworld_pilot/runs/pilot_v1/results_all.csv), [protocol](experiments/permworld_pilot/README.md) |
| Pilot v2 | Adaptation and source-budget diagnostics | [Report](experiments/pilot_v2_diagnostics/DIAGNOSTIC_REPORT.md), [protocol](experiments/pilot_v2_diagnostics/README.md) |
| Pilot v3 | Four-target transfer | [Report](experiments/pilot_v3_multitarget/PILOT_V3_REPORT.md), [task catalog](experiments/pilot_v3_multitarget/TASK_CATALOG.csv), [protocol](experiments/pilot_v3_multitarget/README.md) |
| Pilot v4 | Background-composition exploration | [Report](experiments/pilot_v4_backgrounds/PILOT_V4_REPORT.md), [protocol](experiments/pilot_v4_backgrounds/README.md) |
| Pilot v5 | Two frozen contrasts with five new paired repeats | [Report](experiments/pilot_v5_confirmation/PILOT_V5_REPORT.md), [relations](experiments/pilot_v5_confirmation/mathematics/README.md), [protocol](experiments/pilot_v5_confirmation/README.md) |
| Platform v0.1 | Correctness, data quality and limited learnability calibration | [Platform report](experiments/platform_v01/PLATFORM_REPORT.md), [calibration](experiments/platform_v01/calibration/CALIBRATION_REPORT.md) |
| Task-count v1 | Four-target finite task-count curves | [Report](experiments/task_count_v1/REPORT.md), [raw results](experiments/task_count_v1/raw_test_results.csv), [protocol](experiments/task_count_v1/README.md) |
| Task-count v2 | Budget-by-count interaction | [Report](experiments/task_count_v2/REPORT.md), [paired interactions](experiments/task_count_v2/paired_budget_interactions.csv), [protocol](experiments/task_count_v2/README.md) |

The [publication workflow](publication/README.md) recomputes DMLR results from released tables and rebuilds figures without training, model weights or local reference clones. GPU training belongs to the original frozen protocols; it is not rerun by release acceptance. Checkpoints remain local and excluded from Git.

## Attribution and preservation

Original PermWorld task definitions, generators, transformations and related task/transfer infrastructure come from the [original repository](https://github.com/XuanyuYang223/neurips). Its workshop alignment study is cited background. The newer TMLR representation study is a separate paper and is outside this DMLR manuscript and default reproduction workflow. The [archival upstream audit](publication/evidence/original_permworld_audit.md) remains available for provenance; it is not a DMLR result set or installation dependency.

Historical scientific records, raw results and frozen configurations are preserved. Documentation revisions are tracked separately from the [English translation ledger](maintenance/english_translation/README.md). The draft's authorship, code/data licensing and formal archival release remain open release decisions.
