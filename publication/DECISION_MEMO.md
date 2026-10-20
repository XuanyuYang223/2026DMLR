# DMLR scope and release decisions

## Supported paper

The DMLR paper is a resource-and-science paper with three contributions: a relation-annotated data platform, a composition-and-budget-accounted transfer workflow, and conditional empirical findings. The available record already extends beyond the newest two targets: four-target transfer, six-task calibration, four-target task-count curves, and a separate budget comparison are documented.

The separate TMLR representation study has its own scientific scope. Its encoding, architecture, initialization and answer-control results are excluded from DMLR's claims, main figures and default reproduction. Original workshop alignment work may be cited as background to PermWorld; it is not a missing DMLR experiment or an additional contribution.

## Gap assessment

| Question | Evidence available | Remaining boundary or action |
| --- | --- | --- |
| Is there a reusable resource? | Exact labels, a registry, versioned complete-answer encoding, relation certificates, reproducible pools and a reader | Maintain an installable package and a tested clean-environment onboarding path |
| What is scientifically distinctive? | Same-input and transformed-input relations, joint answer requirements and explicit evidence status | Explain these controls and count templates/instances/encodings separately |
| Are findings broader than two targets? | Four-target v3 and task-count v1, six-task calibration, followed by a focused budget study | Preserve protocol boundaries; do not pool absolute accuracy across evaluator generations |
| Is a background interaction established? | Exploratory v4 followed by five new repeats in v5 | Fixed-points direction did not replicate; longest-run interaction remains uncertain |
| Is there a pure task-count effect? | Two fixed nested compositions and a paired budget comparison | Count changes composition and per-task exposure; the current evidence is conditional |
| Is a formula-use mechanism established? | Registered identities and source-learning records | Correlated improvements do not identify a computation or mechanism |
| What can be reproduced without checkpoints? | Labels, data interfaces, quality checks, released raw tables and derived figures | Release acceptance validates these paths; GPU training and model predictions are separate |
| What still blocks a formal archive? | Reviewable source, protocols and results | Rights-holder-approved licenses, author credit, immutable release/tag and archival identifier |

## Release acceptance before another experiment

The current release check exercises installation in a newly created isolated environment, labels and parameters, encoding, sample reading, small data generation and quality checks, DMLR table recomputation, figure generation and source-only archive verification. The exact checked environment, commands, failures and fixes belong in the release acceptance record linked from [VALIDATION.md](VALIDATION.md). Passing these checks establishes an artifact-use path; it does not constitute an independent rerun of GPU training or test predictions.

Raw data, frozen configurations and scientific manifests remain unchanged. Derived publication material has its own versioned manifest and provenance. Optional upstream audit records remain historical records, outside the default reproduction dependencies.

## A possible later scientific extension

A future study could separate relation availability from exposure over a broader, prespecified task-family sample. Keep the model and full-answer interface fixed; select task sets independently of outcomes; annotate single-task and joint information; measure source mastery and unique as well as repeated task-input exposure. Choose the estimand first: performance at fixed total tokens, performance at matched exposure, or a relation intervention. These answer different questions.

This proposal is not executed by the release check. Further background variants for the existing two targets, extra training budgets and additional hyperparameter searches are outside the present scope.

## Task-library expansion

Use the existing [task-family plan](../experiments/pilot_v5_confirmation/TASK_FAMILY_PLAN.md) and [registry](../experiments/platform_v01/registry/TASK_REGISTRY.json) as the engineering contract. The first three planned families are already implemented. Possible future axes include local versus global order statistics, distribution- or shape-valued outputs, and unary versus binary composition queries. Binary generation and query-level isolation require validation before making a binary benchmark claim.

Each addition should record the template, parameters, output bounds, exact algorithm, independent checker, same-input equivalences, transformed-input relations and joint-query requirements. Proof, finite verification and empirical status stay separate. Representative-task calibration should retain difficult cases and failures.
