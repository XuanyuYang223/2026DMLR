# DMLR contribution statement

## Positioning

**PermWorld: A Relation-Annotated Data Platform for Composition and Budget Studies of Transfer**

PermWorld supplies exact permutation tasks, mathematical transformations and answer-relation records. The DMLR paper combines the mathematical information available from specified source queries with measurements of source learning and held-out adaptation. The paper gives equal weight to this data resource and bounded scientific findings. Derivability, source mastery and transfer benefit are separate claims.

## Three contributions

1. **A relation-annotated data platform.** A unified interface connects inherited and additional registered task templates and parameters over the same permutation object. Versioned answers, exact labels, bounds, orbit-isolated data and correctness checks accompany relations that distinguish single-task, joint-answer and transformed-input requirements, with separate proof, finite-verification and empirical status. All 45 new relation records and 22 historical records have compiled answer maps; seven historical records remain declarative.
2. **A composition-and-budget transfer workflow.** Frozen source sets, paired input pools and initializations, task-token resets, fresh adaptation optimizers, development-only model selection and complete-answer evaluation put mathematical relation annotations alongside measured source learning and adaptation. Actual nonpadding sequence tokens, supervised tokens, samples, unique inputs, repeated exposure and shared baselines remain explicit. The scientific use is to distinguish available mathematical information from learned and transferable behavior; accounting is supporting infrastructure.
3. **Conditional empirical findings with unsuccessful outcomes retained.** The completed experiments show target- and composition-dependent transfer, an unsuccessful/uncertain background confirmation, nonmonotonic fixed-token task-count curves, and budget-dependent count contrasts. Their statistical units and protocol boundaries prevent a small sequence of studies from being presented as a general scaling law.

## Attribution and evidence boundaries

| Component | Origin and available evidence | Defensible statement | Boundary |
| --- | --- | --- | --- |
| Exact permutation environment | Original PermWorld in the `neurips` repository; inherited task declarations, generators and transformations | Exact functions over a common object space make mathematical relations explicit | DMLR did not invent the inherited tasks or the original environment |
| Unified platform v0.1 | `permworld_platform/`, registry, data manifests and correctness certificates | 54 endpoints, 48 canonical mathematical templates, 62 declared instances and 56 materialized unary instances are organized explicitly | Counts use distinct denominators; one encoding is integrated |
| Three focus families | Cycles, global subsequences and permutation blocks | 16 templates and 24 default instances include 22 nonconstant instances at the calibrated lengths | Parameter variants and mathematical aliases are not independent capabilities |
| Auditable transfer workflow | Pilots v1-v5, platform calibration and task-count v1/v2 | Frozen choices, pairing, input isolation and exposure accounting retain weak and negative outcomes | Early scalar pilots and full-answer platform runs are not one pooled benchmark |
| Cross-target transfer and confirmation | Four-target v3, exploratory v4 and five-new-repeat v5 | Transfer is heterogeneous; one selected direction does not reproduce and the second remains uncertain | Small repeat counts and exploratory case selection limit generalization |
| Count-by-budget evidence | Four-target/two-chain v1 and new-repeat budget v2 | Fixed-total-token curves are nonmonotonic in these chains; a larger budget changes the LIS count contrast | Task count also changes composition and per-task exposure; two chains are not a task-set population |

The [DMLR claim ledger](evidence/dmlr_claims.csv) binds numerical statements to released artifacts. Main-figure source tables retain the paired units used in each panel.

## Closest work and supported differences

The [resource comparison](RESOURCE_COMPARISON.md) gives primary sections and pinned
code for nine directly relevant works. Reasoning Gym already has procedural
generation, answer validation, difficulty control, task mixtures and transfer
experiments. CLRS/CLRS-Text already support controlled algorithmic generation and
multi-task execution. Skill-It already uses synthetic tasks and estimated skill
dependencies. DoGE already evaluates target adaptation and logs sample counts;
DoReMi learns domain weights without downstream-task access. Data Mixing Laws
already combines mixture, budget and model-size predictions.

The supported difference is the explicit semantics of exact answers on shared
permutation objects, including jointly sufficient answers and required input
queries, studied alongside source learning and downstream adaptation. An
equivalent systematic relation registry is unverified in the inspected external
sources, not asserted impossible or absent. No head-to-head performance advantage
or universal novelty claim follows from this comparison.

Task definitions, transforms, generators and training infrastructure are inherited
from original PermWorld. Six additional registrations are not six new algorithms:
cycle-length counting already exists in the inherited implementation. The DMLR
extension supplies the unified registry and answer interface, typed query/evidence
records, certificates and the composition/budget studies.

## Boundary with the other paper

The original workshop alignment work is cited only as background to the inherited resource. The newer TMLR encoding, architecture, initialization and answer-control studies belong to the separate representation paper. Their numerical results, figures and comparisons are not DMLR contributions and are not inputs to the default DMLR figure builder or result verification.

The [upstream audit](evidence/original_permworld_audit.md) is preserved as optional archival provenance. Retaining that audit neither transfers the other paper's contributions into DMLR nor makes its repositories, local clones or checkpoints required for this release.

## Statements to avoid

- "Related tasks universally improve transfer," "more tasks are better," or "negative transfer is universal."
- "The model uses the registered mathematical formula" from accuracy or source-learning correlations alone.
- "54 independent mathematical tasks," "four integrated platform encodings," or "all tasks are learnable."
- "A scaling law," "eight times more independent data," or "per-task exposure is matched at 32M tokens."
- "The background interaction is confirmed" from the old exploratory seeds or marginal standard deviations.
- "Unrelated source tasks" when only the absence of a registered relation has been checked.
- "The first mathematical generator, task-relation framework or data-composition study."
- "Other resources lack generation, validation, mixtures, transfer or exposure logs" without source-specific evidence.
- "All 74 relation records are executable," or "six new endpoints mean six new mathematical algorithms."
- "The annotations improve learning" without a comparison isolating the effect of annotations.

Authorship, contributor credit, venue and licensing require explicit decisions by the relevant authors and rights holders; they are not inferred from repository ownership.
