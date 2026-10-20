# Contribution statement

## Proposed positioning

**PermWorld: A Relation-Annotated Testbed for Representation Alignment and Data-Centric Transfer**

PermWorld supplies exact mathematical tasks, controlled input transformations, and executable relations. The combined research program uses these controls first to study representation alignment, then to test how source-task composition, task count, and training budget affect transfer. The contribution is both a reusable experimental resource and a set of bounded scientific findings. Neither a universal benefit from related tasks nor a task-count scaling law is established.

This integration documents work across the original PermWorld/alignment repositories and this DMLR repository. It does not attribute all inherited code or experiments to the DMLR extension. Authorship, contribution credit, venue, and software licensing require an explicit release decision; they are not inferred from Git ownership.

## Claims and attribution boundaries

| Contribution | Origin and available evidence | Defensible statement | Boundary |
| --- | --- | --- | --- |
| Exact permutation task environment | Original PermWorld in the `neurips` repository; inherited generators and multiple representations | Controlled mathematical objects support exact answers and explicit input transformations | The DMLR extension did not invent all legacy tasks; endpoint counts are not counts of independent problems |
| Representation-alignment methodology | Original `neurips` and `2026TMLR` studies; pinned source audit and numerical extracts | Mathematical input correspondences can be tested against incorrect-pairing controls in trained representations | Geometry is a separate outcome from held-out transfer; it does not establish that a network executes a formula |
| Unified, relation-aware platform v0.1 | `permworld_platform/`, executable registry, data and correctness certificates | 54 endpoints, 48 canonical mathematical templates, 62 declared instances and 56 materialized unary instances are organized explicitly | These counts use different denominators; only one input encoding is integrated in this platform release |
| Three documented focus families | Cycles, global subsequences and permutation blocks | 16 templates and 24 default instances, including 22 nonconstant instances at the calibrated lengths | Parameter instances and mathematically equivalent tasks are identified rather than counted as independent capabilities |
| Auditable learning experiments | Pilots v1-v5, platform calibration and task-count v1/v2 | Frozen choices, paired inputs/initializations, development selection, input isolation and complete accounting preserve unsuccessful outcomes | Early pilots and platform experiments use different evaluators and cannot be pooled as a single benchmark |
| Cross-target evidence and failed confirmation | Four-target pilot v3; exploratory v4; five-new-repeat v5 | Related-source transfer is heterogeneous; one selected background effect fails to reproduce in direction and the other remains uncertain | Small repeat counts and exploratory case selection prevent a general background-effect claim |
| Task count depends on composition and budget | Four-target/two-chain v1 and new-repeat 2-by-2 budget v2 | Fixed-total-token task-count curves are nonmonotonic in these chains; a larger budget changes the paired count contrast for LIS | Count changes composition and per-task exposure; two chains do not estimate an average over all task sets |

The [upstream audit](evidence/original_permworld_audit.md) separates the original resource, training studies and alignment results. The [DMLR claim ledger](evidence/dmlr_claims.csv) binds downstream claims to archived artifacts and their hashes. Main-figure source tables retain the statistical units used in each panel.

## Three paper contributions

1. **A controlled, executable resource.** Unify exact tasks, parameter instances, output encoding, mathematical relations, split rules, correctness checks and model-ready samples. Document inherited functionality and new platform engineering separately.
2. **A bridge between geometry and transfer questions.** Place the original alignment controls beside downstream composition experiments using the same mathematical objects. Treat the two outcomes as complementary evidence, without claiming an unmeasured predictive link between them.
3. **A transparent empirical account of conditional transfer.** Retain negative transfer, weak source learning, failed confirmation, chain dependence and budget interactions. State the estimand and repeat count for each result rather than turning a sequence of pilots into an apparent large independent study.

## Statements to avoid

- "Related tasks universally improve transfer," "more tasks are better," or "negative transfer is universal."
- "CKA explains the transfer effect" or "the model uses the registered mathematical formula."
- "54 independent mathematical tasks," "four integrated platform encodings," or "all tasks are learnable."
- "A scaling law," "eight times more independent data," or "all source exposure is matched at 32M tokens."
- "The background interaction is confirmed" based on the old exploratory seeds or marginal standard deviations.
- "Unrelated source tasks" when only the absence of a registered relation has been checked.

## Scientific status

The balanced resource-and-science framing is supported by existing artifacts. A stronger mechanistic paper would require a separately designed bridge between geometric measurements and transfer, or a controlled intervention that separates formula availability, source mastery, composition and exposure. Such an experiment is not run or promised by this integration.
