# Local evidence for the DMLR positioning revision

This is a read-only source and artifact audit for the literature revision based on release-check commit `5cb6b9f`. It does not rerun mathematical enumeration, training, model evaluation, or clean-install acceptance. Frozen scientific files and previous acceptance records are not revised.

## Supported claims and their limits

| Candidate claim | Implementation and released evidence | Defensible wording and limitation |
| --- | --- | --- |
| Multiple exact functions over a common permutation object | [`tasks.py`](../../permworld_platform/tasks.py): `label`, `validate_parameters`, `component_lengths`, `_subsequence_count`; [`objects.py`](../../permworld_platform/objects.py): `validate_permutation`; [`registry.py`](../../permworld_platform/registry.py): `_build_specs`, `list_task_instances`, `task_output_bounds` | Unary tasks operate on the same one-based permutation domain, with exact deterministic labels and explicit parameters. Four binary endpoints also accept a second permutation. This is a concentrated object-space control, not a claim that all endpoints are independent capabilities or that every endpoint has a certified training-data pipeline. |
| Explicit answer, joint, and input-query relations | [`relations.py`](../../permworld_platform/relations.py): `_edge`, `q`, `_historical_edges`, `new_relations`, `eval_expression`, `evaluate_relation`; archived [`RELATIONS.json`](../../experiments/platform_v01/mathematics/RELATIONS.json) | The schema distinguishes single-task versus joint reconstruction, ordered input transforms, parameterized query families, answer maps, proofs, finite checks, empirical observations, and unchecked claims. Not every historical relation is executable: seven records remain declarative inequalities, extra-feature identities, or distribution statements. Twenty-two historical records have compiled answer maps; all 45 platform additions have executable maps. |
| Composition, tokens, exposure, and adaptation can be traced | [`task_count_v2/train.py`](../../experiments/task_count_v2/train.py): `run_pretraining`, `record`, `run_adaptation`, `new_optimizer`, `reset_target_rows`, `schedule`, `token_lr`; [`FROZEN_PROTOCOL.json`](../../experiments/task_count_v2/FROZEN_PROTOCOL.json); [`FROZEN_SELECTIONS.json`](../../experiments/task_count_v2/FROZEN_SELECTIONS.json) | The fixed experiments record named source instances, paired seeds, exact sequence/supervised tokens, sample counts, unique inputs, repeated exposure, learning rates, and development selection. They are diagnostic fixed-mixture experiments, not an adaptive data-selection algorithm. Uniform sample slots do not imply equal token shares. |
| Findings are conditional on targets, compositions, budgets, and repeats | [`raw_test_results.csv`](../../experiments/task_count_v2/raw_test_results.csv), [`paired_budget_interactions.csv`](../../experiments/task_count_v2/paired_budget_interactions.csv), [`budget_interaction_summary.csv`](../../experiments/task_count_v2/budget_interaction_summary.csv), v5 [`PILOT_V5_REPORT.md`](../../experiments/pilot_v5_confirmation/PILOT_V5_REPORT.md) | Known formulas, source-task accuracy, and downstream transfer are three distinct observations. The retained results include failed confirmation, uncertain interactions, negative transfer, and underlearned sources. Two fixed task chains and three paired task-count repeats do not estimate a population law over task compositions. |

## Inherited resource versus the DMLR extension

The 48 legacy endpoint names come from the union of the original `V2_TASK_NAMES`, `V3_TASK_NAMES`, and `PROPERTY32_TASK_NAMES`. `tasks.py` dispatches to the unchanged [vendored mathematical functions](../../experiments/permworld_pilot/vendor/neurips_permutations/math_ops.py). Mathematical definitions, inverse/reverse/complement transformations, existing task/generator infrastructure, and the inherited model implementation are not new DMLR inventions.

The six additional registered endpoints are `cycle_length_count`, `increasing_subsequence_count`, `decreasing_subsequence_count`, `largest_component`, `smallest_component`, and `component_length_count`. In particular, `cycle_length_count` calls an existing legacy primitive. The six registered additions are not six newly invented mathematical algorithms. The three focus families reorganize inherited tasks and add parameterized counts and block-size statistics.

The DMLR additions supported by local artifacts are the unified task/instance registry and complete-answer interface; explicit answer/query relation annotations and their separate evidence classes; independent correctness certificates and orbit-aware generation; development learnability calibration; and the staged paired transfer, task-count, and budget experiments. The original workshop alignment study is background to the resource. Newer TMLR representation controls and their outcomes are outside the DMLR scientific contribution.

The [task counts](../../experiments/platform_v01/registry/TASK_COUNTS.json) separately record 54 endpoints, 48 canonical templates, 62 declared endpoint instances, and 56 default unary materialized instances. These are interface counts with different denominators. Among the focus instances, constants, exact aliases, transform equivalences, and answer-map relations remain explicit. They cannot be compared directly with another resource's task-category count as if both counted independent skills.

## What the relation records actually provide

Examples can be inspected without opening checkpoints:

- `position_partition`: `fixed_points(p) = n - exceedances(p) - deficiencies(p)`. Two same-input source answers and the known input length suffice.
- `fixed_from_two_deficiency_queries`: `fixed_points(p) = n - deficiencies(p) - deficiencies(inverse(p))`. This needs two specified input queries; observing only `deficiencies(p)` does not supply both.
- `lis_length_complement`: `LIS(p) = LDS(complement(p))`. This is a transformed-input equality, not a claim that the two labels coincide on the same input.
- `lis_from_subsequence_histogram`: all increasing-subsequence counts for `k=1,...,n` determine LIS. The default declared instances `k=1,2,3` alone do not meet this query requirement.
- `lis_from_rsk_first_row` and `lds_from_rsk_first_column`: the full RSK shape determines both scalars, but the reverse implication does not follow.

The `query_model`, `uses_transformed_input_queries`, `answer_map`, `parameter_schema`, `proof`, `evidence_class`, and `finite_check` fields make these distinctions machine-readable. Transform lists execute left to right. Transformed arguments can coincide for symmetric permutations; the records do not assume numerically distinct input values.

The archived [certificate](../../experiments/platform_v01/mathematics/CERTIFICATE.json) records independent label checks on 17 implementations and finite checks on 45 new formulas. It reuses 29 historical v5 certificate records by hash. Proof status and finite implementation coverage are separate. Runtime relation constructors initially mark new finite checks as `pending_certificate`; the archived certificate output records the completed coverage. A reader should not treat a freshly constructed metadata record as a new verification run.

The evaluator uses exact task oracles to check a formula; it does not infer relations automatically, establish that the model uses those formulas, or present all required source answers jointly during pretraining. The schema explicitly marks model formula use, arbitrary unlisted reconstructions, and task-family independence as unchecked.

## What the exposure and transfer records actually provide

The released [task exposure table](../../experiments/task_count_v2/task_exposures.csv) includes separate initialization, data, test, sampling, and dropout seeds; task and length; sequence and supervised tokens; sample exposures; unique exposed inputs; repeated exposures; pool size; and coverage. Per-input tables remain under each pretraining run. Longer-budget runs reuse the same 8,192-object pool per repeat and can increase both task-specific coverage and repetition.

[`source_learning.csv`](../../experiments/task_count_v2/source_learning.csv) contains start, midpoint, and endpoint evaluations, with accuracy, answer cross-entropy, complete-answer counts, per-length majority baselines, and exposure at each stage. Training performance is measured on the first 256 training objects per length as a fixed probe, not on all 4,096 objects per length. Development evaluation uses the full development pool. Those recorded scopes should remain explicit in comparisons.

`run_adaptation` loads and hashes the parent model, resets all four held-out task-token rows to the paired random initialization, checks that other parameter values are untouched, and creates an optimizer whose state is empty. It selects among positive-update checkpoints by complete-answer development accuracy, then answer cross-entropy, then earlier update. The archived adaptation curves and selection manifests document those choices; this review does not reopen the local weights to reverify them.

The 4M and 32M runs begin at the same paired initialization but have different absolute-token learning-rate trajectories under the same relative-budget rule. The budget interaction therefore does not isolate update count from the schedule or task-input coverage. Increasing task count also changes composition, relations, answer distributions, and exposure. Source mastery is descriptive evidence, not an identified mechanism of transfer.

## Remaining positioning gaps

The repository does not supply a controlled head-to-head comparison against CLRS, CLRS-Text, Reasoning Gym, Skill-It, DoReMi, DoGE, or Data Mixing Laws. Literature differences should therefore concern documented object spaces, relation semantics, experimental questions, and interfaces, not asserted empirical superiority.

Exact labels and clean documentation alone do not establish novelty. The supportable scientific focus is that the relation/query annotations let researchers state which information a task set contains while separately observing source learning and held-out adaptation under accounted exposure and budget. The present experiments illustrate that these quantities need not move together; they do not explain why.

Binary query-group isolation, some long-context structured outputs, broad task-family sampling, unseen-length transfer, and mechanistic formula-use tests remain outside the validated scope. Licensing, final original-workshop citation metadata, authorship, and stable archival publication identifiers remain unresolved release items.

## Safe verification of this documentation revision

The release-check verifier hashes the paper and other publication files through the original [publication manifest](../MANIFEST.json). Editing prose without a separate revision mechanism would make the old package hash check fail. Re-recording that manifest would instead rewrite a historical acceptance boundary.

A narrow documentation overlay can preserve both goals:

1. Pin baseline commit `5cb6b9f` and verify that the original manifest and `publication/VERIFICATION.json` remain byte-identical to their baseline Git blobs.
2. Add a new revision ledger listing only approved prose, bibliography, and the minimal verifier change. For every changed manifest-covered path, bind its baseline Git-blob hash to the old manifest hash and record the exact new hash. Permit no general hash bypass or scientific-path exemption.
3. Hash new literature-review documents separately in that ledger. Keep old release-check logs, figures, raw result tables, evidence builders, frozen scientific manifests, and package code unchanged.
4. Let the default publication verifier accept only these explicit hash transitions while retaining all existing scientific-source bindings, figure checks, path-preservation assertions, and local-link checks. Do not change arithmetic or plotting pipelines.
5. Write the new verification output under `publication/literature_review/`. Guard against `--record` overwriting the historical manifest once an overlay is active.

No training or model test evaluation is needed for these document checks. A later change outside this exact revision allowlist should fail until separately reviewed and recorded.
