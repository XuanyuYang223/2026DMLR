# Plan for additional task families and reusable generators

This document proposes future work only. It does not add training, targets, backgrounds, or domain experiments. The two confirmation cases and all selection rules are frozen first; this plan does not trigger automatic extensions. It is based on the existing [57-row task catalog](../pilot_v4_backgrounds/TASK_CATALOG.csv) and the current definition audit, independently of the new v5 transfer results.

## Prioritize structural diversity rather than counting task tokens as families

The current catalog contains 48 PermWorld endpoints, of which 32 are registered in the existing scalar pilot. The others include representation outputs, tasks requiring parameters or a second input, and tasks with different output structures. Endpoints, distinct mathematical tasks, parameterized families, and encoding variants must be counted separately. One-line, cycle, and Lehmer representations are encoding variants, not new mathematical tasks; two_cycle_count and three_cycle_count are the k=2 and k=3 instances of cycle_length_count. A different fixed n, random seed, or task-token name does not add a task family.

The next stage should first cover **computational structures** missing from the current experiments, providing distinct references for the local adjacent comparisons and position-inequality counts already studied. Retain four sources per target and a small number of predefined compositions, without increasing the source count yet. The table distinguishes reusable implementations from new implementation work.

| Priority | Family and candidates | Current status | Difference from this round | Key relations and work required before experiments |
| --- | --- | --- | --- | --- |
| 1 | Cycle decomposition: cycle_count, nontrivial_cycle_count, even/odd_cycle_count, cycle_length_count(k) | Existing scalar implementations; k variants belong to one parameterized family | Requires traversal under the permutation action, unlike one-line neighborhoods or pointwise diagonal comparisons | C=F+NT, C=Even+Odd; parity=(n−C) mod2 is a one-way information map. First establish single-task learnability under matched budgets; existing underlearning of cycle_count must be retained |
| 1 | Global ordered subsequences and shape: lis_length, lds_length, rsk_shape | Existing implementations; shape is structured and cannot simply enter the current single-token protocol | Global subsequence/insertion structure, unlike contiguous runs | LIS(p)=LDS(Cp); the first row/column of shape gives LIS/LDS. Reuse scalar LIS/LDS first; define a separate token/loss contract for structured outputs and implementation certificates for general relations |
| 1 | Permutation blocks: global_descents, components | Existing scalar implementations | Global separation of prefixes and suffixes, unlike adjacent descents | components(Cp)=global_descents(p)+1, with a nonidentity answer map. Check concentrated answer distributions and majority baselines before allowing a majority class to obscure underlearning |
| 2 | Displacement statistics: max_displacement, displacement_one_count | Existing scalar implementations | An extremum and a count of the same displacement vector: shared intermediate structure with distinct statistics | Invariant under inversion; do not assume a one-to-one map between them. Preserve counterexamples for single-input/multiple-input joint relations; an unregistered formula does not mean unrelatedness |
| 2 | Inversions/global pairwise order: length, parity; potentially add major_index=Σᵢ i·1[pᵢ>pᵢ₊₁] | length/parity exist; major_index is not implemented | Total pairwise comparisons versus position-weighted adjacency statistics; wider answer support | parity=length mod2; length(Cp)=n(n−1)/2−length(p); major_index(Cp)=n(n−1)/2−major_index(p). Independently verify new endpoints; n=10 still fits the existing numeric tokens, but larger n requires another range check |
| 2 | Pattern structure: existing pattern_avoidance, plus pattern_count(p,pattern) and fixed-window pattern counts | Avoidance exists; count endpoints need implementation. Different patterns are parameters, not independent families | Distinguish subsequence patterns from contiguous-window patterns; these support existence, count, and joint-partition tasks | avoidance=1[count=0] is noninvertible; counts over all relative-order patterns of fixed k sum to C(n,k), and window-pattern counts sum to max(n−k+1,0). Record every member needed for these joint relations and how label-balancing changes the input distribution |

These priorities do not immediately select new experimental targets. Establish definitions, generators, and predefined single-task development runs before freezing targets. If a family cannot be learned under a compatible budget, retain the failure and report it against the declared criteria; do not replace it based on transfer-test performance.

## Future families distinct from the current catalog

The catalog already includes GraphWorld in/out-degree extrema, four row/column/XOR Hamming statistics in MatrixGroupWorld, and linear-form templates in FiniteFieldLegacy. These are not newly discovered domains. If a later stage studies generalization across computational structures, the following new families could be added separately; they must not be folded into conclusions about the two current cases.

| New family | Difference from the catalog | Auditable relation examples | Generator/label requirements |
| --- | --- | --- | --- |
| Graph connectivity and cycle rank | GraphWorld currently has only in/out-degree extrema for directed simple graphs; add undirected components, edges, cycle_rank, and related topological statistics | For undirected graphs, β₁=m−n+c, so edge count and component count jointly determine cycle rank; connected=1[c=1] is a one-way map | Introduce an explicit undirected, loop-free object type; record edge proposals and density parameters separately. Do not reuse directed labels without documenting the semantic change. Enumerate all small-n graphs and cross-check independent BFS/union-find implementations |
| Finite-field matrix rank/kernel | MatrixGroupWorld samples GL(n,F₂), where rank is constantly n; rank learning requires a new distribution that includes singular matrices | rank(A)+nullity(A)=number of columns; the size of ker A is q^nullity(A); rank(A)=rank(Aᵀ); matrix actions or multiple-column information can support joint questions | Use arbitrary m×n matrices over F_q with exact modular arithmetic. Record dimensions and whether rank-stratified proposals alter the natural distribution. If kernel-size outputs are too large, use log_q values as an explicitly registered output map, never silent truncation |
| Quadratic and bilinear forms | FiniteFieldLegacy provides linear forms; quadratic forms require multiplicative cross terms and a different computational structure | In odd characteristic, Q(x+y)−Q(x)−Q(y)=B(x,y), where Q(x)=xᵀMx and B(x,y)=xᵀ(M+Mᵀ)y; this joint relation involves multiple inputs and a new function | Start with odd prime fields to separate characteristic-2 boundary cases. Fix whether matrices/coefficients are task or world parameters in advance. Enumerate small fields; assign x,y,x+y to splits as an entire relation query group to avoid indirect reuse |

These formulas follow from the corresponding definitions, but **this round does not implement or certify these new generators**. Before future training, add machine-readable certificates, boundary conventions, single-task development runs, and a new frozen protocol for the actual implementations. Count the costs and scope of extending mathematical families within an existing domain separately from adding a new domain.

## Register relations as hyperedges with explicit input queries

A pairwise statement that tasks A and B are connected cannot express F=n−D−DI. Each relation should record:

| Field | Contents |
| --- | --- |
| relation_id / version | Stable identifier and definition version; retain old certificates |
| evidence_status | proved, empirically_observed, counterexample, unchecked; an empty field does not mean unrelated |
| domain / conditions | Object space, n/parameter range, length/boundary conventions, distribution assumptions |
| queries | task_id, parameters, required original input ID and transform; one task on two transforms still requires two queries |
| answer_map | Executable answer expression; whether n or other raw features are used; exact, inequality, or probabilistic relation |
| direction / invertibility | One-way or bidirectional; information loss; minimal required member set |
| proof / certificate | Definition-based proof, implementation path/hash, small exhaustive checks, independent implementation, counterexamples |
| query_model | Multiple sources on one input, one source across inputs, or multiple sources across inputs; maximal allowed query set |
| observations | Data version/distribution, sample count, and uncertainty for empirical associations; store separately from formulas |
| training_exposure | Whether answers/paired inputs were actually presented jointly; existence of a relation does not establish model exposure |

For example, reconstruction of fixed_points should be a two-query hyperedge for the same task, rather than a claim that one D label suffices:

```json
{
  "relation_id": "fixed_from_two_deficiency_queries",
  "evidence_status": "proved",
  "domain": "S_n, n>=1",
  "queries": [
    {"task_id": "deficiencies", "input_id": "p", "transform": "identity"},
    {"task_id": "deficiencies", "input_id": "p", "transform": "inverse"}
  ],
  "target": {"task_id": "fixed_points", "input_id": "p"},
  "answer_map": "n - y0 - y1",
  "query_model": "multiple_transformed_inputs",
  "actually_jointly_presented_in_v5_training": false
}
```

Composition analysis should additionally store each task set, ordered slots, whether each hyperedge's query conditions are satisfied, source-to-source and source-to-target relations, and unchecked items. Groups with the same unordered task set but different slot assignments are not automatically identical training conditions. State the exact query model and domain refuted by a counterexample; do not broaden it into mathematical unrelatedness.

## Reusable generator organization

Separate responsibilities in the next platform version, reusing the existing CPU platform's interface principles and keeping older directories read-only:

```text
platform_v_next/
  objects/             permutation, directed_graph, undirected_graph, finite_field_matrix
  tasks/               versioned pure label functions and parameter schemas
  relations/           unary edges, joint-query hyperedges, proofs, counterexamples
  encodings/           object/answer serialization separate from mathematical identity
  generators/          proposal distributions and explicit rejection/stratification policies
  splits/              canonical input IDs, relation-aware equivalence/group assignment
  certificates/        exhaustive fixtures, independent implementations, boundary cases
  manifests/           immutable task order, vocabulary, object/label hashes, RNG streams
```

Pure interfaces can be `label(task_spec, object)`, `transform(action_spec, object)`, and `encode(encoding_spec, object)`. Generators produce objects only; labels come from the same versioned registry. Record label-balancing proposals in the sampling policy, not inside the label function. Parameterized tasks need typed parameter schemas. Separate mathematical task identity from input parameters, encoding, and world-initialization seeds.

`generate_objects(spec, data_seed, role, exclusion_manifest)` returns objects, input hashes, proposal/acceptance counts, and length/parameter histograms; `build_labels(objects, task_registry_version)` separately returns label hashes. Record data/model/sampler/dropout/test RNG seeds instead of using one block name for all seeds. Models can share objects, slot input streams, and initialization across conditions to preserve strict pairing.

Declare split rules before sampling. For single-input tasks, separate objects and equivalence classes under registered transforms. For joint relations involving multiple inputs, assign entire query groups to roles: do not arbitrarily split supervision relating x and x+y across training and test and then claim independence. Not every valid identity requires deduplication under the universal closure of all relations. Define the generalization question and isolation policy first, then preserve auditable collision evidence, without collapsing the available input space or treating mathematical relations themselves as leakage.

Before training, each new family needs exact definitions and ranges, invalid-input rejection, encoding round trips, exhaustive/independent label checks, relation proofs and counterexamples, fixed-split statistics, single-task curves and per-length majority baselines, and frozen compositions and selection budgets. Document differences in per-task exposure, total updates, and learning-rate history between single-task and multitask references. After preselecting a small set of targets, evaluate tests only under the frozen protocol; do not automatically search backward for the best test background.

## Scope of the next stage

First cover the differences among local comparisons, global subsequences, cycle traversal, and global blocks; then decide whether graph topology or finite-field algebra families are warranted. At minimum, distinguish invertible input transforms with identity answer maps, nonidentity answer maps, equal distributions with different computations, same-input joint identities, cross-input joint identities, one-way information maps, and shared structure whose specified label queries remain insufficient. Retain distribution and learnability matching information in each category instead of inferring it from task names.

If confirmation supports only one case, report that conditional result without automatically adding backgrounds for the two existing targets. Even if both replicate, the evidence concerns only those two preselected cases. The next stage should freeze a separate target sample across task families rather than continue searching around these targets for larger effects. This document initiates no experiments.
