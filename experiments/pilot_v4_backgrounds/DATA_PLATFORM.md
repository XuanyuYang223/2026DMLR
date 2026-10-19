# PermWorld data and relation platform

This directory reuses exact definitions from the current task library for fixed-scale background-composition experiments and later explicitly authorized experiments. It adds no mathematical tasks, does not rewrite v1/v2/v3, and does not automatically start training. This round uses only PermWorld, one-line encoding, and the original Transformer. Other-world entries inventory existing interfaces only.

## Definitions and identity

`TASK_CATALOG.csv` has 57 records:48 PermWorld registered endpoints,4 GraphWorld tasks,4 MatrixGroupWorld tasks, and 1 historical finite-field linear-function template. `scalar_pilot_eligible` denotes compatibility with the existing scalar interface; `pilot_v4_roles` denotes actual use in this round. These are distinct concepts.

Each entry records its mathematical definition, input/output ranges, parameters, encoding, generation method, known relations, original source location, and relation-graph references. The following distinctions must not be collapsed into independent task counts:

- n, cycle length k, power exponent, and simple index are parameters or inputs, not new task families. `two_cycle_count`/`three_cycle_count` come from the same `cycle_length_count(p,k)`; k=1 is fixed_points.
- one_line, cycle, Lehmer, and inversion_vector are representations of the same permutation object; four registered `to_*` endpoints convert output representations.
- The 32 scalar properties include exact correspondences, equal distributions, and joint identities. They are not 32 independent statistical replicates.
- The current library has no separate ascents task. double_ascents counts increasing consecutive triples and must not be treated as ascents.

The mathematical input is a one-line permutation `p=(p_1,...,p_n)`, containing exactly 1 through n. Formal experiments fix n=10. Scalar answers use existing number tokens; the one-line prefix has length `2n+4`(24 at n=10). Adding task token, equality sign, answer, and EOS gives 28 training-input tokens. Only answer and EOS positions contribute supervised loss.

## Callable interfaces

`catalog/data_platform.py` imports no Torch and reads no checkpoints or experimental results. It provides:

| Interface | Contract |
| --- | --- |
| `TASK_ORDER` / `labels(p,tasks)` | Fixed 32-task order from vendored `PROPERTY_FUNCTIONS`; unknown tasks fail immediately. |
| `encode_one_line(p)` / `decode_one_line(ids)` | Lossless, canonical existing token encoding for n=1…30; validates bounds, length, delimiters, and permutation validity. |
| `transform(p,action)` | The eight input actions below, without changing task definitions. |
| `orbit(p)` / `orbit_key(p)` | Full eight-action orbit and lexicographically minimal representative, for exact-duplicate and transformation-equivalent leakage checks. |
| `sample_pool(n,count,seed,excluded_orbit_keys)` | Uniform permutation proposals using Python `Random.sample`, rejecting occupied orbits; returns objects and explicit generation metadata. Capacity/rejection-limit violations raise errors. |
| `validate_pools({role:permutations})` | Independently expands all orbits to check within-pool duplicates and between-pool overlap. Sharing one role across tasks/conditions is intentional. |
| `action_composition_table()` / `unit_checks()` | Eight-action composition table, encoding round trips, group actions, deterministic generation, and leakage-rejection checks. |

To preserve historical reproducibility, the experiment engine may continue using the original NumPy `Generator.permutation` proposal stream. This module's `sample_pool` explicitly uses a different Python RNG implementation; **it does not claim identical inputs from the same seed**. Generator, version, seed, input hash, and role must be recorded together. Both reject occupied orbits after uniform permutation proposals; neither claims equiprobable accepted orbit classes. Variable-length experiments must also declare length sampling; this round uses only n=10.

## Input transforms, answer maps, and proof scope

The three action bits specify whether to execute **I, R, C**, in that left-to-right execution order. I cannot be exchanged with R/C. I is inversion, R one-line reversal, and C complement `C(p)_i=n+1−p_i`.

| action | Mathematical function |
| --- | --- |
| 000 | identity |
| 100 | I |
| 010 | R |
| 001 | C |
| 110 | R∘I |
| 101 | C∘I |
| 011 | C∘R |
| 111 | C∘R∘I |

`DIRECT_RELATION_DEFINITIONS.csv` stores 35 defining relations and proof sketches. `RELATION_GRAPH.csv` closes them and their inverses under composition of input actions and affine answer maps, giving 132 edges (including identity and self-invariance edges). The general form is:

```text
right(T_action(p)) = scale * left(p) + n_coefficient * n + constant
```

`proof_basis` gives the direct-relation path composing each edge. This closure is complete only for **declared relations**, not a complete classification of all mathematics. Beyond identity answer maps, the graph contains nontrivial mappings such as `descents(Cp)=n−1−descents(p)` and their composed-action consequences.

`AGGREGATE_RELATIONS.csv` separately stores joint formulas, which must not be presented as single-source task bijections. For example:

```text
fixed_points(p) + exceedances(p) + deficiencies(p) = n
cycle_count(p) = fixed_points(p) + nontrivial_cycle_count(p)
cycle_count(p) = even_cycle_count(p) + odd_cycle_count(p)
peaks + valleys + double_ascents + double_descents = max(n-2,0)
adjacencies(p) = successions(p) + successions(Cp)
```

Direct/composed relations follow from definition-level bijections of counted sets, partitions, or invertible maps; finite checks validate current implementations. Every relation-graph edge passed all 5913 permutations in S1…S7 plus 2048 random n=10 permutations. These checks do not automatically prove arbitrary new mathematical claims. Aggregate formulas use another fixed verification stream; scope and seed are in `RELATION_AUDIT.json`.

`REPLACEMENT_RELATION_COUNTEREXAMPLES.csv` preserves six candidate pairs verified in v3, including this round's two. For each pair and each of eight actions, two n=7 inputs have the same source scalar answer but different transformed target answers. This excludes a deterministic answer map depending only on that source scalar under that action. It does not imply mathematical unrelatedness or exclude other transforms, multi-source relations, or algorithmic connections.

I/R/C relations declare mathematical connections and isolate equivalent inputs. **Formal pretraining does not construct relation-paired inputs, automatically augment transforms, or supply joint answers.** Each original permutation is labeled independently by its sampled task. A relation's existence does not mean training explicitly presents it.

## Background compositions in this round

Targets are fixed_points and longest_decreasing_run. Their B/C candidate slots are fixed as anti_fixed_points/successions and longest_increasing_run/left_to_right_maxima, respectively. Candidates occupy slot0, and three common-background tasks occupy slots 1…3. Both targets are globally excluded from every source group.

| target | g0_v3 common background | g1_recoils common background | g2_deficiencies common background |
| --- | --- | --- | --- |
| fixed_points | descents, peaks, left_to_right_maxima | recoils, peaks, left_to_right_maxima | descents, deficiencies, left_to_right_maxima |
| longest_decreasing_run | descents, peaks, successions | recoils, peaks, successions | descents, deficiencies, successions |

g1 replaces only slot1 descents with its inverse counterpart recoils. Both have the same Eulerian answer distribution under uniform permutations, but different one-line computational structures and observed learning speeds. descents has been removed; this does not add a coexisting descents↔recoils source pair.

g2 replaces only slot2 peaks with deficiencies, exchanging a local three-position extremum count for a position-inequality count and **substantially changing the answer distribution**. Empirical source-train TV for seeds 17/42/73 is approximately 0.760/0.761/0.765; g1 TV is approximately 0.014/0.010/0.013. Backgrounds are prespecified systematic changes, not a matching search. The g2 effect cannot be attributed to a single local-versus-position factor.

Existing single-task step1200 development accuracy (seed17/42/73): descents approximately 99.90/99.71/99.80%, recoils 62.79/74.02/93.07%; peaks 94.82/96.09/96.78%, deficiencies 99.90/99.22/99.12%. All four approach 100% by 4800. Similar endpoints do not establish matching learning trajectories or multitask difficulty. `SOURCE_REFERENCE_CURVES.csv` and `SOURCE_ANSWER_DISTRIBUTIONS.csv` retain complete evidence for three seeds; `BACKGROUND_MATCHING.csv` quantifies differences without post hoc background selection.

`BACKGROUND_RELATION_GRAPH.csv` restricts the full closure to actual nodes in each target/background/condition, distinguishing target-involving and source-internal relations. `BACKGROUND_AGGREGATE_RELATIONS.csv` lists present/missing members of aggregate formulas. For example, fixed_points g2 contains deficiencies but lacks exceedances, so `fixed+exceed+def=n` does not determine the target from that single background-source answer. It remains a mathematical relation requiring disclosure.

## Data roles and freezing contract

Record data objects, mathematical labels, encodings, and experimental roles separately:

1. Fix task order and definition version; construct/reuse source_train, source_dev, target_train, target_dev, and record input-object/label-array hashes. Compatible historical pools can be reused; new roles require uniform orbit exclusion against all known historical pools.
2. Freeze candidates, selection principles, model, and budget before single-task development or background selection. Existing single-task references require model/encoding/length/data/optimization-history compatibility checks; high- and low-budget records cannot be pooled as one reference.
3. Adaptation conditions share target inputs, development data, and selection budgets. Reset optimizers and identically initialize target-specific parameters. Source restoration and target-row resetting require separate training audits; this module does not replace them.
4. Generate independent tests after every model/LR/checkpoint selection is frozen, including full-orbit exclusion across new test seeds. Tests must not select backgrounds, tasks, hyperparameters, or budget extensions.
5. Save data roles, generator, seeds, n, task order, object hashes, label hashes, orbit audits, and phase times. Majority baselines are defined from training labels and reported by length. This round cannot infer length generalization from n=10 alone.

Before adding tasks, register mathematical identity and parameter/encoding membership, exact definitions, input/output ranges, generation rules, and boundary conventions. Then specify relation transforms/answer maps/proof scope and counterexamples or unit checks. Create a new version directory and freeze it before training; do not silently change old runs' task order, vocabulary, or data hashes to expand the library.

## Reproduction and preservation

The following CPU command at repository root rebuilds the catalog and audits. It writes only the v4 directory, does not modify older files, run models, or generate formal v4 test inputs:

This is a pre-freeze catalog-building command. Do not rerun it in place after freezing: audit times and hashes change. Reproduce experiments using the new-output commands in the experiment README; expand the catalog in a new version directory. Preserve archived v4 catalog/audit artifacts.

```powershell
.venv/Scripts/python.exe experiments/pilot_v4_backgrounds/catalog/build_catalog.py
```

`CATALOG_AUDIT.json` records hashes of scripts, vendored mathematics, old v3 catalog code, and all new catalog/relation/reference tables. `SOURCE_REFERENCE_AUDIT.json` records read v3 source-train/single-task-curve hashes and explicitly states no transfer results were read. `unit_checks` covers canonical encoding round trips,64 action compositions, orbit-key invariance, deterministic small-pool generation, between-pool leakage rejection, and invalid-encoding rejection. Small-n fixtures differ from formal experiment data and do not affect test freezing.

Expansion stops after this round. Platform interfaces do not authorize additional domains, architectures, task counts, or experiments.
