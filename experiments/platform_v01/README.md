# PermWorld Data Platform v0.1

This release implements [TASK_FAMILY_PLAN.md](../pilot_v5_confirmation/TASK_FAMILY_PLAN.md), delivering a unified task interface, exact labels, versioned encoding, reproducible data, and limited single-task calibration. pilot_v1–v5 remain read-only; v5's non-replication and uncertain results are unchanged. This round adds no background variants for the two previous targets, transfer combinations, learning-rate search, or budget extensions.

Source code is in [`permworld_platform/`](../../permworld_platform), version `0.1.0`. Use the existing `.venv` from the repository root; no new dependencies or remote code downloads are required. This research platform depends on the vendor code archived in the repository and is not presented as a standalone wheel package.

## Task counts and engineering scope

| Focus family | Canonical templates | Default declared instances | Nonconstant instances at fixed length | Active input encodings |
| --- | ---: | ---: | ---: | ---: |
| Cycle decomposition | 6 | 8 | 8 | 1 |
| Global subsequences | 5 | 9 | 7 | 1 |
| Permutation blocks | 5 | 7 | 7 | 1 |
| Total | **16** | **24** | **22** | **1 shared encoding** |

- Cycles: `cycle_count`, `nontrivial_cycle_count`, `even_cycle_count`, `odd_cycle_count`, `cycle_length_count(k)`, and `longest_cycle`.
- Subsequences: `lis_length`, `lds_length`, `increasing_subsequence_count(k)`, `decreasing_subsequence_count(k)`, and `rsk_shape`.
- Permutation blocks: `components`, `global_descents`, `largest_component`, `smallest_component`, and `component_length_count(k)`.

The four parameterized templates declare k=1,2,3 by default. The counting API accepts k=1…30; additional parameter values do not create new templates. The CPU example also demonstrates k=4 without counting it as another core task template. Existing endpoints such as shortest_cycle and cycle_type remain callable and are not added to the six-template engineering target for cycles.

All **48 legacy endpoints** are retained, with **6 additional registered endpoints**, for a total of **54**. cycle_length_count parameterizes existing counting capabilities, so these six endpoints are not all new mathematical capabilities. Excluding two parameter aliases and four representation-conversion endpoints, the catalog contains **48 canonical mathematical templates**; this is not a count of mathematical independence. The full catalog declares **62 endpoint instances** by default, including aliases and binary tasks. The default dataset materializes **56 unary, non-alias instances**.

Among the 24 focus instances, I₁=D₁=n, leaving 23 distinct functions on the same input and 22 classes that are nonconstant at fixed length. Further merging under registered input transformations that preserve answer values gives 20 classes/19 nonconstant classes; allowing the components/global_descents add-one mapping gives 19/18. The latter counts apply only to registered relations and must not be called counts of independent tasks. cycle_length_count(k=1,2,3) duplicates fixed_points/two_cycle_count/three_cycle_count; D₂ equals the legacy inversion count length, and I₂ is its complement. See [TASK_COUNTS.json](registry/TASK_COUNTS.json) and the [equivalence audit](mathematics/EQUIVALENCE_AUDIT.json).

The 22 nonconstant instances refer to this release's calibration/data settings n=8/10/12 and k=1/2/3. Small n or k>n introduces additional degeneracies; for example, S₁ has only one input. This count must not be extrapolated to imply nonconstancy at every legal length. Per-length distributions and boundary conditions are recorded separately.

## Unified interface and registry

```python
from permworld_platform import label
from permworld_platform.objects import transform
from permworld_platform.registry import task_spec, task_output_bounds

p = (3, 1, 4, 2)
assert label('cycle_count', p) == 1
assert label('increasing_subsequence_count', p, {'k': 2}) == 3
assert label('largest_component', p) == 4
q = transform('inverse', p)
assert label('cycle_count', q) == 1
print(task_spec('component_length_count'))
print(task_output_bounds('increasing_subsequence_count', 30, {'k': 15}))
```

Inputs are one-based permutations of legal length 1…30; right_multiply_simple additionally requires n≥2 and index<n. Functions reject duplicate values, Boolean inputs, implicit type coercion, missing/extra parameters, and invalid additional operands. All labels are exact Python integers, Booleans, or integer structures; counts with k>n are exactly zero.

[TASK_REGISTRY.json](registry/TASK_REGISTRY.json) and its [CSV version](registry/TASK_REGISTRY.csv) record each task's definition, parameters, input constraints, exact algorithm, output type/range, encoding, family, complexity, timing, and relation evidence. [TASK_INSTANCES.csv](registry/TASK_INSTANCES.csv) lists concrete parameter configurations separately. Timings in [TASK_BENCHMARK.csv](registry/TASK_BENCHMARK.csv) include parameter/input validation and label generation, distinct from dataset generation and training costs.

Legacy binary tasks use `operands={'q': q}` and include compose, conjugate, commutator, and bruhat_leq; full input and answer sequence encoding is executable. This release does not materialize binary data pools or certify splits for binary query groups. Do not extend the unary-pool isolation claims to binary tasks.

## Explicitly versioned model interface

The input encoding is `one_line_v1`; the output protocol is **`permworld_sequence_v1`**. One input encoding is exposed. The upstream cycle/Lehmer/inversion-vector representations remain archived and are not counted as integrated encoding variants in this release.

The vocabulary extends the original 188 entries to **203**, preserving all old IDs. Integers below 100 still use one numeric token; larger integers use the existing `<NUM_START>` / base-100 digits / `<NUM_END>` syntax, and structured answers use new nested-list markers. For example, the maximum increasing subsequence count at n=30, k=15 is 155117520 and round-trips exactly in 8 answer tokens including EOS, without truncation or modulo. rsk_shape outputs the complete partition.

Training supervises every answer token and EOS after the prompt. Evaluation generates autoregressively from the prompt and compares the complete answer; the old pilot's single-answer-token accuracy function is not suitable. `encode_example` returns `prompt_ids`, `prompt_length`, `answer_ids` (including EOS), `input_ids`, and `answer_value`. Next-token supervision starts at logit position `prompt_length-1`.

[MODEL_INTERFACE_COMPATIBILITY.csv](registry/MODEL_INTERFACE_COMPATIBILITY.csv) lists output ranges and context support. The listed instances in the three focus families fit within 128 tokens for legal n≤30. Legacy to_cycle, to_reduced_word, and some binary/parameterized permutation outputs may exceed this context at large n. Exact labels and complete sequences can still be generated, but reading them into a fixed 128-token context raises an explicit error. Configure a longer model context separately; silent truncation is prohibited. The new vocabulary means old checkpoints cannot be loaded directly without an explicit migration. All calibration models in this round start from new initializations.

## Fixed data and a complete reading example

The canonical data manifest is [dataset/manifest.json](dataset/manifest.json). The two data/initialization units are 2026102001/617 and 2026102002/829. Each unit includes n=8,10,12, with train=1024, dev=256, and test=256 per length: 18 pools, 9,216 permutations, and 516,096 materialized labels in total.

The generator creates objects first, then calls the unified label function. Each pool uses `PCG64(SeedSequence([data_seed,role_index,n]))`. Uniform permutation proposals are processed in a fixed seed, train/dev/test, and length order, rejecting full inverse/reverse/complement orbit intersections with historical v1–v5 inputs, other roles, and other repeat units. The manifest records proposal/rejection counts, input-content hashes, file hashes, and the instance list. Uniform proposals with sequential orbit exclusion do not give uniform sampling of orbit classes and introduce finite-population dependence across units. This release does not claim strictly IID input pools.

Actual verification is recorded in [quality_full.json](dataset/quality_full.json): RNG replay, recomputation of all labels, duplicate-input checks, and cross-role/unit/history orbit checks passed. [answer_distributions.csv](dataset/answer_distributions.csv) retains per-length answer histograms, observed-constant flags, and training-derived majority baselines. Constant instances and concentrated distributions are retained; observing a constant answer in a small sample does not prove mathematical constancy.

The CPU end-to-end example has been run:

```powershell
.venv/Scripts/python.exe experiments/platform_v01/examples/demo.py
.venv/Scripts/python.exe -m permworld_platform sample --count 2
```

The first command generates a separate small example pool, materializes labels, and reads model samples back. Historical exclusion is explicitly disabled for this interface demonstration, which is not used in calibration. The second command reads canonical data. In Python:

```python
from permworld_platform.data import read_samples

sample = next(read_samples('experiments/platform_v01/dataset', 617, 'train',
                           'increasing_subsequence_count', {'k': 3}))
assert sample['input_ids'][sample['prompt_length']:] == sample['answer_ids']
```

`read_samples` checks object/label file hashes, per-row object IDs, and exact answers. `load_objects` is a low-level convenience reader and does not independently perform full certification. Canonical training additionally freezes the vocabulary, code, and data. Standalone users should retain the corresponding vocabulary.json and check it with `verify-data`, rather than automatically using an updated vocabulary with an old model.

The default generator materializes all 56 unary instances; custom configurations can specify tasks and parameters through `task_instances`. Configurations must satisfy every instance's length constraints. Insufficient input space, an exceeded proposal limit, or a request for binary data produces an explicit error instead of silently repeating samples.

## Relations and correctness

[RELATIONS.json](mathematics/RELATIONS.json) contains 45 newly registered relations and 29 references to v5 certificates. Single-task transformation relations, joint-query relations, and each query's input identifier and transformation are recorded separately. Answer mappings execute through safe JSON expressions without eval. Formula proofs, finite implementation verification, empirical associations, and unchecked items have separate statuses. An unregistered relation is unclassified, not evidence of being "unrelated."

Mathematical verification exhausts S₁…S₇ and adds 64 fixed random inputs at each of n=8/10/12, for 6,105 fixtures. All 272,741 independent label checks and 445,581 pointwise checks of new relations passed. Independent algorithms use union-find, subset enumeration, RSK growth diagrams, and set-based cuts. Historical v5 relations are checked through certificate and definition-source hashes without repeating enumeration. See the [mathematical documentation](mathematics/README.md) and [CERTIFICATE.json](mathematics/CERTIFICATE.json).

## Limited single-task calibration

Two representatives per family were frozen before training: cycle_count and longest_cycle; lis_length and increasing_subsequence_count(k=3); components and largest_component. They cover count/extremum, longest existence/fixed-length count, and block count/largest block, respectively. k=3 avoids the k=1 constant and the k=2 complement relation with the legacy inversion count. Selection does not depend on this round's results.

Two seeds per task give 12 runs. Each run uses 2400 updates, batch size 64, and exactly 800 updates per length; a single learning rate 3e-4 with 100 warmup steps and cosine decay to 3e-5. The Transformer retains 4 layers, width 256, and 8 heads, with BF16 training, FP32 evaluation, and TF32 disabled. Full train/dev curves and per-length metrics are saved every 200 steps. The fixed endpoint is the primary calibration reference. The best development checkpoint is descriptive only, selected by accuracy, answer CE, then earliest step. No learning-rate search was added. Development selection is not independent test performance.

Within a seed, tasks share initial weights and object pools, with a fresh AdamW for each task. Sample and length schedule hashes are recorded for verification. Different multi-token answer lengths change actual token counts and dropout RNG consumption; equal update/sample budgets therefore do not imply equal tokens or identical dropout masks.

The main dataset's test split is reserved for the platform and was not used to evaluate calibration models or select representatives. Data-quality checks inspect test objects/labels, which differs from inspecting model test performance. Mathematical fixtures and the CPU example were never used for training.

Results are available in [CALIBRATION_REPORT.md](calibration/CALIBRATION_REPORT.md), with raw per-seed CSVs, per-length CSVs, curves, and cost tables in the same directory. Majority baselines are determined from training labels at each length. The data-distribution table breaks class ties using canonical JSON string order; calibration uses the smallest numeric value. The six representatives are unaffected. A few nonrepresentative structured tasks have different valid tie choices; mixing these conventions must not be described as a change in model performance.

## Safe reproduction: new directories, same configuration

The canonical files are evidence of the completed run. Do not re-export timed registries or overwrite mathematical certificates in their original locations. To read or verify:

```powershell
.venv/Scripts/python.exe -m permworld_platform sample
.venv/Scripts/python.exe -m permworld_platform.calibration verify
.venv/Scripts/python.exe experiments/platform_v01/calibration/audit.py --forwards
```

To regenerate data with the same configuration and repeat all 12 calibration runs, use new directories:

```powershell
.venv/Scripts/python.exe -m permworld_platform generate --out experiments/platform_v01_reproduction/dataset
.venv/Scripts/python.exe -m permworld_platform verify-data --out experiments/platform_v01_reproduction/dataset
.venv/Scripts/python.exe -m permworld_platform.calibration smoke --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data
.venv/Scripts/python.exe -m permworld_platform.calibration freeze --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data
.venv/Scripts/python.exe -m permworld_platform.calibration train --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data
```

Reproduction reuses this release's canonical registry and mathematical certificates; specify new directories for re-export or re-verification. A data pool with an existing manifest only accepts verification with the same configuration; different configurations or incomplete pools cannot be overwritten. Training reuses only complete runs with matching hashes. Automatic update-level resumption of incomplete training is not guaranteed. Checkpoints remain local, with large .pt files excluded from Git by project convention; source, configuration, data, CSVs, figures, and audits are committed.

## Scope of this round

The platform provides correct generation and reading interfaces, without guaranteeing that every task is learnable with the current model/budget. Two seeds and three lengths present in training provide limited calibration, not evidence of unseen-length extrapolation or a complete sample-efficiency law. Difficult tasks, constant instances, unverified relations, and encoding/context limitations are retained. Changes to training protocols or combinations require a separately frozen plan. This release does not automatically add transfer combinations.