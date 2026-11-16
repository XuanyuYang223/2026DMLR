# Architecture-replication data protocol

This implementation creates only new files under this experiment directory. It
does not modify historical pools, manifests, configurations, checkpoints, or
results. Data generation is independent of the architecture selected by the
development gate: `data_pipeline.py` imports `plan.py`, not the selected-model
configuration. It contains no model construction, forward pass, or training.

## Input scope and reservations

All new objects are permutations of length 10. One object is admitted per complete
inverse/reverse/complement orbit. The separate development-only architecture gate
uses 8,192 training objects and 1,024 development objects; both gate tasks share
these inputs. There is no gate test pool. These are engineering inputs, not an
additional confirmatory repeat.

Each of eight new formal units uses 8,192 source-training, 1,024 source-development,
1,024 target-training, 1,024 target-development, and 2,048 deferred test objects.
The target-training maximum pool receives one frozen random permutation; its
prefixes give nested 256/512/1,024 subsets. All P/D/T/U/A routes within a unit use
the same corresponding target objects. Source conditions share source objects.

Before generating the gate, the capacity check reserves all 9,216 gate objects
and 106,496 formal objects, including the 16,384 deferred test objects. These are
required counts from `plan.py`, not a claim that capacity has already passed.
The conservative available count is `floor(10! / 8)` minus the number of occupied
historical length-10 orbit keys. An insufficient bound or exhausted proposal cap
stops execution; no overlap or isolation rule is relaxed. The gate remains
excluded if the GRU fails and the prespecified Transformer fallback is used.

## Historical coverage and classification

The read-only historical readers and the complete `experiments/**/*.npz` scan
cover pilot v1–v5, platform v0.1 canonical and example inputs, task-count v1/v2,
relation-readout inputs, engineering pools, and all 40 pools from the original
fixed-points confirmation. Original fixed-points protocol, data configuration,
primary/test manifests, every declared pool, and transitive historical bindings
are verified by SHA256. Archived test inputs are excluded as historical objects;
their labels and model predictions do not inform selection.

The preimplementation filename/archive-member inventory contains 219 object
archives and 395 exposure-count archives. The object schemas are `p`, `p+y`, and
the explicitly identified original pilot CKA interface smoke archive containing
`p` alongside features. Only `p` contributes input objects. The 395 original
`input_exposure_counts.npz` files contain nonnegative integer `counts`; they are
bound as non-object accounting records and never converted to permutations.
Unknown archive provenance or schema fails rather than being silently ignored.
There are no NPZ files under the inspected `release_checks/` or `reference/`
directories. The implemented exhaustive scan is of DMLR experiment NPZ inputs,
not a claim about every possible external dataset format.

Repeated references, copied archives, and Stage 2's read-only reuse of canonical
platform inputs do not create additional historical orbit keys. The explicit
sorted exclusion list, classified inventory, original file hashes, and capacity
record are saved before sampling. Historical objects at other lengths remain in
the provenance record but do not consume length-10 capacity.

## Generation, pairing, and immutable records

Gate train/development pools are generated first. Formal non-test pools follow
in repeat order R1–R8, and within each repeat in source-train, source-development,
target-train, target-development order. Tests are generated last in repeat order.
The generator uses uniform permutation proposals, rejecting any already occupied
full orbit. Formal generation adds both gate pools to the historical exclusions,
so no formal role or unit can overlap gate inputs.

PCG64 seeds, role indices, and the target-subset seed are recorded in `plan.py`
and the manifests. Non-test pools use `SeedSequence([data_seed, role_index, 10])`;
formal tests use the separate test seed and role index 4. Every pool records its
file hash, ordered input digest, orbit digest, accepted generation-order digest,
proposal/rejection counts, and elapsed time. The target maximum pool additionally
records its full subset permutation and all three prefix digests. Verification
can replay the exact generator without writing new data pools.

The gate manifest lives at `artifacts/gate_data/manifest.json`; the two archives
are `objects/GATE_train_n10.npz` and `objects/GATE_dev_n10.npz`. The loader is
`load_gate_pool("train" | "dev", out=GATE_OUT)`. Formal manifests and archives
retain the original interface under `artifacts/data/`, with files such as
`objects/R1_target_train_n10.npz` and `load_pool(DEFAULT_OUT, "R1", "target_train")`.
The stored arrays use `p` with shape `(count, 10)` and dtype `int16`.

Historical files are never written. Existing new pools and metadata are never
overwritten. Repeated verification writes a fresh timestamped audit when its
canonical audit filename already exists. An interrupted partial generation is
preserved and rejected for automatic overwrite. Frozen generation code changes
invalidate its existing manifests.

## Test-release gate

No new test objects exist during architecture choice, source training, target
adaptation, or development selection. Test generation requires this experiment's
canonical `artifacts/SELECTIONS_FROZEN.json`, bound to
`artifacts/FROZEN_PROTOCOL.json`, the new data manifest, initialization manifest,
and frozen source files. The gate checks all 32 source completions and all 360
adaptation candidates; their seeds, input digests, initial-model bindings, source
checkpoint hashes, target-row reset and fresh-optimizer records; and the selected
and endpoint checkpoint file hashes. It rechecks the prescribed development
ranking and all 120 selected unit/route/size combinations. It does not rerun models.

The frozen selection timestamp must follow every required completion. The test
manifest binds the complete selection and completion inventory, protocol, primary
data manifest, and checkpoint hashes. Old R1–R8 selections cannot authorize tests
for the new study. New test inputs are generated against historical, gate, all
formal non-test, and earlier new-test exclusions. A changed history, partial test
generation, or mismatched selection freeze fails rather than silently refreshing
the test set.

## Commands and execution order

Run from the repository root with the intended environment. The commands below
are the interface, not a record that generation or training has already run.

```powershell
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/data_pipeline.py capacity
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/data_pipeline.py prepare-gate
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/data_pipeline.py verify-gate
```

Freeze the separate gate protocol, run its authorized development-only training,
and record the prespecified architecture decision. Then prepare the formal data,
freeze the formal experiment with its new initializations, and train:

```powershell
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/data_pipeline.py prepare
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/data_pipeline.py verify
```

Only after every development selection is frozen:

```powershell
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/data_pipeline.py generate-test --selection-manifest experiments/fixed_points_deficiencies_arch_v1/artifacts/SELECTIONS_FROZEN.json
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/data_pipeline.py verify --include-test
```

`--no-replay` still checks all hashes and full orbit separation, but omits exact
RNG replay. It is recorded explicitly. `capacity` reads inputs and prints a JSON
result without generating pools. All lengths other than 10 are excluded from this
new experiment.

## Sampling limitation

The permutation space is finite. Uniform permutation proposals followed by orbit
rejection are not independent uniformly sampled surviving orbits. Global
historical exclusion and sequential admission condition later pools on earlier
ones. Separate units have disjoint input orbits and new seeds, but this finite
sampling design must remain visible in the statistical assumptions. New results
are reported separately from the original eight-unit confirmation.
