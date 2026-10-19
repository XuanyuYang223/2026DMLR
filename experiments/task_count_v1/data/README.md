# Frozen object pools for task-count experiment v1

The generator is [`../data_pipeline.py`](../data_pipeline.py). It imports the
unchanged platform permutation validator, complete inverse/reverse/complement
orbit key, and historical pilot archive reader. Task labels are deliberately not
stored in these archives: the training pipeline uses the platform's exact task
oracles and versioned full-answer encoding.

## Configuration and pairing

| Repeat | Model initialization seed | Data seed | Reserved test seed |
| --- | ---: | ---: | ---: |
| 1 | 1103 | 2026102101 | 2026102151 |
| 2 | 2207 | 2026102102 | 2026102152 |
| 3 | 3301 | 2026102103 | 2026102153 |

At each of lengths 10 and 12, each repeat has 4,096 `source_train` inputs,
256 `source_dev` inputs, 128 `target_train` inputs, and 256 `target_dev` inputs.
The target adaptation sample size is therefore 256, balanced across lengths.
A separate final stage creates 512 `test` inputs at each length for each repeat.
Sampling seeds used by model training are recorded by the training configuration;
they are separate from the object-generation seeds in this table.

All tasks and conditions in a repeat share the relevant object pools. This keeps
the object distribution paired while allowing task definitions and labels to
change. Every object in every role and repeat occupies a unique complete I/R/C
orbit. In particular, development and test transformations of a training input
are also excluded.

The exclusion history consists of the object NPZ files from pilots v1 through
v5, the canonical platform v0.1 dataset, and the platform's runnable example.
`manifest.json` binds all 97 historical archives by SHA-256. No historical target
performance is read by this generator.

## Capacity and reproducibility

Before sampling, the generator subtracts occupied historical orbits from the
conservative lower bound `floor(n! / 8)`. The remaining lower bounds are 325,071
at length 10 and 59,872,104 at length 12. The full request is 15,744 new orbits
per length, including 1,536 reserved final-test orbits. Both checks pass without
weakening isolation.

Each pool uses `PCG64(SeedSequence([data_seed, role_index, n]))`. The role indices
are 0, 1, 2, and 3 in the order shown above. Test pools use `test_seed` and role
index 4. The generation order is repeat, role, then length for training and
development; all test pools come afterward in repeat, length order. Proposals
are uniform permutations, accepted only if their canonical orbit is unoccupied.
This is uniform proposal sampling with rejection, not uniform sampling over
orbit classes. The finite-pool exclusion also creates dependence between pools;
it must not be described as unrestricted independent sampling.

The primary manifest is immutable and records each pool's raw file hash,
canonical input-content hash, ordered orbit hash, proposal count, rejection
count, and timing. Generation refuses partial-output overwrites. `verify`
checks all archived inputs, history hashes, full orbit separation, and exact
RNG replay. `load_objects` checks archive and canonical input-content hashes
before returning an `int16` matrix.

The pretest verification covers 28,416 objects in 24 pools, with zero duplicate
or overlapping orbits. Reserved final tests add 3,072 objects in six pools.

## Final-test gate

The test-generation entry point requires a selection manifest with
`status="frozen"`, `training_complete=true`, a timezone-aware
`frozen_at_utc`, and exactly 108 final target-training selections. The runner is
responsible for constructing this manifest only after all 24 pretraining runs
and 108 target fits have finished. Test generation then binds the selection
manifest's SHA-256 and timestamp. It refuses an altered selection manifest or
changed historical inputs. `test_manifest.json` is separate, so generating tests
does not modify the primary data manifest frozen before training.

The final test pools are used only for the unified final evaluation, with no
subsequent training, configuration choice, or task substitution.

## Commands and reader example

Run from the repository root using the project's Python environment:

```powershell
.venv/Scripts/python.exe experiments/task_count_v1/data_pipeline.py capacity
.venv/Scripts/python.exe experiments/task_count_v1/data_pipeline.py prepare
.venv/Scripts/python.exe experiments/task_count_v1/data_pipeline.py verify
```

For a fresh reproduction, pass `--out experiments/task_count_v1_reproduction/data`
to each command and point the experiment runner at that same directory. Do not
overwrite the released manifests or model selections.

```python
from experiments.task_count_v1.data_pipeline import load_objects
from permworld_platform.encoding import encode_example

pools = "experiments/task_count_v1/data"
objects = load_objects(pools, 1103, "target_train", 10)
p = tuple(map(int, objects[0]))
example = encode_example("cycle_count", p, max_seq_len=128)
assert len(objects) == 128
assert example["input_ids"] == example["prompt_ids"] + example["answer_ids"]
```

After the actual full training and selection freeze, the runner invokes
`generate_test(out, selection_manifest_path)`. Final verification is:

```powershell
.venv/Scripts/python.exe experiments/task_count_v1/data_pipeline.py verify --include-test
```
