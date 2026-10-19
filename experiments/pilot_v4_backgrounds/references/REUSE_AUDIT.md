This round uses only `fixed_points` and `longest_decreasing_run`, with paired model seeds 17,42,73. The inventory reads v3 training/development records and checkpoints only. The script imports no training engine, runs no model, and reads no old/new test inputs, predictions, or accuracies. All checkpoints load with `map_location='cpu'`.

```powershell
.venv/Scripts/python.exe experiments/pilot_v4_backgrounds/references/reuse_inventory.py
```

Exact-reuse plan

| Content | Reusable count | Conditions |
|---|---:|---|
| A candidates at three LRs | 18, forming 6 selection groups | Identical initialization, target,256 training examples, development set, sampling, and selection rule; backgrounds share one A. |
| g0 B/C pretraining | 12 | Identical ordered source tuples, source-training pool, initial model state, sampling, and 4800-step LR trajectory. |
| g0 B/C adaptation candidates | 36 | Identical target/source state, target-row reset, data,800-step budget, and three LRs. |
| g1/g2 pretraining | No existing exact match;24 new runs | g1 replaces only common slot1 descents with recoils; g2 replaces only common slot2 peaks with deficiencies. |
| g1/g2 adaptation candidates | 72 new runs | Fresh adaptation from each new source state, with reset optimizer state. |
| Existing single-task references | 24 | Eight actual source tasks×three seeds; step1200/4800 records are usable with exposure/optimization-step differences disclosed. |

There are 126 unique adaptation candidates and 42 unique development-selection groups. Displaying A/B/C for three backgrounds produces 54 logical rows; repeated A displays are not new runs or independent observations.

Ordered g0 compositions follow. Both interventions change only the specified common slot; related/replacement candidates in slot0 remain fixed.

| Target | B | C |
|---|---|---|
| fixed_points | anti_fixed_points, descents, peaks, left_to_right_maxima | successions, descents, peaks, left_to_right_maxima |
| longest_decreasing_run | longest_increasing_run, descents, peaks, successions | left_to_right_maxima, descents, peaks, successions |

Full v3 has 24 source-training records,21 distinct ordered-composition×seed units, and 18 unordered-composition×seed units. v3 `left_to_right_minima/C` and `longest_decreasing_run/B` have identical ordered source lists and actual parameter hashes for each seed, and can alias the same source checkpoint. Already adapted weights cannot be reused across targets. `fixed_points/C` and `longest_decreasing_run/C` share only an unordered set; swapping slots 0/3 changes task-label assignment to inputs, so this is not exact reuse.

Seeds and data provenance

| Random quantity | seed17 | seed42 | seed73 |
|---|---:|---:|---:|
| Model parameter initialization | 17 | 42 | 73 |
| Train/dev input-generation stream | 20261018 | 20261018 | 2026101803 |
| Source sampling and Torch/CUDA dropout | 1017 | 1042 | 1073 |
| Target-adaptation sampling and Torch/CUDA dropout | 2017 | 2042 | 2073 |

Model seeds are not data-generation seeds. v1 uses one continuous NumPy generator 20261018 to generate development_train, development_dev, then seed17 pretrain/source_dev/target_train/target_dev/test, and then corresponding seed42 pools. v3 reuses seed17/42 inputs and recomputes 32 label columns; target training uses the first 256 examples from the original 1024 pool. seed73 uses the v3 stream 2026101803 to generate source_train8192, source_dev1024, target_train256, target_dev1024 in order, excluding existing full inverse/reverse/complement orbits. All targets/conditions share each seed's pools.

`seed(s)` sets Python random, global NumPy, Torch CPU, and all CUDA seeds. Training separately creates NumPy `default_rng(s+1000)` or `default_rng(s+2000)` for batch sampling. All conditions/LR candidates use identical sampling/dropout seeds. Each step uses 64 distinct indices; source-task slots are shuffled with 16 examples per task. Changing the ordered task list changes the slot-to-label binding; do not match only sets.

Configuration and resetting

- Model: original Transformer with width256,4 layers,8 heads, FFN multiplier 4, dropout0.1, context128, vocabulary188, and tied embedding/output; one-line, fixed n10.
- Source training has 4800 updates:100-step warmup to 3e-4, cosine decay through 2400 to 3e-5, then constant 3e-5 through 4800. The cosine schedule is not stretched anew over 4800 steps.
- Each adaptation candidate has 800 updates at LR3e-4,1e-4, or 3e-5;100-step warmup and cosine decay to 0.1 of initial LR. Full training 256/development1024 evaluation occurs every 100 steps. Excluding step0, select highest development accuracy, lowest development answer CE, earlier checkpoint, then prespecified LR order.
- Adaptation restores only the existing embedding row for the target query token. Because output and embedding share parameters, the output row restores simultaneously. No new head is added; number-answer rows and other source weights transfer. AdamW is recreated with 0 state entries; all endpoint optimizer steps are 800.
- Mask semantics: attention uses strict causal and valid-token masks; loss applies only to answer/EOS with weight 0.5 each. Classification always spans vocabulary188, without a numbers-only class mask. Test answers use only the prompt; EOS is separately evaluated conditioned on the gold answer.

`reuse_inventory.py` recomputes adaptation-initial-state hashes from actual source parameters and initial target rows, checks optimizer counters, sampling-RNG terminal states, tied-row values, model shapes, and parameter finiteness, and recomputes selections from development curves. Any change to inputs, seeds, targets, task-slot order, the three-LR grid, or selection rule breaks exact-reuse prerequisites at the corresponding level.

Original v3 metadata schema for the engine

`multi/s{seed}_{target}_{B|C}/completed.json`:

- `seed`,`tasks`(ordered list),`kind='multi'`,`updates=4800`,`input_tokens=8601600`,`supervised_tokens=614400`,`sample_exposures=307200`,`task_exposures=76800`,`task_fraction=.25`.
- `checkpoint`,`checkpoint_sha256`,`model_state_hash`,`initial_state_hash`,`schedule_sha256`,`train_sha256`,`dev_sha256`, and`actual_unique_per_task`.
- Resolve `checkpoint` relative to **v3 artifacts**, not its completion directory. fixed17/42 actually point to v2; `new_compute` describes new compute in v3 at that time, not v4 costs.
- `curve.csv` sits beside completion metadata. New v3 sources have full train/dev measurements at 0/600/1200/2400/3600/4800; historical fixed17/42 sources lack early curves, with only 2400 and continuation points. Missing points must not be fabricated.

`adapt/s{seed}_{target}_{A|B|C}_lr{lr:.0e}/completed.json`:

- `seed`,`target`,`condition`,`lr`,`best`,`endpoint`(including step, full train/dev accuracy/CE/count),`updates=800`, and budget.
- `best_path`,`endpoint_path`,`best_sha256`,`endpoint_sha256`,`final_model_state_hash`,`schedule_sha256`. Paths are likewise relative to **v3 artifacts**; follow them rather than assuming weights reside beside completion metadata.
- `v3_train_sha256`/`v3_dev_sha256` uniformly bind current v3 data with 32 label columns. v4 should use these top-level fields rather than directly comparing nested legacy fields.
- `initialization` fields `source_state_hash`,`loaded_state_hash`,`adapt_initial_state_hash`,`target_row_hash`,`reset_row_ids`,`optimizer_entries=0`,`optimizer_state_reset=true`,`new_output_head=false` verify actual loading/resetting.
- **Legacy exception:** reused fixed_points seed17/42 records have `initialization.source_path` relative to the old `experiments/` directory; newer v3 records resolve relative to v3 artifacts. Their nested train/dev hashes bind old six-column-label npz files, and `train_slice=[0,256]`. Validate actual original paths, then use top-level v3 data hashes to establish input consistency for the new experiment.
- best.pt usually contains model, config, seed, step, lr, condition, and metrics but no optimizer. endpoint.pt additionally contains optimizer and NumPy/Torch/CUDA RNG states. Checkpoints inherited from v2 may lack `target`; bind it through verified v3 completion metadata and task tokens.

v4 should create its own logical-run records pointing to old files, retaining `reused_from`, source-completion SHA, weight SHA, curve SHA, and new logical background, instead of rewriting old checkpoints or records. Generate new tests independently after v4 configuration freezes, excluding all v1/v2/v3 input orbits. This inventory did not read those test contents.
