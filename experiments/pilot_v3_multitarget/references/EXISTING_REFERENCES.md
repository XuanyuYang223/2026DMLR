Existing single-task reference audit

`candidates_existing.csv` retains per-model development evidence and budgets, distinguishing single-task, four-task, and adaptation records. Reproduce with:

```powershell
.venv/Scripts/python.exe experiments/pilot_v3_multitarget/references/inventory_existing.py
```

The local `reference` checkouts contain no `.pt`, `.pth`, or `.safetensors` files. The NeurIPS specialist and TMLR one-line Transformer are compatible in width 256,4 layers, 8 heads,vocabulary 188, and most optimizer settings. NeurIPS `math_ops.py`, `models.py`, and `passage.py` are byte-identical to the pilot's vendored files. However, those historical models trained over lengths 2..30 or 10..30 for 20,000 updates on different training pools, so they are not paired single-task controls for this round's fixed n10 and 4,800 updates.

Directly usable compatible local single-task records are v1's fixed_points development model (8192 examples, 2400 updates, 100% development accuracy, 34.67% majority baseline) and v2 condition A (256 examples, 800 updates). The former supports retaining the original target; the latter provides a historical small-sample reference. Neither replaces new single-task source baselines, and new data in this round should not directly reuse those old adaptation weights.

Candidate evidence from the TMLR eight-relations table, using n10, identity, validation, and full-vocabulary answer accuracy:

| Task | Accuracy | Training-defined majority baseline | Known pairing |
|---|---:|---:|---|
| exceedances | 100% | 38.28% | deficiencies, inverse |
| left_to_right_minima | 100% | 33.98% | left_to_right_maxima, complement; right_to_left_minima, reverse |
| longest_decreasing_run | 100% | 52.73% | longest_increasing_run, complement |
| descents | 100% | 35.94% | recoils, inverse |
| recoils | 98.83% | 38.28% | descents, inverse |
| peaks | 98.83% | 51.95% | valleys, complement |
| valleys | 100% | 42.97% | peaks, complement |

Each row above has only 256 development examples and comes from one initialization, 95100001. These records can propose candidates only; candidates must pass this round's fixed n10 configuration development gate before freezing. Prioritize exceedances, left_to_right_minima, and longest_decreasing_run, covering position comparisons, records, and contiguous-substring statistics. Related sources deficiencies, right_to_left_minima, and longest_increasing_run cover inverse, reverse, and complement. These three symmetry operations are not statistically independent task families. Valleys could replace one candidate if local peak/valley tasks are desired, but must first pass this round's learnability check.

To retain the original target anchor and save computation, `reusable_local_checkpoints.csv` separately lists 42 local files actually loaded and checked on CPU: the original step0 for both seeds, four B/C4800-step continuations, and best/endpoint checkpoints for A/B/C at three learning rates. Checks verify 188×256 embeddings, tied-output values, and saved optimizer step counts; file and data SHA values are recorded individually. Paths are `experiments/pilot_v2_diagnostics/artifacts/extended/seed{17,42}_{B,C}_4800.pt` and `adapt/s{seed}_{condition}_p{budget}_lr{lr}/`. The first 2400 pretraining steps come from v1 source checkpoints; the latter 2400 inherit the optimizer and RNG with fixed learning rate 3e-5. Adaptation uses a new AdamW and 800 updates. **Reuse only for the fixed_points anchor with exactly unchanged task order, data, and adaptation selection rules.** Generate an independent new test set after this round freezes; previously viewed v1/v2 test results cannot serve as independent confirmation for this round.

Historical results cannot assign fixed difficulty labels to tasks. In another TMLR development setting with 8k updates,lr1e-3, dropout 0, and lengths 10..12, Transformer descents averages 56.67% accuracy and recoils 99.87%, reversing their ordering in the original 20k setting. Multiple factors change simultaneously, preventing attribution to any single cause.

The minimal additional training set is the deduplicated union of final B/C sources, plus new targets that have not passed the same configuration's development gate; the full 32-property library need not be retrained. All candidates share fixed training/development inputs and retain n10 learning curves, train/dev losses, full-vocabulary accuracy, and training-defined majority baselines. If single-task training fails after 800 or 4800 steps, describe it as not fully learned under this configuration and budget. If a task is learned alone but worsens in a mixture, still explain the per-task exposure difference between single-task and four-task training; do not equate it with causal proof of interference. A stricter matched-per-task-exposure control can follow later.
