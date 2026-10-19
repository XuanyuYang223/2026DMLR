# PermWorld v0.1 Single-Task Calibration

This round preselected six tasks, two seeds, and a uniform budget of 2400 updates. It performs no transfer-combination search and removes no task based on performance. The endpoint is the primary fixed-budget reference; the development-best checkpoint is descriptive only. The platform test set was not used in calibration.

The model retains the 4-layer, width-256, 8-head Transformer architecture. For the explicitly versioned new-task and structured-answer interface, the vocabulary expands from the historical 188 entries to 203, yielding 3,244,288 parameters. Its parameter space is not identical to that of historical checkpoints. For each seed, all six tasks start from the same new initialization and each creates AdamW with empty state; no legacy transfer weights are loaded.

Training uses lengths 8, 10, and 12, with 1024 training inputs and 256 development inputs per length. The fixed seed shuffles the length order every three updates, giving exactly 800 updates per length. All tasks share each repeat's data and initial model weights. All three lengths occur in training, so the per-length results are not tests of unseen-length extrapolation.

Accuracy requires generating the entire answer and EOS correctly from the prompt alone. Cross-entropy uses teacher forcing and is a different metric. The output encoding is explicitly versioned to accommodate increasing_subsequence_count(k=3) answers beyond the original single-token range. All six representatives have scalar outputs. The platform correctly encodes structured answers, but this round does not calibrate the learnability of every structured-output task.

| Task | Mean endpoint training accuracy | Mean endpoint development accuracy | Two-seed development range | Development majority baseline | Above majority baseline |
|---|---:|---:|---:|---:|---:|
| cycle_count | 73.88% | 42.84% | 41.93–43.75% | 32.68% | +10.16 pp |
| longest_cycle | 57.41% | 22.53% | 20.57–24.48% | 16.99% | +5.53 pp |
| lis_length | 97.62% | 86.26% | 85.03–87.50% | 41.80% | +44.47 pp |
| increasing_subsequence_count{"k": 3} | 46.61% | 25.26% | 24.48–26.04% | 4.36% | +20.90 pp |
| components | 99.97% | 99.74% | 99.74–99.74% | 77.08% | +22.66 pp |
| largest_component | 99.93% | 99.02% | 98.96–99.09% | 77.08% | +21.94 pp |

Overall means do not conceal the difficulty of cycle tasks: at length 12, cycle_count has mean development accuracy of only 35.74%, below that length's majority baseline of 36.72%. longest_cycle has mean training accuracy of 57.41% and development accuracy of 22.53%, showing both incomplete fitting and a generalization gap. These cannot be attributed to a single mechanism.

increasing_subsequence_count(k=3) reaches 25.26% development accuracy, exceeding the majority baseline by 20.90 percentage points, but its training accuracy is only 46.61%. It remains listed as insufficiently learned under the current configuration. Both permutation-block representatives reach development accuracy near 99%, clearly above their roughly 77% majority baseline. These are limited-budget development observations, not guarantees for entire task families.

Majority answers are selected only from training labels for each task and length, then applied to development data. High raw accuracy may reflect answer imbalance, so gains over majority and per-length curves must also be considered. With only two seeds and development-based checkpoint selection, these results cannot establish reliable transfer ability or test performance. All insufficiently learned tasks are retained; inadequate budget remains inseparable from encoding/optimization difficulty. This round explicitly changes input lengths, training sample counts, and encoding interfaces, so its accuracies are not a controlled comparison with v5.

The 12 training runs used 28,800 updates, 1,843,200 sample exposures, and 52,841,514 actual input tokens. Right-padding computation is reported separately. Cumulative training time was 14.23 minutes, including full train/dev evaluation and saving.

- [Per-seed and per-length results](results_by_seed.csv)
- [Complete train/dev curves](learning_curves.csv)
- [Actual exposures for each input](input_exposures.csv)
- [Per-length budgets](length_exposures.csv)
- [Run costs](compute.csv)
- [Per-length figure](figures/endpoint_by_length.png)
- [Full development-forward and checkpoint audit](CALIBRATION_AUDIT.json)

Run reproduction commands from the repository root. First generate and verify a fresh data directory at `experiments/platform_v01_reproduction/dataset` using the [platform reproduction instructions](../README.md). All outputs below go to a new directory; freeze binds its data and the code hashes at that time. Complete existing runs are skipped after hash verification. Incomplete runs restart from the same initial model, without a guarantee of update-level resumption. The archived SMOKE file is also frozen evidence, so repeat smoke tests in a new directory.

```powershell
.venv/Scripts/python.exe -m permworld_platform.calibration smoke --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data
.venv/Scripts/python.exe -m permworld_platform.calibration freeze --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data
.venv/Scripts/python.exe -m permworld_platform.calibration train --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data
.venv/Scripts/python.exe experiments/platform_v01/calibration/summarize.py --out experiments/platform_v01_reproduction/calibration
.venv/Scripts/python.exe experiments/platform_v01/calibration/audit.py --forwards --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data
```
