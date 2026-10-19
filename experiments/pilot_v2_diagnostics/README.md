# PermWorld second-round diagnostics

This round fixes n=10, target fixed_points, the B/C four-task compositions, one-line encoding, and the v1 Transformer. It examines only 256 target training examples and paired seeds17/42. All v1 files and checkpoints were preserved byte for byte when this round was conducted; new data, models, curves, audits, and CSV files are in this directory's `artifacts/`. Expansion pauses after this round.

## Frozen comparisons

Each seed has 15 adaptation candidates: 3 for random-initialized A, and 3 each for B/C at source budgets of 2400 and 4800 updates. All use learning rates `3e-4, 1e-4, 3e-5`, at most 800 updates, batch 64, the first 256 examples of the v1 target pool, and the original 1024-example development set. Data, tasks, model, optimizer type, warmup 100, and cosine endpoint ratio 0.1 are unchanged. Full train/dev accuracy, answer CE, EOS CE, and two-token CE are recorded every 100 steps. Training-batch loss is separate and must not be confused with full-dataset evaluation loss.

The same within- and across-candidate rule maximizes development answer accuracy, then minimizes development answer CE, then favors earlier updates, then follows the declared learning-rate order. Step0 is recorded but ineligible. The fixed 800-step endpoint is also saved and reported; no candidate stops early. A has no source budget, so the same A results are reused in both the 2400/4800 comparisons, not counted as independent experiments.

V1 lacks historical per-task source losses, complete target training curves, and intermediate checkpoints. Original source-endpoint losses are recomputed from saved checkpoints. Intermediate trajectories are not fabricated; the new adaptation curves are explicitly v2 replays. High-learning-rate endpoints are compared parameter by parameter with the original v1 endpoints to check whether added diagnostics changed the run. Historical v1 test results retain their exploratory status and are not pooled into v2 test averages.

## Extended pretraining budget

V1 source-development results support incomplete learning. B/C therefore inherit the model, AdamW state, NumPy sampling state, and torch/CUDA random states from the original 2400-step checkpoints and train for another 2400 steps. They retain the same 8192-input pool, 25% sampling per task, and 16 examples per task in each batch. The extension fixes the learning rate at the original cosine endpoint, `3e-5`, without restarting warmup or rewriting the first 2400 steps. **This continues the actual original training trajectory; it is not equivalent to training from scratch with a 4800-step cosine schedule.**

Total input tokens are 4,300,800 and 8,601,600 at the two budgets, including answers/EOS; supervised tokens are 307,200 and 614,400. Per-source exposures increase from 38400 to 76800. The available 8192 inputs and each task's actually visited distinct inputs are reported separately. Full source train/dev metrics are computed at the original endpoint and at added steps 3000/3600/4200/4800.

## Initialization, saving, and isolation

Each adaptation strictly loads the source/step0 state_dict and checks all parameter hashes. The tied token_embedding/lm_head row for `<FIXED_POINTS>` is restored to the same seed's step0 row, with its hash recorded. Shared numeric output parameters retain pretrained values; no independent head is added. A new AdamW is created and its initial state confirmed empty. Paired sampling schedule hashes match across all candidates. Source continuation explicitly inherits the optimizer and is distinct from adaptation.

`FROZEN_PROTOCOL.json` records configuration and training-code hashes before training. After all candidates finish, `SELECTIONS_FROZEN.json` records each selection and checkpoint hash; only then is the new test set generated. Each seed receives 2048 new n=10 permutations, excluding complete inverse/reverse/complement orbits from every v1 pool, including independent development and both old test pools, and from the other v2 seed. The new test is used only for unified final evaluation. Raw test readings for all candidate best/endpoint checkpoints are retained; the main table uses only frozen selections, with no further test-based candidate selection.

Majority baselines are defined by the corresponding training-label mode and reported by length. Only length 10 is used here; no cross-length diagnosis is claimed. Answer accuracy uses the same prompt-only prediction and full 188-vocabulary argmax as v1. EOS accuracy is conditioned on the gold answer, not freely generated whole-sequence accuracy.

## Running and reproduction

The environment retains v1's Python 3.13 / torch 2.11.0+cu128 / NumPy 2.5.4; full dependencies are in the sibling `permworld_pilot/requirements-lock.txt`. The original v1 data and step0/pretrain/256-adapt checkpoints are required. Large checkpoints remain local, verified by manifest hashes, and are not committed to Git. Training imports the frozen v1 model/task implementation directly without copying or changing mathematical definitions.

Run from the repository root:

```powershell
.venv/Scripts/python.exe -u experiments/pilot_v2_diagnostics/run_all.py
.venv/Scripts/python.exe experiments/pilot_v2_diagnostics/audit_final.py
.venv/Scripts/python.exe experiments/pilot_v2_diagnostics/summarize.py
.venv/Scripts/python.exe experiments/pilot_v2_diagnostics/audit_final.py
```

`run_all.py` runs inventory→smoke→freeze→train→select→test sequentially and skips stages with completion markers. Fully completed candidates can be reused; incomplete candidates restart from their original starting point, without claiming automatic recovery at any intermediate step. The smoke test checks exact parameter equality between 8 continuous source-continuation steps and 4 steps followed by save/load and another 4 steps. It also verifies that evaluation does not consume dropout RNG.

The first `audit_final.py` checks training and test artifacts. Next, `summarize.py` generates the report, summary CSV files, and curves while referencing the audit result. The final audit updates `artifact_manifest.json` to include those generated artifacts. Neither summarization nor the final audit runs model forwards.

Use a new directory for an independent rerun, preserving these results:

```powershell
.venv/Scripts/python.exe -u experiments/pilot_v2_diagnostics/run_all.py --out experiments/pilot_v2_diagnostics/replication
.venv/Scripts/python.exe experiments/pilot_v2_diagnostics/audit_final.py --out experiments/pilot_v2_diagnostics/replication
.venv/Scripts/python.exe experiments/pilot_v2_diagnostics/summarize.py --out experiments/pilot_v2_diagnostics/replication
.venv/Scripts/python.exe experiments/pilot_v2_diagnostics/audit_final.py --out experiments/pilot_v2_diagnostics/replication
```

To rebuild from v1, first reproduce v1 in a fresh checkout following `../permworld_pilot/README.md`. Numerical checks record the device and environment; bitwise reproduction across CUDA/hardware environments is not guaranteed. Later experiments must not use this round's new test set to select protocols.
