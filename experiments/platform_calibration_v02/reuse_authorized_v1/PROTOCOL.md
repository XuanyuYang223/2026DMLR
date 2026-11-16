# Remaining-instance calibration on the canonical platform pools

## Question and scope

Complete the fixed-budget, development-only single-task calibration for every materialized unary non-alias instance marked `not_run_in_v0.1` in the original learnability index. Selection is the exact intersection with the original dataset manifest, not a selection by model performance. This gives 50 requested instances; all remain visible in the final table.

The author explicitly permits read-only reuse of the platform_v01 pools. The original units pair initialization seed 617 with data seed 2026102001 and sampling/dropout seed 1617, and initialization seed 829 with data seed 2026102002 and sampling/dropout seed 1829. Each has 1,024 training and 256 development inputs at each of n=8,10,12. These are the same data units as the original six-task calibration, not new independent units. Different tasks intentionally share inputs; their train/development split is unchanged. No new input pool is generated and no reserved test pool is opened or forwarded through a model.

POOL_REUSE_AUDIT.json must pass before any training. Its hashes bind all twelve train/development NPZs to both the original frozen calibration and canonical manifest. Archived run/checkpoint/exposure evidence verifies that the completed original calibration used this binding. No incompatible or unbound input is substituted.

## Immutable training and evaluation settings

The authoritative numeric configuration is copied from platform_v01/calibration/FROZEN_CALIBRATION.json: four-layer width-256 eight-head causal Transformer, vocabulary 203, tied embeddings, context 128, dropout 0.1; 2,400 updates, batch 64, learning rate 3e-4, warmup 100, cosine decay to 3e-5, AdamW with weight decay 0.01, betas 0.9/0.999, epsilon 1e-8, and clipping at 1. Use FP32 weights/evaluation, BF16 CUDA training, TF32 disabled. There is one learning rate and no search or training extension.

Loss is the mean over samples of mean cross-entropy over every complete gold answer token and EOS. Length scheduling, index sampling and development evaluations every 200 updates inherit the original functions. Each task starts independently from the paired seed's original initialization, checked against the archived initial-state hash, with a fresh optimizer. Each length receives 800 updates. Tasks are not continued from another task's model.

The fixed update-2400 endpoint is primary. The development-best checkpoint remains a descriptive diagnostic selected by highest exact accuracy, lowest answer CE, earliest positive update. Exact accuracy uses unconstrained greedy generation from the prompt, including the EOS token in the required full answer. The generation budget stays at the original `max_new_tokens=8`. No extra decoding pass, longer decoding budget or teacher-forced answer-token substitute is used to improve the headline accuracy.

## Reader eligibility and output-budget interpretation

The pretraining CPU encoding scan covers only the required canonical train/development samples. If any required sample for an instance exceeds the 128-token context, record the instance as reader-rejected across both units and all lengths. Do not truncate, filter individual examples, resample, or train on a changed pool. The scan identifies `to_reduced_word` as the one such instance; its offending samples are recorded. Thus 49 instances have two planned runs each, while the rejected instance has no invented accuracy.

The original scalar-calibration loader asserted that all answers fit eight generation tokens. The new loader expands eligibility to every context-valid requested task while retaining all gold tokens for training. This removes that scalar-task eligibility assertion only; the numeric training configuration, loss, eight-token generation budget and exact-match definition remain unchanged. The full answer remains the scoring reference, even when its length exceeds the generation budget.

For each instance, unit, role and length, publish the fraction of gold answers that can fit the generation budget. This is a structural upper bound on full-answer accuracy. Scores constrained by this limit are labeled generation-capped; zero full-answer accuracy for an impossible-length answer is not evidence that its mathematical computation was not learned. Loss curves still supervise full answers. No instance is removed for having such a limit.

## Analysis, pairing and stopping

ANALYSIS_PLAN.md specifies the endpoint table, updated index and comparisons. The units are the two inherited data/initialization pairs. Tasks, lengths, checkpoints and shared pools do not increase the independent repeat count. Results are directly comparable in input split, initialization pairing, training settings and decoding budget to v0.1, but different task outputs can impose different difficulty and generation ceilings. Do not interpret this as an independent replication of the original calibration.

Complete the eligible fixed grid and retain negative gains, constants, weak fits and structurally capped scores. Report failures without changing seeds, hyperparameters or scope. No test model forward, transfer mixture, architecture experiment, or family ranking is performed in this stage. The earlier capacity audit remains unchanged; input reuse is the explicit author-approved scope exception for calibration only.

The protocol, analysis plan and bound implementation are committed before any optimizer update, including engineering smoke updates. Engineering checks are separate from formal runs. The runner refuses uncommitted or modified bound files, mismatched inputs, mismatched original initializations or overlapping worker ownership. Outputs use this new subdirectory only.
