# Prespecified model-change replication of the fixed-points confirmation

This experiment reports eight new paired units separately from the original eight-unit fixed-points confirmation. Its sole primary comparison and transfer protocol are inherited unchanged. The model is selected by the development-only gate below, before any formal transfer training. Results are never pooled across the two studies, and neither the original positive contrast nor earlier unsuccessful pilot confirmations are rewritten.

## Question, sources and mathematical scope

The target is `fixed_points`. Every pretrained route has the common slots `descents`, `successions`, and `left_to_right_maxima`, followed by `peaks` (P), `deficiencies` (D), `anti_fixed_points` (T), or `right_to_left_maxima` (U). A is the shared scratch-training baseline within a unit and sample size. This is the original confirmation's task/slot order, not a claim to reproduce the older pilot B groups.

The original definitions, independent-oracle checks and relation certificates are reused by hash. `FP + E + D = n`, `E(p)=D(inverse(p))`, and therefore `FP(p)=n-D(p)-D(inverse(p))`. D on the original input alone is insufficient. Anti-fixed points connect to fixed points through complement/reversal and can also share positional comparisons. U has a reversal relation to the common left-to-right-maxima task. The original same-input and transformed-query combination witnesses remain applicable because no task definition changes. Absence of a registered relation is not mathematical unrelatedness. The models receive no paired-transform examples, formula supervision, or local-position labels.

## Development gate and model decision

Before any optimization, commit the gate protocol, analysis plan, source hashes and isolated gate-data binding. Use the reference four-layer width-256 GRU with the original confirmation's 188-token vocabulary, one-line input, context 128, tied embedding/output weights, embedding/explicit interlayer dropout 0.1, and FP32 weights/evaluation with BF16 CUDA training. Its four one-layer recurrent modules retain the reference's explicit dropout design. The Transformer attention-head and feed-forward fields do not apply to the GRU.

The separate gate has 8192 training and 1024 development permutations at n=10, shared across independent single-task `fixed_points` and `deficiencies` runs. Both start from the same gate step-zero initialization with fresh optimizers and shared sampling/dropout streams. Each receives 4800 updates at batch 64, the original source learning-rate schedule and evaluation steps. The numerical choices and seeds are in `plan.py`. There is no gate test pool, hyperparameter search or continuation from a trained gate model.

Both **fixed-update-4800** complete-answer-plus-EOS development accuracies must be at least 0.90. If both pass, use the GRU. If either fails, use the prespecified eight-layer width-256 Transformer, retaining the original eight heads, feed-forward multiplier four, vocabulary, context, dropout and tied outputs. No tuning follows either outcome. Preserve the gate curves and outcome, exclude its inputs from every formal pool, and discard its trained weights. A fallback is a depth-change replication within the Transformer family, not evidence from a second architecture family. The gate is not a confirmatory repeated unit.

The separately committed formal engineering smoke uses its own sampling/dropout seeds recorded in `plan.py` and a discarded model. Its 112 optimizer updates are engineering checks, not additions to the formal budgets or independent repeated units.

## Unchanged formal protocol

Use one-line input at n=10 and the inherited 188-token tokenizer and scalar answer format. AdamW has betas (0.9,0.999), epsilon 1e-8, weight decay 0.01 and gradient clipping at 1; TF32 is disabled. Loss is the inherited mean of answer-token CE and teacher-forced EOS CE. All parameters are trainable.

There are exactly eight new units, 32 source models, 360 adaptation learning-rate candidates and 120 selected primary results. Every source model receives 4800 updates at batch 64, exactly 16 samples per task per update. All four routes in a unit share step-zero weights, source indices and task-slot assignments. The source LR warms from 3e-6 to 3e-4 through update 100, follows the original cosine decay to 3e-5 at update 2400, and stays there through update 4800. The endpoint is the sole transfer source; no target-driven source checkpoint selection is allowed. Record source training/development accuracy, loss, baselines and exposure at the original scheduled evaluations.

For each route and target size 256,512,1024, start independently from its source endpoint or the unit's scratch initialization. Strictly load the whole model, restore only the fixed_points tied embedding/output row to the same unit's step-zero row, and create a fresh optimizer. Do not continue from another sample size. Each of the inherited LR candidates, in order 3e-4,1e-4,3e-5, receives 800 updates at batch 64, warmup 100 and the original cosine schedule to one tenth of its initial LR. No additional LR search is introduced.

Evaluate development data at step zero and every 100 updates. Step zero is diagnostic and ineligible. Select by greatest complete-answer-plus-EOS accuracy, then lowest answer CE, earliest positive update and declared LR order. Retain the selected LR candidate's update-800 endpoint as a separate diagnostic. Full unconstrained greedy generation must match the answer and EOS. Preserve the historical first-answer-token metric separately without merging evaluation definitions.

## Fresh inputs and supervision

The eight complete seed tuples in `plan.py` precede all training. Each formal unit has 8192 source-training, 1024 source-development, 1024 maximum target-training, 1024 target-development and 2048 deferred test permutations. The target training order supplies nested prefixes 256,512,1024. All routes share corresponding inputs and adaptation random streams. The 1024 labeled target-development examples are additional supervision; a 256-training-example run does not use only 256 target labels overall.

Before gate generation, the capacity preflight must reserve the gate's 9216 objects and all 106496 formal objects, including deferred tests, under the full inverse/reverse/complement orbit rule. Include every historical DMLR model-input pool, explicitly including the original fixed-points experiment, historical engineering pools and the canonical platform pools. Exposure-count archives are accounting records and must be classified rather than treated as permutations. Bind the historical inventory, exclusion list, pool generation order and hashes. No isolation rule is relaxed. All gate/formal roles and units are disjoint by full orbit; failed-gate inputs remain excluded under the fallback.

Uniform permutation proposals followed by sequential global orbit rejection sample a finite population. They do not make the units exactly IID or establish statistical power. Stage 2 reused canonical calibration data and adds no occupied input orbit; its calibration exception does not apply here.

Generate formal test inputs only after all 120 new development selections are recorded and bound to this experiment's protocol, new initializations, data and checkpoints. The test gate must reject old selections even if their R1-R8 labels match. Evaluate all selected and selected-LR endpoint checkpoints on the common new within-unit test pool, preserving 240 checkpoint evaluations and per-example outputs. Test results never alter a source, model, seed, budget, LR, checkpoint or target sample size.

## Analysis, accounting and stopping

The score is `S=0.25*A256+0.5*A512+0.25*A1024`. The sole primary statistic is the eight within-unit `S_D-S_U` differences. The unchanged statistical plan reports every unit, mean, sample SD, range and a two-sided 95% paired t interval with df=7. An interval containing zero means uncertainty, not proof of no effect. Secondary comparisons, all scratch-relative effects, pointwise differences, endpoint diagnostics and source learning remain fully reported. None replaces the primary comparison. Source-learning associations do not establish mediation or use of a registered formula.

Record actual sequence/input and supervised tokens, sample exposure, distinct inputs, updates and elapsed time. Repeated exposures, shared baselines, LR candidates and sample sizes are not independent units. Concurrent run timers are not additive wall time. Keep the original experiment and this model-change replication separate in all tables and interpretations.

Preserve weak source learning, negative transfer, gate failure, uncertainty and any incomplete execution. There is no added repeat, second target, extra source mixture, extended budget or model tuning. Engineering reruns retain their seeds and are documented separately. A failure requiring a frozen-record change stops this sequence. New output stays in this directory, checkpoints stay local, and only the requested branch is eventually pushed by a normal push.
