# Fixed-points deficiencies confirmation: frozen scientific protocol

This is a new, fixed-scope confirmation case motivated by pilot v5 and task-count v1/v2. It does not revise their failed confirmations, negative transfer, or interpretation boundaries. The complete user specification arrived after the preliminary audit: `REQUEST_SCOPE.json`, `PREFLIGHT_REPORT.md`, and `preflight_audit/` remain immutable records of that earlier, incomplete-request stage. This document and `design.py` complete the specification. Repository text and generated reports are English. Delivery is local; no automatic upload or publication is authorized.

## Question and compositions

For `fixed_points`, does replacing a prespecified fourth source with `deficiencies` improve transfer relative to `right_to_left_maxima`, holding the background, source budget, and adaptation procedure fixed?

| Route | Slots 0, 1, 2 | Slot 3 | Role |
|---|---|---|---|
| P | descents, successions, left_to_right_maxima | peaks | Original-background reference |
| D | same | deficiencies | Primary candidate |
| T | same | anti_fixed_points | Prespecified transformed-input-related comparator |
| U | same | right_to_left_maxima | Prespecified computational-structure comparator |
| A | none | none | Shared random-initialization target baseline |

P and D have the task **sets** of the two v5 C groups. Their slot order changes from `[successions, descents, candidate, left_to_right_maxima]` to the order above. T is a new composition, not a direct rerun of an old B group. U is fixed in advance and is not described as mathematically unrelated. Its difficulty, answer distribution, and redundancy are not assumed to match D. All four conditions have exactly 16 examples per source per update and share pool indices, slot assignments, and initialization within each unit.

## Mathematics and epistemic distinctions

The actual legacy task functions, independent oracles, historical certificates, and new combination witnesses are audited in `PREFLIGHT_REPORT.md` and `preflight_audit/`. `RELATION_TABLE.csv` separates proven answer relations, possible shared subcomputations, and hypotheses about trained models. The unchanged v5 exhaustive certificate is reused by hash; only new U and combination checks were run.

In one-based notation, `FP + E + D = n` and `E(p) = D(inverse(p))`, hence `FP(p) = n - D(p) - D(inverse(p))`. D on the original input alone is insufficient. Anti-fixed points satisfy `FP(p) = AF(complement(p)) = AF(reverse(p))`; both can also be computed by positional comparisons. U satisfies `U(p) = L(reverse(p))`, so it has a registered relation to the shared L source. No absence of a registry entry establishes absence of a relation.

The full four scalar answers at the same input do not determine FP in any condition, as demonstrated by n=10 witnesses. With answers at every inverse/reverse/complement transform, D and T determine FP, while audited P and U witnesses still disagree on FP. These are answer-only information statements, not statements that the raw permutation lacks information. Training has no explicit paired-transform queries, formula supervision, or local-position labels. Source learning and downstream gains do not prove use of a formula or a transplantable lemma module.

## Frozen training and selection

The authoritative numeric configuration is `design.py`, copied from the frozen v5 configuration: one-line input at n=10, vocabulary 188, four Transformer layers, width 256, eight attention heads, MLP ratio 4, dropout 0.1, tied input/output embeddings, and maximum context 128. Precision is FP32 weights/evaluation and BF16 CUDA training, with TF32 disabled. AdamW uses betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01, and gradient clipping at 1. The inherited loss averages answer-token CE and teacher-forced EOS CE. The actual prompt includes comma-separated permutation tokens.

Each source model receives 4,800 updates at batch 64. Source LR warms from 3e-6 to 3e-4 over updates 1–100, decays by the inherited cosine schedule to 3e-5 at update 2,400, and stays there through 4,800. Only the 4,800-update endpoint supplies transfer weights. Source train/development evaluations occur at 0, 600, 1,200, 2,400, 3,600, and 4,800. Save all source learning records, including common-source differences between compositions.

Every route and target subset starts independently from its parent endpoint (or step 0 for A). Load the complete model, restore only the target's tied embedding/output row from the same unit's step-0 model, and create a fresh optimizer. Each learning-rate candidate gets 800 updates, batch 64, warmup 100, then the inherited cosine decay to 0.1 times its initial LR. Candidates, in order, are 3e-4, 1e-4, and 3e-5. All parameters are trainable. No continuation from a smaller target subset, frozen prefix, layer search, or cross-model state exchange is allowed.

Evaluate development data at step 0 and every 100 updates. Step 0 is diagnostic and ineligible for selection. Select by (1) highest complete generated answer-plus-EOS exact-match accuracy, (2) lowest answer-token CE, (3) earliest positive update, and (4) declared LR order. There is no extra hyperparameter search. Preserve the selected candidate's update-800 endpoint as a diagnostic. Report old first-answer-token accuracy and teacher-forced EOS separately; do not pool them with full-answer accuracy. A scalar answer has two expected generated tokens, answer and EOS; unconstrained greedy decoding must match both.

## Eight paired units and isolated data

`design.py` records every initialization, pool, subset-order, sampler, dropout, and future-test seed. There are exactly eight new units, 32 source models, 360 target learning-rate candidate runs, and 120 selected primary target results. Shared conditions, checkpoints, sample sizes, learning rates, and A baselines do not create additional independent repeats. Engineering reruns keep the same seeds and are logged separately.

Each unit has 8,192 distinct source training permutations, 1,024 source development permutations, a 1,024-permutation maximum target training pool, 1,024 target development permutations, and a future test pool of 2,048 permutations. An independently seeded order of the target maximum pool gives nested prefixes of 256, 512, and 1,024. **The 1,024 labeled target development examples are additional target supervision**, shared across the LR/checkpoint choices. A 256-example adaptation is not a procedure with only 256 target labels in total.

Before generation, require capacity under full inverse/reverse/complement orbit exclusion. Exclude all historical DMLR inputs, including task-count v2, relation-readout data, and historical engineering pools. Also exclude every previously accepted new pool across roles and units. Store input hashes, generator order, exclusion inventory, proposal counts, and replay verification. The conservative capacity check found 161,553 occupied historical n=10 orbits, leaving at least 292,047; this exceeds the 106,496 requested new orbits including tests. No exclusion rule is relaxed.

The uniform-permutation rejection procedure is a finite-population, globally disjoint design. Later pools depend on earlier accepted orbits; independent seeds do not establish exactly IID units. Orbit-size-dependent rejection can affect the distribution relative to unrestricted uniform permutations. No cross-unit pooling of sample-level observations is used for inferential replication.

Test objects are generated **only after all 120 development choices are frozen** and their bound checkpoints exist. Then evaluate all selected checkpoints and the update-800 endpoint of the selected LR candidate on the same new test inputs within each unit. The planned test inventory is 240 checkpoint evaluations and 491,520 per-example prediction records. Never use test results to change tasks, seeds, training budgets, checkpoints, LR, or sample sizes.

## Analysis and stopping rules

For each unit and route, form `S = 0.25*A256 + 0.5*A512 + 0.25*A1024`, the normalized trapezoid area on log2 sample size. The sole primary comparison is the eight within-unit differences `S_D - S_U`. Report all eight values, mean, sample SD, minimum, maximum, and the paired two-sided 95% t interval with df=7. The interval relies on approximate distributional assumptions with only eight units and a finite population; it is not a power guarantee. An interval containing zero is uncertainty, not proof of no benefit.

Prespecified secondary area comparisons are D−P, U−P, D−T, and P/D/T/U−A. Report all ten condition pairs at each target sample size as secondary descriptive results. Compute paired differences before aggregation. `stats_plan.json` is authoritative. Do not replace an inconclusive primary contrast with a favorable secondary finding.

Source train/development accuracy, loss, train-determined majority baselines, per-task exposure, unique inputs, sequence tokens, supervised tokens, updates, and actual wall time are descriptive evidence. Source learning differences are not causal mediation results. If D and U both improve over P but D does not beat U, the P improvement cannot be attributed specifically to deficiencies. D beating T ranks these particular sources, not all subcomputation versus input-transformation relationships. U does not isolate the absolute effect of adding a source from removing peaks.

Retain weak source learning, chance-like A, negative transfer, failed predictions, and uncertain results. Do not add a unit, target, source composition, budget, architecture, prefix intervention, or learning-rate candidate. Resource failures must be explicit; they never justify silently changing the protocol. Deliver locally and stop after the frozen scope.

## Execution and provenance

`FROZEN_PROTOCOL.json` binds this document, the configuration, statistics plan, code, data manifest, initial-model manifest, and successful smoke record before formal training. Exact commands, environment and timing go in `EXECUTION_LOG.md`; `ENGINEERING_LOG.md` records any failures and repairs. Historical inputs, frozen configurations, numerical results, reports, and all TMLR files remain untouched. Only the new experiment directory receives outputs.
