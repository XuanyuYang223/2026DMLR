# Task-count v2: fixed compositions crossed with token budget

This prespecified follow-up is motivated by task_count_v1 exploration. The primary
target is `lis_length`; the auxiliary target is `fixed_points`. V1 remains a
separate exploratory dataset. No v2 transfer outcome may select tasks, budgets,
hyperparameters, seeds or contrasts. Work is isolated on
`experiment/permworld-task-count-v2` in this directory. Earlier results are kept.

## Fixed 2 by 2 comparisons

For each unchanged v1 chain, retain its exact k=2 and k=16 prefixes and cross
them with 4,000,000 and 32,000,000 nonpadding full-sequence training tokens.
`design/task_sets.csv` lists every task instance and parameter. All four original
held-out targets and their complete same-input aliases remain excluded. All
existing mathematical relations and output bounds are hash-reused from v1.
There are no new task variants, architecture changes or relation-proof searches.
The pool still contains 24 instances from 22 templates. Each set contains k
distinct templates; the input encoding has one variant. The chains have different
memberships at both k values, not merely different orderings.

Each budget begins separately from the same paired random initialization.
The 32M run is not resumed from a 4M checkpoint. Both budgets use learning rate
3e-4 with warmup over 1/24 of their own total token budget and cosine decay to
3e-5 according to consumed tokens. Therefore their absolute-token learning-rate
trajectories differ. Budget and this normalized-progress schedule are tested
together; the experiment does not isolate more updates from LR trajectory.

Sampling is uniform over task instances by examples using the same balanced
slot procedure as v1; unequal sequence lengths imply unequal token shares.
Both budgets reuse the same fixed source input pool. More tokens are repeated
exposure, not an eightfold increase in independent input objects. The 32M budget
is approximately eightfold total tokens, not exact matching of per-task sample
or token exposure between k values. Save actual per-task samples, sequence and
supervised tokens, unique input indices and repeat exposure, as well as tails.
Stop with a whole-example prefix; never truncate an answer to exhaust a budget.
Use the fixed source endpoint, without transfer-driven checkpoint selection.

## New paired repeat units and isolated data

Lengths are 10 and 12. Per repeat and length, use 4,096 source-training inputs,
256 source-development inputs, 128 target-training inputs, 256 target-development
inputs and 512 final-test inputs. Thus each target has 256 distinct adaptation
examples. Source sets, budgets and targets share the applicable pools within a
repeat. Target and source evaluation use the unchanged platform full-answer
generation protocol. A shared object pool does not make labels across tasks
independent observations.

| Repeat | Initialization | Data | Test | Pretrain sampling | Adapt sampling | Pretrain dropout | Adapt dropout |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4103 | 2026110101 | 2026110151 | 2026110181 | 2026110191 | 2026110281 | 2026110291 |
| 2 | 5207 | 2026110102 | 2026110152 | 2026110182 | 2026110192 | 2026110282 | 2026110292 |
| 3 | 6301 | 2026110103 | 2026110153 | 2026110183 | 2026110193 | 2026110283 | 2026110293 |

Check a conservative full inverse/reverse/complement orbit-capacity bound before
generation, including history and all planned test objects. Isolate complete
orbits across roles, repeats, prior pilots/platform inputs and task_count_v1.
Never relax this rule to fill a pool. Only train/development objects are generated
before training; create fresh independent tests after all development selections
are frozen. Uniform proposals followed by orbit rejection are not independent
uniform orbit-class sampling. V1 and v2 inputs remain distinct.

## Unchanged model, loss and adaptation

Use the v1 platform model and vocabulary: 203 tokens, four Transformer layers,
width 256, eight heads, MLP ratio 4, dropout 0.1, tied embeddings/output head,
context 128; `one_line_v1` and `permworld_sequence_v1`. Loss averages answer-token
plus EOS cross-entropy within each sample, then averages samples. Accuracy is
exact greedy generation of the complete answer plus EOS, without gold tokens.
Parameterized I3/D3 answers above 99 retain their complete multi-token encoding.
The generation limit is eight answer tokens, already checked in the v1 design.

Use AdamW with batch 64, LR 3e-4, minimum LR 3e-5, weight decay .01, betas
(.9,.999), epsilon 1e-8 and gradient clip 1, as v1. Evaluation batch is 128.
Every target fit receives 800 updates, 100-update warmup and cosine decay,
and development evaluation every 100 updates. There is one LR candidate and
eight eligible checkpoints. Select highest complete-answer development accuracy,
then lowest answer CE, then earliest update; step zero is not eligible. Also
retain the fixed endpoint as a secondary evaluation. Test outcomes never change
selection. Reinitialize all four original held-out task-token rows, including
their tied head rows, to the shared per-repeat initial values. Load all remaining
pretrained parameters exactly. Reset AdamW state for every adaptation.

Two chains by two k values by two budgets by three repeats give 24 pretrained
models. Each adapts to two targets, giving 48 fits. Six scratch baselines, one per
target and repeat, are trained once and shared across all eight conditions.
There are exactly 54 target fits. Source start/midpoint/endpoint metrics use the
same fixed 256-object training probe per length and complete development pool as
v1. Majority answers are learned from each full source training pool, or each
target's 128 labels per length. Report per-length accuracy and answer CE.

## Primary analysis and stopping rule

Within every chain, target and repeat, calculate in percentage points:

1. D4 = accuracy(k16, 4M) minus accuracy(k2, 4M).
2. D32 = accuracy(k16, 32M) minus accuracy(k2, 32M).
3. Interaction = D32 minus D4.

LIS development-selected interaction is primary; fixed points is auxiliary.
Show every repeat and summarize the three paired differences using mean, sample
SD and range. Do not assess an effect from two marginal SDs. Source tasks, chains,
shared models, reused baselines and targets are not extra independent repeats.
Report all condition-minus-baseline gains; the same baseline cancels in the
primary contrast. Endpoint and per-length comparisons are secondary.

Emphasize LDS and length-three increasing/decreasing subsequence source learning,
exposure and training/development gaps. Such evidence supports associations only,
not a proof of interference, formula use or another learned mechanism. Increasing
k changes composition and available mathematical relations. The experiment tests
whether the count contrast along these fixed chains changes under a larger
budget and its prescribed LR schedule, not a universal count law.

A small engineering smoke and timing check precedes the formal freeze. Retain
flat, negative, unstable and undertrained results. The declared 24+54 runs and
unified independent tests end this round. No extra budget, learning-rate search,
task replacement or follow-up background variants may be added from outcomes.
Deliver raw CSV, interaction figures, source/exposure records, runtime, frozen
configuration, reproduction commands and a Chinese report as explicitly
requested. Code and other documentation remain English. Push shareable artifacts
to the experiment branch; keep checkpoint weights local.
