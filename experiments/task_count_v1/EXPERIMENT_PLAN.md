# First fixed-token task-count experiment

This is a finite exploratory comparison of source-task count under a fixed total
pretraining-token budget, using the existing PermWorld v0.1 interface. The four
targets are fixed in advance: `cycle_count`, `lis_length`, `components`, and
`fixed_points`. No transfer results from this experiment are used to select
sources, hyperparameters, budgets, targets, or reporting contrasts.

The previous pilots and platform release remain unchanged. The study lives on
`experiment/permworld-task-count-v1`, with artifacts in this directory. Repository
documentation and reports are in English; checkpoints remain local.

## Task instances and compositions

`design/configuration.json` and `design/task_sets.csv` contain two fixed nested
chains, each using k=2,4,8,16 source instances. At every k the two sets have
different membership, not just a different order. Their union contains 24
instances from 22 templates; each individual prefix contains distinct templates.
Parameter values and input encodings are not counted as additional templates.

The chains match broad family counts at each k:

| k | Cycle decomposition | Global subsequences | Permutation blocks | Legacy statistics |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 1 | 0 | 0 |
| 4 | 1 | 1 | 1 | 1 |
| 8 | 2 | 2 | 2 | 2 |
| 16 | 4 | 3 | 4 | 5 |

All four targets and their complete same-input answer aliases are excluded from
every source condition. Constant and duplicate instances are excluded. Exact
nonconstancy and inequivalence witnesses are saved at both experimental lengths.
These witnesses do not prove mathematical independence or equal difficulty.
The source pool is restricted to scalar tasks, including parameterized I_3/D_3
counts with multi-token answers.

Input-transform and joint relations remain part of the design. In particular,
alpha at k=16 jointly determines cycle_count from even/odd cycle counts and
fixed_points from either even+odd−nontrivial or n−exceedances−deficiencies. Beta
contains a transformed-input fixed-point relation through anti_fixed_points.
LDS and global_descents also retain transformed-input links to LIS and components.
`design/target_relationships.csv` records the required queries; unlisted relations
are not claimed absent. These legal mathematical relations are not automatically
data leakage.

## Data and repeated units

The input lengths are n=10 and n=12. Each repeated unit has the following numbers
of inputs per length:

| Role | Count per length |
| --- | ---: |
| Source training | 4096 |
| Source development | 256 |
| Target adaptation | 128 |
| Target development | 256 |
| Final test | 512 |

Thus each target adaptation has exactly 256 distinct training inputs. All four
targets share the target pools within a repeated unit; all task-count/chain
conditions share the corresponding source and target pools. The three model
initialization seeds are 1103,2207,3301, with data seeds 2026102101–2026102103 and
test seeds 2026102151–2026102153. Separate sampling/dropout seeds and complete RNG
rules are saved in the frozen training configuration.

Before generation, a conservative capacity bound, floor(n!/8) minus occupied
historical orbits, reserves enough room for all roles and repeats, including the
future tests. Full inverse/reverse/complement orbits are isolated across roles,
repeats, and previous pilot/platform inputs. The rules are never weakened on a
capacity failure. Uniform permutation proposals with sequential orbit rejection
are not claimed to be independent draws from uniform orbit classes.

Only training/development pools are generated initially. Final test objects are
generated after all training is complete and all 108 development selections are
frozen. Test labels and model outputs cannot influence those selections.

## Model, loss, and fixed budgets

The unchanged platform model has a 203-token vocabulary, four Transformer layers,
width 256, eight heads, MLP ratio 4, dropout 0.1, tied embeddings, and context 128.
The input/output protocols are `one_line_v1` and `permworld_sequence_v1`.
Loss first averages answer-token and EOS cross-entropy within each example, then
averages over examples. Accuracy requires the complete autoregressively generated
answer plus EOS, with no gold answer tokens supplied to generation.

There are 24 pretrained models: two chains × four k values × three repeats. Each
has a maximum budget of 4,000,000 nonpadding full-sequence tokens, counting the
prompt, complete answer, and EOS. Whole examples are consumed; the last batch may
be partial and the next example is not truncated. Actual tokens and the unused
tail are reported separately. Supervised, answer-only, and padded token counts
are also recorded. Learning rate follows token progress, with a warmup over 1/24
of the budget to 3e-4 and cosine decay to 3e-5. Tasks are uniformly sampled by
example using balanced task-slot permutations, not uniformly by answer tokens.
Input/length sampling uses a separate RNG stream from task-slot sampling.

Source learning is recorded at initialization, the first update crossing half
the token budget, and the fixed endpoint. Development sets are evaluated fully;
source-training evaluation uses a fixed diagnostic subset, clearly labeled as
such, while majority answers are determined from the full source-training pool.
Pretraining checkpoints are never selected using target transfer outcomes.

Each pretrained endpoint is adapted separately to all four targets, giving 96
adaptations. Twelve random-initialization baselines are trained once (four targets
× three repeats) and shared across k and chain comparisons: 108 target runs total.
Every run has 800 updates, batch size 64, learning rate 3e-4 with 100-update warmup
and cosine decay to 3e-5, and no learning-rate search. Development evaluation is
performed every 100 updates. Checkpoints are selected by highest development
complete-answer accuracy, then lowest answer CE, then earliest update; step zero
is excluded. Fixed endpoints are retained and evaluated as auxiliary results.

All conditions start adaptation with a fresh AdamW state. All four target-task
embedding rows are restored to the same per-repeat initialization, including
their tied output-head rows. Remaining parameters are loaded from the relevant
pretraining endpoint or shared random initialization. Weight hashes, optimizer
reset checks, and paired sampling-stream hashes are saved.

## Analysis and stopping rule

The primary display is development-selected test accuracy versus k for each
target and chain. The corresponding paired difference from the shared baseline
is computed within each repeat. Each figure shows individual repeats and the
mean with descriptive sample SD across the three paired units. Endpoint results,
per-length accuracy, training-derived majority baselines, source learning,
exposure, token costs, and wall-clock timing are auxiliary outputs.

Two chains, four targets, multiple k values, and reused baselines do not create
additional independent repeats. Do not pool these conditions as independent
observations, infer reliability from separate marginal SDs, select a preferred
chain from test performance, or claim a scaling law. Increasing k changes the
composition, available mathematical relations, answer-length distribution, and
per-task exposure. These effects are not separated causally by this experiment.

Negative transfer, no benefit, seed disagreement, and incomplete source/target
learning are retained. A small engineering smoke test and timing check precede
the formal protocol freeze. Completing the declared 24+108 runs and unified test
evaluation ends this study; no automatic budget extension or task replacement
is permitted.
