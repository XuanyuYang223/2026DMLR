# Historical origin of the fixed-points source-replacement hypothesis

This note records the historical motivation for the independently frozen
`fixed_points_deficiencies_v1` study. It was prepared on 2026-11-10 from the
archived reports and tables below, without consulting this study's interim
training outcomes. No historical files, frozen code, configurations, results or
checkpoints were changed. Historical numbers are contextual evidence, not extra
repeats of the new eight-unit confirmation.

## V5: retain the unsuccessful primary confirmation

V5's original fixed-points primary quantity was the background interaction
`(B-C)_deficiencies - (B-C)_original`, with a positive direction prespecified
from v4. B supplied `anti_fixed_points`, C supplied `successions`, and the
background replaced `peaks` with `deficiencies`. Among five new units, the mean
interaction was **-3.0176 pp**, sample SD **15.7473 pp**, with a 95% paired-t
interval **[-22.5704, +16.5352] pp**. Only 2/5 signs matched the prespecified
positive direction. The original confirmation did not reproduce its predicted
mean direction; the interval does not establish absence of a background effect.
The two-contrast simultaneous interval also included zero. These conclusions
remain unchanged. [V5 report](../pilot_v5_confirmation/PILOT_V5_REPORT.md),
[original confirmation summary](../pilot_v5_confirmation/artifacts/confirmation_summary.csv).

The other V5 primary contrast, longest_decreasing_run with a recoils background,
had mean -13.1543 pp, sample SD 12.6746 pp, and 95% paired-t interval
[-28.8919, +2.5833] pp. Its mean retained the exploratory direction, but neither
the individual nor simultaneous interval established a stable effect. It is
not added as a target to this new experiment.

The fixed-points **auxiliary** changes in B-A and C-A were positive in all five
V5 units: means +22.6172 and +25.6348 pp, respectively. This is compatible with
the failed interaction prediction because C improved more than B on average.
The new experiment's historical P/D mapping comes from these old C groups:

| Historical initialization seed | Old C deficiencies minus old C original (pp) |
| ---: | ---: |
| 101 | +12.2559 |
| 211 | +26.0742 |
| 307 | +28.8086 |
| 401 | +32.7637 |
| 503 | +28.2715 |

Their mean is +25.6348 pp, sample SD 7.8587 pp. These values were checked by
subtracting the two archived `C_minus_A` rows within each seed; shared A cancels.
They are an exploratory motivation for a **new** comparison, not a replacement
primary outcome that makes V5 successful. They do not compare deficiencies
with the new U source. [Archived within-seed differences](../pilot_v5_confirmation/artifacts/paired_condition_differences.csv).

## Exact correspondence of task sets, slots and evaluation

| New route | Ordered source tasks, slots 0 to 3 | Historical correspondence |
| --- | --- | --- |
| P | descents, successions, left_to_right_maxima, peaks | Same set as `fixed_points/g0_v3/C` |
| D | descents, successions, left_to_right_maxima, deficiencies | Same set as `fixed_points/g2_deficiencies/C` |
| T | descents, successions, left_to_right_maxima, anti_fixed_points | New composition; neither old B set |
| U | descents, successions, left_to_right_maxima, right_to_left_maxima | New prespecified comparison composition |

The old C slot order was `successions, descents, peaks-or-deficiencies,
left_to_right_maxima`. New slots are old indices `[1, 0, 3, 2]`. The **sets** match
for P/D, but the new shared-stream slot assignment is not an exact replay of old
training. Old B contained `anti_fixed_points, descents, peaks-or-deficiencies,
left_to_right_maxima`; it omitted successions. Calling T a direct replication of
old B would therefore change the claimed comparison.
[Original slot table](../pilot_v5_confirmation/artifacts/compositions.csv),
[new preflight mapping](preflight_audit/COMPOSITIONS.csv).

V5 used n=10, the 188-token legacy vocabulary, four layers, width 256, eight
heads, 4800 source updates and three adaptation learning rates. The new study
inherits this architecture and training configuration, but additionally
requires complete greedy answer-plus-EOS exact match for selection and primary
evaluation. V5 reported first-answer-token accuracy, with EOS diagnostics
separate. A legacy bridge is retained in the new outputs; the two metrics are
not pooled. The new study also uses three nested target-training sizes instead
of V5's single 256-example size. [V5 configuration](../pilot_v5_confirmation/artifacts/config.json),
[new protocol](PROTOCOL.md), [new statistics plan](stats_plan.json).

## Task-count V1: composition-dependent transfer with negative results

V1 used three paired units, two nested task chains, k=2/4/8/16, a fixed 4M-token
pretraining budget, and 256 adaptation examples. Fixed-points transfer was
nonmonotonic: alpha k=8 reached 70.57% mean accuracy and +9.08 pp mean gain over
scratch, but that gain had sample SD 18.73 pp and one of three units was negative.
Alpha k=16 instead had -4.04 pp mean transfer; beta had negative mean transfer at
all four k. Its mean k=16 gain was -11.23 pp. None of these source sets was
selected as a proven optimum. The shared scratch mean was 61.49%.
[V1 report](../task_count_v1/REPORT.md), [condition summaries](../task_count_v1/summary.csv).

The paired fixed-points k16-minus-k2 changes were +0.65 pp (SD 8.06) for alpha
and -4.98 pp (SD 5.51) for beta. LIS, the target that subsequently motivated the
budget study, declined by -5.47 pp (SD 3.92) for alpha and -6.02 pp (SD 2.23) for
beta, with negative signs in all three units in each chain. These are
within-unit changes along fixed compositions, not a pure task-count effect or
a scaling law. Mathematical relationships, answer distributions and per-task
exposure all changed with k. [Paired k-contrast summary](../task_count_v1/k_contrast_summary.csv).

V1 also retained substantial source underlearning. In alpha, fixed-points
transfer and exceedances source performance declined together from k=8 to k=16;
such a correspondence is descriptive and does not establish that inadequate
learning of that task caused the transfer change.

## Task-count V2: a budget interaction, not a deficiencies ablation

V2 introduced three new paired units and crossed k=2/16 with 4M/32M tokens for
the same two chains. LIS was primary; fixed_points was auxiliary. Each entry
below is the mean and sample SD of the **three within-unit** differences
`(k16-k2)_32M - (k16-k2)_4M`, using development-selected checkpoints and pooled
lengths 10 and 12. The two chains share these three units.

| Target | Chain | Mean budget interaction (pp) | Sample SD (pp) |
| --- | --- | ---: | ---: |
| lis_length | alpha | +17.9362 | 7.8941 |
| lis_length | beta | +6.5430 | 5.9594 |
| fixed_points | alpha | +41.4062 | 3.2726 |
| fixed_points | beta | -9.0820 | 11.5532 |

The fixed-points alpha unit interactions were +38.0859, +41.5039 and +44.6289 pp;
beta's were -22.1680, -0.2930 and -4.7852 pp. These opposite directions are
preserved. Alpha k=16 at 32M reached 88.44% mean target accuracy and +23.63 pp
gain over A. The other seven fixed-points conditions had negative mean gains;
beta k=16 at 32M had 43.39% accuracy and -21.42 pp gain. No result here says that
more training generally eliminates negative transfer.
[V2 report](../task_count_v2/REPORT.md), [paired interactions](../task_count_v2/paired_budget_interactions.csv),
[interaction summary](../task_count_v2/budget_interaction_summary.csv),
[condition summary](../task_count_v2/summary.csv).

Alpha k=16 includes exceedances and deficiencies, whose development accuracies
rose from 43.23% to 99.35% and 38.93% to 98.96% when budget increased. It also
contains another registered fixed-points derivation through even/odd/nontrivial
cycle counts. Beta k=16 contains anti_fixed_points; its development accuracy
rose from 44.27% to 63.35% while fixed-points transfer worsened. These observations
motivate examining particular sources, but V2 simultaneously changes the
training of many sources and does not isolate deficiencies.
[Registered source-set relations](../task_count_v2/design/README.md),
[source summaries](../task_count_v2/source_summary.csv).

The 32M runs started from paired random initialization with a longer relative-
budget learning-rate trajectory; they did not continue a fully decayed 4M
checkpoint. Both budgets reused the same input pool, so additional exposure was
not an eightfold increase in independent data. V2 used the platform's 203-token
encoding, lengths 10 and 12 and one adaptation learning rate, unlike the new
legacy-v5-based protocol. These studies motivate the new design but are not
numerically interchangeable with it.

## What the new hypothesis does and does not assert

The historical evidence justifies asking whether replacing the fourth source
with deficiencies has extra value **relative to the prespecified U task** in
one fixed background. It does not establish that answer derivability guarantees
transfer, that deficiencies is a universal preferred source, or that the model
executes `FP=n-E-D`. In particular, D(p) alone does not determine FP(p); the
registered deficiencies derivation needs D(p) and D(inverse(p)). T also permits
positional comparison and relates to FP through complement/reversal, while U
relates to the common prefix-maxima task through reversal. U is not declared
mathematically unrelated or matched to D in difficulty, label distribution or
redundancy. [Current relation table](RELATION_TABLE.csv),
[complete-composition query audit](preflight_audit/JOINT_DERIVABILITY.csv),
[historical relation inventory](../pilot_v5_confirmation/mathematics/audit/RELATION_INVENTORY.csv).

The new sole primary contrast is `S_D-S_U`, where
`S=.25*A256+.5*A512+.25*A1024`, evaluated in eight new paired units. Historical
unit counts, source models, conditions, task sizes and shared baselines do not
increase that number. D-P, U-P, D-T and gains over A are prespecified secondary
comparisons. If D and U both improve on P but D does not outperform U, the P
comparison cannot establish a benefit specific to deficiencies. An uncertain
primary interval remains uncertain rather than evidence of no benefit. No
outcome replaces the old V5 hypothesis or authorizes further targets, budgets,
architectures, source substitutions or mechanistic interventions.

## Runtime wording clarification

The frozen `analysis.py` report generator describes scientific training-run
times separately from other records. Its wording must not be read as excluding
in-run development evaluation from those timers: the training run's elapsed
time includes evaluations and writes inside the timed run. Final-test timing,
data checks and engineering timing are separate records. Overlapping stage and
run timers must not be added. This documentation clarification does not alter
frozen code or any timing values; the final canonical report should use the
explicit interpretation.

## Source bindings

The SHA256 values below bind the archived files consulted for this note. They
are recorded without modifying historical manifests. All paths are relative to
this experiment directory. Tables were read directly; the V5 C-group subtraction
and V2 four interaction means/SDs were cross-checked without model execution.

| Source | SHA256 |
| --- | --- |
| [../pilot_v5_confirmation/PILOT_V5_REPORT.md](../pilot_v5_confirmation/PILOT_V5_REPORT.md) | `c3aa246d26863fca447ca89ffc13bd030e11ce887c154823a58624da63792ddc` |
| [../pilot_v5_confirmation/artifacts/compositions.csv](../pilot_v5_confirmation/artifacts/compositions.csv) | `1e95509c94324dec13b7183bbcfa21ff2226080843b553d3de09bee2721f2607` |
| [../pilot_v5_confirmation/artifacts/config.json](../pilot_v5_confirmation/artifacts/config.json) | `70f5ce3dfd4911528700adb5d0e9dc23401878fe3fbbf6021e3cd2915be55203` |
| [../pilot_v5_confirmation/artifacts/confirmation_summary.csv](../pilot_v5_confirmation/artifacts/confirmation_summary.csv) | `6ae33532253b78cd6cc3cc768adcd19031d0ac0795677a37f1a26ab15bb52fd7` |
| [../pilot_v5_confirmation/artifacts/paired_condition_differences.csv](../pilot_v5_confirmation/artifacts/paired_condition_differences.csv) | `d363946e8c6eee1443d23b9eb1066722d7185dd787fad10c040bf65a322fa965` |
| [../pilot_v5_confirmation/mathematics/audit/RELATION_INVENTORY.csv](../pilot_v5_confirmation/mathematics/audit/RELATION_INVENTORY.csv) | `394b4b034b639450044fdefb32aa63a977306b8b2ef1d60d2c3ddf1ec02cf67e` |
| [../task_count_v1/REPORT.md](../task_count_v1/REPORT.md) | `6f7e6ea23afe817e2203202ed30a7420d3a66c5fd121d7367f3c270da6609015` |
| [../task_count_v1/summary.csv](../task_count_v1/summary.csv) | `8def0701012da94d0d9e630fc752c7539c70e61f6c1b70c1490152961491bbce` |
| [../task_count_v1/k_contrast_summary.csv](../task_count_v1/k_contrast_summary.csv) | `e63a0ad02952df7a2437ae8b286a2859adeace41e9aa0048056170d903dfe8eb` |
| [../task_count_v1/raw_test_results.csv](../task_count_v1/raw_test_results.csv) | `49bdc14e2fb2c7139708a294585400fdcbbcaec15b3c0dc73a50299e1177484c` |
| [../task_count_v2/REPORT.md](../task_count_v2/REPORT.md) | `20a8ac13500fb426db0e84e771c298f494932fa3e7ca7d7f89ef1f75c1f9f6f5` |
| [../task_count_v2/summary.csv](../task_count_v2/summary.csv) | `3ed9f25973aebc9a0bfad70c98ae12b67b6a9325519e9b7cc8e40b481b01a673` |
| [../task_count_v2/budget_interaction_summary.csv](../task_count_v2/budget_interaction_summary.csv) | `85b4c8630a054d49791ca0ca82342ea9a5aa873f3a2478d3eaa4ee3e84423819` |
| [../task_count_v2/paired_budget_interactions.csv](../task_count_v2/paired_budget_interactions.csv) | `67060bfca722eeb8ed07e7d68d6aac4f269b70cba010827eebd99d7a45f1948b` |
| [../task_count_v2/source_summary.csv](../task_count_v2/source_summary.csv) | `9041b52b94ee84b7d9e30cf73f1ff84b74d27820423019d86d37ce521bc75f2d` |
| [../task_count_v2/raw_test_results.csv](../task_count_v2/raw_test_results.csv) | `589043ef94ccdda2d48a751dcbae08f9664fe5c61083c1552cbe8f12b24b4d6c` |
| [../task_count_v2/design/README.md](../task_count_v2/design/README.md) | `b573942b46df970b1449101a17ff5c6ef9c9e088634775c8520d5727e4b85c05` |
| [preflight_audit/COMPOSITIONS.csv](preflight_audit/COMPOSITIONS.csv) | `c2cf470856d0c356f61ce1e7e3ec12caeee87a359144cb3614e76262bef4d81e` |
| [preflight_audit/JOINT_DERIVABILITY.csv](preflight_audit/JOINT_DERIVABILITY.csv) | `828c630a24f7ced3bea1b733d21c4aa04581bc95efd5276f042b525791dc7b48` |
