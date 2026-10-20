# PermWorld: A Relation-Annotated Data Platform for Composition and Budget Studies of Transfer

**DMLR working draft.** Authors, affiliations, venue and license are pending author approval. Repository names do not indicate publication acceptance. This draft synthesizes completed DMLR experiments; the release check adds no model training or test evaluation. Numerical evidence is pinned in the [DMLR evidence ledger](evidence/dmlr_claims.csv) and [figure sources](figures/README.md). Original PermWorld task and generator provenance is credited below; the separate TMLR representation study is outside this manuscript.

## Abstract

How does the composition of a pretraining task set influence learning of a held-out task under a fixed training budget? We present a relation-annotated PermWorld data platform and a controlled transfer evaluation workflow. Building on the original exact-label permutation environment, the platform retains 48 legacy endpoints, exposes 54 endpoints in total, distinguishes templates from parameter instances and encodings, and records single-task and joint relations with explicit query requirements and evidence status. Its complete-answer interface supports scalar and structured outputs, reproducible orbit-isolated data and independent correctness checks. The evaluation workflow records source composition, token budgets, task exposure, paired repeats and development selection. Completed experiments show heterogeneous transfer across targets and compositions. A five-repeat confirmation does not reproduce one exploratory background interaction in direction and leaves the second uncertain. Across four targets and two nested source-task chains, fixed-token task-count curves are nonmonotonic. A new three-repeat budget comparison changes the LIS k16-minus-k2 contrast by +17.94 and +6.54 percentage points in the two chains, while fixed-points behavior remains composition dependent. These findings support a reusable data platform and conditional composition-and-budget conclusions, not a scaling law or evidence that models execute the annotated formulas.

## 1. Introduction

Pretraining data are often described by size or domain proportions. For mathematical data, however, tasks also differ in the computations they require and in how their answers relate. Two tasks can be identical after transforming the input; several answers can jointly determine a target; and a task that appears useful mathematically can be difficult for a model to learn at a fixed budget. A useful testbed must expose these distinctions rather than treating every task name as an independent source of supervision.

Permutations offer a compact setting with exact labels and rich relations. For a permutation p, fixed points, cycle statistics, subsequences and block decompositions share an input object but require different computations. Inverse, reversal and complement transformations expose correspondences specified independently of a model. We use these controls to study whether learning selected source tasks improves learning a held-out task, and how the answer changes with source composition and training budget.

The original PermWorld implementation supplies the inherited task definitions, exact generators, mathematical transformations and task/transfer infrastructure. The DMLR extension unifies those interfaces, extends the task families, records executable relations and investigates transfer through fixed-four-source pilots and later fixed-token task-count and budget experiments. The original workshop alignment study is cited background; newer representation-study results are reserved for the separate TMLR paper.

Our three contributions are: (i) a relation-annotated, executable data platform with explicit task and relation accounting; (ii) a paired transfer workflow with auditable source composition, training-token budgets, exposure and development selection; and (iii) conditional empirical findings across targets, compositions and budgets, including unsuccessful confirmation and negative outcomes. The aim is to make the scientific question reproducible and interpretable, not to claim that more tasks or known relations invariably help.

## 2. Context and scope

Synthetic mathematical environments make exact evaluation and controlled generalization possible. The mathematical reasoning dataset of Saxton et al. supplies structured problems across several mathematical areas; Lample and Charton demonstrate synthetic symbolic-expression training for sequence models. PermWorld emphasizes a complementary control: many exact functions over the same finite object space, accompanied by explicit transformations and joint answer relations. [Saxton et al., 2019](https://arxiv.org/abs/1904.01557); [Lample and Charton, 2019](https://arxiv.org/abs/1912.01412).

Data-mixture methods such as DoReMi optimize domain weights for language-model pretraining. Our fixed compositions are a diagnostic experimental device, not a competing mixture optimizer or a claim about real-world language-model domains. [Xie et al., 2023](https://arxiv.org/abs/2305.10429).

The original PermWorld workshop work also used its mathematical correspondences to study representation alignment. We cite that work as background to the resource, without reproducing its alignment results or treating them as DMLR contributions. The original task definitions and generators are credited to the [pinned PermWorld source](https://github.com/XuanyuYang223/neurips/tree/74f0de2017115f06f11b7285366b66e7e8431d30). Final author-approved workshop citation metadata remain pending.

The present contribution should be compared on exact relations, explicit query requirements, reproducible generation and staged transfer controls. It is not claimed to be the first synthetic mathematical dataset, the first permutation benchmark, or the first study of multitask interference. A submission-specific literature review and final novelty wording remain author review items.

## 3. Resource and mathematical controls

### 3.1 Objects, tasks and counting conventions

Inputs are one-based permutations of length n. The inherited task declarations contain 48 distinct endpoint names across overlapping task lists. The original task definitions, generators and inverse/reverse/complement transformations are retained. Endpoint names, mathematical templates, parameter instances and encodings are distinct counting units.

Platform v0.1 retains the 48 legacy endpoints and adds six registered endpoints. The resulting 54 endpoints correspond to 48 canonical mathematical templates after excluding designated aliases and representation outputs. Default declarations contain 62 endpoint instances, of which 56 unary non-alias instances are materialized in the platform dataset. Binary functions remain callable but do not yet have certified binary query-group data generation. These counts describe interface organization, not mathematical independence.

The three focus families cover cycle decomposition (six templates), global subsequences (five), and permutation blocks (five). Their 16 templates produce 24 default parameter instances. At the data/calibration lengths n=8,10,12, two instances are constant conditional on length: the increasing and decreasing subsequence counts at k=1 both equal n. Consequently there are 22 nonconstant focus instances. Additional equivalences are documented, including cycle-length count at k=1 equaling fixed points and decreasing-pair count equaling inversion count. The registry records definitions, parameters, constraints, exact algorithms, output bounds, complexity and measured generation cost.

### 3.2 Relation types and evidence

We distinguish a single-task relation f(p)=h(g(Tp)), a joint relation f(p)=h(g1(T1p),...,gm(Tmp)), and the input queries required to apply it. A proof from definitions, finite implementation verification and an empirical association are different evidence types. Failure to register a relation is not evidence of mathematical unrelatedness.

For example, fixed points FP, exceedances E and deficiencies D satisfy

> FP(p) + E(p) + D(p) = n.

Every position belongs to exactly one of p_i=i, p_i>i or p_i<i, which proves the identity. Also E(p)=D(p^-1). Thus FP(p)=n-D(p)-D(p^-1), but the latter requires answers on two related inputs. A source set with D(p) alone has not supplied the second answer. Relations are legitimate task structure; they become data-isolation issues only when the split or query protocol violates the declared rules.

Other registered controls include fixed/anti-fixed points under complement, LIS/LDS under complement, and subsequence-count/inversion identities. The platform records 45 new relation entries and 29 references to historical v5 certificates. Historical content is bound by source hashes rather than repeatedly re-enumerated. New-task checks use independent algorithms and exhaustive permutations through n=7 plus fixed larger fixtures: 272,741 label checks and 445,581 relation checks pass over 6,105 fixtures. Finite enumeration validates implementations within its coverage; the mathematical status of a formula is recorded separately.

### 3.3 Encoding, generation and isolation

The active platform input encoding is `one_line_v1`. The output protocol `permworld_sequence_v1` extends the inherited vocabulary from 188 to 203 tokens while preserving old IDs. It supports base-100 integers and nested lists with explicit termination. Large values and structured answers are neither truncated nor reduced modulo a fixed class count. Some legacy outputs can exceed a 128-token context; the reader rejects such examples explicitly.

Data are generated as objects before labels are attached. The canonical platform release contains 9,216 unique permutations and 516,096 labels across two repeat units and lengths 8,10,12. Train/development/test inputs are separated under the full inverse/reverse/complement orbit rule, with historical inputs excluded. The task-count studies use new pools at lengths 10 and 12 and retain the same isolation principle. Uniform proposals followed by sequential orbit rejection are not uniform orbit-class sampling and introduce finite-population dependence across units. We describe the units as paired repeats, not strictly IID samples from an unconstrained population.

Figure 1 summarizes the DMLR platform, typed relation metadata, audited generation and transfer workflow, and development-only calibration. It contains no representation-study results.

![Relation-annotated platform, data audit and development-only calibration](figures/figure1_permworld_resource.png)

*Figure 1. Platform coverage, relation types, archived data checks and development-only learnability calibration. Template/instance counts use distinct denominators. The complete [caption and source tables](figures/README.md) state the scope of each panel; historical audit counts and calibration are not newly rerun training results.*

## 4. Experimental record and protocol boundaries

### 4.1 Transfer pilots and calibration

The pilot sequence is a staged investigation rather than one homogeneous benchmark. Pilot v1 varies target sample size 64/256/1024 for fixed points with two paired seeds. Pilot v2 diagnoses adaptation selection and source learning. Pilot v3 compares A (scratch), B (four sources including a designated related source) and C (one source replacement) for four targets, three paired seeds and 256 target examples. Targets are selected from mathematical definitions and single-task development learnability, not transfer test outcomes.

Pilot v4 examines background combinations for two cases chosen from v3, so its case selection is exploratory. Pilot v5 freezes two v4-derived contrasts and uses five new paired repeat units. It reports the paired difference in B-C between backgrounds, separately from old exploratory results. Source composition, task difficulty and answer distributions cannot be reduced to a single relation-present indicator.

Platform calibration freezes two representatives per new focus family, six tasks in total, before training. Each is trained with two seeds for 2,400 updates. Endpoint development performance is the calibration reference; no model test forward is used. These runs assess learnability at the stated configuration and lengths, not unseen-length generalization.

The early pilots use a single-answer-token evaluator suitable for their scalar tasks. Platform and task-count experiments use complete generated answers including EOS and a 203-token vocabulary. Absolute accuracy across these protocol generations must not be pooled or described as a controlled improvement from one version to another.

### 4.2 Platform-compatible task-count experiments

Task-count v1 freezes two genuinely different nested source chains with k=2,4,8,16. The 24-instance source pool spans the three focus families and legacy statistics; k counts nonconstant instances after excluding complete aliases, not independent mathematical information. All four held-out targets—cycle count, LIS length, components and fixed points—are excluded globally, as are their exact same-input answer aliases. Transformed-input and joint relations remain explicit, including answer mappings that may reveal a target.

V1 trains 24 source models: two chains times four k values times three paired repeats. Each model receives a fixed budget of four million nonpadding sequence tokens, including prompt and answer. Ninety-six adaptations reuse these models across four targets; 12 scratch baselines are shared across k and chain. Each target has 256 examples, balanced across n=10,12.

Task-count v2 is motivated by v1 and reported separately. It retains the same compositions at k=2,16 and compares four million with 32 million tokens using three new paired repeats. LIS is primary and fixed points auxiliary. Twenty-four pretrained models support 48 adaptations, plus six shared scratch baselines. Both budgets start from the same paired random initialization with empty optimizer state. They use the same relative-budget warmup/cosine rule, so their learning rates differ at the same absolute token count; the long run does not continue a fully decayed short endpoint.

The fixed source pool has 8,192 objects per repeat in v2. Extra tokens can expose unseen task-input pairs within that pool as well as repeat already seen pairs. They do not supply eight times as many independent objects. Per-task samples, tokens, unique inputs and repetitions are recorded. Uniform task slots do not imply equal token counts when answer lengths differ.

Both task-count studies use a four-layer, width-256, eight-head Transformer, context 128, dropout 0.1 and tied input/output embeddings. AdamW uses initial learning rate 3e-4, weight decay 0.01, betas (0.9,0.999), epsilon 1e-8 and gradient clipping at 1.0. Training uses BF16 on CUDA with FP32 weights/evaluation and TF32 disabled. The per-example loss is the mean cross-entropy over answer tokens plus EOS, followed by an average over examples; prompt tokens are masked.

Adaptation uses 800 updates, batch size 64, a single 3e-4 initial learning rate, 100 warmup updates and cosine decay to 3e-5. All four held-out task-token rows are reset to paired random initialization before every adaptation, with a fresh AdamW optimizer. Checkpoints are selected every 100 updates by highest complete-answer development accuracy, then lowest answer cross-entropy, then earliest update; update zero is excluded. Independent test inputs are generated after selections freeze. Selected and endpoint tests are both retained, with no test-driven selection.

### 4.3 Statistical units and estimands

For repeat r, define the count contrast at budget b as D_b,r=accuracy(k16,b,r)-accuracy(k2,b,r), and the budget interaction as I_r=D_32M,r-D_4M,r. Differences are computed within repeat before summaries. The shared scratch baseline cancels from this interaction. Reusing a pretrained model across targets or a baseline across chains does not add independent repeats.

Figures show individual repeat values and descriptive mean/sample standard deviation unless explicitly labeled otherwise. V5 retains its original t intervals, including multiplicity-adjusted intervals, with their small-sample limitations. Chains are fixed compositions, not random samples from a population of task sets. Cross-stage panels do not share one pooled error estimate.

## 5. Findings

### 5.1 Exact generation does not guarantee easy learning

Platform calibration distinguishes label correctness from learnability at a fixed budget. Mean endpoint development accuracy is 42.84% for cycle count, 22.53% for longest cycle, 86.26% for LIS, 25.26% for increasing length-3 subsequence count, 99.74% for components and 99.02% for largest component. Their respective training-derived majority baselines are 32.68%,16.99%,41.80%,4.36%,77.08%,77.08%. At n=12, cycle count falls below its majority baseline and length-3 counting remains difficult. High block-task accuracy also needs the high baseline for context. These are development measurements from two repeats, not six independently validated test claims.

### 5.2 Transfer varies across targets and backgrounds

Pilot v3 shows mean B-A differences of -10.12 percentage points for fixed points, +2.59 for exceedances, -1.53 for left-to-right minima and +11.20 for longest decreasing run. B-C changes sign across the three seeds for all four targets. Fixed points is negative relative to scratch in all seeds for both source compositions. Longest decreasing run has a positive mean B gain but one negative seed, while scratch performance remains near the majority baseline. The single-task eligibility check used more data than adaptation; passing it does not guarantee success from 256 examples.

Pilot v5 retains the distinction between discovery and validation. The fixed-points background interaction changes from an exploratory +4.36 points to -3.02 points across five new repeats, with a 95% t interval [-22.57,+16.54]. Its selected direction does not reproduce. The longest-decreasing-run interaction is -13.15 points with interval [-28.89,+2.58], consistent in direction with exploration but still uncertain. Neither supports a stable confirmed background effect. Auxiliary gains shared by B and C are not evidence for a B-C interaction.

### 5.3 Fixed-token task-count curves are nonmonotonic

Across the four targets and two chains in v1, none of the eight mean accuracy curves is nondecreasing at every k step. LIS has a negative k16-minus-k2 contrast in all three repeats of each chain, with means -5.47 and -6.02 points, although all its pretrained condition means exceed the scratch mean. Cycle-count condition means are at or below scratch. Components often benefits but is nonmonotonic. Fixed-points outcomes vary substantially across chains and seeds.

These findings concern two prespecified sequences of task sets at one total-token budget. Increasing k changes the labels learned, available mathematical relations, answer distribution and per-task exposure. The data do not isolate an abstract effect of task number independent of those changes.

### 5.4 A larger budget changes the count contrast

In new v2 repeats, LIS chain alpha changes from D_4M=-6.18 to D_32M=+11.75 points, giving interaction +17.94 (sample SD 7.89). Chain beta changes from -5.79 to +0.75 points, giving +6.54 (SD 5.96). All three interaction values are positive in each chain. A positive interaction does not mean every long-budget k16-minus-k2 comparison is positive: beta retains mixed signs at 32M.

The auxiliary fixed-points result is sharply composition dependent. Its budget interaction is +41.41 points (sample SD 3.27) for alpha and -9.08 (SD 11.55) for beta. At 32M/k16, alpha exceeds the shared scratch mean by 23.63 points while beta falls below it by 21.42. Therefore a longer budget is not a universal cure for negative transfer.

Source records help describe, but do not identify, a mechanism. In alpha/k16, exceedance and deficiency development accuracy rise from about 43.23%/38.93% to 99.35%/98.96%, alongside fixed-points improvement; their joint identity makes this informative. It does not prove formula use. Conversely, alpha/k2 LDS accuracy reaches about 99.09% at 32M while LIS transfer decreases on average, showing that source accuracy alone is not a sufficient transfer predictor. More repetitions, new task-input coverage within the fixed pool, and differing absolute learning-rate trajectories remain part of the budget intervention.

![Transfer, confirmation, task count and budget evidence](figures/figure2_evidence_overview.png)

*Figure 2. Four complementary downstream views: cross-target transfer, exploratory versus new-repeat background interactions, four-target task-count endpoint contrasts and the budget interaction. Every panel retains paired units and its own protocol. See [captions and raw plot tables](figures/README.md); the supplementary figure shows all four k values rather than only endpoints.*

## 6. Limitations and implications

The mathematical environment permits exact evaluation and controlled relations, but it remains a synthetic finite setting. The DMLR platform experiments use one architecture, one active input representation and a small length range. They do not establish transfer to natural language, unseen mathematical domains or longer permutations. Three paired repeats in the task-count studies provide limited uncertainty information; fixed chains do not support task-set population inference.

Known mathematical information, optimization difficulty, answer imbalance and exposure vary jointly in many compositions. The joint derivability of a target is not evidence that a model computes that derivation. Failed confirmation, majority-level target performance and underlearned sources remain scientifically relevant outcomes.

The pipeline changes between early pilots and the platform are intentional and documented. Complete-answer evaluation is necessary for multi-token and structured outputs; it also prevents a single pooled accuracy analysis across the full history. The platform's checks cover declared lengths and algorithms, not every possible parameter/input regime. Binary task splits and some long-context outputs remain engineering gaps.

Resource availability also has limits. Source, data, protocols, raw results and figure evidence are available in Git, but DMLR checkpoints remain local. The release acceptance scope is installation, data interfaces, table recomputation, figure generation and source-only verification; it does not rerun GPU training or model predictions. The acceptance record identifies the tested environment and any failures/fixes. Final licensing, authorship and a stable archival identifier remain release tasks. This draft is not an accepted or formally archived publication.

A stronger general or mechanistic claim would need a separately frozen experiment that separates relation structure, source mastery and exposure across a broader task-family sample. Further background variants for the same two targets would not resolve those gaps by themselves. No such experiment is run or promised by this release check.

## 7. Artifact availability and reproducibility

The [release guide](RELEASE.md) maps each contribution to code, registry, data, mathematical certificates, raw results and reproduction commands. The default figure builder and numerical verification read DMLR-local released tables and registry/audit records only. [Contribution boundaries](CONTRIBUTIONS.md) and a [claim ledger](evidence/dmlr_claims.csv) make the synthesis reviewable without rerunning models. The preserved [upstream audit](evidence/original_permworld_audit.md) is optional archival provenance, outside the default scientific reproduction dependencies.

This release check preserves the raw data, results, frozen configurations and scientific manifests. Derived publication material and documentation revisions have their own provenance records. GPU reproduction belongs to the original frozen protocols and must use fresh output directories. Publication verification requires no training, local checkpoint, untracked reference clone or original working-directory path.

## Appendix A. Evidence and protocol map

| Stage | Scientific question / outcome | Statistical units and scope | Entry point |
| --- | --- | --- | --- |
| Inherited PermWorld resource | Exact task definitions, generators and mathematical transformations | Resource provenance; workshop alignment is background only | [Pinned original source](https://github.com/XuanyuYang223/neurips/tree/74f0de2017115f06f11b7285366b66e7e8431d30) |
| Pilot v1 | Fixed-points sample efficiency | 2 paired seeds; 64/256/1024 samples | [Report](../PILOT_REPORT.md) |
| Pilot v2 | Adaptation and source-budget diagnostics | 2 paired seeds; 256 samples; original and extended budgets | [Report](../experiments/pilot_v2_diagnostics/DIAGNOSTIC_REPORT.md) |
| Pilot v3 | Cross-target composition transfer | 4 targets, 3 paired seeds, 256 samples | [Report](../experiments/pilot_v3_multitarget/PILOT_V3_REPORT.md) |
| Pilot v4 | Background exploration | 2 selected targets, 3 backgrounds, 3 paired seeds | [Report](../experiments/pilot_v4_backgrounds/PILOT_V4_REPORT.md) |
| Pilot v5 | Frozen background confirmation | 2 contrasts, 5 new paired repeats | [Report](../experiments/pilot_v5_confirmation/PILOT_V5_REPORT.md) |
| Platform v0.1 | Resource correctness and learnability | 6 preselected tasks, 2 seeds; development only | [Report](../experiments/platform_v01/PLATFORM_REPORT.md) |
| Task-count v1 | Fixed-total-token finite curves | 4 targets, 2 chains, 4 k values, 3 paired repeats | [Report](../experiments/task_count_v1/REPORT.md) |
| Task-count v2 | Budget-by-count interaction | 2 targets, 2 chains, 2 k values, 2 budgets, 3 new repeats | [Report](../experiments/task_count_v2/REPORT.md) |

## Appendix B. Compute accounting

Platform calibration performs 28,800 updates across 12 runs and about 52.84 million sequence tokens, with recorded training time of 14.23 minutes. Task-count v1 uses 95,999,744 actual pretraining tokens across 24 models and about 85 minutes for the training stage. Task-count v2 uses 431,999,672 actual pretraining tokens plus 82,944,000 adaptation tokens; its recorded training stage is 124.20 minutes and final test pass 30.73 seconds. These are measured local runs, not portable hardware benchmarks or the total cost of the entire historical program. Per-run timings, environment details, supervised-token counts and budget tails remain in the original compute records.

## References

- Saxton, D., Grefenstette, E., Hill, F., and Kohli, P. (2019). [Analysing Mathematical Reasoning Abilities of Neural Models](https://arxiv.org/abs/1904.01557).
- Lample, G., and Charton, F. (2019). [Deep Learning for Symbolic Mathematics](https://arxiv.org/abs/1912.01412).
- Xie, S. M., et al. (2023). [DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/abs/2305.10429).
- Original PermWorld resource and workshop background: [pinned source revision](https://github.com/XuanyuYang223/neurips/tree/74f0de2017115f06f11b7285366b66e7e8431d30). Task definitions, generators and mathematical transformations are inherited; final author-approved software/workshop citation metadata are pending.
