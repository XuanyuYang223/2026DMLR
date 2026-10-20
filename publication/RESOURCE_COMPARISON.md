# Resource and method comparison

This comparison supports the DMLR manuscript's scope, rather than ranking
libraries by endpoint count. It separates data resources and evaluation platforms
from methods that select data or predict mixture performance. A work can provide
more than one of these. Evidence comes from the primary texts, appendices and
official code listed below; external experiments were not rerun.

**Evidence convention.** "Verified" means documented in an identified source,
not independently reproduced here. "Unverified" means this bounded inspection did
not establish the capability; it does not mean unsupported or mathematically
unrelated. A mathematical formula, an estimated learning dependency, and a
correlation in an experiment are different kinds of relation.

## Data resources and evaluation platforms

| Work and organizing unit | Objects, labels, generation and difficulty | Relations, evaluation and evidence |
| --- | --- | --- |
| **Saxton et al., Mathematics Dataset (ICLR 2019)**; mathematical problem modules | Text questions with exact textual answers across school mathematics. Type-compatible module composition and generation entropy control; interpolation and extrapolation settings. | Composed subproblems already express mathematical dependencies. Their role is question construction and generalization testing. [Paper Sections 2.1-2.5, Appendix B](https://arxiv.org/pdf/1904.01557); [official composition implementation](https://github.com/google-deepmind/mathematics_dataset/blob/427f45075f84b8b9774950196ad63867ca20ffb3/mathematics_dataset/util/composition.py). |
| **Lample and Charton, Symbolic Mathematics (ICLR 2020)**; symbolic equation tasks and generation schemes | Expression trees serialized as token sequences; integrals and ODE solutions. Forward, backward and integration-by-parts generation, with operator/length controls. | Mathematical identities drive generation; symbolic/numerical checks accept valid solutions beyond a single string. Mixed-generation training and cross-generation evaluation are already documented. [Paper Sections 3-4, Appendix E](https://arxiv.org/pdf/1912.01412); [official README, Data generation/Evaluation](https://github.com/facebookresearch/SymbolicMathematics/blob/4596d070e1a9a1c2239c923d7d68fda577c8c007/README.md). |
| **CLRS (ICML 2022)**; algorithm executions | Arrays, graphs, strings and geometric objects; typed inputs, final outputs and intermediate hints. Seeded algorithm-specific samplers and input-size control. | Execution traces distinguish algorithms even when final answers coincide. Published evaluation studies algorithm execution and length generalization; current library also supports shared multi-algorithm processors. [Paper Sections 3-4, Appendices A-D](https://proceedings.mlr.press/v162/velickovic22a/velickovic22a.pdf); [official README and extension instructions](https://github.com/google-deepmind/clrs/blob/6d5728b9efadf8648413c8c473929a733caacced/README.md). |
| **CLRS-Text (2024 preprint; formal venue unverified)**; textual algorithm traces | Text serialization of CLRS inputs and selected intermediate states; procedural JSON and Hugging Face generation interfaces. Algorithm-specific training sizes and resampled evaluation sizes. | Generalist LM training and final-output evaluation across trained algorithms. The cited study emphasizes size generalization; this does not rule out other transfer protocols. [Paper Sections 3-5, Appendices A-C](https://arxiv.org/html/2406.04229v1); [official generation README](https://github.com/google-deepmind/clrs/blob/6d5728b9efadf8648413c8c473929a733caacced/clrs/_src/clrs_text/README.md). |
| **Reasoning Gym (NeurIPS 2025, Datasets and Benchmarks)**; reasoning environments | Procedural question/answer/metadata records across mathematical, algorithmic and game tasks; configurable ranges and curricula; task-specific reward/scoring rules. | Weighted mixtures, held-out-task, cross-domain and external-benchmark transfer, and curriculum experiments are established overlap. [Paper Sections 2, 4-5](https://proceedings.neurips.cc/paper_files/paper/2025/file/528d56195a2c77c808494c86fa7c77ad-Paper-Datasets_and_Benchmarks_Track.pdf); [supplement, task/config examples and training approach](https://proceedings.neurips.cc/paper_files/paper/2025/file/528d56195a2c77c808494c86fa7c77ad-Supplemental-Datasets_and_Benchmarks_Track.pdf). Specific verification and accounting semantics are detailed below. |
| **PermWorld DMLR platform**; exact task instances over permutations | Shared one-based permutation object; scalar/list/nested answers, parameters and bounds. Objects are generated before labels, with versioned full-answer encoding and orbit isolation. Binary calls exist but certified binary query-group generation remains incomplete. | Single and joint reconstruction maps state required inputs and evidence class. Frozen transfer studies track source learning, composition, exposure and adaptation. See [`tasks.label`](../permworld_platform/tasks.py), [`relations.py`](../permworld_platform/relations.py), [`data.py`](../permworld_platform/data.py) and the [local audit](literature_review/LOCAL_EVIDENCE_NOTES.md). |

These rows do not count independent capabilities. CLRS algorithm traces,
Reasoning Gym environments, mathematical templates, parameter instances and
PermWorld endpoint names have different meanings. Sorting algorithms can share
an answer function; PermWorld includes aliases and parameter equivalences.
Neither raw endpoint totals nor feature checkmarks establish comparative value.

## Data selection and performance-prediction methods

| Work | Meaning of dependence and selection objective | Composition, budget, transfer and reproduction evidence |
| --- | --- | --- |
| **Skill-It (NeurIPS 2023)** | The formal dependency criterion requires that a uniform pair mixture need no more data to reach a specified metric than target-only training. Practical graph estimators use fixed-budget loss changes; graph-weighted sampling targets evaluation loss. LEGO/addition are synthetic precedents. | Fixed sample/token budgets, continual pretraining, fine-tuning and disjoint evaluation-skill settings. Official code generates synthetic tasks, consumes a supplied graph and logs mixture updates; the README says graph-estimation code is not included, while explaining the estimation runs. [Paper Sections 3-4, Appendix C](https://proceedings.neurips.cc/paper_files/paper/2023/file/70b8505ac79e3e131756f793cd80eb8d-Paper-Conference.pdf); [README](https://github.com/HazyResearch/skill-it/blob/807b3bf7b11970f3a6fb6201cb37dcc1a3335f00/README.md). |
| **DoReMi (NeurIPS 2023)** | Reference-relative excess loss supplies domain weights through distributionally robust proxy training, without downstream-task access. | Token-normalized objectives and fixed main-training steps; weights transfer from a small proxy to larger models. Public weighted sampling, configuration and downstream evaluation interfaces are documented, with explicit differences from the paper implementation. [Paper Section 2, Section 3.1, Appendix C](https://proceedings.neurips.cc/paper_files/paper/2023/file/dcba6be91359358c2355cd920da3fcbd-Paper-Conference.pdf); [official README](https://github.com/sangmichaelxie/doremi/blob/7cde52d1848737aa967ecbdb9e643cf334de160d/README.md). |
| **DoGE (ICML 2024)** | Source/target gradient interactions estimate generalization; proxy optimization learns weights for base-model training. OOD targets supply weight-estimation data despite being excluded from source parameter updates. | The paper includes subsequent target fine-tuning (Section 3.2/Table 3). Code logs per-domain loss, weights and sample counts; exposure reporting is not unique to PermWorld. [Paper Sections 2-3, Appendices B/D/E](https://raw.githubusercontent.com/mlresearch/v235/main/assets/fan24e/fan24e.pdf); [`src/trainer.py`, logging](https://github.com/Olivia-fsm/DoGE/blob/88400e9263dfdd69a24b28aec8533fcb1bf9363d/src/trainer.py#L772); [OOD configuration](https://github.com/Olivia-fsm/DoGE/blob/88400e9263dfdd69a24b28aec8533fcb1bf9363d/config/doge_ood/arxiv_ood.json). |
| **Data Mixing Laws (ICLR 2025)** | Empirical domain-loss models describe mixture responses; a nested prediction procedure combines ratios, training steps and model size. These are learning-loss relationships, not identities among task answers. | Section 4/Algorithm 1 studies these factors together; Appendix B leaves a single joint law unresolved. Official notebooks and pipeline perform step/size extrapolation and capacity-constrained mixture optimization. [Formal paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/cc84bfabe6389d8883fc2071c848f62a-Paper-Conference.pdf); [`pipeline/run.sh`](https://github.com/yegcjs/mixinglaws/blob/8ad3e1244b53294ded55a96ae325f4112ce3bc63/pipeline/run.sh). |

PermWorld does not provide a competing adaptive data-selection method. Its
fixed lists, uniform task slots, paired repeats and fixed-budget endpoints
support diagnostic contrasts. They do not optimize mixture weights, fit a
law over model sizes, or show superiority to these methods.

## Reasoning Gym: what was verified, and what was not

The closest resource already covers much of the engineering scope. At revision
`49b07130b3fcd12f2d064bba7c43869543a0e7e7`, the following distinctions matter:

| Question | Observed evidence | Comparison implication |
| --- | --- | --- |
| Generation and answer validation | [`ProceduralDataset`](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/reasoning_gym/dataset.py) exposes seeded samples and scoring. Its default permits substring partial credit; tasks override scoring. [`test_dataset_common.py`](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/tests/test_dataset_common.py) checks oracle-answer scoring; [`mini_sudoku.py`](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/reasoning_gym/games/mini_sudoku.py) checks uniqueness during construction. | Answer grading, legal-configuration checks and generator correctness checks all exist. They are distinct from a universal independent correctness proof. PermWorld's own finite certificates are also scope-limited. |
| Difficulty and mixtures | [`BaseCurriculum`](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/reasoning_gym/coaching/base_curriculum.py) maps difficulty levels to configurations. [`CompositeDataset`](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/reasoning_gym/composite.py) uses weighted sampling, task configurations, source metadata and optional version IDs. | Neither difficulty control nor mixture/provenance interfaces distinguish PermWorld on their own. |
| Transfer and reproduction | [Algorithmic training YAML](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/training/configs/intra_generalisation/algorithmic_qwen_3b.yaml) mixes three tasks; [evaluation YAML](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/training/evaluations/intra-generalisation/eval_algorithmic_composite.yaml) holds out `number_sorting`. [Training README](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/training/README.md) documents execution and evaluation. | Held-out transfer is already demonstrated. Current YAML uses 100 evaluation examples whereas paper Section 4.1 describes 50; code and published experiment are not silently equated. |
| Budget and observed exposure | Training config records weights, batch/step limits and sequence limits. [Trainer](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/training/trainers/ray_grpo_trainer.py) tracks grouped curriculum scores and recent counts. | A complete cumulative per-task nonpadding-token and unique-input ledger was **unverified**. This is not evidence of no logging, and logging alone would not establish scientific novelty. |

## Relation semantics and input requirements

The external sources above establish mathematical composition, execution traces,
known generative dependencies, empirical transfer and learned domain relations.
A systematic registry with the specific single/joint/query/evidence separation
below was **unverified in each inspected external project**. We do not infer its
absence from an abstract or claim that the underlying tasks lack such relations.

| PermWorld distinction | Concrete record | What it permits, and what it does not establish |
| --- | --- | --- |
| Single-task answer map | `lis_length_complement`: LIS(p) = LDS(complement(p)). | The required answer is on a transformed input, not necessarily the same training example. |
| Joint same-input derivation | `position_partition`: FP(p) = n - E(p) - D(p). | Two source answers and length suffice mathematically. Jointly learning the labels or using that derivation is a separate question. |
| Multiple input queries | `fixed_from_two_deficiency_queries`: FP(p) = n - D(p) - D(inverse(p)). | One task name does not imply both required query answers were supplied. |
| Parameter-family requirements | `lis_from_subsequence_histogram` requires all k=1,...,n counts. | Available k=1,2,3 instances do not satisfy the entire reconstruction requirement. |
| Evidence and execution scope | `proof`, `evidence_class`, `finite_check`, `query_model`, `answer_map`; 45 new and 22 historical executable maps, plus seven declarative historical records. | Definition-based proofs, finite implementation checks and empirical observations have separate status. The registry is neither exhaustive nor fully executable. |

Source: [`relations.py`](../permworld_platform/relations.py), particularly
`_edge`, `q`, `new_relations`, `_historical_edges` and `evaluate_relation`;
[archived records](../experiments/platform_v01/mathematics/RELATIONS.json).
The [local evidence notes](literature_review/LOCAL_EVIDENCE_NOTES.md) distinguish
runtime metadata from completed archived certificate coverage.

This explicit information-sufficiency description is the proposed scientific
use of the platform. It lets a study report three separate outcomes:

1. **Derivability:** an answer map and its required inputs are valid.
2. **Source learning:** a model predicts the relevant source answers under a stated budget.
3. **Transfer:** adaptation improves relative to a paired baseline.

The first does not imply the second or third; source accuracy alone does not
identify formula use. Mathematical query structure is legitimate supervision
structure, not automatically data leakage. Split and query rules determine
whether a protocol violates input isolation.

## PermWorld accounting, inheritance and limits

The [v2 protocol](../experiments/task_count_v2/FROZEN_PROTOCOL.json),
[exposure table](../experiments/task_count_v2/task_exposures.csv),
[source-learning table](../experiments/task_count_v2/source_learning.csv) and
[`train.py`](../experiments/task_count_v2/train.py) bind fixed task sets to
data/initialization/sampling seeds, sequence and supervised tokens, sample
exposures, unique inputs, repetition, and adaptation selection. These are
measured fields rather than assumed equalities. Uniform task slots need not
give equal token shares. Increasing budget reuses a fixed input pool; its
relative-progress schedule changes the absolute-token learning-rate trajectory.

Inherited PermWorld supplies task definitions, generators, mathematical
transforms and task/training infrastructure. DMLR adds the unified registry,
versioned answer interface, typed relation/query records, certificates and
staged transfer studies. Six additional registered endpoints are not six new
algorithms: cycle-length counting already exists in the inherited code. The
workshop alignment work is cited background; new TMLR representation results
are outside the comparison.

The proposed contribution is therefore a particular object and measurement
design, not merely tidier software. Existing experiments illustrate conditional
relationships among composition, source learning and adaptation, retaining
negative transfer, failed confirmation, underlearned sources and small-repeat
uncertainty. They do not establish a general advantage of annotations, a
universal task-count trend, model formula use or a scaling law.

Unresolved questions include the completeness of the relation inventory,
annotation usefulness beyond the present hand-selected families, binary-query
data support, longer inputs, robustness across architectures and larger repeat
samples, and empirical comparison with the cited methods. These are evidence
gaps; this revision does not start new experiments.

## Source audit and publication metadata

All nine works remain in [references.bib](references.bib). The review uses formal
ICLR/ICML/NeurIPS versions where verified, with CLRS-Text explicitly left as a
2024 preprint. It corrects Lample and Charton to ICLR 2020 and confirms the
existing DoReMi author order. The formal Data Mixing Laws author order is
Ye, Liu, Sun, Zhan, Zhou, Qiu.

- [Existing-reference verification](literature_review/EXISTING_REFERENCES.md):
  Saxton, Lample and Charton, DoReMi, including access/version caveats.
- [Resource notes](literature_review/RESOURCE_NOTES.md):
  CLRS, CLRS-Text and Reasoning Gym, with paper sections and pinned code.
- [Method notes](literature_review/METHOD_NOTES.md):
  Skill-It, DoGE and Data Mixing Laws, with definitions and code evidence.
- [Local evidence audit](literature_review/LOCAL_EVIDENCE_NOTES.md):
  what the DMLR implementation and existing results actually support.
- [Revision record](literature_review/REVISION.md):
  protected artifacts, editorial changes and validation scope.
