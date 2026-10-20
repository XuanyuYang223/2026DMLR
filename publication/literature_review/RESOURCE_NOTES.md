# Primary-source notes: CLRS, CLRS-Text, and Reasoning Gym

Reviewed on 2026-10-19. This is a bounded literature and code inspection, not a
reproduction of the other projects. Repository versions and published-paper
protocols are identified separately. An unverified feature is not an absent
feature. Counts of algorithms, generator endpoints, templates, and independent
answer functions are not interchangeable.

## Bibliographic verification

| Work | Verified authors and publication | Primary record |
| --- | --- | --- |
| *The CLRS Algorithmic Reasoning Benchmark* | Petar Velickovic, Adria Puigdomenech Badia, David Budden, Razvan Pascanu, Andrea Banino, Misha Dashevskiy, Raia Hadsell, Charles Blundell; ICML 2022, PMLR 162, pp. 22084-22102. Preserve the authors' diacritics in the bibliography. | [PMLR record and BibTeX](https://proceedings.mlr.press/v162/velickovic22a.html) |
| *The CLRS-Text Algorithmic Reasoning Language Benchmark* | Larisa Markeeva, Sean McLeish, Borja Ibarz, Wilfried Bounsi, Olga Kozlova, Alex Vitvitskyi, Charles Blundell, Tom Goldstein, Avi Schwarzschild, Petar Velickovic; arXiv:2406.04229, 2024. A formal venue was not verified; both the current arXiv record and official code citation identify a preprint. | [arXiv record](https://arxiv.org/abs/2406.04229), [official citation](https://github.com/google-deepmind/clrs/blob/6d5728b9efadf8648413c8c473929a733caacced/clrs/_src/clrs_text/README.md#citation) |
| *Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards* | Zafir Stojanovski, Oliver Stanley, Joe Sharratt, Richard Jones, Abdulhakeem Adefioye, Jean Kaddour, Andreas Kopf; NeurIPS 2025, Datasets and Benchmarks Track, *Advances in Neural Information Processing Systems* 38. Preserve the umlaut in Kopf in the bibliography. | [Formal proceedings record](https://proceedings.neurips.cc/paper_files/paper/2025/hash/528d56195a2c77c808494c86fa7c77ad-Abstract-Datasets_and_Benchmarks_Track.html), [arXiv v2 record](https://arxiv.org/abs/2505.24760v2) |

## CLRS: algorithm execution is the organizing unit

The paper's Sections 3.1-3.3 specify exact algorithm implementations, typed
inputs/outputs, and intermediate execution hints. Arrays, graphs, strings, and
geometric inputs are represented through node/edge/graph data. Importantly,
Section 3.2 explicitly notes that distinct sorting algorithms can have identical
final answers on identical inputs; their traces distinguish them. Therefore its
algorithm count cannot be directly compared to a count of distinct PermWorld
answer functions.

Sections 4.2-4.4 describe canonical data and length generalization; Appendix A
explains extension points, Appendix B illustrates traces, and Appendices C-D
report individual-algorithm results. This original paper primarily evaluates
single-task models but explicitly allows broader learning settings.
[Published paper](https://proceedings.mlr.press/v162/velickovic22a/velickovic22a.pdf).

The inspected official repository is
[`6d5728b9efadf8648413c8c473929a733caacced`](https://github.com/google-deepmind/clrs/tree/6d5728b9efadf8648413c8c473929a733caacced),
committed 2026-10-07. Its README documents seeded `build_sampler` generation,
sample count and input length, TensorFlow Datasets generation, and a shared
processor supporting multiple algorithms. Extension requires input/hint/output
specifications in `clrs/_src/specs.py`, algorithms with probes, and samplers in
`clrs/_src/samplers.py`.
[README: examples, creating datasets, adding algorithms](https://github.com/google-deepmind/clrs/blob/6d5728b9efadf8648413c8c473929a733caacced/README.md).

A systematic registry separating unary answer relations, jointly sufficient
source answers, transformed-input queries, and relation verification status was
not verified in this bounded inspection. This does not imply that the algorithms
lack mathematical relationships.

## CLRS-Text: a directly relevant language-model evaluation platform

Section 3 converts CLRS execution traces to text and exposes selected
intermediate states. Sections 4-5 train a generalist Gemma model across all
thirty algorithms and evaluate final-array exact string matches over resampled
problem sizes. The principal shift is length/interpolation/extrapolation
generalization on trained algorithms, rather than the particular held-out-task
adaptation protocol used here. Appendix A shows trace examples, Appendix B
lists algorithm-specific training sizes, and Appendix C supplies all task
curves. This is already a procedural, multi-task, model-compatible benchmark.
[Paper, Sections 3-5 and Appendices A-C](https://arxiv.org/html/2406.04229v1).

The official CLRS-Text directory documents `clrs_utils.py` for conversion,
`generate_clrs_text.py` for published train/evaluation JSON generation, and
`huggingface_generators.py` for compatible samples. It also links the examples
used in the paper. It inherits CLRS generation and extension interfaces; it
should not be described as a static dataset.
[Pinned official README](https://github.com/google-deepmind/clrs/blob/6d5728b9efadf8648413c8c473929a733caacced/clrs/_src/clrs_text/README.md).

The same systematic relation-registry feature described above remains
**unverified**, not asserted absent. No claim that all possible CLRS-Text
protocols exclude task transfer is warranted.

## Reasoning Gym: substantial overlap must be acknowledged

The formal paper establishes configurable procedural reasoning tasks and
automatic answer rewards (Section 2), held-out within-domain transfer
(Section 4.1), cross-domain transfer including negative entries (Section 4.2),
external-benchmark transfer (Section 4.3), and adaptive curriculum experiments
(Section 5). Section 4.1 uses composite-task GRPO training of
Qwen2.5-3B-Instruct. Appendix A.2 contains task examples, A.3 gives easy/hard
configurations, and A.6 provides the training approach and weighted mixture
configuration. Thus generation, difficulty control, mixing, verification,
curricula, and transfer are established overlap, not PermWorld-exclusive
capabilities.
[Published paper, including appendices](https://proceedings.neurips.cc/paper_files/paper/2025/file/528d56195a2c77c808494c86fa7c77ad-Paper-Datasets_and_Benchmarks_Track.pdf).

The separate official supplement has different section numbering: A.1 task
examples, A.2 easy/hard settings, and C training approach. It includes task
metadata, exact/approximate geometric answer forms, dataset mixture weights,
batch sizes, step limits, prompt/response lengths, and hardware.
[Supplement](https://proceedings.neurips.cc/paper_files/paper/2025/file/528d56195a2c77c808494c86fa7c77ad-Supplemental-Datasets_and_Benchmarks_Track.pdf).

### Pinned code inspection

The inspected repository revision is
[`49b07130b3fcd12f2d064bba7c43869543a0e7e7`](https://github.com/open-thought/reasoning-gym/tree/49b07130b3fcd12f2d064bba7c43869543a0e7e7),
committed 2026-04-17. These are current inspected capabilities, not an assertion
that each was present in the published experiment.

| Capability | Concrete evidence and limits |
| --- | --- |
| Sample interface and scoring | [`reasoning_gym/dataset.py`](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/reasoning_gym/dataset.py): `ProceduralDataset` carries configuration, seed, size, and question/answer/metadata records. `score_answer` defaults to exact-answer full credit with substring partial credit; tasks can override it. `score_answer_cascade` adds optional more permissive matchers. Reseeding supports continuing generation. Scoring is not uniformly final-string exact match. |
| Generator quality checks | [`tests/test_dataset_common.py`](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/tests/test_dataset_common.py) checks that generated oracle answers receive full credit across registered datasets. This is an answer/scorer consistency check, not a universal independent proof of correctness. |
| Task-specific validation | [`reasoning_gym/games/mini_sudoku.py`](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/reasoning_gym/games/mini_sudoku.py): `MiniSudokuConfig.validate` checks legal ranges; `_solve` uses backtracking; `_create_puzzle` invokes `_count_solutions` to preserve uniqueness when removing clues; actual empty-cell count is retained. It would be false to say generators have no correctness checks. |
| Task mixing and provenance | [`reasoning_gym/composite.py`](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/reasoning_gym/composite.py): `DatasetSpec` stores name, weight, and config; `CompositeDataset.__getitem__` uses deterministic weighted sampling by seed/index. Configurations and weights can be updated. Optional dataset version IDs and entry IDs support reconstructing the appropriate source/scoring context. |
| Difficulty and curricula | [`reasoning_gym/coaching/base_curriculum.py`](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/reasoning_gym/coaching/base_curriculum.py): named scalar/range attributes, levels, increment/decrement operations, and generated configurations; contexts distinguish current/accumulated/recent level ranges. These are explicit difficulty interfaces. |
| An actual held-out-task mixture | [`training/configs/intra_generalisation/algorithmic_qwen_3b.yaml`](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/training/configs/intra_generalisation/algorithmic_qwen_3b.yaml) mixes `spell_backward`, `letter_jumble`, and `word_sorting` with weights 0.33/0.34/0.33, dataset size 20,000, train batch size 32, and 500 steps. It records model, context limits, optimizer, rollout, logging, and checkpoints. These fields document intended composition and training limits; they are not a measured per-task token ledger. |
| Held-out evaluation interface | [`training/evaluations/intra-generalisation/eval_algorithmic_composite.yaml`](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/training/evaluations/intra-generalisation/eval_algorithmic_composite.yaml) evaluates `number_sorting`, with seed, generator parameters, saved metadata/results, and repeated sampling. Current YAML specifies 100 examples, while paper Section 4.1 describes 50. This difference is documented, not silently reconciled. |
| Training and reproduction documentation | [`training/README.md`](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/training/README.md) provides the environment, pinned verl revision, GRPO commands, checkpoint conversion, RG evaluation, and lm-evaluation-harness use. This review did not execute these workflows. |
| Learning feedback | [`training/trainers/ray_grpo_trainer.py`](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/training/trainers/ray_grpo_trainer.py): `_compute_correctness_score` delegates to generator scoring; curriculum updates use per-dataset grouped scores and recent sample counts; data/timing metrics are logged. This is evidence of learning/progress tracking. Complete cumulative per-task nonpadding sequence tokens and unique-input accounting were not verified. |

### Relation and comparison limits

The inspected transfer evidence is empirical and the domain taxonomy organizes
task categories. A complete machine-readable distinction among exact answer
formulas, jointly sufficient sets of source answers, transformed-input query
requirements, and proof versus finite-check status was **not verified**.
Individual RG tasks can nonetheless encode mathematical relationships; this
review did not enumerate them all.

The defensible PermWorld distinction to check against its own implementation is
the specific combination of functions on a shared permutation object,
explicit relation/query metadata, and controlled composition/budget/adaptation
experiments. Neither procedural generation nor the presence of transfer
experiments is sufficient novelty. Resource breadth is not a controlled
comparison of independent capabilities.
