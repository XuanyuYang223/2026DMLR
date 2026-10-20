# Original PermWorld evidence and provenance audit

The integrated project has three distinct layers: the original permutation
resource and multitask/geometry studies in `XuanyuYang223/neurips`; the subsequent
representation-pairing study in `XuanyuYang223/2026TMLR`; and the data-centric
transfer experiments and platform extensions in `2026DMLR`. The first two are
substantive prior contributions, not merely implementation dependencies. The
integrated manuscript must attribute them explicitly and keep their numerical
evidence separate from the DMLR experiments.

This audit reads source, protocols and saved result tables. It runs no training,
model forwards or new mathematical experiments. It does not establish individual
authorship, publication acceptance, priority or external novelty. Repository
names are not evidence of a publication venue. The existing sources describe
workshop/manuscript material; the new integrated paper is a draft.

## Pinned sources

| Layer | Canonical repository | Audited revision | Local checkout |
|---|---|---|---|
| Original permutation resource and studies | [XuanyuYang223/neurips](https://github.com/XuanyuYang223/neurips/tree/74f0de2017115f06f11b7285366b66e7e8431d30) | `74f0de2017115f06f11b7285366b66e7e8431d30` | `reference/neurips` |
| Representation-pairing extension | [XuanyuYang223/2026TMLR](https://github.com/XuanyuYang223/2026TMLR/tree/d9a113d20d34db3c21cf69cf546cff4a0ab4904a) | `d9a113d20d34db3c21cf69cf546cff4a0ab4904a` | `reference/2026TMLR` |
| DMLR additions | `XuanyuYang223/2026DMLR` | Per-file source revision recorded in the inventory | Current repository |

Both reference checkouts were clean. Read-only `git ls-remote` returned the same
HEAD revisions during this audit. The actual canonical repository names are
`neurips` and `2026TMLR`; this audit did not infer a separate `2026neurips`
repository. The reference directories are ignored and are not required to read
the exported evidence or regenerate the integrated figures.

[source_inventory.csv](source_inventory.csv) records 42 selected files, their
repository/local paths, pinned URLs, revisions, available contribution and
limitations. It retains both Git-blob and local-working-tree SHA-256 values:
Windows checkout newline conversion can make those hashes differ without a
source change. [UPSTREAM_AUDIT.json](UPSTREAM_AUDIT.json) records the actual
checkout, arithmetic and remote-metadata checks.

## Original resource contributions supported by code

The upstream `math_ops.py` declares 20 V2 tasks, 20 V3 tasks and 32 Property32
statistics. Reading the actual declarations gives **48 distinct endpoint names
in their union**, not 72 independent tasks. Exact oracles cover encodings,
permutation operations, order statistics, cycle statistics, subsequences and
decomposition statistics. The generator and verifier provide executable task
dispatch, mathematical-answer recomputation and encoding checks.

The upstream Passage Math implementation supplies typed task/operand/answer
tokens and unbounded nonnegative integer serialization in base 100. Its
representation interface exposes `one_line`, `cycle`, `lehmer` and
`inversion_vector`. The four-representation transfer protocol crosses these with
eight tasks, trains the one-line row and descents column (11 cells), and holds
out the other 21 cells. These representation variants must not be counted as
additional mathematical task templates. The upstream causal Transformer and
causal MLP share a language-model API; model architecture is not a new DMLR
contribution. [Source implementation](https://github.com/XuanyuYang223/neurips/tree/74f0de2017115f06f11b7285366b66e7e8431d30/src/neurips_permutations)

The upstream record also contains task-count behavior, linear probes, low-shot
adaptation, matched-learning-rate sensitivity, data-by-depth scaling, and
relation-controlled geometry studies. These are prior empirical contributions.
They are not substituted for the later DMLR fixed-token experiment: tasks,
targets, examples, budgets, evaluation splits and statistical units differ.
The original opposite-pool zero-shot means at k=1/2/4/8/16 were
12.25/11.49/11.96/13.64/16.72%, below the 32.83% majority baseline throughout.
This existing task-count evidence already cautions against a monotonic execution
or scaling claim. [Saved behavior summary](https://github.com/XuanyuYang223/neurips/blob/74f0de2017115f06f11b7285366b66e7e8431d30/results/property32-zero-overlap/replicates/behavior_summary.csv)

## Original alignment result suitable for a main figure

The selected original result is the transformation-specific specialist
comparison, not a newly selected DMLR outcome. Its frozen protocol uses 16
single-task specialists at seeds 17, 42 and 101, four-layer Transformers,
one-line encoding, fixed 20,000-update endpoints and the same 4,096 task-free
validation prefixes. CKA is extracted at `<ONE_END>`, before the task query;
the exported layer is `final_norm`. This is one 256-dimensional vector at the
last prefix token, without token mean pooling. Saved probes span every length
n=2 through 30; the exact histogram is included in the exported resource
inventory. Ordinary linear CKA is computed across all 4,096 pooled example
vectors, without within-length adjustment or answer stratification. Those later
TMLR controls are a separate analysis. Neither final-test accuracy nor a new
transfer measurement is involved. [Frozen protocol](https://github.com/XuanyuYang223/neurips/blob/74f0de2017115f06f11b7285366b66e7e8431d30/configs/property_task_geometry.toml),
[feature extraction](https://github.com/XuanyuYang223/neurips/blob/74f0de2017115f06f11b7285366b66e7e8431d30/src/neurips_permutations/cka.py)

The eight designated relations pair descents/recoils, peaks/valleys,
fixed/anti-fixed points, exceedances/deficiencies, LIS/LDS, increasing/decreasing
runs, left-to-right maxima/minima, and global descents/components. The correct
inverse or complement transformation is compared with identity and the original
wrong-transform control. Components use the declared answer offset of one.

The integration exports only the necessary numerical evidence:

- [upstream_alignment_final_norm.csv](upstream_alignment_final_norm.csv): all
  72 relation × seed × condition records at final normalization.
- [upstream_alignment_by_seed.csv](upstream_alignment_by_seed.csv): 24 paired
  triplets and correct-minus-identity/correct-minus-wrong differences.
- [upstream_alignment_by_relation.csv](upstream_alignment_by_relation.csv): eight
  relation means and sample standard deviations over their three seeds.

Every exported row contains source repository, revision, path and both source
hashes. Aggregation from raw rows reproduced the saved upstream condition means
and standard deviations to an absolute tolerance of `1e-14`. Both paired
differences are positive in all 24 seed-level comparisons and all eight
seed-averaged relations. The original primary two-sided relation-level sign
probability is `2/2^8 = 0.0078125`. Seed rows within a relation are not 24
independent mathematical replications. Sample standard deviations are not
confidence intervals. A figure can show each relation's paired CKA contrast;
it must label this as **original validation-prefix representation evidence**,
separate from DMLR test accuracy. [Original results and limits](https://github.com/XuanyuYang223/neurips/blob/74f0de2017115f06f11b7285366b66e7e8431d30/results/property-task-geometry/cka/README.md)

Preserve the accompanying negative result: with four source tasks per model,
mean CKA for 0/1/2/4 direct correspondences was
0.2935/0.2752/0.2662/0.3648. Only one of 12 layout-by-seed curves was monotonic;
the paired endpoint sign test was 0.7744. The direct-relation specialist mean
was 0.1375 versus 0.0903 for other task pairs, while the same-task mean was
0.7096. Other pairs are outside the specified relation set, not mathematically
unrelated. None of these geometry comparisons proves algorithm sharing,
mathematical-formula use or beneficial transfer.

## Representation-pairing work in 2026TMLR

The TMLR continuation explicitly asks how pairing preference depends on
encoding, architecture, initialization, layer, metric and answer controls. Its
PermWorld record contains 64 true 20k specialist endpoints: 16 tasks × two
encodings × two architectures, in one paired data world. Its primary 32
answer-stratified, final-layer, equal-length ordinary-CKA cells have positive
trained pairing preference `Bt` in 26 cells and positive change from the
initial preference `Delta` in 18. Those counts were checked directly against
the saved primary table. They are descriptive coupled cells, not 32 independent
replications. The protocol records `exploratory=true` and `prior_results_seen=true`.
[Primary table](https://github.com/XuanyuYang223/2026TMLR/blob/d9a113d20d34db3c21cf69cf546cff4a0ab4904a/diagnostics/permworld_eight_relations_20261010/final_review/permworld/main32_final_cka_readout.csv)

This continuation distinguishes `Bt` from its initialization-adjusted increment
and retains correct/wrong score changes separately. Positive `Delta` can arise
because the wrong score declines more. The encoding/architecture comparison and
answer controls are real inherited contributions. They must not be described
as DMLR adaptation outcomes or evidence that larger CKA causes transfer.
GraphWorld and MatrixGroupWorld are additional upstream contexts; this DMLR
integration does not pool their results. The original MatrixGroupWorld record
retains 64 passed, two failed and six consistently undefined numerical checks,
and `scientific_analysis_complete=false`; publication packaging does not resolve
that caveat. [TMLR scope and interpretation](https://github.com/XuanyuYang223/2026TMLR/blob/d9a113d20d34db3c21cf69cf546cff4a0ab4904a/README.md)

## What the DMLR work adds

The DMLR pilot copies of `__init__.py`, `math_ops.py`, `models.py` and `passage.py`
were checked against the local pinned `neurips` checkout and are byte-identical.
The reused SDPA implementation matches the pinned TMLR source after newline
normalization. DMLR's additions are the separate paired transfer protocols,
development-only selection and independent-test separation, explicit joint
relations and orbit isolation, the unified task registry/generators and
calibration, and subsequent fixed-token task-count/budget experiments.

Platform v0.1 retains all 48 legacy endpoints and adds six registry endpoints,
for 54. Its counts distinguish 48 canonical mathematical templates, 62 declared
endpoint instances, 56 unary materialized instances, and one exposed input
encoding. The three focus families contain 16 templates and 24 declared
instances, including two constants at fixed length. These numbers do not imply
independent mathematical capabilities or six newly invented mathematical
algorithms. In particular, generalized cycle-length counting reuses an existing
mathematical primitive; aliases and known identities remain explicit.

The DMLR relation registry contains 74 records: 29 reused v5 certificates and
45 platform additions, with 72 labeled as proved from definitions and two as
using the RSK insertion theorem. Formula evidence, finite checks, single versus
joint queries, and transformed-input requirements are separate fields.
[upstream_resource_inventory.json](upstream_resource_inventory.json) makes these
counts and their source bindings available to the publication figures.

## Available publication and artifact material

The original `neurips/paper/` already contains figure/table guides, copy-ready
captions and LaTeX snippets, deterministic figure generators, PNG/SVG figures
and input/output hash manifests. Five permutation-section figures are documented.
Two probe panels explicitly come from an independent PermuFormer2 retraining
snapshot; they must not be silently attributed to the original checkpoint set.
This integration does not copy those panels or their images.

No tracked `.tex` manuscript or `.bib` bibliography was located in either
audited reference snapshot. TMLR supplies extensive HTML/text reports and
plot PDFs, not an identified complete manuscript in the inspected tree.
An integrated paper therefore needs a new draft, checked references and an
author-approved author/contribution list. A repository ownership name or a Git
commit is insufficient evidence for assigning authorship.

Read-only GitHub release API requests confirmed the following metadata at audit
time; **asset payloads were not downloaded or independently rehashed here**:

| Resource | Public release metadata | Assets |
|---|---|---:|
| Original datasets | [permutation-data-2026-09-04](https://github.com/XuanyuYang223/neurips/releases/tag/permutation-data-2026-09-04), non-draft release | 12 |
| Original checkpoints | [permutation-checkpoints-2026-09-04](https://github.com/XuanyuYang223/neurips/releases/tag/permutation-checkpoints-2026-09-04), non-draft release | 16 |
| Representation-pairing artifacts | [representation-selectivity-2026-10-11-caveat](https://github.com/XuanyuYang223/2026TMLR/releases/tag/representation-selectivity-2026-10-11-caveat), non-draft prerelease | 91 |

The original artifact guide specifies dataset/checkpoint checksums and extraction
instructions. TMLR publishes a logical-file mapping and its own limited public
readback audit. These are availability/provenance statements, not new scientific
replications. No LICENSE/COPYING/NOTICE or package license declaration was found
in the inspected upstream snapshot; no new upstream license or relicensing
claim is made here.

## Reproducing this audit

The integrated figures use the committed small exports and need no reference
checkout. Optional re-extraction from the exact pinned, clean local checkouts is:

```powershell
.venv/Scripts/python.exe publication/evidence/audit_original_permworld.py --remote
```

This command verifies pins, hashes source files and reads saved tables. The
optional `--remote` checks GitHub HEAD/release metadata only. It performs no
model work, downloads no weights and writes only the new publication evidence
directory. The audit does not change or upgrade any original scientific claim.
