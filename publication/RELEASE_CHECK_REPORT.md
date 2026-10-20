# DMLR release check

This report accompanies the paper-boundary and clean-installation review starting
from `7ef277d3f2cdd6faef0d2f7c1b844e3b0e1bcf9e`. The initial worktree was clean;
the exact state is retained in [START_STATE.json](../release_checks/START_STATE.json).
The working branch is `publication/permworld-dmlr-release-check`.

## Paper boundary

The DMLR manuscript removes the new TMLR encoding, architecture, initialization
and answer-control results. Their model/cell counts and success counts are absent
from its title, abstract, contributions, result sections, appendix tables, figures
and default reproduction pipeline. The original workshop alignment study appears
only as explicitly cited background. Optional provenance records remain archived;
the new TMLR repository and all historical experiment records are unchanged.

The three retained contributions are a relation-annotated exact data platform;
a paired transfer-evaluation workflow with explicit composition and token-budget
accounting; and conditional findings supported by the existing experiments.
Connecting representation geometry to transfer is not a DMLR contribution.

## Installation findings and repair

The first fresh clone of the starting commit failed installation because it had
neither `pyproject.toml` nor `setup.py`. The clean Python 3.14 virtual environment
initially contained only pip. The actual failed command and output remain in
[attempt00](../release_checks/logs/attempt00/RUN.json).

The repair adds non-editable wheel packaging, a NumPy core dependency and a
Matplotlib publication extra, while retaining the original platform/vendor source
bytes. Mathematical catalogs and certificate resources are explicitly packaged.
The installed `permworld-data` CLI and `permworld_release` API require explicit
disposable-data scope and reject historical-exclusion requests: the wheel does
not include the historical input pools and cannot certify their exclusion.

## Acceptance status

**Passed.** The final run uses Windows 11 build 26200, CPython 3.14.6, NumPy 2.5.4
and Matplotlib 3.11.2. Its new virtual environment initially contained only
pip 26.1.2. System site packages were disabled; Python isolated mode disabled
user packages and inherited import paths. The noneditable wheel was installed
inside that environment and imported from its `Lib/site-packages`, while the
working directory was outside the clone. Neither `reference/`, an old `.venv`,
nor any `.pt` checkpoint existed in the acceptance checkout.

The accepted source was transported as a fresh `git clone --no-local` of the
starting commit plus an exact staged binary patch applied with `git apply --index`.
This allowed acceptance before the final commit. No untracked source-worktree
files were copied and no runtime import used the original workspace. The final
branch can be cloned directly without this pre-commit patch step.
[ACCEPTED_SOURCE.json](../release_checks/ACCEPTED_SOURCE.json) records the candidate
tree, patch digest and 28 executed/package source hashes; the final source matches
that accepted snapshot byte for byte. Subsequent additions are collected evidence,
reports, constraints and derived publication manifests.

| Check | Actual result |
| --- | --- |
| Installation and dependency isolation | Noneditable wheel install and pip check pass; no Torch/GPU runtime installed or imported |
| Exact and parameterized tasks | 12 fixed fixtures, including scalar, structured, binary and parameterized tasks; invalid inputs rejected |
| Full-answer interface | Prompt/answer boundaries, EOS, structured round trips, integer 220 and integer 155117520 verified without truncation |
| Packaged registry/resources | 54 endpoint definitions and 74 relation records load; historical certificates hash-reused, not re-enumerated |
| Disposable generation | Two repeats at n=8,10; 64 objects and 384 labels; zero duplicate or within-fixture cross-role/repeat orbit overlaps |
| Reader and data quality | 16 parameterized training samples read; 72 distribution rows; deterministic replay and corruption rejection pass |
| Installed CLI | Help, generation, verification and sample reading all pass |
| DMLR arithmetic | Eight stage-level checks and 19 claims recomputed from released tables |
| Budget interactions | Four means/sample SDs match archived results and plotted summaries within absolute tolerance 1e-12 |
| Figures | Two main figures and one supplement rebuilt as PNG/SVG/PDF; all 208 plotted-unit and 72 summary rows match exactly |
| Source-only archive | 650 deliverables verified: 646 exact bytes and four explicit prior English revisions; 151 local checkpoints skipped |
| Optional provenance absent | Eight upstream audit/export files absent in the isolated clone; default verifier and figures still pass; all eight restored with identical hashes |
| Preservation and language | Candidate bytes unchanged by commands; no edits to scientific experiment/platform paths; English verification passes |

The disposable fixture explicitly disables historical exclusion and cannot be
used as a research split. The release wrapper rejects missing/true historical
policy when the wheel lacks the required pools. Its verification result reports
historical overlap as unknown (`null`), not a measured zero. The preserved frozen
low-level fixture audit has its original generic fields; interpret them with the
explicit configuration and the wrapper's scoped result.

The final acceptance comprises 21 logged commands with a summed command time of
117.70 seconds, including clone and installation. This is an observed local cost,
not a portable benchmark. Detailed records are in [attempt03/RUN.json](../release_checks/logs/attempt03/RUN.json),
[interface evidence](../release_checks/artifacts/attempt03/ACCEPTANCE.json),
[fixture configuration](../release_checks/artifacts/attempt03/demo_config.json),
[the optional-provenance test](../release_checks/artifacts/attempt03/OPTIONAL_PROVENANCE_ABSENT.json),
and [installed versions](../release_checks/logs/attempt03/07_installed_packages.log).
The [CPU dependency constraints](../release_checks/constraints-cpu-py314.txt)
contain no machine-specific source URI.

## Failures, repairs and fresh reruns

| Attempt | Outcome | Repair or follow-up |
| --- | --- | --- |
| [00](../release_checks/logs/attempt00/RUN.json) | Original commit cannot install: no setup.py/pyproject.toml | Added wheel metadata, dependencies and packaged relative-path resources |
| [01](../release_checks/logs/attempt01/RUN.json) | Candidate patch fails before installation on a PDF | Host Git configured PDF text conversion; export now disables textconv/external diff and emits a full binary patch |
| [02](../release_checks/logs/attempt02/RUN.json) | Complete clean installation and 20-command acceptance pass | Retained; followed by stricter optional-provenance and unknown-history checks |
| [03](../release_checks/logs/attempt03/RUN.json) | New clone and environment; all 21 final commands pass | Accepted source hashes recorded; no extra model work |

The second repair concerns candidate transport, not a scientific result or model
failure. All attempts and command output are retained. The final rerun did not
reuse an earlier virtual environment or inherit its installed packages.

## Budget-interaction arithmetic

For each of three paired repeats, compute
`I = (k16-k2 at 32M) - (k16-k2 at 4M)` from the integer correct/count fields in
the original selected test rows. Then compute the mean and sample SD of those
three interactions. The two chains and shared baselines do not add independent
repeats. This is a recalculation of existing results, not new test evaluation.

| Target | Chain | Interaction mean (pp) | Sample SD (pp) |
| --- | --- | ---: | ---: |
| LIS length, primary | alpha | 17.936198 | 7.894065 |
| LIS length, primary | beta | 6.542969 | 5.959432 |
| fixed_points, auxiliary | alpha | 41.406250 | 3.272577 |
| fixed_points, auxiliary | beta | -9.082031 | 11.553193 |

[The full check](../release_checks/artifacts/attempt03/budget_interactions.json)
retains 48 raw condition rows, 12 per-repeat contrasts, seed metadata and source
hashes. Existing negative and uncertain results are unchanged.

## Reproduction and limits

Follow [RELEASE.md](RELEASE.md) to install the committed branch with
`python -m pip install ".[publication]"` in a new environment. To request the
versions used here, add
`--constraint release_checks/constraints-cpu-py314.txt`. The full acceptance runner
creates its own fresh clone/environment and refuses existing output directories.
Logs retain exact command arguments, working directories, exit codes and times.

No GPU training, neural forward, checkpoint loading, model-selection change or
model test-set evaluation was performed. Label checks on disposable fixtures and
arithmetic on archived tables do not constitute a model replication. No historical
experiment file, frozen configuration, raw result or scientific manifest was
rewritten. The eight optional upstream audit files also remain unchanged; only
publication-derived figure/source/release manifests are refreshed.

## Remaining release work

Author/contributor approval, code and data licensing, archival DOI/release policy
and a checkpoint-distribution decision remain open. The clean CPU installation
does not establish reproducibility of historical GPU training or portability to
every Python version and operating system.
