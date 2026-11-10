"""Write canonical completion documents after all frozen analysis and audits pass.

This post-freeze documentation generator reads existing results only. It does
not change the scientific protocol, frozen analysis outputs, checkpoint choice,
model parameters, predictions or statistical calculations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONDITIONS = ["P", "D", "T", "U", "A"]
UNITS = [f"R{i}" for i in range(1, 9)]
SIZES = [256, 512, 1024]
CANDIDATES = dict(P="peaks", D="deficiencies", T="anti_fixed_points", U="right_to_left_maxima")
COMMON = ["descents", "successions", "left_to_right_maxima"]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(rows, field, scale=1):
    assert len(rows) == 8
    return scale * statistics.mean(float(r[field]) for r in rows)


def stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(str(v) for v in row) + " |" for row in rows]])


def status_gates(artifacts):
    training = read_json(artifacts / "FINAL_TRAINING_AUDIT.json")
    independent = read_json(artifacts / "analysis/INDEPENDENT_RECOMPUTE.json")
    testing = read_json(artifacts / "TEST_COMPLETE.json")
    analysis = read_json(artifacts / "analysis/ANALYSIS_MANIFEST.json")
    loss_figures = read_json(artifacts / "loss_figures/LOSS_FIGURE_MANIFEST.json")
    assert training["status"] == "passed"
    assert (training["source_models"], training["adaptation_candidates"], training["selected_results"]) == (32, 360, 120)
    assert independent["status"] == "pass" and independent["independent_units"] == 8
    assert (independent["evaluated_rows"], independent["raw_prediction_rows"]) == (240, 491520)
    assert testing["status"] == "complete" and (testing["evaluations"], testing["selected_results"]) == (240, 120)
    assert analysis["status"] == "complete" and (analysis["units"], analysis["evaluated_rows"]) == (8, 240)
    assert loss_figures["status"] == "complete" and loss_figures["independent_units"] == 8
    for name, expected in loss_figures["outputs"].items():
        assert sha(artifacts / "loss_figures" / name) == expected
    for name, expected in loss_figures["inputs"].items():
        assert sha(artifacts / name) == expected
    result_sha = sha(artifacts / "results_test.csv")
    assert testing["results_sha256"] == independent["sources"]["results_test.csv"] == analysis["inputs"]["results_test.csv"] == result_sha
    for name, expected in analysis["outputs"].items():
        assert sha(artifacts / "analysis" / name) == expected, f"Derived frozen output changed: {name}"
    protocol = read_json(artifacts / "FROZEN_PROTOCOL.json")
    for name, expected in protocol["code_sha256"].items():
        assert sha(ROOT / name) == expected, f"Frozen code changed: {name}"
    return training, independent, testing, analysis


def summarize_sources(rows):
    endpoint = defaultdict(list)
    for row in rows:
        if int(row["step"]) == 4800:
            endpoint[row["condition"], row["task"], row["split"]].append(row)
    for rows_ in endpoint.values():
        assert len(rows_) == 8 and {r["repeat_id"] for r in rows_} == set(UNITS)
    assert len(endpoint) == 32
    candidates = []
    for condition, task in CANDIDATES.items():
        train, dev = endpoint[condition, task, "train"], endpoint[condition, task, "dev"]
        candidates.append([condition, task, f"{mean(train, 'accuracy', 100):.2f}",
            f"{mean(dev, 'accuracy', 100):.2f}",
            f"{statistics.stdev(float(r['accuracy'])*100 for r in dev):.2f}",
            f"{mean(dev, 'answer_ce'):.4f}", f"{mean(dev, 'majority_accuracy', 100):.2f}"])
    common = []
    for task in COMMON:
        for condition in CANDIDATES:
            train, dev = endpoint[condition, task, "train"], endpoint[condition, task, "dev"]
            common.append([task, condition, f"{mean(train, 'accuracy', 100):.2f}",
                f"{mean(dev, 'accuracy', 100):.2f}", f"{mean(dev, 'majority_accuracy', 100):.2f}",
                f"{mean(dev, 'answer_ce'):.4f}"])
    paired = []
    for task in COMMON:
        for left, right in [("D", "U"), ("D", "P"), ("U", "P"), ("D", "T")]:
            by_condition = {c: {r["repeat_id"]: r for r in endpoint[c, task, "dev"]} for c in (left, right)}
            differences = [100*(float(by_condition[left][r]["accuracy"])-float(by_condition[right][r]["accuracy"])) for r in UNITS]
            paired.append([task, f"{left}-{right}", f"{statistics.mean(differences):+.2f}",
                f"{statistics.stdev(differences):.2f}", f"[{min(differences):+.2f}, {max(differences):+.2f}]"])
    return candidates, common, paired


def costs(artifacts, accounting, result_rows):
    summaries = []
    for stage, count in [("pretrain", 32), ("adapt", 360)]:
        rows = [r for r in accounting if r["stage"] == stage]
        assert len(rows) == count and len({r["run_id"] for r in rows}) == count
        summaries.append(dict(stage=stage, runs=count,
            **{field: sum(int(r[field]) for r in rows) for field in
               ["updates", "sample_exposures", "prompt_tokens", "training_sequence_tokens", "supervised_tokens"]},
            elapsed_seconds=sum(float(r["elapsed_seconds"]) for r in rows)))
    workers = [(p, read_json(p)) for p in sorted(artifacts.glob("ADAPT_WORKER_*.json"))]
    assert workers, "Completed worker summaries are required for actual adaptation wall time."
    units = [unit for _, worker in workers for unit in worker["units"]]
    assert len(units) == len(set(units)) == 8 and set(units) == set(UNITS)
    assert all(w["status"] == "complete" for _, w in workers)
    assert sum(w["candidate_runs"] for _, w in workers) == 360
    first = min(stamp(w["started_at_utc"]) for _, w in workers)
    last = max(stamp(w["finished_at_utc"]) for _, w in workers)
    worker_wall = (last-first).total_seconds()
    test_elapsed = sum(float(row["elapsed_seconds"]) for row in result_rows)
    assert worker_wall > 0 and test_elapsed > 0
    return summaries, workers, worker_wall, test_elapsed


def document_text(artifacts):
    status_gates(artifacts)
    analysis_dir = artifacts / "analysis"
    primary = read_json(analysis_dir / "primary_summary.json")
    assert primary["status"] == "complete" and primary["n"] == 8 and primary["df"] == 7
    results = read_csv(artifacts / "results_test.csv")
    assert len(results) == 240
    by_key = {(r["repeat_id"], r["condition"], int(r["target_samples"]), r["checkpoint_kind"]): r for r in results}
    assert len(by_key) == 240
    summary = read_csv(analysis_dir / "paired_summary.csv")
    scores = read_csv(analysis_dir / "scores_per_repeat.csv")
    score_key = {(r["repeat_id"], r["condition"], r["checkpoint_kind"]): float(r["S_pct"])
                 for r in scores if r["metric"] == "complete"}
    low, high = primary["ci95_low_pp"], primary["ci95_high_pp"]
    if low > 0:
        conclusion = ("The primary interval is entirely positive. This supports additional transfer value of deficiencies "
                      "over the specified U task in this background, budget and adaptation procedure.")
    elif high < 0:
        conclusion = ("The primary interval is entirely negative. The prediction that deficiencies would outperform "
                      "the specified U task is not supported; U performs better in this fixed comparison.")
    else:
        conclusion = ("The primary interval includes zero. Additional transfer value of deficiencies over U remains "
                      "uncertain; this does not prove that the benefit is zero.")
    unit_table = table(["Unit", "S_P (%)", "S_D (%)", "S_T (%)", "S_U (%)", "S_A (%)", "Primary D-U (pp)"],
        [[r, *[f"{score_key[r,c,'selected']:.4f}" for c in CONDITIONS],
          f"{primary['differences_by_repeat_pp'][r]:+.4f}"] for r in UNITS])
    secondary = [r for r in summary if r["metric"] == "complete" and r["checkpoint_kind"] == "selected" and r["scope"] == "area"]
    by_contrast = {row["contrast"]: row for row in secondary}
    gain_d, gain_u = by_contrast["D_minus_A"], by_contrast["U_minus_A"]
    d_a, u_a = float(gain_d["mean_pp"]), float(gain_u["mean_pp"])
    assert abs((d_a-u_a)-primary["mean_pp"]) < 1e-12
    by_size = {n: next(row for row in summary if row["metric"] == "complete" and
               row["checkpoint_kind"] == "selected" and row["scope"] == f"n{n}" and
               row["contrast"] == "D_minus_U") for n in SIZES}
    size_contrasts = " / ".join(f"{float(by_size[n]['mean_pp']):+.4f}" for n in SIZES)
    negative_counts = ", ".join(f"{condition}: {by_contrast[f'{condition}_minus_A']['negative_units']}/8"
                                for condition in ("P", "T", "U"))
    secondary_table = table(["Area contrast", "Role", "Mean (pp)", "Sample SD", "Unit range (pp)", "Positive units"],
        [[r["contrast"], "primary" if r["analysis_role"] == "primary" else "secondary",
          f"{float(r['mean_pp']):+.4f}", f"{float(r['sample_sd_pp']):.4f}",
          f"[{float(r['min_pp']):+.4f}, {float(r['max_pp']):+.4f}]", f"{r['positive_units']}/8"] for r in secondary])
    target_rows = []
    for condition in CONDITIONS:
        for size in SIZES:
            rows = [by_key[r,condition,size,"selected"] for r in UNITS]
            endpoint = [by_key[r,condition,size,"endpoint"] for r in UNITS]
            target_rows.append([condition, size, f"{mean(rows, 'accuracy', 100):.2f}",
                f"{statistics.stdev(float(r['accuracy'])*100 for r in rows):.2f}",
                f"{mean(endpoint, 'accuracy', 100):.2f}", f"{mean(rows, 'answer_ce'):.4f}",
                f"{mean(rows, 'majority_accuracy', 100):.2f}"])
    target_table = table(["Route", "Train examples", "Selected accuracy (%)", "Sample SD", "Endpoint accuracy (%)",
                          "Selected answer CE", "Train-derived majority (%)"], target_rows)
    candidates, common, paired = summarize_sources(read_csv(artifacts / "source_curves.csv"))
    candidate_table = table(["Condition", "Candidate", "Train accuracy (%)", "Dev accuracy (%)", "Dev sample SD",
                             "Dev answer CE", "Dev majority (%)"], candidates)
    common_table = table(["Common task", "Condition", "Train accuracy (%)", "Dev accuracy (%)", "Dev majority (%)", "Dev answer CE"], common)
    common_pair_table = table(["Common task", "Paired dev contrast", "Mean (pp)", "Sample SD", "Unit range (pp)"], paired)
    matching = read_csv(artifacts / "DATA_TASK_MATCHING.csv")
    matching_rows = []
    for role in ["source_train", "source_dev"]:
        rows = [r for r in matching if r["role"] == role and r["task_a"] == "deficiencies" and r["task_b"] == "right_to_left_maxima"]
        assert len(rows) == 8 and {r["repeat_id"] for r in rows} == set(UNITS)
        tv = [float(r["answer_distribution_total_variation"]) for r in rows]
        matching_rows.append([role, f"{statistics.mean(tv):.4f} [{min(tv):.4f}, {max(tv):.4f}]",
            f"{mean(rows,'mean_a'):.4f}", f"{mean(rows,'mean_b'):.4f}",
            f"{mean(rows,'entropy_a_bits'):.4f}", f"{mean(rows,'entropy_b_bits'):.4f}",
            f"{mean(rows,'same_input_answer_agreement',100):.2f}"])
    matching_table = table(["Pool", "D/U answer-distribution TV mean [range]", "D mean answer", "U mean answer",
        "D entropy (bits)", "U entropy (bits)", "Same-input label agreement (%)"], matching_rows)
    budget, workers, worker_wall, test_elapsed = costs(artifacts, read_csv(artifacts / "accounting.csv"), results)
    budget_table = table(["Stage", "Runs", "Updates", "Sample exposures", "Prompt tokens", "Sequence tokens", "Supervised tokens", "Sum of run elapsed seconds"],
        [[r["stage"],r["runs"],r["updates"],r["sample_exposures"],r["prompt_tokens"],r["training_sequence_tokens"],r["supervised_tokens"],
          f"{r['elapsed_seconds']:.2f}"] for r in budget])
    worker_table = table(["Worker units", "Start UTC", "Finish UTC", "Recorded seconds"],
        [[", ".join(w["units"]),w["started_at_utc"],w["finished_at_utc"],f"{w['elapsed_seconds']:.2f}"] for _,w in workers])
    report = f"""# Fixed-points deficiencies confirmation: completed results

All eight new paired units completed the fixed scope: **32 source models,
360 adaptation learning-rate candidates, 120 development-selected results,
240 selected/endpoint evaluations and 491,520 stored per-example predictions**.
The last count reuses eight test pools across conditions and checkpoints; there
are 16,384 distinct test inputs, not 491,520 independent test examples. The final
training audit and independent prediction/statistics reconstruction passed.
This is an independent case within the DMLR relationship-annotated platform and
composition/budget program; no historical conclusion is replaced.

## Sole primary comparison

The score is `S = .25*A256 + .5*A512 + .25*A1024`, the normalized trapezoid area
over log2 target training size, using complete generated answer plus EOS exact
match. Within each unit, D is compared with U before summarization.

**S_D-S_U = {primary['mean_pp']:+.4f} pp**, sample SD {primary['sample_sd_pp']:.4f} pp,
range [{primary['min_pp']:+.4f}, {primary['max_pp']:+.4f}] pp. The paired 95% t
interval is **[{low:+.4f}, {high:+.4f}] pp**, df=7. There are
{primary['positive_units']} positive, {primary['negative_units']} negative and
{primary['zero_units']} zero unit differences. {conclusion}

{unit_table}

![Eight paired primary differences](artifacts/analysis/primary_paired_difference.png)

The interval uses eight differences and the sample SD with denominator seven.
Its approximate-normal, approximately independent unit assumptions are strong
with eight units. Global orbit rejection conditions later pools on earlier pools
in a finite permutation space, so fresh RNG seeds do not establish exact IID
units or guaranteed interval coverage. Eight units are a fixed compute budget,
not a power guarantee. Learning rates, sample sizes, test samples, shared source
models and the shared A baselines do not increase the number of repeats.

## Secondary results and endpoint diagnostics

{secondary_table}

The primary **{primary['mean_pp']:+.4f} pp is a comparison with U**, not a gain
of that size over random initialization. D's mean gain over shared A is
**{d_a:+.4f} pp**, sample SD {float(gain_d['sample_sd_pp']):.4f}, with
{gain_d['positive_units']}/8 positive units and {gain_d['zero_units']} tie.
The negative area-gain counts relative to A are {negative_counts}.
The arithmetic identity `D-U = (D-A) - (U-A)` gives
`{primary['mean_pp']:+.4f} = {d_a:+.4f} - ({u_a:+.4f})` pp for the means.
Thus the primary separation includes U's negative transfer relative to A as
well as D's smaller gain over A. This is a descriptive decomposition, not a
mechanistic or causal mediation result.

The prespecified pointwise D-U means are **{size_contrasts} pp** at
256 / 512 / 1024 examples, respectively. The observed advantage narrows as
target training size increases in this fixed range; no sample size is selected
after testing, and the weighted-area contrast remains the sole primary result.

These secondary contrasts cannot replace the primary comparison. In particular,
an improvement of both D and U over P without D outperforming U does not isolate
a benefit specific to deficiencies. D-T compares two specified sources, not
general classes of shared-subcomputation versus transformed-input relationships.
All ten condition-pair differences at each size are retained in
[paired_differences.csv](artifacts/analysis/paired_differences.csv) and
[paired_summary.csv](artifacts/analysis/paired_summary.csv), including negative
and uncertain results. No favorable subset is selected for the conclusion.

{target_table}

The endpoint is update 800 of the learning-rate candidate selected on
development data, not a second test-selected learning rate. Answer CE is the
teacher-forced answer-token loss; EOS and mean two-token losses remain in the
raw table. The legacy first-answer-token accuracy is separately marked
`legacy_answer_token` in derived tables and is never pooled with the primary
full-answer/EOS metric. Target majority labels come from the corresponding
training subset, with the smallest-answer tie rule, and are applied unchanged
to development and test inputs. All data have n=10.

![Target curves and paired gains over A](artifacts/analysis/target_learning_curves.png)

![Selected-LR training and development curves](artifacts/analysis/adaptation_learning_curves.png)

The separate [loss-curve figures](artifacts/loss_figures/README.md) show saved
answer-token CE and mean answer-plus-EOS CE, with thin per-unit trajectories and
thick eight-unit means. The target plots display the full trajectory of the
development-selected LR candidate for each route/size/unit, not just its selected
checkpoint. This is a selection-dependent display; all 360 raw LR candidate
curves remain available, and the plots perform no new selection or evaluation.

![Selected-LR target answer losses](artifacts/loss_figures/adaptation_answer_ce.png)

## Frozen design and isolation

The common source slots are `descents`, `successions`, `left_to_right_maxima`;
the fourth is P=`peaks`, D=`deficiencies`, T=`anti_fixed_points`, or
U=`right_to_left_maxima`. A starts from the same unit's random initialization.
P/D are the old V5 C **task sets**, with new slot order `[1,0,3,2]` relative to
that history. T is a new composition, not an old B replication.
[HYPOTHESIS_ORIGIN.md](HYPOTHESIS_ORIGIN.md) preserves V5's failed fixed-points
confirmation and uncertain longest-decreasing-run result, along with task-count
V1/V2 negative outcomes and protocol differences. Historical observations are
not pooled with these eight units.

The model and training primitives inherit V5: legacy one-line n=10 input,
188-token vocabulary, four Transformer layers, width 256, eight heads,
4800 source updates with batch 64 and 16 samples per task. Every route and
target size independently starts from its own source endpoint or paired step-0
state. Adaptation loads all source parameters, restores only the fixed_points
tied embedding/output row to its unit's step-0 value, and creates a fresh
optimizer. All parameters adapt for 800 updates at each of three predetermined
learning rates. No frozen prefixes, layer probes or state exchanges are used.
The selected development ordering is highest complete-answer accuracy, lowest
answer CE, earliest positive update, then fixed LR order.

Each unit has 8192 source-training, 1024 source-development, 1024 maximum
target-training, 1024 target-development and 2048 test inputs. Target-training
sizes 256/512/1024 are nested prefixes of one prespecified shuffled pool.
**The additional 1024 labeled target-development examples per unit are target
supervision.** The 256-example condition does not use only 256 target labels in
total. All conditions share paired pools and adaptation streams; all 120
development selections freeze before new test generation. Complete inverse,
reverse and complement orbits isolate roles, units and historical DMLR inputs.
Finite-space rejection and capacity limits remain documented in the data audit.

## Source learning and remaining comparator differences

The candidate table uses all eight unit endpoints at update 4800. Candidate
accuracy differences compare different tasks and distributions, so they do not
constitute a matched difficulty control.

{candidate_table}

{common_table}

The common descents and successions tasks have substantially incomplete training
fit as well as imperfect development performance under this budget. A small
train/development gap therefore does not establish that these tasks were fully
learned. Prefix maxima are much more accurate. No task or unit is removed and
no budget is extended based on source learning.

{common_pair_table}

These source contrasts are paired within unit before averaging. They describe
composition-associated changes; they are not mediator interventions and do not
show that a particular source-learning change caused a target gain or loss.
The [pre-test source-learning note](artifacts/SOURCE_LEARNING_NOTE.md) retains
the interpretation written without target/test outcomes, with complete
[source curves](artifacts/source_curves.csv) and [endpoint summaries](artifacts/analysis/source_endpoint_summary.csv).

{matching_table}

Total variation is half the summed absolute difference of empirical answer
probabilities on the shared numeric support. D/U are not matched in answer
distribution, difficulty or redundancy. U has a reversal relation to the
common prefix-maxima task; it is not an unrelated control. The
[matching table](artifacts/DATA_TASK_MATCHING.csv) retains every unit and other
candidate comparisons. Adaptation retains learned numeric answer embedding/
output rows and all other non-target-task parameters. Such output-parameter
history, label distributions, shared computations, optimization and broader
representations are possible contributors; this experiment does not separate
their roles or establish any as a mechanism.

![Source training and development curves](artifacts/analysis/source_learning_curves.png)

![Source answer-token loss curves](artifacts/loss_figures/source_answer_ce.png)

The [mean answer-plus-EOS source losses](artifacts/loss_figures/source_two_token_ce.png)
and [target losses](artifacts/loss_figures/adaptation_two_token_ce.png) remain
separate from exact-match accuracy. All loss panels explicitly use n=10.

## Mathematics is distinct from learned computation

`FP(p)+E(p)+D(p)=n` and `E(p)=D(inverse(p))` are proved answer identities.
D(p) alone does not provide E(p); the registered deficiencies recovery formula
needs answers on both p and inverse(p). Anti-fixed points has complement and
reversal relationships to FP and can also use positional comparison. All four
source answer vectors on the same input have recorded n=10 insufficiency
witnesses; allowing all registered I/R/C queries makes D and T sufficient in
the specified scalar-answer query model. P/U remain insufficient in that query
model, not mathematically independent of FP. Missing relation annotations do
not prove absence of other relationships.

The training provides no explicit paired transformed queries, formula
supervision or local position labels. Mathematical derivability, source-task
competence and actual transfer benefit are separate claims. Neither a positive
primary result nor D outperforming T would prove use of `FP=n-E-D` or an
independently transplantable lemma module. U helps compare specified source
replacements but does not fully separate adding a task from removing peaks.
See [RELATION_TABLE.csv](RELATION_TABLE.csv) and
[the full-composition audit](preflight_audit/JOINT_DERIVABILITY.csv).

## Actual budgets, time and engineering work

{budget_table}

Every source task has 76,800 sample exposures, 1,996,800 prompt tokens,
2,150,400 complete training-sequence tokens and 153,600 supervised tokens.
There are 26 prompt tokens plus answer and EOS in every 28-token sample, so the
paired source counts here also match source sequence and supervised tokens.
[Per-task exposures](artifacts/exposures.csv) preserve actual unique input counts;
repeated exposures and shared pools are not multiplied into independent data.

{worker_table}

The actual adaptation worker envelope is **{worker_wall:.2f} seconds
({worker_wall/60:.2f} minutes)** from the earliest worker start to the latest
worker finish. Per-run elapsed intervals include in-run evaluations and
checkpoint/table work. The parallel adaptation intervals overlap and contain
resource contention: their sum is accumulated run elapsed time, **not dedicated
GPU-hours or wall-clock duration**. The serial source-run sum is
{budget[0]['elapsed_seconds']:.2f} seconds; source setup/process overhead is not
claimed to be included. Summed final-test record elapsed time is
**{test_elapsed:.2f} seconds**, reported separately. Stage envelopes, per-run
sums and test timers must not be added as if they were disjoint measurements.

Engineering smoke, wiring and scheduling checks used **248 additional optimizer
updates, 444,416 nonpadding sequence tokens and 31,744 supervised tokens**.
These are excluded from scientific budgets and are not new repeats. Engineering
timing is only partially recorded; no complete engineering wall-time or GPU-hour
total is claimed. [ENGINEERING_LOG.md](ENGINEERING_LOG.md) and
[EXECUTION_LOG.md](EXECUTION_LOG.md) retain those checks, environment and commands.

## Acceptance, reproducibility and stopping point

[FINAL_TRAINING_AUDIT.json](artifacts/FINAL_TRAINING_AUDIT.json) replays source
and adaptation streams, validates exposures and initialization/loading/reset
bindings, and independently reconstructs development choices. The raw-prediction
[independent recomputation](artifacts/analysis/INDEPENDENT_RECOMPUTE.json) checks
all 491,520 records against direct fixed-point labels, test input order,
training-derived majority labels and frozen checkpoint metadata. It reconstructs
integer accuracy counts, all weighted scores and paired moments, and the primary
t interval through independent numerical quadrature. These audits add no model
test forwards or training. All 240 evaluation rows are in
[results_test.csv](artifacts/results_test.csv), with [predictions](artifacts/predictions)
and [checkpoint choices](artifacts/SELECTIONS_FROZEN.json).

This canonical report is a documentation supplement produced after the frozen
analysis. It clarifies runtime language and adds source/context summaries
without editing `analysis.py`, its original outputs or any scientific record.
[REPORT_AMENDMENT.json](REPORT_AMENDMENT.json) binds this generator, inputs,
previous document bytes and final document hashes.
The [canonical DMLR paragraph](PAPER_RESULT_DRAFT.md) adds explicit metric and
scratch-baseline context while preserving the
[original frozen-analysis draft](artifacts/analysis/PAPER_RESULT_DRAFT.md).

Only one target, background, architecture, length and budget were tested. This
is a finite three-size comparison, not a scaling law or a general task-family
ranking. Negative transfer, incomplete learning and uncertainty remain in the
complete outputs. The scope is complete and stops here: no additional target,
composition, source substitution, architecture, budget or mechanistic experiment
is authorized by these results. Files and checkpoints remain local; no upload
or publication was performed by this experiment.
"""
    readme = f"""# Fixed-points deficiencies confirmation

**Complete:** eight new paired units, 32 source models, 360 adaptation
learning-rate candidates, 120 development-selected results, 240 selected/endpoint
evaluations and 491,520 per-example prediction records. All eight units and
negative/uncertain outcomes are retained. There are 16,384 distinct test inputs;
reusing them across conditions does not increase the repeat count.

The sole primary contrast is **S_D-S_U = {primary['mean_pp']:+.4f} pp**
(sample SD {primary['sample_sd_pp']:.4f}; 95% paired-t interval
**[{low:+.4f}, {high:+.4f}] pp**, df=7), where
`S=.25*A256+.5*A512+.25*A1024` uses complete generated answer plus EOS.
{conclusion}

For context, the secondary gain over scratch is **S_D-S_A={d_a:+.4f} pp**
({gain_d['positive_units']}/8 positive, {gain_d['zero_units']} tie).
The negative area-gain counts relative to A are {negative_counts}.
D-U narrows from **{size_contrasts} pp** across 256/512/1024 examples.
The primary comparison is therefore not an equally large gain over scratch.

![Primary paired result](artifacts/analysis/primary_paired_difference.png)

This case asks whether deficiencies improves fixed_points transfer more than
the prespecified right_to_left_maxima comparator in the common background
descents/successions/left_to_right_maxima. The V5 one-line n=10 Transformer and
4800-update source budget are fixed. P=peaks and T=anti_fixed_points are
additional specified comparisons; A is the paired shared scratch baseline.
Adaptation uses 800 updates and three frozen learning rates at each nested
256/512/1024 training size, with **1024 additional labeled development examples
per unit**. This is not a claim of only 256 total target labels.

## Review entry points

| Artifact | Purpose |
| --- | --- |
| [RESULT_REPORT.md](RESULT_REPORT.md) | Canonical English result, uncertainty, source-learning comparisons and actual costs |
| [Raw results](artifacts/results_test.csv), [predictions](artifacts/predictions) | All selected and endpoint metrics and per-example outputs |
| [Per-unit scores](artifacts/analysis/scores_per_repeat.csv), [paired differences](artifacts/analysis/paired_differences.csv), [summaries](artifacts/analysis/paired_summary.csv) | Eight-unit comparisons, secondary contrasts and separate legacy metric |
| [PROTOCOL.md](PROTOCOL.md), [frozen binding](artifacts/FROZEN_PROTOCOL.json), [statistics plan](stats_plan.json) | Fixed tasks, seeds, budgets, selection and stopping rules |
| [Mathematical relations](RELATION_TABLE.csv), [query audit](preflight_audit/JOINT_DERIVABILITY.csv) | Proofs, query requirements, available computations and unproved mechanisms |
| [Source-learning note](artifacts/SOURCE_LEARNING_NOTE.md), [curves](artifacts/source_curves.csv), [exposures](artifacts/exposures.csv) | All source tasks, incomplete learning and actual exposure |
| [Loss trajectory figures](artifacts/loss_figures/README.md) | Source and selected-LR target train/dev answer CE and mean answer-plus-EOS CE; n=10, eight units |
| [Historical motivation](HYPOTHESIS_ORIGIN.md) | Preserved V5 failed confirmation and task-count negative/conditional findings |
| [Training audit](artifacts/FINAL_TRAINING_AUDIT.json), [independent recount](artifacts/analysis/INDEPENDENT_RECOMPUTE.json), [test completion](artifacts/TEST_COMPLETE.json) | Passed binding, pairing, selection and statistics checks |
| [Canonical paper paragraph](PAPER_RESULT_DRAFT.md), [original frozen-analysis draft](artifacts/analysis/PAPER_RESULT_DRAFT.md), [claims and limits](artifacts/analysis/CLAIM_EVIDENCE_LIMITS.csv) | Bounded DMLR integration material; original draft retained |
| [Execution commands](EXECUTION_LOG.md), [engineering log](ENGINEERING_LOG.md), [documentation amendment](REPORT_AMENDMENT.json) | Environment, actual commands, timing limits and documentation provenance |

## Reproduce the final readout

After the complete frozen training, selection and test flow in
[EXECUTION_LOG.md](EXECUTION_LOG.md), run from the repository root:

```powershell
.venv/Scripts/python.exe experiments/fixed_points_deficiencies_v1/train.py aggregate
.venv/Scripts/python.exe experiments/fixed_points_deficiencies_v1/final_check.py
.venv/Scripts/python.exe experiments/fixed_points_deficiencies_v1/analysis.py
.venv/Scripts/python.exe experiments/fixed_points_deficiencies_v1/independent_recompute.py
.venv/Scripts/python.exe experiments/fixed_points_deficiencies_v1/plot_losses.py
.venv/Scripts/python.exe experiments/fixed_points_deficiencies_v1/final_report.py
```

The report writer refuses incomplete audit/test scope and does no training or
model inference. The final training audit needs the local checkpoints; the
independent prediction/statistics recount does not load models. Preserve all
frozen records when reproducing. A missing asset is not permission to relax
orbit isolation, change a seed or generate replacement tests.

The statistical units are the eight paired repeats. Source competence, valid
mathematical relations and transfer benefit are distinct. U is related to a
common source and is not difficulty/distribution matched to D. This one-case
result does not prove formula use, establish a scaling law or replace earlier
failed confirmations. The fixed scope is finished; no extension, upload or
publication is performed. Checkpoints remain local.
"""
    paper = f"""# Proposed DMLR result paragraph: canonical completion draft

Motivated by earlier V5 and task-count observations, we conducted a new
prespecified fixed-points source-replacement comparison in eight paired
data/initialization units. Three sources (descents, successions and
left-to-right maxima) were fixed; the fourth was peaks (P), deficiencies (D),
anti-fixed points (T), or right-to-left maxima (U). The legacy one-line n=10
Transformer and 4800-update source budget were fixed. Full-parameter adaptation
used independent fits on nested 256/512/1024 training subsets, 800 updates per
learning-rate candidate, and a separate 1024-example labeled development set.
All 120 development selections froze before independent test generation.
The sole primary outcome was the paired difference in
`S=.25*A256+.5*A512+.25*A1024`, with accuracy defined as complete greedy
answer-plus-EOS exact match. D-U was **{primary['mean_pp']:+.2f} percentage points**
(sample SD {primary['sample_sd_pp']:.2f}; 95% paired-t interval
[{low:+.2f}, {high:+.2f}], df=7), positive in {primary['positive_units']}/8 units.
{conclusion}

The prespecified secondary gain over random initialization was smaller:
**D-A={d_a:+.2f} pp** (sample SD {float(gain_d['sample_sd_pp']):.2f};
{gain_d['positive_units']} positive units and {gain_d['zero_units']} tie).
P, T and U retained negative area gains relative to A in all eight units.
The descriptive identity `D-U=(D-A)-(U-A)` includes U's
{u_a:+.2f} pp mean negative transfer; it is not a mechanism decomposition.
Pointwise D-U means narrowed from {size_contrasts} pp across
256/512/1024 training examples. These secondary observations do not replace
the weighted-area primary comparison or establish a scaling law.

D and U were not matched in source difficulty, answer distribution or
redundancy; U itself has a reversal relation to a common source. Numeric answer
parameters were retained at adaptation, while the target-specific task row
was restored. Neither the positive comparison, source competence, nor answer
derivability establishes formula use or a transplantable lemma module.
Small-sample t assumptions and finite-space orbit exclusion limit interval
coverage. The finding concerns one background, model, budget and length; the
additional development labels are part of target supervision. Historical V5
failed confirmation and task-count negative outcomes remain separate evidence.

This documentation-only draft adds baseline and metric context to the
[original frozen-analysis draft](artifacts/analysis/PAPER_RESULT_DRAFT.md),
which remains unchanged. See [the canonical result report](RESULT_REPORT.md),
[raw metrics](artifacts/results_test.csv), [paired summaries](artifacts/analysis/paired_summary.csv)
and [documentation amendment](REPORT_AMENDMENT.json). No new evaluation or
scientific selection was performed to write this paragraph.
"""
    return report, readme, paper, workers


def run(artifacts=None):
    artifacts = Path(artifacts).resolve() if artifacts else HERE / "artifacts"
    assert artifacts == (HERE / "artifacts").resolve(), "Canonical documents bind this experiment's artifacts."
    report, readme, paper, workers = document_text(artifacts)
    source_paths = [artifacts / name for name in ["FINAL_TRAINING_AUDIT.json", "TEST_COMPLETE.json", "FROZEN_PROTOCOL.json",
        "SELECTIONS_FROZEN.json", "results_test.csv", "source_curves.csv", "accounting.csv", "exposures.csv",
        "DATA_TASK_MATCHING.csv", "SOURCE_LEARNING_NOTE.md", "analysis/ANALYSIS_MANIFEST.json",
        "analysis/INDEPENDENT_RECOMPUTE.json", "analysis/primary_summary.json", "analysis/paired_summary.csv",
        "analysis/scores_per_repeat.csv", "loss_figures/LOSS_FIGURE_MANIFEST.json"]]
    source_paths.extend([path for path, _ in workers])
    source_paths.extend([HERE / "HYPOTHESIS_ORIGIN.md", HERE / "ENGINEERING_LOG.md", HERE / "EXECUTION_LOG.md"])
    sources = {p.relative_to(HERE).as_posix(): sha(p) for p in source_paths}
    immutable_before = {p.relative_to(HERE).as_posix(): sha(p) for p in (artifacts / "analysis").iterdir() if p.is_file()}
    outputs = {"RESULT_REPORT.md": report, "README.md": readme, "PAPER_RESULT_DRAFT.md": paper}
    manifest_path = HERE / "REPORT_AMENDMENT.json"
    previous_manifest = read_json(manifest_path) if manifest_path.exists() else None
    generator_sha = sha(Path(__file__))
    if previous_manifest and previous_manifest["generator_sha256"] == generator_sha and previous_manifest["sources"] == sources:
        assert all(sha(HERE / name) == expected for name, expected in previous_manifest["outputs"].items())
        print(json.dumps(dict(status="already_current", outputs=list(outputs)), indent=2))
        return previous_manifest
    previous = {}
    for name, content in outputs.items():
        path = HERE / name
        if path.exists():
            old_sha = sha(path)
            backup = artifacts / "report_revisions" / f"{path.stem}_{old_sha}.md"
            backup.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists():
                assert sha(backup) == old_sha
            else:
                backup.write_bytes(path.read_bytes())
            previous[name] = dict(sha256=old_sha, retained_at=backup.relative_to(HERE).as_posix())
        path.write_text(content, encoding="utf8")
    assert immutable_before == {p.relative_to(HERE).as_posix(): sha(p) for p in (artifacts / "analysis").iterdir() if p.is_file()}
    if manifest_path.exists():
        prior_sha = sha(manifest_path)
        backup = artifacts / "report_revisions" / f"REPORT_AMENDMENT_{prior_sha}.json"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            backup.write_bytes(manifest_path.read_bytes())
    manifest = dict(status="complete", amendment="Documentation-only canonical completion report, README and paper paragraph",
        created_at_utc=datetime.now(timezone.utc).isoformat(), generator="final_report.py", generator_sha256=generator_sha,
        sources=sources, previous_documents=previous, outputs={name: sha(HERE / name) for name in outputs},
        frozen_analysis_outputs_unchanged=True, frozen_analysis_file_sha256=immutable_before,
        training_runs=0, model_forwards=0,
        changes=["Explicit overlapping timer interpretation", "Actual worker-envelope wall time",
                 "Separate engineering budget and partial timing", "Source mastery and comparator mismatch context",
                 "Primary-versus-scratch and per-size context; original paper draft retained",
                 "Canonical completed-state entry point; historical conclusions preserved"])
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf8")
    print(json.dumps(dict(status="complete", outputs=manifest["outputs"], training_runs=0, model_forwards=0), indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    args = parser.parse_args()
    run(args.artifacts)
