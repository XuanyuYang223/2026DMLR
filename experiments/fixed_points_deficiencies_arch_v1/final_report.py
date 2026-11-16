"""Write new completion documents from this replication's audited tables only.

This writer never imports a model or reruns a selection, inference or training.
Existing reports and derived output directories are not overwritten.
"""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ART = HERE / "artifacts"
ANALYSIS = ART / "analysis"
CONDITIONS = ["P", "D", "T", "U", "A"]
UNITS = [f"R{i}" for i in range(1, 9)]
SIZES = [256, 512, 1024]
CANDIDATES = dict(P="peaks", D="deficiencies", T="anti_fixed_points", U="right_to_left_maxima")
COMMON = ["descents", "successions", "left_to_right_maxima"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def table(headers, values):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in values]])


def format_model(config):
    model = config["model"]
    if config["architecture"] == "gru":
        return f"{model['layers']}-layer width-{model['d_model']} GRU"
    return f"{model['layers']}-layer width-{model['d_model']} Transformer with {model['n_heads']} heads"


def validate():
    for filename in ("README.md", "RESULT_REPORT.md", "PAPER_RESULT_DRAFT.md", "REPORT_MANIFEST.json"):
        if (HERE / filename).exists():
            raise FileExistsError(f"Preserve the existing completion document: {filename}")
    frozen = read(ART / "FROZEN_PROTOCOL.json")
    for path, expected in frozen["code_sha256"].items():
        assert sha(ROOT / path) == expected, path
    training = read(ART / "FINAL_TRAINING_AUDIT.json")
    assert training["status"] == "passed"
    assert (training["source_models"], training["adaptation_candidates"], training["selected_results"]) == (32, 360, 120)
    independent = read(ANALYSIS / "INDEPENDENT_RECOMPUTE.json")
    assert independent["status"] == "pass" and independent["independent_units"] == 8
    assert (independent["evaluated_rows"], independent["raw_prediction_rows"]) == (240, 491520)
    testing = read(ART / "TEST_COMPLETE.json")
    assert testing["status"] == "complete" and (testing["evaluations"], testing["selected_results"]) == (240, 120)
    manifest = read(ANALYSIS / "ANALYSIS_MANIFEST.json")
    assert manifest["status"] == "complete" and manifest["units"] == 8
    assert testing["results_sha256"] == independent["sources"]["results_test.csv"] == manifest["inputs"]["results_test.csv"] == sha(ART / "results_test.csv")
    for filename, expected in manifest["outputs"].items():
        assert sha(ANALYSIS / filename) == expected, filename
    losses = read(ART / "loss_figures/LOSS_FIGURE_MANIFEST.json")
    assert losses["status"] == "complete"
    for filename, expected in losses["outputs"].items():
        assert sha(ART / "loss_figures" / filename) == expected
    return frozen["config"]


def run():
    config = validate()
    model_description = format_model(config)
    gate = read(ART / "GATE_RESULT.json")
    assert gate["status"] == "complete" and gate["selected_architecture"] == config["architecture"]
    assert bool(gate["passed"]) == (config["architecture"] == "gru")
    assert {r["task"] for r in gate["runs"]} == {"fixed_points", "deficiencies"}
    gate_table = table(["Gate task", "Endpoint train accuracy (%)", "Endpoint dev accuracy (%)", "Threshold passed", "Elapsed seconds"],
        [[r["task"], f"{100*r['endpoint_train']['accuracy']:.2f}",
          f"{100*r['endpoint_dev']['accuracy']:.2f}", str(r["criterion_passed"]),
          f"{r['elapsed_seconds']:.2f}"] for r in gate["runs"]])
    primary = read(ANALYSIS / "primary_summary.json")
    mean, sd = primary["mean_pp"], primary["sample_sd_pp"]
    low, high = primary["ci95_low_pp"], primary["ci95_high_pp"]
    if low > 0:
        conclusion = "The primary interval is entirely positive. This supports additional transfer value of deficiencies over the specified U task under this model, background and budget."
    elif high < 0:
        conclusion = "The primary interval is entirely negative. The prediction that deficiencies outperform U is not supported; U performs better in this fixed comparison."
    else:
        conclusion = "The primary interval includes zero. Evidence for additional transfer value of deficiencies over U remains uncertain; this does not prove absence of benefit."
    area = [r for r in rows(ANALYSIS / "paired_summary.csv")
            if r["metric"] == "complete" and r["checkpoint_kind"] == "selected" and r["scope"] == "area"]
    contrasts = {r["contrast"]: r for r in area}
    scratch = {c: contrasts[f"{c}_minus_A"] for c in CANDIDATES}
    secondary_table = table(["Area contrast", "Role", "Mean (pp)", "Sample SD", "Unit range (pp)", "Positive / zero / negative"],
        [[r["contrast"], r["analysis_role"], f"{float(r['mean_pp']):+.4f}", f"{float(r['sample_sd_pp']):.4f}",
          f"[{float(r['min_pp']):+.4f}, {float(r['max_pp']):+.4f}]",
          f"{r['positive_units']} / {r['zero_units']} / {r['negative_units']}"] for r in area])
    scores = {(r["repeat_id"], r["condition"]): float(r["S_pct"])
              for r in rows(ANALYSIS / "scores_per_repeat.csv")
              if r["metric"] == "complete" and r["checkpoint_kind"] == "selected"}
    unit_table = table(["Unit", *[f"S_{c} (%)" for c in CONDITIONS], "Primary D-U (pp)"],
        [[unit, *[f"{scores[unit,c]:.4f}" for c in CONDITIONS],
          f"{primary['differences_by_repeat_pp'][unit]:+.4f}"] for unit in UNITS])
    conditions = {(r["condition"], r["scope"], r["checkpoint_kind"]): r
                  for r in rows(ANALYSIS / "condition_summary.csv") if r["metric"] == "complete"}
    raw = rows(ART / "results_test.csv")
    target_rows = []
    for c in CONDITIONS:
        for n in SIZES:
            selected = conditions[c, f"n{n}", "selected"]
            endpoint = conditions[c, f"n{n}", "endpoint"]
            group = [r for r in raw if r["condition"] == c and int(r["target_samples"]) == n and r["checkpoint_kind"] == "selected"]
            assert len(group) == 8
            target_rows.append([c, n, f"{float(selected['mean_pct']):.2f}", f"{float(selected['sample_sd_pct']):.2f}",
                f"{float(endpoint['mean_pct']):.2f}", f"{statistics.mean(float(r['answer_ce']) for r in group):.4f}",
                f"{100*statistics.mean(float(r['majority_accuracy']) for r in group):.2f}"])
    target_table = table(["Route", "Train examples", "Selected accuracy (%)", "Sample SD", "Endpoint accuracy (%)", "Selected answer CE", "Train-derived majority (%)"], target_rows)
    source = {(r["condition"], r["task"], r["split"]): r for r in rows(ANALYSIS / "source_endpoint_summary.csv")}
    candidate_rows, common_rows = [], []
    for c, task in CANDIDATES.items():
        train, dev = source[c, task, "train"], source[c, task, "dev"]
        candidate_rows.append([c, task, f"{float(train['mean_accuracy_pct']):.2f}", f"{float(dev['mean_accuracy_pct']):.2f}",
            f"{float(dev['sample_sd_pct']):.2f}", f"{float(dev['mean_answer_ce']):.4f}", f"{float(dev['mean_majority_accuracy_pct']):.2f}"])
    for task in COMMON:
        for c in CANDIDATES:
            train, dev = source[c, task, "train"], source[c, task, "dev"]
            common_rows.append([task, c, f"{float(train['mean_accuracy_pct']):.2f}", f"{float(dev['mean_accuracy_pct']):.2f}",
                f"{float(dev['mean_majority_accuracy_pct']):.2f}", f"{float(dev['mean_answer_ce']):.4f}"])
    candidate_table = table(["Condition", "Candidate", "Train accuracy (%)", "Dev accuracy (%)", "Dev sample SD", "Dev answer CE", "Dev majority (%)"], candidate_rows)
    common_table = table(["Common task", "Condition", "Train accuracy (%)", "Dev accuracy (%)", "Dev majority (%)", "Dev answer CE"], common_rows)
    cost_rows = rows(ANALYSIS / "compute_summary.csv")
    cost_table = table(["Stage", "Runs", "Updates", "Sample exposures", "Prompt tokens", "Sequence tokens", "Supervised tokens", "Sum of run elapsed seconds"],
        [[r["stage"], r["runs"], r["updates"], r["sample_exposures"], r["prompt_tokens"], r["training_sequence_tokens"], r["supervised_tokens"], f"{float(r['elapsed_seconds']):.2f}"] for r in cost_rows])
    worker_rows = []
    for prefix in ("SOURCE", "ADAPT"):
        files = sorted(ART.glob(f"{prefix}_WORKER_*.json"))
        for path in files:
            r = read(path)
            worker_rows.append([prefix, ", ".join(r["units"]), r["started_at_utc"], r["finished_at_utc"], f"{r['elapsed_seconds']:.2f}"])
    worker_table = table(["Stage", "Worker units", "Start UTC", "Finish UTC", "Recorded seconds"], worker_rows)
    architecture_note = ("Both prespecified GRU gate tasks passed the endpoint development threshold; the formal model is the four-layer GRU."
        if config["architecture"] == "gru" else
        "The GRU gate did not satisfy both endpoint thresholds, so the prespecified eight-layer Transformer fallback was used. This is a depth-change replication within the Transformer family, not evidence from a different architecture family.")
    scratch_sentence = "; ".join(f"{c}-A {float(scratch[c]['mean_pp']):+.4f} pp (SD {float(scratch[c]['sample_sd_pp']):.4f})" for c in CANDIDATES)
    report = f"""# Fixed-points source-replacement model replication: completed results

The fixed scope is complete: 32 source models, 360 adaptation learning-rate candidates,
120 development-selected results, 240 selected/endpoint evaluations and 491,520
stored prediction records. The records reuse eight test pools containing 16,384
distinct inputs; shared checkpoints and conditions do not create more independent units.
All training and prediction/statistics audits passed.

This study uses **{model_description}**, eight fresh paired units and n=10. It
replicates the original source-replacement design under the prespecified model
selection rule. Its results are reported separately and **never pooled with the
original eight-unit confirmation**. Earlier v5 failed confirmation and task-count
negative outcomes are unchanged.

## Sole primary comparison

S = .25*A256 + .5*A512 + .25*A1024 is the normalized trapezoid area over log2
target training size, using complete generated answer-plus-EOS exact match.
The sole primary paired contrast is **S_D-S_U = {mean:+.4f} pp**, sample SD
{sd:.4f}, range [{primary['min_pp']:+.4f}, {primary['max_pp']:+.4f}], paired 95% t
interval **[{low:+.4f}, {high:+.4f}] pp**, df=7. There are
{primary['positive_units']} positive, {primary['zero_units']} zero and
{primary['negative_units']} negative unit differences.

{conclusion}

{unit_table}

![Each paired primary difference](artifacts/analysis/primary_paired_difference.png)

The interval uses eight within-unit differences. Approximate normality and
independence are assumptions with limited diagnostics, not guarantees. Global
orbit rejection conditions later pools on earlier pools in a finite permutation
space. Learning rates, training sizes, test examples and shared A baselines do
not increase the repeat count. An interval containing zero is uncertainty, not
proof of no effect.

## Scratch-relative context and secondary results

The D-U contrast compares two source replacements, not a gain over scratch.
The scratch-relative results are **{scratch_sentence}**. The identity
D-U=(D-A)-(U-A) is a descriptive decomposition; it does not establish a mechanism.
All prescribed contrasts remain visible regardless of direction.

{secondary_table}

Secondary comparisons cannot replace the primary contrast. All ten pairwise
contrasts at each target size, both checkpoint kinds and the separate legacy
answer-token bridge remain in `artifacts/analysis/paired_differences.csv` and
`paired_summary.csv`.

{target_table}

The endpoint is update 800 of the development-selected learning-rate candidate,
not a test-selected alternative. Answer CE is teacher-forced answer-token loss;
EOS and mean two-token losses are separately retained. Majority labels are
determined from each corresponding training subset. All target sizes also use
1024 labeled development examples per unit; the smallest condition therefore
does not use only 256 target labels overall.

![Target curves and paired gains](artifacts/analysis/target_learning_curves.png)
![Selected-candidate development trajectories](artifacts/analysis/adaptation_learning_curves.png)

## Architecture gate, protocol and isolation

{architecture_note} The gate uses a separate, historically isolated n=10
training/development pool, discarded weights and no test set. Its criterion,
endpoint results and selected model are recorded in `artifacts/GATE_RESULT.json`.
There is no architecture tuning or outcome-dependent addition of units.

{gate_table}

Common source slots are descents, successions and left_to_right_maxima; the
fourth is P=peaks, D=deficiencies, T=anti_fixed_points or U=right_to_left_maxima.
A is the shared random-initialization baseline. Task sets and slot order match
the original eight-unit study. The model change and fresh paired units do not
change one-line encoding, vocabulary 188, loss, optimizer or schedules.

Each source model receives 4800 updates at batch 64, exactly 16 examples per
source per update. The inherited source learning rate decays through update
2400 and remains at its minimum through update 4800. Only the fixed source
endpoint supplies adaptation weights. Each route/size/LR starts independently,
loads all source parameters, restores the fixed_points tied token row to the
same unit's step-0 value, and uses a fresh optimizer. Full-parameter adaptation
uses 800 updates and the same three predetermined learning rates. Development
selection is highest full-answer accuracy, lowest answer CE, earliest positive
step, then declared learning-rate order.

Each unit has 8192 source-training, 1024 source-development, 1024 maximum
target-training, 1024 target-development and 2048 test inputs. Target subsets
256/512/1024 are nested. Complete inverse/reverse/complement orbits separate
roles, units, gate data and historical inputs. All 120 development choices
precede new-test generation. No result changes the task list, seeds, budgets,
learning-rate candidates or model selection rule.

## Source learning and comparator limitations

The following are fixed source endpoints at update 4800 across the eight new
units. Candidate accuracies concern different labels and distributions and do
not constitute matched-difficulty controls.

{candidate_table}

{common_table}

Full source training/development accuracy, losses and majority baselines remain
in `artifacts/source_curves.csv`. Weak fitting and negative transfer are retained;
source performance never filters a unit. Differences in common-source learning
are descriptive associations, not causal mediation evidence.

![Source learning](artifacts/analysis/source_learning_curves.png)
![Source answer losses](artifacts/loss_figures/source_answer_ce.png)
![Target answer losses](artifacts/loss_figures/adaptation_answer_ce.png)

FP+E+D=n and E(p)=D(inverse(p)) are proved identities, but D(p) alone is
insufficient to reconstruct FP. Deficiencies require two registered queries;
anti-fixed points connect through complement/reversal and can also use positional
comparison. U has a reversal relation to the common prefix-maxima source; it is
not an unrelated control. Original definition/relationship certificates are
reused by hash. No explicit paired-transform queries, formula supervision or
local labels are added to training. A positive transfer result cannot establish
internal formula use or a transferable lemma module. D/U remain unmatched in
task difficulty, answer distribution and redundancy. Replacing peaks does not
separate the absolute effects of adding a task and removing another.

## Actual budgets, elapsed time and engineering work

{cost_table}

{worker_table}

Run elapsed sums include internal evaluation and writes. Concurrent intervals
overlap; these sums are neither wall time nor dedicated GPU-hours. Worker start
and finish records provide the execution envelopes. Final test timings are in
`artifacts/results_test.csv`. Per-source exposure records include actual unique
input counts; repeated exposure and shared pools do not create new independent
data. Gate and smoke compute are separate engineering/development costs, not
new confirmation repeats; their exact records remain in the gate artifacts and
`artifacts/SMOKE.json`.

## Acceptance and reproduction

`FINAL_TRAINING_AUDIT.json` independently replays sampling, exposures, loading,
target-row restoration and development choices. `analysis/INDEPENDENT_RECOMPUTE.json`
recounts all prediction records, labels, paired scores and the primary interval
without model execution. No original experiment or publication record is changed.

The runner commands are documented in `PROTOCOL.md`; after formal workers finish:

```powershell
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/post_training.py
```

This completion report uses audited stored records only. The experiment stops
at the prescribed scope, with no additional target, background, architecture,
budget or repeat triggered by its results. Checkpoints remain local.
"""
    paragraph = f"""# Proposed DMLR model-replication paragraph

We repeated the fixed-points source-replacement comparison with {model_description}
in eight fresh paired units, keeping source compositions, n=10, the 4800-update
source budget and downstream selection procedure unchanged. Target adaptations
used nested 256/512/1024 training examples plus 1024 labeled development examples.
The sole primary normalized log-sample accuracy-area contrast, S_D-S_U, was
{mean:+.4f} percentage points (sample SD {sd:.4f}; paired 95% t interval
[{low:+.4f}, {high:+.4f}], df=7). {conclusion}
Scratch-relative area gains were {scratch_sentence}. This makes the comparison
with U explicit rather than describing D-U as a gain of that size over scratch.
The new units are never pooled with the original eight-unit study. {architecture_note}
Mathematical relationships and source learning do not establish formula use;
finite-space sampling and the small repeat count limit interval interpretation.
"""
    readme = f"""# Fixed-points model replication

Completed: 32 source models, 360 adaptation candidates, 120 selected target
results and independent prediction/statistics auditing. Model: {model_description}.

Primary S_D-S_U: **{mean:+.4f} pp**, sample SD {sd:.4f}; paired 95% t interval
**[{low:+.4f}, {high:+.4f}]**, eight new paired units.

{conclusion}

Read [RESULT_REPORT.md](RESULT_REPORT.md) for scratch-relative comparisons,
source mastery, actual costs, the gate outcome and interpretation limits.
[PROTOCOL.md](PROTOCOL.md) and [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md) describe
the prespecified rules. [Primary differences](artifacts/analysis/paired_differences.csv)
and [raw test metrics](artifacts/results_test.csv) retain every unit. Results
are separate from the original eight-unit confirmation and never pooled with it.
No added training or test evaluation is performed by this report writer.
"""
    outputs = {"RESULT_REPORT.md": report, "PAPER_RESULT_DRAFT.md": paragraph, "README.md": readme}
    for filename, text in outputs.items():
        with (HERE / filename).open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    source_paths = [ART / "FROZEN_PROTOCOL.json", ART / "GATE_RESULT.json", ART / "SMOKE.json",
        ART / "FINAL_TRAINING_AUDIT.json", ART / "TEST_COMPLETE.json", ART / "results_test.csv",
        ANALYSIS / "INDEPENDENT_RECOMPUTE.json", ANALYSIS / "ANALYSIS_MANIFEST.json",
        ANALYSIS / "primary_summary.json", ANALYSIS / "paired_summary.csv", ANALYSIS / "scores_per_repeat.csv",
        ANALYSIS / "condition_summary.csv", ANALYSIS / "source_endpoint_summary.csv", ANALYSIS / "compute_summary.csv"]
    source_paths += sorted(ART.glob("*_WORKER_*.json"))
    manifest = dict(status="complete", created_at_utc=datetime.now(timezone.utc).isoformat(),
        generator_sha256=sha(__file__), scientific_recomputation=False, model_forwards=0,
        training_updates=0, original_units_pooled=False,
        inputs={p.relative_to(HERE).as_posix(): sha(p) for p in source_paths},
        outputs={name: sha(HERE / name) for name in outputs})
    with (HERE / "REPORT_MANIFEST.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(dict(status="complete", model=config["architecture"], primary_mean_pp=mean,
                          primary_ci95_pp=[low, high], original_units_pooled=False)))


if __name__ == "__main__":
    run()
