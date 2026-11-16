"""Frozen paired analysis of the new fixed-points confirmation; no model execution."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / "stats_plan.json").read_text(encoding="utf8"))
CONDITIONS = PLAN["conditions"]
SIZES = PLAN["target_samples"]
REPEATS = PLAN["repeat_ids"]
KINDS = ["selected", "endpoint"]
METRICS = ["complete", "legacy_answer_token"]
COLORS = dict(P="#777777", D="#0072B2", T="#D55E00", U="#009E73", A="#AA4499")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    assert rows, f"No rows for {path}"
    with Path(path).open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def t7_cdf(value):
    """Exact odd-df expression after t=sqrt(7)*tan(theta)."""
    angle = math.atan(value / math.sqrt(7))
    integral = (10 * angle + 7.5 * math.sin(2 * angle)
                + 1.5 * math.sin(4 * angle) + math.sin(6 * angle) / 6)
    return .5 + integral / (10 * math.pi)


def t7_quantile(probability=.975):
    assert .5 < probability < 1
    lo, hi = 0.0, 100.0
    for _ in range(90):
        mid = (lo + hi) / 2
        if t7_cdf(mid) < probability:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def summary(values, primary=False):
    values = list(values)
    assert len(values) == 8, "Incomplete confirmation: eight prespecified units are required"
    result = dict(n=8, mean_pp=statistics.mean(values), sample_sd_pp=statistics.stdev(values),
                  min_pp=min(values), max_pp=max(values), positive_units=sum(x > 0 for x in values),
                  zero_units=sum(x == 0 for x in values), negative_units=sum(x < 0 for x in values))
    if primary:
        radius = t7_quantile() * result["sample_sd_pp"] / math.sqrt(8)
        result.update(t_critical=t7_quantile(), df=7, ci95_low_pp=result["mean_pp"] - radius,
                      ci95_high_pp=result["mean_pp"] + radius)
    return result


def load_results(artifacts):
    rows = read_csv(artifacts / "results_test.csv")
    expected = set(itertools.product(REPEATS, CONDITIONS, SIZES, KINDS))
    indexed = {}
    for row in rows:
        key = row["repeat_id"], row["condition"], int(row["target_samples"]), row["checkpoint_kind"]
        assert key in expected and key not in indexed, key
        assert int(row["count"]) == 2048, key
        for count, rate in [("correct", "accuracy"), ("legacy_correct", "legacy_accuracy")]:
            value = int(row[count]) / int(row["count"])
            assert abs(float(row[rate]) - value) < 1e-12, (key, rate)
        assert math.isfinite(float(row["answer_ce"]))
        indexed[key] = row
    assert set(indexed) == expected, f"Missing {len(expected - set(indexed))} planned evaluation rows"
    return rows, indexed


def calculate(indexed):
    scores, differences, summaries, condition_summaries = [], [], [], []
    score_lookup = {}
    for metric in METRICS:
        rate = "accuracy" if metric == "complete" else "legacy_accuracy"
        for kind, condition, repeat in itertools.product(KINDS, CONDITIONS, REPEATS):
            accuracies = {n: float(indexed[repeat, condition, n, kind][rate]) for n in SIZES}
            value = sum(PLAN["weights"][str(n)] * accuracies[n] for n in SIZES) * 100
            score_lookup[metric, kind, repeat, condition] = value
            scores.append(dict(metric=metric, checkpoint_kind=kind, repeat_id=repeat,
                               condition=condition, S_pct=value,
                               **{f"accuracy_{n}_pct": accuracies[n] * 100 for n in SIZES}))
        contrasts = [["D", "U"], *PLAN["secondary_area_contrasts"]]
        for kind in KINDS:
            for scope, pairs in [("area", contrasts), *[(f"n{n}", PLAN["pointwise_contrasts"]) for n in SIZES]]:
                for left, right in pairs:
                    values = []
                    for repeat in REPEATS:
                        if scope == "area":
                            delta = score_lookup[metric, kind, repeat, left] - score_lookup[metric, kind, repeat, right]
                        else:
                            n = int(scope[1:])
                            delta = 100 * (float(indexed[repeat, left, n, kind][rate]) - float(indexed[repeat, right, n, kind][rate]))
                        values.append(delta)
                        differences.append(dict(metric=metric, checkpoint_kind=kind, scope=scope,
                                                contrast=f"{left}_minus_{right}", repeat_id=repeat, difference_pp=delta))
                    primary = metric == "complete" and kind == "selected" and scope == "area" and (left, right) == ("D", "U")
                    summaries.append(dict(metric=metric, checkpoint_kind=kind, scope=scope,
                        contrast=f"{left}_minus_{right}", analysis_role="primary" if primary else "secondary_or_diagnostic",
                        **summary(values, primary)))
        for kind, condition in itertools.product(KINDS, CONDITIONS):
            for n in [*SIZES, "area"]:
                vals = ([score_lookup[metric, kind, r, condition] for r in REPEATS] if n == "area" else
                        [float(indexed[r, condition, n, kind][rate]) * 100 for r in REPEATS])
                condition_summaries.append(dict(metric=metric, checkpoint_kind=kind, condition=condition,
                    scope="area" if n == "area" else f"n{n}", mean_pct=statistics.mean(vals),
                    sample_sd_pct=statistics.stdev(vals), min_pct=min(vals), max_pct=max(vals), n=8))
    return scores, differences, summaries, condition_summaries


def source_summary(artifacts, out):
    path = artifacts / "source_curves.csv"
    if not path.exists():
        return []
    groups = defaultdict(list)
    for row in read_csv(path):
        if int(row["step"]) == 4800:
            groups[row["condition"], row["task"], row["split"]].append(row)
    result = []
    for (condition, task, split), rows in sorted(groups.items()):
        assert len(rows) == 8 and {r["repeat_id"] for r in rows} == set(REPEATS)
        result.append(dict(condition=condition, task=task, split=split, units=8,
            mean_accuracy_pct=100 * statistics.mean(float(r["accuracy"]) for r in rows),
            sample_sd_pct=100 * statistics.stdev(float(r["accuracy"]) for r in rows),
            mean_answer_ce=statistics.mean(float(r["answer_ce"]) for r in rows),
            mean_majority_accuracy_pct=100 * statistics.mean(float(r["majority_accuracy"]) for r in rows),
            examples_per_unit=rows[0]["count"]))
    write_csv(out / "source_endpoint_summary.csv", result)
    return result


def make_plots(artifacts, out, indexed, scores, differences):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42})

    def save(fig, name):
        fig.tight_layout()
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(out / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for condition in CONDITIONS:
        matrix = [[100 * float(indexed[r, condition, n, "selected"]["accuracy"]) for n in SIZES] for r in REPEATS]
        for row in matrix:
            axes[0].plot([8, 9, 10], row, color=COLORS[condition], alpha=.13, linewidth=.8)
        means = [statistics.mean(row[j] for row in matrix) for j in range(3)]
        sds = [statistics.stdev(row[j] for row in matrix) for j in range(3)]
        axes[0].errorbar([8, 9, 10], means, yerr=sds, color=COLORS[condition], marker="o", capsize=3, label=condition)
        if condition != "A":
            gains = [[100 * (float(indexed[r, condition, n, "selected"]["accuracy"]) -
                            float(indexed[r, "A", n, "selected"]["accuracy"])) for n in SIZES] for r in REPEATS]
            for row in gains:
                axes[1].plot([8, 9, 10], row, color=COLORS[condition], alpha=.13, linewidth=.8)
            axes[1].errorbar([8, 9, 10], [statistics.mean(row[j] for row in gains) for j in range(3)],
                yerr=[statistics.stdev(row[j] for row in gains) for j in range(3)], color=COLORS[condition], marker="o", capsize=3, label=f"{condition}-A")
    for ax in axes:
        ax.set_xticks([8, 9, 10], SIZES)
        ax.set_xlabel("Target training examples (log2 spacing)")
        ax.grid(alpha=.2)
        ax.legend(ncol=3)
    axes[0].set(ylabel="Complete answer + EOS accuracy (%)", title="Selected checkpoints; eight paired units")
    axes[1].set(ylabel="Paired gain over shared A (pp)", title="Within-unit differences; mean and sample SD")
    axes[1].axhline(0, color="black", linewidth=.8)
    save(fig, "target_learning_curves")

    primary = [r for r in differences if r["metric"] == "complete" and r["checkpoint_kind"] == "selected" and r["scope"] == "area" and r["contrast"] == "D_minus_U"]
    values = [r["difference_pp"] for r in primary]
    stats = summary(values, True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(REPEATS, values, color=[COLORS["D"] if x >= 0 else COLORS["T"] for x in values], width=.65)
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[0].set(ylabel="S_D - S_U (pp)", title="Unique primary contrast, each paired unit")
    axes[1].errorbar([stats["mean_pp"]], [0], xerr=[[stats["mean_pp"] - stats["ci95_low_pp"]],
                [stats["ci95_high_pp"] - stats["mean_pp"]]], fmt="o", color=COLORS["D"], capsize=5)
    axes[1].scatter(values, [0.2] * 8, s=28, color="#777777", alpha=.8)
    axes[1].axvline(0, color="black", linewidth=.8)
    axes[1].set(yticks=[0, .2], yticklabels=["Mean + 95% t interval", "Eight unit differences"],
                xlabel="S_D - S_U (pp)", title="Paired t interval, df=7")
    axes[1].set_ylim(-.15, .35)
    save(fig, "primary_paired_difference")

    source_file = artifacts / "source_curves.csv"
    if source_file.exists():
        source = read_csv(source_file)
        tasks = ["descents", "successions", "left_to_right_maxima", "peaks", "deficiencies", "anti_fixed_points", "right_to_left_maxima"]
        fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
        for ax, task in zip(axes.flat, tasks):
            for condition in CONDITIONS[:-1]:
                for split, style in [("train", "--"), ("dev", "-")]:
                    grouped = defaultdict(list)
                    for row in source:
                        if row["task"] == task and row["condition"] == condition and row["split"] == split:
                            grouped[int(row["step"])].append(float(row["accuracy"]) * 100)
                    if grouped:
                        steps = sorted(grouped)
                        ax.plot(steps, [statistics.mean(grouped[s]) for s in steps], color=COLORS[condition],
                                linestyle=style, label=f"{condition} {split}")
            ax.set_title(task.replace("_", " "))
            ax.grid(alpha=.2)
            ax.set_ylim(0, 101)
            ax.legend(fontsize=7, ncol=2)
        axes.flat[-1].axis("off")
        axes.flat[-1].text(0, .8, "Eight paired units\nSolid: development\nDashed: training\nFull-answer + EOS exact match\nAll raw curves retained", va="top")
        fig.supxlabel("Pretraining updates")
        fig.supylabel("Mean accuracy (%)")
        save(fig, "source_learning_curves")

    adaptation_file = artifacts / "adaptation_curves.csv"
    if adaptation_file.exists():
        curves = read_csv(adaptation_file)
        chosen = {(r, c, n): float(indexed[r, c, n, "selected"]["lr"]) for r, c, n in itertools.product(REPEATS, CONDITIONS, SIZES)}
        fig, axes = plt.subplots(5, 3, figsize=(13, 14), sharex=True, sharey=True)
        for i, c in enumerate(CONDITIONS):
            for j, n in enumerate(SIZES):
                ax = axes[i, j]
                for split, style in [("train", "--"), ("dev", "-")]:
                    groups = defaultdict(list)
                    for row in curves:
                        if row["condition"] == c and int(row["target_samples"]) == n and row["split"] == split:
                            if float(row["lr"]) == chosen[row["repeat_id"], c, n]:
                                groups[int(row["step"])].append(float(row["accuracy"]) * 100)
                    steps = sorted(groups)
                    if steps:
                        ax.plot(steps, [statistics.mean(groups[s]) for s in steps], style, color=COLORS[c], label=split)
                        ax.fill_between(steps, [min(groups[s]) for s in steps], [max(groups[s]) for s in steps], color=COLORS[c], alpha=.06)
                ax.set_title(f"{c}: {n} training examples")
                ax.set_ylim(0, 101)
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
        fig.supxlabel("Adaptation updates; selected LR candidate per unit")
        fig.supylabel("Full-answer + EOS accuracy (%); band is unit range")
        save(fig, "adaptation_learning_curves")


def reports(artifacts, out, indexed, summaries, source_rows):
    config = json.loads((artifacts / "CONFIG.json").read_text(encoding="utf8"))
    architecture = config["architecture"]
    model_description = (f"{config['model']['layers']}-layer width-{config['model']['d_model']} GRU"
                         if architecture == "gru" else
                         f"{config['model']['layers']}-layer width-{config['model']['d_model']} Transformer with {config['model']['n_heads']} heads")
    primary = next(r for r in summaries if r["analysis_role"] == "primary")
    low, high = primary["ci95_low_pp"], primary["ci95_high_pp"]
    if low > 0:
        conclusion = "The primary interval is positive, supporting additional transfer value of deficiencies over the specified U source in this background and budget."
    elif high < 0:
        conclusion = "The primary interval is negative. The prediction that deficiencies would outperform the specified U source is not supported; U performs better in this fixed comparison."
    else:
        conclusion = "The primary interval includes zero. Evidence for additional transfer value of deficiencies over U remains uncertain; this is not proof of no benefit."
    selected_summary = [r for r in summaries if r["metric"] == "complete" and r["checkpoint_kind"] == "selected" and r["scope"] == "area"]
    table = "\n".join(f"| {r['contrast']} | {r['mean_pp']:+.4f} | {r['sample_sd_pp']:.4f} | [{r['min_pp']:+.4f}, {r['max_pp']:+.4f}] | {r['positive_units']}/8 |" for r in selected_summary)
    baseline = []
    for n in SIZES:
        values = [100 * float(indexed[r, "A", n, "selected"]["majority_accuracy"]) for r in REPEATS]
        baseline.append(f"| {n} | {statistics.mean(values):.4f} | {statistics.stdev(values):.4f} |")
    cost_table = "Accounting is unavailable; completion must be checked before publication."
    accounting_path = artifacts / "accounting.csv"
    if accounting_path.exists():
        accounting = read_csv(accounting_path)
        costs = []
        for stage, expected in [("pretrain", 32), ("adapt", 360)]:
            group = [r for r in accounting if r["stage"] == stage]
            assert len(group) == expected and len({r["run_id"] for r in group}) == expected
            costs.append(dict(stage=stage, runs=len(group),
                **{field: sum(int(r[field]) for r in group) for field in
                   ["updates", "sample_exposures", "prompt_tokens", "training_sequence_tokens", "supervised_tokens"]},
                elapsed_seconds=sum(float(r["elapsed_seconds"]) for r in group),
                independent_inputs_per_run_min=min(int(r["independent_inputs"]) for r in group),
                independent_inputs_per_run_max=max(int(r["independent_inputs"]) for r in group)))
        write_csv(out / "compute_summary.csv", costs)
        cost_table = ("| Stage | Runs | Updates | Samples | Sequence tokens | Supervised tokens | Runtime (hours) |\n"
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n" +
            "\n".join(f"| {r['stage']} | {r['runs']} | {r['updates']} | {r['sample_exposures']} | "
                      f"{r['training_sequence_tokens']} | {r['supervised_tokens']} | {r['elapsed_seconds']/3600:.4f} |" for r in costs))
    report = f"""# Fixed-points source-replacement model replication

Status: all 32 pretraining models, 360 learning-rate candidates and 120 selected
adaptations are represented by complete test records. Selected and fixed-endpoint
results remain separate. This is the prespecified model-change replication of
the original fixed-points source-replacement confirmation, using {model_description}.
These eight new units are reported separately and never pooled with the original
eight units. Historical failed confirmation and negative transfer are retained.

## Primary result

The prespecified score is S = 0.25 A256 + 0.5 A512 + 0.25 A1024, using complete
generated answer plus EOS exact match. The sole primary comparison S_D-S_U is
**{primary['mean_pp']:+.4f} percentage points**, sample SD {primary['sample_sd_pp']:.4f},
range [{primary['min_pp']:+.4f}, {primary['max_pp']:+.4f}], with a paired 95% t interval
**[{low:+.4f}, {high:+.4f}]**, df=7. {primary['positive_units']} of eight differences
are positive, {primary['negative_units']} negative and {primary['zero_units']} zero.

{conclusion}

![Each paired primary difference](primary_paired_difference.png)

## All prespecified area contrasts

All differences are computed within unit before aggregation. Secondary contrasts
are descriptive and cannot replace the primary contrast.

| Contrast | Mean (pp) | Sample SD | Unit range (pp) | Positive units |
| --- | ---: | ---: | --- | ---: |
{table}

The per-size table contains all ten pairwise differences at each of the three
sample sizes. It retains every unit and both selected/endpoint diagnostics;
`metric=legacy_answer_token` is an explicitly separate evaluation bridge.
It is not pooled with the complete-answer metric.

![Target sample-size curves and paired gains](target_learning_curves.png)

## Protocol and supervision

Common sources are descents, successions and left_to_right_maxima; the fourth
slot is P=peaks, D=deficiencies, T=anti_fixed_points or U=right_to_left_maxima.
A is one shared scratch adaptation per unit/sample-size/LR candidate. The task
sets and common-first slot order match the original eight-unit experiment.
Historically P/D correspond to the old v5 C task sets, and T was a new
composition rather than a direct replay of an old B group.

The tokenizer/input/loss/optimizer and schedules inherit the original confirmation:
one-line n=10, vocabulary 188, {model_description},
4800 pretraining updates, batch 64 with 16 samples per task. Each adaptation
starts independently from the source endpoint or paired step-0 state, resets
the fixed_points tied token row to its step-0 value, and uses a fresh optimizer.
All parameters adapt for 800 updates at each of three fixed learning rates.
Development selection uses full-answer accuracy, answer CE, earliest positive
step, then fixed LR order. This full-answer selection/evaluation differs from
the legacy v5 answer-token metric and is disclosed rather than merged.

Training subsets are nested 256/512/1024. **Each unit additionally uses 1024
labeled target development examples** for selection. A 256-example adaptation
is therefore not a procedure with only 256 target labels in total. All 120
selections freeze before independent new tests are generated. No test-driven
task, checkpoint, LR, sample-size or budget choice occurs.

| Target training examples | Training-derived majority test accuracy (%) | SD across units |
| --- | ---: | ---: |
{chr(10).join(baseline)}

## Source learning and accounting

All source training/development curves, full-answer accuracy, loss and
training-derived majority baselines are retained. `source_endpoint_summary.csv`
separates common-source performance by composition and split; the original
per-unit records remain in `../source_curves.csv`. Training/exposure records
report actual tokens, supervised tokens, samples, unique inputs, updates and
runtime in `../exposures.csv` and `../accounting.csv`. Equal update counts alone
are not asserted to match all resources. Engineering checks/retries are
separate from the eight scientific units.

{cost_table}

Runtime is the sum of recorded scientific training-run elapsed intervals, not
total elapsed wall time or dedicated GPU-hours. Each run includes its internal
train/development evaluations and writes; concurrent runs overlap. Final tests,
data checks and engineering checks have separate records. Unique inputs are per run; summing reused
inputs across conditions would not measure independent data.

## Interpretation limits

The paired t interval assumes approximately independent normal unit differences;
eight units provide limited diagnostics. Global inverse/reverse/complement orbit
rejection draws from a finite universe and introduces dependence, despite new
RNG streams. The interval has assumption-sensitive coverage, and eight units
are a fixed compute budget rather than a power guarantee. Conditions, samples,
models reused across sample sizes and shared baselines do not enlarge n.

FP+E+D=n and E(p)=D(inverse(p)) are mathematical identities. D(p) alone is not
enough for that reconstruction. Anti-fixed points are connected through
complement/reversal and can also use positional comparison. U is related to
the common left-to-right-record task by reversal; it is not mathematically
unrelated, nor matched to D in difficulty, distribution or redundancy.
Training supplies no paired-transform queries, formula supervision or local
position labels. Positive transfer cannot prove formula use or an independently
transplantable lemma module. Source-learning correlations are descriptive,
not mediator-causal evidence.

If D and U both improve over P without D exceeding U, improvement over the old
background is not specific evidence for deficiencies. D-versus-T concerns
these two sources only, not a general hierarchy of shared computation and
transformation relations. U does not fully separate adding a task from removing
peaks. No outcomes trigger another target, architecture, budget or background.

## Artifacts and reproduction

See `scores_per_repeat.csv`, `paired_differences.csv`, `paired_summary.csv`,
`condition_summary.csv`, `primary_summary.json`, `../results_test.csv`, and
`../predictions/`. Independent raw-prediction recounting is implemented in
`../../independent_recompute.py`, separately from the analysis calculations.

```powershell
.venv/Scripts/python.exe experiments/fixed_points_deficiencies_arch_v1/analysis.py --artifacts experiments/fixed_points_deficiencies_arch_v1/artifacts
.venv/Scripts/python.exe experiments/fixed_points_deficiencies_arch_v1/independent_recompute.py --artifacts experiments/fixed_points_deficiencies_arch_v1/artifacts
```

No upload, publication or additional experiment is performed by analysis.
"""
    (out / "REPORT.md").write_text(report, encoding="utf8")
    (out / "README.md").write_text(f"""# New fixed-points confirmation results

Eight paired units; 32 source models; 360 adaptation candidates; 120 selected
results. Primary S_D-S_U: {primary['mean_pp']:+.4f} pp, paired 95% t interval
[{low:+.4f}, {high:+.4f}], sample SD {primary['sample_sd_pp']:.4f} pp.

{conclusion}

Read [REPORT.md](REPORT.md), [the primary statistics](primary_summary.json),
[per-unit differences](paired_differences.csv), and
[sample-size curves](target_learning_curves.png). The results apply to the fixed
four-source background, n=10, inherited v5 budget and specified adaptation
procedure, with {model_description}. The original eight-unit confirmation and
prior v5/task-count results remain separate and are never pooled. No further experiment
or publication is automatically launched.
""", encoding="utf8")
    (out / "PAPER_RESULT_DRAFT.md").write_text(f"""# Proposed DMLR result paragraph

We replicated the prespecified fixed-points source replacement with
{model_description} in eight new paired units, separately from the original
eight-unit confirmation. With
descents, successions and left-to-right maxima held fixed, the fourth source
was peaks (P), deficiencies (D), anti-fixed points (T), or right-to-left maxima
(U). Under the inherited v5 training budget, independent full-parameter target
adaptations used nested 256/512/1024 training examples and a separate 1024-example
labeled development set. All choices preceded new-test generation. The sole
primary contrast in normalized log-sample accuracy area, S_D-S_U, was
{primary['mean_pp']:+.2f} percentage points (sample SD {primary['sample_sd_pp']:.2f};
95% paired t interval [{low:+.2f}, {high:+.2f}], df=7).
{conclusion} This comparison concerns specified source replacements in one
background; U is itself related to a common source through input reversal.
Neither mathematical answer derivability nor observed source competence proves
internal formula use. The eight-unit interval is assumption-sensitive under
finite-space orbit exclusion. Earlier unsuccessful v5 confirmation and task-count
negative outcomes are retained as separate evidence.
""", encoding="utf8")
    write_csv(out / "CLAIM_EVIDENCE_LIMITS.csv", [
        dict(claim="Conditional value of deficiencies versus U", evidence="primary_summary.json; eight paired S_D-S_U values", limit="One background, budget, length and architecture; uncertainty is retained"),
        dict(claim="Effect of the specified source replacements", evidence="paired_summary.csv; D-P, U-P, D-T and all source-A contrasts", limit="Secondary descriptive results do not replace the primary comparison; removal of peaks is not separated from addition"),
        dict(claim="Mathematical answer relationships", evidence="../../../fixed_points_deficiencies_v1/preflight_audit/JOINT_DERIVABILITY.csv; ../FROZEN_PROTOCOL.json historical certificate bindings", limit="A valid oracle formula does not establish model implementation or transfer"),
        dict(claim="Source learning differs across compositions", evidence="../source_curves.csv; source_endpoint_summary.csv", limit="Descriptive association, not a causal mediation analysis"),
        dict(claim="Training budgets and pairing are auditable", evidence="../exposures.csv; ../accounting.csv; initialization and selection bindings", limit="Equal updates do not alone ensure equal tokens or independent-input coverage"),
        dict(claim="Full-answer evaluation with legacy bridge", evidence="../results_test.csv; ../predictions/", limit="Full-answer/EOS and answer-token metrics are distinct and never pooled"),
    ])


def run(artifacts, out=None, plots=True):
    artifacts = Path(artifacts).resolve()
    assert artifacts == (HERE / "artifacts").resolve(), "Only this replication may be analyzed"
    out = Path(out).resolve() if out else artifacts / "analysis"
    if not out.is_relative_to(artifacts) or out.exists():
        raise ValueError("Use a fresh analysis directory inside this replication")
    out.mkdir(parents=True)
    rows, indexed = load_results(artifacts)
    scores, differences, summaries, condition_summaries = calculate(indexed)
    write_csv(out / "scores_per_repeat.csv", scores)
    write_csv(out / "paired_differences.csv", differences)
    write_csv(out / "paired_summary.csv", summaries)
    write_csv(out / "condition_summary.csv", condition_summaries)
    primary = next(r for r in summaries if r["analysis_role"] == "primary")
    primary["differences_by_repeat_pp"] = {r["repeat_id"]: r["difference_pp"] for r in differences
        if r["metric"] == "complete" and r["checkpoint_kind"] == "selected" and r["scope"] == "area" and r["contrast"] == "D_minus_U"}
    primary.update(status="complete", hypothesis_origin="Prespecified model-change replication of fixed_points_deficiencies_v1; eight new units never pooled with the original eight", result_table_sha256=sha(artifacts / "results_test.csv"), statistics_plan_sha256=sha(HERE / "stats_plan.json"))
    write_json(out / "primary_summary.json", primary)
    source_rows = source_summary(artifacts, out)
    if plots:
        make_plots(artifacts, out, indexed, scores, differences)
    reports(artifacts, out, indexed, summaries, source_rows)
    write_json(out / "ANALYSIS_MANIFEST.json", dict(status="complete", units=8, evaluated_rows=len(rows),
        inputs={"results_test.csv": sha(artifacts / "results_test.csv"), "stats_plan.json": sha(HERE / "stats_plan.json")},
        outputs={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in {"ANALYSIS_MANIFEST.json", "INDEPENDENT_RECOMPUTE.json"}},
        training_runs=0, model_forwards=0))
    return primary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.artifacts, args.out, not args.no_plots), indent=2))
