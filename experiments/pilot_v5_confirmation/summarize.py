"""Figures and reports from saved CSV records only; never trains or selects models."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paired_statistics as ps

HERE = Path(__file__).resolve().parent
TARGETS = ("fixed_points", "longest_decreasing_run")
METRIC_LABELS = ("Change in B - A", "Change in C - A", "Interaction: change in B - C")
COLORS = {"A": "#69778d", "B": "#1683ad", "C": "#d28032"}


def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def pp(value):
    return f"{100 * float(value):+.2f}"


def pct(value):
    return f"{100 * float(value):.2f}%"


def truth(value):
    return str(value).lower() in ("true", "1")


def cell_key(row):
    return row["target"], int(row["seed"]), row["background"], row["condition"]


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def savefig(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def resolve_report_links(text, report_path, out):
    """Resolve template links against their real source and output directories.

    Template artifact links refer to the selected --out, whereas documentation
    links refer to this experiment's source directory. The report is stored in
    HERE for the delivered default run and inside --out for a separate rerun.
    """
    def rewrite(match):
        destination = match.group("destination")
        if destination.startswith(("http://", "https://", "#")):
            return match.group(0)
        filename, separator, fragment = destination.partition("#")
        target = out / filename[len("artifacts/"):] if filename.startswith("artifacts/") else HERE / filename
        try:
            link = os.path.relpath(target.resolve(), report_path.parent.resolve()).replace("\\", "/")
        except ValueError:  # Windows cannot express a relative path across drives.
            link = target.resolve().as_posix()
        if separator:
            link += separator + fragment
        if any(character.isspace() for character in link) or "(" in link or ")" in link:
            link = "<" + link + ">"
        return match.group("prefix") + link + ")"

    return re.sub(r"(?P<prefix>!?\[[^\]\n]*\]\()(?P<destination>[^\)\n]+)\)", rewrite, text)


def exploratory():
    out = HERE / "exploratory"
    differences, summaries = ps.run("exploratory")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    backgrounds = (("g1_recoils", "g1 recoils - g0", "#27659b", -0.08),
                   ("g2_deficiencies", "g2 deficiencies - g0", "#c36930", 0.08))
    for ti, target in enumerate(TARGETS):
        for mi, metric in enumerate(ps.METRICS):
            ax = axes[ti, mi]
            for background, label, color, offset in backgrounds:
                data = [r for r in differences if r["target"] == target and r["new_background"] == background]
                ax.plot([i + offset for i in range(3)], [100 * r[metric] for r in data],
                        marker="o", color=color, label=label, linewidth=1.1)
                for i, row in enumerate(data):
                    ax.annotate(f"{100 * row[metric]:+.2f}", (i + offset, 100 * row[metric]),
                                xytext=(2, 6 if offset < 0 else -13), textcoords="offset points", fontsize=8)
            ax.axhline(0, color="#666666", linewidth=0.8)
            ax.set_xticks(range(3), ["17", "42", "73"])
            ax.set_title(f"{target}\n{METRIC_LABELS[mi]}", fontsize=10)
            ax.set_ylabel("Percentage points")
            ax.grid(alpha=0.18)
            if ti == 1:
                ax.set_xlabel("Exploratory paired seed")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("v4 exploratory evidence: differences computed within each paired seed", fontsize=13)
    fig.tight_layout()
    savefig(fig, out / "exploratory_paired_background_changes")
    body = table(["Target", "Background change", "seed", "Δ(B−A) pp", "Δ(C−A) pp", "Δ(B−C) pp"],
                 [[r["target"], r["new_background"] + "−g0", r["seed"],
                   *(f"{100 * r[m]:+.4f}" for m in ps.METRICS)] for r in differences])
    summary_table = table(["Target", "Background change", "Mean Δ(B−A) pp", "Mean Δ(C−A) pp", "Mean Δ(B−C) pp", "Per-seed interaction range pp"],
                          [[target, background + "−g0",
                            *(f"{100 * next(r['mean'] for r in summaries if r['target'] == target and r['new_background'] == background and r['metric'] == m):+.4f}" for m in ps.METRICS),
                            f"[{100 * min(r['delta_B_minus_C'] for r in differences if r['target'] == target and r['new_background'] == background):+.4f}, {100 * max(r['delta_B_minus_C'] for r in differences if r['target'] == target and r['new_background'] == background):+.4f}]"]
                           for target in TARGETS for background, *_ in backgrounds])
    report = f"""# Per-seed paired recomputation of existing v4 results

This file reads only existing `pilot_v4_backgrounds/artifacts/results_selected.csv`, without new training or reading v5 confirmation results. Old seeds 17/42/73 provide exploratory evidence only. All results are accuracies of development-selected checkpoints on the same new v4 test pools; A is shared across backgrounds.

For each seed, compute `Δ(B−A)=(B_new−A)−(B_original−A)` and `Δ(C−A)=(C_new−A)−(C_original−A)`, then the primary interaction `Δ(B−C)=(B_new−C_new)−(B_original−C_original)`. Shared A cancels, so the first two terms equal the background changes in B and C themselves; the interaction is their difference. Values below are percentage points; CSV files retain raw accuracy proportions.

{body}

{summary_table}

For fixed_points, the deficiencies background versus the original increases mean B−A by 16.59pp and C−A by 12.22pp, leaving a primary interaction of only +4.36pp. All three seeds are positive, ranging from about +0.34 to +11.33pp. Improvement in B alone cannot all be attributed to the advantage of the related source over its replacement.

For longest_decreasing_run, the recoils background versus the original has a mean primary interaction of −12.00pp, negative in two seeds and positive in one. This mainly reflects change in B, while C changes less on average. The other two exploratory compositions are retained without adding confirmation contrasts from their results.

These are direct within-seed differences; separate SDs for the two groups are not used to judge background-effect reliability. Three exploratory blocks and contrasts chosen from the same results cannot provide independent confirmation; no significance claim is made for those blocks. The next stage fixes the positive FP g2−g0 and negative LDR g1−g0 directions, using five new paired data/initialization units; see `../stats_plan.json`. Exploration and confirmation are reported separately, not pooled as eight repeats.

![Background differences by seed](exploratory_paired_background_changes.png)

Recomputed raw data: [exploratory_paired_differences.csv](exploratory_paired_differences.csv). All 12 interactions were independently checked against the old v4 interaction CSV and agree within 1e−12. That old table was not used in recomputation; see the [audit](exploratory_statistics_audit.json).
"""
    (out / "EXPLORATORY_ANALYSIS_ZH.md").write_text(report, encoding="utf-8")
    print(json.dumps({"stage": "exploratory", "rows": len(differences), "report": str(out / "EXPLORATORY_ANALYSIS_ZH.md")}))


def result_tables(out, selected, candidates, plan):
    index = {cell_key(row): row for row in selected}
    displayed, deltas, aggregate, endpoints = [], [], [], []
    for contrast in plan["contrasts"]:
        target = contrast["target"]
        for background in (contrast["original_background"], contrast["new_background"]):
            for rep in plan["repetitions"]:
                seed = rep["seed"]
                values = {condition: index[target, seed, "shared" if condition == "A" else background, condition]
                          for condition in ("A", "B", "C")}
                for condition, row in values.items():
                    displayed.append({**row, "display_background": background,
                                      "shared_A_display_only": condition == "A"})
                for name, left, right in (("B_minus_A", "B", "A"), ("C_minus_A", "C", "A"), ("B_minus_C", "B", "C")):
                    deltas.append({"target": target, "seed": seed, "background": background,
                                   "contrast": name, "test_difference": float(values[left]["test_accuracy"]) - float(values[right]["test_accuracy"]),
                                   "dev_difference": float(values[left]["dev_accuracy"]) - float(values[right]["dev_accuracy"])})
            for condition in ("A", "B", "C"):
                rows = [r for r in displayed if r["target"] == target and r["display_background"] == background and r["condition"] == condition]
                aggregate.append({"target": target, "background": background, "condition": condition,
                                  **ps.summarize_values([float(r["test_accuracy"]) for r in rows]),
                                  "majority_accuracy_mean": ps.average(float(r["majority_accuracy"]) for r in rows)})
    assert len(displayed) == 60 and len(deltas) == 60
    for row in selected:
        end = [r for r in candidates if cell_key(r) == cell_key(row) and truth(r["selected_lr"]) and r["checkpoint"] == "endpoint"]
        assert len(end) == 1
        end = end[0]
        endpoints.append({**{k: row[k] for k in ("target", "seed", "background", "condition", "lr")},
                          "selected_step": row["adapt_step"], "endpoint_step": end["adapt_step"],
                          "selected_dev_accuracy": row["dev_accuracy"], "endpoint_dev_accuracy": end["dev_accuracy"],
                          "selected_test_accuracy": row["test_accuracy"], "endpoint_test_accuracy": end["test_accuracy"],
                          "endpoint_minus_selected_test": float(end["test_accuracy"]) - float(row["test_accuracy"])})
    ps.write_csv(out / "results_display_ABC.csv", displayed)
    ps.write_csv(out / "paired_condition_differences.csv", deltas)
    ps.write_csv(out / "results_summary.csv", aggregate)
    ps.write_csv(out / "selected_vs_endpoint.csv", endpoints)
    return displayed, deltas, aggregate, endpoints


def confirmation_plots(out, differences, summaries, plan):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, contrast in zip(axes, plan["contrasts"]):
        data = [r for r in differences if r["target"] == contrast["target"]]
        summary = next(r for r in summaries if r["target"] == contrast["target"] and r["primary"])
        color = "#1683ad" if contrast["expected_direction"] > 0 else "#d28032"
        values = [100 * r["delta_B_minus_C"] for r in data]
        ax.scatter(range(5), values, s=55, color=color, zorder=3)
        for i, value in enumerate(values):
            ax.annotate(f"{value:+.2f}", (i, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
        center = 100 * summary["mean"]
        ax.errorbar(5.4, center, yerr=100 * (summary["family95_upper"] - summary["mean"]), fmt="none",
                    ecolor=color, linewidth=6, alpha=0.20, capsize=6, label="Family 95% t interval")
        ax.errorbar(5.4, center, yerr=100 * (summary["ci95_upper"] - summary["mean"]), fmt="D",
                    color=color, linewidth=1.5, capsize=4, label="Mean and 95% t interval")
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_xticks([0, 1, 2, 3, 4, 5.4], [str(r["seed"]) for r in data] + ["Mean"])
        ax.set_ylabel("Paired interaction difference (percentage points)")
        ax.set_xlabel("New paired repeat / initialization seed")
        ax.set_title(contrast["target"] + "\n" + contrast["new_background"] + " - " + contrast["original_background"], fontsize=11)
        ax.grid(axis="y", alpha=0.18)
        ax.margins(y=0.2)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Fixed confirmation contrasts: change in (B - C), five new repeats\nIntervals assume a repeat-level t model; n = 5", fontsize=12)
    fig.tight_layout()
    savefig(fig, out / "figures" / "confirmation_interactions_by_repeat")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ti, contrast in enumerate(plan["contrasts"]):
        data = [r for r in differences if r["target"] == contrast["target"]]
        for mi, metric in enumerate(ps.METRICS):
            ax = axes[ti, mi]
            vals = [100 * r[metric] for r in data]
            ax.scatter(range(5), vals, color="#1683ad" if mi < 2 else "#984c8d", s=45)
            for i, val in enumerate(vals):
                ax.annotate(f"{val:+.2f}", (i, val), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
            ax.axhline(0, color="#555555", linewidth=0.8)
            ax.axhline(ps.average(vals), color="#888888", linestyle=":", linewidth=1, label="Paired mean")
            ax.set_xticks(range(5), [str(r["seed"]) for r in data])
            ax.set_title(contrast["target"] + "\n" + METRIC_LABELS[mi], fontsize=10)
            ax.set_ylabel("Percentage points")
            ax.set_xlabel("New paired repeat")
            ax.margins(y=0.22)
            ax.grid(alpha=0.18)
    fig.suptitle("Primary interaction and its auxiliary B/A and C/A components", fontsize=13)
    fig.tight_layout()
    savefig(fig, out / "figures" / "confirmation_background_components")


def source_tables(out, differences, plan):
    curves = ps.read_csv(out / "source_curves.csv")
    exposures = ps.read_csv(out / "source_exposures.csv")
    assert len(exposures) == 160
    index = {(r["target"], int(r["seed"]), r["background"], r["condition"], r["task"], int(r["step"]), r["split"]): r for r in curves}
    learning = []
    for exposure in exposures:
        row = {**exposure, "role": "candidate" if int(exposure["slot"]) == 0 else "common"}
        for split in ("train", "dev"):
            final = index[(*cell_key(exposure), exposure["task"], 4800, split)]
            for metric in ("accuracy", "answer_ce", "two_token_ce", "eos_ce", "majority_accuracy", "count"):
                row["multi4800_" + split + "_" + metric] = float(final[metric])
            row["multi4800_" + split + "_above_majority"] = float(final["accuracy"]) - float(final["majority_accuracy"])
        row["train_minus_dev_accuracy"] = row["multi4800_train_accuracy"] - row["multi4800_dev_accuracy"]
        learning.append(row)
    old_rows = []
    for task in sorted({r["task"] for r in exposures}):
        for seed in (17, 42, 73):
            path = HERE.parent / "pilot_v3_multitarget" / "artifacts" / "single" / f"s{seed}_{task}" / "curve.csv"
            for row in ps.read_csv(path):
                if int(row["step"]) in (1200, 4800):
                    old_rows.append({**row, "reference_scope": "historical_only_different_data_and_initialization_not_paired",
                                     "source_csv": str(path.relative_to(HERE.parent.parent)),
                                     "source_sha256": ps.sha256(path)})
    current = {(r["target"], int(r["seed"]), r["background"], r["condition"], int(r["slot"])): r for r in learning}
    changes, contexts, common = [], [], []
    for difference in differences:
        target, seed = difference["target"], difference["seed"]
        context = {k: difference[k] for k in ("target", "seed", "original_background", "new_background", *ps.METRICS)}
        unchanged_common = []
        for condition in ("B", "C"):
            for slot in range(4):
                old = current[target, seed, difference["original_background"], condition, slot]
                new = current[target, seed, difference["new_background"], condition, slot]
                same = old["task"] == new["task"]
                delta = new["multi4800_dev_accuracy"] - old["multi4800_dev_accuracy"] if same else ""
                changes.append({"target": target, "seed": seed, "condition": condition, "slot": slot,
                                "original_background": difference["original_background"], "new_background": difference["new_background"],
                                "old_task": old["task"], "new_task": new["task"], "same_mathematical_task": same,
                                "old_dev_accuracy": old["multi4800_dev_accuracy"], "new_dev_accuracy": new["multi4800_dev_accuracy"],
                                "same_task_dev_accuracy_change": delta,
                                "old_dev_majority": old["multi4800_dev_majority_accuracy"], "new_dev_majority": new["multi4800_dev_majority_accuracy"]})
                if slot == 0:
                    assert same
                    context[condition + "_candidate_task"] = old["task"]
                    context[condition + "_candidate_original_dev_accuracy"] = old["multi4800_dev_accuracy"]
                    context[condition + "_candidate_new_dev_accuracy"] = new["multi4800_dev_accuracy"]
                    context[condition + "_candidate_dev_change"] = delta
                elif same:
                    unchanged_common.append(delta)
        context["unchanged_common_max_abs_dev_change"] = max(map(abs, unchanged_common))
        contexts.append(context)
        for background in (difference["original_background"], difference["new_background"]):
            for slot in (1, 2, 3):
                b, c = current[target, seed, background, "B", slot], current[target, seed, background, "C", slot]
                assert b["task"] == c["task"]
                common.append({"target": target, "seed": seed, "background": background, "slot": slot, "task": b["task"],
                               "B_dev_accuracy": b["multi4800_dev_accuracy"], "C_dev_accuracy": c["multi4800_dev_accuracy"],
                               "B_minus_C_dev_accuracy": b["multi4800_dev_accuracy"] - c["multi4800_dev_accuracy"]})
    ps.write_csv(out / "source_learning_table.csv", learning)
    ps.write_csv(out / "historical_single_references.csv", old_rows)
    ps.write_csv(out / "source_background_changes.csv", changes)
    ps.write_csv(out / "source_interaction_context.csv", contexts)
    ps.write_csv(out / "common_source_BC.csv", common)
    return curves, learning, contexts, common


def learning_plots(out, source_curves, selected, plan):
    target_curves = ps.read_csv(out / "target_curves.csv")
    selected_index = {cell_key(row): row for row in selected}
    for contrast in plan["contrasts"]:
        target = contrast["target"]
        for background in (contrast["original_background"], contrast["new_background"]):
            fig, axes = plt.subplots(5, 2, figsize=(12, 17), sharex=True)
            for i, rep in enumerate(plan["repetitions"]):
                seed = rep["seed"]
                for j, condition in enumerate(("B", "C")):
                    ax = axes[i, j]
                    rows = [r for r in source_curves if cell_key(r) == (target, seed, background, condition)]
                    for task in sorted({r["task"] for r in rows}):
                        dev = sorted((r for r in rows if r["task"] == task and r["split"] == "dev"), key=lambda r: int(r["step"]))
                        train = sorted((r for r in rows if r["task"] == task and r["split"] == "train"), key=lambda r: int(r["step"]))
                        line = ax.plot([int(r["step"]) for r in dev], [float(r["accuracy"]) for r in dev], label=task)[0]
                        ax.plot([int(r["step"]) for r in train], [float(r["accuracy"]) for r in train], linestyle="--", linewidth=0.8, color=line.get_color(), alpha=0.6)
                    ax.set_ylim(0, 1.03)
                    ax.set_title(f"Repeat {seed}, {condition}", fontsize=10)
                    ax.set_ylabel("Accuracy")
                    ax.grid(alpha=0.18)
                    if i == 0:
                        ax.legend(fontsize=7)
                    if i == 4:
                        ax.set_xlabel("Pretraining updates")
            fig.suptitle(f"{target}, {background}: source learning\nSolid = dev, dashed = train; n = 10", fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            savefig(fig, out / "figures" / f"source_learning_{target}_{background}")
            fig, axes = plt.subplots(5, 2, figsize=(12, 17), sharex=True)
            for i, rep in enumerate(plan["repetitions"]):
                seed = rep["seed"]
                for condition in ("A", "B", "C"):
                    key = (target, seed, "shared" if condition == "A" else background, condition)
                    choice = selected_index[key]
                    rows = sorted((r for r in target_curves if cell_key(r) == key and float(r["lr"]) == float(choice["lr"])), key=lambda r: int(r["step"]))
                    assert len(rows) == 9
                    for j, metric in enumerate(("accuracy", "answer_ce")):
                        axes[i, j].plot([int(r["step"]) for r in rows], [float(r["dev_" + metric]) for r in rows], color=COLORS[condition], label=f"{condition}, LR={choice['lr']}")
                        axes[i, j].plot([int(r["step"]) for r in rows], [float(r["train_" + metric]) for r in rows], color=COLORS[condition], linestyle="--", alpha=0.55)
                        chosen = next(r for r in rows if int(r["step"]) == int(choice["adapt_step"]))
                        axes[i, j].scatter(int(chosen["step"]), float(chosen["dev_" + metric]), color=COLORS[condition], marker="o", s=28)
                for j, metric in enumerate(("accuracy", "answer_ce")):
                    ax = axes[i, j]
                    ax.set_title(f"Repeat {seed}: {metric}", fontsize=10)
                    ax.set_ylabel(metric)
                    ax.grid(alpha=0.18)
                    if j == 0:
                        ax.set_ylim(0, 1.03)
                    ax.legend(fontsize=7)
                    if i == 4:
                        ax.set_xlabel("Adaptation updates")
            fig.suptitle(f"{target}, {background}: selected-LR target curves\nSolid = dev, dashed = train, point = dev-selected checkpoint; all LR curves retained in CSV", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            savefig(fig, out / "figures" / f"target_learning_{target}_{background}")


def compute_tables(out):
    runs = []
    for kind, directory in (("pretrain", "multi"), ("adapt", "adapt")):
        paths = sorted((out / directory).glob("*/completed.json"))
        assert len(paths) == (40 if kind == "pretrain" else 150)
        for path in paths:
            meta = js(path)
            runs.append({"kind": kind, "run_id": path.parent.name,
                         **{key: meta[key] for key in ("target", "background", "seed", "condition", "updates", "input_tokens", "supervised_tokens", "sample_exposures", "seconds", "new_compute")},
                         "timer_scope": "training, full train/dev evaluation, checkpoint save and run bookkeeping"})
    summaries = []
    for kind in ("pretrain", "adapt"):
        values = [r for r in runs if r["kind"] == kind]
        summaries.append({"kind": kind, "runs": len(values),
                          **{key: sum(r[key] for r in values) for key in ("updates", "input_tokens", "supervised_tokens", "sample_exposures", "seconds")}})
    timings = [js(path) for path in sorted(out.glob("timing_*.json"))]
    ps.write_csv(out / "compute_by_run.csv", runs)
    ps.write_csv(out / "compute_summary.csv", summaries)
    ps.write_csv(out / "stage_timings.csv", timings)
    return summaries, timings


def source_result_findings(learning, contexts, summaries):
    """Describe all repeats and explicit counterexamples; no causal modeling."""
    def rows(target, background, condition, task):
        return [r for r in learning if (r["target"], r["background"], r["condition"], r["task"])
                == (target, background, condition, task)]

    def average_field(values, field):
        return ps.average(float(r[field]) for r in values)

    fp = [r for r in contexts if r["target"] == "fixed_points"]
    ldr = [r for r in contexts if r["target"] == "longest_decreasing_run"]
    af0 = rows("fixed_points", "g0_v3", "B", "anti_fixed_points")
    af2 = rows("fixed_points", "g2_deficiencies", "B", "anti_fixed_points")
    lir0 = rows("longest_decreasing_run", "g0_v3", "B", "longest_increasing_run")
    lir1 = rows("longest_decreasing_run", "g1_recoils", "B", "longest_increasing_run")
    fp_source_target_opposed = sum(r["B_candidate_dev_change"] < 0 and r["delta_B_minus_A"] > 0 for r in fp)
    dev_by_key = {(r["target"], int(r["seed"]), r["background"], r["condition"], r["task"]): float(r["multi4800_dev_accuracy"]) for r in learning}
    peak_changes = [(r["seed"], condition,
                     dev_by_key["longest_decreasing_run", r["seed"], "g1_recoils", condition, "peaks"]
                     - dev_by_key["longest_decreasing_run", r["seed"], "g0_v3", condition, "peaks"])
                    for r in ldr for condition in ("B", "C")]
    peak_positive = "; ".join(f"{condition}/seed{seed}: {pp(value)}pp" for seed, condition, value in peak_changes if value > 0)
    ldr_exceptions = "; ".join(
        f"seed{r['seed']}: LIR source dev changes by {pp (r['B_candidate_dev_change'])}pp, target B by {pp (r['delta_B_minus_A'])}pp and C by {pp (r['delta_C_minus_A'])}pp, yet the primary interaction is {pp (r['delta_B_minus_C'])}pp"
        for r in ldr if r["B_candidate_dev_change"] < 0 and r["delta_B_minus_C"] > 0)
    component = {(r["target"], r["metric"]): r["mean"] for r in summaries}
    return f"""The independent description of source learning before testing is saved in [SOURCE_LEARNING_NOTES.md](SOURCE_LEARNING_NOTES.md); its audit records that TEST_COMPLETE did not yet exist. Only the following section juxtaposes source records with frozen target results, distinguishing evidence from interpretation.

**FP: joint improvement in B/C appears again, but the prespecified positive replacement interaction does not reproduce.** Mean anti_fixed_points source dev changes from {pct (average_field (af0, 'multi4800_dev_accuracy'))} to {pct (average_field (af2, 'multi4800_dev_accuracy'))}, both near 99%. In {fp_source_target_opposed}/5 repeats this source dev decreases while B target accuracy increases. Failure to learn the related candidate in the original background therefore cannot uniformly explain this target, and candidate-source accuracy cannot directly proxy target benefit. Mean background changes in B−A and C−A are {pp (component['fixed_points', 'delta_B_minus_A'])}/{pp (component['fixed_points', 'delta_C_minus_A'])}pp. Both improve in all five repeats, but C improves more on average; the primary interaction is {pp (component['fixed_points', 'delta_B_minus_C'])}pp. deficiencies is well learned in both B/C new backgrounds, and both admit the formula for cross-input joint recovery of FP. This is an important design fact, but the round does not prove that the model used the formula, so joint gains cannot be directly attributed to this pathway.

**LDR: lower source learning accompanies lower B target benefit, but does not establish an interaction mechanism.** Mean LIR source dev changes from {pct (average_field (lir0, 'multi4800_dev_accuracy'))} to {pct (average_field (lir1, 'multi4800_dev_accuracy'))}, decreasing in all five repeats. New-background mean train is {pct (average_field (lir1, 'multi4800_train_accuracy'))}, and mean development majority baseline is {pct (average_field (lir1, 'multi4800_dev_majority_accuracy'))}. Both train and development are low, supporting insufficient source learning under this budget rather than only a generalization gap after training fit. B target accuracy also decreases in all five repeats, with mean B−A background change {pp (component['longest_decreasing_run', 'delta_B_minus_A'])}pp. C−A simultaneously changes by {pp (component['longest_decreasing_run', 'delta_C_minus_A'])}pp, leaving mean primary interaction {pp (component['longest_decreasing_run', 'delta_B_minus_C'])}pp. Important counterexamples are retained: {ldr_exceptions}. The interaction depends on both B and C; looking only at declining B-source learning misses this.

**Common sources are not uniformly harmed.** Across the two LDR conditions,10 same-task peaks background differences contain {sum (value < 0 for _, _, value in peak_changes)} decreases and {sum (value > 0 for _, _, value in peak_changes)} increases; positive entries are {peak_positive}. Thus the v4 observation that peaks always declined does not reproduce entry by entry. Changes in common sources and successions depend on seed/condition; the composition effect is not a simultaneous decline in all sources.

Existing records support these concurrent changes, low training-learning levels, and counterexamples. They cannot distinguish shared computational structure, optimization speed/gradient competition, answer distributions, use of joint relations, or adaptation-stage plasticity. No mediator interventions, gradient experiments, or explicit transformed-input queries were performed, and no mechanism was established."""


def write_report(out, differences, summaries, selected, aggregate, contexts, learning, common, costs, timings, plan):
    old = ps.read_csv(HERE / "exploratory" / "exploratory_paired_differences.csv")
    comparisons = []
    primary = [row for row in summaries if row["primary"]]
    judgments = []
    for row in primary:
        values = [float(r["delta_B_minus_C"]) for r in old if r["target"] == row["target"] and r["new_background"] == row["new_background"]]
        comparisons.append({"target": row["target"], "original_background": row["original_background"], "new_background": row["new_background"],
                            "exploratory_n": 3, "exploratory_mean_interaction": ps.average(values), "exploratory_sample_sd": ps.sample_sd(values),
                            "confirmation_n": 5, "confirmation_mean_interaction": row["mean"], "confirmation_sample_sd": row["sample_sd"],
                            "confirmation_ci95_lower": row["ci95_lower"], "confirmation_ci95_upper": row["ci95_upper"],
                            "confirmation_family95_lower": row["family95_lower"], "confirmation_family95_upper": row["family95_upper"],
                            "pooled": False})
        individual = row["ci95_lower"] > 0 if row["prespecified_direction"] > 0 else row["ci95_upper"] < 0
        family = row["family95_lower"] > 0 if row["prespecified_direction"] > 0 else row["family95_upper"] < 0
        if not row["mean_matches_direction"]:
            verdict = "The prespecified mean direction did not reproduce"
        elif family:
            verdict = "The mean direction reproduced, and simultaneous intervals for both contrasts exclude 0 under the model assumptions"
        elif individual:
            verdict = "The mean direction reproduced; the individual t interval excludes 0, but the simultaneous interval still includes 0"
        else:
            verdict = "The mean direction reproduced, but individual and simultaneous intervals include 0; direction and magnitude remain unstable"
        judgments.append (f"- **{row['target']}**: {verdict}. Mean interaction {pp (row['mean'])}pp; {row['direction_matching_count']}/5 repeats match the prespecified direction. 95% paired-t interval [{pp (row['ci95_lower'])}, {pp (row['ci95_upper'])}]pp; two-contrast Bonferroni simultaneous interval [{pp (row['family95_lower'])}, {pp (row['family95_upper'])}]pp.")
    ps.write_csv(out / "confirmation_vs_exploration.csv", comparisons)
    primary_table = table(["Target", "v4 exploratory mean pp", "v5 confirmation mean±sample SD pp", "Per-repeat range pp", "95% paired-t interval pp", "Two-contrast simultaneous interval pp"],
                          [[r["target"], pp(c["exploratory_mean_interaction"]), f"{pp(r['mean'])} ± {100*r['sample_sd']:.2f}",
                            f"[{pp(r['min'])}, {pp(r['max'])}]", f"[{pp(r['ci95_lower'])}, {pp(r['ci95_upper'])}]",
                            f"[{pp(r['family95_lower'])}, {pp(r['family95_upper'])}]"] for r, c in zip(primary, comparisons)])
    repeat_table = table(["Target", "New repeat/initialization seed", "Δ(B−A) pp", "Δ(C−A) pp", "Primary Δ(B−C) pp"],
                         [[r["target"], r["seed"], *(pp(r[m]) for m in ps.METRICS)] for r in differences])
    level_table = table(["Target", "Background", "A accuracy", "B accuracy", "C accuracy", "Target majority baseline"],
                        [[contrast["target"], background,
                          *(pct(next(r["mean"] for r in aggregate if r["target"] == contrast["target"] and r["background"] == background and r["condition"] == c)) for c in ("A", "B", "C")),
                          pct(next(r["majority_accuracy_mean"] for r in aggregate if r["target"] == contrast["target"] and r["background"] == background))]
                         for contrast in plan["contrasts"] for background in (contrast["original_background"], contrast["new_background"])])
    diagnostic_table = table(["Target", "Two-sided sign-flip p", "Two-sided sign p", "Direction matches/nonzero n", "Leave-one-out mean range pp"],
                            [[r["target"], f"{r['sign_flip_two_sided_p']:.4f}", f"{r['sign_test_two_sided_p']:.4f}",
                              f"{r['direction_matching_count']}/{r['sign_test_nonzero_n']}",
                              f"[{pp(r['leave_one_mean_min'])}, {pp(r['leave_one_mean_max'])}]"] for r in primary])
    source_context_table = table(["Target", "Repeat", "Target interaction pp", "B candidate source dev background difference pp", "C candidate source dev background difference pp", "Max absolute unchanged-common-source dev difference pp"],
                                 [[r["target"], r["seed"], pp(r["delta_B_minus_C"]), pp(r["B_candidate_dev_change"]),
                                   pp(r["C_candidate_dev_change"]), f"{100*r['unchanged_common_max_abs_dev_change']:.2f}"] for r in contexts])
    source_aggregate = []
    for contrast in plan["contrasts"]:
        for background in (contrast["original_background"], contrast["new_background"]):
            for condition in ("B", "C"):
                matched = [r for r in learning if r["target"] == contrast["target"] and r["background"] == background and r["condition"] == condition and int(r["slot"]) == 0]
                source_aggregate.append([contrast["target"], background, condition, matched[0]["task"],
                                         pct(ps.average(r["multi4800_train_accuracy"] for r in matched)),
                                         pct(ps.average(r["multi4800_dev_accuracy"] for r in matched)),
                                         f"[{pct(min(r['multi4800_dev_accuracy'] for r in matched))}, {pct(max(r['multi4800_dev_accuracy'] for r in matched))}]",
                                         pct(ps.average(r["multi4800_dev_majority_accuracy"] for r in matched))])
    source_summary_table = table(["Target", "Background", "Condition", "Candidate source", "Mean train", "Mean dev", "Dev range", "Dev majority baseline"], source_aggregate)
    cost_table = table(["Stage", "Runs", "Updates", "Input tokens", "Supervised tokens", "Run timer minutes"],
                       [[r["kind"], r["runs"], r["updates"], r["input_tokens"], r["supervised_tokens"], f"{r['seconds']/60:.2f}"] for r in costs])
    seed_table = table(["Repeat", "Initialization seed", "Train/dev generation seed", "Test generation seed"],
                       [[f"R{i+1}", r["seed"], r["data_generation_seed"], r["test_data_seed"]] for i, r in enumerate(plan["repetitions"])])
    lr_counts = dict(Counter(r["lr"] for r in selected))
    audit_path = out / "FINAL_AUDIT.json"
    audit_passed = audit_path.exists() and js(audit_path).get("status") == "passed"
    audit_text = "The final independent audit passed; see FINAL_AUDIT.json for complete status and itemized records" if audit_passed else "The final independent audit has not passed; run audit_final.py before treating delivery as complete"
    total_seconds = sum(r["seconds"] for r in costs)
    stage_seconds = sum(r["seconds"] for r in timings)
    main_lines = "\n".join(judgments)
    source_max = max(abs(float(r["B_minus_C_dev_accuracy"])) for r in common)
    source_findings = source_result_findings(learning, contexts, summaries)
    report = f"""# pilot_v5: confirmation of fixed background-interaction hypotheses

This round completes two fixed contrasts proposed from v4 exploration, with five new paired data/initialization repeats. The primary quantity is computed within each repeat: `Δ(B−C)=(B−C)_new_background−(B−C)_original_background`. It is not a comparison of separate SDs for two background means.

{main_lines}

This validation is limited to these two preselected cases at n=10, 256 target examples, and fixed budgets. The old three seeds are exploratory; the new five repeats are confirmatory. They are not pooled as eight repeats. Uniform new tests were generated only after training and development selection finished; all negative gains, unlearned tasks, and inconsistent results are retained.

## Recompute exploration before freezing confirmation

All targets' g1−g0 and g2−g0 differences were recomputed from v4's 42 unique selected raw rows. The 12 per-seed background differences and plots are in the [exploratory analysis](exploratory/EXPLORATORY_ANALYSIS_ZH.md). FP g2−g0 exploratory mean Δ(B−A)/Δ(C−A)/Δ(B−C) is +16.59/+12.22/+4.36pp, with primary interactions +1.42/+0.34/+11.33pp by seed. LDR g1−g0 is −11.56/+0.44/−12.00pp, with primary interactions −20.07/+2.00/−17.92pp. Shared A cancels exactly in background differences.

Prespecified H1: fixed_points, original→deficiencies background, expecting a positive primary interaction. H2: longest_decreasing_run, original→recoils background, expecting a negative primary interaction. Directions come from v4 exploration, not independent theoretical predictions. This round does not confirm the other two background contrasts, change directions from new results, switch to one-sided tests, or add repeats.

| Target | Candidate B / candidate C | Original three common sources | New three common sources |
| --- | --- | --- | --- |
| fixed_points | anti_fixed_points / successions | descents, peaks, left_to_right_maxima | descents, deficiencies, left_to_right_maxima |
| longest_decreasing_run | longest_increasing_run / left_to_right_maxima | descents, peaks, successions | recoils, peaks, successions |

Candidates occupy fixed slot0 and common sources slots 1–3. Neither target enters any pretraining group. Mathematical task relations and instance isolation are distinct checks. Group definitions and statistical methods were hash-frozen before new formal training; see the [frozen configuration](artifacts/FROZEN_PROTOCOL.json) and [stats_plan.json](stats_plan.json).

## Confirmation results from five new repeats

{primary_table}

{repeat_table}

![Primary interactions and intervals by repeat](artifacts/figures/confirmation_interactions_by_repeat.png)

![B−A/C−A components of the primary interaction](artifacts/figures/confirmation_background_components.png)

The following means over five repeats provide context on accuracy levels and majority baselines. A is trained and selected once per target/repeat and repeated in background display tables. The 60 display rows correspond to 50 unique selected models, not 60 independent selections.

{level_table}

Intervals use the five paired interaction differences directly, with SD denominator n−1. Individual intervals use df=4 and t=.975 quantile 2.776445. Bonferroni simultaneous intervals for the two primary contrasts have 97.5% marginal coverage each and critical value 3.495406. The latter do not require independence between the two target contrasts, but t coverage still assumes approximately independent normal repeat-level values. Five repeats cannot test normality or tails. Distinct RNG/initialization streams also do not imply strictly IID pools, because cross-block orbit exclusion creates finite-population sampling-without-replacement dependence. Intervals include finite-test-pool random variation and do not cover all task families.

{diagnostic_table}

The sign-flip diagnostic enumerates 32 sign assignments and requires independent sign symmetry around zero; it is not a treatment-randomization test guaranteed by this design. The sign test additionally assumes independent repeats with positive/negative probability 1/2 and concerns sign/median tendencies rather than the mean. With five nonzero repeats, both have minimum two-sided p=0.0625, so conventional significance cannot be promised. Leave-one-out means describe sensitivity to one repeat and do not replace the primary analysis. Auxiliary Δ(B−A)/Δ(C−A) and source learning are descriptive only, without expanding the testing family or selecting significant findings afterward.

## Joint mathematical relations and data isolation

Actual task implementations and independent checks are documented in the [mathematical relations](mathematics/README.md) and [audit results](mathematics/MATH_AUDIT.json). The inventory distinguishes formulas/proofs, empirical observations, and unchecked claims. Exhaustive and random larger-n checks validate implementations; finite verification is not a general proof. In particular, distinguish recovering a target jointly from source answers on the same input from recovering it when additional transformed-input queries are allowed.

At each position i, p(i) is exactly one of equal to, greater than, or less than i, so `fixed_points(p)+exceedances (p)+deficiencies (p)=n`. Same-input exceedances and deficiencies answers jointly determine fixed_points. The new FP background contains deficiencies but not exceedances, so this identity alone does not establish same-input joint recovery by the current group. Also, `exceedances (p)=deficiencies (inverse(p))`; if an extra inverse-permutation query is allowed, `fixed_points(p)=n−deficiencies (p)−deficiencies (inverse(p))`. This round supplies no explicit paired queries or joint-reasoning supervision.

**Both B and C in the FP deficiencies background have this exact cross-input joint relation; C is not a control without target mathematical relations.** The mathematical audit provides n=10 counterexamples for all 12 v4 groups: equal n and equal four-source scalar answers on the same input do not uniquely recover the target. Restricting queries to four sources × eight I/R/C transformations, or 32 answers, all six B groups recover the target from transformed candidate queries, and FP g2 C recovers it from two deficiencies queries. Each of the remaining five C groups has an n=10 insufficiency counterexample. This excludes only the specified answer-query model, not computation from raw input or other unchecked relations. See [TARGET_JOINT_DERIVABILITY.csv](mathematics/audit/TARGET_JOINT_DERIVABILITY.csv) and [explicit counterexamples](mathematics/audit/COUNTEREXAMPLES.csv).

Verification completed exhaustive S1…S7 coverage of 5913 inputs plus 4096 n=10 fixtures. Twenty definitions under eight transformations received 1,601,440 independent label comparisons, and 28 formulas/bounds each received 10,009 checks. Fixtures validate mathematical implementations only; they were not used for training or transfer-configuration selection. Empirically observed associations and unchecked mechanisms are separately recorded in [OBSERVED_ASSOCIATIONS.csv](mathematics/audit/OBSERVED_ASSOCIATIONS.csv) and [UNCHECKED_CLAIMS.csv](mathematics/audit/UNCHECKED_CLAIMS.csv).

The related candidate anti_fixed_points connects to fixed_points through complement/reversal input transforms; LIR/LDR connect through complement or reversal. descents, runs, peaks/valleys also share local comparison structure. These are legitimate mathematical relations, not automatically leakage. Leakage checks concern actual train/dev/test inputs and full inverse/reverse/complement orbit overlap. Whether a group's complete source scalar answers recover a target, and whether transformed queries are required, is specified group by group in the mathematical inventory and counterexamples. Unchecked relations are not declared absent.

## Auxiliary source-learning results and limits on mechanisms

Each four-source run has 4800 steps with batch64,16 examples per task per step, and exactly 25% sampling. Each task receives 76800 exposures,2150400 input tokens, and 153600 supervised tokens. Actual unique-input counts, stream hashes, length-specific (n=10) train/dev accuracy, answer CE, EOS CE, and majority baselines are retained in [source_learning_table.csv](artifacts/source_learning_table.csv) and [source_exposures.csv](artifacts/source_exposures.csv).

{source_summary_table}

{source_context_table}

These 10 entries cover all repeats, not cases selected by transfer magnitude. Candidate-source columns compare the same task's development accuracy across backgrounds; the common-source maximum includes only unchanged task definitions. Replaced peaks/deficiencies or descents/recoils tasks have their levels/baselines juxtaposed, without treating different-task accuracy differences as learning changes. The maximum absolute within-background common-task B/C development difference is {100*source_max:.2f}pp; see [common_source_BC.csv](artifacts/common_source_BC.csv) for full records.

Source records support descriptions that composition changes accompany changed source learning and that some sources remain unlearned under the current budget. They do not prove that source-accuracy changes cause transfer interactions. Sources and targets may jointly depend on optimization, shared computation, answer distributions, representation competition, and finite samples; these mediators were not independently manipulated. Mathematical derivability does not mean the Transformer learned or used the formula. Repeats with opposing source/transfer changes are fully retained, without post hoc explanations limited to intuitive cases.

{source_findings}

Old v3 single-task records are copied as historical references in [historical_single_references.csv](artifacts/historical_single_references.csv); no single-task retraining was performed. They use old seeds/pools and are not strictly paired controls for new multitask runs. Single1200 matches only per-task exposure, with different LR history and total updates; single4800 has four times the multitask per-task exposure. New-multitask minus old-single-task gaps therefore cannot directly identify composition interference. Complete new source and target curves are retained; selected-LR train/dev plots are in the [figures directory](artifacts/figures).

## Configuration, actual budget, and reproduction

{seed_table}

Every repeat uses a freshly initialized model and an independent data-seed stream, without reusing old weights as new repeats. The five new repeats are isolated from each other and all v1–v4 inputs using the same full transformation-orbit rule. Each block has source_train8192/source_dev1024/target_train256/target_dev1024 and test 2048. Targets/backgrounds/conditions share corresponding inputs within a block; task definitions determine target labels. Source sampling/dropout seed=initialization seed+1000, adaptation=+2000. Full generation order and data hashes are in seed_provenance, DATA_READY, and test_data_audit.

The model remains a4-layer Transformer, width256,8 heads, MLP4, dropout.1, tied input/output embedding, vocabulary188, PermWorld one-line, n=10, with 3,240,448 parameters. Each training forward has 28 tokens, with supervision only on answer/EOS; prefix tokens do not contribute loss. B/C each receive 4800 steps:100-step warmup to 3e−4, cosine decay to 3e−5 at 2400, then constant through 4800. Each adaptation condition retains three LRs 3e−4/1e−4/3e−5,800 steps each,100-step warmup and cosine decay to 0.1 of the initial value. No new hyperparameter search was added.

Full train/dev evaluation occurs every 100 steps, excluding step0. Selection prioritizes highest development answer accuracy, lowest development answer CE, earliest step, then fixed LR order. LR counts for 50 unique selected models: {lr_counts}. Adaptation strictly loads source parameters, restores the target token's tied embedding/output row to the block's step0 value, and uses fresh AdamW state. AdamW wd=.01, betas=.9/.999, eps=1e−8, clip1; BF16 training, FP32 weights/evaluation, TF32 disabled.

All 50 development selections are frozen before generating five new test pools. All 300 best/endpoint results across LRs are retained diagnostically, without test-based model reselection. Primary analysis uses only 50 selected rows; endpoint comparisons are in [selected_vs_endpoint.csv](artifacts/selected_vs_endpoint.csv). {audit_text}. Commands, file responsibilities, and resumption instructions are in the [README](README.md).

{cost_table}

New training timers total {total_seconds/60:.2f} minutes; engine-stage wall time totals {stage_seconds/60:.2f} minutes. Training timers include training, full train/dev evaluation, checkpoint saving, and bookkeeping, not pure GPU time. The two measures are not added; independent auditing and plotting are separate. All 40 pretraining runs and 150 adaptation candidates are new, totaling 312000 updates and 559104000 input tokens. Budgets were not extended, and no new single-task training was performed. Checkpoints/configuration are retained for later CKA integration.

## Next-stage task-family expansion plan and stopping point

See [TASK_FAMILY_PLAN.md](TASK_FAMILY_PLAN.md), frozen before new transfer results. Prioritize existing cycle-decomposition, global ordered-subsequence, and permutation-block families to provide different computational references for studied local comparisons and position-inequality counts. Next add displacement/pairwise-order and pattern counts; parameter instances are not independent families. Future genuinely new families include undirected connectivity/cycle rank, finite-field rank/kernel with singular matrices, and quadratic/bilinear forms over odd-characteristic fields. These new generators are not implemented or certified in this round. Each family should receive independent oracle checks and single-task development before compositions are frozen, retaining unlearned candidates. Two preselected cases cannot establish a universal ranking of families.

The platform should separate mathematical task_id, parameters, encoding, and input transforms. Single-task relations record domain conditions, direction, input actions, and answer maps. Joint relations use multi-input/multi-source hyperedges to record required source sets, query transformations, formulas/proofs, and verification scope; observed associations and unchecked markers are stored separately. Generators reuse base-instance→transform→label oracle→uniform split/orbit deduplication→encoding, recording versions, seeds, actual pool hashes, sampling slots, and exposures. Legitimate relation-derived tasks are not automatically leakage; instance isolation and relation categories are audited separately.

This round completes only five new repeats for the two prespecified contrasts. Effect direction, magnitude uncertainty, and mechanistic conjectures are reported separately. No more background variants for these targets, extra repeats, training budgets, or learning rates are added, pending further guidance.

## Main deliverables

- [Paired differences by repeat](artifacts/confirmation_paired_differences.csv), [Primary/auxiliary summaries](artifacts/confirmation_summary.csv), [Exploration and confirmation separately](artifacts/confirmation_vs_exploration.csv).
- [50 unique primary results](artifacts/results_selected.csv), [All 300 candidate best/endpoint results](artifacts/results_all_candidates.csv); Raw CSV per repeat: [101](artifacts/results_seed101.csv), [211](artifacts/results_seed211.csv), [307](artifacts/results_seed307.csv), [401](artifacts/results_seed401.csv), [503](artifacts/results_seed503.csv).
- [Source learning](artifacts/source_learning_table.csv), [Actual exposures](artifacts/source_exposures.csv), [Same-source changes across backgrounds](artifacts/source_background_changes.csv), [Source and target interactions side by side](artifacts/source_interaction_context.csv).
- [Statistical audit](artifacts/confirmation_statistics_audit.json), [Final independent audit](artifacts/FINAL_AUDIT.json), [Mathematical verification](mathematics/MATH_AUDIT.json), [Frozen statistical plan](stats_plan.json).
"""
    path = HERE / "PILOT_V5_REPORT.md" if out == (HERE / "artifacts").resolve() else out / "PILOT_V5_REPORT.md"
    path.write_text(resolve_report_links(report, path, out), encoding="utf-8")
    key_results = {"selected_unique": 50, "display_rows": 60, "candidate_test_rows": 300,
                   "primary_summaries": primary, "exploration_confirmation": comparisons,
                   "new_training_timer_seconds": total_seconds, "engine_stage_seconds": stage_seconds,
                   "statistics_plan_sha256": ps.sha256(HERE / "stats_plan.json"),
                   "old_and_new_pooled": False, "further_backgrounds_added": False}
    (out / "key_results.json").write_text(json.dumps(key_results, indent=2) + "\n", encoding="utf-8")
    return path


def confirmation(out):
    out = out.resolve()
    assert (out / "TEST_COMPLETE.json").exists(), "No completed report before all tests finish."
    plan = js(HERE / "stats_plan.json")
    selected, candidates = ps.read_csv(out / "results_selected.csv"), ps.read_csv(out / "results_all_candidates.csv")
    assert len(selected) == 50 and len(candidates) == 300
    assert len({cell_key(r) for r in selected}) == 50 and sum(truth(r["selected_primary"]) for r in candidates) == 50
    assert all(truth(r["selected_primary"]) for r in selected)
    assert sum(r["condition"] == "A" for r in selected) == 10
    (out / "figures").mkdir(exist_ok=True)
    differences, summaries = ps.run("confirmation", out / "results_selected.csv", out)
    displayed, deltas, aggregate, endpoints = result_tables(out, selected, candidates, plan)
    confirmation_plots(out, differences, summaries, plan)
    curves, learning, contexts, common = source_tables(out, differences, plan)
    learning_plots(out, curves, selected, plan)
    costs, timings = compute_tables(out)
    report = write_report(out, differences, summaries, selected, aggregate, contexts, learning, common, costs, timings, plan)
    print(json.dumps({"status": "complete", "report": str(report), "primary_contrasts": 2, "paired_repeats": 5,
                      "selected_unique": 50, "candidate_test_rows": 300, "figures": len(list((out / "figures").glob("*.png")))}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("exploratory", "confirmation"))
    parser.add_argument("--out", type=Path, default=HERE / "artifacts")
    args = parser.parse_args()
    if args.mode == "exploratory":
        exploratory()
    else:
        confirmation(args.out)


if __name__ == "__main__":
    main()
