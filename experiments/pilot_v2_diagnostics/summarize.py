"""Summarize saved v2 artifacts only: no model import, training, or forward pass."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "artifacts"
FIG = OUT / "figures"
REPORT = HERE / "DIAGNOSTIC_REPORT.md"
OLD = HERE.parent / "permworld_pilot" / "runs" / "pilot_v1"
GROUPS = [("A", 0), ("B", 2400), ("C", 2400), ("B", 4800), ("C", 4800)]
PROTOCOLS = [
    "original_v1_endpoint_new_test",
    "replay_fixed_lr_endpoint",
    "dev_selected",
    "selected_lr_endpoint",
]
COLORS = {"A": "#606c80", "B": "#167eab", "C": "#d68434"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def integer(row, key):
    return int(float(row[key]))


def number(row, key):
    return float(row[key])


def pct(value):
    return f"{100 * float(value):.2f}%"


def pp(value):
    return f"{100 * float(value):+.2f}"


def mdtable(headers, rows):
    return "\n".join([
        "| " + " | ".join(map(str, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ])


def group_key(row):
    return integer(row, "seed"), row["condition"], integer(row, "pretrain_updates")


def save_fig(fig, name):
    fig.savefig(FIG / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def source_tables():
    rows = read_csv(OUT / "source_learning_2400.csv")
    for path in sorted((OUT / "extended").glob("*.csv")):
        rows.extend(read_csv(path))
    rows.sort(key=lambda r: (integer(r, "seed"), r["condition"], r["task"], integer(r, "step"), r["split"]))
    write_csv(OUT / "source_learning_all.csv", rows)
    endpoint = [r for r in rows if integer(r, "step") in (2400, 4800)]
    write_csv(OUT / "source_learning_endpoints.csv", endpoint)
    grouped = defaultdict(dict)
    for row in endpoint:
        grouped[(integer(row, "seed"), row["condition"], integer(row, "step"), row["task"])][row["split"]] = row
    wide = []
    for (seed, condition, budget, task), pair in sorted(grouped.items()):
        row = dict(seed=seed, condition=condition, pretrain_updates=budget, task=task, length=10)
        for split in ("train", "dev"):
            for metric in ("accuracy", "answer_ce", "two_token_ce", "eos_ce", "eos_accuracy", "majority_accuracy", "majority_label", "count"):
                row[f"{split}_{metric}"] = pair[split][metric]
        wide.append(row)
    write_csv(OUT / "source_learning_table.csv", wide)
    for seed in sorted({integer(r, "seed") for r in rows}):
        for condition in ("B", "C"):
            subset = [r for r in rows if integer(r, "seed") == seed and r["condition"] == condition]
            tasks = list(dict.fromkeys(r["task"] for r in subset))
            fig, axes = plt.subplots(len(tasks), 2, figsize=(11, 3 * len(tasks)), squeeze=False)
            for i, task in enumerate(tasks):
                for split, linestyle in (("train", "-"), ("dev", "--")):
                    values = sorted([r for r in subset if r["task"] == task and r["split"] == split], key=lambda r: integer(r, "step"))
                    x = [integer(r, "step") for r in values]
                    for j, metric in enumerate(("accuracy", "answer_ce")):
                        axes[i, j].plot(x, [number(r, metric) for r in values], linestyle, marker="o", label=split)
                baseline = next(number(r, "majority_accuracy") for r in subset if r["task"] == task and r["split"] == "dev")
                axes[i, 0].axhline(baseline, color="grey", linestyle=":", label="dev majority")
                axes[i, 0].set_ylim(0, 1.03)
                for j in range(2):
                    axes[i, j].set_title(task)
                    axes[i, j].set_xlabel("Pretraining updates")
                    axes[i, j].set_ylabel("Answer accuracy" if j == 0 else "Answer CE")
                    axes[i, j].grid(alpha=.2)
                    axes[i, j].legend(fontsize=8)
            fig.suptitle(f"Source learning: seed {seed}, condition {condition}; 2400 checkpoint + new continuation")
            fig.tight_layout()
            save_fig(fig, f"source_seed{seed}_{condition}.png")
    return rows, wide


def target_plots(curves, selections):
    for selection in selections:
        seed, condition, budget = int(selection["seed"]), selection["condition"], int(selection["budget"])
        lr = float(selection["lr"])
        subset = [r for r in curves if group_key(r) == (seed, condition, budget)]
        chosen = sorted([r for r in subset if number(r, "lr") == lr], key=lambda r: integer(r, "step"))
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for split, style in (("train", "-"), ("dev", "--")):
            for axis, metric in zip(axes, ("accuracy", "answer_ce")):
                axis.plot([integer(r, "step") for r in chosen], [number(r, f"{split}_{metric}") for r in chosen], style, marker="o", label=split)
                axis.axvline(int(selection["step"]), color="black", alpha=.25, linestyle=":")
        axes[0].axhline(number(chosen[0], "dev_majority_accuracy"), color="grey", linestyle=":", label="dev majority")
        axes[0].set_ylim(0, 1.03)
        for axis, label in zip(axes, ("Answer accuracy", "Answer CE")):
            axis.set(xlabel="Adaptation update", ylabel=label)
            axis.grid(alpha=.2)
            axis.legend()
        fig.suptitle(f"seed {seed}, {condition}, pretrain={budget}, selected LR={lr:g}; dev selection at {selection['step']}")
        fig.tight_layout()
        stem = f"seed{seed}_{condition}_p{budget}"
        save_fig(fig, f"target_selected_{stem}.png")

        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        for rate in sorted({number(r, "lr") for r in subset}, reverse=True):
            values = sorted([r for r in subset if number(r, "lr") == rate], key=lambda r: integer(r, "step"))
            for i, split in enumerate(("train", "dev")):
                for j, metric in enumerate(("accuracy", "answer_ce")):
                    axes[i, j].plot([integer(r, "step") for r in values], [number(r, f"{split}_{metric}") for r in values], marker=".", label=f"LR {rate:g}")
        for i, split in enumerate(("train", "dev")):
            for j, metric in enumerate(("accuracy", "answer_ce")):
                axes[i, j].set(xlabel="Adaptation update", ylabel=f"{split} {metric}")
                axes[i, j].grid(alpha=.2)
                axes[i, j].legend()
            axes[i, 0].set_ylim(0, 1.03)
        fig.suptitle(f"Equal-budget LR candidates: seed {seed}, {condition}, pretrain={budget}; no test-based selection")
        fig.tight_layout()
        save_fig(fig, f"target_lr_comparison_{stem}.png")


def protocol_tables(comparison, seeds):
    # A has zero pretraining. It is explicitly copied into both comparison panels,
    # never counted as a second independent run or included twice in a mean.
    expanded = []
    for row in comparison:
        budget = integer(row, "pretrain_updates")
        for panel in ((2400, 4800) if row["condition"] == "A" else (budget,)):
            expanded.append(dict(comparison_pretrain_budget=panel, reused_A0=int(row["condition"] == "A"), **row))
    write_csv(OUT / "protocol_comparison_paired.csv", expanded)
    grouped = defaultdict(list)
    for row in expanded:
        grouped[(int(row["comparison_pretrain_budget"]), row["protocol"], row["condition"])].append(row)
    averaged = []
    for (budget, protocol, condition), rows in sorted(grouped.items()):
        assert {integer(r, "seed") for r in rows} == set(seeds)
        out = dict(comparison_pretrain_budget=budget, protocol=protocol, condition=condition, n_seeds=len(rows), actual_pretrain_updates=0 if condition == "A" else budget, reused_A0=int(condition == "A"))
        for metric in ("test_accuracy", "test_answer_ce", "dev_accuracy", "dev_ce", "train_accuracy", "majority_accuracy"):
            values = [number(r, metric) for r in rows]
            out[metric + "_mean"] = mean(values)
            out[metric + "_sample_sd"] = stdev(values) if len(values) > 1 else 0.0
            out[metric + "_min"] = min(values)
            out[metric + "_max"] = max(values)
        for r in rows:
            out[f"seed{r['seed']}_test_accuracy"] = number(r, "test_accuracy")
        averaged.append(out)
    write_csv(OUT / "protocol_comparison_mean.csv", averaged)
    deltas = []
    for seed in seeds:
        for budget in (2400, 4800):
            for protocol in PROTOCOLS:
                rows = {r["condition"]: r for r in expanded if integer(r, "seed") == seed and int(r["comparison_pretrain_budget"]) == budget and r["protocol"] == protocol}
                if set(rows) != {"A", "B", "C"}:
                    continue
                for first, second in (("B", "A"), ("C", "A"), ("B", "C")):
                    deltas.append(dict(seed=seed, comparison_pretrain_budget=budget, protocol=protocol, contrast=f"{first}-{second}", test_accuracy_delta=number(rows[first], "test_accuracy") - number(rows[second], "test_accuracy"), dev_accuracy_delta=number(rows[first], "dev_accuracy") - number(rows[second], "dev_accuracy")))
    write_csv(OUT / "paired_deltas.csv", deltas)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    labels = {"original_v1_endpoint_new_test": "v1 endpoint\nnew test", "replay_fixed_lr_endpoint": "fixed LR\nreplay endpoint", "dev_selected": "dev-selected\nLR + checkpoint", "selected_lr_endpoint": "selected LR\nendpoint"}
    for axis, budget in zip(axes, (2400, 4800)):
        protocols = [p for p in PROTOCOLS if p != "original_v1_endpoint_new_test" or budget == 2400]
        for offset, condition in enumerate(("A", "B", "C")):
            values = [next(r for r in averaged if r["comparison_pretrain_budget"] == budget and r["protocol"] == p and r["condition"] == condition) for p in protocols]
            x = [i + (offset - 1) * .24 for i in range(len(protocols))]
            axis.bar(x, [r["test_accuracy_mean"] for r in values], width=.23, label=condition + (" (same A0)" if condition == "A" else ""), color=COLORS[condition], alpha=.8)
            for i, row in enumerate(values):
                axis.plot([x[i]] * len(seeds), [row[f"seed{s}_test_accuracy"] for s in seeds], ".", color="black", markersize=4)
        axis.set_xticks(range(len(protocols)), [labels[p] for p in protocols], fontsize=8)
        axis.set(title=f"Pretraining comparison budget = {budget}", ylabel="New isolated test answer accuracy", ylim=(0, 1))
        axis.grid(axis="y", alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("256 distinct target examples; mean of two paired seeds, dots are individual seeds")
    fig.tight_layout()
    save_fig(fig, "protocol_comparison.png")
    return averaged, deltas


def compute_costs():
    rows = []
    for path in sorted((OUT / "adapt").glob("*/completed.json")):
        m = read_json(path)
        rows.append(dict(stage="v2_adapt_candidate", seed=m["seed"], condition=m["condition"], pretrain_updates=m["budget"], lr=m["lr"], updates=m["updates"], input_tokens=m["input_tokens"], supervised_tokens=m["supervised_tokens"], sample_exposures=m["sample_exposures"], seconds=m["seconds"], new_compute=1, timing_scope="training, step0 and later train/dev measurements, candidate best checkpoint saves, and replay-difference check when applicable; initialization and final endpoint save excluded"))
    for path in sorted((OUT / "extended").glob("*.json")):
        m = read_json(path)
        rows.append(dict(stage="v2_pretrain_extension", seed=m["seed"], condition=m["condition"], pretrain_updates=4800, lr=m["lr"], updates=m["additional_updates"], input_tokens=m["additional_input_tokens"], supervised_tokens=m["additional_supervised_tokens"], sample_exposures=m["additional_updates"] * 64, seconds=m["seconds"], new_compute=1, timing_scope="continuation loop, source train/dev measurements, and CPU NumPy sampler replay verification; final save excluded"))
    for m in read_csv(OLD / "actual_budgets.csv"):
        if m["stage"] == "pretrain":
            rows.append(dict(stage="v1_pretraining_reused", seed=m["seed"], condition=m["condition"], pretrain_updates=m["updates"], lr="historical schedule", updates=m["updates"], input_tokens=m["input_tokens"], supervised_tokens=m["supervised_tokens"], sample_exposures=integer(m, "updates") * 64, seconds=m["seconds"], new_compute=0, timing_scope="historical v1 recorded training; not incurred again"))
    write_csv(OUT / "compute_costs.csv", rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["stage"]].append(row)
    totals = []
    for stage, group in sorted(grouped.items()):
        total = dict(stage=stage, runs=len(group), new_compute=group[0]["new_compute"])
        for key in ("updates", "input_tokens", "supervised_tokens", "sample_exposures", "seconds"):
            total[key] = sum(float(r[key]) for r in group)
        totals.append(total)
    write_csv(OUT / "compute_cost_totals.csv", totals)
    stages = [read_json(p) for p in sorted(OUT.glob("timing_*.json"))]
    if stages:
        write_csv(OUT / "stage_wall_times.csv", stages)
    replay = []
    for path in sorted((OUT / "adapt").glob("*/completed.json")):
        m = read_json(path)
        if m["replay_v1"]:
            replay.append(dict(seed=m["seed"], condition=m["condition"], pretrain_updates=m["budget"], lr=m["lr"], **m["replay_v1"], replay_final_state_hash=m["final_model_state_hash"]))
    write_csv(OUT / "replay_v1_weight_differences.csv", replay)
    return rows, totals, stages, replay


def replacement_table():
    histograms = read_csv(OUT / "replacement_distributions.csv")
    rows = []
    for seed in sorted({integer(r, "seed") for r in histograms}):
        distributions = {}
        for task in ("anti_fixed_points", "successions"):
            values = sorted([r for r in histograms if integer(r, "seed") == seed and r["task"] == task], key=lambda r: integer(r, "answer"))
            distributions[task] = [number(r, "probability") for r in values]
        tv = .5 * sum(abs(a - b) for a, b in zip(*distributions.values()))
        for task, probabilities in distributions.items():
            rows.append(dict(seed=seed, task=task, mean=sum(i * p for i, p in enumerate(probabilities)), entropy_bits=-sum(p * math.log2(p) for p in probabilities if p), replacement_total_variation=tv))
    write_csv(OUT / "replacement_matching_summary.csv", rows)
    return rows


def late_training_diagnostics(curves):
    """Use the same 400->800 window for every candidate, independent of selection."""
    grouped = defaultdict(dict)
    for row in curves:
        seed, condition, budget = group_key(row)
        grouped[(seed, condition, budget, number(row, "lr"))][integer(row, "step")] = row
    rows = []
    for (seed, condition, budget, lr), steps in sorted(grouped.items()):
        assert 400 in steps and 800 in steps
        before, after = steps[400], steps[800]
        row = dict(seed=seed, condition=condition, pretrain_updates=budget, lr=lr, start_step=400, end_step=800, window_rule="same fixed window for all candidates; no checkpoint/LR selection or test scores")
        for split in ("train", "dev"):
            for metric in ("accuracy", "answer_ce"):
                key = f"{split}_{metric}"
                row[f"{key}_400"] = number(before, key)
                row[f"{key}_800"] = number(after, key)
                row[f"{key}_change"] = number(after, key) - number(before, key)
        row["train_ce_decreased"] = int(row["train_answer_ce_change"] < -1e-12)
        row["dev_ce_increased"] = int(row["dev_answer_ce_change"] > 1e-12)
        row["dev_accuracy_decreased"] = int(row["dev_accuracy_change"] < -1e-12)
        row["train_ce_down_and_dev_ce_up"] = row["train_ce_decreased"] * row["dev_ce_increased"]
        row["train_ce_down_dev_ce_up_dev_accuracy_down"] = row["train_ce_down_and_dev_ce_up"] * row["dev_accuracy_decreased"]
        rows.append(row)
    assert len(rows) == 30
    write_csv(OUT / "late_training_diagnostics.csv", rows)
    summary = []
    for condition, budget in GROUPS:
        candidates = [r for r in rows if r["condition"] == condition and r["pretrain_updates"] == budget]
        item = dict(condition=condition, pretrain_updates=budget, candidate_count=len(candidates), start_step=400, end_step=800)
        for key in ("train_ce_decreased", "dev_ce_increased", "dev_accuracy_decreased", "train_ce_down_and_dev_ce_up", "train_ce_down_dev_ce_up_dev_accuracy_down"):
            item[key] = sum(r[key] for r in candidates)
        summary.append(item)
    write_csv(OUT / "late_training_summary.csv", summary)
    return rows, summary


def source_budget_diagnostics(sources):
    endpoints = defaultdict(dict)
    for row in sources:
        endpoints[(row["seed"], row["condition"], row["task"])][row["pretrain_updates"]] = row
    changes = []
    for (seed, condition, task), budgets in sorted(endpoints.items()):
        before, after = budgets[2400], budgets[4800]
        row = dict(seed=seed, condition=condition, task=task, from_updates=2400, to_updates=4800)
        for split in ("train", "dev"):
            for metric in ("accuracy", "answer_ce"):
                key = f"{split}_{metric}"
                row[f"{key}_2400"] = number(before, key)
                row[f"{key}_4800"] = number(after, key)
                row[f"{key}_change"] = number(after, key) - number(before, key)
        row["dev_accuracy_increased"] = int(row["dev_accuracy_change"] > 1e-12)
        row["dev_answer_ce_decreased"] = int(row["dev_answer_ce_change"] < -1e-12)
        changes.append(row)
    assert len(changes) == 16
    write_csv(OUT / "source_budget_changes.csv", changes)
    common_tasks = sorted(set(r["task"] for r in sources if r["condition"] == "B") & set(r["task"] for r in sources if r["condition"] == "C"))
    common = []
    for budget in (2400, 4800):
        for task in common_tasks:
            row = dict(pretrain_updates=budget, task=task)
            for condition in ("B", "C"):
                group = [r for r in sources if r["condition"] == condition and r["pretrain_updates"] == budget and r["task"] == task]
                row[f"{condition}_n_seeds"] = len(group)
                for split in ("train", "dev"):
                    for metric in ("accuracy", "answer_ce"):
                        row[f"{condition}_{split}_{metric}_mean"] = mean(number(r, f"{split}_{metric}") for r in group)
            row["B_minus_C_dev_accuracy"] = row["B_dev_accuracy_mean"] - row["C_dev_accuracy_mean"]
            row["B_minus_C_dev_answer_ce"] = row["B_dev_answer_ce_mean"] - row["C_dev_answer_ce_mean"]
            common.append(row)
    write_csv(OUT / "common_source_tasks_comparison.csv", common)
    return changes, common


def report(config, selections, sources, averaged, deltas, totals, stages, replay, replacements, late, late_summary):
    seeds = config["seeds"]
    comparison = read_csv(OUT / "protocol_comparison.csv")
    test_audit = read_json(OUT / "test_data_audit.json")
    smoke = read_json(OUT / "smoke.json")
    all_completed = [read_json(p) for p in sorted((OUT / "adapt").glob("*/completed.json"))]
    lr_selection_counts = [dict(lr=lr, selected_groups=sum(float(s["lr"]) == lr for s in selections), total_groups=len(selections), note="10 actual condition/seed/budget groups; A0 counted once per seed") for lr in config["learning_rates"]]
    write_csv(OUT / "selected_learning_rate_counts.csv", lr_selection_counts)
    lr_counts_text = ", ".join(f"{r['lr']:g} selected in {r['selected_groups']}/{r['total_groups']} groups" for r in lr_selection_counts)
    changed_lr_selections = [s for s in selections if float(s["lr"]) != config["learning_rates"][0]]
    changed_lr_text = "; ".join(f"seed{s['seed']} {s['condition']}/{s['budget']} selects {s['lr']:g}, {s['step']} steps" for s in changed_lr_selections) or "All groups select the original learning rate"

    def avg(protocol, budget, condition, metric="test_accuracy_mean"):
        return next(r[metric] for r in averaged if r["protocol"] == protocol and r["comparison_pretrain_budget"] == budget and r["condition"] == condition)

    def byseed(protocol, budget, condition, seed):
        return next(r for r in comparison if r["protocol"] == protocol and integer(r, "pretrain_updates") == budget and r["condition"] == condition and integer(r, "seed") == seed)

    delta_lines = []
    for budget in (2400, 4800):
        a = avg("dev_selected", budget, "A")
        b, c = avg("dev_selected", budget, "B"), avg("dev_selected", budget, "C")
        delta_lines.append(f"{budget} pretraining steps and development selection: B−A={pp(b-a)}, C−A={pp(c-a)}, B−C={pp(b-c)} percentage points.")
    gains = {c: avg("dev_selected", 2400, c) - avg("original_v1_endpoint_new_test", 2400, c) for c in ("A", "B", "C")}
    budget_effects = []
    for seed in seeds:
        for condition in ("B", "C"):
            for protocol in ("replay_fixed_lr_endpoint", "dev_selected", "selected_lr_endpoint"):
                before = byseed(protocol, 2400, condition, seed)
                after = byseed(protocol, 4800, condition, seed)
                budget_effects.append(dict(seed=seed, condition=condition, protocol=protocol, from_pretrain_updates=2400, to_pretrain_updates=4800, test_accuracy_2400=number(before, "test_accuracy"), test_accuracy_4800=number(after, "test_accuracy"), test_accuracy_delta=number(after, "test_accuracy") - number(before, "test_accuracy"), dev_accuracy_delta=number(after, "dev_accuracy") - number(before, "dev_accuracy")))
    write_csv(OUT / "target_budget_deltas.csv", budget_effects)
    selected_budget_effects = [r for r in budget_effects if r["protocol"] == "dev_selected"]
    heterogeneous_conditions = [condition for condition in ("B", "C") if min(r["test_accuracy_delta"] for r in selected_budget_effects if r["condition"] == condition) < 0 < max(r["test_accuracy_delta"] for r in selected_budget_effects if r["condition"] == condition)]
    endpoint_drops = []
    for s in selections:
        best = byseed("dev_selected", s["budget"], s["condition"], s["seed"])
        end = byseed("selected_lr_endpoint", s["budget"], s["condition"], s["seed"])
        endpoint_drops.append(dict(seed=s["seed"], condition=s["condition"], pretrain_updates=s["budget"], selected_lr=s["lr"], best_step=s["step"], best_train_accuracy=number(best, "train_accuracy"), endpoint_train_accuracy=number(end, "train_accuracy"), best_dev_accuracy=number(best, "dev_accuracy"), endpoint_dev_accuracy=number(end, "dev_accuracy"), best_dev_ce=number(best, "dev_ce"), endpoint_dev_ce=number(end, "dev_ce"), dev_accuracy_drop=number(best, "dev_accuracy") - number(end, "dev_accuracy"), dev_ce_increase=number(end, "dev_ce") - number(best, "dev_ce"), test_accuracy_drop=number(best, "test_accuracy") - number(end, "test_accuracy")))
    write_csv(OUT / "best_vs_endpoint.csv", endpoint_drops)
    early = sum(r["best_step"] < 800 for r in endpoint_drops)
    early_test_improved = sum(r["best_step"] < 800 and r["test_accuracy_drop"] > 1e-12 for r in endpoint_drops)
    early_test_worsened = sum(r["best_step"] < 800 and r["test_accuracy_drop"] < -1e-12 for r in endpoint_drops)
    overfit = [r for r in endpoint_drops if r["best_step"] < 800 and r["dev_ce_increase"] > 0 and r["dev_accuracy_drop"] > 0]
    late_ce_divergence = sum(r["train_ce_down_and_dev_ce_up"] for r in late)
    late_acc_deterioration = sum(r["train_ce_down_dev_ce_up_dev_accuracy_down"] for r in late)
    replay_exact = sum(bool(r["bitwise_same"]) for r in replay)
    low_source = [r for r in sources if r["pretrain_updates"] == 4800 and number(r, "dev_accuracy") < .9]
    source_changes, common_sources = source_budget_diagnostics(sources)
    source_dev_accuracy_improved = sum(r["dev_accuracy_increased"] for r in source_changes)
    source_dev_ce_improved = sum(r["dev_answer_ce_decreased"] for r in source_changes)
    historical = read_csv(OUT / "v1_observed_target_curves.csv")
    historical_ce_worse = historical_accuracy_worse = 0
    for seed in seeds:
        for condition in ("A", "B", "C"):
            selected = {integer(r, "step"): r for r in historical if integer(r, "seed") == seed and r["condition"] == condition}
            if 400 in selected and 800 in selected:
                historical_ce_worse += number(selected[800], "dev_answer_ce") > number(selected[400], "dev_answer_ce")
                historical_accuracy_worse += number(selected[800], "dev_accuracy") < number(selected[400], "dev_accuracy")
    selected_neg = [(budget, c) for budget in (2400, 4800) for c in ("B", "C") if avg("dev_selected", budget, c) < avg("dev_selected", budget, "A")]
    new_total_seconds = sum(float(r["seconds"]) for r in totals if r["new_compute"])
    extension = next(r for r in totals if r["stage"] == "v2_pretrain_extension")
    wall_total = sum(r["wall_seconds"] for r in stages)
    timed_stage_names = ", ".join(r["stage"] for r in stages)
    inventory_timing_note = "Inventory ran through a separate CLI without a timing_inventory record, so its duration is excluded from the wall-time total above." if not any(r["stage"] == "inventory" for r in stages) else "Inventory is included among the timed stages."
    checked = all(m["initialization"]["loaded_state_hash"] == m["initialization"]["source_state_hash"] and m["initialization"]["optimizer_state_reset"] and m["initialization"]["optimizer_entries"] == 0 for m in all_completed)
    target_rows_equal = all(len({m["initialization"]["target_row_hash"] for m in all_completed if m["seed"] == seed}) == 1 for seed in seeds)
    data_equal = all(len({m["initialization"]["train_sha256"] for m in all_completed if m["seed"] == seed}) == len({m["initialization"]["dev_sha256"] for m in all_completed if m["seed"] == seed}) == 1 for seed in seeds)
    schedule_equal = all(len({m["schedule_sha256"] for m in all_completed if m["seed"] == seed}) == 1 for seed in seeds)
    assert checked and target_rows_equal and data_equal and schedule_equal

    lines = [
        "# PermWorld second-round diagnostic report (2026-10-18)",
        "",
        f"Completed two paired seeds {seeds}, 30 learning-rate candidate adaptations of 800 updates each, 4 pretraining extensions of 2400 additional updates each, and evaluation on a new isolated test set generated after selections were frozen. The target is fixed_points with 256 distinct training examples; task composition, one-line encoding, and architecture follow pilot_v1. No tasks or targets were added.",
        "",
        "**" + " ".join(delta_lines) + "**",
        "",
        ("Some development-selected means remain below A; negative transfer is retained in the results. " if selected_neg else "All development-selected B/C means are at least as high as A, but there are only two seeds. ") + f"After doubling the budget, {len(low_source)}/16 source-task × seed × condition combinations still have development accuracy below 90%. The 90% threshold describes incomplete mastery diagnostically, not a universal difficulty standard across tasks. These results cannot attribute transfer mechanisms to the known mathematical relation.",
        "",
        "## Tasks, data, and fair comparison",
        "",
        "The target is fixed_points(p)=#{i:p(i)=i}, at fixed n=10. B pretrains on anti_fixed_points, descents, peaks, left_to_right_maxima; C replaces only anti_fixed_points with successions. The relation fixed_points(p)=anti_fixed_points(c(p)), c(p)(i)=11−p(i), is an identity under input complement. Successions is not registered as having this direct relation; mathematical unrelatedness is not asserted.",
        "",
        "Both seeds retain the first 256 v1 target training inputs and 1024 development inputs. B/C share 8192 pretraining inputs and 1024 source development inputs, with 16/64 examples per task in every batch. All adaptation candidates share same-seed data, sampling sequences, and a maximum 800 updates. No parameters are added: target-specific initialization restores only the existing target-task token's tied embedding/output row to the shared same-seed step0 value; numeric output parameters continue to transfer. Each adaptation creates AdamW afresh and verifies an empty state.",
        "",
        f"The new test set was generated after all 10 development selections were frozen, with 2048 inputs per seed. Complete identity/inverse/reverse/complement orbits exclude every v1 pool, including historical tests, and cross-seed new test inputs. Explicit orbit overlap is {test_audit['overlap']}. Within each seed, all conditions, candidates, and protocols share the same new test set. All tables below use this test set; old v1 scores are not pooled into averages. Original v1 endpoints are reevaluated on the new test. The old development set is reused and the question arose from historical exploration, so this remains diagnostic exploration.",
        "",
        "## Historical records and supplementary diagnostics",
        "",
        "Historical records retain sparse last-batch two-token pretraining losses, source endpoint accuracies, target development accuracy/answer CE every 400 updates (only steps 400 and 800), and last-batch training loss. They lack per-source historical train/dev loss trajectories, full-target-training trajectories, and intermediate checkpoints. This round recomputes full source train/dev accuracy, answer CE, two-token CE, EOS accuracy, and majority baselines by length from original checkpoints. Historical intermediate models cannot be recovered from endpoints; new trajectories are explicitly labeled replay rather than presented as historical measurements.",
        "",
        "The length-stratified baseline has only n=10 in this round. The majority label comes from the corresponding training pool, then its accuracy is measured on train/dev/test. Source-table answer CE uses the full 188-token vocabulary; training loss averages answer and EOS CE and must not be conflated with answer CE.",
        "",
        f"Strict weight-load hash checks passed={checked}; same-seed target-specific rows match={target_rows_equal}; target data hashes match={data_equal}; adaptation sampling hashes match={schedule_equal}; save/resume 4+4 updates are bitwise equal={smoke['resume_state_bitwise_equal']}. Original pretraining optimizer steps are 2400 and saved adaptation steps 800; new adaptation optimizer states start at 0.",
        "",
        "V1 did not save adaptation-start parameter fingerprints. Historical load/reset conclusions rely on code, saved optimizer steps, and reconstruction checks, not direct historical start-state hashes. Every v2 candidate saves initialization.json before running, recording pre/post-load hashes, target-row hashes, and empty optimizer state.",
        "",
        f"Across the 6 historical 256-example adaptations from step400 to 800, development answer CE rises in {historical_ce_worse} groups and development accuracy declines in {historical_accuracy_worse}. Worse CE may reflect confidence/calibration deterioration and does not always mean accuracy overfitting. Historical last-batch mixed two-token loss is not a substitute for full-training-set loss.",
        "",
        f"High-learning-rate protocol replays are bitwise identical to original v1 endpoints in {replay_exact}/{len(replay)} units. Maximum absolute parameter differences are in the [replay table](artifacts/replay_v1_weight_differences.csv)." + (" Replays without bitwise reproduction do not replace original checkpoints; new-test results for the originals are reported separately." if replay_exact < len(replay) else " The replayed full training curves can be associated with identical endpoint weights; sparse historical records remain separately available."),
        "",
        mdtable(["Seed", "Condition", "Pretraining updates", "Bitwise equal", "Maximum parameter difference"], [[r["seed"], r["condition"], r["pretrain_updates"], r["bitwise_same"], f"{r['parameter_max_abs_difference']:.6g}"] for r in replay]),
        "",
        "## Source learning and replacement difficulty",
        "",
        "The table gives original- and double-budget endpoints for each seed. Accuracy and CE use complete pools. Full two-token/EOS metrics and training majority baselines are in the [source-learning table](artifacts/source_learning_table.csv); extension measurements every 600 updates are in the [complete source-curve CSV](artifacts/source_learning_all.csv).",
        "",
        mdtable(["seed", "Condition/updates", "Task", "Train acc", "Dev acc", "Train CE", "Dev CE", "Dev majority"], [[r["seed"], f"{r['condition']}/{r['pretrain_updates']}", r["task"], pct(r["train_accuracy"]), pct(r["dev_accuracy"]), f"{float(r['train_answer_ce']):.4f}", f"{float(r['dev_answer_ce']):.4f}", pct(r["dev_majority_accuracy"])] for r in sources]),
        "",
        "Answer distributions use the same 8192 source training inputs; extending training does not change the input distribution:",
        "",
        mdtable(["seed", "Task", "Label mean", "Entropy (bits)", "B/C total variation"], [[r["seed"], r["task"], f"{r['mean']:.4f}", f"{r['entropy_bits']:.4f}", f"{r['replacement_total_variation']:.4f}"] for r in replacements]),
        "",
    ]
    for budget in (2400, 4800):
        b = [r for r in sources if r["pretrain_updates"] == budget and r["task"] == "anti_fixed_points"]
        c = [r for r in sources if r["pretrain_updates"] == budget and r["task"] == "successions"]
        lines.append(f"{budget} updates, mean development accuracy for the replaced tasks is anti_fixed_points={pct(mean(float(r['dev_accuracy']) for r in b))}, successions={pct(mean(float(r['dev_accuracy']) for r in c))}; their mean development answer CE is {mean(float(r['dev_answer_ce']) for r in b):.4f}/{mean(float(r['dev_answer_ce']) for r in c):.4f}. This is a learning difference under the current multitask optimization conditions, not an independent estimate of intrinsic task difficulty.")
        lines.append("")
    lines += [
        f"Among 16 paired source units from the original to doubled budget, development accuracy improves in {source_dev_accuracy_improved}/16 and development answer CE decreases in {source_dev_ce_improved}/16. Per-unit train/dev changes are in [source_budget_changes.csv](artifacts/source_budget_changes.csv). This records the actual impact of continued training while retaining units without mastery; improvement is not equivalent to having learned a task.",
        "",
        "The three common B/C sources also reach different learning levels in the two compositions. The following development results average both seeds; B/C use identical common-task inputs, labels, sampling ratios, and total budgets:",
        "",
        mdtable(["Pretraining updates", "Common task", "B dev acc", "C dev acc", "B−C pp", "B dev CE", "C dev CE"], [[r["pretrain_updates"], r["task"], pct(r["B_dev_accuracy_mean"]), pct(r["C_dev_accuracy_mean"]), pp(r["B_minus_C_dev_accuracy"]), f"{r['B_dev_answer_ce_mean']:.4f}", f"{r['C_dev_answer_ce_mean']:.4f}"] for r in common_sources]),
        "",
        "Task replacement accompanies source-capability differences across the whole multitask model, including common tasks. Later replacement matching should therefore examine both candidate and common-task learning curves and final capability. This observation identifies neither gradient conflict, mathematical relations, nor another specific mechanism, and does not prove intrinsic difficulty of any single task. Full train/dev means are in [common_source_tasks_comparison.csv](artifacts/common_source_tasks_comparison.csv).",
        "",
    ]
    lines += [
        "## Adaptation protocol and new-test comparison",
        "",
        "Every condition/seed/pretraining budget has the same three learning rates 3e−4,1e−4,3e−5 and 800 updates each, with 100 warmup steps followed by cosine decay to 0.1 of the initial rate. Full train/dev metrics are measured every 100 updates. Step0 is diagnostic only and ineligible for selection. Ties are resolved by highest development answer accuracy, lowest answer CE, earliest update, then declared learning-rate order. A checkpoint is selected within each candidate and then a learning rate using the same rule; test scores never participate.",
        "",
        "The doubled budget continues from the original 2400-update checkpoint, preserving optimizer and random states, for another 2400 updates at the original endpoint learning rate 3e−5. It tests continuation at a fixed final rate, not a fresh 4800-step cosine plan. Both budgets use the same adaptation selection rule. A always has pretraining budget 0 and is reused in both panels, without extra training or counting as another seed.",
        "",
        mdtable(["Comparison pretraining budget", "Protocol", "A", "B", "C"], [[budget, protocol, *(pct(avg(protocol, budget, c)) if any(r["comparison_pretrain_budget"] == budget and r["protocol"] == protocol and r["condition"] == c for r in averaged) else "—" for c in ("A", "B", "C"))] for budget in (2400, 4800) for protocol in PROTOCOLS if budget == 2400 or protocol != "original_v1_endpoint_new_test"]),
        "",
        "`original_v1_endpoint_new_test` evaluates original 800-step weights at the original learning rate on the new test; `replay_fixed_lr_endpoint` is the retrained endpoint at that rate; `dev_selected` is the development-selected learning rate and checkpoint; `selected_lr_endpoint` is the 800-step endpoint at the same selected rate. Tables show arithmetic means across two seeds. Per-seed values and descriptive standard deviations are in the [protocol means CSV](artifacts/protocol_comparison_mean.csv) and [paired comparison CSV](artifacts/protocol_comparison_paired.csv); no significance inference is made.",
        "",
        "![Protocol comparison](artifacts/figures/protocol_comparison.png)",
        "",
        mdtable(["seed", "Condition/pretraining", "Selected LR", "Selected step", "Selected dev acc", "Endpoint dev acc", "Selected test acc", "Endpoint test acc"], [[s["seed"], f"{s['condition']}/{s['budget']}", f"{s['lr']:g}", s["step"], pct(s["dev_accuracy"]), pct(byseed("selected_lr_endpoint", s["budget"], s["condition"], s["seed"])["dev_accuracy"]), pct(byseed("dev_selected", s["budget"], s["condition"], s["seed"])["test_accuracy"]), pct(byseed("selected_lr_endpoint", s["budget"], s["condition"], s["seed"])["test_accuracy"])] for s in selections]),
        "",
        f"Development-selected learning-rate counts: {lr_counts_text}; {changed_lr_text}. Each seed's A0 is counted once;10 groups are not 10 independent seeds. With equally budgeted 800-step candidates, the results do not support uniformly lowering the rate as a universal repair. This does not exclude lower rates requiring more updates, different warmup/decay schedules, or better adaptation strategies outside the candidate set. See [selected_learning_rate_counts.csv](artifacts/selected_learning_rate_counts.csv).",
        "",
        f"At the fixed 2400-update pretraining budget, development selection changes new-test accuracy relative to the original v1 endpoint by A {pp(gains['A'])}, B {pp(gains['B'])}, C {pp(gains['C'])} percentage points. Both learning rate and stopping point change, so this evaluates their joint adaptation protocol; the difference cannot be attributed entirely to either factor.",
        "",
        f"Of 10 selections, {early} occur before 800 updates; in {len(overfit)} of these, development accuracy decreases and development CE rises from the best development checkpoint to the endpoint. This count is affected by selection: the selected checkpoint having development accuracy at least as high as the endpoint is almost inherent in the rule, and alone does not prove overfitting. See the [curve CSV](artifacts/target_curves.csv) and [best-versus-endpoint differences](artifacts/best_vs_endpoint.csv). Corresponding new-test differences are frozen-protocol validation only, never used for reselection.",
        "",
        f"Among these {early} early selections, only {early_test_improved} selected checkpoints have higher new-test accuracy than their same-rate 800-step endpoints, while {early_test_worsened} are lower. Development selection does not guarantee new-test benefits. Failed selections are fully retained under the frozen plan, without using the new test to choose endpoints or rates retrospectively or to retrain.",
        "",
        f"Separately, a uniform fixed 400→800-step comparison over all 30 candidates, independent of selected rates or checkpoints, finds training answer CE decreasing while development answer CE increases in {late_ce_divergence}/30; development accuracy also declines in {late_acc_deterioration}/30. The former supports improving training fit with worsening development loss; only the latter additionally shows accuracy deterioration. Neither proves a sole cause of negative transfer. Sparse 100-step evaluation cannot equate this with numerical optimization instability or rule out transient instability at unsampled steps. This is an exploratory fixed-window diagnostic;30 candidates sharing data/initialization are not 30 independent seeds. Raw metrics and flags are in [late_training_diagnostics.csv](artifacts/late_training_diagnostics.csv).",
        "",
        mdtable(["Condition/pretraining", "Candidates", "Train CE decreases", "Dev CE increases", "Dev acc decreases", "Train CE↓ and dev CE↑", "Previous and dev acc↓"], [[f"{r['condition']}/{r['pretrain_updates']}", r["candidate_count"], r["train_ce_decreased"], r["dev_ce_increased"], r["dev_accuracy_decreased"], r["train_ce_down_and_dev_ce_up"], r["train_ce_down_dev_ce_up_dev_accuracy_down"]] for r in late_summary]),
        "",
        "New-test differences after development selection for doubled versus original pretraining budget: " + "; ".join(f"{c} {pp(avg('dev_selected', 4800, c)-avg('dev_selected', 2400, c))} percentage points" for c in ("B", "C")) + ". Increased-budget results are separate from fixed-budget results; differences from twice the computation are not task-composition effects under the original fixed budget.",
        "",
        "Per-seed budget effects follow (development-selected 4800−2400 test accuracy, percentage points). A mean benefit need not occur in every seed:",
        "",
        mdtable(["seed", "B budget effect", "C budget effect"], [[seed, *(pp(next(r["test_accuracy_delta"] for r in selected_budget_effects if r["seed"] == seed and r["condition"] == condition)) for condition in ("B", "C"))] for seed in seeds]),
        "",
        (f"{', '.join(heterogeneous_conditions)} budget effects have opposite signs across the two seeds, showing substantial heterogeneity; their means cannot be interpreted as stable benefits." if heterogeneous_conditions else "Budget effects have only two paired random units; their means do not establish stable benefits.") + "Per-seed budget differences for all protocols are in [target_budget_deltas.csv](artifacts/target_budget_deltas.csv).",
        "",
        "Differences by paired seed (percentage points):",
        "",
        mdtable(["seed", "Budget", "Protocol", "B−A", "C−A", "B−C"], [[seed, budget, protocol, *(pp(next(r["test_accuracy_delta"] for r in deltas if r["seed"] == seed and r["comparison_pretrain_budget"] == budget and r["protocol"] == protocol and r["contrast"] == contrast)) for contrast in ("B-A", "C-A", "B-C"))] for seed in seeds for budget in (2400, 4800) for protocol in ("dev_selected", "selected_lr_endpoint")]),
        "",
        "## Compute budget and runtime",
        "",
        "Each original B/C pretraining run uses 2400 updates×64=153600 sample exposures, 4300800 input tokens including answer/EOS, and 307200 supervised tokens. The doubled budget totals 307200 exposures, 8601600 input tokens, and 614400 supervised tokens. Each of four tasks receives exactly 25%. The pool has 8192 inputs; distinct inputs actually visited per task and sampling hashes are in [source_budgets.csv](artifacts/source_budgets.csv). Continuation uses the same pool, increasing exposures rather than distinct examples.",
        "",
        "Each adaptation uses 800×64=51200 exposures, 1433600 input tokens, and 102400 supervised tokens; each of 256 target training examples is exposed 200 times on average. 3 candidates per group total 2400 updates. Reusing A gives 30 actual candidates, not 36.",
        "",
        mdtable(["Stage", "Runs", "New compute", "Total updates", "Input tokens", "Supervised tokens", "Recorded loop seconds"], [[r["stage"], r["runs"], bool(r["new_compute"]), f"{int(r['updates']):,}", f"{int(r['input_tokens']):,}", f"{int(r['supervised_tokens']):,}", f"{r['seconds']:.1f}"] for r in totals]),
        "",
        f"Recorded training and associated evaluation/saving timers total {new_total_seconds:.1f} seconds, including {extension['seconds']:.1f} seconds and {int(extension['input_tokens']):,} additional input tokens for extended pretraining. Recorded-stage wall time totals {wall_total:.1f} seconds ({wall_total/60:.1f} minutes), covering{timed_stage_names}; it overlaps training timers and must not be added to them. {inventory_timing_note} This is not end-to-end duration for all diagnostic work. Adaptation timing includes step0 and later full train/dev evaluations, candidate best-checkpoint saves, and replay-weight comparisons where applicable; it excludes initialization and final endpoint saving. Source-extension timing includes training, source train/dev measurements, and CPU NumPy sampler replay checks, but excludes final saving. These are not pure GPU training times; exact diagnostic-forward tokens were not recorded. Model training tokens and evaluation costs are distinguished. Smoke-test updates are counted separately in smoke-stage wall time, not formal training tokens.",
        "",
        mdtable(["Stage", "Wall seconds"], [[r["stage"], f"{r['wall_seconds']:.2f}"] for r in stages]),
        "",
        "## Supported explanations and unresolved factors",
        "",
        "1. **There is direct evidence of incomplete source mastery and mismatched learning levels under current multitask conditions.** Full source train/dev results and continuation curves show B/C capability differences for both replaced and common tasks. Similar answer distributions do not ensure matched learning conditions or final capability. Low training and development scores cannot be explained solely by source overfitting, and common-task differences do not isolate gradient conflict or a mathematical-relation mechanism.",
        f"2. **The fixed window shows some late-training generalization deterioration; same-data, same-budget candidates directly test protocol sensitivity.** From 400→800 steps, training answer CE decreases while development answer CE increases in {late_ce_divergence}/30 candidates, with development accuracy also declining in {late_acc_deterioration}/30. This evidence does not depend on best-checkpoint selection. The development gap between best and endpoint is constrained by selection and does not independently prove overfitting. New-test differences between development selection and the original protocol are frozen-protocol validation. Joint learning-rate/checkpoint selection cannot alone identify the cause of negative transfer; failed selections are retained.",
        "3. **Checks found no inconsistency in loading, target-specific rows, optimizer reset, or data/evaluation across conditions.** Hash/state audits, continuation smoke tests, and original-high-LR replay comparisons provide inspectable evidence. These engineering errors are not supported as explanations of the observed differences.",
        "4. **Mechanisms remain unidentified.** Final replacement-task capability, gradient conflicts, representation changes, and complement-input inductive bias were not controlled. Longer pretraining changes both source capability and parameter location; it is not a mechanism-isolating intervention. No CKA sweep was conducted; compatible state_dicts and configurations remain available.",
        "5. **Statistical and selection scope is limited.** Both initialization and data change across two random units, so their effects cannot be estimated separately. One length, target, and task replacement cannot generalize to the library. Repeated development-set reuse may introduce selection bias. All candidate new-test scores are disclosed only after freezing, without further rate selection, retraining, or configuration changes. The old v1 test retains its exploratory status.",
        "",
        "## Next-step recommendation and stopping boundary",
        "",
        "**Prioritize replacement-task matching, then add targets, and only later increase source-task count.** Source-development capability differences remain unresolved. Match replacements using single-task/source-development learning curves, label distributions, and convergence capability while target transfer tests remain unseen, reducing confounds in interpreting B/C mechanisms. Matching must include train/dev curves and final capability for the replaced task and all three common sources, because replacement also accompanies common-task changes here. V2 does not change tasks. If later work retains these replacements, first establish more adequate source learning under equal compute budgets. After matching, add a few prespecified targets with known relations to test reproducibility beyond one target. Increasing task count now would add both confounds and cost.",
        "",
        "Expansion stops here pending further guidance. All completed results, including negative transfer and incompletely learned sources, are retained.",
        "",
        "## Artifacts and reproduction",
        "",
        "- All candidate tests by seed: [seed17](artifacts/results_seed17.csv), [seed42](artifacts/results_seed42.csv); [raw protocol comparison](artifacts/protocol_comparison.csv).",
        "- [Full configuration and frozen code hashes](artifacts/FROZEN_PROTOCOL.json), [frozen selections](artifacts/SELECTIONS_FROZEN.json), [new-test isolation audit](artifacts/test_data_audit.json).",
        "- [Source-learning table](artifacts/source_learning_table.csv), [actual budgets](artifacts/source_budgets.csv), [candidate development results](artifacts/candidate_development_results.csv), [complete target curves](artifacts/target_curves.csv), [actual cost](artifacts/compute_costs.csv).",
        "- Selected-rate target train/dev curves by seed/condition/budget are `artifacts/figures/target_selected_*.png`; three-rate comparisons are `target_lr_comparison_*.png`; source-learning curves are `source_seed*.png`. Figures use answer accuracy and answer CE, without presenting two-token training loss as answer CE.",
        "- [Reproduction instructions](README.md). This report and figures are generated from saved artifacts by `.venv/Scripts/python.exe experiments/pilot_v2_diagnostics/summarize.py`; the script has no model forwards and does not select or rerun training.",
    ]
    final_audit = OUT / "FINAL_AUDIT.json"
    if final_audit.exists():
        lines += ["", f"Final artifact audit status: {read_json(final_audit)['status']}; see [FINAL_AUDIT.json](artifacts/FINAL_AUDIT.json). The audit also verifies that selection freezing precedes new-test generation, test/selection hashes, frozen source hashes, and byte preservation of v1 files."]
    document = "\n".join(lines) + "\n"
    if REPORT.parent == OUT:
        document = document.replace("(artifacts/", "(./").replace("`artifacts/", "`./")
        document = document.replace("[Reproduction instructions](README.md)", f"[Reproduction instructions]({(HERE / 'README.md').as_posix()})")
    REPORT.write_text(document, encoding="utf-8")
    summary = dict(seeds=seeds, target_n=256, complete=True, selected_negative_mean_conditions=[dict(budget=b, condition=c) for b, c in selected_neg], source_dev_below_90_at_4800=len(low_source), total_source_seed_condition_units=16, source_dev_accuracy_increased_with_extension=source_dev_accuracy_improved, source_dev_ce_decreased_with_extension=source_dev_ce_improved, replay_bitwise_equal=replay_exact, replay_total=len(replay), selected_learning_rate_counts={f"{r['lr']:g}": r["selected_groups"] for r in lr_selection_counts}, selected_checkpoints_before_endpoint=early, early_selected_test_better_than_same_lr_endpoint=early_test_improved, early_selected_test_worse_than_same_lr_endpoint=early_test_worsened, selection_conditioned_best_to_endpoint_deterioration_count=len(overfit), fixed_400_to_800_train_ce_down_dev_ce_up=late_ce_divergence, fixed_400_to_800_train_ce_down_dev_ce_up_dev_acc_down=late_acc_deterioration, fixed_window_candidate_count=len(late), protocol_improvement_pp={k: 100 * v for k, v in gains.items()}, extension_improvement_pp={c: 100 * (avg("dev_selected", 4800, c) - avg("dev_selected", 2400, c)) for c in ("B", "C")}, extension_effect_by_seed_pp={f"seed{r['seed']}_{r['condition']}": 100 * r["test_accuracy_delta"] for r in selected_budget_effects}, new_training_and_associated_evaluation_seconds=new_total_seconds, recorded_stage_wall_seconds=wall_total, recorded_stages=[r["stage"] for r in stages], recommendation="replacement-task matching including common-source learning first; do not expand now", report="DIAGNOSTIC_REPORT.md")
    (OUT / "key_results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main():
    global OUT, FIG, REPORT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT, help="Saved diagnostics artifacts directory; no model code is imported.")
    args = parser.parse_args()
    OUT = args.out.resolve()
    FIG = OUT / "figures"
    REPORT = HERE / "DIAGNOSTIC_REPORT.md" if OUT == (HERE / "artifacts").resolve() else OUT / "DIAGNOSTIC_REPORT.md"
    assert (OUT / "TEST_COMPLETE.json").exists(), "Complete frozen test stage before summarizing."
    config = read_json(OUT / "FROZEN_PROTOCOL.json")["config"]
    selections = read_json(OUT / "SELECTIONS_FROZEN.json")["selections"]
    assert len(config["seeds"]) == 2 and len(selections) == 10
    assert read_json(OUT / "TEST_COMPLETE.json")["rows"] == 60
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    _, sources = source_tables()
    curves = read_csv(OUT / "target_curves.csv")
    target_plots(curves, selections)
    late, late_summary = late_training_diagnostics(curves)
    averaged, deltas = protocol_tables(read_csv(OUT / "protocol_comparison.csv"), config["seeds"])
    _, totals, stages, replay = compute_costs()
    replacements = replacement_table()
    summary = report(config, selections, sources, averaged, deltas, totals, stages, replay, replacements, late, late_summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
