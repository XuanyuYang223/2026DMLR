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
    lr_counts_text = "、".join(f"{r['lr']:g}选中{r['selected_groups']}/{r['total_groups']}组" for r in lr_selection_counts)
    changed_lr_selections = [s for s in selections if float(s["lr"]) != config["learning_rates"][0]]
    changed_lr_text = "；".join(f"seed{s['seed']} {s['condition']}/{s['budget']}选择{s['lr']:g}、{s['step']}步" for s in changed_lr_selections) or "全部组均选择原学习率"

    def avg(protocol, budget, condition, metric="test_accuracy_mean"):
        return next(r[metric] for r in averaged if r["protocol"] == protocol and r["comparison_pretrain_budget"] == budget and r["condition"] == condition)

    def byseed(protocol, budget, condition, seed):
        return next(r for r in comparison if r["protocol"] == protocol and integer(r, "pretrain_updates") == budget and r["condition"] == condition and integer(r, "seed") == seed)

    delta_lines = []
    for budget in (2400, 4800):
        a = avg("dev_selected", budget, "A")
        b, c = avg("dev_selected", budget, "B"), avg("dev_selected", budget, "C")
        delta_lines.append(f"{budget}步预训练、开发集选择后，B−A={pp(b-a)}、C−A={pp(c-a)}、B−C={pp(b-c)}个百分点。")
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
    inventory_timing_note = "inventory经独立CLI执行且没有timing_inventory记录，因此其耗时不在上述墙钟合计内。" if not any(r["stage"] == "inventory" for r in stages) else "inventory已包含在有计时记录的阶段中。"
    checked = all(m["initialization"]["loaded_state_hash"] == m["initialization"]["source_state_hash"] and m["initialization"]["optimizer_state_reset"] and m["initialization"]["optimizer_entries"] == 0 for m in all_completed)
    target_rows_equal = all(len({m["initialization"]["target_row_hash"] for m in all_completed if m["seed"] == seed}) == 1 for seed in seeds)
    data_equal = all(len({m["initialization"]["train_sha256"] for m in all_completed if m["seed"] == seed}) == len({m["initialization"]["dev_sha256"] for m in all_completed if m["seed"] == seed}) == 1 for seed in seeds)
    schedule_equal = all(len({m["schedule_sha256"] for m in all_completed if m["seed"] == seed}) == 1 for seed in seeds)
    assert checked and target_rows_equal and data_equal and schedule_equal

    lines = [
        "# PermWorld 第二轮诊断报告（2026-10-18）",
        "",
        f"已实际完成两个配对种子{seeds}、30次学习率候选适配（每次800更新）、4次预训练延长（每次新增2400更新），以及冻结选择后生成的新隔离测试集评估。目标固定为256个不同训练样本的 fixed_points，任务组合、one-line编码和模型结构均沿用pilot_v1。没有增加任务或目标。",
        "",
        "**" + " ".join(delta_lines) + "**",
        "",
        ("开发集选择后仍有低于A的均值，负迁移保留在结果中；" if selected_neg else "开发集选择后的B/C均值均不低于A，但仅有两个种子；") + f"两倍预算后仍有{len(low_source)}/16个源任务×种子×条件组合的开发准确率低于90%。90%仅用于描述未充分掌握的诊断门槛，不是跨任务统一难度标准。不能从这些结果把迁移机制归因于已知数学关系。",
        "",
        "## 任务、数据与公平比较",
        "",
        "目标 fixed_points(p)=#{i:p(i)=i}，固定n=10。B预训练 anti_fixed_points、descents、peaks、left_to_right_maxima；C仅把anti_fixed_points替换成successions。关系为fixed_points(p)=anti_fixed_points(c(p))，c(p)(i)=11−p(i)。这是输入complement下的恒等式。successions没有被登记为该直接关系任务，未断言数学上无关。",
        "",
        "两种子各沿用v1目标训练前256条及1024条开发输入；B/C共用8192条预训练输入、1024条源开发输入，四任务每批各16/64。适配所有候选共享同种子数据、采样序列与最大800更新。没有新增参数；目标专有初始化处理仅将既有目标task token的tied embedding/output行恢复到同种子共享step0值，数字输出参数继续迁移。每次适配新建AdamW并检查状态为空。",
        "",
        f"新测试在10个开发集选择全部冻结之后生成，每种子2048条；按identity/inverse/reverse/complement完整轨道排除v1所有数据（包含历史测试）及跨种子新测试输入。显式轨道交集为{test_audit['overlap']}。两个种子均在同种子各条件、候选和协议之间共享同一新测试集。下列表全部是这批新测试结果，v1旧测试分数没有混入均值。v1原endpoint重新在新测试上评估。旧开发集被复用，且本轮问题来源于历史探索，所以这仍是诊断性探索。",
        "",
        "## 历史记录与补充诊断",
        "",
        "历史记录保存了预训练稀疏的最后一批双token损失、源endpoint准确率、目标每400更新（仅400和800步）的开发准确率/答案CE和最后一批训练损失。缺少逐源任务历史训练/开发损失轨迹、目标完整训练集轨迹及中间模型checkpoint。此次对原checkpoint补测完整源训练/开发集准确率、答案CE、双token CE、EOS准确率和按长度多数基线。历史中间模型无法从endpoint恢复，新重跑曲线明确记为replay，不伪装成历史测量。",
        "",
        "按长度基线在本轮只有n=10一层；多数标签由对应训练池确定，再测量该标签在训练/开发/测试池的命中率。源任务表中的答案CE使用全188词表；训练损失是答案与EOS的双token平均CE，两者不可混用。",
        "",
        f"权重严格加载hash检查通过={checked}；同种子目标专有行一致={target_rows_equal}；目标数据哈希一致={data_equal}；适配采样序列哈希一致={schedule_equal}；保存/继续训练4+4更新逐位一致={smoke['resume_state_bitwise_equal']}。原预训练优化器步数为2400，原适配保存步数为800；新的适配优化器状态从0开始。",
        "",
        "历史v1没有保存适配起点参数指纹，历史加载/重置判断依据代码、保存的优化器步数及本次重建检查，不能声称掌握历史起点的直接哈希证据。v2每个候选实际运行前保存initialization.json，具有加载前后hash、目标行hash及空优化器状态记录。",
        "",
        f"历史6组256样本适配从400到800步，有{historical_ce_worse}组开发答案CE上升、{historical_accuracy_worse}组开发准确率下降。CE恶化可能反映置信度/校准恶化，不能全部写成准确率过拟合；历史最后一批混合双token loss也不能代替完整训练集损失。",
        "",
        f"高学习率原协议重跑与v1原endpoint在{replay_exact}/{len(replay)}个单元逐位相同。最大参数绝对差见[replay表](artifacts/replay_v1_weight_differences.csv)。" + ("未逐位复现的重跑不替代原checkpoint，原checkpoint新测试结果单列。" if replay_exact < len(replay) else "重跑补出的完整训练曲线可与相同权重终点关联，历史稀疏记录仍单独保留。"),
        "",
        mdtable(["种子", "条件", "预训练更新", "逐位相同", "最大参数差"], [[r["seed"], r["condition"], r["pretrain_updates"], r["bitwise_same"], f"{r['parameter_max_abs_difference']:.6g}"] for r in replay]),
        "",
        "## 源任务学习与替换难度",
        "",
        "下表为每种子的原预算和两倍预算endpoint。准确率与CE全部为完整池测量；更完整的双token/EOS指标和训练多数基线保存在[源学习表](artifacts/source_learning_table.csv)，延长阶段每600更新的结果见[全部源曲线CSV](artifacts/source_learning_all.csv)。",
        "",
        mdtable(["seed", "条件/更新", "任务", "训练acc", "开发acc", "训练CE", "开发CE", "开发多数"], [[r["seed"], f"{r['condition']}/{r['pretrain_updates']}", r["task"], pct(r["train_accuracy"]), pct(r["dev_accuracy"]), f"{float(r['train_answer_ce']):.4f}", f"{float(r['dev_answer_ce']):.4f}", pct(r["dev_majority_accuracy"])] for r in sources]),
        "",
        "答案分布使用相同8192条源训练输入，延长训练不改变输入分布：",
        "",
        mdtable(["seed", "任务", "标签均值", "熵(bit)", "B/C总变差"], [[r["seed"], r["task"], f"{r['mean']:.4f}", f"{r['entropy_bits']:.4f}", f"{r['replacement_total_variation']:.4f}"] for r in replacements]),
        "",
    ]
    for budget in (2400, 4800):
        b = [r for r in sources if r["pretrain_updates"] == budget and r["task"] == "anti_fixed_points"]
        c = [r for r in sources if r["pretrain_updates"] == budget and r["task"] == "successions"]
        lines.append(f"{budget}更新时，被替换任务开发准确率均值：anti_fixed_points={pct(mean(float(r['dev_accuracy']) for r in b))}，successions={pct(mean(float(r['dev_accuracy']) for r in c))}；开发答案CE均值分别为{mean(float(r['dev_answer_ce']) for r in b):.4f}/{mean(float(r['dev_answer_ce']) for r in c):.4f}。这是当前多任务优化条件下的学习差异，尚不能当作任务内在难度的独立估计。")
        lines.append("")
    lines += [
        f"原预算→两倍预算的16个配对源任务单元中，{source_dev_accuracy_improved}/16个开发准确率提高、{source_dev_ce_improved}/16个开发答案CE下降；逐项训练/开发指标变化见[source_budget_changes.csv](artifacts/source_budget_changes.csv)。这说明继续训练对源学习的实际影响，同时保留仍未充分掌握的单元，不能以“有提升”代替“已经学会”。",
        "",
        "B/C共有的三个源任务在两种组成下也呈现不同学习水平。下表是两种子平均开发结果，B/C使用相同公共任务输入、标签、采样比例和总预算：",
        "",
        mdtable(["预训练更新", "公共任务", "B开发acc", "C开发acc", "B−C百分点", "B开发CE", "C开发CE"], [[r["pretrain_updates"], r["task"], pct(r["B_dev_accuracy_mean"]), pct(r["C_dev_accuracy_mean"]), pp(r["B_minus_C_dev_accuracy"]), f"{r['B_dev_answer_ce_mean']:.4f}", f"{r['C_dev_answer_ce_mean']:.4f}"] for r in common_sources]),
        "",
        "任务替换伴随整个多任务模型的源能力差异，包括公共任务；因此后续替换匹配需要同时核对替换任务和公共任务的学习曲线与最终能力。该观察没有识别梯度冲突、数学关系或其他具体机制，也不能证明任何单任务的内在难度。完整训练/开发均值见[common_source_tasks_comparison.csv](artifacts/common_source_tasks_comparison.csv)。",
        "",
    ]
    lines += [
        "## 适配协议与新测试对照",
        "",
        "各条件/种子/预训练预算拥有同样三个学习率3e−4、1e−4、3e−5，各800更新；100步warmup后余弦衰减到0.1倍。每100更新测量完整训练集与开发集，step0仅诊断，不参与选择。按开发答案准确率最大、答案CE最小、更新步最早、声明学习率顺序依次打破平局。先在各候选中选checkpoint，再以同一规则选择学习率；测试分数从不参与选择。",
        "",
        "两倍预算从原2400更新checkpoint继续，保留原优化器与随机数状态，用原末端学习率3e−5继续2400更新。它检验固定末端学习率的继续训练，并非从头训练4800步的新余弦计划。两预算使用同一适配选择规则。A的预训练预算恒为0，在两预算面板中复用同一运行，未额外训练或当成新增种子。",
        "",
        mdtable(["比较预训练预算", "协议", "A", "B", "C"], [[budget, protocol, *(pct(avg(protocol, budget, c)) if any(r["comparison_pretrain_budget"] == budget and r["protocol"] == protocol and r["condition"] == c for r in averaged) else "—" for c in ("A", "B", "C"))] for budget in (2400, 4800) for protocol in PROTOCOLS if budget == 2400 or protocol != "original_v1_endpoint_new_test"]),
        "",
        "`original_v1_endpoint_new_test`为原学习率、800步原权重在新测试上的结果；`replay_fixed_lr_endpoint`为同原学习率重新训练的终点；`dev_selected`为开发选择的学习率及checkpoint；`selected_lr_endpoint`为同一选中学习率在800步的终点。表为两个种子算术均值；每种子结果及描述性标准差见[协议均值CSV](artifacts/protocol_comparison_mean.csv)和[配对对照CSV](artifacts/protocol_comparison_paired.csv)，不作显著性推断。",
        "",
        "![协议对照](artifacts/figures/protocol_comparison.png)",
        "",
        mdtable(["seed", "条件/预训练", "选中LR", "选中步", "选中开发acc", "终点开发acc", "选中测试acc", "终点测试acc"], [[s["seed"], f"{s['condition']}/{s['budget']}", f"{s['lr']:g}", s["step"], pct(s["dev_accuracy"]), pct(byseed("selected_lr_endpoint", s["budget"], s["condition"], s["seed"])["dev_accuracy"]), pct(byseed("dev_selected", s["budget"], s["condition"], s["seed"])["test_accuracy"]), pct(byseed("selected_lr_endpoint", s["budget"], s["condition"], s["seed"])["test_accuracy"])] for s in selections]),
        "",
        f"开发集选择的学习率计数：{lr_counts_text}；{changed_lr_text}。每种子的A0只计一次，10组不是10个独立随机种子。当前同为800步的候选预算下，结果不支持把简单统一降低学习率视为普遍修复；这不排除低学习率需要更多更新、不同warmup/衰减日程，或候选集合以外的适配策略更合适。计数见[selected_learning_rate_counts.csv](artifacts/selected_learning_rate_counts.csv)。",
        "",
        f"固定2400预训练预算下，开发选择相对原v1 endpoint的新测试准确率变化：A {pp(gains['A'])}、B {pp(gains['B'])}、C {pp(gains['C'])}个百分点。这同时改变学习率和停止位置，只能评价这一联合适配协议；不能把差值全部归因于其中一个因素。",
        "",
        f"10个选择中{early}个早于800更新；其中{len(overfit)}个从最佳开发checkpoint到终点出现开发准确率下降且开发CE上升。该计数受开发集选择规则影响：被选checkpoint的开发准确率不低于终点几乎是规则本身的结果，不能单凭它证明过拟合。完整曲线见[曲线CSV](artifacts/target_curves.csv)，最佳与终点差值见[best_vs_endpoint](artifacts/best_vs_endpoint.csv)；对应新测试差值只作为冻结后的验证，不用于重新选择。",
        "",
        f"在这{early}个提前选择中，只有{early_test_improved}个选中checkpoint的新测试准确率高于同学习率800步终点，{early_test_worsened}个更低。开发选择没有保证在新测试上获益；这些失败选择按冻结方案完整保留，不根据新测试反向挑选终点、学习率或重新训练。",
        "",
        f"另对全部30候选统一比较固定400→800步，不依赖选中学习率或checkpoint：{late_ce_divergence}/30出现训练答案CE下降且开发答案CE上升，其中{late_acc_deterioration}/30同时开发准确率下降。前者支持训练拟合继续改善但开发损失退化的现象，只有后者同时表明准确率也退化；均不证明其是负迁移的唯一原因。每100步的稀疏评估不能把这些现象等同于数值优化不稳定，也不能完全排除未采样时间点的瞬时不稳定。这是探索性固定窗口诊断，30个共享数据/初始化的候选不是30个独立种子。原始指标和标记见[late_training_diagnostics.csv](artifacts/late_training_diagnostics.csv)。",
        "",
        mdtable(["条件/预训练", "候选数", "训练CE下降", "开发CE上升", "开发acc下降", "训练CE↓且开发CE↑", "前项且开发acc↓"], [[f"{r['condition']}/{r['pretrain_updates']}", r["candidate_count"], r["train_ce_decreased"], r["dev_ce_increased"], r["dev_accuracy_decreased"], r["train_ce_down_and_dev_ce_up"], r["train_ce_down_dev_ce_up_dev_accuracy_down"]] for r in late_summary]),
        "",
        "开发选择后两倍预训练相对原预算的新测试差值：" + "；".join(f"{c} {pp(avg('dev_selected', 4800, c)-avg('dev_selected', 2400, c))}个百分点" for c in ("B", "C")) + "。增加预训练预算与固定预算结果分开呈现，不能把两倍计算带来的差值当作原固定预算条件下的任务组成效应。",
        "",
        "预算变化的逐种子效应如下（开发选择后的4800−2400测试准确率，百分点）；均值不能代表每个种子都受益：",
        "",
        mdtable(["seed", "B预算变化", "C预算变化"], [[seed, *(pp(next(r["test_accuracy_delta"] for r in selected_budget_effects if r["seed"] == seed and r["condition"] == condition)) for condition in ("B", "C"))] for seed in seeds]),
        "",
        (f"{'、'.join(heterogeneous_conditions)}的预算效应在两个种子之间异号，说明当前预算效应有明显异质性，不能将其均值解释为稳定收益。" if heterogeneous_conditions else "预算效应只有两个配对随机单元，不能从其均值推断稳定收益。") + "各协议的逐种子预算差值见[target_budget_deltas.csv](artifacts/target_budget_deltas.csv)。",
        "",
        "逐配对种子差值如下（百分点）：",
        "",
        mdtable(["seed", "预算", "协议", "B−A", "C−A", "B−C"], [[seed, budget, protocol, *(pp(next(r["test_accuracy_delta"] for r in deltas if r["seed"] == seed and r["comparison_pretrain_budget"] == budget and r["protocol"] == protocol and r["contrast"] == contrast)) for contrast in ("B-A", "C-A", "B-C"))] for seed in seeds for budget in (2400, 4800) for protocol in ("dev_selected", "selected_lr_endpoint")]),
        "",
        "## 计算预算与运行时间",
        "",
        "原B/C预训练每次2400更新×64=153600次样本曝光、4300800输入tokens（含答案与EOS）、307200监督tokens；两倍预算总计307200次曝光、8601600输入tokens、614400监督tokens。四任务严格各25%；预训练池8192条，实际每任务看过的不同输入数与采样hash见[source_budgets.csv](artifacts/source_budgets.csv)。延长继续使用相同池，增加的是曝光次数，不是不同训练样本数。",
        "",
        "每次适配800×64=51200次曝光、1433600输入tokens、102400监督tokens；256条目标训练样本平均曝光200遍。每组3个候选共2400更新。A复用使本轮实际候选数为30，而非36。",
        "",
        mdtable(["阶段", "运行数", "新增计算", "更新合计", "输入tokens", "监督tokens", "记录循环秒数"], [[r["stage"], r["runs"], bool(r["new_compute"]), f"{int(r['updates']):,}", f"{int(r['input_tokens']):,}", f"{int(r['supervised_tokens']):,}", f"{r['seconds']:.1f}"] for r in totals]),
        "",
        f"本轮记录的训练及关联评估/保存计时合计{new_total_seconds:.1f}秒；其中新增预训练{extension['seconds']:.1f}秒、额外{int(extension['input_tokens']):,}输入tokens。仅已记录阶段的墙钟合计{wall_total:.1f}秒（{wall_total/60:.1f}分钟），实际覆盖{timed_stage_names}；与上述训练计时重叠，不应相加。{inventory_timing_note}这不是包含全部诊断工作的端到端总时长。适配计时包含step0和后续完整训练/开发评估、候选内best checkpoint保存、适用时的重放权重差异检查；不含初始化和最终endpoint保存。源延长计时包含训练、源训练/开发测量和CPU NumPy采样器重放核验，不含最后保存。均不是纯GPU训练时间，未记录诊断forward的精确tokens。模型训练tokens与评估成本严格区分。smoke小步更新另计在smoke阶段墙钟，不计入正式训练tokens。",
        "",
        mdtable(["阶段", "墙钟秒数"], [[r["stage"], f"{r['wall_seconds']:.2f}"] for r in stages]),
        "",
        "## 得到支持的解释与仍无法区分的因素",
        "",
        "1. **源任务未充分掌握、当前多任务条件下学习水平不匹配有直接证据。** 完整源训练/开发结果及继续训练曲线说明被替换任务和公共任务均存在B/C能力差异；仅答案分布近似并不等于学习条件或最终能力匹配。训练与开发同时较低的任务不能仅用源过拟合解释，公共任务差异也没有单独识别梯度冲突或数学关系机制。",
        f"2. **固定窗口显示部分训练后期泛化退化，适配协议敏感性由同数据同预算候选直接检验。** 统一400→800步中，{late_ce_divergence}/30候选训练答案CE下降而开发答案CE上升，{late_acc_deterioration}/30还伴随开发准确率下降。该证据不依赖挑选最佳checkpoint；最佳开发点与终点的开发差距受选择规则约束，不能独立证明过拟合。开发选择与原协议的新测试差异是冻结后的验证；LR和checkpoint联合选择的影响不能单独识别负迁移的原因，失败的选择也完整保留。",
        "3. **检查未发现加载、目标专有行、优化器重置或条件间数据/评估不一致。** 哈希与状态审计、继续训练smoke、原高LR重跑差异表给出了可检查证据；这些工程错误作为本轮差异的解释不受现有证据支持。",
        "4. **机制仍无法识别。** 没有控制替换任务最终能力、梯度冲突、表示变化或输入complement归纳偏置。更长预训练同时改变源能力与参数位置，不是隔离机制的干预。本轮没有开展CKA扫描；兼容state_dict及配置保留供后续接入。",
        "5. **统计与选择范围有限。** 两个随机单元同时改变初始化和数据，不能分开估计两者；单一长度、目标和任务替换不能外推到任务库。开发集长期复用可能有选择偏差；全部候选的新测试数值只作冻结后披露，不据此追选学习率、重训或改配置。旧v1测试保持其探索性质。",
        "",
        "## 下一步建议与停止边界",
        "",
        "**优先检查替换任务匹配，其次才增加目标任务，最后再考虑扩大源任务数量。** 当前任务组合中的源开发能力差异尚未消除，先在不看目标迁移测试的条件下，以单任务/源开发学习曲线、标签分布及收敛能力匹配替换任务，能直接减少B/C机制解释的混杂。匹配检查同时覆盖被替换任务及三个公共源任务的训练/开发曲线和最终能力，因为本轮任务替换也伴随公共任务能力变化。当前v2不更换任务；若后续保留现有替换任务，则先建立更充分且等计算预算的源学习条件。匹配检查之后，再增加少量预先指定且具已知关系的目标，检验单目标结论是否可复现；扩大任务数量现在会同时增加混杂与计算量。",
        "",
        "本轮到此停止扩展，等待后续指导。所有已运行结果（包括负迁移和源任务未充分学会）均保留。",
        "",
        "## 产物与复现",
        "",
        "- 每种子全部候选测试：[seed17](artifacts/results_seed17.csv)、[seed42](artifacts/results_seed42.csv)；[原始协议对照](artifacts/protocol_comparison.csv)。",
        "- [完整配置与冻结代码hash](artifacts/FROZEN_PROTOCOL.json)、[选择冻结记录](artifacts/SELECTIONS_FROZEN.json)、[新测试隔离审计](artifacts/test_data_audit.json)。",
        "- [源学习表](artifacts/source_learning_table.csv)、[实际预算](artifacts/source_budgets.csv)、[候选开发结果](artifacts/candidate_development_results.csv)、[完整目标曲线](artifacts/target_curves.csv)、[实际成本](artifacts/compute_costs.csv)。",
        "- 目标每seed/condition/budget的选中LR训练/开发曲线为`artifacts/figures/target_selected_*.png`，三LR比较为`target_lr_comparison_*.png`，源学习曲线为`source_seed*.png`。图使用答案准确率和答案CE，不把双token训练loss伪装成答案CE。",
        "- [复现说明](README.md)。本报告与图由`.venv/Scripts/python.exe experiments/pilot_v2_diagnostics/summarize.py`从已保存产物生成；该脚本没有任何模型forward，不选择或重跑训练。",
    ]
    final_audit = OUT / "FINAL_AUDIT.json"
    if final_audit.exists():
        lines += ["", f"最终产物审计状态：{read_json(final_audit)['status']}，见[FINAL_AUDIT.json](artifacts/FINAL_AUDIT.json)。审计另核验选择冻结先于新测试生成、测试/选择hash、源码冻结hash和v1文件逐字节保留。"]
    document = "\n".join(lines) + "\n"
    if REPORT.parent == OUT:
        document = document.replace("(artifacts/", "(./").replace("`artifacts/", "`./")
        document = document.replace("[复现说明](README.md)", f"[复现说明]({(HERE / 'README.md').as_posix()})")
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
