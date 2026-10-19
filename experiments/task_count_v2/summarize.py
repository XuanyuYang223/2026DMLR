"""Paired budget-interaction reporting; never trains, evaluates or selects a model.

Only completed independent test records are accepted. The experimental unit is
the paired repeat, not a condition, target, source task or reused baseline.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ("development_selected", "fixed_endpoint")
PRIMARY = CHECKPOINTS[0]
ARCHIVED_TEST_SHA = "589043ef94ccdda2d48a751dcbae08f9664fe5c61083c1552cbe8f12b24b4d6c"
FOCUS = ("lds_length", "increasing_subsequence_count_k3", "decreasing_subsequence_count_k3")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    assert rows, path
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def moments(values):
    values = list(map(float, values))
    assert len(values) == 3
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "min": min(values), "max": max(values)}


def metric_valid(row):
    count, correct = int(row["count"]), int(row["correct"])
    assert count > 0 and 0 <= correct <= count
    assert abs(float(row["accuracy"]) - correct / count) < 1e-12
    assert abs(float(row["majority_accuracy"]) - int(row["majority_correct"]) / count) < 1e-12


def pre_id(chain, k, budget, seed):
    return f"{chain}_k{k}_b{budget}_s{seed}"


def target_id(chain, k, budget, target, seed):
    return f"{chain}_k{k}_b{budget}_{target}_s{seed}"


def collect(out):
    protocol = read_json(out / "FROZEN_PROTOCOL.json")
    selections = read_json(out / "FROZEN_SELECTIONS.json")
    training = read_json(out / "TRAINING_COMPLETE.json")
    test = read_json(out / "TEST_COMPLETE.json")
    assert protocol["status"] == selections["status"] == "frozen"
    assert selections["training_complete"] and test["status"] == "passed" and test["no_test_based_selection"]
    assert test["raw_results_sha256"] == sha(out / "raw_test_results.csv")
    assert test["frozen_selection_sha256"] == sha(out / "FROZEN_SELECTIONS.json")
    protocol_sha = sha(out / "FROZEN_PROTOCOL.json")
    assert selections["frozen_protocol_sha256"] == training["frozen_protocol_sha256"] == protocol_sha
    cfg = protocol["config"]
    targets = [s["task_id"] for s in protocol["target_specs"]]
    chains, ks = list(protocol["chains"]), cfg["k_values"]
    budgets = cfg["pretraining_budgets_nonpadding_sequence_tokens"]
    seeds = [r["initialization_seed"] for r in cfg["repeats"]]
    assert targets == ["lis_length", "fixed_points"] and len(chains) == 2
    assert ks == [2, 16] and budgets == [4_000_000, 32_000_000] and seeds == [4103, 5207, 6301]
    assert cfg["scope"] == {"pretraining_models": 24, "adaptations_from_pretraining": 48, "shared_random_initialization_baselines": 6, "total_target_training_runs": 54}
    conditions = [("baseline", 0, 0)] + [(c, k, b) for c in chains for k in ks for b in budgets]
    expected_pre = {pre_id(c, k, b, s) for c, k, b in conditions[1:] for s in seeds}
    expected_adapt = {target_id(c, k, b, t, s) for c, k, b in conditions for t in targets for s in seeds}
    records, curves, source_curves, exposures = [], [], [], []
    for stage, expected in (("pretraining", expected_pre), ("adaptation", expected_adapt)):
        paths = sorted((out / stage).glob("*/COMPLETE.json"))
        assert {p.parent.name for p in paths} == expected
        for path in paths:
            rec = read_json(path)
            assert rec["run_id"] == path.parent.name and rec["frozen_protocol_sha256"] == protocol_sha
            names = ["source_curves.csv", "task_exposures.csv"] if stage == "pretraining" else ["curves.csv"]
            for name in names:
                assert sha(path.parent / name) == rec["artifact_sha256"][name]
            rec["stage"] = stage
            records.append(rec)
            if stage == "pretraining":
                assert rec["input_tokens"] + rec["budget_tail_tokens"] == rec["budget_tokens"]
                assert rec["uniform_task_count_max_minus_min"] <= 1
                assert rec["pretraining_started_from"] == "paired_random_initialization"
                assert rec["continued_from_short_budget"] is False
                assert rec["optimizer_started_empty"] and rec["fixed_pool_size"] == 8192
                assert rec["fixed_pool_unique_inputs_exposed"] <= 8192
                source_curves.extend(read_csv(path.parent / "source_curves.csv"))
                exposures.extend(read_csv(path.parent / "task_exposures.csv"))
            else:
                assert rec["updates"] == cfg["adaptation_updates"] == 800
                curves.extend(read_csv(path.parent / "curves.csv"))
    pre = {r["run_id"]: r for r in records if r["stage"] == "pretraining"}
    for c in chains:
        for k in ks:
            for s in seeds:
                assert pre[pre_id(c, k, budgets[0], s)]["initial_state_sha256"] == pre[pre_id(c, k, budgets[1], s)]["initial_state_sha256"]
    rows = read_csv(out / "raw_test_results.csv")
    assert len(rows) == 324 == test["result_rows"]
    lookup = {}
    identities = {r["run_id"]: r for r in records if r["stage"] == "adaptation"}
    selected = {r["run_id"]: r for r in selections["selections"]}
    assert set(selected) == expected_adapt
    for row in rows:
        metric_valid(row)
        assert row["run_id"] in expected_adapt and row["checkpoint"] in CHECKPOINTS and row["split"] == "test"
        rec = identities[row["run_id"]]
        for name in ("target", "chain", "condition", "initialization_seed", "data_seed", "pretrain_sampling_seed", "adaptation_sampling_seed", "k", "budget_tokens"):
            assert str(row[name]) == str(rec[name])
        assert int(row["selected_update"]) == (rec["best_update"] if row["checkpoint"] == PRIMARY else 800)
        assert selected[row["run_id"]]["selected_update"] == rec["best_update"]
        key = row["run_id"], row["checkpoint"], row["length"]
        assert key not in lookup
        lookup[key] = row
    lengths = [str(n) for n in cfg["lengths"]] + ["all"]
    for rid in expected_adapt:
        for checkpoint in CHECKPOINTS:
            assert all((rid, checkpoint, n) in lookup for n in lengths)
            parts = [lookup[rid, checkpoint, n] for n in lengths if n != "all"]
            total = lookup[rid, checkpoint, "all"]
            for field in ("count", "correct", "majority_correct"):
                assert sum(int(r[field]) for r in parts) == int(total[field])
    by_run = defaultdict(list)
    for row in curves:
        metric_valid(row)
        by_run[row["run_id"]].append(row)
    for rid, rec in identities.items():
        dev = [r for r in by_run[rid] if r["split"] == "target_dev" and r["length"] == "all" and int(r["update"]) > 0]
        assert len(dev) == 8
        best = min(dev, key=lambda r: (-float(r["accuracy"]), float(r["answer_ce"]), int(r["update"])))
        assert int(best["update"]) == rec["best_update"]
    source_keys = set()
    for row in source_curves:
        metric_valid(row)
        key = tuple(row[x] for x in ("run_id", "source_task", "stage", "split", "length"))
        assert key not in source_keys
        source_keys.add(key)
        assert row["stage"] in ("start", "midpoint", "endpoint")
    assert len(source_curves) == 3888
    for row in exposures:
        assert int(row["unique_inputs_exposed"]) <= int(row["pool_size"]) == 4096
        assert int(row["repeat_exposures"]) == int(row["sample_exposures"]) - int(row["unique_inputs_exposed"])
    return {"protocol": protocol, "config": cfg, "records": records, "rows": rows, "curves": curves, "source_curves": source_curves, "exposures": exposures, "targets": targets, "chains": chains, "ks": ks, "budgets": budgets, "seeds": seeds, "lengths": lengths, "conditions": conditions, "test": test, "training": training}


def target_summaries(data):
    lookup = {(r["target"], r["chain"], int(r["k"]), int(r["budget_tokens"]), int(r["initialization_seed"]), r["checkpoint"], r["length"]): r for r in data["rows"]}
    assert len(lookup) == len(data["rows"])
    paired, summaries, diagnostics = [], [], []
    for row in data["rows"]:
        if row["condition"] == "A":
            assert row["chain"] == "baseline" and int(row["k"]) == int(row["budget_tokens"]) == 0
            continue
        base = lookup[row["target"], "baseline", 0, 0, int(row["initialization_seed"]), row["checkpoint"], row["length"]]
        for field in ("count", "majority_correct", "majority_answer", "data_seed", "adaptation_sampling_seed"):
            assert row[field] == base[field]
        paired.append({**{k: row[k] for k in ("target", "chain", "k", "budget_tokens", "initialization_seed", "data_seed", "pretrain_sampling_seed", "adaptation_sampling_seed", "checkpoint", "length")}, "pretrained_run_id": row["run_id"], "shared_baseline_run_id": base["run_id"], "test_count": int(row["count"]), "pretrained_accuracy": float(row["accuracy"]), "baseline_accuracy": float(base["accuracy"]), "paired_delta_pp": 100 * (float(row["accuracy"]) - float(base["accuracy"])), "pretrained_selected_update": int(row["selected_update"]), "baseline_selected_update": int(base["selected_update"]), "majority_accuracy": float(row["majority_accuracy"])})
    deltas = {(r["target"], r["chain"], int(r["k"]), int(r["budget_tokens"]), int(r["initialization_seed"]), r["checkpoint"], r["length"]): r for r in paired}
    for t in data["targets"]:
        for c, k, b in data["conditions"]:
            for checkpoint in CHECKPOINTS:
                for n in data["lengths"]:
                    group = [lookup[t, c, k, b, s, checkpoint, n] for s in data["seeds"]]
                    acc = moments(r["accuracy"] for r in group)
                    gains = moments([0] * 3 if c == "baseline" else [deltas[t, c, k, b, s, checkpoint, n]["paired_delta_pp"] for s in data["seeds"]])
                    summaries.append({"target": t, "chain": c, "k": k, "budget_tokens": b, "checkpoint": checkpoint, "length": n, "independent_paired_repeats": 3, **{f"accuracy_{name}": value for name, value in acc.items()}, **{f"paired_delta_pp_{name}": value for name, value in gains.items()}, "majority_accuracy_mean": statistics.mean(float(r["majority_accuracy"]) for r in group), "uncertainty": "Descriptive sample SD of three paired repeats, not a confidence interval"})
            for s in data["seeds"]:
                for n in data["lengths"]:
                    best, end = lookup[t, c, k, b, s, PRIMARY, n], lookup[t, c, k, b, s, CHECKPOINTS[1], n]
                    diagnostics.append({"run_id": best["run_id"], "target": t, "chain": c, "k": k, "budget_tokens": b, "initialization_seed": s, "length": n, "selected_update": int(best["selected_update"]), "endpoint_update": 800, "development_selected_test_accuracy": float(best["accuracy"]), "endpoint_test_accuracy": float(end["accuracy"]), "selected_minus_endpoint_test_pp": 100 * (float(best["accuracy"]) - float(end["accuracy"]))})
    assert len(paired) == 288 and len(summaries) == 108
    return lookup, paired, summaries, diagnostics


def interactions(data, lookup):
    raw, summary = [], []
    for t in data["targets"]:
        for c in data["chains"]:
            for checkpoint in CHECKPOINTS:
                for n in data["lengths"]:
                    group = []
                    for s in data["seeds"]:
                        values = {(k, b): lookup[t, c, k, b, s, checkpoint, n] for k in data["ks"] for b in data["budgets"]}
                        a = lookup[t, "baseline", 0, 0, s, checkpoint, n]
                        d4 = 100 * (float(values[16, 4_000_000]["accuracy"]) - float(values[2, 4_000_000]["accuracy"]))
                        d32 = 100 * (float(values[16, 32_000_000]["accuracy"]) - float(values[2, 32_000_000]["accuracy"]))
                        row = {"target": t, "target_role": "primary" if t == "lis_length" else "auxiliary", "chain": c, "checkpoint": checkpoint, "length": n, "initialization_seed": s, "data_seed": a["data_seed"], "pretrain_sampling_seed": a["pretrain_sampling_seed"], "adaptation_sampling_seed": a["adaptation_sampling_seed"], "shared_baseline_run_id": a["run_id"], "shared_baseline_accuracy": float(a["accuracy"]), "d4_k16_minus_k2_pp": d4, "d32_k16_minus_k2_pp": d32, "paired_budget_interaction_pp": d32 - d4, "baseline_cancels_exactly": True}
                        for (k, b), r in values.items():
                            tag = f"k{k}_b{b}"
                            row[f"{tag}_accuracy"] = float(r["accuracy"])
                            row[f"{tag}_gain_over_a_pp"] = 100 * (float(r["accuracy"]) - float(a["accuracy"]))
                            row[f"{tag}_run_id"] = r["run_id"]
                        raw.append(row)
                        group.append(row)
                    row = {"target": t, "target_role": group[0]["target_role"], "chain": c, "checkpoint": checkpoint, "length": n, "independent_paired_repeats": 3}
                    for field in ("d4_k16_minus_k2_pp", "d32_k16_minus_k2_pp", "paired_budget_interaction_pp"):
                        row.update({f"{field}_{name}": value for name, value in moments(r[field] for r in group).items()})
                    row.update({f"seed{r['initialization_seed']}_interaction_pp": r["paired_budget_interaction_pp"] for r in group})
                    row["positive_repeat_interactions"] = sum(r["paired_budget_interaction_pp"] > 0 for r in group)
                    row["uncertainty"] = "Within-repeat differences first, then descriptive sample SD over three paired repeats; no marginal-SD significance inference"
                    summary.append(row)
    assert len(raw) == 72 and len(summary) == 24
    return raw, summary


def source_and_cost(data):
    source_rows, majorities, compute = [], [], []
    source_lookup = {(r["run_id"], r["source_task"], r["stage"], r["split"], r["length"]): r for r in data["source_curves"]}
    for r in data["source_curves"]:
        train = source_lookup[r["run_id"], r["source_task"], r["stage"], "source_train", r["length"]]
        dev = source_lookup[r["run_id"], r["source_task"], r["stage"], "source_dev", r["length"]]
        source_rows.append({**r, "above_majority_pp": 100 * (float(r["accuracy"]) - float(r["majority_accuracy"])), "train_probe_minus_dev_pp": 100 * (float(train["accuracy"]) - float(dev["accuracy"]))})
        if r["stage"] == "start" and r["length"] != "all":
            majorities.append({"kind": "source", "run_id": r["run_id"], "task": r["source_task"], **{k: r[k] for k in ("chain", "k", "budget_tokens", "initialization_seed", "data_seed", "length")}, "evaluation_split": r["split"], "majority_answer": r["majority_answer"], "majority_accuracy": r["majority_accuracy"], "count": r["count"], "majority_label_source": "complete 4096-object source training pool for this length"})
    for r in data["curves"] + data["rows"]:
        if r["condition"] != "A" or r["length"] == "all":
            continue
        if r["split"] == "test":
            if r["checkpoint"] != PRIMARY:
                continue
        elif int(r["update"]) != 0:
            continue
        majorities.append({"kind": "target", "run_id": r["run_id"], "task": r["target"], "chain": "shared", "k": "shared", "budget_tokens": "shared", **{k: r[k] for k in ("initialization_seed", "data_seed", "length")}, "evaluation_split": r["split"], "majority_answer": r["majority_answer"], "majority_accuracy": r["majority_accuracy"], "count": r["count"], "majority_label_source": "128 target training labels for this length"})
    fields = ("run_id", "initialization_seed", "data_seed", "pretrain_sampling_seed", "adaptation_sampling_seed", "chain", "k", "budget_tokens", "condition", "target", "updates", "sample_exposures", "input_tokens", "padded_input_tokens", "supervised_tokens", "answer_tokens", "budget_tail_tokens", "uniform_task_count_max_minus_min", "seconds", "peak_memory_bytes", "best_update", "initial_state_sha256", "length_index_schedule_sha256", "fixed_pool_size", "fixed_pool_unique_inputs_exposed", "timer_scope")
    for rec in data["records"]:
        cat = f"pretraining_{rec['budget_tokens']//1_000_000}M" if rec["stage"] == "pretraining" else "baseline_adaptation" if rec["condition"] == "A" else "pretrained_adaptation"
        phases = rec.get("phase_seconds", {})
        compute.append({"stage": rec["stage"], "category": cat, **{key: rec.get(key, "") for key in fields}, "training_updates_seconds": phases.get("training_updates", ""), "source_evaluation_seconds": phases.get("source_evaluation", ""), "target_evaluation_seconds": phases.get("target_evaluation", "")})
    assert len(compute) == 78
    groups = defaultdict(list)
    source_fields = ("chain", "k", "budget_tokens", "source_task", "stage", "split", "length")
    for r in source_rows:
        groups[tuple(r[k] for k in source_fields)].append(r)
    summaries = []
    for key, group in sorted(groups.items()):
        assert len({r["initialization_seed"] for r in group}) == 3
        summaries.append({**dict(zip(source_fields, key)), "independent_paired_repeats": 3, **{f"accuracy_{name}": value for name, value in moments(r["accuracy"] for r in group).items()}, "majority_accuracy_mean": statistics.mean(float(r["majority_accuracy"]) for r in group), "above_majority_pp_mean": statistics.mean(r["above_majority_pp"] for r in group), "answer_ce_mean": statistics.mean(float(r["answer_ce"]) for r in group), "train_probe_minus_dev_pp_mean": statistics.mean(r["train_probe_minus_dev_pp"] for r in group), "repeat_units_at_or_below_majority": sum(r["above_majority_pp"] <= 0 for r in group)})
    exposure_groups = defaultdict(list)
    for r in data["exposures"]:
        exposure_groups[r["run_id"], r["source_task"]].append(r)
    totals = []
    for (rid, task), group in sorted(exposure_groups.items()):
        assert len(group) == 2 and {r["length"] for r in group} == {"10", "12"}
        total = {k: group[0][k] for k in ("run_id", "chain", "k", "budget_tokens", "initialization_seed", "data_seed", "source_task")}
        for field in ("input_tokens", "padded_input_tokens", "supervised_tokens", "answer_tokens", "sample_exposures", "unique_inputs_exposed", "repeat_exposures", "pool_size"):
            total[field] = sum(int(r[field]) for r in group)
        total["pool_coverage"] = total["unique_inputs_exposed"] / total["pool_size"]
        total["mean_exposures_per_pool_input"] = total["sample_exposures"] / total["pool_size"]
        assert total["pool_size"] == 8192
        totals.append(total)
    assert len(totals) == 216
    return source_rows, summaries, majorities, compute, totals


def source_budget_contrasts(data, source, exposures):
    curves = {(r["chain"], int(r["k"]), int(r["budget_tokens"]), int(r["initialization_seed"]), r["source_task"], r["split"]): r for r in source if r["stage"] == "endpoint" and r["length"] == "all"}
    expo = {(r["chain"], int(r["k"]), int(r["budget_tokens"]), int(r["initialization_seed"]), r["source_task"]): r for r in exposures}
    result = []
    for (c, k, b, s, task), small in expo.items():
        if b != 4_000_000:
            continue
        large = expo[c, k, 32_000_000, s, task]
        row = {"chain": c, "k": k, "initialization_seed": s, "source_task": task, "focus_task": task in FOCUS, "small_run_id": small["run_id"], "large_run_id": large["run_id"], "fixed_pool_size_each": 8192}
        for field in ("sample_exposures", "input_tokens", "supervised_tokens", "unique_inputs_exposed", "repeat_exposures", "mean_exposures_per_pool_input"):
            row[f"b4M_{field}"] = small[field]
            row[f"b32M_{field}"] = large[field]
            row[f"ratio_32M_over_4M_{field}"] = large[field] / small[field] if small[field] else ""
        for split in ("source_train", "source_dev"):
            a, b = curves[c, k, 4_000_000, s, task, split], curves[c, k, 32_000_000, s, task, split]
            row[f"b4M_{split}_accuracy"] = float(a["accuracy"])
            row[f"b32M_{split}_accuracy"] = float(b["accuracy"])
            row[f"{split}_accuracy_change_pp"] = 100 * (float(b["accuracy"]) - float(a["accuracy"]))
            row[f"b4M_{split}_answer_ce"] = float(a["answer_ce"])
            row[f"b32M_{split}_answer_ce"] = float(b["answer_ce"])
            row[f"{split}_majority_accuracy"] = float(a["majority_accuracy"])
            assert a["majority_accuracy"] == b["majority_accuracy"]
        result.append(row)
    assert len(result) == 108
    return result


def summarize_source_contrasts(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["chain"], row["k"], row["source_task"]].append(row)
    result = []
    for (chain, k, task), group in sorted(grouped.items()):
        assert len({r["initialization_seed"] for r in group}) == 3
        row = {"chain": chain, "k": k, "source_task": task, "independent_paired_repeats": 3}
        for field in ("source_dev_accuracy_change_pp", "source_train_accuracy_change_pp"):
            row.update({f"{field}_{name}": value for name, value in moments(r[field] for r in group).items()})
        row["positive_dev_changes"] = sum(r["source_dev_accuracy_change_pp"] > 0 for r in group)
        row.update({f"seed{r['initialization_seed']}_dev_change_pp": r["source_dev_accuracy_change_pp"] for r in group})
        result.append(row)
    assert len(result) == 36
    return result


def v1_exploratory_reference(out):
    """Keep the hypothesis-generating evidence separate from all v2 estimates."""
    path = ROOT / "experiments/task_count_v1/k_contrast_summary.csv"
    selected = [r for r in read_csv(path) if r["target"] in ("lis_length", "fixed_points") and r["checkpoint"] == PRIMARY and r["length"] == "all" and r["k_from"] == "2" and r["k_to"] == "16"]
    assert len(selected) == 4
    rows = [{"evidence_stage": "v1_exploration_only", "budget_tokens": 4_000_000, **r, "source_sha256": sha(path), "pooled_with_v2": False} for r in selected]
    write_csv(out / "v1_exploratory_reference.csv", rows)
    return rows


def learning_rate_evidence(out, cfg):
    rows = []
    def at(token, budget):
        progress = min(1., token / budget)
        warmup = 1 / 24
        if progress <= warmup:
            return cfg["learning_rate"] * progress / warmup
        return cfg["minimum_lr"] + .5 * (cfg["learning_rate"] - cfg["minimum_lr"]) * (1 + math.cos(math.pi * (progress - warmup) / (1 - warmup)))
    for budget in cfg["pretraining_budgets_nonpadding_sequence_tokens"]:
        points = sorted(set([int(budget * i / 400) for i in range(401)] + [budget / 24, 4_000_000]))
        for tokens in points:
            if tokens <= budget:
                rows.append({"budget_tokens": budget, "consumed_nonpadding_tokens": tokens, "relative_progress": tokens / budget, "learning_rate": at(tokens, budget), "scope": "Frozen analytic schedule; actual update-end token locations remain in each updates.csv"})
    write_csv(out / "learning_rate_schedule.csv", rows)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for budget in cfg["pretraining_budgets_nonpadding_sequence_tokens"]:
        subset = [r for r in rows if r["budget_tokens"] == budget]
        ax.plot([r["consumed_nonpadding_tokens"] / 1e6 for r in subset], [r["learning_rate"] for r in subset], label=f"{budget//1_000_000}M budget")
    ax.axvline(4, color="gray", linestyle=":", linewidth=1)
    ax.set(xlabel="Consumed nonpadding sequence tokens (millions)", ylabel="Learning rate", title="Same relative-budget rule; different absolute-token LR trajectories")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend()
    for suffix in ("png", "svg"):
        fig.savefig(out / "figures" / f"learning_rate_schedule.{suffix}", dpi=180)
    plt.close(fig)
    return {"4M_at_4M": at(4_000_000, 4_000_000), "32M_at_4M": at(4_000_000, 32_000_000), "warmup_4M": 4_000_000 / 24, "warmup_32M": 32_000_000 / 24}


def figures(out, data, lookup, paired, summaries, raw_interactions, int_summary, source_summary):
    directory = out / "figures"
    directory.mkdir(exist_ok=True)
    colors = {4_000_000: "#1769aa", 32_000_000: "#c45c17"}
    markers = dict(zip(data["seeds"], ["o", "s", "^"]))
    seed_colors = dict(zip(data["seeds"], ["#1769aa", "#c45c17", "#29874a"]))
    means = {(r["target"], r["chain"], r["k"], r["budget_tokens"], r["checkpoint"], r["length"]): r for r in summaries}
    for t in data["targets"]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey="row", constrained_layout=True)
        for col, c in enumerate(data["chains"]):
            for b in data["budgets"]:
                color = colors[b]
                for s in data["seeds"]:
                    vals = [100 * float(lookup[t, c, k, b, s, PRIMARY, "all"]["accuracy"]) for k in data["ks"]]
                    baseline = 100 * float(lookup[t, "baseline", 0, 0, s, PRIMARY, "all"]["accuracy"])
                    axes[0, col].plot([0, 1], vals, color=color, alpha=.4, lw=.8, marker=markers[s], ms=4)
                    axes[1, col].plot([0, 1], [v-baseline for v in vals], color=color, alpha=.4, lw=.8, marker=markers[s], ms=4)
                group = [means[t, c, k, b, PRIMARY, "all"] for k in data["ks"]]
                axes[0, col].errorbar([0, 1], [100*r["accuracy_mean"] for r in group], yerr=[100*r["accuracy_sample_sd"] for r in group], color=color, lw=2.2, capsize=4, label=f"{b//1_000_000}M mean +/- sample SD")
                axes[1, col].errorbar([0, 1], [r["paired_delta_pp_mean"] for r in group], yerr=[r["paired_delta_pp_sample_sd"] for r in group], color=color, lw=2.2, capsize=4)
            baseline = means[t, "baseline", 0, 0, PRIMARY, "all"]
            axes[0, col].axhline(100*baseline["accuracy_mean"], color="black", ls="--", lw=1, label="Shared A mean")
            axes[1, col].axhline(0, color="black", ls="--", lw=1)
            axes[0, col].set(title=c.removeprefix("chain_"), ylim=(0, 100), ylabel="Complete-answer test accuracy (%)")
            axes[1, col].set(ylabel="Paired gain over shared A (pp)", xlabel="Source-task instances k")
            for ax in axes[:, col]:
                ax.set_xticks([0, 1], ["2", "16"])
                ax.grid(axis="y", alpha=.2)
            axes[0, col].legend(fontsize=8, loc="lower left")
        fig.suptitle(f"{t}: fixed compositions, two budgets\nThree paired repeats; small markers o/s/^ = {data['seeds']}; development-selected checkpoint", fontsize=11)
        for suffix in ("png", "svg"):
            fig.savefig(directory / f"{t}_budget_comparison.{suffix}", dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
        selected = [r for r in raw_interactions if r["target"] == t and r["checkpoint"] == PRIMARY and r["length"] == "all"]
        stats = [r for r in int_summary if r["target"] == t and r["checkpoint"] == PRIMARY and r["length"] == "all"]
        fields = ["d4_k16_minus_k2_pp", "d32_k16_minus_k2_pp", "paired_budget_interaction_pp"]
        titles = ["4M: k16 minus k2", "32M: k16 minus k2", "Interaction: D32 minus D4"]
        for ax, field, title in zip(axes, fields, titles):
            for ci, c in enumerate(data["chains"]):
                group = [r for r in selected if r["chain"] == c]
                for offset, r in zip([-.12, 0, .12], group):
                    ax.scatter(ci + offset, r[field], marker=markers[r["initialization_seed"]], color=seed_colors[r["initialization_seed"]], s=42, label=str(r["initialization_seed"]) if ci == 0 else None)
                mean = next(r for r in stats if r["chain"] == c)
                ax.errorbar(ci, mean[f"{field}_mean"], yerr=mean[f"{field}_sample_sd"], fmt="D", color="black", capsize=5, ms=5, label="Mean +/- sample SD" if ci == 0 else None)
            ax.axhline(0, color="gray", ls="--", lw=1)
            ax.set_xticks([0, 1], ["alpha", "beta"])
            ax.set(title=title, ylabel="Paired difference (percentage points)", xlim=(-.5, 1.5))
            ax.grid(axis="y", alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{t}: within-repeat budget interaction\nThree independent paired units; the same scratch baseline cancels", fontsize=11)
        for suffix in ("png", "svg"):
            fig.savefig(directory / f"{t}_budget_interaction.{suffix}", dpi=180)
        plt.close(fig)
    # Matched source learning, with training-probe and development distinguished.
    relevant = [r for r in source_summary if r["source_task"] in FOCUS and r["length"] == "all"]
    for task in FOCUS:
        groups = sorted({(r["chain"], int(r["k"])) for r in relevant if r["source_task"] == task})
        fig, axes = plt.subplots(1, len(groups), figsize=(4*len(groups), 4.3), squeeze=False, constrained_layout=True)
        for ax, (c, k) in zip(axes[0], groups):
            for b in data["budgets"]:
                for split, style in (("source_train", "--"), ("source_dev", "-")):
                    group = [next(r for r in relevant if r["source_task"] == task and r["chain"] == c and int(r["k"]) == k and int(r["budget_tokens"]) == b and r["stage"] == stage and r["split"] == split) for stage in ("start", "midpoint", "endpoint")]
                    ax.plot([0, .5, 1], [100*r["accuracy_mean"] for r in group], style, color=colors[b], marker="o", label=f"{b//1_000_000}M {'train probe' if split=='source_train' else 'dev'}")
                    if split == "source_dev" and b == data["budgets"][0]:
                        ax.axhline(100*group[0]["majority_accuracy_mean"], color="gray", ls=":", label="Train-derived majority on dev")
            ax.set(title=f"{c.removeprefix('chain_')} k={k}", xlabel="Relative budget stage", ylabel="Complete-answer accuracy (%)", ylim=(0, 100))
            ax.set_xticks([0, .5, 1], ["start", "mid", "end"])
            ax.grid(axis="y", alpha=.2)
        handles, labels = axes[0, -1].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=8, loc="outside lower center", ncol=3)
        fig.suptitle(f"{task}: source learning, mean of three paired repeats", fontsize=11)
        for suffix in ("png", "svg"):
            fig.savefig(directory / f"source_{task}.{suffix}", dpi=180)
        plt.close(fig)
    return learning_rate_evidence(out, data["config"])


def write_report(out, data, summaries, raw_interactions, int_summary, source, source_summaries, exposures, source_contrasts, compute, diagnostics, lr, old_reference):
    # Outcome-specific prose belongs only to this archived result; tables always recompute.
    archived_result = sha(out / "raw_test_results.csv") == ARCHIVED_TEST_SHA
    totals = {}
    for category in dict.fromkeys(r["category"] for r in compute):
        group = [r for r in compute if r["category"] == category]
        totals[category] = {"runs": len(group), **{f: sum(float(r[f]) for r in group) for f in ("updates", "input_tokens", "supervised_tokens", "sample_exposures", "seconds")}}
    pre = [r for r in data["records"] if r["stage"] == "pretraining"]
    primary = [r for r in int_summary if r["checkpoint"] == PRIMARY and r["length"] == "all"]
    endpoint = [r for r in source if r["stage"] == "endpoint" and r["split"] == "source_dev" and r["length"] == "all"]
    result = {"status": "completed", "pretraining_models": 24, "target_training_runs": 54, "independent_paired_repeat_units": 3, "raw_test_rows": len(data["rows"]), "primary_target": "lis_length", "auxiliary_target": "fixed_points", "total_by_category": totals, "primary_interactions": primary, "training_stage_wall_seconds": data["training"]["stage_seconds"], "test_seconds": data["test"]["seconds"], "pretraining_token_tail_min": min(r["budget_tail_tokens"] for r in pre), "pretraining_token_tail_max": max(r["budget_tail_tokens"] for r in pre), "source_endpoint_records": len(endpoint), "source_endpoint_at_or_below_majority": sum(r["above_majority_pp"] <= 0 for r in endpoint), "lr_schedule_values": lr, "raw_results_sha256": sha(out / "raw_test_results.csv"), "frozen_protocol_sha256": sha(out / "FROZEN_PROTOCOL.json"), "frozen_selection_sha256": sha(out / "FROZEN_SELECTIONS.json"), "no_test_based_selection": True, "additional_training_triggered": False}
    write_json(out / "SUMMARY.json", result)
    text = ["# PermWorld task-count v2：任务数量与训练预算的配对交互\n\n", "本轮已完成冻结范围内的 **24 个预训练模型、48 次预训练后适配及 6 个共享随机初始化基线**，共 54 次目标训练。LIS 长度是主要目标，fixed_points 是辅助目标；每个目标仍只有 256 个适配样本。三个新的配对重复为 4103、5207、6301。学习率方案预先固定；checkpoint 只依据开发集选择。冻结选择后生成独立测试输入并统一评估。没有根据结果追加预算、换任务或增加搜索。\n\n", "本轮假设来自 v1 对 LIS 随 k 增大下降及 fixed_points 依赖任务链的探索观察。v1 的旧结果不是本轮重复，不与新种子合并统计。两条链只保留原名单的 k=2、16，分别从同一配对随机初始化训练到 400 万或 3200 万 tokens；每条链内任务组成在两档预算之间不变。\n\n", "## 主要结果：先做逐重复差分，再汇总\n\n", "D4 表示 400 万预算下 k16−k2，D32 表示 3200 万预算下 k16−k2；交互差为 D32−D4。正交互表示更大预算使 k16 相对 k2 的表现改善，不等于所有条件的绝对准确率都提高，也不等于已经消除负迁移。下表单位均为百分点，± 为三个配对差的样本标准差，不是置信区间。\n\n", "| 目标 | 链 | D4 均值 | D32 均值 | 交互差均值 ± SD | 三个交互差范围 |\n|---|---|---:|---:|---:|---:|\n"]
    # The primary table always contains v2 only; old seed outcomes appear separately.
    for r in primary:
        text.append(f"| {r['target']} | {r['chain'].removeprefix('chain_')} | {r['d4_k16_minus_k2_pp_mean']:+.2f} | {r['d32_k16_minus_k2_pp_mean']:+.2f} | {r['paired_budget_interaction_pp_mean']:+.2f} ± {r['paired_budget_interaction_pp_sample_sd']:.2f} | [{r['paired_budget_interaction_pp_min']:+.2f}, {r['paired_budget_interaction_pp_max']:+.2f}] |\n")
    text.append("\n逐重复结果如下。三种差值均在同一链、同一初始化、同一数据及采样重复单元内计算，不能由两组各自标准差替代。共享随机初始化基线在交互差中精确抵消。\n\n| 目标 | 链 | 初始化种子 | D4 | D32 | D32−D4 |\n|---|---|---:|---:|---:|---:|\n")
    for r in raw_interactions:
        if r["checkpoint"] == PRIMARY and r["length"] == "all":
            text.append(f"| {r['target']} | {r['chain'].removeprefix('chain_')} | {r['initialization_seed']} | {r['d4_k16_minus_k2_pp']:+.2f} | {r['d32_k16_minus_k2_pp']:+.2f} | {r['paired_budget_interaction_pp']:+.2f} |\n")
    text.append("\n![LIS 预算交互](figures/lis_length_budget_interaction.png)\n\n![fixed_points 预算交互](figures/fixed_points_budget_interaction.png)\n\n")
    for t in data["targets"]:
        findings = [r for r in primary if r["target"] == t]
        text.append(f"**{t}**：")
        text.append("；".join(f"{r['chain'].removeprefix('chain_')} 链交互均值 {r['paired_budget_interaction_pp_mean']:+.2f} 个百分点，{r['positive_repeat_interactions']}/3 个重复为正" for r in findings) + "。这些是当前有限重复的实际观察，不能推广为所有任务组成的预算规律。\n\n")
    if archived_result:
        text.append("LIS 在新种子的 4M 条件下复现了 v1 的方向：两条链均为三个重复的 k16 低于 k2。增加预算后，alpha 的 k16 在三个重复都反超 k2；beta 的交互虽三个重复均为正，但 32M 下 k16−k2 本身只有两个重复为正、一个为负，平均接近零。因此 beta 更适合描述为原有 k16 劣势缩小，而非稳定建立 k16 优势。这里的两条链共享三个重复单元，不能把六个链内交互当作六个独立重复。\n\nfixed_points 呈现相反的链间方向：alpha 的 32M k16 出现明显改善，beta 的 32M k16 反而更差；后者的交互范围从 −22.17 到 −0.29 个百分点，幅度不稳定。预算效果依赖任务组成及目标，不能概括为扩大预算普遍消除负迁移。\n\n")
    text.append("## v1 探索证据：独立列出，不并入本轮统计\n\n旧种子为 1103、2207、3301，只有 4M 预算。下表直接保留 v1 的 k16−k2 配对差及源 CSV 哈希；不同测试输入和重复单元不与 v2 合并，也不能把跨轮差异仅归于训练预算。\n\n| 目标 | 链 | v1 D4 均值 ± SD（百分点） | v1 逐种子差（1103 / 2207 / 3301） |\n|---|---|---:|---|\n")
    for r in old_reference:
        text.append(f"| {r['target']} | {r['chain'].removeprefix('chain_')} | {float(r['paired_change_pp_mean']):+.2f} ± {float(r['paired_change_pp_sample_sd']):.2f} | {float(r['seed1103_change_pp']):+.2f} / {float(r['seed2207_change_pp']):+.2f} / {float(r['seed3301_change_pp']):+.2f} |\n")
    text.append("\n[v1 探索参照 CSV](v1_exploratory_reference.csv)仅解释假设来源；本轮的交互估计全部来自 v2。\n\n")
    text.append("## 各条件准确率与共享 A 的收益\n\n| 目标 | 链 | k | 预算 | 测试准确率均值 | 相对 A 的配对收益均值 ± SD（百分点） |\n|---|---|---:|---:|---:|---:|\n")
    for r in summaries:
        if r["checkpoint"] == PRIMARY and r["length"] == "all":
            text.append(f"| {r['target']} | {r['chain'].removeprefix('chain_')} | {r['k']} | {r['budget_tokens']//1_000_000}M | {100*r['accuracy_mean']:.2f}% | {r['paired_delta_pp_mean']:+.2f} ± {r['paired_delta_pp_sample_sd']:.2f} |\n")
    if archived_result:
        text.append("\nLIS 的 32M 四个条件在每个重复均高于其共享 A；不过 alpha k2 的平均准确率从 57.03% 降到 54.10%，说明正交互也可以包含小 k 表现下降。Alpha k16 从 50.85% 升到 65.85%，才是其反超的主要变化。Beta 的 k2 和 k16 都随预算改善，分别为 54.82%→61.46% 和 49.02%→62.21%。\n\nfixed_points 只有 alpha k16 的 32M 条件获得平均正迁移：88.44%，相对 A 为 +23.63 个百分点，三个重复均为正。其余七个条件的平均收益均为负；beta k16 的 32M 仅 43.39%，相对 A 为 −21.42 个百分点。大的 alpha 交互不能掩盖这些保留的负结果。\n\n")
    text.append("\n![LIS 准确率和基线差](figures/lis_length_budget_comparison.png)\n\n![fixed_points 准确率和基线差](figures/fixed_points_budget_comparison.png)\n\n## 预算增加究竟改变了什么\n\n")
    text.append(f"两档预算均从配对的随机初始化和全新 AdamW 开始；32M 没有接续 4M 衰减终点。相对预算进度规则相同，但热身长度分别为 {lr['warmup_4M']:,.2f} 与 {lr['warmup_32M']:,.2f} tokens。在累计 4M tokens 处，4M 曲线学习率是 {lr['4M_at_4M']:.6g}，32M 曲线仍为 {lr['32M_at_4M']:.6g}。因此本轮比较的是更大预算及其预先固定的学习率轨迹，不单独识别更新次数和学习率轨迹各自的作用。实际逐更新 LR 在各运行的 updates.csv 中，解析规则在 [learning_rate_schedule.csv](learning_rate_schedule.csv)。\n\n![两档预算的学习率轨迹](figures/learning_rate_schedule.png)\n\n")
    text.append("源训练池在两档预算完全相同：每个重复、每个长度 4096 个对象，共 8192 个，不是八倍独立数据。但同一任务在 4M 时未必见过整个输入池，尤其是 k16；32M 可以同时提高该任务对固定池的覆盖率并增加已见任务—输入对的重复暴露，不能把每个新增样本都称为已见任务—输入对的重复。任务按样本均匀采样；不同任务的 prompt/答案长度不同，token 份额并不相等。最后只取容纳于预算的完整样本前缀，保留尾差。32M 是约八倍总 tokens，不宣称 k16 的每任务暴露与 k2 的 4M 严格匹配。每个任务的独立输入数最多 8192；跨任务把相同输入重复计数不能成为更多独立对象。每任务覆盖率、首次暴露与重复暴露的实际差异保留在 task_exposures.csv 和 source_budget_comparisons.csv。\n\n")
    text.append("## 源任务学习与重复暴露\n\n训练准确率使用每长度固定前 256 个源训练对象的诊断子集；开发集使用每长度全部 256 个对象。多数答案从每长度完整 4096 个源训练标签确定。曲线分别在起点、首次跨过半预算的更新及终点保存。以下聚焦 LDS、I3、D3，其他任务完整保留在 CSV 中。\n\n| 链 | k | 源任务 | 预算 | 开发准确率 | 训练诊断准确率 | 开发多数基线 | 样本暴露均值 | 独立输入均值 |\n|---|---:|---|---:|---:|---:|---:|---:|---:|\n")
    for c in data["chains"]:
        for k in data["ks"]:
            for task in FOCUS:
                for b in data["budgets"]:
                    rows = [r for r in source_summaries if r["chain"] == c and int(r["k"]) == k and int(r["budget_tokens"]) == b and r["source_task"] == task and r["stage"] == "endpoint" and r["length"] == "all"]
                    if not rows:
                        continue
                    dev = next(r for r in rows if r["split"] == "source_dev")
                    train = next(r for r in rows if r["split"] == "source_train")
                    e = [r for r in exposures if r["chain"] == c and int(r["k"]) == k and int(r["budget_tokens"]) == b and r["source_task"] == task]
                    text.append(f"| {c.removeprefix('chain_')} | {k} | {task} | {b//1_000_000}M | {100*dev['accuracy_mean']:.2f}% | {100*train['accuracy_mean']:.2f}% | {100*dev['majority_accuracy_mean']:.2f}% | {statistics.mean(r['sample_exposures'] for r in e):,.0f} | {statistics.mean(r['unique_inputs_exposed'] for r in e):,.1f} |\n")
    text.append(f"\n全部 {len(endpoint)} 个源任务×模型终点记录中，{sum(r['above_majority_pp'] <= 0 for r in endpoint)} 个开发准确率未超过各自多数基线。它们重叠使用任务和模型，不能作为 {len(endpoint)} 个独立实验。训练/开发差距、源任务学习改善及迁移交互同时发生，只能描述关联，不能直接证明模型干扰、数学恒等式利用或其反面。\n\n")
    below = {b: sum(r["above_majority_pp"] <= 0 for r in endpoint if int(r["budget_tokens"]) == b) for b in data["budgets"]}
    text.append(f"按预算分别计，4M 有 {below[4_000_000]}/108 个源任务终点未超过多数基线，32M 仍有 {below[32_000_000]}/108 个。重点源任务的预算变化按同一重复配对后如下；这些数值没有把不同任务当作重复。\n\n| 链 | k | 源任务 | 32M−4M 开发准确率变化均值 ± SD（百分点） | 逐重复为正数量 |\n|---|---:|---|---:|---:|\n")
    for r in summarize_source_contrasts(source_contrasts):
        if r["source_task"] in FOCUS:
            text.append(f"| {r['chain'].removeprefix('chain_')} | {r['k']} | {r['source_task']} | {r['source_dev_accuracy_change_pp_mean']:+.2f} ± {r['source_dev_accuracy_change_pp_sample_sd']:.2f} | {r['positive_dev_changes']}/3 |\n")
    if archived_result:
        text.append("\nLDS 和三元子序列计数的开发表现都随预算改善，k16 每任务独立输入覆盖也从约 5100 增至约 8190，接近固定池的 8192 上限；每任务样本暴露约 8064→64513。同时 k16 的 LDS 仍低于 alpha k2 的 99.09%，三元计数也远未充分学会。Beta k2 的 D3 在 32M 时训练诊断为 55.66%、开发为 40.30%，约 15.36 个百分点的差距说明更多重复暴露没有自动解决泛化。Alpha k2 的 LDS 从 84.57% 提升到 99.09%，而其 LIS 迁移均值下降，这直接限制了“相关源任务准确率更高就必然迁移更好”的简单解释。\n\n与 fixed_points 有明确联合关系的 alpha k16 exceedances 和 deficiencies，开发准确率分别从 43.23%→99.35% 与 38.93%→98.96%，同时 fixed_points 迁移明显改善；这是值得后续研究的对应现象。但 even_cycle_count 仍接近多数基线，不能说所有相关数学量都被充分学会。Beta k16 的 anti_fixed_points 也从 44.27% 提高到 63.35%，其 fixed_points 迁移却恶化。合法公式存在及相关源任务进步均不足以证明模型利用了公式，当前记录不能区分公式利用、一般表示变化和优化过程等机制。\n\n")
    for task in FOCUS:
        text.append(f"![{task} 源任务曲线](figures/source_{task}.png)\n\n")
    text.append("## 配置、隔离和适配核对\n\n模型与 v1 相同：平台当前 203-token 词表，4 层、宽度 256、8 头 Transformer、FFN 比例 4、dropout 0.1、上下文 128、共享 embedding/输出头。使用 one_line_v1 和 permworld_sequence_v1。损失先在每个样本的答案 token 加 EOS 上平均交叉熵，再平均样本；准确率要求完整自由生成答案加 EOS。I3/D3 的多 token 整数答案没有截断或取模。\n\n每次适配均重置全部四个原留出任务的专有 token 行（含共享输出头），其余预训练权重准确加载，创建全新优化器。800 次更新、batch 64、初始 LR 3e-4、100 次热身后余弦衰减到 3e-5；每 100 次更新评估开发集，共八个合格 checkpoint。按开发准确率最高、答案 CE 最低、最早更新的固定规则选择；起点不参与选择。所有条件只有一个 LR 候选。\n\n全输入按完整 inverse/reverse/complement 轨道隔离，覆盖角色、重复、历史 pilot、平台及 v1。生成前容量检查，不静默放宽规则。每目标适配 128×2=256 个不同对象，开发 256×2，最终测试 512×2。长度 10、12 都出现在训练中，按长度结果不是长度外推。开发选择冻结后才生成最终测试。\n\n")
    text.append("| 初始化 | 数据 | 测试 | 源采样 | 适配采样 | 源 dropout | 适配 dropout |\n|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in data["config"]["repeats"]:
        text.append("| " + " | ".join(str(r[k]) for k in ("initialization_seed", "data_seed", "test_seed", "pretrain_sampling_seed", "adaptation_sampling_seed", "pretrain_dropout_seed", "adaptation_dropout_seed")) + " |\n")
    text.append("\n目标多数答案仅从每长度的 128 个训练标签估计；新测试上的均值如下。\n\n| 目标 | n=10 多数准确率 | n=12 多数准确率 |\n|---|---:|---:|\n")
    for t in data["targets"]:
        vals = [next(r["majority_accuracy_mean"] for r in summaries if r["target"] == t and r["chain"] == "baseline" and r["checkpoint"] == PRIMARY and r["length"] == n) for n in ("10", "12")]
        text.append(f"| {t} | {100*vals[0]:.2f}% | {100*vals[1]:.2f}% |\n")
    adapts = [r for r in data["records"] if r["stage"] == "adaptation"]
    diagnostic = [r for r in diagnostics if r["length"] == "all"]
    text.append(f"\n54 次适配中 {sum(r['best_update'] < 800 for r in adapts)} 次开发选择早于更新 800。新测试上，预选 checkpoint 比终点高的有 {sum(r['selected_minus_endpoint_test_pp'] > 0 for r in diagnostic)} 次，低的有 {sum(r['selected_minus_endpoint_test_pp'] < 0 for r in diagnostic)} 次，其余相同。这是预先指定的辅助比较，不用于重新选择模型。完整终点交互与按长度结果在原始 CSV 和交互 CSV 中。\n\n")
    text.append("| 目标 | 链 | 开发预选交互均值 | 固定终点交互均值（百分点） |\n|---|---|---:|---:|\n")
    for r in primary:
        end = next(x for x in int_summary if x["target"] == r["target"] and x["chain"] == r["chain"] and x["checkpoint"] == "fixed_endpoint" and x["length"] == "all")
        text.append(f"| {r['target']} | {r['chain'].removeprefix('chain_')} | {r['paired_budget_interaction_pp_mean']:+.2f} | {end['paired_budget_interaction_pp_mean']:+.2f} |\n")
    if archived_result:
        text.append("\n四个目标×链的平均交互方向在固定终点相同，主要方向并非只出现在开发预选 checkpoint；具体幅度和种子模式仍随 checkpoint 改变。终点结果只是预先指定的辅助诊断，不取代主结果。\n\n")
    text.append("## 实际计算成本\n\n| 阶段 | 运行数 | 更新数 | 非 padding 序列 tokens | 监督 tokens | 累计运行分钟 |\n|---|---:|---:|---:|---:|---:|\n")
    for cat, r in totals.items():
        text.append(f"| {cat} | {r['runs']} | {r['updates']:,.0f} | {r['input_tokens']:,.0f} | {r['supervised_tokens']:,.0f} | {r['seconds']/60:.2f} |\n")
    text.append(f"\n训练阶段墙钟时间 {result['training_stage_wall_seconds']/60:.2f} 分钟，独立测试 {result['test_seconds']/60:.2f} 分钟。各预训练模型尾差为 {result['pretraining_token_tail_min']}–{result['pretraining_token_tail_max']} tokens。运行计时包含相应训练、开发评估和 checkpoint/CSV 写入，不含数据编码；阶段计时与运行计时重叠，不能相加。每次运行的 padded tokens、监督 tokens、样本暴露、内存及时间见 compute.csv。\n\n")
    smoke = read_json(out / "SMOKE.json")
    source_update_seconds = sum(float(r["training_updates_seconds"]) for r in compute if r["stage"] == "pretraining")
    text.append(f"Smoke 固定小批次测速为 {smoke['timing_tokens_per_second']:,.0f} tokens/s，据此预估全部源模型的纯更新耗时 {smoke['estimated_all_pretraining_training_only_seconds']/60:.2f} 分钟。实际源模型纯更新累计 {source_update_seconds/60:.2f} 分钟，含源评估及写入的预训练运行累计 {sum(r['seconds'] for r in pre)/60:.2f} 分钟；前述 {result['training_stage_wall_seconds']/60:.2f} 分钟还含全部目标适配和阶段开销。Smoke 是资源可行性估算，正式成本以实际记录为准，不能用估计值替代。\n\n")
    text.append("## 证据边界与收口\n\n三个新配对重复是统计重复单元。两条链、两个目标、多个 k/预算、共享源模型及共享随机基线不能增加独立重复数。这里只展示逐重复差、均值、样本 SD 和范围，未从各条件边际 SD 推断显著性，也没有进行结果驱动的最佳任务链选择。少量重复与有限测试输入留下较大不确定性。\n\n同一源集合内预算变化没有换任务，但跨 k 仍改变组成、数学关系和每任务暴露。两档预算还伴随不同的绝对 token 学习率轨迹；固定池重复训练与增加独立数据也未分离。Alpha k16 的 fixed_points 联合可推导关系与 beta 的输入变换关系均保留，合法数学关系不是数据泄漏。现有记录没有证明模型采用这些公式。\n\n负收益、零收益、种子分歧和训练不足全部保留。本轮仅回答两条既定链、两档预算、一个模型、两个长度和 256 个目标样本下的有限预算交互；不宣称 scaling law。本轮已按冻结范围结束，不追加预算或替换任务。\n\n- [逐重复原始结果](raw_test_results.csv)、[逐重复预算交互](paired_budget_interactions.csv)、[交互汇总](budget_interaction_summary.csv)、[相对共享 A 的收益](paired_deltas.csv)\n- [所有源任务学习](source_learning.csv)、[源任务预算配对比较](source_budget_comparisons.csv)、[每任务每长度暴露](task_exposures.csv)、[每任务合计暴露](task_exposure_totals.csv)\n- [目标训练/开发曲线](adaptation_curves.csv)、[按长度多数基线](majority_per_length.csv)、[实际计算成本](compute.csv)\n- [冻结配置](FROZEN_PROTOCOL.json)、[任务名单及数学关系](design/README.md)、[复现命令](README.md)\n")
    if (out / "RUN_AUDIT.json").exists():
        audit = read_json(out / "RUN_AUDIT.json")
        assert audit["status"] == "pass" and audit["raw_result_rows"] == 324
        text.append(f"\n[独立运行审计](RUN_AUDIT.json)已通过，覆盖 24 个源模型、54 次目标训练、135 个正式权重文件及 12 对从同一初始化开始的预算运行；重放源和目标采样，核对权重加载、四个留出 token 行、优化器重置、开发选择及全部 324 行测试记录。另复算 {audit['development_checkpoints_recomputed']} 个开发 checkpoint，未增加测试神经网络前向。审计用时 {audit['seconds']:.2f} 秒，详见 [预算配对](audit_budget_pairing.csv)、[逐运行核算](audit_run_accounting.csv)及[开发前向核验](audit_development_forwards.csv)。\n")
    text.append("\n工程 smoke 最初遇到缩小版输入清单与归档哈希不一致，在进入训练前停止；修正为复制原始清单及归档的精确字节后，最终 smoke 通过。初始设置失败证据保留于 smoke/failed_manifest_setup，不能与正式模型训练失败混为一谈。[SMOKE.json](SMOKE.json)记录最终参数化任务、多 token 答案、损失、保存加载、适配和计时检查。[分析自检](SUMMARY_SELF_CHECK.json)验证四格差分符号、共享基线抵消以及缺失/重复行拒绝。\n")
    if (out / "PRESERVATION_AFTER.json").exists():
        text.append("\n[历史保留核验](PRESERVATION_AFTER.json)记录此前结果及 v1 的保留情况。\n")
    (out / "REPORT.md").write_text("".join(text), encoding="utf-8")
    return result


def reproduction_readme(out):
    text = """# PermWorld task-count v2: budget interaction

Completed scope: 24 pretrained models, 48 pretrained adaptations and six shared
scratch baselines. LIS length is primary; fixed points is auxiliary. Each unchanged
v1 chain contributes k2 and k16 at 4M and 32M sequence-token budgets, with three new
paired repeats. The Chinese [report](REPORT.md) is explicitly requested; code and
other documentation are English. V1 exploration is not pooled with these data.

Start with [paired interactions](paired_budget_interactions.csv),
[interaction summary](budget_interaction_summary.csv), [raw test results](raw_test_results.csv)
and the [frozen plan](EXPERIMENT_PLAN.md). Each interaction subtracts the within-repeat
k16-minus-k2 effect at 4M from that at 32M. Shared baselines cancel exactly.
Three repeats remain three statistical units, regardless of reused models or targets.

Both budgets begin at the same paired random initialization with empty optimizers;
32M does not continue a decayed 4M endpoint. Both use warmup over 1/24 of their
own budget and cosine decay in normalized token progress. The absolute-token LR
trajectories differ: at 4M tokens the short run has reached 3e-5 while the 32M run
is still near its peak. See [schedule CSV](learning_rate_schedule.csv) and
[schedule plot](figures/learning_rate_schedule.png). The same finite 8192-object
source pool is reused, so eightfold tokens do not mean eightfold independent data
or exact per-task exposure matching. A task may not have covered the full pool at
4M, especially at k16: additional tokens can expose previously unseen task-input
pairs within that fixed pool as well as repeat already seen pairs. The per-task
unique-input and repeated-exposure counts distinguish these cases; not every
additional training example is a repeated task-input pair. Input IDs and counts remain in each source
run's input_exposures.csv; task_exposures.csv records per-task/per-length samples,
tokens, unique inputs and repeated exposures.

## Reproduce the frozen scope

From the repository root, use the existing project Python environment and a fresh
output directory. CUDA is required for training/testing. Historical input pools,
including v1, are needed for the unchanged full inverse/reverse/complement orbit
exclusion. Do not overwrite the completed output or relax the data rules.

```powershell
.venv/Scripts/python.exe experiments/task_count_v2/design.py --out experiments/task_count_v2_reproduction/design
Copy-Item -LiteralPath experiments/task_count_v2/EXPERIMENT_PLAN.md -Destination experiments/task_count_v2_reproduction/EXPERIMENT_PLAN.md
.venv/Scripts/python.exe experiments/task_count_v2/data_pipeline.py prepare --out experiments/task_count_v2_reproduction/data
.venv/Scripts/python.exe experiments/task_count_v2/train.py smoke --out experiments/task_count_v2_reproduction
.venv/Scripts/python.exe experiments/task_count_v2/train.py freeze --out experiments/task_count_v2_reproduction
.venv/Scripts/python.exe experiments/task_count_v2/train.py train --out experiments/task_count_v2_reproduction
.venv/Scripts/python.exe experiments/task_count_v2/train.py select --out experiments/task_count_v2_reproduction
.venv/Scripts/python.exe experiments/task_count_v2/train.py test --out experiments/task_count_v2_reproduction
.venv/Scripts/python.exe experiments/task_count_v2/summarize.py --out experiments/task_count_v2_reproduction
```

The design command creates the fresh directory. Freeze binds the copied plan,
design, code, smoke evidence and training/development data. Test inputs are only
generated after all 54 development selections are frozen. Completed runs are
verified before reuse; partial runs fail visibly, without silent extension.
The reporting script never imports Torch or runs a model: it verifies identities,
hashes, development choices and paired accounting, then derives tables and plots.

For an exact UTF-8 training log instead of PowerShell's output-file defaults,
replace the training command with the following (an existing log is refused):

```powershell
.venv/Scripts/python.exe experiments/task_count_v2/run_logged.py --log experiments/task_count_v2_reproduction/execution.log -- .venv/Scripts/python.exe experiments/task_count_v2/train.py train --out experiments/task_count_v2_reproduction
```

The initial engineering smoke had a manifest setup mismatch before any model
training. Its evidence remains in smoke/failed_manifest_setup; the corrected
smoke copied exact manifest/archive bytes and passed. This is separate from
formal-run outcomes. SUMMARY_SELF_CHECK.json checks interaction arithmetic and
rejects duplicate/missing records using synthetic numbers without model calls.

For the archived run, verify frozen code/data without training:

```powershell
.venv/Scripts/python.exe experiments/task_count_v2/train.py verify
```

The completed [independent audit](RUN_AUDIT.json) passed all 24 source and 54 target
runs, 135 formal checkpoint files, 12 budget pairs and all 324 raw result rows.
It independently rebuilt all three initializations and replayed sampling, checked
fresh optimizers and target-row resets, and recomputed two development checkpoints.
It performed no new test neural forwards. See [budget pairing](audit_budget_pairing.csv),
[run accounting](audit_run_accounting.csv) and [development forwards](audit_development_forwards.csv).

After reproducing training/testing, reproduce the audit and update the report:

```powershell
.venv/Scripts/python.exe experiments/task_count_v2/audit_run.py --out experiments/task_count_v2_reproduction --distributions-only
.venv/Scripts/python.exe experiments/task_count_v2/audit_run.py --out experiments/task_count_v2_reproduction --forwards
.venv/Scripts/python.exe experiments/task_count_v2/summarize.py --out experiments/task_count_v2_reproduction
```

The `--forwards` option uses local checkpoints for development-only numerical
checks. To create a new release manifest, first place the canonical experiment
scripts alongside the new output, then inventory and verify the finalized bytes:

```powershell
Copy-Item -LiteralPath experiments/task_count_v2/train.py,experiments/task_count_v2/design.py,experiments/task_count_v2/data_pipeline.py,experiments/task_count_v2/summarize.py,experiments/task_count_v2/audit_run.py,experiments/task_count_v2/release.py,experiments/task_count_v2/run_logged.py,experiments/task_count_v2/SUMMARY_SELF_CHECK.json -Destination experiments/task_count_v2_reproduction
.venv/Scripts/python.exe experiments/task_count_v2/release.py --out experiments/task_count_v2_reproduction
.venv/Scripts/python.exe experiments/task_count_v2/release.py --out experiments/task_count_v2_reproduction --verify
```

For the original archived output, the release writer is run once after all
artifacts are final. Full verification includes local weights; a Git checkout
without weights explicitly uses source-only verification:

```powershell
.venv/Scripts/python.exe experiments/task_count_v2/release.py --verify
.venv/Scripts/python.exe experiments/task_count_v2/release.py --verify --source-only
```

All derived CSVs and figures can be regenerated from the delivered raw
tables without weights using `summarize.py`. Any regenerated file changes can
invalidate the old release inventory; verify the archive before regenerating,
or write derived output in a copied directory.

## Artifacts

- FROZEN_PROTOCOL.json: complete model, loss, optimizer, seeds, budgets and hashes.
- FROZEN_SELECTIONS.json: all development-only checkpoint decisions.
- raw_test_results.csv and results_seed*.csv: selected and endpoint tests at n10,
  n12 and their equally weighted aggregate; no condition is discarded.
- paired_budget_interactions.csv: D4, D32 and D32-minus-D4 for each repeat.
- v1_exploratory_reference.csv: the four old 4M k16-minus-k2 summaries and seed
  values, with a source hash; never pooled with v2 or used as new repeats.
- budget_interaction_summary.csv: three-repeat descriptive mean, sample SD and
  range of those paired differences, with individual values retained.
- paired_deltas.csv and summary.csv: condition minus the same repeat's shared A.
- source_learning.csv, source_summary.csv and source_focus.csv: all source
  start/mid/end learning, and LDS/I3/D3 focus; training is a fixed diagnostic probe.
- source_budget_comparisons.csv: matched 32M-minus-4M source development/training
  changes and actual exposure ratios; these are associations, not mechanisms.
- source_budget_summary.csv: descriptive paired source-change means, sample SD,
  ranges and per-repeat development changes; tasks are never pooled as repeats.
- task_exposures.csv and task_exposure_totals.csv: samples, total/supervised tokens,
  unique objects, repeat exposures and pool coverage, by length and summed.
- adaptation_curves.csv, checkpoint_diagnostics.csv and majority_per_length.csv:
  target curves, preselected-versus-endpoint diagnostics and train-derived baselines.
- compute.csv: actual tokens, updates, runtime, memory, per-run unique source inputs.
- figures/: budget interactions, accuracy/gains, source learning and LR trajectories.
- SUMMARY.json: completed counts, timings and evidence bindings.

Weights remain local and Git-ignored. Other artifacts are shareable on the dedicated
experiment branch. Archived Windows checkpoint paths may need separator
normalization for a POSIX consumer; fresh runs write native paths. Reading the CSVs
and reproducing plots needs no weights. Negative or undertrained results are retained.
The completed frozen scope does not trigger extra training or task replacement.
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def summarize(out):
    data = collect(out)
    lookup, paired, summaries, diagnostics = target_summaries(data)
    raw_interactions, int_summary = interactions(data, lookup)
    source, source_summaries, majorities, compute, exposure_totals = source_and_cost(data)
    source_contrasts = source_budget_contrasts(data, source, exposure_totals)
    write_csv(out / "source_budget_summary.csv", summarize_source_contrasts(source_contrasts))
    old_reference = v1_exploratory_reference(out)
    for filename, rows in (("paired_deltas.csv", paired), ("summary.csv", summaries), ("paired_budget_interactions.csv", raw_interactions), ("budget_interaction_summary.csv", int_summary), ("checkpoint_diagnostics.csv", diagnostics), ("source_learning.csv", source), ("source_summary.csv", source_summaries), ("source_focus.csv", [r for r in source if r["source_task"] in FOCUS]), ("source_budget_comparisons.csv", source_contrasts), ("adaptation_curves.csv", data["curves"]), ("task_exposures.csv", data["exposures"]), ("task_exposure_totals.csv", exposure_totals), ("compute.csv", compute), ("majority_per_length.csv", majorities)):
        write_csv(out / filename, rows)
    for seed in data["seeds"]:
        write_csv(out / f"results_seed{seed}.csv", [r for r in data["rows"] if int(r["initialization_seed"]) == seed])
    lr = figures(out, data, lookup, paired, summaries, raw_interactions, int_summary, source_summaries)
    result = write_report(out, data, summaries, raw_interactions, int_summary, source, source_summaries, exposure_totals, source_contrasts, compute, diagnostics, lr, old_reference)
    reproduction_readme(out)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(summarize(args.out.resolve()), indent=2))
