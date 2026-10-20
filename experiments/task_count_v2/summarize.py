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
    text = ["# PermWorld task-count v2: paired interaction between task count and training budget\n\n", "The frozen scope is complete: **24 pretrained models, 48 adaptations from pretrained models, and 6 shared random-initialization baselines**, totaling 54 target-training runs. LIS length is the primary target and fixed_points is auxiliary; each target still uses only 256 adaptation examples. The three new paired repeats are 4103, 5207, and 6301. Learning-rate schedules were fixed in advance; checkpoint selection used development data only. Independent test inputs were generated and evaluated together after selections were frozen. No outcome triggered extra budget, task replacement, or additional search.\n\n", "This hypothesis follows v1's exploratory observations that LIS performance declined with increasing k and that fixed_points transfer depended on the task chain. Old v1 results are not repeats of this study and are not pooled with the new seeds. Both chains retain only k=2 and k=16 from their original lists, trained from the same paired random initialization to 4M or 32M tokens; task composition within each chain and k is unchanged across budgets.\n\n", "## Primary results: compute within-repeat differences before summarizing\n\n", "D4 denotes k16−k2 at the 4M budget, D32 denotes k16−k2 at 32M, and the interaction is D32−D4. A positive interaction means the larger budget improves k16 relative to k2; it does not imply higher absolute accuracy in every condition or the elimination of negative transfer. All values below are percentage points. The ± term is the sample standard deviation of the three paired differences, not a confidence interval.\n\n", "| Target | Chain | Mean D4 | Mean D32 | Mean interaction ± SD | Range of three interactions |\n|---|---|---:|---:|---:|---:|\n"]
    # The primary table always contains v2 only; old seed outcomes appear separately.
    for r in primary:
        text.append(f"| {r['target']} | {r['chain'].removeprefix('chain_')} | {r['d4_k16_minus_k2_pp_mean']:+.2f} | {r['d32_k16_minus_k2_pp_mean']:+.2f} | {r['paired_budget_interaction_pp_mean']:+.2f} ± {r['paired_budget_interaction_pp_sample_sd']:.2f} | [{r['paired_budget_interaction_pp_min']:+.2f}, {r['paired_budget_interaction_pp_max']:+.2f}] |\n")
    text.append("\nIndividual repeats are shown below. All three differences are computed within the same chain, initialization, data, and sampling repeat unit; they cannot be replaced by the two conditions' separate standard deviations. The shared random-initialization baseline cancels exactly in the interaction.\n\n| Target | Chain | Initialization seed | D4 | D32 | D32−D4 |\n|---|---|---:|---:|---:|---:|\n")
    for r in raw_interactions:
        if r["checkpoint"] == PRIMARY and r["length"] == "all":
            text.append(f"| {r['target']} | {r['chain'].removeprefix('chain_')} | {r['initialization_seed']} | {r['d4_k16_minus_k2_pp']:+.2f} | {r['d32_k16_minus_k2_pp']:+.2f} | {r['paired_budget_interaction_pp']:+.2f} |\n")
    text.append("\n![LIS budget interaction](figures/lis_length_budget_interaction.png)\n\n![fixed_points budget interaction](figures/fixed_points_budget_interaction.png)\n\n")
    for t in data["targets"]:
        findings = [r for r in primary if r["target"] == t]
        text.append(f"**{t}**: ")
        text.append("; ".join(f"{r['chain'].removeprefix('chain_')} has a mean interaction of {r['paired_budget_interaction_pp_mean']:+.2f} percentage points, positive in {r['positive_repeat_interactions']}/3 repeats" for r in findings) + ". These are observations from the current limited repeats, not a budget law for all task compositions.\n\n")
    if archived_result:
        text.append("At 4M, the new LIS seeds reproduce v1's direction: k16 is below k2 in all three repeats of both chains. With the larger budget, alpha k16 overtakes k2 in all three repeats. Beta also has positive interactions in all three repeats, but its 32M k16−k2 difference itself is positive in two repeats and negative in one, with a mean near zero. Beta therefore supports a reduction of the original k16 disadvantage rather than a stable k16 advantage. The two chains share three repeat units; their six within-chain interactions cannot be treated as six independent repeats.\n\nfixed_points shows opposite directions across chains: alpha k16 improves markedly at 32M, whereas beta k16 worsens. Beta's interaction ranges from −22.17 to −0.29 percentage points, indicating unstable magnitude. Budget effects depend on task composition and target; these results do not establish that larger budgets generally eliminate negative transfer.\n\n")
    text.append("## V1 exploratory evidence: reported separately, not pooled with this study\n\nThe old seeds were 1103, 2207, and 3301, with only a 4M budget. The table directly preserves v1's paired k16−k2 differences and the source CSV hash. Different test inputs and repeat units are not pooled with v2, and cross-round differences cannot be attributed solely to training budget.\n\n| Target | Chain | V1 mean D4 ± SD (pp) | V1 seed differences (1103 / 2207 / 3301) |\n|---|---|---:|---|\n")
    for r in old_reference:
        text.append(f"| {r['target']} | {r['chain'].removeprefix('chain_')} | {float(r['paired_change_pp_mean']):+.2f} ± {float(r['paired_change_pp_sample_sd']):.2f} | {float(r['seed1103_change_pp']):+.2f} / {float(r['seed2207_change_pp']):+.2f} / {float(r['seed3301_change_pp']):+.2f} |\n")
    text.append("\nThe [v1 exploratory reference CSV](v1_exploratory_reference.csv) explains only the hypothesis's origin; all interaction estimates in this study come from v2.\n\n")
    text.append("## Condition accuracy and gains over shared A\n\n| Target | Chain | k | Budget | Mean test accuracy | Mean paired gain over A ± SD (pp) |\n|---|---|---:|---:|---:|---:|\n")
    for r in summaries:
        if r["checkpoint"] == PRIMARY and r["length"] == "all":
            text.append(f"| {r['target']} | {r['chain'].removeprefix('chain_')} | {r['k']} | {r['budget_tokens']//1_000_000}M | {100*r['accuracy_mean']:.2f}% | {r['paired_delta_pp_mean']:+.2f} ± {r['paired_delta_pp_sample_sd']:.2f} |\n")
    if archived_result:
        text.append("\nAll four LIS conditions at 32M exceed their shared A in every repeat. However, alpha k2 mean accuracy falls from 57.03% to 54.10%, showing that a positive interaction can include poorer small-k performance. Alpha k16 improves from 50.85% to 65.85%, which is the main change behind its reversal. Beta k2 and k16 both improve with budget, from 54.82%→61.46% and 49.02%→62.21%, respectively.\n\nFor fixed_points, only alpha k16 at 32M has positive mean transfer: 88.44% accuracy and +23.63 percentage points over A, positive in all three repeats. The other seven conditions have negative mean gains. Beta k16 at 32M reaches only 43.39%, or −21.42 percentage points relative to A. The large alpha interaction must not obscure these retained negative results.\n\n")
    text.append("\n![LIS accuracy and baseline gains](figures/lis_length_budget_comparison.png)\n\n![fixed_points accuracy and baseline gains](figures/fixed_points_budget_comparison.png)\n\n## What changes with the larger budget\n\n")
    text.append(f"Both budgets start from paired random initialization with fresh AdamW optimizers; 32M does not continue the decayed 4M endpoint. The relative-budget progress rule is identical, but warmup lasts {lr['warmup_4M']:,.2f} and {lr['warmup_32M']:,.2f} tokens, respectively. At 4M consumed tokens, the 4M schedule has learning rate {lr['4M_at_4M']:.6g}, while the 32M schedule is still at {lr['32M_at_4M']:.6g}. This comparison therefore combines the larger budget with its prespecified learning-rate trajectory; it does not separately identify the effects of update count and learning-rate trajectory. Actual per-update learning rates are in each run's updates.csv, and the analytic rule is in [learning_rate_schedule.csv](learning_rate_schedule.csv).\n\n![Learning-rate trajectories for both budgets](figures/learning_rate_schedule.png)\n\n")
    text.append("Both budgets use exactly the same source-training pool: 4096 objects per repeat and length, or 8192 total, not eight times as much independent data. A task may not have seen the entire pool at 4M, especially at k16. The 32M run can both increase that task's coverage of the fixed pool and repeat previously seen task-input pairs; not every additional example is a repeated task-input pair. Tasks are sampled uniformly by examples, while different prompt/answer lengths imply unequal token shares. The last batch uses only the whole-example prefix that fits the budget, retaining the unused tail. The 32M budget is approximately eight times total tokens; k16 per-task exposure is not claimed to match k2 at 4M exactly. Each task can expose at most 8192 distinct inputs; counting the same input again across tasks does not create more independent objects. Actual differences in per-task coverage, first exposures, and repeated exposures are preserved in task_exposures.csv and source_budget_comparisons.csv.\n\n")
    text.append("## Source learning and repeated exposure\n\nTraining accuracy uses the fixed first 256 source-training objects per length as a diagnostic probe; development uses all 256 objects per length. Majority answers are determined from the full 4096 source-training labels per length. Curves are saved at initialization, the first update crossing half the budget, and the endpoint. The focus below is LDS, I3, and D3; all other tasks remain in the CSVs.\n\n| Chain | k | Source task | Budget | Dev accuracy | Training-probe accuracy | Dev majority baseline | Mean sample exposures | Mean distinct inputs |\n|---|---:|---|---:|---:|---:|---:|---:|---:|\n")
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
    text.append(f"\nOf the {len(endpoint)} source-task × model endpoint records, {sum(r['above_majority_pp'] <= 0 for r in endpoint)} have development accuracy at or below their own majority baseline. These records reuse tasks and models and are not {len(endpoint)} independent experiments. Concurrent training/development gaps, source-learning improvements, and transfer interactions are associations; they do not directly establish interference, use of mathematical identities, or the absence of either.\n\n")
    below = {b: sum(r["above_majority_pp"] <= 0 for r in endpoint if int(r["budget_tokens"]) == b) for b in data["budgets"]}
    text.append(f"By budget, {below[4_000_000]}/108 source-task endpoints at 4M do not exceed their majority baseline, and {below[32_000_000]}/108 still do not at 32M. Budget changes for the focus tasks are paired within repeat below; different tasks are not treated as repeats.\n\n| Chain | k | Source task | Mean 32M−4M dev-accuracy change ± SD (pp) | Positive repeat count |\n|---|---:|---|---:|---:|\n")
    for r in summarize_source_contrasts(source_contrasts):
        if r["source_task"] in FOCUS:
            text.append(f"| {r['chain'].removeprefix('chain_')} | {r['k']} | {r['source_task']} | {r['source_dev_accuracy_change_pp_mean']:+.2f} ± {r['source_dev_accuracy_change_pp_sample_sd']:.2f} | {r['positive_dev_changes']}/3 |\n")
    if archived_result:
        text.append("\nLDS and length-three subsequence-count development performance both improve with budget. At k16, distinct input coverage per task rises from about 5100 to about 8190, near the fixed-pool limit of 8192; sample exposure per task is approximately 8064→64513. Nevertheless, k16 LDS remains below alpha k2's 99.09%, and the three-element counts remain far from fully learned. At 32M, beta k2 D3 reaches 55.66% on the training probe versus 40.30% on development. This gap of about 15.36 percentage points shows that more repeated exposure does not automatically resolve generalization. Alpha k2 LDS improves from 84.57% to 99.09% while its mean LIS transfer falls, directly limiting the simple explanation that higher related-source accuracy necessarily yields better transfer.\n\nAlpha k16 exceedances and deficiencies have an explicit joint relationship to fixed_points. Their development accuracies improve from 43.23%→99.35% and 38.93%→98.96%, respectively, alongside markedly improved fixed_points transfer. This correspondence merits future investigation. However, even_cycle_count remains near its majority baseline, so not all related mathematical quantities are fully learned. Beta k16 anti_fixed_points also improves from 44.27% to 63.35% while fixed_points transfer worsens. The existence of a valid formula and improvements on related sources are insufficient to prove that the model uses the formula. Current records cannot distinguish formula use, broader representation changes, and optimization mechanisms.\n\n")
    for task in FOCUS:
        text.append(f"![{task} source-learning curves](figures/source_{task}.png)\n\n")
    text.append("## Configuration, isolation, and adaptation checks\n\nThe model is unchanged from v1: the platform's current 203-token vocabulary, a 4-layer Transformer of width 256 with 8 heads, FFN ratio 4, dropout 0.1, context 128, and tied embedding/output head. Encodings are one_line_v1 and permworld_sequence_v1. Loss averages answer-token-plus-EOS cross-entropy within each example before averaging examples; accuracy requires free generation of the complete answer plus EOS. Multi-token I3/D3 integer answers are neither truncated nor reduced modulo a bound.\n\nEvery adaptation resets all four original held-out task-token rows, including the tied output-head rows, loads all remaining pretrained parameters exactly, and creates a fresh optimizer. It uses 800 updates, batch 64, initial LR 3e-4, 100-update warmup, then cosine decay to 3e-5. Development evaluation occurs every 100 updates, giving eight eligible checkpoints. Selection uses highest development accuracy, then lowest answer CE, then earliest update; initialization is ineligible. Every condition has exactly one LR candidate.\n\nComplete inverse/reverse/complement orbits isolate inputs across roles, repeats, historical pilots, the platform, and v1. Capacity is checked before generation, and rules are never silently relaxed. Each target has 128×2=256 distinct adaptation objects, 256×2 development objects, and 512×2 final-test objects. Both lengths 10 and 12 appear in training, so per-length results are not length extrapolation. Final tests are generated only after development selections are frozen.\n\n")
    text.append("| Initialization | Data | Test | Source sampling | Adapt sampling | Source dropout | Adapt dropout |\n|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in data["config"]["repeats"]:
        text.append("| " + " | ".join(str(r[k]) for k in ("initialization_seed", "data_seed", "test_seed", "pretrain_sampling_seed", "adaptation_sampling_seed", "pretrain_dropout_seed", "adaptation_dropout_seed")) + " |\n")
    text.append("\nTarget majority answers are estimated only from the 128 training labels per length. Their mean accuracies on the new test are shown below.\n\n| Target | n=10 majority accuracy | n=12 majority accuracy |\n|---|---:|---:|\n")
    for t in data["targets"]:
        vals = [next(r["majority_accuracy_mean"] for r in summaries if r["target"] == t and r["chain"] == "baseline" and r["checkpoint"] == PRIMARY and r["length"] == n) for n in ("10", "12")]
        text.append(f"| {t} | {100*vals[0]:.2f}% | {100*vals[1]:.2f}% |\n")
    adapts = [r for r in data["records"] if r["stage"] == "adaptation"]
    diagnostic = [r for r in diagnostics if r["length"] == "all"]
    text.append(f"\nDevelopment selects an update earlier than 800 in {sum(r['best_update'] < 800 for r in adapts)} of the 54 adaptations. On the new test, the preselected checkpoint exceeds the endpoint in {sum(r['selected_minus_endpoint_test_pp'] > 0 for r in diagnostic)} runs and falls below it in {sum(r['selected_minus_endpoint_test_pp'] < 0 for r in diagnostic)}; the rest tie. This is a prespecified auxiliary comparison, not a new model-selection step. Complete endpoint interactions and per-length results remain in the raw and interaction CSVs.\n\n")
    text.append("| Target | Chain | Mean development-selected interaction | Mean fixed-endpoint interaction (pp) |\n|---|---|---:|---:|\n")
    for r in primary:
        end = next(x for x in int_summary if x["target"] == r["target"] and x["chain"] == r["chain"] and x["checkpoint"] == "fixed_endpoint" and x["length"] == "all")
        text.append(f"| {r['target']} | {r['chain'].removeprefix('chain_')} | {r['paired_budget_interaction_pp_mean']:+.2f} | {end['paired_budget_interaction_pp_mean']:+.2f} |\n")
    if archived_result:
        text.append("\nAll four target × chain mean interactions retain their directions at the fixed endpoint, so the main directions are not confined to development-selected checkpoints. Magnitudes and seed patterns still change with the checkpoint. Endpoint results are a prespecified auxiliary diagnostic and do not replace the primary results.\n\n")
    text.append("## Actual compute cost\n\n| Stage | Runs | Updates | Nonpadding sequence tokens | Supervised tokens | Cumulative run minutes |\n|---|---:|---:|---:|---:|---:|\n")
    for cat, r in totals.items():
        text.append(f"| {cat} | {r['runs']} | {r['updates']:,.0f} | {r['input_tokens']:,.0f} | {r['supervised_tokens']:,.0f} | {r['seconds']/60:.2f} |\n")
    text.append(f"\nTraining-stage wall time was {result['training_stage_wall_seconds']/60:.2f} minutes, and independent testing took {result['test_seconds']/60:.2f} minutes. Unused token tails range from {result['pretraining_token_tail_min']}–{result['pretraining_token_tail_max']} per pretrained model. Run timers include the respective training, development evaluation, and checkpoint/CSV writes but exclude data encoding. Stage and run timers overlap and must not be added. Per-run padded tokens, supervised tokens, sample exposures, memory, and time remain in compute.csv.\n\n")
    smoke = read_json(out / "SMOKE.json")
    source_update_seconds = sum(float(r["training_updates_seconds"]) for r in compute if r["stage"] == "pretraining")
    text.append(f"The fixed small-batch smoke timing was {smoke['timing_tokens_per_second']:,.0f} tokens/s, projecting {smoke['estimated_all_pretraining_training_only_seconds']/60:.2f} minutes for all source-model training updates alone. Actual source-model updates took {source_update_seconds/60:.2f} cumulative minutes, and source runs including evaluations and writes took {sum(r['seconds'] for r in pre)/60:.2f} minutes. The {result['training_stage_wall_seconds']/60:.2f}-minute stage time also includes all target adaptations and stage overhead. Smoke timing is a resource-feasibility estimate; actual recorded costs, not estimates, define the completed experiment's cost.\n\n")
    text.append("## Evidence limits and stopping point\n\nThe three new paired repeats are the statistical units. Two chains, two targets, multiple k/budget conditions, shared source models, and shared random baselines do not create extra independent repeats. We show individual paired differences, means, sample SDs, and ranges, without inferring significance from separate marginal condition SDs or selecting a preferred chain from outcomes. The small repeat count and finite test inputs leave substantial uncertainty.\n\nBudget changes within a source set do not replace tasks, but changing k still changes composition, mathematical relationships, and exposure per task. The budgets also imply different absolute-token learning-rate trajectories; repeated training on a fixed pool and adding independent data are not separately identified. Alpha k16's joint fixed_points derivations and beta's transformed-input relations remain part of the design. Valid mathematical relationships are not data leakage, and these records do not show that the model uses the formulas.\n\nNegative gains, zero gains, seed disagreement, and incomplete learning are all retained. This study addresses only the finite budget interaction for two fixed chains, two budgets, one model, two lengths, and 256 target examples; it makes no scaling-law claim. The frozen scope is complete, with no added budget or task replacement.\n\n- [Per-repeat raw results](raw_test_results.csv), [per-repeat budget interactions](paired_budget_interactions.csv), [interaction summary](budget_interaction_summary.csv), [gains over shared A](paired_deltas.csv)\n- [All source learning](source_learning.csv), [paired source-budget comparisons](source_budget_comparisons.csv), [per-task/per-length exposure](task_exposures.csv), [per-task exposure totals](task_exposure_totals.csv)\n- [Target training/development curves](adaptation_curves.csv), [majority baselines by length](majority_per_length.csv), [actual compute cost](compute.csv)\n- [Frozen configuration](FROZEN_PROTOCOL.json), [task lists and mathematical relationships](design/README.md), [reproduction commands](README.md)\n")
    if (out / "RUN_AUDIT.json").exists():
        audit = read_json(out / "RUN_AUDIT.json")
        assert audit["status"] == "pass" and audit["raw_result_rows"] == 324
        text.append(f"\nThe [independent run audit](RUN_AUDIT.json) passed, covering 24 source models, 54 target fits, 135 formal checkpoint files, and 12 budget pairs starting from the same initialization. It replayed source and target sampling and checked parameter loading, all four held-out token rows, optimizer resets, development selection, and all 324 test rows. It also recomputed {audit['development_checkpoints_recomputed']} development checkpoints without any new test neural forwards. The audit took {audit['seconds']:.2f} seconds; see [budget pairing](audit_budget_pairing.csv), [per-run accounting](audit_run_accounting.csv), and [development-forward checks](audit_development_forwards.csv).\n")
    text.append("\nThe initial engineering smoke encountered a mismatch between its reduced input manifest and archived hashes and stopped before training. Copying the exact original manifest/archive bytes corrected the setup, and the final smoke passed. The initial setup-failure evidence remains in smoke/failed_manifest_setup and must not be conflated with formal training failure. [SMOKE.json](SMOKE.json) records the final parameterized-task, multi-token-answer, loss, save/load, adaptation, and timing checks. The [analysis self-check](SUMMARY_SELF_CHECK.json) verifies the four-cell difference signs, shared-baseline cancellation, and rejection of missing/duplicate rows.\n")
    if (out / "PRESERVATION_AFTER.json").exists():
        text.append("\nThe [historical preservation check](PRESERVATION_AFTER.json) records preservation of previous results and v1.\n")
    (out / "REPORT.md").write_text("".join(text), encoding="utf-8")
    return result


def reproduction_readme(out):
    text = """# PermWorld task-count v2: budget interaction

Completed scope: 24 pretrained models, 48 pretrained adaptations and six shared
scratch baselines. LIS length is primary; fixed points is auxiliary. Each unchanged
v1 chain contributes k2 and k16 at 4M and 32M sequence-token budgets, with three new
paired repeats. The [report](REPORT.md), its generator, and this README were revised
to English at the user's request; experimental results are unchanged.
V1 exploration is not pooled with these data.

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

The original RELEASE_MANIFEST.json remains the evidence of the pretranslation
archive, and frozen scientific records are unchanged. The user-requested English
revision is recorded in the [translation ledger documentation](../../maintenance/english_translation/task_count_v2/README.md).
Release verification resolves that explicit ledger rather than rewriting old
hashes. Do not rerun the release writer to erase the original archive's provenance.
Full verification includes local weights; a Git checkout without weights
explicitly uses source-only verification:

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
