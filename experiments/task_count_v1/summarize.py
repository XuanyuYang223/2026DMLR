"""Summarize completed task-count results without importing or running a model.

All checkpoint selections must already be frozen and all independent tests must
be complete. The script preserves the three paired repeat units and never treats
shared pretrained models, reused scratch baselines or task-set chains as extra
independent observations. It performs no statistical or training selection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUT = Path(__file__).resolve().parent
CHECKPOINTS = ("development_selected", "fixed_endpoint")
PRIMARY = CHECKPOINTS[0]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    assert rows, f"No rows for {path}"
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_key(spec):
    return spec["task_id"] + "".join(f"_{key}{value}" for key, value in sorted(spec["params"].items()))


def moments(values):
    values = [float(v) for v in values]
    assert len(values) == 3, f"Expected three paired repeat units, received {len(values)}"
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "min": min(values), "max": max(values)}


def valid_metric(row):
    count, correct = int(row["count"]), int(row["correct"])
    assert count > 0 and 0 <= correct <= count
    assert abs(float(row["accuracy"]) - correct / count) < 1e-12
    assert abs(float(row["majority_accuracy"]) - int(row["majority_correct"]) / count) < 1e-12


def collect(out):
    protocol = read_json(out / "FROZEN_PROTOCOL.json")
    selected = read_json(out / "FROZEN_SELECTIONS.json")
    training = read_json(out / "TRAINING_COMPLETE.json")
    test = read_json(out / "TEST_COMPLETE.json")
    assert protocol["status"] == selected["status"] == "frozen"
    assert selected["training_complete"] is True and test["status"] == "passed"
    assert test["no_test_based_selection"] is True
    assert test["raw_results_sha256"] == sha(out / "raw_test_results.csv")
    assert test["frozen_selection_sha256"] == sha(out / "FROZEN_SELECTIONS.json")
    protocol_sha = sha(out / "FROZEN_PROTOCOL.json")
    assert selected["frozen_protocol_sha256"] == protocol_sha
    assert training["frozen_protocol_sha256"] == protocol_sha
    cfg = protocol["config"]
    targets = [task_key(s) for s in protocol["target_specs"]]
    chains, ks = list(protocol["chains"]), cfg["k_values"]
    seeds = [r["initialization_seed"] for r in cfg["repeats"]]
    assert len(targets) == 4 and len(chains) == 2 and ks == [2, 4, 8, 16] and len(seeds) == 3
    assert cfg["scope"] == {"pretraining_models": 24, "adaptations_from_pretraining": 96, "shared_random_initialization_baselines": 12, "total_target_training_runs": 108}
    expected_pre = {f"{chain}_k{k}_s{seed}" for chain in chains for k in ks for seed in seeds}
    expected_adapt = {f"{chain}_k{k}_{target}_s{seed}" for chain, k in [("baseline", 0)] + [(c, k) for c in chains for k in ks] for target in targets for seed in seeds}
    assert len(expected_pre) == 24 and len(expected_adapt) == 108
    records, curves, source_curves, exposures = [], [], [], []
    for stage, expected in (("pretraining", expected_pre), ("adaptation", expected_adapt)):
        paths = sorted((out / stage).glob("*/COMPLETE.json"))
        assert {p.parent.name for p in paths} == expected, f"Incomplete or extra {stage} runs"
        for path in paths:
            rec = read_json(path)
            assert rec["run_id"] == path.parent.name and rec["frozen_protocol_sha256"] == protocol_sha
            # Independent audit handles weight bytes; these checks bind every table read here.
            names = ["source_curves.csv", "task_exposures.csv"] if stage == "pretraining" else ["curves.csv"]
            for name in names:
                assert sha(path.parent / name) == rec["artifact_sha256"][name]
            rec["stage"] = stage
            records.append(rec)
            if stage == "pretraining":
                assert rec["input_tokens"] + rec["budget_tail_tokens"] == cfg["pretraining_budget_nonpadding_sequence_tokens"]
                assert rec["uniform_task_count_max_minus_min"] <= 1
                source_curves.extend(read_csv(path.parent / "source_curves.csv"))
                exposures.extend(read_csv(path.parent / "task_exposures.csv"))
            else:
                assert rec["updates"] == cfg["adaptation_updates"] == 800
                curves.extend(read_csv(path.parent / "curves.csv"))
    rows = read_csv(out / "raw_test_results.csv")
    assert len(rows) == 108 * 2 * 3 == test["result_rows"]
    lookup, identities = {}, {r["run_id"]: r for r in records if r["stage"] == "adaptation"}
    selections = {r["run_id"]: r for r in selected["selections"]}
    assert set(selections) == expected_adapt
    for row in rows:
        valid_metric(row)
        assert row["run_id"] in expected_adapt and row["checkpoint"] in CHECKPOINTS and row["split"] == "test"
        rec, selection = identities[row["run_id"]], selections[row["run_id"]]
        for name in ["target", "chain", "condition", "initialization_seed", "data_seed", "pretrain_sampling_seed", "adaptation_sampling_seed", "k"]:
            assert str(row[name]) == str(rec[name])
        expected_update = rec["best_update"] if row["checkpoint"] == PRIMARY else 800
        assert int(row["selected_update"]) == expected_update
        assert selection["selected_update"] == rec["best_update"]
        key = (row["run_id"], row["checkpoint"], row["length"])
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
    # Recompute the frozen development selection from its original development table.
    by_run = defaultdict(list)
    for row in curves:
        valid_metric(row)
        by_run[row["run_id"]].append(row)
    for rid, rec in identities.items():
        dev = [r for r in by_run[rid] if r["split"] == "target_dev" and r["length"] == "all" and int(r["update"]) > 0]
        assert len(dev) == 8
        best = min(dev, key=lambda r: (-float(r["accuracy"]), float(r["answer_ce"]), int(r["update"])))
        assert int(best["update"]) == rec["best_update"]
    # Each source must have exactly one record per stage, split and length.
    source_keys = set()
    for row in source_curves:
        valid_metric(row)
        key = (row["run_id"], row["source_task"], row["stage"], row["split"], row["length"])
        assert key not in source_keys
        source_keys.add(key)
        assert row["stage"] in ("start", "midpoint", "endpoint")
    assert len(source_curves) == 3 * 2 * sum(ks) * 3 * 2 * 3
    return {"protocol": protocol, "config": cfg, "records": records, "rows": rows, "curves": curves, "source_curves": source_curves, "exposures": exposures, "targets": targets, "chains": chains, "ks": ks, "seeds": seeds, "lengths": lengths, "test": test, "training": training}


def summarize_targets(data):
    rows, cfg = data["rows"], data["config"]
    lookup = {(r["target"], r["chain"], int(r["k"]), int(r["initialization_seed"]), r["checkpoint"], r["length"]): r for r in rows}
    assert len(lookup) == len(rows)
    paired, summary, diagnostics = [], [], []
    for row in rows:
        if row["condition"] == "A":
            assert row["chain"] == "baseline" and int(row["k"]) == 0
            continue
        base = lookup[row["target"], "baseline", 0, int(row["initialization_seed"]), row["checkpoint"], row["length"]]
        for field in ("count", "majority_correct", "majority_answer", "data_seed", "adaptation_sampling_seed"):
            assert row[field] == base[field], (row["run_id"], field)
        paired.append({"target": row["target"], "chain": row["chain"], "k": int(row["k"]), "initialization_seed": int(row["initialization_seed"]), "data_seed": int(row["data_seed"]), "pretrain_sampling_seed": int(row["pretrain_sampling_seed"]), "adaptation_sampling_seed": int(row["adaptation_sampling_seed"]), "checkpoint": row["checkpoint"], "length": row["length"], "pretrained_run_id": row["run_id"], "shared_baseline_run_id": base["run_id"], "test_count": int(row["count"]), "pretrained_accuracy": float(row["accuracy"]), "baseline_accuracy": float(base["accuracy"]), "paired_delta_pp": 100 * (float(row["accuracy"]) - float(base["accuracy"])), "pretrained_selected_update": int(row["selected_update"]), "baseline_selected_update": int(base["selected_update"]), "majority_accuracy": float(row["majority_accuracy"])})
    delta_lookup = {(r["target"], r["chain"], r["k"], r["initialization_seed"], r["checkpoint"], r["length"]): r for r in paired}
    conditions = [("baseline", 0)] + [(chain, k) for chain in data["chains"] for k in data["ks"]]
    for target in data["targets"]:
        for chain, k in conditions:
            for checkpoint in CHECKPOINTS:
                for length in data["lengths"]:
                    group = [lookup[target, chain, k, seed, checkpoint, length] for seed in data["seeds"]]
                    acc = moments(r["accuracy"] for r in group)
                    majority = moments(r["majority_accuracy"] for r in group)
                    delta = moments([0] * 3 if chain == "baseline" else [delta_lookup[target, chain, k, seed, checkpoint, length]["paired_delta_pp"] for seed in data["seeds"]])
                    summary.append({"target": target, "chain": chain, "k": k, "checkpoint": checkpoint, "length": length, "independent_paired_repeats": 3, "accuracy_mean": acc["mean"], "accuracy_sample_sd": acc["sample_sd"], "accuracy_min": acc["min"], "accuracy_max": acc["max"], "majority_accuracy_mean": majority["mean"], "above_majority_pp_mean": 100 * (acc["mean"] - majority["mean"]), "paired_delta_pp_mean": delta["mean"], "paired_delta_pp_sample_sd": delta["sample_sd"], "paired_delta_pp_min": delta["min"], "paired_delta_pp_max": delta["max"], "error_bar_interpretation": "Descriptive sample SD over three paired repeat units; not a confidence interval"})
            for seed in data["seeds"]:
                for length in data["lengths"]:
                    best = lookup[target, chain, k, seed, PRIMARY, length]
                    end = lookup[target, chain, k, seed, CHECKPOINTS[1], length]
                    diagnostics.append({"run_id": best["run_id"], "target": target, "chain": chain, "k": k, "initialization_seed": seed, "length": length, "selected_update": int(best["selected_update"]), "endpoint_update": 800, "development_selected_test_accuracy": float(best["accuracy"]), "endpoint_test_accuracy": float(end["accuracy"]), "selected_minus_endpoint_test_pp": 100 * (float(best["accuracy"]) - float(end["accuracy"])), "interpretation": "Both predeclared checkpoints are reported; test differences do not change selection"})
    assert len(paired) == 96 * 2 * 3 and len(summary) == 4 * 9 * 2 * 3
    return paired, summary, diagnostics


def source_and_cost(data):
    compute, source_rows, majority_rows = [], [], []
    source_lookup = {(r["run_id"], r["source_task"], r["stage"], r["split"], r["length"]): r for r in data["source_curves"]}
    for row in data["source_curves"]:
        train = source_lookup[row["run_id"], row["source_task"], row["stage"], "source_train", row["length"]]
        dev = source_lookup[row["run_id"], row["source_task"], row["stage"], "source_dev", row["length"]]
        source_rows.append({**row, "above_majority_pp": 100 * (float(row["accuracy"]) - float(row["majority_accuracy"])), "train_probe_minus_dev_pp": 100 * (float(train["accuracy"]) - float(dev["accuracy"]))})
        if row["stage"] == "start" and row["length"] != "all":
            majority_rows.append({"kind": "source", "run_id": row["run_id"], "task": row["source_task"], "chain": row["chain"], "k": row["k"], "initialization_seed": row["initialization_seed"], "data_seed": row["data_seed"], "length": row["length"], "evaluation_split": row["split"], "majority_answer": row["majority_answer"], "majority_accuracy": row["majority_accuracy"], "count": row["count"], "majority_label_source": "complete source training pool for this length"})
    for row in data["curves"] + data["rows"]:
        if row["condition"] != "A" or row["length"] == "all":
            continue
        if row["split"] == "test":
            if row["checkpoint"] != PRIMARY:
                continue
        elif int(row["update"]) != 0:
            continue
        majority_rows.append({"kind": "target", "run_id": row["run_id"], "task": row["target"], "chain": "shared_across_all_conditions", "k": "shared", "initialization_seed": row["initialization_seed"], "data_seed": row["data_seed"], "length": row["length"], "evaluation_split": row["split"], "majority_answer": row["majority_answer"], "majority_accuracy": row["majority_accuracy"], "count": row["count"], "majority_label_source": "128 target training labels for this length"})
    keys = ["run_id", "initialization_seed", "data_seed", "pretrain_sampling_seed", "adaptation_sampling_seed", "chain", "k", "condition", "target", "updates", "sample_exposures", "input_tokens", "padded_input_tokens", "supervised_tokens", "answer_tokens", "budget_tail_tokens", "uniform_task_count_max_minus_min", "seconds", "peak_memory_bytes", "best_update", "initial_state_sha256", "length_index_schedule_sha256", "timer_scope"]
    for rec in data["records"]:
        category = "pretraining" if rec["stage"] == "pretraining" else "baseline_adaptation" if rec["condition"] == "A" else "pretrained_adaptation"
        phases = rec.get("phase_seconds", {})
        compute.append({"stage": rec["stage"], "category": category, **{key: rec.get(key, "") for key in keys},
                        "training_updates_seconds": phases.get("training_updates", ""),
                        "source_evaluation_seconds": phases.get("source_evaluation", ""),
                        "target_evaluation_seconds": phases.get("target_evaluation", "")})
    assert len(compute) == 132
    return source_rows, majority_rows, compute


def k_contrasts(data, paired):
    """Preserve the same repeat for adjacent and total changes along each chain."""
    lookup = {(r["target"], r["chain"], r["k"], r["initialization_seed"], r["checkpoint"], r["length"]): r for r in paired}
    contrasts = [(2, 4), (4, 8), (8, 16), (2, 16)]
    raw, summary = [], []
    for target in data["targets"]:
        for chain in data["chains"]:
            for checkpoint in CHECKPOINTS:
                for length in data["lengths"]:
                    for start, end in contrasts:
                        group = []
                        for seed in data["seeds"]:
                            a = lookup[target, chain, start, seed, checkpoint, length]
                            b = lookup[target, chain, end, seed, checkpoint, length]
                            assert a["shared_baseline_run_id"] == b["shared_baseline_run_id"]
                            value = 100 * (b["pretrained_accuracy"] - a["pretrained_accuracy"])
                            assert abs(value - (b["paired_delta_pp"] - a["paired_delta_pp"])) < 1e-10
                            group.append(value)
                            raw.append({"target": target, "chain": chain, "checkpoint": checkpoint, "length": length,
                                "k_from": start, "k_to": end, "initialization_seed": seed, "data_seed": a["data_seed"],
                                "pretrain_sampling_seed": a["pretrain_sampling_seed"], "adaptation_sampling_seed": a["adaptation_sampling_seed"],
                                "from_run_id": a["pretrained_run_id"], "to_run_id": b["pretrained_run_id"],
                                "shared_baseline_run_id": a["shared_baseline_run_id"],
                                "accuracy_from": a["pretrained_accuracy"], "accuracy_to": b["pretrained_accuracy"],
                                "paired_accuracy_change_pp": value, "baseline_cancels_exactly": True})
                        stats = moments(group)
                        summary.append({"target": target, "chain": chain, "checkpoint": checkpoint, "length": length,
                            "k_from": start, "k_to": end, "independent_paired_repeats": 3,
                            "paired_change_pp_mean": stats["mean"], "paired_change_pp_sample_sd": stats["sample_sd"],
                            "paired_change_pp_min": stats["min"], "paired_change_pp_max": stats["max"],
                            **{f"seed{seed}_change_pp": value for seed, value in zip(data["seeds"], group)},
                            "interpretation": "Descriptive within-repeat change along a fixed chain; no extra independent repeats or significance test"})
    assert len(raw) == 576 and len(summary) == 192
    return raw, summary


def source_summary(source_rows):
    """Average only matched task/condition rows across the three repeat units."""
    groups = defaultdict(list)
    for row in source_rows:
        key = tuple(row[field] for field in ("chain", "k", "source_task", "stage", "split", "length"))
        groups[key].append(row)
    summary = []
    for key, group in sorted(groups.items()):
        assert len({r["initialization_seed"] for r in group}) == 3
        acc = moments(r["accuracy"] for r in group)
        gain = moments(r["above_majority_pp"] for r in group)
        summary.append({**dict(zip(("chain", "k", "source_task", "stage", "split", "length"), key)),
            "independent_paired_repeats": 3, "accuracy_mean": acc["mean"], "accuracy_sample_sd": acc["sample_sd"],
            "accuracy_min": acc["min"], "accuracy_max": acc["max"],
            "majority_accuracy_mean": statistics.mean(float(r["majority_accuracy"]) for r in group),
            "above_majority_pp_mean": gain["mean"], "above_majority_pp_sample_sd": gain["sample_sd"],
            "answer_ce_mean": statistics.mean(float(r["answer_ce"]) for r in group),
            "train_probe_minus_dev_pp_mean": statistics.mean(float(r["train_probe_minus_dev_pp"]) for r in group),
            "repeat_units_at_or_below_majority": sum(float(r["above_majority_pp"]) <= 0 for r in group)})
    return summary


def figures(out, data, paired, summary):
    directory = out / "figures"
    directory.mkdir(exist_ok=True)
    colors = dict(zip(data["chains"], ["#1769aa", "#c45c17"]))
    markers = dict(zip(data["seeds"], ["o", "s", "^"]))
    raw = {(r["target"], r["chain"], int(r["k"]), int(r["initialization_seed"]), r["checkpoint"]): r for r in data["rows"] if r["length"] == "all"}
    delta = {(r["target"], r["chain"], r["k"], r["initialization_seed"], r["checkpoint"]): r for r in paired if r["length"] == "all"}
    totals = {(r["target"], r["chain"], r["k"], r["checkpoint"]): r for r in summary if r["length"] == "all"}
    for target in data["targets"]:
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), constrained_layout=True)
        for chain in data["chains"]:
            color = colors[chain]
            for seed in data["seeds"]:
                axes[0].plot(data["ks"], [100 * float(raw[target, chain, k, seed, PRIMARY]["accuracy"]) for k in data["ks"]], color=color, alpha=.45, linewidth=.9, marker=markers[seed], markersize=4, label=f"{chain.removeprefix('chain_')} / seed {seed}")
                axes[1].plot(data["ks"], [delta[target, chain, k, seed, PRIMARY]["paired_delta_pp"] for k in data["ks"]], color=color, alpha=.45, linewidth=.9, marker=markers[seed], markersize=4)
            values = [totals[target, chain, k, PRIMARY] for k in data["ks"]]
            axes[0].errorbar(data["ks"], [100 * r["accuracy_mean"] for r in values], yerr=[100 * r["accuracy_sample_sd"] for r in values], color=color, linewidth=2, capsize=3, label=f"{chain.removeprefix('chain_')} mean +/- SD")
            axes[1].errorbar(data["ks"], [r["paired_delta_pp_mean"] for r in values], yerr=[r["paired_delta_pp_sample_sd"] for r in values], color=color, linewidth=2, capsize=3)
        baseline = totals[target, "baseline", 0, PRIMARY]
        for seed in data["seeds"]:
            value = float(raw[target, "baseline", 0, seed, PRIMARY]["accuracy"])
            axes[0].axhline(100 * value, color="0.55", linestyle=":", linewidth=.7, alpha=.6)
        axes[0].axhline(100 * baseline["accuracy_mean"], color="0.15", linestyle="--", linewidth=1.2, label="shared scratch mean")
        axes[0].axhline(100 * baseline["majority_accuracy_mean"], color="#508846", linestyle=":", linewidth=1.1, label="train-derived majority mean")
        axes[1].axhline(0, color="0.15", linestyle="--", linewidth=1)
        axes[0].set_ylabel("Independent-test complete-answer accuracy (%)")
        axes[1].set_ylabel("Paired test accuracy minus shared scratch (pp)")
        axes[0].set_ylim(-2, 102)
        for axis in axes:
            axis.set_xscale("log", base=2)
            axis.set_xticks(data["ks"], labels=[str(k) for k in data["ks"]])
            axis.set_xlabel("Number of source task instances k")
            axis.grid(alpha=.2)
        axes[0].legend(fontsize=7, ncol=2, loc="best")
        fig.suptitle(f"{target}: development-selected checkpoints\nThree paired repeats; error bars are descriptive sample SD, not confidence intervals", fontsize=11)
        for suffix in ("png", "svg"):
            fig.savefig(directory / f"{target}_task_count.{suffix}", dpi=170)
        plt.close(fig)
    # A compact secondary display preserves endpoint results without choosing between protocols.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, target in zip(axes.flat, data["targets"]):
        for chain in data["chains"]:
            for checkpoint, style in ((PRIMARY, "-"), (CHECKPOINTS[1], "--")):
                values = [totals[target, chain, k, checkpoint] for k in data["ks"]]
                axis.plot(data["ks"], [r["paired_delta_pp_mean"] for r in values], style, marker="o", color=colors[chain], label=f"{chain.removeprefix('chain_')} / {'dev-selected' if checkpoint == PRIMARY else 'endpoint'}")
        axis.axhline(0, color="0.3", linewidth=.8)
        axis.set_xscale("log", base=2)
        axis.set_xticks(data["ks"], labels=data["ks"])
        axis.set_xlabel("Source task instances k")
        axis.set_ylabel("Mean paired gain over scratch (pp)")
        axis.set_title(target)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("Predeclared development selection and fixed endpoint: descriptive comparison")
    for suffix in ("png", "svg"):
        fig.savefig(directory / f"checkpoint_comparison.{suffix}", dpi=170)
    plt.close(fig)


def findings(data, paired, summary, source_rows):
    """Describe all four prespecified targets using the completed primary results."""
    means = {(r["target"], r["chain"], r["k"]): r for r in summary if r["checkpoint"] == PRIMARY and r["length"] == "all"}
    deltas = {(r["target"], r["chain"], r["k"], r["initialization_seed"]): r["paired_delta_pp"] for r in paired if r["checkpoint"] == PRIMARY and r["length"] == "all"}
    nondecreasing = sum(all(means[t, c, a]["accuracy_mean"] <= means[t, c, b]["accuracy_mean"] for a, b in zip(data["ks"], data["ks"][1:])) for t in data["targets"] for c in data["chains"])
    def accuracy(target, chain, k):
        return 100 * means[target, chain, k]["accuracy_mean"]
    def gain(target, chain, k):
        return means[target, chain, k]["paired_delta_pp_mean"]
    def changes(target, chain):
        return [deltas[target, chain, 16, s] - deltas[target, chain, 2, s] for s in data["seeds"]]
    text = ["## Findings from the completed test\n\n",
        f"The response to increasing source count depends on the target and chain: {nondecreasing}/8 target-chain mean curves are nondecreasing across every k. A higher count does not give a uniform improvement in this fixed-token comparison.\n\n"]
    lis = [changes("lis_length", c) for c in data["chains"]]
    text.append(f"**LIS retains transfer gains, but adding tasks weakens them.** Alpha falls from {accuracy('lis_length','chain_alpha',2):.2f}% at k=2 to {accuracy('lis_length','chain_alpha',16):.2f}% at k=16; beta falls from {accuracy('lis_length','chain_beta',2):.2f}% to {accuracy('lis_length','chain_beta',16):.2f}%. The paired k16-minus-k2 differences are negative in {sum(v<0 for v in lis[0])}/3 alpha repeats and {sum(v<0 for v in lis[1])}/3 beta repeats. Nevertheless, all eight condition means remain above the {accuracy('lis_length','baseline',0):.2f}% scratch reference in this run. This distinguishes reduced transfer benefit from negative transfer.\n\n")
    cy_nonpositive = sum(gain("cycle_count", c, k) <= 0 for c in data["chains"] for k in data["ks"])
    text.append(f"**Cycle count shows no mean transfer gain.** {cy_nonpositive}/8 pretrained condition means are at or below scratch ({accuracy('cycle_count','baseline',0):.2f}%). The mean k16-minus-k2 change is {statistics.mean(changes('cycle_count','chain_alpha')):+.2f} pp for alpha and {statistics.mean(changes('cycle_count','chain_beta')):+.2f} pp for beta, with mixed repeat signs. The small changes and paired spreads do not support a reliable count trend.\n\n")
    co_positive = sum(gain("components", c, k) > 0 for c in data["chains"] for k in data["ks"])
    text.append(f"**Components has modest gains above a strong scratch model.** Scratch already reaches {accuracy('components','baseline',0):.2f}%. {co_positive}/8 condition means are higher; k=16 gains are {gain('components','chain_alpha',16):+.2f} pp for alpha and {gain('components','chain_beta',16):+.2f} pp for beta. Alpha's k16-minus-k2 change is positive in {sum(v>0 for v in changes('components','chain_alpha'))}/3 repeats; beta's is positive in {sum(v>0 for v in changes('components','chain_beta'))}/3. Neither mean curve improves at every step.\n\n")
    fp8 = [deltas["fixed_points", "chain_alpha", 8, s] for s in data["seeds"]]
    text.append(f"**Fixed points is strongly nonmonotonic and chain dependent.** Alpha's k=8 accuracy is {accuracy('fixed_points','chain_alpha',8):.2f}%, with paired gains " + ", ".join(f"{v:+.2f}" for v in fp8) + f" pp in seed order {data['seeds']}. Its mean gain {gain('fixed_points','chain_alpha',8):+.2f} pp has sample SD {statistics.stdev(fp8):.2f} pp; the negative repeat remains part of the result. Alpha then falls to {accuracy('fixed_points','chain_alpha',16):.2f}% at k=16 ({gain('fixed_points','chain_alpha',16):+.2f} pp versus scratch). Beta has negative mean transfer at all four k, including {gain('fixed_points','chain_beta',16):+.2f} pp at k=16. No best-k configuration is selected from these tests. Alpha16's known joint target formulas therefore do not guarantee a transfer benefit under this training budget.\n\n")
    source = defaultdict(list)
    for row in source_rows:
        if row["stage"] == "endpoint" and row["split"] == "source_dev" and row["length"] == "all":
            source[row["chain"], int(row["k"]), row["source_task"]].append(row)
    def sm(chain, k, task, field="accuracy"):
        rows = source[chain, k, task]
        assert len(rows) == 3
        return 100 * statistics.mean(float(r[field]) for r in rows)
    text.append("**Source records support underlearning and exposure dilution as issues to examine, without proving a mechanism.** "
        f"For a source present throughout alpha, LDS development accuracy decreases from {sm('chain_alpha',2,'lds_length'):.2f}% at k=2 to {sm('chain_alpha',16,'lds_length'):.2f}% at k=16 (majority {sm('chain_alpha',16,'lds_length','majority_accuracy'):.2f}%). Beta's decreasing-subsequence-count source (subsequence length 3) falls from {sm('chain_beta',2,'decreasing_subsequence_count_k3'):.2f}% to {sm('chain_beta',16,'decreasing_subsequence_count_k3'):.2f}% (majority {sm('chain_beta',16,'decreasing_subsequence_count_k3','majority_accuracy'):.2f}%). These changes accompany the weaker LIS transfer at larger task counts, while per-task exposure also falls. "
        f"Nontrivial-cycle count in alpha16 is {sm('chain_alpha',16,'nontrivial_cycle_count'):.2f}% against the same {sm('chain_alpha',16,'nontrivial_cycle_count','majority_accuracy'):.2f}% majority baseline; alpha16 even/odd-cycle sources are {sm('chain_alpha',16,'even_cycle_count'):.2f}%/{sm('chain_alpha',16,'odd_cycle_count'):.2f}% against {sm('chain_alpha',16,'even_cycle_count','majority_accuracy'):.2f}%/{sm('chain_alpha',16,'odd_cycle_count','majority_accuracy'):.2f}%. Availability of a joint identity is therefore separate from demonstrated learning of its source statistics. "
        f"For fixed points, alpha's exceedances source falls from {sm('chain_alpha',8,'exceedances'):.2f}% at k=8 to {sm('chain_alpha',16,'exceedances'):.2f}% at k=16, concurrent with its target decline. In contrast, the k=16 largest/smallest-component sources remain at {sm('chain_alpha',16,'largest_component'):.2f}%/{sm('chain_alpha',16,'smallest_component'):.2f}% in alpha and {sm('chain_beta',16,'largest_component'):.2f}%/{sm('chain_beta',16,'smallest_component'):.2f}% in beta, above their {sm('chain_alpha',16,'largest_component','majority_accuracy'):.2f}% majority baseline. These are descriptive examples from the complete source table; they do not distinguish optimization interference, reduced exposure, representation effects or formula use.\n\n")
    return "".join(text)


def report(out, data, paired, summary, diagnostics, source_rows, compute):
    cfg = data["config"]
    total = {}
    for category in ("pretraining", "baseline_adaptation", "pretrained_adaptation"):
        rows = [r for r in compute if r["category"] == category]
        total[category] = {"runs": len(rows), **{key: sum(float(r[key]) for r in rows) for key in ["seconds", "updates", "input_tokens", "padded_input_tokens", "supervised_tokens", "sample_exposures"]}}
    pre = [r for r in compute if r["category"] == "pretraining"]
    result = {"status": "complete", "independent_paired_repeat_units": 3, "pretraining_models": 24, "target_training_runs": 108, "raw_test_rows": len(data["rows"]), "paired_delta_rows": len(paired), "source_learning_rows": len(source_rows), "cost_by_category": total, "pretraining_budget_tail_tokens_min": min(int(r["budget_tail_tokens"]) for r in pre), "pretraining_budget_tail_tokens_max": max(int(r["budget_tail_tokens"]) for r in pre), "pretraining_update_min": min(int(r["updates"]) for r in pre), "pretraining_update_max": max(int(r["updates"]) for r in pre), "test_seconds": data["test"]["seconds"], "training_stage_wall_seconds": data["training"]["stage_seconds"], "raw_test_results_sha256": sha(out / "raw_test_results.csv"), "frozen_protocol_sha256": sha(out / "FROZEN_PROTOCOL.json"), "frozen_selections_sha256": sha(out / "FROZEN_SELECTIONS.json"), "test_based_selection": False, "additional_training_requested": False, "uncertainty_note": "Sample SD of three paired repetitions is descriptive; shared models, targets and scratch baselines do not increase independent sample size."}
    write_json(out / "SUMMARY.json", result)
    primary = [r for r in summary if r["checkpoint"] == PRIMARY and r["length"] == "all"]
    index = {(r["target"], r["chain"], r["k"]): r for r in primary}
    text = ["# First PermWorld task-count experiment\n\n",
        "All 24 pretraining models and 108 target-training runs completed. The four held-out targets are cycle_count, lis_length, components and fixed_points. Each pretraining model is reused for all four targets; each of the 12 scratch runs is shared across all eight source-set conditions. This report retains the complete prespecified scope, including negative transfer, no gain and incomplete source learning. No task replacement, learning-rate search or budget extension follows the results.\n\n",
        "The primary result uses the checkpoint selected by development accuracy, then answer cross-entropy, then earliest update. A fixed 800-update endpoint is also reported for every run. Both were fixed before generating the independent test objects; test performance never changes selection. All accuracies require generation of the complete answer and EOS from the prompt alone.\n\n",
        findings(data, paired, summary, source_rows),
        "## Independent-test results\n\n",
        "Values below are means over three paired repeat units. Parentheses show sample SD of the **within-repeat gain over the shared scratch baseline**, in percentage points. These are descriptive spreads, not confidence intervals or tests of significance.\n\n",
        "| Target | Chain | k=2: accuracy; gain (SD) | k=4 | k=8 | k=16 | Shared scratch accuracy |\n|---|---|---:|---:|---:|---:|---:|\n"]
    for target in data["targets"]:
        baseline = index[target, "baseline", 0]
        for chain in data["chains"]:
            values = [index[target, chain, k] for k in data["ks"]]
            cells = [f"{100*r['accuracy_mean']:.2f}%; {r['paired_delta_pp_mean']:+.2f} ({r['paired_delta_pp_sample_sd']:.2f})" for r in values]
            text.append(f"| {target} | {chain} | " + " | ".join(cells) + f" | {100*baseline['accuracy_mean']:.2f}% |\n")
    text.append("\nThe following k=16 versus k=2 comparison describes the change along each fixed chain for development-selected checkpoints; it is a paired k=16 minus k=2 accuracy difference. The shared scratch baseline cancels within each repeat. The SD is computed from those three differences, not from marginal condition SDs. This is neither a significance test nor a scaling-law fit.\n\n| Target | Chain | Mean k16 minus k2 (sample SD) | Three individual differences |\n|---|---|---:|---|\n")
    for target in data["targets"]:
        for chain in data["chains"]:
            values = []
            for seed in data["seeds"]:
                subset = [r for r in paired if r["target"] == target and r["chain"] == chain and r["initialization_seed"] == seed and r["checkpoint"] == PRIMARY and r["length"] == "all"]
                values.append(next(r["paired_delta_pp"] for r in subset if r["k"] == 16) - next(r["paired_delta_pp"] for r in subset if r["k"] == 2))
            text.append(f"| {target} | {chain} | {statistics.mean(values):+.2f} ({statistics.stdev(values):.2f}) pp | " + ", ".join(f"{v:+.2f}" for v in values) + " pp |\n")
    text.append("\n[Per-repeat k contrasts](paired_k_contrasts.csv) and [their paired summaries](k_contrast_summary.csv) also retain the adjacent 2-to-4, 4-to-8 and 8-to-16 changes, both checkpoint policies and each length. These contrasts reuse the same three units and do not create additional independent replications.\n")
    text.append("\nThe plots show each repeat separately and retain the two chains. Scratch baselines repeat visually to aid comparison but were trained only once per target and repeat.\n\n")
    text.extend(f"![{target}: independent-test accuracy and paired gain over shared scratch](figures/{target}_task_count.png)\n\n" for target in data["targets"])
    text.append("\n## What the experiment controls, and what changes with k\n\n"
        "k counts nonconstant task instances after removing exact same-input aliases, not independent mathematical information. The pool has 24 instances from 22 templates. Every prefix has distinct templates, so its template count also equals k. Two nested chains have different membership at every k. Their broad family counts match: cycle/subsequence at k=2; one cycle/subsequence/block/legacy instance at k=4; two of each at k=8; and 4/3/4/5 at k=16. The k=2 sets consequently do not cover every family. Definitions and membership were frozen before training, using no transfer outcome.\n\n"
        "Increasing k changes task composition, mathematical relationships, answer distributions and per-task exposure. Alpha at k=16 contains cycle_count=even_cycle_count+odd_cycle_count and fixed_points=n-exceedances-deficiencies (also even+odd-nontrivial_cycle_count). Beta lacks those registered complete same-input query sets but has fixed_points(p)=anti_fixed_points(complement(p)). Transformed LIS/LDS and component/global-descent relations are retained. These legal relationships are not input leakage; neither their availability nor observed learning curves establish a learned mechanism. Other joint derivations remain unclassified. The curves do not isolate a pure causal count effect.\n\n"
        "Task sampling is balanced by examples, with maximum task exposure difference at most one example at the stopping boundary. Total nonpadding prompt-plus-answer sequence tokens are capped at 4,000,000 per pretrained model; supervised tokens are recorded separately. Variable prompt and answer lengths therefore produce unequal token shares across tasks despite uniform example sampling. A final whole-example prefix is used without truncating answers, and the unspent token tail is reported. Learning rate follows consumed-token progress.\n\n"
        "## Source learning and adaptation diagnostics\n\n"
        "Source evaluation occurs at initialization, the first update crossing half the token budget, and the fixed endpoint. Training accuracy uses a fixed 256-object probe per length; source development uses all 256 objects per length. Majority labels come from each complete 4,096-object training pool per length. These curves may reveal underlearning or covariation with transfer, but they do not identify a mechanism.\n\n")
    endpoint_source = [r for r in source_rows if r["stage"] == "endpoint" and r["split"] == "source_dev" and r["length"] == "all"]
    low = [r for r in endpoint_source if float(r["above_majority_pp"]) <= 0]
    text.append(f"Across the {len(endpoint_source)} source-task/run endpoint records, {len(low)} have development accuracy at or below their train-derived majority baseline. These records are overlapping tasks and shared models, not independent observations. Full per-task/per-length values, losses, exposure and probe-to-development gaps remain in the CSVs.\n\n")
    adapts = [r for r in data["records"] if r["stage"] == "adaptation"]
    earlier = sum(r["best_update"] < 800 for r in adapts)
    selected_end = [r for r in diagnostics if r["length"] == "all"]
    plus = sum(r["selected_minus_endpoint_test_pp"] > 0 for r in selected_end)
    minus = sum(r["selected_minus_endpoint_test_pp"] < 0 for r in selected_end)
    text.append(f"Development selects an update earlier than 800 in {earlier}/108 runs. On the independent test, the preselected checkpoint is higher than the endpoint in {plus} runs and lower in {minus}; the rest tie. These descriptive counts do not select a new protocol. See [checkpoint diagnostics](checkpoint_diagnostics.csv) and [comparison figure](figures/checkpoint_comparison.png).\n\n")
    text.append("## Fixed data, model and budget\n\n"
        "All conditions share the same role-specific inputs within each repeat. Inputs are isolated by complete inverse/reverse/complement orbits across source training, source development, target training, target development, final test and prior experiments. Generation checks conservative finite-domain capacity before rejection sampling and never relaxes isolation. Lengths 10 and 12 occur in every training/evaluation role, so per-length performance is not unseen-length extrapolation. Each target uses 128 training examples per length (256 total), 256 development examples per length, and 512 final test examples per length.\n\n"
        "The model is the platform's 4-layer, width-256, 8-head Transformer with the current 203-token vocabulary and context length 128. Input encoding is one_line_v1 and sequence encoding is permworld_sequence_v1. Loss averages answer-token-plus-EOS cross-entropy within each example, then across examples. The parameterized I3/D3 sources exercise exact multi-token values above 99. At adaptation, all four held-out task-token rows in the tied embedding/output matrix are reset to the paired initial values; other pretrained weights are loaded exactly. AdamW is freshly created. Training uses 800 updates, batch size 64, initial LR 3e-4, 100-update warmup and cosine decay to 3e-5. There is one learning-rate candidate and eight eligible development checkpoints per condition.\n\n"
        "| Initialization seed | Data seed | Pretraining sampling seed | Adaptation sampling seed |\n|---:|---:|---:|---:|\n")
    for repeat in cfg["repeats"]:
        text.append(f"| {repeat['initialization_seed']} | {repeat['data_seed']} | {repeat['pretrain_sampling_seed']} | {repeat['adaptation_sampling_seed']} |\n")
    text.append("\nTarget majority answers are chosen separately from the 128 training labels at each length. Their mean independent-test accuracies are shown below; they are shared across conditions and are not fitted to test labels.\n\n| Target | n=10 majority | n=12 majority |\n|---|---:|---:|\n")
    for target in data["targets"]:
        majority = [next(r["majority_accuracy_mean"] for r in summary if r["target"] == target and r["chain"] == "baseline" and r["checkpoint"] == PRIMARY and r["length"] == n) for n in ["10", "12"]]
        text.append(f"| {target} | {100*majority[0]:.2f}% | {100*majority[1]:.2f}% |\n")
    text.append("\n## Actual compute\n\n| Stage | Runs | Updates | Nonpadding sequence tokens | Supervised tokens | Cumulative run minutes |\n|---|---:|---:|---:|---:|---:|\n")
    for name, values in total.items():
        text.append(f"| {name} | {values['runs']} | {values['updates']:,.0f} | {values['input_tokens']:,.0f} | {values['supervised_tokens']:,.0f} | {values['seconds']/60:.2f} |\n")
    text.append(f"\nPretraining uses {result['pretraining_update_min']:,}–{result['pretraining_update_max']:,} updates per model; unused token tails range from {result['pretraining_budget_tail_tokens_min']} to {result['pretraining_budget_tail_tokens_max']}. Independent test evaluation took {result['test_seconds']/60:.2f} minutes. Training-stage wall time was {result['training_stage_wall_seconds']/60:.2f} minutes. Per-run timers include training, development evaluations and checkpoint/table writes; encoding and audit overhead are separate. The stage timer overlaps the run timers and must not be added to them. Padded sequence tokens, answer-only tokens, actual exposures and per-run memory are retained in [compute.csv](compute.csv) and [task exposures](task_exposures.csv).\n\n")
    text.append("## Limits and next decision\n\n"
        "There are three paired repeat units, not 24 independent pretrained observations per target or 108 independent transfer replications. Conditions share initialization, objects, target sampling and scratch baselines; four targets also share each pretrained model. Pairing is preserved by subtracting the corresponding baseline inside each repeat. The two fixed chains are compositions, not random samples of all possible source sets. Small repeat counts and finite test sets leave substantial uncertainty; marginal standard deviations of two conditions cannot establish a difference.\n\n"
        "This is a finite task-count curve at two lengths, one model, one token budget and 256 target examples. It is not a sample-efficiency curve or evidence of a scaling law. The next priority is to separate reduced source exposure from composition and source learnability in a separately prespecified comparison, using the cross-chain LIS decline and chain-dependent fixed-points behavior as evidence to investigate. Merely increasing k beyond 16 would leave those explanations mixed. Any composition controls or budget changes require a new protocol; no additional training or source replacement is run here.\n\n"
        "- [Raw test results](raw_test_results.csv), [paired gains](paired_deltas.csv), [summary](summary.csv)\n"
        "- [Source learning](source_learning.csv), [matched source summaries](source_summary.csv), [target training/development curves](adaptation_curves.csv), [majority baselines by length](majority_per_length.csv)\n"
        "- [Frozen protocol](FROZEN_PROTOCOL.json), [development selections](FROZEN_SELECTIONS.json), [task design and relationships](design/README.md)\n"
        "- [Reproduction instructions](README.md)\n")
    if (out / "PRESERVATION_VERIFICATION.json").exists():
        text.append("\nThe separate [historical preservation verification](PRESERVATION_VERIFICATION.json) checked the archived file bytes, including local checkpoints, and resolved the prior English translation through its existing ledger. Historical scientific manifests were not rewritten for this study.\n")
    if (out / "RUN_AUDIT.json").exists():
        audited = read_json(out / "RUN_AUDIT.json")
        assert audited["status"] == "pass" and audited["raw_result_rows"] == 648
        text.append(f"\nThe [independent completed-run audit](RUN_AUDIT.json) passed, checking all 24 source and 108 target runs, replaying their recorded sampling streams, rebuilding the three initializations, and validating checkpoint/configuration bindings and development-only selection. It recomputed {audited.get('development_checkpoints_recomputed', 0)} development checkpoint forwards and performed no new test neural forwards. See [per-run accounting](audit_run_accounting.csv) and [development-forward checks](audit_development_forwards.csv) for the detailed evidence.\n")
    if (out / "SOURCE_DISTRIBUTION_AUDIT.json").exists():
        text.append("\n[Complete source answer distributions](source_answer_distributions.csv) and their [distribution audit](SOURCE_DISTRIBUTION_AUDIT.json) are retained. Different tasks have different answer supports and imbalance, so raw accuracies across tasks are interpreted relative to their own training-derived majority references.\n")
    (out / "REPORT.md").write_text("".join(text), encoding="utf-8")
    return result


def reproduction_readme(out, data):
    # Only references supported command-line stages; never imports the runner.
    text = """# PermWorld task-count experiment v1

This directory contains the completed first fixed-token task-count experiment:
24 pretrained models, four held-out targets, three paired repeat units, 96
pretrained adaptations and 12 shared scratch baselines. Historical pilot and
platform evidence remain separate. Checkpoints are retained locally and ignored
by Git; raw CSVs, code, frozen configuration, reports and figures are shareable.

Start with [REPORT.md](REPORT.md), [raw_test_results.csv](raw_test_results.csv),
[paired_deltas.csv](paired_deltas.csv), and the [frozen design](design/README.md).
The primary checkpoint is selected exclusively by development accuracy, CE and
earliest update; fixed endpoints are also reported without test-based selection.
Figures show each repeat and descriptive sample SD. Shared baselines and shared
pretrained models never count as extra independent repetitions.

The completed-run [independent audit](RUN_AUDIT.json), [per-run accounting](audit_run_accounting.csv),
and [development-forward checks](audit_development_forwards.csv) provide the
validation evidence. The [environment inventory](ENVIRONMENT.json) records
Windows 11, RTX 5070, Python 3.13.14, PyTorch 2.11.0+cu128, NumPy 2.5.4,
Matplotlib 3.11.2 and installed package versions. This package inventory was
collected after the run; runtime environment evidence was also captured in
`SMOKE.json` and `TRAINING_STARTED.json` before formal result reporting.

## Reproduction

Run commands from the repository root using the project environment. Use a fresh
output directory so the completed experiment and its hashes remain untouched.
The canonical code has fixed design, seed and budget constants; copying output
does not permit changing the scientific protocol. CUDA is required for formal
training and evaluation. Data generation also requires the retained historical
object pools used for orbit exclusion.

```powershell
.venv/Scripts/python.exe experiments/task_count_v1/design.py --out experiments/task_count_v1_reproduction/design
Copy-Item -LiteralPath experiments/task_count_v1/EXPERIMENT_PLAN.md -Destination experiments/task_count_v1_reproduction/EXPERIMENT_PLAN.md
.venv/Scripts/python.exe experiments/task_count_v1/data_pipeline.py prepare --out experiments/task_count_v1_reproduction/data
.venv/Scripts/python.exe experiments/task_count_v1/train.py smoke --out experiments/task_count_v1_reproduction
.venv/Scripts/python.exe experiments/task_count_v1/train.py freeze --out experiments/task_count_v1_reproduction
.venv/Scripts/python.exe experiments/task_count_v1/train.py train --out experiments/task_count_v1_reproduction
.venv/Scripts/python.exe experiments/task_count_v1/train.py select --out experiments/task_count_v1_reproduction
.venv/Scripts/python.exe experiments/task_count_v1/train.py test --out experiments/task_count_v1_reproduction
.venv/Scripts/python.exe experiments/task_count_v1/summarize.py --out experiments/task_count_v1_reproduction
```

The design command creates the fresh parent directory. The copied experiment
plan is required because freezing binds its bytes, together with the design,
smoke result, source code and training-data files. The commands invoke the
canonical experiment scripts while writing new data and results under the
reproduction directory; no new choice of tasks or hyperparameters is made.

After the complete test and summary, optionally reproduce the independent audit
and release inventory. The audit's distribution step must precede its complete
run; `--forwards` recomputes development metrics only. The release inventory
expects local copies of the six experiment scripts alongside the new results.

```powershell
.venv/Scripts/python.exe experiments/task_count_v1/audit_run.py --out experiments/task_count_v1_reproduction --distributions-only
.venv/Scripts/python.exe experiments/task_count_v1/audit_run.py --out experiments/task_count_v1_reproduction --forwards
Copy-Item -LiteralPath experiments/task_count_v1/train.py,experiments/task_count_v1/design.py,experiments/task_count_v1/data_pipeline.py,experiments/task_count_v1/summarize.py,experiments/task_count_v1/audit_run.py,experiments/task_count_v1/release.py -Destination experiments/task_count_v1_reproduction
.venv/Scripts/python.exe experiments/task_count_v1/release.py --out experiments/task_count_v1_reproduction
.venv/Scripts/python.exe experiments/task_count_v1/release.py --out experiments/task_count_v1_reproduction --verify
```

Test inputs are generated by the test stage only after all development selections
are frozen. Completed runs are reused only after verification. An incomplete run
fails visibly; do not overwrite its evidence or silently increase the budget.

To verify the original frozen protocol without training or evaluating a model:

```powershell
.venv/Scripts/python.exe experiments/task_count_v1/train.py verify
```

Once the release manifest exists, verify every delivered byte, or explicitly
verify only Git deliverables in a checkout without the local checkpoints:

```powershell
.venv/Scripts/python.exe experiments/task_count_v1/release.py --verify
.venv/Scripts/python.exe experiments/task_count_v1/release.py --verify --source-only
```

The summarizer imports only standard-library utilities and Matplotlib. It checks
all expected run identities, table hashes, recorded selections and pairing,
then writes derived tables/plots. It never loads a checkpoint or selects a model
using test data. An independent audit, when present, provides additional checks
beyond these reporting safeguards.

The primary CSVs and figures can be analyzed without checkpoint files. This
archived run was produced on Windows, and some checkpoint metadata stores
relative paths with backslashes. A consumer that re-evaluates archived weights
on POSIX must normalize those path separators; the frozen training code is not
rewritten to alter archived metadata. Fresh reproduction generates paths for
its own operating system. The reporting script uses run identifiers and local
CSV paths, without consuming checkpoint path strings.

## Output map

- `FROZEN_PROTOCOL.json`: complete hyperparameters, seed streams, code/data hashes.
- `FROZEN_SELECTIONS.json`: all 108 development-only checkpoint decisions.
- `raw_test_results.csv`, `results_seed*.csv`: selected and endpoint results at
  n=10, n=12 and their equally weighted aggregate.
- `paired_deltas.csv`, `summary.csv`: within-repeat subtraction from shared A;
  three-repeat means and sample SD, with both checkpoint policies preserved.
- `paired_k_contrasts.csv`, `k_contrast_summary.csv`: paired k16-minus-k2 and
  adjacent k changes within each chain and repeat, with individual values and
  descriptive mean/sample SD; the shared scratch baseline cancels exactly.
- `checkpoint_diagnostics.csv`: selected-minus-endpoint test differences.
- `source_learning.csv`: start/midpoint/endpoint accuracy, loss, majority and
  training-probe/development gap for every source task.
- `source_summary.csv`: matched task/chain/k/stage/length summaries over the three
  repeat units; no pooling of different source tasks as independent replicates.
- `adaptation_curves.csv`: all target training/development evaluations.
- `task_exposures.csv`, `compute.csv`: actual example/token exposures, updates,
  padded versus nonpadding tokens, budget tails and runtime.
- `majority_per_length.csv`: labels derived only from the relevant training pool,
  evaluated separately on training probes, development and final target tests.
- `SUMMARY.json`: completion counts, costs and bindings to the raw evidence.
- `figures/`: PNG and SVG figures; no additional model evaluation.

No transfer condition, target or unsuccessful run is discarded based on outcomes.
This round ends after the frozen comparison; it does not trigger extra training.
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def summarize(out):
    data = collect(out)
    paired, summary, diagnostics = summarize_targets(data)
    changes, change_summary = k_contrasts(data, paired)
    source, majority, compute = source_and_cost(data)
    write_csv(out / "paired_deltas.csv", paired)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paired_k_contrasts.csv", changes)
    write_csv(out / "k_contrast_summary.csv", change_summary)
    write_csv(out / "checkpoint_diagnostics.csv", diagnostics)
    write_csv(out / "source_learning.csv", source)
    write_csv(out / "source_summary.csv", source_summary(source))
    write_csv(out / "adaptation_curves.csv", data["curves"])
    write_csv(out / "task_exposures.csv", data["exposures"])
    write_csv(out / "compute.csv", compute)
    write_csv(out / "majority_per_length.csv", majority)
    for seed in data["seeds"]:
        write_csv(out / f"results_seed{seed}.csv", [row for row in data["rows"] if int(row["initialization_seed"]) == seed])
    figures(out, data, paired, summary)
    result = report(out, data, paired, summary, diagnostics, source, compute)
    reproduction_readme(out, data)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(summarize(args.out.resolve()), indent=2))
