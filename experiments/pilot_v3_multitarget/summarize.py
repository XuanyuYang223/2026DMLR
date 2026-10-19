"""Summarize completed v3 CSV/JSON artifacts; no model import or forward pass."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "artifacts"
COLORS = {"A": "#68778d", "B": "#1683ad", "C": "#d28032"}
CONTRASTS = [("B", "A"), ("C", "A"), ("B", "C")]
LABELS = {
    "single_not_mastered": "Single task not fully learned under this configuration",
    "multi_degradation": "Learned alone but degraded in multitask training",
    "insufficient_degradation_evidence": "Insufficient evidence of multitask degradation",
}


def js(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, values):
    values = list(values)
    if not values:
        raise ValueError(f"Empty required table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in values for k in r)))
        writer.writeheader()
        writer.writerows(values)


def num(row, key):
    return float(row[key])


def integer(row, key):
    return int(float(row[key]))


def truth(value):
    return str(value).lower() in ("true", "1")


def pct(value):
    return f"{100 * float(value):.2f}%"


def pp(value):
    return f"{100 * float(value):+.2f}"


def stats(values, prefix):
    values = list(values)
    return {prefix + "_mean": mean(values), prefix + "_sample_sd": stdev(values) if len(values) > 1 else 0,
            prefix + "_min": min(values), prefix + "_max": max(values)}


def mdtable(headers, values):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |",
                       "| " + " | ".join("---" for _ in headers) + " |",
                       *("| " + " | ".join(map(str, r)) + " |" for r in values)])


def savefig(fig, out, name):
    fig.savefig(out / "figures" / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def result_tables(out, cfg, selected, candidates):
    targets, seeds = cfg["targets"], cfg["seeds"]
    index = {(r["target"], integer(r, "seed"), r["condition"]): r for r in selected}
    assert len(index) == 36
    summary, deltas, delta_summary, endpoints = [], [], [], []
    for target in targets:
        for condition in "ABC":
            group = [index[target, seed, condition] for seed in seeds]
            record = dict(target=target, condition=condition, n_seeds=len(seeds), target_samples=256, pretrain_updates=0 if condition == "A" else 4800)
            for metric in ("test_accuracy", "test_answer_ce", "dev_accuracy", "dev_ce", "train_accuracy", "majority_accuracy"):
                record.update(stats((num(r, metric) for r in group), metric))
            record.update({f"seed{seed}_test_accuracy": num(index[target, seed, condition], "test_accuracy") for seed in seeds})
            summary.append(record)
            for row in group:
                endpoint = next(r for r in candidates if r["target"] == target and integer(r, "seed") == integer(row, "seed")
                                and r["condition"] == condition and num(r, "lr") == num(row, "lr") and r["checkpoint"] == "endpoint")
                endpoints.append(dict(target=target, seed=row["seed"], condition=condition, lr=row["lr"], selected_step=row["adapt_step"],
                                      selected_dev_accuracy=row["dev_accuracy"], endpoint_dev_accuracy=endpoint["dev_accuracy"],
                                      selected_test_accuracy=row["test_accuracy"], endpoint_test_accuracy=endpoint["test_accuracy"],
                                      endpoint_minus_selected_test=num(endpoint, "test_accuracy") - num(row, "test_accuracy")))
        for first, second in CONTRASTS:
            current = []
            for seed in seeds:
                a, b = index[target, seed, first], index[target, seed, second]
                record = dict(target=target, seed=seed, contrast=f"{first}-{second}",
                              test_accuracy_delta=num(a, "test_accuracy") - num(b, "test_accuracy"),
                              dev_accuracy_delta=num(a, "dev_accuracy") - num(b, "dev_accuracy"),
                              first_test_accuracy=a["test_accuracy"], second_test_accuracy=b["test_accuracy"])
                deltas.append(record)
                current.append(record)
            delta_summary.append(dict(target=target, contrast=f"{first}-{second}", n_seeds=len(seeds),
                                      **stats((r["test_accuracy_delta"] for r in current), "delta"),
                                      negative_seeds=sum(r["test_accuracy_delta"] < 0 for r in current),
                                      zero_seeds=sum(r["test_accuracy_delta"] == 0 for r in current),
                                      positive_seeds=sum(r["test_accuracy_delta"] > 0 for r in current)))
    write_csv(out / "results_summary.csv", summary)
    write_csv(out / "paired_deltas.csv", deltas)
    write_csv(out / "paired_deltas_summary.csv", delta_summary)
    write_csv(out / "selected_vs_endpoint.csv", endpoints)
    return summary, deltas, delta_summary, endpoints


def result_plots(out, cfg, selected, summary, deltas, delta_summary):
    targets, seeds = cfg["targets"], cfg["seeds"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for axis, target in zip(axes.flat, targets):
        for offset, condition in enumerate("ABC"):
            group = sorted([r for r in selected if r["target"] == target and r["condition"] == condition], key=lambda r: integer(r, "seed"))
            axis.plot(range(len(seeds)), [num(r, "test_accuracy") for r in group], marker="o", color=COLORS[condition], label=condition)
        baseline = sorted([r for r in selected if r["target"] == target and r["condition"] == "A"], key=lambda r: integer(r, "seed"))
        axis.plot(range(len(seeds)), [num(r, "majority_accuracy") for r in baseline], ":", color="grey", label="train-majority baseline")
        axis.set(title=target, xlabel="Paired seed", ylabel="Isolated test answer accuracy", ylim=(0, 1.03))
        axis.set_xticks(range(len(seeds)), [str(s) for s in seeds])
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("256 target examples; development-selected checkpoint; all three paired seeds")
    fig.tight_layout()
    savefig(fig, out, "target_accuracy_by_seed.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for axis, target in zip(axes.flat, targets):
        for i, seed in enumerate(seeds):
            subset = [next(r for r in deltas if r["target"] == target and r["seed"] == seed and r["contrast"] == f"{a}-{b}") for a, b in CONTRASTS]
            axis.plot([j + (i - 1) * .07 for j in range(3)], [100 * r["test_accuracy_delta"] for r in subset], marker="o", alpha=.8, label=f"seed {seed}")
        axis.axhline(0, color="black", lw=.8)
        axis.set_xticks(range(3), [f"{a}-{b}" for a, b in CONTRASTS])
        axis.set(title=target, ylabel="Paired accuracy difference (percentage points)")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("Every paired-seed contrast; signs and magnitudes retained")
    fig.tight_layout()
    savefig(fig, out, "paired_deltas_by_seed.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for axis, (first, second) in zip(axes, CONTRASTS):
        contrast = f"{first}-{second}"
        group = [next(r for r in delta_summary if r["target"] == target and r["contrast"] == contrast) for target in targets]
        axis.errorbar(range(4), [100 * r["delta_mean"] for r in group], yerr=[100 * r["delta_sample_sd"] for r in group], fmt="D", color="black", capsize=5, label="mean +/- sample SD")
        for i, seed in enumerate(seeds):
            values = [next(r["test_accuracy_delta"] for r in deltas if r["target"] == target and r["seed"] == seed and r["contrast"] == contrast) for target in targets]
            axis.scatter([j + (i - 1) * .06 for j in range(4)], [100 * v for v in values], s=25, alpha=.7, label=f"seed {seed}")
        axis.axhline(0, color="grey", lw=.8)
        axis.set_xticks(range(4), [t.replace("_", "\n") for t in targets], fontsize=8)
        axis.set(title=contrast, ylabel="Accuracy difference (percentage points)")
        axis.grid(axis="y", alpha=.2)
        axis.legend(fontsize=7)
    fig.suptitle("Descriptive n=3 per target; shared inputs across targets, no independent-task inference")
    fig.tight_layout()
    savefig(fig, out, "paired_deltas_summary.png")


def source_tables(out, protocol):
    allrows, metas = [], []
    for kind in ("single", "multi"):
        for path in sorted((out / kind).glob("*/completed.json")):
            meta = js(path)
            metas.append(dict(path=path, **meta))
            for r in rows(path.parent / "curve.csv"):
                allrows.append(dict(**r, new_compute=meta["new_compute"]))
    write_csv(out / "source_learning_all.csv", allrows)
    endpoint_index = {(r["run_id"], integer(r, "step"), r["task"], r["split"]): r for r in allrows}
    wide = []
    metrics = ("accuracy", "answer_ce", "two_token_ce", "eos_ce", "eos_accuracy", "majority_accuracy", "majority_label", "count")
    for meta in metas:
        if meta["kind"] != "single":
            continue
        run = meta["path"].parent.name
        for step in (1200, 4800):
            for task in meta["tasks"]:
                record = dict(seed=meta["seed"], task=task, step=step, length=10, exposures=step * 64)
                for split in ("train", "dev"):
                    r = endpoint_index[run, step, task, split]
                    record.update({split + "_" + m: r[m] for m in metrics})
                wide.append(record)
    write_csv(out / "single_task_references.csv", wide)
    comparisons, common = [], []
    for group in protocol["groups"]:
        target = group["target"]
        for seed in protocol["config"]["seeds"]:
            for condition in "BC":
                run = f"s{seed}_{target}_{condition}"
                for task in group["sources"][condition]:
                    record = dict(target=target, seed=seed, condition=condition, task=task, length=10,
                                  role="related" if task == group["related"] else "replacement" if task == group["replacement"] else "common",
                                  multi_updates=4800, multi_task_exposures=76800, single1200_exposures=76800, single4800_exposures=307200)
                    for name, name_run, step in (("single1200", f"s{seed}_{task}", 1200), ("single4800", f"s{seed}_{task}", 4800), ("multi4800", run, 4800)):
                        for split in ("train", "dev"):
                            r = endpoint_index[name_run, step, task, split]
                            record.update({name + "_" + split + "_" + m: r[m] for m in metrics})
                    single, multi = num(record, "single4800_dev_accuracy"), num(record, "multi4800_dev_accuracy")
                    record["multi_minus_single4800_dev_accuracy"] = multi - single
                    record["multi_minus_single1200_dev_accuracy"] = multi - num(record, "single1200_dev_accuracy")
                    record["classification"] = "single_not_mastered" if single < .90 else "multi_degradation" if single - multi >= .05 else "insufficient_degradation_evidence"
                    record["classification_zh"] = LABELS[record["classification"]]
                    comparisons.append(record)
            for task in sorted(set(group["sources"]["B"]) & set(group["sources"]["C"])):
                record = dict(target=target, seed=seed, task=task, length=10)
                for condition in "BC":
                    for split in ("train", "dev"):
                        r = endpoint_index[f"s{seed}_{target}_{condition}", 4800, task, split]
                        record.update({condition + "_" + split + "_" + m: r[m] for m in metrics})
                record["B_minus_C_dev_accuracy"] = num(record, "B_dev_accuracy") - num(record, "C_dev_accuracy")
                record["B_minus_C_dev_answer_ce"] = num(record, "B_dev_answer_ce") - num(record, "C_dev_answer_ce")
                common.append(record)
    assert len(comparisons) == 96 and len(common) == 36
    write_csv(out / "single_vs_multi_sources.csv", comparisons)
    write_csv(out / "common_source_differences.csv", common)
    classification = []
    for target in protocol["config"]["targets"]:
        for condition in "BC":
            for key in LABELS:
                classification.append(dict(target=target, condition=condition, classification=key, classification_zh=LABELS[key],
                                           task_seed_occurrences=sum(r["target"] == target and r["condition"] == condition and r["classification"] == key for r in comparisons)))
    write_csv(out / "source_classification_summary.csv", classification)
    learning_summary = []
    for group in protocol["groups"]:
        for condition in "BC":
            for task in group["sources"][condition]:
                group_rows = [r for r in comparisons if r["target"] == group["target"] and r["condition"] == condition and r["task"] == task]
                record = dict(target=group["target"], condition=condition, task=task, role=group_rows[0]["role"], n_seeds=len(group_rows))
                for metric in ("single1200_dev_accuracy", "single4800_dev_accuracy", "multi4800_dev_accuracy", "multi4800_dev_answer_ce", "multi4800_dev_majority_accuracy", "multi_minus_single4800_dev_accuracy"):
                    record.update(stats((num(r, metric) for r in group_rows), metric))
                learning_summary.append(record)
    write_csv(out / "source_learning_summary.csv", learning_summary)
    return allrows, metas, comparisons, common, classification, learning_summary


def source_plots(out, protocol, allrows, common):
    idx = defaultdict(list)
    for r in allrows:
        idx[r["run_id"], r["task"], r["split"]].append(r)
    for values in idx.values():
        values.sort(key=lambda r: integer(r, "step"))
    for group in protocol["groups"]:
        target = group["target"]
        tasks = list(dict.fromkeys(group["sources"]["B"] + group["sources"]["C"]))
        for seed in protocol["config"]["seeds"]:
            fig, axes = plt.subplots(len(tasks), 2, figsize=(12, 2.6 * len(tasks)), squeeze=False)
            for i, task in enumerate(tasks):
                specs = [("single", f"s{seed}_{task}", "#45505b")]
                specs += [(c, f"s{seed}_{target}_{c}", COLORS[c]) for c in "BC" if task in group["sources"][c]]
                for label, run, color in specs:
                    for split, style in (("train", ":"), ("dev", "-")):
                        values = idx[run, task, split]
                        curve_label = f"{label} {split}"
                        if values and integer(values[0], "step") > 0:
                            curve_label += f" (starts {integer(values[0], 'step')})"
                        for axis, metric in zip(axes[i], ("accuracy", "answer_ce")):
                            axis.plot([integer(r, "step") for r in values], [num(r, metric) for r in values], style, marker=".", color=color, label=curve_label, alpha=.85)
                baseline = num(idx[f"s{seed}_{task}", task, "dev"][0], "majority_accuracy")
                axes[i, 0].axhline(baseline, color="grey", linestyle="--", alpha=.5, label="dev majority")
                axes[i, 0].set_ylim(0, 1.03)
                for axis, ylabel in zip(axes[i], ("Answer accuracy", "Answer CE")):
                    axis.axvline(1200, color="grey", alpha=.25)
                    axis.set(title=task, xlabel="Total optimizer updates", ylabel=ylabel)
                    axis.grid(alpha=.2)
                    axis.legend(fontsize=7, ncol=2)
            historical_note = "\nHistorical B/C full-source curves begin at 2400; earlier points unavailable" if target == "fixed_points" and seed in (17, 42) else ""
            fig.suptitle(f"{target}, seed {seed}; single 4800 = 4x task exposure; single 1200 differs in LR history" + historical_note)
            fig.tight_layout()
            savefig(fig, out, f"source_learning_{target}_seed{seed}.png")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for axis, group in zip(axes.flat, protocol["groups"]):
        target = group["target"]
        tasks = sorted(set(group["sources"]["B"]) & set(group["sources"]["C"]))
        for i, seed in enumerate(protocol["config"]["seeds"]):
            values = [next(r for r in common if r["target"] == target and r["seed"] == seed and r["task"] == task) for task in tasks]
            axis.plot([j + (i - 1) * .06 for j in range(3)], [100 * r["B_minus_C_dev_accuracy"] for r in values], marker="o", label=f"seed {seed}")
        axis.axhline(0, color="black", lw=.8)
        axis.set_xticks(range(3), [t.replace("_", "\n") for t in tasks], fontsize=8)
        axis.set(title=target, ylabel="Common-source dev B - C (percentage points)")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, out, "common_source_differences.png")


def source_effects(out, comparisons, source_metas):
    """Describe both signs; retain the separately defined mastery classification."""
    multi_states = {m["path"].parent.name: m["model_state_hash"] for m in source_metas if m["kind"] == "multi"}
    unique = {}
    for r in comparisons:
        run_id = f"s{r['seed']}_{r['target']}_{r['condition']}"
        state = multi_states[run_id]
        key = (r["seed"], r["task"], state)
        if key not in unique:
            unique[key] = dict(seed=r["seed"], task=r["task"], multi_model_state_hash=state,
                               single4800_dev_accuracy=num(r, "single4800_dev_accuracy"),
                               multi4800_dev_accuracy=num(r, "multi4800_dev_accuracy"),
                               multi_minus_single4800_dev_accuracy=r["multi_minus_single4800_dev_accuracy"],
                               single4800_task_exposures=r["single4800_exposures"],
                               multi4800_task_exposures=r["multi_task_exposures"],
                               logical_runs=[], logical_task_occurrences=0)
        record = unique[key]
        assert record["multi_minus_single4800_dev_accuracy"] == r["multi_minus_single4800_dev_accuracy"]
        record["logical_runs"].append(run_id)
        record["logical_task_occurrences"] += 1
    values = []
    for record in unique.values():
        values.append(dict(record, logical_runs=";".join(record["logical_runs"])))
    values.sort(key=lambda r: (r["seed"], r["task"], r["multi_model_state_hash"]))
    write_csv(out / "source_learning_effects_unique.csv", values)
    summaries = []
    for scope, records in (("logical_task_occurrences", comparisons), ("unique_seed_task_source_state", values)):
        deltas = [r["multi_minus_single4800_dev_accuracy"] for r in records]
        summaries.append(dict(scope=scope, n_records=len(records),
                              improved_at_least_5pp=sum(d >= .05 for d in deltas),
                              decreased_at_least_5pp=sum(d <= -.05 for d in deltas),
                              within_5pp=sum(-.05 < d < .05 for d in deltas),
                              **stats(deltas, "multi_minus_single4800_dev_accuracy")))
    write_csv(out / "source_learning_effect_counts.csv", summaries)
    return dict(rows=values, counts=summaries)


def target_curves(out, cfg, selections):
    curves = rows(out / "target_curves.csv")
    assert len(curves) == 108 * 9
    late = []
    for path in sorted((out / "adapt").glob("*/completed.json")):
        meta = js(path)
        points = {integer(r, "step"): r for r in rows(path.parent / "curve.csv")}
        r = dict(target=meta["target"], seed=meta["seed"], condition=meta["condition"], lr=meta["lr"])
        for split in ("train", "dev"):
            for metric in ("accuracy", "answer_ce"):
                key = split + "_" + metric
                r[key + "_400"] = num(points[400], key)
                r[key + "_800"] = num(points[800], key)
                r[key + "_delta_800_minus_400"] = num(points[800], key) - num(points[400], key)
        late.append(r)
    write_csv(out / "late_training_diagnostics.csv", late)
    for target in cfg["targets"]:
        for seed in cfg["seeds"]:
            fig, axes = plt.subplots(3, 2, figsize=(12, 9))
            for i, condition in enumerate("ABC"):
                selected = next(r for r in selections if r["target"] == target and r["seed"] == seed and r["condition"] == condition)
                subset = [r for r in curves if r["target"] == target and integer(r, "seed") == seed and r["condition"] == condition and num(r, "lr") == selected["lr"]]
                subset.sort(key=lambda r: integer(r, "step"))
                assert len(subset) == 9
                for axis, metric in zip(axes[i], ("accuracy", "answer_ce")):
                    for split, style in (("train", "-"), ("dev", "--")):
                        axis.plot([integer(r, "step") for r in subset], [num(r, split + "_" + metric) for r in subset], style, marker="o", label=split)
                    axis.axvline(selected["step"], color="black", alpha=.3, linestyle=":", label="dev-selected update")
                    axis.set(title=f"{condition}; selected LR {selected['lr']:g}", xlabel="Adaptation updates", ylabel=metric)
                    axis.grid(alpha=.2)
                    axis.legend(fontsize=8)
                axes[i, 0].axhline(num(subset[0], "dev_majority_accuracy"), color="grey", linestyle=":")
                axes[i, 0].set_ylim(0, 1.03)
            fig.suptitle(f"{target}, seed {seed}; full target train/development curves; 256 train examples")
            fig.tight_layout()
            savefig(fig, out, f"target_selected_{target}_seed{seed}.png")
    return late


def costs(out, source_metas):
    records = []
    for meta in source_metas:
        record = dict(kind=meta["kind"], run_id=meta["path"].parent.name, seed=meta["seed"], target="", new_compute=meta["new_compute"])
        record.update({key: meta[key] for key in ("updates", "input_tokens", "supervised_tokens", "sample_exposures", "seconds")})
        record.update(task_exposures=meta["task_exposures"], task_fraction=meta["task_fraction"], tasks=";".join(meta["tasks"]),
                      checkpoint=meta["checkpoint"], checkpoint_sha256=meta["checkpoint_sha256"])
        record["timer_scope"] = "new: training/full train-dev evaluations/periodic checkpoint saves plus bookkeeping; reused: historical v1+v2 training timers, includes continuation sampler replay"
        records.append(record)
    for path in sorted((out / "adapt").glob("*/completed.json")):
        meta = js(path)
        record = dict(kind="adapt", run_id=path.parent.name, seed=meta["seed"], target=meta["target"], new_compute=meta["new_compute"])
        record.update({key: meta[key] for key in ("updates", "input_tokens", "supervised_tokens", "sample_exposures", "seconds")})
        record.update(task_exposures=meta["sample_exposures"], task_fraction=1, tasks=meta["target"], checkpoint=meta["endpoint_path"], checkpoint_sha256=meta["endpoint_sha256"])
        record["timer_scope"] = "new: training/full train-dev evaluations/best and endpoint saves, excluding initialization; reused: v2 timer with different save/check bookkeeping scope"
        records.append(record)
    write_csv(out / "compute_by_run.csv", records)
    summary = []
    for kind in ("single", "multi", "adapt"):
        for new in (True, False):
            group = [r for r in records if r["kind"] == kind and r["new_compute"] == new]
            if not group:
                continue
            summary.append(dict(kind=kind, new_compute=new, runs=len(group),
                                **{key: sum(r[key] for r in group) for key in ("updates", "input_tokens", "supervised_tokens", "sample_exposures", "seconds")}))
    write_csv(out / "compute_summary.csv", summary)
    timings = [js(path) for path in sorted(out.glob("timing_*.json"))]
    write_csv(out / "stage_timings.csv", timings)
    return records, summary, timings


def composition_reuse(out, source_metas):
    multi = [m for m in source_metas if m["kind"] == "multi"]
    ordered_groups, unordered_groups, state_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    for m in multi:
        ordered_groups[tuple(m["tasks"])].append(m["path"].parent.name)
        unordered_groups[tuple(sorted(m["tasks"]))].append(m["path"].parent.name)
        state_groups[m["model_state_hash"]].append(m["path"].parent.name)
    records = []
    for m in multi:
        records.append(dict(run_id=m["path"].parent.name, seed=m["seed"], ordered_tasks=";".join(m["tasks"]),
                            unordered_tasks=";".join(sorted(m["tasks"])), model_state_hash=m["model_state_hash"],
                            same_ordered_composition_runs=";".join(ordered_groups[tuple(m["tasks"])]),
                            same_unordered_composition_runs=";".join(unordered_groups[tuple(sorted(m["tasks"]))]),
                            same_model_state_runs=";".join(state_groups[m["model_state_hash"]]),
                            identical_state_run_count=len(state_groups[m["model_state_hash"]]), new_compute=m["new_compute"]))
    write_csv(out / "source_composition_reuse.csv", records)
    return dict(logical_runs=len(multi), unique_ordered_compositions=len(ordered_groups),
                unique_unordered_compositions=len(unordered_groups), unique_model_states=len(state_groups),
                duplicate_state_groups=sum(len(group) > 1 for group in state_groups.values()),
                duplicate_state_extra_runs=sum(len(group) - 1 for group in state_groups.values()))


def report(out, report_path, protocol, selected, summary, delta_summary, comparisons, common, learning_summary, compute, timings, late, composition, effects):
    cfg, groups = protocol["config"], protocol["groups"]
    catalog = js(HERE / "catalog" / "CATALOG_AUDIT.json")
    matching = rows(out / "replacement_matching.csv")
    audit = js(out / "FINAL_AUDIT.json") if (out / "FINAL_AUDIT.json").exists() else None
    preservation_passed = bool(audit and audit.get("status") == "passed" and audit.get("old_files_byte_preserved", 0) > 0)
    preservation_text = (
        f"Final audit status=passed: byte hashes of {audit['old_files_byte_preserved']} historical files, including v1/v2 code, configurations, data, results, and checkpoints, were checked; preservation passed."
        if preservation_passed else
        "Historical byte preservation awaits the final audit. FINAL_AUDIT.json is missing, is not passed, or lacks the preserved-file count; this report cannot claim the preservation check passed."
    )
    audit_text = f"{audit.get('status', 'status field missing')}; see FINAL_AUDIT.json" if audit else "Not generated yet; run audit_final.py, then rerun this script to update the report"
    linkbase = os.path.relpath(out, report_path.parent).replace("\\", "/")
    def link(file):
        return f"{linkbase}/{file}"
    primary = []
    for target in cfg["targets"]:
        records = {r["condition"]: r for r in summary if r["target"] == target}
        contrasts = {r["contrast"]: r for r in delta_summary if r["target"] == target}
        primary.append([target, *[pct(records[c]["test_accuracy_mean"]) for c in "ABC"],
                        *[f"{pp(contrasts[c]['delta_mean'])} ± {100 * contrasts[c]['delta_sample_sd']:.2f}" for c in ("B-A", "C-A", "B-C")]])
    detailed = [[r["target"], r["seed"], r["condition"], pct(r["test_accuracy"]), pct(r["dev_accuracy"]), r["lr"], r["adapt_step"], pct(r["majority_accuracy"]), "Historical weights reused" if not truth(r["new_compute"]) else "Trained this round"] for r in selected]
    recurrence = []
    for target in cfg["targets"]:
        records = {r["contrast"]: r for r in delta_summary if r["target"] == target}
        recurrence.append([target, *[f"{records[c]['negative_seeds']}/{records[c]['zero_seeds']}/{records[c]['positive_seeds']}" for c in ("B-A", "C-A", "B-C")]])
    relation_defs = {
        "fixed_points": "Fixed-point count # {i:p_i=i}", "exceedances": "Exceedance count # {i:p_i>i}",
        "left_to_right_minima": "Number of strict new minima scanning left to right", "longest_decreasing_run": "Length of the longest contiguous strictly decreasing run (at least 1)",
    }
    transforms = {"complement": "C(p)_i=n+1-p_i", "inverse": "I(p)=p⁻¹", "reverse": "R(p)_i=p_{n+1-i}"}
    group_table = [[g["target"], relation_defs[g["target"]], g["related"], transforms[g["transform"]], g["replacement"], ", ".join(g["sources"]["B"][1:])] for g in groups]
    gates = [[r["target"], pct(r["accuracy"]), pct(r["majority_accuracy"]), r["pass_gate"]] for r in protocol["target_gates"]]
    match_table = [[r["target"], r["related"], r["replacement"], f"{num(r, 'tv'):.4f}", f"{num(r, 'endpoint_dev_accuracy_gap'):.4f}", f"{num(r, 'mean_curve_accuracy_gap'):.4f}", f"{num(r, 'score'):.4f}", r["selected"]] for r in matching]
    used_sources = {task for group in groups for tasklist in group["sources"].values() for task in tasklist}
    unused_candidates = {r["replacement"] for r in matching} - used_sources
    unused_refs = [r for r in rows(out / "single_task_references.csv") if integer(r, "step") == 4800 and r["task"] in unused_candidates]
    candidate_reference_text = (
        "Single-task records are retained for candidates outside the final source sets. The table uses actual 4800-step endpoints. Development accuracy<90% means incomplete learning under this configuration, not intrinsic task difficulty. Unselected candidates generally have only the design seed17 reference; three-seed coverage is not added.\n\n" +
        mdtable(["Unselected candidate", "seed", "Single-task train", "Single-task dev", "Dev majority baseline", "90% descriptive threshold"],
                [[r["task"], r["seed"], pct(r["train_accuracy"]), pct(r["dev_accuracy"]), pct(r["dev_majority_accuracy"]), "Not fully learned" if num(r, "dev_accuracy") < .90 else "Reached"] for r in unused_refs])
        if unused_refs else "No trained replacement candidate was excluded from every final source set in this round; all candidate single-task records remain in single_task_references.csv."
    )
    source_table = [[r["target"], r["condition"], r["task"], pct(r["single1200_dev_accuracy_mean"]), pct(r["single4800_dev_accuracy_mean"]), pct(r["multi4800_dev_accuracy_mean"]), f"{r['multi4800_dev_answer_ce_mean']:.3f}", pct(r["multi4800_dev_majority_accuracy_mean"])] for r in learning_summary]
    classcounts = Counter(r["classification"] for r in comparisons)
    effect_counts_table = [["Raw logical task occurrences" if r["scope"] == "logical_task_occurrences" else "Deduplicated by seed×task×source state",
                           r["n_records"], r["improved_at_least_5pp"], r["decreased_at_least_5pp"], r["within_5pp"]] for r in effects["counts"]]
    improvements = sorted([r for r in effects["rows"] if r["multi_minus_single4800_dev_accuracy"] >= .05],
                          key=lambda r: r["multi_minus_single4800_dev_accuracy"], reverse=True)[:3]
    decreases = sorted([r for r in effects["rows"] if r["multi_minus_single4800_dev_accuracy"] <= -.05],
                       key=lambda r: r["multi_minus_single4800_dev_accuracy"])[:3]
    extreme_effects = [("Improvement", r) for r in improvements] + [("Decline", r) for r in decreases]
    effects_examples = (mdtable(["Direction", "seed", "Source task", "Single 4800 dev", "Multi 4800 dev", "Difference pp", "Logical runs containing this state"],
                               [[sign, r["seed"], r["task"], pct(r["single4800_dev_accuracy"]), pct(r["multi4800_dev_accuracy"]),
                                 pp(r["multi_minus_single4800_dev_accuracy"]), r["logical_runs"]] for sign, r in extreme_effects])
                        if extreme_effects else "No source record reaches the descriptive difference threshold of±5 percentage points.")
    common_abs = [abs(r["B_minus_C_dev_accuracy"]) for r in common]
    negatives = {contrast: sum(r["contrast"] == contrast and r["delta_mean"] < 0 for r in delta_summary) for contrast in ("B-A", "C-A", "B-C")}
    allnegative = {contrast: sum(r["contrast"] == contrast and r["negative_seeds"] == 3 for r in delta_summary) for contrast in ("B-A", "C-A", "B-C")}
    bcswitch = sum(r["contrast"] == "B-C" and r["negative_seeds"] > 0 and r["positive_seeds"] > 0 for r in delta_summary)
    new_seconds = sum(r["seconds"] for r in compute if r["new_compute"])
    stage_seconds = sum(r["seconds"] for r in timings)
    cost_table = [[r["kind"], "New" if r["new_compute"] else "Historical reuse", r["runs"], f"{r['updates']:,}", f"{r['input_tokens']:,}", f"{r['supervised_tokens']:,}", f"{r['seconds']/60:.2f}"] for r in compute]
    ce_worse = sum(r["dev_answer_ce_delta_800_minus_400"] > 0 for r in late)
    acc_worse = sum(r["dev_accuracy_delta_800_minus_400"] < 0 for r in late)
    selected_lrs = Counter(float(r["lr"]) for r in selected)
    unanimous_negative = [target for target in cfg["targets"] if all(
        next(r["negative_seeds"] for r in delta_summary if r["target"] == target and r["contrast"] == contrast) == len(cfg["seeds"])
        for contrast in ("B-A", "C-A"))]
    observed_detail = ("Targets with both B−A and C−A negative in all three seeds are " +
                       (", ".join(unanimous_negative) if unanimous_negative else "None") +
                       "; these are observed directions across three seeds, not statistical significance or universal negative transfer on unseen tasks.")
    if "longest_decreasing_run" in cfg["targets"]:
        ldr = {(integer(r, "seed"), r["condition"]): r for r in selected if r["target"] == "longest_decreasing_run"}
        seeds = cfg["seeds"]
        offsets = {c: [num(ldr[seed, c], "test_accuracy") - num(ldr[seed, c], "majority_accuracy") for seed in seeds] for c in "AC"}
        gains = {seed: num(ldr[seed, "B"], "test_accuracy") - num(ldr[seed, "A"], "test_accuracy") for seed in seeds}
        positive_seeds = [seed for seed in seeds if gains[seed] > 0]
        other_seeds = [seed for seed in seeds if gains[seed] <= 0]
        observed_detail += (
            " For longest_decreasing_run, A/C test performance is close to the training-mode baseline: per-seed A minus baseline is " +
            ", ".join(pp(value) for value in offsets["A"]) + ", and C minus baseline is " +
            ", ".join(pp(value) for value in offsets["C"]) + " percentage points (seed order " +
            "/".join(map(str, seeds)) + "). B improves over A for seeds " +
            "/".join(map(str, positive_seeds)) + ", while seeds without improvement are " +
            ("/".join(map(str, other_seeds)) or "None") + "; corresponding B−A differences are " +
            ", ".join(pp(gains[seed]) for seed in seeds) + " percentage points."
        )
        related = next(g["related"] for g in groups if g["target"] == "longest_decreasing_run")
        related_dev = [next(num(r, "multi4800_dev_accuracy") for r in comparisons if r["target"] == "longest_decreasing_run" and r["seed"] == seed and r["condition"] == "B" and r["task"] == related) for seed in seeds]
        observed_detail += (
            f" B's related source {related} has development accuracies " + ", ".join(pct(value) for value in related_dev) +
            ", showing source learning and target benefit both varying with seed. Seeds change both input pools and initialization; three paired observations cannot establish a causal source."
        )
    observed_detail += " Target eligibility was established with 8192 single-task training examples; this does not imply A can learn with 256 examples and 800 adaptation steps. Target eligibility and small-sample learning failure therefore require separate interpretation."
    bc_means = [r["delta_mean"] for r in delta_summary if r["contrast"] == "B-C"]
    composition_variable = bcswitch > 0 or (min(bc_means) < 0 < max(bc_means))
    if composition_variable and any(common_abs) and classcounts["multi_degradation"]:
        recommendations = (
            "Prioritize deeper composition analysis: B−C changes direction with target or seed, common-source learning varies across B/C compositions, and some tasks are learned alone but degrade in mixtures."
            "First hold task count, target sample size, and total budget fixed while controlling matching, per-task exposures, and optimization trajectories around common-source learning differences; then add a few more independent target families."
        )
    elif classcounts["multi_degradation"] == 0 and classcounts["single_not_mastered"] == 0:
        recommendations = (
            "Prioritize a few independent task families: current sources show no learning degradation under the predefined threshold, either alone or in mixtures. Test whether specific target statistics and transformation relations determine transfer direction."
            "Keep task count and budget fixed with the same development-selection protocol, avoiding confounding task expansion with budget changes."
        )
    else:
        recommendations = (
            "Prioritize deeper reference and matching analysis for composition learning: incomplete single-task learning and mixture degradation currently obstruct attribution of transfer differences to mathematical relations."
            "First control per-task exposures and optimization trajectories and examine better-matched replacements. If effects still depend on the target after controlling these differences, add independent families."
        )
    recommendations += (
        "These are research priorities based on this round's observations, not significance tests. All four targets remain in PermWorld, and fixed_points/exceedances both concern position comparisons."
        "Increasing source-task count is not yet a priority, to avoid changing task count, exposure, and composition simultaneously."
    )
    report_text = f"""# Pilot v3: negative transfer and task-composition diagnostics across multiple targets

This round completed a fixed-scale comparison of four targets, 256 training examples per target, and three paired seeds (17, 42, 73). Main results use 36 development-selected A/B/C checkpoints; all 216 test results from best and endpoint checkpoints of 108 learning-rate candidates are retained. All choices used mathematical definitions, training data, and development data only, followed by unified evaluation on an independent new test set generated after selections froze.

Mean B−A is negative for {negatives['B-A']}/4 targets, with all three seeds negative for {allnegative['B-A']}/4; the corresponding C−A counts are {negatives['C-A']}/4 and {allnegative['C-A']}/4. B−C changes sign across seeds for {bcswitch}/4 targets. These are observed differences, not strong statistical inference from three seeds or a direct interpretation of B−C as the pure causal effect of a mathematical relation.

## Main results

Accuracies are means over three seeds. Differences are in percentage points; ± denotes the sample standard deviation of per-seed differences, not a confidence interval.

{mdtable(['Target','A','B','C','B−A mean±SD','C−A mean±SD','B−C mean±SD'], primary)}

![Per-seed accuracy for each target]({link('figures/target_accuracy_by_seed.png')})

![Paired differences and descriptive summaries]({link('figures/paired_deltas_summary.png')})

Counts of negative / zero / positive seeds follow; no-benefit and opposite-direction results are retained.

{mdtable(['Target','B−A negative/zero/positive','C−A negative/zero/positive','B−C negative/zero/positive'], recurrence)}

{observed_detail}

Raw main results for each target and seed follow ([CSV]({link('results_selected.csv')})). Test majority baselines use the most frequent answer among that seed's 256 target training examples, not the test-set mode.

{mdtable(['Target','seed','Condition','New-test accuracy','Dev accuracy','Selected LR','Selected step','Majority baseline','Training provenance'], detailed)}

## Task catalog, target freezing, and composition

`TASK_CATALOG.csv` inventories {catalog['catalog_rows']} registered/historical interface records, including {catalog['permworld_registered_endpoints']} PermWorld endpoints, 32 scalar property tasks, and 4 output-encoding conversion endpoints. Catalog rows are not independent mathematical task counts: input lengths and parameter values are parameter variants; one-line/cycle/Lehmer/inversion-vector are encoding variants of the same mathematical object, with conversions labeled separately. GraphWorld, MatrixGroupWorld, and historical finite-field tasks are inventoried only, not run. Relation checks cover all 5913 permutations of n=1…7 plus 2048 random n=10 permutations. Finite checks verify implementation; identities follow from definitions.

The four targets and compositions follow. In every row, target(p)=related_source(T(p)), with identity answer mapping; the three common B/C tasks remain fixed. Inverse, reverse, and complement cover three input transformations. Fixed points and exceedances share the broader positional-comparison family, so task diversity must not be overstated.

All four B relations have identity answer mappings. Nontrivial answer mappings and multitask additive identities are not covered. The catalog includes such other relations, but this round excludes them under learnability and limited-scale constraints.

{mdtable(['Target','Definition','Related source in B','Input transform T','Replacement source in C','Three common sources'], group_table)}

The plan froze before single-task development experiments. Target eligibility requires seed17 single-task 4800-step development accuracy≥90%, at least 15 percentage points above the training-mode development baseline. Transfer test results were not used to select targets.

{mdtable(['Target','Single-task dev accuracy','Training-mode dev baseline','Passed predefined gate'], gates)}

Replacements minimize the sum of three terms within predefined candidate sets using seed17: source-training answer-histogram total variation TV, the absolute 4800-step single-task development-accuracy difference, and the mean absolute development-accuracy difference at steps 600/1200/2400/3600/4800. Ties follow candidate order. Fixed_points retains successions as a historical anchor, and exceedances also has only one predefined candidate; neither is improved through multiple-candidate matching. All lengths are 10, so length distributions match exactly. Remaining difficulty and answer-distribution differences are explicitly retained.

{mdtable(['Target','Related source','Replacement candidate','TV','Absolute endpoint accuracy gap','Mean absolute curve gap','Total score','Selected'], match_table)}

{candidate_reference_text}

C means only that the designated direct input-transformation/answer-mapping relation is not annotated for this experiment; it never means complete mathematical unrelatedness. The catalog retains other known relations. Counterexamples refute only the tested direct identities, not all more complex relations. Similar answer distributions do not imply similar learning processes.

## Single-task and mixture learning references

Every actual source task has single-task references for three seeds, sharing each seed's initialization, 8192-example source training pool, and 1024-example development pool with mixture training. Length-stratified records cover only n=10; no length extrapolation is claimed. Full train/dev accuracy, answer CE, EOS CE, two-token CE, majority baselines, and learning curves are in the [single/multitask comparison]({link('single_vs_multi_sources.csv')}) and [complete source curves]({link('source_learning_all.csv')}).

New single- and multitask runs record full metrics at steps 0/600/1200/2400/3600/4800. Reused fixed_points source models for seeds17/42 lack full train/dev curves before 2400; only the original 2400-step endpoints and v2 continuation records at 3000/3600/4200/4800 remain. Their B/C curves start at 2400, without interpolated or fabricated earlier data. New single-task curves do not fill these historical multitask gaps.

Single-task 4800-step training exposes its task 307200 times, 4 times the 76800 per-task exposures in four-task 4800-step training. A1200-step single-task reference with 76800 exposures is also saved. The single-task 1200 learning-rate trajectory/final rate differs from mixture 4800, and distinct visited inputs may differ. Thus the following classes are descriptive diagnostics, not identification of pure task interference: single-task 4800-step development accuracy<90% is labeled "Single task not fully learned under this configuration"; ≥90% with mixture accuracy at least 5 percentage points lower is "Learned alone but degraded in multitask training"; all others are "Insufficient evidence of multitask degradation".

Across 96 target×condition×seed×source-task occurrences, the three class counts are {classcounts['single_not_mastered']}, {classcounts['multi_degradation']}, {classcounts['insufficient_degradation_evidence']}. Sources repeat across compositions, so these are not 96 independent tasks; the first class also does not establish intrinsic learning difficulty.

These classes depend on single-task learning and do not separately identify mixture improvements: a task not mastered alone but improving by at least 5 percentage points in a mixture remains in the first class. We therefore separately count both directions of mixture 4800 minus single-task 4800 development accuracy. The±5-point threshold is descriptive only and changes neither the frozen protocol nor the classes above. Counts include all logical task occurrences and a version deduplicated by seed×task×source endpoint-state hash. Deduplicated records still share tasks, inputs, initializations, and compositions, so they are not independent samples.

{mdtable(['Counting scope','Records','Improvement≥5pp','Decline≥5pp','Absolute difference<5pp'], effect_counts_table)}

The three largest improvements and three largest declines follow, or all entries if a direction has fewer than three. Logical runs sharing a state appear in the same row. See the full deduplicated table in [source_learning_effects_unique.csv]({link('source_learning_effects_unique.csv')}). These are descriptive source-development examples, not test-based selection.

{effects_examples}

Mixture improvements and declines may depend on seed and optimization. Single-task 4800 and mixture 4800 share total optimizer updates and global learning-rate trajectories, but single-task per-task exposure is 4 times higher and mixtures introduce task alternation. These differences prevent isolating pure interference. Single-task 1200 matches only per-task exposure, with different learning-rate history and total updates. Improvements do not directly prove causal synergy, nor do declines directly prove pure task interference. Both directions are retained so that failure to learn alone is not mistaken for failure in every mixture.

The following values average three seeds. Raw tables retain per-seed and training-set metrics.

{mdtable(['Target','Condition','Source task','Single 1200 dev','Single 4800 dev','Multi 4800 dev','Multi dev CE','Multi dev majority baseline'], source_table)}

The mean absolute B/C development-accuracy difference for common tasks is {100 * mean(common_abs):.2f} percentage points, with maximum {100 * max(common_abs):.2f} percentage points across 36 task×seed×target occurrences. Common-task learning differences are retained as composition-training results; changing only the replaced task does not imply equal representations or learning states for other tasks.

![Common-source differences]({link('figures/common_source_differences.png')})

The 24 logical pretraining runs correspond to only {composition['unique_ordered_compositions']} distinct ordered source compositions and {composition['unique_unordered_compositions']} unordered sets. Across three seeds there are {composition['unique_model_states']} distinct endpoint-state hashes and {composition['duplicate_state_groups']} duplicate-hash groups ({composition['duplicate_state_extra_runs']} extra duplicate runs). Equal seeds, inputs, and ordered compositions can produce identical weights. Equal unordered sets with different slot orders need not be bitwise identical. The engine still executes every logical run, and new compute counts actual execution; duplicate states are not independent source-model evidence. See the [composition/state deduplication audit]({link('source_composition_reuse.csv')}).

## Training, adaptation, and testing protocol

The original PermWorld one-line tokenizer/vocabulary and Transformer are retained (4 layers,width 256,8 heads,dropout 0.1, tied embedding/output weights). All four targets are excluded from every pretraining source set. Source train/dev pools contain 8192/1024 inputs; target train/dev/new-test pools contain 256/1024/2048. Within each seed, targets and A/B/C reuse the same permutations with labels generated by their respective definitions. Seeds have distinct pools; shared inputs across targets mean cross-target results are not independent samples.

B/C each use four tasks,batch 64 with exactly 16 examples per task, and 4800 updates. Learning rate warms up for 100 steps to 3e−4, decays by cosine to 3e−5 at 2400, and stays 3e−5 until 4800; no automatic further extension is performed. AdamW(weight_decay=.01, betas=.9/.999, eps=1e−8), gradient clipping 1, FP32 weights/evaluation, BF16 training, TF32 disabled. Training loss weights answer-token and EOS cross-entropy equally. Accuracy is answer argmax over the full 188-token vocabulary given the input/task prompt; EOS is reported separately.

Each A/B/C has the same three rates (3e−4,1e−4,3e−5),800 steps each, and identical sampling order. After 100 warmup steps, cosine decay reaches 0.1 of the initial rate. Full train/dev metrics are computed every 100 steps. Selection maximizes development answer accuracy, then minimizes answer CE, favors earlier updates, and breaks remaining ties by declared rate order; step0 is ineligible. The existing target-specific tied token/output row is restored to same-seed step0, optimizer state is emptied, and source parameters are loaded strictly. Selected-rate counts are {dict(sorted(selected_lrs.items(), reverse=True))}. In the fixed 400→800 candidate window, development CE rises in {ce_worse}/108 and development accuracy declines in {acc_worse}/108. The endpoint gap after best-development selection is not treated as independent proof of overfitting.

`DESIGN_FROZEN.json` freezes design, code, and data; `FROZEN_PROTOCOL.json` freezes targets/compositions; `SELECTIONS_FROZEN.json` freezes 36 development selections and weight hashes. Only afterward is a new test set generated beyond round two: all v1/v2 inputs, v3 train/dev inputs, and their complete inverse/reverse/complement orbits are excluded, and new-test seeds are mutually orbit-isolated. Audited overlap is 0. The new test does not select targets, replacements, rates, or checkpoints. All 108 candidates' best/endpoints are evaluated together after freezing; disclosing 216 rows introduces no further selection.

For fixed_points seeds17/42,4 v2 double-budget pretrained weights and 18 adaptation candidates are reused exactly and reevaluated on this round's new test after file-hash and configuration/data compatibility checks. These retain historical exploratory status and are not independent retrained repetitions. Seed73 and the other three targets are newly trained. {preservation_text} Full final audit: {audit_text}.

## Compute costs and limitations

Training timers in the table total {new_seconds/60:.2f} new minutes; stage wall times total {stage_seconds/60:.2f} minutes. Training timers include full evaluations, saving, and some bookkeeping. Historical reused timers have slightly different scopes and are not counted as new compute. Stage times additionally include labeling/isolation checks, selection, and 216 test evaluations, and cannot be added to training timers. Summary plotting and final audits run separately and are excluded from engine-stage wall time. See per-row timing scopes in `compute_by_run.csv` and stage times in `stage_timings.csv`.

{mdtable(['Type','Cost provenance','Runs','Updates','Input tokens','Supervised tokens','Training timer minutes'], cost_table)}

Each new mixture run processes 8,601,600 input tokens and 614,400 supervised tokens, with 25% sampling per source. Each adaptation candidate processes 1,433,600 input tokens and 102,400 supervised tokens. Input tokens count the full 28-token forward training sequence, including answer/EOS; supervised tokens count the two loss positions. Timers include full evaluation and saving, but these training-token counts exclude evaluation tokens.

This round compares transfer only at 256 examples, 4800 pretraining steps, and 800 adaptation steps; it establishes no full sample-efficiency curve or scaling law. It has only three paired seeds, four learnability-screened PermWorld targets, and one length/model/encoding; design choices use seed17 development records. Best development values across rate candidates have selection bias. Test evaluation is frozen, but multiple-candidate results are disclosure only and must not select a new best main conclusion. Without a cross-budget sweep, effects are conditional on v2's specific extended-budget rate trajectory, not budget-independent. Answer distributions, known transformation relations, source learning, joint-training effects, and initialization may jointly explain B/C differences.

One concrete confound is exceedances C replacing deficiencies with recoils while descents remains common, introducing the known source–source relation recoils(I(p))=descents(p) (the catalog audit's descent_inverse relation). B/C change both internal source-set relations and the designated target–source relation; differences cannot be attributed solely to the latter.

## Next-stage recommendation

{recommendations}

Expansion stops here pending further guidance.

## Files and reproduction

- [All candidate raw results for three seeds]({link('results_seed17.csv')}), [seed42]({link('results_seed42.csv')}), [seed73]({link('results_seed73.csv')}); [all 216 rows]({link('results_all_candidates.csv')}), [36 main results]({link('results_selected.csv')}).
- [Paired differences]({link('paired_deltas.csv')}), [difference summaries]({link('paired_deltas_summary.csv')}), [endpoint versus development selection]({link('selected_vs_endpoint.csv')}); all means and sample SDs are in [results_summary.csv]({link('results_summary.csv')}).
- [Single/multitask comparison]({link('single_vs_multi_sources.csv')}), [common-source differences]({link('common_source_differences.csv')}), [candidate matching differences]({link('replacement_matching.csv')}).
- [All target train/dev curves]({link('target_curves.csv')}); figures are in `{linkbase}/figures/target_selected_*.png` and `source_learning_*.png`; original per-run curves/weight hashes are in `adapt/`, `single/`, and `multi/`.
- [Frozen configuration]({link('FROZEN_PROTOCOL.json')}), [compute costs]({link('compute_summary.csv')}), [stage timings]({link('stage_timings.csv')}), [audit]({link('FINAL_AUDIT.json')}); see the experiment directory's `README.md` for the task catalog and reproduction commands.
"""
    report_path.write_text(report_text, encoding="utf-8")
    key_results = dict(selected_rows=len(selected), all_candidate_rows=216, n_seeds=3, target_samples=256,
                       targets=cfg["targets"], negative_mean_targets=negatives, all_negative_seed_targets=allnegative,
                       composition_sign_switch_targets=bcswitch, source_classification_counts=dict(classcounts),
                       source_composition_reuse=composition,
                       source_learning_effect_counts=effects["counts"],
                       new_training_timer_seconds=new_seconds, recorded_engine_stage_seconds=stage_seconds,
                       report=str(report_path), no_test_based_selection=True)
    (out / "key_results.json").write_text(json.dumps(key_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    assert (out / "TEST_COMPLETE.json").exists(), "Experiment incomplete; no completed-results report generated."
    protocol, selection = js(out / "FROZEN_PROTOCOL.json"), js(out / "SELECTIONS_FROZEN.json")
    completion = js(out / "TEST_COMPLETE.json")
    assert completion["candidate_rows"] == 216 and completion["selected_rows"] == 36
    selected, candidates = rows(out / "results_selected.csv"), rows(out / "results_all_candidates.csv")
    assert len(selected) == 36 and len(candidates) == 216 and sum(truth(r["selected_primary"]) for r in candidates) == 36
    assert all(truth(r["selected_primary"]) for r in selected)
    (out / "figures").mkdir(exist_ok=True)
    summary, deltas, delta_summary, endpoints = result_tables(out, protocol["config"], selected, candidates)
    result_plots(out, protocol["config"], selected, summary, deltas, delta_summary)
    source_rows, metas, comparisons, common, classifications, learning_summary = source_tables(out, protocol)
    effects = source_effects(out, comparisons, metas)
    source_plots(out, protocol, source_rows, common)
    late = target_curves(out, protocol["config"], selection["selections"])
    run_costs, compute, timings = costs(out, metas)
    composition = composition_reuse(out, metas)
    report_path = HERE / "PILOT_V3_REPORT.md" if out == DEFAULT_OUT.resolve() else out / "PILOT_V3_REPORT.md"
    report(out, report_path, protocol, selected, summary, delta_summary, comparisons, common, learning_summary, compute, timings, late, composition, effects)
    print(json.dumps(dict(status="complete", report=str(report_path), figures=len(list((out / "figures").glob("*.png"))), selected_rows=len(selected)), ensure_ascii=False))


if __name__ == "__main__":
    main()
