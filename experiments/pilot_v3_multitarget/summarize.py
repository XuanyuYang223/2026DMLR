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
    "single_not_mastered": "本配置单任务未充分学会",
    "multi_degradation": "单任务能学会、组合训练下降",
    "insufficient_degradation_evidence": "不足以认定组合训练下降",
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
        f"最终审计status=passed，已核对v1/v2代码、配置、数据、结果和checkpoint等{audit['old_files_byte_preserved']}个历史文件的字节hash，保留检查通过。"
        if preservation_passed else
        "历史文件字节保留仍待最终审计确认；FINAL_AUDIT.json缺失、状态非passed或缺少已保留文件数量，不能在本报告中宣称保留检查通过。"
    )
    audit_text = f"{audit.get('status', '缺失status字段')}，见 FINAL_AUDIT.json" if audit else "尚未生成；运行 audit_final.py 后再运行本脚本更新报告"
    linkbase = os.path.relpath(out, report_path.parent).replace("\\", "/")
    def link(file):
        return f"{linkbase}/{file}"
    primary = []
    for target in cfg["targets"]:
        records = {r["condition"]: r for r in summary if r["target"] == target}
        contrasts = {r["contrast"]: r for r in delta_summary if r["target"] == target}
        primary.append([target, *[pct(records[c]["test_accuracy_mean"]) for c in "ABC"],
                        *[f"{pp(contrasts[c]['delta_mean'])} ± {100 * contrasts[c]['delta_sample_sd']:.2f}" for c in ("B-A", "C-A", "B-C")]])
    detailed = [[r["target"], r["seed"], r["condition"], pct(r["test_accuracy"]), pct(r["dev_accuracy"]), r["lr"], r["adapt_step"], pct(r["majority_accuracy"]), "历史权重复用" if not truth(r["new_compute"]) else "本轮训练"] for r in selected]
    recurrence = []
    for target in cfg["targets"]:
        records = {r["contrast"]: r for r in delta_summary if r["target"] == target}
        recurrence.append([target, *[f"{records[c]['negative_seeds']}/{records[c]['zero_seeds']}/{records[c]['positive_seeds']}" for c in ("B-A", "C-A", "B-C")]])
    relation_defs = {
        "fixed_points": "固定点数 # {i:p_i=i}", "exceedances": "超越数 # {i:p_i>i}",
        "left_to_right_minima": "从左至右严格新最小值的个数", "longest_decreasing_run": "最长连续严格下降段长度（至少1）",
    }
    transforms = {"complement": "C(p)_i=n+1-p_i", "inverse": "I(p)=p⁻¹", "reverse": "R(p)_i=p_{n+1-i}"}
    group_table = [[g["target"], relation_defs[g["target"]], g["related"], transforms[g["transform"]], g["replacement"], ", ".join(g["sources"]["B"][1:])] for g in groups]
    gates = [[r["target"], pct(r["accuracy"]), pct(r["majority_accuracy"]), r["pass_gate"]] for r in protocol["target_gates"]]
    match_table = [[r["target"], r["related"], r["replacement"], f"{num(r, 'tv'):.4f}", f"{num(r, 'endpoint_dev_accuracy_gap'):.4f}", f"{num(r, 'mean_curve_accuracy_gap'):.4f}", f"{num(r, 'score'):.4f}", r["selected"]] for r in matching]
    used_sources = {task for group in groups for tasklist in group["sources"].values() for task in tasklist}
    unused_candidates = {r["replacement"] for r in matching} - used_sources
    unused_refs = [r for r in rows(out / "single_task_references.csv") if integer(r, "step") == 4800 and r["task"] in unused_candidates]
    candidate_reference_text = (
        "未纳入最终源集合的候选也保留单任务记录。下表由实际4800步终点生成，开发准确率<90%仅表示本配置未充分学会，不表示任务固有困难；未选候选通常只有设计seed17参照，不补齐三个种子。\n\n" +
        mdtable(["未选候选", "seed", "单任务train", "单任务dev", "开发多数基线", "90%描述性门槛"],
                [[r["task"], r["seed"], pct(r["train_accuracy"]), pct(r["dev_accuracy"]), pct(r["dev_majority_accuracy"]), "未充分学会" if num(r, "dev_accuracy") < .90 else "达到"] for r in unused_refs])
        if unused_refs else "本次没有已训练但未纳入任何最终源集合的替换候选；全部候选单任务记录仍保留在 single_task_references.csv。"
    )
    source_table = [[r["target"], r["condition"], r["task"], pct(r["single1200_dev_accuracy_mean"]), pct(r["single4800_dev_accuracy_mean"]), pct(r["multi4800_dev_accuracy_mean"]), f"{r['multi4800_dev_answer_ce_mean']:.3f}", pct(r["multi4800_dev_majority_accuracy_mean"])] for r in learning_summary]
    classcounts = Counter(r["classification"] for r in comparisons)
    effect_counts_table = [["原始逻辑任务出现次数" if r["scope"] == "logical_task_occurrences" else "seed×任务×source state去重",
                           r["n_records"], r["improved_at_least_5pp"], r["decreased_at_least_5pp"], r["within_5pp"]] for r in effects["counts"]]
    improvements = sorted([r for r in effects["rows"] if r["multi_minus_single4800_dev_accuracy"] >= .05],
                          key=lambda r: r["multi_minus_single4800_dev_accuracy"], reverse=True)[:3]
    decreases = sorted([r for r in effects["rows"] if r["multi_minus_single4800_dev_accuracy"] <= -.05],
                       key=lambda r: r["multi_minus_single4800_dev_accuracy"])[:3]
    extreme_effects = [("改善", r) for r in improvements] + [("下降", r) for r in decreases]
    effects_examples = (mdtable(["方向", "seed", "源任务", "单4800 dev", "组合4800 dev", "差值pp", "出现的逻辑run"],
                               [[sign, r["seed"], r["task"], pct(r["single4800_dev_accuracy"]), pct(r["multi4800_dev_accuracy"]),
                                 pp(r["multi_minus_single4800_dev_accuracy"]), r["logical_runs"]] for sign, r in extreme_effects])
                        if extreme_effects else "没有source记录达到±5个百分点的描述性差异门槛。")
    common_abs = [abs(r["B_minus_C_dev_accuracy"]) for r in common]
    negatives = {contrast: sum(r["contrast"] == contrast and r["delta_mean"] < 0 for r in delta_summary) for contrast in ("B-A", "C-A", "B-C")}
    allnegative = {contrast: sum(r["contrast"] == contrast and r["negative_seeds"] == 3 for r in delta_summary) for contrast in ("B-A", "C-A", "B-C")}
    bcswitch = sum(r["contrast"] == "B-C" and r["negative_seeds"] > 0 and r["positive_seeds"] > 0 for r in delta_summary)
    new_seconds = sum(r["seconds"] for r in compute if r["new_compute"])
    stage_seconds = sum(r["seconds"] for r in timings)
    cost_table = [[r["kind"], "新增" if r["new_compute"] else "历史复用", r["runs"], f"{r['updates']:,}", f"{r['input_tokens']:,}", f"{r['supervised_tokens']:,}", f"{r['seconds']/60:.2f}"] for r in compute]
    ce_worse = sum(r["dev_answer_ce_delta_800_minus_400"] > 0 for r in late)
    acc_worse = sum(r["dev_accuracy_delta_800_minus_400"] < 0 for r in late)
    selected_lrs = Counter(float(r["lr"]) for r in selected)
    unanimous_negative = [target for target in cfg["targets"] if all(
        next(r["negative_seeds"] for r in delta_summary if r["target"] == target and r["contrast"] == contrast) == len(cfg["seeds"])
        for contrast in ("B-A", "C-A"))]
    observed_detail = ("本轮观察到B−A和C−A都在三个种子中为负的目标为 " +
                       ("、".join(unanimous_negative) if unanimous_negative else "无") +
                       "；这描述三个种子的实际方向，不代表统计显著性或未见任务上的普遍负迁移。")
    if "longest_decreasing_run" in cfg["targets"]:
        ldr = {(integer(r, "seed"), r["condition"]): r for r in selected if r["target"] == "longest_decreasing_run"}
        seeds = cfg["seeds"]
        offsets = {c: [num(ldr[seed, c], "test_accuracy") - num(ldr[seed, c], "majority_accuracy") for seed in seeds] for c in "AC"}
        gains = {seed: num(ldr[seed, "B"], "test_accuracy") - num(ldr[seed, "A"], "test_accuracy") for seed in seeds}
        positive_seeds = [seed for seed in seeds if gains[seed] > 0]
        other_seeds = [seed for seed in seeds if gains[seed] <= 0]
        observed_detail += (
            " longest_decreasing_run 的A/C测试表现接近训练众数基线：逐种子A减基线为 " +
            "、".join(pp(value) for value in offsets["A"]) + "，C减基线为 " +
            "、".join(pp(value) for value in offsets["C"]) + " 个百分点（按seed " +
            "/".join(map(str, seeds)) + "排序）。B相对A收益出现在seed " +
            "/".join(map(str, positive_seeds)) + "，没有收益的seed为 " +
            ("/".join(map(str, other_seeds)) or "无") + "；对应B−A为 " +
            "、".join(pp(gains[seed]) for seed in seeds) + " 个百分点。"
        )
        related = next(g["related"] for g in groups if g["target"] == "longest_decreasing_run")
        related_dev = [next(num(r, "multi4800_dev_accuracy") for r in comparisons if r["target"] == "longest_decreasing_run" and r["seed"] == seed and r["condition"] == "B" and r["task"] == related) for seed in seeds]
        observed_detail += (
            f" B的相关源{related}开发准确率依次为 " + "、".join(pct(value) for value in related_dev) +
            "，显示源学习状态与目标收益同时随seed变化；seed也同时改变输入池与初始化，三个配对观察不足以认定因果来源。"
        )
    observed_detail += " 目标资格门槛是在8192条单任务训练样本下通过，不意味着A在256条样本及800步适配预算下也能学会；因此目标资格与本轮小样本学习失败需分别解释。"
    bc_means = [r["delta_mean"] for r in delta_summary if r["contrast"] == "B-C"]
    composition_variable = bcswitch > 0 or (min(bc_means) < 0 < max(bc_means))
    if composition_variable and any(common_abs) and classcounts["multi_degradation"]:
        recommendations = (
            "优先深化组合分析：B−C的方向随具体目标或seed改变，公共源任务的学习状态也随B/C组合变化，同时存在单任务学会而组合下降的记录。"
            "先保持任务数、目标样本数和总预算，围绕公共源任务学习差异做匹配/每任务暴露量/优化轨迹控制，再增加少量更独立的目标任务族。"
        )
    elif classcounts["multi_degradation"] == 0 and classcounts["single_not_mastered"] == 0:
        recommendations = (
            "优先增加少量独立任务族：当前源任务单独及组合均未显示预设门槛下的学习下降，应检验目标具体统计及变换关系是否决定迁移方向。"
            "继续固定任务数和预算，保留相同开发选择协议，避免把任务扩展与预算变化混在一起。"
        )
    else:
        recommendations = (
            "优先深化组合学习参照与匹配分析：当前单任务未充分学会或组合下降的记录，使迁移差异难以归因于数学关系。"
            "先控制每任务暴露量和优化轨迹，并检查更匹配的替换项；若这些差异得到控制后效果仍依赖具体目标，再增加独立任务族。"
        )
    recommendations += (
        "上述是基于本轮观察的研究优先级，不是显著性检验。四个目标仍在PermWorld，fixed_points与exceedances同属位置比较类；"
        "暂不优先扩大源任务数量，以免同时改变任务数量、暴露量和组成。"
    )
    report_text = f"""# Pilot v3：多个目标上的负迁移与任务组成诊断

本轮已完成四个目标、每目标256个训练样本、三个配对种子（17、42、73）的固定规模比较。主结果为开发集选定的36个 A/B/C checkpoint；全部108个学习率候选的 best 与 endpoint 共216条测试结果均保留。所有方案只依据数学定义、训练集和开发集选定，统一在冻结选择后生成的独立新测试上评估。

本轮 B−A 的均值为负出现在 {negatives['B-A']}/4 个目标，其中 {allnegative['B-A']}/4 个目标三个种子均为负；C−A 相应为 {negatives['C-A']}/4 和 {allnegative['C-A']}/4。B−C 在 {bcswitch}/4 个目标上随种子改变符号。这里报告观察到的差异，不用三个种子作强统计推断，也不把 B−C 直接解释为某条数学关系的纯因果效应。

## 主要结果

准确率为三个种子的均值；差值单位为百分点，± 为逐种子差值的样本标准差（不是置信区间）。

{mdtable(['目标','A','B','C','B−A 均值±SD','C−A 均值±SD','B−C 均值±SD'], primary)}

![每个目标的逐种子准确率]({link('figures/target_accuracy_by_seed.png')})

![配对差值及描述性汇总]({link('figures/paired_deltas_summary.png')})

负 / 零 / 正的种子个数如下；无收益和方向相反的结果都保留。

{mdtable(['目标','B−A 负/零/正','C−A 负/零/正','B−C 负/零/正'], recurrence)}

{observed_detail}

每个目标、每个种子的原始主结果如下（[CSV]({link('results_selected.csv')})）。测试集的多数基线用该种子256条目标训练数据中频率最高的答案构造，不用测试集众数。

{mdtable(['目标','seed','条件','新测试准确率','开发准确率','所选LR','所选步数','多数基线','训练来源'], detailed)}

## 任务目录、目标冻结与组成

`TASK_CATALOG.csv` 盘点了 {catalog['catalog_rows']} 条当前注册/历史接口记录，其中 PermWorld 有 {catalog['permworld_registered_endpoints']} 个端点、32个标量 property 任务和4个输出编码转换端点。目录行数不是独立数学任务数：输入长度和参数取值是参数变体，one-line/cycle/Lehmer/inversion-vector 是同一数学对象的编码变体，编码转换单独标注。GraphWorld、MatrixGroupWorld 和历史有限域内容只作目录盘点，本轮没有运行。关系检查覆盖 n=1…7 的全部5913个置换以及2048个 n=10 随机置换；有限检查验证实现，恒等关系依据定义成立。

四个目标及组成如下。对每一行，目标(p)=相关源任务(T(p))，答案映射均为恒等；B/C 共有的三个任务固定。反演、反序、补值覆盖三类输入变换。固定点与超越数同属较宽的位置比较任务族，不能夸大任务多样性。

四条B关系的答案映射全为恒等；没有覆盖非平凡答案映射或多任务加和恒等式。目录收录了这些其他关系，本轮按可学习性和有限规模约束未纳入。

{mdtable(['目标','定义','B的相关源','输入变换T','C的替换源','三个公共源'], group_table)}

计划在单任务开发实验前冻结。目标入选门槛为 seed17 的单任务4800步开发准确率≥90%，且至少比训练众数开发基线高15个百分点；没有使用迁移测试结果挑目标。

{mdtable(['目标','单任务开发准确率','训练众数开发基线','预设门槛通过'], gates)}

替换任务在预先声明的候选集合内，依据 seed17 源训练答案直方图总变差 TV、单任务4800步开发准确率差、600/1200/2400/3600/4800步开发准确率差的均值三项相加选最小值；并列按候选顺序。fixed_points 保留原 successions 替换作为历史锚点，exceedances 也只有一个预设候选，故这两项没有通过多候选优化改善匹配。所有输入长度均为10，长度分布完全相同，剩余难度和答案分布差异明确保留。

{mdtable(['目标','相关源','替换候选','TV','终点准确率差绝对值','曲线差绝对值均值','总分','选中'], match_table)}

{candidate_reference_text}

C只表示没有被标注为具有本实验对应的直接输入变换/答案映射关系，绝不表示数学上完全无关。目录保存了其他已知关系；反例检查只否定检查过的直接恒等式，不证明不存在更复杂关系。答案分布相近也不意味着学习过程相近。

## 单任务与组合学习参照

所有实际源任务均有三个种子的单任务参照，使用与组合训练相同的每种子初始化、8192条源训练池和1024条开发池。各任务按长度的记录只有 n=10；不声称长度外推。全量 train/dev 准确率、answer CE、EOS CE、双token CE、多数基线和学习曲线保留在 [单/多任务对照]({link('single_vs_multi_sources.csv')}) 与 [完整源曲线]({link('source_learning_all.csv')})。

新训练的单任务及多任务记录0/600/1200/2400/3600/4800步全量指标；但fixed_points的seed17/42复用源模型缺少2400步之前的全量train/dev曲线，仅保留2400步原预算终点和v2续训3000/3600/4200/4800步记录。对应B/C图线从2400步开始，早期空缺未插值或补造；本轮新增单任务曲线不填补这些历史多任务记录。

单任务4800步每任务暴露307200次，是四任务4800步中单个任务76800次暴露的4倍；另保存单任务1200步76800次暴露参照。单任务1200步的学习率轨迹/最终学习率与组合4800步不同，且看到的不同输入数可能不同。因此下列分类是描述性诊断，不能单独识别纯任务干扰效应：单任务4800步开发准确率<90%记为“本配置单任务未充分学会”；≥90%且组合低至少5个百分点记为“单任务能学会、组合训练下降”；其余为“不足以认定组合训练下降”。

按目标×条件×种子×源任务共96个出现次数，三类分别为 {classcounts['single_not_mastered']}、{classcounts['multi_degradation']}、{classcounts['insufficient_degradation_evidence']}。同一源任务在不同组合中重复出现，这不是96个独立任务；第一类也不是任务具有内在学习困难的结论。

这三个类别按单任务学习程度定义，不能涵盖组合改善：单任务未充分学会而组合提高至少5个百分点的记录仍落在第一类。因此另行按组合4800减单任务4800的开发准确率统计正负两方向，±5个百分点仅作描述门槛，不改变冻结协议或上述分类。既报告全部逻辑任务出现次数，也按seed×任务×source终点state hash去掉完全相同模型的重复；去重后仍共享任务、输入、初始化和源组成，仍不是独立样本。

{mdtable(['计数范围','记录数','改善≥5pp','下降≥5pp','差值绝对值<5pp'], effect_counts_table)}

下面分别保留最大三个改善和最大三个下降记录（若某方向不足三个则全部列出），相同state重复出现的逻辑run放在同一行。完整去重表见 [source_learning_effects_unique.csv]({link('source_learning_effects_unique.csv')})。这些只是源开发指标的描述性例子，未用于测试选择。

{effects_examples}

组合改善和下降都可能依赖具体seed及优化过程。单任务4800步和组合4800步具有相同的总优化器更新数及全局学习率轨迹，但单任务的每任务暴露为组合的4倍，组合还引入任务交替；这些差别使纯任务干扰仍无法被单独识别。单任务1200步只匹配每任务暴露，学习率历史与总更新数不同。故改善不能直接证明因果协同，下降也不能直接证明纯任务干扰。不同方向共同保留，避免把“单任务未学会”误读为“所有组合也未学会”。

以下为三个种子均值。原始表保留每个种子与训练集指标。

{mdtable(['目标','条件','源任务','单1200 dev','单4800 dev','组合4800 dev','组合dev CE','组合dev多数基线'], source_table)}

公共任务 B/C 开发准确率差的绝对值平均为 {100 * mean(common_abs):.2f} 个百分点，最大为 {100 * max(common_abs):.2f} 个百分点（36个任务×种子×目标出现次数）。公共任务学习差异作为组成训练结果保留；不能假设仅被替换任务不同、其他任务表征/学习状态就相同。

![公共源任务差异]({link('figures/common_source_differences.png')})

另外，24个逻辑预训练run只对应 {composition['unique_ordered_compositions']} 个不同的有序源组合、{composition['unique_unordered_compositions']} 个不同的无序源集合；三个种子合计 {composition['unique_model_states']} 个不同终点模型state hash，{composition['duplicate_state_groups']} 组hash重复（重复额外run数 {composition['duplicate_state_extra_runs']}）。相同seed、输入和有序源组合可以训练出完全一致权重；无序集合相同但槽位顺序不同不保证逐位相同。引擎仍执行各逻辑run，新增计算按实际执行计数，不能把重复状态当独立source模型证据。详见 [组合及state去重审计]({link('source_composition_reuse.csv')})。

## 训练、适配和测试协议

使用原 PermWorld one-line tokenizer/vocabulary、Transformer（4层、宽256、8头、dropout0.1、绑定embedding/output权重）。所有四个目标从所有预训练源集合排除。源池8192、源开发1024；目标训练256、开发1024、新测试2048，每种子在目标间与 A/B/C 间复用相同置换输入、按相应数学定义生成标签。三个种子使用各自输入池；目标间共享输入意味着跨目标结果并非独立样本。

B/C均四任务，batch64每任务精确16，4800次更新，学习率前100步warmup至3e−4、至2400步余弦下降到3e−5，再保持3e−5到4800步；没有继续自动延长。AdamW(weight_decay=.01, betas=.9/.999, eps=1e−8)、梯度裁剪1、FP32权重/评估、BF16训练、TF32关闭。训练损失为答案token和EOS各占一半的交叉熵；准确率为给定输入与任务prompt后的完整188词表argmax答案，EOS单独报告。

每个 A/B/C 拥有相同三档LR（3e−4、1e−4、3e−5）、各800步、相同采样顺序，前100步warmup后余弦到初始LR的0.1。每100步计算全部训练/开发指标；先最大化开发答案准确率，再最小化开发answer CE，再最早更新步，再按声明LR顺序破同分，step0不入选。目标已有专有token/output绑定行恢复为同种子step0值，优化器状态清空，源参数严格加载。选择LR计数为 {dict(sorted(selected_lrs.items(), reverse=True))}。候选400→800步固定窗口中，{ce_worse}/108个开发CE上升，{acc_worse}/108个开发准确率下降；不把经最佳开发点选择后的终点差距直接当作独立过拟合证据。

`DESIGN_FROZEN.json` 冻结设计、源码和数据；`FROZEN_PROTOCOL.json` 冻结目标/组成；`SELECTIONS_FROZEN.json` 冻结36项开发选择和权重hash。随后才生成第二轮之后的新测试：统一排除所有 v1/v2 输入、v3训练/开发输入及其 inverse/reverse/complement 完整变换轨道，并隔离新测试种子间轨道，审计交集为0。新测试不用于选择目标、替换任务、LR或checkpoint。全部108候选的best/endpoint在冻结后统一评估，披露216行不产生追加选择。

fixed_points 的seed17/42精确复用v2两倍预算的4个预训练权重和18个适配候选，文件hash与配置/数据兼容性核对后在本轮新测试上重新评估。这部分保留历史探索性质，不能称作独立重新训练的重复实验；seed73及其余三个目标为本轮新训练。{preservation_text} 完整最终审计：{audit_text}。

## 计算成本与局限

下表训练timer总计新增 {new_seconds/60:.2f} 分钟；阶段墙钟合计 {stage_seconds/60:.2f} 分钟。训练timer包含各自全量评估、保存和部分记账开销，复用历史timer范围略有不同，不将历史复用时间计为本轮新增。阶段时间另包含标签/隔离检查、选择和216次测试等，不能与训练timer相加；汇总绘图和最终审计另行执行，未计入引擎阶段墙钟。计时范围逐行见 `compute_by_run.csv`，阶段时间见 `stage_timings.csv`。

{mdtable(['类型','成本来源','运行数','更新数','输入tokens','监督tokens','训练timer分钟'], cost_table)}

每个新组合run实际输入tokens为8,601,600、监督tokens614,400，每源任务采样比例25%；每适配候选输入tokens1,433,600、监督tokens102,400。输入tokens计完整28-token训练序列的前向输入，包含答案与EOS；监督tokens计两个损失位置。全量评估与保存耗时包含在timer中，但上述训练token计数不包含评估token。

本轮只比较固定256样本、固定4800步预训练和固定800步适配下的迁移表现，不建立完整样本效率曲线或scaling law。仅三个配对种子、四个经过可学习性筛选的PermWorld目标、单一长度/模型/编码；设计选择使用seed17开发记录。学习率候选的最佳开发值有选择偏差，测试保持冻结评估但多候选结果仅供披露；不能在本轮测试上挑选下一份“最佳”主结论。没有跨预算扫描，因此本轮效应以v2延长预算的特定学习率轨迹为条件，不能声称预算无关。相同答案分布、已知变换关系、源任务学习程度、共同训练影响和初始化种子仍可能共同解释B/C差异。

一个具体混杂是exceedances的C用recoils替换deficiencies，而公共源包含descents，因此引入已知源—源关系recoils(I(p))=descents(p)（见目录审计的descent_inverse关系）。B/C同时改变源集合内部的关系结构和指定的目标—源关系，不能把差值只归因于后者。

## 下一阶段建议

{recommendations}

本轮到此停止扩展，等待后续指导。

## 文件与复现

- [三个种子全部候选原始结果]({link('results_seed17.csv')})、[seed42]({link('results_seed42.csv')})、[seed73]({link('results_seed73.csv')})；[216行全集]({link('results_all_candidates.csv')})、[36行主结果]({link('results_selected.csv')})。
- [配对差值]({link('paired_deltas.csv')})、[差值汇总]({link('paired_deltas_summary.csv')})、[终点与开发选择对照]({link('selected_vs_endpoint.csv')})；所有均值与样本SD见 [results_summary.csv]({link('results_summary.csv')})。
- [单/多任务对照]({link('single_vs_multi_sources.csv')})、[公共源任务差异]({link('common_source_differences.csv')})、[候选匹配差异]({link('replacement_matching.csv')})。
- [全部目标train/dev曲线]({link('target_curves.csv')})；图在 `{linkbase}/figures/target_selected_*.png` 和 `source_learning_*.png`，原始每run曲线/权重hash在 `adapt/`、`single/`、`multi/`。
- [冻结配置]({link('FROZEN_PROTOCOL.json')})、[计算成本]({link('compute_summary.csv')})、[阶段耗时]({link('stage_timings.csv')})、[审计]({link('FINAL_AUDIT.json')})；任务目录与复现命令见实验目录 `README.md`。
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
