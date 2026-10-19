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
    body = table(["目标", "背景改变", "seed", "Δ(B−A) pp", "Δ(C−A) pp", "Δ(B−C) pp"],
                 [[r["target"], r["new_background"] + "−g0", r["seed"],
                   *(f"{100 * r[m]:+.4f}" for m in ps.METRICS)] for r in differences])
    summary_table = table(["目标", "背景改变", "Δ(B−A)均值 pp", "Δ(C−A)均值 pp", "Δ(B−C)均值 pp", "交互逐seed范围 pp"],
                          [[target, background + "−g0",
                            *(f"{100 * next(r['mean'] for r in summaries if r['target'] == target and r['new_background'] == background and r['metric'] == m):+.4f}" for m in ps.METRICS),
                            f"[{100 * min(r['delta_B_minus_C'] for r in differences if r['target'] == target and r['new_background'] == background):+.4f}, {100 * max(r['delta_B_minus_C'] for r in differences if r['target'] == target and r['new_background'] == background):+.4f}]"]
                           for target in TARGETS for background, *_ in backgrounds])
    report = f"""# v4 现有结果的逐种子配对重算

本文件只读取既有 `pilot_v4_backgrounds/artifacts/results_selected.csv`，没有新增训练，也不读取 v5 验证结果。旧种子17/42/73仅为探索证据。全部结果使用各自按开发集选出的checkpoint在同一v4新测试池上的准确率；A在各背景共享。

对于每个种子，先计算 `Δ(B−A)=(B新−A)−(B原−A)`、`Δ(C−A)=(C新−A)−(C原−A)`，再计算主交互 `Δ(B−C)=(B新−C新)−(B原−C原)`。共享A使前两项分别等于B与C本身的背景差；交互等于前两项之差。以下数字为百分点，CSV保留准确率比例原值。

{body}

{summary_table}

fixed_points的deficiencies背景相对原背景，B−A平均增加16.59pp、C−A平均增加12.22pp，因此主交互只有+4.36pp；三个种子均为正，但幅度从约+0.34到+11.33pp。不能把B单独的改善全部归于相关源相对替换源的优势。

longest_decreasing_run的recoils背景相对原背景，主交互平均−12.00pp，三个种子中两个为负、一个为正；主要表现为B的变化，而C平均变化较小。另两组探索组合也保留，未据其结果增加验证对比。

这些是直接逐种子相减的结果，不通过两组各自的标准差判断背景差异是否可靠。三个探索块和依据同批结果挑出的对比不能提供独立验证；本轮不对这三块作显著性承诺。下一步固定验证FP的g2−g0正方向、LDR的g1−g0负方向，使用五个新数据/初始化配对单元，统计方案见 `../stats_plan.json`。探索与验证分别报告，不合并成八重复。

![逐种子背景差](exploratory_paired_background_changes.png)

原始重算数据：[exploratory_paired_differences.csv](exploratory_paired_differences.csv)。全部12条交互另与v4旧交互CSV逐项核对，在1e−12精度内一致；该旧表没有参与重算，见[审计](exploratory_statistics_audit.json)。
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
    ldr_exceptions = "；".join(
        f"seed{r['seed']}的LIR源dev变化{pp(r['B_candidate_dev_change'])}pp，目标B变化{pp(r['delta_B_minus_A'])}pp、C变化{pp(r['delta_C_minus_A'])}pp，主交互却为{pp(r['delta_B_minus_C'])}pp"
        for r in ldr if r["B_candidate_dev_change"] < 0 and r["delta_B_minus_C"] > 0)
    component = {(r["target"], r["metric"]): r["mean"] for r in summaries}
    return f"""源学习的测试前独立描述已保存在[SOURCE_LEARNING_NOTES.md](SOURCE_LEARNING_NOTES.md)，其审计记录当时TEST_COMPLETE尚不存在。以下才将源记录与冻结后的目标结果并列，区分证据和解释。

**FP：B/C共同改善再次出现，预指定的正替换交互未复现。** anti_fixed_points的源dev均值由{pct(average_field(af0, 'multi4800_dev_accuracy'))}变为{pct(average_field(af2, 'multi4800_dev_accuracy'))}，两边都已接近99%；{fp_source_target_opposed}/5个重复中该源dev下降但B的目标准确率提高。因此“原背景没有学会相关候选”不能作为这一目标的统一解释，也不能用候选源准确率直接代理目标收益。B−A与C−A的背景改变均值分别为{pp(component['fixed_points', 'delta_B_minus_A'])}/{pp(component['fixed_points', 'delta_C_minus_A'])}pp，五个重复的两项都提高，但后者平均更大；主交互为{pp(component['fixed_points', 'delta_B_minus_C'])}pp。deficiencies在B/C新背景中都学得很高，也都具有跨输入联合恢复FP的公式，这是重要的设计事实；本轮并未证明模型利用该公式，不能把共同收益直接归因于这条路径。

**LDR：源学习下降与B目标收益下降共同出现，但不能据此推出交互机制。** LIR源dev均值由{pct(average_field(lir0, 'multi4800_dev_accuracy'))}变为{pct(average_field(lir1, 'multi4800_dev_accuracy'))}，五次均下降；新背景train均值{pct(average_field(lir1, 'multi4800_train_accuracy'))}、dev多数基线均值{pct(average_field(lir1, 'multi4800_dev_majority_accuracy'))}。训练和开发都低，支持当前预算下源学习不足，而不是仅有训练拟合后的泛化落差。五次B目标准确率也均下降，B−A背景改变均值{pp(component['longest_decreasing_run', 'delta_B_minus_A'])}pp；C−A同时改变{pp(component['longest_decreasing_run', 'delta_C_minus_A'])}pp，所以主交互均值为{pp(component['longest_decreasing_run', 'delta_B_minus_C'])}pp。保留关键反例：{ldr_exceptions}。交互取决于B和C两边，单看B源下降会漏掉它。

**公共源并非一致受损。** LDR的peaks跨两个条件共10条同任务背景差，{sum(value < 0 for _, _, value in peak_changes)}条降低、{sum(value > 0 for _, _, value in peak_changes)}条提高；正向项为{peak_positive}。这也说明v4“peaks全部下降”的源观察在新重复中不逐项成立。公共源以及successions的变化存在seed/条件差异，不能把组合效应描述为所有源同步变差。

已有记录支持上述并列变化、低训练学习水平和反例。它们仍无法区分共享计算结构、优化速度/梯度竞争、答案分布、联合关系利用或适配阶段可塑性等解释；本轮没有中介量干预、梯度实验或显式变换查询，也没有证明任何机制。"""


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
            verdict = "预指定均值方向未复现"
        elif family:
            verdict = "均值方向复现，且模型假设下的两对比同时区间排除0"
        elif individual:
            verdict = "均值方向复现；单项t区间排除0，同时区间仍包含0"
        else:
            verdict = "均值方向复现，但单项和同时区间仍包含0，方向与幅度尚不稳定"
        judgments.append(f"- **{row['target']}**：{verdict}。交互均值{pp(row['mean'])}pp，{row['direction_matching_count']}/5个重复符合预指定方向；95%配对t区间[{pp(row['ci95_lower'])}, {pp(row['ci95_upper'])}]pp，两项Bonferroni同时区间[{pp(row['family95_lower'])}, {pp(row['family95_upper'])}]pp。")
    ps.write_csv(out / "confirmation_vs_exploration.csv", comparisons)
    primary_table = table(["目标", "v4探索均值 pp", "v5验证均值±样本SD pp", "逐重复范围 pp", "95%配对t区间 pp", "两项同时区间 pp"],
                          [[r["target"], pp(c["exploratory_mean_interaction"]), f"{pp(r['mean'])} ± {100*r['sample_sd']:.2f}",
                            f"[{pp(r['min'])}, {pp(r['max'])}]", f"[{pp(r['ci95_lower'])}, {pp(r['ci95_upper'])}]",
                            f"[{pp(r['family95_lower'])}, {pp(r['family95_upper'])}]"] for r, c in zip(primary, comparisons)])
    repeat_table = table(["目标", "新重复/初始化seed", "Δ(B−A) pp", "Δ(C−A) pp", "主Δ(B−C) pp"],
                         [[r["target"], r["seed"], *(pp(r[m]) for m in ps.METRICS)] for r in differences])
    level_table = table(["目标", "背景", "A准确率", "B准确率", "C准确率", "目标多数基线"],
                        [[contrast["target"], background,
                          *(pct(next(r["mean"] for r in aggregate if r["target"] == contrast["target"] and r["background"] == background and r["condition"] == c)) for c in ("A", "B", "C")),
                          pct(next(r["majority_accuracy_mean"] for r in aggregate if r["target"] == contrast["target"] and r["background"] == background))]
                         for contrast in plan["contrasts"] for background in (contrast["original_background"], contrast["new_background"])])
    diagnostic_table = table(["目标", "双侧符号翻转p", "双侧符号p", "符合方向/非零n", "留一均值范围 pp"],
                            [[r["target"], f"{r['sign_flip_two_sided_p']:.4f}", f"{r['sign_test_two_sided_p']:.4f}",
                              f"{r['direction_matching_count']}/{r['sign_test_nonzero_n']}",
                              f"[{pp(r['leave_one_mean_min'])}, {pp(r['leave_one_mean_max'])}]"] for r in primary])
    source_context_table = table(["目标", "重复", "目标交互 pp", "B候选源dev背景差 pp", "C候选源dev背景差 pp", "未换公共源最大绝对dev差 pp"],
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
    source_summary_table = table(["目标", "背景", "条件", "候选源", "train均值", "dev均值", "dev范围", "dev多数基线"], source_aggregate)
    cost_table = table(["阶段", "run数", "更新数", "输入tokens", "监督tokens", "运行计时分钟"],
                       [[r["kind"], r["runs"], r["updates"], r["input_tokens"], r["supervised_tokens"], f"{r['seconds']/60:.2f}"] for r in costs])
    seed_table = table(["重复", "初始化seed", "训练/开发生成seed", "测试生成seed"],
                       [[f"R{i+1}", r["seed"], r["data_generation_seed"], r["test_data_seed"]] for i, r in enumerate(plan["repetitions"])])
    lr_counts = dict(Counter(r["lr"] for r in selected))
    audit_path = out / "FINAL_AUDIT.json"
    audit_passed = audit_path.exists() and js(audit_path).get("status") == "passed"
    audit_text = "最终独立审计通过，完整状态和逐项记录见FINAL_AUDIT.json" if audit_passed else "最终独立审计尚未通过，需运行audit_final.py后再视为交付完成"
    total_seconds = sum(r["seconds"] for r in costs)
    stage_seconds = sum(r["seconds"] for r in timings)
    main_lines = "\n".join(judgments)
    source_max = max(abs(float(r["B_minus_C_dev_accuracy"])) for r in common)
    source_findings = source_result_findings(learning, contexts, summaries)
    report = f"""# pilot_v5：固定假设的背景交互验证

本轮完成两个由v4探索结果提出的固定对比、五个新数据/初始化配对重复。主量是在每个重复内先计算的 `Δ(B−C)=(B−C)新背景−(B−C)原背景`，不是两个背景均值各自标准差的比较。

{main_lines}

以上只验证这两个预选案例、n=10、256个目标样本和固定预算。旧三个seed属于探索，新五个重复属于验证；没有将它们合并为八次重复。训练及开发选择已完成后才生成统一新测试，保留所有负收益、未学会和不一致结果。

## 先重算探索证据，再冻结验证

从v4的42条唯一selected原始行重算所有目标的g1−g0、g2−g0，12条逐seed背景差和图见[探索说明](exploratory/EXPLORATORY_ANALYSIS_ZH.md)。FP g2−g0的Δ(B−A)/Δ(C−A)/Δ(B−C)探索均值为+16.59/+12.22/+4.36pp，主交互逐seed为+1.42/+0.34/+11.33pp；LDR g1−g0为−11.56/+0.44/−12.00pp，主交互逐seed为−20.07/+2.00/−17.92pp。共享A在背景差中严格消去。

预先固定H1：fixed_points，原背景→deficiencies背景，期望主交互为正；H2：longest_decreasing_run，原背景→recoils背景，期望主交互为负。方向来自v4探索，不是独立提出的理论预测。此次不再验证另外两个背景对比，也不依据新结果更换方向、改用单侧检验或新增重复。

| 目标 | 候选B / 候选C | 原三个公共源 | 新三个公共源 |
| --- | --- | --- | --- |
| fixed_points | anti_fixed_points / successions | descents, peaks, left_to_right_maxima | descents, deficiencies, left_to_right_maxima |
| longest_decreasing_run | longest_increasing_run / left_to_right_maxima | descents, peaks, successions | recoils, peaks, successions |

候选固定slot0，公共源slots1–3。两个目标均不进入任一预训练组；数学任务关系与实例隔离是不同层面的检查。组定义和统计方法在新正式训练之前按hash冻结，详见[冻结配置](artifacts/FROZEN_PROTOCOL.json)、[stats_plan.json](stats_plan.json)。

## 五个新重复的验证结果

{primary_table}

{repeat_table}

![逐重复主交互及区间](artifacts/figures/confirmation_interactions_by_repeat.png)

![主交互的B−A/C−A分量](artifacts/figures/confirmation_background_components.png)

下表为五重复的准确率均值，仅帮助理解水平与多数基线；A每目标/重复只训练和选择一次，在背景展示表中重复显示。60条展示行对应50个唯一选中模型，并非60次独立选择。

{level_table}

区间直接基于五个配对交互差，SD使用n−1分母。单项区间采用df=4、t=.975分位2.776445；两个主要对比的Bonferroni同时区间，每项边际覆盖97.5%，临界值3.495406。后者不要求两个目标对比之间独立，但t覆盖仍依赖重复层面的近似独立正态假设。五次重复不足以检验正态性或尾部；不同RNG/初始化流也不等于严格IID数据池，因为跨块orbit排除构成有限总体不放回依赖。区间包含有限测试池的随机变化，不能当作所有任务族的区间。

{diagnostic_table}

符号翻转诊断穷举32种符号，要求零中心下独立符号对称，不是由本实验设计保证的处理随机化检验；符号检验另要求独立重复下正负概率各1/2，针对符号/中位数倾向而非均值。五个非零重复时两者双侧p最小都为0.0625，所以不能承诺常规显著性。留一均值只描述对单个重复的敏感性，不替代主分析。辅助Δ(B−A)/Δ(C−A)及源学习只报告描述结果，不增加检验家族或事后筛选显著结果。

## 联合数学关系与数据隔离

实际任务实现与独立核验见[数学关系说明](mathematics/README.md)、[核验结果](mathematics/MATH_AUDIT.json)。清单明确区分有公式/证明、数据观察和未检查；穷举与随机大n核验是实现检查，不能将有限核验代替一般证明。尤其要区分同一输入的源答案能否联合恢复目标，和允许对变换输入另做查询时能否恢复目标。

对于每个位置i，p(i)与i恰好满足等于、大于、小于之一，所以 `fixed_points(p)+exceedances(p)+deficiencies(p)=n`。若同时提供同输入的exceedances和deficiencies答案，则可求fixed_points；本轮FP新背景只有deficiencies，没有exceedances，因此不能仅凭该等式声称当前组同输入联合给出了fixed_points。又有 `exceedances(p)=deficiencies(inverse(p))`，故允许额外逆排列查询时，`fixed_points(p)=n−deficiencies(p)−deficiencies(inverse(p))`；本轮没有给模型显式成对查询或联合推理监督。

**FP的deficiencies背景中，B和C都具有上述严格的跨输入联合关系，不能把C解释为不含目标数学关系的对照。** 数学核验已为v4全部12组提供n=10的反例：相同n及同一输入的四源标量答案不能唯一恢复目标。限定到四源×I/R/C八变换的32个答案时，六个B都可从候选变换查询恢复目标，FP g2 C可用两次deficiencies查询恢复；其余五个C各有n=10不充分反例。这只排除指定答案查询模型，不排除从raw input计算目标或其他未检查关系。清单见[TARGET_JOINT_DERIVABILITY.csv](mathematics/audit/TARGET_JOINT_DERIVABILITY.csv)和[显式反例](mathematics/audit/COUNTEREXAMPLES.csv)。

核验实际完成S1…S7穷举5913个输入与4096个n=10 fixture；20个定义在八变换上完成1,601,440次独立标签比对，28条公式/界各验证10,009次。fixture仅为数学实现核对，没有送入训练或用于选择迁移配置。仅由数据观察的关联与未检查机制分别见[OBSERVED_ASSOCIATIONS.csv](mathematics/audit/OBSERVED_ASSOCIATIONS.csv)、[UNCHECKED_CLAIMS.csv](mathematics/audit/UNCHECKED_CLAIMS.csv)。

相关候选anti_fixed_points与fixed_points由补值/反向输入变换联系，LIR/LDR由补值或反向变换联系。descents、runs、peaks/valleys等还共享局部比较结构。它们是合法数学关系，不能自动标为泄漏；本轮泄漏检查针对训练、开发、测试的实际输入及inverse/reverse/complement完整轨道重叠。是否能从某组全部源标量答案恢复目标，以及是否需要变换查询，以数学清单逐组条目和反例为准；没有检查的关系不写成不存在。

## 源学习辅助结果及机制边界

每个四源run4800步，batch64，每步每任务16条，精确25%采样。各任务76800次暴露、2150400输入tokens、153600监督tokens；实际不同输入数、流hash、每长度(n=10)训练/开发准确率、答案CE、EOS CE与多数基线逐条保存于[source_learning_table.csv](artifacts/source_learning_table.csv)与[source_exposures.csv](artifacts/source_exposures.csv)。

{source_summary_table}

{source_context_table}

上述10条为全部重复，而不是按迁移幅度挑出的案例。候选源两列比较的是同一任务跨背景的dev准确率；公共源最大值仅覆盖没有更换定义的任务。更换的peaks/deficiencies或descents/recoils只并列其水平与基线，不把不同任务的准确率相减当作学习改变。相同背景内公共任务B/C dev差的绝对值最大为{100*source_max:.2f}pp，完整记录见[common_source_BC.csv](artifacts/common_source_BC.csv)。

源记录能支持“组合改变与源学习状态改变同时出现”及“某些源在当前预算仍未学会”的描述，不能证明源准确率变化导致迁移交互。源与目标可能共同受优化、共享计算、答案分布、表示竞争和有限样本影响；本轮没有对这些中介量做独立干预。数学上可推导也不等于Transformer实际学会或利用了公式。若某个重复源与迁移变化方向不一致，仍完整保留，不能事后只解释符合直觉的重复。

{source_findings}

旧v3单任务记录作为历史参照复制到[historical_single_references.csv](artifacts/historical_single_references.csv)，本轮没有重训单任务。它们使用旧seed/输入池，与新多任务不构成严格配对对照；单1200步只匹配每任务暴露，学习率轨迹与总更新仍不同，单4800步暴露为多任务每任务的四倍。因此不能把新多任务−旧单任务差直接解释为组合干扰效应。完整新源曲线与目标曲线均保留，选中LR的train/dev图位于[figures目录](artifacts/figures)。

## 配置、实际预算与复现

{seed_table}

各重复重新初始化模型并生成独立seed流的数据，没有用旧权重充当新重复。五个新重复之间及与v1–v4所有旧输入，按统一的完整变换轨道规则隔离。每块source_train8192/source_dev1024/target_train256/target_dev1024，测试2048；同一块各目标、背景、条件共享相应输入，目标标签由定义确定。source采样/dropout seed为初始化seed+1000，adapt为+2000；完整生成顺序与数据hash见seed_provenance、DATA_READY及test_data_audit。

模型保持4层Transformer，width256、8 heads、MLP4、dropout.1、绑定输入/output embedding、词表188，PermWorld one-line，n=10；参数3,240,448。每训练前向28 tokens，其中答案/EOS两位置监督，前缀不计loss。B/C各4800步，LR先100步warmup到3e−4，2400步余弦到3e−5后保持到4800。每适配条件沿用三档LR 3e−4/1e−4/3e−5，各800步，100步warmup和余弦到初值0.1。没有新增超参数搜索。

每100步全量评估train/dev，排除step0，依次按最高dev答案准确率、最低dev答案CE、最早step、固定LR顺序选择。50个唯一模型选择LR计数：{lr_counts}。适配严格加载源参数、将目标token的绑定embedding/output行重置为同块step0值、使用全新AdamW状态。AdamW wd=.01、betas=.9/.999、eps=1e−8、clip1；BF16训练、FP32权重/评估、TF32关闭。

先冻结全部50项开发选择，再生成五个新测试池；300条全部LR的best/endpoint测试结果保留为诊断，不据测试重选模型。主分析只使用50条selected；训练终点对照保留于[selected_vs_endpoint.csv](artifacts/selected_vs_endpoint.csv)。{audit_text}。实际命令、文件职责与继续运行方式见[README](README.md)。

{cost_table}

新增训练计时合计{total_seconds/60:.2f}分钟；引擎各阶段墙钟合计{stage_seconds/60:.2f}分钟。训练计时包括训练、全量train/dev评估、checkpoint保存和记账，不等同于纯GPU计算时间；二者不相加，独立审计与绘图另计。40个预训练run和150个适配候选全部新增，合计312000次更新、559104000输入tokens；没有延长预算，也没有新增单任务训练。checkpoint和配置保留，便于后续CKA接口使用。

## 下一阶段任务族扩展方案与收口

详见冻结在新迁移结果之前的[TASK_FAMILY_PLAN.md](TASK_FAMILY_PLAN.md)。优先复用现有循环分解、全局有序子序列、置换分块族，使已研究的局部比较和位置不等计数获得不同计算结构的参照；其次加入位移/成对次序与模式计数，参数实例不重复计成独立族。后续真正需要新增的族包括无向图连通/循环秩、含奇异矩阵的有限域rank/kernel、奇特征域二次型/双线性形式；这些新生成器本轮尚未实现或认证。各族先完成独立oracle核验与单任务开发，再冻结组合，保留不能学会的候选；不由两个预选案例外推任务族的普遍排序。

平台应把数学task_id、参数、编码、输入变换分开；单任务关系记录域条件、方向、输入作用与答案映射，联合关系用多输入/多源超边记录需要的源集合、查询变换、公式/证明与测试范围，另存观察关联及未检查标记。生成器复用base-instance→变换→标签oracle→统一split/orbit去重→编码，并存版本、seed、实际pool hash、采样slot与暴露量。关系导出的合法任务不自动称为泄漏，实例隔离和关系类别分别审计。

本轮只完成两个预指定对比的五次新重复。对效应方向、幅度不确定性和机制猜想分别报告，不追加这两个目标的背景变体、额外重复、训练预算或学习率，等待后续指导。

## 主要交付

- [逐重复配对差](artifacts/confirmation_paired_differences.csv)、[主/辅助汇总](artifacts/confirmation_summary.csv)、[探索与验证分表](artifacts/confirmation_vs_exploration.csv)。
- [50条唯一主结果](artifacts/results_selected.csv)、[300条全部候选best/endpoint](artifacts/results_all_candidates.csv)；每重复原始CSV：[101](artifacts/results_seed101.csv)、[211](artifacts/results_seed211.csv)、[307](artifacts/results_seed307.csv)、[401](artifacts/results_seed401.csv)、[503](artifacts/results_seed503.csv)。
- [源学习](artifacts/source_learning_table.csv)、[实际暴露](artifacts/source_exposures.csv)、[背景间同源变化](artifacts/source_background_changes.csv)、[源与目标交互并列](artifacts/source_interaction_context.csv)。
- [统计审计](artifacts/confirmation_statistics_audit.json)、[最终独立审计](artifacts/FINAL_AUDIT.json)、[数学核验](mathematics/MATH_AUDIT.json)、[冻结统计方案](stats_plan.json)。
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
