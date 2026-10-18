"""Summarize saved v4 CSV/JSON only; no model imports or inference."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "artifacts"
V3 = HERE.parent / "pilot_v3_multitarget" / "artifacts"
COLORS = {"A": "#69778d", "B": "#1683ad", "C": "#d28032"}


def js(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, values):
    values = list(values)
    if not values:
        raise ValueError(f"Empty required output: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(dict.fromkeys(k for r in values for k in r)))
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


def table(headers, values):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |",
                       "| " + " | ".join("---" for _ in headers) + " |",
                       *("| " + " | ".join(map(str, r)) + " |" for r in values)])


def savefig(fig, out, name):
    fig.savefig(out / "figures" / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def identity(row):
    return row["target"], row["background"], integer(row, "seed"), row["condition"]


def result_tables(out, cfg, selected, candidates):
    index = {identity(r): r for r in selected}
    assert len(index) == 42
    expanded, summary, deltas, delta_summary, interactions, interaction_summary, endpoints = [], [], [], [], [], [], []
    for target in cfg["targets"]:
        for background in cfg["backgrounds"]:
            for seed in cfg["seeds"]:
                for c in "ABC":
                    row = index[target, "shared" if c == "A" else background, seed, c]
                    expanded.append(dict(row, comparison_background=background, shared_A_display=c == "A"))
            for c in "ABC":
                group = [r for r in expanded if r["target"] == target and r["comparison_background"] == background and r["condition"] == c]
                record = dict(target=target, background=background, condition=c, n_paired_seeds=len(group), shared_A_display=c == "A")
                for metric in ("test_accuracy", "test_answer_ce", "dev_accuracy", "dev_ce", "train_accuracy", "majority_accuracy"):
                    record.update(stats((num(r, metric) for r in group), metric))
                summary.append(record)
            for first, second in (("B", "A"), ("C", "A"), ("B", "C")):
                group = []
                for seed in cfg["seeds"]:
                    a = index[target, "shared" if first == "A" else background, seed, first]
                    b = index[target, "shared" if second == "A" else background, seed, second]
                    record = dict(target=target, background=background, seed=seed, contrast=f"{first}-{second}",
                                  test_accuracy_delta=num(a, "test_accuracy") - num(b, "test_accuracy"),
                                  dev_accuracy_delta=num(a, "dev_accuracy") - num(b, "dev_accuracy"))
                    deltas.append(record)
                    group.append(record)
                ds = [r["test_accuracy_delta"] for r in group]
                delta_summary.append(dict(target=target, background=background, contrast=f"{first}-{second}", n_seeds=3,
                                          **stats(ds, "delta"), negative_seeds=sum(v < 0 for v in ds), zero_seeds=sum(v == 0 for v in ds), positive_seeds=sum(v > 0 for v in ds)))
        for earlier, later in combinations(cfg["backgrounds"], 2):
            group = []
            for seed in cfg["seeds"]:
                early = next(r for r in deltas if r["target"] == target and r["background"] == earlier and r["seed"] == seed and r["contrast"] == "B-C")
                late = next(r for r in deltas if r["target"] == target and r["background"] == later and r["seed"] == seed and r["contrast"] == "B-C")
                record = dict(target=target, seed=seed, earlier_background=earlier, later_background=later, interaction=f"{later}-{earlier}",
                              earlier_B_minus_C=early["test_accuracy_delta"], later_B_minus_C=late["test_accuracy_delta"],
                              interaction_test_accuracy=late["test_accuracy_delta"] - early["test_accuracy_delta"],
                              interaction_dev_accuracy=late["dev_accuracy_delta"] - early["dev_accuracy_delta"],
                              BC_sign_reversal=early["test_accuracy_delta"] * late["test_accuracy_delta"] < 0)
                interactions.append(record)
                group.append(record)
            values = [r["interaction_test_accuracy"] for r in group]
            interaction_summary.append(dict(target=target, interaction=f"{later}-{earlier}", earlier_background=earlier, later_background=later,
                                            n_seeds=3, **stats(values, "interaction"), negative_seeds=sum(v < 0 for v in values),
                                            zero_seeds=sum(v == 0 for v in values), positive_seeds=sum(v > 0 for v in values),
                                            BC_sign_reversal_seeds=sum(r["BC_sign_reversal"] for r in group)))
    for row in selected:
        endpoint = next(r for r in candidates if identity(r) == identity(row) and num(r, "lr") == num(row, "lr") and r["checkpoint"] == "endpoint")
        endpoints.append(dict(target=row["target"], background=row["background"], seed=row["seed"], condition=row["condition"], lr=row["lr"],
                              selected_step=row["adapt_step"], selected_dev_accuracy=row["dev_accuracy"], endpoint_dev_accuracy=endpoint["dev_accuracy"],
                              selected_train_accuracy=row["train_accuracy"], endpoint_train_accuracy=endpoint["train_accuracy"],
                              selected_dev_answer_ce=row["dev_ce"], endpoint_dev_answer_ce=endpoint["dev_ce"],
                              selected_test_accuracy=row["test_accuracy"], endpoint_test_accuracy=endpoint["test_accuracy"],
                              endpoint_minus_selected_test=num(endpoint, "test_accuracy") - num(row, "test_accuracy")))
    assert len(expanded) == 54 and len(deltas) == 54 and len(interactions) == 18 and len(endpoints) == 42
    for name, data in (("results_display_ABC.csv", expanded), ("results_summary.csv", summary), ("paired_deltas.csv", deltas),
                       ("paired_deltas_summary.csv", delta_summary), ("background_interactions.csv", interactions),
                       ("background_interactions_summary.csv", interaction_summary), ("selected_vs_endpoint.csv", endpoints)):
        write_csv(out / name, data)
    return expanded, summary, deltas, delta_summary, interactions, interaction_summary, endpoints


def result_plots(out, cfg, expanded, deltas, interactions, interaction_summary):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for i, target in enumerate(cfg["targets"]):
        for j, background in enumerate(cfg["backgrounds"]):
            ax = axes[i, j]
            for c in "ABC":
                values = sorted([r for r in expanded if r["target"] == target and r["comparison_background"] == background and r["condition"] == c], key=lambda r: integer(r, "seed"))
                ax.plot(range(3), [num(r, "test_accuracy") for r in values], marker="o", color=COLORS[c], label=c + (" (shared)" if c == "A" else ""))
            baseline = sorted([r for r in expanded if r["target"] == target and r["comparison_background"] == background and r["condition"] == "A"], key=lambda r: integer(r, "seed"))
            ax.plot(range(3), [num(r, "majority_accuracy") for r in baseline], color="grey", linestyle=":", label="train-majority baseline")
            ax.set(title=f"{target}\n{background}", ylabel="New-test answer accuracy", ylim=(0, 1.03), xlabel="Paired block (data + initialization)")
            ax.set_xticks(range(3), cfg["seeds"])
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    fig.suptitle("256 target samples; A shown in three backgrounds but trained/selected once")
    fig.tight_layout()
    savefig(fig, out, "accuracy_by_background_seed.png")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for i, target in enumerate(cfg["targets"]):
        for j, contrast in enumerate(("B-A", "C-A", "B-C")):
            ax = axes[i, j]
            for seed in cfg["seeds"]:
                values = [next(r["test_accuracy_delta"] for r in deltas if r["target"] == target and r["background"] == bg and r["seed"] == seed and r["contrast"] == contrast) for bg in cfg["backgrounds"]]
                ax.plot(range(3), [100 * v for v in values], marker="o", label=f"block {seed}")
            ax.axhline(0, color="black", lw=.8)
            ax.set_xticks(range(3), cfg["backgrounds"], rotation=15)
            ax.set(title=f"{target}\n{contrast}", ylabel="Paired accuracy difference (pp)")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle("Composition contrasts across prespecified backgrounds; same tests within each block")
    fig.tight_layout()
    savefig(fig, out, "contrasts_by_background_seed.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    names = [f"{later}-{earlier}" for earlier, later in combinations(cfg["backgrounds"], 2)]
    for ax, target in zip(axes, cfg["targets"]):
        group = [next(r for r in interaction_summary if r["target"] == target and r["interaction"] == name) for name in names]
        ax.errorbar(range(3), [100 * r["interaction_mean"] for r in group], yerr=[100 * r["interaction_sample_sd"] for r in group], fmt="D", color="black", capsize=5, label="mean +/- sample SD")
        for i, seed in enumerate(cfg["seeds"]):
            vals = [next(r["interaction_test_accuracy"] for r in interactions if r["target"] == target and r["seed"] == seed and r["interaction"] == name) for name in names]
            ax.plot([j + (i - 1) * .05 for j in range(3)], [100 * v for v in vals], marker="o", alpha=.65, label=f"block {seed}")
        ax.axhline(0, color="grey", lw=.8)
        ax.set_xticks(range(3), [name.replace("-g", "\n- g") for name in names], fontsize=8)
        ax.set(title=target, ylabel="Interaction: change in B - C (pp)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Paired background interactions; descriptive n=3, not independent backgrounds or a causal attribution")
    fig.tight_layout()
    savefig(fig, out, "background_interactions.png")


def source_tables(out, cfg, interactions):
    curve = rows(out / "source_curves.csv")
    exposure = rows(out / "source_exposures.csv")
    assert len(exposure) == 144
    index = {(r["target"], r["background"], integer(r, "seed"), r["condition"], r["task"], integer(r, "step"), r["split"]): r for r in curve}
    metas = [dict(path=path, **js(path)) for path in sorted((out / "multi").glob("*/completed.json"))]
    mi = {(m["target"], m["background"], m["seed"], m["condition"]): m for m in metas}
    single_cache, learning, common = {}, [], []
    for r in exposure:
        seed, task = integer(r, "seed"), r["task"]
        if (seed, task) not in single_cache:
            single_cache[seed, task] = {(integer(a, "step"), a["split"]): a for a in rows(V3 / "single" / f"s{seed}_{task}" / "curve.csv")}
        rec = dict(r, role="candidate" if integer(r, "slot") == 0 else "common")
        for split in ("train", "dev"):
            src = index[r["target"], r["background"], seed, r["condition"], task, 4800, split]
            for metric in ("accuracy", "answer_ce", "two_token_ce", "eos_ce", "majority_accuracy", "count"):
                rec[f"multi4800_{split}_{metric}"] = src[metric]
            for step in (1200, 4800):
                src = single_cache[seed, task][step, split]
                for metric in ("accuracy", "answer_ce", "majority_accuracy"):
                    rec[f"single{step}_{split}_{metric}"] = src[metric]
        rec["multi_minus_single4800_dev_accuracy"] = num(rec, "multi4800_dev_accuracy") - num(rec, "single4800_dev_accuracy")
        rec["multi_state_hash"] = mi[r["target"], r["background"], seed, r["condition"]]["model_state_hash"]
        learning.append(rec)
    li = {(r["target"], r["background"], integer(r, "seed"), r["condition"], integer(r, "slot")): r for r in learning}
    for g in cfg["groups"]:
        for seed in cfg["seeds"]:
            for slot in (1, 2, 3):
                b, c = li[g["target"], g["background"], seed, "B", slot], li[g["target"], g["background"], seed, "C", slot]
                assert b["task"] == c["task"]
                common.append(dict(target=g["target"], background=g["background"], seed=seed, slot=slot, task=b["task"],
                                   B_dev_accuracy=b["multi4800_dev_accuracy"], C_dev_accuracy=c["multi4800_dev_accuracy"],
                                   B_minus_C_dev_accuracy=num(b, "multi4800_dev_accuracy") - num(c, "multi4800_dev_accuracy"),
                                   B_dev_answer_ce=b["multi4800_dev_answer_ce"], C_dev_answer_ce=c["multi4800_dev_answer_ce"]))
    changes, contexts = [], []
    for interaction in interactions:
        target, seed, early, late = interaction["target"], interaction["seed"], interaction["earlier_background"], interaction["later_background"]
        context = dict(interaction)
        same_common_changes = []
        for condition in "BC":
            for slot in range(4):
                a, b = li[target, early, seed, condition, slot], li[target, late, seed, condition, slot]
                same = a["task"] == b["task"]
                change = num(b, "multi4800_dev_accuracy") - num(a, "multi4800_dev_accuracy") if same else ""
                changes.append(dict(target=target, seed=seed, condition=condition, interaction=interaction["interaction"], earlier_background=early,
                                    later_background=late, slot=slot, role="candidate" if slot == 0 else "common", earlier_task=a["task"], later_task=b["task"],
                                    same_mathematical_task=same, earlier_dev_accuracy=a["multi4800_dev_accuracy"], later_dev_accuracy=b["multi4800_dev_accuracy"],
                                    same_task_dev_accuracy_change=change, earlier_state_hash=a["multi_state_hash"], later_state_hash=b["multi_state_hash"]))
                if slot == 0:
                    assert same
                    context[f"{condition}_candidate_task"] = a["task"]
                    context[f"{condition}_candidate_dev_change"] = change
                elif same:
                    same_common_changes.append(abs(change))
        context["same_common_task_change_abs_mean"] = mean(same_common_changes)
        context["same_common_task_change_abs_max"] = max(same_common_changes)
        context["same_common_task_condition_occurrences"] = len(same_common_changes)
        contexts.append(context)
    assert len(common) == 54 and len(changes) == 144 and len(contexts) == 18
    unique = {}
    for r in learning:
        key = integer(r, "seed"), r["task"], r["multi_state_hash"]
        unique.setdefault(key, dict(seed=key[0], task=key[1], multi_state_hash=key[2], logical_occurrences=0,
                                    multi_minus_single4800_dev_accuracy=r["multi_minus_single4800_dev_accuracy"]))["logical_occurrences"] += 1
    effects = []
    for scope, records in (("logical_occurrences", learning), ("unique_seed_task_source_state", list(unique.values()))):
        values = [r["multi_minus_single4800_dev_accuracy"] for r in records]
        effects.append(dict(scope=scope, count=len(values), improved_at_least_5pp=sum(v >= .05 for v in values), decreased_at_least_5pp=sum(v <= -.05 for v in values), within_5pp=sum(-.05 < v < .05 for v in values)))
    for name, records in (("source_learning_table.csv", learning), ("common_source_BC.csv", common), ("source_background_slot_changes.csv", changes),
                          ("interaction_source_context.csv", contexts), ("source_effect_counts.csv", effects), ("source_effects_unique.csv", unique.values())):
        write_csv(out / name, records)
    return curve, metas, learning, common, changes, contexts, effects


def source_plots(out, cfg, curve):
    for g in cfg["groups"]:
        target, bg = g["target"], g["background"]
        tasks = list(dict.fromkeys(g["sources"]["B"] + g["sources"]["C"]))
        fig, axes = plt.subplots(len(tasks), 3, figsize=(15, 2.4 * len(tasks)), sharey=True, squeeze=False)
        for i, task in enumerate(tasks):
            for j, seed in enumerate(cfg["seeds"]):
                ax = axes[i, j]
                single = rows(V3 / "single" / f"s{seed}_{task}" / "curve.csv")
                dev = sorted([r for r in single if r["split"] == "dev"], key=lambda r: integer(r, "step"))
                ax.plot([integer(r, "step") for r in dev], [num(r, "accuracy") for r in dev], color="grey", linestyle=":", label="single dev (4x exposure)")
                for c in "BC":
                    if task not in g["sources"][c]:
                        continue
                    for split, style in (("train", ":"), ("dev", "-")):
                        values = sorted([r for r in curve if r["target"] == target and r["background"] == bg and integer(r, "seed") == seed and r["condition"] == c and r["task"] == task and r["split"] == split], key=lambda r: integer(r, "step"))
                        label = f"{c} {split}" + (" (starts 2400)" if integer(values[0], "step") == 2400 else "")
                        ax.plot([integer(r, "step") for r in values], [num(r, "accuracy") for r in values], style, color=COLORS[c], label=label)
                ax.axhline(num(dev[0], "majority_accuracy"), color="grey", alpha=.4, linestyle="--")
                ax.set(title=f"{task}; block {seed}", xlabel="Total optimizer updates", ylabel="Answer accuracy", ylim=(0, 1.03))
                ax.grid(alpha=.2)
                ax.legend(fontsize=6)
        fig.suptitle(f"{target}; {bg}; source learning (full CE metrics preserved in CSV)")
        fig.tight_layout()
        savefig(fig, out, f"source_curves_{target}_{bg}.png")


def target_plots_and_late(out, cfg, choices):
    curve = rows(out / "target_curves.csv")
    assert len(curve) == 1134
    late = []
    for path in sorted((out / "adapt").glob("*/completed.json")):
        m = js(path)
        cp = {integer(r, "step"): r for r in rows(path.parent / "curve.csv")}
        record = {k: m[k] for k in ("target", "background", "seed", "condition", "lr")}
        for split in ("train", "dev"):
            for metric in ("accuracy", "answer_ce"):
                key = split + "_" + metric
                record[key + "_400"] = num(cp[400], key)
                record[key + "_800"] = num(cp[800], key)
                record[key + "_change"] = num(cp[800], key) - num(cp[400], key)
        late.append(record)
    write_csv(out / "late_training_diagnostics.csv", late)
    for target in cfg["targets"]:
        for seed in cfg["seeds"]:
            cells = [("shared", "A"), *[(bg, c) for bg in cfg["backgrounds"] for c in "BC"]]
            fig, axes = plt.subplots(7, 2, figsize=(12, 16))
            for i, (bg, condition) in enumerate(cells):
                chosen = next(r for r in choices if identity(r) == (target, bg, seed, condition))
                values = sorted([r for r in curve if identity(r) == (target, bg, seed, condition) and num(r, "lr") == float(chosen["lr"])], key=lambda r: integer(r, "step"))
                assert len(values) == 9
                for ax, metric in zip(axes[i], ("accuracy", "answer_ce")):
                    for split, style in (("train", "-"), ("dev", "--")):
                        ax.plot([integer(r, "step") for r in values], [num(r, split + "_" + metric) for r in values], style, marker=".", label=split)
                    ax.axvline(chosen["step"], color="grey", linestyle=":", label="dev-selected step")
                    ax.set(title=f"{bg}, {condition}, LR {chosen['lr']:g}", xlabel="Adaptation update", ylabel=metric)
                    ax.grid(alpha=.2)
                    ax.legend(fontsize=7)
                axes[i, 0].set_ylim(0, 1.03)
                axes[i, 0].axhline(num(values[0], "dev_majority_accuracy"), color="grey", linestyle=":", alpha=.5)
            fig.suptitle(f"{target}, block {seed}; A once, all three backgrounds; selected LR curves")
            fig.tight_layout()
            savefig(fig, out, f"target_selected_{target}_seed{seed}.png")
    return late


def compute_tables(out, source_metas):
    records = []
    for kind, metas in (("pretrain", source_metas), ("adapt", [dict(path=path, **js(path)) for path in sorted((out / "adapt").glob("*/completed.json"))])):
        for m in metas:
            record = {k: m[k] for k in ("target", "background", "seed", "condition", "new_compute", "updates", "input_tokens", "supervised_tokens", "sample_exposures", "seconds")}
            record.update(kind=kind, run_id=m["path"].parent.name,
                          timer_scope="new training/full train-dev evaluation/checkpoint and bookkeeping; reused timers historical, not v4 added cost")
            records.append(record)
    assert len(records) == 162
    write_csv(out / "compute_by_run.csv", records)
    summary = []
    for kind in ("pretrain", "adapt"):
        for new in (True, False):
            group = [r for r in records if r["kind"] == kind and r["new_compute"] == new]
            summary.append(dict(kind=kind, new_compute=new, runs=len(group), **{k: sum(r[k] for r in group) for k in ("updates", "input_tokens", "supervised_tokens", "sample_exposures", "seconds")}))
    write_csv(out / "compute_summary.csv", summary)
    timing = [js(path) for path in sorted(out.glob("timing_*.json"))]
    write_csv(out / "stage_timings.csv", timing)
    state_groups = defaultdict(list)
    for m in source_metas:
        state_groups[m["model_state_hash"]].append(m["path"].parent.name)
    reuse = [dict(run_id=m["path"].parent.name, seed=m["seed"], ordered_tasks=";".join(m["tasks"]),
                  model_state_hash=m["model_state_hash"], same_state_runs=";".join(state_groups[m["model_state_hash"]]),
                  same_state_count=len(state_groups[m["model_state_hash"]]), new_compute=m["new_compute"]) for m in source_metas]
    write_csv(out / "source_state_reuse.csv", reuse)
    return summary, timing, len(state_groups)


def candidate_matching(out, cfg):
    """Describe the retained pair using frozen, pre-v4 evidence; no selection."""
    distributions = {(integer(r, "seed"), r["task"]): r for r in rows(HERE / "catalog" / "SOURCE_ANSWER_DISTRIBUTIONS.csv")}
    references = {(integer(r, "seed"), r["task"], integer(r, "step"), r["split"]): r for r in rows(HERE / "catalog" / "SOURCE_REFERENCE_CURVES.csv")}
    prior = rows(V3 / "replacement_matching.csv")
    records = []
    for target in cfg["targets"]:
        g = next(g for g in cfg["groups"] if g["target"] == target)
        first, second = g["related"], g["replacement"]
        original = next(r for r in prior if r["target"] == target and r["related"] == first and r["replacement"] == second and truth(r["selected"]))
        for seed in cfg["seeds"]:
            a, b = distributions[seed, first], distributions[seed, second]
            ah, bh = json.loads(a["histogram"]), json.loads(b["histogram"])
            assert len(ah) == len(bh) and sum(ah) == integer(a, "count") and sum(bh) == integer(b, "count")
            tv = .5 * sum(abs(x / sum(ah) - y / sum(bh)) for x, y in zip(ah, bh))
            record = dict(target=target, seed=seed, B_candidate=first, C_candidate=second, length=10, answer_tv=tv,
                          source_train_count_B=a["count"], source_train_count_C=b["count"],
                          evidence="frozen catalog SOURCE_ANSWER_DISTRIBUTIONS and SOURCE_REFERENCE_CURVES; no new v4 training used")
            for condition, task in (("B", first), ("C", second)):
                for step in (1200, 4800):
                    r = references[seed, task, step, "dev"]
                    record[f"{condition}_single_dev_{step}"] = num(r, "accuracy")
                    record[f"{condition}_single_dev_answer_ce_{step}"] = num(r, "answer_ce")
                record[f"{condition}_reference_sha256"] = references[seed, task, 4800, "dev"]["source_sha256"]
            record["mean_absolute_dev_curve_gap"] = mean(abs(num(references[seed, first, step, "dev"], "accuracy") - num(references[seed, second, step, "dev"], "accuracy")) for step in cfg["source_eval_steps"] if step)
            record["v3_selection_evidence"] = os.path.relpath(V3 / "replacement_matching.csv", out).replace("\\", "/")
            record["v3_design_seed"] = 17
            record["v3_design_seed_score"] = original["score"]
            record["v4_candidate_choice"] = "retained v3 pair; diagnostic only, no new candidate selection"
            records.append(record)
    assert len(records) == 6
    write_csv(out / "candidate_matching.csv", records)
    return records


def case_observations(out, cfg, selected, deltas, interactions, learning, common):
    seeds = cfg["seeds"]
    si = {identity(r): r for r in selected}
    li = {(r["target"],r["background"],integer(r,"seed"),r["condition"],r["task"]):r for r in learning}
    def vector(values, formatter=pp):
        return "/".join(formatter(v) for v in values)
    def dv(target, bg, contrast):
        return [next(r["test_accuracy_delta"] for r in deltas if r["target"]==target and r["background"]==bg and r["seed"]==seed and r["contrast"]==contrast) for seed in seeds]
    target_text = (
        f"以下数列均按配对块{'/'.join(map(str,seeds))}排序。fixed_points在g0/g1的B−A和C−A均为3/3负，"
        f"g2的B−A变为{vector(dv('fixed_points','g2_deficiencies','B-A'))}pp，C−A为{vector(dv('fixed_points','g2_deficiencies','C-A'))}pp。"
        "原固定点负迁移依赖背景，不能泛化成这两个候选组成的固有属性；g2中块42的B收益仅约0.24pp，三个同号不能当作已确立的普遍收益。"
        "g2同时提高B和C，且两者都含deficiencies的多输入联合目标关系，不能把绝对迁移改善全归于B的直接相关候选。\n\n"
        f"longest_decreasing_run的g0 B−A为{vector(dv('longest_decreasing_run','g0_v3','B-A'))}pp，"
        f"g1缩至{vector(dv('longest_decreasing_run','g1_recoils','B-A'))}pp，g2为{vector(dv('longest_decreasing_run','g2_deficiencies','B-A'))}pp。"
        "g1的B/C均在训练众数对应的测试基线附近，原g0的块17/73收益消失；g2保留部分收益，块42从原无收益变为小幅正收益。"
    )
    fits = [r for r in rows(out/'selected_vs_endpoint.csv') if r['target']=='longest_decreasing_run' and r['background']=='g1_recoils']
    fit_text = (
        "g1的LDR适配不是整个训练过程都无法拟合。六项开发选择均退回LR=3e−5、step100的早期checkpoint；"
        f"所选train为{pct(min(num(r,'selected_train_accuracy') for r in fits))}–{pct(max(num(r,'selected_train_accuracy') for r in fits))}，"
        f"同LR800步终点train已升至{pct(min(num(r,'endpoint_train_accuracy') for r in fits))}–{pct(max(num(r,'endpoint_train_accuracy') for r in fits))}，"
        f"终点test却降至{pct(min(num(r,'endpoint_test_accuracy') for r in fits))}–{pct(max(num(r,'endpoint_test_accuracy') for r in fits))}。"
        "这支持后期拟合提高而泛化恶化、开发规则退回早期基线附近模型，不能把所选train接近众数误写成全程没有学到训练数据。"
    )
    fit_table = [[r['seed'],r['condition'],r['lr'],r['selected_step'],pct(r['selected_train_accuracy']),pct(r['endpoint_train_accuracy']),pct(r['selected_dev_accuracy']),pct(r['endpoint_dev_accuracy']),pct(r['selected_test_accuracy']),pct(r['endpoint_test_accuracy'])] for r in fits]
    source_table=[]
    for target,task in [('fixed_points','anti_fixed_points'),('longest_decreasing_run','longest_increasing_run')]:
        for bg in cfg['backgrounds']:
            group=[li[target,bg,seed,'B',task] for seed in seeds]
            source_table.append([target,bg,task,vector([num(r,'multi4800_train_accuracy') for r in group],pct),vector([num(r,'multi4800_dev_accuracy') for r in group],pct),vector([num(r,'single1200_dev_accuracy') for r in group],pct)])
    peak_changes=[num(li[t,'g1_recoils',seed,c,'peaks'],'multi4800_dev_accuracy')-num(li[t,'g0_v3',seed,c,'peaks'],'multi4800_dev_accuracy') for t in cfg['targets'] for seed in seeds for c in 'BC']
    examples=[]
    for seed,bg in [(17,'g1_recoils'),(42,'g2_deficiencies')]:
        old,new=li['fixed_points','g0_v3',seed,'B','anti_fixed_points'],li['fixed_points',bg,seed,'B','anti_fixed_points']
        examples.append(f"fixed_points块{seed}从g0到{bg}，anti_fixed_points源dev {pct(old['multi4800_dev_accuracy'])}→{pct(new['multi4800_dev_accuracy'])}，目标B测试{pct(si['fixed_points','g0_v3',seed,'B']['test_accuracy'])}→{pct(si['fixed_points',bg,seed,'B']['test_accuracy'])}")
    success_source=[num(li['fixed_points','g2_deficiencies',seed,'C','successions'],'multi4800_dev_accuracy')-num(li['fixed_points','g0_v3',seed,'C','successions'],'multi4800_dev_accuracy') for seed in seeds]
    success_target=[num(si['fixed_points','g2_deficiencies',seed,'C'],'test_accuracy')-num(si['fixed_points','g0_v3',seed,'C'],'test_accuracy') for seed in seeds]
    common_examples=[]
    for task in ('descents','successions','deficiencies'):
        values=[next(r['B_minus_C_dev_accuracy'] for r in common if r['target']=='longest_decreasing_run' and r['background']=='g2_deficiencies' and r['seed']==seed and r['task']==task) for seed in seeds]
        common_examples.append(f"{task}为{vector(values)}pp")
    improvements=[r for r in learning if r['target']=='longest_decreasing_run' and r['condition']=='B' and r['task']=='successions' and r['multi_minus_single4800_dev_accuracy']>=.05]
    source_text=(
        f"peaks在g1−g0的两目标×B/C×三块中有{sum(v<0 for v in peak_changes)}/{len(peak_changes)}项开发准确率下降。"
        "LDR的相关源LIR在g1的train/dev均较低，且低于其单任务1200步参照；这与块17/73目标收益消失并列出现，但块42原本没有收益，不能称所有块目标同向下降。\n\n"
        "候选源终点准确率不足以解释迁移："+'；'.join(examples)+'。'
        f"此外fixed_points的g2相对g0，C候选successions源dev变化为{vector(success_source)}pp，而C目标测试变化为{vector(success_target)}pp，三个块均是源候选下降、目标提高。\n\n"
        "LDR的g2内公共源B−C开发差异："+'；'.join(common_examples)+'。'
        "因此目标B/C差异同时伴随公共descents/successions学习变化，deficiencies则接近相同，不能分解为独立机制。组合也非只损害源任务：LDR-B的successions相对单任务4800的改善包括"+
        '；'.join(f"{r['background']}块{r['seed']} {pp(r['multi_minus_single4800_dev_accuracy'])}pp" for r in improvements)+
        "。不等每任务暴露、任务交替与三个块的优化差异仍限制干扰/协同归因。"
    )
    return dict(target_text=target_text,fit_text=fit_text,fit_table=fit_table,source_table=source_table,source_text=source_text)


def report(out, cfg, selected, summary, deltas, delta_summary, interactions, interaction_summary, learning, common, contexts, effects, compute, timing, late, state_count):
    report_path = HERE / "PILOT_V4_REPORT.md" if out == DEFAULT_OUT.resolve() else out / "PILOT_V4_REPORT.md"
    prefix = os.path.relpath(out, report_path.parent).replace("\\", "/")
    def link(name):
        return f"{prefix}/{name}"
    addendum_link = os.path.relpath(HERE / "catalog_addenda" / "README.md", report_path.parent).replace("\\", "/")
    matching_link = os.path.relpath(HERE / "catalog" / "BACKGROUND_MATCHING.csv", report_path.parent).replace("\\", "/")
    previous_matching_link = os.path.relpath(V3 / "replacement_matching.csv", report_path.parent).replace("\\", "/")
    matching = rows(HERE / "catalog" / "BACKGROUND_MATCHING.csv")
    matching_table = [[r["seed"], r["background"], f"{r['removed']}→{r['added']}", f"{num(r,'answer_tv'):.5f}",
                       pct(r["removed_dev_1200"]), pct(r["added_dev_1200"]), pct(r["removed_dev_4800"]), pct(r["added_dev_4800"]),
                       f"{100*num(r,'mean_absolute_curve_gap'):.2f}"] for r in matching]
    tv_ranges = {bg: (min(num(r, "answer_tv") for r in matching if r["background"] == bg),
                      max(num(r, "answer_tv") for r in matching if r["background"] == bg)) for bg in cfg["backgrounds"] if bg != "g0_v3"}
    candidate_records = candidate_matching(out, cfg)
    cases=case_observations(out,cfg,selected,deltas,interactions,learning,common)
    candidate_table = [[r["target"], r["seed"], r["B_candidate"], r["C_candidate"], f"{r['answer_tv']:.5f}",
                        pct(r["B_single_dev_1200"]), pct(r["C_single_dev_1200"]), pct(r["B_single_dev_4800"]), pct(r["C_single_dev_4800"]),
                        f"{100*r['mean_absolute_dev_curve_gap']:.2f}"] for r in candidate_records]
    audit = js(out / "FINAL_AUDIT.json") if (out / "FINAL_AUDIT.json").exists() else {}
    preserved_count = audit.get("historical_files_byte_preserved", audit.get("old_files_byte_preserved", 0))
    preserved = audit.get("status") == "passed" and preserved_count > 0
    audit_text = f"最终审计passed；{preserved_count}个历史文件字节hash已核对" if preserved else "最终审计及历史字节保留结论待确认，见FINAL_AUDIT.json"
    main_table = []
    for target in cfg["targets"]:
        for bg in cfg["backgrounds"]:
            s = {r["condition"]: r for r in summary if r["target"] == target and r["background"] == bg}
            d = {r["contrast"]: r for r in delta_summary if r["target"] == target and r["background"] == bg}
            main_table.append([target, bg, *[pct(s[c]["test_accuracy_mean"]) for c in "ABC"],
                               *[f"{pp(d[c]['delta_mean'])} ± {100*d[c]['delta_sample_sd']:.2f}" for c in ("B-A", "C-A", "B-C")]])
    interaction_table = [[r["target"], r["interaction"], *[pp(next(x['interaction_test_accuracy'] for x in interactions if x['target']==r['target'] and x['interaction']==r['interaction'] and x['seed']==seed)) for seed in cfg['seeds']], f"{pp(r['interaction_mean'])} ± {100*r['interaction_sample_sd']:.2f}",
                          f"{r['negative_seeds']}/{r['zero_seeds']}/{r['positive_seeds']}", r["BC_sign_reversal_seeds"]] for r in interaction_summary]
    raw_table = [[r["target"], r["background"], r["seed"], r["condition"], pct(r["test_accuracy"]), pct(r["majority_accuracy"]), r["lr"], r["adapt_step"], "复用" if not truth(r["new_compute"]) else "新增"] for r in selected]
    bg_table = [[g["target"], g["background"], g["related"], g["replacement"], *g["common"]] for g in cfg["groups"]]
    signs = []
    for target in cfg["targets"]:
        crossing = []
        for seed in cfg["seeds"]:
            values = [r["test_accuracy_delta"] for r in deltas if r["target"] == target and r["seed"] == seed and r["contrast"] == "B-C"]
            if min(values) < 0 < max(values):
                crossing.append(seed)
        consistent = [r["interaction"] for r in interaction_summary if r["target"] == target and (r["negative_seeds"] == 3 or r["positive_seeds"] == 3)]
        signs.append(f"{target}：B−C随背景改变符号的配对块为{crossing or '无'}；三个块同方向的背景交互为{consistent or '无'}。")
    strongest = sorted(contexts, key=lambda r: abs(r["interaction_test_accuracy"]), reverse=True)[:4]
    context_table = [[r["target"], r["seed"], r["interaction"], pp(r["interaction_test_accuracy"]),
                      pp(r["B_candidate_dev_change"]), pp(r["C_candidate_dev_change"]), f"{100*r['same_common_task_change_abs_max']:.2f}"] for r in strongest]
    common_max = max(abs(r["B_minus_C_dev_accuracy"]) for r in common)
    effects_table = [[r["scope"], r["count"], r["improved_at_least_5pp"], r["decreased_at_least_5pp"], r["within_5pp"]] for r in effects]
    cost_table = [[r["kind"], "新增" if r["new_compute"] else "历史复用", r["runs"], f"{r['updates']:,}", f"{r['input_tokens']:,}", f"{r['supervised_tokens']:,}", f"{r['seconds']/60:.2f}"] for r in compute]
    new_seconds = sum(r["seconds"] for r in compute if r["new_compute"])
    stage_seconds = sum(r["seconds"] for r in timing)
    unique_lr_counts = Counter(num(r, "lr") for r in selected)
    ce_worse = sum(r["dev_answer_ce_change"] > 0 for r in late)
    acc_worse = sum(r["dev_accuracy_change"] < 0 for r in late)
    text = f"""# Pilot v4：背景任务是否改变组成效应

完成两个探索性目标、三个预先冻结背景、三个配对块、固定256目标训练样本的有限比较。主结果为42个唯一开发选择；共享A仅训练/选择一次，加入三个背景比较后展示54行，不能计作54个独立训练。全部126个学习率候选的best和endpoint在同一新测试协议下评估，252行全部保留，不按测试重选。

本轮最清楚的现象是：固定点在原背景的负迁移可随背景改变，LDR在g1的原有收益消失，但这不能由候选源的终点准确率单独解释。下文同时保留效应幅度、逐块差异、训练终点和反例；n=3仅作描述，不作为背景、任务或初始化的一般因果结论。

## 主结果与背景交互

准确率为三个配对块均值；差值以百分点计，±为逐块差值的样本SD，不是置信区间。背景间A为同一权重与同一测试预测，均值重复显示仅便于对照。

{table(['目标','背景','A','B','C','B−A均值±SD','C−A均值±SD','B−C均值±SD'], main_table)}

{cases['target_text']}

![逐块准确率]({link('figures/accuracy_by_background_seed.png')})

![背景上的组成差值]({link('figures/contrasts_by_background_seed.png')})

交互定义为Δ(B−C)=(B−C)_后背景−(B−C)_前背景，按同一配对块和相同新测试输入相减。正数表示后背景更有利于B相对C，不等于B本身必然改善；可同时检查B−A和C−A。

{table(['目标','后背景−前背景',*[f'块{s}交互pp' for s in cfg['seeds']],'交互均值±SD(pp)','负/零/正块数','B−C符号翻转块数'], interaction_table)}

{' '.join(signs)} 符号记录须结合幅度解读：LDR的g1块17 B−C仅−0.0488pp，等于2048个测试样本中的一个；g2块42 B−C为+0.6836pp，而开发B−C为−0.3906pp。这些接近零的差异不足以支持稳定优劣，不能用符号翻转计数代替效应量。

![背景交互]({link('figures/background_interactions.png')})

42项唯一主结果如下（[CSV]({link('results_selected.csv')})）；A的background为shared。每种子84行全部候选原始结果见报告末链接。多数基线使用该块256条目标训练数据的众数，在测试集计算命中率。

{table(['目标','背景','配对块','条件','新测试准确率','多数基线','选择LR','选择步','训练来源'], raw_table)}

{cases['fit_text']}

{table(['块','条件','选择LR','选择步','所选train','终点train','所选dev','终点dev','所选test','终点test'], cases['fit_table'])}

## 有限设计与既有数据平台

案例fixed_points和longest_decreasing_run根据v3结果选定，属于探索性案例；不代表完整任务库，也不是独立于v3的目标选择。相关/替换候选保持v3不变：fixed_points为anti_fixed_points/successions，longest_decreasing_run为longest_increasing_run/left_to_right_maxima。两条目标—相关源关系都使用补值C(p)_i=n+1−p_i及恒等答案映射：target(p)=related(C(p))；不覆盖所有数学关系。

{table(['目标','背景','B候选slot0','C候选slot0','公共slot1','公共slot2','公共slot3'], bg_table)}

g0_v3保留原背景；g1_recoils只将公共slot1的descents换成recoils；g2_deficiencies只将公共slot2的peaks换成deficiencies。新增背景在本轮迁移训练/测试之前依据定义及既有单任务参照冻结，没有声称难度或答案分布完全匹配。每个背景内部B/C仅slot0不同，公共三个slot相同。跨背景则任务本身和集合内部已知关系结构也可能变化，不能将效应仅归因于候选与目标的直接关系，也不能称C数学上无关。目录、输入变换/答案映射和复用检查分别保存在实验根目录TASK_CATALOG.csv、DATA_PLATFORM.md及references/REUSE_AUDIT.md。

冻结的 [BACKGROUND_MATCHING.csv]({matching_link}) 显示：g1被换任务的源训练答案分布TV为{tv_ranges['g1_recoils'][0]:.5f}–{tv_ranges['g1_recoils'][1]:.5f}，但recoils的单任务早期学习明显慢于descents；g2的TV为{tv_ranges['g2_deficiencies'][0]:.5f}–{tv_ranges['g2_deficiencies'][1]:.5f}，答案分布差异很大，早期学习轨迹也不同。总体等分布不等于有限池直方图完全一致，更不等于优化难度匹配。下表是冻结前已有记录的描述，背景没有根据这些数值搜索最优得分。

{table(['块','背景','替换','答案TV','原任务dev1200','新任务dev1200','原任务dev4800','新任务dev4800','曲线绝对差均值pp'], matching_table)}

同样需要保留B/C候选对本身的剩余差异，不能只核对公共背景替换。以下 [candidate_matching.csv]({link('candidate_matching.csv')}) 从冻结的源答案直方图和既有单任务参考曲线计算，三背景共用同一候选对；没有使用本轮新增预训练、适配或测试来挑候选。原选择证据为 [v3 replacement_matching.csv]({previous_matching_link})：fixed_points保持历史anti_fixed_points/successions锚点，longest_decreasing_run沿用v3基于seed17开发记录选定的longest_increasing_run/left_to_right_maxima；本轮只是描述其三个配对块上的匹配程度。

{table(['目标','块','B候选','C候选','答案TV','B单dev1200','C单dev1200','B单dev4800','C单dev4800','曲线绝对差均值pp'], candidate_table)}

这张表与背景匹配表分别描述两类替换：TV接近不保证学习速度相同，4800步都学会也不保证早期轨迹一致。它们是保留的混杂说明，不是本轮重新优化组成的依据。

组合冻结后、查看本轮新迁移结果之前，定义级科学复核补充了原目录未覆盖的关系；未改组合、数据、预算或选择规则，详见 [catalog_addenda/README.md]({addendum_link})。记D=deficiencies，I=反演，R=一行反序，C=补值，则 fixed_points(p)=n−D(p)−D(I(p))，anti_fixed_points(p)=n−D(C(p))−D(R(I(p)))。因此fixed_points的g2中B/C都含有能通过多个变换输入联合重建目标的背景任务，不能把C理解为不含目标数学联系；单个输入的一个D标签不足以确定目标。正式训练不提供这种配对查询、联合答案或关系损失，池内轨道隔离也不保留不同的变换等价输入。这是隐含数学联系，并非单标签泄漏或模型实际使用该恒等式的证据。

g1是移除descents再换成recoils，并非新增一对共存的descents/recoils源任务。对于longest_decreasing_run，LDR/LIR/descents/peaks共享局部比较符号序列：descents计下降边，LDR/LIR分别取最长连续下降/上升段，peaks计局部峰；这些标量答案不能互相决定。g1移除descents但保留peaks，g2保留descents但移除peaks并加入位置不等计数，对计算结构的改变不同。背景的边缘答案分布、学习动态、源—源/多输入关系和局部计算结构均随之变化，单一关系图边数不足以概括本实验。

模型/数据/预算保持v3兼容：PermWorld n=10、one-line encoding、Transformer 4层/256宽/8头/dropout0.1、188词表与绑定embedding/output，FP32权重/评估、BF16训练、TF32关闭。源池8192与源开发1024，目标训练256与开发1024，最终新测试每块2048。两个目标、各背景和A/B/C在块内共享输入池；此共享并不使跨目标或背景成为独立样本。

配对块标签17/42/73联合索引初始化和数据池，不能当作数据生成RNG seed：s17/s42源于顺序使用RNG20261018，s73源于RNG2026101803；模型初始化seed分别17/42/73。预训练采样/dropout seed=块标签+1000，适配为+2000，新测试RNG2026101805。训练/开发池及step0权重逐字节复用v3，seed_provenance.json记录来源。块间差异混合输入、初始化及优化随机性，本轮未独立干预这些因素。

## 源学习、暴露与解释限制

每预训练run4800步，batch64的slot0…3每步精确16条，四任务采样比例各25%。各slot累计76800次暴露；input/label/index stream hash逐slot核对，同块所有背景/条件共享输入索引及slot赋值。对固定数学任务的相同slot，标签流也相同。完整144行 [source_exposures.csv]({link('source_exposures.csv')}) 保留不同输入数、真实暴露、tokens和hash；不能将76800次暴露当76800个不同输入。

全部源任务train/dev准确率、answer/EOS/two-token CE、按n=10多数基线和曲线保留，[source_learning_table.csv]({link('source_learning_table.csv')})另连接v3相同输入/初始化的单任务1200/4800步参照。没有新增单任务计算。单任务4800与组合4800具有相同总更新数及全局LR轨迹，但每任务暴露是组合4倍，任务交替也不同；单任务1200只匹配每任务暴露，LR历史和总更新不同，无法据此孤立识别纯干扰。

相对单任务4800步的源开发差异，改善和下降都保留。±5pp仅为描述门槛，去重按seed×任务×source终点state hash；同一任务在多个组合的出现次数并非独立样本。36逻辑源run共有{state_count}个不同终点state hash。

{table(['范围','记录数','改善≥5pp','下降≥5pp','绝对差<5pp'], effects_table)}

背景内部相同公共任务B/C开发准确率差的绝对值最大为{100*common_max:.2f}pp。跨背景逐slot检查：[source_background_slot_changes.csv]({link('source_background_slot_changes.csv')})对同一数学任务计算变化；若slot内任务被替换，仅展示两边任务和指标，差值留空，避免把两个任务的准确率差当成同一任务学习改变。

以下为按目标交互绝对值排序的最多四个描述性案例。候选B/C源开发变化是同一任务在后背景减前背景；公共源列只统计未换任务的slot。这里只并列保存状态变化，不拟合相关系数、不作因果归因，也未用这些表追加实验或选择主结果。

{table(['目标','块','后−前背景','目标交互pp','B候选源dev变化pp','C候选源dev变化pp','同任务公共源最大绝对变化pp'], context_table)}

下面进一步列出两个B候选的源学习状态，数列按17/42/73排列。单1200步只匹配每任务暴露，LR历史和总更新仍不同。

{table(['目标','背景','B候选源','组合train三块','组合dev三块','单1200 dev三块'], cases['source_table'])}

{cases['source_text']}

g0复用源曲线有历史空缺：fixed_points块17/42只有2400及3000/3600/4200/4800全量train/dev记录；对应图线从2400开始，未补造早期点。新训练run及其余兼容v3记录有0/600/1200/2400/3600/4800步指标。按长度参照只有n=10，不宣称长度外推。

## 适配选择、隔离及预算

B/C各4800步，源LR为100步warmup至3e−4、2400步余弦到3e−5、随后常数3e−5至4800；不再延长。AdamW(weight_decay=.01, betas=.9/.999, eps=1e−8)，梯度裁剪1。每适配条件同三档LR（3e−4/1e−4/3e−5）、每档800步、warmup100后余弦到初始LR的0.1；所有块内候选共享适配数据及采样顺序。目标已有专有token的绑定embedding/output行恢复同块step0值，严格加载预训练权重，重置优化器，无新输出头。

每100步评估全量训练/开发，依次按最高开发答案准确率、最低开发answer CE、最早步数、声明LR顺序选择，step0排除。答案准确率在完整词表prompt-only argmax计算；答案与EOS各占训练loss一半。42项唯一选择的LR计数为{dict(sorted(unique_lr_counts.items(), reverse=True))}；全部候选400→800固定窗口有{ce_worse}/126开发CE上升、{acc_worse}/126开发准确率下降，保留最佳开发与训练终点对照，不把选择后差距直接解释为独立过拟合证据。

FROZEN_PROTOCOL锁定六个目标/背景组成、源码与既有证据；SELECTIONS_FROZEN锁定42项选择及checkpoint hash后才生成独立新测试。新测试排除v1/v2/v3所有输入及v4训练/开发的inverse/reverse/complement完整轨道，三个新测试池也隔离；块内目标/背景/条件统一评估。测试不用于选择背景、LR或checkpoint。{audit_text}；完整证据见FINAL_AUDIT.json。

预训练36逻辑run中12复用、24新增；126适配候选中18个共享A与36个g0 B/C复用，72个新增。精确复用基于有序任务tuple、输入池、初始化、采样及预算兼容，不仅比较无序任务集合。历史权重在新测试上重评估，不是本轮重新训练的独立重复。所有252候选best/endpoint测试记录保留；测试次数及其评估开销在本轮阶段时间中，不计入训练token数。

{table(['阶段','来源','run数','更新数','训练input tokens','监督tokens','训练timer分钟'], cost_table)}

本轮新增训练timer合计{new_seconds/60:.2f}分钟，引擎阶段墙钟合计{stage_seconds/60:.2f}分钟。timer包含训练、全量train/dev评估、保存及部分记账，复用timer来自历史阶段且范围略异，不计本轮新增；两种时间不能相加。审计与绘图另行运行，不在引擎阶段计时。输入tokens计完整28-token训练前向序列，监督tokens计答案/EOS两个位置；不含评估tokens。本轮单任务参照为既有记录复用，不再付训练成本。

## 结论边界与下一步

这是一项两个经v3结果挑选案例、三个背景、三个配对块的有限探索。背景和任务共享数据、初始化、来源权重；样本数固定256，预训练固定4800、适配每档800，不建立样本效率曲线或scaling law。非正收益、负迁移、方向翻转以及未充分学会的源/目标均保留。目标先前在8192条单任务数据下通过可学习性门槛，不能保证在256条条件下也学会；需要结合多数基线解读小样本结果。

两个目标每个背景下的C具有相同无序任务集合，只是successions与left_to_right_maxima在slot0/slot3互换。相同索引/slot流下，任务分配到的具体输入随之不同，因此C的训练权重/表现可以不同；目标任务没有进入预训练输入，不能把跨目标C差异解释成目标名称改变了预训练。这也进一步限制两个案例的独立性。

下一阶段优先完善“关系类型×背景计算结构”的任务目录和有限组合对照，而非直接增加源任务数量。当前证据是：固定点负迁移可在g2改变，LDR收益在g1消失，公共源学习同时变化，且候选源准确率与目标收益存在反例。候选库宜明确覆盖以下类别，把分布匹配、学习曲线匹配和关系类别分开标注：

1. **直接可逆变换及答案映射。** 保留complement/reverse/inverse下恒等答案关系作为锚点，增加非恒等仿射映射参照，例如同一descents任务在补值输入上满足d(Cp)=n−1−d(p)。这是同任务的变换/映射变体，不计作新独立任务。
2. **边缘等分布但计算结构不同。** 以现有descents/recoils/exceedances等Eulerian统计作候选，区分one-line下邻接位置比较、逆位置比较和对角线位置比较；g1表明TV接近仍会伴随早期学习和组合效果差异，不能只匹配答案直方图。
3. **多源或多输入联合关系。** 标注fixed/exceedances/deficiencies的位置划分、D(p)与D(Ip)联合恢复fixed，以及cycle_count=fixed_points+nontrivial_cycle_count；区分同输入多源答案与变换输入多查询。以后若检验模型是否利用这些关系，须另行冻结成对输入/关系约束干预，不能拿本轮隐含关系当已利用机制。
4. **共享局部计算结构但标量不互定。** 用已有LIR/LDR/descents/peaks/valleys及局部三元组计数组织背景，区分共同中间比较序列、等式与上下界，避免将共同计算统计的保留/删除简化成关系边数变化。
5. **单向信息关系及其他计算族。** 现有cycle_count到parity的模2映射、cycle分解相关计数可提供不同于局部/位置统计的结构，但应先复核单任务可学习性；v3的cycle_count候选曾未达学习门槛，不能直接把未学会任务当已匹配对照。

组合层面先保留固定四源数、固定预算和有序slot配对，明确哪些组仅改边缘分布/学习速度，哪些组同时改变局部结构或联合关系。若跨目标比较相同C集合，还应单独处理slot到输入的分配。以上为后续研究建议，并未执行新目标、成对查询、slot控制或训练；不依据本轮候选测试最高分再搜索最优背景。

本轮已按预设规模停止；没有追加学习率、预算、背景或目标，等待后续指导。

## 文件

- [42唯一主结果]({link('results_selected.csv')})、[54背景对照展示行]({link('results_display_ABC.csv')})、[252全部候选]({link('results_all_candidates.csv')})。
- 每块原始CSV：[17]({link('results_seed17.csv')})、[42]({link('results_seed42.csv')})、[73]({link('results_seed73.csv')})。
- [配对组成差值]({link('paired_deltas.csv')})、[背景交互逐块]({link('background_interactions.csv')})、[交互汇总]({link('background_interactions_summary.csv')})、[终点对照]({link('selected_vs_endpoint.csv')})。
- [源学习/单任务参照]({link('source_learning_table.csv')})、[公共源B/C差异]({link('common_source_BC.csv')})、[源与目标交互并列记录]({link('interaction_source_context.csv')})。
- [沿用B/C候选对的分布与难度差异]({link('candidate_matching.csv')})，选择来源为 [v3冻结前的匹配记录]({previous_matching_link})。
- [完整目标曲线]({link('target_curves.csv')})、[源曲线]({link('source_curves.csv')})、[冻结配置]({link('FROZEN_PROTOCOL.json')})、[成本]({link('compute_summary.csv')})、[阶段耗时]({link('stage_timings.csv')})；复现命令见README.md。
- [独立统计重算与CSV绑定审计]({link('STATISTICS_AUDIT.json')})，与权重/协议最终审计分开记录。
- [冻结后的定义级关系复核补充]({addendum_link})；它在查看本轮新迁移结果之前完成，不触发新实验，也不改写冻结目录。
"""
    report_path.write_text(text, encoding="utf-8")
    (out / "key_results.json").write_text(json.dumps(dict(selected_unique=42, display_rows=54, candidate_rows=252, target_samples=256,
        paired_blocks=cfg["seeds"], summary=summary, interactions=interaction_summary, source_unique_states=state_count,
        new_training_timer_seconds=new_seconds, engine_stage_seconds=stage_seconds, audit_preservation_passed=preserved), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    assert (out / "TEST_COMPLETE.json").exists(), "Experiment incomplete: no completed-results report generated."
    complete, protocol = js(out / "TEST_COMPLETE.json"), js(out / "FROZEN_PROTOCOL.json")
    cfg = protocol["config"]
    assert complete["selected_rows"] == 42 and complete["candidate_rows"] == 252
    selected, candidates = rows(out / "results_selected.csv"), rows(out / "results_all_candidates.csv")
    assert len(selected) == 42 and len(candidates) == 252 and sum(truth(r["selected_primary"]) for r in candidates) == 42
    assert all(truth(r["selected_primary"]) for r in selected)
    assert sum(r["condition"] == "A" for r in selected) == 6 and all(r["background"] == "shared" for r in selected if r["condition"] == "A")
    (out / "figures").mkdir(exist_ok=True)
    expanded, summary, deltas, delta_summary, interactions, interaction_summary, endpoints = result_tables(out, cfg, selected, candidates)
    result_plots(out, cfg, expanded, deltas, interactions, interaction_summary)
    curve, metas, learning, common, changes, contexts, effects = source_tables(out, cfg, interactions)
    source_plots(out, cfg, curve)
    late = target_plots_and_late(out, cfg, js(out / "SELECTIONS_FROZEN.json")["selections"])
    compute, timing, state_count = compute_tables(out, metas)
    path = report(out, cfg, selected, summary, deltas, delta_summary, interactions, interaction_summary, learning, common, contexts, effects, compute, timing, late, state_count)
    print(json.dumps(dict(status="complete", report=str(path), unique_selected=42, candidate_rows=252, figures=len(list((out / "figures").glob("*.png")))), ensure_ascii=False))


if __name__ == "__main__":
    main()
