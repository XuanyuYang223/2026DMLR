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
        f"The following sequences are ordered by paired block {'/'.join (map (str, seeds))}. For fixed_points, B−A and C−A are negative in all 3/3 blocks in g0/g1; "
        f"in g2, B−A becomes {vector (dv('fixed_points','g2_deficiencies','B-A'))}pp and C−A is {vector (dv('fixed_points','g2_deficiencies','C-A'))}pp. "
        "The original fixed_points negative transfer depends on the background and cannot be generalized as an intrinsic property of the two candidate compositions. The g2 B gain in block42 is only about 0.24pp; three matching signs do not establish a universal benefit. "
        "g2 improves both B and C, and both contain the multi-input joint target relation involving deficiencies. The absolute transfer improvement cannot be attributed entirely to the directly related candidate in B.\n\n"
        f"For longest_decreasing_run, g0 B−A is {vector (dv('longest_decreasing_run','g0_v3','B-A'))}pp; "
        f"it contracts to {vector (dv('longest_decreasing_run','g1_recoils','B-A'))}pp in g1 and is {vector (dv('longest_decreasing_run','g2_deficiencies','B-A'))}pp in g2. "
        "Both B/C in g1 are near the test baseline corresponding to the training mode, and the original g0 gains in blocks 17/73 disappear. g2 retains some benefit, while block42 changes from no benefit to a small positive gain."
    )
    fits = [r for r in rows(out/'selected_vs_endpoint.csv') if r['target']=='longest_decreasing_run' and r['background']=='g1_recoils']
    fit_text = (
        "LDR adaptation in g1 is not unable to fit throughout training. All six development selections return to early checkpoints at LR=3e−5, step100; "
        f"selected training accuracy is {pct (min (num (r,'selected_train_accuracy') for r in fits))}–{pct (max (num (r,'selected_train_accuracy') for r in fits))}, "
        f"whereas at the same LR the step800 endpoint training accuracy rises to {pct (min (num (r,'endpoint_train_accuracy') for r in fits))}–{pct (max (num (r,'endpoint_train_accuracy') for r in fits))}, "
        f"but endpoint test accuracy falls to {pct (min (num (r,'endpoint_test_accuracy') for r in fits))}–{pct (max (num (r,'endpoint_test_accuracy') for r in fits))}. "
        "This supports improved late training fit with worse generalization and selection returning to an early model near baseline. Selected training accuracy near the mode must not be described as failure to learn the training data throughout the run."
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
        examples.append (f"fixed_points block{seed}, g0 to {bg}: anti_fixed_points source dev {pct (old['multi4800_dev_accuracy'])}→{pct (new['multi4800_dev_accuracy'])}, target B test {pct (si['fixed_points','g0_v3', seed,'B']['test_accuracy'])}→{pct (si['fixed_points', bg, seed,'B']['test_accuracy'])}")
    success_source=[num(li['fixed_points','g2_deficiencies',seed,'C','successions'],'multi4800_dev_accuracy')-num(li['fixed_points','g0_v3',seed,'C','successions'],'multi4800_dev_accuracy') for seed in seeds]
    success_target=[num(si['fixed_points','g2_deficiencies',seed,'C'],'test_accuracy')-num(si['fixed_points','g0_v3',seed,'C'],'test_accuracy') for seed in seeds]
    common_examples=[]
    for task in ('descents','successions','deficiencies'):
        values=[next(r['B_minus_C_dev_accuracy'] for r in common if r['target']=='longest_decreasing_run' and r['background']=='g2_deficiencies' and r['seed']==seed and r['task']==task) for seed in seeds]
        common_examples.append (f"{task}: {vector (values)}pp")
    improvements=[r for r in learning if r['target']=='longest_decreasing_run' and r['condition']=='B' and r['task']=='successions' and r['multi_minus_single4800_dev_accuracy']>=.05]
    source_text=(
        f"For peaks, g1−g0 lowers development accuracy in {sum (v<0 for v in peak_changes)}/{len (peak_changes)} entries across two targets × B/C × three blocks. "
        "The LDR-related source LIR has low train/dev accuracy in g1, below its single-task step1200 reference. This accompanies loss of target benefit in blocks 17/73, but block42 had no original benefit, so target performance does not decline in every block.\n\n"
        "Candidate-source endpoint accuracy alone cannot explain transfer: "+'; '.join (examples)+'. '
        f"For fixed_points g2 versus g0, C candidate successions source dev changes by {vector (success_source)}pp while C target test changes by {vector (success_target)}pp; in all three blocks, the candidate source worsens while the target improves.\n\n"
        "LDR g2 common-source B−C development differences: "+'; '.join (common_examples)+'. '
        "Target B/C differences thus accompany changes in common descents/successions learning, while deficiencies is almost unchanged; these cannot be decomposed into independent mechanisms. Compositions do not only harm sources: improvements in LDR-B successions over single-task step4800 include "+
        '; '.join (f"{r['background']} block{r['seed']} {pp (r['multi_minus_single4800_dev_accuracy'])}pp" for r in improvements)+
        ". Unequal per-task exposure, task alternation, and optimization differences across three blocks still limit interference/synergy attribution."
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
    audit_text = f"Final audit passed; byte hashes checked for {preserved_count} historical files" if preserved else "Final audit and historical byte-preservation conclusion remain pending; see FINAL_AUDIT.json"
    main_table = []
    for target in cfg["targets"]:
        for bg in cfg["backgrounds"]:
            s = {r["condition"]: r for r in summary if r["target"] == target and r["background"] == bg}
            d = {r["contrast"]: r for r in delta_summary if r["target"] == target and r["background"] == bg}
            main_table.append([target, bg, *[pct(s[c]["test_accuracy_mean"]) for c in "ABC"],
                               *[f"{pp(d[c]['delta_mean'])} ± {100*d[c]['delta_sample_sd']:.2f}" for c in ("B-A", "C-A", "B-C")]])
    interaction_table = [[r["target"], r["interaction"], *[pp(next(x['interaction_test_accuracy'] for x in interactions if x['target']==r['target'] and x['interaction']==r['interaction'] and x['seed']==seed)) for seed in cfg['seeds']], f"{pp(r['interaction_mean'])} ± {100*r['interaction_sample_sd']:.2f}",
                          f"{r['negative_seeds']}/{r['zero_seeds']}/{r['positive_seeds']}", r["BC_sign_reversal_seeds"]] for r in interaction_summary]
    raw_table = [[r["target"], r["background"], r["seed"], r["condition"], pct (r["test_accuracy"]), pct (r["majority_accuracy"]), r["lr"], r["adapt_step"], "Reused" if not truth (r["new_compute"]) else "New"] for r in selected]
    bg_table = [[g["target"], g["background"], g["related"], g["replacement"], *g["common"]] for g in cfg["groups"]]
    signs = []
    for target in cfg["targets"]:
        crossing = []
        for seed in cfg["seeds"]:
            values = [r["test_accuracy_delta"] for r in deltas if r["target"] == target and r["seed"] == seed and r["contrast"] == "B-C"]
            if min(values) < 0 < max(values):
                crossing.append(seed)
        consistent = [r["interaction"] for r in interaction_summary if r["target"] == target and (r["negative_seeds"] == 3 or r["positive_seeds"] == 3)]
        signs.append (f"{target}: paired blocks where B−C changes sign across backgrounds: {crossing or 'none'}; background interactions with the same direction in all three blocks: {consistent or 'none'}.")
    strongest = sorted(contexts, key=lambda r: abs(r["interaction_test_accuracy"]), reverse=True)[:4]
    context_table = [[r["target"], r["seed"], r["interaction"], pp(r["interaction_test_accuracy"]),
                      pp(r["B_candidate_dev_change"]), pp(r["C_candidate_dev_change"]), f"{100*r['same_common_task_change_abs_max']:.2f}"] for r in strongest]
    common_max = max(abs(r["B_minus_C_dev_accuracy"]) for r in common)
    effects_table = [[r["scope"], r["count"], r["improved_at_least_5pp"], r["decreased_at_least_5pp"], r["within_5pp"]] for r in effects]
    cost_table = [[r["kind"], "New" if r["new_compute"] else "Historical reuse", r["runs"], f"{r['updates']:,}", f"{r['input_tokens']:,}", f"{r['supervised_tokens']:,}", f"{r['seconds']/60:.2f}"] for r in compute]
    new_seconds = sum(r["seconds"] for r in compute if r["new_compute"])
    stage_seconds = sum(r["seconds"] for r in timing)
    unique_lr_counts = Counter(num(r, "lr") for r in selected)
    ce_worse = sum(r["dev_answer_ce_change"] > 0 for r in late)
    acc_worse = sum(r["dev_accuracy_change"] < 0 for r in late)
    text = f"""# Pilot v4: do background tasks change composition effects?

A limited comparison is complete for two exploratory targets, three prespecified backgrounds, three paired blocks, and 256 fixed target-training examples. There are 42 unique development-selected primary results. Shared A is trained/selected once; showing it in three background comparisons produces 54 display rows, not 54 independent training runs. Best and endpoint checkpoints for all 126 learning-rate candidates were evaluated under the same new-test protocol. All 252 rows are retained, with no test-based reselection.

The clearest observation is that fixed_points negative transfer in the original background can change with the background, while the original LDR benefit disappears in g1. Candidate-source endpoint accuracy alone cannot explain this. Effect sizes, block-level differences, training endpoints, and counterexamples are all retained below. With n=3 these are descriptive observations, not general causal conclusions about backgrounds, tasks, or initialization.

## Primary results and background interactions

Accuracies are means over three paired blocks; differences are percentage points, and ± denotes the sample SD of block-level differences, not a confidence interval. A uses the same weights and test predictions across backgrounds; its mean is repeated only for comparison.

{table(['Target','Background','A','B','C','B−A mean±SD','C−A mean±SD','B−C mean±SD'], main_table)}

{cases['target_text']}

![Accuracy by block]({link('figures/accuracy_by_background_seed.png')})

![Composition contrasts across backgrounds]({link('figures/contrasts_by_background_seed.png')})

The interaction is Δ(B−C)=(B−C)_later_background−(B−C)_earlier_background, differenced within the same paired block on identical new-test inputs. Positive values mean the later background favors B relative to C, not necessarily that B itself improves; B−A and C−A provide additional context.

{table(['Target','Later−earlier background',*[f'Block{s} interaction pp' for s in cfg['seeds']],'Interaction mean±SD (pp)','Negative/zero/positive blocks','Blocks with B−C sign reversal'], interaction_table)}

{' '.join (signs)} Signs must be interpreted alongside magnitude: LDR g1 block17 B−C is only −0.0488pp, one of 2048 test examples; g2 block42 B−C is +0.6836pp, whereas development B−C is −0.3906pp. These near-zero differences do not establish stable superiority. Sign-flip counts cannot replace effect sizes.

![Background interaction]({link('figures/background_interactions.png')})

The 42 unique primary results are below ([CSV]({link('results_selected.csv')})); A has background=shared. Links at the end provide all 84 candidate rows per seed. The majority baseline predicts the mode of that block's 256 target-training labels and measures accuracy on the test set.

{table(['Target','Background','Paired block','Condition','New-test accuracy','Majority baseline','Selected LR','Selected step','Training origin'], raw_table)}

{cases['fit_text']}

{table(['Block','Condition','Selected LR','Selected step','Selected train','Endpoint train','Selected dev','Endpoint dev','Selected test','Endpoint test'], cases['fit_table'])}

## Limited design and existing data platform

The fixed_points and longest_decreasing_run cases were selected from v3 results and are exploratory. They do not represent the full task library, and target selection is not independent of v3. Related/replacement candidates remain unchanged: anti_fixed_points/successions for fixed_points, and longest_increasing_run/left_to_right_maxima for longest_decreasing_run. Both target–related-source relations use complement C(p)_i=n+1−p_i and an identity answer map: target(p)=related(C(p)). They do not cover all mathematical relations.

{table(['Target','Background','B candidate slot0','C candidate slot0','Common slot1','Common slot2','Common slot3'], bg_table)}

g0_v3 preserves the original background; g1_recoils replaces only common slot1 descents with recoils; g2_deficiencies replaces only common slot2 peaks with deficiencies. New backgrounds were frozen from definitions and existing single-task references before this round's transfer training/testing, without claiming fully matched difficulty or answer distributions. Within each background only slot0 differs between B/C, and the three common slots match. Across backgrounds, task definitions and known relations within the set can both change. Effects cannot be attributed solely to the direct candidate–target relation, and C must not be called mathematically unrelated. The catalog, input transformations/answer maps, and reuse checks are in TASK_CATALOG.csv, DATA_PLATFORM.md, and references/REUSE_AUDIT.md in the experiment directory.

Frozen [BACKGROUND_MATCHING.csv]({matching_link}) shows source-training answer-distribution TV of {tv_ranges['g1_recoils'][0]:.5f}–{tv_ranges['g1_recoils'][1]:.5f} for the g1 replacement, but early single-task learning is much slower for recoils than descents. g2 TV is {tv_ranges['g2_deficiencies'][0]:.5f}–{tv_ranges['g2_deficiencies'][1]:.5f}, a large distribution difference, with different early learning trajectories as well. Equal population distributions do not imply identical finite-pool histograms or matched optimization difficulty. The table describes existing records before freezing; backgrounds were not searched for an optimal score using these values.

{table(['Block','Background','Replacement','Answer TV','Original dev1200','New dev1200','Original dev4800','New dev4800','Mean absolute curve gap pp'], matching_table)}

Remaining differences within the B/C candidate pairs must also be retained, beyond checking common-background replacements. [candidate_matching.csv]({link('candidate_matching.csv')}) below uses frozen source-answer histograms and existing single-task reference curves; the same pair is used in all three backgrounds. This round's new pretraining, adaptation, and test results were not used to choose candidates. The original selection evidence is [v3 replacement_matching.csv]({previous_matching_link}): fixed_points retains the historical anti_fixed_points/successions anchor, and longest_decreasing_run retains the longest_increasing_run/left_to_right_maxima pair selected in v3 using seed17 development records. This round only describes matching across the three paired blocks.

{table(['Target','Block','B candidate','C candidate','Answer TV','B single dev1200','C single dev1200','B single dev4800','C single dev4800','Mean absolute curve gap pp'], candidate_table)}

This table and the background-matching table describe two types of replacement. Similar TV does not guarantee equal learning speed; both tasks being learned by step4800 does not guarantee matching early trajectories. These are retained confounding differences, not grounds for reoptimizing composition in this round.

After composition freezing and before inspecting new transfer results, a definition-level scientific review added relations absent from the original catalog; compositions, data, budgets, and selection rules were unchanged. See [catalog_addenda/README.md]({addendum_link}). Let D=deficiencies, I=inverse, R=one-line reversal, and C=complement. Then fixed_points(p)=n−D(p)−D(I(p)), and anti_fixed_points(p)=n−D(C(p))−D(R(I(p))). Thus both B/C in fixed_points g2 contain a background task that jointly reconstructs the target across transformed inputs; C cannot be interpreted as lacking target relations. One D label on one input does not determine the target. Training supplies no such paired queries, joint answers, or relation loss, and within-pool orbit isolation does not retain distinct transformation-equivalent inputs. This is an implicit mathematical relation, not single-label leakage or evidence that the model used the identity.

g1 removes descents and replaces it with recoils; it does not add a coexisting descents/recoils pair. For longest_decreasing_run, LDR/LIR/descents/peaks share the local comparison-sign sequence: descents counts descending edges, LDR/LIR take the longest consecutive descending/ascending segment, and peaks counts local peaks. These scalar answers do not determine each other. g1 removes descents but retains peaks; g2 retains descents, removes peaks, and adds a position-inequality count. They alter computational structure differently. Marginal answer distributions, learning dynamics, source–source/multi-input relations, and local computational structure all change; a single relation-graph edge count cannot summarize this experiment.

Model/data/budgets remain v3-compatible: PermWorld n=10, one-line encoding, Transformer with 4 layers/width256/8 heads/dropout0.1, vocabulary188, and tied embedding/output; FP32 weights/evaluation, BF16 training, TF32 disabled. Source train/dev pools have 8192/1024 inputs, target train/dev have 256/1024, and the final new test has 2048 per block. Targets, backgrounds, and A/B/C share input pools within a block; this sharing does not make targets or backgrounds independent samples.

Paired-block labels 17/42/73 jointly index initialization and data pools; they are not data-generation RNG seeds. s17/s42 were generated sequentially with RNG 20261018, and s73 with RNG 2026101803; model initialization seeds are 17/42/73. Pretraining sampling/dropout seed=block label +1000, adaptation=+2000, and new-test RNG 2026101805. Training/development pools and step0 weights are reused byte-for-byte from v3; seed_provenance.json records their origins. Block differences combine input, initialization, and optimization randomness; these factors were not independently manipulated.

## Source learning, exposure, and limits of interpretation

Each pretraining run has 4800 steps. Each batch of 64 contains exactly 16 examples in each slot0…3, giving exactly 25% task sampling. Each slot receives 76800 exposures. Input/label/index-stream hashes are checked per slot; all backgrounds/conditions in a block share input indices and slot assignment. The same mathematical task in the same slot also has the same label stream. All 144 rows of [source_exposures.csv]({link('source_exposures.csv')}) retain unique-input counts, actual exposures, tokens, and hashes. 76800 exposures are not 76800 distinct inputs.

All source train/dev accuracies, answer/EOS/two-token CE, n=10 majority baselines, and curves are retained. [source_learning_table.csv]({link('source_learning_table.csv')}) additionally links v3 single-task step1200/4800 references using identical inputs/initialization. No new single-task computation was performed. Single-task4800 and multitask 4800 have the same total updates and global LR trajectory, but single-task per-task exposure is four times larger, with different task alternation. Single-task1200 matches only per-task exposure; LR history and total updates differ. These comparisons cannot isolate pure interference.

Both increases and decreases in source-development accuracy relative to single-task4800 are retained. ±5pp is only a descriptive threshold; deduplication uses seed×task×source-endpoint-state hash. Multiple appearances of a task in compositions are not independent samples. The 36 logical source runs have {state_count} distinct endpoint-state hashes.

{table(['Scope','Records','Improvement≥5pp','Decline≥5pp','Absolute gap<5pp'], effects_table)}

Within-background B/C development differences for the same common task reach an absolute maximum of {100*common_max:.2f}pp. Across backgrounds, [source_background_slot_changes.csv]({link('source_background_slot_changes.csv')}) computes changes only for the same mathematical task in each slot. When a slot's task is replaced, it shows both tasks and metrics but leaves the difference blank, avoiding interpretation of different-task accuracy differences as a change in learning the same task.

The following are up to four descriptive cases ranked by absolute target interaction. Candidate B/C source-development changes are later-minus-earlier background differences for the same task; common-source columns cover only unchanged slots. These tables juxtapose state changes without fitting correlations or making causal attributions, and were not used to add experiments or select primary results.

{table(['Target','Block','Later−earlier background','Target interaction pp','B candidate source dev change pp','C candidate source dev change pp','Max absolute same-task common-source change pp'], context_table)}

Source-learning states for the two B candidates are listed below in 17/42/73 order. Single-task1200 matches only per-task exposure; LR history and total updates still differ.

{table(['Target','Background','B candidate source','Multitask train across 3 blocks','Multitask dev across 3 blocks','Single1200 dev across 3 blocks'], cases['source_table'])}

{cases['source_text']}

Reused g0 source curves have historical gaps: fixed_points blocks 17/42 contain full train/dev records only at 2400 and 3000/3600/4200/4800. Their plots start at 2400; earlier points were not fabricated. Newly trained runs and other compatible v3 records contain metrics at 0/600/1200/2400/3600/4800. Length-specific references cover only n=10, with no length-extrapolation claim.

## Adaptation selection, isolation, and budget

B/C each receive 4800 steps. Source LR warms up for 100 steps to 3e−4, follows cosine decay to 3e−5 at 2400, then stays at 3e−5 through 4800; no extension was added. AdamW (weight_decay=.01, betas=.9/.999, eps=1e−8), gradient clipping 1. Every adaptation condition uses the same three LRs (3e−4/1e−4/3e−5),800 steps each,100-step warmup and cosine decay to 0.1 of initial LR. All candidates within a block share adaptation data and sampling order. The existing target token's tied embedding/output row is restored to the block's step0 value. Pretrained weights are strictly loaded, the optimizer is reset, and no new output head is added.

Full training/development evaluation occurs every 100 steps. Selection prioritizes highest development answer accuracy, lowest development answer CE, earliest step, then declared LR order; step0 is excluded. Answer accuracy uses prompt-only argmax over the full vocabulary; answer and EOS each contribute half of training loss. LR counts among 42 unique selections are {dict (sorted (unique_lr_counts.items(), reverse=True))}. Over the fixed400→800 window, {ce_worse}/126 candidates have increased development CE and {acc_worse}/126 decreased development accuracy. Best-development and endpoint comparisons are retained; post-selection gaps alone are not treated as independent evidence of overfitting.

FROZEN_PROTOCOL locks six target/background compositions, code, and prior evidence. Independent new tests are generated only after SELECTIONS_FROZEN locks 42 selections and checkpoint hashes. New tests exclude the full inverse/reverse/complement orbits of all v1/v2/v3 inputs and v4 train/dev inputs; the three new test pools are isolated from each other as well. Targets/backgrounds/conditions are evaluated uniformly within a block. Tests do not select backgrounds, LRs, or checkpoints. {audit_text}; see FINAL_AUDIT.json for complete evidence.

Of 36 logical pretraining runs,12 are reused and 24 new. Of 126 adaptation candidates,18 shared-A and 36 g0 B/C candidates are reused, and 72 are new. Exact reuse requires compatibility of ordered task tuples, input pools, initialization, sampling, and budgets, not merely unordered task sets. Historical weights are reevaluated on new tests, not retrained as independent repeats. All 252 best/endpoint candidate test records are retained. Test evaluation costs appear in stage timings, not training-token totals.

{table(['Stage','Origin','Runs','Updates','Training input tokens','Supervised tokens','Training timer minutes'], cost_table)}

New training timers total {new_seconds/60:.2f} minutes; engine-stage wall time totals {stage_seconds/60:.2f} minutes. Timers include training, full train/dev evaluation, saving, and some bookkeeping. Reuse timers come from historical stages with somewhat different scope and are not new compute. The two timing measures must not be added. Auditing and plotting run separately, outside engine-stage timing. Input tokens count the full 28-token training forward sequence; supervised tokens count answer/EOS positions; evaluation tokens are excluded. Single-task references are reused records with no additional training cost.

## Limits and next steps

This is a limited exploration of two cases selected from v3 results, three backgrounds, and three paired blocks. Backgrounds and tasks share data, initialization, and source weights. Target samples are fixed at 256, pretraining at 4800 steps, and adaptation at 800 per LR; this establishes neither a sample-efficiency curve nor a scaling law. Nonpositive gains, negative transfer, direction reversals, and insufficiently learned sources/targets are retained. Targets previously passed a learnability threshold with 8192 single-task examples; this does not ensure learning with 256. Small-sample outcomes require interpretation alongside majority baselines.

For each background, C has the same unordered task set for both targets, with successions and left_to_right_maxima swapped between slot0/slot3. Under identical index/slot streams, the specific inputs assigned to tasks therefore differ, so C weights/performance can differ. The target task never enters pretraining inputs; cross-target C differences cannot be interpreted as the target name changing pretraining. This further limits independence of the two cases.

The next priority is a task catalog and limited composition controls spanning relation type × background computational structure, rather than immediately increasing source-task count. Current evidence shows that fixed_points negative transfer can change in g2, LDR benefits disappear in g1, common-source learning changes concurrently, and candidate-source accuracy has counterexamples as a proxy for target gains. Candidate libraries should distinguish distribution matching, learning-curve matching, and relation categories:

1. **Direct invertible transformations and answer maps.** Retain identity-answer relations under complement/reverse/inverse as anchors; add nonidentity affine-map references, such as the same descents task on complemented inputs satisfying d(Cp)=n−1−d(p). These are transformation/map variants of one task, not new independent tasks.
2. **Equal marginal distributions with different computational structures.** Existing Eulerian statistics such as descents/recoils/exceedances can distinguish adjacent-position comparisons, inverse-position comparisons, and diagonal-position comparisons under one-line encoding. g1 shows that similar TV can coexist with different early learning and composition effects; matching answer histograms alone is insufficient.
3. **Multi-source or multi-input joint relations.** Register the fixed/exceedances/deficiencies position partition, joint recovery of fixed from D(p) and D(Ip), and cycle_count=fixed_points+nontrivial_cycle_count. Distinguish multiple source answers on one input from multiple queries on transformed inputs. Testing whether models use these relations requires separately frozen paired-input/relation-constraint interventions; implicit relations here are not evidence of an implemented mechanism.
4. **Shared local computational structure without mutually determining scalars.** Organize backgrounds using existing LIR/LDR/descents/peaks/valleys and local triple counts. Distinguish a shared intermediate comparison sequence, equalities, and bounds, rather than reducing retention/removal of shared computations to relation-edge counts.
5. **One-way information relations and other computational families.** The existing cycle_count-to-parity modulo 2 map and cycle-decomposition counts offer structures beyond local/position statistics, but require single-task learnability review first. The v3 cycle_count candidate failed its learning threshold; an unlearned task cannot simply be treated as a matched control.

At the composition level, retain four sources, fixed budgets, and ordered-slot pairing. Specify which groups change only marginal distributions/learning speed and which also change local structure or joint relations. Cross-target comparisons of identical C sets must separately address slot-to-input assignment. These are future recommendations: no new targets, paired queries, slot controls, or training were executed, and no search for the best background was performed using candidate test maxima.

This round stopped at the prespecified scale, with no additional learning rates, budgets, backgrounds, or targets, pending further guidance.

## Files

- [42 unique primary results]({link('results_selected.csv')}), [54 background-comparison display rows]({link('results_display_ABC.csv')}), [All 252 candidate results]({link('results_all_candidates.csv')}).
- Raw CSV per block: [17]({link('results_seed17.csv')}), [42]({link('results_seed42.csv')}), [73]({link('results_seed73.csv')}).
- [Paired composition differences]({link('paired_deltas.csv')}), [Background interactions by block]({link('background_interactions.csv')}), [Interaction summary]({link('background_interactions_summary.csv')}), [Endpoint comparison]({link('selected_vs_endpoint.csv')}).
- [Source learning/single-task references]({link('source_learning_table.csv')}), [Common-source B/C differences]({link('common_source_BC.csv')}), [Source and target interactions side by side]({link('interaction_source_context.csv')}).
- [Distribution and difficulty differences for retained B/C candidate pairs]({link('candidate_matching.csv')}); selection evidence: [v3 pre-freeze matching records]({previous_matching_link}).
- [Full target curves]({link('target_curves.csv')}), [Source curves]({link('source_curves.csv')}), [Frozen configuration]({link('FROZEN_PROTOCOL.json')}), [Cost]({link('compute_summary.csv')}), [Stage timings]({link('stage_timings.csv')}); see README.md for reproduction commands.
- [Independent statistical recomputation and CSV-binding audit]({link('STATISTICS_AUDIT.json')}); recorded separately from the final weights/protocol audit.
- [Post-freeze definition-level relation review addendum]({addendum_link}); completed before inspecting new transfer results, without triggering new experiments or rewriting the frozen catalog.
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
