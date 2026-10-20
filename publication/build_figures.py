"""Rebuild publication figures from archived tables, without model evaluation.

Each panel retains its own protocol, estimand and repeated unit. Development
calibration and held-out transfer accuracy are never pooled. All sources are
tracked DMLR tables: upstream research audits are optional historical documents,
not figure inputs. No checkpoint is opened and no training/test forward is run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "publication/figures"
BLUE, ORANGE, GREEN, PURPLE = "#276B91", "#C66B35", "#3B8066", "#80538E"
INK, MUTED, GRID = "#23313D", "#657681", "#DCE3E7"
COLORS = {"chain_alpha": BLUE, "chain_beta": ORANGE}
TARGET_LABELS = {"fixed_points": "Fixed points", "exceedances": "Exceedances",
    "left_to_right_minima": "Left-to-right minima", "longest_decreasing_run": "Longest decreasing run",
    "cycle_count": "Cycle count", "lis_length": "LIS length", "components": "Block count"}
SOURCE_BINDINGS = {}
PLOT_DATA = []
PLOT_SUMMARIES = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(relative, scope):
    path = ROOT / relative
    assert path.is_file(), path
    SOURCE_BINDINGS[path.relative_to(ROOT).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size, "scope": scope}
    return path


def read_csv(relative, scope):
    with bind(relative, scope).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(relative, scope):
    return json.loads(bind(relative, scope).read_text(encoding="utf-8-sig"))


def write_csv(path, rows):
    assert rows
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def record(figure, panel, study, estimand, group, seeds, values, unit):
    assert len(seeds) == len(values) and len(set(seeds)) == len(seeds)
    assert len(seeds) >= 2
    for seed, value in zip(seeds, values):
        PLOT_DATA.append({"figure": figure, "panel": panel, "study": study, "estimand": estimand,
            "group": group, "repeat_id": seed, "value": value, "unit": unit})
    mean, sd = statistics.mean(values), statistics.stdev(values)
    PLOT_SUMMARIES.append({"figure": figure, "panel": panel, "study": study, "estimand": estimand,
        "group": group, "repeat_units": len(seeds), "mean": mean, "sample_sd": sd,
        "min": min(values), "max": max(values), "unit": unit,
        "uncertainty": "Descriptive sample SD; not a confidence interval or additional independent observations"})
    return mean, sd


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.titlesize": 11, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 9,
        "axes.edgecolor": "#A7B3BB", "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "permworld-dmlr-release-v2",
        "savefig.facecolor": "white", "figure.facecolor": "white"})


def save(fig, out, stem, title):
    for extension in ("png", "svg", "pdf"):
        metadata = {"Title": title}
        if extension == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        elif extension == "svg":
            metadata["Date"] = None
        fig.savefig(out / f"{stem}.{extension}", dpi=300, metadata=metadata)
        if extension == "svg":
            path = out / f"{stem}.{extension}"
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf8").splitlines()) + "\n",
                            encoding="utf8", newline="\n")
    plt.close(fig)


def heading(ax, letter, title, subtitle):
    ax.set_title(f"{letter}  {title}", loc="left", pad=39 if "\n" in subtitle else 29)
    ax.text(0, 1.025, subtitle, transform=ax.transAxes, fontsize=8.2, color=MUTED,
            va="bottom", ha="left", clip_on=False)


def paired_points(ax, y, values, color, seeds, figure, panel, study, estimand, group, unit="percentage points", hollow=False):
    mean, sd = record(figure, panel, study, estimand, group, seeds, values, unit)
    markers = ["o", "s", "^", "v", "P"]
    offsets = [0] if len(values) == 1 else [(i - (len(values)-1)/2)*.055 for i in range(len(values))]
    for i, (value, offset) in enumerate(zip(values, offsets)):
        ax.scatter(value, y+offset, s=23, marker=markers[i], facecolor="white" if hollow else color,
                   edgecolor=color, linewidth=.9, alpha=.77, zorder=4)
    ax.errorbar(mean, y, xerr=sd, fmt="D", color=color, markeredgecolor="white",
                markersize=5.8, capsize=3, linewidth=1.5, zorder=6)
    return mean, sd


def finish_effect_axis(ax, labels, xmax=None):
    ax.axvline(0, color=MUTED, linewidth=.8, linestyle="--", zorder=0)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels)-.55, -.55)
    ax.grid(axis="x", color=GRID, linewidth=.65, zorder=0)
    ax.tick_params(axis="y", length=0)
    if xmax:
        ax.set_xlim(*xmax)


def card(ax, xy, width, height, title, body, color=BLUE, fontsize=9):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=.8, edgecolor=color, facecolor="#F6F9FA", transform=ax.transAxes))
    ax.text(x+.025, y+height-.035, title, transform=ax.transAxes, color=color, fontsize=10,
            weight="bold", ha="left", va="top")
    ax.text(x+.025, y+height-.105, body, transform=ax.transAxes, fontsize=fontsize,
            va="top", ha="left", linespacing=1.6)


def resource_panel(ax, counts):
    ax.set_axis_off()
    heading(ax, "a", "A reusable mathematical task platform", "Endpoint names, templates and instances are different counts")
    assert counts["legacy_endpoints_preserved"] == 48 and counts["total_registry_endpoints"] == 54
    assert counts["canonical_mathematical_templates"] == 48
    assert counts["all_declared_endpoint_instances_including_aliases_and_binary"] == 62
    assert counts["default_unary_manifest_instances"] == 56
    assert counts["input_encoding_variants"] == ["one_line_v1"]
    card(ax, (.015, .55), .97, .40, "Platform v0.1: exact labels, shared interface",
         "48 retained + 6 added = 54 task endpoints\n"
         "48 canonical templates; 62 declared instances\n"
         "56 materialized unary instances; one-line input\n"
         "Versioned integer / structured-answer encoding", BLUE, 9.1)
    card(ax, (.015, .06), .97, .42, "Three focus families",
         "Family                              Templates / instances\n"
         "Cycle decomposition                          6 / 8\n"
         "Global subsequences                          5 / 9\n"
         "Permutation blocks                           5 / 7", GREEN, 9.0)
    ax.text(.025, -.025, "Focus: 16 templates / 24 instances / 22 nonconstant.\nCounts do not establish mathematical independence.",
            transform=ax.transAxes, fontsize=8.0, color=MUTED, va="top")


def relation_panel(ax, relations):
    ax.set_axis_off()
    assert len(relations["relations"]) == 74
    heading(ax, "b", "Relationships are explicit design objects", "74 relation records; typed evidence and query conditions")
    card(ax, (.015, .68), .97, .25, "Transformed-input relation",
        r"$\mathrm{LIS}(p)=\mathrm{LDS}(C(p))$"+"\nThe source answer uses a complemented input.", BLUE, 9.2)
    card(ax, (.015, .36), .97, .26, "Joint answers on the same input",
        r"$\mathrm{FP}(p)=n-\mathrm{exceedances}(p)-\mathrm{deficiencies}(p)$"+"\nFormula availability does not establish model use.", PURPLE, 8.9)
    ax.text(.035, .285, "Exact generators → full-orbit isolated data", transform=ax.transAxes,
            fontsize=9.4, weight="bold", color=GREEN)
    ax.text(.035, .215, "Paired training → adaptation → audited archive", transform=ax.transAxes,
            fontsize=9.4, weight="bold", color=GREEN)
    ax.text(.035, .085, "Versioned complete-answer + EOS evaluation\nInput isolation and legal task relations are separate checks.",
            transform=ax.transAxes, fontsize=8.7, linespacing=1.6)
    ax.text(.035, -.025, "Aliases and constants remain visible.\nMissing relations mean unclassified, not unrelated.",
            transform=ax.transAxes, fontsize=8.0, color=MUTED, va="top")


def calibration_panel(ax, records):
    tasks = ["cycle_count", "longest_cycle", "lis_length", "increasing_subsequence_count", "components", "largest_component"]
    names = ["Cycle count", "Longest cycle", "LIS length", "Increasing triples", "Block count", "Largest block"]
    selected = [r for r in records if r["checkpoint"] == "endpoint" and r["split"] == "dev" and r["length"] == "all"]
    assert len(selected) == 12
    heading(ax, "d", "Learnability is measured, not assumed", "Platform calibration · dev endpoint · 2 seeds · n=8,10,12")
    for y, task in enumerate(tasks):
        rows = sorted([r for r in selected if r["task_id"] == task], key=lambda r:int(r["initialization_seed"]))
        assert len(rows) == 2
        seeds = [int(r["initialization_seed"]) for r in rows]
        acc = [100*float(r["accuracy"]) for r in rows]
        majority = [100*float(r["majority_accuracy"]) for r in rows]
        mean, _ = paired_points(ax, y, acc, GREEN, seeds, "figure1", "d", "platform_v0.1",
            "Fixed-endpoint complete-answer dev accuracy", task, "% accuracy")
        majority_mean, _ = record("figure1", "d", "platform_v0.1", "Train-derived majority accuracy on dev", task, seeds, majority, "% accuracy")
        ax.scatter(majority_mean, y, marker="|", color=MUTED, s=150, linewidth=1.7, zorder=5)
        ax.plot([majority_mean, mean], [y,y], color=GREEN, alpha=.2, linewidth=3, zorder=1)
    ax.set_yticks(range(6), names)
    ax.set_ylim(5.6,-.6)
    ax.set_xlim(0,105)
    ax.set_xticks([0,25,50,75,100])
    ax.set_xlabel("Complete-answer development accuracy (%)")
    ax.grid(axis="x", color=GRID, linewidth=.65)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Line2D([0],[0],marker="D",color=GREEN,label="Mean ± sample SD",markersize=5),
        Line2D([0],[0],marker="|",color=MUTED,linestyle="none",label="Training-majority baseline",markersize=10)],
        loc="lower left", bbox_to_anchor=(0,-.27), frameon=False, ncol=2, fontsize=7.8)


def data_and_protocol_panel(ax, quality):
    ax.set_axis_off()
    heading(ax, "c", "Data and compute have explicit audit trails",
            "Archived platform audit · fixed inputs · paired training budgets")
    assert quality["status"] == "passed"
    assert quality["objects_checked"] == 9216 and quality["labels_recomputed"] == 516096
    assert quality["duplicate_inputs"] == 0
    assert quality["cross_role_or_repeat_orbit_overlap"] == 0
    assert quality["historical_orbit_overlap"] == 0
    card(ax, (.015, .54), .97, .41, "Input and label checks",
         "9,216 inputs; 516,096 exact labels recomputed\n"
         "0 duplicate inputs; 0 cross-role / repeat orbit overlaps\n"
         "0 historical orbit overlaps; RNG streams replayed\n"
         "Inverse, reversal and complement define isolation", BLUE, 8.9)
    card(ax, (.015, .055), .97, .40, "Transfer protocol",
         "Freeze tasks, input pools and selection rules\n"
         "Account for sequence tokens and per-task exposure\n"
         "Pair adaptation data, initialization and repeats\n"
         "Evaluate full generated answers + EOS", GREEN, 9.0)
    ax.text(.025, -.025, "Audit counts describe the archived unary dataset.\nBinary-pair isolation was not certified; no new model run here.",
            transform=ax.transAxes, fontsize=8.0, color=MUTED, va="top")


def load_evidence():
    data = {}
    data["counts"] = read_json("experiments/platform_v01/registry/TASK_COUNTS.json", "Platform endpoint/template/instance count conventions")
    data["relations"] = read_json("experiments/platform_v01/mathematics/RELATIONS.json", "Typed formulas, input transforms and joint-answer metadata")
    data["quality"] = read_json("experiments/platform_v01/dataset/quality_full.json", "Archived platform unary input and exact-label audit; not a new generation or model evaluation")
    data["calibration"] = read_csv("experiments/platform_v01/calibration/results_by_seed.csv", "Endpoint, dev, length=all; six prespecified tasks and two seeds")
    data["v3"] = read_csv("experiments/pilot_v3_multitarget/artifacts/paired_deltas.csv", "Development-selected target test deltas; B-A and C-A; three paired seeds")
    data["v4"] = read_csv("experiments/pilot_v5_confirmation/exploratory/exploratory_paired_differences.csv", "V4 exploratory background contrasts selected in advance of v5; old three seeds")
    data["v5"] = read_csv("experiments/pilot_v5_confirmation/artifacts/confirmation_paired_differences.csv", "V5 primary delta(B-C); five new paired repeats; not pooled with v4")
    data["tc1_contrasts"] = read_csv("experiments/task_count_v1/paired_k_contrasts.csv", "Development-selected, length=all, k2-to-k16 paired contrast; three seeds")
    data["tc1_raw"] = read_csv("experiments/task_count_v1/raw_test_results.csv", "Development-selected full-answer+EOS test accuracy; two fixed chains, four targets, k2/4/8/16")
    data["tc2"] = read_csv("experiments/task_count_v2/paired_budget_interactions.csv", "Development-selected, length=all; [k16-k2]32M-[k16-k2]4M within each new repeat")
    return data


def figure2(data, out):
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.9), layout="constrained")
    fig.get_layout_engine().set(hspace=.14, wspace=.12)
    fig.suptitle("Transfer evidence across the DMLR program", fontsize=17, weight="bold")
    # The separate panels deliberately do not share an estimand or one pooled N.
    ax = axes[0,0]
    heading(ax, "a", "Transfer varies across targets", "Pilot v3 · 4 sources · 256 target examples · 3 paired seeds")
    targets = ["fixed_points", "exceedances", "left_to_right_minima", "longest_decreasing_run"]
    for y,t in enumerate(targets):
        for contrast, shift, color in (("B-A",-.15,BLUE),("C-A",.15,ORANGE)):
            rows = sorted([r for r in data["v3"] if r["target"] == t and r["contrast"] == contrast],key=lambda r:int(r["seed"]))
            assert [int(r["seed"]) for r in rows] == [17,42,73]
            values = [100*float(r["test_accuracy_delta"]) for r in rows]
            paired_points(ax,y+shift,values,color,[int(r["seed"]) for r in rows],"figure2","a","pilot_v3",contrast,t)
    finish_effect_axis(ax,[TARGET_LABELS[t] for t in targets],(-25,30))
    ax.set_xlabel("Gain over shared scratch A (percentage points)")
    ax.legend(handles=[Line2D([0],[0],color=BLUE,marker="D",label="B − A"),Line2D([0],[0],color=ORANGE,marker="D",label="C − A")],
              loc="lower right",frameon=False,ncol=2,fontsize=8.5)
    ax = axes[0,1]
    heading(ax,"b","Background effects require confirmation","V4 exploration: 3 units; V5 confirmation: 5 new units\nSame prespecified contrasts; stages kept separate")
    targets = ["fixed_points", "longest_decreasing_run"]
    for y,t in enumerate(targets):
        background = "g2_deficiencies" if t == "fixed_points" else "g1_recoils"
        for key,shift,color in (("v4",-.15,MUTED),("v5",.15,PURPLE)):
            rows = sorted([r for r in data[key] if r["target"] == t and r["new_background"] == background and r["original_background"] == "g0_v3"],key=lambda r:int(r["seed"]))
            assert len(rows) == (3 if key=="v4" else 5)
            paired_points(ax,y+shift,[100*float(r["delta_B_minus_C"]) for r in rows],color,[int(r["seed"]) for r in rows],
                          "figure2","b",f"pilot_{key}","(B-C)new background - (B-C)original",t,hollow=key=="v4")
    finish_effect_axis(ax,["Fixed points\n+ deficiencies background","Longest decreasing run\n+ recoils background"],(-38,36))
    ax.set_xlabel("Background interaction (percentage points)")
    ax.legend(handles=[Line2D([0],[0],color=MUTED,marker="D",label="V4 exploration, n=3"),Line2D([0],[0],color=PURPLE,marker="D",label="V5 confirmation, n=5")],
              loc="lower right",frameon=False,fontsize=8.2)
    ax = axes[1,0]
    heading(ax,"c","More tasks are not uniformly better","Task-count v1 · 4M sequence tokens · 3 new paired units")
    targets = ["cycle_count", "lis_length", "components", "fixed_points"]
    for y,t in enumerate(targets):
        for chain,shift in (("chain_alpha",-.15),("chain_beta",.15)):
            rows=sorted([r for r in data["tc1_contrasts"] if r["target"]==t and r["chain"]==chain and r["checkpoint"]=="development_selected" and r["length"]=="all" and r["k_from"]=="2" and r["k_to"]=="16"],key=lambda r:int(r["initialization_seed"]))
            assert [int(r["initialization_seed"]) for r in rows]==[1103,2207,3301]
            paired_points(ax,y+shift,[float(r["paired_accuracy_change_pp"]) for r in rows],COLORS[chain],[int(r["initialization_seed"]) for r in rows],
                          "figure2","c","task_count_v1","Accuracy k16 - k2 at 4M",t+"/"+chain)
    finish_effect_axis(ax,[TARGET_LABELS[t] for t in targets],(-16,13))
    ax.set_xlabel("Task-count contrast: k16 − k2 (percentage points)")
    ax.legend(handles=[Line2D([0],[0],color=BLUE,marker="D",label="Chain α"),Line2D([0],[0],color=ORANGE,marker="D",label="Chain β")],
              loc="lower right",frameon=False,ncol=2,fontsize=8.5)
    ax = axes[1,1]
    heading(ax,"d","Budget changes the task-count effect","Task-count v2 · 4M vs 32M · 3 fresh paired units\nSame finite input pools; different absolute-token LR paths")
    for y,t in enumerate(["lis_length","fixed_points"]):
        for chain,shift in (("chain_alpha",-.15),("chain_beta",.15)):
            rows=sorted([r for r in data["tc2"] if r["target"]==t and r["chain"]==chain and r["checkpoint"]=="development_selected" and r["length"]=="all"],key=lambda r:int(r["initialization_seed"]))
            assert [int(r["initialization_seed"]) for r in rows]==[4103,5207,6301]
            paired_points(ax,y+shift,[float(r["paired_budget_interaction_pp"]) for r in rows],COLORS[chain],[int(r["initialization_seed"]) for r in rows],
                          "figure2","d","task_count_v2","(k16-k2)32M - (k16-k2)4M",t+"/"+chain)
    finish_effect_axis(ax,["LIS length\nprimary target","Fixed points\nauxiliary target"],(-35,53))
    ax.set_xlabel("Budget interaction: D32 − D4 (percentage points)")
    ax.legend(handles=[Line2D([0],[0],color=BLUE,marker="D",label="Chain α"),Line2D([0],[0],color=ORANGE,marker="D",label="Chain β")],
              loc="lower right",frameon=False,ncol=2,fontsize=8.5)
    fig.supxlabel("Small symbols = individual paired units; diamonds/whiskers = mean ± sample SD. Panels have distinct estimands and protocols.",fontsize=8.3)
    save(fig,out,"figure2_evidence_overview","DMLR evidence across targets, composition, confirmation, task count and budget")


def supplementary_count_curves(data,out):
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.7),layout="constrained")
    fig.get_layout_engine().set(hspace=.1,wspace=.08)
    fig.suptitle("Task-count v1: complete finite curves at fixed 4M tokens",fontsize=16,weight="bold")
    targets=["cycle_count","lis_length","components","fixed_points"]
    rows=[r for r in data["tc1_raw"] if r["checkpoint"]=="development_selected" and r["length"]=="all"]
    lookup={(r["target"],r["chain"],int(r["k"]),int(r["initialization_seed"])):r for r in rows}
    assert len(rows)==108 and len(lookup)==len(rows)
    seeds=[1103,2207,3301]
    ks=[2,4,8,16]
    for letter,ax,t in zip("abcd",axes.flat,targets):
        heading(ax,letter,TARGET_LABELS[t],"Development-selected full-answer test accuracy · 3 paired repeats")
        base=[100*float(lookup[t,"baseline",0,s]["accuracy"]) for s in seeds]
        baseline_mean,baseline_sd=record("figureS1",letter,"task_count_v1","Shared scratch accuracy",t,seeds,base,"% accuracy")
        ax.axhline(baseline_mean,color=MUTED,linestyle="--",linewidth=1,label="Shared scratch A")
        for chain in COLORS:
            means,sds=[],[]
            for k in ks:
                vals=[100*float(lookup[t,chain,k,s]["accuracy"]) for s in seeds]
                mean,sd=record("figureS1",letter,"task_count_v1","Test accuracy",f"{t}/{chain}/k{k}",seeds,vals,"% accuracy")
                means.append(mean);sds.append(sd)
            for si,s in enumerate(seeds):
                ax.plot(range(4),[100*float(lookup[t,chain,k,s]["accuracy"]) for k in ks],color=COLORS[chain],alpha=.28,lw=.8,marker=["o","s","^"][si],ms=3)
            ax.errorbar(range(4),means,yerr=sds,color=COLORS[chain],marker="D",markersize=4,linewidth=1.8,capsize=3,label="Chain "+("α" if chain=="chain_alpha" else "β"))
        ax.set_xticks(range(4),ks)
        ax.set(xlabel="Source-task instances k",ylabel="Test accuracy (%)",ylim=(0,100))
        ax.grid(axis="y",color=GRID,linewidth=.65)
        ax.legend(frameon=False,fontsize=8,ncol=3,loc="lower left")
    fig.supxlabel("Fixed lists and shared baselines; changing k also changes composition, relations and per-task exposure. Not a scaling law.",fontsize=8.2)
    save(fig,out,"figureS1_full_task_count_curves","Complete task-count v1 accuracy curves with individual paired repeats")


def captions():
    return """# Publication figures and source map

All figures are regenerated by `publication/build_figures.py` from retained tables.
The script imports no training code, opens no weights and performs no model
forward. PNG (300 dpi), SVG and PDF versions contain identical numerical content.
`PLOT_DATA.csv` preserves every plotted repeated-unit value; `PLOT_SUMMARIES.csv`
records descriptive means and sample SD. `SOURCE_MAP.json` binds original input
tables and the builder by SHA256, with filtering and scope notes. Original
experiment results and checkpoints are untouched.

| Figure | Browse | Vector downloads |
| --- | --- | --- |
| 1: resource, data audit and calibration | [PNG](figure1_permworld_resource.png) | [PDF](figure1_permworld_resource.pdf), [SVG](figure1_permworld_resource.svg) |
| 2: transfer evidence across stages | [PNG](figure2_evidence_overview.png) | [PDF](figure2_evidence_overview.pdf), [SVG](figure2_evidence_overview.svg) |
| S1: all four-target task-count curves | [PNG](figureS1_full_task_count_curves.png) | [PDF](figureS1_full_task_count_curves.pdf), [SVG](figureS1_full_task_count_curves.svg) |

Download [every plotted value](PLOT_DATA.csv), [descriptive summaries](PLOT_SUMMARIES.csv)
and the [source/hash map](SOURCE_MAP.json).

## Figure 1: PermWorld as a relation-aware experimental resource

**a. Resource scope.** The platform retains 48 inherited endpoints and adds six.
Its 48 canonical templates,
62 declared instances and 56 materialized unary instances use different counting
conventions. The three focus families contain 16 templates and 24 instances,
including two fixed-length constants, leaving 22 nonconstant instances at the
release lengths. The active input encoding is one-line; output encoding is
versioned for integer and structured answers. Counts come directly from the
tracked DMLR registry. They do not establish mathematical independence.

**b. Relation-aware workflow.** Registered transformed-input and same-input joint
relations specify which answers and input transformations are needed. The examples
LIS(p)=LDS(C(p)) and FP(p)=n−exceedances(p)−deficiencies(p) are previously documented
formulas, not new empirical discoveries. Exact labels, whole-orbit input isolation,
paired training and full-answer evaluation are separate parts of the design.
Formula availability does not prove use by a model, and unlisted relations are
unclassified rather than unrelated.

**c. Data and protocol audit.** The archived platform unary-data audit checked
9,216 inputs and recomputed 516,096 exact labels. Duplicate inputs, cross-role or
repeat orbit overlaps, and historical orbit overlaps were all zero; pool RNG
streams were replayed. Orbit isolation uses the group generated by inverse,
reversal and complement. These are archived audit results, not a newly generated
dataset or an additional scientific replication. Binary-pair isolation was not
certified. The workflow separately freezes tasks and pools, records actual token
and task exposure, pairs repeats, and evaluates complete generated answers plus
EOS. The earlier scalar-answer pilots in Figure 2 are explicitly distinguished
from this platform protocol.

**d. Platform calibration.** Six representatives were fixed before training, with
two seeds (617,829), 2400 updates, and n=8,10,12. Points are fixed-endpoint
development accuracy for the entire generated answer plus EOS, pooled over lengths
by sample count within a seed. Vertical ticks show the mean training-derived
majority baseline on development. Small symbols show each seed; diamonds and
whiskers show their mean and sample SD. These are development calibration results,
not final-test transfer results or evidence of unseen-length generalization.

## Figure 2: the DMLR evidence sequence

The four panels have different estimands and protocols; their vertical groups
and horizontal values are not pooled into one benchmark score. Small symbols
represent paired repeat units and diamonds/whiskers are their mean ± sample SD,
not confidence intervals. The sample SD is computed after each within-repeat
difference, never from two separate marginal SDs.

**a. Pilot v3.** B−A and C−A test-accuracy gains for four targets at 256 adaptation
examples and four source tasks. B includes a registered related source, C replaces
that source, and A is the shared scratch baseline. Three seeds (17,42,73) are
retained individually. The legacy protocol evaluates a scalar answer token and
uses development-only learning-rate/checkpoint selection. Related-source presence
is not a pure causal intervention on mathematical relatedness, and C is not
claimed to be unrelated.

**b. Exploration followed by confirmation.** The fixed-points deficiencies
background and decreasing-run recoils background were selected from v4 exploration.
For each unit, the plotted interaction is (B−C)new background−(B−C)original.
Old v4 seeds (17,42,73) and five fresh v5 repeats (101,211,307,401,503) remain
separate, not pooled as eight repeats. The fixed-points prespecified mean direction
did not reproduce; the decreasing-run point estimate retained its direction but
the original v5 paired-t intervals included zero for both targets. This figure
shows descriptive SD rather than those inferential intervals; the latter remain
in the archived report. V3/v4 reuse some training artifacts, so their panels are
not independent replications of one another.

**c. Task-count v1.** Development-selected complete-answer-plus-EOS test contrasts
k16−k2 for all four held-out targets, separately for the two fixed nested chains.
Three paired units (1103,2207,3301), n=10/12, 256 target examples and a fixed 4M
nonpadding prompt-plus-answer token budget are used. Figure S1 retains the complete
k=2,4,8,16 curves. Increasing k changes composition, available relations and
per-task exposure; this is not a composition-free count effect or a scaling law.

**d. Task-count v2.** The paired budget interaction is
[(accuracy k16−accuracy k2)32M−(accuracy k16−accuracy k2)4M], separately for each
chain, with LIS primary and fixed points auxiliary. Three further repeats
(4103,5207,6301) start both budgets from the same paired initialization and use the
same fixed source-input pool. Relative-budget LR schedules are identical but
absolute-token LR trajectories differ. More tokens increase repeated exposure
and per-task coverage within the pool, not its number of independent objects.
Shared scratch baselines cancel and shared models do not create extra repeats.

## Figure S1: complete task-count v1 curves

All four held-out targets, both prespecified chains and all k values are retained.
Each thin line is one repeat; thick lines and error bars are means and sample SD.
The horizontal dashed line is the shared scratch mean, trained only once per
target/repeat. All panels use a 0–100% axis. Source underlearning, negative transfer
and seed disagreement remain visible; no best chain or k was chosen from tests.

## Reproduction

Run from the repository root in the installed publication environment:

```powershell
python publication/build_figures.py --out build/publication-figures
```

Use `--out` with a fresh directory to preserve the released figure bytes. The
builder depends only on tracked DMLR registry and result tables. Source audits,
upstream repositories, `reference/` and local checkpoints are not required.
TMLR-specific experiments and original workshop alignment results are outside
this DMLR figure package.
"""


def build(out):
    out.mkdir(parents=True,exist_ok=True)
    SOURCE_BINDINGS.clear()
    PLOT_DATA.clear()
    PLOT_SUMMARIES.clear()
    style()
    data=load_evidence()
    fig,axes=plt.subplots(2,2,figsize=(12.6,9.3),layout="constrained",gridspec_kw={"height_ratios":[1.13,1]})
    fig.get_layout_engine().set(hspace=.16,wspace=.13)
    fig.suptitle("PermWorld: exact tasks, explicit relationships, auditable evidence",fontsize=17,weight="bold")
    resource_panel(axes[0,0],data["counts"])
    relation_panel(axes[0,1],data["relations"])
    data_and_protocol_panel(axes[1,0],data["quality"])
    calibration_panel(axes[1,1],data["calibration"])
    save(fig,out,"figure1_permworld_resource","PermWorld resource, mathematical relationships, data audit and development calibration")
    figure2(data,out)
    supplementary_count_curves(data,out)
    write_csv(out/"PLOT_DATA.csv",PLOT_DATA)
    write_csv(out/"PLOT_SUMMARIES.csv",PLOT_SUMMARIES)
    (out/"README.md").write_text(captions(),encoding="utf-8")
    result={"schema_version":"permworld-dmlr-figures-v2","status":"complete","builder_sha256":sha(Path(__file__)),"matplotlib_version":matplotlib.__version__,
        "scope":"DMLR resource and transfer evidence; upstream alignment studies are excluded",
        "training_or_model_evaluations_run":False,"source_files":SOURCE_BINDINGS,
        "plotted_repeat_values":len(PLOT_DATA),"plotted_group_summaries":len(PLOT_SUMMARIES),
        "output_sha256":{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="SOURCE_MAP.json"}}
    (out/"SOURCE_MAP.json").write_text(json.dumps(result,indent=2,ensure_ascii=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":"complete","figure_sets":3,"plotted_repeat_values":len(PLOT_DATA),"output":str(out)}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    build(args.out.resolve())
