"""Visualize saved source and selected-LR adaptation losses, without inference.

Selection-dependent adaptation plots describe the candidate chosen by the
already frozen development rule. They do not revise that selection or discard
the other candidates' raw learning curves.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
UNITS = [f"R{i}" for i in range(1, 9)]
CONDITIONS = ["P", "D", "T", "U", "A"]
SIZES = [256, 512, 1024]
COLORS = dict(P="#777777", D="#0072B2", T="#D55E00", U="#009E73", A="#AA4499")
COMMON = ["descents", "successions", "left_to_right_maxima"]
CANDIDATES = dict(P="peaks", D="deficiencies", T="anti_fixed_points", U="right_to_left_maxima")
SOURCE_STEPS = [0, 600, 1200, 2400, 3600, 4800]
ADAPT_STEPS = list(range(0, 801, 100))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def csvrows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def validate(artifacts):
    selections = read_json(artifacts / "SELECTIONS_FROZEN.json")
    assert selections["status"] == "frozen" and selections["training_complete"] is True
    assert selections["candidate_runs"] == 360 and len(selections["selections"]) == 120
    chosen = {(r["repeat_id"], r["condition"], int(r["target_samples"])): r["run_id"] for r in selections["selections"]}
    assert set(chosen) == set(itertools.product(UNITS, CONDITIONS, SIZES))
    sources, adaptation = csvrows(artifacts / "source_curves.csv"), csvrows(artifacts / "adaptation_curves.csv")
    source_lookup = {}
    for row in sources:
        key = row["repeat_id"], row["condition"], row["task"], int(row["step"]), row["split"]
        assert key not in source_lookup
        source_lookup[key] = row
    expected_source = {(unit, condition, task, step, split) for unit, condition in itertools.product(UNITS, CANDIDATES)
                       for task, step, split in itertools.product([*COMMON, CANDIDATES[condition]], SOURCE_STEPS, ["train", "dev"])}
    assert set(source_lookup) == expected_source and len(source_lookup) == 1536
    candidate_lookup = {}
    for row in adaptation:
        key = row["run_id"], int(row["step"]), row["split"]
        assert key not in candidate_lookup
        candidate_lookup[key] = row
    runs = {f"{r}_{c}_n{n}_lr{i}" for r,c,n,i in itertools.product(UNITS, CONDITIONS, SIZES, range(3))}
    expected_adaptation = set(itertools.product(runs, ADAPT_STEPS, ["train", "dev"]))
    assert set(candidate_lookup) == expected_adaptation and len(candidate_lookup) == 6480
    for row in [*sources, *adaptation]:
        assert int(row["length"]) == 10
        for field in ["answer_ce", "two_token_ce"]:
            assert math.isfinite(float(row[field])) and float(row[field]) >= 0
    return source_lookup, candidate_lookup, chosen


def run(artifacts):
    artifacts = Path(artifacts).resolve()
    out = artifacts / "loss_figures"
    source, adaptation, chosen = validate(artifacts)
    inputs = {name: sha(artifacts / name) for name in ["SELECTIONS_FROZEN.json", "source_curves.csv", "adaptation_curves.csv"]}
    script_sha = sha(Path(__file__))
    manifest_path = out / "LOSS_FIGURE_MANIFEST.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if previous["inputs"] == inputs and previous["generator_sha256"] == script_sha:
            assert all(sha(out/name) == value for name,value in previous["outputs"].items())
            print(json.dumps(dict(status="already_current", figures=12, model_forwards=0), indent=2))
            return previous
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "svg.hashsalt": "fixed_points_deficiencies_loss_figures_v1"})
    generated = []

    def save(fig, name):
        fig.tight_layout(rect=(.02, .025, 1, .96))
        for suffix in ("png", "svg", "pdf"):
            path = out / f"{name}.{suffix}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            generated.append(path)
        plt.close(fig)

    metric_names = {"answer_ce": "Answer-token cross-entropy", "two_token_ce": "Mean answer + EOS cross-entropy"}
    for metric, label in metric_names.items():
        fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True)
        for ax, task in zip(axes.flat, [*COMMON, *CANDIDATES.values()]):
            for condition in CANDIDATES:
                if task not in [*COMMON, CANDIDATES[condition]]:
                    continue
                for split, style in [("train", "--"), ("dev", "-")]:
                    values = [[float(source[unit, condition, task, step, split][metric]) for step in SOURCE_STEPS] for unit in UNITS]
                    for row in values:
                        ax.plot(SOURCE_STEPS, row, color=COLORS[condition], linestyle=style, linewidth=.6, alpha=.09)
                    ax.plot(SOURCE_STEPS, [statistics.mean(row[j] for row in values) for j in range(len(SOURCE_STEPS))],
                            color=COLORS[condition], linestyle=style, linewidth=1.9, label=f"{condition} {split}")
            ax.set_title(task.replace("_", " "))
            ax.set_yscale("symlog", linthresh=.01, linscale=1)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
            ax.legend(fontsize=6.5, ncol=2)
        axes.flat[-1].axis("off")
        axes.flat[-1].text(0, .9, "n=10; eight paired units\n\nThick lines: unit means\nThin lines: individual units\nSolid: development\nDashed: training\n\nY: symlog, linear below 0.01\nAll values retained, no clipping", va="top")
        fig.suptitle(f"Source learning losses: {label}; n=10")
        fig.supxlabel("Pretraining updates (fixed endpoint: 4800)")
        fig.supylabel(f"{label} (symlog; linear below 0.01)")
        save(fig, f"source_{metric}")

        fig, axes = plt.subplots(5, 3, figsize=(13.5, 15), sharex=True)
        for i, condition in enumerate(CONDITIONS):
            for j, size in enumerate(SIZES):
                ax = axes[i, j]
                for split, style in [("train", "--"), ("dev", "-")]:
                    values = [[float(adaptation[chosen[unit, condition, size], step, split][metric])
                               for step in ADAPT_STEPS] for unit in UNITS]
                    for row in values:
                        ax.plot(ADAPT_STEPS, row, color=COLORS[condition], linestyle=style, linewidth=.65, alpha=.13)
                    ax.plot(ADAPT_STEPS, [statistics.mean(row[k] for row in values) for k in range(len(ADAPT_STEPS))],
                            color=COLORS[condition], linestyle=style, linewidth=2, label=split)
                ax.set_title(f"{condition}: {size} target training examples")
                ax.set_yscale("symlog", linthresh=.01, linscale=1)
                ax.set_ylim(bottom=0)
                ax.grid(alpha=.2)
                ax.legend(fontsize=7)
        fig.suptitle(f"Adaptation: {label}; n=10\nFrozen selected-LR candidates; thick means / thin individual units (n=8)")
        fig.supxlabel("Adaptation updates; selection-dependent display, all 360 raw candidate curves retained")
        fig.supylabel(f"{label} (symlog; linear below 0.01)")
        save(fig, f"adaptation_{metric}")
    note = out / "README.md"
    note.write_text("""# Saved learning-loss trajectories

These figures read stored training/development curves only. They add no training,
model evaluation, checkpoint selection or statistical test. All panels use n=10.
Thick lines are means of the eight paired units; thin lines retain each unit.
Solid lines show development, and dashed lines show training. Y axes explicitly
use symlog scaling, with a linear region below 0.01, to preserve large initial
losses and readable later losses, including exact zero. All saved loss values
are plotted without clipping or alteration. Panels autoscale independently.

Source panels include every condition/task at all six prespecified evaluations.
Target panels show the entire 0-to-800 trajectory of each frozen selected-LR
candidate, including updates after the selected checkpoint. They are
selection-dependent descriptions, not independent evidence for a new checkpoint
rule. The raw adaptation table retains all three LR candidates for every route,
sample size and unit: 360 runs, not only the 120 displayed candidates.

`answer_ce` is teacher-forced answer-token cross-entropy. `two_token_ce` is the
mean of answer-token and teacher-forced EOS cross-entropies. These losses are
distinct from the complete generated answer/EOS exact-match accuracy.

| Stage | Answer loss | Mean answer + EOS loss |
| --- | --- | --- |
| Source | [PNG](source_answer_ce.png), [SVG](source_answer_ce.svg), [PDF](source_answer_ce.pdf) | [PNG](source_two_token_ce.png), [SVG](source_two_token_ce.svg), [PDF](source_two_token_ce.pdf) |
| Adaptation | [PNG](adaptation_answer_ce.png), [SVG](adaptation_answer_ce.svg), [PDF](adaptation_answer_ce.pdf) | [PNG](adaptation_two_token_ce.png), [SVG](adaptation_two_token_ce.svg), [PDF](adaptation_two_token_ce.pdf) |

The separate manifest records source-table/selection hashes and this plotting
script hash. Frozen `analysis.py` outputs remain unchanged.
""", encoding="utf8")
    generated.append(note)
    result = dict(status="complete", generator="plot_losses.py", generator_sha256=script_sha,
        inputs=inputs, outputs={p.name: sha(p) for p in generated}, independent_units=8,
        length=10, source_curve_rows=1536, adaptation_curve_rows=6480,
        raw_adaptation_candidates=360, displayed_selected_lr_candidates=120,
        selection_scope="Frozen selected-LR candidates; no reselection; all candidate curves retained",
        metrics=list(metric_names), y_scale="symlog", linear_threshold=.01,
        loss_values_modified_or_clipped=False, training_runs=0, model_forwards=0,
        frozen_analysis_outputs_modified=False)
    manifest_path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf8")
    print(json.dumps(dict(status="complete", figures=12, source_curve_rows=1536,
                         adaptation_curve_rows=6480, training_runs=0, model_forwards=0), indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    run(parser.parse_args().artifacts)
