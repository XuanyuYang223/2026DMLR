"""Render the two saved fixed-points studies without statistical recomputation.

Only archived CSV/JSON tables and documentation are read. No experiment modules,
data pools, model checkpoints, training routines or evaluators are imported.
Every plotted scientific ordinate is copied from a saved CSV field; horizontal
offsets only separate overlapping unit markers. Existing outputs are preserved.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "publication/figures/case_integration_v2"
FILTER = {"metric": "complete", "checkpoint_kind": "selected", "scope": "area"}
CONTRASTS = ["P_minus_A", "D_minus_A", "T_minus_A", "U_minus_A", "D_minus_U"]
LABELS = ["P−A", "D−A", "T−A", "U−A", "D−U"]
UNITS = [f"R{i}" for i in range(1, 9)]
OFFSETS = [-0.245, -0.175, -0.105, -0.035, 0.035, 0.105, 0.175, 0.245]
STUDIES = [
    {"id": "original_transformer", "panel": "a", "label": "Original Transformer confirmation",
     "title": "Original confirmation\n4-layer Transformer · eight paired units",
     "directory": "experiments/fixed_points_deficiencies_v1", "color": "#2874A6"},
    {"id": "new_gru", "panel": "b", "label": "New GRU architecture replication",
     "title": "Architecture replication\n4-layer GRU · eight new paired units",
     "directory": "experiments/fixed_points_deficiencies_arch_v1", "color": "#B86624"},
]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return [(line, row) for line, row in enumerate(csv.DictReader(handle), start=2)]


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def collect():
    """Select archived fields verbatim; never calculate means, SDs or intervals."""
    points, summaries, mapping, bindings = [], [], [], {}
    plot = {}
    configs = []
    for study in STUDIES:
        folder = ROOT / study["directory"]
        difference_path = folder / "artifacts/analysis/paired_differences.csv"
        summary_path = folder / "artifacts/analysis/paired_summary.csv"
        protocol_path = folder / "artifacts/FROZEN_PROTOCOL.json"
        report_path = folder / "RESULT_REPORT.md"
        for path in (difference_path, summary_path, protocol_path, report_path):
            bindings[relative(path)] = sha(path)
        protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
        config = protocol["config"]
        assert config["n"] == 10 and config["target_sizes"] == [256, 512, 1024]
        assert config["model"]["layers"] == 4 and config["model"]["d_model"] == 256
        assert len(config["repeats"]) == 8
        if study["id"] == "new_gru":
            assert config["architecture"] == "gru"
        configs.append(config)
        differences = [(line, row) for line, row in read_csv(difference_path)
                       if all(row[k] == value for k, value in FILTER.items()) and row["contrast"] in CONTRASTS]
        saved_summaries = [(line, row) for line, row in read_csv(summary_path)
                           if all(row[k] == value for k, value in FILTER.items()) and row["contrast"] in CONTRASTS]
        assert len(differences) == 40 and len(saved_summaries) == 5
        for x, contrast in enumerate(CONTRASTS):
            selected = [(line, row) for line, row in differences if row["contrast"] == contrast]
            assert len(selected) == 8 and {row["repeat_id"] for _, row in selected} == set(UNITS)
            summary = [(line, row) for line, row in saved_summaries if row["contrast"] == contrast]
            assert len(summary) == 1
            summary_line, summary_row = summary[0]
            assert summary_row["n"] == "8"
            if contrast == "D_minus_U":
                assert summary_row["analysis_role"] == "primary" and summary_row["df"] == "7"
                assert summary_row["ci95_low_pp"] and summary_row["ci95_high_pp"]
            else:
                assert summary_row["analysis_role"] == "secondary_or_diagnostic"
                assert not summary_row["ci95_low_pp"] and not summary_row["ci95_high_pp"]
            summary_meta = {"study_id": study["id"], "panel": study["panel"],
                "source_path": relative(summary_path), "source_sha256": bindings[relative(summary_path)],
                "source_csv_line": str(summary_line)}
            summaries.append({**summary_meta, **summary_row})
            ordered = {row["repeat_id"]: (line, row) for line, row in selected}
            plot[(study["id"], contrast)] = {"points": [ordered[unit][1] for unit in UNITS], "summary": summary_row}
            for unit, offset in zip(UNITS, OFFSETS):
                line, row = ordered[unit]
                meta = {"study_id": study["id"], "panel": study["panel"],
                    "source_path": relative(difference_path), "source_sha256": bindings[relative(difference_path)],
                    "source_csv_line": str(line)}
                points.append({**meta, **row})
                mapping.append({**meta, **FILTER, "contrast": contrast, "series_role": "paired_unit",
                    "repeat_id": unit, "source_field": "difference_pp", "value_as_recorded": row["difference_pp"],
                    "plot_x": str(x + offset), "vertical_unit": "percentage_points", "interval_type": "none"})
            for field, role in [("mean_pp", "saved_mean"), ("ci95_low_pp", "primary_ci_low"), ("ci95_high_pp", "primary_ci_high")]:
                if field != "mean_pp" and contrast != "D_minus_U":
                    continue
                mapping.append({**summary_meta, **FILTER, "contrast": contrast, "series_role": role,
                    "repeat_id": "", "source_field": field, "value_as_recorded": summary_row[field],
                    "plot_x": str(x), "vertical_unit": "percentage_points",
                    "interval_type": "saved_paired_mean_95pct_t_df7" if field != "mean_pp" else "none"})
    for key in ("n", "encoding", "vocab_size", "sources", "batch", "pretrain_updates", "adaptation_updates",
                "learning_rates", "target_sizes", "counts", "target_dev"):
        assert configs[0][key] == configs[1][key], key
    old_seeds = {(r["model_seed"], r["data_seed"]) for r in configs[0]["repeats"]}
    new_seeds = {(r["model_seed"], r["data_seed"]) for r in configs[1]["repeats"]}
    assert old_seeds.isdisjoint(new_seeds)
    return plot, points, summaries, mapping, bindings


def draw(plot, out):
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 11, "xtick.labelsize": 12,
        "ytick.labelsize": 10, "svg.fonttype": "none", "axes.unicode_minus": True,
        "savefig.facecolor": "white", "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 7.2), sharey=True)
    fig.subplots_adjust(left=0.080, right=0.985, bottom=0.255, top=0.755, wspace=0.10)
    fig.suptitle("Fixed-points transfer: within-study paired contrasts", x=0.080, y=0.965,
                 ha="left", fontsize=18, fontweight="bold")
    fig.text(0.080, 0.916, r"Score $S=\frac{1}{4}A_{256}+\frac{1}{2}A_{512}+\frac{1}{4}A_{1024}$; complete generated answer + EOS",
             fontsize=11.5, color="#334155")
    legend = [
        Line2D([], [], marker="o", markersize=6, linestyle="none", color="#737B86", label="One paired unit"),
        Line2D([], [], marker="D", markersize=6, linestyle="none", color="#111827", label="Saved mean"),
        Line2D([], [], color="#111827", linewidth=1.8, marker="_", markersize=9,
               label="Saved 95% t interval (D−U only)"),
    ]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.074, 0.875),
               ncol=3, frameon=False, fontsize=10.5, handletextpad=0.7, columnspacing=2.2)
    for ax, study in zip(axes, STUDIES):
        ax.axvspan(3.54, 4.52, color="#F1F5F9", zorder=0)
        ax.axvline(3.5, color="#CBD5E1", linewidth=0.8, zorder=1)
        ax.axhline(0, color="#475569", linewidth=1.2, zorder=2)
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.8, zorder=0)
        for x, contrast in enumerate(CONTRASTS):
            row = plot[(study["id"], contrast)]
            ys = [float(point["difference_pp"]) for point in row["points"]]
            ax.scatter([x + offset for offset in OFFSETS], ys, s=39, color=study["color"],
                       alpha=0.90, edgecolor="white", linewidth=0.55, zorder=4)
            saved = row["summary"]
            if contrast == "D_minus_U":
                low, high = float(saved["ci95_low_pp"]), float(saved["ci95_high_pp"])
                ax.vlines(x, low, high, color="#111827", linewidth=1.65, zorder=3)
                ax.hlines([low, high], x - 0.10, x + 0.10, color="#111827", linewidth=1.65, zorder=3)
            ax.scatter(x, float(saved["mean_pp"]), marker="D", s=48, color="#111827",
                       edgecolor="white", linewidth=0.8, zorder=5)
        ax.set_xlim(-0.5, 4.52)
        ax.set_ylim(-18, 20)
        ax.set_xticks(range(5), LABELS)
        ax.set_yticks([-15, -10, -5, 0, 5, 10, 15, 20])
        ax.set_title(study["title"], loc="left", pad=16, linespacing=1.45)
        ax.text(-0.03, 1.14, study["panel"], transform=ax.transAxes, ha="right", va="top",
                fontweight="bold", fontsize=15)
        ax.text(4.02, 18.75, "Primary", ha="center", va="center", fontsize=9.5, color="#475569")
        ax.tick_params(axis="both", length=0, pad=8)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#CBD5E1")
    axes[0].set_ylabel("Difference in score S (percentage points)", labelpad=10)
    fig.text(0.080, 0.174, "P: peaks     D: deficiencies     T: anti_fixed_points     U: right_to_left_maxima", fontsize=10.5)
    fig.text(0.080, 0.137, "A: scratch. Shared sources: descents, successions, left_to_right_maxima.", fontsize=10.5)
    fig.text(0.080, 0.079, "Dots are paired within each study; the studies use separate units and are not paired or pooled across panels.",
             fontsize=10, color="#475569")
    fig.text(0.080, 0.045, "Only D−U has an interval: the saved paired-mean 95% t interval (df = 7), not a standard-deviation bar.",
             fontsize=10, color="#475569")
    # Fail visibly if an archived ordinate would be hidden by the fixed layout.
    for values in plot.values():
        assert all(-18 < float(row["difference_pp"]) < 20 for row in values["points"])
        assert -18 < float(values["summary"]["mean_pp"]) < 20
        if values["summary"]["contrast"] == "D_minus_U":
            assert -18 < float(values["summary"]["ci95_low_pp"]) < float(values["summary"]["ci95_high_pp"]) < 20
    fig.savefig(out / "worked_case.png", dpi=300)
    fig.savefig(out / "worked_case.svg", metadata={"Title": "Fixed-points transfer: within-study paired contrasts",
        "Description": "Archived paired differences and saved means; only D-U includes the saved paired 95% t interval. Separate original Transformer and new GRU studies."})
    plt.close(fig)


def documentation(summaries):
    table = ["| Study | Contrast | Saved mean (pp) | Saved primary interval (pp) |",
             "| --- | --- | --- | --- |"]
    for row in summaries:
        interval = f"[{row['ci95_low_pp']}, {row['ci95_high_pp']}]" if row["contrast"] == "D_minus_U" else "Not shown; secondary comparison"
        table.append(f"| {row['study_id']} | {row['contrast']} | {row['mean_pp']} | {interval} |")
    return """# Worked fixed-points case: two separate studies

![Within-study paired contrasts](worked_case.png)

[Vector SVG](worked_case.svg) · [Exact plotted unit values](PLOT_DATA.csv) ·
[Saved summary rows](PLOT_SUMMARIES.csv) · [Coordinate/source map](SOURCE_MAP.csv)

## Figure caption

**Fixed-points transfer under two separately registered model protocols.**
The left panel is the original four-layer, width-256 Transformer confirmation;
the right is the four-layer, width-256 GRU replication selected by its prespecified
development-only gate. Both use one-line length-10 permutations, the legacy
188-token vocabulary, and the same four-source compositions and training rules.
The panels each show eight paired units. The GRU study uses new data and new
initializations; the two sets of eight units are neither paired nor pooled across
studies. Horizontal offsets identify display positions only.

The score is S = 1/4 A256 + 1/2 A512 + 1/4 A1024, where A is complete generated
answer-plus-EOS test accuracy after development selection. Each contrast includes
every saved unit difference, with a black diamond for its saved mean. The only
interval is the already reported paired-mean 95% t interval for the sole primary
contrast D−U (df=7); it is not sample SD. No interval is newly computed or added
for secondary contrasts. Zero denotes no difference for that contrast.

Shared sources are descents, successions and left_to_right_maxima. The fourth
source is P=peaks, D=deficiencies, T=anti_fixed_points, or U=right_to_left_maxima.
A is random initialization with direct target training. The first four contrasts
give scratch-relative context: D−U must not be reported as D's gain over scratch.
Within each unit, all routes reuse the same corresponding data and A baseline;
these repeated comparisons do not increase the number of independent repeats.

Pretraining uses 4800 updates, batch 64 and 16 examples per task per update.
Each target subset starts independently from its source endpoint or paired scratch
initialization. Adaptation uses 800 updates and the three frozen learning rates
3e-4/1e-4/3e-5, selected by development accuracy, answer loss, earliest step and
fixed learning-rate order. The target training sizes 256/512/1024 also use 1024
labeled development examples per unit; the smallest condition does not use only
256 target labels overall.
The test pools are generated only after the complete development selections
freeze. Both studies retain endpoint diagnostics elsewhere; this figure includes
only development-selected checkpoints.

The original primary interval is positive. The GRU primary interval includes
zero, leaving its additional D-over-U transfer evidence uncertain rather than
proving no benefit. The figure does not estimate a between-architecture treatment
effect: architectures also have different parameter counts and fresh finite-space
data units. Global orbit exclusion conditions later samples on earlier samples;
the small-sample t interval has approximate sampling assumptions. No result proves
formula use or a transferable internal lemma module.

## Exact saved summary values

The following strings are copied directly from the original summary CSV fields,
without recomputation, rescaling or new rounding.

""" + "\n".join(table) + """

## Source selection and provenance

Each experiment supplies `artifacts/analysis/paired_differences.csv` and
`paired_summary.csv`, filtered by `metric=complete`, `checkpoint_kind=selected`,
and `scope=area`. Contrasts are P_minus_A, D_minus_A, T_minus_A, U_minus_A and
D_minus_U. Point coordinates use `difference_pp`; black diamonds use `mean_pp`;
the two primary intervals use `ci95_low_pp` and `ci95_high_pp`. All scientific
values are taken from saved fields. Numeric conversion for plotting does not
recalculate any statistic. `SOURCE_MAP.csv` retains each raw string, source path,
source SHA256, source CSV line, filter, field name and display coordinate.

`PLOT_DATA.csv` and `PLOT_SUMMARIES.csv` copy the selected original rows and add
provenance columns. Original row values, including the secondary sample SD fields,
remain unchanged; sample SD is not plotted. `MANIFEST.json` binds source tables,
frozen protocols, result reports, builder and every newly generated artifact.
The builder checks that source hashes are unchanged after rendering.

No experiment analysis script, data pool, checkpoint, training code or model
evaluator is loaded. Original analyses, predictions, scientific tables and
protocols are not modified. The figure is a presentation-only derivative.

## Reproduction

From the repository root, with Matplotlib installed:

```powershell
.venv/Scripts/python.exe -B publication/case_integration_v2/build_worked_case.py
```

The default output directory must be absent or empty. Existing outputs are never
overwritten. To reproduce without replacing this bundle:

```powershell
.venv/Scripts/python.exe -B publication/case_integration_v2/build_worked_case.py --out publication/figures/case_integration_v2/reproduction
```

Environment and executed command are recorded in the manifest. Review the PNG
for legibility; the SVG preserves editable text and vector marks.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(DEFAULT_OUT):
        raise ValueError("New figure output must remain within publication/figures/case_integration_v2")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Preserve existing figure output; choose a fresh reproduction subdirectory")
    plot, points, summaries, mapping, sources = collect()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "PLOT_DATA.csv", points)
    write_csv(out / "PLOT_SUMMARIES.csv", summaries)
    write_csv(out / "SOURCE_MAP.csv", mapping)
    draw(plot, out)
    write_text(out / "README.md", documentation(summaries))
    assert all(sha(ROOT / name) == digest for name, digest in sources.items()), "An original source changed while rendering"
    outputs = {relative(path): sha(path) for path in sorted(out.iterdir()) if path.is_file()}
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Presentation-only figure; no new analysis, data generation, training or evaluation",
        "command": ".venv/Scripts/python.exe -B " + relative(__file__) + (" --out " + relative(out) if out != DEFAULT_OUT else ""),
        "python": sys.version, "system": platform.platform(), "matplotlib": matplotlib.__version__,
        "builder_sha256": sha(__file__), "source_sha256": sources, "outputs": outputs,
        "filters": FILTER, "contrasts": CONTRASTS, "unit_ids_within_each_study": UNITS,
        "independent_repeat_count": "Eight paired units in each separate study; no pooled or across-study paired estimate",
        "point_rows": len(points), "summary_rows": len(summaries), "mapped_ordinates": len(mapping),
        "statistics_recomputed": False, "model_forward_calls": 0, "checkpoint_files_opened": 0,
        "data_pool_files_opened": 0, "scientific_ordinates": "Copied from saved CSV fields; exact source strings retained",
        "interval_policy": "Only primary D_minus_U: saved paired-mean 95% t interval, df 7; no SD bars or secondary intervals",
        "display_only": {"horizontal_offsets_by_repeat": dict(zip(UNITS, OFFSETS)), "shared_y_limits_pp": [-18, 20]},
        "sources_unchanged_after_render": True}
    write_text(out / "MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "complete", "output_directory": relative(out), "point_rows": len(points),
        "summary_rows": len(summaries), "sources_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
