"""Create presentation-only curve pages from immutable calibration CSVs.

No model, checkpoint, input pool, training routine, or evaluation routine is
loaded. Existing files and output directories are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ART = HERE / "artifacts"
FINAL = ART / "final"
SOURCES = (FINAL / "learning_curves.csv", FINAL / "endpoint_summary.csv")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ART / "presentation_v1")
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ART.resolve()) or out.is_relative_to(FINAL.resolve()) or out.exists():
        raise ValueError("Use a new output directory inside artifacts, outside final; no overwrite is allowed")

    source_hashes = {relative(path): sha(path) for path in SOURCES}
    final_manifest_hash = sha(FINAL / "MANIFEST.json")
    curves = read_csv(SOURCES[0])
    summary = read_csv(SOURCES[1])
    pooled = [r for r in summary if r["length"] == "all" and r["status"] == "completed_shared_canonical_units"]
    if len(pooled) != 49 or len({r["instance_id"] for r in pooled}) != 49:
        raise ValueError("Expected the complete recorded inventory of 49 measured instances")
    by_curve = defaultdict(list)
    for row in curves:
        if row["length"] == "all":
            by_curve[row["instance_id"], int(row["initialization_seed"]), row["split"]].append(row)
    for rows in by_curve.values():
        rows.sort(key=lambda r: int(r["step"]))
    if len(by_curve) != 49 * 2 * 2:
        raise ValueError("Pooled curves do not cover each instance, unit and split exactly once")

    out.mkdir(parents=True)
    page_records = []
    plotted_points = 0
    for page, start in enumerate(range(0, len(pooled), 10), 1):
        page_rows = pooled[start:start + 10]
        figure, axes = plt.subplots(5, 2, figsize=(15, 16), sharex=True, sharey=True)
        figure.subplots_adjust(left=.075, right=.985, bottom=.055, top=.885, hspace=.43, wspace=.10)
        title = figure.suptitle("Shared canonical units: training and development curves", fontsize=16, y=.988)
        subtitle = figure.text(.5, .963, f"Pooled across trained lengths 8, 10 and 12 | Page {page}",
                               ha="center", va="top", fontsize=11, color="#444444")
        for axis, row in zip(axes.flat, page_rows):
            iid = row["instance_id"]
            for seed, color in ((617, "#1769aa"), (829, "#db7417")):
                for split, style in (("train", "--"), ("dev", "-")):
                    points = by_curve[iid, seed, split]
                    if [int(p["step"]) for p in points] != list(range(0, 2401, 200)):
                        raise ValueError(f"Unexpected recorded update inventory: {iid}, {seed}, {split}")
                    axis.plot([int(p["step"]) for p in points], [float(p["accuracy"]) for p in points],
                              color=color, linestyle=style, linewidth=1.5, label=f"{seed} {split}")
                    plotted_points += len(points)
            ceiling = float(row["dev_generation_compatible_fraction_mean"])
            axis.axhline(ceiling, color="black", linestyle=":", linewidth=1,
                         label="Mean dev generation ceiling")
            axis.set_title(textwrap.fill(iid, 45), fontsize=10, pad=8)
            axis.set_ylim(-.02, 1.02)
            axis.set_xlim(0, 2400)
            axis.set_xticks([0, 600, 1200, 1800, 2400])
            axis.tick_params(labelsize=9, labelbottom=True)
            axis.grid(alpha=.2)
        for axis in list(axes.flat)[len(page_rows):]:
            axis.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        legend = figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .941),
                               ncol=5, frameon=False, fontsize=10, handlelength=2.8,
                               columnspacing=1.7)
        figure.supxlabel("Optimizer updates", fontsize=12, y=.025)
        figure.supylabel("Complete generated answer-plus-EOS accuracy", fontsize=12, x=.018)
        figure.text(.5, .008, "Same two data/initialization units as v0.1; no new model evaluation.",
                    ha="center", va="bottom", fontsize=9, color="#555555")
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        title_box = title.get_window_extent(renderer)
        subtitle_box = subtitle.get_window_extent(renderer)
        legend_box = legend.get_window_extent(renderer)
        panel_top = max(axis.title.get_window_extent(renderer).y1 for axis in axes.flat if axis.get_visible())
        if not (title_box.y0 > subtitle_box.y1 and subtitle_box.y0 > legend_box.y1 and legend_box.y0 > panel_top):
            raise AssertionError("Presentation header overlaps; preserve this attempted output directory")
        names = []
        for extension in ("png", "svg"):
            name = f"learning_curves_{page:02d}.{extension}"
            figure.savefig(out / name, dpi=160)
            names.append(name)
        page_records.append(dict(page=page, instances=[r["instance_id"] for r in page_rows], files=names,
                                 header_nonoverlap_checked=True))
        plt.close(figure)

    if source_hashes != {relative(path): sha(path) for path in SOURCES}:
        raise AssertionError("Source CSV changed during presentation generation")
    if sha(FINAL / "MANIFEST.json") != final_manifest_hash:
        raise AssertionError("The audited final manifest changed during presentation generation")
    readme = """# Readable calibration curve pages

These presentation-only figures redraw the saved pooled learning curves with
separate title, subtitle and legend bands. The earlier audited `final/` output,
its manifest and the frozen report generator are unchanged.

Each measured instance has the same four recorded curves: training and
development for the original initialization/data pairings 617 and 829. The
dotted line is the recorded mean development generation ceiling. Values are
read directly from `../final/learning_curves.csv` and
`../final/endpoint_summary.csv`; no smoothing, new averaging, checkpoint
selection, model execution, input generation or test evaluation is performed.
The context-rejected instance has no invented curve. Lengths 8, 10 and 12 all
occur in training, and these are the same data units as the v0.1 calibration.

The inherited eight-token generation budget limits some structured answers.
The majority-answer comparator is uncapped; see
`../../BASELINE_AND_REPRODUCTION_NOTES.md` before interpreting negative gains.
Numerical results and the complete endpoint table remain in
`../final/RESULT_REPORT.md` and its source CSVs.

Reproduce from the repository root into a **new** destination:

```powershell
.venv/Scripts/python.exe -B experiments/platform_calibration_v02/reuse_authorized_v1/plot_readable_curves.py --out experiments/platform_calibration_v02/reuse_authorized_v1/artifacts/presentation_reproduction
```

The writer rejects existing destinations. `MANIFEST.json` records source CSV
hashes, the unchanged final-manifest hash, script hash, figure inventory and
layout checks. Both PNG and SVG versions are provided.
"""
    with (out / "README.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(readme)
    write_json(out / "MANIFEST.json", dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        purpose="Presentation-only layout correction; existing scientific outputs remain unchanged",
        source_sha256=source_hashes, final_manifest_sha256=final_manifest_hash,
        script_path=relative(Path(__file__)), script_sha256=sha(Path(__file__)),
        matplotlib_version=matplotlib.__version__, completed_instances=len(pooled),
        pooled_curve_rows=plotted_points, data_recomputed=False, model_forward_calls=0,
        training_updates=0, test_inputs_opened=False, pages=page_records,
        outputs={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(json.dumps(dict(status="passed", output=relative(out), pages=len(page_records),
                          instances=len(pooled), plotted_points=plotted_points)))


if __name__ == "__main__":
    main()
