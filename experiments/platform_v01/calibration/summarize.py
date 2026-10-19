"""Summarize the finite v0.1 calibration, retaining every preselected task."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from permworld_platform.calibration import CFG, TASKS, read_json, run_id, task_slug, write_csv, write_json


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out = args.out.resolve()
    all_curves, selected, compute, exposures, per_length_exposures = [], [], [], [], []
    completed = []
    for repeat in CFG["repeats"]:
        for task in TASKS:
            rid = run_id(task, repeat)
            run_dir = out / "runs" / rid
            record = read_json(run_dir / "COMPLETE.json")
            curve = rows(run_dir / "curves.csv")
            all_curves.extend(curve)
            completed.append(record)
            dev_rows = [r for r in curve if r["split"] == "dev" and r["length"] == "all" and int(r["step"]) > 0]
            chosen = min(dev_rows, key=lambda r: (-float(r["accuracy"]), float(r["answer_ce"]), int(r["step"])))
            assert int(chosen["step"]) == record["best_step"]
            for checkpoint, update in [("endpoint", CFG["updates"]), ("dev_best", record["best_step"])]:
                for row in curve:
                    if int(row["step"]) == update:
                        selected.append({"checkpoint": checkpoint, **row,
                            "above_majority_pp": 100 * (float(row["accuracy"]) - float(row["majority_accuracy"])),
                            "role": "primary fixed-budget reference" if checkpoint == "endpoint" else "descriptive development-selected reference"})
            compute.append({k: record[k] for k in ["run_id", "initialization_seed", "data_seed", "task_id", "family", "params",
                "updates", "sample_exposures", "input_tokens", "padded_input_tokens", "supervised_tokens", "answer_tokens", "seconds",
                "parameter_count", "peak_memory_bytes", "initial_state_sha256", "best_step"]})
            exposures.extend(rows(run_dir / "input_exposures.csv"))
            for length in CFG["lengths"]:
                n = str(length)
                values = [int(r["exposures"]) for r in exposures if r["run_id"] == rid and int(r["length"]) == length]
                per_length_exposures.append({"run_id": rid, "task_id": task["task_id"], "params": json.dumps(task["params"]),
                    "initialization_seed": repeat["initialization_seed"], "data_seed": repeat["data_seed"], "length": length,
                    "updates": record["length_updates"][n], "distinct_train_inputs": len(values),
                    "exposures": sum(values), "min_exposure_per_input": min(values), "max_exposure_per_input": max(values),
                    **record["tokens_by_length"][n]})
    write_csv(out / "learning_curves.csv", all_curves)
    write_csv(out / "results_by_seed.csv", selected)
    for repeat in CFG["repeats"]:
        seed = repeat["initialization_seed"]
        write_csv(out / f"results_seed{seed}.csv", [r for r in selected if int(r["initialization_seed"]) == seed])
    write_csv(out / "compute.csv", compute)
    write_csv(out / "input_exposures.csv", exposures)
    write_csv(out / "length_exposures.csv", per_length_exposures)
    task_summary = []
    for task in TASKS:
        for length in [*CFG["lengths"], "all"]:
            task_rows = [r for r in selected if r["task_id"] == task["task_id"] and r["split"] == "dev"
                         and r["checkpoint"] == "endpoint" and str(r["length"]) == str(length)]
            accuracy = np.asarray([float(r["accuracy"]) for r in task_rows])
            baseline = np.asarray([float(r["majority_accuracy"]) for r in task_rows])
            train_rows = [r for r in selected if r["task_id"] == task["task_id"] and r["split"] == "train"
                         and r["checkpoint"] == "endpoint" and str(r["length"]) == str(length)]
            train_accuracy = np.asarray([float(r["accuracy"]) for r in train_rows])
            task_summary.append({"task_id": task["task_id"], "params": json.dumps(task["params"]), "family": task["family"],
                "length": length, "repeats": len(task_rows), "endpoint_train_accuracy_mean": train_accuracy.mean(),
                "endpoint_dev_accuracy_mean": accuracy.mean(), "endpoint_dev_accuracy_min": accuracy.min(),
                "endpoint_dev_accuracy_max": accuracy.max(), "majority_dev_accuracy_mean": baseline.mean(),
                "endpoint_above_majority_pp_mean": 100 * (accuracy - baseline).mean(),
                "status": "retained; limited fixed-budget development evidence, not a task exclusion criterion"})
    write_csv(out / "task_summary.csv", task_summary)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    colors = {CFG["repeats"][0]["initialization_seed"]: "#1767a6", CFG["repeats"][1]["initialization_seed"]: "#d76623"}
    for family in dict.fromkeys(t["family"] for t in TASKS):
        tasks = [t for t in TASKS if t["family"] == family]
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        for col, task in enumerate(tasks):
            for repeat in CFG["repeats"]:
                seed = repeat["initialization_seed"]
                for split, style in [("train", "--"), ("dev", "-")]:
                    cr = [r for r in all_curves if r["task_id"] == task["task_id"] and int(r["initialization_seed"]) == seed
                          and r["length"] == "all" and r["split"] == split]
                    cr.sort(key=lambda r: int(r["step"]))
                    xs = [int(r["step"]) for r in cr]
                    axes[0, col].plot(xs, [100 * float(r["accuracy"]) for r in cr], style, color=colors[seed], label=f"{seed} {split}")
                    axes[1, col].plot(xs, [float(r["answer_ce"]) for r in cr], style, color=colors[seed], label=f"{seed} {split}")
                baseline = next(float(r["majority_accuracy"]) for r in cr)
                axes[0, col].axhline(100 * baseline, color=colors[seed], linestyle=":", alpha=.65)
            axes[0, col].set_title(task_slug(task))
            axes[0, col].set_ylim(0, 102)
            for row in range(2):
                axes[row, col].set_xlabel("Updates")
                axes[row, col].grid(alpha=.2)
                axes[row, col].legend(fontsize=8)
            axes[0, col].set_ylabel("Exact answer + EOS accuracy (%)")
            axes[1, col].set_ylabel("Teacher-forced answer token CE")
        fig.suptitle(f"{family}: fixed-budget single-task calibration; dotted = train-derived dev majority")
        for suffix in ("png", "svg"):
            fig.savefig(figures / f"{family}_curves.{suffix}", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, task in zip(axes.flat, TASKS):
        for repeat in CFG["repeats"]:
            seed = repeat["initialization_seed"]
            task_rows = [r for r in selected if r["task_id"] == task["task_id"] and int(r["initialization_seed"]) == seed
                         and r["checkpoint"] == "endpoint" and r["split"] == "dev" and r["length"] != "all"]
            task_rows.sort(key=lambda r: int(r["length"]))
            axis.plot([int(r["length"]) for r in task_rows], [100 * float(r["accuracy"]) for r in task_rows],
                      "o-", color=colors[seed], label=f"{seed} endpoint")
            axis.plot([int(r["length"]) for r in task_rows], [100 * float(r["majority_accuracy"]) for r in task_rows],
                      ":", color=colors[seed], label=f"{seed} majority")
        axis.set_title(task_slug(task))
        axis.set_xlabel("Permutation length")
        axis.set_ylabel("Dev exact accuracy (%)")
        axis.set_ylim(0, 102)
        axis.set_xticks(CFG["lengths"])
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    for suffix in ("png", "svg"):
        fig.savefig(figures / f"endpoint_by_length.{suffix}", dpi=150)
    plt.close(fig)
    timings = {"runs": len(compute), "updates": sum(r["updates"] for r in compute),
        "input_tokens": sum(r["input_tokens"] for r in compute), "padded_input_tokens": sum(r["padded_input_tokens"] for r in compute),
        "supervised_tokens": sum(r["supervised_tokens"] for r in compute),
        "sample_exposures": sum(r["sample_exposures"] for r in compute),
        "run_seconds": sum(r["seconds"] for r in compute), "test_evaluated": False,
        "training_timer_scope": "Full train/dev evaluations and checkpoint saves included; data encoding, summaries/audits excluded"}
    write_json(out / "SUMMARY.json", timings)
    text = ["# PermWorld v0.1 Single-Task Calibration\n\n", "This round preselected six tasks, two seeds, and a uniform budget of 2400 updates. It performs no transfer-combination search and removes no task based on performance. The endpoint is the primary fixed-budget reference; the development-best checkpoint is descriptive only. The platform test set was not used in calibration.\n\n",
            f"The model retains the 4-layer, width-256, 8-head Transformer architecture. For the explicitly versioned new-task and structured-answer interface, the vocabulary expands from the historical 188 entries to 203, yielding {compute[0]['parameter_count']:,} parameters. Its parameter space is not identical to that of historical checkpoints. For each seed, all six tasks start from the same new initialization and each creates AdamW with empty state; no legacy transfer weights are loaded.\n\n",
            "Training uses lengths 8, 10, and 12, with 1024 training inputs and 256 development inputs per length. The fixed seed shuffles the length order every three updates, giving exactly 800 updates per length. All tasks share each repeat's data and initial model weights. All three lengths occur in training, so the per-length results are not tests of unseen-length extrapolation.\n\n",
            "Accuracy requires generating the entire answer and EOS correctly from the prompt alone. Cross-entropy uses teacher forcing and is a different metric. The output encoding is explicitly versioned to accommodate increasing_subsequence_count(k=3) answers beyond the original single-token range. All six representatives have scalar outputs. The platform correctly encodes structured answers, but this round does not calibrate the learnability of every structured-output task.\n\n",
            "| Task | Mean endpoint training accuracy | Mean endpoint development accuracy | Two-seed development range | Development majority baseline | Above majority baseline |\n|---|---:|---:|---:|---:|---:|\n"]
    for row in task_summary:
        if row["length"] == "all":
            text.append(f"| {row['task_id']}{row['params'] if row['params'] != '{}' else ''} | {100 * row['endpoint_train_accuracy_mean']:.2f}% | {100 * row['endpoint_dev_accuracy_mean']:.2f}% | {100 * row['endpoint_dev_accuracy_min']:.2f}–{100 * row['endpoint_dev_accuracy_max']:.2f}% | {100 * row['majority_dev_accuracy_mean']:.2f}% | {row['endpoint_above_majority_pp_mean']:+.2f} pp |\n")
    cy12 = next(r for r in task_summary if r["task_id"] == "cycle_count" and r["length"] == 12)
    longest = next(r for r in task_summary if r["task_id"] == "longest_cycle" and r["length"] == "all")
    subseq = next(r for r in task_summary if r["task_id"] == "increasing_subsequence_count" and r["length"] == "all")
    text.extend([f"\nOverall means do not conceal the difficulty of cycle tasks: at length 12, cycle_count has mean development accuracy of only {100 * cy12['endpoint_dev_accuracy_mean']:.2f}%, below that length's majority baseline of {100 * cy12['majority_dev_accuracy_mean']:.2f}%. longest_cycle has mean training accuracy of {100 * longest['endpoint_train_accuracy_mean']:.2f}% and development accuracy of {100 * longest['endpoint_dev_accuracy_mean']:.2f}%, showing both incomplete fitting and a generalization gap. These cannot be attributed to a single mechanism.\n",
        f"\nincreasing_subsequence_count(k=3) reaches {100 * subseq['endpoint_dev_accuracy_mean']:.2f}% development accuracy, exceeding the majority baseline by {subseq['endpoint_above_majority_pp_mean']:.2f} percentage points, but its training accuracy is only {100 * subseq['endpoint_train_accuracy_mean']:.2f}%. It remains listed as insufficiently learned under the current configuration. Both permutation-block representatives reach development accuracy near 99%, clearly above their roughly 77% majority baseline. These are limited-budget development observations, not guarantees for entire task families.\n",
        "\nMajority answers are selected only from training labels for each task and length, then applied to development data. High raw accuracy may reflect answer imbalance, so gains over majority and per-length curves must also be considered. With only two seeds and development-based checkpoint selection, these results cannot establish reliable transfer ability or test performance. All insufficiently learned tasks are retained; inadequate budget remains inseparable from encoding/optimization difficulty. This round explicitly changes input lengths, training sample counts, and encoding interfaces, so its accuracies are not a controlled comparison with v5.\n",
        f"\nThe {timings['runs']} training runs used {timings['updates']:,} updates, {timings['sample_exposures']:,} sample exposures, and {timings['input_tokens']:,} actual input tokens. Right-padding computation is reported separately. Cumulative training time was {timings['run_seconds'] / 60:.2f} minutes, including full train/dev evaluation and saving.\n",
        "\n- [Per-seed and per-length results](results_by_seed.csv)\n- [Complete train/dev curves](learning_curves.csv)\n- [Actual exposures for each input](input_exposures.csv)\n- [Per-length budgets](length_exposures.csv)\n- [Run costs](compute.csv)\n- [Per-length figure](figures/endpoint_by_length.png)\n- [Full development-forward and checkpoint audit](CALIBRATION_AUDIT.json)\n",
        "\nRun reproduction commands from the repository root. First generate and verify a fresh data directory at `experiments/platform_v01_reproduction/dataset` using the [platform reproduction instructions](../README.md). All outputs below go to a new directory; freeze binds its data and the code hashes at that time. Complete existing runs are skipped after hash verification. Incomplete runs restart from the same initial model, without a guarantee of update-level resumption. The archived SMOKE file is also frozen evidence, so repeat smoke tests in a new directory.\n\n```powershell\n.venv/Scripts/python.exe -m permworld_platform.calibration smoke --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data\n.venv/Scripts/python.exe -m permworld_platform.calibration freeze --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data\n.venv/Scripts/python.exe -m permworld_platform.calibration train --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data\n.venv/Scripts/python.exe experiments/platform_v01/calibration/summarize.py --out experiments/platform_v01_reproduction/calibration\n.venv/Scripts/python.exe experiments/platform_v01/calibration/audit.py --forwards --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data\n```\n"])
    (out / "CALIBRATION_REPORT.md").write_text("".join(text), encoding="utf-8")
    print(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
