"""Recompute published task-count v2 interactions without models or dependencies.

Only archived CSV tables are read. The primary slice is development-selected,
length=all. Each interaction is computed within one paired repeat before means
and sample standard deviations are calculated across the three repeats.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys


TOLERANCE = 1e-12
TARGETS = ("lis_length", "fixed_points")
CHAINS = ("chain_alpha", "chain_beta")
SEEDS = (4103, 5207, 6301)
BUDGETS = (4_000_000, 32_000_000)
KS = (2, 16)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, context):
    actual, expected = float(actual), float(expected)
    require(
        math.isfinite(actual)
        and math.isfinite(expected)
        and math.isclose(actual, expected, rel_tol=0, abs_tol=TOLERANCE),
        f"{context}: recomputed {actual!r}, published {expected!r}",
    )


def summary(values):
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def run(repo, rebuilt_figures=None):
    repo = repo.resolve()
    source_paths = {
        "raw_test_results": repo / "experiments/task_count_v2/raw_test_results.csv",
        "budget_interaction_summary": repo / "experiments/task_count_v2/budget_interaction_summary.csv",
        "figure_plot_data": repo / "publication/figures/PLOT_DATA.csv",
        "figure_plot_summaries": repo / "publication/figures/PLOT_SUMMARIES.csv",
    }
    loaded = {}
    for key, path in source_paths.items():
        require(path.is_file(), f"Required published source is missing: {path}")
        loaded[key] = read_csv(path)

    raw = [
        row for row in loaded["raw_test_results"][1]
        if row["checkpoint"] == "development_selected"
        and row["length"] == "all"
        and row["chain"] in CHAINS
        and row["target"] in TARGETS
    ]
    require(len(raw) == 48, f"Expected 48 selected pretrained rows, found {len(raw)}")
    lookup = {}
    paired_seed_metadata = {}
    seed_fields = (
        "data_seed", "test_seed", "pretrain_sampling_seed", "adaptation_sampling_seed",
        "pretrain_dropout_seed", "adaptation_dropout_seed",
    )
    for row in raw:
        key = (
            row["target"], row["chain"], int(row["k"]),
            int(row["budget_tokens"]), int(row["initialization_seed"]),
        )
        require(key not in lookup, f"Duplicate selected result key: {key}")
        require(key[2] in KS and key[3] in BUDGETS and key[4] in SEEDS,
                f"Unexpected selected design unit: {key}")
        require(row["condition"] == "pretrained" and row["split"] == "test",
                f"Unexpected condition or archived split for {key}")
        require(row["task_id"] == row["target"] and json.loads(row["params"]) == {},
                f"Unexpected task or parameters for {key}")
        require(int(row["seed"]) == key[4], f"Inconsistent repeat identifiers: {key}")
        correct, count = int(row["correct"]), int(row["count"])
        require(count == 1024 and 0 <= correct <= count, f"Invalid counts: {key}")
        close(correct / count, row["accuracy"], f"Correct/count accuracy: {key}")
        lookup[key] = row
        metadata = {name: int(row[name]) for name in seed_fields}
        if key[4] in paired_seed_metadata:
            require(metadata == paired_seed_metadata[key[4]],
                    f"Seed metadata differ within paired repeat {key[4]}")
        else:
            paired_seed_metadata[key[4]] = metadata

    published = {}
    for row in loaded["budget_interaction_summary"][1]:
        if row["checkpoint"] != "development_selected" or row["length"] != "all":
            continue
        key = row["target"], row["chain"]
        require(key not in published, f"Duplicate published summary: {key}")
        published[key] = row
    require(set(published) == {(t, c) for t in TARGETS for c in CHAINS},
            "Published summary does not contain exactly the four expected groups")

    plotted_summaries = {}
    for row in loaded["figure_plot_summaries"][1]:
        if row["figure"] != "figure2" or row["panel"] != "d":
            continue
        require(row["study"] == "task_count_v2", "Unexpected Figure 2d study")
        require(row["estimand"] == "(k16-k2)32M - (k16-k2)4M",
                "Unexpected Figure 2d estimand")
        key = tuple(row["group"].split("/"))
        require(key not in plotted_summaries, f"Duplicate plotted summary: {key}")
        plotted_summaries[key] = row
    require(set(plotted_summaries) == set(published),
            "Figure summaries do not contain exactly the four expected groups")

    plotted_units = {}
    for row in loaded["figure_plot_data"][1]:
        if row["figure"] != "figure2" or row["panel"] != "d":
            continue
        require(row["study"] == "task_count_v2", "Unexpected Figure 2d unit study")
        key = (*row["group"].split("/"), int(row["repeat_id"]))
        require(key not in plotted_units, f"Duplicate plotted unit: {key}")
        plotted_units[key] = row
    require(len(plotted_units) == 12, "Expected 12 plotted paired interactions")

    groups = []
    for target in TARGETS:
        for chain in CHAINS:
            key = target, chain
            repeats = []
            for seed in SEEDS:
                accuracies = {}
                input_rows = []
                for budget in BUDGETS:
                    for k in KS:
                        row = lookup[target, chain, k, budget, seed]
                        accuracy = int(row["correct"]) / int(row["count"])
                        accuracies[budget, k] = accuracy
                        input_rows.append({
                            "run_id": row["run_id"], "budget_tokens": budget, "k": k,
                            "correct": int(row["correct"]), "count": int(row["count"]),
                            "accuracy": accuracy,
                        })
                d4 = 100 * (accuracies[4_000_000, 16] - accuracies[4_000_000, 2])
                d32 = 100 * (accuracies[32_000_000, 16] - accuracies[32_000_000, 2])
                interaction = d32 - d4
                close(interaction, published[key][f"seed{seed}_interaction_pp"],
                      f"Published per-repeat interaction: {key}, {seed}")
                close(interaction, plotted_units[target, chain, seed]["value"],
                      f"Plotted per-repeat interaction: {key}, {seed}")
                repeats.append({
                    "initialization_seed": seed, **paired_seed_metadata[seed],
                    "source_rows": input_rows,
                    "d4_k16_minus_k2_pp": d4,
                    "d32_k16_minus_k2_pp": d32,
                    "paired_budget_interaction_pp": interaction,
                })
            require(int(published[key]["independent_paired_repeats"]) == len(SEEDS),
                    f"Wrong published repeat count: {key}")
            require(int(plotted_summaries[key]["repeat_units"]) == len(SEEDS),
                    f"Wrong plotted repeat count: {key}")
            group = {"target": target, "chain": chain, "independent_paired_repeats": 3,
                     "repeats": repeats, "statistics": {}}
            for metric in (
                "d4_k16_minus_k2_pp",
                "d32_k16_minus_k2_pp",
                "paired_budget_interaction_pp",
            ):
                result = summary([unit[metric] for unit in repeats])
                for statistic, value in result.items():
                    close(value, published[key][f"{metric}_{statistic}"],
                          f"Published {metric} {statistic}: {key}")
                group["statistics"][metric] = result
            interaction_summary = group["statistics"]["paired_budget_interaction_pp"]
            for statistic in ("mean", "sample_sd", "min", "max"):
                close(interaction_summary[statistic], plotted_summaries[key][statistic],
                      f"Figure interaction {statistic}: {key}")
            groups.append(group)

    rebuilt_check = {"requested": rebuilt_figures is not None}
    if rebuilt_figures is not None:
        rebuilt_check["status"] = "passed"
        rebuilt_check["files"] = []
        for name, published_key in (
            ("PLOT_DATA.csv", "figure_plot_data"),
            ("PLOT_SUMMARIES.csv", "figure_plot_summaries"),
        ):
            rebuilt = rebuilt_figures.resolve() / name
            require(rebuilt.is_file(), f"Rebuilt table missing: {rebuilt}")
            rebuilt_table = read_csv(rebuilt)
            require(rebuilt_table == loaded[published_key],
                    f"Rebuilt figure table differs in header, rows or cell values: {name}")
            rebuilt_check["files"].append({
                "name": name, "rows": len(rebuilt_table[1]), "sha256": digest(rebuilt),
                "comparison": "Exact header, row order and string-cell equality",
            })

    return {
        "schema_version": "permworld-dmlr-budget-interaction-check-v1",
        "status": "passed",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "dependencies": "Python standard library only",
        "training_or_model_evaluations_run": False,
        "archived_result_slice": {
            "checkpoint": "development_selected", "length": "all",
            "targets": list(TARGETS), "chains": list(CHAINS),
            "initialization_seeds": list(SEEDS), "raw_pretrained_rows": len(raw),
        },
        "formula": "I = 100 * [(accuracy(k16,32M)-accuracy(k2,32M))"
                   " - (accuracy(k16,4M)-accuracy(k2,4M))]",
        "uncertainty": "Sample SD across three within-repeat differences; "
                       "shared models/baselines are not additional repeats",
        "absolute_tolerance": TOLERANCE,
        "sources": {
            name: {"path": path.relative_to(repo).as_posix(),
                   "sha256": digest(path), "bytes": path.stat().st_size}
            for name, path in source_paths.items()
        },
        "groups": groups,
        "rebuilt_figure_table_check": rebuilt_check,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="Clean repository checkout")
    parser.add_argument("--out", required=True, type=Path, help="New JSON evidence file")
    parser.add_argument("--rebuilt-figures", type=Path,
                        help="Optional rebuilt figure directory for exact CSV comparison")
    args = parser.parse_args()
    output = args.out.resolve()
    require(not output.exists(), f"Evidence file exists; use a fresh --out path: {output}")
    report = run(args.repo, args.rebuilt_figures)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "groups_checked": len(report["groups"]),
        "paired_repeat_interactions_checked": 12, "output": str(output),
    }))


if __name__ == "__main__":
    main()
