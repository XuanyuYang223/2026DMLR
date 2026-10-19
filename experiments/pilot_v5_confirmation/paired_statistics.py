"""Frozen paired-data analysis; no model imports, training, or method selection.

This module deliberately avoids the name statistics.py so it cannot shadow the
Python standard-library statistics module used by the existing project.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4 = HERE.parent / "pilot_v4_backgrounds" / "artifacts"
METRICS = ("delta_B_minus_A", "delta_C_minus_A", "delta_B_minus_C")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Required table is empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def average(values):
    values = list(values)
    return sum(values) / len(values)


def sample_sd(values):
    values = list(values)
    center = average(values)
    return math.sqrt(sum((x - center) ** 2 for x in values) / (len(values) - 1))


def t4_cdf(value):
    u = value / math.sqrt(value * value + 4)
    return 0.5 + 0.75 * u - 0.25 * u ** 3


def t4_quantile(probability):
    if not 0.5 < probability < 1:
        raise ValueError("Only positive df=4 quantiles are required")
    lo, hi = 0.0, 1000.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if t4_cdf(mid) < probability:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sign_diagnostics(values):
    values = list(values)
    observed = abs(sum(values))
    count = sum(abs(sum(s * x for s, x in zip(signs, values))) >= observed - 1e-12
                for signs in itertools.product((-1, 1), repeat=len(values)))
    nonzero = [x for x in values if x != 0]
    positive = sum(x > 0 for x in nonzero)
    n = len(nonzero)
    tail = sum(math.comb(n, k) for k in range(min(positive, n - positive) + 1)) / 2 ** n
    return {
        "sign_flip_two_sided_p": count / 2 ** len(values),
        "sign_flip_patterns": 2 ** len(values),
        "sign_test_two_sided_p": min(1.0, 2 * tail),
        "sign_test_nonzero_n": n,
        "positive_count": positive,
        "negative_count": n - positive,
        "zero_count": len(values) - n,
    }


def summarize_values(values, primary=False, direction=0):
    values = list(values)
    result = {"n": len(values), "mean": average(values), "sample_sd": sample_sd(values),
              "min": min(values), "max": max(values)}
    if primary:
        if len(values) != 5:
            raise ValueError("Confirmation requires all five prespecified repeat units")
        se = result["sample_sd"] / math.sqrt(5)
        t95, tfamily = t4_quantile(0.975), t4_quantile(0.9875)
        leave_one = [average(values[:i] + values[i + 1:]) for i in range(5)]
        result.update({
            "standard_error": se, "df": 4,
            "t_95_critical": t95, "t_family_critical": tfamily,
            "ci95_lower": result["mean"] - t95 * se,
            "ci95_upper": result["mean"] + t95 * se,
            "family95_lower": result["mean"] - tfamily * se,
            "family95_upper": result["mean"] + tfamily * se,
            "prespecified_direction": direction,
            "direction_matching_count": sum(x * direction > 0 for x in values),
            "mean_matches_direction": result["mean"] * direction > 0,
            "leave_one_mean_min": min(leave_one), "leave_one_mean_max": max(leave_one),
        })
        result.update(sign_diagnostics(values))
    return result


def paired_differences(raw, contrasts, seeds):
    index = {}
    for row in raw:
        key = (row["target"], int(row["seed"]), row["background"], row["condition"])
        if key in index:
            raise ValueError(f"Duplicate selected cell: {key}")
        index[key] = row
    output = []
    for contrast in contrasts:
        target = contrast["target"]
        original = contrast["original_background"]
        new = contrast["new_background"]
        for seed in seeds:
            A = index[target, seed, "shared", "A"]
            B0 = index[target, seed, original, "B"]
            C0 = index[target, seed, original, "C"]
            B1 = index[target, seed, new, "B"]
            C1 = index[target, seed, new, "C"]
            cells = (A, B0, C0, B1, C1)
            for key in ("data_generation_seed", "model_initialization_seed", "test_count"):
                if len({str(cell[key]) for cell in cells}) != 1:
                    raise ValueError(f"Unpaired {key}: {target}, {seed}, {new}")
            a, b0, c0, b1, c1 = [float(row["test_accuracy"]) for row in cells]
            row = {"target": target, "seed": seed,
                   "model_initialization_seed": int(A["model_initialization_seed"]),
                   "data_generation_seed": int(A["data_generation_seed"]),
                   "original_background": original, "new_background": new,
                   "test_count": int(A["test_count"]), "A_shared": a,
                   "B_original": b0, "C_original": c0, "B_new": b1, "C_new": c1,
                   "B_minus_A_original": b0 - a, "B_minus_A_new": b1 - a,
                   "C_minus_A_original": c0 - a, "C_minus_A_new": c1 - a,
                   "B_minus_C_original": b0 - c0, "B_minus_C_new": b1 - c1,
                   "delta_B_minus_A": (b1 - a) - (b0 - a),
                   "delta_C_minus_A": (c1 - a) - (c0 - a),
                   "delta_B_minus_C": (b1 - c1) - (b0 - c0)}
            if abs(row["delta_B_minus_C"] - (row["delta_B_minus_A"] - row["delta_C_minus_A"])) > 1e-12:
                raise AssertionError("Paired interaction algebra failed")
            output.append(row)
    return output


def summarize_differences(differences, contrasts, confirmation):
    summaries = []
    for contrast in contrasts:
        matched = [row for row in differences if row["target"] == contrast["target"]
                   and row["new_background"] == contrast["new_background"]]
        for metric in METRICS:
            primary = confirmation and metric == "delta_B_minus_C"
            summaries.append({"target": contrast["target"],
                              "original_background": contrast["original_background"],
                              "new_background": contrast["new_background"], "metric": metric,
                              "primary": primary,
                              **summarize_values([row[metric] for row in matched], primary,
                                                 contrast.get("expected_direction", 0))})
    return summaries


def self_check():
    assert abs(t4_quantile(0.975) - 2.7764451051977987) < 1e-10
    assert abs(t4_cdf(t4_quantile(0.9875)) - 0.9875) < 1e-14
    assert sign_diagnostics([1, 2, 3, 4, 5])["sign_flip_two_sided_p"] == 0.0625
    assert sign_diagnostics([1, 2, 3, 4, 5])["sign_test_two_sided_p"] == 0.0625
    assert sign_diagnostics([0, 0, 0, 0, 0])["sign_test_two_sided_p"] == 1
    assert sign_diagnostics([1, -1, 2, -2, 0])["sign_flip_two_sided_p"] == 1
    return {"passed": True, "t_95_critical": t4_quantile(0.975),
            "t_family_critical": t4_quantile(0.9875),
            "smallest_two_sided_exact_p_n5": 0.0625}


def run(mode, input_path=None, out=None):
    plan = json.loads((HERE / "stats_plan.json").read_text(encoding="utf-8"))
    confirmation = mode == "confirmation"
    if confirmation:
        contrasts = plan["contrasts"]
        seeds = [row["seed"] for row in plan["repetitions"]]
        input_path = Path(input_path or HERE / "artifacts" / "results_selected.csv")
        out = Path(out or HERE / "artifacts")
    else:
        contrasts = [{"target": target, "original_background": "g0_v3", "new_background": new}
                     for target in ("fixed_points", "longest_decreasing_run")
                     for new in ("g1_recoils", "g2_deficiencies")]
        seeds = [17, 42, 73]
        input_path = Path(input_path or V4 / "results_selected.csv")
        out = Path(out or HERE / "exploratory")
    raw = read_csv(input_path)
    if confirmation:
        expected = {row["seed"]: row for row in plan["repetitions"]}
        if len(raw) != 50:
            raise ValueError("Expected 50 unique selected models; A is shared")
        for row in raw:
            exp = expected[int(row["seed"])]
            for key in ("model_initialization_seed", "data_generation_seed"):
                if int(row[key]) != exp[key]:
                    raise ValueError(f"Unexpected confirmation seed provenance: {row}")
    differences = paired_differences(raw, contrasts, seeds)
    summaries = summarize_differences(differences, contrasts, confirmation)
    write_csv(out / f"{mode}_paired_differences.csv", differences)
    write_csv(out / f"{mode}_summary.csv", summaries)
    audit = {"mode": mode, "input": str(input_path.resolve()), "input_sha256": sha256(input_path),
             "plan_sha256": sha256(HERE / "stats_plan.json"),
             "statistics_sha256": sha256(Path(__file__)), "seed_count": len(seeds),
             "difference_rows": len(differences), "summary_rows": len(summaries),
             "self_check": self_check(), "unit": "accuracy fraction; multiply by 100 for percentage points",
             "pairing": "Within-repeat differences computed directly from selected raw A/B/C accuracies; shared A cancels algebraically."}
    if not confirmation:
        old = read_csv(V4 / "background_interactions.csv")
        for row in differences:
            match = [r for r in old if r["target"] == row["target"] and int(r["seed"]) == row["seed"]
                     and r["earlier_background"] == row["original_background"]
                     and r["later_background"] == row["new_background"]]
            assert len(match) == 1
            assert abs(float(match[0]["interaction_test_accuracy"]) - row["delta_B_minus_C"]) < 1e-12
        audit["independent_v4_interaction_reconciliation"] = "all 12 exact within 1e-12"
    (out / f"{mode}_statistics_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return differences, summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("exploratory", "confirmation", "self-check"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.mode == "self-check":
        print(json.dumps(self_check(), indent=2))
    else:
        differences, summaries = run(args.mode, args.input, args.out)
        print(json.dumps({"mode": args.mode, "difference_rows": len(differences),
                          "summary_rows": len(summaries)}, indent=2))
