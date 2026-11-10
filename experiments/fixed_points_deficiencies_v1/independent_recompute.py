"""Recompute confirmation statistics from integer prediction counts, without models.

This audit deliberately does not import the training, analysis, or data pipeline
modules. Target labels are computed directly from the permutation definition.
The Student critical value uses numerical density quadrature, independently of
the analytic CDF used by analysis.py.
"""
from __future__ import annotations

import argparse
import ast
import csv
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPEATS = [f"R{i}" for i in range(1, 9)]
CONDITIONS = ["P", "D", "T", "U", "A"]
SIZES = [256, 512, 1024]
KINDS = ["selected", "endpoint"]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual, expected, label, tolerance=1e-10):
    assert math.isfinite(float(actual)) and math.isfinite(float(expected)), label
    assert abs(float(actual) - float(expected)) <= tolerance, (label, actual, expected)


def eos_id():
    """Read literal tokenizer blocks without importing any implementation."""
    path = ROOT / "experiments/permworld_pilot/vendor/neurips_permutations/passage.py"
    assignments = {node.target.id: node.value for node in ast.parse(path.read_text(encoding="utf8")).body
                   if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
    tokens = [f"{i:02d}" for i in range(100)]
    for name in ("ORIGINAL_FIXED_TOKENS", "EXTENDED_FIXED_TOKENS", "PROPERTY_FIXED_TOKENS"):
        tokens.extend(ast.literal_eval(assignments[name]))
    assert len(tokens) == 188 and len(set(tokens)) == 188
    return tokens.index("<EOS>"), digest(path)


def t7_numerical_cdf(value, panels=2048):
    """Composite Simpson integration of the normalized Student(7) density."""
    assert panels % 2 == 0
    normalizer = math.gamma(4) / (math.sqrt(7 * math.pi) * math.gamma(3.5))
    width = value / panels
    density = lambda x: normalizer * (1 + x*x/7) ** -4
    integral = density(0) + density(value)
    integral += 4 * math.fsum(density(i * width) for i in range(1, panels, 2))
    integral += 2 * math.fsum(density(i * width) for i in range(2, panels, 2))
    return .5 + integral * width / 3


def t7_numerical_quantile():
    low, high = 0.0, 10.0
    for _ in range(48):
        middle = (low + high) / 2
        if t7_numerical_cdf(middle) < .975:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def load_pools(artifacts):
    data_dir = artifacts / "data"
    manifest = read_json(data_dir / "manifest.json")
    test_manifest = read_json(data_dir / "test_manifest.json")
    assert test_manifest["primary_manifest_sha256"] == digest(data_dir / "manifest.json")
    assert test_manifest["selection_freeze"]["sha256"] == digest(artifacts / "SELECTIONS_FROZEN.json")
    pools = {}
    for entry in [*manifest["pools"], *test_manifest["pools"]]:
        if entry["role"] not in {"target_train", "test"}:
            continue
        path = data_dir / entry["path"]
        assert digest(path) == entry["sha256"], path
        with np.load(path, allow_pickle=False) as archive:
            values = archive["p"].copy()
        expected = 1024 if entry["role"] == "target_train" else 2048
        assert values.shape == (expected, 10) and int(entry["count"]) == expected
        assert np.all(np.sort(values, axis=1) == np.arange(1, 11)), path
        pools[entry["repeat_id"], entry["role"]] = values
    assert set(pools) == set(itertools.product(REPEATS, ["target_train", "test"]))
    return pools


def recompute_prediction_file(path, result, selection, train, test, eos):
    correct = legacy_correct = count = 0
    losses = []
    train_labels = [sum(int(v) == i + 1 for i, v in enumerate(p)) for p in train]
    counts = [train_labels.count(i) for i in range(11)]
    majority = max(range(11), key=lambda i: counts[i])
    test_labels = [sum(int(v) == i + 1 for i, v in enumerate(p)) for p in test]
    majority_correct = test_labels.count(majority)
    kind = result["checkpoint_kind"]
    expected_step = int(selection["step"]) if kind == "selected" else 800
    with path.open(encoding="utf8") as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            assert index < len(test) and record["example_index"] == index
            assert record["permutation"] == test[index].tolist(), (path, index, "test input")
            for field in ("repeat_id", "condition", "checkpoint_kind", "run_id"):
                assert record[field] == result[field], (path, field)
            assert int(record["target_samples"]) == int(result["target_samples"])
            assert int(record["step"]) == int(result["step"]) == expected_step
            close(record["lr"], result["lr"], "prediction learning rate", 1e-15)
            assert record["checkpoint_sha256"] == result["checkpoint_sha256"] == selection[f"{kind}_sha256"]
            expected_gold = [test_labels[index], eos]
            assert record["gold_answer_ids"] == expected_gold
            generated = record["predicted_ids"]
            assert isinstance(generated, list) and 1 <= len(generated) <= 2
            assert all(isinstance(value, int) and 0 <= value < 188 for value in generated)
            assert (len(generated) == 1) == (generated[0] == eos)
            full = generated == expected_gold
            first = generated[0] == expected_gold[0]
            assert bool(record["correct"]) == full
            assert bool(record["legacy_correct"]) == first
            assert int(record["legacy_answer_prediction_id"]) == generated[0]
            assert bool(record["terminated_by_eos"]) == (eos in generated)
            assert bool(record["truncated"]) == (eos not in generated)
            loss = float(record["answer_nll"])
            assert math.isfinite(loss) and loss >= 0
            losses.append(loss)
            correct += full
            legacy_correct += first
            count += 1
    assert count == 2048 == int(result["count"])
    assert correct == int(result["correct"])
    assert legacy_correct == int(result["legacy_correct"])
    assert majority == int(result["majority_label"])
    assert majority_correct == int(result["majority_correct"])
    close(result["accuracy"], correct/count, "complete accuracy", 1e-12)
    close(result["legacy_accuracy"], legacy_correct/count, "legacy accuracy", 1e-12)
    close(result["majority_accuracy"], majority_correct/count, "training-derived majority", 1e-12)
    # GPU float32 batch reductions and Python summation have different order.
    close(result["answer_ce"], math.fsum(losses)/count, "per-example answer NLL", 5e-6)
    close(result["two_token_ce"], (float(result["answer_ce"])+float(result["eos_ce"]))/2,
          "two-token mean loss", 1e-10)
    assert digest(path) == result["predictions_sha256"]
    return {"complete": Fraction(correct, count), "legacy_answer_token": Fraction(legacy_correct, count)}


def run(artifacts, analysis_dir=None):
    artifacts = Path(artifacts).resolve()
    analysis_dir = Path(analysis_dir).resolve() if analysis_dir else artifacts / "analysis"
    selections_path = artifacts / "SELECTIONS_FROZEN.json"
    frozen = read_json(selections_path)
    assert frozen["status"] == "frozen" and frozen["training_complete"] is True
    assert frozen["candidate_runs"] == 360 and len(frozen["selections"]) == 120
    selected = {(r["repeat_id"], r["condition"], int(r["target_samples"])): r for r in frozen["selections"]}
    assert set(selected) == set(itertools.product(REPEATS, CONDITIONS, SIZES))
    pools = load_pools(artifacts)
    eos, tokenizer_sha = eos_id()
    rows = read_csv(artifacts / "results_test.csv")
    expected = set(itertools.product(REPEATS, CONDITIONS, SIZES, KINDS))
    raw = {}
    hashes = {}
    for result in rows:
        repeat, condition, size, kind = key = (result["repeat_id"], result["condition"],
            int(result["target_samples"]), result["checkpoint_kind"])
        assert key in expected and key not in raw
        selection = selected[repeat, condition, size]
        assert result["run_id"] == selection["run_id"]
        close(result["lr"], selection["lr"], "frozen learning rate", 1e-15)
        assert result["selection_manifest_sha256"] == digest(selections_path)
        test = pools[repeat, "test"]
        input_sha = hashlib.sha256(np.asarray(test, dtype="<i2").tobytes()).hexdigest()
        assert result["test_input_sha256"] == input_sha
        path = artifacts / "predictions" / f"{repeat}_{condition}_n{size}_{kind}.jsonl"
        raw[key] = recompute_prediction_file(path, result, selection,
            pools[repeat, "target_train"][:size], test, eos)
        hashes[path.name] = digest(path)
    assert set(raw) == expected
    scores = {}
    for repeat, condition, kind, metric in itertools.product(REPEATS, CONDITIONS, KINDS,
                                                           ["complete", "legacy_answer_token"]):
        scores[metric, kind, repeat, condition] = (
            raw[repeat, condition, 256, kind][metric]/4 + raw[repeat, condition, 512, kind][metric]/2
            + raw[repeat, condition, 1024, kind][metric]/4) * 100
    score_rows = read_csv(analysis_dir / "scores_per_repeat.csv")
    assert len(score_rows) == len(scores) == 160
    assert len({(r["metric"], r["checkpoint_kind"], r["repeat_id"], r["condition"]) for r in score_rows}) == 160
    for row in score_rows:
        key = row["metric"], row["checkpoint_kind"], row["repeat_id"], row["condition"]
        close(row["S_pct"], scores[key], "score from exact raw counts")
    difference_groups = {}
    difference_rows = read_csv(analysis_dir / "paired_differences.csv")
    assert len(difference_rows) == 1216
    for row in difference_rows:
        metric, kind, repeat, scope = row["metric"], row["checkpoint_kind"], row["repeat_id"], row["scope"]
        left, right = row["contrast"].split("_minus_")
        if scope == "area":
            difference = scores[metric, kind, repeat, left] - scores[metric, kind, repeat, right]
        else:
            size = int(scope[1:])
            difference = (raw[repeat, left, size, kind][metric] - raw[repeat, right, size, kind][metric])*100
        close(row["difference_pp"], difference, "paired difference from exact raw counts")
        group = difference_groups.setdefault((metric, kind, scope, row["contrast"]), {})
        assert repeat not in group
        group[repeat] = float(difference)
    summary_rows = read_csv(analysis_dir / "paired_summary.csv")
    assert len(summary_rows) == len(difference_groups) == 152
    assert len({(r["metric"], r["checkpoint_kind"], r["scope"], r["contrast"]) for r in summary_rows}) == 152
    for row in summary_rows:
        group = difference_groups[row["metric"], row["checkpoint_kind"], row["scope"], row["contrast"]]
        assert set(group) == set(REPEATS)
        values = list(group.values())
        assert int(row["n"]) == 8
        for field, value in [("mean_pp", statistics.mean(values)), ("sample_sd_pp", statistics.stdev(values)),
                             ("min_pp", min(values)), ("max_pp", max(values))]:
            close(row[field], value, field)
    primary = read_json(analysis_dir / "primary_summary.json")
    primary_values = {repeat: float(scores["complete", "selected", repeat, "D"] -
                      scores["complete", "selected", repeat, "U"]) for repeat in REPEATS}
    values = list(primary_values.values())
    mean, sd = statistics.mean(values), statistics.stdev(values)
    critical = t7_numerical_quantile()
    radius = critical * sd/math.sqrt(8)
    close(primary["t_critical"], critical, "independent Student quadrature", 2e-10)
    close(t7_numerical_cdf(critical, panels=4096), .975, "quadrature refinement", 2e-12)
    for field, value in [("mean_pp", mean), ("sample_sd_pp", sd), ("min_pp", min(values)),
                         ("max_pp", max(values)), ("ci95_low_pp", mean-radius), ("ci95_high_pp", mean+radius)]:
        close(primary[field], value, field)
    assert primary["df"] == 7 and primary["n"] == 8
    for repeat, value in primary_values.items():
        close(primary["differences_by_repeat_pp"][repeat], value, repeat)
    report = dict(status="pass", independent_units=8, evaluated_rows=len(rows),
        raw_prediction_rows=2048*len(rows), metric="Complete generated answer plus EOS exact match",
        primary_differences_by_repeat_pp=primary_values, mean_pp=mean, sample_sd_pp=sd,
        min_pp=min(values), max_pp=max(values), ci95_low_pp=mean-radius, ci95_high_pp=mean+radius,
        df=7, t_critical_numerical_quadrature=critical,
        checks=["Independent fixed-point labels", "Published test-input order", "Frozen selection/checkpoint metadata",
                "Full generation and legacy answer-token integer counts", "Training-prefix majority classes",
                "Mean per-example answer NLL", "All per-unit scores and paired differences", "All paired summary moments",
                "Primary 95% t interval from independent numerical quadrature"],
        answer_nll_absolute_tolerance=5e-6,
        sources={"results_test.csv": digest(artifacts / "results_test.csv"),
                 "SELECTIONS_FROZEN.json": digest(selections_path), "tokenizer": tokenizer_sha,
                 "data/manifest.json": digest(artifacts / "data/manifest.json"),
                 "data/test_manifest.json": digest(artifacts / "data/test_manifest.json")},
        raw_prediction_sha256=hashes, model_forwards=0, training_runs=0,
        scope="Recomputes tabular statistics from stored predictions, not GPU training or model inference."
    )
    destination = analysis_dir / "INDEPENDENT_RECOMPUTE.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    parser.add_argument("--analysis-dir", type=Path)
    args = parser.parse_args()
    result = run(args.artifacts, args.analysis_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "raw_prediction_sha256"}, indent=2))
