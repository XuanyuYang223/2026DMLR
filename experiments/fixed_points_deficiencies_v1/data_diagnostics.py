"""Pretest label distributions and independent oracle checks; no model execution.

Only the four already frozen training/development roles are opened. Target
development distributions are repeated for the three declared training-prefix
baselines, without treating those repeated distributions as independent data.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fixed_points_deficiencies_v1 import data_pipeline as data
from experiments.fixed_points_deficiencies_v1.design import REPEATS, SOURCES, TARGET
from permworld_platform import label

TASKS = sorted({TARGET, *(task for tasks in SOURCES.values() for task in tasks)})
ORACLE_PATH = ROOT / "experiments/pilot_v5_confirmation/mathematics/audit_relations.py"


def csvwrite(path, rows):
    with Path(path).open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def histogram(values):
    return np.bincount(values, minlength=11)


def entropy(counts):
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def run(out=HERE / "artifacts"):
    out = Path(out)
    filenames = ["DATA_LABEL_DISTRIBUTIONS.csv", "DATA_LABEL_SUMMARY.csv",
                 "DATA_TASK_MATCHING.csv", "DATA_LABEL_AUDIT.json"]
    if any((out / name).exists() for name in filenames):
        raise FileExistsError("diagnostics already exist; preserve them and use a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data_out = HERE / "artifacts/data"
    manifest = data._assert_manifest(data_out)
    spec = importlib.util.spec_from_file_location("frozen_v5_independent_definitions", ORACLE_PATH)
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)
    distributions, summaries, matching = [], [], []
    checks = inputs = 0
    for repeat in REPEATS:
        rid = repeat["repeat_id"]
        labels = {}
        for role in data.ROLES:
            permutations = data.load_pool(data_out, rid, role)
            matrix = np.empty((len(permutations), len(TASKS)), dtype=np.int16)
            for i, values in enumerate(permutations):
                p = tuple(map(int, values))
                expected = independent.reference(p)
                expected["right_to_left_maxima"] = sum(
                    all(p[j] < p[k] for j in range(k + 1, len(p))) for k in range(len(p)))
                for j, task in enumerate(TASKS):
                    actual = int(label(task, p))
                    if actual != expected[task]:
                        raise AssertionError((rid, role, i, task, p, actual, expected[task]))
                    matrix[i, j] = actual
                    checks += 1
            labels[role] = matrix
            inputs += len(permutations)
        for role in data.ROLES:
            sizes = [256, 512, 1024] if role.startswith("target") else [0]
            for size in sizes:
                values = labels[role][:size] if role == "target_train" else labels[role]
                baseline_values = labels["target_train"][:size] if size else labels["source_train"]
                record = next(r for r in manifest["pools"] if r["repeat_id"] == rid and r["role"] == role)
                for j, task in enumerate(TASKS):
                    counts = histogram(values[:, j])
                    train_counts = histogram(baseline_values[:, j])
                    majority = int(train_counts.argmax())  # Smallest answer breaks ties.
                    identity = dict(repeat_id=rid, role=role, target_samples=size or "",
                                    n=10, task=task, sample_count=len(values),
                                    majority_training_count=len(baseline_values), majority_train_answer=majority,
                                    majority_accuracy=float(counts[majority]/len(values)))
                    for answer, count in enumerate(counts):
                        distributions.append(dict(**identity, answer=answer, count=int(count),
                                                  fraction=float(count/len(values))))
                    summaries.append(dict(**identity, mean=float(values[:, j].mean()),
                        sample_sd=float(values[:, j].std(ddof=1)), entropy_bits=entropy(counts),
                        distinct_answers=int((counts > 0).sum()),
                        minimum_answer=int(values[:, j].min()), maximum_answer=int(values[:, j].max()),
                        constant_on_pool=bool((counts > 0).sum() == 1), input_archive_sha256=record["sha256"]))
        for role in ("source_train", "source_dev"):
            for first, second in [("deficiencies", "right_to_left_maxima"),
                                  ("deficiencies", "peaks"), ("deficiencies", "anti_fixed_points"),
                                  ("right_to_left_maxima", "left_to_right_maxima")]:
                a, b = labels[role][:, TASKS.index(first)], labels[role][:, TASKS.index(second)]
                ah, bh = histogram(a), histogram(b)
                matching.append(dict(repeat_id=rid, role=role, n=10, task_a=first, task_b=second,
                    sample_count=len(a), answer_distribution_total_variation=float(np.abs(ah-bh).sum()/(2*len(a))),
                    mean_a=float(a.mean()), mean_b=float(b.mean()),
                    entropy_a_bits=entropy(ah), entropy_b_bits=entropy(bh),
                    same_input_answer_agreement=float((a == b).mean()),
                    interpretation="Descriptive pool differences; not a difficulty or causal-mechanism test"))
        print(f"[data diagnostics] {rid}: independent labels verified for all pretest objects", flush=True)
    csvwrite(out / filenames[0], distributions)
    csvwrite(out / filenames[1], summaries)
    csvwrite(out / filenames[2], matching)
    report = dict(status="pass", created_at_utc=data.utc_now(), distinct_input_count=inputs,
        independent_label_comparisons=checks, task_count=len(TASKS), tasks=TASKS,
        distribution_rows=len(distributions), summary_rows=len(summaries), matching_rows=len(matching),
        constant_task_pools=sum(row["constant_on_pool"] for row in summaries),
        majority_rule="Smallest answer among maximum-frequency training labels; source roles use source_train, target roles use the corresponding target_train prefix",
        target_dev_repetition="Target development frequencies repeat for each target subset size to expose the size-specific training-majority baseline; these are the same 1024 inputs, not additional data",
        labels_not_supervision="Labels of tasks not trained on a role are descriptive audits only; they are never supplied to a model or used to change the task selection",
        population_relation="U(p)=L(reverse(p)) and reversal is a bijection, hence U and L have identical answer distributions under uniform permutations on S_n. A finite realized pool need not have identical histograms. The orbit-excluded sampling design and realized history can also alter the unrestricted population distribution.",
        unmatched_comparison="D and U do not have matched answer distributions; these tables quantify the difference. Distribution matching does not establish equal learning difficulty, redundancy or transfer value.",
        tests_opened=False, models_loaded=False, model_forwards=0,
        primary_data_manifest_sha256=data.sha(data_out / "manifest.json"),
        source_bindings={p.relative_to(ROOT).as_posix(): data.sha(p) for p in
                         [Path(__file__), ORACLE_PATH, ROOT / "permworld_platform/tasks.py",
                          ROOT / "experiments/permworld_pilot/vendor/neurips_permutations/math_ops.py"]},
        output_sha256={name: data.sha(out / name) for name in filenames[:3]},
        seconds=time.perf_counter()-started)
    data.write_json(out / filenames[3], report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "artifacts")
    run(parser.parse_args().out)
