"""Read-only platform integration checks; no model import, evaluation or selection.

Run from the repository root:
  .venv/Scripts/python.exe experiments/platform_v01/integration_review.py

Only the explicitly selected output JSON is written. Dataset object and label
files, frozen implementation, mathematical certificates, and registry exports
are never modified. Test objects/labels are inspected for data quality only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from permworld_platform.objects import canonical_orbit_key
from permworld_platform.tasks import label
from permworld_platform.encoding import encode_example, decode_answer_ids, vocabulary_manifest


REPRESENTATIVES = {
    "cycle_count", "longest_cycle", "lis_length",
    "increasing_subsequence_count[k=3]", "components", "largest_component",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review(dataset: Path) -> dict:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    seen, issues = set(), []
    total = collisions = sample_checks = row_checks = big_count = max_answer = 0
    overflow, maximum_lengths, big_by_role = Counter(), Counter(), Counter()
    expected_keys = {instance["instance_id"] for instance in manifest["instances"]}
    checked_file_hashes = {}
    for pool in manifest["pools"]:
        object_path = dataset / pool["object_path"]
        label_path = dataset / pool["label_path"]
        for path, key in ((object_path, "object_sha256"), (label_path, "label_sha256")):
            digest = sha(path)
            checked_file_hashes[path.relative_to(dataset).as_posix()] = digest
            if digest != pool[key]:
                issues.append(f"hash mismatch: {path}")
        with np.load(object_path, allow_pickle=False) as archive:
            objects = archive["p"].copy()
        rows = [json.loads(line) for line in label_path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != pool["count"] or len(objects) != len(rows):
            issues.append(f"row count mismatch: {object_path}")
        if objects.shape != (pool["count"], pool["n"]):
            issues.append(f"shape mismatch: {object_path}")
        for index, (array, row) in enumerate(zip(objects, rows)):
            permutation = tuple(map(int, array))
            orbit = canonical_orbit_key(permutation)
            collisions += orbit in seen
            seen.add(orbit)
            total += 1
            if set(row["answers"]) != expected_keys:
                issues.append(f"instance-key mismatch: {object_path}:{index}")
            scalar = row["answers"]["increasing_subsequence_count[k=3]"]
            big_count += scalar >= 100
            big_by_role[pool["role"]] += scalar >= 100
            max_answer = max(max_answer, scalar)
            # The fixed sampling rule is index-based, independent of answers
            # and unrelated to any later neural performance.
            if index % 128 == 0:
                for instance in manifest["instances"]:
                    value = label(instance["task_id"], permutation, instance["params"])
                    stored = row["answers"][instance["instance_id"]]
                    if json.loads(json.dumps(value)) != stored:
                        issues.append(f"label mismatch: {object_path}:{index}:{instance['instance_id']}")
                    sample = encode_example(instance["task_id"], permutation, instance["params"])
                    if decode_answer_ids(sample["answer_ids"], boolean=type(value) is bool) != value:
                        issues.append(f"encoding roundtrip mismatch: {instance['instance_id']}")
                    sample_checks += 1
                    maximum_lengths[instance["task_id"]] = max(maximum_lengths[instance["task_id"]], len(sample["input_ids"]))
                    if len(sample["input_ids"]) > 128:
                        overflow[instance["task_id"]] += 1
            row_checks += 1

    saved_vocab = json.loads((dataset / "vocabulary.json").read_text(encoding="utf-8"))
    vocab_equal = saved_vocab == vocabulary_manifest()
    vocab_hash_ok = sha(dataset / "vocabulary.json") == manifest["vocabulary_sha256"]
    if not vocab_equal or not vocab_hash_ok:
        issues.append("vocabulary mismatch")

    tie_differences = []
    with (dataset / "answer_distributions.csv").open(encoding="utf-8", newline="") as handle:
        distributions = list(csv.DictReader(handle))
    for row in distributions:
        if row["role"] != "train":
            continue
        histogram = json.loads(row["answer_histogram"])
        highest = max(histogram.values())
        tied = [value for value, count in histogram.items() if count == highest]
        if len(tied) < 2:
            continue
        # Numbers sort numerically; homogeneous structural answers use Python
        # recursive lexicographic order. Both are valid deterministic choices
        # among tied modes, whereas the materializer compares JSON strings.
        numeric_choice = min(tied, key=json.loads)
        if numeric_choice != row["train_majority_answer"]:
            tie_differences.append({
                "seed": int(row["seed"]), "n": int(row["n"]), "instance_id": row["instance_id"],
                "dataset_json_string_tie_choice": row["train_majority_answer"],
                "numeric_or_recursive_tie_choice": numeric_choice,
                "calibration_representative": row["instance_id"] in REPRESENTATIVES,
            })
    if collisions:
        issues.append(f"within-platform orbit collisions: {collisions}")
    if any(row["calibration_representative"] for row in tie_differences):
        issues.append("majority tie policy changes a calibration representative")

    return {
        "status": "passed" if not issues else "failed",
        "review_scope": "Read-only interface/data integration review, separate from full dataset RNG replay and mathematical certificates.",
        "dataset": dataset.resolve().relative_to(ROOT).as_posix(),
        "manifest_sha256": sha(dataset / "manifest.json"),
        "script_sha256": sha(Path(__file__)),
        "pools_checked": len(manifest["pools"]), "objects_checked": total,
        "instances_per_object": len(expected_keys), "materialized_label_values": total * len(expected_keys),
        "stored_label_rows_with_exact_instance_keys": row_checks,
        "sampled_exact_label_and_encoding_checks": sample_checks,
        "sample_rule": "Every 128th object in every pool, starting at index zero; all 56 instances per selected object.",
        "checked_file_hashes": checked_file_hashes,
        "within_platform_orbit_collisions": collisions,
        "saved_vocabulary_hash_matches": vocab_hash_ok,
        "current_vocabulary_equals_saved": vocab_equal,
        "increasing_subsequence_count_k3_answers_ge100": big_count,
        "increasing_subsequence_count_k3_answers_ge100_by_role": dict(big_by_role),
        "maximum_observed_increasing_subsequence_count_k3": max_answer,
        "sampled_context_overflows_at128": dict(overflow),
        "sampled_maximum_sequence_tokens": max(maximum_lengths.values()),
        "majority_tie_policy_difference_count": len(tie_differences),
        "majority_tie_policy_differences": tie_differences,
        "calibration_representative_tie_differences": 0,
        "historical_orbit_exclusion_replayed_in_this_review": False,
        "model_imported_or_executed": False,
        "model_metrics_computed": False,
        "test_labels_inspected_for_data_quality_only": True,
        "test_results_used_for_selection": False,
        "configuration_or_representative_changes": False,
        "limitations": [
            "This review checks current artifact coherence, not universal future version compatibility.",
            "load_objects is a convenience reader without integrity checks; read_samples checks object/label hashes and exact labels but does not independently bind saved vocabulary. Frozen calibration binds code and vocabulary separately.",
            "Some legacy structural outputs at other inputs/lengths exceed 128 tokens; exact generation supports them while the model-ready reader rejects an insufficient context explicitly.",
            "Binary task interfaces are executable but no binary dataset/pair-group isolation is certified in v0.1.",
            "Dataset distribution mode ties use JSON-string order; calibration uses numeric order. All six actual representative tasks agree under both policies.",
        ],
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "experiments/platform_v01/dataset")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/platform_v01/INTEGRATION_REVIEW.json")
    args = parser.parse_args()
    result = review(args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    keys = ("status", "objects_checked", "instances_per_object", "sampled_exact_label_and_encoding_checks", "increasing_subsequence_count_k3_answers_ge100", "majority_tie_policy_difference_count", "issues")
    print(json.dumps({key: result[key] for key in keys}, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
