"""Independent finite correctness and equivalence certificates for platform v0.1.

Run ``python -m permworld_platform.certificates --out <fresh-directory>``.
Old v5 identities are hash-referenced, never re-enumerated by this command.
Fixtures are mathematical checks only: no model, training or held-out selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from itertools import permutations
import json
from pathlib import Path
import random
import time

from . import relations
from .objects import transform
from .tasks import label

FOCUS = {
    "cycle_decomposition": ["cycle_count", "nontrivial_cycle_count", "even_cycle_count", "odd_cycle_count", "cycle_length_count", "longest_cycle"],
    "global_subsequence": ["lis_length", "lds_length", "increasing_subsequence_count", "decreasing_subsequence_count", "rsk_shape"],
    "permutation_blocks": ["components", "global_descents", "largest_component", "smallest_component", "component_length_count"],
}
PARAMETERIZED = {"cycle_length_count", "increasing_subsequence_count", "decreasing_subsequence_count", "component_length_count"}
ROOT = Path(__file__).resolve().parents[1]


def independent_cycle_lengths(p: tuple[int, ...]) -> tuple[int, ...]:
    """Undirected graph connectivity by union-find, not cycle traversal."""
    parent = list(range(len(p)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, value in enumerate(p):
        left, right = find(i), find(value - 1)
        if left != right:
            parent[left] = right
    return tuple(sorted(Counter(find(i) for i in range(len(p))).values(), reverse=True))


def independent_block_lengths(p: tuple[int, ...]) -> tuple[int, ...]:
    """Enumerate set-equality cuts instead of the production prefix maximum."""
    cuts = [0] + [j for j in range(1, len(p) + 1) if set(p[:j]) == set(range(1, j + 1))]
    return tuple(right - left for left, right in zip(cuts, cuts[1:]))


def independent_subsequences(p: tuple[int, ...]) -> tuple[dict[int, int], dict[int, int]]:
    """Enumerate every nonempty index subset, independently of count DP/LIS tails."""
    inc, dec = Counter(), Counter()
    for mask in range(1, 1 << len(p)):
        values = [value for i, value in enumerate(p) if mask & (1 << i)]
        if all(a < b for a, b in zip(values, values[1:])):
            inc[len(values)] += 1
        if all(a > b for a, b in zip(values, values[1:])):
            dec[len(values)] += 1
    return inc, dec


def independent_rsk_shape(p: tuple[int, ...]) -> tuple[int, ...]:
    """Permutation-matrix growth diagram, independent of sequential row insertion.

    If NW, NE and SW shapes are a,b,c, the local rule starts with the
    permutation-matrix cell as carry and emits max(b_i,c_i)+carry;
    the next carry is min(b_i,c_i)-a_i. No production bumping code is used.
    """
    n = len(p)
    previous = [()] * (n + 1)
    for row in range(1, n + 1):
        current = [()]
        for column in range(1, n + 1):
            nw, ne, sw = previous[column - 1], previous[column], current[column - 1]
            carry = int(p[row - 1] == column)
            result = []
            for index in range(max(len(nw), len(ne), len(sw)) + 1):
                a = nw[index] if index < len(nw) else 0
                b = ne[index] if index < len(ne) else 0
                c = sw[index] if index < len(sw) else 0
                result.append(max(b, c) + carry)
                carry = min(b, c) - a
            while result and result[-1] == 0:
                result.pop()
            current.append(tuple(result))
        previous = current
    return previous[-1]


def independent_answers(p: tuple[int, ...]) -> dict:
    cycles = independent_cycle_lengths(p)
    blocks = independent_block_lengths(p)
    inc, dec = independent_subsequences(p)
    n = len(p)
    return {
        "cycle_count": len(cycles), "nontrivial_cycle_count": sum(x > 1 for x in cycles),
        "even_cycle_count": sum(x % 2 == 0 for x in cycles), "odd_cycle_count": sum(x % 2 == 1 for x in cycles),
        "longest_cycle": max(cycles), "shortest_cycle": min(cycles),
        "lis_length": max(inc), "lds_length": max(dec), "rsk_shape": independent_rsk_shape(p),
        "components": len(blocks), "global_descents": sum(min(p[:j]) > max(p[j:]) for j in range(1, n)),
        "largest_component": max(blocks), "smallest_component": min(blocks),
        "cycle_length_count": Counter(cycles), "component_length_count": Counter(blocks),
        "increasing_subsequence_count": inc, "decreasing_subsequence_count": dec,
    }


def instance_rows() -> list[dict]:
    return [{"family": family, "task_id": task, "params": {"k": k} if k else {},
             "instance_id": f"{task}[k={k}]" if k else task}
            for family, tasks in FOCUS.items() for task in tasks
            for k in ([1, 2, 3] if task in PARAMETERIZED else [None])]


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write empty certificate table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _finite_equivalence(instances: list[dict], fixtures: list[tuple[int, ...]], values: dict) -> dict:
    groups = defaultdict(list)
    for instance in instances:
        iid = instance["instance_id"]
        groups[tuple(values[iid])].append(iid)
    same_input = [group for group in groups.values() if len(group) > 1]
    constant = []
    for instance in instances:
        iid = instance["instance_id"]
        by_n = defaultdict(set)
        for p, value in zip(fixtures, values[iid]):
            by_n[len(p)].add(value)
        if all(len(outputs) == 1 for outputs in by_n.values()):
            constant.append(iid)
    expected_duplicates = [["increasing_subsequence_count[k=1]", "decreasing_subsequence_count[k=1]"]]
    if sorted(map(sorted, same_input)) != sorted(map(sorted, expected_duplicates)):
        raise AssertionError(f"Unexpected finite same-input equivalences: {same_input}")
    proofs = [{"members": expected_duplicates[0], "evidence_class": "proved_from_definitions",
               "reason": "Both equal n; constant conditional on length", "relation_ids": [
                   "increasing_subsequence_count_k1_constant", "decreasing_subsequence_count_k1_constant"]}]
    transformed = [
        {"members": ["lis_length", "lds_length"], "transform": "complement", "answer_map": "identity", "relation_id": "lis_length_complement"},
        {"members": ["increasing_subsequence_count[k=2]", "decreasing_subsequence_count[k=2]"], "transform": "complement", "answer_map": "identity", "relation_id": "increasing_subsequence_count_complement"},
        {"members": ["increasing_subsequence_count[k=3]", "decreasing_subsequence_count[k=3]"], "transform": "complement", "answer_map": "identity", "relation_id": "increasing_subsequence_count_complement"},
    ]
    # One concrete different answer is sufficient to disprove an exact same-input
    # duplicate, even though finite equality by itself never proves equivalence.
    witnesses = []
    for left_index, left in enumerate(instances):
        for right in instances[left_index + 1:]:
            a, b = left["instance_id"], right["instance_id"]
            if values[a] == values[b]:
                continue
            index = next(i for i, (x, z) in enumerate(zip(values[a], values[b])) if x != z)
            witnesses.append({"left": a, "right": b, "input": list(fixtures[index]),
                              "left_answer": values[a][index], "right_answer": values[b][index]})
    by_family = []
    for family, tasks in FOCUS.items():
        subset = [row for row in instances if row["family"] == family]
        by_family.append({"family": family, "template_count": len(tasks), "declared_parameter_instances": len(subset),
                          "conditional_constant_instances": sum(row["instance_id"] in constant for row in subset)})
    return {
        "status": "passed", "scope": "The 16 named focus templates and their 24 declared instances, n=1..7 exhaustive plus n=8,10,12 fixtures",
        "counting_contract": "A template is a parameterized mathematical function; each declared parameter assignment is an instance. Encoding choices never increase mathematical counts.",
        "by_family": by_family, "focus_templates": 16, "focus_parameter_instances": 24,
        "same_input_distinct_functions": 23, "nonconstant_same_input_classes_conditional_on_n": 22,
        "value_preserving_declared_transform_classes_including_constant": 20,
        "nonconstant_value_preserving_declared_transform_classes": 19,
        "conditional_constant_instances": constant, "proved_same_input_equivalences": proofs,
        "proved_value_preserving_transformed_equivalences": transformed,
        "additional_invertible_answer_map_equivalence": {
            "members": ["components", "global_descents"], "transform": "complement",
            "answer_map": "components = global_descents + 1", "relation_id": "components_complement_global_descents",
            "note": "Not collapsed in the value-preserving count. Quotienting also by this declared affine map gives 19 classes, 18 nonconstant. Not a complete classification under arbitrary answer maps."},
        "legacy_aliases": [
            {"instance": "cycle_length_count[k=1]", "same_input_legacy": "fixed_points"},
            {"instance": "cycle_length_count[k=2]", "same_input_legacy": "two_cycle_count"},
            {"instance": "cycle_length_count[k=3]", "same_input_legacy": "three_cycle_count"},
            {"instance": "decreasing_subsequence_count[k=2]", "same_input_legacy": "length"},
            {"instance": "increasing_subsequence_count[k=2]", "legacy": "length", "answer_map": "n(n-1)/2 - length"}],
        "nonduplicate_same_input_witnesses": witnesses,
        "limitations": ["Counts describe this declared inventory and equivalence policy, not a number of independent information sources.",
                        "Known joint formulas and one-way maps do not collapse task templates.",
                        "Different task families are not asserted unrelated.",
                        "Constant within length does not mean constant when varying n; input length alone predicts these labels."]}


def run(out: Path) -> dict:
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    artifact_names = ["CERTIFICATE.json", "RELATIONS.json", "NEW_RELATION_CHECKS.csv", "LABEL_CHECKS.csv", "EQUIVALENCE_AUDIT.json", "HISTORICAL_REUSE.json", "DEFINITION_DISTRIBUTIONS.csv"]
    if any((out / name).exists() for name in artifact_names):
        raise FileExistsError("Certificate output already exists; choose a fresh --out directory")
    historical = relations.historical_references()
    fixtures = [p for n in range(1, 8) for p in permutations(range(1, n + 1))]
    exhaustive_count = len(fixtures)
    rng = random.Random(2026101907)
    for n in [8, 10, 12]:
        for _ in range(64):
            fixtures.append(tuple(rng.sample(range(1, n + 1), n)))
    fixture_hash = hashlib.sha256(json.dumps(fixtures, separators=(",", ":")).encode()).hexdigest()
    instances = instance_rows()
    values = {row["instance_id"]: [] for row in instances}
    checks = Counter()
    distributions = defaultdict(Counter)
    ranges = {}
    for p in fixtures:
        expected = independent_answers(p)
        for task, answer in expected.items():
            ks = range(1, len(p) + 2) if task in PARAMETERIZED else [None]
            for k in ks:
                params = {"k": k} if k else None
                reference = answer[k] if k else answer
                actual = label(task, p, params)
                if actual != reference:
                    raise AssertionError(f"Independent label mismatch: {task} k={k} p={p}: {actual}!={reference}")
                checks[task] += 1
        for row in instances:
            actual = label(row["task_id"], p, row["params"] or None)
            values[row["instance_id"]].append(actual)
            distributions[(row["instance_id"], len(p))][json.dumps(actual)] += 1
            ranges.setdefault(row["instance_id"], set()).add(actual)
    relation_checks = []
    new_edges = relations.new_relations()
    for edge in new_edges:
        count = 0
        for p in fixtures:
            ks = sorted({1, 2, 3, len(p), len(p) + 1}) if edge["parameter_schema"] else [None]
            for k in ks:
                result = relations.evaluate_relation(edge, p, {"k": k} if k else None)
                if not result["passed"]:
                    raise AssertionError(f"Relation failure {edge['relation_id']} {p} k={k}: {result}")
                count += 1
        relation_checks.append({"relation_id": edge["relation_id"], "evidence_class": edge["evidence_class"],
                               "pointwise_checks": count, "status": "passed", "proof_scope": edge["scope"],
                               "finite_scope": "S1..S7 exhaustive; 64 seeded samples each at n8,n10,n12; parameterized relations k in {1,2,3,n,n+1}"})
    equivalences = _finite_equivalence(instances, fixtures, values)
    registry = relations.relation_registry()
    checked_by_id = {row["relation_id"]: row for row in relation_checks}
    for edge in registry["relations"]:
        if edge["origin"] == "platform_v0.1_new_relation":
            edge["finite_check"].update(status="passed", pointwise_checks=checked_by_id[edge["relation_id"]]["pointwise_checks"],
                                        fixture_sha256=fixture_hash)
    label_rows = [{"task_id": task, "independent_label_comparisons": count, "status": "passed",
                   "reference_algorithm": ("union-find on undirected permutation graph" if "cycle" in task else
                       "all index subsets" if task in {"lis_length", "lds_length", "increasing_subsequence_count", "decreasing_subsequence_count"} else
                       "permutation-matrix growth diagram" if task == "rsk_shape" else "all set-equality direct-sum cuts / all cross-cut comparisons"),
                   "scope": "S1..S7 exhaustive; n8,n10,n12 fixed random fixtures; all count k1..n+1"}
                  for task, count in sorted(checks.items())]
    dist_rows = []
    for (instance, n), histogram in sorted(distributions.items()):
        total = sum(histogram.values())
        dist_rows.append({"instance_id": instance, "n": n, "input_count": total,
                          "sampling": "exhaustive_uniform" if n <= 7 else "fixed_random_fixtures_not_training",
                          "distinct_answers": len(histogram), "constant_on_checked_inputs": len(histogram) == 1,
                          "majority_fraction": max(histogram.values()) / total,
                          "histogram_json": json.dumps(histogram, sort_keys=True)})
    _json(out / "HISTORICAL_REUSE.json", historical)
    _json(out / "RELATIONS.json", registry)
    _csv(out / "LABEL_CHECKS.csv", label_rows)
    _csv(out / "NEW_RELATION_CHECKS.csv", relation_checks)
    _csv(out / "DEFINITION_DISTRIBUTIONS.csv", dist_rows)
    _json(out / "EQUIVALENCE_AUDIT.json", equivalences)
    source_files = [Path(__file__), Path(relations.__file__), ROOT / "permworld_platform/tasks.py",
                    ROOT / "permworld_platform/objects.py", ROOT / "experiments/permworld_pilot/vendor/neurips_permutations/math_ops.py"]
    certificate = {
        "status": "passed", "schema_version": "permworld-certificates-v0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started,
        "exhaustive_n": list(range(1, 8)), "exhaustive_input_count": exhaustive_count,
        "random_n": [8, 10, 12], "random_count_per_n": 64, "random_seed": 2026101907,
        "total_fixtures": len(fixtures), "fixture_sha256": fixture_hash,
        "task_implementations_independently_compared": len(checks),
        "independent_label_comparisons": sum(checks.values()), "new_relation_count": len(new_edges),
        "new_relation_pointwise_checks": sum(row["pointwise_checks"] for row in relation_checks),
        "historical_relation_count_reused": len(registry["relations"]) - len(new_edges),
        "historical_formula_rechecks_run": 0,
        "finite_checks_are_not_universal_proofs": True,
        "fixtures_passed_to_models": False, "new_training_started": False,
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        "artifact_sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in artifact_names if name != "CERTIFICATE.json"},
    }
    _json(out / "CERTIFICATE.json", certificate)
    return certificate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Fresh destination; never overwrite old certificates")
    args = parser.parse_args()
    certificate = run(args.out)
    print(json.dumps(certificate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
