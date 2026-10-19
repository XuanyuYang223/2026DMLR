"""Machine-readable task registry, explicit instances, bounds and export CLI.

Run ``python -m permworld_platform.registry --out experiments/platform_v01/registry``.
Mathematical templates, parameter instances, aliases and representation-only
endpoints are separate fields and are never silently counted as new tasks.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

from .tasks import BINARY_TASKS, LEGACY_PATH, LEGACY_TASK_IDS, NEW_TASK_IDS, PARAMETER_K_TASKS, TASK_IDS, label, legacy, validate_parameters

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_VERSION = "permworld_registry_v0.1"
ENCODING_VERSION = "permworld_sequence_v1"
CATALOG_PATH = ROOT / "experiments/pilot_v4_backgrounds/TASK_CATALOG.csv"
FOCUS_FAMILIES = {
    "cycle_decomposition": ("cycle_count", "nontrivial_cycle_count", "even_cycle_count", "odd_cycle_count", "cycle_length_count", "longest_cycle"),
    "global_subsequence": ("lis_length", "lds_length", "increasing_subsequence_count", "decreasing_subsequence_count", "rsk_shape"),
    "permutation_blocks": ("components", "global_descents", "largest_component", "smallest_component", "component_length_count"),
}
ALIASES = {
    "two_cycle_count": {"task_id": "cycle_length_count", "params": {"k": 2}},
    "three_cycle_count": {"task_id": "cycle_length_count", "params": {"k": 3}},
}
ENCODING_ENDPOINTS = ("to_cycle", "to_lehmer", "to_inversion_vector", "to_reduced_word")

_NEW = {
    "cycle_length_count": ("Number of cycles of length exactly k in the disjoint cycle decomposition of p.", "Traverse each permutation orbit once and count orbit sizes equal to k.", "O(n)"),
    "increasing_subsequence_count": ("Number of index tuples i1<...<ik with p(i1)<...<p(ik); k>n returns 0.", "Dynamic programming d[l,i]=sum_{j<i,p(j)<p(i)} d[l-1,j], with d[1,i]=1.", "O(k*n^2) time; O(n) auxiliary memory; integer arithmetic exact"),
    "decreasing_subsequence_count": ("Number of index tuples i1<...<ik with p(i1)>...>p(ik); k>n returns 0.", "Dynamic programming d[l,i]=sum_{j<i,p(j)>p(i)} d[l-1,j], with d[1,i]=1.", "O(k*n^2) time; O(n) auxiliary memory; integer arithmetic exact"),
    "largest_component": ("Largest size among the unique finest direct-sum indecomposable components of p.", "Record every boundary i where max(p(1),...,p(i))=i; take maximum gap between consecutive boundaries, including 0 and n.", "O(n)"),
    "smallest_component": ("Smallest size among the unique finest direct-sum indecomposable components of p.", "Record every boundary i where max(p(1),...,p(i))=i; take minimum gap between consecutive boundaries, including 0 and n.", "O(n)"),
    "component_length_count": ("Number of size-k components in the unique finest direct-sum indecomposable decomposition of p.", "Record every boundary i where max(p(1),...,p(i))=i; count consecutive boundary gaps equal to k.", "O(n)"),
}


def _parameter_schema(task_id):
    if task_id in PARAMETER_K_TASKS:
        return {"k": {"type": "integer", "minimum": 1, "maximum": 30, "required": True, "declared_instances": [1, 2, 3], "boundary": "k>n returns zero"}}
    if task_id == "pattern_avoidance":
        return {"pattern": {"type": "permutation", "min_length": 1, "max_length": 30, "required": True, "declared_instances": [[1, 2, 3]], "boundary": "pattern length > n implies avoidance"}}
    if task_id == "power":
        return {"exponent": {"type": "integer", "minimum": 0, "required": True, "declared_instances": [2]}}
    if task_id == "right_multiply_simple":
        return {"index": {"type": "integer", "minimum": 1, "maximum": "n-1", "required": True, "declared_instances": [1]}}
    return {}


def _output_type(task_id):
    if task_id in {"cycle_type", "rsk_shape"}:
        return "integer_partition"
    if task_id == "to_cycle":
        return "cycle_decomposition"
    if task_id in {"to_lehmer", "to_inversion_vector"}:
        return "integer_vector"
    if task_id == "to_reduced_word":
        return "integer_sequence"
    if task_id in {"inverse", "compose", "power", "conjugate", "commutator", "right_multiply_simple"}:
        return "permutation"
    if task_id in {"pattern_avoidance", "bruhat_leq"}:
        return "boolean"
    return "integer"


def _complexity(task_id):
    if task_id in _NEW:
        return _NEW[task_id][2]
    if task_id in {"lis_length", "lds_length"}:
        return "O(n log n) time; O(n) memory"
    if task_id == "rsk_shape":
        return "O(n^2) upper bound; O(n) tableau storage"
    if task_id == "pattern_avoidance":
        return "O(binomial(n,m)*m log m), m=pattern length; early termination allowed"
    if task_id == "power":
        return "O(n log(exponent+1))"
    if task_id == "bruhat_leq":
        return "O(n^2)"
    if task_id == "to_reduced_word":
        return "O(n^3) worst-case for repeated leftmost-descent scans"
    if task_id in {"length", "to_lehmer", "to_inversion_vector"}:
        return "O(n) bounded-width bitset operations for n<=30"
    if task_id == "cycle_type":
        return "O(n log n) upper bound including sorting cycle lengths"
    return "O(n)"


def _algorithm(task_id, definition):
    if task_id in _NEW:
        return _NEW[task_id][1]
    methods = {
        "rsk_shape": "Exact Schensted row insertion; each incoming value bumps the first strictly larger entry; return row lengths.",
        "lis_length": "Patience-sorting tails with binary search (bisect_left).",
        "lds_length": "Patience-sorting LIS on negated values.",
        "components": "Count prefix boundaries max(p(1),...,p(i))=i, including i=n.",
        "global_descents": "Compare each running prefix minimum with the following suffix maximum.",
        "cycle_type": "Traverse permutation cycles and sort their lengths in descending order.",
        "bruhat_leq": "Rank-matrix criterion r_p(i,j)>=r_q(i,j) for all i,j.",
        "power": "Nonnegative integer exponentiation by repeated squaring of exact permutation composition.",
    }
    if task_id in methods:
        return methods[task_id]
    if "cycle" in task_id:
        return "Traverse each permutation orbit once; compute the declared statistic of cycle lengths."
    return "Unmodified vendored exact deterministic function; " + definition


def _build_specs():
    with CATALOG_PATH.open(encoding="utf-8-sig", newline="") as handle:
        historical = {row["task_id"]: row for row in csv.DictReader(handle) if row["world"] == "PermWorld"}
    specs = []
    for task_id in TASK_IDS:
        row = historical.get(task_id, {})
        definition = _NEW[task_id][0] if task_id in _NEW else row["definition"]
        family = next((family for family, ids in FOCUS_FAMILIES.items() if task_id in ids), None)
        if family is None:
            if task_id in {"two_cycle_count", "three_cycle_count", "shortest_cycle", "cycle_type"}:
                family = "cycle_decomposition"
            else:
                family = row.get("mathematical_family", "permutation_operations")
        output_type = _output_type(task_id)
        specs.append({
            "registry_version": REGISTRY_VERSION, "task_id": task_id, "task_version": "1.0.0",
            "mathematical_definition": definition, "parameters": _parameter_schema(task_id),
            "legal_input_lengths": {"minimum": 2 if task_id == "right_multiply_simple" else 1, "maximum": 30, "parameter_constraints": "index<n" if task_id == "right_multiply_simple" else None},
            "input_object": "permutation_one_based", "input_arity": 2 if task_id in BINARY_TASKS else 1,
            "operand_schema": {"q": {"type": "permutation", "length": "same as p", "required": True}} if task_id in BINARY_TASKS else {},
            "operand_convention": {"compose": "p o q", "conjugate": "p o q o p^-1 (p is conjugator)", "commutator": "p o q o p^-1 o q^-1", "bruhat_leq": "p <= q in strong Bruhat order"}.get(task_id),
            "family": family, "focus_template": any(task_id in ids for ids in FOCUS_FAMILIES.values()),
            "identity_kind": "parameter_alias" if task_id in ALIASES else "representation_endpoint" if task_id in ENCODING_ENDPOINTS else "parameterized_template" if _parameter_schema(task_id) else "task_template",
            "canonical_alias": ALIASES.get(task_id), "exact_answer_method": _algorithm(task_id, definition),
            "complexity": _complexity(task_id),
            "output_type": output_type, "output_range": {"symbolic": "0..binomial(n,k)" if "subsequence_count" in task_id else "0..floor(n/k)" if task_id in {"cycle_length_count", "component_length_count"} else "1..n" if task_id in {"largest_component", "smallest_component"} else row.get("output_range"), "support_kind": "outer bounds unless stated otherwise", "executable_bounds": "task_output_bounds(task_id,n,params)"},
            "output_encoding": ENCODING_VERSION,
            "input_encoding_variants": ["one_line_v1"],
            "upstream_encodings_not_exposed_in_platform_v01": ["canonical_cycle", "lehmer", "inversion_vector"],
            "encoding_variants_are_new_tasks": False,
            "legacy_scalar_protocol": task_id in legacy.PROPERTY32_TASK_NAMES,
            "legacy_single_numeric_token_compatible_all_legal_lengths": output_type in {"integer", "boolean"} and task_id not in {"length", "increasing_subsequence_count", "decreasing_subsequence_count"},
            "generation": {"object": "uniform permutation proposals; explicit generator split/rejection policy applies", "label": "exact deterministic function, independent of split and sampling policy"},
            "generation_time": {"status": "measured_at_registry_export", "artifact": "TASK_BENCHMARK.csv", "includes": "input validation plus exact label; excludes object proposal, encoding and IO"},
            "known_relations": {"status": "see versioned relation registry; unlisted does not mean unrelated", "historical_annotation": row.get("known_relations"), "historical_annotation_evidence": "catalog text alone is not a proof or verification certificate", "registry_api": "permworld_platform.relations.list_relations", "reference": "experiments/platform_v01/mathematics"},
            "verification": {"historical_semantics": "unmodified vendor function" if task_id in LEGACY_TASK_IDS else "new implementation", "implementation_certificate": "experiments/platform_v01/mathematics/CERTIFICATE.json" if any(task_id in ids for ids in FOCUS_FAMILIES.values()) else None, "interface_certificate": "experiments/platform_v01/registry/CORE_INTERFACE_AUDIT.json", "relation_certificates": "historical v5 certificates reused by hash; new relations separately certified", "scope_note": "New-family exhaustive/independent label checks cover the focus instances; every endpoint additionally has a hand-derived interface fixture. Unlisted tasks are not claimed independently reverified over the full domain."},
            "provenance": {"implementation": "permworld_platform/tasks.py", "legacy_source": str(LEGACY_PATH.relative_to(ROOT)).replace("\\", "/") if task_id in LEGACY_TASK_IDS or task_id == "cycle_length_count" else None, "historical_catalog": str(CATALOG_PATH.relative_to(ROOT)).replace("\\", "/") if row else None},
        })
    return specs


_SPECS = _build_specs()
_BY_ID = {spec["task_id"]: spec for spec in _SPECS}


def list_task_specs() -> list[dict]:
    return copy.deepcopy(_SPECS)


def task_spec(task_id: str) -> dict:
    try:
        return copy.deepcopy(_BY_ID[task_id])
    except KeyError as exc:
        raise KeyError(f"unknown task_id {task_id!r}") from exc


def instance_id(task_id: str, params=None) -> str:
    parameters = validate_parameters(task_id, params)
    if not parameters:
        return task_id
    pieces = []
    for key, value in sorted(parameters.items()):
        text = ",".join(str(x) for x in value) if isinstance(value, (list, tuple)) else str(value)
        pieces.append(f"{key}={text}")
    return task_id + "[" + ";".join(pieces) + "]"


def list_task_instances(*, include_aliases: bool = False, include_binary: bool = False, focus_only: bool = False) -> list[dict]:
    """Finite declared manifest, separate from all runtime-legal parameter values.

    Default returns all unary endpoints except the two legacy parameter aliases.
    Four representation endpoints remain usable but carry their identity_kind.
    """
    instances = []
    for spec in _SPECS:
        task_id = spec["task_id"]
        if (not include_aliases and task_id in ALIASES) or (not include_binary and task_id in BINARY_TASKS) or (focus_only and not spec["focus_template"]):
            continue
        if task_id in PARAMETER_K_TASKS:
            parameter_sets = [{"k": k} for k in (1, 2, 3)]
        else:
            parameter_sets = [{"pattern": [1, 2, 3]}] if task_id == "pattern_avoidance" else [{"exponent": 2}] if task_id == "power" else [{"index": 1}] if task_id == "right_multiply_simple" else [{}]
        for params in parameter_sets:
            instances.append({"instance_id": instance_id(task_id, params), "task_id": task_id, "params": params, "input_arity": spec["input_arity"], "family": spec["family"], "focus_template": spec["focus_template"], "identity_kind": spec["identity_kind"], "output_type": spec["output_type"], "output_encoding": ENCODING_VERSION})
    return instances


def task_output_bounds(task_id: str, n: int, params=None) -> dict:
    """Executable safe bounds, not a claim that every value in the interval occurs."""
    spec = task_spec(task_id)
    parameters = validate_parameters(task_id, params)
    if type(n) is not int or not spec["legal_input_lengths"]["minimum"] <= n <= 30:
        raise ValueError("unsupported input length for task")
    if task_id == "right_multiply_simple" and parameters["index"] >= n:
        raise ValueError("index must be smaller than n")
    output_type = spec["output_type"]
    result = {"type": output_type, "support_kind": "safe_outer_bounds"}
    if output_type == "integer":
        minimum, maximum = 0, n
        if task_id in {"cycle_count", "longest_cycle", "shortest_cycle", "lis_length", "lds_length", "longest_increasing_run", "longest_decreasing_run", "components", "largest_component", "smallest_component", "left_to_right_maxima", "left_to_right_minima", "right_to_left_maxima", "right_to_left_minima"}:
            minimum = 1
        if task_id in {"descents", "recoils", "successions", "adjacencies", "exceedances", "deficiencies", "global_descents", "max_displacement"}:
            maximum = n - 1
        if task_id in {"peaks", "valleys"}:
            maximum = (n - 1) // 2
        if task_id in {"double_ascents", "double_descents"}:
            maximum = max(n - 2, 0)
        if task_id in {"two_cycle_count", "even_cycle_count", "nontrivial_cycle_count"}:
            maximum = n // 2
        if task_id == "three_cycle_count":
            maximum = n // 3
        if task_id in {"cycle_length_count", "component_length_count"}:
            maximum = n // parameters["k"]
        if task_id in {"increasing_subsequence_count", "decreasing_subsequence_count"}:
            maximum = math.comb(n, parameters["k"]) if parameters["k"] <= n else 0
            minimum = n if parameters["k"] == 1 else 0
        if task_id == "length":
            maximum = n * (n - 1) // 2
        if task_id == "parity":
            maximum = 1
        result.update(minimum=minimum, maximum=maximum)
    elif output_type == "boolean":
        result.update(minimum=0, maximum=1, values=[False, True])
    elif output_type == "integer_partition":
        result.update(element_minimum=1, element_maximum=n, minimum_length=1, maximum_length=n, sum=n, order="nonincreasing")
    elif output_type == "permutation":
        result.update(element_minimum=1, element_maximum=n, length=n, unique_elements=True)
    elif output_type == "cycle_decomposition":
        result.update(element_minimum=1, element_maximum=n, total_elements=n, canonicalization="minimum first per cycle; cycles ordered by their minima; includes singletons")
    elif output_type == "integer_vector":
        result.update(element_minimum=0, element_maximum=max(0, n - 1), length=n, per_position_maximum=[n - i for i in range(1, n + 1)])
    else:
        result.update(element_minimum=1, element_maximum=max(0, n - 1), minimum_length=0, maximum_length=n * (n - 1) // 2)
    return result


def count_summary() -> dict:
    """Engineering counts; the certificates additionally report empirical collisions."""
    focus = list_task_instances(focus_only=True)
    families = {}
    for family, ids in FOCUS_FAMILIES.items():
        instances = [row for row in focus if row["family"] == family]
        families[family] = {"canonical_templates": len(ids), "declared_instances": len(instances), "parameterized_templates": sum(task_id in PARAMETER_K_TASKS for task_id in ids), "implemented_input_encoding_variants": 1}
    return {
        "legacy_endpoints_preserved": len(LEGACY_TASK_IDS), "new_registry_endpoints": len(NEW_TASK_IDS),
        "total_registry_endpoints": len(TASK_IDS), "parameter_alias_endpoints": len(ALIASES),
        "representation_only_endpoints": len(ENCODING_ENDPOINTS),
        "canonical_mathematical_templates": len(TASK_IDS) - len(ALIASES) - len(ENCODING_ENDPOINTS),
        "all_declared_endpoint_instances_including_aliases_and_binary": len(list_task_instances(include_aliases=True, include_binary=True)),
        "canonical_declared_instances_including_binary": len(list_task_instances(include_binary=True)),
        "default_unary_manifest_instances": len(list_task_instances()),
        "focus_templates": sum(len(ids) for ids in FOCUS_FAMILIES.values()), "focus_declared_instances": len(focus),
        "focus_nonconstant_instances": len(focus) - 2,
        "focus_same_input_distinct_functions": 23,
        "focus_value_preserving_declared_transform_classes": 20,
        "focus_nonconstant_value_preserving_declared_transform_classes": 19,
        "effective_count_scope": "Focus instances only; collapse the two k=1 constants, and declared complement/reversal identity-answer pairs LIS/LDS and increasing/decreasing count k=2,3. This is not a classification under arbitrary answer maps. See mathematics/EQUIVALENCE_AUDIT.json for formulas and distinctness witnesses.",
        "input_encoding_variants": ["one_line_v1"],
        "upstream_encodings_not_exposed_in_platform_v01": ["canonical_cycle", "lehmer", "inversion_vector"],
        "output_protocol_versions": [ENCODING_VERSION], "families": families,
        "exact_equivalences": [
            {"left": "two_cycle_count", "right": "cycle_length_count[k=2]", "class": "parameter_alias"},
            {"left": "three_cycle_count", "right": "cycle_length_count[k=3]", "class": "parameter_alias"},
            {"left": "fixed_points", "right": "cycle_length_count[k=1]", "class": "same_input_identical_label"},
            {"left": "length", "right": "decreasing_subsequence_count[k=2]", "class": "same_input_identical_label"},
            {"left": "increasing_subsequence_count[k=1]", "right": "decreasing_subsequence_count[k=1]", "class": "both equal n; constant conditional on length"},
        ],
        "interpretation": "48 is a canonical template count, not a claim of 48 mathematically independent tasks. Input-transform and answer-map equivalences are kept in the relation registry, not silently erased; global constant tasks remain visible."
    }


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def model_interface_compatibility(task_id: str, n: int, params=None, *, max_seq_len: int = 128) -> dict:
    """A conservative full-sequence bound for a fixed parameter instance.

    A context failure is explicit and does not remove the task from the exact
    data platform. The encoder accepts an optional larger context budget.
    """
    from .encoding import encode_value
    spec = task_spec(task_id)
    parameters = validate_parameters(task_id, params)
    bounds = task_output_bounds(task_id, n, parameters)
    prompt_max = 2 * n + 6
    if spec["input_arity"] == 2:
        prompt_max += 2 * n + 2
    if parameters:
        prompt_max += 2 + sum(1 + len(encode_value(value)) for value in parameters.values())
    output_type = spec["output_type"]
    if output_type in {"integer", "boolean"}:
        answer_max = len(encode_value(bounds["maximum"])) + 1
    elif output_type == "cycle_decomposition":
        answer_max = 4 * n + 2
    elif output_type == "integer_sequence":
        length = bounds["maximum_length"]
        answer_max = 2 * length + 2 if length else 3
    else:
        answer_max = 2 * n + 2
    return {
        "task_id": task_id, "n": n, "params": parameters,
        "encoding_version": ENCODING_VERSION, "max_seq_len": max_seq_len,
        "full_sequence_upper_bound_tokens": prompt_max + answer_max,
        "all_inputs_fit_at_this_length_and_parameters": prompt_max + answer_max <= max_seq_len,
        "historical_single_numeric_answer_token_supported": output_type in {"integer", "boolean"} and bounds["maximum"] < 100,
        "full_answer_evaluation_required": True,
        "excessive_context_policy": "raise ValueError; never truncate",
        "note": "Safe bound may exceed actual sequence lengths; arbitrary power exponents require a separate parameter-specific check."
    }


def export_registry(out: str | Path, *, benchmark_samples: int = 64) -> dict:
    """Export schemas and deterministic benchmark inputs; timings are machine-specific."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    specs = list_task_specs()
    benchmarks = []
    compatibility = []
    rng = random.Random(2026101907)
    for n in (10, 30):
        inputs = []
        for _ in range(benchmark_samples):
            p = list(range(1, n + 1)); rng.shuffle(p)
            q = list(range(1, n + 1)); rng.shuffle(q)
            inputs.append((tuple(p), tuple(q)))
        for instance in list_task_instances(include_aliases=True, include_binary=True):
            compatibility.append(dict(instance_id=instance["instance_id"], **model_interface_compatibility(instance["task_id"], n, instance["params"])))
            kwargs = {"params": instance["params"]}
            # Include validation, exact function dispatch, and label calculation.
            start = time.perf_counter_ns()
            for p, q in inputs:
                label(instance["task_id"], p, operands={"q": q} if instance["input_arity"] == 2 else None, **kwargs)
            elapsed = time.perf_counter_ns() - start
            benchmarks.append({"instance_id": instance["instance_id"], "task_id": instance["task_id"], "n": n, "samples": benchmark_samples, "mean_microseconds_per_label": elapsed / benchmark_samples / 1000, "includes_validation": True, "seed": 2026101907})
    for spec in specs:
        spec["generation_time"]["measurements"] = [row for row in benchmarks if row["task_id"] == spec["task_id"]]
        spec["model_interface_compatibility"] = [row for row in compatibility if row["task_id"] == spec["task_id"]]
    # Attach exact relation IDs and evidence classes without promoting catalog
    # annotations to proofs or running any unchanged historical verification.
    from .relations import list_relations
    relations = list_relations()
    certificate_path = ROOT / "experiments/platform_v01/mathematics/CERTIFICATE.json"
    if certificate_path.exists():
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        if certificate.get("status") != "passed":
            raise ValueError("existing mathematical certificate has not passed")
        for relative, expected in certificate["source_sha256"].items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"mathematical certificate source changed: {relative}")
        relation_path = certificate_path.parent / "RELATIONS.json"
        if hashlib.sha256(relation_path.read_bytes()).hexdigest() != certificate["artifact_sha256"]["RELATIONS.json"]:
            raise ValueError("certified relation artifact changed")
        relations = json.loads(relation_path.read_text(encoding="utf-8"))["relations"]
    for spec in specs:
        refs = []
        for relation in relations:
            mentioned = set(relation.get("required_tasks", []))
            for query in [relation.get("target", {}), *relation.get("queries", []), relation.get("query_family", {})]:
                if "task_id" in query:
                    mentioned.add(query["task_id"])
            if spec["task_id"] in mentioned:
                refs.append({"relation_id": relation["relation_id"], "evidence_class": relation.get("evidence_class"), "verification": relation.get("finite_check")})
        spec["known_relations"]["relation_references"] = refs
    manifest = {"registry_version": REGISTRY_VERSION, "source_sha256": hashlib.sha256(LEGACY_PATH.read_bytes()).hexdigest(), "historical_catalog_sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(), "counts": count_summary(), "tasks": specs, "instances": list_task_instances(include_aliases=True, include_binary=True)}
    (out / "TASK_REGISTRY.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "TASK_COUNTS.json").write_text(json.dumps(count_summary(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(out / "TASK_REGISTRY.csv", specs)
    _write_csv(out / "TASK_INSTANCES.csv", manifest["instances"])
    _write_csv(out / "TASK_BENCHMARK.csv", benchmarks)
    _write_csv(out / "MODEL_INTERFACE_COMPATIBILITY.csv", compatibility)
    return {"out": str(out), "task_endpoints": len(specs), "benchmark_rows": len(benchmarks), "counts": count_summary()}


def audit_interface(out: str | Path) -> dict:
    """Small hand-derived full-endpoint contract and rejection check.

    This complements the independent exhaustive new-family certificate. It
    does not rerun historical v5 relation verification or neural training.
    """
    from .encoding import decode_answer_ids, encode_example
    from .objects import canonical_orbit_key, transform, validate_permutation
    p, q = (3, 1, 4, 2), (2, 4, 1, 3)
    expected = {
        "to_cycle": ((1, 3, 4, 2),), "to_lehmer": (2, 0, 1, 0),
        "to_inversion_vector": (1, 2, 0, 0), "to_reduced_word": (2, 3, 1),
        "length": 3, "descents": 2, "fixed_points": 0, "parity": 1,
        "cycle_type": (4,), "rsk_shape": (2, 2), "lis_length": 2, "lds_length": 2,
        "pattern_avoidance": True, "inverse": q, "compose": (1, 2, 3, 4),
        "power": (4, 3, 2, 1), "conjugate": q, "commutator": (1, 2, 3, 4),
        "right_multiply_simple": (1, 3, 4, 2), "bruhat_leq": False,
        "peaks": 1, "exceedances": 2, "recoils": 1, "valleys": 1,
        "double_ascents": 0, "double_descents": 0, "successions": 0,
        "adjacencies": 0, "anti_fixed_points": 0, "deficiencies": 2,
        "left_to_right_maxima": 2, "left_to_right_minima": 2,
        "right_to_left_maxima": 2, "right_to_left_minima": 2,
        "cycle_count": 1, "two_cycle_count": 0, "three_cycle_count": 0,
        "even_cycle_count": 1, "odd_cycle_count": 0, "longest_cycle": 4,
        "shortest_cycle": 4, "nontrivial_cycle_count": 1,
        "longest_increasing_run": 2, "longest_decreasing_run": 2,
        "global_descents": 0, "components": 1, "max_displacement": 2,
        "displacement_one_count": 2, "cycle_length_count": 0,
        "increasing_subsequence_count": 3, "decreasing_subsequence_count": 3,
        "largest_component": 4, "smallest_component": 4, "component_length_count": 0,
    }
    if set(expected) != set(TASK_IDS):
        raise AssertionError("hand-derived fixture does not cover every endpoint")
    records = []
    for task_id in TASK_IDS:
        params = {"k": 2} if task_id in PARAMETER_K_TASKS else {"pattern": (1, 2, 3)} if task_id == "pattern_avoidance" else {"exponent": 2} if task_id == "power" else {"index": 1} if task_id == "right_multiply_simple" else {}
        operands = {"q": q} if task_id in BINARY_TASKS else None
        value = label(task_id, p, params, operands)
        assert value == expected[task_id], (task_id, value, expected[task_id])
        assert type(value) is type(expected[task_id]), task_id
        sample = encode_example(task_id, p, params, operands, max_seq_len=128)
        assert decode_answer_ids(sample["answer_ids"], boolean=type(value) is bool) == value
        records.append({"task_id": task_id, "params": params, "p": p, "q": q if operands else None, "expected": expected[task_id], "actual": value, "exact_type_pass": True, "encoding_roundtrip_pass": True})
    invalid = [
        ("empty permutation", lambda: label("descents", ())),
        ("too long", lambda: label("descents", tuple(range(1, 32)))),
        ("duplicate", lambda: label("descents", (1, 1))),
        ("noninteger", lambda: label("descents", (1.0, 2))),
        ("boolean entry", lambda: label("descents", (True, 2))),
        ("string input", lambda: label("descents", "12")),
        ("unknown task", lambda: label("missing", p)),
        ("missing k", lambda: label("cycle_length_count", p)),
        ("zero k", lambda: label("cycle_length_count", p, {"k": 0})),
        ("too large k", lambda: label("cycle_length_count", p, {"k": 31})),
        ("boolean k", lambda: label("cycle_length_count", p, {"k": True})),
        ("extra parameter", lambda: label("descents", p, {"k": 1})),
        ("malformed parameter mapping", lambda: label("descents", p, [])),
        ("malformed pattern", lambda: label("pattern_avoidance", p, {"pattern": [1, 1]})),
        ("negative power", lambda: label("power", p, {"exponent": -1})),
        ("boolean power", lambda: label("power", p, {"exponent": True})),
        ("invalid simple index", lambda: label("right_multiply_simple", p, {"index": 4})),
        ("boolean simple index", lambda: label("right_multiply_simple", p, {"index": True})),
        ("missing operand", lambda: label("compose", p)),
        ("extra operand", lambda: label("compose", p, operands={"q": q, "x": q})),
        ("wrong operand length", lambda: label("compose", p, operands={"q": (1, 2)})),
        ("unexpected operand", lambda: label("descents", p, operands={"q": q})),
        ("unknown transform", lambda: transform("not_a_transform", p)),
        ("too short context", lambda: encode_example("rsk_shape", tuple(range(30, 0, -1)), max_seq_len=127)),
    ]
    rejection_records = []
    for name, call in invalid:
        try:
            call()
        except (TypeError, ValueError, KeyError) as exc:
            rejection_records.append({"case": name, "rejected": True, "exception": type(exc).__name__})
        else:
            raise AssertionError(f"invalid input was accepted: {name}")
    boundary_checks = 0
    for instance in list_task_instances(include_aliases=True, include_binary=True):
        for n in (max(1, task_spec(instance["task_id"])["legal_input_lengths"]["minimum"]), 30):
            for values in (tuple(range(1, n + 1)), tuple(range(n, 0, -1))):
                operands = {"q": values[::-1]} if instance["input_arity"] == 2 else None
                value = label(instance["task_id"], values, instance["params"], operands)
                bounds = task_output_bounds(instance["task_id"], n, instance["params"])
                if bounds["type"] in {"integer", "boolean"}:
                    assert bounds["minimum"] <= value <= bounds["maximum"]
                sample = encode_example(instance["task_id"], values, instance["params"], operands)
                info = model_interface_compatibility(instance["task_id"], n, instance["params"])
                assert len(sample["input_ids"]) <= info["full_sequence_upper_bound_tokens"], (instance, info)
                assert decode_answer_ids(sample["answer_ids"], boolean=type(value) is bool) == value
                assert canonical_orbit_key(values) == canonical_orbit_key(transform("inverse", values))
                boundary_checks += 1
    huge = label("increasing_subsequence_count", tuple(range(1, 31)), {"k": 15})
    assert huge == 155117520
    large_sample = encode_example("increasing_subsequence_count", tuple(range(1, 31)), {"k": 15}, max_seq_len=128)
    assert decode_answer_ids(large_sample["answer_ids"]) == huge
    status = {"status": "passed", "hand_derived_endpoint_labels": len(records), "invalid_input_rejections": len(rejection_records), "boundary_encoding_and_upper_bound_checks": boundary_checks, "large_integer_exact_value": huge, "large_integer_answer_tokens_including_eos": len(large_sample["answer_ids"]), "historical_relation_rechecks": 0, "cases": records, "rejections": rejection_records}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "CORE_INTERFACE_AUDIT.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in status.items() if key not in {"cases", "rejections"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/platform_v01/registry")
    parser.add_argument("--benchmark-samples", type=int, default=64)
    parser.add_argument("--audit-interface", action="store_true", help="Run endpoint fixtures, parameter/type rejection and encoding boundary checks.")
    args = parser.parse_args()
    if args.benchmark_samples < 1:
        parser.error("--benchmark-samples must be positive")
    print(json.dumps(export_registry(args.out, benchmark_samples=args.benchmark_samples), ensure_ascii=False, indent=2))
    if args.audit_interface:
        print(json.dumps(audit_interface(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
