"""Versioned mathematical hyperedges with explicit input queries.

Expressions are JSON trees interpreted by a small allowlist, never Python eval.
Finite certificates validate implementations; proofs and empirical observations
remain separate evidence classes.  Absence of an edge means unclassified.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

RELATIONS_VERSION = "permworld-relations-v0.1"
ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "experiments/pilot_v5_confirmation/mathematics"
V5_INVENTORY_SHA256 = "394b4b034b639450044fdefb32aa63a977306b8b2ef1d60d2c3ddf1ec02cf67e"
V5_DETAIL_SHA256 = "abe146533bd3d6d74895ff54b1437a847e63c09e59d12be5dac11f04007b6d4d"


def q(task_id: str, transforms: Sequence[str] = (), **params: Any) -> dict:
    return {"task_id": task_id, "params": params, "input_id": "p",
            "transforms": list(transforms)}


def op(name: str, *args: Any) -> dict:
    return {"op": name, "args": list(args)}


def y(index: int) -> dict:
    return {"query": index}


N = {"context": "n"}
K = {"context": "k"}


def _edge(relation_id: str, target: dict, queries: list[dict], answer_map: Any,
          formula: str, proof: str, *, params: dict | None = None,
          status: str = "proved_from_definitions", query_family: dict | None = None,
          scope: str = "S_n, 1 <= n <= 30 in platform v0.1; mathematical statement for every n >= 1") -> dict:
    all_queries = queries + ([query_family] if query_family else []) + [target]
    transformed = any(item.get("transforms") for item in all_queries)
    multiple = len(queries) > 1 or query_family is not None
    result = {
        "relation_id": relation_id, "version": RELATIONS_VERSION,
        "evidence_class": status, "formula": formula, "scope": scope,
        "relation_type": "joint" if multiple else "single_task",
        "query_model": ("multiple_transformed_inputs" if multiple and transformed else
                        "same_input_multiple_answers" if multiple else
                        "transformed_input" if transformed else "same_input"),
        "uses_transformed_input_queries": transformed,
        "requires_distinct_input_values": False,
        "input_query_note": "Transformed query arguments can coincide for symmetric permutations; no assumption of distinct numerical inputs is required.",
        "input_ids": ["p"], "target": target, "queries": queries,
        "answer_map": answer_map, "parameter_schema": params or {},
        "proof": proof, "finite_check": {"status": "pending_certificate", "artifact":
            "experiments/platform_v01/mathematics/NEW_RELATION_CHECKS.csv"},
        "training_exposure": "Relation metadata does not imply joint presentation or model use.",
        "mechanism_claim": False,
    }
    if query_family is not None:
        result["query_family"] = query_family
    return result


def historical_references() -> dict:
    """Check byte identity only; do not rerun historical label/identity checks."""
    checks = {
        "audit/RELATION_INVENTORY.csv": V5_INVENTORY_SHA256,
        "audit/MATHEMATICS_AUDIT.json": V5_DETAIL_SHA256,
    }
    for relative, expected in checks.items():
        actual = hashlib.sha256((V5 / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Historical certificate changed: {relative}")
    detail = json.loads((V5 / "audit/MATHEMATICS_AUDIT.json").read_text(encoding="utf-8"))
    # The inventory certificate itself is fixed above; bind its underlying
    # implementation and the referenced empirical/unchecked tables as well.
    implementation_relative = "experiments/permworld_pilot/vendor/neurips_permutations/math_ops.py"
    implementation_expected = detail["protected_sha256"][implementation_relative.replace("/", "\\")]
    if hashlib.sha256((ROOT / implementation_relative).read_bytes()).hexdigest() != implementation_expected:
        raise RuntimeError("Legacy task definitions changed; historical proofs require review")
    for name in ["OBSERVED_ASSOCIATIONS.csv", "UNCHECKED_CLAIMS.csv"]:
        expected = detail["output_sha256"][name]
        if hashlib.sha256((V5 / "audit" / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Historical evidence changed: {name}")
        checks[f"audit/{name}"] = expected
    return {"status": "reused_hash_verified", "root": str(V5.relative_to(ROOT)).replace("\\", "/"),
            "sha256": checks, "historical_rechecks_run": 0,
            "implementation_sha256": {implementation_relative: implementation_expected},
            "previous_status": detail["status"], "previous_exhaustive_n": detail["exhaustive_n"],
            "previous_random_n": detail["random_n"], "previous_random_inputs": detail["random_inputs"],
            "previous_independent_label_comparisons": detail["definition_label_comparisons"]}


def _historical_edges() -> list[dict]:
    historical_references()
    # Executable mappings for the historical hyperedges directly used by this
    # platform. Other formulas/bounds retain their complete original certificate.
    compiled = {
        "position_partition": (q("fixed_points"), [q("exceedances"), q("deficiencies")], op("sub", N, y(0), y(1))),
        "inverse_exceedance_deficiency": (q("exceedances"), [q("deficiencies", ["inverse"])], y(0)),
        "fixed_from_two_deficiency_queries": (q("fixed_points"), [q("deficiencies"), q("deficiencies", ["inverse"])], op("sub", N, y(0), y(1))),
        "anti_fixed_from_two_deficiency_queries": (q("anti_fixed_points"), [q("deficiencies", ["complement"]), q("deficiencies", ["complement", "inverse"])], op("sub", N, y(0), y(1))),
        "fixed_complement": (q("fixed_points"), [q("anti_fixed_points", ["complement"])], y(0)),
        "fixed_reverse": (q("fixed_points"), [q("anti_fixed_points", ["reverse"])], y(0)),
        "fixed_inverse": (q("fixed_points"), [q("fixed_points", ["inverse"])], y(0)),
        "anti_fixed_inverse": (q("anti_fixed_points"), [q("anti_fixed_points", ["inverse"])], y(0)),
        "run_complement": (q("longest_decreasing_run"), [q("longest_increasing_run", ["complement"])], y(0)),
        "run_reverse": (q("longest_decreasing_run"), [q("longest_increasing_run", ["reverse"])], y(0)),
        "descent_recoil_inverse": (q("descents"), [q("recoils", ["inverse"])], y(0)),
        "descent_complement": (q("descents", ["complement"]), [q("descents")], op("sub", N, 1, y(0))),
        "descent_reverse": (q("descents", ["reverse"]), [q("descents")], op("sub", N, 1, y(0))),
        "peaks_reverse": (q("peaks", ["reverse"]), [q("peaks")], y(0)),
        "peaks_complement": (q("peaks"), [q("valleys", ["complement"])], y(0)),
        "successions_inverse": (q("successions", ["inverse"]), [q("successions")], y(0)),
        "successions_reverse_complement": (q("successions", ["reverse", "complement"]), [q("successions")], y(0)),
        "adjacency_two_successions": (q("adjacencies"), [q("successions"), q("successions", ["complement"])], op("add", y(0), y(1))),
        "deficiencies_inverse_reverse_complement": (q("deficiencies", ["inverse", "reverse", "complement"]), [q("deficiencies")], y(0)),
        "cycle_fixed_partition": (q("cycle_count"), [q("fixed_points"), q("nontrivial_cycle_count")], op("add", y(0), y(1))),
        "cycle_parity_partition": (q("cycle_count"), [q("even_cycle_count"), q("odd_cycle_count")], op("add", y(0), y(1))),
        "permutation_parity_cycles": (q("parity"), [q("cycle_count")], op("mod", op("sub", N, y(0)), 2)),
    }
    with (V5 / "audit/RELATION_INVENTORY.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        rid = row["relation_id"]
        if rid in compiled:
            target, queries, mapping = compiled[rid]
            edge = _edge(rid, target, queries, mapping, row["formula"], row["proof"])
        else:
            edge = {"relation_id": rid, "version": RELATIONS_VERSION,
                    "evidence_class": row["evidence_class"], "formula": row["formula"],
                    "relation_type": row["relation_type"], "scope": row["mathematical_scope"],
                    "required_tasks": row["required_tasks"].split(";"),
                    "proof": row["proof"], "answer_map": None,
                    "executable": False,
                    "execution_note": "Historical inequality, extra-feature identity or distribution statement; retained declaratively with original certificate.",
                    "mechanism_claim": False}
        edge["origin"] = "pilot_v5_unchanged_certificate"
        edge["finite_check"] = {"status": "reused_hash_verified", "inputs_checked": int(row["inputs_checked"]),
            "artifact": "experiments/pilot_v5_confirmation/mathematics/audit/RELATION_INVENTORY.csv",
            "artifact_sha256": V5_INVENTORY_SHA256, "relation_id": rid,
            "historical_rechecks_run": 0}
        result.append(edge)
    return result


def new_relations() -> list[dict]:
    """New edges, separately certified from immutable historical formulas."""
    edges: list[dict] = []
    add = edges.append
    positive_k = {"k": {"type": "integer", "minimum": 1, "maximum": 30}}
    add(_edge("cycle_length_one_fixed", q("cycle_length_count", k=1), [q("fixed_points")], y(0),
        "cycle_length_count(p,1)=fixed_points(p)", "A singleton orbit consists exactly of one fixed point."))
    add(_edge("cycle_length_two_alias", q("cycle_length_count", k=2), [q("two_cycle_count")], y(0),
        "cycle_length_count(p,2)=two_cycle_count(p)", "The retained legacy endpoint is the k=2 instance by its actual definition."))
    add(_edge("cycle_length_three_alias", q("cycle_length_count", k=3), [q("three_cycle_count")], y(0),
        "cycle_length_count(p,3)=three_cycle_count(p)", "The retained legacy endpoint is the k=3 instance by its actual definition."))
    for task in ["cycle_count", "nontrivial_cycle_count", "even_cycle_count", "odd_cycle_count", "longest_cycle", "shortest_cycle", "cycle_length_count"]:
        params = {"k": K} if task == "cycle_length_count" else {}
        add(_edge(f"{task}_inverse", q(task, **params), [q(task, ["inverse"], **params)], y(0),
            f"{task}(p)= {task}(inverse(p))", "Inversion reverses the direction of each disjoint cycle and preserves its members and length.",
            params=positive_k if params else None))
    add(_edge("parity_from_even_cycles", q("parity"), [q("even_cycle_count")], op("mod", y(0), 2),
        "parity(p)=even_cycle_count(p) mod 2", "An even-length cycle is odd, an odd-length cycle is even, and cycle signs multiply."))
    for task, reducer, formula in [("cycle_count", "sum", "C=sum_k c_k"),
                                    ("longest_cycle", "last_positive_index", "longest_cycle=max{k:c_k>0}"),
                                    ("shortest_cycle", "first_positive_index", "shortest_cycle=min{k:c_k>0}")]:
        add(_edge(f"{task}_from_cycle_histogram", q(task), [], op(reducer, {"queries": "all"}), formula,
            "The disjoint cycle lengths form a multiset partition of all n vertices; the complete histogram determines its size and extrema.",
            query_family={"task_id": "cycle_length_count", "params": {"k": {"index": "j"}}, "input_id": "p", "transforms": [], "index_range": [1, N]}))
    add(_edge("cycle_histogram_weighted_size", {"context": "n"}, [], op("weighted_sum_1based", {"queries": "all"}),
        "n=sum_{k=1}^n k*cycle_length_count(p,k)", "Every vertex belongs to exactly one disjoint cycle.",
        query_family={"task_id": "cycle_length_count", "params": {"k": {"index": "j"}}, "input_id": "p", "transforms": [], "index_range": [1, N]}))
    for task, other in [("lis_length", "lds_length"), ("increasing_subsequence_count", "decreasing_subsequence_count")]:
        params = {"k": K} if "count" in task else {}
        for transform in ["complement", "reverse"]:
            add(_edge(f"{task}_{transform}", q(task, **params), [q(other, [transform], **params)], y(0),
                f"{task}(p)= {other}({transform}(p))", "Complement reverses value comparisons; reversal reverses index order. Either maps increasing subsequences bijectively to decreasing subsequences of equal length.",
                params=positive_k if params else None))
    for task in ["lis_length", "lds_length", "increasing_subsequence_count", "decreasing_subsequence_count"]:
        params = {"k": K} if "count" in task else {}
        add(_edge(f"{task}_inverse", q(task, **params), [q(task, ["inverse"], **params)], y(0),
            f"{task}(p)={task}(inverse(p))", "Transposition of permutation graph points bijects monotone k-point subsets; sorting the new index coordinates preserves monotonicity type.",
            params=positive_k if params else None))
    for task in ["increasing_subsequence_count", "decreasing_subsequence_count"]:
        add(_edge(f"{task}_k1_constant", q(task, k=1), [], N,
            f"{task}(p,1)=n", "Every one-position subset is a monotone subsequence. Constant conditional on input length; not a learnability target."))
    add(_edge("decreasing_pairs_inversions", q("decreasing_subsequence_count", k=2), [q("length")], y(0),
        "decreasing_subsequence_count(p,2)=length(p)", "Both definitions count exactly pairs i<j with p_i>p_j; legacy length is inversion count, not input size."))
    add(_edge("increasing_pairs_complement_inversions", q("increasing_subsequence_count", k=2), [q("length")], op("sub", op("choose2", N), y(0)),
        "increasing_subsequence_count(p,2)=n(n-1)/2-length(p)", "Every two-position subset has distinct values and is either increasing or decreasing, exclusively."))
    add(_edge("monotone_pair_partition", {"context": "choose_n_2"}, [q("increasing_subsequence_count", k=2), q("decreasing_subsequence_count", k=2)], op("add", y(0), y(1)),
        "I_2(p)+D_2(p)=n(n-1)/2", "Partition all pairs by their strict comparison. This does not extend to k>=3 because mixed patterns also exist."))
    add(_edge("lis_from_subsequence_histogram", q("lis_length"), [], op("last_positive_index", {"queries": "all"}),
        "LIS(p)=max{k:I_k(p)>0}", "LIS is the greatest length for which an increasing subsequence exists.",
        query_family={"task_id": "increasing_subsequence_count", "params": {"k": {"index": "j"}}, "input_id": "p", "transforms": [], "index_range": [1, N]}))
    add(_edge("lis_from_rsk_first_row", q("lis_length"), [q("rsk_shape")], op("first", y(0)),
        "LIS(p)=rsk_shape(p)[0]", "The first row of row insertion stores minimal tails for increasing subsequences: insertion extends its length exactly when a longer increasing subsequence becomes possible.",
        status="proved_rsk_insertion_theorem"))
    add(_edge("lds_from_rsk_first_column", q("lds_length"), [q("rsk_shape")], op("len", y(0)),
        "LDS(p)=len(rsk_shape(p))", "The row-bumping lemma identifies the maximum decreasing subsequence length with the number of nonempty insertion rows (Schensted's theorem). Full theorem, not a claim derived from the finite checks.",
        status="proved_rsk_insertion_theorem"))
    add(_edge("rsk_shape_size", {"context": "n"}, [q("rsk_shape")], op("sum", y(0)),
        "sum(rsk_shape(p))=n", "Each insertion adds exactly one box and never deletes any box."))
    add(_edge("components_complement_global_descents", q("components"), [q("global_descents", ["complement"])], op("add", y(0), 1),
        "components(p)=global_descents(complement(p))+1", "At each internal cut j, all prefix values of p are smaller than all suffix values iff the first j values are {1,...,j}. Complement reverses this comparison. Direct-sum components count internal cuts plus one."))
    for task in ["components", "largest_component", "smallest_component", "component_length_count"]:
        params = {"k": K} if task == "component_length_count" else {}
        for transforms, suffix in [(["inverse"], "inverse"), (["reverse", "complement"], "reverse_complement")]:
            add(_edge(f"{task}_{suffix}", q(task, **params), [q(task, transforms, **params)], y(0),
                f"{task}(p)={task}({suffix}(p))", "Inversion preserves every initial interval invariant under p. Reverse-complement reverses the ordered direct-sum decomposition and preserves its component-length multiset.",
                params=positive_k if params else None))
    for task, reducer in [("components", "sum"), ("largest_component", "last_positive_index"), ("smallest_component", "first_positive_index")]:
        add(_edge(f"{task}_from_component_histogram", q(task), [], op(reducer, {"queries": "all"}),
            f"{task} reconstructed from all component_length_count(p,k), k=1..n",
            "The unique maximal splitting at invariant initial intervals partitions p into indecomposable direct-sum blocks. Its full length histogram determines count and extrema.",
            query_family={"task_id": "component_length_count", "params": {"k": {"index": "j"}}, "input_id": "p", "transforms": [], "index_range": [1, N]}))
    add(_edge("component_histogram_weighted_size", {"context": "n"}, [], op("weighted_sum_1based", {"queries": "all"}),
        "n=sum_{k=1}^n k*component_length_count(p,k)", "Every position belongs to one maximal indecomposable direct-sum block.",
        query_family={"task_id": "component_length_count", "params": {"k": {"index": "j"}}, "input_id": "p", "transforms": [], "index_range": [1, N]}))
    for edge in edges:
        edge["origin"] = "platform_v0.1_new_relation"
        edge["executable"] = True
    return edges


def relation_registry() -> dict:
    """Return proved relations plus explicit pointers to observations/unknowns."""
    return {"schema_version": RELATIONS_VERSION, "historical_certificates": historical_references(),
            "transformation_order": "Apply listed transforms left to right to the original input_id.",
            "absence_semantics": "No registered edge means not classified; never mathematically unrelated.",
            "relations": _historical_edges() + new_relations(),
            "empirical_associations": {"evidence_class": "empirically_observed", "source":
                "experiments/pilot_v5_confirmation/mathematics/audit/OBSERVED_ASSOCIATIONS.csv",
                "note": "Historical v4 learning co-movements, not identities or causal explanations; kept separate from executable formulas."},
            "unchecked": [
                {"claim": "Arbitrary unlisted joint reconstruction among tasks", "evidence_class": "unchecked"},
                {"claim": "A trained model actually computes a registered formula", "evidence_class": "unchecked"},
                {"claim": "Input-orbit relation equivalence implies equal learning difficulty", "evidence_class": "unchecked"},
                {"claim": "All tasks in distinct families are independent", "evidence_class": "unchecked"}]}


def list_relations() -> list[dict]:
    """Convenience interface consumed by the task metadata exporter."""
    return [{**edge, "evidence_status": edge["evidence_class"]}
            for edge in relation_registry()["relations"]]


def eval_expression(expr: Any, answers: Sequence[Any], context: Mapping[str, int]) -> Any:
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return expr
    if not isinstance(expr, dict):
        raise ValueError("Expression must be a JSON object or numeric literal")
    if set(expr) == {"query"}:
        return answers[expr["query"]]
    if set(expr) == {"queries"} and expr["queries"] == "all":
        return list(answers)
    if set(expr) == {"context"}:
        return context[expr["context"]]
    if set(expr) == {"index"}:
        return context[expr["index"]]
    if set(expr) != {"op", "args"}:
        raise ValueError(f"Unknown expression shape: {expr}")
    args = [eval_expression(arg, answers, context) for arg in expr["args"]]
    name = expr["op"]
    if name == "add":
        return sum(args)
    if name == "sub":
        return args[0] - sum(args[1:])
    if name == "mod":
        return args[0] % args[1]
    if name == "choose2":
        return args[0] * (args[0] - 1) // 2
    if name == "sum":
        return sum(args[0])
    if name == "len":
        return len(args[0])
    if name == "first":
        return args[0][0]
    if name == "weighted_sum_1based":
        return sum((i + 1) * value for i, value in enumerate(args[0]))
    if name == "first_positive_index":
        return next(i + 1 for i, value in enumerate(args[0]) if value > 0)
    if name == "last_positive_index":
        return max(i + 1 for i, value in enumerate(args[0]) if value > 0)
    raise ValueError(f"Expression operation not allowlisted: {name}")


def evaluate_relation(relation: dict, p: Sequence[int], params: dict | None = None) -> dict:
    """Execute a reconstruction against the registry oracle, no eval or code loading."""
    from .registry import label
    from .objects import transform

    if relation.get("answer_map") is None:
        raise ValueError("This historical relation is declarative, not a pointwise reconstruction")
    context = {"n": len(p), "choose_n_2": len(p) * (len(p) - 1) // 2, **(params or {})}

    def query_value(query: dict, ctx: dict) -> Any:
        if "context" in query:
            return ctx[query["context"]]
        if query["input_id"] != "p":
            raise ValueError("v0.1 relation evaluator accepts original input_id p only")
        transformed = tuple(p)
        for action in query.get("transforms", []):
            transformed = transform(action, transformed)
        query_params = {key: eval_expression(value, [], ctx) if isinstance(value, dict) else value
                        for key, value in query.get("params", {}).items()}
        return label(query["task_id"], transformed, params=query_params or None)

    answers = [query_value(query, context) for query in relation.get("queries", [])]
    family = relation.get("query_family")
    if family is not None:
        lo, hi = (eval_expression(value, [], context) for value in family["index_range"])
        answers.extend(query_value(family, {**context, "j": j}) for j in range(lo, hi + 1))
    predicted = eval_expression(relation["answer_map"], answers, context)
    actual = query_value(relation["target"], context)
    return {"relation_id": relation["relation_id"], "actual": actual,
            "reconstructed": predicted, "query_answers": answers, "passed": actual == predicted}
