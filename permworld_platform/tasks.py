"""Exact task labels, preserving all 48 historical PermWorld endpoints.

The vendored mathematical functions remain authoritative for historical
tasks. Platform validation deliberately limits lengths to 1..30 and makes
all additional operands/parameters explicit. New count labels are Python
integers and are never truncated, reduced modulo a base, or coerced to a
single-token range.
"""
from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path

from .objects import validate_permutation

LEGACY_PATH = Path(__file__).resolve().parents[1] / "experiments/permworld_pilot/vendor/neurips_permutations/math_ops.py"
_module_spec = importlib.util.spec_from_file_location("permworld_platform._legacy_math_ops", LEGACY_PATH)
if _module_spec is None or _module_spec.loader is None:
    raise ImportError(f"cannot load historical math definitions from {LEGACY_PATH}")
legacy = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(legacy)

LEGACY_TASK_IDS = tuple(dict.fromkeys(legacy.V2_TASK_NAMES + legacy.V3_TASK_NAMES + legacy.PROPERTY32_TASK_NAMES))
NEW_TASK_IDS = (
    "cycle_length_count", "increasing_subsequence_count", "decreasing_subsequence_count",
    "largest_component", "smallest_component", "component_length_count",
)
TASK_IDS = LEGACY_TASK_IDS + NEW_TASK_IDS
PARAMETER_K_TASKS = frozenset({"cycle_length_count", "increasing_subsequence_count", "decreasing_subsequence_count", "component_length_count"})
BINARY_TASKS = frozenset({"compose", "conjugate", "commutator", "bruhat_leq"})


def component_lengths(permutation) -> tuple[int, ...]:
    """Sizes, in left-to-right order, of the finest direct-sum decomposition.

    A boundary after position i occurs iff max(p[0:i]) == i. Taking every
    boundary gives the unique direct-sum indecomposable components.
    """
    p = validate_permutation(permutation)
    maximum = previous = 0
    lengths = []
    for i, value in enumerate(p, 1):
        maximum = max(maximum, value)
        if maximum == i:
            lengths.append(i - previous)
            previous = i
    return tuple(lengths)


def _subsequence_count(p: tuple[int, ...], k: int, increasing: bool) -> int:
    if k > len(p):
        return 0
    counts = [1] * len(p)
    for _length in range(2, k + 1):
        counts = [sum(counts[j] for j in range(i) if (p[j] < p[i]) == increasing) for i in range(len(p))]
    return sum(counts)


def _mapping(value, name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def validate_parameters(task_id: str, params=None) -> dict:
    """Validate exact named parameters; no silent defaults for parameterized tasks."""
    if task_id not in TASK_IDS:
        raise KeyError(f"unknown task_id {task_id!r}")
    parameters = _mapping(params, "params")
    expected = {"k"} if task_id in PARAMETER_K_TASKS else {
        "pattern_avoidance": {"pattern"}, "power": {"exponent"},
        "right_multiply_simple": {"index"},
    }.get(task_id, set())
    if set(parameters) != expected:
        raise ValueError(f"{task_id} requires exactly parameters {sorted(expected)}; got {sorted(parameters)}")
    if "k" in parameters:
        k = parameters["k"]
        if type(k) is not int:
            raise TypeError("k must be a built-in integer")
        if not 1 <= k <= 30:
            raise ValueError("k must be in 1..30 (k>n yields zero)")
    if "pattern" in parameters:
        parameters["pattern"] = validate_permutation(parameters["pattern"])
    if "exponent" in parameters:
        exponent = parameters["exponent"]
        if type(exponent) is not int:
            raise TypeError("exponent must be a built-in integer")
        if exponent < 0:
            raise ValueError("exponent must be nonnegative")
    if "index" in parameters:
        index = parameters["index"]
        if type(index) is not int:
            raise TypeError("index must be a built-in integer")
        if not 1 <= index < 30:
            raise ValueError("index must be in 1..29 and smaller than input length")
    return parameters


def label(task_id: str, permutation, params=None, operands=None):
    """Return the exact Python label for a registered task.

    Binary operands are supplied as ``operands={"q": q}``. compose means
    p o q; conjugate means p o q o p^-1 (p is the conjugator); commutator
    means p o q o p^-1 o q^-1; bruhat_leq tests p <= q. Parameterized
    pattern_avoidance takes params={"pattern": (...)}, power takes
    params={"exponent": e}, and right_multiply_simple takes {"index": i}.
    These conventions match the frozen historical functions exactly.
    """
    p = validate_permutation(permutation)
    parameters = validate_parameters(task_id, params)
    other = _mapping(operands, "operands")
    if task_id in BINARY_TASKS:
        if set(other) != {"q"}:
            raise ValueError(f"{task_id} requires exactly operands={{'q': permutation}}")
        q = validate_permutation(other["q"])
        if len(q) != len(p):
            raise ValueError("binary operands must have equal lengths")
        return getattr(legacy, task_id)(p, q)
    if other:
        raise ValueError(f"{task_id} accepts no additional operands")
    if task_id in legacy.PROPERTY_FUNCTIONS:
        return legacy.PROPERTY_FUNCTIONS[task_id](p)
    unary = {
        "to_cycle": legacy.canonical_cycles, "to_lehmer": legacy.lehmer_code,
        "to_inversion_vector": legacy.inversion_vector, "to_reduced_word": legacy.reduced_coxeter_word,
        "length": legacy.inversion_count, "parity": legacy.parity,
        "cycle_type": legacy.cycle_type, "rsk_shape": legacy.rsk_shape, "inverse": legacy.inverse,
    }
    if task_id in unary:
        return unary[task_id](p)
    if task_id == "pattern_avoidance":
        return legacy.avoids_pattern(p, parameters["pattern"])
    if task_id == "power":
        return legacy.power(p, parameters["exponent"])
    if task_id == "right_multiply_simple":
        return legacy.right_multiply_simple(p, parameters["index"])
    if task_id == "cycle_length_count":
        return legacy.cycle_length_count(p, parameters["k"])
    if task_id in {"increasing_subsequence_count", "decreasing_subsequence_count"}:
        return _subsequence_count(p, parameters["k"], task_id == "increasing_subsequence_count")
    if task_id == "largest_component":
        return max(component_lengths(p))
    if task_id == "smallest_component":
        return min(component_lengths(p))
    if task_id == "component_length_count":
        return component_lengths(p).count(parameters["k"])
    raise RuntimeError(f"registered task {task_id} has no implementation")
