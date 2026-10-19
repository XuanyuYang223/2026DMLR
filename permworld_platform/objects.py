"""Validated PermWorld objects and declared input transformations (v0.1)."""
from __future__ import annotations

from collections.abc import Iterable

MIN_LENGTH = 1
MAX_LENGTH = 30
TRANSFORM_ACTIONS = tuple(f"{i:03b}" for i in range(8))


def validate_permutation(permutation: Iterable[int]) -> tuple[int, ...]:
    """Return p in S_n for 1 <= n <= 30; reject coercions and malformed input."""
    if isinstance(permutation, (str, bytes)):
        raise TypeError("permutation must be an iterable of built-in integers")
    try:
        p = tuple(permutation)
    except TypeError as exc:
        raise TypeError("permutation must be iterable") from exc
    if any(type(value) is not int for value in p):
        raise TypeError("permutation entries must be built-in integers, not booleans")
    if not MIN_LENGTH <= len(p) <= MAX_LENGTH:
        raise ValueError(f"platform v0.1 supports {MIN_LENGTH} <= n <= {MAX_LENGTH}")
    if set(p) != set(range(1, len(p) + 1)):
        raise ValueError("permutation must contain each integer 1..n exactly once")
    return p


def transform(action: str, permutation: Iterable[int]) -> tuple[int, ...]:
    """Apply identity/I/R/C or an action bit string in inverse, reverse, complement order.

    For example '101' means C(I(p)); '110' means R(I(p)). These eight
    actions form the dihedral symmetries of a permutation plot. They are
    input transformations, not additional task templates or encodings.
    """
    p = validate_permutation(permutation)
    aliases = {"identity": "000", "inverse": "100", "reverse": "010", "complement": "001"}
    bits = aliases.get(action, action)
    if bits not in TRANSFORM_ACTIONS:
        raise ValueError(f"unknown transformation {action!r}")
    if bits[0] == "1":
        inverse = [0] * len(p)
        for i, value in enumerate(p, 1):
            inverse[value - 1] = i
        p = tuple(inverse)
    if bits[1] == "1":
        p = p[::-1]
    if bits[2] == "1":
        p = tuple(len(p) + 1 - value for value in p)
    return p


def canonical_orbit_key(permutation: Iterable[int]) -> tuple[int, ...]:
    """Lexicographically smallest I/R/C image; length is implicit in the tuple."""
    p = validate_permutation(permutation)
    return min(transform(action, p) for action in TRANSFORM_ACTIONS)
