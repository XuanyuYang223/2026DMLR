"""Read-only reproduction of the blocked n=8 calibration capacity check.

Run from the repository root with Python -B. This standalone program imports no
project modules, creates no files or data, and does not access a model. It prints
JSON to stdout. The canonical v0.1 pools alone suffice to prove the shortfall;
additional historical exclusions can only reduce the available capacity.
"""

import collections
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LENGTH = 8
TRANSFORMS = (
    "identity", "reverse", "complement", "reverse_complement",
    "inverse", "inverse_reverse", "inverse_complement",
    "inverse_reverse_complement",
)


def inverse(permutation):
    result = [0] * len(permutation)
    for position, value in enumerate(permutation, 1):
        result[value - 1] = position
    return tuple(result)


def orbit(permutation):
    p = tuple(map(int, permutation))
    n = len(p)
    reverse = p[::-1]
    complement = tuple(n + 1 - value for value in p)
    reverse_complement = complement[::-1]
    return (
        p, reverse, complement, reverse_complement,
        inverse(p), inverse(reverse), inverse(complement),
        inverse(reverse_complement),
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    all_orbits = {}
    fixed_counts = [0] * len(TRANSFORMS)
    for p in itertools.permutations(range(1, LENGTH + 1)):
        variants = orbit(p)
        for index, variant in enumerate(variants):
            fixed_counts[index] += variant == p
        all_orbits[min(variants)] = len(set(variants))

    historical = set()
    canonical = set()
    inputs = []
    expected_values = list(range(1, LENGTH + 1))
    for path in sorted((ROOT / "experiments").rglob("*.npz")):
        if "platform_calibration_v02" in path.parts:
            continue
        with np.load(path, allow_pickle=False) as archive:
            if "p" not in archive.files:
                continue
            values = archive["p"]
            if values.ndim != 2 or values.shape[1] != LENGTH:
                continue
            permutations = [tuple(map(int, row)) for row in values]
        if any(sorted(p) != expected_values for p in permutations):
            raise ValueError(f"Invalid permutation in {path}")
        keys = {min(orbit(p)) for p in permutations}
        historical.update(keys)
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith("experiments/platform_v01/dataset/data/"):
            canonical.update(keys)
        inputs.append({
            "path": relative,
            "sha256": digest(path),
            "rows": len(permutations),
            "distinct_orbits": len(keys),
        })

    config_path = ROOT / "experiments/platform_v01/calibration/FROZEN_CALIBRATION.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))["config"]
    required = len(config["repeats"]) * (
        config["train_per_length"] + config["dev_per_length"]
    )
    available = len(set(all_orbits) - historical)
    assert sum(fixed_counts) == len(TRANSFORMS) * len(all_orbits)
    assert sum(all_orbits.values()) == 40320
    print(json.dumps({
        "status": "blocked" if available < required else "capacity_only_pass",
        "n": LENGTH,
        "total_orbits": len(all_orbits),
        "transform_order": TRANSFORMS,
        "burnside_fixed_counts": fixed_counts,
        "burnside_orbits": sum(fixed_counts) // len(TRANSFORMS),
        "orbit_size_distribution": dict(collections.Counter(all_orbits.values())),
        "canonical_platform_occupied_orbits": len(canonical),
        "canonical_only_available_orbits": len(set(all_orbits) - canonical),
        "experiments_p_array_occupied_orbits": len(historical),
        "available_before_other_history": available,
        "required_new_train_dev_orbits": required,
        "shortfall": max(0, required - available),
        "test_orbits_requested_in_this_bound": 0,
        "configuration_path": config_path.relative_to(ROOT).as_posix(),
        "configuration_sha256": digest(config_path),
        "historical_inputs": inputs,
        "scope": "Experiments NPZ p-arrays only; sufficient impossibility bound, not a complete audit of all historical formats/reference inputs.",
        "no_project_imports": True,
        "training_or_model_evaluation": False,
        "new_data_generation": False,
        "writes_files": False,
    }, indent=2))


if __name__ == "__main__":
    main()
