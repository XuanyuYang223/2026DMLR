"""Object-only, orbit-isolated data for the first fixed-token task-count study.

The primary manifest is immutable. Test inputs are generated in a separate stage
after the complete set of adaptation selections has been frozen. Task labels are
computed by the unchanged PermWorld platform exact oracles at consumption time.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from permworld_platform.data import historical_inputs
from permworld_platform.objects import canonical_orbit_key, validate_permutation

DEFAULT_OUT = ROOT / "experiments/task_count_v1/data"
DATA_CONFIG = {
    "version": "task_count_orbit_objects_v1",
    "lengths": [10, 12],
    "replicates": [
        {"repeat": 1, "model_seed": 1103, "data_seed": 2026102101, "test_seed": 2026102151},
        {"repeat": 2, "model_seed": 2207, "data_seed": 2026102102, "test_seed": 2026102152},
        {"repeat": 3, "model_seed": 3301, "data_seed": 2026102103, "test_seed": 2026102153},
    ],
    "split_order": ["source_train", "source_dev", "target_train", "target_dev"],
    "per_length_counts": {"source_train": 4096, "source_dev": 256,
                          "target_train": 128, "target_dev": 256, "test": 512},
    "test_generation": "after all 24 pretraining runs and 108 target training selections are frozen",
    "proposal": "uniform permutation proposals; reject complete inverse/reverse/complement orbits",
    "rng": "numpy.PCG64(SeedSequence([data_seed, role_index, n])); test uses test_seed and role_index=4",
    "generation_order": "all training/development pools in repeat, role, length order; test pools last in repeat, length order",
    "historical_exclusion": "pilot_v1 through pilot_v5, the canonical platform_v01/dataset, and platform_v01 runnable-example objects",
    "pairing": "all source tasks and all target tasks use the same role-specific objects within a repeat; conditions do not alter object pools",
    "capacity_rule": "floor(n!/8) minus occupied historical orbits must cover every requested role across all repeats, including future tests",
    "max_proposals_per_pool": 2000000,
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf8")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _history():
    occupied, bindings = historical_inputs()
    platform_dir = ROOT / "experiments/platform_v01/dataset/data"
    if not platform_dir.is_dir():
        raise FileNotFoundError("the canonical platform v0.1 dataset is required for historical exclusion")
    raw_seen = set()
    historical_platform_paths = list(platform_dir.glob("*.npz"))
    historical_platform_paths += list((ROOT / "experiments/platform_v01/examples").rglob("*.npz"))
    for path in sorted(historical_platform_paths):
        with np.load(path, allow_pickle=False) as archive:
            if "p" not in archive or archive["p"].ndim != 2:
                raise ValueError(f"invalid historical platform archive: {path}")
            for row in archive["p"]:
                p = tuple(map(int, row))
                if p not in raw_seen:
                    raw_seen.add(p)
                    occupied.add(canonical_orbit_key(p))
        bindings[path.relative_to(ROOT).as_posix()] = sha(path)
    if not bindings:
        raise ValueError("historical archives are missing; refusing to weaken isolation")
    return occupied, dict(sorted(bindings.items()))


def _capacity(occupied):
    counts = Counter(map(len, occupied))
    requested = len(DATA_CONFIG["replicates"]) * sum(DATA_CONFIG["per_length_counts"].values())
    rows = []
    for n in DATA_CONFIG["lengths"]:
        lower_bound = math.factorial(n) // 8
        remaining = lower_bound - counts[n]
        row = {"n": n, "permutations": math.factorial(n), "maximum_orbit_size": 8,
               "orbit_count_lower_bound": lower_bound, "historical_occupied_orbits": counts[n],
               "available_orbit_lower_bound": remaining,
               "requested_new_orbits_all_roles_and_repeats": requested,
               "reserved_future_test_orbits": 512 * len(DATA_CONFIG["replicates"]),
               "conservative_capacity_sufficient": remaining >= requested}
        rows.append(row)
        if remaining < requested:
            raise RuntimeError(f"conservative orbit capacity is insufficient: {row}; no isolation rule was relaxed")
    return rows


def _descriptors(include_test=False):
    for rep in DATA_CONFIG["replicates"]:
        for role_index, role in enumerate(DATA_CONFIG["split_order"]):
            for n in DATA_CONFIG["lengths"]:
                yield rep, role, role_index, n
    if include_test:
        for rep in DATA_CONFIG["replicates"]:
            for n in DATA_CONFIG["lengths"]:
                yield rep, "test", 4, n


def _sample(rep, role, role_index, n, occupied):
    rng_seed = rep["test_seed"] if role == "test" else rep["data_seed"]
    rng = np.random.default_rng(np.random.SeedSequence([rng_seed, role_index, n]))
    accepted, attempts = [], 0
    count = DATA_CONFIG["per_length_counts"][role]
    while len(accepted) < count:
        attempts += 1
        if attempts > DATA_CONFIG["max_proposals_per_pool"]:
            raise RuntimeError(f"proposal cap reached for repeat={rep['repeat']} role={role} n={n}; refusing overlap")
        p = tuple(map(int, rng.permutation(n) + 1))
        key = canonical_orbit_key(p)
        if key in occupied:
            continue
        occupied.add(key)
        accepted.append(p)
    return accepted, attempts, rng_seed


def _write_pool(out, rep, role, role_index, n, occupied):
    start = time.perf_counter()
    objects, proposals, rng_seed = _sample(rep, role, role_index, n, occupied)
    path = out / "objects" / f"s{rep['model_seed']}_{role}_n{n}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite object archive: {path}")
    np.savez_compressed(path, p=np.asarray(objects, dtype=np.int16))
    record = {"repeat": rep["repeat"], "model_seed": rep["model_seed"],
              "data_seed": rep["data_seed"], "rng_seed": rng_seed,
              "test_seed": rep["test_seed"], "role": role, "role_index": role_index,
              "n": n, "count": len(objects), "path": path.relative_to(out).as_posix(),
              "sha256": sha(path), "canonical_input_digest": json_hash(objects),
              "orbit_digest": json_hash([canonical_orbit_key(p) for p in objects]),
              "proposals": proposals, "rejected": proposals - len(objects),
              "seconds": time.perf_counter() - start}
    print(f"[data] repeat={rep['repeat']} {role} n={n} count={len(objects)} rejected={record['rejected']}", flush=True)
    return record


def prepare(out=DEFAULT_OUT):
    """Create immutable training/development pools, reserving capacity for tests."""
    out = Path(out)
    if (out / "manifest.json").exists():
        manifest = _read(out / "manifest.json")
        if manifest["config"] != DATA_CONFIG:
            raise ValueError("existing dataset configuration differs; choose a new output directory")
        verify(out, include_test=False, replay=False)
        return manifest
    if (out / "objects").exists():
        raise FileExistsError("partial data generation exists; preserve it and choose a fresh output directory")
    start = time.perf_counter()
    occupied, history = _history()
    capacity = _capacity(occupied)
    write_json(out / "DATA_CONFIG.json", DATA_CONFIG)
    write_json(out / "capacity_audit.json", {"status": "pass", "generated_at_utc": utc_now(),
                                           "method": DATA_CONFIG["capacity_rule"], "lengths": capacity})
    history_count = len(occupied)
    pools = [_write_pool(out, rep, role, role_index, n, occupied)
             for rep, role, role_index, n in _descriptors()]
    manifest = {"status": "complete", "created_at_utc": utc_now(), "config": DATA_CONFIG,
                "config_sha256": json_hash(DATA_CONFIG), "pools": pools,
                "history_bindings": history, "historical_unique_orbits": history_count,
                "historical_orbits_by_length": dict(sorted(Counter(map(len, occupied)).items())),
                "capacity_audit": capacity, "test_generated": False,
                "source_bindings": {"permworld_platform/objects.py": sha(ROOT / "permworld_platform/objects.py"),
                                    "permworld_platform/data.py": sha(ROOT / "permworld_platform/data.py")},
                "seconds": time.perf_counter() - start}
    # Counts in the historical-only view must not include newly generated pools.
    for record in pools:
        key = record["n"]
        manifest["historical_orbits_by_length"][key] -= record["count"]
    write_json(out / "manifest.json", manifest)
    return manifest


def _assert_manifest(out):
    manifest = _read(out / "manifest.json")
    if manifest["status"] != "complete" or manifest["config"] != DATA_CONFIG:
        raise ValueError("incomplete or incompatible primary manifest")
    if manifest["config_sha256"] != json_hash(DATA_CONFIG):
        raise ValueError("configuration digest mismatch")
    for rel, digest in manifest["source_bindings"].items():
        if sha(ROOT / rel) != digest:
            raise ValueError(f"platform isolation source changed: {rel}")
    return manifest


def _pool_array(out, record):
    path = out / record["path"]
    if sha(path) != record["sha256"]:
        raise ValueError(f"object archive digest mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"p"}:
            raise ValueError("unexpected fields in object archive")
        p = archive["p"].copy()
    if p.dtype != np.int16 or p.shape != (record["count"], record["n"]):
        raise ValueError("object archive dtype or shape differs from the manifest")
    objects = [tuple(map(int, row)) for row in p]
    if json_hash(objects) != record["canonical_input_digest"]:
        raise ValueError("canonical input digest mismatch")
    return p


def load_objects(out, seed, role, n):
    """Return an int16 permutation matrix after checking frozen file/content hashes."""
    out = Path(out)
    manifest = _assert_manifest(out)
    records = manifest["pools"]
    if role == "test":
        test = _read(out / "test_manifest.json")
        if test["primary_manifest_sha256"] != sha(out / "manifest.json"):
            raise ValueError("test pool primary-manifest binding mismatch")
        records = test["pools"]
    rows = [r for r in records if (r["model_seed"], r["role"], r["n"]) == (seed, role, n)]
    if len(rows) != 1:
        raise ValueError(f"unknown or duplicate pool: seed={seed}, role={role}, n={n}")
    return _pool_array(out, rows[0])


def _selection_gate(path):
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    value = _read(path)
    if value.get("status") != "frozen" or value.get("training_complete") is not True:
        raise ValueError("test generation requires a frozen selection manifest with training_complete=true")
    if not isinstance(value.get("selections"), list) or len(value["selections"]) != 108:
        raise ValueError("all 108 target-training selections must be frozen before generating test inputs")
    when = datetime.fromisoformat(value["frozen_at_utc"].replace("Z", "+00:00"))
    if when.tzinfo is None or when > datetime.now(timezone.utc):
        raise ValueError("selection freeze timestamp must be timezone-aware and no later than test generation")
    try:
        portable_path = path.relative_to(ROOT).as_posix()
    except ValueError:
        portable_path = str(path)
    return {"path": portable_path, "sha256": sha(path), "frozen_at_utc": value["frozen_at_utc"],
            "selection_count": len(value["selections"])}


def generate_test(out=DEFAULT_OUT, selection_manifest_path=None):
    """Generate the independent test pool only after final training selections."""
    if selection_manifest_path is None:
        raise ValueError("a final frozen selection manifest is required")
    out = Path(out)
    gate = _selection_gate(selection_manifest_path)
    manifest = _assert_manifest(out)
    if (out / "test_manifest.json").exists():
        previous = _read(out / "test_manifest.json")
        if previous["selection_freeze"]["sha256"] != gate["sha256"]:
            raise ValueError("test inputs are already bound to another selection freeze")
        verify(out, include_test=True, replay=False)
        return previous
    if list((out / "objects").glob("*_test_*.npz")):
        raise FileExistsError("partial test generation exists; refusing to overwrite it")
    start = time.perf_counter()
    occupied, history = _history()
    if history != manifest["history_bindings"]:
        raise ValueError("historical inputs changed since training data were frozen")
    for record in manifest["pools"]:
        for row in _pool_array(out, record):
            key = canonical_orbit_key(tuple(map(int, row)))
            if key in occupied:
                raise ValueError("pre-existing orbit collision detected before test generation")
            occupied.add(key)
    test_rows = [_write_pool(out, rep, role, role_index, n, occupied)
                 for rep, role, role_index, n in _descriptors(include_test=True) if role == "test"]
    result = {"status": "complete", "created_at_utc": utc_now(),
              "primary_manifest_sha256": sha(out / "manifest.json"),
              "selection_freeze": gate, "pools": test_rows,
              "seconds": time.perf_counter() - start,
              "test_use": "one unified final evaluation; no new training or model selection after test generation"}
    write_json(out / "test_manifest.json", result)
    return result


def verify(out=DEFAULT_OUT, include_test=False, replay=True):
    """Audit every object, the complete orbit separation, and optional RNG replay."""
    out = Path(out)
    start = time.perf_counter()
    manifest = _assert_manifest(out)
    occupied, history = _history()
    if history != manifest["history_bindings"]:
        raise ValueError("historical file bindings differ from the frozen manifest")
    capacity = _capacity(occupied)
    rows = list(manifest["pools"])
    test = None
    if include_test:
        test = _read(out / "test_manifest.json")
        if test["primary_manifest_sha256"] != sha(out / "manifest.json"):
            raise ValueError("test manifest primary binding mismatch")
        gate = _selection_gate(test["selection_freeze"]["path"])
        if gate != test["selection_freeze"]:
            raise ValueError("selection freeze changed after test generation")
        if datetime.fromisoformat(test["created_at_utc"]) < datetime.fromisoformat(gate["frozen_at_utc"].replace("Z", "+00:00")):
            raise ValueError("test generation predates final selection freeze")
        rows.extend(test["pools"])
    descriptors = list(_descriptors(include_test))
    if len(rows) != len(descriptors):
        raise ValueError("missing or extra object pools")
    counts = Counter()
    orbit_digests = []
    for record, (rep, role, role_index, n) in zip(rows, descriptors):
        expected = (rep["repeat"], rep["model_seed"], rep["data_seed"], rep["test_seed"], role, role_index, n, DATA_CONFIG["per_length_counts"][role])
        actual = tuple(record[key] for key in ["repeat", "model_seed", "data_seed", "test_seed", "role", "role_index", "n", "count"])
        if actual != expected:
            raise ValueError("pool identity/order differs from frozen generation order")
        array = _pool_array(out, record)
        objects = [validate_permutation(tuple(map(int, p))) for p in array]
        keys = [canonical_orbit_key(p) for p in objects]
        if len(set(keys)) != len(keys) or any(key in occupied for key in keys):
            raise ValueError("duplicate orbit within or across pool/history boundaries")
        if json_hash(keys) != record["orbit_digest"]:
            raise ValueError("orbit digest mismatch")
        if replay:
            generated, proposals, rng_seed = _sample(rep, role, role_index, n, occupied)
            if generated != objects or proposals != record["proposals"] or rng_seed != record["rng_seed"]:
                raise ValueError("object pool cannot be reproduced by its frozen RNG stream")
        else:
            occupied.update(keys)
        counts[role] += len(objects)
        orbit_digests.append(record["orbit_digest"])
    result = {"status": "pass", "created_at_utc": utc_now(), "include_test": include_test,
              "rng_replayed": replay, "pool_count": len(rows), "object_count": sum(counts.values()),
              "objects_by_role": dict(counts), "within_pool_duplicate_orbits": 0,
              "between_pool_overlapping_orbits": 0, "historical_overlapping_orbits": 0,
              "history_file_count": len(history), "capacity": capacity,
              "ordered_pool_orbit_digest": json_hash(orbit_digests),
              "primary_manifest_sha256": sha(out / "manifest.json"),
              "test_manifest_sha256": sha(out / "test_manifest.json") if include_test else None,
              "seconds": time.perf_counter() - start}
    write_json(out / ("quality_with_test.json" if include_test else "quality_pretest.json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify", "generate-test", "capacity"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument("--selection-manifest", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.out)
    elif args.command == "verify":
        result = verify(args.out, args.include_test, not args.no_replay)
    elif args.command == "generate-test":
        result = generate_test(args.out, args.selection_manifest)
    else:
        occupied, files = _history()
        result = {"status": "pass", "capacity": _capacity(occupied), "historical_file_count": len(files)}
    print(json.dumps({k: v for k, v in result.items() if k not in {"pools", "history_bindings", "config"}}, indent=2))


if __name__ == "__main__":
    main()
