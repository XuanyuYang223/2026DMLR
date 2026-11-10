"""Fixed-size, object-only pools with complete historical I/R/C orbit exclusion.

No test object is generated until all 120 development selections are frozen.
The historical readers are used read-only. This module never writes old assets.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fixed_points_deficiencies_v1.design import CFG, REPEATS
from experiments.task_count_v2 import data_pipeline as archived_data
from permworld_platform.objects import canonical_orbit_key, validate_permutation

DEFAULT_OUT = HERE / "artifacts" / "data"
ROLES = ["source_train", "source_dev", "target_train", "target_dev"]
COUNTS = dict(source_train=8192, source_dev=1024, target_train=1024,
              target_dev=1024, test=2048)
assert CFG["n"] == 10 and CFG["counts"] == COUNTS and len(REPEATS) == 8
assert [r["repeat_id"] for r in REPEATS] == [f"R{i}" for i in range(1, 9)]
for key in ("model_seed", "data_seed", "test_seed", "subset_seed"):
    assert len({r[key] for r in REPEATS}) == 8

DATA_CONFIG = {
    "version": "fixed_points_deficiencies_orbit_objects_v1",
    "n": 10, "repeats": REPEATS, "counts": COUNTS, "split_order": ROLES,
    "target_sizes": [256, 512, 1024],
    "generation_order": "all pretest pools in repeat then role order; tests last in repeat order",
    "proposal": "uniform permutations; reject all previously occupied inverse/reverse/complement orbits",
    "rng": "numpy.PCG64(SeedSequence([data_seed, role_index, n])); test uses test_seed and role_index=4",
    "target_order": "accepted maximum pool reordered by PCG64(SeedSequence([subset_seed,n])).permutation(1024); nested prefixes",
    "history": "all archived DMLR permutation NPZs outside this experiment, including pilots, platform examples, task-count v1/v2, historical smoke pools and relation_readout_v1",
    "capacity": "floor(n!/8) minus all historically occupied n=10 orbits must cover every new pool including future tests",
    "test_gate": "120 distinct frozen selections covering 8 units x 5 routes x 3 sizes; training_complete=true",
    "pairing": "all conditions share role-specific inputs within a unit; only task definitions/labels differ",
    "sampling_limitations": "finite permutation space; uniform permutation proposals followed by orbit rejection are not iid uniform surviving orbits; global exclusion conditions later pools on earlier acceptance",
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


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf8")


def _relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def _bound_file(bindings, path, expected=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"required historical asset is missing: {path}")
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError(f"historical asset changed: {path}")
    bindings[_relative(path)] = actual


def _check_historical_bindings(bindings, values):
    for rel, digest in values.items():
        _bound_file(bindings, ROOT / rel.replace("\\", "/"), digest)


def _history():
    """Require complete archived inventories, then cover every historical NPZ.

    The additional all-NPZ scan closes omissions of historical smoke/readout
    pools in the earlier study's historical reader. No outcome table is read.
    """
    occupied, bindings = archived_data._history()
    v2 = ROOT / "experiments/task_count_v2/data"
    primary = _read(v2 / "manifest.json")
    tests = _read(v2 / "test_manifest.json")
    for name in ("DATA_CONFIG.json", "manifest.json", "test_manifest.json"):
        _bound_file(bindings, v2 / name)
    _check_historical_bindings(bindings, primary["history_bindings"])
    if tests["primary_manifest_sha256"] != sha(v2 / "manifest.json"):
        raise ValueError("historical task-count v2 test binding changed")
    records = primary["pools"] + tests["pools"]
    expected = {v2 / r["path"] for r in records}
    if len(records) != 30 or expected != set((v2 / "objects").glob("*.npz")):
        raise ValueError("historical task-count v2 pool inventory is incomplete or unbound")
    for row in records:
        _bound_file(bindings, v2 / row["path"], row["sha256"])
        archived_data._pool_array(v2, row)

    platform = ROOT / "experiments/platform_v01/dataset"
    platform_manifest = _read(platform / "manifest.json")
    _bound_file(bindings, platform / "manifest.json")
    expected = {platform / r["object_path"] for r in platform_manifest["pools"]}
    if len(expected) != 18 or expected != set((platform / "data").glob("*.npz")):
        raise ValueError("canonical platform object inventory is incomplete or unbound")
    for row in platform_manifest["pools"]:
        _bound_file(bindings, platform / row["object_path"], row["object_sha256"])
    _check_historical_bindings(bindings, platform_manifest["historical_files"])

    readout = ROOT / "experiments/relation_readout_v1"
    readout_manifest = _read(readout / "data_manifest.json")
    _bound_file(bindings, readout / "data_manifest.json")
    _bound_file(bindings, readout / "FROZEN_PROTOCOL.json", readout_manifest["protocol_sha256"])
    _check_historical_bindings(bindings, readout_manifest["historical_bindings"])
    rows = readout_manifest["pools"]
    if {(r["seed"], r["n"]) for r in rows} != {(s, n) for s in (4103, 5207, 6301) for n in (10, 12)}:
        raise ValueError("relation-readout historical inventory is incomplete")
    expected = {readout / r["path"].replace("\\", "/") for r in rows}
    if len(rows) != 6 or expected != set((readout / "data").glob("*.npz")):
        raise ValueError("relation-readout contains duplicate or unbound pools")
    for row in rows:
        path = readout / row["path"].replace("\\", "/")
        _bound_file(bindings, path, row["sha256"])
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["p"] or archive["p"].shape != (512, row["n"]):
                raise ValueError(f"unexpected relation-readout pool: {path}")

    # Bind and inspect all archived object files, including old engineering runs.
    # New experiment files are excluded here and checked as separate new pools.
    inventory, raw_seen = [], set()
    for path in sorted((ROOT / "experiments").rglob("*.npz")):
        if path.is_relative_to(HERE):
            continue
        _bound_file(bindings, path)
        with np.load(path, allow_pickle=False) as archive:
            if "p" not in archive or archive["p"].ndim != 2:
                raise ValueError(f"unclassified historical NPZ; inspect before proceeding: {path}")
            array = archive["p"]
            if not np.issubdtype(array.dtype, np.integer):
                raise ValueError(f"non-integer historical permutations: {path}")
            for row in array:
                p = tuple(map(int, row))
                if p not in raw_seen:
                    validate_permutation(p)
                    raw_seen.add(p)
                    occupied.add(canonical_orbit_key(p))
            inventory.append({"path": _relative(path), "sha256": bindings[_relative(path)],
                              "count": len(array), "n": array.shape[1]})
    if len(inventory) < 179:
        raise ValueError("historical inventory is smaller than the 179 archives inspected before implementation")
    return occupied, dict(sorted(bindings.items())), inventory


def _capacity(occupied):
    historical = sum(len(key) == 10 for key in occupied)
    lower = math.factorial(10) // 8
    requested = len(REPEATS) * sum(COUNTS.values())
    result = {"n": 10, "permutations": math.factorial(10), "maximum_orbit_size": 8,
              "orbit_count_lower_bound": lower, "historical_occupied_orbits": historical,
              "available_orbit_lower_bound": lower - historical,
              "requested_new_orbits_all_roles_and_units": requested,
              "reserved_future_test_orbits": len(REPEATS) * COUNTS["test"],
              "conservative_capacity_sufficient": lower - historical >= requested}
    if not result["conservative_capacity_sufficient"]:
        raise RuntimeError(f"insufficient orbit capacity; no rule was relaxed: {result}")
    return result


def _descriptors(include_test=False):
    for rep in REPEATS:
        for role_index, role in enumerate(ROLES):
            yield rep, role, role_index
    if include_test:
        for rep in REPEATS:
            yield rep, "test", 4


def _sample(rep, role, role_index, occupied):
    seed = rep["test_seed"] if role == "test" else rep["data_seed"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, role_index, 10]))
    accepted, attempts = [], 0
    while len(accepted) < COUNTS[role]:
        attempts += 1
        if attempts > DATA_CONFIG["max_proposals_per_pool"]:
            raise RuntimeError(f"proposal cap reached for {rep['repeat_id']} {role}; refusing overlap")
        p = tuple(map(int, rng.permutation(10) + 1))
        key = canonical_orbit_key(p)
        if key in occupied:
            continue
        occupied.add(key)
        accepted.append(p)
    raw_digest = json_hash(accepted)
    order = list(range(len(accepted)))
    if role == "target_train":
        subset_rng = np.random.default_rng(np.random.SeedSequence([rep["subset_seed"], 10]))
        order = list(map(int, subset_rng.permutation(len(accepted))))
        accepted = [accepted[i] for i in order]
    return accepted, attempts, seed, raw_digest, order


def _write_pool(out, rep, role, role_index, occupied):
    start = time.perf_counter()
    objects, proposals, seed, raw_digest, order = _sample(rep, role, role_index, occupied)
    path = out / "objects" / f"{rep['repeat_id']}_{role}_n10.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite object archive: {path}")
    np.savez_compressed(path, p=np.asarray(objects, dtype=np.int16))
    row = {**rep, "role": role, "role_index": role_index, "rng_seed": seed,
           "n": 10, "count": len(objects), "path": path.relative_to(out).as_posix(),
           "sha256": sha(path), "canonical_input_digest": json_hash(objects),
           "orbit_digest": json_hash([canonical_orbit_key(p) for p in objects]),
           "accepted_generation_order_digest": raw_digest,
           "proposals": proposals, "rejected": proposals - len(objects),
           "seconds": time.perf_counter() - start}
    if role == "target_train":
        row["subset_permutation"] = order
        row["nested_prefix_digests"] = {str(size): json_hash(objects[:size]) for size in (256, 512, 1024)}
    print(f"[data] {rep['repeat_id']} {role}: {len(objects)} accepted, {row['rejected']} rejected", flush=True)
    return row


def _source_bindings():
    paths = [Path(__file__), HERE / "design.py", ROOT / "permworld_platform/objects.py",
             ROOT / "permworld_platform/data.py", ROOT / "experiments/task_count_v2/data_pipeline.py"]
    return {_relative(path): sha(path) for path in paths}


def prepare(out=DEFAULT_OUT):
    """Create training/development objects and reserve capacity for all tests."""
    out = Path(out)
    if (out / "manifest.json").exists():
        verify(out, include_test=False, replay=False)
        return _read(out / "manifest.json")
    if (out / "objects").exists():
        raise FileExistsError("partial object generation exists; preserve it and choose a fresh directory")
    start = time.perf_counter()
    occupied, bindings, inventory = _history()
    capacity = _capacity(occupied)
    historical_counts = dict(sorted(Counter(map(len, occupied)).items()))
    # Store the explicit exclusion list before generating any new object.
    out.mkdir(parents=True, exist_ok=True)
    exclusions = out / "historical_orbit_exclusions.jsonl"
    if exclusions.exists():
        raise FileExistsError("partial exclusion record exists; preserve it and choose a fresh directory")
    with exclusions.open("w", encoding="utf8", newline="\n") as handle:
        for key in sorted(occupied, key=lambda p: (len(p), p)):
            handle.write(json.dumps(key, separators=(",", ":")) + "\n")
    write_json(out / "DATA_CONFIG.json", DATA_CONFIG)
    write_json(out / "capacity_audit.json", {"status": "pass", "created_at_utc": utc_now(), **capacity})
    write_json(out / "historical_inventory.json", {"files": inventory, "bindings": bindings})
    pools = [_write_pool(out, rep, role, index, occupied) for rep, role, index in _descriptors()]
    manifest = {"status": "complete", "created_at_utc": utc_now(), "config": DATA_CONFIG,
                "config_sha256": json_hash(DATA_CONFIG), "pools": pools,
                "history_bindings": bindings, "historical_inventory_sha256": sha(out / "historical_inventory.json"),
                "historical_orbits_by_length": historical_counts,
                "historical_exclusion_path": exclusions.name, "historical_exclusion_sha256": sha(exclusions),
                "capacity_audit": capacity, "test_generated": False,
                "source_bindings": _source_bindings(), "numpy_version": np.__version__,
                "seconds": time.perf_counter() - start}
    write_json(out / "manifest.json", manifest)
    return manifest


def _assert_manifest(out):
    manifest = _read(out / "manifest.json")
    if manifest["status"] != "complete" or manifest["config"] != DATA_CONFIG:
        raise ValueError("incomplete or incompatible primary data manifest")
    if manifest["config_sha256"] != json_hash(DATA_CONFIG):
        raise ValueError("data configuration hash mismatch")
    if manifest["source_bindings"] != _source_bindings():
        raise ValueError("frozen data-generation source changed; do not silently modify existing data")
    if _read(out / "DATA_CONFIG.json") != DATA_CONFIG:
        raise ValueError("standalone frozen data configuration changed")
    return manifest


def _pool_array(out, record):
    path = out / record["path"]
    if sha(path) != record["sha256"]:
        raise ValueError(f"object archive hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["p"]:
            raise ValueError("unexpected object archive fields")
        array = archive["p"].copy()
    if array.dtype != np.int16 or array.shape != (record["count"], 10):
        raise ValueError("object archive shape/dtype differs from frozen manifest")
    objects = [tuple(map(int, row)) for row in array]
    if json_hash(objects) != record["canonical_input_digest"]:
        raise ValueError("canonical input hash mismatch")
    if record["role"] == "target_train":
        order = record["subset_permutation"]
        if sorted(order) != list(range(1024)):
            raise ValueError("invalid frozen target ordering")
        raw = [objects[i] for i in np.argsort(order)]
        if json_hash(raw) != record["accepted_generation_order_digest"]:
            raise ValueError("target generation order cannot be recovered")
        for size in (256, 512, 1024):
            if json_hash(objects[:size]) != record["nested_prefix_digests"][str(size)]:
                raise ValueError("nested target subset hash mismatch")
    return array


def load_pool(out, repeat_id, role):
    """Load a hash-checked pool; take target_train[:size] for nested subsets."""
    out = Path(out)
    role = "test" if role == "target_test" else role
    manifest = _assert_manifest(out)
    records = manifest["pools"]
    if role == "test":
        test = _read(out / "test_manifest.json")
        if test["primary_manifest_sha256"] != sha(out / "manifest.json"):
            raise ValueError("test inputs do not bind to the primary data manifest")
        records = test["pools"]
    found = [r for r in records if (r["repeat_id"], r["role"]) == (repeat_id, role)]
    if len(found) != 1:
        raise ValueError(f"missing/duplicate object pool: {repeat_id} {role}")
    return _pool_array(out, found[0])


def _selection_gate(path):
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    value = _read(path)
    if value.get("status") != "frozen" or value.get("training_complete") is not True:
        raise ValueError("test generation requires status=frozen and training_complete=true")
    selections = value.get("selections", [])
    if len(selections) != 120:
        raise ValueError("all 120 development selections must freeze before any test generation")
    expected = {(r["repeat_id"], c, n) for r in REPEATS for c in ("P", "D", "T", "U", "A") for n in (256, 512, 1024)}
    actual = {(r["repeat_id"], r["condition"], r["target_samples"]) for r in selections}
    if actual != expected or len({r["run_id"] for r in selections}) != 120:
        raise ValueError("selection inventory must cover every prescribed unit/route/size exactly once")
    when = datetime.fromisoformat(value["frozen_at_utc"].replace("Z", "+00:00"))
    if when.tzinfo is None or when > datetime.now(timezone.utc):
        raise ValueError("selection freeze time must be timezone-aware and not in the future")
    try:
        location = path.relative_to(ROOT).as_posix()
    except ValueError:
        location = str(path)
    return {"path": location, "sha256": sha(path), "frozen_at_utc": value["frozen_at_utc"], "selection_count": 120}


def generate_test(out=DEFAULT_OUT, selection_manifest=None):
    """Generate independent test inputs only after the complete selection freeze."""
    if selection_manifest is None:
        raise ValueError("a complete frozen selection manifest is required")
    out = Path(out)
    gate = _selection_gate(selection_manifest)
    manifest = _assert_manifest(out)
    if (out / "test_manifest.json").exists():
        old = _read(out / "test_manifest.json")
        if old["selection_freeze"] != gate:
            raise ValueError("existing tests belong to a different selection freeze")
        verify(out, include_test=True, replay=False)
        return old
    if list((out / "objects").glob("*_test_*.npz")):
        raise FileExistsError("partial test generation exists; preserve it without overwriting")
    start = time.perf_counter()
    occupied, bindings, _ = _history()
    if bindings != manifest["history_bindings"]:
        raise ValueError("historical assets changed after pretest data freeze")
    for record in manifest["pools"]:
        for row in _pool_array(out, record):
            key = canonical_orbit_key(tuple(map(int, row)))
            if key in occupied:
                raise ValueError("orbit overlap discovered before test generation")
            occupied.add(key)
    pools = [_write_pool(out, rep, role, index, occupied)
             for rep, role, index in _descriptors(True) if role == "test"]
    result = {"status": "complete", "created_at_utc": utc_now(),
              "primary_manifest_sha256": sha(out / "manifest.json"),
              "selection_freeze": gate, "pools": pools, "seconds": time.perf_counter() - start,
              "test_use": "one unified final evaluation after all development choices; never used for selection"}
    write_json(out / "test_manifest.json", result)
    return result


def verify(out=DEFAULT_OUT, include_test=False, replay=True):
    """Verify frozen files, global orbit separation, and exact RNG/order replay."""
    out = Path(out)
    start = time.perf_counter()
    manifest = _assert_manifest(out)
    occupied, bindings, inventory = _history()
    if bindings != manifest["history_bindings"]:
        raise ValueError("historical assets/inventory changed after the data freeze")
    if sha(out / "historical_inventory.json") != manifest["historical_inventory_sha256"]:
        raise ValueError("historical inventory file changed")
    if _read(out / "historical_inventory.json") != {"files": inventory, "bindings": bindings}:
        raise ValueError("historical inventory does not match the actual files")
    exclusion = out / manifest["historical_exclusion_path"]
    if sha(exclusion) != manifest["historical_exclusion_sha256"]:
        raise ValueError("historical exclusion list changed")
    with exclusion.open(encoding="utf8") as handle:
        excluded_keys = {tuple(json.loads(line)) for line in handle}
    if excluded_keys != occupied:
        raise ValueError("explicit historical orbit exclusions are incomplete")
    capacity = _capacity(occupied)
    records = list(manifest["pools"])
    if include_test:
        test = _read(out / "test_manifest.json")
        if test["primary_manifest_sha256"] != sha(out / "manifest.json"):
            raise ValueError("test data primary-manifest binding changed")
        gate = _selection_gate(test["selection_freeze"]["path"])
        if gate != test["selection_freeze"]:
            raise ValueError("selection freeze changed after test generation")
        if datetime.fromisoformat(test["created_at_utc"]) < datetime.fromisoformat(gate["frozen_at_utc"].replace("Z", "+00:00")):
            raise ValueError("test generation predates selection freeze")
        records.extend(test["pools"])
    descriptors = list(_descriptors(include_test))
    if len(records) != len(descriptors):
        raise ValueError("missing or extra new pools")
    expected_paths = {out / r["path"] for r in records}
    actual_paths = set((out / "objects").glob("*.npz"))
    if not include_test:
        actual_paths = {p for p in actual_paths if "_test_" not in p.name}
    if expected_paths != actual_paths:
        raise ValueError("unbound or missing object archive")
    counts = Counter()
    for record, (rep, role, index) in zip(records, descriptors):
        if any(record[k] != value for k, value in rep.items()):
            raise ValueError("unit identity/seeds differ from frozen protocol")
        if (record["role"], record["role_index"], record["n"], record["count"]) != (role, index, 10, COUNTS[role]):
            raise ValueError("pool identity/order differs from frozen protocol")
        objects = [validate_permutation(tuple(map(int, row))) for row in _pool_array(out, record)]
        keys = [canonical_orbit_key(p) for p in objects]
        if len(set(keys)) != len(keys) or any(k in occupied for k in keys):
            raise ValueError("duplicate/overlapping orbit within a pool or across units/roles/history")
        if json_hash(keys) != record["orbit_digest"]:
            raise ValueError("orbit digest mismatch")
        if replay:
            reproduced, proposals, seed, raw_digest, order = _sample(rep, role, index, occupied)
            if reproduced != objects or proposals != record["proposals"] or seed != record["rng_seed"]:
                raise ValueError("frozen RNG stream does not reproduce objects/proposals")
            if raw_digest != record["accepted_generation_order_digest"]:
                raise ValueError("accepted generation order differs on replay")
            if role == "target_train" and order != record["subset_permutation"]:
                raise ValueError("nested target order differs on replay")
        else:
            occupied.update(keys)
        counts[role] += len(objects)
    result = {"status": "pass", "created_at_utc": utc_now(), "include_test": include_test,
              "rng_replayed": replay, "pool_count": len(records), "object_count": sum(counts.values()),
              "objects_by_role": dict(counts), "within_pool_duplicate_orbits": 0,
              "between_pool_overlapping_orbits": 0, "historical_overlapping_orbits": 0,
              "historical_npz_count": len(inventory), "history_binding_count": len(bindings),
              "capacity": capacity, "target_subsets": "verified nested randomized 256/512/1024 prefixes",
              "primary_manifest_sha256": sha(out / "manifest.json"),
              "test_manifest_sha256": sha(out / "test_manifest.json") if include_test else None,
              "seconds": time.perf_counter() - start}
    write_json(out / ("quality_with_test.json" if include_test else "quality_pretest.json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["capacity", "prepare", "verify", "generate-test"])
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
        occupied, bindings, inventory = _history()
        result = {"status": "pass", "capacity": _capacity(occupied),
                  "history_binding_count": len(bindings), "historical_npz_count": len(inventory)}
    print(json.dumps({k: v for k, v in result.items() if k not in {"pools", "history_bindings", "config"}}, indent=2))


if __name__ == "__main__":
    main()
