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

from experiments.fixed_points_deficiencies_arch_v1.plan import BASE_CFG as CFG, REPEATS, COUNTS, GATE, ORIGINAL, ROUTES, SIZES
from experiments.task_count_v2 import data_pipeline as archived_data
from permworld_platform.objects import canonical_orbit_key, validate_permutation

DEFAULT_OUT = HERE / "artifacts" / "data"
GATE_OUT = HERE / "artifacts" / "gate_data"
ROLES = ["source_train", "source_dev", "target_train", "target_dev"]
assert CFG["n"] == 10 and CFG["counts"] == COUNTS and len(REPEATS) == 8
assert [r["repeat_id"] for r in REPEATS] == [f"R{i}" for i in range(1, 9)]
for key in ("model_seed", "data_seed", "test_seed", "subset_seed"):
    assert len({r[key] for r in REPEATS}) == 8

DATA_CONFIG = {
    "version": "fixed_points_deficiencies_architecture_orbit_objects_v1",
    "n": 10, "repeats": REPEATS, "counts": COUNTS, "split_order": ROLES,
    "target_sizes": [256, 512, 1024],
    "generation_order": "separate gate train then dev; all formal pretest pools in repeat then role order; tests last in repeat order",
    "proposal": "uniform permutations; reject all previously occupied inverse/reverse/complement orbits",
    "rng": "numpy.PCG64(SeedSequence([data_seed, role_index, n])); test uses test_seed and role_index=4",
    "target_order": "accepted maximum pool reordered by PCG64(SeedSequence([subset_seed,n])).permutation(1024); nested prefixes",
    "history": "all archived DMLR permutation NPZs outside this experiment, including the original fixed_points_deficiencies_v1 40 pools; exposure-count archives are explicitly classified as non-object records",
    "capacity": "floor(n!/8) minus historically occupied n=10 orbits must cover gate train/dev and all eight formal units including deferred tests; never relax isolation",
    "test_gate": "120 new protocol/checkpoint-bound frozen selections after all 32 source and 360 adaptation runs complete",
    "gate_counts": {"train": GATE["train_count"], "dev": GATE["dev_count"]},
    "gate_data_seed": GATE["data_seed"],
    "pairing": "all conditions share role-specific inputs within a unit; only task definitions/labels differ",
    "sampling_limitations": "finite permutation space; uniform permutation proposals followed by orbit rejection are not iid uniform surviving orbits; global exclusion conditions later pools on earlier acceptance",
    "max_proposals_per_pool": 2000000,
}
GATE_DATA_CONFIG = {
    "version": "fixed_points_architecture_gate_objects_v1", "n": 10,
    "data_seed": GATE["data_seed"], "counts": DATA_CONFIG["gate_counts"],
    "roles": ["train", "dev"], "no_test_pool": True,
    "rng": "numpy.PCG64(SeedSequence([data_seed, role_index, 10]))",
    "isolation": "Complete historical and between-role inverse/reverse/complement exclusion; gate inputs also excluded from every formal role/unit",
    "capacity_reservation": "Gate train/dev and every formal pool including deferred tests reserved before gate generation",
}
GATE_REP = {"repeat_id": "GATE", "data_seed": GATE["data_seed"]}


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
    if not path.resolve().is_relative_to(HERE):
        raise ValueError("writes must remain inside this new experiment directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf8") == text:
            return
        raise FileExistsError(f"refusing to overwrite existing record: {path}")
    with path.open("x", encoding="utf8", newline="\n") as handle:
        handle.write(text)


def _new_output(out):
    out = Path(out).resolve()
    if not out.is_relative_to(HERE):
        raise ValueError("new data/audits must remain inside this experiment directory")
    return out


def _write_audit(out, name, result):
    path = out / name
    if path.exists():
        path = out / (Path(name).stem + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    write_json(path, result)


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

    # The previous study excluded its own directory; this replication must not.
    old_data = ORIGINAL / "artifacts" / "data"
    old_protocol = ORIGINAL / "artifacts" / "FROZEN_PROTOCOL.json"
    old_primary = _read(old_data / "manifest.json")
    old_tests = _read(old_data / "test_manifest.json")
    _bound_file(bindings, old_protocol)
    for name in ("DATA_CONFIG.json", "manifest.json", "test_manifest.json"):
        _bound_file(bindings, old_data / name)
    if _read(old_protocol)["data_manifest_sha256"] != sha(old_data / "manifest.json"):
        raise ValueError("original fixed-points protocol/data binding changed")
    if old_tests["primary_manifest_sha256"] != sha(old_data / "manifest.json"):
        raise ValueError("original fixed-points test/data binding changed")
    _check_historical_bindings(bindings, old_primary["history_bindings"])
    old_records = old_primary["pools"] + old_tests["pools"]
    expected = {old_data / r["path"] for r in old_records}
    if len(old_records) != 40 or expected != set((old_data / "objects").glob("*.npz")):
        raise ValueError("original fixed-points 40-pool inventory is incomplete or unbound")
    if {(r["repeat_id"], r["role"]) for r in old_records} != {
            (f"R{i}", role) for i in range(1, 9) for role in [*ROLES, "test"]}:
        raise ValueError("original fixed-points roles/units are incomplete")
    for record in old_records:
        _bound_file(bindings, old_data / record["path"], record["sha256"])

    # Bind and inspect all archived object files, including old engineering runs.
    # New experiment files are excluded here and checked as separate new pools.
    inventory, raw_seen = [], set()
    for path in sorted((ROOT / "experiments").rglob("*.npz")):
        if path.is_relative_to(HERE):
            continue
        _bound_file(bindings, path)
        with np.load(path, allow_pickle=False) as archive:
            fields = set(archive.files)
            if path.name == "input_exposure_counts.npz":
                if not path.is_relative_to(ORIGINAL) or archive.files != ["counts"]:
                    raise ValueError(f"unknown exposure archive provenance/schema: {path}")
                counts = archive["counts"]
                if counts.ndim not in (1, 2) or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
                    raise ValueError(f"invalid non-object exposure archive: {path}")
                inventory.append({"path": _relative(path), "sha256": bindings[_relative(path)],
                    "classification": "non_object_exposure_counts", "shape": list(counts.shape),
                    "dtype": str(counts.dtype), "excluded_from_orbit_counts": True})
                continue
            known_object_schema = fields in ({"p"}, {"p", "y"})
            feature_path = ROOT / "experiments/permworld_pilot/runs/pilot_v1/cka_interface_smoke.npz"
            known_feature_schema = path == feature_path and fields == {
                "p", "labels", "input_end_index", "layer1", "layer2", "layer3", "layer4", "final_norm"}
            if not known_object_schema and not known_feature_schema:
                raise ValueError(f"unclassified historical NPZ schema {sorted(fields)}: {path}")
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
                              "classification": "permutation_pool", "archive_fields": sorted(fields),
                              "count": len(array), "n": array.shape[1]})
    if sum(r["classification"] == "permutation_pool" for r in inventory) < 219:
        raise ValueError("historical inventory is smaller than 179 prior plus 40 original fixed-points object archives")
    if sum(r["classification"] == "non_object_exposure_counts" for r in inventory) != 395:
        raise ValueError("original fixed-points exposure archive inventory differs from the inspected 395 files")
    return occupied, dict(sorted(bindings.items())), inventory


def _capacity(occupied):
    historical = sum(len(key) == 10 for key in occupied)
    lower = math.factorial(10) // 8
    requested = len(REPEATS) * sum(COUNTS.values())
    gate_requested = GATE["train_count"] + GATE["dev_count"]
    total_requested = requested + gate_requested
    result = {"n": 10, "permutations": math.factorial(10), "maximum_orbit_size": 8,
              "orbit_count_lower_bound": lower, "historical_occupied_orbits": historical,
              "available_orbit_lower_bound": lower - historical,
              "requested_new_orbits_all_roles_and_units": requested,
              "requested_gate_orbits": gate_requested,
              "requested_new_orbits_including_gate": total_requested,
              "reserved_future_test_orbits": len(REPEATS) * COUNTS["test"],
              "conservative_capacity_sufficient": lower - historical >= total_requested}
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
    count = GATE_DATA_CONFIG["counts"][role] if role in ("train", "dev") else COUNTS[role]
    while len(accepted) < count:
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
    paths = [Path(__file__), HERE / "plan.py", ORIGINAL / "design.py", ROOT / "permworld_platform/objects.py",
             ROOT / "permworld_platform/data.py", ROOT / "experiments/task_count_v2/data_pipeline.py"]
    return {_relative(path): sha(path) for path in paths}


def _write_history(out, occupied, bindings, inventory, capacity):
    out = _new_output(out)
    out.mkdir(parents=True, exist_ok=True)
    exclusions = out / "historical_orbit_exclusions.jsonl"
    with exclusions.open("x", encoding="utf8", newline="\n") as handle:
        for key in sorted(occupied, key=lambda p: (len(p), p)):
            handle.write(json.dumps(key, separators=(",", ":")) + "\n")
    write_json(out / "capacity_audit.json", {"status": "pass", "created_at_utc": utc_now(), **capacity})
    write_json(out / "historical_inventory.json", {"files": inventory, "bindings": bindings})
    return exclusions


def prepare_gate(out=GATE_OUT):
    """Create the development-only gate objects after reserving the whole study."""
    out = _new_output(out)
    if (out / "manifest.json").exists():
        verify_gate(out, replay=False)
        return _read(out / "manifest.json")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("partial gate data exist; preserve them without overwriting")
    start = time.perf_counter()
    occupied, bindings, inventory = _history()
    capacity = _capacity(occupied)
    historical_counts = dict(sorted(Counter(map(len, occupied)).items()))
    exclusions = _write_history(out, occupied, bindings, inventory, capacity)
    write_json(out / "DATA_CONFIG.json", GATE_DATA_CONFIG)
    pools = [_write_pool(out, GATE_REP, role, index, occupied)
             for index, role in enumerate(GATE_DATA_CONFIG["roles"])]
    result = {"status": "complete", "created_at_utc": utc_now(), "config": GATE_DATA_CONFIG,
        "config_sha256": json_hash(GATE_DATA_CONFIG), "pools": pools,
        "history_bindings": bindings, "historical_inventory_sha256": sha(out / "historical_inventory.json"),
        "historical_orbits_by_length": historical_counts,
        "historical_exclusion_path": exclusions.name, "historical_exclusion_sha256": sha(exclusions),
        "capacity_audit": capacity, "test_generated": False, "test_allowed": False,
        "source_bindings": _source_bindings(), "numpy_version": np.__version__,
        "seconds": time.perf_counter() - start}
    write_json(out / "manifest.json", result)
    return result


def _assert_gate_manifest(out=GATE_OUT):
    out = Path(out)
    value = _read(out / "manifest.json")
    if value.get("status") != "complete" or value["config"] != GATE_DATA_CONFIG:
        raise ValueError("incomplete or incompatible gate data")
    if value["config_sha256"] != json_hash(GATE_DATA_CONFIG) or value["source_bindings"] != _source_bindings():
        raise ValueError("gate configuration or generation code changed")
    if _read(out / "DATA_CONFIG.json") != GATE_DATA_CONFIG or value.get("test_allowed") is not False:
        raise ValueError("gate configuration/test policy changed")
    expected = [("train", 0, GATE["train_count"]), ("dev", 1, GATE["dev_count"])]
    if [(r["role"], r["role_index"], r["count"]) for r in value["pools"]] != expected:
        raise ValueError("gate must contain exactly its prescribed train and development pools")
    if {out / r["path"] for r in value["pools"]} != set((out / "objects").glob("*.npz")):
        raise ValueError("gate has missing or unbound object archives")
    return value


def load_gate_pool(role, out=GATE_OUT):
    if role not in ("train", "dev"):
        raise ValueError("the architecture gate has only train/dev pools")
    out = Path(out)
    manifest = _assert_gate_manifest(out)
    return _pool_array(out, next(r for r in manifest["pools"] if r["role"] == role))


def _verify_history_record(out, manifest, occupied, bindings, inventory):
    if bindings != manifest["history_bindings"]:
        raise ValueError("historical assets/inventory changed after freeze")
    if sha(out / "historical_inventory.json") != manifest["historical_inventory_sha256"]:
        raise ValueError("historical inventory file changed")
    if _read(out / "historical_inventory.json") != {"files": inventory, "bindings": bindings}:
        raise ValueError("historical inventory differs from actual files")
    exclusion = out / manifest["historical_exclusion_path"]
    if sha(exclusion) != manifest["historical_exclusion_sha256"]:
        raise ValueError("historical exclusion list changed")
    with exclusion.open(encoding="utf8") as handle:
        excluded = {tuple(json.loads(line)) for line in handle}
    if excluded != occupied:
        raise ValueError("historical exclusion list is incomplete")


def _add_gate(occupied, bindings, inventory, out=GATE_OUT, replay=False):
    """Verify the gate against history, then add its objects to formal exclusions."""
    out = Path(out)
    manifest = _assert_gate_manifest(out)
    _verify_history_record(out, manifest, occupied, bindings, inventory)
    for row in manifest["pools"]:
        if row["repeat_id"] != "GATE" or row["data_seed"] != GATE["data_seed"] or row["n"] != 10:
            raise ValueError("gate input identity differs from the frozen plan")
        objects = [validate_permutation(tuple(map(int, p))) for p in _pool_array(out, row)]
        keys = [canonical_orbit_key(p) for p in objects]
        if len(keys) != len(set(keys)) or any(key in occupied for key in keys):
            raise ValueError("gate orbit duplicates or historical/train-development overlap")
        if json_hash(keys) != row["orbit_digest"]:
            raise ValueError("gate orbit digest changed")
        if replay:
            expected, attempts, seed, raw_digest, _ = _sample(GATE_REP, row["role"], row["role_index"], occupied)
            if (expected, attempts, seed, raw_digest) != (objects, row["proposals"], row["rng_seed"], row["accepted_generation_order_digest"]):
                raise ValueError("gate objects/order/proposals differ from exact RNG replay")
        else:
            occupied.update(keys)
    for name in ("manifest.json", "DATA_CONFIG.json", "historical_inventory.json", "historical_orbit_exclusions.jsonl", "capacity_audit.json"):
        _bound_file(bindings, out / name)
    for row in manifest["pools"]:
        _bound_file(bindings, out / row["path"], row["sha256"])
        inventory.append({"path": _relative(out / row["path"]), "sha256": row["sha256"],
            "classification": "separate_architecture_gate_pool", "count": row["count"], "n": 10})
    return manifest


def verify_gate(out=GATE_OUT, replay=True):
    out = _new_output(out)
    start = time.perf_counter()
    occupied, bindings, inventory = _history()
    capacity = _capacity(occupied)
    manifest = _add_gate(occupied, bindings, inventory, out, replay)
    result = {"status": "pass", "created_at_utc": utc_now(), "rng_replayed": replay,
        "pool_count": 2, "object_count": GATE["train_count"] + GATE["dev_count"],
        "historical_overlapping_orbits": 0, "between_pool_overlapping_orbits": 0,
        "within_pool_duplicate_orbits": 0, "test_pool_count": 0,
        "capacity": capacity, "gate_manifest_sha256": sha(out / "manifest.json"),
        "seconds": time.perf_counter() - start}
    if manifest["capacity_audit"] != capacity:
        raise ValueError("capacity reservation changed")
    _write_audit(out, "quality_gate.json", result)
    return result


def prepare(out=DEFAULT_OUT):
    """Create training/development objects and reserve capacity for all tests."""
    out = _new_output(out)
    if (out / "manifest.json").exists():
        verify(out, include_test=False, replay=False)
        return _read(out / "manifest.json")
    if (out / "objects").exists():
        raise FileExistsError("partial object generation exists; preserve it and choose a fresh directory")
    start = time.perf_counter()
    occupied, bindings, inventory = _history()
    capacity = _capacity(occupied)
    _add_gate(occupied, bindings, inventory)
    historical_counts = dict(sorted(Counter(map(len, occupied)).items()))
    # Store the explicit exclusion list before generating any new object.
    out.mkdir(parents=True, exist_ok=True)
    exclusions = out / "historical_orbit_exclusions.jsonl"
    if exclusions.exists():
        raise FileExistsError("partial exclusion record exists; preserve it and choose a fresh directory")
    with exclusions.open("x", encoding="utf8", newline="\n") as handle:
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
                "gate_manifest_path": _relative(GATE_OUT / "manifest.json"),
                "gate_manifest_sha256": sha(GATE_OUT / "manifest.json"),
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
    if manifest["gate_manifest_path"] != _relative(GATE_OUT / "manifest.json") or manifest["gate_manifest_sha256"] != sha(GATE_OUT / "manifest.json"):
        raise ValueError("formal data do not bind to the verified separate gate pools")
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
    artifacts = HERE / "artifacts"
    if path.resolve() != (artifacts / "SELECTIONS_FROZEN.json").resolve():
        raise ValueError("test generation requires this experiment's canonical new selection manifest")
    # This import is intentionally deferred until after the architecture decision
    # and formal registration exist. It also checks gate-result/protocol bindings.
    from experiments.fixed_points_deficiencies_arch_v1.protocol import verify as verify_formal_protocol
    verify_formal_protocol()
    if value.get("status") != "frozen" or value.get("training_complete") is not True:
        raise ValueError("test generation requires status=frozen and training_complete=true")
    selections = value.get("selections", [])
    if len(selections) != 120:
        raise ValueError("all 120 development selections must freeze before any test generation")
    expected = {(r["repeat_id"], c, n) for r in REPEATS for c in ("P", "D", "T", "U", "A") for n in (256, 512, 1024)}
    actual = {(r["repeat_id"], r["condition"], r["target_samples"]) for r in selections}
    if actual != expected or len({r["run_id"] for r in selections}) != 120:
        raise ValueError("selection inventory must cover every prescribed unit/route/size exactly once")
    protocol_path = artifacts / "FROZEN_PROTOCOL.json"
    protocol = _read(protocol_path)
    protocol_hash = sha(protocol_path)
    if protocol.get("status") != "frozen" or value.get("protocol_sha256") != protocol_hash:
        raise ValueError("selection does not bind to this experiment's frozen architecture protocol")
    config = protocol["config"]
    for name in ("n", "repeats", "sources", "target_sizes", "counts", "pretrain_updates", "adaptation_updates", "learning_rates"):
        if config[name] != CFG[name]:
            raise ValueError(f"formal protocol differs from the architecture-independent plan: {name}")
    for rel, digest in protocol["code_sha256"].items():
        if sha(ROOT / rel) != digest:
            raise ValueError(f"frozen formal source changed: {rel}")
    if protocol["data_manifest_sha256"] != sha(DEFAULT_OUT / "manifest.json"):
        raise ValueError("formal protocol/data binding changed")
    if protocol["initial_manifest_sha256"] != sha(artifacts / "INITIALS.json"):
        raise ValueError("formal protocol/initialization binding changed")
    manifest = _assert_manifest(DEFAULT_OUT)
    initials = _read(artifacts / "INITIALS.json")["initials"]
    if len(initials) != 8 or {r["repeat_id"] for r in initials} != {r["repeat_id"] for r in REPEATS}:
        raise ValueError("new initialization inventory is incomplete")
    initial_by_id = {r["repeat_id"]: r for r in initials}
    records = {(r["repeat_id"], r["role"]): r for r in manifest["pools"]}
    hashes, inputs = {}, {}

    def checked_hash(checkpoint, expected_hash):
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_relative_to(artifacts):
            raise ValueError("checkpoint binding escapes the new experiment")
        if checkpoint not in hashes:
            hashes[checkpoint] = sha(checkpoint)
        if hashes[checkpoint] != expected_hash:
            raise ValueError(f"checkpoint changed: {checkpoint}")

    def input_hash(unit, role, size=None):
        key = (unit, role, size)
        if key not in inputs:
            array = _pool_array(DEFAULT_OUT, records[(unit, role)])
            if size is not None:
                array = array[:size]
            inputs[key] = hashlib.sha256(np.asarray(array, dtype="<i2").tobytes()).hexdigest()
        return inputs[key]

    selected_by_id = {(r["repeat_id"], r["condition"], r["target_samples"]): r for r in selections}
    source_paths, adapt_paths, completion_hashes = set(), set(), {}
    latest_completion = datetime.min.replace(tzinfo=timezone.utc)
    for rep in REPEATS:
        unit = rep["repeat_id"]
        initial = initial_by_id[unit]
        if any(initial.get(key) != number for key, number in rep.items()):
            raise ValueError("new initialization seeds do not match the frozen paired unit")
        checked_hash(artifacts / initial["path"], initial["sha256"])
        sources, source_streams, target_row_hashes = {}, set(), set()
        for condition in ROUTES[:-1]:
            folder = artifacts / "pretrain" / f"{unit}_{condition}"
            completed = folder / "completed.json"
            source_paths.add(completed)
            row = _read(completed)
            completion_hashes[_relative(completed)] = sha(completed)
            if (row.get("status"), row.get("protocol_sha256"), row.get("updates")) != ("complete", protocol_hash, CFG["pretrain_updates"]):
                raise ValueError("all 32 new source endpoints must complete under the frozen protocol")
            if (row["repeat_id"], row["condition"], row["run_id"], row["model_seed"], row["data_seed"], row["sampling_seed"], row["dropout_seed"]) != (
                    unit, condition, folder.name, rep["model_seed"], rep["data_seed"], rep["source_sampling_seed"], rep["source_dropout_seed"]):
                raise ValueError("source identity or seed binding changed")
            if row["tasks"] != CFG["sources"][condition] or row["initial_sha256"] != initial["sha256"] or row["initial_model_sha256"] != initial["model_sha256"]:
                raise ValueError("source composition or initialization binding changed")
            if row["source_input_sha256"] != input_hash(unit, "source_train") or row["source_dev_input_sha256"] != input_hash(unit, "source_dev"):
                raise ValueError("source input binding changed")
            checked_hash(folder / "endpoint.pt", row["endpoint_sha256"])
            sources[condition] = row
            source_streams.add(row["stream_sha256"])
            latest_completion = max(latest_completion, datetime.fromisoformat(row["finished_at_utc"].replace("Z", "+00:00")))
        if len(source_streams) != 1:
            raise ValueError("source conditions did not share the paired sampling stream")
        streams_by_size = {size: set() for size in SIZES}
        for condition in ROUTES:
            for size in SIZES:
                candidates = []
                for lr_index, lr in enumerate(CFG["learning_rates"]):
                    run_id = f"{unit}_{condition}_n{size}_lr{lr_index}"
                    folder = artifacts / "adapt" / run_id
                    completed = folder / "completed.json"
                    adapt_paths.add(completed)
                    row = _read(completed)
                    completion_hashes[_relative(completed)] = sha(completed)
                    if (row.get("status"), row.get("protocol_sha256"), row.get("updates")) != ("complete", protocol_hash, CFG["adaptation_updates"]):
                        raise ValueError("all 360 new adaptation candidates must complete before tests")
                    identity = (unit, condition, size, run_id, lr_index, lr, rep["model_seed"], rep["data_seed"], rep["adapt_sampling_seed"], rep["adapt_dropout_seed"])
                    if tuple(row[k] for k in ("repeat_id", "condition", "target_samples", "run_id", "lr_index", "lr", "model_seed", "data_seed", "sampling_seed", "dropout_seed")) != identity:
                        raise ValueError("adaptation identity or seed binding changed")
                    if row["train_input_sha256"] != input_hash(unit, "target_train", size) or row["dev_input_sha256"] != input_hash(unit, "target_dev"):
                        raise ValueError("adaptation data binding changed")
                    audit = row["load_audit"]
                    source_rel = initial["path"] if condition == "A" else f"pretrain/{unit}_{condition}/endpoint.pt"
                    source_hash = initial["sha256"] if condition == "A" else sources[condition]["endpoint_sha256"]
                    model_hash = initial["model_sha256"] if condition == "A" else sources[condition]["endpoint_model_sha256"]
                    if (audit["source_checkpoint"], audit["source_checkpoint_sha256"], audit["source_model_sha256"], audit["loaded_model_sha256"], audit["initial_sha256"]) != (source_rel, source_hash, model_hash, model_hash, initial["sha256"]):
                        raise ValueError("adaptation source/model/initialization binding changed")
                    if any(audit.get(k) is not True for k in ("strict_complete_load", "target_row_matches_step0", "only_target_tied_row_reset", "tied_weights", "full_parameter_training")) or audit.get("fresh_optimizer_state_entries") != 0:
                        raise ValueError("adaptation load/reset/optimizer audit failed")
                    target_row_hashes.add(audit["target_row_sha256"])
                    checked_hash(folder / "best.pt", row["best_sha256"])
                    checked_hash(folder / "endpoint.pt", row["endpoint_sha256"])
                    metric = row["best"]["metrics"]
                    expected_key = [-metric["accuracy"], metric["answer_ce"], row["best"]["step"]]
                    if row["best"]["selection_key"] != expected_key:
                        raise ValueError("candidate development selection key changed")
                    streams_by_size[size].add(row["stream_sha256"])
                    candidates.append(row)
                    latest_completion = max(latest_completion, datetime.fromisoformat(row["finished_at_utc"].replace("Z", "+00:00")))
                winner = min(candidates, key=lambda r: (*r["best"]["selection_key"], r["lr_index"]))
                selected = selected_by_id[(unit, condition, size)]
                for key in ("run_id", "lr", "lr_index"):
                    if selected[key] != winner[key]:
                        raise ValueError("frozen selection is not the prescribed development winner")
                folder_rel = f"adapt/{winner['run_id']}"
                expected_selected = (folder_rel + "/best.pt", winner["best_sha256"], folder_rel + "/endpoint.pt", winner["endpoint_sha256"], winner["best"]["step"], winner["best"]["metrics"]["accuracy"], winner["best"]["metrics"]["answer_ce"])
                if tuple(selected[k] for k in ("selected_path", "selected_sha256", "endpoint_path", "endpoint_sha256", "step", "dev_accuracy", "dev_answer_ce")) != expected_selected:
                    raise ValueError("selected checkpoint or metric binding changed")
        if len(target_row_hashes) != 1 or any(len(v) != 1 for v in streams_by_size.values()):
            raise ValueError("adaptation routes do not share target-row initialization and paired data streams")
    if source_paths != set((artifacts / "pretrain").glob("*/completed.json")) or adapt_paths != set((artifacts / "adapt").glob("*/completed.json")):
        raise ValueError("missing or extra formal training completions")
    if value.get("candidate_runs") != 360:
        raise ValueError("selection candidate inventory is incomplete")
    when = datetime.fromisoformat(value["frozen_at_utc"].replace("Z", "+00:00"))
    if when.tzinfo is None or when > datetime.now(timezone.utc):
        raise ValueError("selection freeze time must be timezone-aware and not in the future")
    if when < latest_completion:
        raise ValueError("selection freeze predates a required training completion")
    try:
        location = path.relative_to(ROOT).as_posix()
    except ValueError:
        location = str(path)
    return {"path": location, "sha256": sha(path), "frozen_at_utc": value["frozen_at_utc"],
        "selection_count": 120, "protocol_sha256": protocol_hash,
        "source_completions": len(source_paths), "adaptation_completions": len(adapt_paths),
        "completion_bindings_sha256": json_hash(completion_hashes), "checkpoint_files_hash_checked": len(hashes)}


def generate_test(out=DEFAULT_OUT, selection_manifest=None):
    """Generate independent test inputs only after the complete selection freeze."""
    if selection_manifest is None:
        raise ValueError("a complete frozen selection manifest is required")
    out = _new_output(out)
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
    # The full archived inventory is needed to verify the frozen gate binding.
    occupied, bindings, gate_inventory = _history()
    _add_gate(occupied, bindings, gate_inventory)
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
    out = _new_output(out)
    start = time.perf_counter()
    manifest = _assert_manifest(out)
    occupied, bindings, inventory = _history()
    capacity = _capacity(occupied)
    _add_gate(occupied, bindings, inventory)
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
    if manifest["capacity_audit"] != capacity:
        raise ValueError("full-study capacity reservation changed")
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
    _write_audit(out, "quality_with_test.json" if include_test else "quality_pretest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["capacity", "prepare-gate", "verify-gate", "prepare", "verify", "generate-test"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gate-out", type=Path, default=GATE_OUT)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument("--selection-manifest", type=Path)
    args = parser.parse_args()
    if args.command == "prepare-gate":
        result = prepare_gate(args.gate_out)
    elif args.command == "verify-gate":
        result = verify_gate(args.gate_out, not args.no_replay)
    elif args.command == "prepare":
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
