"""Reproducible object-first sampling, exact labels, and immutable data manifests."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .objects import canonical_orbit_key, validate_permutation
from .registry import list_task_instances, instance_id, task_spec, task_output_bounds
from .tasks import label
from .encoding import encode_example, vocabulary_manifest

ROOT = Path(__file__).resolve().parents[1]
DATA_VERSION = "permworld_objects_labels_v1"
DEFAULT_CONFIG = dict(
    version=DATA_VERSION, lengths=[8, 10, 12],
    replicates=[dict(model_seed=617, data_seed=2026102001), dict(model_seed=829, data_seed=2026102002)],
    per_length_counts=dict(train=1024, dev=256, test=256),
    proposal="uniform permutation conditional on length; sequential rejection of occupied I/R/C orbits",
    split_order=["train", "dev", "test"],
    historical_exclusion=True, max_proposals_per_pool=2000000,
    test_use="reserved platform holdout; not evaluated or used to select calibration tasks/configuration",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def historical_inputs():
    """Read only the previous pilot object archives, never their target outcomes."""
    seen, files, raw_seen = set(), {}, set()
    dirs = ["permworld_pilot/runs/pilot_v1", "pilot_v2_diagnostics/artifacts",
            "pilot_v3_multitarget/artifacts", "pilot_v4_backgrounds/artifacts",
            "pilot_v5_confirmation/artifacts"]
    for folder in dirs:
        for path in sorted((ROOT / "experiments" / folder).rglob("*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                if "p" not in archive or archive["p"].ndim != 2:
                    continue
                for row in archive["p"]:
                    p = tuple(map(int, row))
                    if p not in raw_seen:
                        raw_seen.add(p)
                        seen.add(canonical_orbit_key(p))
            files[path.relative_to(ROOT).as_posix()] = sha(path)
    return seen, files


def pool_path(out, seed, role, n):
    return Path(out) / "data" / f"s{seed}_{role}_n{n}.npz"


def load_objects(dataset_dir, seed, role):
    manifest = json.loads((Path(dataset_dir) / "manifest.json").read_text(encoding="utf8"))
    if role not in manifest["config"]["split_order"]:
        raise ValueError(f"unknown split {role}")
    if seed not in [r["model_seed"] for r in manifest["config"]["replicates"]]:
        raise ValueError(f"unknown replicate {seed}")
    values = []
    for n in manifest["config"]["lengths"]:
        with np.load(pool_path(dataset_dir, seed, role, n), allow_pickle=False) as archive:
            values.extend(tuple(map(int, p)) for p in archive["p"])
    return values


def _normal(value):
    return json.loads(json.dumps(value))


def read_samples(dataset_dir, seed, role, task_id, params=None, *, max_seq_len=128):
    """Read materialized labels and return model-ready samples after exact checks."""
    out = Path(dataset_dir)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf8"))
    iid = instance_id(task_id, params or {})
    if iid not in {i["instance_id"] for i in manifest["instances"]}:
        raise ValueError(f"{iid} is not materialized in this dataset; explicitly build a new configuration")
    for n in manifest["config"]["lengths"]:
        path = pool_path(out, seed, role, n)
        spec = next((r for r in manifest["pools"] if (r["seed"], r["role"], r["n"]) == (seed, role, n)), None)
        if spec is None:
            raise ValueError("unknown seed/split/length in frozen dataset")
        if sha(path) != spec["object_sha256"] or sha(path.with_suffix(".labels.jsonl")) != spec["label_sha256"]:
            raise ValueError("dataset object/label bytes differ from manifest")
        objects = np.load(path, allow_pickle=False)["p"]
        records = [json.loads(line) for line in path.with_suffix(".labels.jsonl").read_text(encoding="utf8").splitlines()]
        if len(records) != len(objects):
            raise ValueError("object/label row count mismatch")
        for p_array, row in zip(objects, records):
            p = tuple(map(int, p_array))
            if row["input_id"] != json_hash(p):
                raise ValueError("object/label identity mismatch")
            result = encode_example(task_id, p, params, max_seq_len=max_seq_len)
            if _normal(result["answer_value"]) != row["answers"][iid]:
                raise ValueError("materialized label does not match the current exact oracle")
            yield result


def generate_dataset(out, config=None):
    out = Path(out)
    config = _normal(config or DEFAULT_CONFIG)
    if (out / "manifest.json").exists():
        old_manifest = json.loads((out / "manifest.json").read_text(encoding="utf8"))
        if old_manifest["config"] != config:
            raise ValueError("existing immutable dataset has another config; use a new --out")
        verify_dataset(out, recompute_labels=False)
        return old_manifest
    if (out / "data").exists():
        raise ValueError("partial output already exists; retain it and choose a new --out")
    if config["version"] != DATA_VERSION or config["split_order"] != ["train", "dev", "test"]:
        raise ValueError("unsupported data version or split order")
    if len(set(config["lengths"])) != len(config["lengths"]):
        raise ValueError("duplicate lengths")
    for n in config["lengths"]:
        validate_permutation(tuple(range(1, n + 1)))
    if len({r["model_seed"] for r in config["replicates"]}) != len(config["replicates"]):
        raise ValueError("duplicate model-seed replicate IDs")
    if len({r["data_seed"] for r in config["replicates"]}) != len(config["replicates"]):
        raise ValueError("data RNG seeds must differ")
    if any(type(c) is not int or c <= 0 for c in config["per_length_counts"].values()):
        raise ValueError("split counts must be positive integers")
    requested_per_length = sum(config["per_length_counts"].values()) * len(config["replicates"])
    if any(requested_per_length > math.factorial(n) for n in config["lengths"]):
        raise ValueError("requested unique objects exceed raw permutation space, even before orbit exclusion")
    instances = list_task_instances()
    if "task_instances" in config:
        instances = []
        for task in config["task_instances"]:
            spec = task_spec(task["task_id"])
            if spec["input_arity"] != 1:
                raise ValueError("binary tasks require a pair-aware generator; v0.1 only materializes unary pools")
            params = task.get("params", {})
            instances.append(dict(instance_id=instance_id(task["task_id"], params), task_id=task["task_id"],
                params=params, input_arity=1, family=spec["family"], focus_template=spec["focus_template"],
                identity_kind=spec["identity_kind"], output_type=spec["output_type"], output_encoding=spec["output_encoding"]))
    if not instances or len({i["instance_id"] for i in instances}) != len(instances):
        raise ValueError("task instance list must be nonempty and unique")
    for inst in instances:
        for n in config["lengths"]:
            try:
                task_output_bounds(inst["task_id"], n, inst["params"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"unsupported configured instance {inst['instance_id']} at n={n}: {exc}") from exc
    start = time.perf_counter()
    seen, history = historical_inputs() if config["historical_exclusion"] else (set(), {})
    history_count = len(seen)
    # This release materializes unary tasks; binary tasks remain executable via explicit operands.
    instances = [i for i in instances if i.get("input_arity", 1) == 1]
    manifests, distribution, timings = [], [], []
    vocab = vocabulary_manifest()
    write_json(out / "vocabulary.json", vocab)
    for rep in config["replicates"]:
        for role_index, role in enumerate(config["split_order"]):
            for n in config["lengths"]:
                rng = np.random.default_rng(np.random.SeedSequence([rep["data_seed"], role_index, n]))
                count = config["per_length_counts"][role]
                accepted, attempts = [], 0
                while len(accepted) < count:
                    attempts += 1
                    if attempts > config["max_proposals_per_pool"]:
                        raise RuntimeError(f"orbit capacity/proposal limit reached for seed{rep['model_seed']} {role} n{n}; no silent overlap")
                    p = tuple(map(int, rng.permutation(n) + 1))
                    key = canonical_orbit_key(p)
                    if key in seen:
                        continue
                    seen.add(key)
                    accepted.append(p)
                path = pool_path(out, rep["model_seed"], role, n)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, p=np.asarray(accepted, dtype=np.int16))
                rows = [dict(input_id=json_hash(p), answers={}) for p in accepted]
                for inst in instances:
                    iid, task_id, params = inst["instance_id"], inst["task_id"], inst["params"]
                    t0 = time.perf_counter()
                    answers = [label(task_id, p, params) for p in accepted]
                    elapsed = time.perf_counter() - t0
                    for row, value in zip(rows, answers):
                        row["answers"][iid] = value
                    histogram = Counter(json.dumps(v, separators=(",", ":")) for v in answers)
                    distribution.append(dict(seed=rep["model_seed"], data_seed=rep["data_seed"], role=role,
                        n=n, instance_id=iid, task_id=task_id, count=count, distinct_answers=len(histogram),
                        observed_constant=len(histogram) == 1, largest_class_fraction=max(histogram.values()) / count,
                        answer_histogram=json.dumps(dict(sorted(histogram.items())), separators=(",", ":"))))
                    timings.append(dict(seed=rep["model_seed"], role=role, n=n, instance_id=iid,
                        count=count, total_seconds=elapsed, microseconds_per_label=elapsed * 1e6 / count))
                labels_path = path.with_suffix(".labels.jsonl")
                with labels_path.open("w", encoding="utf8", newline="\n") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                record = dict(seed=rep["model_seed"], data_seed=rep["data_seed"], role=role, n=n,
                    count=count, proposals=attempts, accepted=count, rejected=attempts-count,
                    rng="numpy.PCG64(SeedSequence([data_seed,role_index,n]))",
                    object_path=path.relative_to(out).as_posix(), object_sha256=sha(path),
                    canonical_object_content_sha256=json_hash(accepted),
                    label_path=labels_path.relative_to(out).as_posix(), label_sha256=sha(labels_path),
                    unique_inputs=count, unique_orbits=count)
                manifests.append(record)
                print(f"[data] seed{rep['model_seed']} {role} n{n}: {count} objects, {len(instances)} label instances", flush=True)
    # Train-derived majority labels, never a mode fitted to dev/test.
    train_modes = {}
    for row in distribution:
        if row["role"] == "train":
            hist = json.loads(row["answer_histogram"])
            train_modes[(row["seed"], row["n"], row["instance_id"])] = min(hist, key=lambda k: (-hist[k], k))
    for row in distribution:
        mode = train_modes[(row["seed"], row["n"], row["instance_id"])]
        row["train_majority_answer"] = mode
        row["train_majority_accuracy"] = json.loads(row["answer_histogram"]).get(mode, 0) / row["count"]
    write_csv(out / "answer_distributions.csv", distribution)
    write_csv(out / "generation_timing.csv", timings)
    metadata = dict(version=DATA_VERSION, config=config, instances=instances, pools=manifests,
        historical_files=history, historical_unique_orbits=history_count,
        vocabulary_sha256=sha(out / "vocabulary.json"), generation_seconds=time.perf_counter()-start,
        generated_unix=time.time(), numpy_version=np.__version__,
        relation_isolation="full inverse/reverse/complement orbit across roles, lengths implicitly, and replicates; historical pools excluded",
        sampling_limitations="uniform proposals with sequential orbit rejection; not uniform orbit classes; no replacement induces finite-population dependence",
        binary_dataset_policy="no materialized binary pools; explicit pair input interface supported; not claiming pair-group split certification")
    write_json(out / "manifest.json", metadata)
    verify_dataset(out, recompute_labels=False)
    return metadata


def verify_dataset(out, *, recompute_labels=True):
    out = Path(out)
    meta = json.loads((out / "manifest.json").read_text(encoding="utf8"))
    seen, history_files = historical_inputs() if meta["config"]["historical_exclusion"] else (set(), {})
    if history_files != meta["historical_files"]:
        raise ValueError("historical object archives changed")
    count = checks = 0
    for row in meta["pools"]:
        path = out / row["object_path"]
        labels_path = out / row["label_path"]
        if sha(path) != row["object_sha256"] or sha(labels_path) != row["label_sha256"]:
            raise ValueError("dataset hash mismatch")
        objects = np.load(path, allow_pickle=False)["p"]
        records = [json.loads(s) for s in labels_path.read_text(encoding="utf8").splitlines()]
        if len(objects) != row["count"] or len(records) != len(objects):
            raise ValueError("pool count mismatch")
        if objects.shape != (row["count"], row["n"]):
            raise ValueError("pool input length mismatch")
        role_index = meta["config"]["split_order"].index(row["role"])
        rng = np.random.default_rng(np.random.SeedSequence([row["data_seed"], role_index, row["n"]]))
        replayed, attempts, replay_keys = [], 0, set()
        while len(replayed) < row["count"]:
            attempts += 1
            if attempts > meta["config"]["max_proposals_per_pool"]:
                raise ValueError("RNG replay exceeded proposal bound")
            p = tuple(map(int, rng.permutation(row["n"]) + 1))
            key = canonical_orbit_key(p)
            if key in seen or key in replay_keys:
                continue
            replay_keys.add(key)
            replayed.append(p)
        if replayed != [tuple(map(int, p)) for p in objects] or attempts != row["proposals"]:
            raise ValueError("RNG replay mismatch")
        for p_array, record in zip(objects, records):
            p = tuple(map(int, p_array))
            key = canonical_orbit_key(p)
            if key in seen:
                raise ValueError("duplicate or cross-split/historical orbit collision")
            seen.add(key)
            if record["input_id"] != json_hash(p):
                raise ValueError("input ID mismatch")
            if recompute_labels:
                for inst in meta["instances"]:
                    if _normal(label(inst["task_id"], p, inst["params"])) != record["answers"][inst["instance_id"]]:
                        raise ValueError("incorrect exact label")
                    checks += 1
            count += 1
    if sha(out / "vocabulary.json") != meta["vocabulary_sha256"]:
        raise ValueError("vocabulary changed")
    audit = dict(status="passed", objects_checked=count, labels_recomputed=checks,
                 duplicate_inputs=0, cross_role_or_repeat_orbit_overlap=0, historical_orbit_overlap=0,
                 all_pool_rng_streams_replayed=True,
                 empirical_constant_tasks_retained=True, binary_pair_isolation_checked=False)
    write_json(out / ("quality_full.json" if recompute_labels else "quality.json"), audit)
    return audit
