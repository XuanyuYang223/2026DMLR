"""Verify archived calibration-to-canonical-pool provenance without model forwards.

Only canonical training/development pools are read. Reserved test input files
are never opened. Historical records are read-only. A new audit file is written
once; existing output is rejected rather than replaced.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "experiments/platform_v01/calibration"
DATA = ROOT / "experiments/platform_v01/dataset"
OUT = Path(__file__).resolve().parent / "POOL_REUSE_AUDIT.json"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve the existing audit: {OUT}")
    started = time.monotonic()
    bindings = {}

    def bind(path, expected=None):
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Historical hash mismatch: {path}")
        bindings[path.relative_to(ROOT).as_posix()] = actual
        return actual

    frozen_path = OLD / "FROZEN_CALIBRATION.json"
    frozen = read_json(frozen_path)
    frozen_hash = bind(frozen_path)
    cfg = frozen["config"]
    manifest_path = DATA / "manifest.json"
    manifest = read_json(manifest_path)
    bind(manifest_path, frozen["registry_data_certificate_evidence_sha256"]["../dataset/manifest.json"])
    for rel, expected in frozen["code_and_registry_sha256"].items():
        bind(ROOT / rel, expected)
    for name in ("TRAINING_STARTED.json", "CALIBRATION_COMPLETE.json"):
        record = read_json(OLD / name)
        bind(OLD / name)
        assert record["frozen_protocol_sha256"] == frozen_hash
    old_audit = read_json(OLD / "CALIBRATION_AUDIT.json")
    bind(OLD / "CALIBRATION_AUDIT.json")
    assert old_audit["status"] == "passed"
    assert old_audit["test_forward_calls"] == 0

    expected_paths = set()
    pool_records = []
    for repeat in cfg["repeats"]:
        seed = repeat["initialization_seed"]
        paired = next(r for r in manifest["config"]["replicates"] if r["model_seed"] == seed)
        assert paired["data_seed"] == repeat["data_seed"]
        for role in ("train", "dev"):
            for n in cfg["lengths"]:
                path = DATA / "data" / f"s{seed}_{role}_n{n}.npz"
                expected_paths.add(path.resolve())
                frozen_rel = "../dataset/data/" + path.name
                row = next(r for r in manifest["pools"] if (r["seed"], r["role"], r["n"]) == (seed, role, n))
                assert (DATA / row["object_path"]).resolve() == path.resolve()
                assert frozen["training_development_data_sha256"][frozen_rel] == row["object_sha256"]
                digest = bind(path, row["object_sha256"])
                with np.load(path, allow_pickle=False) as archive:
                    values = archive["p"]
                    assert values.shape == (cfg[f"{role}_per_length"], n)
                    assert np.issubdtype(values.dtype, np.integer)
                    assert np.all(np.sort(values, axis=1) == np.arange(1, n + 1))
                pool_records.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=digest,
                    initialization_seed=seed, data_seed=repeat["data_seed"], role=role,
                    length=n, rows=cfg[f"{role}_per_length"], frozen_and_manifest_hashes_match=True))
    assert expected_paths == {(OLD / p).resolve() for p in frozen["training_development_data_sha256"]}
    assert len(pool_records) == 12

    initializer_records = []
    run_records = []
    exposure_records_checked = 0
    for repeat in cfg["repeats"]:
        seed = repeat["initialization_seed"]
        initial_path = OLD / "initial" / f"seed{seed}.pt"
        initial_file_hash = bind(initial_path)
        initial = torch.load(initial_path, map_location="cpu", weights_only=False)
        assert initial["config"] == cfg and initial["initialization_seed"] == seed
        initial_hash = state_hash(initial["model"])
        initializer_records.append(dict(path=initial_path.relative_to(ROOT).as_posix(),
            sha256=initial_file_hash, model_state_sha256=initial_hash, initialization_seed=seed))
        del initial
        rng = np.random.default_rng(repeat["sampling_and_dropout_seed"])
        expected_schedule = hashlib.sha256()
        expected_exposures = {n: np.zeros(cfg["train_per_length"], dtype=np.int64) for n in cfg["lengths"]}
        order = []
        for _ in range(cfg["updates"]):
            if not order:
                order = list(map(int, rng.permutation(cfg["lengths"])))
            n = order.pop()
            indices = rng.choice(cfg["train_per_length"], cfg["batch_size"], replace=False)
            expected_schedule.update(np.asarray([n, *indices], dtype="<i8").tobytes())
            expected_exposures[n][indices] += 1
        for spec in cfg["representatives"]:
            slug = spec["task_id"] + "".join(f"_{k}{v}" for k, v in sorted(spec["params"].items()))
            rid = f"s{seed}_{slug}"
            folder = OLD / "runs" / rid
            complete_path = folder / "COMPLETE.json"
            complete = read_json(complete_path)
            bind(complete_path)
            assert complete["frozen_protocol_sha256"] == frozen_hash
            assert complete["initial_state_sha256"] == initial_hash
            assert complete["length_and_index_schedule_sha256"] == expected_schedule.hexdigest()
            assert complete["test_evaluated"] is False
            assert complete["updates"] == cfg["updates"]
            for key in ("initialization_seed", "data_seed", "sampling_and_dropout_seed"):
                assert complete[key] == repeat[key]
            exposure_path = folder / "input_exposures.csv"
            bind(exposure_path)
            exposures = read_csv(exposure_path)
            assert len(exposures) == len(cfg["lengths"]) * cfg["train_per_length"]
            assert len({(int(row["length"]), int(row["input_index"])) for row in exposures}) == len(exposures)
            for row in exposures:
                assert row["run_id"] == rid
                assert int(row["exposures"]) == expected_exposures[int(row["length"])][int(row["input_index"])]
            exposure_records_checked += len(exposures)
            for name, digest in complete["checkpoint_sha256"].items():
                bind(folder / name, digest)
            endpoint = torch.load(folder / "endpoint.pt", map_location="cpu", weights_only=False)
            assert endpoint["config"] == cfg
            assert endpoint["step"] == cfg["updates"]
            assert endpoint["identity"]["run_id"] == rid
            assert endpoint["initial_state_sha256"] == initial_hash
            assert state_hash(endpoint["model"]) != initial_hash
            for key in ("initialization_seed", "data_seed", "sampling_and_dropout_seed"):
                assert endpoint["identity"][key] == repeat[key]
            assert {int(s["step"]) for s in endpoint["optimizer"]["state"].values()} == {cfg["updates"]}
            del endpoint
            run_records.append(dict(run_id=rid, initialization_seed=seed, data_seed=repeat["data_seed"],
                completed_path=complete_path.relative_to(ROOT).as_posix(),
                frozen_protocol_sha256=frozen_hash, initial_state_sha256=initial_hash,
                schedule_sha256=expected_schedule.hexdigest(), exposure_records_checked=len(exposures),
                completed_updates=cfg["updates"], endpoint_identity_and_config_match=True,
                archived_checkpoint_file_hashes_match=True))

    expected_run_ids = {r["run_id"] for r in run_records}
    actual_run_ids = {p.parent.name for p in (OLD / "runs").glob("*/COMPLETE.json")}
    assert expected_run_ids == actual_run_ids and len(run_records) == 12
    changed = [rel for rel, digest in bindings.items() if sha(ROOT / rel) != digest]
    assert not changed, changed
    result = dict(status="passed", created_at_utc=datetime.now(timezone.utc).isoformat(),
        command=".venv/Scripts/python.exe -B experiments/platform_calibration_v02/reuse_authorized_v1/verify_pool_reuse.py",
        audit_source_sha256=sha(Path(__file__)), seconds=time.monotonic() - started,
        conclusion="The completed v0.1 calibration records, code, checkpoints and replayed index streams bind the same canonical training/development pool bytes and seed-to-unit pairing requested for reuse.",
        evidence_boundary="Archived provenance and sampling-index replay, not gradient replay or a new model evaluation; the original completed run records supply the historical execution evidence.",
        canonical_train_dev_pools_verified=len(pool_records), historical_runs_verified=len(run_records),
        historical_checkpoint_files_hashed=24, initializer_files_loaded=2, endpoint_files_loaded=12,
        exposure_records_checked=exposure_records_checked, sampling_streams_replayed=2,
        reserved_test_input_files_opened=0, model_forward_calls=0, new_training_updates=0,
        historical_files_changed=changed, seed_pairing=cfg["repeats"], pools=pool_records,
        initializers=initializer_records, historical_runs=run_records,
        historical_file_bindings=dict(sorted(bindings.items())))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "seconds", "canonical_train_dev_pools_verified", "historical_runs_verified", "exposure_records_checked", "reserved_test_input_files_opened", "model_forward_calls")}))


if __name__ == "__main__":
    main()
