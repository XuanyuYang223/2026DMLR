"""Read-only audit of finite calibration outputs; optionally repeat dev forwards."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from permworld_platform.calibration import (
    CFG, TASKS, evaluate, load_checkpoint, load_pools, majority_answers, make_model,
    read_json, run_id, runtime, sha, state_hash, verify_freeze, write_csv, write_json,
)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def audit(out, data_dir, forwards):
    began = time.monotonic()
    verify_freeze(out, data_dir)
    assert read_json(out / "CALIBRATION_COMPLETE.json")["runs"] == 12
    frozen = sha(out / "FROZEN_CALIBRATION.json")
    all_rows, forwards_rows = [], []
    initial_hashes, schedule_hashes = {}, {}
    device = runtime("cuda" if forwards else "cpu")
    for repeat in CFG["repeats"]:
        seed = repeat["initialization_seed"]
        initial_hashes[seed] = state_hash(load_checkpoint(out / "initial" / f"seed{seed}.pt")["model"])
        rebuilt = make_model(seed, torch.device("cpu"))
        assert state_hash(rebuilt.state_dict()) == initial_hashes[seed], "Fresh initialization seed does not reproduce saved weights"
        del rebuilt
        rng = np.random.default_rng(repeat["sampling_and_dropout_seed"])
        order, expected_schedule = [], hashlib.sha256()
        expected_exposures = {n: np.zeros(1024, dtype=np.int64) for n in CFG["lengths"]}
        for _ in range(2400):
            if not order:
                order = list(map(int, rng.permutation(CFG["lengths"])))
            n = order.pop()
            indices = rng.choice(1024, 64, replace=False)
            expected_schedule.update(np.asarray([n, *indices], dtype="<i8").tobytes())
            expected_exposures[n][indices] += 1
        schedule_hashes[seed] = set()
        for task in TASKS:
            rid = run_id(task, repeat)
            folder = out / "runs" / rid
            completed = read_json(folder / "COMPLETE.json")
            assert completed["frozen_protocol_sha256"] == frozen
            assert completed["updates"] == 2400 and completed["sample_exposures"] == 153600
            assert completed["initial_state_sha256"] == initial_hashes[seed]
            assert completed["test_evaluated"] is False and completed["optimizer_started_empty"] is True
            schedule_hashes[seed].add(completed["length_and_index_schedule_sha256"])
            assert completed["length_and_index_schedule_sha256"] == expected_schedule.hexdigest()
            curve = read_csv(folder / "curves.csv")
            assert len(curve) == 13 * 2 * 4
            for split, count in [("train", 1024), ("dev", 256)]:
                for step in range(0, 2401, 200):
                    values = [r for r in curve if r["split"] == split and int(r["step"]) == step]
                    detailed = [r for r in values if r["length"] != "all"]
                    total = next(r for r in values if r["length"] == "all")
                    assert sorted(int(r["length"]) for r in detailed) == [8, 10, 12]
                    assert all(int(r["count"]) == count for r in detailed)
                    for r in values:
                        assert float(r["accuracy"]) == int(r["correct"]) / int(r["count"])
                        assert float(r["majority_accuracy"]) == int(r["majority_correct"]) / int(r["count"])
                    for key in ["count", "correct", "majority_correct"]:
                        assert int(total[key]) == sum(int(r[key]) for r in detailed)
            candidates = [r for r in curve if r["split"] == "dev" and r["length"] == "all" and int(r["step"]) > 0]
            best = min(candidates, key=lambda r: (-float(r["accuracy"]), float(r["answer_ce"]), int(r["step"])))
            assert int(best["step"]) == completed["best_step"]
            exposures = read_csv(folder / "input_exposures.csv")
            assert len(exposures) == 3072
            for n in [8, 10, 12]:
                values = [r for r in exposures if int(r["length"]) == n]
                assert sorted(int(r["input_index"]) for r in values) == list(range(1024))
                assert sum(int(r["exposures"]) for r in values) == 800 * 64
                for r in values:
                    assert int(r["exposures"]) == expected_exposures[n][int(r["input_index"])]
                assert completed["length_updates"][str(n)] == 800
            for metric in ["input_tokens", "padded_input_tokens", "supervised_tokens", "answer_tokens"]:
                assert sum(v[metric] for v in completed["tokens_by_length"].values()) == completed[metric]
            pools = load_pools(data_dir, task, repeat) if forwards else None
            for name, expected_step in [("best", completed["best_step"]), ("endpoint", 2400)]:
                path = folder / f"{name}.pt"
                assert sha(path) == completed["checkpoint_sha256"][path.name]
                payload = load_checkpoint(path)
                assert payload["step"] == expected_step
                assert payload["initial_state_sha256"] == initial_hashes[seed]
                assert state_hash(payload["model"]) != initial_hashes[seed]
                opt_steps = {int(state["step"]) for state in payload["optimizer"]["state"].values()}
                assert opt_steps == {expected_step}
                if forwards:
                    model = make_model(seed, device)
                    model.load_state_dict(payload["model"], strict=True)
                    rechecked = evaluate(model, pools["dev"], majority_answers(pools["train"]), device)
                    for value in rechecked:
                        saved = next(r for r in curve if r["split"] == "dev" and str(r["length"]) == str(value["length"]) and int(r["step"]) == expected_step)
                        for key in ["accuracy", "majority_accuracy", "count", "correct"]:
                            assert float(saved[key]) == value[key], (rid, name, key)
                        assert abs(float(saved["answer_ce"]) - value["answer_ce"]) < 1e-6
                        forwards_rows.append({"run_id": rid, "checkpoint": name, "step": expected_step, **value})
                    del model
                all_rows.append({"run_id": rid, "checkpoint": name, "step": expected_step,
                                 "checkpoint_sha256": sha(path), "optimizer_step": expected_step, "verified": True})
            del payload
        assert len(schedule_hashes[seed]) == 1, "Task representatives received different sample schedules"
    write_csv(out / "checkpoint_audit.csv", all_rows)
    if forwards:
        write_csv(out / "development_forward_audit.csv", forwards_rows)
    result = {"status": "passed", "unix": time.time(), "seconds": time.monotonic() - began,
              "runs": 12, "checkpoint_loads": 26, "checkpoint_audits": len(all_rows),
              "curve_rows": 12 * 104, "input_exposure_records": 12 * 3072,
              "frozen_code_registry_data_verified": True, "paired_initialization_and_sample_schedules": True,
              "initializers_rebuilt_from_seed": 2, "sampling_streams_replayed_from_seed": 2,
              "development_selection_recomputed": True, "optimizer_counters_verified": True,
              "development_forwards_repeated": forwards, "test_forward_calls": 0,
              "scope": "Training/evaluation replay uses the frozen model and evaluator; independent oracle certificates cover label correctness separately"}
    write_json(out / "CALIBRATION_AUDIT.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "experiments" / "platform_v01" / "dataset" / "data")
    parser.add_argument("--forwards", action="store_true")
    args = parser.parse_args()
    audit(args.out.resolve(), args.data_dir.resolve(), args.forwards)
