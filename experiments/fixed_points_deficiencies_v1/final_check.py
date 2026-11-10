"""Read-only independent reconstruction of training and selection invariants.

Run only after the fixed training inventory completes. This audit never trains,
generates data, evaluates a model, or changes frozen scientific artifacts. Its
own derived audit record is the only output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from design import CFG, CONDITIONS, REPEATS, ROUTES, SIZES, SOURCES, TARGET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(weights):
    digest = hashlib.sha256()
    for key in sorted(weights):
        digest.update(key.encode())
        digest.update(weights[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load(path):
    value = torch.load(path, map_location="cpu", weights_only=False)
    assert state_hash(value["model"]) == value["model_sha256"], path
    return value


def csvrows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def replay(seed, updates, count, slots):
    rng = np.random.default_rng(seed)
    exposures = np.zeros((slots, count), dtype=np.int64)
    digest = hashlib.sha256()
    for _ in range(updates):
        indices = rng.choice(count, 64, replace=False)
        assigned = np.tile(np.arange(slots), 64//slots)
        rng.shuffle(assigned)
        digest.update(np.asarray(indices, dtype="<i4").tobytes())
        digest.update(np.asarray(assigned, dtype="<i1").tobytes())
        for slot in range(slots):
            np.add.at(exposures[slot], indices[assigned == slot], 1)
    return digest.hexdigest(), exposures


def audit(out):
    out = Path(out).resolve()
    started = time.perf_counter()
    frozen = read(out / "FROZEN_PROTOCOL.json")
    protocol_hash = sha(out / "FROZEN_PROTOCOL.json")
    for name, digest in frozen["code_sha256"].items():
        assert sha(ROOT / name) == digest, name
    selections = read(out / "SELECTIONS_FROZEN.json")
    assert selections["training_complete"] is True and selections["status"] == "frozen"
    assert len(selections["selections"]) == 120
    selected_by_key = {(row["repeat_id"], row["condition"], row["target_samples"]): row
                       for row in selections["selections"]}
    assert len(selected_by_key) == 120
    source_records, adaptation_records, selection_records = [], [], []
    binding_paths = [out / "FROZEN_PROTOCOL.json", out / "SELECTIONS_FROZEN.json"]
    for repeat in REPEATS:
        repeat_id = repeat["repeat_id"]
        initial_path = out / "initial" / f"{repeat_id}.pt"
        initial = load(initial_path)
        assert initial["repeat"] == repeat
        expected_source_stream, expected_source_exposures = replay(
            repeat["source_sampling_seed"], CFG["pretrain_updates"], CFG["source_pool"], 4)
        expected_adapt = {size: replay(repeat["adapt_sampling_seed"], CFG["adaptation_updates"], size, 1)
                          for size in SIZES}
        for condition in ROUTES:
            source_path = initial_path if condition == "A" else out / "pretrain" / f"{repeat_id}_{condition}" / "endpoint.pt"
            source = initial if condition == "A" else load(source_path)
            source_sha = sha(source_path)
            if condition != "A":
                source_folder = source_path.parent
                completed = read(source_folder / "completed.json")
                assert completed["protocol_sha256"] == protocol_hash
                assert completed["endpoint_sha256"] == source_sha
                assert completed["initial_sha256"] == sha(initial_path)
                assert completed["tasks"] == SOURCES[condition]
                assert source["step"] == completed["updates"] == 4800
                assert source["model_sha256"] != initial["model_sha256"]
                assert completed["stream_sha256"] == expected_source_stream
                actual_counts = np.load(source_folder / "input_exposure_counts.npz")["counts"]
                assert np.array_equal(actual_counts, expected_source_exposures)
                exposure_rows = csvrows(source_folder / "exposures.csv")
                assert len(exposure_rows) == 4
                for slot, row in enumerate(exposure_rows):
                    count = int(actual_counts[slot].sum())
                    assert int(row["task_slot"]) == slot and row["task"] == SOURCES[condition][slot]
                    assert count == int(row["sample_exposures"]) == 4800*16
                    assert int(row["unique_inputs"]) == int((actual_counts[slot] > 0).sum())
                    assert int(row["prompt_tokens"]) == count*26
                    assert int(row["training_sequence_tokens"]) == count*28
                    assert int(row["supervised_tokens"]) == count*2
                curves = csvrows(source_folder / "curves.csv")
                expected_keys = {(task, str(step), split) for task in SOURCES[condition]
                                 for step in CFG["source_eval_steps"] for split in ["train", "dev"]}
                assert {(r["task"], r["step"], r["split"]) for r in curves} == expected_keys
                assert len(curves) == len(expected_keys)
                for row in curves:
                    count = 8192 if row["split"] == "train" else 1024
                    assert int(row["count"]) == count
                    assert float(row["accuracy"]) == int(row["correct"])/count
                source_records.append(dict(repeat_id=repeat_id, condition=condition,
                                           stream_and_exposure_replay=True, endpoint_changed=True, all_curve_rows=True))
                binding_paths.extend([source_folder / "completed.json", source_folder / "curves.csv",
                                      source_folder / "input_exposure_counts.npz"])
            for size in SIZES:
                candidate_best = []
                for lr_index, lr in enumerate(CFG["learning_rates"]):
                    run_id = f"{repeat_id}_{condition}_n{size}_lr{lr_index}"
                    folder = out / "adapt" / run_id
                    completed = read(folder / "completed.json")
                    assert completed["protocol_sha256"] == protocol_hash
                    assert completed["run_id"] == run_id and completed["updates"] == 800
                    assert completed["lr"] == lr and completed["target_development_labels"] == 1024
                    load_audit = read(folder / "load_audit.json")
                    assert load_audit == completed["load_audit"]
                    assert load_audit["source_checkpoint_sha256"] == source_sha
                    assert load_audit["source_model_sha256"] == source["model_sha256"] == load_audit["loaded_model_sha256"]
                    assert load_audit["fresh_optimizer_state_entries"] == 0
                    assert load_audit["full_parameter_training"] and load_audit["tied_weights"]
                    target_row = load_audit["target_row"]
                    expected_weights = dict(source["model"])
                    for name in ["token_embedding.weight", "lm_head.weight"]:
                        expected_weights[name] = expected_weights[name].clone()
                        expected_weights[name][target_row].copy_(initial["model"][name][target_row])
                    assert state_hash(expected_weights) == load_audit["adaptation_initial_model_sha256"]
                    row_hash = hashlib.sha256(initial["model"]["token_embedding.weight"][target_row].numpy().tobytes()).hexdigest()
                    assert row_hash == load_audit["target_row_sha256"]
                    del expected_weights
                    stream, counts = expected_adapt[size]
                    assert completed["stream_sha256"] == stream
                    actual_counts = np.load(folder / "input_exposure_counts.npz")["counts"]
                    assert np.array_equal(actual_counts, counts[0])
                    assert int(actual_counts.sum()) == completed["sample_exposures"] == 800*64
                    assert completed["independent_inputs"] == int((actual_counts > 0).sum())
                    assert completed["prompt_tokens"] == 800*64*26
                    assert completed["training_sequence_tokens"] == 800*64*28
                    assert completed["supervised_tokens"] == 800*64*2
                    curves = csvrows(folder / "curves.csv")
                    expected_keys = {(str(step), split) for step in range(0, 801, 100) for split in ["train", "dev"]}
                    assert {(r["step"], r["split"]) for r in curves} == expected_keys and len(curves) == len(expected_keys)
                    development = [r for r in curves if r["split"] == "dev" and int(r["step"]) > 0]
                    winner = min(development, key=lambda r: (-float(r["accuracy"]), float(r["answer_ce"]), int(r["step"])))
                    key = (-float(winner["accuracy"]), float(winner["answer_ce"]), int(winner["step"]), lr_index)
                    assert completed["best"]["selection_key"] == list(key[:3])
                    candidate_best.append((key, run_id, completed))
                    for kind in ["best", "endpoint"]:
                        assert sha(folder / f"{kind}.pt") == completed[f"{kind}_sha256"]
                    adaptation_records.append(dict(run_id=run_id, complete_model_load_and_only_tied_row_reset=True,
                        empty_initial_optimizer=True, paired_sampling_replay=True, within_lr_selection_recomputed=True))
                    binding_paths.extend([folder / "completed.json", folder / "curves.csv", folder / "load_audit.json",
                                          folder / "input_exposure_counts.npz"])
                key, chosen_id, chosen = min(candidate_best, key=lambda item: item[0])
                selected = selected_by_key[(repeat_id, condition, size)]
                assert selected["run_id"] == chosen_id
                assert selected["step"] == key[2] and selected["lr_index"] == key[3]
                assert selected["selected_sha256"] == chosen["best_sha256"]
                assert selected["endpoint_sha256"] == chosen["endpoint_sha256"]
                selection_records.append(dict(repeat_id=repeat_id, condition=condition, target_samples=size,
                    selected_run_id=chosen_id, selected_step=key[2], selected_lr_index=key[3], recomputed=True))
        del initial
    assert len(source_records) == 32 and len(adaptation_records) == 360 and len(selection_records) == 120
    result = dict(status="passed", audited_at_utc=datetime.now(timezone.utc).isoformat(),
                  audit_script_sha256=sha(Path(__file__)), elapsed_seconds=time.perf_counter()-started,
                  source_models=32, adaptation_candidates=360, selected_results=120,
                  scope="CPU audit of stored training, parameter, exposure and selection evidence; no new model evaluation",
                  source_checks=source_records, adaptation_checks=adaptation_records, selection_checks=selection_records,
                  evidence_sha256={p.relative_to(out).as_posix(): sha(p) for p in binding_paths})
    path = out / "FINAL_TRAINING_AUDIT.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["status", "source_models", "adaptation_candidates", "selected_results", "elapsed_seconds"]}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "artifacts")
    audit(parser.parse_args().out)
