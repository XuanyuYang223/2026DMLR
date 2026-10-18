"""Independent CPU-only provenance audit. No training or model forward calls.

Run after TEST_COMPLETE, then summarize, then rerun to include the report/figures
in the manifest. Checkpoints are read with map_location='cpu'.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OLD = HERE.parent / "permworld_pilot/runs/pilot_v1"
V2 = HERE.parent / "pilot_v2_diagnostics/artifacts"
sys.path.insert(0, str(HERE.parent / "permworld_pilot/vendor"))
from neurips_permutations.math_ops import PROPERTY_FUNCTIONS
from neurips_permutations.passage import TOKEN_TO_ID, VOCABULARY

torch.set_num_threads(2)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2 ** 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(state, reset_token=None, initial_row=None):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if reset_token is not None and key in ["token_embedding.weight", "lm_head.weight"]:
            value = value.clone()
            value[reset_token].copy_(initial_row)
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message):
    check(math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-12), message)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def csvwrite(path, values):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def orbit_key(values):
    p = tuple(map(int, values))
    inverse = tuple(p.index(value) + 1 for value in range(1, len(p) + 1))
    candidates = []
    for base in [p, inverse]:
        for q in [base, base[::-1]]:
            candidates.extend([q, tuple(len(q) + 1 - value for value in q)])
    return min(candidates)


def metric_key(row):
    return (-float(row["dev_accuracy"]), float(row["dev_answer_ce"]), int(row["step"]))


def truth(value):
    return value is True or value == "True"


def audit(out):
    started = time.monotonic()
    design = read(out / "DESIGN_FROZEN.json")
    cfg = design["config"]
    protocol = read(out / "FROZEN_PROTOCOL.json")
    ready = read(out / "DATA_READY.json")
    selection = read(out / "SELECTIONS_FROZEN.json")
    evidence = read(out / "FROZEN_EVIDENCE.json")
    test_audit = read(out / "test_data_audit.json")
    finish = read(out / "TEST_COMPLETE.json")
    seeds, targets, lrs = cfg["seeds"], cfg["targets"], cfg["learning_rates"]
    check(seeds == [17, 42, 73] and len(targets) == 4, "Unexpected seed/target scope")
    check(protocol["config"] == cfg == read(out / "config_plan.json"), "Configuration disagreement")
    check(len(VOCABULARY) == cfg["vocab_size"] == 188, "Vocabulary drift")
    for name, digest in design["code"].items():
        check(sha(ROOT / name) == digest, f"Frozen source changed: {name}")
    check(design["data_sha256"] == sha(out / "DATA_READY.json"), "Data manifest changed")
    check(protocol["design_sha256"] == sha(out / "DESIGN_FROZEN.json"), "Design hash changed")
    check(protocol["matching_sha256"] == sha(out / "replacement_matching.csv"), "Matching changed")
    for task, digest in protocol["single_refs"].items():
        check(sha(out / "single" / f"s17_{task}" / "completed.json") == digest,
              f"Design reference changed: {task}")
    check(selection["protocol_sha256"] == sha(out / "FROZEN_PROTOCOL.json"), "Protocol binding changed")
    check(evidence["status"] == "passed", "Frozen evidence did not pass")
    check(evidence["frozen_protocol_sha256"] == sha(out / "FROZEN_PROTOCOL.json"),
          "Frozen evidence protocol binding changed")
    check(evidence["design_unix"] == design["unix"] and evidence["protocol_unix"] == protocol["unix"],
          "Frozen evidence design/protocol timestamp binding changed")
    check(evidence["created_before_new_test_generation"] and evidence["neural_calls"] == 0,
          "Frozen evidence test timing or neural-call scope changed")
    for name, rec in evidence["files"].items():
        path = (out / name).resolve()
        check(path.stat().st_size == rec["bytes"] and sha(path) == rec["sha256"],
              f"Locked design evidence changed: {name}")
    check(test_audit["selection_sha256"] == finish["selection_sha256"] == sha(out / "SELECTIONS_FROZEN.json"),
          "Selection/test binding changed")
    chronology = [design["unix"], read(out / "DESIGN_REFERENCES_COMPLETE.json")["unix"], protocol["unix"],
                  evidence["created_unix"],
                  read(out / "SINGLE_REFERENCES_COMPLETE.json")["unix"], read(out / "PRETRAIN_COMPLETE.json")["unix"],
                  read(out / "ADAPT_COMPLETE.json")["unix"], selection["unix"], test_audit["unix"], finish["unix"]]
    check(chronology == sorted(chronology), "Freeze/train/test chronological order failed")
    preserved = read(out / "preservation.json")
    for name, digest in preserved.items():
        check(sha(ROOT / name) == digest, f"Old artifact changed: {name}")

    def resolve(name):
        return (out / name).resolve()

    # Independently recompute orbit keys; old inputs are read only to exclude
    # their orbits. No old test predictions/accuracies are used in this audit.
    old_keys = set()
    old_paths = sorted((OLD / "data").glob("*.npz")) + sorted((V2 / "data").glob("*.npz"))
    for path in old_paths:
        with np.load(path) as data:
            old_keys.update(orbit_key(p) for p in data["p"])
    pools, pool_hashes, v3_keys = {}, {}, set()
    task_index = {task: index for index, task in enumerate(PROPERTY_FUNCTIONS)}
    labels_checked = 0

    def verify_labels(permutations, answers, label):
        nonlocal labels_checked
        check(answers.shape == (len(permutations), 32), f"Answer matrix shape: {label}")
        for task in cfg["design_tasks"]:
            actual = np.asarray([PROPERTY_FUNCTIONS[task](tuple(map(int, p))) for p in permutations], dtype=np.int16)
            check(np.array_equal(actual, answers[:, task_index[task]]), f"Mathematical labels changed: {label}/{task}")
            labels_checked += len(permutations)
    for rec in ready["pools"]:
        seed, role = rec["seed"], rec["role"]
        path = out / "data" / f"s{seed}_{role}.npz"
        check(sha(path) == rec["sha256"], f"Dataset hash changed: {path}")
        with np.load(path) as archive:
            p, y = archive["p"], archive["y"]
        check(p.shape == (rec["count"], 10) and y.shape == (rec["count"], 32), f"Dataset shape: {path}")
        check(np.all(np.sort(p, axis=1) == np.arange(1, 11)), f"Invalid permutation: {path}")
        verify_labels(p, y, path.name)
        check(hashlib.sha256(p.tobytes()).hexdigest() == rec["input_sha256"], f"Input hash: {path}")
        keys = {orbit_key(q) for q in p}
        check(len(keys) == len(p) and keys.isdisjoint(v3_keys), f"Cross-role orbit overlap: {path}")
        v3_keys.update(keys)
        if seed == 73:
            check(keys.isdisjoint(old_keys), f"New seed overlaps history: {path}")
        else:
            old_role = {"source_train": "pretrain", "source_dev": "source_dev",
                        "target_train": "target_train", "target_dev": "target_dev"}[role]
            with np.load(OLD / "data" / f"s{seed}_{old_role}.npz") as archive:
                check(np.array_equal(p, archive["p"][:len(p)]), f"Original input reuse mismatch: {path}")
                for old_index, task in enumerate(["fixed_points", "anti_fixed_points", "successions", "descents", "peaks", "left_to_right_maxima"]):
                    check(np.array_equal(y[:, task_index[task]], archive["y"][:len(p), old_index]),
                          f"Original labels mismatch: {path}, {task}")
        pools[(seed, role)] = (p, y)
        pool_hashes[(seed, role)] = rec["sha256"]
    check(len(pools) == 12, "Expected twelve train/development pools")
    excluded = old_keys | v3_keys
    check(sorted(rec["seed"] for rec in test_audit["pools"]) == seeds, "Fresh-test seed coverage")
    for rec in test_audit["pools"]:
        path = out / "data" / f"s{rec['seed']}_test.npz"
        check(sha(path) == rec["sha256"], f"Test hash changed: {path}")
        with np.load(path) as archive:
            p, y = archive["p"], archive["y"]
        keys = {orbit_key(q) for q in p}
        check(p.shape == (2048, 10) and len(p) == len(keys) == rec["unique_orbits"] == rec["count"] == 2048,
              "Test duplicates/count")
        check(keys.isdisjoint(excluded), "Fresh test orbit leakage")
        check(np.all(np.sort(p, axis=1) == np.arange(1, 11)), "Invalid test permutation")
        verify_labels(p, y, path.name)
        excluded.update(keys)
        pools[(rec["seed"], "test")] = (p, y)

    # Load every relevant saved checkpoint on CPU, validating tied output,
    # shapes, finite parameters, optimizer counters, and parameter hashes.
    checkpoint_records = {}
    expected_shapes = None

    def checkpoint(path, expected_step=None, expected_optimizer=None, expected_sha=None):
        nonlocal expected_shapes
        path = path.resolve()
        digest = sha(path)
        if expected_sha is not None:
            check(digest == expected_sha, f"Checkpoint SHA mismatch: {path}")
        cp = torch.load(path, map_location="cpu", weights_only=False)
        state = cp["model"]
        shapes = {name: tuple(value.shape) for name, value in state.items()}
        if expected_shapes is None:
            expected_shapes = shapes
        check(shapes == expected_shapes, f"Model shape/key drift: {path}")
        check(shapes["token_embedding.weight"] == (188, 256), f"Embedding mismatch: {path}")
        check(torch.equal(state["token_embedding.weight"], state["lm_head.weight"]), f"Tied output mismatch: {path}")
        check(all(not value.is_floating_point() or bool(torch.isfinite(value).all()) for value in state.values()),
              f"Nonfinite saved parameters: {path}")
        if expected_step is not None:
            check(cp.get("step", 0) == expected_step, f"Checkpoint step: {path}")
        optimizer_steps = sorted({int(value["step"]) for value in cp.get("optimizer", {}).get("state", {}).values()})
        if expected_optimizer is not None:
            check(optimizer_steps == [expected_optimizer], f"Optimizer steps: {path}")
            for group in cp["optimizer"]["param_groups"]:
                check(group["weight_decay"] == .01 and tuple(group["betas"]) == (.9, .999)
                      and group["eps"] == 1e-8, f"Optimizer config: {path}")
        parameter_hash = state_hash(state)
        record = dict(path=os.path.relpath(path, out).replace("\\", "/"), sha256=digest,
                      state_sha256=parameter_hash, step=cp.get("step", 0), cpu_loaded=True,
                      optimizer_steps=";".join(map(str, optimizer_steps)), tied_embedding_equal=True)
        previous = checkpoint_records.get(path)
        check(previous is None or previous == record, f"Checkpoint mutated during audit: {path}")
        checkpoint_records[path] = record
        return cp, parameter_hash

    initial_states, initial_hashes = {}, {}
    for seed in seeds:
        path = out / "initial" / f"seed{seed}.pt"
        cp, initial_hashes[seed] = checkpoint(path, expected_sha=ready["initials"][str(seed)])
        initial_states[seed] = cp["model"]
        if seed in [17, 42]:
            old_cp, old_hash = checkpoint(OLD / f"seed{seed}" / "step0.pt")
            check(old_hash == initial_hashes[seed], "Original initialization mismatch")
            del old_cp

    def replay_schedule(seed, task_count, updates, train_count, offset):
        rng = np.random.default_rng(seed + offset)
        digest = hashlib.sha256()
        seen = [set() for _ in range(task_count)]
        for _ in range(updates):
            indices = rng.choice(train_count, 64, replace=False)
            slots = np.tile(np.arange(task_count), 64 // task_count)
            rng.shuffle(slots)
            digest.update(indices.tobytes())
            digest.update(slots.tobytes())
            for index in range(task_count):
                seen[index].update(map(int, indices[slots == index]))
        return digest.hexdigest(), rng.bit_generator.state, [len(values) for values in seen]

    source_schedules = {(seed, size): replay_schedule(seed, size, 4800, 8192, 1000)
                        for seed in seeds for size in [1, 4]}
    adapt_schedules = {seed: replay_schedule(seed, 1, 800, 256, 2000) for seed in seeds}
    groups = {group["target"]: group for group in protocol["groups"]}
    check(set(groups) == set(targets), "Target composition coverage")
    source_tasks = sorted({task for group in groups.values() for tasks in group["sources"].values() for task in tasks})
    for target, group in groups.items():
        b, c = group["sources"]["B"], group["sources"]["C"]
        check(len(b) == len(c) == len(set(b)) == len(set(c)) == 4, "Each composition must have four tasks")
        check(b[1:] == c[1:] and b[0] != c[0], "Replacement must preserve three common task slots")
        check(b[0] == group["related"] and c[0] == group["replacement"], "Composition relation binding")
        check(not set(b + c).intersection(targets), "Target included in pretraining")

    source_metas, single_metas, source_states = {}, {}, {}
    for kind, folders in [("single", sorted((out / "single").glob("*/completed.json"))),
                          ("multi", sorted((out / "multi").glob("*/completed.json")))]:
        for meta_path in folders:
            meta = read(meta_path)
            seed, tasks = meta["seed"], meta["tasks"]
            size = len(tasks)
            check(size == (1 if kind == "single" else 4), "Source task count")
            check(meta["updates"] == 4800 and meta["input_tokens"] == 8601600
                  and meta["supervised_tokens"] == 614400 and meta["sample_exposures"] == 307200,
                  f"Source budget mismatch: {meta_path}")
            check(meta["task_exposures"] == 307200 // size and meta["task_fraction"] == 1 / size,
                  f"Per-task exposure mismatch: {meta_path}")
            check(meta["initial_state_hash"] == initial_hashes[seed], f"Source initialization: {meta_path}")
            check(meta["train_sha256"] == pool_hashes[(seed, "source_train")]
                  and meta["dev_sha256"] == pool_hashes[(seed, "source_dev")], f"Source data mismatch: {meta_path}")
            schedule_hash, rng_state, unique = source_schedules[(seed, size)]
            check(meta["schedule_sha256"] == schedule_hash, f"Source sampler mismatch: {meta_path}")
            check(meta["actual_unique_per_task"] == dict(zip(tasks, unique)), f"Unique sample counts: {meta_path}")
            path = resolve(meta["checkpoint"])
            cp, final_hash = checkpoint(path, 4800, 4800, meta["checkpoint_sha256"])
            check(cp["seed"] == seed, "Source checkpoint seed")
            if meta["new_compute"]:
                check(cp["tasks"] == tasks and cp["config"] == cfg, "New source checkpoint task/config binding")
            check(final_hash == meta["model_state_hash"], f"Source state hash: {meta_path}")
            check(cp["numpy_rng"] == rng_state, f"Source final sampler RNG: {meta_path}")
            close(cp["optimizer"]["param_groups"][0]["lr"], 3e-5, "Source final learning rate")
            curve = rows(meta_path.parent / "curve.csv")
            for task in tasks:
                for split in ["train", "dev"]:
                    endpoint = [r for r in curve if r["task"] == task and r["split"] == split and int(r["step"]) == 4800]
                    check(len(endpoint) == 1, f"Source endpoint measurement missing: {meta_path}")
                    check(int(endpoint[0]["length"]) == 10 and int(endpoint[0]["count"]) == (8192 if split == "train" else 1024),
                          "Source length/count mismatch")
                    yy = pools[(seed, "source_train")][1][:, task_index[task]]
                    mode = int(np.bincount(yy).argmax())
                    eval_y = pools[(seed, "source_train" if split == "train" else "source_dev")][1][:, task_index[task]]
                    close(endpoint[0]["majority_accuracy"], np.mean(eval_y == mode), "Source majority baseline")
                    check(int(endpoint[0]["majority_label"]) == mode, "Source majority label")
            if kind == "single":
                check(meta["new_compute"] is True, "Unexpected reused single-task control")
                single_metas[(seed, tasks[0])] = meta
                for step in [1200, 2400]:
                    partial, _ = checkpoint(meta_path.parent / f"step{step}.pt", step, step)
                    check(partial["tasks"] == tasks and partial["seed"] == seed, "Single-task partial binding")
                    del partial
                check({int(r["step"]) for r in curve} == set(cfg["source_eval_steps"]), "Single-task learning curve coverage")
            else:
                target, condition = next((target, condition) for target in targets for condition in ["B", "C"]
                                         if meta_path.parent.name == f"s{seed}_{target}_{condition}")
                check(tasks == groups[target]["sources"][condition], "Source composition changed")
                source_metas[(seed, target, condition)] = meta
                source_states[(seed, target, condition)] = cp["model"]
                if meta["new_compute"]:
                    for step in [1200, 2400]:
                        partial, _ = checkpoint(meta_path.parent / f"step{step}.pt", step, step)
                        check(partial["tasks"] == tasks and partial["seed"] == seed, "Multitask partial binding")
                        del partial
                else:
                    check(seed in [17, 42] and target == "fixed_points", "Unexpected source reuse")
                    check(path == (V2 / "extended" / f"seed{seed}_{condition}_4800.pt").resolve(), "Reused source path")
                    check(cp["config"]["model"] == cfg["model"] and cp["config"]["sources"][condition] == tasks,
                          "Reused source model/composition binding")
                    check(cp["optimizer_continued"] and cp["rng_continued"] and cp["sampler_replay_4800_verified"],
                          "Reused continuation provenance")
                    check(cp["source_checkpoint_sha256"] == sha(OLD / f"seed{seed}" / f"pretrain_{condition}.pt"),
                          "Original prefix binding")
            del cp
    check(len(source_metas) == 24 and sum(not r["new_compute"] for r in source_metas.values()) == 4,
          "Expected twenty new and four reused pretraining runs")
    check(set(single_metas) == {(17, task) for task in cfg["design_tasks"]}
          | {(seed, task) for seed in [42, 73] for task in source_tasks}, "Single-task coverage mismatch")
    ordered_compositions, unordered_compositions = {}, {}
    for key, meta in source_metas.items():
        ordered_compositions.setdefault((key[0], tuple(meta["tasks"])), []).append(key)
        unordered_compositions.setdefault((key[0], tuple(sorted(meta["tasks"]))), []).append(key)
    for keys in ordered_compositions.values():
        check(len({source_metas[key]["model_state_hash"] for key in keys}) == 1,
              "Identical ordered compositions and paired training unexpectedly produced different source weights")

    # Recompute target gates and replacement choice from source training labels
    # and single-task development curves only.
    single_curves = {task: rows(out / "single" / f"s17_{task}" / "curve.csv") for task in cfg["design_tasks"]}

    def source_dev(task, step):
        return next(r for r in single_curves[task] if r["split"] == "dev" and int(r["step"]) == step)

    matching = rows(out / "replacement_matching.csv")
    for target, group in groups.items():
        dev = source_dev(target, 4800)
        check(float(dev["accuracy"]) >= .9 and float(dev["accuracy"]) >= float(dev["majority_accuracy"]) + .15,
              f"Target development gate failed: {target}")
        related, scores = group["related"], []
        labels = pools[(17, "source_train")][1]
        p = np.bincount(labels[:, task_index[related]], minlength=11) / len(labels)
        for order, candidate in enumerate(cfg["plans"][target]["candidates"]):
            q = np.bincount(labels[:, task_index[candidate]], minlength=11) / len(labels)
            tv = float(np.abs(p - q).sum() / 2)
            end = abs(float(source_dev(related, 4800)["accuracy"]) - float(source_dev(candidate, 4800)["accuracy"]))
            gap = float(np.mean([abs(float(source_dev(related, step)["accuracy"]) - float(source_dev(candidate, step)["accuracy"]))
                                 for step in cfg["source_eval_steps"] if step]))
            score = tv + end + gap
            row = next(r for r in matching if r["target"] == target and r["replacement"] == candidate)
            for field, expected in [("tv", tv), ("endpoint_dev_accuracy_gap", end), ("mean_curve_accuracy_gap", gap), ("score", score)]:
                close(row[field], expected, f"Replacement score {target}/{candidate}/{field}")
            scores.append((score, order, candidate))
        check(min(scores)[2] == group["replacement"], f"Replacement selection changed: {target}")

    metas = [read(path) for path in sorted((out / "adapt").glob("*/completed.json"))]
    check(len(metas) == 108 and sum(not m["new_compute"] for m in metas) == 18,
          "Expected ninety new and eighteen reused adaptation candidates")
    all_selections = selection["selections"]
    check(len(all_selections) == 36, "Selection count")
    source_reset_hashes = {}
    for meta in metas:
        seed, target, condition, lr = meta["seed"], meta["target"], meta["condition"], meta["lr"]
        check(meta["updates"] == 800 and meta["input_tokens"] == 1433600 and meta["supervised_tokens"] == 102400
              and meta["sample_exposures"] == 51200, "Adaptation budget mismatch")
        check(meta["schedule_sha256"] == adapt_schedules[seed][0], "Adaptation sampler mismatch")
        check(meta["v3_train_sha256"] == pool_hashes[(seed, "target_train")]
              and meta["v3_dev_sha256"] == pool_hashes[(seed, "target_dev")], "Adaptation data mismatch")
        init = meta["initialization"]
        check(init["optimizer_entries"] == 0 and init["optimizer_state_reset"] and not init["new_output_head"],
              "Adaptation optimizer/head initialization")
        token = TOKEN_TO_ID[f"<{target.upper()}>"]
        check(init["reset_row_ids"] == [token], "Wrong target token reset")
        initial_row = initial_states[seed]["token_embedding.weight"][token]
        check(init["target_row_hash"] == tensor_hash(initial_row), "Unpaired target initialization")
        source_state = initial_states[seed] if condition == "A" else source_states[(seed, target, condition)]
        key = (seed, target, condition)
        if key not in source_reset_hashes:
            source_reset_hashes[key] = (state_hash(source_state), state_hash(source_state, token, initial_row))
        source_hash, reset_hash = source_reset_hashes[key]
        check(init["source_state_hash"] == init["loaded_state_hash"] == source_hash, "Source parameters not loaded")
        check(init["adapt_initial_state_hash"] == reset_hash, "Incorrect reset initial model")
        if meta["new_compute"]:
            path = resolve(init["source_path"])
            check(init["train_sha256"] == pool_hashes[(seed, "target_train")]
                  and init["dev_sha256"] == pool_hashes[(seed, "target_dev")], "New adaptation data binding")
        else:
            check(seed in [17, 42] and target == "fixed_points", "Unexpected adaptation reuse")
            path = HERE.parent / init["source_path"]
            check(init["train_sha256"] == sha(OLD / "data" / f"s{seed}_target_train.npz")
                  and init["train_slice"] == [0, 256]
                  and init["dev_sha256"] == sha(OLD / "data" / f"s{seed}_target_dev.npz"), "Reused adaptation data binding")
        check(sha(path) == init["source_sha256"], "Adaptation source file changed")
        curve = rows(out / "adapt" / meta["run_id"] / "curve.csv")
        check(sorted(int(r["step"]) for r in curve) == list(range(0, 801, 100)), "Adaptation curve budget")
        train_answers = pools[(seed, "target_train")][1][:, task_index[target]]
        mode = int(np.bincount(train_answers).argmax())
        for row in curve:
            check(int(row["seed"]) == seed and row["target"] == target and row["condition"] == condition,
                  "Adaptation curve identity")
            close(row["lr"], lr, "Adaptation curve learning rate")
            for split, expected_count in [("train", 256), ("dev", 1024)]:
                check(int(row[split + "_count"]) == expected_count, "Adaptation curve sample count")
                eval_y = pools[(seed, "target_train" if split == "train" else "target_dev")][1][:, task_index[target]]
                close(row[split + "_majority_accuracy"], np.mean(eval_y == mode), "Adaptation curve majority baseline")
                check(0 <= float(row[split + "_accuracy"]) <= 1, "Invalid adaptation curve accuracy")
                for field in ["answer_ce", "eos_ce", "two_token_ce"]:
                    value = float(row[split + "_" + field])
                    check(math.isfinite(value) and value >= 0, "Nonfinite/negative adaptation curve CE")
        best = min([r for r in curve if int(r["step"]) > 0], key=metric_key)
        endpoint = next(r for r in curve if int(r["step"]) == 800)
        for which, expected in [("best", best), ("endpoint", endpoint)]:
            for field in ["step", "train_accuracy", "dev_accuracy", "dev_answer_ce"]:
                close(meta[which][field], expected[field], f"Saved metric disagrees with curve: {meta['run_id']} {which}")
            path = resolve(meta[which + "_path"])
            if not meta["new_compute"]:
                budget = 0 if condition == "A" else 4800
                original_folder = V2 / "adapt" / f"s{seed}_{condition}_p{budget}_lr{lr:.0e}"
                check(path == (original_folder / f"{which}.pt").resolve(), "Reused adaptation checkpoint path")
                original_meta = read(original_folder / "completed.json")
                check(meta[which + "_sha256"] == original_meta[which + "_sha256"], "Reused adaptation checkpoint SHA")
            cp, actual_hash = checkpoint(path, int(expected["step"]), 800 if which == "endpoint" else None,
                                         meta[which + "_sha256"])
            check(cp["seed"] == seed and cp["condition"] == condition and cp["lr"] == lr, "Adaptation checkpoint identity")
            check(cp["config"]["model"] == cfg["model"], "Adaptation checkpoint model config")
            if meta["new_compute"]:
                check(cp["target"] == target and cp["config"] == cfg, "New adaptation target/config binding")
            for field in ["step", "train_accuracy", "dev_accuracy", "dev_answer_ce"]:
                close(cp["metrics"][field], meta[which][field], "Checkpoint metrics disagree")
            if which == "endpoint":
                check(actual_hash == meta["final_model_state_hash"], "Final adaptation parameter hash")
                check(cp["numpy_rng"] == adapt_schedules[seed][1], "Adaptation final RNG")
                close(cp["optimizer"]["param_groups"][0]["lr"], lr * .1, "Adaptation final learning rate")
            del cp
    for seed in seeds:
        for target in targets:
            for condition in ["A", "B", "C"]:
                candidates = [m for m in metas if m["seed"] == seed and m["target"] == target and m["condition"] == condition]
                check(sorted(m["lr"] for m in candidates) == sorted(lrs), "Unequal learning-rate selection budget")
                chosen = min(candidates, key=lambda m: (*metric_key(m["best"]), lrs.index(m["lr"])))
                saved = next(r for r in all_selections if r["seed"] == seed and r["target"] == target and r["condition"] == condition)
                check(chosen["lr"] == saved["lr"] and chosen["best"]["step"] == saved["step"], "Development selection mismatch")
                for which in ["best", "endpoint"]:
                    check(chosen[which + "_path"] == saved[which + "_path"]
                          and chosen[which + "_sha256"] == saved[which + "_sha256"], "Frozen selection checkpoint mismatch")

    raw = rows(out / "results_all_candidates.csv")
    selected = rows(out / "results_selected.csv")
    check(len(raw) == finish["candidate_rows"] == 216 and len(selected) == finish["selected_rows"] == 36, "Test result row counts")
    check(selected == [r for r in raw if truth(r["selected_primary"])], "Selected result CSV mismatch")
    check(len({(r["seed"], r["target"], r["condition"], r["lr"], r["checkpoint"]) for r in raw}) == 216,
          "Duplicate candidate test row")
    expected_keys = {(seed, target, condition, lr, which) for seed in seeds for target in targets
                     for condition in ["A", "B", "C"] for lr in lrs for which in ["best", "endpoint"]}
    check({(int(r["seed"]), r["target"], r["condition"], float(r["lr"]), r["checkpoint"]) for r in raw} == expected_keys,
          "Candidate test grid incomplete")
    for seed in seeds:
        check(rows(out / f"results_seed{seed}.csv") == [r for r in raw if int(r["seed"]) == seed], "Per-seed raw CSV mismatch")
        check(sum(int(r["seed"]) == seed for r in raw) == 72, "Per-seed result count")
    for row in raw:
        seed, target, condition, lr = int(row["seed"]), row["target"], row["condition"], float(row["lr"])
        meta = next(m for m in metas if m["seed"] == seed and m["target"] == target and m["condition"] == condition and m["lr"] == lr)
        chosen = next(r for r in all_selections if r["seed"] == seed and r["target"] == target and r["condition"] == condition)
        which = row["checkpoint"]
        check(row["checkpoint_sha256"] == meta[which + "_sha256"], "Test checkpoint mismatch")
        check(truth(row["selected_lr"]) == (lr == chosen["lr"])
              and truth(row["selected_primary"]) == (lr == chosen["lr"] and which == "best"), "Test selection labels")
        check(int(row["adapt_step"]) == meta[which]["step"] and int(row["test_count"]) == 2048, "Test sample/update count")
        for field in ["dev_accuracy", "train_accuracy"]:
            close(row[field], meta[which][field], "Test metadata mismatch")
        close(row["dev_ce"], meta[which]["dev_answer_ce"], "Test development CE mismatch")
        answer_accuracy = float(row["test_accuracy"])
        check(0 <= answer_accuracy <= 1 and abs(answer_accuracy * 2048 - round(answer_accuracy * 2048)) < 1e-8,
              "Invalid accuracy count")
        check(0 <= float(row["test_eos_accuracy"]) <= 1, "Invalid test EOS accuracy")
        for field in ["test_answer_ce", "test_eos_ce", "test_two_token_ce"]:
            check(math.isfinite(float(row[field])) and float(row[field]) >= 0, "Nonfinite/negative test CE")
        close(row["test_two_token_ce"], (float(row["test_answer_ce"]) + float(row["test_eos_ce"])) / 2,
              "Test CE aggregation mismatch")
        y = pools[(seed, "target_train")][1][:, task_index[target]]
        mode = int(np.bincount(y).argmax())
        baseline = np.mean(pools[(seed, "test")][1][:, task_index[target]] == mode)
        close(row["majority_accuracy"], baseline, "Test majority baseline mismatch")

    new_sources = [m for m in source_metas.values() if m["new_compute"]]
    new_adapt = [m for m in metas if m["new_compute"]]
    record = dict(status="passed", audit_unix=time.time(), audit_seconds=time.monotonic() - started,
        cpu_only_checkpoint_loads=True, tests_forwarded_by_this_audit=0, old_files_byte_preserved=len(preserved),
        frozen_source_hashes_verified=True, frozen_evidence_verified=True,
        frozen_evidence_files_verified=len(evidence["files"]), full_orbit_data_isolation_verified=True,
        mathematical_label_values_recomputed=labels_checked,
        fixed_anchor_data_reuse_verified=True, target_gate_and_replacement_choice_recomputed=True,
        targets=4, paired_seeds=seeds, target_samples=256, source_tasks=source_tasks,
        single_task_runs=len(single_metas), single_task_checkpoints_at_1200_2400_4800_verified=True,
        source_runs=24, source_runs_reused=4, adaptation_candidates=108, adaptation_candidates_reused=18,
        ordered_composition_seed_units=len(ordered_compositions),
        unordered_composition_seed_units=len(unordered_compositions),
        identical_ordered_composition_parameter_hashes_verified=True,
        development_selections_recomputed=36, candidate_test_rows=216, selected_test_rows=36,
        checkpoints_cpu_loaded=len(checkpoint_records), paired_source_and_adaptation_samplers_verified=True,
        target_row_and_loaded_parameter_hashes_verified=True, fresh_adaptation_optimizer_evidence_verified=True,
        saved_optimizer_counters_verified=True, selection_before_new_test_generation=True,
        logical_pretrain_updates=24 * 4800, new_pretrain_updates=sum(m["updates"] for m in new_sources),
        logical_adapt_updates=108 * 800, new_adapt_updates=sum(m["updates"] for m in new_adapt),
        new_single_task_updates=sum(m["updates"] for m in single_metas.values()),
        new_source_training_seconds=sum(m["seconds"] for m in new_sources),
        new_adaptation_seconds=sum(m["seconds"] for m in new_adapt),
        new_single_task_seconds=sum(m["seconds"] for m in single_metas.values()),
        limitations=["Single-task4800 has four times the per-task exposures of multitask4800.",
                     "Single-task1200 matches per-task exposures only: optimizer update counts and LR trajectories differ.",
                     "Targets share inputs and initializations; four targets and three seeds are not twelve independent task families.",
                     "Fixed-points seeds17/42 reuse old development selections and weights; only the new test inputs are independent.",
                     "This audit checks saved evidence and parameters; it does not rerun model inference or establish causal interference."])
    write(out / "FINAL_AUDIT.json", record)
    write(out / "environment.json", dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
        numpy=np.__version__, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", cpu_audit=True,
        threads=torch.get_num_threads(), training_precision=cfg["precision"],
        note="Training precision is frozen protocol metadata; this script performs only CPU tensor inspection."))
    csvwrite(out / "checkpoint_audit.csv", list(checkpoint_records.values()))
    # Include code/report/catalog outside artifacts and separately account for
    # referenced historical weights. Rerun after summary generation.
    files = {p.resolve() for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts
             and p.name != "artifact_manifest.json" and p.suffix != ".log"}
    files.update(p.resolve() for p in out.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                 and p.name != "artifact_manifest.json" and p.suffix != ".log")
    manifest = {os.path.relpath(path, out).replace("\\", "/"): dict(bytes=path.stat().st_size, sha256=sha(path))
                for path in sorted(files)}
    external = {record["path"]: dict(sha256=record["sha256"], state_sha256=record["state_sha256"])
                for path, record in checkpoint_records.items() if not path.is_relative_to(HERE)}
    write(out / "artifact_manifest.json", dict(path_base=str(out), files=manifest, historical_checkpoints=external))
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "artifacts")
    args = parser.parse_args()
    audit(args.out.resolve())


if __name__ == "__main__":
    main()
