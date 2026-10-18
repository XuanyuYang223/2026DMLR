"""Independent CPU-only audit of the finite v4 background experiment.

Does not import the training engine, run model forwards, or change historical
artifacts. Run after TEST_COMPLETE; rerun after reporting to refresh manifest.
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
V1 = HERE.parent / "permworld_pilot/runs/pilot_v1"
V2 = HERE.parent / "pilot_v2_diagnostics/artifacts"
V3 = HERE.parent / "pilot_v3_multitarget/artifacts"
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
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def state_hash(state, token=None, initial_row=None):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if token is not None and name in ["token_embedding.weight", "lm_head.weight"]:
            value = value.clone()
            value[token].copy_(initial_row)
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message):
    check(math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-12), message)


def truth(value):
    check(value in [True, False, "True", "False"], f"Invalid Boolean {value!r}")
    return value is True or value == "True"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def csvwrite(path, values):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(dict.fromkeys(k for r in values for k in r)))
        writer.writeheader()
        writer.writerows(values)


def orbit_key(values):
    p = tuple(map(int, values))
    inverse = tuple(p.index(v) + 1 for v in range(1, len(p) + 1))
    orbit = []
    for base in [p, inverse]:
        for q in [base, base[::-1]]:
            orbit.extend([q, tuple(len(q) + 1 - v for v in q)])
    return min(orbit)


def best_key(row):
    return (-float(row["dev_accuracy"]), float(row["dev_answer_ce"]), int(row["step"]))


def source_lr(step):
    return .00003 if step > 2400 else .0003 * (step / 100 if step <= 100 else
        .1 + .9 * .5 * (1 + math.cos(math.pi * (step - 100) / 2300)))


def audit(out):
    started = time.monotonic()
    frozen = read(out / "FROZEN_PROTOCOL.json")
    cfg, groups = frozen["config"], frozen["groups"]
    ready, smoke = read(out / "DATA_READY.json"), read(out / "SMOKE.json")
    selection = read(out / "SELECTIONS_FROZEN.json")
    test_audit, finish = read(out / "test_data_audit.json"), read(out / "TEST_COMPLETE.json")
    seeds, targets, backgrounds = cfg["seeds"], cfg["targets"], cfg["backgrounds"]
    lrs = cfg["learning_rates"]
    check(seeds == [17, 42, 73], "Unexpected seed scope")
    check(targets == ["fixed_points", "longest_decreasing_run"], "Unexpected target scope")
    check(backgrounds == ["g0_v3", "g1_recoils", "g2_deficiencies"], "Unexpected backgrounds")
    check(lrs == [.0003, .0001, .00003] and cfg["target_samples"] == 256, "Adaptation scope drift")
    check(cfg == read(out / "config.json"), "Config disagreement")
    check(cfg["vocab_size"] == len(VOCABULARY) == 188 and cfg["n"] == 10, "Vocab/length mismatch")
    check(smoke["status"] == "passed" and smoke["common_task_sequences_equal"], "Smoke incomplete")
    check(read(HERE / "catalog/CATALOG_AUDIT.json")["status"] == "passed", "Catalog incomplete")
    check(read(HERE / "references/reuse_inventory.json")["status"] == "passed", "Reuse audit incomplete")
    check(frozen["data_sha256"] == sha(out / "DATA_READY.json"), "Data manifest changed")
    check(frozen["compositions_sha256"] == sha(out / "compositions.csv"), "Composition table changed")
    for path, digest in frozen["code"].items():
        check(sha(ROOT / path) == digest, f"Frozen source changed: {path}")
    for path, digest in frozen["evidence"].items():
        check(sha((out / path).resolve()) == digest, f"Frozen evidence changed: {path}")
    check(selection["protocol_sha256"] == sha(out / "FROZEN_PROTOCOL.json"), "Selection protocol binding")
    check(test_audit["selection_sha256"] == finish["selection_sha256"] == sha(out / "SELECTIONS_FROZEN.json"),
          "Test selection binding")
    chronology = [ready["unix"], smoke["unix"], frozen["unix"], read(out / "PRETRAIN_COMPLETE.json")["unix"],
                  read(out / "ADAPT_COMPLETE.json")["unix"], selection["unix"], test_audit["unix"], finish["unix"]]
    check(chronology == sorted(chronology), "Freeze/train/select/test chronology")
    preserved = read(out / "preservation.json")
    for path, digest in preserved.items():
        check(sha(ROOT / path) == digest, f"Historical artifact changed: {path}")

    def resolve(path):
        return (out / path).resolve()

    def reference(path):
        return os.path.relpath(path.resolve(), out).replace("\\", "/")

    group_map = {(g["target"], g["background"]): g for g in groups}
    check(len(groups) == len(group_map) == 6, "Composition coverage")
    v3_groups = {g["target"]: g for g in read(V3 / "FROZEN_PROTOCOL.json")["groups"]}
    expected_cells = {(seed, target, "shared", "A") for seed in seeds for target in targets}
    expected_cells |= {(seed, target, background, condition) for seed in seeds for target in targets
                       for background in backgrounds for condition in ["B", "C"]}
    expected_source_cells = {cell for cell in expected_cells if cell[-1] != "A"}
    for target in targets:
        for background in backgrounds:
            group = group_map[(target, background)]
            for condition in ["B", "C"]:
                expected = list(v3_groups[target]["sources"][condition])
                if background == "g1_recoils":
                    check(expected[1] == "descents", "Unexpected common slot1")
                    expected[1] = "recoils"
                elif background == "g2_deficiencies":
                    check(expected[2] == "peaks", "Unexpected common slot2")
                    expected[2] = "deficiencies"
                check(group["sources"][condition] == expected, "Background slot manipulation changed")
                check(len(set(expected)) == 4 and not set(expected).intersection(targets), "Task count or target leakage")
            check(group["sources"]["B"][1:] == group["sources"]["C"][1:] == group["common"],
                  "B/C common tasks differ")
    source_tasks = sorted({t for g in groups for ts in g["sources"].values() for t in ts})
    checked_tasks = sorted(set(source_tasks + targets))
    task_index = {t: i for i, t in enumerate(PROPERTY_FUNCTIONS)}
    label_checks = 0

    def verify_arrays(p, y, count, label):
        nonlocal label_checks
        check(p.shape == (count, 10) and y.shape == (count, 32), f"Data shape: {label}")
        check(np.all(np.sort(p, axis=1) == np.arange(1, 11)), f"Invalid permutation: {label}")
        for task in checked_tasks:
            answers = np.asarray([PROPERTY_FUNCTIONS[task](tuple(map(int, q))) for q in p], dtype=np.int16)
            check(np.array_equal(answers, y[:, task_index[task]]), f"Label mismatch: {label}/{task}")
            label_checks += count

    pool_data, pool_hashes, current_keys = {}, {}, set()
    for row in ready["pools"]:
        seed, role = row["seed"], row["role"]
        path = out / "data" / f"s{seed}_{role}.npz"
        origin = V3 / "data" / path.name
        check(sha(path) == row["sha256"] == sha(origin), f"Exact v3 data reuse: {path}")
        check(resolve(row["source_file"]) == origin.resolve(), "Data provenance path")
        check(row["data_generation_seed"] == cfg["data_generation_seeds"][str(seed)], "Data generation seed")
        with np.load(path, allow_pickle=False) as archive:
            p, y = archive["p"], archive["y"]
        verify_arrays(p, y, row["count"], path.name)
        check(hashlib.sha256(p.tobytes()).hexdigest() == row["input_sha256"], "Input tensor SHA")
        keys = {orbit_key(q) for q in p}
        check(len(keys) == len(p) and keys.isdisjoint(current_keys), "Train/dev cross-role or cross-seed orbit leakage")
        current_keys.update(keys)
        pool_data[(seed, role)], pool_hashes[(seed, role)] = (p, y), row["sha256"]
    check(len(pool_data) == 12, "Train/development pool count")
    historical_keys = set()
    for root in [V1, V2, V3]:
        for path in sorted((root / "data").glob("*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                historical_keys.update(orbit_key(q) for q in archive["p"])
    blocked = historical_keys | current_keys
    check(test_audit["excluded_orbits"] == len(blocked), "Historical exclusion count")
    check(sorted(r["seed"] for r in test_audit["pools"]) == seeds, "Test seed coverage")
    for row in test_audit["pools"]:
        path = out / "data" / f"s{row['seed']}_test.npz"
        check(sha(path) == row["sha256"], "Test SHA changed")
        check(row["data_generation_seed"] == cfg["new_test_seed"], "New test generation seed")
        with np.load(path, allow_pickle=False) as archive:
            p, y = archive["p"], archive["y"]
        verify_arrays(p, y, 2048, path.name)
        keys = {orbit_key(q) for q in p}
        check(len(keys) == row["unique_orbits"] == row["count"] == 2048 and keys.isdisjoint(blocked), "New test orbit isolation")
        blocked.update(keys)
        pool_data[(row["seed"], "test")] = (p, y)

    checkpoints, expected_shapes = {}, None

    def checkpoint(path, expected_sha=None, step=None, optimizer_step=None):
        nonlocal expected_shapes
        path = path.resolve()
        digest = sha(path)
        if expected_sha:
            check(digest == expected_sha, f"Checkpoint SHA: {path}")
        cp = torch.load(path, map_location="cpu", weights_only=False)
        state = cp["model"]
        shapes = {k: tuple(v.shape) for k, v in state.items()}
        if expected_shapes is None:
            expected_shapes = shapes
        check(shapes == expected_shapes and shapes["token_embedding.weight"] == (188, 256), "Model shape mismatch")
        check(torch.equal(state["token_embedding.weight"], state["lm_head.weight"]), "Tied output mismatch")
        check(all(not v.is_floating_point() or bool(torch.isfinite(v).all()) for v in state.values()), "Nonfinite model parameters")
        if step is not None:
            check(cp.get("step", 0) == step, f"Checkpoint step: {path}")
        steps = sorted({int(s["step"]) for s in cp.get("optimizer", {}).get("state", {}).values()})
        if optimizer_step is not None:
            check(steps == [optimizer_step], f"Optimizer counter: {path}")
            for group in cp["optimizer"]["param_groups"]:
                check(group["weight_decay"] == .01 and tuple(group["betas"]) == (.9, .999)
                      and group["eps"] == 1e-8, "Optimizer configuration")
        parameter_hash = state_hash(state)
        record = dict(path=reference(path), sha256=digest, state_sha256=parameter_hash,
                      step=cp.get("step", 0), optimizer_steps=";".join(map(str, steps)), cpu_loaded=True,
                      tied_embedding_values_equal=True)
        check(path not in checkpoints or checkpoints[path] == record, "Checkpoint changed during audit")
        checkpoints[path] = record
        return cp, parameter_hash

    initial_states, initial_hashes = {}, {}
    for seed in seeds:
        path = out / "initial" / f"seed{seed}.pt"
        check(sha(path) == sha(V3 / "initial" / path.name) == ready["initials"][str(seed)], "Initial reuse changed")
        cp, initial_hashes[seed] = checkpoint(path, ready["initials"][str(seed)])
        initial_states[seed] = cp["model"]

    def schedule(seed, task_count, updates, count, offset):
        rng, digest = np.random.default_rng(seed + offset), hashlib.sha256()
        indices, slots = [], []
        for _ in range(updates):
            index = rng.choice(count, 64, replace=False)
            slot = np.tile(np.arange(task_count), 64 // task_count)
            rng.shuffle(slot)
            digest.update(index.tobytes()); digest.update(slot.tobytes())
            indices.append(index); slots.append(slot)
        return digest.hexdigest(), rng.bit_generator.state, np.asarray(indices), np.asarray(slots)

    source_schedules = {s: schedule(s, 4, 4800, 8192, 1000) for s in seeds}
    adaptation_schedules = {s: schedule(s, 1, 800, 256, 2000) for s in seeds}
    slot_rows = rows(out / "sampling_slots.csv")
    exposure_rows = rows(out / "source_exposures.csv")
    check(len(slot_rows) == 12 and len(exposure_rows) == 144, "Slot/exposure table size")
    slot_evidence = {}
    for seed in seeds:
        _, _, indices, slots = source_schedules[seed]
        p, y = pool_data[(seed, "source_train")]
        for slot in range(4):
            stream = indices[slots == slot]
            evidence = dict(exposures=len(stream), unique_inputs=len(set(stream.tolist())),
                            index_stream_sha256=hashlib.sha256(stream.tobytes()).hexdigest(),
                            input_stream_sha256=hashlib.sha256(p[stream].tobytes()).hexdigest())
            saved = next(r for r in slot_rows if int(r["seed"]) == seed and int(r["slot"]) == slot)
            for field, value in evidence.items():
                check(str(saved[field]) == str(value), f"Slot stream mismatch: {seed}/{slot}/{field}")
            slot_evidence[(seed, slot)] = (stream, evidence)

    source_metas, source_states = {}, {}
    for meta_path in sorted((out / "multi").glob("*/completed.json")):
        meta = read(meta_path)
        cell = (meta["seed"], meta["target"], meta["background"], meta["condition"])
        seed, target, background, condition = cell
        check(cell in expected_source_cells and cell not in source_metas, "Source run coverage")
        tasks = group_map[(target, background)]["sources"][condition]
        check(meta["tasks"] == tasks and meta["kind"] == "multi", "Source task tuple")
        check(meta["updates"] == 4800 and meta["input_tokens"] == 8601600 and meta["supervised_tokens"] == 614400
              and meta["sample_exposures"] == 307200 and meta["task_exposures"] == 76800
              and meta["task_fraction"] == .25, "Source budget equality")
        check(meta["initial_state_hash"] == initial_hashes[seed], "Source initial weights")
        check(meta["train_sha256"] == pool_hashes[(seed, "source_train")]
              and meta["dev_sha256"] == pool_hashes[(seed, "source_dev")], "Source data binding")
        check(meta["schedule_sha256"] == source_schedules[seed][0], "Paired source sampler")
        path = resolve(meta["checkpoint"])
        cp, final_hash = checkpoint(path, meta["checkpoint_sha256"], 4800, 4800)
        check(final_hash == meta["model_state_hash"] and cp["numpy_rng"] == source_schedules[seed][1], "Source parameter/RNG hash")
        check(cp["seed"] == seed and cp["config"]["model"] == cfg["model"], "Source checkpoint model/seed")
        close(cp["optimizer"]["param_groups"][0]["lr"], source_lr(4800), "Source final LR")
        expected_new = background != "g0_v3"
        check(meta["new_compute"] == expected_new, "Source reuse scope")
        if expected_new:
            check(cp["tasks"] == tasks and cp["config"] == cfg, "New source configuration")
            for step in [1200, 2400]:
                partial, _ = checkpoint(meta_path.parent / f"step{step}.pt", step=step, optimizer_step=step)
                check(partial["tasks"] == tasks and partial["seed"] == seed, "Partial source identity")
                close(partial["optimizer"]["param_groups"][0]["lr"], source_lr(step), "Source LR trajectory")
                del partial
        else:
            old_folder = V3 / "multi" / f"s{seed}_{target}_{condition}"
            old_meta = read(old_folder / "completed.json")
            check(resolve(meta["reused_from"]) == old_folder.resolve(), "g0 source must use its own v3 record")
            check(path == (V3 / old_meta["checkpoint"]).resolve() and meta["checkpoint_sha256"] == old_meta["checkpoint_sha256"],
                  "Source exact historical checkpoint")
            for field in ["tasks", "schedule_sha256", "model_state_hash", "initial_state_hash", "train_sha256", "dev_sha256"]:
                check(meta[field] == old_meta[field], "Source reuse metadata changed")
        curve = rows(meta_path.parent / "curve.csv")
        for slot, task in enumerate(tasks):
            stream, evidence = slot_evidence[(seed, slot)]
            check(meta["actual_unique_per_task"][task] == evidence["unique_inputs"], "Source unique count")
            exposure = next(r for r in exposure_rows if (int(r["seed"]), r["target"], r["background"], r["condition"], int(r["slot"])) == (*cell, slot))
            check(exposure["task"] == task and int(exposure["exposures"]) == 76800
                  and int(exposure["input_tokens"]) == 76800 * 28 and int(exposure["supervised_tokens"]) == 153600
                  and float(exposure["task_fraction"]) == .25, "Source exposure budget")
            for field in ["index_stream_sha256", "input_stream_sha256"]:
                check(exposure[field] == evidence[field], "Unpaired task slot exposure")
            label_digest = hashlib.sha256(pool_data[(seed, "source_train")][1][stream, task_index[task]].tobytes()).hexdigest()
            check(exposure["label_stream_sha256"] == label_digest, "Task label stream binding")
            train_y = pool_data[(seed, "source_train")][1][:, task_index[task]]
            mode = int(np.bincount(train_y).argmax())
            for split, count in [("train", 8192), ("dev", 1024)]:
                endpoint = next(r for r in curve if r["task"] == task and r["split"] == split and int(r["step"]) == 4800)
                check(int(endpoint["count"]) == count and int(endpoint["length"]) == 10, "Source endpoint count/length")
                eval_y = pool_data[(seed, "source_train" if split == "train" else "source_dev")][1][:, task_index[task]]
                close(endpoint["majority_accuracy"], np.mean(eval_y == mode), "Source majority baseline")
        source_metas[cell], source_states[cell] = meta, cp["model"]
        del cp
    check(set(source_metas) == expected_source_cells and len(source_metas) == 36, "Expected36 source cells")
    check(sum(m["new_compute"] for m in source_metas.values()) == 24, "Expected24 new source runs")

    # Existing single-task references are evidence only, never additional v4
    # training. Their1200 update point matches exposure, not optimizer history.
    single_reference_count = 0
    for seed in seeds:
        for task in source_tasks:
            folder = V3 / "single" / f"s{seed}_{task}"
            meta = read(folder / "completed.json")
            check(meta["tasks"] == [task] and meta["updates"] == 4800 and meta["task_exposures"] == 307200, "Single-reference budget")
            check(meta["train_sha256"] == pool_hashes[(seed, "source_train")]
                  and meta["dev_sha256"] == pool_hashes[(seed, "source_dev")], "Single-reference data compatibility")
            for step in [1200, 4800]:
                cp, digest = checkpoint(folder / f"step{step}.pt", meta["checkpoint_sha256"] if step == 4800 else None,
                                        step, step)
                check(cp["tasks"] == [task] and cp["seed"] == seed, "Single-reference checkpoint identity")
                if step == 4800:
                    check(digest == meta["model_state_hash"], "Single-reference parameter hash")
                del cp
            single_reference_count += 1
    check(single_reference_count == 24, "Single-reference coverage")

    metas, initial_reconstructions = {}, {}
    selections = selection["selections"]
    check(len(selections) == 42, "Unique selection count")
    check({(r["seed"], r["target"], r["background"], r["condition"]) for r in selections} == expected_cells,
          "A must be selected once in shared background")
    for meta_path in sorted((out / "adapt").glob("*/completed.json")):
        meta = read(meta_path)
        cell = (meta["seed"], meta["target"], meta["background"], meta["condition"])
        seed, target, background, condition = cell
        lr = meta["lr"]
        check(cell in expected_cells and lr in lrs and (*cell, lr) not in metas, "Adaptation candidate coverage")
        check(meta["updates"] == 800 and meta["input_tokens"] == 1433600 and meta["supervised_tokens"] == 102400
              and meta["sample_exposures"] == 51200, "Adaptation budget equality")
        check(meta["schedule_sha256"] == adaptation_schedules[seed][0], "Adaptation paired sampler")
        check(meta["v4_train_sha256"] == pool_hashes[(seed, "target_train")]
              and meta["v4_dev_sha256"] == pool_hashes[(seed, "target_dev")], "Adaptation data binding")
        expected_new = condition != "A" and background != "g0_v3"
        check(meta["new_compute"] == expected_new, "Adaptation reuse scope")
        init = meta["initialization"]
        check(init["optimizer_entries"] == 0 and init["optimizer_state_reset"] and init["new_output_head"] is False,
              "Fresh adaptation optimizer/head")
        token = TOKEN_TO_ID[f"<{target.upper()}>"]
        initial_row = initial_states[seed]["token_embedding.weight"][token]
        check(init["reset_row_ids"] == [token] and init["target_row_hash"] == tensor_hash(initial_row), "Paired target token initialization")
        expected_source = out / "initial" / f"seed{seed}.pt" if condition == "A" else resolve(source_metas[cell]["checkpoint"])
        check(resolve(init["source_path"]) == expected_source.resolve() and sha(expected_source) == init["source_sha256"],
              "Normalized source path/hash")
        check(init["train_sha256"] == pool_hashes[(seed, "target_train")]
              and init["dev_sha256"] == pool_hashes[(seed, "target_dev")], "Normalized adaptation data hashes")
        if cell not in initial_reconstructions:
            state = initial_states[seed] if condition == "A" else source_states[cell]
            initial_reconstructions[cell] = (state_hash(state), state_hash(state, token, initial_row))
        loaded, restored = initial_reconstructions[cell]
        check(init["loaded_state_hash"] == init["source_state_hash"] == loaded
              and init["adapt_initial_state_hash"] == restored, "Loaded/reset source weights")
        if not expected_new:
            old_folder = V3 / "adapt" / f"s{seed}_{target}_{condition}_lr{lr:.0e}"
            old_meta = read(old_folder / "completed.json")
            check(resolve(meta["reused_from"]) == old_folder.resolve()
                  and meta["reused_metadata_sha256"] == sha(old_folder / "completed.json"), "Reuse record origin")
            check(meta["original_initialization_audit"] == old_meta["initialization"], "Original initialization audit preservation")
        curve = rows(meta_path.parent / "curve.csv")
        check([int(r["step"]) for r in curve] == list(range(0, 801, 100)), "Equal adaptation evaluation budget")
        mode = int(np.bincount(pool_data[(seed, "target_train")][1][:, task_index[target]]).argmax())
        for row in curve:
            check((int(row["seed"]), row["target"], row["background"], row["condition"]) == cell, "Curve identity")
            close(row["lr"], lr, "Curve learning rate")
            for split, count in [("train", 256), ("dev", 1024)]:
                check(int(row[split + "_count"]) == count, "Curve sample count")
                yy = pool_data[(seed, "target_train" if split == "train" else "target_dev")][1][:, task_index[target]]
                close(row[split + "_majority_accuracy"], np.mean(yy == mode), "Curve majority baseline")
                for field in ["answer_ce", "eos_ce", "two_token_ce"]:
                    check(math.isfinite(float(row[split + "_" + field])) and float(row[split + "_" + field]) >= 0,
                          "Invalid development/training loss")
        best = min((r for r in curve if int(r["step"]) > 0), key=best_key)
        endpoint = curve[-1]
        for which, expected in [("best", best), ("endpoint", endpoint)]:
            for field in ["step", "train_accuracy", "dev_accuracy", "dev_answer_ce"]:
                close(meta[which][field], expected[field], "Best/endpoint curve mismatch")
            path = resolve(meta[which + "_path"])
            if not expected_new:
                check(path == (V3 / old_meta[which + "_path"]).resolve()
                      and meta[which + "_sha256"] == old_meta[which + "_sha256"], "Exact adaptation checkpoint reuse")
            cp, digest = checkpoint(path, meta[which + "_sha256"], int(expected["step"]), 800 if which == "endpoint" else None)
            check(cp["seed"] == seed and cp["condition"] == condition and cp["lr"] == lr, "Adaptation checkpoint identity")
            check(cp["config"]["model"] == cfg["model"], "Adaptation model configuration")
            if expected_new:
                check(cp["config"] == cfg and cp["target"] == target and cp["background"] == background, "New adaptation configuration")
            for field in ["step", "train_accuracy", "dev_accuracy", "dev_answer_ce"]:
                close(cp["metrics"][field], meta[which][field], "Saved checkpoint metrics")
            if which == "endpoint":
                check(digest == meta["final_model_state_hash"] and cp["numpy_rng"] == adaptation_schedules[seed][1],
                      "Adaptation final parameter/RNG hash")
                close(cp["optimizer"]["param_groups"][0]["lr"], .1 * lr, "Adaptation final LR")
            del cp
        metas[(*cell, lr)] = meta
    check(len(metas) == 126 and sum(m["new_compute"] for m in metas.values()) == 72, "Unique candidate/reuse totals")
    check(sum(key[3] == "A" for key in metas) == 18, "A baseline was duplicated across backgrounds")
    for cell in expected_cells:
        candidates = [metas[(*cell, lr)] for lr in lrs]
        chosen = min(candidates, key=lambda m: (*best_key(m["best"]), lrs.index(m["lr"])))
        saved = next(r for r in selections if (r["seed"], r["target"], r["background"], r["condition"]) == cell)
        check(chosen["lr"] == saved["lr"] and chosen["best"]["step"] == saved["step"], "Independent development selection mismatch")
        close(chosen["best"]["dev_accuracy"], saved["dev_accuracy"], "Selected development accuracy")
        close(chosen["best"]["dev_answer_ce"], saved["dev_ce"], "Selected development CE")
        for which in ["best", "endpoint"]:
            check(chosen[which + "_path"] == saved[which + "_path"] and chosen[which + "_sha256"] == saved[which + "_sha256"],
                  "Selection checkpoint hash binding")

    raw, selected = rows(out / "results_all_candidates.csv"), rows(out / "results_selected.csv")
    check(len(raw) == finish["candidate_rows"] == 252 and len(selected) == finish["selected_rows"] == 42, "Test row totals")
    check(selected == [r for r in raw if truth(r["selected_primary"])], "Selected raw rows mismatch")
    expected_result_keys = {(*cell, lr, which) for cell in expected_cells for lr in lrs for which in ["best", "endpoint"]}
    actual_result_keys = {(int(r["seed"]), r["target"], r["background"], r["condition"], float(r["lr"]), r["checkpoint"]) for r in raw}
    check(actual_result_keys == expected_result_keys and len(actual_result_keys) == len(raw), "Full test candidate grid")
    for seed in seeds:
        check(rows(out / f"results_seed{seed}.csv") == [r for r in raw if int(r["seed"]) == seed], "Per-seed raw result mismatch")
    for row in raw:
        cell = (int(row["seed"]), row["target"], row["background"], row["condition"])
        seed, target, background, condition = cell
        lr, which = float(row["lr"]), row["checkpoint"]
        meta = metas[(*cell, lr)]
        choice = next(r for r in selections if (r["seed"], r["target"], r["background"], r["condition"]) == cell)
        check(row["checkpoint_sha256"] == meta[which + "_sha256"], "Test checkpoint provenance")
        check(truth(row["selected_lr"]) == (lr == choice["lr"])
              and truth(row["selected_primary"]) == (lr == choice["lr"] and which == "best"), "Result selection flags")
        check(int(row["adapt_step"]) == meta[which]["step"] and int(row["test_count"]) == 2048, "Test counts")
        check(truth(row["new_compute"]) == meta["new_compute"], "Compute reuse flag")
        for field in ["dev_accuracy", "train_accuracy"]:
            close(row[field], meta[which][field], "Result development/train metric binding")
        close(row["dev_ce"], meta[which]["dev_answer_ce"], "Result development CE binding")
        accuracy = float(row["test_accuracy"])
        check(0 <= accuracy <= 1 and abs(accuracy * 2048 - round(accuracy * 2048)) < 1e-8, "Invalid test accuracy")
        for field in ["test_answer_ce", "test_eos_ce", "test_two_token_ce"]:
            check(math.isfinite(float(row[field])) and float(row[field]) >= 0, "Invalid test loss")
        close(row["test_two_token_ce"], (float(row["test_answer_ce"]) + float(row["test_eos_ce"])) / 2, "Test CE aggregation")
        mode = int(np.bincount(pool_data[(seed, "target_train")][1][:, task_index[target]]).argmax())
        close(row["majority_accuracy"], np.mean(pool_data[(seed, "test")][1][:, task_index[target]] == mode), "Test majority baseline")

    # Optional summarized A/B/C rows are a display expansion, not54 selections.
    # The primary engine outputs above must retain only the42 unique cells.
    source_new = [m for m in source_metas.values() if m["new_compute"]]
    adapt_new = [m for m in metas.values() if m["new_compute"]]
    record = dict(status="passed", audit_unix=time.time(), audit_seconds=time.monotonic() - started,
        cpu_only_checkpoint_loads=True, model_forward_calls=0, tests_forwarded_by_this_audit=0,
        historical_files_byte_preserved=len(preserved), frozen_code_and_evidence_verified=True,
        chronological_freeze_before_new_training_and_test=True, exact_v3_train_dev_initial_reuse_verified=True,
        full_orbit_test_isolation_verified=True, mathematical_labels_recomputed=label_checks,
        targets=targets, backgrounds=backgrounds, paired_seed_blocks=seeds,
        source_runs=36, source_runs_new=24, source_runs_reused=12,
        source_per_slot_input_and_label_streams_verified=144,
        adaptation_candidates=126, adaptation_candidates_new=72, adaptation_candidates_reused=54,
        shared_A_candidates=18, independently_recomputed_selections=42, selected_test_rows=42, candidate_test_rows=252,
        logical_ABC_display_rows=54, shared_A_not_independent_background_observations=True,
        compatible_single_task_references=single_reference_count, checkpoints_cpu_loaded=len(checkpoints),
        source_loaded_parameter_and_target_row_reset_hashes_verified=True,
        fresh_adaptation_optimizer_records_and_saved_counters_verified=True,
        new_pretrain_updates=sum(m["updates"] for m in source_new),
        new_adaptation_updates=sum(m["updates"] for m in adapt_new),
        new_pretrain_input_tokens=sum(m["input_tokens"] for m in source_new),
        new_adaptation_input_tokens=sum(m["input_tokens"] for m in adapt_new),
        new_pretrain_seconds=sum(m["seconds"] for m in source_new),
        new_adaptation_seconds=sum(m["seconds"] for m in adapt_new),
        limitations=["Two targets were selected after v3 outcomes; this is an exploratory conditional comparison.",
            "Paired block IDs jointly index old data pools and model initialization; their effects are not independently varied.",
            "A is shared across backgrounds. Display duplication does not add samples, training, selections or independent observations.",
            "Single-task4800 has four times the per-task exposure of multitask4800; single1200 matches exposure but not optimizer/LR history.",
            "This CPU audit checks saved evidence and weights; it does not independently rerun neural inference."])
    write(out / "FINAL_AUDIT.json", record)
    training_environment = read(V3 / "environment.json")
    write(out / "environment.json", dict(python=sys.version, platform=platform.platform(), torch=torch.__version__, numpy=np.__version__,
        cuda_build=torch.version.cuda, cpu_audit_threads=torch.get_num_threads(),
        training_precision=cfg["precision"], historical_training_device=training_environment.get("device"),
        device_metadata_source="pilot_v3_multitarget/artifacts/environment.json; no GPU query or execution by this audit"))
    csvwrite(out / "checkpoint_audit.csv", list(checkpoints.values()))
    files = {p.resolve() for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts
             and p.name != "artifact_manifest.json" and p.suffix != ".log"}
    files.update(p.resolve() for p in out.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                 and p.name != "artifact_manifest.json" and p.suffix != ".log")
    manifest = {reference(path): dict(bytes=path.stat().st_size, sha256=sha(path)) for path in sorted(files)}
    external = {rec["path"]: dict(sha256=rec["sha256"], state_sha256=rec["state_sha256"])
                for path, rec in checkpoints.items() if not path.is_relative_to(HERE)}
    write(out / "artifact_manifest.json", dict(path_base=str(out), files=manifest, historical_checkpoints=external))
    print(json.dumps(record, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "artifacts")
    args = parser.parse_args()
    audit(args.out.resolve())


if __name__ == "__main__":
    main()
