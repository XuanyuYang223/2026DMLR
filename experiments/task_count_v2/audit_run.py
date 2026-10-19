"""Independent accounting and checkpoint audit of the task-count budget study.

This script performs no training and no test-set neural forwards. Optional
forwards re-evaluate one multi-token source development checkpoint and one
development-selected target checkpoint. Training schedules are reconstructed
from the documented NumPy streams, not from the runner's schedule function.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.task_count_v2 import train, design, data_pipeline
from permworld_platform import calibration as platform


def csv_rows(path):
    with Path(path).open(encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def metadata_path(base, relative):
    """Resolve native Windows metadata paths portably without rewriting archives."""
    return Path(base) / str(relative).replace("\\", "/")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def equal_float(actual, expected, label, tolerance=1e-9):
    actual, expected = float(actual), float(expected)
    check(math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= tolerance,
          f"{label}: {actual} != {expected}")


def independent_schedule(seed, pool_size):
    rng = np.random.default_rng(seed)
    while True:
        for n in rng.permutation([10, 12]).tolist():
            yield n, rng.choice(pool_size, 64, replace=False)


def lr_expected(progress, warmup):
    if progress <= warmup:
        return 3e-4 * progress / warmup
    return 3e-5 + .5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * (progress - warmup) / (1 - warmup)))


def verify_update_numerics(rows, *, source, budget=None):
    """Reject unstable logs and verify both branches of the frozen LR schedule."""
    prior_tokens = 0
    prior_warmup_lr, prior_decay_lr = None, None
    warmup = 1 / 24 if source else 100 / 800
    for index, row in enumerate(rows, 1):
        numeric_fields = ["loss", "gradient_norm", "learning_rate"]
        if "elapsed_seconds" in row:
            numeric_fields.append("elapsed_seconds")
        for field in numeric_fields:
            value = float(row[field])
            check(math.isfinite(value) and value >= 0, f"nonfinite or negative update {field} at {index}")
        tokens = int(row["cumulative_input_tokens"])
        check(tokens > prior_tokens, f"non-increasing consumed tokens at update {index}")
        prior_tokens = tokens
        progress = tokens / budget if source else int(row["update"]) / 800
        check(0 < progress <= 1, f"out-of-budget schedule progress at update {index}")
        lr = float(row["learning_rate"])
        check(lr > 0, f"nonpositive update learning rate at {index}")
        equal_float(lr, lr_expected(progress, warmup), f"predefined learning rate at update {index}")
        if progress <= warmup:
            check(prior_warmup_lr is None or lr >= prior_warmup_lr, "warmup learning rate decreased")
            prior_warmup_lr = lr
        else:
            check(prior_decay_lr is None or lr <= prior_decay_lr, "cosine learning rate increased")
            prior_decay_lr = lr


def totals_zero():
    return dict(input_tokens=0, padded_input_tokens=0, supervised_tokens=0, answer_tokens=0, sample_exposures=0)


def batch_counts(records):
    return {"input_tokens": sum(len(r["input_ids"]) for r in records),
            "padded_input_tokens": max(len(r["input_ids"]) for r in records) * len(records),
            "supervised_tokens": sum(len(r["answer_ids"]) for r in records),
            "answer_tokens": sum(len(r["answer_ids"]) - 1 for r in records),
            "sample_exposures": len(records)}


def add_counts(totals, counts):
    for key, value in counts.items():
        totals[key] += value


def verify_metric_groups(rows, group_fields, per_length_counts):
    groups = {}
    for row in rows:
        if "elapsed_seconds" in row:
            check(math.isfinite(float(row["elapsed_seconds"])) and float(row["elapsed_seconds"]) >= 0,
                  "nonfinite or negative evaluation elapsed time")
        key = tuple(row[field] for field in group_fields)
        group = groups.setdefault(key, {})
        check(row["length"] not in group, f"duplicate metric group {key}/{row['length']}")
        group[row["length"]] = row
    for key, values in groups.items():
        check(set(values) == {"10", "12", "all"}, f"missing lengths in metrics {key}")
        for n in ("10", "12"):
            row = values[n]
            check(int(row["count"]) == per_length_counts[int(n)], f"wrong evaluation count {key}/{n}")
            for correct, accuracy in (("correct", "accuracy"), ("eos_generated", "eos_accuracy"),
                                      ("majority_correct", "majority_accuracy")):
                count = int(row[correct])
                check(0 <= count <= int(row["count"]), f"invalid correct count {key}/{n}/{correct}")
                equal_float(row[accuracy], count / int(row["count"]), f"accuracy quotient {key}/{n}/{accuracy}")
            for loss in ("answer_ce", "eos_ce", "sequence_ce"):
                check(math.isfinite(float(row[loss])) and float(row[loss]) >= 0, f"invalid loss {key}/{loss}")
        all_row = values["all"]
        total_count = sum(int(values[n]["count"]) for n in ("10", "12"))
        check(int(all_row["count"]) == total_count, f"wrong aggregate count {key}")
        for field in ("correct", "eos_generated", "majority_correct"):
            check(int(all_row[field]) == sum(int(values[n][field]) for n in ("10", "12")), f"aggregate {key}/{field}")
        for field in ("accuracy", "eos_accuracy", "majority_accuracy", "answer_ce", "eos_ce", "sequence_ce"):
            expected = sum(float(values[n][field]) * int(values[n]["count"]) for n in ("10", "12")) / total_count
            equal_float(all_row[field], expected, f"aggregate {key}/{field}")
    return groups


def independent_majority(training):
    result = {}
    for n, records in training.items():
        histogram = Counter(tuple(r["answer_ids"]) for r in records)
        answers = {tuple(r["answer_ids"]): r["answer_value"] for r in records}
        result[n] = min(histogram, key=lambda tokens: (-histogram[tokens], answers[tokens]))
    return result


def verify_majority(rows, evaluated, training, context):
    modes = independent_majority(training)
    for row in rows:
        if row["length"] == "all":
            continue
        n = int(row["length"])
        expected = sum(tuple(r["answer_ids"]) == modes[n] for r in evaluated[n])
        check(int(row["majority_correct"]) == expected, f"training-derived majority mismatch {context}/{n}")


@lru_cache(maxsize=3)
def checkpoint(path):
    return platform.load_checkpoint(Path(path))


def verify_checkpoint(cp, updates, initial, freeze_hash, expected_lr):
    check(cp["frozen_protocol_sha256"] == freeze_hash, "checkpoint protocol binding")
    check(cp["config"] == train.CFG, "checkpoint frozen configuration")
    check(cp["initial_state_sha256"] == initial["state_sha256"], "checkpoint initial-state binding")
    check(set(cp["model"]) == set(initial["model"]), "checkpoint state keys")
    for name, value in cp["model"].items():
        check(value.shape == initial["model"][name].shape and value.dtype == initial["model"][name].dtype,
              f"checkpoint shape/dtype {name}")
        check(bool(torch.isfinite(value).all()), f"nonfinite checkpoint parameter {name}")
    check(torch.equal(cp["model"]["lm_head.weight"], cp["model"]["token_embedding.weight"]), "tied output weights diverged")
    optimizer = cp["optimizer"]
    check(bool(optimizer["state"]), "trained checkpoint has empty optimizer")
    parameter_ids = [i for group in optimizer["param_groups"] for i in group["params"]]
    check(set(parameter_ids) == set(optimizer["state"]), "optimizer is missing trained parameter state")
    for state in optimizer["state"].values():
        check(int(state["step"]) == updates, "optimizer step proves incorrect reset/update accounting")
        check(bool(torch.isfinite(state["exp_avg"]).all()) and bool(torch.isfinite(state["exp_avg_sq"]).all()), "nonfinite optimizer state")
    for group in optimizer["param_groups"]:
        equal_float(group["lr"], expected_lr, "checkpoint optimizer LR")
        equal_float(group["weight_decay"], .01, "checkpoint weight decay")
        check(tuple(group["betas"]) == (.9, .999), "checkpoint optimizer betas")


def verify_artifacts(run_dir, complete, freeze_hash):
    check(complete["frozen_protocol_sha256"] == freeze_hash, "run protocol binding")
    check(complete["optimizer_started_empty"] is True and complete["test_evaluated"] is False,
          "training provenance flags")
    for relative, digest in complete["artifact_sha256"].items():
        check(platform.sha(metadata_path(run_dir, relative)) == digest, f"run artifact hash mismatch {run_dir}/{relative}")


def audit_source(out, repeat, chain, k, budget, initial, freeze_hash):
    rid = train.pretrain_id(repeat, chain, k, budget)
    directory = out / "pretraining" / rid
    complete = platform.read_json(directory / "COMPLETE.json")
    verify_artifacts(directory, complete, freeze_hash)
    check(complete["budget_tokens"] == budget and complete["learning_rate_budget_tokens"] == budget,
          "source budget/LR identity")
    check(complete["pretraining_started_from"] == "paired_random_initialization" and
          complete["continued_from_short_budget"] is False, "source run continued a trained endpoint")
    check(complete["initial_state_sha256"] == initial["state_sha256"], "source initial state differs from paired random model")
    initial_path = out / "initial" / f"seed{repeat['initialization_seed']}.pt"
    check(metadata_path(out, complete["initial_checkpoint"]).resolve() == initial_path.resolve(),
          "source initial checkpoint is not the paired random model")
    specs = design.CHAINS[chain][:k]
    keys = [design.task_key(s) for s in specs]
    pools = {design.task_key(s): train.pool(out, repeat, "source_train", s) for s in specs}
    updates = csv_rows(directory / "updates.csv")
    check(len(updates) == complete["updates"], f"update count {rid}")
    verify_update_numerics(updates, source=True, budget=budget)
    totals = totals_zero()
    by_task = {(key, n): totals_zero() for key in keys for n in (10, 12)}
    exposures = {(key, n): np.zeros(4096, dtype=np.int64) for key in keys for n in (10, 12)}
    def stage_exposures():
        result = {}
        for key in keys:
            for length in (10, 12, "all"):
                lengths = (10, 12) if length == "all" else (length,)
                count = sum(by_task[key, n]["sample_exposures"] for n in lengths)
                unique = sum(int(np.count_nonzero(exposures[key, n])) for n in lengths)
                result[key, str(length)] = {
                    "source_sample_exposures": count,
                    "source_unique_inputs_exposed": unique,
                    "source_repeat_exposures": count - unique,
                    "source_input_tokens": sum(by_task[key, n]["input_tokens"] for n in lengths),
                    "source_supervised_tokens": sum(by_task[key, n]["supervised_tokens"] for n in lengths)}
        return result
    stage_exposure_snapshots = {"start": stage_exposures()}
    stream = independent_schedule(repeat["pretrain_sampling_seed"], 4096)
    slots_rng = np.random.default_rng(np.random.SeedSequence([repeat["pretrain_sampling_seed"], 1]))
    object_hash, slot_hash, combined_hash = (hashlib.sha256() for _ in range(3))
    sample_hash, short_prefix_hash = hashlib.sha256(), hashlib.sha256()
    short_prefix_tokens, short_prefix_count, short_prefix_closed = 0, 0, False
    update, next_cost = 0, None
    first_midpoint = None
    while totals["input_tokens"] < budget:
        n, indices = next(stream)
        slots = np.concatenate([slots_rng.permutation(k) for _ in range(64 // k)])
        batch, accepted = [], []
        running = 0
        for index, slot in zip(indices.tolist(), slots.tolist()):
            key = keys[slot]
            row = pools[key][n][index]
            if totals["input_tokens"] + running + len(row["input_ids"]) > budget:
                next_cost = len(row["input_ids"])
                break
            batch.append(row)
            accepted.append((key, index, slot))
            running += len(row["input_ids"])
        if not batch:
            break
        update += 1
        check(update <= len(updates), f"runner stopped before independent budget replay {rid}")
        counts = batch_counts(batch)
        add_counts(totals, counts)
        logged = updates[update - 1]
        check((int(logged["update"]), int(logged["length"]), int(logged["batch_size"])) == (update, n, len(batch)), f"update replay {rid}/{update}")
        for field in ("input_tokens", "padded_input_tokens", "supervised_tokens", "answer_tokens"):
            check(int(logged[field]) == counts[field], f"token accounting {rid}/{update}/{field}")
        check(int(logged["cumulative_input_tokens"]) == totals["input_tokens"], "cumulative token accounting")
        equal_float(logged["learning_rate"], lr_expected(totals["input_tokens"] / budget, 1 / 24), "token LR replay")
        width = max(len(row["input_ids"]) for row in batch)
        for row, (key, index, slot) in zip(batch, accepted):
            single = batch_counts([row])
            single["padded_input_tokens"] = width
            add_counts(by_task[key, n], single)
            exposures[key, n][index] += 1
            if not short_prefix_closed:
                if short_prefix_tokens + len(row["input_ids"]) > 4_000_000:
                    short_prefix_closed = True
                else:
                    short_prefix_hash.update(np.asarray((n, index, slot), dtype="<i8").tobytes())
                    short_prefix_tokens += len(row["input_ids"])
                    short_prefix_count += 1
        object_bytes = np.asarray([n, *[index for _, index, _ in accepted]], dtype="<i8").tobytes()
        slot_bytes = np.asarray([slot for _, _, slot in accepted], dtype="<i8").tobytes()
        object_hash.update(object_bytes)
        slot_hash.update(slot_bytes)
        combined_hash.update(object_bytes)
        combined_hash.update(slot_bytes)
        sample_hash.update(np.asarray([(n, index, slot) for _, index, slot in accepted], dtype="<i8").tobytes())
        if first_midpoint is None and totals["input_tokens"] >= budget / 2:
            first_midpoint = (update, totals["input_tokens"])
            stage_exposure_snapshots["midpoint"] = stage_exposures()
        if len(batch) < 64:
            break
    check(update == complete["updates"], "extra update after token stopping")
    for field, value in totals.items():
        check(complete[field] == value, f"run total {rid}/{field}")
    tail = budget - totals["input_tokens"]
    check(tail == complete["budget_tail_tokens"] and 0 <= tail < (next_cost or 128), "token budget tail bound")
    check(next_cost == complete["next_unconsumed_sequence_tokens"], "next unconsumed sequence cost")
    for key, digest in (("length_index_schedule_sha256", object_hash), ("task_slot_schedule_sha256", slot_hash),
                        ("length_index_and_slot_schedule_sha256", combined_hash), ("sample_stream_sha256", sample_hash)):
        check(complete[key] == digest.hexdigest(), f"independent schedule digest {rid}/{key}")
    if budget == 4_000_000:
        check(complete["paired_short_budget_sample_prefix"] is None, "short run unexpectedly has a parent prefix")
        check(short_prefix_hash.hexdigest() == sample_hash.hexdigest(), "short stream prefix reconstruction")
    else:
        prefix = complete["paired_short_budget_sample_prefix"]
        check(prefix["budget_tokens"] == 4_000_000 and prefix["sample_exposures"] == short_prefix_count and
              prefix["sha256"] == short_prefix_hash.hexdigest() and prefix["matches_short_budget"] is True and
              prefix["same_initialization"] is True, "long budget exact short-prefix evidence")
    stage_exposure_snapshots["endpoint"] = stage_exposures()
    exposure_rows = csv_rows(directory / "input_exposures.csv")
    check(len(exposure_rows) == k * 2 * 4096, "source exposure table row count")
    seen = set()
    for row in exposure_rows:
        identity = row["source_task"], int(row["length"]), int(row["input_index"])
        check(identity not in seen, "duplicate source exposure record")
        seen.add(identity)
        check(int(row["exposures"]) == exposures[identity[0], identity[1]][identity[2]], "source input exposure replay")
    task_rows = csv_rows(directory / "task_exposures.csv")
    check(len(task_rows) == 2 * k, "source task exposure table size")
    for row in task_rows:
        key = row["source_task"], int(row["length"])
        for field, expected in by_task[key].items():
            check(int(row[field]) == expected, f"per-task token replay {rid}/{key}/{field}")
        check(int(row["unique_inputs_exposed"]) == int(np.count_nonzero(exposures[key])), "unique source inputs")
        unique = int(np.count_nonzero(exposures[key]))
        check(int(row["repeat_exposures"]) == by_task[key]["sample_exposures"] - unique,
              "repeated source exposure accounting")
        check(int(row["pool_size"]) == 4096, "source pool grew with the token budget")
        equal_float(row["pool_coverage"], unique / 4096, "source input pool coverage")
        equal_float(row["mean_exposures_per_pool_input"], by_task[key]["sample_exposures"] / 4096,
                    "source mean exposure per input")
        equal_float(row["sample_fraction"], by_task[key]["sample_exposures"] / totals["sample_exposures"], "task fraction")
    union_unique = sum(int(np.count_nonzero(sum(exposures[key, n] for key in keys))) for n in (10, 12))
    check(complete["fixed_pool_unique_inputs_exposed"] == union_unique, "source unique input union accounting")
    check(complete["fixed_pool_size"] == 8192, "source fixed pool size")
    task_counts = [sum(by_task[key, n]["sample_exposures"] for n in (10, 12)) for key in keys]
    check(max(task_counts) - min(task_counts) <= 1, "source tasks were not sampled uniformly")
    check(complete["uniform_task_count_max_minus_min"] == max(task_counts) - min(task_counts), "source balance summary")
    curves = csv_rows(directory / "source_curves.csv")
    check(len(curves) == k * 3 * 2 * 3, "source learning-curve row count")
    groups = verify_metric_groups(curves, ["source_task", "stage", "split"], {10: 256, 12: 256})
    stage_points = {"start": (0, 0), "midpoint": first_midpoint, "endpoint": (update, totals["input_tokens"])}
    for row in curves:
        check((int(row["update"]), int(row["input_tokens"])) == stage_points[row["stage"]], "source evaluation timing")
        for field, value in stage_exposure_snapshots[row["stage"]][row["source_task"], row["length"]].items():
            check(int(row[field]) == value, f"stage source exposure accounting {rid}/{row['stage']}/{field}")
    for spec in specs:
        key = design.task_key(spec)
        for role in ("source_train", "source_dev"):
            evaluation_pool = {n: r[:256] for n, r in pools[key].items()} if role == "source_train" else train.pool(out, repeat, role, spec)
            for stage in stage_points:
                verify_majority(list(groups[key, stage, role].values()), evaluation_pool, pools[key], f"{rid}/{key}/{role}")
    cp = checkpoint(str(directory / "endpoint.pt"))
    verify_checkpoint(cp, update, initial, freeze_hash, lr_expected(totals["input_tokens"] / budget, 1 / 24))
    check(cp["identity"]["run_id"] == rid and cp["identity"]["budget_tokens"] == budget,
          "source checkpoint model/budget identity")
    check(cp["updates"] == update and cp["tokens"] == totals, "source checkpoint accounting")
    check(platform.state_hash(cp["model"]) == cp["state_sha256"], "source checkpoint parameter hash")
    check(train.row_hash(cp["model"]) == complete["target_rows_before_adaptation_sha256"], "source target rows hash")
    return {"run_id": rid, "stage": "pretraining", "budget_tokens": budget, "updates": update, **totals,
            "budget_tail_tokens": tail, "uniform_task_count_max_minus_min": max(task_counts) - min(task_counts),
            "unique_input_union": union_unique, "fixed_input_pool_size": 8192,
            "checkpoint_count": 1, "status": "pass"}


def expected_reset_hash(parent, initial):
    state = dict(parent["model"])
    weight = state["token_embedding.weight"].clone()
    weight[train.target_rows()] = initial["model"]["token_embedding.weight"][train.target_rows()]
    state["token_embedding.weight"] = weight
    state["lm_head.weight"] = weight
    return platform.state_hash(state)


def audit_target(out, repeat, chain, k, budget, spec, initial, freeze_hash):
    rid = train.adaptation_id(repeat, chain, k, budget, spec)
    directory = out / "adaptation" / rid
    complete = platform.read_json(directory / "COMPLETE.json")
    verify_artifacts(directory, complete, freeze_hash)
    check(complete["budget_tokens"] == budget, "adaptation source-budget identity")
    check(complete["updates"] == 800 and complete["optimizer_initial_state_entries"] == 0, "fixed adaptation budget/fresh optimizer")
    training = train.pool(out, repeat, "target_train", spec)
    development = train.pool(out, repeat, "target_dev", spec)
    updates = csv_rows(directory / "updates.csv")
    check(len(updates) == 800, "target update table length")
    verify_update_numerics(updates, source=False)
    totals, exposures = totals_zero(), {n: np.zeros(128, dtype=np.int64) for n in (10, 12)}
    digest, length_updates = hashlib.sha256(), Counter()
    stream = independent_schedule(repeat["adaptation_sampling_seed"], 128)
    for update, logged in enumerate(updates, 1):
        n, indices = next(stream)
        records = [training[n][i] for i in indices]
        counts = batch_counts(records)
        add_counts(totals, counts)
        exposures[n][indices] += 1
        length_updates[n] += 1
        digest.update(np.asarray([n, *indices], dtype="<i8").tobytes())
        check((int(logged["update"]), int(logged["length"])) == (update, n), "target schedule replay")
        for field in ("input_tokens", "padded_input_tokens", "supervised_tokens", "answer_tokens"):
            check(int(logged[field]) == counts[field], f"target token count {rid}/{update}/{field}")
        check(int(logged["cumulative_input_tokens"]) == totals["input_tokens"], "target cumulative tokens")
        equal_float(logged["learning_rate"], lr_expected(update / 800, 100 / 800), "adaptation LR")
    check(length_updates == {10: 400, 12: 400}, "adaptation length balance")
    check(complete["length_updates"] == {str(n): v for n, v in length_updates.items()}, "reported length balance")
    check(digest.hexdigest() == complete["length_index_schedule_sha256"], "paired adaptation sampling hash")
    for field, value in totals.items():
        check(complete[field] == value, f"target final totals {rid}/{field}")
    exposure_rows = csv_rows(directory / "input_exposures.csv")
    check(len(exposure_rows) == 256, "target input exposure row count")
    seen = set()
    for row in exposure_rows:
        key = int(row["length"]), int(row["input_index"])
        check(key not in seen, "duplicate adaptation input exposure")
        seen.add(key)
        check(int(row["exposures"]) == exposures[key[0]][key[1]], "adaptation input exposure replay")
    curves = csv_rows(directory / "curves.csv")
    check(len(curves) == 9 * 2 * 3, "target learning-curve row count")
    for row in curves:
        at = int(row["update"])
        expected_tokens = 0 if at == 0 else int(updates[at - 1]["cumulative_input_tokens"])
        check(int(row["input_tokens"]) == expected_tokens, "target evaluation token position")
    # Train and development have different counts, so check them separately.
    for role, pool in (("target_train", training), ("target_dev", development)):
        subset = [r for r in curves if r["split"] == role]
        verify_metric_groups(subset, ["update", "split"], {n: len(rows) for n, rows in pool.items()})
        check({int(r["update"]) for r in subset} == set(range(0, 801, 100)), "target evaluation schedule")
        verify_majority(subset, pool, training, f"{rid}/{role}")
    dev_rows = [r for r in curves if r["split"] == "target_dev" and r["length"] == "all" and int(r["update"]) > 0]
    best = min(dev_rows, key=lambda r: (-float(r["accuracy"]), float(r["answer_ce"]), int(r["update"])))
    endpoint = next(r for r in dev_rows if int(r["update"]) == 800)
    check(complete["best_update"] == int(best["update"]), "development checkpoint selection replay")
    for key, row in (("best_dev", best), ("endpoint_dev", endpoint)):
        for field in ("accuracy", "answer_ce", "eos_ce", "sequence_ce", "majority_accuracy"):
            equal_float(complete[key][field], row[field], f"reported {key}/{field}")
    parent_path = metadata_path(out, complete["parent_checkpoint"])
    check(platform.sha(parent_path) == complete["parent_checkpoint_sha256"], "parent checkpoint hash")
    expected_parent = (out / "initial" / f"seed{repeat['initialization_seed']}.pt" if chain == "baseline" else
                       out / "pretraining" / train.pretrain_id(repeat, chain, k, budget) / "endpoint.pt")
    check(parent_path.resolve() == expected_parent.resolve(), "wrong pretrained parent")
    parent = checkpoint(str(parent_path))
    check(complete["loaded_state_sha256"] == platform.state_hash(parent["model"]), "loaded pretraining parameter hash")
    check(complete["loaded_target_rows_sha256"] == train.row_hash(parent["model"]), "loaded target row hash")
    check(complete["adaptation_initial_state_sha256"] == expected_reset_hash(parent, initial), "only the four tied target rows reset")
    check(complete["reset_target_rows_sha256"] == initial["target_rows_sha256"], "paired target-specific initialization")
    check(complete["reset_target_token_ids"] == train.target_rows(), "target reset identity")
    for name, row in (("best.pt", best), ("endpoint.pt", endpoint)):
        cp = checkpoint(str(directory / name))
        at = int(row["update"])
        verify_checkpoint(cp, at, initial, freeze_hash, lr_expected(at / 800, 100 / 800))
        check(cp["identity"]["run_id"] == rid and cp["identity"]["target"] == design.task_key(spec) and
              cp["identity"]["budget_tokens"] == budget, "target checkpoint model/budget identity")
        check(cp["update"] == at and cp["parent_checkpoint_sha256"] == complete["parent_checkpoint_sha256"], "target checkpoint metadata")
        check(cp["adaptation_initial_state_sha256"] == complete["adaptation_initial_state_sha256"], "checkpoint reset-state evidence")
        equal_float(cp["development"]["accuracy"], row["accuracy"], "checkpoint development accuracy")
        equal_float(cp["development"]["answer_ce"], row["answer_ce"], "checkpoint development tie-break loss")
        expected_tokens = {field: sum(int(r[field]) for r in updates[:at]) for field in ("input_tokens", "padded_input_tokens", "supervised_tokens", "answer_tokens")}
        expected_tokens["sample_exposures"] = at * 64
        check(cp["tokens"] == expected_tokens, "selected checkpoint consumed-token accounting")
    return {"run_id": rid, "stage": "adaptation", "budget_tokens": budget, "updates": 800, **totals,
            "best_update": int(best["update"]), "checkpoint_count": 2, "status": "pass"}


def audit_test(out, expected_targets):
    selection = platform.read_json(out / "FROZEN_SELECTIONS.json")
    check(selection["status"] == "frozen" and selection["training_complete"] is True, "selection freeze")
    entries = {row["run_id"]: row for row in selection["selections"]}
    check(len(selection["selections"]) == 54 and set(entries) == expected_targets, "selection unique model IDs")
    test_manifest = platform.read_json(out / "data/test_manifest.json")
    check(test_manifest["selection_freeze"]["sha256"] == platform.sha(out / "FROZEN_SELECTIONS.json"), "test generation selection binding")
    check(test_manifest["primary_manifest_sha256"] == platform.sha(out / "data/manifest.json"), "test data primary binding")
    quality = platform.read_json(out / "data/quality_with_test.json")
    check(quality["status"] == "pass" and quality["rng_replayed"] is True and quality["object_count"] == 31488, "final data orbit/RNG audit")
    check(quality["test_manifest_sha256"] == platform.sha(out / "data/test_manifest.json"), "final data quality binding")
    times = [platform.read_json(out / name)[field] for name, field in (
        ("FROZEN_PROTOCOL.json", "frozen_at_utc"), ("TRAINING_STARTED.json", "started_utc"),
        ("TRAINING_COMPLETE.json", "completed_utc"), ("FROZEN_SELECTIONS.json", "frozen_at_utc"))]
    times.append(test_manifest["created_at_utc"])
    check([datetime.fromisoformat(v) for v in times] == sorted(datetime.fromisoformat(v) for v in times), "freeze/train/select/test ordering")
    test_complete = platform.read_json(out / "TEST_COMPLETE.json")
    check(test_complete["status"] == "passed" and test_complete["result_rows"] == 324, "unified test completion")
    check(test_complete["raw_results_sha256"] == platform.sha(out / "raw_test_results.csv"), "raw test-result hash")
    rows = csv_rows(out / "raw_test_results.csv")
    check(len(rows) == 324, "test-result row count")
    groups = verify_metric_groups(rows, ["run_id", "checkpoint"], {10: 512, 12: 512})
    check(len(groups) == 108, "selected/endpoint test model count")
    for rid, entry in entries.items():
        complete = platform.read_json(out / "adaptation" / rid / "COMPLETE.json")
        check(entry["selected_update"] == complete["best_update"] and entry["endpoint_update"] == 800, "frozen selected updates")
        check(entry["selected_checkpoint_sha256"] == complete["artifact_sha256"]["best.pt"] and
              entry["endpoint_checkpoint_sha256"] == complete["artifact_sha256"]["endpoint.pt"], "frozen checkpoint identities")
        repeat = next(r for r in train.REPEATS if r["initialization_seed"] == entry["initialization_seed"])
        spec = next(s for s in design.TARGETS if design.task_key(s) == entry["target"])
        evaluated = train.pool(out, repeat, "test", spec)
        training = train.pool(out, repeat, "target_train", spec)
        saved = platform.read_json(out / "test_results" / f"{rid}.json")
        check(saved["frozen_selection_sha256"] == platform.sha(out / "FROZEN_SELECTIONS.json"), "individual test-result selection binding")
        for kind, update, dev in (("development_selected", entry["selected_update"], entry["best_development"]),
                                  ("fixed_endpoint", 800, entry["endpoint_development"])):
            group = groups[rid, kind]
            for row in group.values():
                check(int(row["selected_update"]) == update and int(row["k"]) == entry["k"] and row["chain"] == entry["chain"], "test checkpoint identity")
                check(int(row["budget_tokens"]) == entry["budget_tokens"], "test source-budget identity")
                equal_float(row["development_accuracy"], dev["accuracy"], "test selection provenance")
            verify_majority(list(group.values()), evaluated, training, f"test/{rid}/{kind}")
        check(len(saved["rows"]) == 6, "individual test-result row count")
        for row in saved["rows"]:
            raw = groups[rid, row["checkpoint"]][str(row["length"])]
            for field in ("accuracy", "answer_ce", "eos_ce", "sequence_ce", "majority_accuracy"):
                equal_float(raw[field], row[field], "individual test-result versus raw CSV")
    baseline_ids = {r["run_id"] for r in rows if r["chain"] == "baseline"}
    check(len(baseline_ids) == 6 and sum(r["chain"] == "baseline" for r in rows) == 36,
          "shared baselines were duplicated as independent training/test models")
    return {"selection_count": 54, "unique_baseline_models": 6, "checkpoint_evaluations": 108,
            "raw_result_rows": len(rows), "test_neural_forwards_in_audit": 0}


def development_forwards(out):
    device = platform.runtime("cuda")
    repeat = train.REPEATS[0]
    source_spec = next(s for s in design.CHAINS["chain_alpha"] if s["task_id"] == "increasing_subsequence_count")
    source_dir = out / "pretraining" / train.pretrain_id(repeat, "chain_alpha", 16, 32_000_000)
    target_spec = design.TARGETS[0]
    target_dir = out / "adaptation" / train.adaptation_id(repeat, "chain_alpha", 16, 32_000_000, target_spec)
    checks = [
        (source_dir / "endpoint.pt", source_spec, "source_train", "source_dev",
         [r for r in csv_rows(source_dir / "source_curves.csv") if r["stage"] == "endpoint" and
          r["source_task"] == design.task_key(source_spec) and r["split"] == "source_dev"]),
        (target_dir / "best.pt", target_spec, "target_train", "target_dev",
         [r for r in csv_rows(target_dir / "curves.csv") if int(r["update"]) ==
          platform.read_json(target_dir / "COMPLETE.json")["best_update"] and r["split"] == "target_dev"]),
    ]
    results = []
    for path, spec, train_role, dev_role, recorded in checks:
        model = platform.make_model(repeat["initialization_seed"], device)
        model.load_state_dict(checkpoint(str(path))["model"], strict=True)
        evaluated = platform.evaluate(model, train.pool(out, repeat, dev_role, spec),
            platform.majority_answers(train.pool(out, repeat, train_role, spec)), device)
        by_length = {str(r["length"]): r for r in recorded}
        for row in evaluated:
            old = by_length[str(row["length"])]
            for field in ("count", "correct", "majority_correct", "eos_generated"):
                check(int(old[field]) == row[field], f"independent development forward {path}/{field}")
            for field in ("accuracy", "answer_ce", "eos_ce", "sequence_ce"):
                equal_float(old[field], row[field], f"independent development forward {path}/{field}", tolerance=3e-6)
            results.append({"checkpoint": path.relative_to(out).as_posix(), "task": design.task_key(spec),
                            "split": dev_role, **row, "status": "pass"})
        del model
        torch.cuda.empty_cache()
    platform.write_csv(out / "audit_development_forwards.csv", results)
    return {"development_checkpoints_recomputed": 2, "development_rows_recomputed": len(results),
            "test_neural_forwards_in_audit": 0}


def audit_budget_pairing(out):
    """Check the paired fresh starts, shared stream prefix, and fixed input pools.

    The separate schedule replay already checks every sampled object and slot for
    both budgets. Here we compare their common prefix and exposure tables. The
    short run's final partial batch, if present, is a prefix of a full long-run
    batch; its padded counts may therefore differ and are not compared directly.
    """
    rows = []
    common_identity_fields = ("initialization_seed", "data_seed", "test_seed",
        "pretrain_sampling_seed", "pretrain_dropout_seed", "initial_state_sha256")
    prefix_fields = ("update", "length", "batch_size", "input_tokens",
                     "padded_input_tokens", "supervised_tokens", "answer_tokens",
                     "cumulative_input_tokens")
    metric_fields = ("count", "correct", "accuracy", "answer_ce", "sequence_ce",
                     "eos_ce", "eos_accuracy", "eos_generated", "majority_accuracy", "majority_correct")
    for repeat in train.REPEATS:
        for chain in design.CHAINS:
            for k in (2, 16):
                directories = [out / "pretraining" / train.pretrain_id(repeat, chain, k, budget)
                               for budget in (4_000_000, 32_000_000)]
                short, long = [platform.read_json(path / "COMPLETE.json") for path in directories]
                check(all(short[field] == long[field] for field in common_identity_fields),
                      "paired source budgets differ in initialization or data/sampling/dropout seed")
                logs = [csv_rows(path / "updates.csv") for path in directories]
                short_full = len(logs[0]) - int(int(logs[0][-1]["batch_size"]) < 64)
                check(len(logs[1]) > len(logs[0]), "long run did not extend the sampling stream")
                for a, b in zip(logs[0][:short_full], logs[1][:short_full]):
                    check(all(a[field] == b[field] for field in prefix_fields),
                          "budget pair does not share the complete-batch sampling prefix")
                check(float(logs[0][0]["learning_rate"]) != float(logs[1][0]["learning_rate"]),
                      "budget-specific relative-progress schedules unexpectedly identical")
                equal_float(float(logs[0][0]["learning_rate"]) / float(logs[1][0]["learning_rate"]),
                            8., "initial warmup LR ratio between paired budgets")
                start_metrics = []
                for path in directories:
                    start_metrics.append({(r["source_task"], r["split"], r["length"]): r
                        for r in csv_rows(path / "source_curves.csv") if r["stage"] == "start"})
                check(set(start_metrics[0]) == set(start_metrics[1]), "budget-pair source start metric keys")
                for key, a in start_metrics[0].items():
                    for field in metric_fields:
                        equal_float(a[field], start_metrics[1][key][field], "paired initial model forward")
                exposure_tables = []
                for path in directories:
                    exposure_tables.append({(r["source_task"], r["length"], r["input_index"]): int(r["exposures"])
                                            for r in csv_rows(path / "input_exposures.csv")})
                check(set(exposure_tables[0]) == set(exposure_tables[1]), "budget pairs use different source input pools")
                check(all(exposure_tables[1][key] >= count for key, count in exposure_tables[0].items()),
                      "long-run exposure table is not an extension of the short-run stream")
                rows.append({"initialization_seed": repeat["initialization_seed"],
                    "data_seed": repeat["data_seed"], "sampling_seed": repeat["pretrain_sampling_seed"],
                    "chain": chain, "k": k, "short_run_id": short["run_id"], "long_run_id": long["run_id"],
                    "same_random_initial_state": True, "continued_from_short_endpoint": False,
                    "same_source_pool": True, "fixed_input_pool_size": 8192,
                    "common_full_batches": short_full,
                    "short_final_partial_batch": int(logs[0][-1]["batch_size"]) < 64,
                    "short_input_tokens": short["input_tokens"], "long_input_tokens": long["input_tokens"],
                    "actual_token_ratio": long["input_tokens"] / short["input_tokens"],
                    "short_unique_input_union": short["fixed_pool_unique_inputs_exposed"],
                    "long_unique_input_union": long["fixed_pool_unique_inputs_exposed"],
                    "short_first_update_lr": logs[0][0]["learning_rate"],
                    "long_first_update_lr": logs[1][0]["learning_rate"],
                    "learning_rate_trajectories_differ": True,
                    "all_task_input_exposures_monotone": True, "status": "pass"})
    platform.write_csv(out / "audit_budget_pairing.csv", rows)
    return {"fresh_initialization_budget_pairs": len(rows),
            "paired_budget_full_batch_prefixes_verified": len(rows),
            "fixed_source_pool_and_monotone_exposure_pairs_verified": len(rows),
            "independent_data_increase_claim": False,
            "budget_pairing_table_sha256": platform.sha(out / "audit_budget_pairing.csv")}


def source_distributions(out):
    """Full source-pool label summaries, independent of source evaluation subsets."""
    train.verify_freeze(out)
    started = time.monotonic()
    rows = []
    for repeat in train.REPEATS:
        for spec in design.SOURCE_POOL:
            training = train.pool(out, repeat, "source_train", spec)
            modes = independent_majority(training)
            for role in ("source_train", "source_dev"):
                by_length = training if role == "source_train" else train.pool(out, repeat, role, spec)
                for n, records in by_length.items():
                    histogram = Counter(int(r["answer_value"]) for r in records)
                    check(all(type(r["answer_value"]) is int for r in records), "source distribution expects prespecified scalar tasks")
                    mode_value = next(r["answer_value"] for r in training[n] if tuple(r["answer_ids"]) == modes[n])
                    rows.append({"seed": repeat["initialization_seed"], "data_seed": repeat["data_seed"],
                        "source_task": design.task_key(spec), "task_id": spec["task_id"],
                        "params": json.dumps(spec.get("params", {}), sort_keys=True),
                        "split": role, "length": n, "count": len(records),
                        "distinct_answers": len(histogram), "observed_constant": len(histogram) == 1,
                        "minimum_answer": min(histogram), "maximum_answer": max(histogram),
                        "largest_class_fraction": max(histogram.values()) / len(records),
                        "training_majority_answer": mode_value,
                        "training_majority_accuracy": histogram.get(mode_value, 0) / len(records),
                        "mean_answer_plus_eos_tokens": sum(len(r["answer_ids"]) for r in records) / len(records),
                        "mean_sequence_tokens": sum(len(r["input_ids"]) for r in records) / len(records),
                        "answer_histogram": json.dumps(dict(sorted(histogram.items())), separators=(",", ":")),
                        "scope": "complete source role pool; not the first256 training evaluation subset"})
        train.encoded_pool.cache_clear()
    check(len(rows) == len(design.SOURCE_POOL) * 3 * 2 * 2, "full source distribution cardinality")
    output = out / "source_answer_distributions.csv"
    platform.write_csv(output, rows)
    result = {"status": "pass", "completed_utc": train.utc(), "seconds": time.monotonic() - started,
              "source_instances": len(design.SOURCE_POOL), "rows": len(rows),
              "labels_summarized": sum(row["count"] for row in rows),
              "source_train_objects_per_length": 4096, "source_dev_objects_per_length": 256,
              "frozen_protocol_sha256": platform.sha(out / "FROZEN_PROTOCOL.json"),
              "primary_data_manifest_sha256": platform.sha(out / "data/manifest.json"),
              "csv_sha256": platform.sha(output), "test_inputs_opened": False}
    platform.write_json(out / "SOURCE_DISTRIBUTION_AUDIT.json", result)
    return result


def audit(out, forwards=False, include_test=True):
    started = time.monotonic()
    torch.set_num_threads(1)
    frozen = train.verify_freeze(out)
    freeze_hash = platform.sha(out / "FROZEN_PROTOCOL.json")
    expected_source, expected_target = set(), set()
    for stage, repeat, chain, k, budget, spec in train.formal_runs():
        rid = train.pretrain_id(repeat, chain, k, budget) if stage == "pretraining" else train.adaptation_id(repeat, chain, k, budget, spec)
        (expected_source if stage == "pretraining" else expected_target).add(rid)
    check(len(expected_source) == 24 and len(expected_target) == 54, "frozen study run cardinality")
    for stage, expected in (("pretraining", expected_source), ("adaptation", expected_target)):
        actual = {p.parent.name for p in (out / stage).glob("*/COMPLETE.json")}
        check(actual == expected, f"missing or extra completed runs in {stage}")
    check(platform.read_json(out / "TRAINING_COMPLETE.json")["frozen_protocol_sha256"] == freeze_hash, "training completion binding")
    rows, initial_hashes = [], {}
    for repeat in train.REPEATS:
        seed = repeat["initialization_seed"]
        initial = checkpoint(str(out / "initial" / f"seed{seed}.pt"))
        check(initial["frozen_protocol_sha256"] == freeze_hash, "initial checkpoint binding")
        check(initial["vocabulary"] == platform.vocabulary() and initial["config"] == train.CFG, "initial vocabulary/configuration binding")
        model = platform.make_model(seed, torch.device("cpu"))
        initial_hashes[seed] = platform.state_hash(model.state_dict())
        check(initial_hashes[seed] == initial["state_sha256"] == platform.state_hash(initial["model"]), "initial state reconstructible from seed")
        check(train.row_hash(initial["model"]) == initial["target_rows_sha256"], "initial target row digest")
        del model
        for stage, rep, chain, k, budget, spec in train.formal_runs():
            if rep["initialization_seed"] != seed:
                continue
            row = (audit_source(out, rep, chain, k, budget, initial, freeze_hash) if stage == "pretraining" else
                   audit_target(out, rep, chain, k, budget, spec, initial, freeze_hash))
            rows.append(row)
            print(f"[audit] {row['run_id']}: pass", flush=True)
        train.encoded_pool.cache_clear()
        checkpoint.cache_clear()
    platform.write_csv(out / "audit_run_accounting.csv", rows)
    pairing_result = audit_budget_pairing(out)
    test_result = audit_test(out, expected_target) if include_test else {"test_audit": "not requested", "test_neural_forwards_in_audit": 0}
    forwards_result = development_forwards(out) if forwards else {"development_checkpoints_recomputed": 0}
    result = {"status": "pass", "completed_utc": train.utc(), "seconds": time.monotonic() - started,
              "frozen_protocol_sha256": freeze_hash, "pretraining_runs": 24, "target_training_runs": 54,
              "initializations_independently_rebuilt": 3,
              "checkpoints_loaded_and_checked": 135,
              "source_schedules_replayed": 24, "adaptation_schedules_replayed": 54,
              "shared_object_stream_pairing": "Each source run was independently replayed from the same repeat-specific object RNG; token stopping can truncate different prefixes.",
              "target_pairing": "All 18 target fits per repeat independently reproduce one common length/index stream and one reset target-row initialization.",
              "optimizer_reset_evidence": "Frozen constructor and smoke, empty-state flags, and optimizer step counters equal each selected checkpoint's adaptation updates.",
              "source_evaluation_scope": "First 256 training objects per length, full 256 development objects per length; majority fit on all 4096 source training objects per length.",
              "pretraining_input_tokens": sum(r["input_tokens"] for r in rows if r["stage"] == "pretraining"),
              "adaptation_input_tokens": sum(r["input_tokens"] for r in rows if r["stage"] == "adaptation"),
              **pairing_result, **test_result, **forwards_result}
    distribution_audit = platform.read_json(out / "SOURCE_DISTRIBUTION_AUDIT.json")
    check(distribution_audit["frozen_protocol_sha256"] == freeze_hash, "full-source distribution protocol binding")
    check(distribution_audit["csv_sha256"] == platform.sha(out / "source_answer_distributions.csv"), "full-source distribution CSV binding")
    result["full_source_distribution_rows"] = distribution_audit["rows"]
    platform.write_json(out / "RUN_AUDIT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=train.DEFAULT_OUT)
    parser.add_argument("--forwards", action="store_true")
    parser.add_argument("--pretest-only", action="store_true")
    parser.add_argument("--distributions-only", action="store_true")
    args = parser.parse_args()
    result = source_distributions(args.out.resolve()) if args.distributions_only else audit(args.out.resolve(), args.forwards, not args.pretest_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
