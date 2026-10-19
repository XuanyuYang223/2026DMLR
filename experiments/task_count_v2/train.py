"""Frozen task-count experiment using PermWorld v0.1 exact sequence supervision.

Run smoke, freeze, train, select, and test in that order. Completed formal runs
are immutable and reusable; a partial run fails visibly rather than changing
its budget or silently resuming with a different sampling stream.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from permworld_platform import calibration as platform
from permworld_platform.encoding import encode_example
from experiments.task_count_v2 import design, data_pipeline

DEFAULT_OUT = ROOT / "experiments" / "task_count_v2"
REPEATS = [
    {"initialization_seed": 4103, "data_seed": 2026110101, "test_seed": 2026110151,
     "pretrain_sampling_seed": 2026110181, "adaptation_sampling_seed": 2026110191,
     "pretrain_dropout_seed": 2026110281, "adaptation_dropout_seed": 2026110291},
    {"initialization_seed": 5207, "data_seed": 2026110102, "test_seed": 2026110152,
     "pretrain_sampling_seed": 2026110182, "adaptation_sampling_seed": 2026110192,
     "pretrain_dropout_seed": 2026110282, "adaptation_dropout_seed": 2026110292},
    {"initialization_seed": 6301, "data_seed": 2026110103, "test_seed": 2026110153,
     "pretrain_sampling_seed": 2026110183, "adaptation_sampling_seed": 2026110193,
     "pretrain_dropout_seed": 2026110283, "adaptation_dropout_seed": 2026110293},
]
CFG = {
    "version": "permworld_task_count_v2",
    "repeats": REPEATS,
    "lengths": [10, 12],
    "k_values": [2, 16],
    "pretraining_budgets_nonpadding_sequence_tokens": [4_000_000, 32_000_000],
    "budget_pairing": "Both budgets start from the same paired random initialization with fresh optimizers, source objects, sampling and dropout seeds. Neither continues a short-budget checkpoint. The 32M stream extends the same sampled example stream, but a 4M tail can be a partial batch.",
    "fixed_input_pool": "4096 objects per length and repeat reused at both budgets; more tokens mean repeated exposure, not proportionally more independent data",
    "pretraining_checkpoint": "fixed token-budget endpoint only",
    "pretraining_schedule": "cosine in consumed nonpadding sequence tokens; warmup first 1/24 budget, floor 3e-5",
    "pretraining_sampling": "Each batch contains a concatenation of random task permutations; balanced within k slots. Independent length/index and task-slot RNGs. Same streams across chains at fixed k; final batch is a prefix fitting remaining tokens.",
    "pretraining_length_schedule": "Seeded random permutation of lengths every two updates; paired across chains at fixed k until token-budget stopping",
    "source_train_evaluation": "fixed first 256 training objects per length, for each task; majority computed from complete source training pool",
    "source_evaluation_stages": ["start", "midpoint", "endpoint"],
    "source_midpoint": "first completed update with consumed tokens >= half of that run's budget",
    "adaptation_updates": 800,
    "adaptation_samples_per_length": 128,
    "adaptation_eval_every": 100,
    "adaptation_warmup_updates": 100,
    "adaptation_selection": "maximum development complete-answer accuracy, minimum answer CE, earliest update; update zero excluded",
    "adaptation_learning_rate_candidates": [3e-4],
    "target_parameter_initialization": "All four held-out task-token rows of the tied embedding/output matrix are reset to the paired random initialization before every adaptation, including scratch. Other pretrained parameters are loaded exactly. Fresh AdamW state.",
    "batch_size": 64,
    "evaluation_batch_size": platform.CFG["evaluation_batch_size"],
    "learning_rate": 3e-4,
    "minimum_lr": 3e-5,
    "weight_decay": .01,
    "betas": [.9, .999],
    "epsilon": 1e-8,
    "gradient_clip": 1.,
    "model": copy.deepcopy(platform.CFG["model"]),
    "vocabulary_size": len(platform.vocabulary()),
    "max_new_tokens": platform.CFG["max_new_tokens"],
    "loss": platform.CFG["loss"],
    "accuracy": platform.CFG["accuracy"],
    "precision": platform.CFG["precision"],
    "test_policy": "Create independent test pools only after all 54 development selections have been frozen; report selected and fixed endpoint checkpoints, without further selection",
    "scope": {"pretraining_models": 24, "adaptations_from_pretraining": 48,
              "shared_random_initialization_baselines": 6, "total_target_training_runs": 54},
}
write_json = platform.write_json
read_json = platform.read_json
write_csv = platform.write_csv
sha = platform.sha
state_hash = platform.state_hash


def utc():
    return datetime.now(timezone.utc).isoformat()


def data_out(out):
    return out / "data"


def identity(repeat, chain="baseline", k=0, budget=0, spec=None):
    value = {**repeat, "seed": repeat["initialization_seed"], "chain": chain, "k": k,
             "budget_tokens": budget}
    value["condition"] = "A" if chain == "baseline" else "pretrained"
    if spec is not None:
        value.update(target=design.task_key(spec), task_id=spec["task_id"],
                     params=json.dumps(spec.get("params", {}), sort_keys=True))
    return value


def pretrain_id(repeat, chain, k, budget):
    return f"{chain}_k{k}_b{budget}_s{repeat['initialization_seed']}"


def adaptation_id(repeat, chain, k, budget, spec):
    return f"{chain}_k{k}_b{budget}_{design.task_key(spec)}_s{repeat['initialization_seed']}"


def new_optimizer(model, device):
    optimizer = platform.optimizer_for(model, device)
    assert len(optimizer.state) == 0
    return optimizer


def target_rows():
    return [platform.token_id(f"<{spec['task_id'].upper()}>") for spec in design.HELDOUT_TARGETS]


def row_hash(state):
    return state_hash({"target_rows": state["token_embedding.weight"][target_rows()]})


def initial_checkpoint(out, repeat, device):
    path = out / "initial" / f"seed{repeat['initialization_seed']}.pt"
    if path.exists():
        cp = platform.load_checkpoint(path)
        assert cp["frozen_protocol_sha256"] == sha(out / "FROZEN_PROTOCOL.json")
        assert state_hash(cp["model"]) == cp["state_sha256"]
        return cp, path
    model = platform.make_model(repeat["initialization_seed"], device)
    cp = {"model": model.state_dict(), "initialization_seed": repeat["initialization_seed"],
          "state_sha256": state_hash(model.state_dict()), "target_rows_sha256": row_hash(model.state_dict()),
          "frozen_protocol_sha256": sha(out / "FROZEN_PROTOCOL.json"),
          "vocabulary": platform.vocabulary(), "config": CFG}
    platform.save_checkpoint(path, cp)
    del model
    return platform.load_checkpoint(path), path


def reset_target_rows(model, initial):
    before = state_hash(model.state_dict())
    before_rows = row_hash(model.state_dict())
    initial_rows = initial["model"]["token_embedding.weight"][target_rows()]
    with torch.no_grad():
        model.token_embedding.weight[target_rows()] = initial_rows.to(model.token_embedding.weight.device)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
    assert row_hash(model.state_dict()) == row_hash(initial["model"])
    return {"loaded_state_sha256": before, "loaded_target_rows_sha256": before_rows,
            "adaptation_initial_state_sha256": state_hash(model.state_dict()),
            "reset_target_rows_sha256": row_hash(model.state_dict()), "reset_target_token_ids": target_rows()}


@functools.lru_cache(maxsize=192)
def encoded_pool(dataset, seed, role, task_id, params_json):
    spec_params = json.loads(params_json)
    result = {}
    for n in CFG["lengths"]:
        objects = data_pipeline.load_objects(Path(dataset), seed, role, n)
        records = [encode_example(task_id, tuple(map(int, p)), spec_params,
                                  max_seq_len=CFG["model"]["max_seq_len"]) for p in objects]
        assert records
        for r in records:
            assert r["input_ids"][r["prompt_length"]:] == r["answer_ids"]
            assert r["answer_ids"][-1] == platform.token_id("<EOS>")
            assert 1 < len(r["answer_ids"]) <= CFG["max_new_tokens"]
        result[n] = records
    return result


def pool(out, repeat, role, spec):
    return encoded_pool(str(data_out(out).resolve()), repeat["initialization_seed"], role,
                        spec["task_id"], json.dumps(spec.get("params", {}), sort_keys=True))


def schedule(seed, count_by_length, batch_size=64):
    rng = np.random.default_rng(seed)
    while True:
        for n in rng.permutation(CFG["lengths"]).tolist():
            yield int(n), rng.choice(count_by_length[int(n)], batch_size, replace=False)


def token_lr(tokens_after_update, budget):
    progress = min(tokens_after_update / budget, 1.)
    warmup = 1 / 24
    if progress <= warmup:
        return CFG["learning_rate"] * progress / warmup
    cosine_progress = (progress - warmup) / (1 - warmup)
    return CFG["minimum_lr"] + .5 * (CFG["learning_rate"] - CFG["minimum_lr"]) * (1 + math.cos(math.pi * cosine_progress))


def adaptation_lr(update):
    if update <= CFG["adaptation_warmup_updates"]:
        return CFG["learning_rate"] * update / CFG["adaptation_warmup_updates"]
    progress = (update - CFG["adaptation_warmup_updates"]) / (CFG["adaptation_updates"] - CFG["adaptation_warmup_updates"])
    return CFG["minimum_lr"] + .5 * (CFG["learning_rate"] - CFG["minimum_lr"]) * (1 + math.cos(math.pi * progress))


def zero_totals():
    return {key: 0 for key in ("input_tokens", "padded_input_tokens", "supervised_tokens", "answer_tokens", "sample_exposures")}


def update_totals(totals, step_result, count):
    for key in totals:
        totals[key] += count if key == "sample_exposures" else step_result[key]


def checked_complete(run_dir, out):
    path = run_dir / "COMPLETE.json"
    if not path.exists():
        return None
    result = read_json(path)
    assert result["frozen_protocol_sha256"] == sha(out / "FROZEN_PROTOCOL.json")
    for relative, digest in result["artifact_sha256"].items():
        assert sha(run_dir / relative) == digest, f"Completed artifact changed: {run_dir / relative}"
    return result


def begin_run(run_dir, out, info):
    if (run_dir / "STARTED.json").exists():
        raise RuntimeError(f"Incomplete run retained at {run_dir}; investigate before an explicit same-protocol restart")
    write_json(run_dir / "STARTED.json", {**info, "started_utc": utc(),
               "frozen_protocol_sha256": sha(out / "FROZEN_PROTOCOL.json")})


def run_pretraining(out, repeat, chain, k, budget, device):
    rid = pretrain_id(repeat, chain, k, budget)
    run_dir = out / "pretraining" / rid
    if checked_complete(run_dir, out):
        print(f"Reuse pretraining {rid}", flush=True)
        return
    info = {"run_id": rid, **identity(repeat, chain, k, budget)}
    begin_run(run_dir, out, info)
    specs = design.CHAINS[chain][:k]
    source = {design.task_key(s): {role: pool(out, repeat, role, s) for role in ("source_train", "source_dev")} for s in specs}
    majority = {key: platform.majority_answers(values["source_train"]) for key, values in source.items()}
    initial, initial_path = initial_checkpoint(out, repeat, device)
    model = platform.make_model(repeat["initialization_seed"], device)
    model.load_state_dict(initial["model"], strict=True)
    assert state_hash(model.state_dict()) == initial["state_sha256"]
    optimizer = new_optimizer(model, device)
    platform.seed_all(repeat["pretrain_dropout_seed"])
    counts = {n: len(next(iter(source.values()))["source_train"][n]) for n in CFG["lengths"]}
    stream = schedule(repeat["pretrain_sampling_seed"], counts)
    task_rng = np.random.default_rng(np.random.SeedSequence([repeat["pretrain_sampling_seed"], 1]))
    exposure = {key: {n: np.zeros(counts[n], dtype=np.int64) for n in CFG["lengths"]} for key in source}
    task_totals = {key: {n: zero_totals() for n in CFG["lengths"]} for key in source}
    totals, curves, updates_log = zero_totals(), [], []
    schedule_hash = hashlib.sha256()
    object_stream_hash = hashlib.sha256()
    task_slot_hash = hashlib.sha256()
    sample_stream_hash = hashlib.sha256()
    paired_prefix_hash = hashlib.sha256()
    paired_prefix_count = 0
    short_budget = min(CFG["pretraining_budgets_nonpadding_sequence_tokens"])
    paired_short = None
    if budget > short_budget:
        paired_short = checked_complete(out / "pretraining" / pretrain_id(repeat, chain, k, short_budget), out)
        assert paired_short is not None, "Complete the paired short-budget run first"
        assert paired_short["initial_state_sha256"] == initial["state_sha256"]
    started = time.monotonic()
    phase_seconds = {"training_updates": 0., "source_evaluation": 0.}
    torch.cuda.reset_peak_memory_stats(device)
    update, midpoint_done, next_sequence_tokens = 0, False, None

    def record(stage):
        evaluation_start = time.monotonic()
        for key, task_pools in source.items():
            for role in ("source_train", "source_dev"):
                records = {n: rows[:256] if role == "source_train" else rows for n, rows in task_pools[role].items()}
                for metric in platform.evaluate(model, records, majority[key], device):
                    metric_lengths = CFG["lengths"] if metric["length"] == "all" else [int(metric["length"])]
                    exposed_samples = sum(task_totals[key][n]["sample_exposures"] for n in metric_lengths)
                    unique_objects = sum(int(np.count_nonzero(exposure[key][n])) for n in metric_lengths)
                    curves.append({**info, "source_task": key, "stage": stage, "update": update,
                                   "input_tokens": totals["input_tokens"], "split": role,
                                   "source_sample_exposures": exposed_samples,
                                   "source_unique_inputs_exposed": unique_objects,
                                   "source_repeat_exposures": exposed_samples - unique_objects,
                                   "source_input_tokens": sum(task_totals[key][n]["input_tokens"] for n in metric_lengths),
                                   "source_supervised_tokens": sum(task_totals[key][n]["supervised_tokens"] for n in metric_lengths),
                                   "evaluation_scope": "first256_per_length" if role == "source_train" else "full_development",
                                   **metric, "elapsed_seconds": time.monotonic() - started})
        write_csv(run_dir / "source_curves.csv", curves)
        phase_seconds["source_evaluation"] += time.monotonic() - evaluation_start

    record("start")
    keys = list(source)
    while totals["input_tokens"] < budget:
        n, indices = next(stream)
        slots = np.concatenate([task_rng.permutation(k) for _ in range(CFG["batch_size"] // k)])
        remaining = budget - totals["input_tokens"]
        records, accepted = [], []
        batch_tokens = 0
        for index, slot in zip(indices.tolist(), slots.tolist()):
            key = keys[slot]
            row = source[key]["source_train"][n][index]
            row_tokens = len(row["input_ids"])
            if batch_tokens + row_tokens > remaining:
                next_sequence_tokens = row_tokens
                break
            records.append(row)
            accepted.append((key, index, slot))
            batch_tokens += row_tokens
        if not records:
            break
        update += 1
        update_start = time.monotonic()
        result = platform.step(model, optimizer, records, token_lr(totals["input_tokens"] + batch_tokens, budget), device)
        phase_seconds["training_updates"] += time.monotonic() - update_start
        assert result["input_tokens"] == batch_tokens
        update_totals(totals, result, len(records))
        for key, index, slot in accepted:
            row = source[key]["source_train"][n][index]
            exposure[key][n][index] += 1
            task_totals[key][n]["input_tokens"] += len(row["input_ids"])
            task_totals[key][n]["supervised_tokens"] += len(row["answer_ids"])
            task_totals[key][n]["answer_tokens"] += len(row["answer_ids"]) - 1
            task_totals[key][n]["sample_exposures"] += 1
            task_totals[key][n]["padded_input_tokens"] += max(len(r["input_ids"]) for r in records)
        stream_values = np.asarray([n, *[i for _, i, _ in accepted]], dtype="<i8").tobytes()
        object_stream_hash.update(stream_values)
        task_slot_hash.update(np.asarray([slot for _, _, slot in accepted], dtype="<i8").tobytes())
        sample_values = np.asarray([(n, index, slot) for _, index, slot in accepted], dtype="<i8")
        sample_stream_hash.update(sample_values.tobytes())
        if paired_short is not None and paired_prefix_count < paired_short["sample_exposures"]:
            take = min(len(accepted), paired_short["sample_exposures"] - paired_prefix_count)
            paired_prefix_hash.update(sample_values[:take].tobytes())
            paired_prefix_count += take
        schedule_hash.update(stream_values)
        schedule_hash.update(np.asarray([slot for _, _, slot in accepted], dtype="<i8").tobytes())
        updates_log.append({**info, "update": update, "length": n, "batch_size": len(records),
                            "learning_rate": token_lr(totals["input_tokens"], budget), **result,
                            "budget_fraction": totals["input_tokens"] / budget,
                            "cumulative_input_tokens": totals["input_tokens"],
                            "elapsed_seconds": time.monotonic() - started})
        if not midpoint_done and totals["input_tokens"] >= budget // 2:
            record("midpoint")
            midpoint_done = True
        if update % 200 == 0:
            print(json.dumps({"run": rid, "update": update, "tokens": totals["input_tokens"],
                              "loss": round(result["loss"], 6),
                              "seconds": round(time.monotonic() - started, 2)}), flush=True)
        if len(records) < CFG["batch_size"]:
            break
    tail = budget - totals["input_tokens"]
    assert 0 <= tail < (next_sequence_tokens or CFG["model"]["max_seq_len"])
    assert midpoint_done
    if paired_short is not None:
        assert paired_prefix_count == paired_short["sample_exposures"]
        assert paired_prefix_hash.hexdigest() == paired_short["sample_stream_sha256"]
    record("endpoint")
    platform.save_checkpoint(run_dir / "endpoint.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "identity": info, "updates": update, "tokens": totals, "config": CFG,
        "frozen_protocol_sha256": sha(out / "FROZEN_PROTOCOL.json"),
        "state_sha256": state_hash(model.state_dict()), "initial_state_sha256": initial["state_sha256"],
        "task_rng": task_rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all()})
    write_csv(run_dir / "updates.csv", updates_log)
    write_csv(run_dir / "input_exposures.csv", [{**info, "source_task": key, "length": n,
        "input_index": i, "exposures": int(amount)} for key, by_length in exposure.items()
        for n, values in by_length.items() for i, amount in enumerate(values)])
    task_rows = [{**info, "source_task": key, "length": n, **values,
                 "sample_fraction": values["sample_exposures"] / totals["sample_exposures"],
                 "unique_inputs_exposed": int(np.count_nonzero(exposure[key][n])),
                 "repeat_exposures": values["sample_exposures"] - int(np.count_nonzero(exposure[key][n])),
                 "pool_coverage": float(np.count_nonzero(exposure[key][n]) / counts[n]),
                 "mean_exposures_per_pool_input": values["sample_exposures"] / counts[n],
                 "pool_size": counts[n]} for key, by_length in task_totals.items() for n, values in by_length.items()]
    write_csv(run_dir / "task_exposures.csv", task_rows)
    counts_by_task = [sum(values[n]["sample_exposures"] for n in CFG["lengths"]) for values in task_totals.values()]
    assert max(counts_by_task) - min(counts_by_task) <= 1
    artifacts = ["endpoint.pt", "source_curves.csv", "updates.csv", "input_exposures.csv", "task_exposures.csv"]
    write_json(run_dir / "COMPLETE.json", {**info, "completed_utc": utc(), "seconds": time.monotonic() - started,
        "updates": update, **totals, "budget_tail_tokens": tail,
        "phase_seconds": phase_seconds,
        "next_unconsumed_sequence_tokens": next_sequence_tokens, "uniform_task_count_max_minus_min": max(counts_by_task) - min(counts_by_task),
        "initial_checkpoint": str(initial_path.relative_to(out)), "initial_state_sha256": initial["state_sha256"],
        "pretraining_started_from": "paired_random_initialization",
        "continued_from_short_budget": False,
        "learning_rate_budget_tokens": budget,
        "fixed_pool_unique_inputs_exposed": sum(int(np.count_nonzero(sum(exposure[key][n] for key in keys))) for n in CFG["lengths"]),
        "fixed_pool_size": sum(counts.values()),
        "target_rows_before_adaptation_sha256": row_hash(model.state_dict()),
        "length_index_and_slot_schedule_sha256": schedule_hash.hexdigest(),
        "length_index_schedule_sha256": object_stream_hash.hexdigest(), "task_slot_schedule_sha256": task_slot_hash.hexdigest(),
        "sample_stream_sha256": sample_stream_hash.hexdigest(),
        "paired_short_budget_sample_prefix": None if paired_short is None else {
            "budget_tokens": short_budget, "sample_exposures": paired_prefix_count,
            "sha256": paired_prefix_hash.hexdigest(), "matches_short_budget": True,
            "same_initialization": True,
            "note": "Exact sampled-example prefix; final short-budget partial batch can differ in batch grouping, and learning rates differ from the first update"},
        "optimizer_started_empty": True, "test_evaluated": False, "parameter_count": sum(p.numel() for p in model.parameters()),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "timer_scope": "training, source train/development evaluation, checkpoint and CSV writes; excludes data encoding",
        "frozen_protocol_sha256": sha(out / "FROZEN_PROTOCOL.json"),
        "artifact_sha256": {name: sha(run_dir / name) for name in artifacts}})
    print(f"Completed pretraining {rid}: {totals['input_tokens']} tokens, tail {tail}", flush=True)
    del model, optimizer, initial
    torch.cuda.empty_cache()


def run_adaptation(out, repeat, chain, k, budget, spec, device):
    rid = adaptation_id(repeat, chain, k, budget, spec)
    run_dir = out / "adaptation" / rid
    if checked_complete(run_dir, out):
        print(f"Reuse adaptation {rid}", flush=True)
        return
    info = {"run_id": rid, **identity(repeat, chain, k, budget, spec)}
    begin_run(run_dir, out, info)
    pools = {role: pool(out, repeat, role, spec) for role in ("target_train", "target_dev")}
    assert all(len(v) == 128 for v in pools["target_train"].values())
    majority = platform.majority_answers(pools["target_train"])
    initial, initial_path = initial_checkpoint(out, repeat, device)
    model = platform.make_model(repeat["initialization_seed"], device)
    parent_path = initial_path if chain == "baseline" else out / "pretraining" / pretrain_id(repeat, chain, k, budget) / "endpoint.pt"
    parent = platform.load_checkpoint(parent_path)
    model.load_state_dict(parent["model"], strict=True)
    assert state_hash(model.state_dict()) == state_hash(parent["model"])
    load_audit = reset_target_rows(model, initial)
    # Verify reset did not touch any parameter outside the four tied task rows.
    reset_state = model.state_dict()
    for name, original in parent["model"].items():
        expected = original.clone()
        if name in ("token_embedding.weight", "lm_head.weight"):
            expected[target_rows()] = initial["model"][name][target_rows()]
        assert torch.equal(reset_state[name].cpu(), expected.cpu()), name
    optimizer = new_optimizer(model, device)
    platform.seed_all(repeat["adaptation_dropout_seed"])
    stream = schedule(repeat["adaptation_sampling_seed"], {n: 128 for n in CFG["lengths"]})
    schedule_hash = hashlib.sha256()
    exposures = {n: np.zeros(128, dtype=np.int64) for n in CFG["lengths"]}
    length_updates = Counter()
    totals, curves, updates_log = zero_totals(), [], []
    best_key, best_step, best_dev = None, None, None
    started = time.monotonic()
    phase_seconds = {"training_updates": 0., "target_evaluation": 0.}
    torch.cuda.reset_peak_memory_stats(device)

    def record(update):
        evaluation_start = time.monotonic()
        dev = None
        for role in ("target_train", "target_dev"):
            for metrics in platform.evaluate(model, pools[role], majority, device):
                curves.append({**info, "update": update, "split": role, **metrics,
                               "input_tokens": totals["input_tokens"], "elapsed_seconds": time.monotonic() - started})
                if role == "target_dev" and metrics["length"] == "all":
                    dev = metrics
        write_csv(run_dir / "curves.csv", curves)
        phase_seconds["target_evaluation"] += time.monotonic() - evaluation_start
        return dev

    def checkpoint(update, dev):
        return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "update": update,
            "identity": info, "config": CFG, "development": dev, "tokens": totals.copy(),
            "frozen_protocol_sha256": sha(out / "FROZEN_PROTOCOL.json"),
            "initial_state_sha256": initial["state_sha256"], **load_audit,
            "parent_checkpoint": str(parent_path.relative_to(out)), "parent_checkpoint_sha256": sha(parent_path),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}

    record(0)
    for update in range(1, CFG["adaptation_updates"] + 1):
        n, indices = next(stream)
        records = [pools["target_train"][n][i] for i in indices]
        update_start = time.monotonic()
        result = platform.step(model, optimizer, records, adaptation_lr(update), device)
        phase_seconds["training_updates"] += time.monotonic() - update_start
        update_totals(totals, result, len(records))
        exposures[n][indices] += 1
        length_updates[n] += 1
        schedule_hash.update(np.asarray([n, *indices], dtype="<i8").tobytes())
        updates_log.append({**info, "update": update, "length": n, "learning_rate": adaptation_lr(update),
                           **result, "cumulative_input_tokens": totals["input_tokens"]})
        if update % CFG["adaptation_eval_every"] == 0:
            dev = record(update)
            key = (-dev["accuracy"], dev["answer_ce"], update)
            if best_key is None or key < best_key:
                best_key, best_step, best_dev = key, update, dev
                platform.save_checkpoint(run_dir / "best.pt", checkpoint(update, dev))
            if update % 200 == 0:
                print(json.dumps({"run": rid, "update": update, "tokens": totals["input_tokens"],
                    "loss": round(result["loss"], 6), "dev_accuracy": dev["accuracy"],
                    "seconds": round(time.monotonic() - started, 2)}), flush=True)
    assert dict(length_updates) == {n: CFG["adaptation_updates"] // 2 for n in CFG["lengths"]}
    platform.save_checkpoint(run_dir / "endpoint.pt", checkpoint(update, dev))
    write_csv(run_dir / "updates.csv", updates_log)
    write_csv(run_dir / "input_exposures.csv", [{**info, "length": n, "input_index": i, "exposures": int(amount)}
              for n, values in exposures.items() for i, amount in enumerate(values)])
    artifacts = ["best.pt", "endpoint.pt", "curves.csv", "updates.csv", "input_exposures.csv"]
    write_json(run_dir / "COMPLETE.json", {**info, "completed_utc": utc(), "seconds": time.monotonic() - started,
        "updates": update, **totals, "best_update": best_step, "best_dev": best_dev, "endpoint_dev": dev,
        "phase_seconds": phase_seconds,
        "length_updates": dict(length_updates), "initial_state_sha256": initial["state_sha256"], **load_audit,
        "parent_checkpoint": str(parent_path.relative_to(out)), "parent_checkpoint_sha256": sha(parent_path),
        "length_index_schedule_sha256": schedule_hash.hexdigest(), "optimizer_started_empty": True,
        "optimizer_initial_state_entries": 0, "test_evaluated": False,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "timer_scope": "training, adaptation train/development evaluation, checkpoint and CSV writes; excludes data encoding",
        "frozen_protocol_sha256": sha(out / "FROZEN_PROTOCOL.json"),
        "artifact_sha256": {name: sha(run_dir / name) for name in artifacts}})
    print(json.dumps({"run": rid, "completed": True, "best_update": best_step,
                      "dev_best_accuracy": best_dev["accuracy"], "dev_endpoint_accuracy": dev["accuracy"],
                      "seconds": round(time.monotonic() - started, 2)}), flush=True)
    del model, optimizer, initial, parent
    torch.cuda.empty_cache()


def formal_runs():
    for repeat in REPEATS:
        for chain in design.CHAINS:
            for k in CFG["k_values"]:
                for budget in CFG["pretraining_budgets_nonpadding_sequence_tokens"]:
                    yield "pretraining", repeat, chain, k, budget, None
        for spec in design.TARGETS:
            yield "adaptation", repeat, "baseline", 0, 0, spec
        for chain in design.CHAINS:
            for k in CFG["k_values"]:
                for budget in CFG["pretraining_budgets_nonpadding_sequence_tokens"]:
                    for spec in design.TARGETS:
                        yield "adaptation", repeat, chain, k, budget, spec


def train_all(out, device):
    verify_freeze(out)
    start_path = out / "TRAINING_STARTED.json"
    if not start_path.exists():
        write_json(start_path, {"started_utc": utc(), "frozen_protocol_sha256": sha(out / "FROZEN_PROTOCOL.json"),
            "environment": environment(device), "scope": CFG["scope"]})
    started = time.monotonic()
    for stage, repeat, chain, k, budget, spec in formal_runs():
        rid = pretrain_id(repeat, chain, k, budget) if stage == "pretraining" else adaptation_id(repeat, chain, k, budget, spec)
        try:
            if stage == "pretraining":
                run_pretraining(out, repeat, chain, k, budget, device)
            else:
                run_adaptation(out, repeat, chain, k, budget, spec, device)
        except Exception:
            write_json(out / stage / rid / f"FAILURE_{int(time.time())}.json",
                {"utc": utc(), "traceback": traceback.format_exc(), "no_budget_extension": True})
            raise
    verify_freeze(out)
    write_json(out / "TRAINING_COMPLETE.json", {"completed_utc": utc(), "stage_seconds": time.monotonic() - started,
        "scope": CFG["scope"], "frozen_protocol_sha256": sha(out / "FROZEN_PROTOCOL.json"), "test_evaluated": False})


def environment(device):
    return {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "parameter_count": 3244288, "encoding_version": platform.encoding().ENCODING_VERSION}


def freeze(out):
    if (out / "FROZEN_PROTOCOL.json").exists():
        verify_freeze(out)
        return
    smoke_result = read_json(out / "SMOKE.json")
    assert smoke_result["status"] == "passed"
    assert not (out / "TRAINING_STARTED.json").exists()
    quality = data_pipeline.verify(data_out(out), include_test=False)
    files = sorted((ROOT / "permworld_platform").glob("*.py"))
    files += [Path(__file__).resolve(), Path(design.__file__).resolve(), Path(data_pipeline.__file__).resolve(),
              ROOT / "experiments" / "task_count_v1" / "design.py",
              platform.VENDOR / "neurips_permutations" / "models.py", platform.VENDOR / "sdpa.py",
              platform.VENDOR / "neurips_permutations" / "math_ops.py",
              platform.VENDOR / "neurips_permutations" / "passage.py"]
    evidence = [out / "SMOKE.json", out / "EXPERIMENT_PLAN.md"]
    if (out / "ENVIRONMENT.json").exists():
        evidence.append(out / "ENVIRONMENT.json")
    evidence += sorted((out / "design").glob("*.json"))
    evidence += sorted(data_out(out).rglob("*.npz"))
    evidence += [p for p in sorted(data_out(out).glob("*.json")) if "test" not in p.name.lower() and "quality" not in p.name.lower() and "verify" not in p.name.lower()]
    assert any("manifest" in p.name.lower() for p in evidence), "Missing training data manifest"
    assert (out / "design" / "FROZEN_DESIGN.json").exists()
    write_json(out / "FROZEN_PROTOCOL.json", {
        "status": "frozen", "frozen_at_utc": utc(), "config": CFG,
        "target_specs": design.TARGETS, "heldout_target_specs_for_reset": design.HELDOUT_TARGETS, "chains": design.CHAINS,
        "vocabulary": platform.vocabulary(), "data_config": data_pipeline.DATA_CONFIG,
        "code_sha256": {os.path.relpath(p, out).replace("\\", "/"): sha(p) for p in files},
        "design_and_data_sha256": {os.path.relpath(p, out).replace("\\", "/"): sha(p) for p in evidence},
        "test_generated": False, "formal_training_started": False,
        "hypothesis_scope": "Two v1 nested chains at k=2/16, budgets 4M/32M from paired initializations, primary LIS and auxiliary fixed points, three new paired repeats; fixed scope with no extensions"})


def verify_freeze(out):
    value = read_json(out / "FROZEN_PROTOCOL.json")
    assert value["status"] == "frozen"
    assert value["config"] == CFG
    assert value["target_specs"] == design.TARGETS and value["chains"] == design.CHAINS
    assert value["heldout_target_specs_for_reset"] == design.HELDOUT_TARGETS
    assert value["data_config"] == data_pipeline.DATA_CONFIG
    assert value["vocabulary"] == platform.vocabulary()
    for group in ("code_sha256", "design_and_data_sha256"):
        for relative, digest in value[group].items():
            assert sha(out / relative) == digest, f"Frozen artifact changed: {relative}"
    return value


def select(out):
    verify_freeze(out)
    assert (out / "TRAINING_COMPLETE.json").exists()
    selections = []
    for stage, repeat, chain, k, budget, spec in formal_runs():
        rid = pretrain_id(repeat, chain, k, budget) if stage == "pretraining" else adaptation_id(repeat, chain, k, budget, spec)
        run_dir = out / stage / rid
        result = checked_complete(run_dir, out)
        assert result is not None, rid
        if stage == "adaptation":
            selections.append({"run_id": rid, **identity(repeat, chain, k, budget, spec),
                "selected_update": result["best_update"], "endpoint_update": CFG["adaptation_updates"],
                "best_development": result["best_dev"], "endpoint_development": result["endpoint_dev"],
                "selected_checkpoint": str((run_dir / "best.pt").relative_to(out)),
                "selected_checkpoint_sha256": result["artifact_sha256"]["best.pt"],
                "endpoint_checkpoint": str((run_dir / "endpoint.pt").relative_to(out)),
                "endpoint_checkpoint_sha256": result["artifact_sha256"]["endpoint.pt"]})
    assert len(selections) == 54
    path = out / "FROZEN_SELECTIONS.json"
    if path.exists():
        existing = read_json(path)
        assert existing["selections"] == selections
    else:
        write_json(path, {"status": "frozen", "training_complete": True, "frozen_at_utc": utc(),
                         "frozen_protocol_sha256": sha(out / "FROZEN_PROTOCOL.json"),
                         "selection_rule": CFG["adaptation_selection"], "selections": selections,
                         "test_generated_before_selection": False})
    print(f"Frozen {len(selections)} development-only selections", flush=True)


def test(out, device):
    verify_freeze(out)
    selected = read_json(out / "FROZEN_SELECTIONS.json")
    assert selected["status"] == "frozen" and selected["training_complete"] is True
    assert len(selected["selections"]) == 54
    data_pipeline.generate_test(data_out(out), out / "FROZEN_SELECTIONS.json")
    data_pipeline.verify(data_out(out), include_test=True)
    complete_path = out / "TEST_COMPLETE.json"
    if complete_path.exists():
        completed = read_json(complete_path)
        assert completed["frozen_selection_sha256"] == sha(out / "FROZEN_SELECTIONS.json")
        assert completed["raw_results_sha256"] == sha(out / "raw_test_results.csv")
        print("Reuse completed independent test evaluation", flush=True)
        return
    started = time.monotonic()
    all_rows = []
    for entry in selected["selections"]:
        repeat = next(r for r in REPEATS if r["initialization_seed"] == entry["initialization_seed"])
        spec = next(s for s in design.TARGETS if design.task_key(s) == entry["target"])
        test_pools = pool(out, repeat, "test", spec)
        majority = platform.majority_answers(pool(out, repeat, "target_train", spec))
        result_path = out / "test_results" / f"{entry['run_id']}.json"
        if result_path.exists():
            saved = read_json(result_path)
            assert saved["frozen_selection_sha256"] == sha(out / "FROZEN_SELECTIONS.json")
            all_rows.extend(saved["rows"])
            continue
        model = platform.make_model(repeat["initialization_seed"], device)
        rows = []
        for checkpoint_name, path_key, sha_key, step_key in (
                ("development_selected", "selected_checkpoint", "selected_checkpoint_sha256", "selected_update"),
                ("fixed_endpoint", "endpoint_checkpoint", "endpoint_checkpoint_sha256", "endpoint_update")):
            path = out / entry[path_key]
            assert sha(path) == entry[sha_key]
            checkpoint = platform.load_checkpoint(path)
            model.load_state_dict(checkpoint["model"], strict=True)
            for metrics in platform.evaluate(model, test_pools, majority, device):
                rows.append({"run_id": entry["run_id"], **identity(repeat, entry["chain"], entry["k"], entry["budget_tokens"], spec),
                             "checkpoint": checkpoint_name, "selected_update": entry[step_key],
                             "development_accuracy": entry["best_development" if checkpoint_name == "development_selected" else "endpoint_development"]["accuracy"],
                             "split": "test", **metrics})
        write_json(result_path, {"completed_utc": utc(), "rows": rows,
            "frozen_selection_sha256": sha(out / "FROZEN_SELECTIONS.json"), "development_only_selection": True})
        all_rows.extend(rows)
        print(f"Test evaluated {entry['run_id']}", flush=True)
        del model
    assert len(all_rows) == 54 * 2 * 3
    write_csv(out / "raw_test_results.csv", all_rows)
    write_json(complete_path, {"status": "passed", "completed_utc": utc(), "seconds": time.monotonic() - started,
        "adaptation_runs": 54, "checkpoint_evaluations": 108, "result_rows": len(all_rows),
        "frozen_selection_sha256": sha(out / "FROZEN_SELECTIONS.json"),
        "raw_results_sha256": sha(out / "raw_test_results.csv"), "no_test_based_selection": True})


def smoke_two_budgets(out, device):
    """Use production pretrain/adapt functions at tiny fixed budgets, never tests."""
    mini_out = out / "smoke" / "end_to_end"
    if (mini_out / "COMPLETE.json").exists():
        return read_json(mini_out / "COMPLETE.json")
    saved_cfg = copy.deepcopy(CFG)
    mini_budgets = [4_000, 32_000]
    repeat = REPEATS[0]
    chain = next(iter(design.CHAINS))
    started = time.monotonic()
    try:
        CFG["pretraining_budgets_nonpadding_sequence_tokens"] = mini_budgets
        CFG["adaptation_updates"] = 8
        CFG["adaptation_eval_every"] = 4
        CFG["adaptation_warmup_updates"] = 1
        # These are byte-identical copies of pretest inputs. Smoke never opens
        # the final test, and neither smoke weights nor losses affect selection.
        manifest = read_json(data_out(out) / "manifest.json")
        data_out(mini_out).mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_out(out) / "manifest.json", data_out(mini_out) / "manifest.json")
        for record in manifest["pools"]:
            if record["model_seed"] == repeat["initialization_seed"]:
                destination = data_out(mini_out) / record["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(data_out(out) / record["path"], destination)
        write_json(mini_out / "FROZEN_PROTOCOL.json", {
            "status": "smoke_only", "formal_data_selection": False,
            "budgets": mini_budgets, "adaptation_updates": 8})
        for budget in mini_budgets:
            run_pretraining(mini_out, repeat, chain, 2, budget, device)
        for spec in design.TARGETS:
            run_adaptation(mini_out, repeat, "baseline", 0, 0, spec, device)
            for budget in mini_budgets:
                run_adaptation(mini_out, repeat, chain, 2, budget, spec, device)
        pretraining = [checked_complete(mini_out / "pretraining" / pretrain_id(repeat, chain, 2, b), mini_out)
                       for b in mini_budgets]
        assert pretraining[0]["initial_state_sha256"] == pretraining[1]["initial_state_sha256"]
        assert pretraining[1]["paired_short_budget_sample_prefix"]["matches_short_budget"]
        adaptations = [checked_complete(path.parent, mini_out) for path in sorted((mini_out / "adaptation").glob("*/COMPLETE.json"))]
        assert len(adaptations) == 6
        assert len({a["reset_target_rows_sha256"] for a in adaptations}) == 1
        assert len({a["length_index_schedule_sha256"] for a in adaptations}) == 1
        assert all(a["optimizer_started_empty"] for a in adaptations)
        result = {"status": "passed", "seconds": time.monotonic() - started,
            "pretraining_models": 2, "adaptations": len(adaptations),
            "tiny_budgets": mini_budgets, "paired_initialization": True,
            "long_budget_sample_prefix_matches_short": True,
            "baseline_and_pretrained_adaptation_schedules_match": True,
            "four_heldout_rows_reset_identically": True, "fresh_optimizers": True,
            "development_selection_and_complete_answers_checked": True,
            "production_functions_exercised": ["run_pretraining", "run_adaptation"],
            "test_inputs_opened": False,
            "artifact_root": str(mini_out.relative_to(out))}
        write_json(mini_out / "COMPLETE.json", result)
        return result
    finally:
        CFG.clear()
        CFG.update(saved_cfg)


def smoke(out, device):
    """Small isolated protocol checks and fixed 32-update timing; no formal training."""
    if (out / "FROZEN_PROTOCOL.json").exists():
        raise RuntimeError("Do not rewrite smoke evidence after freezing")
    started = time.monotonic()
    rng = np.random.default_rng(917301)
    specs = design.CHAINS[next(iter(design.CHAINS))]
    assert len(specs) == 16
    samples_by_k = {}
    for k in CFG["k_values"]:
        records = [encode_example(s["task_id"], tuple(map(int, rng.permutation(12) + 1)), s.get("params", {}),
                                  max_seq_len=128) for s in specs[:k]]
        samples_by_k[k] = records
        for row in records:
            assert platform.encoding().decode_answer_ids(row["answer_ids"]) == row["answer_value"]
    records = [encode_example("increasing_subsequence_count", tuple(range(1, 13)), {"k": 3}),
               encode_example("increasing_subsequence_count", tuple(range(12, 0, -1)), {"k": 3}),
               *samples_by_k[16]]
    assert records[0]["answer_value"] == 220 and records[1]["answer_value"] == 0
    assert len(records[0]["answer_ids"]) > len(records[1]["answer_ids"])
    model = platform.make_model(917301, device)
    initial = {"model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
    optimizer = new_optimizer(model, device)
    for k in CFG["k_values"]:
        platform.step(model, optimizer, samples_by_k[k], 3e-4, device)
    # The exact per-example loss must differ from a token-global average when
    # example answer lengths differ. Check the formula with independent loops.
    model.eval()
    ids, valid, targets, _, _ = platform.collate(records, device)
    with torch.no_grad():
        logits = model(ids, attention_mask=valid)
        losses = platform.token_loss(logits, targets)
        actual = (losses.sum(1) / (targets != -100).sum(1)).mean()
        independent = torch.stack([torch.nn.functional.cross_entropy(logits[i, targets[i] != -100].float(),
                    targets[i, targets[i] != -100]) for i in range(len(records))]).mean()
        assert torch.allclose(actual, independent, atol=1e-6)
        corruption = ids.clone()
        for i, row in enumerate(records):
            corruption[i, row["prompt_length"]:] = platform.token_id("<BOS>")
        corrupted_logits = model(corruption, attention_mask=valid)
        for i, row in enumerate(records):
            assert torch.equal(logits[i, :row["prompt_length"]], corrupted_logits[i, :row["prompt_length"]])
        padded = torch.cat([ids, torch.full((len(records), 3), platform.token_id("<PAD>"), device=device)], dim=1)
        padded_valid = torch.cat([valid, torch.zeros((len(records), 3), dtype=torch.bool, device=device)], dim=1)
        assert torch.allclose(logits, model(padded, attention_mask=padded_valid)[:, :ids.shape[1]], atol=1e-5, rtol=1e-5)
    platform.seed_all(917302)
    platform.step(model, optimizer, records, 3e-4, device)
    checkpoint_path = out / "smoke" / "resume.pt"
    platform.save_checkpoint(checkpoint_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                             **platform.rng_state(rng)})
    platform.step(model, optimizer, records, 3e-4, device)
    expected = state_hash(model.state_dict())
    restored = platform.make_model(917301, device)
    cp = platform.load_checkpoint(checkpoint_path)
    restored.load_state_dict(cp["model"], strict=True)
    restored_optimizer = new_optimizer(restored, device)
    restored_optimizer.load_state_dict(cp["optimizer"])
    platform.restore_rng(cp)
    platform.step(restored, restored_optimizer, records, 3e-4, device)
    assert state_hash(restored.state_dict()) == expected
    reset = reset_target_rows(restored, initial)
    optimizer = new_optimizer(restored, device)
    target = design.TARGETS[0]
    target_records = [encode_example(target["task_id"], tuple(map(int, rng.permutation(12) + 1)), target.get("params", {})) for _ in range(8)]
    platform.step(restored, optimizer, target_records, 3e-4, device)
    restored.eval()
    contaminated = copy.deepcopy(target_records)
    for row in contaminated:
        row["input_ids"][row["prompt_length"]:] = [platform.token_id("<BOS>")] * len(row["answer_ids"])
        row["answer_ids"] = [platform.token_id("<BOS>")] * len(row["answer_ids"])
    assert platform.greedy_answers(restored, target_records, device) == platform.greedy_answers(restored, contaminated, device)
    torch_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    toy_metrics = platform.evaluate(restored, {12: target_records}, platform.majority_answers({12: target_records}), device)
    assert torch.equal(torch_before, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))
    # Timing is fixed in advance and cannot change the formal model or budget.
    timing_records = (samples_by_k[16] * 4)
    if device.type == "cuda":
        torch.cuda.synchronize()
    timed_start = time.monotonic()
    timing_tokens = 0
    for _ in range(32):
        timing_tokens += platform.step(restored, optimizer, timing_records, 3e-4, device)["input_tokens"]
    if device.type == "cuda":
        torch.cuda.synchronize()
    timing_seconds = time.monotonic() - timed_start
    del restored, model, optimizer, restored_optimizer
    torch.cuda.empty_cache()
    lr_checks = []
    for fraction in (0., 1 / 24, .125, .5, 1.):
        small_lr = token_lr(fraction * 4_000_000, 4_000_000)
        large_lr = token_lr(fraction * 32_000_000, 32_000_000)
        assert math.isclose(small_lr, large_lr, rel_tol=1e-12, abs_tol=1e-15)
        lr_checks.append({"budget_fraction": fraction, "learning_rate": small_lr})
    assert token_lr(4_000_000, 32_000_000) > token_lr(4_000_000, 4_000_000)
    end_to_end = smoke_two_budgets(out, device)
    write_json(out / "SMOKE.json", {"status": "passed", "completed_utc": utc(), "seconds": time.monotonic() - started,
        "environment": environment(device), "k_values_checked": CFG["k_values"],
        "parameterized_samples_checked": True, "multi_token_answer_value": 220,
        "multi_token_answer_ids": records[0]["answer_ids"], "per_example_answer_plus_eos_loss_checked": True,
        "save_resume_bitwise": True, "causal_answer_leakage_absent": True, "explicit_padding_checked": True,
        "generation_independent_of_gold": True, "evaluation_preserves_rng": True,
        "fresh_adaptation_optimizer": True, "target_row_reset": reset,
        "toy_adaptation_and_evaluation": toy_metrics, "formal_test_opened": False,
        "two_budget_production_pipeline": end_to_end,
        "learning_rate_relative_budget_checks": lr_checks,
        "learning_rate_4M_tokens_short_budget": token_lr(4_000_000, 4_000_000),
        "learning_rate_4M_tokens_long_budget": token_lr(4_000_000, 32_000_000),
        "timing_updates": 32, "timing_seconds": timing_seconds, "timing_input_tokens": timing_tokens,
        "timing_tokens_per_second": timing_tokens / timing_seconds,
        "estimated_4M_training_only_seconds": 4_000_000 * timing_seconds / timing_tokens,
        "estimated_32M_training_only_seconds": 32_000_000 * timing_seconds / timing_tokens,
        "estimated_all_pretraining_training_only_seconds": 12 * 36_000_000 * timing_seconds / timing_tokens,
        "timing_note": "Fixed smoke batch, training only; excludes formal evaluations, encoding, checkpoint and file I/O"})
    print(json.dumps({"smoke": "passed", "timing_seconds": timing_seconds,
                      "tokens_per_second": timing_tokens / timing_seconds}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["smoke", "freeze", "train", "select", "test", "verify"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.stage == "freeze":
        freeze(out)
    elif args.stage == "verify":
        verify_freeze(out)
        print("Frozen task-count protocol verified")
    elif args.stage == "select":
        select(out)
    else:
        device = platform.runtime(args.device)
        if args.stage in ("train", "test") and device.type != "cuda":
            raise RuntimeError("Formal training/evaluation uses the frozen CUDA configuration")
        {"smoke": smoke, "train": train_all, "test": test}[args.stage](out, device)


if __name__ == "__main__":
    main()
