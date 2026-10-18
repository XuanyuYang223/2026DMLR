"""CPU-only v3 reuse inventory for the finite two-target v4 background study.

Reads saved weights, training/development data and metrics; never reads tests
or calls a model. Existing v1/v2/v3 files are never written.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V3 = ROOT / "experiments/pilot_v3_multitarget/artifacts"
OLD = ROOT / "experiments/permworld_pilot/runs/pilot_v1"
sys.path.insert(0, str(ROOT / "experiments/permworld_pilot/vendor"))
from neurips_permutations.passage import TOKEN_TO_ID

TARGETS = ["fixed_points", "longest_decreasing_run"]
BACKGROUNDS = ["g0_v3", "g1_recoils", "g2_deficiencies"]
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
    for key, value in sorted(state.items()):
        if token is not None and key in ["token_embedding.weight", "lm_head.weight"]:
            value = value.clone()
            value[token].copy_(initial_row)
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def csvwrite(path, values):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(dict.fromkeys(key for row in values for key in row)))
        writer.writeheader()
        writer.writerows(values)


def rel(path):
    return os.path.relpath(path.resolve(), ROOT).replace("\\", "/")


def replay(seed, task_count, count, updates, offset):
    rng = np.random.default_rng(seed + offset)
    digest = hashlib.sha256()
    for _ in range(updates):
        indices = rng.choice(count, 64, replace=False)
        slots = np.tile(np.arange(task_count), 64 // task_count)
        rng.shuffle(slots)
        digest.update(indices.tobytes())
        digest.update(slots.tobytes())
    return digest.hexdigest(), rng.bit_generator.state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    frozen = read(V3 / "FROZEN_PROTOCOL.json")
    cfg = frozen["config"]
    assert cfg["seeds"] == [17, 42, 73] and cfg["pretrain_updates"] == 4800
    assert cfg["adaptation_updates"] == 800 and cfg["learning_rates"] == [.0003, .0001, .00003]
    assert read(V3 / "FINAL_AUDIT.json")["status"] == "passed"
    assert sha(V3 / "DESIGN_FROZEN.json") == frozen["design_sha256"]
    for path, expected in read(V3 / "DESIGN_FROZEN.json")["code"].items():
        assert sha(ROOT / path) == expected, path
    data_ready = read(V3 / "DATA_READY.json")
    data_records = []
    for row in data_ready["pools"]:
        path = V3 / "data" / f"s{row['seed']}_{row['role']}.npz"
        assert sha(path) == row["sha256"]
        with np.load(path, allow_pickle=False) as archive:
            p, y = archive["p"], archive["y"]
        assert hashlib.sha256(p.tobytes()).hexdigest() == row["input_sha256"]
        assert p.shape == (row["count"], 10) and y.shape == (row["count"], 32)
        data_records.append(dict(**row, path=rel(path), data_generation_seed=20261018 if row["seed"] in [17, 42] else 2026101803,
                                 data_generation_stream="shared v1 stream, role order" if row["seed"] in [17, 42] else "v3 new-data stream; only seed73 roles generated"))
    data_hash = {(row["seed"], row["role"]): row["sha256"] for row in data_records}
    checkpoints, shapes = {}, None

    def load(path, expected_sha=None, expected_steps=None):
        nonlocal shapes
        path = path.resolve()
        actual_sha = sha(path)
        if expected_sha:
            assert actual_sha == expected_sha, path
        cp = torch.load(path, map_location="cpu", weights_only=False)
        state = cp["model"]
        actual_shapes = {k: tuple(v.shape) for k, v in state.items()}
        if shapes is None:
            shapes = actual_shapes
        assert shapes == actual_shapes and actual_shapes["token_embedding.weight"] == (188, 256)
        assert torch.equal(state["token_embedding.weight"], state["lm_head.weight"])
        assert all(not v.is_floating_point() or bool(torch.isfinite(v).all()) for v in state.values())
        steps = sorted({int(s["step"]) for s in cp.get("optimizer", {}).get("state", {}).values()})
        if expected_steps is not None:
            assert cp["step"] == expected_steps and steps == [expected_steps], path
        digest = state_hash(state)
        item = dict(path=rel(path), sha256=actual_sha, state_sha256=digest,
                    step=cp.get("step", 0), optimizer_steps=";".join(map(str, steps)), cpu_loaded=True,
                    tied_embedding_values_equal=True)
        if path in checkpoints:
            assert checkpoints[path] == item
        checkpoints[path] = item
        return cp, digest

    initials, init_hashes = {}, {}
    seed_records = []
    for seed in cfg["seeds"]:
        cp, init_hashes[seed] = load(V3 / "initial" / f"seed{seed}.pt", data_ready["initials"][str(seed)])
        initials[seed] = cp["model"]
        seed_records.append(dict(seed=seed, model_initialization_seed=seed,
            training_data_generation_seed=20261018 if seed in [17, 42] else 2026101803,
            source_sample_rng_seed=seed + 1000, source_torch_cpu_dropout_seed=seed + 1000,
            source_cuda_dropout_seed=seed + 1000, adaptation_sample_rng_seed=seed + 2000,
            adaptation_torch_cpu_dropout_seed=seed + 2000, adaptation_cuda_dropout_seed=seed + 2000,
            initial_path=rel(V3 / "initial" / f"seed{seed}.pt"), initial_sha256=data_ready["initials"][str(seed)],
            initial_state_sha256=init_hashes[seed],
            note="Data streams are not reseeded by model seed. No target-specific or LR-specific sampler/dropout reseed."))
    source_schedules = {seed: replay(seed, 4, 8192, 4800, 1000) for seed in cfg["seeds"]}
    adapt_schedules = {seed: replay(seed, 1, 256, 800, 2000) for seed in cfg["seeds"]}
    groups = {g["target"]: g for g in frozen["groups"]}
    source_rows, source_lookup, source_states, composition_index = [], {}, {}, {}
    for meta_path in sorted((V3 / "multi").glob("*/completed.json")):
        meta = read(meta_path)
        seed = meta["seed"]
        target, condition = next((t, c) for t in cfg["targets"] for c in ["B", "C"]
                                 if meta_path.parent.name == f"s{seed}_{t}_{c}")
        tasks = tuple(meta["tasks"])
        assert list(tasks) == groups[target]["sources"][condition]
        assert meta["updates"] == 4800 and meta["task_fraction"] == .25 and meta["task_exposures"] == 76800
        assert meta["schedule_sha256"] == source_schedules[seed][0]
        assert meta["initial_state_hash"] == init_hashes[seed]
        assert meta["train_sha256"] == data_hash[(seed, "source_train")]
        assert meta["dev_sha256"] == data_hash[(seed, "source_dev")]
        path = (V3 / meta["checkpoint"]).resolve()
        cp, digest = load(path, meta["checkpoint_sha256"], 4800)
        assert digest == meta["model_state_hash"] and cp["numpy_rng"] == source_schedules[seed][1]
        assert cp["config"]["model"] == cfg["model"]
        assert math.isclose(cp["optimizer"]["param_groups"][0]["lr"], .00003, abs_tol=1e-15)
        source_states[(seed, target, condition)] = cp["model"]
        record = dict(seed=seed, origin_target=target, origin_condition=condition,
            ordered_tasks_json=json.dumps(tasks), unordered_tasks_json=json.dumps(sorted(tasks)),
            updates=4800, task_exposures=76800, task_fraction=.25,
            checkpoint=rel(path), checkpoint_sha256=meta["checkpoint_sha256"], model_state_hash=digest,
            completion_record=rel(meta_path), completion_sha256=sha(meta_path),
            curve_path=rel(meta_path.parent / "curve.csv"), curve_sha256=sha(meta_path.parent / "curve.csv"),
            source_train_sha256=meta["train_sha256"], source_dev_sha256=meta["dev_sha256"],
            initial_state_hash=meta["initial_state_hash"], schedule_sha256=meta["schedule_sha256"],
            original_v3_new_compute=meta["new_compute"], used_in_v4_g0=target in TARGETS)
        source_rows.append(record)
        source_lookup[(seed, target, condition)] = record
        composition_index.setdefault((seed, tasks), []).append(record)
        del cp
    assert len(source_rows) == 24 and len(composition_index) == 21
    for matches in composition_index.values():
        assert len({r["model_state_hash"] for r in matches}) == 1

    adaptation_rows, metadata = [], []
    selections = read(V3 / "SELECTIONS_FROZEN.json")
    assert selections["protocol_sha256"] == sha(V3 / "FROZEN_PROTOCOL.json")
    reset_cache = {}
    for meta_path in sorted((V3 / "adapt").glob("*/completed.json")):
        meta = read(meta_path)
        seed, target, condition, lr = meta["seed"], meta["target"], meta["condition"], meta["lr"]
        assert meta["updates"] == 800 and meta["schedule_sha256"] == adapt_schedules[seed][0]
        assert meta["v3_train_sha256"] == data_hash[(seed, "target_train")]
        assert meta["v3_dev_sha256"] == data_hash[(seed, "target_dev")]
        init = meta["initialization"]
        token = TOKEN_TO_ID[f"<{target.upper()}>"]
        assert init["optimizer_entries"] == 0 and init["optimizer_state_reset"]
        assert init["new_output_head"] is False and init["reset_row_ids"] == [token]
        initial_row = initials[seed]["token_embedding.weight"][token]
        assert init["target_row_hash"] == tensor_hash(initial_row)
        key = (seed, target, condition)
        source = initials[seed] if condition == "A" else source_states[key]
        if key not in reset_cache:
            reset_cache[key] = (state_hash(source), state_hash(source, token, initial_row))
        assert init["loaded_state_hash"] == init["source_state_hash"] == reset_cache[key][0]
        assert init["adapt_initial_state_hash"] == reset_cache[key][1]
        curve = rows(meta_path.parent / "curve.csv")
        assert [int(r["step"]) for r in curve] == list(range(0, 801, 100))
        best = min((r for r in curve if int(r["step"]) > 0), key=lambda r: (-float(r["dev_accuracy"]), float(r["dev_answer_ce"]), int(r["step"])))
        assert int(best["step"]) == meta["best"]["step"]
        paths, parameter_hashes = {}, {}
        for which in ["best", "endpoint"]:
            paths[which] = (V3 / meta[which + "_path"]).resolve()
            cp, digest = load(paths[which], meta[which + "_sha256"], 800 if which == "endpoint" else None)
            assert cp["seed"] == seed and cp["condition"] == condition and cp["lr"] == lr
            assert cp["config"]["model"] == cfg["model"] and cp["step"] == meta[which]["step"]
            parameter_hashes[which] = digest
            if which == "endpoint":
                assert digest == meta["final_model_state_hash"] and cp["numpy_rng"] == adapt_schedules[seed][1]
            del cp
        selected = next(r for r in selections["selections"] if r["seed"] == seed and r["target"] == target and r["condition"] == condition)
        in_scope = target in TARGETS
        adaptation_rows.append(dict(seed=seed, target=target, condition=condition, lr=lr,
            v4_reuse_role=("A_shared_across_backgrounds" if condition == "A" else "g0_v3_BC") if in_scope else "outside_two_target_scope",
            scope_reusable=in_scope, selected_lr=lr == selected["lr"], best_step=meta["best"]["step"], endpoint_step=800,
            completion_record=rel(meta_path), completion_sha256=sha(meta_path),
            curve_path=rel(meta_path.parent / "curve.csv"), curve_sha256=sha(meta_path.parent / "curve.csv"),
            best_path=rel(paths["best"]), best_sha256=meta["best_sha256"], best_state_hash=parameter_hashes["best"],
            endpoint_path=rel(paths["endpoint"]), endpoint_sha256=meta["endpoint_sha256"], endpoint_state_hash=parameter_hashes["endpoint"],
            source_state_hash=init["source_state_hash"], adapt_initial_state_hash=init["adapt_initial_state_hash"],
            target_token_id=token, target_row_hash=init["target_row_hash"], optimizer_entries_at_start=0,
            schedule_sha256=meta["schedule_sha256"], v3_train_sha256=meta["v3_train_sha256"], v3_dev_sha256=meta["v3_dev_sha256"],
            original_v3_new_compute=meta["new_compute"]))
        metadata.append(meta)
    assert len(adaptation_rows) == 108
    for seed in cfg["seeds"]:
        for target in cfg["targets"]:
            for condition in ["A", "B", "C"]:
                candidates = [m for m in metadata if m["seed"] == seed and m["target"] == target and m["condition"] == condition]
                chosen = min(candidates, key=lambda m: (-m["best"]["dev_accuracy"], m["best"]["dev_answer_ce"], m["best"]["step"], cfg["learning_rates"].index(m["lr"])))
                selected = next(r for r in selections["selections"] if r["seed"] == seed and r["target"] == target and r["condition"] == condition)
                assert chosen["lr"] == selected["lr"] and chosen["best_sha256"] == selected["best_sha256"]

    plan = []
    needed_single_tasks = set()
    for seed in cfg["seeds"]:
        for target in TARGETS:
            for background in BACKGROUNDS:
                for condition in ["B", "C"]:
                    tasks = list(groups[target]["sources"][condition])
                    if background == "g1_recoils":
                        assert tasks[1] == "descents"
                        tasks[1] = "recoils"
                    elif background == "g2_deficiencies":
                        assert tasks[2] == "peaks"
                        tasks[2] = "deficiencies"
                    assert len(set(tasks)) == 4 and not set(tasks).intersection(TARGETS)
                    needed_single_tasks.update(tasks)
                    matches = composition_index.get((seed, tuple(tasks)), [])
                    plan.append(dict(seed=seed, target=target, background=background, condition=condition,
                        ordered_tasks_json=json.dumps(tasks), exact_v3_source_match=bool(matches),
                        matching_v3_records_json=json.dumps([r["completion_record"] for r in matches]),
                        preferred_checkpoint=(source_lookup[(seed, target, condition)]["checkpoint"] if background == "g0_v3" else matches[0]["checkpoint"] if matches else ""),
                        adaptation_reusable=background == "g0_v3", adaptation_reuse_candidates=3 if background == "g0_v3" else 0))
    assert len(plan) == 36 and sum(r["exact_v3_source_match"] for r in plan) == 12
    assert all(r["exact_v3_source_match"] == (r["background"] == "g0_v3") for r in plan)
    single_rows = []
    for meta_path in sorted((V3 / "single").glob("*/completed.json")):
        m = read(meta_path)
        seed, task = m["seed"], m["tasks"][0]
        scope = task in needed_single_tasks
        rec = dict(seed=seed, task=task, scope_reusable=scope, completion_record=rel(meta_path),
                   completion_sha256=sha(meta_path), curve_path=rel(meta_path.parent / "curve.csv"),
                   curve_sha256=sha(meta_path.parent / "curve.csv"), source_train_sha256=m["train_sha256"],
                   source_dev_sha256=m["dev_sha256"], model_state_hash=m["model_state_hash"])
        if scope:
            for step in [1200, 4800]:
                path = meta_path.parent / f"step{step}.pt"
                cp, digest = load(path, m["checkpoint_sha256"] if step == 4800 else None, step)
                rec[f"checkpoint_{step}"] = rel(path)
                rec[f"checkpoint_{step}_sha256"] = sha(path)
                rec[f"checkpoint_{step}_state_hash"] = digest
                if step == 4800:
                    assert digest == m["model_state_hash"]
                del cp
        single_rows.append(rec)
    assert sum(r["scope_reusable"] for r in single_rows) == 24
    summary = dict(status="passed", seconds=time.monotonic() - started, cpu_only=True,
        model_forward_calls=0, test_inputs_read=0, test_metrics_read=0, new_test_inputs_read=0,
        v3_protocol_sha256=sha(V3 / "FROZEN_PROTOCOL.json"),
        v3_selection_sha256=sha(V3 / "SELECTIONS_FROZEN.json"),
        v4_targets=TARGETS, v4_backgrounds=BACKGROUNDS, paired_model_seeds=cfg["seeds"],
        v3_source_records=24, v3_unique_ordered_composition_seed_units=21,
        v3_unique_unordered_composition_seed_units=len({(r["seed"], r["unordered_tasks_json"]) for r in source_rows}),
        v3_adaptation_candidates=108, v4_reuse_A_candidates=18, v4_reuse_A_selection_groups=6,
        v4_g0_reuse_source_runs=12, v4_g0_reuse_BC_candidates=36,
        v4_g1_g2_new_source_runs=24, v4_g1_g2_new_adaptation_candidates=72,
        v4_unique_adaptation_candidates=126, v4_unique_selection_groups=42,
        v4_logical_ABC_display_rows=54, v4_reusable_single_task_references=24,
        v4_required_source_tasks=sorted(needed_single_tasks), cpu_checkpoints_loaded=len(checkpoints),
        exact_new_background_source_matches=0, task_order_matters=True,
        compatibility=dict(model=cfg["model"], vocab_size=188, encoding="one_line", length=10,
            source_updates=4800, batch=64, task_fraction=.25, source_lr=cfg["source_lr"],
            adaptation_updates=800, learning_rates=cfg["learning_rates"], adaptation_eval_every=100,
            selection=cfg["selection"], target_samples=256, optimizer="fresh AdamW per target/LR",
            task_row_reset="restore only target query token's existing tied embedding/output row from same saved seed initializer",
            loss="0.5 * (answer cross entropy + teacher-conditioned EOS cross entropy); all188 classes",
            masking="strict causal attention combined with valid-token mask; no task-specific output-class restriction"),
        path_rule="CSV paths are relative to repository root. Original v3 completion paths are relative to v3 artifacts; nested initialization.source_path has legacy exception for reused v2 records.")
    csvwrite(out / "data_inventory.csv", data_records)
    csvwrite(out / "seed_provenance.csv", seed_records)
    csvwrite(out / "source_inventory.csv", source_rows)
    csvwrite(out / "adaptation_inventory.csv", adaptation_rows)
    csvwrite(out / "single_task_inventory.csv", single_rows)
    csvwrite(out / "v4_reuse_plan.csv", plan)
    csvwrite(out / "checkpoint_inventory.csv", list(checkpoints.values()))
    write(out / "reuse_inventory.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
