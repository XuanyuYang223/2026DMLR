"""Source-only endpoint audit and reporting for the fixed remaining-task grid.

Run only after all eligible formal runs finish. This program never constructs a
model, generates inputs, opens reserved test pools, or executes a model forward.
Historical files and existing report outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import textwrap
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / "experiments/platform_v01/calibration"
ART = HERE / "artifacts"
LEVELS = ["8", "10", "12", "all"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def params(value):
    return json.loads(value) if isinstance(value, str) else value


def task_key(record):
    return record["task_id"], json.dumps(params(record["params"]), sort_keys=True)


def close(a, b, tolerance=1e-10):
    assert math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance), (a, b)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def source_audit():
    started = time.monotonic()
    frozen = read_json(HERE / "FROZEN_PROTOCOL.json")
    frozen_hash = sha(HERE / "FROZEN_PROTOCOL.json")
    assert sha(ART / "FROZEN_CALIBRATION.json") == frozen_hash
    cfg = frozen["config"]
    old_frozen = read_json(OLD / "FROZEN_CALIBRATION.json")
    assert sha(OLD / "FROZEN_CALIBRATION.json") == frozen["inherited_config_sha256"]
    stable_keys = ("lengths", "train_per_length", "dev_per_length", "updates", "batch_size",
                   "eval_every", "evaluation_batch_size", "learning_rate", "warmup", "minimum_lr",
                   "weight_decay", "betas", "epsilon", "gradient_clip", "model", "precision",
                   "max_new_tokens", "repeats")
    for key in stable_keys:
        assert cfg[key] == old_frozen["config"][key], key
    bindings = {relative(HERE / "FROZEN_PROTOCOL.json"): frozen_hash}
    for group in ("code_sha256", "train_dev_sha256"):
        for rel, expected in frozen[group].items():
            assert sha(ROOT / rel) == expected, rel
            bindings[rel] = expected
    assert read_json(HERE / "POOL_REUSE_AUDIT.json")["status"] == "passed"
    enc = read_json(HERE / "ENCODING_PREFLIGHT.json")
    assert sha(HERE / "ENCODING_PREFLIGHT.csv") == enc["table_sha256"]
    assert frozen["accepted_instances"] == enc["accepted_instances"]
    assert frozen["rejected_instances"] == enc["rejected_instances"]
    accepted, rejected = enc["accepted_instances"], enc["rejected_instances"]
    assert len(accepted) == 49 and len(rejected) == 1
    specs = {task_key(s): s for s in accepted}
    expected_keys = {(s["instance_id"], r["initialization_seed"]) for s in accepted for r in cfg["repeats"]}
    completions = []
    for path in sorted((ART / "runs").glob("*/COMPLETE.json")):
        record = read_json(path)
        assert task_key(record) in specs, path
        spec = specs[task_key(record)]
        completions.append((path, record, spec))
    actual_keys = [(s["instance_id"], c["initialization_seed"]) for _, c, s in completions]
    assert len(actual_keys) == 98 and len(set(actual_keys)) == 98
    assert set(actual_keys) == expected_keys, "The fixed formal grid is incomplete or contains extra runs"

    initial_states, schedules, expected_exposures = {}, {}, {}
    for repeat in cfg["repeats"]:
        seed = repeat["initialization_seed"]
        entry = frozen["reused_initial_checkpoint_sha256"][str(seed)]
        old_path = ROOT / entry["path"]
        assert sha(old_path) == entry["sha256"]
        bindings[entry["path"]] = entry["sha256"]
        payload = torch.load(old_path, map_location="cpu", weights_only=False)
        initial_states[seed] = state_hash(payload["model"])
        assert payload["initialization_seed"] == seed
        del payload
        new_path = ART / "initial" / f"seed{seed}.pt"
        payload = torch.load(new_path, map_location="cpu", weights_only=False)
        assert state_hash(payload["model"]) == initial_states[seed]
        bindings[relative(new_path)] = sha(new_path)
        del payload
        rng = np.random.default_rng(repeat["sampling_and_dropout_seed"])
        order, h = [], hashlib.sha256()
        counts = {n: np.zeros(cfg["train_per_length"], dtype=np.int64) for n in cfg["lengths"]}
        for _ in range(cfg["updates"]):
            if not order:
                order = list(map(int, rng.permutation(cfg["lengths"])))
            n = order.pop()
            indices = rng.choice(cfg["train_per_length"], cfg["batch_size"], replace=False)
            h.update(np.asarray([n, *indices], dtype="<i8").tobytes())
            counts[n][indices] += 1
        schedules[seed], expected_exposures[seed] = h.hexdigest(), counts
        old_record = read_json(OLD / "runs" / f"s{seed}_cycle_count" / "COMPLETE.json")
        assert schedules[seed] == old_record["length_and_index_schedule_sha256"]
        assert initial_states[seed] == old_record["initial_state_sha256"]

    all_curves, audited_runs, accounting, exposure_summary = [], [], [], []
    curve_map, run_map = {}, {}
    for path, record, spec in completions:
        folder, seed, iid = path.parent, record["initialization_seed"], spec["instance_id"]
        assert folder.name == record["run_id"]
        repeat = next(r for r in cfg["repeats"] if r["initialization_seed"] == seed)
        assert record["frozen_protocol_sha256"] == frozen_hash
        assert record["updates"] == cfg["updates"]
        assert record["initial_state_sha256"] == initial_states[seed]
        assert record["length_and_index_schedule_sha256"] == schedules[seed]
        assert record["test_evaluated"] is False and record["optimizer_started_empty"] is True
        assert record["sample_exposures"] == cfg["updates"] * cfg["batch_size"]
        for name in ("initialization_seed", "data_seed", "sampling_and_dropout_seed"):
            assert record[name] == repeat[name]
        for name, expected in record["checkpoint_sha256"].items():
            assert sha(folder / name) == expected, folder / name
            bindings[relative(folder / name)] = expected
        payload = torch.load(folder / "endpoint.pt", map_location="cpu", weights_only=False)
        assert payload["config"] == cfg
        assert payload["identity"]["run_id"] == record["run_id"]
        assert payload["step"] == cfg["updates"]
        assert payload["initial_state_sha256"] == initial_states[seed]
        assert state_hash(payload["model"]) != initial_states[seed]
        assert {int(s["step"]) for s in payload["optimizer"]["state"].values()} == {cfg["updates"]}
        del payload
        curve_path = folder / "curves.csv"
        curve = read_csv(curve_path)
        expected_curve = {(step, split, n) for step in range(0, cfg["updates"] + 1, cfg["eval_every"])
                          for split in ("train", "dev") for n in LEVELS}
        identities = [(int(r["step"]), r["split"], r["length"]) for r in curve]
        assert len(identities) == len(expected_curve) and set(identities) == expected_curve
        for row in curve:
            assert row["run_id"] == record["run_id"] and int(row["initialization_seed"]) == seed
            n = row["length"]
            expected_count = cfg[f"{row['split']}_per_length"] * (len(cfg["lengths"]) if n == "all" else 1)
            assert int(row["count"]) == expected_count
            for a, count_key in (("accuracy", "correct"), ("majority_accuracy", "majority_correct"), ("eos_accuracy", "eos_generated")):
                close(row[a], int(row[count_key]) / int(row["count"]))
            assert 0 <= int(row["correct"]) <= int(row["eos_generated"]) <= int(row["count"])
            all_curves.append({"instance_id": iid, **row})
        for step in range(0, cfg["updates"] + 1, cfg["eval_every"]):
            for split in ("train", "dev"):
                values = [r for r in curve if int(r["step"]) == step and r["split"] == split]
                pooled = next(r for r in values if r["length"] == "all")
                for field in ("count", "correct", "majority_correct", "eos_generated"):
                    assert int(pooled[field]) == sum(int(r[field]) for r in values if r["length"] != "all")
        candidates = [r for r in curve if r["split"] == "dev" and r["length"] == "all" and int(r["step"]) > 0]
        best = min(candidates, key=lambda r: (-float(r["accuracy"]), float(r["answer_ce"]), int(r["step"])))
        endpoint = next(r for r in candidates if int(r["step"]) == cfg["updates"])
        assert int(best["step"]) == record["best_step"]
        for field in ("accuracy", "answer_ce", "eos_ce", "sequence_ce", "count", "correct", "majority_accuracy"):
            close(endpoint[field], record["endpoint_dev"][field])
            close(best[field], record["best_dev"][field])
        exp_path = folder / "input_exposures.csv"
        exposures = read_csv(exp_path)
        exp_keys = [(int(r["length"]), int(r["input_index"])) for r in exposures]
        exp_expected = {(n, i) for n in cfg["lengths"] for i in range(cfg["train_per_length"])}
        assert len(exp_keys) == len(exp_expected) and set(exp_keys) == exp_expected
        for row in exposures:
            assert row["run_id"] == record["run_id"]
            assert int(row["exposures"]) == expected_exposures[seed][int(row["length"])][int(row["input_index"])]
        for n in cfg["lengths"]:
            counts = expected_exposures[seed][n]
            assert record["length_updates"][str(n)] == cfg["updates"] // len(cfg["lengths"])
            exposure_summary.append(dict(instance_id=iid, run_id=record["run_id"], seed=seed, length=n,
                unique_training_inputs=int(np.count_nonzero(counts)), sample_exposures=int(counts.sum()),
                minimum_exposures=int(counts.min()), maximum_exposures=int(counts.max()),
                updates=record["length_updates"][str(n)], **record["tokens_by_length"][str(n)],
                source_path=relative(exp_path)))
        for field in ("input_tokens", "padded_input_tokens", "supervised_tokens", "answer_tokens"):
            assert record[field] == sum(r[field] for r in record["tokens_by_length"].values())
        accounting.append(dict(instance_id=iid, family=spec["family"], run_id=record["run_id"], seed=seed,
            updates=record["updates"], sample_exposures=record["sample_exposures"],
            unique_training_inputs=sum(int(np.count_nonzero(c)) for c in expected_exposures[seed].values()),
            **{k: record[k] for k in ("input_tokens", "padded_input_tokens", "supervised_tokens", "answer_tokens", "seconds")},
            completed_unix=record["completed_unix"], source_path=relative(path)))
        for p in (path, curve_path, exp_path):
            bindings[relative(p)] = sha(p)
        audited_runs.append(dict(instance_id=iid, run_id=record["run_id"], seed=seed,
            endpoint_step=cfg["updates"], best_step=record["best_step"],
            best_dev_accuracy=record["best_dev"]["accuracy"], endpoint_dev_accuracy=record["endpoint_dev"]["accuracy"],
            best_minus_endpoint_pp=100 * (record["best_dev"]["accuracy"] - record["endpoint_dev"]["accuracy"]),
            source_path=relative(path), passed=True))
        curve_map[iid, seed], run_map[iid, seed] = curve, record

    workers = []
    for seed in (617, 829):
        path = ART / f"WORKER_{seed}.json"
        if path.exists():
            workers.append({"path": relative(path), "record": read_json(path)})
            bindings[relative(path)] = sha(path)
            worker = workers[-1]["record"]
            assert worker["status"] == "complete" and worker["seed"] == seed
            assert worker["frozen_protocol_sha256"] == frozen_hash
            assert set(worker["run_ids"]) == {r["run_id"] for r in audited_runs if r["seed"] == seed}
    assert len(workers) == 2, "Both fixed workers must finish before the final report"
    wall_seconds = None
    if len(workers) == 2 and all("started_at_utc" in w["record"] and "finished_at_utc" in w["record"] for w in workers):
        starts = [datetime.fromisoformat(w["record"]["started_at_utc"]) for w in workers]
        finishes = [datetime.fromisoformat(w["record"]["finished_at_utc"]) for w in workers]
        wall_seconds = (max(finishes) - min(starts)).total_seconds()
    audit = dict(status="passed", created_at_utc=datetime.now(timezone.utc).isoformat(),
        frozen_protocol_sha256=frozen_hash, formal_runs=len(audited_runs), requested_instances=50,
        completed_instances=49, reader_rejected_instances=1, units=2,
        checkpoint_files_hashed=2 * len(audited_runs), endpoint_checkpoints_loaded=len(audited_runs),
        curves_checked=len(all_curves), input_exposure_records_checked=len(audited_runs) * 3072,
        original_initial_states_and_sample_streams_match=True, development_choices_rechecked=True,
        model_forward_calls=0, reserved_test_input_files_opened=0, new_training_updates=0,
        audit_seconds=time.monotonic() - started, worker_wall_seconds=wall_seconds, workers=workers,
        accumulated_run_elapsed_seconds=sum(r["seconds"] for r in accounting),
        timing_scope="Worker envelope is wall time when both worker records exist. Sum of run elapsed includes overlapping concurrent work and is not dedicated GPU-hours.",
        input_bindings=dict(sorted(bindings.items())))
    return frozen, enc, curve_map, run_map, all_curves, audited_runs, accounting, exposure_summary, audit


def produce_tables(frozen, enc, curve_map, run_map):
    index_path = ROOT / "experiments/platform_v01/LEARNABILITY_INDEX.csv"
    old_index = read_csv(index_path)
    specs = {s["instance_id"]: s for s in enc["accepted_instances"] + enc["rejected_instances"]}
    scope = [r["instance_id"] for r in old_index if r["instance_id"] in specs]
    rejected = {s["instance_id"] for s in enc["rejected_instances"]}
    enc_rows = read_csv(HERE / "ENCODING_PREFLIGHT.csv")
    raw, summary = [], []
    for iid in scope:
        spec = specs[iid]
        for seed in (617, 829):
            for length in LEVELS:
                base = dict(instance_id=iid, task_id=spec["task_id"], params=json.dumps(spec["params"], sort_keys=True),
                            family=spec["family"], seed=seed, length=length,
                            status="reader_rejected_context" if iid in rejected else "completed_shared_canonical_units")
                for role in ("train", "dev"):
                    e = [r for r in enc_rows if r["instance_id"] == iid and int(r["seed"]) == seed
                         and r["split"] == role and (length == "all" or r["length"] == length)]
                    assert len(e) == (3 if length == "all" else 1)
                    count, compatible = sum(int(r["count"]) for r in e), sum(int(r["cap_compatible_count"]) for r in e)
                    base[f"{role}_count"] = count
                    base[f"{role}_generation_compatible_fraction"] = compatible / count
                    base[f"{role}_context_rejected_count"] = sum(int(r["context_rejected_count"]) for r in e)
                base["generation_limited"] = base["dev_generation_compatible_fraction"] < 1
                if iid not in rejected:
                    curve, record = curve_map[iid, seed], run_map[iid, seed]
                    for role in ("train", "dev"):
                        row = next(r for r in curve if int(r["step"]) == frozen["config"]["updates"] and r["split"] == role and r["length"] == length)
                        assert int(row["count"]) == base[f"{role}_count"]
                        for field in ("accuracy", "majority_accuracy", "answer_ce", "eos_ce", "sequence_ce", "correct", "majority_correct"):
                            base[f"{role}_{field}"] = int(row[field]) if field.endswith("correct") else float(row[field])
                        assert float(row["accuracy"]) <= base[f"{role}_generation_compatible_fraction"] + 1e-12
                    base["gain_over_majority_pp"] = 100 * (base["dev_accuracy"] - base["dev_majority_accuracy"])
                    base["run_id"] = record["run_id"]
                    base["source_path"] = relative(ART / "runs" / record["run_id"] / "curves.csv")
                else:
                    base["source_path"] = relative(HERE / "ENCODING_PREFLIGHT.json")
                raw.append(base)
        for length in LEVELS:
            rows = [r for r in raw if r["instance_id"] == iid and r["length"] == length]
            row = dict(instance_id=iid, family=spec["family"], length=length, status=rows[0]["status"], units=0 if iid in rejected else 2,
                dev_generation_compatible_fraction_mean=statistics.mean(r["dev_generation_compatible_fraction"] for r in rows),
                train_generation_compatible_fraction_mean=statistics.mean(r["train_generation_compatible_fraction"] for r in rows),
                generation_limited=any(r["generation_limited"] for r in rows))
            if iid not in rejected:
                for field in ("train_accuracy", "dev_accuracy", "train_majority_accuracy", "dev_majority_accuracy", "gain_over_majority_pp", "train_answer_ce", "dev_answer_ce", "train_sequence_ce", "dev_sequence_ce"):
                    row[field + "_mean"] = statistics.mean(r[field] for r in rows)
                for r in rows:
                    row[f"dev_accuracy_seed{r['seed']}"] = r["dev_accuracy"]
                    row[f"train_accuracy_seed{r['seed']}"] = r["train_accuracy"]
            summary.append(row)
    pooled = {r["instance_id"]: r for r in summary if r["length"] == "all"}
    updated = []
    for old in old_index:
        row = dict(old)
        row["original_v01_status"] = old["status"]
        row["original_index_path"] = relative(index_path)
        row["data_unit_provenance"] = "same canonical units and seed pairing as v0.1; not additional independent units"
        row["model_test_evaluated"] = "False"
        if old["instance_id"] in rejected:
            row.update(status="reader_rejected_context_v0.2", endpoint_dev_mean="", majority_dev_mean="",
                       result_source_path=relative(HERE / "ENCODING_PREFLIGHT.json"), current_scope="requested_context_rejected")
        elif old["instance_id"] in pooled:
            p = pooled[old["instance_id"]]
            row.update(status="calibrated_shared_canonical_units_v0.2", seeds="617;829", lengths="8;10;12",
                       endpoint_dev_mean=p["dev_accuracy_mean"], majority_dev_mean=p["dev_majority_accuracy_mean"],
                       result_source_path="endpoint_summary.csv", current_scope="completed_new_task_calibration",
                       generation_compatible_fraction=p["dev_generation_compatible_fraction_mean"],
                       generation_limited=p["generation_limited"])
        elif old["status"] == "calibrated_fixed_budget_dev_only":
            row.update(result_source_path="experiments/platform_v01/calibration/task_summary.csv", current_scope="inherited_v01_calibration")
        else:
            row.update(result_source_path=relative(index_path), current_scope="excluded_nonmaterialized_binary_or_alias",
                       data_unit_provenance="not run: excluded nonmaterialized binary or alias")
        updated.append(row)
    assert len(raw) == 400 and len(summary) == 200 and len(updated) == 62
    return raw, summary, updated


def make_plots(out, summary, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = out / "figures"
    directory.mkdir()
    instances = [r["instance_id"] for r in summary if r["length"] == "all" and r["status"].startswith("completed")]
    for page, start in enumerate(range(0, len(instances), 10), 1):
        figure, axes = plt.subplots(5, 2, figsize=(15, 16), sharex=True, sharey=True)
        for axis, iid in zip(axes.flat, instances[start:start + 10]):
            for seed, color in ((617, "#1769aa"), (829, "#db7417")):
                for split, style in (("train", "--"), ("dev", "-")):
                    rows = sorted((r for r in curves if r["instance_id"] == iid and r["length"] == "all"
                                   and int(r["initialization_seed"]) == seed and r["split"] == split), key=lambda r: int(r["step"]))
                    axis.plot([int(r["step"]) for r in rows], [float(r["accuracy"]) for r in rows],
                              color=color, linestyle=style, label=f"{seed} {split}")
            ceiling = next(r["dev_generation_compatible_fraction_mean"] for r in summary if r["instance_id"] == iid and r["length"] == "all")
            axis.axhline(ceiling, color="black", linestyle=":", linewidth=1, label="Mean dev generation ceiling")
            axis.set_title(textwrap.fill(iid, 42), fontsize=10)
            axis.set_ylim(-0.02, 1.02)
            axis.grid(alpha=.2)
            axis.set_xlabel("Updates")
            axis.set_ylabel("Complete-answer accuracy")
        for axis in list(axes.flat)[len(instances[start:start + 10]):]:
            axis.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=3, fontsize=9)
        figure.suptitle("Shared canonical units: pooled training/development curves", y=.995)
        figure.tight_layout(rect=(0, 0, 1, .96))
        for extension in ("png", "svg"):
            figure.savefig(directory / f"learning_curves_{page:02d}.{extension}", dpi=160)
        plt.close(figure)


def report_text(summary, accounting, audit):
    pooled = [r for r in summary if r["length"] == "all"]
    completed = [r for r in pooled if r["units"] == 2]
    capped = [r for r in completed if r["generation_limited"]]
    below = [r for r in completed if r["gain_over_majority_pp_mean"] < 0]
    token_totals = {k: sum(r[k] for r in accounting) for k in ("input_tokens", "padded_input_tokens", "supervised_tokens", "answer_tokens", "updates", "sample_exposures")}
    wall = "not recorded" if audit["worker_wall_seconds"] is None else f"{audit['worker_wall_seconds']:.2f} seconds"
    table = ["| Instance | Family | Length | Dev617 (%) | Dev829 (%) | Mean dev (%) | Mean train (%) | Dev majority (%) | Gain (pp) | Dev ceiling (%) | Status |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in summary:
        values = [r["instance_id"], r["family"], r["length"]]
        if r["units"]:
            values += [f"{100*r[k]:.2f}" for k in ("dev_accuracy_seed617", "dev_accuracy_seed829", "dev_accuracy_mean", "train_accuracy_mean", "dev_majority_accuracy_mean")]
            values += [f"{r['gain_over_majority_pp_mean']:+.2f}", f"{100*r['dev_generation_compatible_fraction_mean']:.2f}",
                       "completed; generation-capped" if r["generation_limited"] else "completed"]
        else:
            values += ["not measured"] * 7 + ["reader-rejected: context"]
        table.append("| " + " | ".join(values) + " |")
    text = f"""# Remaining platform-instance calibration results

The fixed grid is complete: 50 requested instances, 49 trained with both inherited units (98 runs), and one explicit reader rejection. All runs use the endpoint at update 2400 as the primary development reference. No model test forward or new data pool was produced. The rejected `to_reduced_word` instance has no invented accuracy; no input was truncated, filtered or replaced.

## Data reuse and comparability

The training/development inputs and seed-to-unit pairing are the same canonical pools used by the original six-task calibration. The pool-reuse audit verifies their hashes, archived run/checkpoint bindings and index streams. These results are directly comparable in data split, initialization pairing, training configuration and decoding budget, and they are not independent data units. Different tasks share inputs intentionally. Reserved test inputs remain unopened and no model forward uses them.

The original capacity failure remains an immutable historical record. The author resolved it for this calibration by permitting read-only reuse; the separate transfer experiment still requires fresh historical-orbit exclusion. This stage does not produce a new transfer result.

## Endpoint results and interpretation

The table below includes every requested instance at each trained length and pooled across lengths. Means summarize the same two units; no confidence interval or family ranking is claimed. There are {len(capped)} completed instances with a pooled development generation ceiling below one and {len(below)} completed instances with a mean endpoint development score below their training-derived majority baseline. These are descriptive counts, not independent statistical trials or task-removal rules.

The model still generates at most eight tokens including EOS. Complete gold answers remain the loss and exact-match reference. A ceiling below one therefore identifies protocol-limited exact accuracy: a zero score on an impossible-length answer does not establish that the underlying mathematical computation is unlearnable. Context-valid structured tasks remain in the results. Loss curves supervise their entire gold answer. All three lengths occur in training; per-length development differences are not unseen-length extrapolation.

The majority answer is determined from the corresponding training pool separately for each length. `endpoint_by_unit.csv` includes both train/development baselines, all losses, exact counts and per-unit cap fractions. `endpoint_summary.csv` provides the unrounded table values. `LEARNABILITY_INDEX_v02.csv` preserves the original order and marks the six original calibrations as inherited, the new completions as shared-unit calibration, the context rejection explicitly, and nonmaterialized binary/alias exclusions without changing the old index.

""" + "\n".join(table) + f"""

## Curves, checkpoint diagnostics and actual cost

`learning_curves.csv` retains every training/development record. The pooled curve pages in `figures/` retain both units and show the generation ceiling. `checkpoint_diagnostics.csv` reports the selected development-best and fixed endpoint; the selected checkpoint never replaces the endpoint primary reference.

Actual formal totals from `accounting.csv`: {token_totals['updates']} optimizer updates, {token_totals['sample_exposures']} example exposures, {token_totals['input_tokens']} nonpadding sequence tokens, {token_totals['padded_input_tokens']} padded sequence tokens, {token_totals['supervised_tokens']} supervised tokens and {token_totals['answer_tokens']} non-EOS answer tokens. Repeated exposures are not additional independent inputs. `exposure_summary.csv` gives per-task/unit/length unique input counts, exposure extrema and actual token counts; individual run directories retain each input-index count.

Accumulated run elapsed time is {audit['accumulated_run_elapsed_seconds']:.2f} seconds. Concurrent runs overlap, so this sum is not wall time or dedicated GPU-hours. Worker start/finish envelope: {wall}. Per-run timing includes training, training/development evaluation and checkpoint saving, and excludes data encoding and this report/audit.

## Audit and limitations

`SOURCE_ONLY_AUDIT.json` passed: the complete fixed inventory, frozen source/data hashes, original initialization states, all index/exposure streams, checkpoint hashes, endpoint optimizer counters, curve arithmetic and development selections were checked. No model forward was required for this audit. Statistical evidence remains limited to two shared units, one architecture, one fixed optimization/decoding configuration and the trained lengths. The generation cap and context rejection constrain the calibrated interface; incomplete fitting, generalization gaps and negative gains remain visible and are not reasons to delete runs.

No extra task, repeat, decoding-budget search, training extension or model test evaluation follows this report. The next separately requested transfer stage has its own fresh-unit and test-gating requirements.
"""
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ART / "final")
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(HERE) or out.exists():
        raise ValueError("Use a fresh output directory inside reuse_authorized_v1; existing outputs are immutable")
    frozen, enc, curves, runs, all_curves, diagnostics, accounting, exposures, audit = source_audit()
    raw, summary, index = produce_tables(frozen, enc, curves, runs)
    out.mkdir(parents=True)
    write_csv(out / "endpoint_by_unit.csv", raw)
    write_csv(out / "endpoint_summary.csv", summary)
    write_csv(out / "LEARNABILITY_INDEX_v02.csv", index)
    write_csv(out / "learning_curves.csv", all_curves)
    write_csv(out / "checkpoint_diagnostics.csv", diagnostics)
    write_csv(out / "accounting.csv", accounting)
    write_csv(out / "exposure_summary.csv", exposures)
    write_json(out / "SOURCE_ONLY_AUDIT.json", audit)
    make_plots(out, summary, all_curves)
    with (out / "RESULT_REPORT.md").open("x", encoding="utf-8") as f:
        f.write(report_text(summary, accounting, audit))
    with (out / "README.md").open("x", encoding="utf-8") as f:
        f.write("# Completed remaining-instance calibration\n\nRead [RESULT_REPORT.md](RESULT_REPORT.md) for the complete endpoint table, shared-unit interpretation, generation limits and cost. [SOURCE_ONLY_AUDIT.json](SOURCE_ONLY_AUDIT.json) records the passed no-forward audit. [endpoint_by_unit.csv](endpoint_by_unit.csv) and [endpoint_summary.csv](endpoint_summary.csv) are the numerical source tables. [LEARNABILITY_INDEX_v02.csv](LEARNABILITY_INDEX_v02.csv) updates status without changing v0.1 records.\n\nRebuild into a fresh directory from the repository root with `.venv/Scripts/python.exe -B experiments/platform_calibration_v02/reuse_authorized_v1/report_and_audit.py --out experiments/platform_calibration_v02/reuse_authorized_v1/artifacts/final_reproduction`. Local checkpoints are required for source-only checkpoint verification; no model forward or new training runs. The existing output is never overwritten.\n")
    outputs = {relative(p): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    write_json(out / "MANIFEST.json", dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        frozen_protocol_sha256=audit["frozen_protocol_sha256"], analysis_source_sha256=sha(Path(__file__)),
        outputs=outputs, scientific_unit="Two original canonical data/initialization pairs shared with v0.1"))
    print(json.dumps(dict(status="passed", formal_runs=audit["formal_runs"], endpoint_rows=len(raw),
                         summary_rows=len(summary), index_rows=len(index), out=relative(out))))


if __name__ == "__main__":
    main()
