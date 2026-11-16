"""Run the fixed post-training sequence after both adaptation workers succeed.

No additional training is performed. The only model evaluation is the frozen
inventory of 120 selected and 120 corresponding endpoint checkpoints. Completed
test entries are verified and reused by train.py if an interrupted attempt is
resumed. Every invocation retains separate command logs and stage records.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.fixed_points_deficiencies_arch_v1.design import CONDITIONS, REPEATS, ROUTES, SIZES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "artifacts"
LOGS = OUT / "execution_logs"
SCRIPT_DIR = HERE.relative_to(ROOT).as_posix()


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def gates():
    protocol_hash = sha(OUT / "FROZEN_PROTOCOL.json")
    repeats = {r["repeat_id"] for r in REPEATS}
    expected_sources = {f"{repeat}_{condition}" for repeat in repeats for condition in CONDITIONS}
    actual_source_files = list((OUT / "pretrain").glob("*/completed.json"))
    assert len(actual_source_files) == 32
    assert {p.parent.name for p in actual_source_files} == expected_sources
    for path in actual_source_files:
        value = read(path)
        assert value["status"] == "complete" and value["updates"] == 4800
        assert value["run_id"] == path.parent.name and value["protocol_sha256"] == protocol_hash
        assert (path.parent / "endpoint.pt").is_file()
    expected_adaptations = {f"{repeat}_{condition}_n{size}_lr{index}"
                           for repeat in repeats for condition in ROUTES for size in SIZES for index in range(3)}
    actual_adaptation_files = list((OUT / "adapt").glob("*/completed.json"))
    assert len(actual_adaptation_files) == 360
    assert {p.parent.name for p in actual_adaptation_files} == expected_adaptations
    for path in actual_adaptation_files:
        value = read(path)
        assert value["status"] == "complete" and value["updates"] == 800
        assert value["run_id"] == path.parent.name and value["protocol_sha256"] == protocol_hash
        assert (path.parent / "endpoint.pt").is_file() and (path.parent / "best.pt").is_file()
    workers = sorted(OUT.glob("ADAPT_WORKER_*.json"))
    assert len(workers) == 2, "Exactly two completed worker records are required."
    observed_units = set()
    for path in workers:
        worker = read(path)
        assert worker["status"] == "complete" and worker["candidate_runs"] == 180
        assert len(worker["units"]) == 4 and len(set(worker["units"])) == 4
        assert not observed_units.intersection(worker["units"]), "Worker units overlap."
        observed_units.update(worker["units"])
        assert worker["protocol_sha256"] == protocol_hash
    assert observed_units == repeats
    return dict(status="passed", source_models=32, source_updates=4800, adaptation_candidates=360,
                adaptation_updates=800, workers=2, disjoint_units=sorted(observed_units),
                protocol_sha256=protocol_hash,
                worker_records={p.name: sha(p) for p in workers},
                launch_requirement="Parent orchestrator observes both worker process exit codes equal zero")


def preserve_replay_proof(attempt_id):
    assert attempt_id and all(c.isalnum() or c in "_-" for c in attempt_id)
    data = OUT / "data"
    primary_hash = sha(data / "manifest.json")
    test_hash = sha(data / "test_manifest.json")
    assert read(data / "test_manifest.json")["primary_manifest_sha256"] == primary_hash
    # Data verification preserves its first record and timestamps later ones.
    # Select a genuine verifier output, not an earlier preserved proof copy.
    pattern = re.compile(r"quality_with_test(?:_\d{8}T\d{12}Z)?\.json")
    candidates = sorted(p for p in data.glob("quality_with_test*.json") if pattern.fullmatch(p.name))
    valid, skipped, conflicting = [], [], []
    for path in candidates:
        record = read(path)
        full_replay = (record.get("status") == "pass" and record.get("rng_replayed") is True
                       and record.get("include_test") is True)
        if not full_replay:
            skipped.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                                reason="Not a passed full test-input RNG replay"))
            continue
        if (record.get("primary_manifest_sha256"), record.get("test_manifest_sha256")) != (primary_hash, test_hash):
            conflicting.append(path.relative_to(ROOT).as_posix())
            continue
        when = datetime.fromisoformat(record["created_at_utc"].replace("Z", "+00:00"))
        if when.tzinfo is None:
            raise ValueError(f"Full-replay audit timestamp has no timezone: {path}")
        valid.append((when, path, record))
    if conflicting:
        raise ValueError(f"Full-replay audit bindings conflict with the current manifests: {conflicting}")
    if not valid:
        raise ValueError("No passed full-replay audit matches the current primary and test manifests")
    semantic = lambda record: {key: value for key, value in record.items()
                               if key not in {"created_at_utc", "seconds"}}
    expected = semantic(valid[0][2])
    if any(semantic(record) != expected for _, _, record in valid[1:]):
        raise ValueError("Matching full-replay audits disagree beyond their timestamp and elapsed time")
    _, source, proof = max(valid, key=lambda item: (item[0], item[1].name))
    source_hash = sha(source)
    destination = data / "quality_with_test_full_replay_before_evaluation.json"
    if destination.exists():
        previous = read(destination)
        if semantic(previous) != expected:
            raise ValueError("The preserved earlier full-replay proof disagrees with the current matching audit")
    snapshot = LOGS / f"{attempt_id}_quality_with_test_full_replay.json"
    with snapshot.open("xb") as handle:
        handle.write(source.read_bytes())
    if not destination.exists():
        with destination.open("xb") as handle:
            handle.write(source.read_bytes())
    assert sha(snapshot) == source_hash
    binding_path = LOGS / f"{attempt_id}_replay_proof_selection.json"
    exclusive_json(binding_path, dict(status="passed", selected_source=source.relative_to(ROOT).as_posix(),
        selected_source_sha256=source_hash, primary_manifest_sha256=primary_hash,
        test_manifest_sha256=test_hash, selection_rule="Latest matching full replay; all matching records must agree except timestamp/elapsed time",
        matching_records=[dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)) for _, p, _ in valid],
        skipped_records=skipped, attempt_snapshot=snapshot.relative_to(ROOT).as_posix(),
        attempt_snapshot_sha256=sha(snapshot), canonical_path=destination.relative_to(ROOT).as_posix(),
        canonical_sha256=sha(destination)))
    print(json.dumps(dict(status="preserved", canonical_path=destination.relative_to(ROOT).as_posix(),
                          canonical_sha256=sha(destination), attempt_snapshot=snapshot.relative_to(ROOT).as_posix(),
                          attempt_snapshot_sha256=sha(snapshot), selected_source=source.relative_to(ROOT).as_posix(),
                          selected_source_sha256=source_hash,
                          selection_record=binding_path.relative_to(ROOT).as_posix()), indent=2))


def commands(attempt_id):
    def python(script, *arguments):
        return [sys.executable, "-B", "-u", f"{SCRIPT_DIR}/{script}", *arguments]
    return [
        ("aggregate", python("train.py", "aggregate")),
        ("verify_training", python("train.py", "verify")),
        ("freeze_selection", python("train.py", "select")),
        ("independent_training_audit", python("final_check.py")),
        ("plot_losses", python("plot_losses.py")),
        ("generate_test", python("data_pipeline.py", "generate-test", "--selection-manifest",
                                 f"{SCRIPT_DIR}/artifacts/SELECTIONS_FROZEN.json")),
        ("verify_test_full_replay", python("data_pipeline.py", "verify", "--include-test")),
        ("preserve_test_replay_proof", python("post_training.py", "--preserve-replay-proof", "--attempt-id", attempt_id)),
        ("evaluate_frozen_test_inventory", python("train.py", "test")),
        ("analyze", python("analysis.py")),
        ("independent_prediction_statistics_audit", python("independent_recompute.py")),
        ("canonical_report", python("final_report.py")),
    ]


def run():
    LOGS.mkdir(parents=True, exist_ok=True)
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    record_path = LOGS / f"{attempt_id}_post_training_commands.jsonl"
    status_path = LOGS / f"{attempt_id}_post_training_status.json"
    start = utc()
    begin = time.perf_counter()
    base = dict(attempt_id=attempt_id, started_at_utc=start, controller_sha256=sha(__file__),
                python=sys.executable, cwd=str(ROOT), training_runs=0,
                model_evaluation_scope="Only the planned 240 selected/endpoint evaluations; completed entries are reused",
                command_records=record_path.relative_to(ROOT).as_posix())
    try:
        gate = gates()
    except Exception as error:
        exclusive_json(status_path, dict(**base, status="failed_precondition", finished_at_utc=utc(),
                                         error_type=type(error).__name__, error=str(error), elapsed_seconds=time.perf_counter()-begin))
        raise
    exclusive_json(LOGS / f"{attempt_id}_gate.json", gate)
    plan = commands(attempt_id)
    exclusive_json(LOGS / f"{attempt_id}_plan.json", dict(**base, gate=gate,
                   stages=[dict(stage=name, argv=argv, command_display=subprocess.list2cmdline(argv)) for name, argv in plan]))
    completed = []
    with record_path.open("x", encoding="utf-8", newline="\n") as records:
        for index, (name, argv) in enumerate(plan, 1):
            log = LOGS / f"{attempt_id}_{index:02d}_{name}.log"
            started_at = utc()
            stage_begin = time.perf_counter()
            command = dict(attempt_id=attempt_id, stage_index=index, stage=name, argv=argv,
                           command_display=subprocess.list2cmdline(argv), cwd=str(ROOT),
                           started_at_utc=started_at, output_log=log.relative_to(ROOT).as_posix())
            records.write(json.dumps(dict(event="start", **command))+"\n")
            records.flush()
            print(f"START {index:02d}/{len(plan):02d} {name} {started_at}", flush=True)
            execution_error = None
            with log.open("xb") as output:
                try:
                    process = subprocess.run(argv, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, check=False)
                    exit_code = process.returncode
                except Exception as error:
                    execution_error = dict(type=type(error).__name__, message=str(error))
                    output.write((json.dumps(execution_error)+"\n").encode("utf-8"))
                    exit_code = -1
            result = dict(**command, finished_at_utc=utc(), elapsed_seconds=time.perf_counter()-stage_begin,
                          exit_code=exit_code, output_log_sha256=sha(log))
            if execution_error:
                result["execution_error"] = execution_error
            records.write(json.dumps(dict(event="end", **result))+"\n")
            records.flush()
            print(f"END   {index:02d}/{len(plan):02d} {name} exit={exit_code} seconds={result['elapsed_seconds']:.3f}", flush=True)
            if exit_code != 0:
                exclusive_json(status_path, dict(**base, status="failed", failed_stage=name,
                    exit_code=exit_code, finished_at_utc=utc(), elapsed_seconds=time.perf_counter()-begin,
                    completed_stages=completed, failure=result))
                return exit_code if 0 < exit_code < 256 else 1
            completed.append(result)
    try:
        test_complete = read(OUT / "TEST_COMPLETE.json")
        assert test_complete["status"] == "complete" and test_complete["evaluations"] == 240
        assert read(OUT / "FINAL_TRAINING_AUDIT.json")["status"] == "passed"
        assert read(OUT / "analysis" / "INDEPENDENT_RECOMPUTE.json")["status"] == "pass"
        assert read(HERE / "REPORT_MANIFEST.json")["status"] == "complete"
    except Exception as error:
        exclusive_json(status_path, dict(**base, status="failed_final_artifact_gate", finished_at_utc=utc(),
            elapsed_seconds=time.perf_counter()-begin, completed_stages=completed,
            error_type=type(error).__name__, error=str(error)))
        raise
    result = dict(**base, status="complete", finished_at_utc=utc(), elapsed_seconds=time.perf_counter()-begin,
                  gate=gate, completed_stage_count=len(completed), completed_stages=completed,
                  final_artifacts={str(p.relative_to(ROOT)): sha(p) for p in [OUT / "TEST_COMPLETE.json",
                    OUT / "FINAL_TRAINING_AUDIT.json", OUT / "analysis" / "INDEPENDENT_RECOMPUTE.json",
                    HERE / "REPORT_MANIFEST.json", HERE / "README.md", HERE / "RESULT_REPORT.md"]})
    exclusive_json(status_path, result)
    completion = OUT / "POST_TRAINING_COMPLETE.json"
    exclusive_json(completion, result)
    print(json.dumps(dict(status="complete", attempt_id=attempt_id, stages=len(completed),
                          elapsed_seconds=result["elapsed_seconds"], training_runs=0)), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preserve-replay-proof", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attempt-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.preserve_replay_proof:
        preserve_replay_proof(args.attempt_id)
    else:
        sys.exit(run())
