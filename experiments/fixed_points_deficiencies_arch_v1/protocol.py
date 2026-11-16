"""Commit-checked gate and formal registrations; never rewrite a frozen record."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.fixed_points_deficiencies_arch_v1.plan import GATE, GRU_MODEL, HERE, ORIGINAL, TRANSFORMER_MODEL

OUT = HERE / "artifacts"
GATE_FREEZE = HERE / "GATE_FROZEN.json"
FORMAL_FREEZE = OUT / "FROZEN_PROTOCOL.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_committed(path):
    path = Path(path).resolve()
    if git("rev-parse", f"HEAD:{rel(path)}") != git("hash-object", str(path)):
        raise ValueError(f"Commit the unchanged bound file before optimization: {rel(path)}")


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def inherited_paths():
    paths = [ROOT / "experiments/permworld_pilot/pilot.py",
             ROOT / "experiments/pilot_v3_multitarget/pilot_v3.py",
             ROOT / "experiments/pilot_v5_confirmation/artifacts/config.json",
             ORIGINAL / "design.py", ORIGINAL / "RELATION_TABLE.csv",
             ORIGINAL / "PREFLIGHT_REPORT.md", ORIGINAL / "preflight_audit/AUDIT.json"]
    for folder in ("experiments/permworld_pilot/vendor", "permworld_platform",
                   "experiments/task_count_v1", "experiments/task_count_v2"):
        source = ROOT / folder
        paths += sorted(source.rglob("*.py") if folder.endswith("vendor") else source.glob("*.py"))
    reuse = HERE / "RELATION_REUSE.json"
    if reuse.exists():
        paths += [ROOT / name for name in read(reuse)["sources"]]
    return sorted(set(paths))


def gate_paths():
    names = ["plan.py", "models.py", "gate_runner.py", "scalar_gate_helpers.py",
             "data_pipeline.py", "protocol.py", "GATE_PROTOCOL.md", "GATE_ANALYSIS_PLAN.md",
             "PROTOCOL.md", "ANALYSIS_PLAN.md", "stats_plan.json", "DATA_PROTOCOL.md"]
    return sorted(set([HERE / name for name in names] + inherited_paths()))


def formal_paths():
    paths = sorted(HERE.glob("*.py")) + [HERE / name for name in
        ("PROTOCOL.md", "ANALYSIS_PLAN.md", "stats_plan.json", "DATA_PROTOCOL.md",
         "GATE_PROTOCOL.md", "GATE_ANALYSIS_PLAN.md", "RELATION_REUSE.json", "RELATION_TABLE.csv")]
    return sorted(set(paths + inherited_paths()))


def check_code(record, committed):
    for name, expected in record["code_sha256"].items():
        path = ROOT / name
        if sha(path) != expected:
            raise ValueError(f"Frozen source bytes changed: {name}")
        if committed:
            require_committed(path)


def verify_gate(committed=True):
    record = read(GATE_FREEZE)
    assert record["status"] == "frozen" and record["config"] == GATE
    if committed:
        require_committed(GATE_FREEZE)
    check_code(record, committed)
    manifest = ROOT / record["gate_data_manifest_path"]
    assert sha(manifest) == record["gate_data_manifest_sha256"]
    if committed:
        require_committed(manifest)
    for name, expected in record["gate_input_sha256"].items():
        assert sha(ROOT / name) == expected, name
        if committed:
            require_committed(ROOT / name)
    return dict(status="passed", protocol_sha256=sha(GATE_FREEZE),
                committed_files_required=committed)


def verify_gate_decision(check_files=False):
    frozen, decision = read(GATE_FREEZE), read(OUT / "GATE_RESULT.json")
    assert decision["status"] == "complete"
    assert decision["frozen_gate_protocol_sha256"] == sha(GATE_FREEZE)
    assert decision["gate_data_manifest_sha256"] == frozen["gate_data_manifest_sha256"]
    assert decision["threshold"] == GATE["threshold"] and decision["criterion"] == GATE["criterion"]
    records, bindings = [], {}
    for task in GATE["tasks"]:
        folder = OUT / "gate" / task
        path = folder / "COMPLETE.json"
        record = read(path)
        assert record["status"] == "complete" and record["task"] == task
        assert record["architecture"] == "gru" and record["model"] == GRU_MODEL
        assert record["updates"] == GATE["updates"] and record["batch"] == GATE["batch"]
        assert record["frozen_gate_protocol_sha256"] == sha(GATE_FREEZE)
        assert record["test_evaluated"] is False and record["optimizer_started_empty"] is True
        for name in ("model_seed", "data_seed", "sampling_seed", "dropout_seed"):
            assert record[name] == GATE[name]
        endpoint = record["endpoint_dev"]
        assert endpoint["step"] == GATE["updates"] and endpoint["count"] == GATE["dev_count"]
        assert endpoint["accuracy"] == endpoint["correct"] / endpoint["count"]
        assert record["criterion_passed"] == (endpoint["accuracy"] >= GATE["threshold"])
        if check_files:
            for name, expected in record["file_sha256"].items():
                assert sha(folder / name) == expected, folder / name
            with (folder / "curves.csv").open(encoding="utf-8-sig", newline="") as handle:
                curves = list(csv.DictReader(handle))
            rows = [r for r in curves if r["split"] == "dev" and int(r["step"]) == GATE["updates"]]
            assert len(rows) == 1
            for name in ("accuracy", "correct", "count", "answer_ce"):
                assert float(rows[0][name]) == endpoint[name]
        records.append(record)
        bindings[rel(path)] = sha(path)
    assert decision["runs"] == records
    assert len({r["initial_model_sha256"] for r in records}) == 1
    assert len({r["input_stream_sha256"] for r in records}) == 1
    assert records[0]["input_hashes"] == records[1]["input_hashes"]
    passed = all(r["endpoint_dev"]["accuracy"] >= GATE["threshold"] for r in records)
    assert decision["passed"] == passed
    assert decision["selected_architecture"] == ("gru" if passed else "transformer")
    assert decision["selected_model_config"] == (GRU_MODEL if passed else TRANSFORMER_MODEL)
    assert decision["gate_weights_reused_for_formal_runs"] is False
    assert decision["test_evaluated"] is False and decision["gate_is_confirmatory_unit"] is False
    return bindings


def freeze_gate():
    if GATE_FREEZE.exists():
        return verify_gate()
    assert not (OUT / "gate").exists(), "Gate registration must precede engineering or gate optimization"
    assert not FORMAL_FREEZE.exists()
    manifest_path = OUT / "gate_data/manifest.json"
    manifest = read(manifest_path)
    assert manifest["status"] == "complete"
    from experiments.fixed_points_deficiencies_arch_v1 import data_pipeline
    quality = data_pipeline.verify_gate(replay=True)
    assert quality["status"] == "pass"
    assert len(manifest["pools"]) == 2
    inputs = {rel(manifest_path.parent / row["path"]): row["sha256"]
              for row in manifest["pools"]}
    record = dict(status="frozen", frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        parent_commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
        config=GATE, code_sha256={rel(p): sha(p) for p in gate_paths()},
        gate_data_manifest_path=rel(manifest_path), gate_data_manifest_sha256=sha(manifest_path),
        gate_input_sha256=inputs, gate_optimizer_updates_already_performed=0,
        formal_models_already_trained=0, test_objects_generated=False,
        rule="Commit this registration and all bound files before any smoke or gate optimization")
    write_new(GATE_FREEZE, record)
    return verify_gate(committed=False)


def verify(committed=True):
    from experiments.fixed_points_deficiencies_arch_v1.design import CFG
    record = read(FORMAL_FREEZE)
    assert record["status"] == "frozen" and record["config"] == CFG
    if committed:
        require_committed(FORMAL_FREEZE)
    check_code(record, committed)
    for name, path in (("data_manifest_sha256", OUT / "data/manifest.json"),
                       ("initial_manifest_sha256", OUT / "INITIALS.json"),
                       ("gate_result_sha256", OUT / "GATE_RESULT.json"),
                       ("gate_protocol_sha256", GATE_FREEZE)):
        assert sha(path) == record[name], name
        if committed:
            require_committed(path)
    for entry in read(OUT / "INITIALS.json")["initials"]:
        assert sha(OUT / entry["path"]) == entry["sha256"]
    assert verify_gate_decision() == record["gate_completion_sha256"]
    return dict(status="passed", protocol_sha256=sha(FORMAL_FREEZE),
                committed_files_required=committed, architecture=CFG["architecture"])


def freeze():
    from experiments.fixed_points_deficiencies_arch_v1.design import CFG
    if FORMAL_FREEZE.exists():
        return verify()
    verify_gate()
    gate_completions = verify_gate_decision(check_files=True)
    assert not any(p.is_file() for name in ("pretrain", "adapt", "smoke")
                   for p in (OUT / name).rglob("*")), "Preserve any partial formal/engineering execution; registration must precede optimization"
    assert not (OUT / "SMOKE.json").exists(), "Formal registration precedes its engineering updates"
    assert not (OUT / "data/test_manifest.json").exists()
    quality = read(OUT / "data/quality_pretest.json")
    assert quality["status"] == "pass" and quality["rng_replayed"]
    initials = read(OUT / "INITIALS.json")["initials"]
    assert len(initials) == 8 and len({r["model_sha256"] for r in initials}) == 8
    for row in initials:
        assert sha(OUT / row["path"]) == row["sha256"]
    record = dict(status="frozen", frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        parent_commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
        config=CFG, code_sha256={rel(p): sha(p) for p in formal_paths()},
        data_manifest_sha256=sha(OUT / "data/manifest.json"),
        initial_manifest_sha256=sha(OUT / "INITIALS.json"),
        gate_result_sha256=sha(OUT / "GATE_RESULT.json"), gate_protocol_sha256=sha(GATE_FREEZE),
        gate_completion_sha256=gate_completions,
        formal_models_already_trained=0, formal_smoke_optimizer_updates_already_performed=0,
        test_objects_generated=False,
        expected=dict(source_models=32, adaptation_candidates=360, selections=120,
                      test_checkpoint_evaluations=240, test_prediction_rows=491520),
        smoke_policy="Run the separately protocol-bound engineering smoke after this commit; require its passed record before formal training",
        unit_policy="Eight new units, never pooled with original confirmation",
        rule="No tuning, outcome-driven extension or test reselection")
    write_new(FORMAL_FREEZE, record)
    write_new(OUT / "CONFIG.json", CFG)
    return verify(committed=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze-gate", "verify-gate", "freeze", "verify"))
    args = parser.parse_args()
    result = {"freeze-gate": freeze_gate, "verify-gate": verify_gate,
              "freeze": freeze, "verify": verify}[args.command]()
    print(json.dumps(result, indent=2))
