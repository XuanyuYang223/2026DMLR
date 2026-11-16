"""Bind the author-approved shared-pool calibration before any optimizer step."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / "experiments/platform_v01/calibration"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def main():
    destination = HERE / "FROZEN_PROTOCOL.json"
    alias = HERE / "artifacts/FROZEN_CALIBRATION.json"
    if destination.exists() or alias.exists():
        raise FileExistsError("Preserve the existing freeze; this command is single-use")
    original = read(OLD / "FROZEN_CALIBRATION.json")
    reuse = read(HERE / "POOL_REUSE_AUDIT.json")
    encoding = read(HERE / "ENCODING_PREFLIGHT.json")
    if reuse["status"] != "passed" or reuse["historical_files_changed"]:
        raise RuntimeError("Canonical-pool reuse provenance did not pass")
    if reuse["reserved_test_input_files_opened"] or reuse["model_forward_calls"]:
        raise RuntimeError("The reuse audit exceeded its read-only scope")
    assert encoding["scoped_instance_count"] == 50
    assert len(encoding["accepted_instances"]) == 49
    assert len(encoding["rejected_instances"]) == 1
    configuration = copy.deepcopy(original["config"])
    configuration.update(
        version="platform_calibration_v02/shared_canonical_units_v1",
        representatives=encoding["accepted_instances"],
        representative_selection="All remaining materialized unary non-alias instances; one whole-instance reader rejection is retained explicitly",
        scope="49 eligible instances times two original paired units; one additional instance recorded as reader-rejected; no test forwards or transfer",
        reporting="Fixed update-2400 complete-answer development accuracy under the unchanged eight-token generation budget, with structural ceilings disclosed",
        pairing="Same canonical train/dev pools, seed pairing and initialization as v0.1; shared data units, not independent additional repeats",
    )
    pools = {}
    for relative, expected in original["training_development_data_sha256"].items():
        path = (OLD / relative).resolve()
        if sha(path) != expected:
            raise RuntimeError(f"Canonical pool changed: {path}")
        pools[rel(path)] = expected
    assert len(pools) == 12 and all("_test_" not in key for key in pools)
    required = [HERE / name for name in (
        "loader_runner.py", "report_and_audit.py", "freeze.py", "verify_pool_reuse.py", "encoding_preflight.py",
        "PROTOCOL.md", "ANALYSIS_PLAN.md", "AUTHOR_SCOPE_DECISION.md",
        "POOL_REUSE_AUDIT.json", "ENCODING_PREFLIGHT.json", "ENCODING_PREFLIGHT.csv",
    )]
    required.extend(ROOT / p for p in original["code_and_registry_sha256"])
    required += [OLD / "FROZEN_CALIBRATION.json", OLD.parent / "dataset/manifest.json",
                 OLD.parent / "LEARNABILITY_INDEX.csv", OLD.parent / "registry/TASK_REGISTRY.json"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    initial = {}
    for repeat in configuration["repeats"]:
        seed = repeat["initialization_seed"]
        path = OLD / "initial" / f"seed{seed}.pt"
        initial[str(seed)] = {"path": rel(path), "sha256": sha(path)}
    payload = {
        "status": "frozen_before_any_new_optimizer_update",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "inherited_config_path": rel(OLD / "FROZEN_CALIBRATION.json"),
        "inherited_config_sha256": sha(OLD / "FROZEN_CALIBRATION.json"),
        "config": configuration,
        "accepted_instances": encoding["accepted_instances"],
        "rejected_instances": encoding["rejected_instances"],
        "requested_instances": 50,
        "planned_formal_runs": 98,
        "independent_new_data_units": 0,
        "shared_canonical_data_units": 2,
        "code_sha256": {rel(p): sha(p) for p in sorted(set(required))},
        "train_dev_sha256": dict(sorted(pools.items())),
        "reused_initial_checkpoint_sha256": initial,
        "commit_requirement": "This protocol and every bound source/plan file must match HEAD before any smoke or formal optimizer update",
        "reserved_test_input_files_opened": 0,
        "test_model_forwards_planned": 0,
        "new_input_generation_planned": False,
        "python": sys.version,
        "freeze_command": ".venv/Scripts/python.exe -B experiments/platform_calibration_v02/reuse_authorized_v1/freeze.py",
    }
    data = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf8")
    alias.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(data)
    with alias.open("xb") as handle:
        handle.write(data)
    print(json.dumps({"protocol": rel(destination), "sha256": sha(destination),
                      "formal_runs": 98, "reader_rejected_instances": 1,
                      "next": "Commit bound files before invoking loader_runner.py smoke"}))


if __name__ == "__main__":
    main()
