"""Freeze and verify the local confirmation before formal training."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from design import CFG, HERE, ROOT

OUT = HERE / "artifacts"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def paths():
    own = [HERE / name for name in ["design.py", "data_pipeline.py", "train.py", "protocol.py",
        "analysis.py", "independent_recompute.py", "stats_plan.json", "PROTOCOL.md", "RELATION_TABLE.csv"]]
    inherited = [ROOT / "experiments/pilot_v3_multitarget/pilot_v3.py",
        ROOT / "experiments/permworld_pilot/pilot.py",
        ROOT / "experiments/pilot_v5_confirmation/artifacts/config.json",
        ROOT / "experiments/task_count_v2/data_pipeline.py"]
    inherited += sorted((ROOT / "experiments/permworld_pilot/vendor").rglob("*.py"))
    inherited += sorted((ROOT / "permworld_platform").glob("*.py"))
    return own + inherited


def verify():
    frozen = read(OUT / "FROZEN_PROTOCOL.json")
    assert frozen["status"] == "frozen" and frozen["config"] == CFG
    for path, expected in frozen["code_sha256"].items():
        assert sha(ROOT / path) == expected, path
    for key, path in [("data_manifest_sha256", "data/manifest.json"),
        ("initial_manifest_sha256", "INITIALS.json"), ("smoke_sha256", "SMOKE.json")]:
        assert sha(OUT / path) == frozen[key], key
    return {"status": "passed", "protocol_sha256": sha(OUT / "FROZEN_PROTOCOL.json"),
            "bound_files": len(frozen["code_sha256"])}


def freeze():
    if (OUT / "FROZEN_PROTOCOL.json").exists():
        return verify()
    assert not list((OUT / "pretrain").glob("*/completed.json"))
    assert not list((OUT / "adapt").glob("*/completed.json"))
    assert not (OUT / "data/test_manifest.json").exists()
    assert read(OUT / "data/quality_pretest.json")["status"] == "pass"
    assert read(OUT / "data/quality_pretest.json")["rng_replayed"]
    assert read(OUT / "SMOKE.json")["status"] == "passed"
    initials = read(OUT / "INITIALS.json")["initials"]
    assert len(initials) == 8 and len({r["model_sha256"] for r in initials}) == 8
    for row in initials:
        assert sha(OUT / row["path"]) == row["sha256"]
    record = dict(status="frozen", frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        starting_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        config=CFG, code_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in paths()},
        data_manifest_sha256=sha(OUT / "data/manifest.json"),
        initial_manifest_sha256=sha(OUT / "INITIALS.json"), smoke_sha256=sha(OUT / "SMOKE.json"),
        formal_models_already_trained=0, test_objects_generated=False,
        expected=dict(source_models=32, adaptation_candidates=360, selections=120,
                      test_checkpoint_evaluations=240, test_prediction_rows=491520),
        rule="No outcome-based extension or reselection. Local delivery only.")
    (OUT / "FROZEN_PROTOCOL.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf8")
    (OUT / "CONFIG.json").write_text(json.dumps(CFG, indent=2)+"\n", encoding="utf8")
    return verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "verify"])
    args = parser.parse_args()
    print(json.dumps(globals()[args.command](), indent=2))
