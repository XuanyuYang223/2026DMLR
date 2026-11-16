"""Record existing mathematical evidence and the pre-optimization local state."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.fixed_points_deficiencies_arch_v1.plan import HERE, ORIGINAL, all_new_seeds


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def save(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    paths = [ORIGINAL / name for name in ("PROTOCOL.md", "RELATION_TABLE.csv", "PREFLIGHT_REPORT.md",
        "preflight_audit/AUDIT.json", "preflight_audit/COMPOSITIONS.csv",
        "preflight_audit/COUNTEREXAMPLES.csv", "preflight_audit/JOINT_DERIVABILITY.csv")]
    old = read(ORIGINAL / "preflight_audit/AUDIT.json")
    assert old["status"] == "passed"
    historical = old["historical_certificates"]
    for name, expected in historical["sha256"].items():
        path = ROOT / historical["root"] / name
        assert sha(path) == expected, path
        paths.append(path)
    for name, expected in historical["implementation_sha256"].items():
        assert sha(ROOT / name) == expected, name
        paths.append(ROOT / name)
    save(HERE / "RELATION_REUSE.json", dict(
        status="reused_unchanged", recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        original_audit_path=relative(ORIGINAL / "preflight_audit/AUDIT.json"),
        mathematical_rechecks_run=0, task_definitions_changed=False,
        sources={relative(p): sha(p) for p in sorted(set(paths))},
        scope="Existing proofs, independent-oracle checks, same-input and transformed-query witnesses are reused by hash; changing architecture does not change the mathematical functions",
        distinctions=["Proven answer relations", "Possible shared subcomputations", "Unverified trained-model mechanisms"],
        caveat="D(p) alone is insufficient; no absence of a registry entry proves unrelatedness"))
    copy = HERE / "RELATION_TABLE.csv"
    with copy.open("xb") as handle:
        handle.write((ORIGINAL / "RELATION_TABLE.csv").read_bytes())
    raw_capacity = HERE / "artifacts/preflight/CAPACITY_PREFLIGHT.json"
    raw = raw_capacity.read_bytes()
    capacity = json.loads(raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"))
    assert capacity["status"] == "pass" and capacity["capacity"]["conservative_capacity_sufficient"]
    save(HERE / "artifacts/preflight/CAPACITY_AUDIT.json", dict(
        **capacity, original_console_path=relative(raw_capacity), original_console_sha256=sha(raw_capacity),
        original_console_encoding="UTF-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "UTF-8",
        capacity_source_sha256_at_execution="84a5c6827f13c48c40cad164ba985014d70ba273ea22e1c9bc1835a341c3bf42",
        note="Numeric capacity result copied without recomputation. A subsequent prefreeze edit strengthens only the deferred-test protocol gate, not history or capacity calculations."))
    save(HERE / "START_STATE.json", dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        head=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
        status_short=git("status", "--short"), working_directory=str(ROOT),
        new_seed_values=all_new_seeds(),
        seed_audit="Read-only word-boundary search of historical experiment Python/JSON/Markdown metadata found no selected formal/gate/engineering seed. Earlier unselected gate stream suggestions2026111602/3 matched old smoke code and were replaced before any gate registration or training.",
        preserved_untracked_file="LOCAL_STATE_SUMMARY.md", new_optimizer_updates=0,
        gate_generated=False, formal_generated=False,
        reference_gru_source={"path": "reference/2026TMLR/experiments/permworld_encoding_archdev_20261009/common.py",
            "sha256": sha(ROOT / "reference/2026TMLR/experiments/permworld_encoding_archdev_20261009/common.py")},
        publication_policy="Normal branch push at completion; no checkpoint upload, force-push or merge"))
    print("Recorded mathematical reuse, capacity transcript and pre-optimization state")


if __name__ == "__main__":
    main()
