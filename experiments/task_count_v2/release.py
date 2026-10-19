"""Inventory the completed study, or verify its archived bytes without training.

Writing requires the final test, independent audit, report, figures, and local
checkpoints. Full verification checks every inventoried file. --source-only is
an explicit verification mode for a Git checkout without the local .pt files.
Paths in this manifest use POSIX separators on every operating system.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST_NAME = "RELEASE_MANIFEST.json"
BASE_COMMIT = "a4d43fc7dafe1bce7e9f86ec4beee78068973ce7"
GIT_FILE_LIMIT = 100 * 1024 * 1024
TEMP_SUFFIXES = {".tmp", ".temp", ".part", ".partial", ".swp", ".pyc", ".pyo"}
REQUIRED_FILES = [
    "train.py", "design.py", "data_pipeline.py", "summarize.py", "audit_run.py", "release.py",
    "EXPERIMENT_PLAN.md", "SMOKE.json", "FROZEN_PROTOCOL.json", "FROZEN_SELECTIONS.json",
    "TRAINING_COMPLETE.json", "TEST_COMPLETE.json", "RUN_AUDIT.json",
    "SOURCE_DISTRIBUTION_AUDIT.json", "source_answer_distributions.csv", "audit_run_accounting.csv",
    "REPORT.md", "README.md", "SUMMARY.json", "raw_test_results.csv",
    "paired_deltas.csv", "summary.csv", "paired_budget_interactions.csv", "budget_interaction_summary.csv",
    "checkpoint_diagnostics.csv", "source_learning.csv", "source_summary.csv", "adaptation_curves.csv",
    "task_exposures.csv", "task_exposure_totals.csv", "compute.csv", "majority_per_length.csv",
    "source_focus.csv", "source_budget_comparisons.csv", "learning_rate_schedule.csv",
    "smoke/end_to_end/COMPLETE.json",
    "results_seed4103.csv", "results_seed5207.csv", "results_seed6301.csv",
    "design/FROZEN_DESIGN.json", "design/configuration.json",
    "data/manifest.json", "data/test_manifest.json", "data/quality_with_test.json",
]
REQUIRED_FILES += [f"figures/{name}.{extension}" for name in (
    "lis_length_budget_comparison", "fixed_points_budget_comparison",
    "lis_length_budget_interaction", "fixed_points_budget_interaction",
    "source_lds_length", "source_increasing_subsequence_count_k3",
    "source_decreasing_subsequence_count_k3", "learning_rate_schedule")
    for extension in ("png", "svg")]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_file(base, value):
    """Consume only safe, relative POSIX paths; do not inherit host separators."""
    check(isinstance(value, str) and "\\" not in value, f"Non-POSIX manifest path: {value!r}")
    relative = PurePosixPath(value)
    check(not relative.is_absolute() and relative.parts and
          all(part not in ("..", ".") and ":" not in part for part in relative.parts),
          f"Invalid manifest path: {value!r}")
    target = base.joinpath(*relative.parts)
    check(target.resolve().is_relative_to(base.resolve()), f"Path escapes study directory: {value}")
    return target


def inventory_paths(out):
    for path in sorted(out.rglob("*")):
        relative = path.relative_to(out)
        if "__pycache__" in relative.parts or relative.as_posix() == MANIFEST_NAME:
            continue
        if path.suffix.lower() in TEMP_SUFFIXES or path.name.endswith("~") or path.name.startswith(".#"):
            continue
        check(not path.is_symlink(), f"Release artifacts must not be symlinks: {relative.as_posix()}")
        if path.is_file():
            yield relative.as_posix(), path


def csv_count(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def completion_gate(out):
    for relative in REQUIRED_FILES:
        path = relative_file(out, relative)
        check(path.is_file() and path.stat().st_size > 0, f"Missing required output: {relative}")
    report_text = (out / "REPORT.md").read_text(encoding="utf-8-sig")
    check(any("\u4e00" <= character <= "\u9fff" for character in report_text),
          "The requested Chinese report is missing")
    protocol = read_json(out / "FROZEN_PROTOCOL.json")
    selections = read_json(out / "FROZEN_SELECTIONS.json")
    tested = read_json(out / "TEST_COMPLETE.json")
    audited = read_json(out / "RUN_AUDIT.json")
    summary = read_json(out / "SUMMARY.json")
    check(protocol["status"] == selections["status"] == "frozen", "Protocol/selections are not frozen")
    check(selections["training_complete"] is True and len(selections["selections"]) == 54,
          "The 54 development selections are incomplete")
    check(tested["status"] == "passed" and tested["checkpoint_evaluations"] == 108,
          "Independent testing is incomplete")
    check(tested["raw_results_sha256"] == sha(out / "raw_test_results.csv"), "Raw test CSV binding changed")
    check(tested["frozen_selection_sha256"] == sha(out / "FROZEN_SELECTIONS.json"), "Test selection binding changed")
    check(audited["status"] == "pass" and audited.get("raw_result_rows") == 324,
          "A passing audit of the final test is required")
    check(audited["pretraining_runs"] == 24 and audited["target_training_runs"] == 54,
          "Independent audit has the wrong run counts")
    check(audited["frozen_protocol_sha256"] == sha(out / "FROZEN_PROTOCOL.json"), "Audit protocol binding changed")
    check(summary["status"] == "completed" and summary["pretraining_models"] == 24 and
          summary["target_training_runs"] == 54, "Summary is incomplete")
    check(summary["raw_results_sha256"] == sha(out / "raw_test_results.csv"), "Summary input binding changed")
    check(summary["frozen_protocol_sha256"] == sha(out / "FROZEN_PROTOCOL.json"), "Summary protocol binding changed")
    quality = read_json(out / "data" / "quality_with_test.json")
    check(quality["status"] == "pass" and quality["include_test"] is True, "Final orbit/data audit is missing")
    distributions = read_json(out / "SOURCE_DISTRIBUTION_AUDIT.json")
    check(distributions["status"] == "pass", "Source distribution audit failed")
    check(distributions["csv_sha256"] == sha(out / "source_answer_distributions.csv"), "Source distribution CSV binding changed")
    check(csv_count(out / "raw_test_results.csv") == 324, "Wrong raw test row count")
    check(csv_count(out / "audit_run_accounting.csv") == 78, "Wrong accounting audit row count")
    for stage, expected in (("pretraining", 24), ("adaptation", 54)):
        completed = list((out / stage).glob("*/COMPLETE.json"))
        check(len(completed) == expected, f"Wrong completed run count in {stage}")
        for path in completed:
            record = read_json(path)
            for relative, digest in record["artifact_sha256"].items():
                artifact = relative_file(path.parent, relative)
                check(artifact.is_file(), f"Missing completed-run artifact: {artifact}")
                # Hashes are all computed once by the final inventory below;
                # completion records are checked against it afterwards.
                check(isinstance(digest, str) and len(digest) == 64, f"Malformed artifact digest: {artifact}")
    for seed in (4103, 5207, 6301):
        check((out / "initial" / f"seed{seed}.pt").is_file(), f"Missing initial checkpoint for seed {seed}")
    if audited.get("development_checkpoints_recomputed", 0):
        check((out / "audit_development_forwards.csv").is_file(), "Missing recorded development-forward audit")
    return {"pretraining_models": 24, "target_training_runs": 54,
            "paired_repeat_units": 3, "raw_test_rows": 324, "checkpoint_test_evaluations": 108}


def git_value(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def write_manifest(out):
    counts = completion_gate(out)
    files = {}
    for relative, path in inventory_paths(out):
        local = path.suffix.lower() == ".pt"
        size = path.stat().st_size
        check(local or size < GIT_FILE_LIMIT, f"Git deliverable is at least 100 MiB: {relative} ({size} bytes)")
        files[relative] = {"bytes": size, "sha256": sha(path),
                           "delivery": "local_checkpoint" if local else "git_deliverable"}
    for stage in ("pretraining", "adaptation"):
        for path in (out / stage).glob("*/COMPLETE.json"):
            for relative, digest in read_json(path)["artifact_sha256"].items():
                key = relative_file(path.parent, relative).relative_to(out).as_posix()
                check(files[key]["sha256"] == digest, f"Completed-run artifact changed: {key}")
    categories = {category: {"files": sum(v["delivery"] == category for v in files.values()),
        "bytes": sum(v["bytes"] for v in files.values() if v["delivery"] == category)}
        for category in ("git_deliverable", "local_checkpoint")}
    result = {"version": "task_count_v2_release_manifest_v1", "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "path_base": "directory containing RELEASE_MANIFEST.json", "path_separator": "/",
        "base_commit": BASE_COMMIT, "branch": git_value("branch", "--show-current"),
        "worktree_head_at_manifest_creation": git_value("rev-parse", "HEAD"),
        "checkpoint_policy": "All .pt files remain local and are ignored by Git; all other inventoried files are Git deliverables.",
        "report_language": "Chinese, as explicitly requested for task-count v2",
        "verification_modes": {"default": "Check every listed file, including local checkpoints.",
                               "source_only": "Check all Git deliverables; explicitly skip all .pt checkpoints."},
        "exclusions": [MANIFEST_NAME, "__pycache__", "temporary files and Python bytecode"],
        "counts": {**counts, "files": len(files), "bytes": sum(v["bytes"] for v in files.values()),
                   "by_delivery": categories}, "files": files}
    target = out / MANIFEST_NAME
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    return {"status": "written", **result["counts"]}


def verify_manifest(out, source_only=False):
    value = read_json(out / MANIFEST_NAME)
    check(value["version"] == "task_count_v2_release_manifest_v1", "Unknown manifest version")
    check(value["base_commit"] == BASE_COMMIT, "Unexpected base commit")
    expected, skipped = set(), 0
    for relative, record in value["files"].items():
        path = relative_file(out, relative)
        check(record["delivery"] in ("local_checkpoint", "git_deliverable"), f"Unknown delivery type: {relative}")
        local = record["delivery"] == "local_checkpoint"
        check(local == (path.suffix.lower() == ".pt"), f"Wrong checkpoint classification: {relative}")
        if source_only and local:
            skipped += 1
            continue
        check(local or record["bytes"] < GIT_FILE_LIMIT, f"Oversized Git deliverable: {relative}")
        expected.add(relative)
        check(path.is_file() and not path.is_symlink(), f"Missing or invalid artifact: {relative}")
        check(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], f"Artifact changed: {relative}")
    actual = {relative for relative, path in inventory_paths(out) if not (source_only and path.suffix.lower() == ".pt")}
    check(actual == expected, f"Manifest inventory mismatch: extra={sorted(actual - expected)}, missing={sorted(expected - actual)}")
    return {"status": "passed", "mode": "source_only" if source_only else "full",
            "files_verified": len(expected), "local_checkpoints_explicitly_skipped": skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--source-only", action="store_true", help="Verify Git deliverables, explicitly skipping local .pt files")
    args = parser.parse_args()
    if args.source_only and not args.verify:
        parser.error("--source-only requires --verify")
    out = args.out.resolve()
    result = verify_manifest(out, args.source_only) if args.verify else write_manifest(out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
