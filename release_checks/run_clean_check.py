"""Create a fresh clone/venv and log CPU-only release acceptance commands.

Each invocation requires a new work directory. No existing checkout/environment
is deleted or reused. This script uses only the bootstrap interpreter's stdlib.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Git repository URL or path, never a copied worktree")
    parser.add_argument("--branch", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--patch", type=Path, help="Binary Git-index patch to apply to the fresh clone before installation")
    parser.add_argument("--expect-missing-packaging", action="store_true")
    args = parser.parse_args()
    work, logs = args.work.resolve(), args.logs.resolve()
    if work.exists() or logs.exists():
        raise SystemExit("Use new work and log directories; completed or failed attempts are retained")
    work.mkdir(parents=True)
    logs.mkdir(parents=True)
    repo, environment = work / "repo", work / "venv"
    outside = work / "outside_checkout"
    outside.mkdir()
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    records = []
    result = {"schema_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "bootstrap_python": sys.version, "bootstrap_executable": sys.executable,
              "platform": platform.platform(), "work_directory": str(work),
              "source": args.source, "branch": args.branch,
              "isolated_python_flag": True, "system_site_packages": False,
              "commands": records, "training_runs": 0, "neural_forwards": 0,
              "status": "running"}

    def save():
        (logs / "RUN.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")

    def run(name, command, cwd, expected=0):
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, encoding="utf8", errors="replace")
        record = {"name": name, "argv": list(map(str, command)), "cwd": str(cwd),
                  "exit_code": completed.returncode, "expected_exit_code": expected,
                  "seconds": time.perf_counter() - started, "log": name + ".log"}
        records.append(record)
        (logs / record["log"]).write_text(completed.stdout, encoding="utf8")
        save()
        print(json.dumps(record), flush=True)
        if completed.returncode != expected:
            result["status"] = "failed"
            result["failed_command"] = name
            save()
            raise SystemExit(f"{name} failed; see {logs / record['log']}")
        return completed.stdout

    run("01_clone", ["git", "clone", "--no-local", "--single-branch", "--branch", args.branch,
                     args.source, str(repo)], work)
    result["checkout_head"] = run("02_revision", ["git", "rev-parse", "HEAD"], repo).strip()
    run("03_initial_status", ["git", "status", "--porcelain=v1"], repo)
    if args.patch:
        patch = args.patch.resolve()
        result["candidate_patch_sha256"] = hashlib.sha256(patch.read_bytes()).hexdigest()
        run("03a_apply_candidate", ["git", "apply", "--index", "--whitespace=nowarn", str(patch)], repo)
        result["candidate_git_tree"] = run("03b_candidate_tree", ["git", "write-tree"], repo).strip()
    assert not (repo / ".venv").exists() and not (repo / "reference").exists()
    assert not list(repo.rglob("*.pt")), "Clean source checkout unexpectedly contains checkpoints"
    run("04_create_venv", [sys.executable, "-I", "-m", "venv", str(environment)], work)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    configuration = (environment / "pyvenv.cfg").read_text(encoding="utf8")
    assert "include-system-site-packages = false" in configuration
    (logs / "pyvenv.cfg").write_text(configuration, encoding="utf8")
    run("05_initial_packages", [str(python), "-I", "-m", "pip", "list", "--format=json"], outside)
    expected = 1 if args.expect_missing_packaging else 0
    run("06_install", [str(python), "-I", "-m", "pip", "--isolated", "install", str(repo) + "[publication]"], outside, expected)
    if args.expect_missing_packaging:
        assert not (repo / "pyproject.toml").exists() and not (repo / "setup.py").exists()
        result["status"] = "expected_packaging_failure_recorded"
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        return

    run("07_installed_packages", [str(python), "-I", "-m", "pip", "freeze", "--all"], outside)
    run("08_dependency_check", [str(python), "-I", "-m", "pip", "check"], outside)
    run("09_interfaces", [str(python), "-I", str(repo / "release_checks/acceptance.py"),
                          "--repo", str(repo), "--out", str(work / "acceptance")], outside)
    run("10_dmlr_recompute", [str(python), "-I", str(repo / "publication/evidence/build_dmlr_evidence.py")], outside)
    run("11_figures", [str(python), "-I", str(repo / "publication/build_figures.py"),
                       "--out", str(work / "rebuilt_figures")], outside)
    run("12_budget_interactions", [str(python), "-I", str(repo / "release_checks/check_budget_interactions.py"),
                                   "--repo", str(repo), "--out", str(work / "budget_interactions.json"),
                                   "--rebuilt-figures", str(work / "rebuilt_figures")], outside)
    run("13_source_only", [str(python), "-I", str(repo / "experiments/task_count_v2/release.py"),
                           "--verify", "--source-only"], outside)
    run("14_publication_verify", [str(python), "-I", str(repo / "publication/verify.py")], outside)
    run("15_language_verification", [str(python), "-I", str(repo / "maintenance/english_translation/task_count_v2/verify.py")], outside)
    run("15a_optional_provenance_absent", [str(python), "-I", str(repo / "release_checks/check_optional_provenance.py"),
                                          "--repo", str(repo), "--out", str(work / "optional_provenance_check")], outside)
    run("16_final_status", ["git", "status", "--porcelain=v1"], repo)
    run("17_candidate_bytes_unchanged", ["git", "diff", "--exit-code"], repo)
    run("18_scientific_paths_unchanged", ["git", "diff", "--exit-code", result["checkout_head"],
                                         "--", "experiments", "permworld_platform"], repo)
    result["status"] = "passed"
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save()


if __name__ == "__main__":
    main()
