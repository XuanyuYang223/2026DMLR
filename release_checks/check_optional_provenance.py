"""Prove default DMLR verification/figures work with optional audits absent.

Run only in an isolated acceptance clone. Each optional file is moved to a new
outside directory and restored in finally; no recursive move/delete is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    if out.exists() or out.is_relative_to(repo):
        raise SystemExit("Use a fresh output directory outside the acceptance clone")
    out.mkdir(parents=True)
    names = ("audit_original_permworld.py", "original_permworld_audit.md", "source_inventory.csv",
             "upstream_alignment_by_relation.csv", "upstream_alignment_by_seed.csv",
             "upstream_alignment_final_norm.csv", "UPSTREAM_AUDIT.json", "upstream_resource_inventory.json")
    evidence = (repo / "publication/evidence").resolve()
    moved, hashes, commands = [], {}, []
    try:
        for name in names:
            source, destination = (evidence / name).resolve(), (out / name).resolve()
            assert source.parent == evidence and source.is_relative_to(repo)
            assert destination.parent == out and not destination.exists()
            hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
            os.replace(source, destination)
            moved.append((source, destination))
        assert all(not (evidence / name).exists() for name in names)
        for name, argv in (
            ("verify", [sys.executable, "-I", str(repo / "publication/verify.py")]),
            ("figures", [sys.executable, "-I", str(repo / "publication/build_figures.py"),
                         "--out", str(out / "figures")]),
        ):
            process = subprocess.run(argv, cwd=out, text=True, encoding="utf8", errors="replace",
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (out / (name + ".log")).write_text(process.stdout, encoding="utf8")
            commands.append({"name": name, "argv": argv, "exit_code": process.returncode})
            if process.returncode:
                raise RuntimeError(f"DMLR {name} depended on optional provenance: {process.stdout}")
    finally:
        for source, destination in reversed(moved):
            assert source.parent == evidence and destination.parent == out and not source.exists()
            os.replace(destination, source)
            assert hashlib.sha256(source.read_bytes()).hexdigest() == hashes[source.name]
    report = {"status": "passed", "optional_files_absent_during_commands": list(names),
              "all_files_restored_with_identical_sha256": True, "sha256": hashes,
              "commands": commands, "training_runs": 0, "model_forwards": 0}
    (out / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
