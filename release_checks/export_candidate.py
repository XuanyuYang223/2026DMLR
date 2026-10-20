"""Export exact staged Git bytes for pre-commit isolated-checkout acceptance.

Disable external diff/text conversion: PDF textconv output is not an applicable
binary patch. The generated patch is a temporary transport artifact, not a new
source dependency. Final publication commits the accepted source files normally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-against", type=Path)
    args = parser.parse_args()
    path = args.out.resolve()
    if path.exists():
        raise SystemExit("Use a fresh export path; previous snapshots are retained")
    command = ["git", "diff", "--cached", "--binary", "--full-index", "--no-textconv", "--no-ext-diff", "HEAD"]
    raw = subprocess.check_output(command)
    path.write_bytes(raw)
    if args.check_against:
        subprocess.run(["git", "-C", str(args.check_against.resolve()), "apply", "--check", "--index",
                        "--whitespace=nowarn", str(path)], check=True)
    print(json.dumps({"argv": command, "patch": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                      "bytes": len(raw), "index_tree": subprocess.check_output(["git", "write-tree"], text=True).strip(),
                      "read_only_apply_check_passed": bool(args.check_against)}, indent=2))
