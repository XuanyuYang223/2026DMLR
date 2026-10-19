"""Record or verify release bytes, including locally retained checkpoint files."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "PLATFORM_MANIFEST.json"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
        for rel, spec in manifest["files"].items():
            path = ROOT / rel
            assert path.is_file() and path.stat().st_size == spec["bytes"] and sha(path) == spec["sha256"], rel
        print(json.dumps(dict(status="passed", files=len(manifest["files"]))))
        return
    platform_audit = json.loads((HERE / "PLATFORM_AUDIT.json").read_text(encoding="utf8"))
    assert platform_audit["status"] == "passed"
    assert (HERE / "PLATFORM_REPORT.md").exists()
    paths = {ROOT / "README.md"}
    for directory in (ROOT / "permworld_platform", HERE):
        paths.update(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p != MANIFEST)
    result = dict(version="0.1.0", path_base="repository root", checkpoint_git_policy="retained locally, .pt ignored by git",
                  files={p.relative_to(ROOT).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(paths)})
    MANIFEST.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(dict(status="written", files=len(result["files"]), bytes=sum(s["bytes"] for s in result["files"].values()))))


if __name__ == "__main__":
    main()
