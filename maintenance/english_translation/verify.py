"""Verify the English translation without rewriting historical experiment evidence.

Run from any directory. Historical hashes refer to BASE_REVISION; a translated
file is accepted only when its original Git bytes and current English bytes both
match the separate translation ledger. All other archived files require an exact
hash match. This does not make the translated files byte-identical to history.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE_REVISION = "c4aeb499a0c7f2dc59b70e438f66c5b6dad4244a"
LEDGER = HERE / "TRANSLATION_MANIFEST.json"
HAN = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002fa1f]")
CJK_PUNCTUATION = re.compile("[\u3001\u3002\uff01\uff08\uff09\uff0c\uff1a\uff1b\uff1f]")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def decode(data):
    encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    return data.decode(encoding)


def original(path):
    return git("show", f"{BASE_REVISION}:{path}")


def compare_python(before, after):
    """Require the same executable AST except for string-literal wording."""
    old, new = ast.parse(before), ast.parse(after)
    changed = 0

    def compare(a, b):
        nonlocal changed
        assert type(a) is type(b), f"AST node type changed: {type(a).__name__}"
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            if a.value != b.value:
                assert HAN.search(a.value) or CJK_PUNCTUATION.search(a.value), "An unrelated English string changed; review explicitly"
                assert not HAN.search(b.value), "Untranslated Python string"
                changed += 1
            return
        if isinstance(a, ast.AST):
            for field in a._fields:
                compare(getattr(a, field), getattr(b, field))
        elif isinstance(a, list):
            assert len(a) == len(b), "AST list length changed"
            for left, right in zip(a, b):
                compare(left, right)
        else:
            assert a == b, f"Non-string AST value changed: {a!r} -> {b!r}"

    compare(old, new)
    return changed


def compare_csv(before, after):
    old = list(csv.reader(io.StringIO(before)))
    new = list(csv.reader(io.StringIO(after)))
    assert len(old) == len(new), "CSV row count changed"
    changed = 0
    for left, right in zip(old, new):
        assert len(left) == len(right), "CSV column count changed"
        for a, b in zip(left, right):
            if a != b:
                assert HAN.search(a) and not HAN.search(b), "Non-localized CSV cell changed"
                changed += 1
    return changed


def validate_translation(path, before_bytes, after_bytes):
    before, after = decode(before_bytes), decode(after_bytes)
    assert HAN.search(before), f"Unexpected non-translation edit: {path}"
    assert not HAN.search(after), f"Untranslated content: {path}"
    suffix = Path(path).suffix
    assert suffix in (".md", ".py", ".csv"), f"Unexpected translated format: {path}"
    if suffix == ".py":
        return {"changed_string_literals": compare_python(before, after), "executable_ast_preserved": True}
    if suffix == ".csv":
        return {"translated_cells": compare_csv(before, after), "all_other_cells_preserved": True}
    return {"review": "English prose; numerical statements, equations, commands, and links reviewed separately"}


def record():
    changed = git("diff", "--name-only", "-z", BASE_REVISION).decode().split("\0")
    files = {}
    for rel in sorted(filter(None, changed)):
        if rel.startswith("maintenance/english_translation/"):
            continue
        assert (ROOT / rel).is_file(), f"Deleted file: {rel}"
        before, after = original(rel), (ROOT / rel).read_bytes()
        checks = validate_translation(rel, before, after)
        files[rel] = {"original_sha256": digest(before), "english_sha256": digest(after), **checks}
    assert files, "No translations found"
    result = {
        "schema_version": 1,
        "base_revision": BASE_REVISION,
        "scope": "English translation of documentation, report templates, and textual CSV classifications",
        "historical_manifests_rewritten": False,
        "training_or_model_evaluation_run": False,
        "files": files,
    }
    LEDGER.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"ledger_written": str(LEDGER.relative_to(ROOT)), "translated_files": len(files)}))


def verify(skip_archived_bytes=False):
    ledger = json.loads(LEDGER.read_text(encoding="utf8"))
    assert ledger["base_revision"] == BASE_REVISION
    files = ledger["files"]
    originals = {}
    python_files = csv_cells = 0
    # A later publication entry page is a substantive documentation revision,
    # not another translation. Keep the original translation ledger immutable
    # and validate the intermediate English Git bytes through an explicit bridge.
    bridge_paths = [ROOT / "maintenance/documentation_revisions" / name for name in
                    ("publication_integration.json", "release_check.json")]
    bridges = [json.loads(path.read_text(encoding="utf8")) for path in bridge_paths if path.exists()]
    documentation_revisions = 0
    for rel, entry in files.items():
        before, after = original(rel), (ROOT / rel).read_bytes()
        assert digest(before) == entry["original_sha256"], f"Original Git content mismatch: {rel}"
        relevant = [bridge for bridge in bridges if rel in bridge["files"]]
        if digest(after) != entry["english_sha256"] and relevant:
            expected = entry["english_sha256"]
            intermediate = None
            for bridge in relevant:
                assert set(bridge["files"]) == {"README.md"}, "Unexpected documentation bridge scope"
                assert bridge["base_revision"] in {
                    "95e84e6af5de2c0a692da2f64bf95745455f5c3c", "7ef277d3f2cdd6faef0d2f7c1b844e3b0e1bcf9e"}
                revision = bridge["files"][rel]
                prior = git("show", f"{bridge['base_revision']}:{rel}")
                assert digest(prior) == revision["before_sha256"] == expected
                if intermediate is None:
                    intermediate = prior
                expected = revision["after_sha256"]
                documentation_revisions += 1
            assert digest(after) == expected, f"Publication documentation changed: {rel}"
            assert not HAN.search(decode(after)), f"Non-English publication documentation: {rel}"
            after = intermediate
        assert digest(after) == entry["english_sha256"], f"English translation changed: {rel}"
        checks = validate_translation(rel, before, after)
        python_files += int(checks.get("executable_ast_preserved", False))
        csv_cells += checks.get("translated_cells", 0)
        originals[rel] = digest(before)

    # Check tracked and non-ignored untracked text; dependencies and Git history
    # are not repository content. Binary model/data/image artifacts stay intact.
    candidates = set(filter(None, git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")))
    text_files = 0
    for rel in sorted(candidates):
        if Path(rel).suffix.lower() in (".pt", ".npz", ".png", ".jpg", ".pdf"):
            continue
        try:
            text = decode((ROOT / rel).read_bytes())
        except UnicodeError:
            continue
        assert not HAN.search(text), f"Remaining Han characters: {rel}"
        text_files += 1

    checked, translated_references = {}, 0

    def check(path, expected):
        nonlocal translated_references
        path = path.resolve()
        rel = path.relative_to(ROOT).as_posix()
        if rel in files:
            assert originals[rel] == expected, f"Historical translation origin mismatch: {rel}"
            translated_references += 1
        elif not skip_archived_bytes:
            if rel not in checked:
                with path.open("rb") as handle:
                    checked[rel] = hashlib.file_digest(handle, "sha256").hexdigest()
            assert checked[rel] == expected, f"Non-translated archived file changed: {rel}"

    platform = ROOT / "experiments/platform_v01"
    preservation = json.loads((platform / "preservation.json").read_text())
    release = json.loads((platform / "PLATFORM_MANIFEST.json").read_text())
    for rel, expected in preservation["files"].items():
        check(ROOT / rel, expected)
    for rel, item in release["files"].items():
        check(ROOT / rel, item["sha256"])
    # v4/v5 also froze prose as design evidence. Validate those references via
    # original Git bytes without rebinding the scientific protocol.
    for pilot in ("pilot_v4_backgrounds", "pilot_v5_confirmation"):
        frozen = json.loads((ROOT / "experiments" / pilot / "artifacts/FROZEN_PROTOCOL.json").read_text())
        for rel, expected in frozen["code"].items():
            check(ROOT / rel, expected)

    result = {
        "status": "passed",
        "base_revision": BASE_REVISION,
        "translated_files": len(files),
        "python_files_with_same_executable_ast_except_translated_strings": python_files,
        "csv_text_cells_translated_all_other_cells_preserved": csv_cells,
        "repository_text_files_scanned": text_files,
        "remaining_han_files": 0,
        "unique_untranslated_archived_files_hash_verified": len(checked),
        "historical_references_resolved_through_translation_ledger": translated_references,
        "subsequent_documentation_revisions_verified": documentation_revisions,
        "archived_byte_checks_skipped": skip_archived_bytes,
        "historical_manifests_rewritten": False,
        "training_or_model_evaluation_run": False,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="Create the ledger after reviewing the English translation")
    parser.add_argument("--skip-archived-bytes", action="store_true", help="Only check translated files and language coverage")
    parser.add_argument("--report", type=Path, help="Write a new verification result, separate from historical audits")
    args = parser.parse_args()
    if args.record:
        record()
    else:
        result = verify(args.skip_archived_bytes)
        if args.report:
            args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
