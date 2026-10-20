"""Verify the English revision without changing completed scientific evidence."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STUDY = ROOT / "experiments/task_count_v2"
BASE = "d5f2b5c22e241322c619ada0e9ec7642b5b1eced"
APPROVED = {"REPORT.md", "README.md", "summarize.py", "release.py"}
HAN = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]")
CJK_PUNCTUATION = re.compile("[\u3001\u3002\uff01\uff08\uff09\uff0c\uff1a\uff1b\uff1f]")
NUMBER = re.compile(r"[+\-\u2212]?\d[\d,]*(?:\.\d+)?")
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def before(name):
    return git("show", f"{BASE}:experiments/task_count_v2/{name}")


def check_generator(old, new):
    changed = []

    def compare(a, b):
        assert type(a) is type(b), "Generator AST node type changed"
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            if a.value != b.value:
                assert HAN.search(a.value) or CJK_PUNCTUATION.search(a.value) or a.value.startswith(
                    "# PermWorld task-count v2: budget interaction"), "Unrelated English literal changed"
                assert not HAN.search(b.value), "Untranslated generator literal"
                changed.append(a.lineno)
            return
        if isinstance(a, ast.AST):
            for field in a._fields:
                compare(getattr(a, field), getattr(b, field))
        elif isinstance(a, list):
            assert len(a) == len(b), "Generator AST list length changed"
            for x, y in zip(a, b):
                compare(x, y)
        else:
            assert a == b, f"Non-string generator AST value changed: {a!r} -> {b!r}"

    compare(ast.parse(old), ast.parse(new))
    return {"executable_ast_preserved_except_language_strings": True,
            "changed_string_literals": len(changed)}


def numeric_tables(text):
    return [[NUMBER.findall(cell) for cell in line.split("|")[1:-1]]
            for line in text.splitlines() if line.startswith("|")]


def verify():
    ledger = json.loads((HERE / "TRANSLATION_MANIFEST.json").read_text(encoding="utf-8"))
    assert ledger["schema_version"] == 1 and ledger["base_revision"] == BASE
    assert set(ledger["files"]) == APPROVED
    archived_release = (STUDY / "RELEASE_MANIFEST.json").read_bytes()
    assert digest(archived_release) == ledger["original_release_manifest_sha256"]
    assert archived_release == before("RELEASE_MANIFEST.json")
    archive = json.loads(archived_release)["files"]
    for name, entry in ledger["files"].items():
        old, new = before(name), (STUDY / name).read_bytes()
        assert digest(old) == entry["original_sha256"] == archive[name]["sha256"], name
        assert digest(new) == entry["english_sha256"] and len(new) == entry["english_bytes"], name
        assert not HAN.search(new.decode("utf-8-sig")), name
    generator = check_generator(before("summarize.py").decode("utf-8-sig"),
                                (STUDY / "summarize.py").read_text(encoding="utf-8-sig"))
    old_report = before("REPORT.md").decode("utf-8-sig")
    new_report = (STUDY / "REPORT.md").read_text(encoding="utf-8-sig")
    assert numeric_tables(old_report) == numeric_tables(new_report), "Report numeric table cells changed"
    old_links, new_links = Counter(LINK.findall(old_report)), Counter(LINK.findall(new_report))
    assert not old_links - new_links, "An original report evidence link was removed"
    # The existing generator condition now sees the already completed
    # preservation audit, which did not yet exist when the old report was made.
    assert new_links - old_links in (Counter(), Counter({"PRESERVATION_AFTER.json": 1})), "Unexpected evidence link added"
    for name in ("README.md", "REPORT.md"):
        for link in LINK.findall((STUDY / name).read_text(encoding="utf-8-sig")):
            if "://" not in link and not link.startswith("#"):
                assert (STUDY / link.split("#")[0]).exists(), f"Broken link: {link}"
    changed = git("diff", "--name-only", BASE, "--", "experiments/task_count_v2").decode().splitlines()
    assert set(changed) == {"experiments/task_count_v2/" + name for name in APPROVED}, changed
    untracked = git("ls-files", "--others", "--exclude-standard", "--", "experiments/task_count_v2").decode().splitlines()
    assert not untracked, f"Unexpected study artifacts: {untracked}"

    # Reuse the earlier ledger and language scan, explicitly skipping its large
    # historical byte audit. The current v2 release check is run separately.
    spec = importlib.util.spec_from_file_location("earlier_english_revision", HERE.parent / "verify.py")
    earlier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(earlier)
    previous = earlier.verify(skip_archived_bytes=True)
    decoded_json = 0

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from strings(item)

    paths = set(git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0"))
    for path in sorted(paths):
        if Path(path).suffix.lower() != ".json":
            continue
        value = json.loads((ROOT / path).read_text(encoding="utf-8-sig"))
        assert not any(HAN.search(s) for s in strings(value)), f"Escaped non-English JSON text: {path}"
        decoded_json += 1
    return {"status": "passed", "base_revision": BASE, "translated_or_language_policy_files": 4,
            **generator, "report_numeric_table_cells_preserved": True,
            "report_evidence_links_preserved": True,
            "additional_report_links": dict(new_links - old_links), "original_release_manifest_preserved": True,
            "other_tracked_study_artifacts_unchanged": True,
            "earlier_translation_verification": previous,
            "decoded_json_files_scanned": decoded_json, "remaining_han_files": 0,
            "training_or_model_evaluation_run": False,
            "v2_checkpoint_byte_verification": "Run release.py --verify separately; do not infer from this check"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify()
    print(json.dumps(result, indent=2))
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
