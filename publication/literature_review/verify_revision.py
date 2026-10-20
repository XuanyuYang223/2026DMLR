"""Check this editorial revision without training, predictions or certificate reruns."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = "5cb6b9f3898c4d8c2857b5f5530d9785c8124a38"
EDITED = {
    "publication/PAPER_DRAFT.md", "publication/CONTRIBUTIONS.md",
    "publication/README.md", "publication/references.bib", "publication/verify.py",
}
OUTPUTS = {"REVISION.json", "VERIFICATION.json", "PUBLICATION_VERIFICATION.json",
           "validation.log", "publication_verify.log", "english_verify.log"}
EXPECTED_REFERENCES = {
    "saxton2019analysing": "2019", "lample2020symbolic": "2020",
    "velickovic2022clrs": "2022", "markeeva2024clrstext": "2024",
    "stojanovski2025reasoninggym": "2025", "chen2023skillit": "2023",
    "xie2023doremi": "2023", "fan2024doge": "2024", "ye2025datamixing": "2025",
}


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def new_document_files():
    return sorted([ROOT / "publication/RESOURCE_COMPARISON.md"] + [
        path for path in HERE.rglob("*") if path.is_file()
        and "__pycache__" not in path.parts and path.name not in OUTPUTS
    ])


def record():
    ledger = {
        "schema_version": 1,
        "revision": "literature-positioning-v1",
        "baseline_commit": BASE,
        "scope": "Literature and editorial revision; scientific artifacts and prior acceptance unchanged",
        "changed_files": {
            rel: {"before_sha256": digest(git("show", f"{BASE}:{rel}")),
                  "after_sha256": digest((ROOT / rel).read_bytes())}
            for rel in sorted(EDITED)
        },
        "added_files": {
            path.relative_to(ROOT).as_posix(): {"sha256": digest(path.read_bytes())}
            for path in new_document_files()
        },
        "excluded_outputs": sorted(OUTPUTS),
        "output_scope": "Revision ledger and execution reports are recorded separately to avoid self-hash cycles.",
        "training_runs": 0,
        "model_forwards": 0,
    }
    (HERE / "REVISION.json").write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf8")


def section(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]


def parse_bib(text):
    """Read brace-delimited entries, checking nested accent braces and duplicate keys."""
    entries = {}
    for match in re.finditer(r"(?m)^@(\w+)\{([^,\s]+),", text):
        depth = 1
        end = match.end()
        while end < len(text) and depth:
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
            end += 1
        assert depth == 0, f"Unbalanced bibliography entry: {match.group(2)}"
        key = match.group(2)
        assert key not in entries, f"Duplicate bibliography key: {key}"
        entries[key] = text[match.end():end - 1]
    return entries


def verify():
    ledger = json.loads((HERE / "REVISION.json").read_text(encoding="utf8"))
    assert ledger["baseline_commit"] == BASE
    assert set(ledger["changed_files"]) == EDITED
    baseline_paths = set(git("ls-tree", "-r", "--name-only", BASE).decode().splitlines())
    changed = set(filter(None, git("diff", "--name-only", "-z", BASE).decode().split("\0")))
    existing_changes = changed & baseline_paths
    assert existing_changes == EDITED, f"Unexpected changed baseline paths: {existing_changes ^ EDITED}"
    new_paths = (changed - baseline_paths) | set(filter(
        None, git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")))
    for rel in new_paths:
        assert rel == "publication/RESOURCE_COMPARISON.md" or rel.startswith(
            "publication/literature_review/"), f"Unexpected new file: {rel}"
    for rel, info in ledger["changed_files"].items():
        assert digest(git("show", f"{BASE}:{rel}")) == info["before_sha256"]
        assert digest((ROOT / rel).read_bytes()) == info["after_sha256"]
    expected_added = {p.relative_to(ROOT).as_posix() for p in new_document_files()}
    assert set(ledger["added_files"]) == expected_added
    for rel, info in ledger["added_files"].items():
        assert rel not in baseline_paths
        assert digest((ROOT / rel).read_bytes()) == info["sha256"]

    protected_records = ["publication/MANIFEST.json", "publication/VERIFICATION.json",
                         "release_checks/ACCEPTED_SOURCE.json", "release_checks/START_STATE.json"]
    for rel in protected_records:
        assert (ROOT / rel).read_bytes() == git("show", f"{BASE}:{rel}")
    old = git("show", f"{BASE}:publication/PAPER_DRAFT.md").decode("utf-8-sig")
    current = (ROOT / "publication/PAPER_DRAFT.md").read_text(encoding="utf-8-sig")
    unchanged_sections = [
        ("## 4. Experimental record and protocol boundaries", "## 5. Findings"),
        ("## 5. Findings", "## 6. Limitations and implications"),
        ("## Appendix A. Evidence and protocol map", "## References"),
    ]
    for start, end in unchanged_sections:
        assert section(old, start, end).replace("\r\n", "\n") == section(
            current, start, end).replace("\r\n", "\n"), f"Scientific text changed: {start}"

    bib = parse_bib((ROOT / "publication/references.bib").read_text(encoding="utf8"))
    assert set(bib) == set(EXPECTED_REFERENCES)
    for key, year in EXPECTED_REFERENCES.items():
        assert re.search(r"year\s*=\s*\{" + year + r"\}", bib[key])
        for field in ("author", "title", "url"):
            assert re.search(field + r"\s*=\s*\{", bib[key]), (key, field)
        url = re.search(r"url\s*=\s*\{([^}]+)\}", bib[key]).group(1)
        assert url in current, f"Bibliography entry missing from manuscript references: {key}"

    documents = [ROOT / rel for rel in EDITED if rel.endswith((".md", ".bib"))]
    documents += new_document_files()
    table_blocks = 0
    for path in documents:
        if path.suffix not in {".md", ".bib", ".py", ".json", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8-sig")
        assert not re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text), f"Non-English prose: {path}"
        if path.suffix != ".md":
            continue
        width = None
        for line in text.splitlines():
            if line.startswith("|"):
                cells = len(re.split(r"(?<!\\)\|", line))
                if width is None:
                    width = cells
                    table_blocks += 1
                assert cells == width, f"Inconsistent table columns in {path}: {line}"
            else:
                width = None

    return {
        "status": "passed", "baseline_commit": BASE,
        "changed_baseline_paths": sorted(existing_changes),
        "new_document_hashes_checked": len(expected_added),
        "external_bibliography_entries": len(bib),
        "markdown_table_blocks_checked": table_blocks,
        "unchanged_manuscript_sections": [item[0] for item in unchanged_sections],
        "historical_release_records": "unchanged",
        "all_other_baseline_tracked_paths": "unchanged",
        "scientific_source_and_figure_hashes": "checked separately by publication/verify.py",
        "external_capability_review": "primary-source inspection; no external software reproduction",
        "training_runs": 0, "model_forwards": 0, "checkpoint_files_opened": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="Record only the new editorial ledger")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report and args.report.resolve() != (HERE / "VERIFICATION.json").resolve():
        parser.error("Write this revision's report only to publication/literature_review/VERIFICATION.json")
    if args.record:
        record()
    result = verify()
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))
