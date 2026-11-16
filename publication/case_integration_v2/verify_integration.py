"""Verify the new manuscript layer from saved evidence, without model execution."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PAPER = ROOT / "publication/PAPER_DRAFT_v2.md"
LEDGER = ROOT / "publication/evidence/dmlr_claims_v2.csv"
FIGURE = ROOT / "publication/figures/case_integration_v2/final"
BASE = "5f5342fe0d3653c917f9e58ff0792d6d87f87616"
CALIBRATION = ROOT / "experiments/platform_calibration_v02/reuse_authorized_v1/artifacts/final/RESULT_REPORT.md"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return Path(path).read_text(encoding="utf-8-sig")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def local(name):
    path = (ROOT / name).resolve()
    assert path.is_relative_to(ROOT), name
    return path


def numerical_text(line):
    # Link destinations and inline anchor IDs are navigation, not scientific values.
    return re.sub(r"\]\([^)]*\)", "]", line)


def run():
    paper = read(PAPER)
    lines = paper.splitlines()
    abstract = re.search(r"(?ms)^## Abstract\s*\n(.*?)(?=^## )", paper).group(1).strip()
    abstract_words = len(abstract.split())
    assert abstract_words <= 200, abstract_words
    assert not re.search(r"[\u4e00-\u9fff]", paper), "Repository prose must be English"
    assert not re.search(r"\b(frozen|authorized|release check|stopping point)\b", paper, re.I)
    with LEDGER.open(encoding="utf-8-sig", newline="") as handle:
        ledger = list(csv.DictReader(handle))
    assert ledger
    required = {"claim_id", "paper_path", "paper_locator", "paper_excerpt", "source_path",
                "source_locator", "source_excerpt", "source_sha256", "evidence_scope"}
    assert required <= set(ledger[0])
    source_text, source_hashes, covered = {}, {}, set()
    for row in ledger:
        assert row["paper_path"] == PAPER.relative_to(ROOT).as_posix()
        match = re.search(r"line\s+(\d+)", row["paper_locator"], re.I)
        assert match, row["paper_locator"]
        start = int(match.group(1)) - 1
        excerpt_lines = row["paper_excerpt"].splitlines()
        assert excerpt_lines and lines[start:start+len(excerpt_lines)] == excerpt_lines, row["claim_id"]
        covered.update(range(start, start+len(excerpt_lines)))
        source = local(row["source_path"])
        if source not in source_text:
            source_text[source] = read(source)
            source_hashes[row["source_path"]] = sha(source)
        assert source_hashes[row["source_path"]] == row["source_sha256"], row["claim_id"]
        assert row["source_excerpt"] and row["source_excerpt"] in source_text[source], row["claim_id"]
        source_start = re.search(r"line\s+(\d+)", row["source_locator"], re.I)
        assert source_start, row["source_locator"]
        offset = int(source_start.group(1))-1
        evidence_lines = row["source_excerpt"].splitlines()
        assert source_text[source].splitlines()[offset:offset+len(evidence_lines)] == evidence_lines, row["claim_id"]
    numeric_lines = {i for i, line in enumerate(lines) if re.search(r"\d", numerical_text(line))}
    missing = sorted(numeric_lines-covered)
    assert not missing, [(i+1, lines[i]) for i in missing]

    calibration = read(CALIBRATION)
    start = calibration.index("| Instance | Family | Length |")
    end = calibration.index("\n\n", start)
    exact_table = calibration[start:end]
    assert exact_table in paper, "The complete displayed calibration table must be copied verbatim"
    table_rows = [line for line in exact_table.splitlines()[2:] if line.startswith("|")]
    assert len(table_rows) == 200
    assert sum("reader-rejected" in line for line in table_rows) == 4

    # Only the new figure may be a repository-local link inside the manuscript.
    local_links = []
    for target in re.findall(r"\]\(([^)]+)\)", paper):
        if not re.match(r"https?://|mailto:|#", target):
            local_links.append(target)
            assert target == "figures/case_integration_v2/final/worked_case.png", target
            assert (PAPER.parent / target).is_file()
    assert len(local_links) == 1

    figure = json.loads(read(FIGURE / "MANIFEST.json"))
    assert figure["statistics_recomputed"] is False and figure["model_forward_calls"] == 0
    assert figure["checkpoint_files_opened"] == 0 and figure["data_pool_files_opened"] == 0
    assert (figure["point_rows"], figure["summary_rows"], figure["mapped_ordinates"]) == (80, 10, 94)
    for category in ("source_sha256", "outputs"):
        for name, expected in figure[category].items():
            assert sha(local(name)) == expected, name
    assert sha(HERE / "build_worked_case_final.py") == figure["builder_sha256"]
    preview = json.loads(read(FIGURE.parent / "MANIFEST.json"))
    assert sha(HERE / "build_worked_case.py") == preview["builder_sha256"]
    for name, expected in preview["outputs"].items():
        assert sha(local(name)) == expected, name
    for name in ("PLOT_DATA.csv", "PLOT_SUMMARIES.csv", "SOURCE_MAP.csv"):
        assert (FIGURE / name).read_bytes() == (FIGURE.parent / name).read_bytes()

    changed_historical = git("diff", "--name-only", "--diff-filter=MD", BASE, "--").splitlines()
    assert set(changed_historical) <= {"STAGE_STATUS.md"}, changed_historical
    assert not git("ls-files", "--", "LOCAL_STATE_SUMMARY.md")
    checkpoint_additions = git("diff", "--name-only", "--diff-filter=A", BASE, "--", "*.pt")
    assert not checkpoint_additions
    stage2 = json.loads(read(ROOT / "experiments/platform_calibration_v02/reuse_authorized_v1/artifacts/final/SOURCE_ONLY_AUDIT.json"))
    stage3 = json.loads(read(ROOT / "experiments/fixed_points_deficiencies_arch_v1/artifacts/analysis/INDEPENDENT_RECOMPUTE.json"))
    assert stage2["status"] == "passed" and stage2["formal_runs"] == 98
    assert stage3["status"] == "pass" and stage3["raw_prediction_rows"] == 491520

    publication_files = [PAPER, LEDGER, ROOT / "publication/AUTHOR_DECISIONS.md",
                         *sorted(HERE.glob("*.py")), HERE / "REVISION.md",
                         *sorted(FIGURE.glob("*"))]
    return dict(status="passed", checked_at_utc=datetime.now(timezone.utc).isoformat(),
        base_commit=BASE, current_commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
        abstract_words=abstract_words, ledger_rows=len(ledger), numeric_manuscript_lines=len(numeric_lines),
        uncovered_numeric_lines=[], complete_calibration_table_rows=len(table_rows),
        context_rejection_rows=4, exact_source_excerpts=True, source_locator_checks=True,
        figure_ordinates_copied=94, figure_sources_unchanged=True, preview_outputs_unchanged=True,
        preview_numeric_tables_unchanged=True,
        historical_changes=changed_historical, checkpoint_additions=[],
        local_state_summary_untracked=True, source_sha256=source_hashes,
        publication_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in publication_files if p.is_file()},
        scope="Text, source excerpts, hashes, copied table/figure data and recorded audit status; no scientific statistics recomputed",
        model_forward_calls=0, training_runs=0, checkpoint_files_opened=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh JSON path; existing audits are never overwritten")
    args = parser.parse_args()
    destination = args.output.resolve()
    assert destination.is_relative_to(HERE)
    assert not destination.exists(), "Preserve the previous audit"
    try:
        record = run()
    except Exception as error:
        record = dict(status="failed", at_utc=datetime.now(timezone.utc).isoformat(),
                      error_type=type(error).__name__, error=str(error), verifier_sha256=sha(__file__),
                      training_runs=0, model_forward_calls=0)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
        raise
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: record[k] for k in ("status", "abstract_words", "ledger_rows", "numeric_manuscript_lines", "complete_calibration_table_rows")}, indent=2))
