"""Verify the integrated publication package without training or model forwards."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
BASE = "95e84e6af5de2c0a692da2f64bf95745455f5c3c"
MANIFEST = HERE / "MANIFEST.json"
ALLOWED_PRIOR_EDITS = {"README.md", "maintenance/english_translation/verify.py"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def write_manifest():
    paths = [p for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts
             and p.name not in {"MANIFEST.json", "VERIFICATION.json"}]
    paths += [ROOT / p for p in ALLOWED_PRIOR_EDITS]
    paths += list((ROOT / "maintenance/documentation_revisions").glob("*"))
    data = {"schema_version": 1, "base_revision": BASE,
            "scope": "Publication synthesis; no new training or model test evaluation",
            "allowed_prior_file_edits": sorted(ALLOWED_PRIOR_EDITS),
            "files": {p.relative_to(ROOT).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
                      for p in sorted(paths) if p.is_file()}}
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n", encoding="utf8")


def verify():
    manifest = load(MANIFEST)
    assert manifest["base_revision"] == BASE
    for rel, info in manifest["files"].items():
        path = ROOT / rel
        assert path.is_file() and sha(path) == info["sha256"], f"Publication file changed: {rel}"
    changed = set(filter(None, git("diff", "--name-only", "-z", BASE).decode().split("\0")))
    unexpected = {p for p in changed if p not in ALLOWED_PRIOR_EDITS
                  and not p.startswith(("publication/", "maintenance/documentation_revisions/"))}
    assert not unexpected, f"Historical artifact edits: {sorted(unexpected)}"

    bindings = {}
    figures = load(HERE / "figures/SOURCE_MAP.json")
    assert sha(HERE / "build_figures.py") == figures["builder_sha256"]
    for rel, info in figures["source_files"].items():
        bindings[rel] = info["sha256"]
    for rel, expected in figures["output_sha256"].items():
        assert sha(HERE / "figures" / rel) == expected, f"Figure output changed: {rel}"
    evidence = load(HERE / "evidence/dmlr_recompute_checks.json")
    assert evidence["status"] == "pass" and len(evidence["checks"]) == 8
    assert evidence["training_runs"] == evidence["neural_forwards"] == 0
    for rel, expected in evidence["source_artifact_sha256"].items():
        if rel in bindings:
            assert bindings[rel] == expected
        bindings[rel] = expected
    for rel, expected in bindings.items():
        assert sha(ROOT / rel) == expected, f"Scientific source changed: {rel}"

    links = 0
    broken = []
    documents = [ROOT / "README.md", *HERE.rglob("*.md")]
    for document in documents:
        text = document.read_text(encoding="utf-8-sig")
        for href in re.findall(r"\]\(([^\s)]+)(?:\s+[^)]*)?\)", text):
            href = unquote(href.strip("<>"))
            if href.startswith(("https://", "http://", "mailto:", "#")):
                continue
            target = (document.parent / href.split("#", 1)[0]).resolve()
            links += 1
            if not target.exists():
                broken.append(f"{document.relative_to(ROOT)} -> {href}")
    assert not broken, "Broken local links: " + repr(broken)

    # The supported data-to-model-sample path runs on CPU without changing data.
    import sys
    sys.path.insert(0, str(ROOT))
    from permworld_platform import label
    from permworld_platform.encoding import encode_example
    from permworld_platform.data import read_samples
    p = (3, 1, 4, 2)
    assert label("cycle_count", p) == 1
    assert label("increasing_subsequence_count", p, {"k": 2}) == 3
    row = encode_example("increasing_subsequence_count", p, {"k": 2}, max_seq_len=128)
    assert row["input_ids"][row["prompt_length"]:] == row["answer_ids"]
    sample = next(read_samples(ROOT / "experiments/platform_v01/dataset", 617,
                               "train", "increasing_subsequence_count", {"k": 3}))
    assert sample["input_ids"][sample["prompt_length"]:] == sample["answer_ids"]

    # Report availability of optional upstream clones without treating their
    # presence as an independent repetition of the saved upstream audit.
    upstream_count = 0
    missing_clones = set()
    with (HERE / "evidence/source_inventory.csv").open(encoding="utf8", newline="") as handle:
        inventory = list(csv.DictReader(handle))
    # Pinned re-extraction has its own saved audit; optional clones are not required
    # for the source-only publication verification or figure rebuilding.
    for directory in ("reference/neurips", "reference/2026TMLR"):
        if not (ROOT / directory / ".git").exists():
            missing_clones.add(directory)
        else:
            upstream_count += 1
    audit = load(HERE / "evidence/UPSTREAM_AUDIT.json")
    assert audit["status"] == "passed"
    result = {"status": "passed", "base_revision": BASE,
              "publication_files_hash_verified": len(manifest["files"]),
              "scientific_source_files_hash_verified": len(bindings),
              "local_links_checked": links, "broken_local_links": 0,
              "derived_evidence_stages_checked": len(evidence["checks"]),
              "plot_values": figures["plotted_repeat_values"],
              "upstream_inventory_entries": len(inventory),
              "optional_upstream_clones_present": upstream_count,
              "optional_upstream_clones_absent": sorted(missing_clones),
              "cpu_label_encoding_and_canonical_sample_read": "passed",
              "historical_experiment_paths_changed": 0,
              "training_runs": 0, "neural_forwards": 0,
              "checkpoint_files_opened": 0,
              "note": "Validates source bindings and saved arithmetic audits; not a new model replication"}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="Record after reviewing the publication package")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.record:
        write_manifest()
    result = verify()
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
