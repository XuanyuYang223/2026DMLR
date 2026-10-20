"""Export small, pinned upstream evidence without training or model forwards.

Optional regeneration requires the two ignored reference checkouts. Publication
figures consume the exported CSVs and do not require those checkouts or weights.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tomllib
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REPOS = {
    "neurips": (ROOT / "reference/neurips", "XuanyuYang223/neurips", "74f0de2017115f06f11b7285366b66e7e8431d30"),
    "tmlr": (ROOT / "reference/2026TMLR", "XuanyuYang223/2026TMLR", "d9a113d20d34db3c21cf69cf546cff4a0ab4904a"),
    "dmlr": (ROOT, "XuanyuYang223/2026DMLR", None),
}


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def binding(repo_key, relative):
    checkout, repository, revision = REPOS[repo_key]
    revision = revision or git(checkout, "rev-parse", "HEAD").decode().strip()
    path = checkout / relative
    content = path.read_bytes()
    blob = git(checkout, "show", f"{revision}:{relative}")
    return {"source_repository": repository, "source_revision": revision,
            "source_repository_path": relative, "source_local_path": path.relative_to(ROOT).as_posix(),
            "source_url": f"https://github.com/{repository}/blob/{revision}/{relative}",
            "source_worktree_sha256": sha(content), "source_git_blob_sha256": sha(blob),
            "source_worktree_bytes": len(content), "source_git_blob_bytes": len(blob)}


def main(remote=False):
    OUT.mkdir(parents=True, exist_ok=True)
    checkouts = {}
    for key in ("neurips", "tmlr"):
        checkout, repository, revision = REPOS[key]
        assert git(checkout, "rev-parse", "HEAD").decode().strip() == revision
        status = git(checkout, "status", "--porcelain").decode()
        assert not status, f"Reference checkout changed: {checkout}"
        checkouts[key] = {"repository": repository, "revision": revision,
                          "remote": git(checkout, "remote", "get-url", "origin").decode().strip(),
                          "clean_worktree": True}
        if remote:
            checkouts[key]["remote_head_observed"] = git(ROOT, "ls-remote", checkouts[key]["remote"], "HEAD").decode().strip()

    inventory = []
    def add(key, path, category, contribution, limitation):
        inventory.append({**binding(key, path), "category": category,
                          "supported_contribution": contribution, "limitation": limitation})

    original = [
        ("README.md", "overview", "Original permutation resource, task-count, transfer and geometry studies", "Repository study record; no authorship or external novelty inference"),
        ("src/neurips_permutations/math_ops.py", "exact_oracles", "20 V2, 20 V3 and 32 Property32 names with 48 distinct endpoints in their union", "Endpoint names are not independent mathematical capabilities"),
        ("src/neurips_permutations/passage.py", "encoding", "Typed Passage Math grammar, base-100 integers, four input representations", "Different versions use different vocabulary/context configurations"),
        ("src/neurips_permutations/generate.py", "generator", "Seeded data generation and task dispatch", "Generator existence does not establish downstream learnability"),
        ("src/neurips_permutations/verify.py", "verification", "Mathematical-answer and encoding verification tools", "Not rerun on the full upstream corpus in this integration"),
        ("src/neurips_permutations/models.py", "models", "Shared causal Transformer and causal MLP APIs", "DMLR adopts Transformer code; architecture is inherited"),
        ("src/neurips_permutations/training.py", "training", "Multitask causal-language-model training infrastructure", "Its protocol is distinct from DMLR fixed-sequence-token training"),
        ("PROPERTY_TASK_GEOMETRY.md", "geometry_protocol", "Eight relation families, specialists and fixed-four-task controls", "CKA is not transfer accuracy or proof of a shared algorithm"),
        ("configs/property_task_geometry.toml", "frozen_geometry_config", "Seeds 17/42/101, 20k updates, 4096 validation prefixes, eight relations", "Shared model/input structure prevents treating all rows as independent"),
        ("src/neurips_permutations/property_task_geometry_cka.py", "geometry_implementation", "Specialist, bundle and transformed-input CKA analysis", "No activations or model forwards recomputed here"),
        ("src/neurips_permutations/cka.py", "geometry_feature_extraction", "Last-prefix ONE_END hidden vector; ordinary linear CKA across pooled probe vectors", "No token mean pooling, within-length adjustment or answer control in this original panel"),
        ("results/property-task-geometry/cka/probe_manifests.json", "geometry_probe_provenance", "4096 task-free validation prefixes spanning n=2 through 30", "Saved metadata inspected; original raw shards not regenerated"),
        ("results/property-task-geometry/cka/symmetry_cka.csv", "geometry_raw", "Per-relation/seed/condition/layer CKA", "Validation-prefix geometry, not independent-test behavior"),
        ("results/property-task-geometry/cka/symmetry_summary.csv", "geometry_summary", "Relation/condition means and sample SD over three seeds", "SD is not a confidence interval"),
        ("results/property-task-geometry/cka/specialist_pair_summary.csv", "geometry_task_pairs", "Pair-level specialist comparison", "Task pairs share specialists"),
        ("results/property-task-geometry/cka/specialist_group_summary.csv", "geometry_groups", "Related/other/same-task CKA summaries", "Other means outside eight designated relations, not unrelated"),
        ("results/property-task-geometry/cka/bundle_cell_cka.csv", "composition_raw", "Fixed-four-task geometry at 0/1/2/4 direct correspondences", "Layout-by-seed cells share anchors"),
        ("results/property-task-geometry/cka/bundle_summary.csv", "composition_summary", "Nonmonotonic fixed-four-task geometry result", "Does not establish transfer effects"),
        ("results/property-task-geometry/cka/README.md", "geometry_interpretation", "Relation-level statistics and preserved negative composition result", "Upstream interpretation; not new DMLR evidence"),
        ("results/property32-zero-overlap/replicates/behavior_summary.csv", "task_count_behavior", "Original opposite-pool zero-shot task-count curve", "Different data, targets and training budget; do not pool with DMLR"),
        ("src/neurips_permutations/representation_transfer.py", "representation_transfer", "Four representations by eight tasks; 11 trained and 21 held-out cells", "Partial transfer is not uniformly successful"),
        ("paper/FIGURES.md", "publication_figures", "Original main and supplementary figure guide", "Figure material exists; no full manuscript source located"),
        ("paper/PERMUTATION_SECTION_FIGURES.md", "publication_figures", "Five standalone plots with captions and provenance", "Probe panels use a separately identified PermuFormer2 retraining source"),
        ("paper/TABLES.md", "publication_tables", "Copy-ready tables and LaTeX snippets", "Snippets are not a complete manuscript or bibliography"),
        ("paper/figures/permutation-section/manifest.json", "figure_provenance", "Exact source/output hashes and independent probe-source attribution", "Working-tree newline conversions can differ from Git blob hashes"),
        ("ARTIFACTS.md", "artifact_release", "Versioned datasets/checkpoints with checksums and restoration instructions", "Asset payloads not downloaded or reverified in this audit"),
        ("pyproject.toml", "package_metadata", "Runnable package entry points and dependencies", "No package license declaration found in this snapshot"),
    ]
    for path, category, contribution, limitation in original:
        add("neurips", path, category, contribution, limitation)
    for path, category, contribution, limitation in [
        ("README.md", "alignment_overview", "PermWorld/GraphWorld/MatrixGroupWorld representation-pairing program", "Domains and dependency structures are not pooled independent replications"),
        ("diagnostics/permworld_eight_relations_20261010/protocol.json", "alignment_protocol", "64 PermWorld specialists across two encodings and architectures", "Exploratory; prior results seen; one paired data world"),
        ("diagnostics/permworld_eight_relations_20261010/final_review/permworld/main32_final_cka_readout.csv", "alignment_primary_cells", "32 answer-controlled pairing cells and initialization decomposition", "26 positive Bt and 18 positive Delta are descriptive coupled cells"),
        ("experiments/longrun_attention.py", "attention_runtime", "Equivalent SDPA implementation reused in DMLR", "Runtime implementation, not new DMLR model architecture"),
        ("docs/upstream_provenance.json", "upstream_provenance", "Explicit neurips source snapshot and license inspection", "Does not grant a new license or establish individual authorship"),
        ("artifacts_manifest.json", "artifact_manifest", "Logical file-to-content/release-asset mapping", "No large checkpoint assets downloaded here"),
        ("docs/representation_reproduction.md", "reproduction", "Saved-table replay and full artifact restoration instructions", "Table arithmetic does not reproduce every model forward"),
        ("docs/representation_publication_audit.json", "publication_audit", "Recorded public readback and limited clean-clone checks", "Matrix numerical verification caveat remains unresolved"),
    ]:
        add("tmlr", path, category, contribution, limitation)
    for path, category, contribution, limitation in [
        ("experiments/permworld_pilot/README.md", "dmlr_provenance", "Explicit pinned source reuse and separate transfer runner", "Alignment findings are not reinterpreted as transfer results"),
        ("permworld_platform/registry.py", "dmlr_registry", "Unified legacy/new templates, parameters and aliases", "Engineering/template counts are not independence claims"),
        ("experiments/platform_v01/registry/TASK_COUNTS.json", "dmlr_resource_counts", "54 endpoints, 48 canonical templates, 16 focus templates/24 instances", "Count definitions and constants must remain explicit"),
        ("experiments/platform_v01/mathematics/RELATIONS.json", "dmlr_relations", "74 registered relations; single/joint/transformed-input requirements", "An absent relation is unclassified, not unrelated"),
        ("experiments/platform_v01/PLATFORM_REPORT.md", "dmlr_platform", "Versioned 203-token encoding, exact generators and limited calibration", "Six calibrated representatives do not calibrate the whole library"),
        ("experiments/task_count_v1/REPORT.md", "dmlr_fixed_token_count", "Paired 4M-token source-count transfer experiment", "Composition changes with count; no scaling law"),
        ("experiments/task_count_v2/REPORT.md", "dmlr_budget_interaction", "Frozen count-by-budget comparison with independent new repeats", "Finite two-budget/four-condition study; no new training in integration"),
    ]:
        add("dmlr", path, category, contribution, limitation)
    write_csv("source_inventory.csv", inventory)

    source_path = "results/property-task-geometry/cka/symmetry_cka.csv"
    provenance = binding("neurips", source_path)
    source_rows = read_csv(REPOS["neurips"][0] / source_path)
    final = [row for row in source_rows if row["layer"] == "final_norm"]
    assert len(final) == 72
    raw = [{**row, **provenance, "probe_split": "validation", "probe_inputs": 4096} for row in final]
    write_csv("upstream_alignment_final_norm.csv", raw)
    pairs = {}
    for row in final:
        key = row["pair_id"], int(row["model_seed"])
        assert row["condition"] not in pairs.setdefault(key, {})
        pairs[key][row["condition"]] = float(row["linear_cka"])
    paired = []
    for (pair_id, seed), values in sorted(pairs.items()):
        assert set(values) == {"identity", "correct", "wrong"}
        paired.append({"pair_id": pair_id, "model_seed": seed,
                       **{f"{key}_cka": values[key] for key in ("identity", "correct", "wrong")},
                       "correct_minus_identity": values["correct"] - values["identity"],
                       "correct_minus_wrong": values["correct"] - values["wrong"], **provenance})
    assert len(paired) == 24
    write_csv("upstream_alignment_by_seed.csv", paired)
    summary_lookup = {(r["pair_id"], r["condition"]): r for r in read_csv(REPOS["neurips"][0] / "results/property-task-geometry/cka/symmetry_summary.csv")}
    relation_rows = []
    for pair_id in sorted({r["pair_id"] for r in paired}):
        rows = [r for r in paired if r["pair_id"] == pair_id]
        assert len(rows) == 3 and {r["model_seed"] for r in rows} == {17, 42, 101}
        result = {"pair_id": pair_id, "seed_count": 3, "unit": "mathematical_relation_after_seed_averaging"}
        for key in ("identity_cka", "correct_cka", "wrong_cka", "correct_minus_identity", "correct_minus_wrong"):
            result[key + "_mean"] = statistics.mean(r[key] for r in rows)
            result[key + "_sample_sd"] = statistics.stdev(r[key] for r in rows)
        for condition in ("identity", "correct", "wrong"):
            saved = summary_lookup[pair_id, condition]
            assert abs(result[condition + "_cka_mean"] - float(saved["final_layer_cka_mean"])) < 1e-14
            assert abs(result[condition + "_cka_sample_sd"] - float(saved["final_layer_cka_sample_sd"])) < 1e-14
        relation_rows.append({**result, **provenance})
    write_csv("upstream_alignment_by_relation.csv", relation_rows)

    tree = ast.parse((REPOS["neurips"][0] / "src/neurips_permutations/math_ops.py").read_text())
    names = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
             if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id in ("V2_TASK_NAMES", "V3_TASK_NAMES", "PROPERTY32_TASK_NAMES")}
    config = tomllib.loads((REPOS["neurips"][0] / "configs/property_task_geometry.toml").read_text())
    probe_manifests = json.loads((REPOS["neurips"][0] / "results/property-task-geometry/cka/probe_manifests.json").read_text())
    probe = probe_manifests["identity"]
    assert probe["example_count"] == sum(probe["n_histogram"].values()) == 4096
    assert set(map(int, probe["n_histogram"])) == set(range(2, 31))
    assert all(p["n_histogram"] == probe["n_histogram"] for p in probe_manifests.values())
    platform_counts = json.loads((ROOT / "experiments/platform_v01/registry/TASK_COUNTS.json").read_text())
    relations = json.loads((ROOT / "experiments/platform_v01/mathematics/RELATIONS.json").read_text())["relations"]
    tmlr_rows = read_csv(REPOS["tmlr"][0] / "diagnostics/permworld_eight_relations_20261010/final_review/permworld/main32_final_cka_readout.csv")
    resource = {"original_task_grids": names, "original_grid_sizes": {k: len(v) for k, v in names.items()},
                "original_distinct_endpoint_union": len(set().union(*map(set, names.values()))),
                "original_input_representations": ["one_line", "cycle", "lehmer", "inversion_vector"],
                "dmlr_platform_counts": platform_counts, "dmlr_registered_relations": len(relations),
                "dmlr_relation_origins": dict(Counter(r["origin"] for r in relations)),
                "dmlr_relation_evidence_classes": dict(Counter(r["evidence_class"] for r in relations)),
                "original_geometry_model_seeds": config["model_seeds"],
                "original_geometry_probe_inputs": config["analysis"]["probe_examples"],
                "original_geometry_probe": {"input_encoding": "one_line", "split": "validation",
                    "task_query_present": False, "feature_layer": "final_norm",
                    "feature_position": "last task-free prefix token <ONE_END>",
                    "token_mean_pooling": False, "feature_width": config["model"]["d_model"],
                    "similarity": "ordinary centered linear CKA across all 4096 pooled example vectors",
                    "length_adjustment": "none; not computed separately within length",
                    "answer_stratification": False, "n_histogram": probe["n_histogram"],
                    "probe_sha256": probe["probe_sha256"], "probe_seed": probe["seed"],
                    "selection": probe["selection"], "probe_manifest_binding": binding("neurips", "results/property-task-geometry/cka/probe_manifests.json"),
                    "feature_code_binding": binding("neurips", "src/neurips_permutations/cka.py")},
                "original_geometry_updates": config["training"]["max_steps"],
                "tmlr_primary_cells": len(tmlr_rows), "tmlr_positive_Bt": sum(float(r["Bt"]) > 0 for r in tmlr_rows),
                "tmlr_positive_Delta": sum(float(r["Delta"]) > 0 for r in tmlr_rows),
                "tmlr_unit_caveat": "One paired data world; shared initial weights and models; 32 cells are not independent repeats",
                "source_bindings": {p: binding(k, p) for k, p in [("neurips", "src/neurips_permutations/math_ops.py"), ("neurips", "src/neurips_permutations/passage.py"), ("neurips", "configs/property_task_geometry.toml"), ("dmlr", "experiments/platform_v01/registry/TASK_COUNTS.json"), ("dmlr", "experiments/platform_v01/mathematics/RELATIONS.json")]}}
    (OUT / "upstream_resource_inventory.json").write_text(json.dumps(resource, indent=2) + "\n", encoding="utf8")
    missing = {}
    for key in ("neurips", "tmlr"):
        paths = git(REPOS[key][0], "ls-files").decode().splitlines()
        missing[key] = {"tex_sources": [p for p in paths if p.endswith(".tex")],
                        "bib_sources": [p for p in paths if p.endswith(".bib")],
                        "license_notice_files": [p for p in paths if Path(p).name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))]}
    releases = []
    if remote:
        for repository, tag in [("XuanyuYang223/neurips", "permutation-data-2026-09-04"),
                                ("XuanyuYang223/neurips", "permutation-checkpoints-2026-09-04"),
                                ("XuanyuYang223/2026TMLR", "representation-selectivity-2026-10-11-caveat")]:
            url = f"https://api.github.com/repos/{repository}/releases/tags/{tag}"
            request = urllib.request.Request(url, headers={"User-Agent": "PermWorld-evidence-audit"})
            with urllib.request.urlopen(request, timeout=30) as response:
                value = json.load(response)
            releases.append({"api_url": url, "url": value["html_url"], "tag": value["tag_name"],
                             "draft": value["draft"], "prerelease": value["prerelease"],
                             "asset_count": len(value["assets"]), "asset_bytes": sum(a["size"] for a in value["assets"]),
                             "asset_payloads_downloaded": False})
    vendor_reuse = []
    for name in ("__init__.py", "math_ops.py", "models.py", "passage.py"):
        source = REPOS["neurips"][0] / "src/neurips_permutations" / name
        vendor = ROOT / "experiments/permworld_pilot/vendor/neurips_permutations" / name
        assert source.read_text() == vendor.read_text()
        vendor_reuse.append({"source": source.relative_to(ROOT).as_posix(),
                             "copy": vendor.relative_to(ROOT).as_posix(),
                             "exact_bytes": source.read_bytes() == vendor.read_bytes(),
                             "newline_normalized_text_equal": True,
                             "source_sha256": sha(source.read_bytes()), "copy_sha256": sha(vendor.read_bytes())})
    attention_source = REPOS["tmlr"][0] / "experiments/longrun_attention.py"
    attention_copy = ROOT / "experiments/permworld_pilot/vendor/sdpa.py"
    assert attention_source.read_text() == attention_copy.read_text()
    vendor_reuse.append({"source": attention_source.relative_to(ROOT).as_posix(),
                         "copy": attention_copy.relative_to(ROOT).as_posix(),
                         "exact_bytes": attention_source.read_bytes() == attention_copy.read_bytes(),
                         "newline_normalized_text_equal": True,
                         "source_sha256": sha(attention_source.read_bytes()), "copy_sha256": sha(attention_copy.read_bytes())})
    result = {"status": "passed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "checkouts": checkouts, "source_inventory_rows": len(inventory), "missing_publication_inventory": missing,
              "alignment_raw_export_rows": len(raw), "alignment_seed_units": len(paired), "alignment_relation_units": len(relation_rows),
              "positive_seed_contrasts": {key: sum(r[key] > 0 for r in paired) for key in ("correct_minus_identity", "correct_minus_wrong")},
              "positive_relation_contrasts": {key: sum(r[key + "_mean"] > 0 for r in relation_rows) for key in ("correct_minus_identity", "correct_minus_wrong")},
              "relation_level_two_sided_sign_p_when_all_eight_positive": 2 / 2**8,
              "original_summary_recomputed_from_saved_raw_rows": True,
              "training_or_model_forwards_run": False, "upstream_modified": False,
              "verified_vendor_reuse": vendor_reuse,
              "remote_release_metadata": releases, "exported_data_sha256": {p.name: sha(p.read_bytes()) for p in OUT.glob("upstream_*.csv")}}
    (OUT / "UPSTREAM_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": result["status"], "inventory_rows": len(inventory), "raw_alignment_rows": len(raw), "relation_units": len(relation_rows)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", action="store_true", help="Also read GitHub HEAD and release metadata; never download weights")
    args = parser.parse_args()
    main(args.remote)
