"""Prespecified v1 compositions for the task-count by training-budget study.

The hypotheses come from v1 outcomes. This module never reads v2 outcomes and
reuses, rather than repeats, the unchanged v1 mathematical checks by byte hash.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.task_count_v1 import design as v1

HELDOUT_TARGETS = copy.deepcopy(v1.TARGETS)
TARGETS = [v1.spec("lis_length"), v1.spec("fixed_points")]
EVALUATION_TARGETS = TARGETS
K_VALUES = [2, 16]
BUDGETS = [4_000_000, 32_000_000]
LENGTHS = list(v1.LENGTHS)
CHAINS = copy.deepcopy(v1.CHAINS)
SOURCE_POOL = copy.deepcopy(v1.SOURCE_POOL)
task_key = v1.task_key
broad_family = v1.broad_family
spec = v1.spec


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def csv_read(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configuration():
    return copy.deepcopy({
        "schema_version": "task-count-budget-design-v2",
        "targets": TARGETS, "primary_target": "lis_length", "auxiliary_target": "fixed_points",
        "heldout_targets": HELDOUT_TARGETS, "k_values": K_VALUES,
        "budgets_nonpadding_sequence_tokens": BUDGETS, "lengths": LENGTHS,
        "chains": CHAINS, "source_pool": SOURCE_POOL,
        "k_definition": v1.configuration()["k_definition"],
        "selection_rule": "Keep the exact ordered v1 prefixes at k=2 and k=16. The two targets and the budget contrast were requested after v1 exploration; freeze before any v2 transfer outcomes. No source replacement, difficulty reselection or transfer-driven exclusion.",
        "family_counts": {str(k): dict(sorted(Counter(broad_family(t) for t in CHAINS["chain_alpha"][:k]).items())) for k in K_VALUES},
        "template_count_note": v1.configuration()["template_count_note"],
        "exclusions": copy.deepcopy(v1.EXCLUSIONS),
        "relations_policy": v1.configuration()["relations_policy"],
        "hypothesis_origin": "The v1 exploratory LIS decline from k2 to k16, and chain-dependent fixed-points transfer, motivate this new independent budget interaction study. V1 and v2 results are reported separately.",
        "primary_statistic": "Within each target, chain and paired repeat: [accuracy(k16,32M)-accuracy(k2,32M)]-[accuracy(k16,4M)-accuracy(k2,4M)], using development-selected checkpoints.",
        "auxiliary_statistics": ["Each budget's paired k16-minus-k2", "Each condition minus the shared random-initialization baseline", "Fixed-endpoint checkpoints", "Per-length results", "Source learning and exposure"],
        "interpretation": "Only budget changes within each fixed source set. Comparing k still changes composition and relations. Both budgets start at the same paired initialization; normalized-progress LR schedules produce different absolute-token LR trajectories. The 32M budget reuses the same finite input pool, not eight times as many independent objects. It is approximately eight times total tokens, not exact per-task exposure matching.",
        "independent_paired_repeats": 3, "new_v2_transfer_results_read": False,
        "new_formula_or_witness_checks": 0, "additional_model_selection": False,
    })


README = """# Frozen task-count by budget design

This is a follow-up hypothesis motivated by v1 exploration, frozen before v2
transfer outcomes. The primary target is LIS length; fixed points is auxiliary.
All four v1 held-out task IDs remain excluded from every source set and are reset
at adaptation. The exact v1 chains are retained only at k=2 and k=16. Each source
set is crossed with 4M and 32M nonpadding sequence tokens. There are three new
paired repeat units, 24 pretrained endpoints and 54 target fits including six
shared scratch baselines.

`configuration.json` records the hypotheses and complete ordered chains.
`task_sets.csv` records the memberships. Both k2 sets contain one cycle and one
subsequence instance; k16 sets contain four cycle, three subsequence, four block
and five legacy instances. The pool has 24 instances and 22 templates. Each
individual source set has k distinct templates. There is one encoding.

`v1_evidence_reuse.json` verifies the v1 frozen design and all original evidence
by SHA256. Existing inequivalence witnesses and output-encoding bounds are reused,
not rerun. `target_relationships.csv` retains all four original held-out targets
at the selected k values. `relation_inventory.json` retains the proof, finite
verification, empirical association and unchecked distinctions from the platform.
No missing relation is relabeled as mathematical unrelatedness. Legal transformed
input and joint answer relations are not automatically data leakage.

Alpha k16 can derive fixed points from n minus exceedances minus deficiencies or
even plus odd minus nontrivial cycle counts. Beta retains the transformed-input
anti-fixed-point relation. LDS relates to LIS under complement or reversal, so
its presence is retained and annotated. Having source labels in a set does not
show that the model learned or used a formula.

The primary contrast is the within-repeat difference between the 32M and 4M
k16-minus-k2 effects. Baseline gains, endpoints and source learning are auxiliary.
Chains, shared baselines and target reuse are not extra independent repeats.
Three-repeat sample SD is descriptive; differences are computed before their
uncertainty is summarized. This study cannot establish a scaling law or a pure
composition-free count effect.

Recreate into a fresh directory:

```powershell
.venv/Scripts/python.exe experiments/task_count_v2/design.py --out experiments/task_count_v2_reproduction/design
```

An existing frozen directory is only verified, never silently updated.
"""


def build_design(out=None):
    out = Path(out) if out is not None else Path(__file__).with_suffix("")
    freeze = out / "FROZEN_DESIGN.json"
    cfg = configuration()
    if freeze.exists():
        previous = json.loads(freeze.read_text(encoding="utf-8"))
        if previous["configuration"] != cfg or previous["design_source_sha256"] != sha(__file__):
            raise RuntimeError("Frozen design changed; use a fresh output directory")
        for name, digest in previous["artifact_sha256"].items():
            if sha(out / name) != digest:
                raise RuntimeError(f"Frozen artifact changed: {name}")
        return previous
    base = ROOT / "experiments/task_count_v1/design"
    old = json.loads((base / "FROZEN_DESIGN.json").read_text(encoding="utf-8"))
    assert old["status"] == "frozen"
    assert old["design_source_sha256"] == sha(v1.__file__)
    assert old["configuration"]["chains"] == CHAINS
    verified = {"experiments/task_count_v1/design.py": sha(v1.__file__), "experiments/task_count_v1/design/FROZEN_DESIGN.json": sha(base / "FROZEN_DESIGN.json")}
    for name, digest in old["artifact_sha256"].items():
        assert sha(base / name) == digest, name
        verified[f"experiments/task_count_v1/design/{name}"] = digest
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Refusing a nonempty incomplete design directory")
    out.mkdir(parents=True, exist_ok=True)
    for k in K_VALUES:
        members = [{task_key(t) for t in c[:k]} for c in CHAINS.values()]
        assert members[0] != members[1] and all(len(s) == k for s in members)
        assert all(not ({task_key(t) for t in HELDOUT_TARGETS} & s) for s in members)
    for name in ("source_pool.csv", "encoding_bounds.csv", "inequivalence_audit.json", "relation_inventory.json", "relation_reuse.json"):
        (out / name).write_bytes((base / name).read_bytes())
    for name in ("task_sets.csv", "target_relationships.csv"):
        csv_write(out / name, [row for row in csv_read(base / name) if int(row["k"]) in K_VALUES])
    counts = json.loads((base / "counts.json").read_text(encoding="utf-8"))
    counts["prefixes"] = [row for row in counts["prefixes"] if row["k"] in K_VALUES]
    dump(out / "counts.json", counts)
    dump(out / "configuration.json", cfg)
    dump(out / "v1_evidence_reuse.json", {"status": "reused_hash_verified", "sha256": verified, "historical_formula_rechecks_run": 0, "new_witness_evaluations": 0, "v1_transfer_evidence_role": "Hypothesis motivation only; no v2 outcomes read", "all_existing_task_lists_unchanged": True})
    (out / "README.md").write_text(README, encoding="utf-8")
    result = {"status": "frozen", "configuration": cfg, "design_source_sha256": sha(__file__), "artifact_sha256": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}, "new_v2_transfer_results_read": False, "historical_formula_rechecks_run": 0, "nonconstancy_witness_count": old["nonconstancy_witness_count"], "pairwise_inequivalence_witness_count": old["pairwise_inequivalence_witness_count"], "witnesses_reused_not_recomputed": True}
    dump(freeze, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_suffix(""))
    args = parser.parse_args()
    result = build_design(args.out)
    print(json.dumps({"status": result["status"], "output": str(args.out), "source_instances": len(SOURCE_POOL), "evaluated_targets": [task_key(t) for t in TARGETS], "formula_rechecks": 0}))
