"""Outcome-independent task-instance design for the first task-count experiment.

This module imports no training code and reads no transfer or calibration results.
The existing mathematical certificates are reused by hash. New label evaluations
produce counterexamples to equality and constancy at the experiment's lengths;
they do not repeat the historical identity audits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from permworld_platform.encoding import encode_example
from permworld_platform.registry import task_output_bounds, task_spec
from permworld_platform.relations import historical_references
from permworld_platform.tasks import label


def spec(task_id, **params):
    return {"task_id": task_id, "params": params}


TARGETS = [spec(x) for x in ("cycle_count", "lis_length", "components", "fixed_points")]
K_VALUES = [2, 4, 8, 16]
LENGTHS = [10, 12]
CHAINS = {
    "chain_alpha": [
        spec("nontrivial_cycle_count"), spec("lds_length"),
        spec("largest_component"), spec("descents"),
        spec("even_cycle_count"), spec("increasing_subsequence_count", k=3),
        spec("global_descents"), spec("exceedances"),
        spec("odd_cycle_count"), spec("cycle_length_count", k=2),
        spec("decreasing_subsequence_count", k=3), spec("smallest_component"),
        spec("component_length_count", k=1), spec("length"),
        spec("recoils"), spec("deficiencies"),
    ],
    "chain_beta": [
        spec("longest_cycle"), spec("decreasing_subsequence_count", k=3),
        spec("smallest_component"), spec("peaks"),
        spec("cycle_length_count", k=3), spec("lds_length"),
        spec("component_length_count", k=2), spec("valleys"),
        spec("nontrivial_cycle_count"), spec("even_cycle_count"),
        spec("increasing_subsequence_count", k=3), spec("global_descents"),
        spec("largest_component"), spec("anti_fixed_points"),
        spec("longest_increasing_run"), spec("longest_decreasing_run"),
    ],
}


def task_key(task):
    """Same scalar-parameter slug convention as platform calibration.task_slug."""
    return task["task_id"] + "".join(f"_{k}{v}" for k, v in sorted(task["params"].items()))


SOURCE_POOL = sorted({task_key(t): t for tasks in CHAINS.values() for t in tasks}.values(), key=task_key)


def broad_family(task):
    family = task_spec(task["task_id"])["family"]
    return family if family in {"cycle_decomposition", "global_subsequence", "permutation_blocks"} else "legacy_statistics"


EXCLUSIONS = [
    {"instances": [task_key(t) for t in TARGETS], "reason": "All four held-out targets are excluded globally, not just for their own adaptation."},
    {"instances": ["cycle_length_count_k1"], "reason": "Exact same-input answer alias of held-out fixed_points: c_1(p)=FP(p).", "relation_ids": ["cycle_length_one_fixed"]},
    {"instances": ["two_cycle_count", "three_cycle_count"], "reason": "Complete aliases of the retained cycle_length_count k=2 and k=3 instances.", "relation_ids": ["cycle_length_two_alias", "cycle_length_three_alias"]},
    {"instances": ["decreasing_subsequence_count_k2"], "reason": "Exact same-input answer alias of retained length (inversion count).", "relation_ids": ["decreasing_pairs_inversions"]},
    {"instances": ["increasing_subsequence_count_k1", "decreasing_subsequence_count_k1"], "reason": "Equal to n; constant conditional on length and exact aliases of each other.", "relation_ids": ["increasing_subsequence_count_k1_constant", "decreasing_subsequence_count_k1_constant"]},
    {"instances": ["all k>n count instances"], "reason": "Constant zero for the corresponding input length; none are instantiated."},
    {"instances": ["rsk_shape", "cycle_type", "all representation and permutation-valued endpoints", "all binary endpoints"], "reason": "Outside this prespecified scalar experiment. Shape, cycle type and representations can expose target answers by deterministic postprocessing; they are not mislabeled as exact answer aliases."},
    {"instances": ["increasing_subsequence_count_k2"], "reason": "Not a full alias: equals binomial(n,2)-length(p). Omitted to use k=3 counts and exercise multi-token answers; this is a design restriction, not an unrelatedness claim."},
]


def configuration():
    """Return an independent JSON-safe copy; importing this module freezes nothing."""
    return copy.deepcopy({
        "schema_version": "task-count-design-v1", "targets": TARGETS,
        "k_values": K_VALUES, "lengths": LENGTHS, "chains": CHAINS,
        "source_pool": SOURCE_POOL, "k_definition": "Number of nonconstant task instances after removing complete same-input answer aliases; not number of independent mathematical quantities.",
        "selection_rule": "Two human-specified nested, family-stratified chains chosen from exact definitions, before this experiment's training and transfer outcomes. Prefix family slots are cycle, subsequence, block, legacy; repeated at k=8; k=16 adds two cycles, one subsequence, two blocks and three legacy statistics. Different chain membership is intentional. All 24 pool instances occur in the union.",
        "family_counts": {str(k): dict(sorted(Counter(broad_family(t) for t in CHAINS["chain_alpha"][:k]).items())) for k in K_VALUES},
        "template_count_note": "The pool has 22 templates and 24 instances: cycle_length_count and component_length_count each have two k values. Each individual prefix uses distinct templates, so its template count equals its k. One input encoding: one_line_v1; sequence protocol permworld_sequence_v1.",
        "exclusions": EXCLUSIONS,
        "relations_policy": "Transformed-input, answer-map and joint relations are retained and annotated. Absence of a registered derivation means unclassified, never mathematically unrelated.",
        "interpretation": "A limited task-count curve along two fixed compositions. Increasing k changes composition, available relations, answer distribution and per-task exposure at fixed total tokens. These effects are not identified separately, and the two chains are not independent statistical repeats.",
        "witness_policy": {"lengths": LENGTHS, "seed": 2026102701, "random_proposals_per_length": 1024, "training_examples": False, "purpose": "Counterexamples to same-input equality and constancy, not task difficulty selection or train/dev/test performance."},
        "no_transfer_results_read": True, "additional_model_selection": False,
    })


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixtures(n):
    identity = tuple(range(1, n + 1))
    proposals = [identity, identity[::-1]]
    proposals += [identity[j:] + identity[:j] for j in range(1, n)]
    for i in range(n - 1):
        p = list(identity)
        p[i], p[i + 1] = p[i + 1], p[i]
        proposals.append(tuple(p))
    rng = random.Random(2026102701 + n)
    for _ in range(1024):
        p = list(identity)
        rng.shuffle(p)
        proposals.append(tuple(p))
    return list(dict.fromkeys(proposals))


def _witness_audit():
    counters, differences, values_over_99, encoding_rows = [], [], [], []
    all_tasks = SOURCE_POOL + TARGETS
    input_hashes = {}
    for n in LENGTHS:
        fixtures = _fixtures(n)
        input_hashes[str(n)] = hashlib.sha256(json.dumps(fixtures).encode()).hexdigest()
        outputs = {task_key(t): [label(t["task_id"], p, t["params"]) for p in fixtures] for t in all_tasks}
        for task in all_tasks:
            key, vals = task_key(task), outputs[task_key(task)]
            different = next((i for i, value in enumerate(vals) if value != vals[0]), None)
            if different is None:
                raise AssertionError(f"No nonconstancy witness for {key}, n={n}")
            counters.append({"task_key": key, "n": n, "input_a": list(fixtures[0]), "answer_a": vals[0], "input_b": list(fixtures[different]), "answer_b": vals[different]})
            encoded = [encode_example(task["task_id"], p, task["params"], max_seq_len=128) for p in fixtures]
            max_answer = max(len(row["answer_ids"]) for row in encoded)
            if max_answer > 8:
                raise AssertionError(f"Generation bound too small for {key}: {max_answer}")
            encoding_rows.append({"task_key": key, "n": n, "observed_min": min(vals), "observed_max": max(vals), "answer_token_max_including_eos": max_answer, "input_sequence_max_tokens": max(len(row["input_ids"]) for row in encoded), "exact_bounds": json.dumps(task_output_bounds(task["task_id"], n, task["params"])), "fixtures": len(fixtures)})
            if any(v >= 100 for v in vals):
                i = next(i for i, v in enumerate(vals) if v >= 100)
                values_over_99.append({"task_key": key, "n": n, "permutation": list(fixtures[i]), "answer": vals[i], "answer_ids": encoded[i]["answer_ids"]})
        for a, b in itertools.combinations(all_tasks, 2):
            # Target/target comparisons are harmless but not required for exclusion.
            ka, kb = task_key(a), task_key(b)
            index = next((i for i, (va, vb) in enumerate(zip(outputs[ka], outputs[kb])) if va != vb), None)
            if index is None:
                raise AssertionError(f"Potential same-input alias at n={n}: {ka}, {kb}")
            differences.append({"task_a": ka, "task_b": kb, "n": n, "permutation": list(fixtures[index]), "answer_a": outputs[ka][index], "answer_b": outputs[kb][index]})
    return {"status": "passed", "nonconstant_witnesses": counters, "inequality_witnesses": differences, "multi_token_answer_witnesses": values_over_99, "input_sha256": input_hashes, "fixtures_used_for_training": False, "interpretation": "Each exact counterexample proves that its two functions are not identically equal on S_n, or that its task is not constant on S_n. It does not establish independence, equal difficulty or absence of derivability."}, encoding_rows


def _reuse_relations():
    base = ROOT / "experiments/platform_v01/mathematics"
    certificate = json.loads((base / "CERTIFICATE.json").read_text(encoding="utf-8"))
    if certificate["status"] != "passed":
        raise RuntimeError("The platform certificate did not pass")
    checked = {}
    for relative, digest in certificate["source_sha256"].items():
        if _sha(ROOT / relative) != digest:
            raise RuntimeError(f"Unchanged mathematical implementation required: {relative}")
        checked[relative] = digest
    for name in ("RELATIONS.json", "NEW_RELATION_CHECKS.csv", "LABEL_CHECKS.csv", "EQUIVALENCE_AUDIT.json"):
        digest = certificate["artifact_sha256"][name]
        if _sha(base / name) != digest:
            raise RuntimeError(f"Mathematical evidence changed: {name}")
        checked[str((base / name).relative_to(ROOT)).replace("\\", "/")] = digest
    checked[str((base / "CERTIFICATE.json").relative_to(ROOT)).replace("\\", "/")] = _sha(base / "CERTIFICATE.json")
    historical = historical_references()
    registry = json.loads((base / "RELATIONS.json").read_text(encoding="utf-8"))
    return registry, {"status": "reused_hash_verified", "historical_formula_rechecks_run": 0, "platform_formula_rechecks_run": 0, "sha256": checked, "historical": historical}


def _target_relationships(chain, k):
    keys = {task_key(t) for t in CHAINS[chain][:k]}
    rows = []
    def add(target, relation, evidence, formula, requires, transforms="none", available=True, note=""):
        rows.append({"chain": chain, "k": k, "target": target, "relation_id": relation, "evidence_class": evidence, "formula": formula, "required_source_instances": ";".join(requires), "input_transforms": transforms, "all_required_instances_present": bool(available), "same_input_derivation_available": bool(available and transforms == "none"), "note": note})
    add("cycle_count", "cycle_parity_partition", "proved_from_definitions", "cycle_count(p)=even_cycle_count(p)+odd_cycle_count(p)", ["even_cycle_count", "odd_cycle_count"], available={"even_cycle_count", "odd_cycle_count"} <= keys)
    add("fixed_points", "position_partition", "proved_from_definitions", "fixed_points(p)=n-exceedances(p)-deficiencies(p)", ["exceedances", "deficiencies"], available={"exceedances", "deficiencies"} <= keys)
    add("fixed_points", "composition_cycle_parity_and_fixed_partition", "symbolic_composition_of_reused_proved_formulas", "fixed_points(p)=even_cycle_count(p)+odd_cycle_count(p)-nontrivial_cycle_count(p)", ["even_cycle_count", "odd_cycle_count", "nontrivial_cycle_count"], available={"even_cycle_count", "odd_cycle_count", "nontrivial_cycle_count"} <= keys, note="Substitute cycle_parity_partition into cycle_fixed_partition; arithmetic rearrangement only, no new empirical check.")
    add("fixed_points", "fixed_from_two_deficiency_queries", "proved_from_definitions", "fixed_points(p)=n-deficiencies(p)-deficiencies(inverse(p))", ["deficiencies"], "identity and inverse", "deficiencies" in keys, "Needs answers at p and inverse(p), which usually are distinct. Training data isolation uses full I/R/C orbits; this legal mathematical relationship is not input leakage.")
    add("fixed_points", "fixed_complement", "proved_from_definitions", "fixed_points(p)=anti_fixed_points(complement(p))", ["anti_fixed_points"], "complement", "anti_fixed_points" in keys)
    add("fixed_points", "fixed_reverse", "proved_from_definitions", "fixed_points(p)=anti_fixed_points(reverse(p))", ["anti_fixed_points"], "reverse", "anti_fixed_points" in keys)
    for transform in ("complement", "reverse"):
        add("lis_length", f"lis_length_{transform}", "proved_from_definitions", f"lis_length(p)=lds_length({transform}(p))", ["lds_length"], transform, "lds_length" in keys)
    add("components", "components_complement_global_descents", "proved_from_definitions", "components(p)=global_descents(complement(p))+1", ["global_descents"], "complement", "global_descents" in keys)
    for target, task in (("cycle_count", "cycle_length_count"), ("lis_length", "increasing_subsequence_count"), ("components", "component_length_count")):
        add(target, f"{target}_from_full_histogram", "proved_from_definitions", f"Complete {task}(p,j), j=1..n determines {target}(p)", [f"{task}_k1..kn"], available=False, note="Only a few histogram entries are sources. The complete certified histogram identity does not establish reconstruction from this subset.")
    for target in (t["task_id"] for t in TARGETS):
        add(target, "other_joint_reconstruction", "unchecked", "Other joint derivations have not been exhaustively classified", [], available=False, note="No complete derivation search; unlisted does not mean unrelated. A model's use of any formula is untested.")
    return rows


README = """# Frozen task-count design

The task lists were specified from mathematical definitions before training this
experiment. This module reads no transfer metrics or calibration performance.
Both chains are fixed prefixes, with different membership at every k=2,4,8,16.
They use the same broad family counts: (1 cycle, 1 subsequence) at k=2;
(1,1,1 block,1 legacy) at k=4; (2,2,2,2) at k=8; (4,3,4,5) at k=16.
Thus k=2 does not span all families. At k=16 their intersection has 8 instances
and their union has all 24 pool instances. Family strata do not match difficulty.

`configuration.json` contains the complete ordered lists and selection rule;
`task_sets.csv` gives every membership and its position. `source_pool.csv` binds
instances to registry templates, mathematical families and exact definitions.
There are 24 pool instances from 22 templates; each chain prefix happens to use
distinct templates, so its template count equals k. There is one input encoding.
The term k is not a count of independent mathematical information.

All four held-out targets are excluded from every source set. The fixed-point
alias cycle_length_count(k=1), old two/three-cycle aliases, D_2=inversion-count,
and conditional-on-length constants are excluded. Non-scalar tasks are outside
this scalar experiment. Registry tasks omitted for scope are not called aliases
or unrelated. Exclusions and their reasons are explicitly listed in the config.

## Retained relationships and interpretation

`target_relationships.csv` distinguishes same-input reconstruction from transformed
queries and incomplete histogram availability. In particular, alpha at k=16 can
reconstruct cycle_count=even_cycle_count+odd_cycle_count and fixed_points via
either n-exceedances-deficiencies or even+odd-nontrivial_cycle_count. The latter
is a symbolic composition of two previously certified partition identities.
Beta does not contain those complete same-input query sets, but it retains
fixed_points(p)=anti_fixed_points(complement(p)). Both chains eventually contain
LDS(complement(p))=LIS(p) and global_descents(complement(p))+1=components(p).
Some relations require labels at different transformed inputs; the metadata
does not imply these inputs are jointly presented during training. Legal task
relationships are not automatically input leakage. Additional joint derivations
are unclassified, and no mechanism claim is made about a trained model.

`relation_inventory.json` reuses the platform's full 74-entry inventory with its
proof, finite-check and historical observation distinctions. Its old empirical
references are retained as metadata and are not read to select the lists.
`relation_reuse.json` records byte-hash reuse of existing evidence: no unchanged
historical formula is rechecked. Source-source links, including inverse
invariance, monotone subsequence complement/reversal and block identities, remain
available in that inventory; missing edges mean unclassified.

## New validation, without model selection

`inequivalence_audit.json` records exact witnesses at both n=10 and n=12 showing
each selected source is nonconstant and no source has the same answer function
as any other source or target. Fixtures are identity/reversal, shifts, adjacent
swaps and 1,024 seeded uniform proposals per length. They are not train/dev/test
data. One unequal answer disproves identical functions; this is not a proof of
independence, matched difficulty or absence of joint derivability.
`encoding_bounds.csv` checks exact output bounds and complete sequence encoding.
The I_3 and D_3 sources include multi-token integer answers up to 220 at n=12;
eight generation tokens are sufficient for all selected scalar outputs.

Increasing k also changes composition, available relations, answer-length
distribution and exposure per task. Results establish only finite curves for
these two lists, not a pure causal effect of count or a scaling law. The chains
share models, targets and baselines and do not create extra independent repeats.

Recreate into a fresh output directory from the repository root:

```powershell
.venv/Scripts/python.exe experiments/task_count_v1/design.py --out experiments/task_count_v1_reproduction/design
```

Reusing an existing output verifies its frozen configuration, source and artifact
hashes. It never silently changes the design.
"""


def build_design(out=None):
    out = Path(out) if out is not None else Path(__file__).with_suffix("")
    freeze = out / "FROZEN_DESIGN.json"
    cfg = configuration()
    source_hash = _sha(__file__)
    if freeze.exists():
        prior = json.loads(freeze.read_text(encoding="utf-8"))
        if prior["configuration"] != cfg or prior["design_source_sha256"] != source_hash:
            raise RuntimeError("Refusing to change a frozen task design; use a fresh output")
        for name, digest in prior["artifact_sha256"].items():
            if _sha(out / name) != digest:
                raise RuntimeError(f"Frozen design artifact changed: {name}")
        return prior
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Refusing to populate an incomplete/nonempty design directory")
    out.mkdir(parents=True, exist_ok=True)
    for k in K_VALUES:
        sets = [{task_key(t) for t in chain[:k]} for chain in CHAINS.values()]
        assert all(len(s) == k for s in sets) and sets[0] != sets[1]
        counts = [Counter(broad_family(t) for t in chain[:k]) for chain in CHAINS.values()]
        assert counts[0] == counts[1]
    assert len(SOURCE_POOL) == 24
    assert len({t["task_id"] for t in SOURCE_POOL}) == 22
    assert all(task_spec(t["task_id"])["output_type"] == "integer" for t in SOURCE_POOL + TARGETS)
    witnesses, encodings = _witness_audit()
    relations, reuse = _reuse_relations()
    rows, summaries, relationship_rows = [], [], []
    for chain_name, chain in CHAINS.items():
        for k in K_VALUES:
            summaries.append({"chain": chain_name, "k": k, "task_instances": k, "templates": len({t["task_id"] for t in chain[:k]}), "input_encoding_variants": 1, "broad_family_counts": dict(sorted(Counter(broad_family(t) for t in chain[:k]).items()))})
            for position, task in enumerate(chain[:k], 1):
                rows.append({"chain": chain_name, "k": k, "position": position, "task_key": task_key(task), "task_id": task["task_id"], "params": json.dumps(task["params"], sort_keys=True), "broad_family": broad_family(task), "registry_family": task_spec(task["task_id"])["family"]})
            relationship_rows.extend(_target_relationships(chain_name, k))
    pool_rows = []
    for task in SOURCE_POOL:
        info = task_spec(task["task_id"])
        pool_rows.append({"task_key": task_key(task), "task_id": task["task_id"], "params": json.dumps(task["params"], sort_keys=True), "broad_family": broad_family(task), "registry_family": info["family"], "definition": info["mathematical_definition"], "algorithm": info["exact_answer_method"], "output_type": info["output_type"], "same_input_target_alias": False, "nonconstant_n10_and_n12": True, "relation_status": "See proved/finite/empirical/unchecked metadata; absence does not mean unrelated"})
    _dump(out / "configuration.json", cfg)
    _dump(out / "counts.json", {"pool_instances": 24, "pool_templates": 22, "input_encoding_variants": 1, "prefixes": summaries})
    _csv(out / "task_sets.csv", rows)
    _csv(out / "source_pool.csv", pool_rows)
    _csv(out / "target_relationships.csv", relationship_rows)
    _csv(out / "encoding_bounds.csv", encodings)
    _dump(out / "inequivalence_audit.json", witnesses)
    _dump(out / "relation_inventory.json", relations)
    _dump(out / "relation_reuse.json", reuse)
    (out / "README.md").write_text(README, encoding="utf-8")
    result = {"status": "frozen", "configuration": cfg, "design_source_sha256": source_hash, "artifact_sha256": {p.name: _sha(p) for p in sorted(out.iterdir()) if p.is_file()}, "nonconstancy_witness_count": len(witnesses["nonconstant_witnesses"]), "pairwise_inequivalence_witness_count": len(witnesses["inequality_witnesses"]), "transfer_results_read": False, "historical_formula_rechecks_run": 0}
    _dump(freeze, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).with_suffix(""))
    args = parser.parse_args()
    result = build_design(args.out)
    print(json.dumps({"status": result["status"], "source_pool_instances": len(SOURCE_POOL), "source_pool_templates": len({t["task_id"] for t in SOURCE_POOL}), "nonconstancy_witnesses": result["nonconstancy_witness_count"], "pairwise_inequivalence_witnesses": result["pairwise_inequivalence_witness_count"], "output": str(args.out)}))
