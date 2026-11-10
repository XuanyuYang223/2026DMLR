"""CPU mathematical preflight; no research data generation or model execution."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from permworld_platform import label
from permworld_platform.objects import TRANSFORM_ACTIONS, transform
from permworld_platform.relations import historical_references

COMMON = ["descents", "successions", "left_to_right_maxima"]
CANDIDATES = {"P": "peaks", "D": "deficiencies", "T": "anti_fixed_points", "U": "right_to_left_maxima"}
V5 = ROOT / "experiments/pilot_v5_confirmation"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(out):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    old_certificates = historical_references()
    oracle_path = V5 / "mathematics/audit_relations.py"
    spec = importlib.util.spec_from_file_location("v5_definition_oracle", oracle_path)
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)
    v5 = read(V5 / "artifacts/config.json")
    groups = v5["groups"]
    composition_rows = []
    for condition, candidate in CANDIDATES.items():
        sources = COMMON + [candidate]
        background = {"P": "g0_v3", "D": "g2_deficiencies"}.get(condition)
        old = next((g["sources"]["C"] for g in groups if g["target"] == "fixed_points"
                    and g["background"] == background), None)
        if old:
            assert set(old) == set(sources)
            assert [old.index(task) for task in sources] == [1, 0, 3, 2]
        composition_rows.append(dict(condition=condition, sources=json.dumps(sources),
            old_C_background=background or "not_an_old_C_set", old_slot_order=json.dumps(old),
            new_slots_as_old_slot_indices=json.dumps([1, 0, 3, 2] if old else None),
            exact_old_training_replay=False, target_present=False))

    with (V5 / "mathematics/audit/COUNTEREXAMPLES.csv").open(encoding="utf-8-sig", newline="") as handle:
        historical = list(csv.DictReader(handle))
    def witness(query_model):
        row = next(r for r in historical if r["target"] == "fixed_points" and
                   r["background"] == "g0_v3" and r["condition"] == "C" and
                   r["query_model"] == query_model)
        return tuple(json.loads(row["p"])), tuple(json.loads(row["q"]))

    comparisons = 0
    def answers(p, tasks, actions):
        nonlocal comparisons
        values = []
        for action in actions:
            q = transform(action, p)
            old_answers = independent.reference(q)
            for task in tasks:
                expected = (sum(all(q[i] > q[j] for j in range(i + 1, len(q)))
                                for i in range(len(q))) if task == "right_to_left_maxima"
                            else old_answers[task])
                actual = label(task, q)
                assert actual == expected, (task, q, expected, actual)
                comparisons += 1
                values.append(actual)
        return values

    collisions = []
    for condition, candidate in CANDIDATES.items():
        sources = COMMON + [candidate]
        p, q = witness("all_eight_transforms" if condition == "U" else "same_input")
        for mode, actions in [("same_input", ["000"]), ("all_eight_transforms", TRANSFORM_ACTIONS)]:
            if mode == "all_eight_transforms" and condition in {"D", "T"}:
                continue
            if mode == "all_eight_transforms":
                p, q = witness(mode)
            yp, yq = answers(p, sources, actions), answers(q, sources, actions)
            assert yp == yq and label("fixed_points", p) != label("fixed_points", q)
            collisions.append(dict(condition=condition, query_model=mode, n=len(p),
                sources=json.dumps(sources), actions=json.dumps(list(actions)),
                p=json.dumps(p), q=json.dumps(q), source_answers=json.dumps(yp),
                fixed_points_p=label("fixed_points", p), fixed_points_q=label("fixed_points", q),
                conclusion="No deterministic target map from this specified answer vector and n on all S_10",
                raw_input_scope="The full permutation is not part of this answer-vector sufficiency claim"))

    # Only U's oracle and new combination bounds are enumerated. Historical formulas
    # above are reused by hash; their original exhaustive checks are not rerun.
    new_checks = 0
    for n in range(1, 7):
        for p in itertools.permutations(range(1, n + 1)):
            u = label("right_to_left_maxima", p)
            oracle = sum(all(p[i] > p[j] for j in range(i + 1, n)) for i in range(n))
            assert u == oracle
            d, left = label("descents", p), label("left_to_right_maxima", p)
            assert 1 <= u <= n and u <= d + 1 and left <= n - d and left + u <= n + 1
            new_checks += 1

    derivability = []
    for condition in CANDIDATES:
        derivability.append(dict(condition=condition, same_input="disproved by saved n=10 counterexample",
            eight_transform_queries=("recoverable: FP(p)=n-D(p)-D(inverse(p))" if condition == "D" else
                                     "recoverable: FP(p)=AF(complement(p))=AF(reverse(p))" if condition == "T" else
                                     "disproved by saved n=10 counterexample"),
            learned_mechanism="untested", new_training_run=False))
    write_csv(out / "COMPOSITIONS.csv", composition_rows)
    write_csv(out / "COUNTEREXAMPLES.csv", collisions)
    write_csv(out / "JOINT_DERIVABILITY.csv", derivability)
    bindings = [oracle_path, V5 / "artifacts/config.json", V5 / "mathematics/audit/COUNTEREXAMPLES.csv",
                ROOT / "permworld_platform/tasks.py", ROOT / "permworld_platform/objects.py",
                ROOT / "experiments/task_count_v2/data_pipeline.py",
                ROOT / "experiments/relation_readout_v1/data_manifest.json"]
    bindings += [ROOT / "experiments/pilot_v4_backgrounds/catalog" / name for name in
                 ("DIRECT_RELATION_DEFINITIONS.csv", "RELATION_GRAPH.csv", "RELATION_AUDIT.csv", "RELATION_AUDIT.json")]
    report = dict(status="passed", scope="Instruction-independent mathematical/protocol preflight only",
        historical_certificates=old_certificates, historical_full_enumerations_rerun=0,
        new_U_oracle_and_combination_bound_fixtures=new_checks, new_U_exhaustive_lengths=list(range(1,7)),
        witness_label_cross_checks=comparisons, counterexamples=len(collisions),
        input_length_scope="Counterexamples establish insufficiency on S_10; no training lengths have been selected",
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in bindings},
        elapsed_seconds=time.perf_counter()-started, formal_protocol_frozen=False,
        research_inputs_generated=0, training_runs=0, model_forwards=0, checkpoints_opened=0)
    (out / "AUDIT.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Fresh audit directory; existing paths are rejected")
    run(parser.parse_args().out)
