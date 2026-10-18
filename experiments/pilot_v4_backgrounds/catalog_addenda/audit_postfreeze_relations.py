"""Definition-only, CPU-only addendum; never read model/transfer artifacts.

Run with -B. Existing output is protected; use --out for a reproduction file.
The original catalog, protocol, task composition, and training are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import time
from datetime import datetime, timezone

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE / 'catalog'))
import data_platform as dp


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(out):
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}; supply --out NEW_FILE.json')
    started = time.perf_counter()
    protected = sorted(p for p in (BASE / 'catalog').rglob('*') if p.is_file())
    protected += [BASE / 'TASK_CATALOG.csv', BASE / 'DATA_PLATFORM.md', BASE / 'design.py',
                  BASE / 'artifacts/FROZEN_PROTOCOL.json']
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    f = dp.PROPERTY_FUNCTIONS
    names = ['fixed_points', 'anti_fixed_points', 'deficiencies', 'descents',
             'recoils', 'successions', 'peaks', 'longest_decreasing_run',
             'longest_increasing_run']
    counts = {k: 0 for k in ['fixed_two_def_queries', 'anti_fixed_two_def_queries',
        'peaks_reverse', 'successions_reverse_complement', 'deficiencies_RCI',
        'decreasing_run_bounds', 'increasing_run_bounds',
        'succession_descents_bound', 'succession_recoils_bound']}

    def check(p):
        n = len(p)
        v = {t: f[t](p) for t in names}
        d = v['descents']
        equations = {
            'fixed_two_def_queries': v['fixed_points'] == n - v['deficiencies']
                - f['deficiencies'](dp.transform(p, '100')),
            'anti_fixed_two_def_queries': v['anti_fixed_points'] == n
                - f['deficiencies'](dp.transform(p, '001'))
                - f['deficiencies'](dp.transform(p, '110')),
            'peaks_reverse': v['peaks'] == f['peaks'](dp.transform(p, '010')),
            'successions_reverse_complement': v['successions'] ==
                f['successions'](dp.transform(p, '011')),
            'deficiencies_RCI': v['deficiencies'] ==
                f['deficiencies'](dp.transform(p, '111')),
            'decreasing_run_bounds': (n + (n-d) - 1) // (n-d) <=
                v['longest_decreasing_run'] <= d + 1,
            'increasing_run_bounds': (n + d) // (d+1) <=
                v['longest_increasing_run'] <= n-d,
            'succession_descents_bound': v['successions'] + d <= n-1,
            'succession_recoils_bound': v['successions'] + v['recoils'] <= n-1,
        }
        for name, holds in equations.items():
            if not holds:
                raise AssertionError((name, p, v))
            counts[name] += 1

    for n in range(1, 8):
        for p in itertools.permutations(range(1, n+1)):
            check(p)
    rng = random.Random(2026101806)
    for _ in range(2048):
        check(tuple(rng.sample(range(1, 11), 10)))

    p = (1, 3, 2, *range(4, 11))
    q = (2, 3, 1, *range(4, 11))
    assert f['deficiencies'](p) == f['deficiencies'](q) == 1
    assert f['fixed_points'](p) == 8 and f['fixed_points'](q) == 7
    with (BASE / 'catalog/RELATION_GRAPH.csv').open(encoding='utf-8-sig', newline='') as stream:
        graph = list(csv.DictReader(stream))
    supplement_edges = []
    for task, action in [('peaks', '010'), ('successions', '011'), ('deficiencies', '111')]:
        present = any(row['left'] == task and row['right'] == task and
                      row['action'] == action for row in graph)
        supplement_edges.append(dict(task=task, action=action,
            already_in_frozen_graph=present, answer_mapping='identity'))

    after = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    assert before == after, 'Protected file changed during this read-only audit'
    assert 'torch' not in sys.modules
    report = dict(status='passed',
        audit_type='post-freeze definition-only scientific addendum',
        generated_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-started,
        protocol_and_compositions_changed=False,
        transfer_results_or_checkpoints_read=False,
        proof_scope='All n>=1 by the partition, inverse bijection and local comparison proofs in README.md; finite checks validate implementation.',
        transform_convention='bits I,R,C applied in that order; 110=R o I=I o C; 111=C o R o I',
        exhaustive_n=list(range(1, 8)), exhaustive_inputs=5913,
        random_n=10, random_seed=2026101806, random_inputs=2048,
        checked_inputs_per_relation=7961, total_identity_or_bound_checks=sum(counts.values()),
        checks=counts, supplement_self_edges=supplement_edges,
        same_input_single_deficiency_counterexample=dict(
            p=p, q=q, deficiencies_p=1, deficiencies_q=1, fixed_points_p=8, fixed_points_q=7),
        protected_file_hashes_before=before, protected_file_hashes_after=after,
        protected_files_unchanged=True,
        supplemental_source_hashes={p.name: sha(p) for p in [Path(__file__), HERE/'README.md']},
        vendor_math_sha256=sha(ROOT/'experiments/permworld_pilot/vendor/neurips_permutations/math_ops.py'),
        no_formal_test_inputs_generated=True,
        finite_checks_are_not_a_standalone_general_proof=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','checked_inputs_per_relation',
        'total_identity_or_bound_checks','protected_files_unchanged','elapsed_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=HERE/'RELATION_ADDENDUM_AUDIT.json')
    run(parser.parse_args().out.resolve())
