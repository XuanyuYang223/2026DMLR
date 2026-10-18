"""Lock and independently verify the evidence used for the v3 design freeze.

CPU only: reads source train/dev data, saved metrics, and checkpoint bytes.
Does not import the training engine or Torch, execute a model, or read tests.
Run after `pilot_v3.py freeze` and before the first v3 test is generated.
Existing evidence is verified byte-for-byte and is never overwritten.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / 'experiments/permworld_pilot/vendor'))
from neurips_permutations.math_ops import PROPERTY_FUNCTIONS

TASKS = list(PROPERTY_FUNCTIONS)
TASK_INDEX = {t: i for i, t in enumerate(TASKS)}


def read(path):
    return Path(path).read_text(encoding='utf-8-sig')


def js(path):
    return json.loads(read(path))


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def truth(value):
    if value is True or value == 'True': return True
    if value is False or value == 'False': return False
    raise AssertionError(f'Unexpected Boolean: {value!r}')


def close(a, b, description):
    assert math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12), description


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE / 'artifacts')
    args = parser.parse_args()
    out = args.out.resolve()
    output = out / 'FROZEN_EVIDENCE.json'
    started = time.perf_counter()
    existing = js(output) if output.exists() else None
    # Check existence only; no test contents or neural transfer outcomes are read.
    if existing is None:
        assert not list((out / 'data').glob('*_test.npz')), 'First evidence lock must precede new test generation'
        assert not (out / 'TEST_COMPLETE.json').exists(), 'New tests already evaluated'
    protocol_path = out / 'FROZEN_PROTOCOL.json'
    protocol = js(protocol_path)
    design = js(out / 'DESIGN_FROZEN.json')
    cfg = protocol['config']
    assert cfg == design['config']
    assert cfg['matching_design_seed'] == 17
    assert cfg['targets'] == ['fixed_points', 'exceedances', 'left_to_right_minima', 'longest_decreasing_run']
    assert cfg['n'] == 10 and cfg['pretrain_updates'] == 4800
    assert cfg['target_samples'] == 256
    assert design['unix'] <= protocol['unix'] <= time.time()
    assert sha(out / 'DESIGN_FROZEN.json') == protocol['design_sha256']
    assert sha(out / 'replacement_matching.csv') == protocol['matching_sha256']
    for name, expected in design['code'].items():
        assert sha(ROOT / name) == expected, name

    locked = {}

    def lock(path):
        path = Path(path).resolve()
        # Relative paths are portable when moving the entire experiment tree.
        import os
        name = os.path.relpath(path, out).replace('\\', '/')
        record = dict(sha256=sha(path), bytes=path.stat().st_size)
        if name in locked: assert locked[name] == record
        locked[name] = record
        return record['sha256']

    for name in ['FROZEN_PROTOCOL.json', 'DESIGN_FROZEN.json', 'DATA_READY.json',
                 'DESIGN_REFERENCES_COMPLETE.json', 'replacement_matching.csv',
                 'target_development_gates.csv']:
        lock(out / name)
    for path in [HERE / 'TASK_CATALOG.csv', HERE / 'catalog/CATALOG_AUDIT.json',
                 HERE / 'catalog/RELATION_AUDIT.csv',
                 HERE / 'catalog/REPLACEMENT_RELATION_COUNTEREXAMPLES.csv',
                 HERE / 'catalog/build_catalog.py', Path(__file__)]:
        lock(path)
    catalog_audit = js(HERE / 'catalog/CATALOG_AUDIT.json')
    assert catalog_audit['status'] == 'passed'
    assert sha(HERE / 'TASK_CATALOG.csv') == catalog_audit['catalog_sha256']
    assert catalog_audit['neural_calls'] == 0 and catalog_audit['neural_results_read'] is False
    for name, expected in catalog_audit['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    catalog = rows(HERE / 'TASK_CATALOG.csv')
    eligible = {r['task_id'] for r in catalog if r['world'] == 'PermWorld' and truth(r['pilot_v3_eligible'])}
    assert set(cfg['design_tasks']) <= eligible
    relation_rows = rows(HERE / 'catalog/RELATION_AUDIT.csv')
    counterexamples = rows(HERE / 'catalog/REPLACEMENT_RELATION_COUNTEREXAMPLES.csv')
    assert all(int(r['failures']) == 0 and int(r['checks']) == 7961 for r in relation_rows)

    pool_records = js(out / 'DATA_READY.json')['pools']
    distributions = {}
    arrays = {}
    for split, role in [('train', 'source_train'), ('dev', 'source_dev')]:
        path = out / 'data' / f's17_{role}.npz'
        expected = next(r for r in pool_records if r['seed'] == 17 and r['role'] == role)
        assert lock(path) == expected['sha256']
        with np.load(path, allow_pickle=False) as archive:
            permutations, labels = archive['p'], archive['y']
        assert permutations.shape == (expected['count'], cfg['n'])
        assert labels.shape == (expected['count'], len(TASKS))
        arrays[split] = labels
        distributions[split] = {}
        for task in cfg['design_tasks']:
            # Independently check the saved column against the mathematical function.
            gold = np.asarray([PROPERTY_FUNCTIONS[task](tuple(map(int, p))) for p in permutations], dtype=np.int16)
            assert np.array_equal(gold, labels[:, TASK_INDEX[task]]), (split, task)
            counts = np.bincount(gold, minlength=cfg['n'] + 1)
            probs = counts / len(gold)
            distributions[split][task] = dict(counts=counts.tolist(), probabilities=probs.tolist(),
                mean=float(gold.mean()), entropy_bits=float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))))

    metrics = {}
    checkpoint_records = {}
    for task in cfg['design_tasks']:
        folder = out / 'single' / f's17_{task}'
        meta_path = folder / 'completed.json'
        assert lock(meta_path) == protocol['single_refs'][task]
        lock(folder / 'curve.csv')
        meta = js(meta_path)
        assert meta['seed'] == 17 and meta['tasks'] == [task] and meta['kind'] == 'single'
        assert meta['updates'] == 4800 and meta['task_exposures'] == 307200
        assert meta['train_sha256'] == sha(out / 'data/s17_source_train.npz')
        assert meta['dev_sha256'] == sha(out / 'data/s17_source_dev.npz')
        checkpoints = {}
        for step in [1200, 2400, 4800]:
            path = folder / f'step{step}.pt'
            checkpoints[str(step)] = lock(path)
        assert checkpoints['4800'] == meta['checkpoint_sha256']
        assert (out / meta['checkpoint']).resolve() == (folder / 'step4800.pt').resolve()
        checkpoint_records[task] = checkpoints
        curve = rows(folder / 'curve.csv')
        assert len(curve) == 2 * len(cfg['source_eval_steps'])
        assert {(int(r['step']), r['split']) for r in curve} == {
            (s, split) for s in cfg['source_eval_steps'] for split in ['train', 'dev']}
        majority = int(np.bincount(arrays['train'][:, TASK_INDEX[task]]).argmax())
        for r in curve:
            assert int(r['seed']) == 17 and r['kind'] == 'single' and r['task'] == task
            assert int(r['length']) == 10 and int(r['majority_label']) == majority
            ys = arrays[r['split']][:, TASK_INDEX[task]]
            assert int(r['count']) == len(ys)
            close(r['majority_accuracy'], np.mean(ys == majority), (task, r['split'], 'majority'))
            assert 0 <= float(r['accuracy']) <= 1
            assert math.isfinite(float(r['answer_ce'])) and float(r['answer_ce']) >= 0
            metrics[task, int(r['step']), r['split']] = r

    def dev(task, step):
        return metrics[task, step, 'dev']

    gates = []
    matches = []
    groups = []
    saved_gates = rows(out / 'target_development_gates.csv')
    saved_matches = rows(out / 'replacement_matching.csv')
    assert len(saved_gates) == 4
    assert len(saved_matches) == sum(len(p['candidates']) for p in cfg['plans'].values()) == 6
    for target in cfg['targets']:
        endpoint = dev(target, 4800)
        acc, baseline = float(endpoint['accuracy']), float(endpoint['majority_accuracy'])
        gate = dict(target=target, accuracy=acc, majority_accuracy=baseline,
                    pass_gate=acc >= .90 and acc >= baseline + .15)
        assert gate['pass_gate'], target
        gates.append(gate)
        for source in [saved_gates, protocol['target_gates']]:
            saved = next(r for r in source if r['target'] == target)
            close(saved['accuracy'], acc, target)
            close(saved['majority_accuracy'], baseline, target)
            assert truth(saved['pass_gate']) == gate['pass_gate']
        plan = cfg['plans'][target]
        related = plan['related']
        # The three declared actions are involutions; offset 0 makes either
        # direction of the catalog certificate equivalent to the requested one.
        certificates = [r for r in relation_rows if {r['left'], r['right']} == {target, related}
                        and r['input_transform'] == plan['transform'] and int(r['offset']) == 0]
        assert certificates and plan['transform'] in ['inverse', 'reverse', 'complement'], target
        a = np.asarray(distributions['train'][related]['probabilities'])
        scores = []
        for rank, candidate in enumerate(plan['candidates']):
            b = np.asarray(distributions['train'][candidate]['probabilities'])
            tv = float(np.abs(a - b).sum() / 2)
            end_gap = abs(float(dev(related, 4800)['accuracy']) - float(dev(candidate, 4800)['accuracy']))
            curve_gap = float(np.mean([abs(float(dev(related, s)['accuracy']) - float(dev(candidate, s)['accuracy']))
                                       for s in cfg['source_eval_steps'] if s]))
            record = dict(target=target, related=related, replacement=candidate, tv=tv,
                          endpoint_dev_accuracy_gap=end_gap, mean_curve_accuracy_gap=curve_gap,
                          score=tv + end_gap + curve_gap, candidate_order=rank, selected=False)
            scores.append(record)
        min(scores, key=lambda r: (r['score'], r['candidate_order']))['selected'] = True
        for r in scores:
            saved = next(s for s in saved_matches if s['target'] == target and s['replacement'] == r['replacement'])
            assert saved['related'] == r['related'] and int(saved['candidate_order']) == r['candidate_order']
            for field in ['tv', 'endpoint_dev_accuracy_gap', 'mean_curve_accuracy_gap', 'score']:
                close(saved[field], r[field], (target, r['replacement'], field))
            assert truth(saved['selected']) == r['selected']
            matches.append(r)
        selected = next(r['replacement'] for r in scores if r['selected'])
        group = dict(target=target, sources={'B': [related, *plan['common']], 'C': [selected, *plan['common']]},
                     related=related, replacement=selected, transform=plan['transform'],
                     answer_map='identity', selection_reason=plan['reason'])
        assert group == next(g for g in protocol['groups'] if g['target'] == target)
        for tasks in group['sources'].values():
            assert len(set(tasks)) == 4 and not set(tasks).intersection(cfg['targets'])
        witnesses = [r for r in counterexamples if r['target'] == target and r['replacement_candidate'] == selected]
        assert len(witnesses) == 8
        assert all(not truth(r['arbitrary_source_answer_to_transformed_target_map']) for r in witnesses)
        groups.append(group)

    evidence = dict(version='pilot_v3_design_evidence/v1', status='passed',
        design_unix=design['unix'], protocol_unix=protocol['unix'],
        frozen_protocol_sha256=sha(protocol_path), files=dict(sorted(locked.items())),
        source_distributions=distributions, source_label_checks=sum(len(a) for a in arrays.values()) * len(cfg['design_tasks']),
        design_seed=17, single_tasks=cfg['design_tasks'], checkpoints=checkpoint_records,
        recomputed_target_gates=gates, recomputed_matching=matches, recomputed_groups=groups,
        catalog_status='passed', selected_relations_verified=True,
        old_and_new_test_contents_read=False, neural_calls=0, torch_imported='torch' in sys.modules,
        limitations=['Hashes and saved-metric arithmetic do not independently repeat model forwards.',
                    'This evidence lock is created after development and before the new test; it does not replace the earlier design freeze.',
                    'Single4800 matches total updates; single1200 matches per-task exposure, with different LR history.'])
    assert evidence['torch_imported'] is False
    if existing is not None:
        for key, value in evidence.items():
            assert existing.get(key) == value, f'Frozen evidence differs: {key}; refusing overwrite'
        assert existing['created_unix'] >= protocol['unix']
        print(json.dumps(dict(status='verified_existing_without_write', path=str(output),
                              files=len(locked), gates=len(gates), candidate_pairs=len(matches))), flush=True)
        return
    now = time.time()
    evidence.update(created_unix=now, created_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
        seconds=time.perf_counter() - started, created_before_new_test_generation=True,
        created_before_adapt_complete=not (out / 'ADAPT_COMPLETE.json').exists())
    # Exclusive create prevents overwriting a parallel evidence writer.
    with output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(status='created', path=str(output), files=len(locked), gates=len(gates),
                          candidate_pairs=len(matches), seconds=evidence['seconds'])), flush=True)


if __name__ == '__main__':
    main()
