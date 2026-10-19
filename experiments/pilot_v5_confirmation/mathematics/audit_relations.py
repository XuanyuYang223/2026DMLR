"""CPU-only definition audit; no model, training input, or new result access.

Outputs are append-only: reproduce into a new directory with --out.
The reference implementation below does not call vendor label functions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V4 = ROOT / 'experiments/pilot_v4_backgrounds'
VENDOR = ROOT / 'experiments/permworld_pilot/vendor/neurips_permutations/math_ops.py'
spec = importlib.util.spec_from_file_location('audited_vendor_math', VENDOR)
vendor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vendor)
GROUPS = json.loads((V4 / 'artifacts/FROZEN_PROTOCOL.json').read_text(encoding='utf-8'))['config']['groups']
ACTIONS = tuple(''.join(a) for a in itertools.product('01', repeat=3))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inverse(p):
    # Sort graph points on their value coordinate, independently of vendor code.
    return tuple(i for _, i in sorted((v, i) for i, v in enumerate(p, 1)))


def transform(p, action):
    if action[0] == '1':
        p = inverse(p)
    if action[1] == '1':
        p = p[::-1]
    if action[2] == '1':
        p = tuple(len(p) + 1 - x for x in p)
    return p


def reference(p):
    """Independent scalar definitions; runs by enumerating all contiguous intervals."""
    n = len(p)
    inv = inverse(p)
    s = tuple(int(p[i] > p[i + 1]) for i in range(n - 1))
    increasing = max(j - i for i in range(n) for j in range(i + 1, n + 1)
                     if all(p[k] < p[k + 1] for k in range(i, j - 1)))
    decreasing = max(j - i for i in range(n) for j in range(i + 1, n + 1)
                     if all(p[k] > p[k + 1] for k in range(i, j - 1)))
    cycles = []
    unseen = set(range(1, n + 1))
    while unseen:
        start = min(unseen)
        cycle = [start]
        cursor = p[start - 1]
        while cursor != start:
            cycle.append(cursor)
            cursor = p[cursor - 1]
        unseen.difference_update(cycle)
        cycles.append(cycle)
    return dict(
        fixed_points=sum(p[i - 1] == i for i in range(1, n + 1)),
        anti_fixed_points=sum(p[i - 1] == n + 1 - i for i in range(1, n + 1)),
        exceedances=sum(p[i - 1] > i for i in range(1, n + 1)),
        deficiencies=sum(p[i - 1] < i for i in range(1, n + 1)),
        descents=sum(s), recoils=sum(inv[i] > inv[i + 1] for i in range(n - 1)),
        peaks=sum(s[i] == 0 and s[i + 1] == 1 for i in range(n - 2)),
        valleys=sum(s[i] == 1 and s[i + 1] == 0 for i in range(n - 2)),
        double_ascents=sum(s[i] == s[i + 1] == 0 for i in range(n - 2)),
        double_descents=sum(s[i] == s[i + 1] == 1 for i in range(n - 2)),
        successions=sum(p[i + 1] - p[i] == 1 for i in range(n - 1)),
        adjacencies=sum(abs(p[i + 1] - p[i]) == 1 for i in range(n - 1)),
        longest_increasing_run=increasing, longest_decreasing_run=decreasing,
        left_to_right_maxima=sum(all(p[j] < p[i] for j in range(i)) for i in range(n)),
        cycle_count=len(cycles), nontrivial_cycle_count=sum(len(c) > 1 for c in cycles),
        even_cycle_count=sum(len(c) % 2 == 0 for c in cycles),
        odd_cycle_count=sum(len(c) % 2 == 1 for c in cycles),
        parity=sum(p[i] > p[j] for i in range(n) for j in range(i + 1, n)) % 2,
    )


# All statements have definition-level proofs; finite checks verify their implementations.
RELATIONS = [
    ('position_partition', 'same_input_joint_identity', 'F(p)+E(p)+D(p)=n',
     'At every position exactly one of p_i=i, p_i>i, p_i<i holds.', 'fixed_points;exceedances;deficiencies'),
    ('inverse_exceedance_deficiency', 'transformed_input_identity', 'E(p)=D(Ip)',
     'The graph point (i,p_i) with p_i>i becomes (p_i,i) below the diagonal after inversion.', 'exceedances;deficiencies'),
    ('fixed_from_two_deficiency_queries', 'multiple_transformed_inputs_joint_identity', 'F(p)=n-D(p)-D(Ip)',
     'Substitute E(p)=D(Ip) into the position partition.', 'fixed_points;deficiencies'),
    ('anti_fixed_from_two_deficiency_queries', 'multiple_transformed_inputs_joint_identity', 'AF(p)=n-D(Cp)-D(ICp)',
     'Apply the fixed-point two-query identity to Cp and use F(Cp)=AF(p).', 'anti_fixed_points;deficiencies'),
    ('fixed_complement', 'transformed_input_identity', 'F(p)=AF(Cp)',
     'n+1-p_i=n+1-i if and only if p_i=i.', 'fixed_points;anti_fixed_points'),
    ('fixed_reverse', 'transformed_input_identity', 'F(p)=AF(Rp)',
     'Reindex the reversed position j=n+1-i in the anti-diagonal equality.', 'fixed_points;anti_fixed_points'),
    ('fixed_inverse', 'transformed_input_identity', 'F(p)=F(Ip)',
     'A point is fixed by a bijection if and only if it is fixed by its inverse.', 'fixed_points'),
    ('anti_fixed_inverse', 'transformed_input_identity', 'AF(p)=AF(Ip)',
     'Graph transposition preserves the anti-diagonal i+p_i=n+1.', 'anti_fixed_points'),
    ('run_complement', 'transformed_input_identity', 'LDR(p)=LIR(Cp)',
     'Complement changes every strict descent in a contiguous interval into an ascent, preserving length.', 'longest_decreasing_run;longest_increasing_run'),
    ('run_reverse', 'transformed_input_identity', 'LDR(p)=LIR(Rp)',
     'Reversal maps contiguous decreasing intervals bijectively to increasing intervals of equal length.', 'longest_decreasing_run;longest_increasing_run'),
    ('descent_recoil_inverse', 'transformed_input_identity', 'd(p)=recoils(Ip)',
     'Recoils is the descent count of the inverse, and I is involutive.', 'descents;recoils'),
    ('descent_complement', 'transformed_input_affine_identity', 'd(Cp)=n-1-d(p)',
     'Each of the n-1 unequal adjacent comparisons reverses its sign.', 'descents'),
    ('descent_reverse', 'transformed_input_affine_identity', 'd(Rp)=n-1-d(p)',
     'Reversal reverses each of the n-1 adjacent comparisons.', 'descents'),
    ('peaks_reverse', 'transformed_input_identity', 'peaks(Rp)=peaks(p)',
     'The two neighbors exchange places while the center remains larger than both.', 'peaks'),
    ('peaks_complement', 'transformed_input_identity', 'peaks(p)=valleys(Cp)',
     'Complement reverses both inequalities at each interior position.', 'peaks;valleys'),
    ('interior_partition', 'same_input_joint_identity', 'peaks+valleys+double_ascents+double_descents=max(n-2,0)',
     'The two comparison bits at each interior position belong to exactly one of 01,10,00,11.', 'peaks;valleys;double_ascents;double_descents'),
    ('peak_valley_endpoints', 'same_input_identity_with_extra_input_features', 'peaks-valleys=s_last-s_first (0 when n=1)',
     'The number of 0-to-1 transitions minus 1-to-0 transitions telescopes along the comparison-bit sequence.', 'peaks;valleys;adjacent_comparison_endpoints'),
    ('successions_inverse', 'transformed_input_identity', 'successions(Ip)=successions(p)',
     'An edge from (i,a) to (i+1,a+1) maps bijectively under transposition to an edge from (a,i) to (a+1,i+1).', 'successions'),
    ('successions_reverse_complement', 'transformed_input_identity', 'successions(CRp)=successions(p)',
     'Reverse and complement together preserve an adjacent upward difference of one.', 'successions'),
    ('adjacency_two_successions', 'multiple_transformed_inputs_joint_identity', 'adjacencies(p)=successions(p)+successions(Cp)',
     'An absolute difference of one is exactly an upward or downward difference of one, exchanged by complement.', 'adjacencies;successions'),
    ('successions_descents_bound', 'same_input_inequality', 'successions(p)+d(p)<=n-1',
     'Every succession is an ascent, so successions and descents count disjoint adjacent positions.', 'successions;descents'),
    ('successions_recoils_bound', 'same_input_inequality', 'successions(p)+recoils(p)<=n-1',
     'Apply the previous bound to Ip and use succession inverse-invariance.', 'successions;recoils'),
    ('decreasing_run_bounds', 'same_input_inequality', 'ceil(n/(n-d))<=LDR<=d+1',
     'Cut at ascents into n-d decreasing runs; the largest is at least the average and uses at most all d descent edges.', 'longest_decreasing_run;descents'),
    ('increasing_run_bounds', 'same_input_inequality', 'ceil(n/(d+1))<=LIR<=n-d',
     'Cut at descents into d+1 increasing runs; use the average and the n-1-d available ascent edges.', 'longest_increasing_run;descents'),
    ('deficiencies_inverse_reverse_complement', 'transformed_input_identity', 'D(CRIp)=D(p)',
     'RC exchanges above- and below-diagonal graph points, and inverse exchanges E and D once more.', 'deficiencies'),
    ('cycle_fixed_partition', 'same_input_joint_identity', 'cycle_count=F+nontrivial_cycle_count',
     'Partition disjoint cycles into singleton and non-singleton cycles.', 'cycle_count;fixed_points;nontrivial_cycle_count'),
    ('cycle_parity_partition', 'same_input_joint_identity', 'cycle_count=even_cycle_count+odd_cycle_count',
     'Partition disjoint cycles by parity of their lengths.', 'cycle_count;even_cycle_count;odd_cycle_count'),
    ('permutation_parity_cycles', 'same_input_noninvertible_answer_map', 'parity=(n-cycle_count) mod 2',
     'A k-cycle is a product of k-1 transpositions; multiply the cycle signs.', 'parity;cycle_count'),
]


def equations(p, by_action):
    n = len(p)
    a, c, r, inv, ic, cr, cri = (by_action[k] for k in ['000','001','010','100','110','011','111'])
    d = a['descents']
    s = [int(p[i] > p[i + 1]) for i in range(n - 1)]
    return dict(
        position_partition=a['fixed_points'] + a['exceedances'] + a['deficiencies'] == n,
        inverse_exceedance_deficiency=a['exceedances'] == inv['deficiencies'],
        fixed_from_two_deficiency_queries=a['fixed_points'] == n-a['deficiencies']-inv['deficiencies'],
        anti_fixed_from_two_deficiency_queries=a['anti_fixed_points'] == n-c['deficiencies']-ic['deficiencies'],
        fixed_complement=a['fixed_points'] == c['anti_fixed_points'],
        fixed_reverse=a['fixed_points'] == r['anti_fixed_points'],
        fixed_inverse=a['fixed_points'] == inv['fixed_points'],
        anti_fixed_inverse=a['anti_fixed_points'] == inv['anti_fixed_points'],
        run_complement=a['longest_decreasing_run'] == c['longest_increasing_run'],
        run_reverse=a['longest_decreasing_run'] == r['longest_increasing_run'],
        descent_recoil_inverse=d == inv['recoils'],
        descent_complement=c['descents'] == n-1-d,
        descent_reverse=r['descents'] == n-1-d,
        peaks_reverse=a['peaks'] == r['peaks'],
        peaks_complement=a['peaks'] == c['valleys'],
        interior_partition=sum(a[k] for k in ['peaks','valleys','double_ascents','double_descents']) == max(n-2,0),
        peak_valley_endpoints=a['peaks']-a['valleys'] == (s[-1]-s[0] if s else 0),
        successions_inverse=a['successions'] == inv['successions'],
        successions_reverse_complement=a['successions'] == cr['successions'],
        adjacency_two_successions=a['adjacencies'] == a['successions']+c['successions'],
        successions_descents_bound=a['successions']+d <= n-1,
        successions_recoils_bound=a['successions']+a['recoils'] <= n-1,
        decreasing_run_bounds=(n+n-d-1)//(n-d) <= a['longest_decreasing_run'] <= d+1,
        increasing_run_bounds=(n+d)//(d+1) <= a['longest_increasing_run'] <= n-d,
        deficiencies_inverse_reverse_complement=a['deficiencies'] == cri['deficiencies'],
        cycle_fixed_partition=a['cycle_count'] == a['fixed_points']+a['nontrivial_cycle_count'],
        cycle_parity_partition=a['cycle_count'] == a['even_cycle_count']+a['odd_cycle_count'],
        permutation_parity_cycles=a['parity'] == (n-a['cycle_count']) % 2,
    )


def run(out):
    started = time.perf_counter()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Refusing to overwrite an audit directory; use --out NEW_DIRECTORY')
    out.mkdir(parents=True, exist_ok=True)
    protected = [VENDOR, V4/'design.py', V4/'TASK_CATALOG.csv', V4/'DATA_PLATFORM.md',
                 V4/'artifacts/FROZEN_PROTOCOL.json', V4/'PILOT_V4_REPORT.md',
                 V4/'artifacts/source_learning_table.csv']
    protected += [p for directory in ['catalog','catalog_addenda'] for p in (V4/directory).rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts]
    before = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(protected))}
    all_groups = [(g, cond) for g in GROUPS for cond in ['B','C']]
    identities = [(g['target'],g['background'],c) for g,c in all_groups]
    seen_same = {key:{} for key in identities}
    seen_transform = {key:{} for key in identities}
    same_witness = {}
    transform_witness = {}
    relation_counts = Counter()
    definition_checks = 0
    exhaustive_histograms = {}
    fixture_n10 = []

    def check(p, exhaustive):
        nonlocal definition_checks
        by_action = {action: reference(transform(p, action)) for action in ACTIONS}
        for action, labels in by_action.items():
            q = transform(p, action)
            for task, expected in labels.items():
                actual = vendor.parity(q) if task == 'parity' else vendor.PROPERTY_FUNCTIONS[task](q)
                assert actual == expected, (task, p, action, actual, expected)
                definition_checks += 1
        eq = equations(p, by_action)
        assert set(eq) == {r[0] for r in RELATIONS}
        for name, holds in eq.items():
            assert holds, (name, p)
            relation_counts[name] += 1
        if exhaustive:
            hist = exhaustive_histograms.setdefault(len(p), {'descents':Counter(), 'recoils':Counter()})
            for task in hist:
                hist[task][by_action['000'][task]] += 1
        # Same-input counterexamples use actual n=10 domain, not just a smaller-n witness.
        if len(p) != 10:
            return
        fixture_n10.append(p)
        for g, condition in all_groups:
            key = (g['target'], g['background'], condition)
            target = by_action['000'][g['target']]
            source = tuple(by_action['000'][t] for t in g['sources'][condition])
            complete = tuple(by_action[a][t] for a in ACTIONS for t in g['sources'][condition])
            for signature, store, witnesses, mode in [(source,seen_same,same_witness,'same_input'),
                                                      (complete,seen_transform,transform_witness,'all_eight_transforms')]:
                if key in witnesses:
                    continue
                previous = store[key].get(signature)
                if previous is not None and previous[1] != target:
                    witnesses[key] = dict(target=key[0],background=key[1],condition=key[2],
                        query_model=mode,n=10,sources=json.dumps(g['sources'][condition]),
                        p=json.dumps(previous[0]),q=json.dumps(p),
                        source_answers_p=json.dumps(signature),source_answers_q=json.dumps(signature),
                        target_p=previous[1],target_q=target,
                        implication='No deterministic map from this ordered source-answer vector and n to target on all S_10.')
                else:
                    store[key][signature] = (p,target)

    for n in range(1,8):
        for p in itertools.permutations(range(1,n+1)):
            check(p, True)
    rng = random.Random(2026101811)
    for _ in range(4096):
        check(tuple(rng.sample(range(1,11),10)), False)
    assert len(same_witness) == 12, 'Need a same-input n=10 counterexample for every current group'
    for n,hist in exhaustive_histograms.items():
        assert hist['descents'] == hist['recoils']

    relation_rows = [dict(relation_id=rid,evidence_class='proved_from_definitions',relation_type=kind,
        formula=formula,mathematical_scope='All permutations in S_n, n>=1; lengths known',
        required_tasks=tasks,proof=proof,inputs_checked=relation_counts[rid],
        implementation_checked=True,source='vendor math_ops.py plus independent audit_relations.reference',
        mechanism_claim='Does not establish that a trained model computes or uses this relation')
        for rid,kind,formula,proof,tasks in RELATIONS]
    relation_rows.append(dict(relation_id='descent_recoil_uniform_distribution',evidence_class='proved_from_definitions',
        relation_type='equal_population_marginal_under_explicit_distribution',formula='Law(d(P))=Law(recoils(P)) for P uniform on S_n',
        mathematical_scope='Uniform distribution on all permutations; not arbitrary orbit-filtered finite samples',
        required_tasks='descents;recoils',proof='Inverse is a bijection of S_n and recoils(p)=d(Ip).',
        inputs_checked=5913,implementation_checked=True,source='exact histograms on S_1 through S_7',
        mechanism_claim='Equal answer marginals do not imply equal learning dynamics.'))
    joint_rows = []
    for g,condition in all_groups:
        key = (g['target'],g['background'],condition)
        same = same_witness[key]
        formulas = []
        if g['target'] == 'fixed_points' and condition == 'B':
            formulas.append('F(p)=AF(Cp)')
        if g['target'] == 'longest_decreasing_run' and condition == 'B':
            formulas.append('LDR(p)=LIR(Cp)')
        if g['target'] == 'fixed_points' and g['background'] == 'g2_deficiencies':
            formulas.append('F(p)=n-D(p)-D(Ip)')
        transform_status = 'proven_recoverable' if formulas else ('disproved_by_n10_counterexample' if key in transform_witness else 'not_resolved_by_this_audit')
        if formulas:
            assert key not in transform_witness
        in_confirmation = (g['target']=='fixed_points' and g['background'] in ['g0_v3','g2_deficiencies']) or (g['target']=='longest_decreasing_run' and g['background'] in ['g0_v3','g1_recoils'])
        joint_rows.append(dict(target=key[0],background=key[1],condition=key[2],
            in_frozen_v5_confirmation=in_confirmation,sources=json.dumps(g['sources'][condition]),n=10,
            same_input_four_source_scalar_answers='not_sufficient_proven_by_counterexample',
            same_input_witness_p=same['p'],same_input_witness_q=same['q'],
            transformed_source_query_model='ordered exact answers to all four tasks on each of I/R/C eight transforms',
            transformed_source_query_status=transform_status,available_reconstruction_formulas='; '.join(formulas),
            transformed_query_witness='COUNTEREXAMPLES.csv' if key in transform_witness else '',
            input_known_length=True,target_in_source_set=key[0] in g['sources'][condition],
            training_exposes_joint_answers=False,training_exposes_paired_transform_queries=False,
            limitation='Answer-vector sufficiency is different from computing the target from the full raw input; no claim about arbitrary transformations or learned mechanisms.'))

    write_csv(out/'RELATION_INVENTORY.csv', relation_rows)
    write_csv(out/'TARGET_JOINT_DERIVABILITY.csv', joint_rows)
    write_csv(out/'COUNTEREXAMPLES.csv', list(same_witness.values())+list(transform_witness.values()))
    # These are old exploratory observations, recomputed from the old source table only.
    with (V4/'artifacts/source_learning_table.csv').open(encoding='utf-8-sig',newline='') as stream:
        old = list(csv.DictReader(stream))
    ix = {(r['target'],r['background'],r['condition'],r['seed'],r['task']):r for r in old}
    observation_rows = []
    for task,targets,conditions in [('peaks',['fixed_points','longest_decreasing_run'],['B','C']),
                                     ('longest_increasing_run',['longest_decreasing_run'],['B'])]:
        diffs=[]
        for target,condition,seed in itertools.product(targets,conditions,['17','42','73']):
            before_row=ix[(target,'g0_v3',condition,seed,task)]
            after_row=ix[(target,'g1_recoils',condition,seed,task)]
            delta=100*(float(after_row['multi4800_dev_accuracy'])-float(before_row['multi4800_dev_accuracy']))
            diffs.append(delta)
            observation_rows.append(dict(observation_id=f'{target}_{condition}_s{seed}_{task}',
                evidence_class='observed_only',target=target,condition=condition,seed=seed,source_task=task,
                comparison='g1_recoils-g0_v3',source_dev_accuracy_change_pp=delta,
                evidence_path='experiments/pilot_v4_backgrounds/artifacts/source_learning_table.csv',
                interpretation='Finite-run association; no identity, generalization guarantee, gradient-conflict proof, or target-transfer mediation claim.'))
        assert all(d < 0 for d in diffs)
    write_csv(out/'OBSERVED_ASSOCIATIONS.csv', observation_rows)
    write_csv(out/'UNCHECKED_CLAIMS.csv', [
        dict(claim_id='trained_model_uses_partition',evidence_class='not_checked',claim='Models use F=n-D-DI internally',reason='Requires a separate representation or intervention study; identity alone is insufficient.'),
        dict(claim_id='gradient_conflict_mechanism',evidence_class='not_checked',claim='Gradient conflict causes background effects',reason='No frozen gradient intervention or causal mediation measurement.'),
        dict(claim_id='all_possible_joint_relations',evidence_class='not_checked',claim='All mathematical or algorithmic relations between tasks have been classified',reason='This audit covers listed formulas and specific same-input/eight-transform answer query models only.'),
        dict(claim_id='all_lengths_model_transfer',evidence_class='not_checked',claim='Transfer findings hold for lengths other than n=10',reason='Training and evaluation use n=10; proofs of task identities do not prove model generalization.'),
        dict(claim_id='cross_family_certificates',evidence_class='not_checked',claim='Suggested future task-family relations are implementation-certified here',reason='Future-plan mathematical definitions need new implementation fixtures before experiments.'),
    ])
    assert before == {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(protected))}
    assert 'torch' not in sys.modules
    report=dict(status='passed',generated_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-started,exhaustive_n=list(range(1,8)),exhaustive_inputs=5913,
        random_n=10,random_inputs=4096,random_seed=2026101811,
        random_input_sha256=hashlib.sha256(json.dumps(fixture_n10,separators=(',',':')).encode()).hexdigest(),
        fixture_purpose='Definition-level verification only; never passed to models or used to select a training configuration.',
        task_implementations_independently_compared=20,definition_label_comparisons=definition_checks,
        checks_per_relation=dict(relation_counts),same_input_counterexample_groups=len(same_witness),
        all_eight_transform_counterexample_groups=len(transform_witness),
        mathematical_proof_scope='Proof sketches establish listed universal statements; finite enumeration validates implementations.',
        query_scope='Joint sufficiency from scalar source answers plus n, not from the raw permutation which trivially determines every label.',
        protected_files_unchanged=True,protected_sha256=before,
        code_sha256=sha(Path(__file__)),no_gpu=True,no_training=True,
        new_validation_data_or_results_read=False,
        output_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'MATHEMATICS_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','elapsed_seconds','definition_label_comparisons',
        'same_input_counterexample_groups','all_eight_transform_counterexample_groups']},ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=HERE/'audit')
    run(parser.parse_args().out.resolve())
