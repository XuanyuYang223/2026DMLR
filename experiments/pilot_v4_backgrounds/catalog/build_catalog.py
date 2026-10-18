"""Extend the existing catalog into a reusable, CPU-only v4 data/relations layer.

Reads mathematical code and historical source-only train/dev references, never
transfer outcomes or study test arrays. All writes remain in pilot_v4_backgrounds.
"""
from __future__ import annotations

import csv
from collections import deque
import importlib.util
import itertools
import json
from pathlib import Path
import random
import time

import numpy as np

import data_platform as platform

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent.parent
OLD = ROOT / 'experiments/pilot_v3_multitarget'
spec = importlib.util.spec_from_file_location('v3_catalog_readonly', OLD / 'catalog/build_catalog.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
previous.OUT = HERE
previous.BASE = BASE

TARGETS = ('fixed_points', 'longest_decreasing_run')
ANCHORS = {
    'fixed_points': dict(related='anti_fixed_points', replacement='successions',
                         common=['descents','peaks','left_to_right_maxima']),
    'longest_decreasing_run': dict(related='longest_increasing_run', replacement='left_to_right_maxima',
                                   common=['descents','peaks','successions']),
}
BACKGROUND_EDITS = {
    'g0_v3': None,
    'g1_recoils': ('descents','recoils'),
    'g2_deficiencies': ('peaks','deficiencies'),
}


def js(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def csvread(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def write(path, obj): path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
csvwrite = previous.write_csv
sha = previous.sha


def design():
    groups=[]
    for target, anchor in ANCHORS.items():
        for background, edit in BACKGROUND_EDITS.items():
            common=[edit[1] if edit and task==edit[0] else task for task in anchor['common']]
            sources={c:[anchor[name],*common] for c,name in [('B','related'),('C','replacement')]}
            for values in sources.values():
                assert len(set(values))==4 and not set(values)&set(TARGETS)
            assert sources['B'][1:]==sources['C'][1:]
            groups.append(dict(target=target,background=background,related=anchor['related'],
                replacement=anchor['replacement'],sources=sources,
                candidate_slot=0,changed_slot=None if edit is None else 1+anchor['common'].index(edit[0]),
                old_background_task=None if edit is None else edit[0],
                new_background_task=None if edit is None else edit[1],
                selection_basis='predeclared systematic one-common-task change; exact mathematics and existing source-only single-task train/dev curves',
                difficulty_or_answer_distribution_matched=False,
                transformed_training_pairs=False))
    return groups


def proof_for(row):
    rid=row['relation_id']
    if rid=='descent_inverse': return 'Recoil is defined as the number of descents of the inverse; inversion is involutive.'
    if 'fixed' in rid: return 'Reindex graph points (i,p_i); diagonal/antidiagonal membership is preserved or exchanged by the stated reflection.'
    if 'exceedance' in rid: return 'Put j=p_i; p_i>i iff p_inverse(j)<j, giving a bijection of counted positions.'
    if 'decomposition' in rid: return 'Value complement exchanges a globally descending split with a direct-sum split; components count adds the final block.'
    if 'record' in rid: return 'A record condition is a strict inequality against every preceding/following value; reflection or graph transposition reindexes the same witness points.'
    if 'subsequence' in rid or rid.startswith(('lis_length','lds_length')): return 'Reflect/transposition the permutation graph; increasing/decreasing subsequence chains map bijectively with unchanged cardinality.'
    if 'run' in rid: return 'Value complement or reversal exchanges strictly increasing/decreasing contiguous intervals without changing length.'
    if any(word in rid for word in ['peak','double','rtl_record']): return 'The stated reflection flips all strict inequalities in the local pattern or record condition; eligible positions are in bijection.'
    if 'cycle' in rid: return 'Inversion reverses each disjoint cycle and preserves its length; every named cycle statistic is unchanged.'
    if 'displacement' in rid: return 'Inversion sends graph point (i,p_i) to (p_i,i), preserving the absolute coordinate difference.'
    return 'Exact equality follows by applying the stated bijection to the count/maximum in the registered definition.'


def direct_relations():
    definitions=[]
    for r in previous.relations():
        definitions.append(dict(relation_id=r['relation_id'],left=r['left'],right=r['right'],
            action=dict(identity='000',inverse='100',reverse='010',complement='001')[r['input_transform']],
            input_transform=r['input_transform'],scale=1,n_coefficient=0,constant=r['offset'],
            relation_type=r['relation_type'],proof_scope='all permutations, n>=1; derivation from definitions',
            proof_sketch=proof_for(r),origin=r['origin']))
    for task in ['anti_fixed_points','successions','adjacencies']:
        definitions.append(dict(relation_id=task+'_inverse_self',left=task,right=task,
            action='100',input_transform='inverse',scale=1,n_coefficient=0,constant=0,
            relation_type='self_invariance',proof_scope='all permutations, n>=1; derivation from definitions',
            proof_sketch=('Graph transposition preserves the antidiagonal.' if task=='anti_fixed_points' else
                          'An adjacent consecutive pair is represented by two permutation-graph points with both coordinate differences 1 (or absolute difference 1); graph transposition preserves this condition.'),
            origin='existing_definitions_v4_certificate'))
    for action, name in [('001','complement'),('010','reverse')]:
        definitions.append(dict(relation_id='descent_'+name+'_affine',left='descents',right='descents',
            action=action,input_transform=name,scale=-1,n_coefficient=1,constant=-1,
            relation_type='nonidentity_answer_mapping',proof_scope='all permutations, n>=1; derivation from definitions',
            proof_sketch='Every one of the n-1 adjacent unequal pairs reverses its comparison direction.',
            origin='existing_definitions_v4_certificate'))
    return definitions


def relation_closure(definitions):
    composition=platform.action_composition_table(); edges={t:[] for t in platform.TASK_ORDER}
    for r in definitions:
        edges[r['left']].append(r)
        inverse=dict(r,relation_id=r['relation_id']+'__reverse_direction',left=r['right'],right=r['left'],
            scale=r['scale'],n_coefficient=-r['scale']*r['n_coefficient'],constant=-r['scale']*r['constant'])
        edges[inverse['left']].append(inverse)
    output=[]
    for start in platform.TASK_ORDER:
        first=(start,'000',1,0,0); pending=deque([(first,[])])
        seen={first}
        while pending:
            (task,action,scale,a,b),path=pending.popleft()
            output.append(dict(left=start,right=task,action=action,scale=scale,n_coefficient=a,constant=b,
                formula=f'{task}(T_{action}(p))={scale}*{start}(p)+({a})*n+({b})',
                proof_basis=json.dumps(path),proof_scope='all n>=1 by composition/inversion of the cited definition identities',
                closure_scope='Complete closure of the declared edges under D8 input actions and affine answer-map composition; not a complete catalog of all mathematical relationships'))
            for edge in edges[task]:
                e=edge['scale']; state=(edge['right'],composition[action,edge['action']],e*scale,
                    e*a+edge['n_coefficient'],e*b+edge['constant'])
                if state not in seen:
                    seen.add(state);pending.append((state,path+[edge['relation_id']]))
            assert len(seen)<=len(platform.TASK_ORDER)*8*2
    return output


def audit_graph(graph):
    rng=random.Random(2026102204)
    ps=[p for n in range(1,8) for p in itertools.permutations(range(1,n+1))]
    exact=len(ps);ps += [tuple(rng.sample(range(1,11),10)) for _ in range(2048)]
    for p in ps:
        values={a:dict(zip(platform.TASK_ORDER,platform.labels(platform.transform(p,a)))) for a in platform.ACTIONS}
        for r in graph:
            assert values[r['action']][r['right']]==r['scale']*values['000'][r['left']]+r['n_coefficient']*len(p)+r['constant'],(r,p)
    for r in graph:r.update(exhaustive_n='1..7',random_n=10,checks=len(ps),failures=0)
    return dict(status='passed',exhaustive_inputs=exact,random_inputs=2048,random_n=10,random_seed=2026102204,
                derived_edges=len(graph),total_edge_checks=len(graph)*len(ps))


def aggregates():
    specs=[
        ('position_partition',['fixed_points','exceedances','deficiencies'],'fixed_points(p)+exceedances(p)+deficiencies(p)=n',
         'Every position satisfies exactly one of p_i=i, p_i>i, p_i<i.'),
        ('cycle_fixed_partition',['cycle_count','fixed_points','nontrivial_cycle_count'],'cycle_count=fixed_points+nontrivial_cycle_count',
         'Partition disjoint cycles into length one and length at least two.'),
        ('cycle_parity_partition',['cycle_count','even_cycle_count','odd_cycle_count'],'cycle_count=even_cycle_count+odd_cycle_count',
         'Partition disjoint cycles by length parity.'),
        ('interior_partition',['peaks','valleys','double_ascents','double_descents'],'peaks+valleys+double_ascents+double_descents=max(n-2,0)',
         'The two adjacent comparison signs at every interior position have exactly four possibilities.'),
        ('succession_adjacency',['adjacencies','successions'],'adjacencies(p)=successions(p)+successions(Cp)',
         'An absolute adjacent difference of one is either an upward or downward difference; complement swaps the two.'),
        ('inversion_encodings',['length','to_lehmer','to_inversion_vector','to_reduced_word'],
         'length(p)=sum(Lehmer(p))=sum(inversion_vector(p))=len(reduced_word(p))',
         'The first three count each inversion once; the registered reduced word uses one adjacent swap per inversion.'),
        ('permutation_parity',['parity','cycle_count'],'parity(p)=(n-cycle_count(p)) mod 2',
         'A cycle of length k is a product of k-1 transpositions.'),
    ]
    return [dict(relation_id=i,members=json.dumps(m),input_transform='identity and complement' if i=='succession_adjacency' else 'identity',
                 formula=f,answer_mapping='joint formula; not a one-source identity map',proof_scope='all n>=1 by the stated partition/bijection',
                 proof_sketch=p,checked_inputs=7961,check_reference='RELATION_AUDIT.json aggregate_identity_checks') for i,m,f,p in specs]


def background_references(groups):
    used=sorted({t for g in groups for ts in g['sources'].values() for t in ts})
    index={t:i for i,t in enumerate(platform.TASK_ORDER)}
    references=[];matching=[];histograms=[];source_hashes={};curves={}
    for seed in [17,42,73]:
        pool_path=OLD/'artifacts/data'/f's{seed}_source_train.npz'
        source_hashes[pool_path.relative_to(ROOT).as_posix()]=sha(pool_path)
        with np.load(pool_path,allow_pickle=False) as d:y=d['y'];p=d['p']
        probs={}
        for task in used:
            gold=np.asarray([platform.PROPERTY_FUNCTIONS[task](tuple(map(int,q))) for q in p])
            assert np.array_equal(y[:,index[task]],gold)
            counts=np.bincount(gold,minlength=11);probs[task]=counts/len(gold)
            histograms.append(dict(seed=seed,task=task,length=10,count=len(gold),histogram=json.dumps(counts.tolist()),
                mean=float(gold.mean()),std=float(gold.std()),majority_fraction=float(probs[task].max()),
                entropy_bits=float(-np.sum(probs[task][probs[task]>0]*np.log2(probs[task][probs[task]>0])))))
            path=OLD/'artifacts/single'/f's{seed}_{task}'/'curve.csv'
            source_hashes[path.relative_to(ROOT).as_posix()]=sha(path)
            raw=csvread(path);curves[seed,task]={int(r['step']):float(r['accuracy']) for r in raw if r['split']=='dev'}
            references.extend(dict(**r,source_path=path.relative_to(ROOT).as_posix(),source_sha256=source_hashes[path.relative_to(ROOT).as_posix()]) for r in raw)
        for background,edit in BACKGROUND_EDITS.items():
            if edit is None:continue
            old,new=edit
            matching.append(dict(seed=seed,background=background,removed=old,added=new,length=10,
                answer_tv=float(np.abs(probs[old]-probs[new]).sum()/2),
                removed_dev_1200=curves[seed,old][1200],added_dev_1200=curves[seed,new][1200],
                removed_dev_4800=curves[seed,old][4800],added_dev_4800=curves[seed,new][4800],
                mean_absolute_curve_gap=float(np.mean([abs(curves[seed,old][st]-curves[seed,new][st]) for st in [600,1200,2400,3600,4800]])),
                basis='diagnostic description only; backgrounds predetermined, no score-based selection'))
    csvwrite(HERE/'SOURCE_REFERENCE_CURVES.csv',references)
    csvwrite(HERE/'SOURCE_ANSWER_DISTRIBUTIONS.csv',histograms)
    csvwrite(HERE/'BACKGROUND_MATCHING.csv',matching)
    return dict(source_tasks=used,curve_rows=len(references),source_hashes=source_hashes,transfer_outcomes_read=False)


def main():
    started=time.perf_counter();HERE.mkdir(parents=True,exist_ok=True)
    groups=design();definitions=direct_relations();graph=relation_closure(definitions)
    audit=audit_graph(graph);units=platform.unit_checks();assert units['model_imported'] is False
    # Keep the earlier finite-check corpus for aggregate identities and the two
    # v4 replacement pairs (alongside the other four previously audited pairs).
    old_relations=previous.relations();aggregate_audit=previous.audit(old_relations)
    replacement_audit=previous.candidate_mapping_audit()
    rows=previous.perm_rows(old_relations)+previous.other_rows()
    used={t for g in groups for ts in g['sources'].values() for t in ts}
    for row in rows:
        row['scalar_pilot_eligible']=row.pop('pilot_v3_eligible')
        task=row['task_id'];roles=[]
        if row['world']=='PermWorld':
            if task in TARGETS:roles.append('target')
            if task in {a['related'] for a in ANCHORS.values()}:roles.append('related_source')
            if task in {a['replacement'] for a in ANCHORS.values()}:roles.append('replacement_source')
            if task in {t for g in groups for t in g['sources']['B'][1:]}:roles.append('background_source')
            row['input_range'] += {'power':'; exponent k is an integer >=0','right_multiply_simple':'; simple index 1<=i<n',
                                  'pattern_avoidance':'; pattern is a permutation in S_k',
                                  'compose':'; second permutation in S_n','conjugate':'; two permutations in S_n; first is conjugator',
                                  'commutator':'; second permutation in S_n','bruhat_leq':'; second permutation in S_n'}.get(task,'')
            if task in platform.TASK_ORDER:
                row['generation'] += '; uniform proposals with orbit rejection are not claimed uniform over orbit classes'
        row['pilot_v4_roles']=';'.join(roles) if roles else 'not_used'
        row['relation_graph_reference']='catalog/RELATION_GRAPH.csv and AGGREGATE_RELATIONS.csv'
        row['available_composite_actions']=';'.join(sorted({r['action'] for r in graph if task in [r['left'],r['right']]})) if row['world']=='PermWorld' else ''
    csvwrite(BASE/'TASK_CATALOG.csv',rows)
    csvwrite(HERE/'DIRECT_RELATION_DEFINITIONS.csv',definitions)
    csvwrite(HERE/'RELATION_GRAPH.csv',graph)
    csvwrite(HERE/'AGGREGATE_RELATIONS.csv',aggregates())
    annotated=[]
    for g in groups:
        for condition,tasks in g['sources'].items():
            available={g['target'],*tasks}
            for r in graph:
                if r['left'] in available and r['right'] in available and not(r['action']=='000' and r['left']==r['right']):
                    annotated.append(dict(target=g['target'],background=g['background'],condition=condition,
                        left=r['left'],right=r['right'],relation_role='involves_target' if g['target'] in [r['left'],r['right']] else 'within_sources',
                        action=r['action'],formula=r['formula'],proof_basis=r['proof_basis']))
    csvwrite(HERE/'BACKGROUND_RELATION_GRAPH.csv',annotated)
    aggregate_applicability=[]
    for g in groups:
        for condition,tasks in g['sources'].items():
            available={g['target'],*tasks}
            for r in aggregates():
                members=set(json.loads(r['members']))
                present=members&available
                if present:
                    aggregate_applicability.append(dict(target=g['target'],background=g['background'],condition=condition,
                        relation_id=r['relation_id'],formula=r['formula'],present_members=';'.join(sorted(present)),
                        missing_members=';'.join(sorted(members-available)),all_named_members_available=members<=available,
                        can_claim_target_determined_by_present_source_answers=False,
                        interpretation='Membership audit only. A joint identity is not an available one-source answer map; derivability needs all arguments and separate algebra verification.'))
    csvwrite(HERE/'BACKGROUND_AGGREGATE_RELATIONS.csv',aggregate_applicability)
    references=background_references(groups)
    write(HERE/'BACKGROUND_DESIGN.json',dict(status='mathematics_and_source_reference_reviewed',targets=TARGETS,groups=groups,
        rationale={
            'g0_v3':'Historical anchor with exact original ordered source slots.',
            'g1_recoils':'Replace descents by its inverse counterpart; preserve population Eulerian marginal, change one-line computational structure and observed learning trajectory. Descents is removed, so this does not add a co-present descents/recoils pair.',
            'g2_deficiencies':'Replace local three-position peak count by positional inequality count; comparable late single-task mastery, different early dynamics, output range and distribution.'},
        global_target_exclusion=True,slot0_candidate_fixed=True,only_one_background_slot_changed=True,
        claims_not_made=['difficulty matched','answer distributions matched for g2','all other mathematical relations equal','mathematical unrelatedness of C','transformed input pairs used in pretraining'],
        no_v4_transfer_outcomes_used=True))
    write(HERE/'RELATION_AUDIT.json',dict(**aggregate_audit,derived_graph=audit,unit_checks=units,replacement_relation_audit=replacement_audit))
    write(HERE/'SOURCE_REFERENCE_AUDIT.json',references)
    sources=[Path(__file__),HERE/'data_platform.py',OLD/'catalog/build_catalog.py',
             ROOT/'experiments/permworld_pilot/vendor/neurips_permutations/math_ops.py',
             ROOT/'experiments/permworld_pilot/vendor/neurips_permutations/passage.py']
    files=[BASE/'TASK_CATALOG.csv',*sorted(HERE.glob('*.csv')),HERE/'BACKGROUND_DESIGN.json',HERE/'RELATION_AUDIT.json',HERE/'SOURCE_REFERENCE_AUDIT.json']
    result=dict(status='passed',catalog_rows=len(rows),permworld_registered_endpoints=48,permworld_scalar_property32=32,
        graph_edges=len(graph),direct_definition_edges=len(definitions),group_count=len(groups),source_tasks=sorted(used),
        historical_transfer_results_read=False,v4_transfer_results_read=False,neural_calls=0,
        catalog_sha256=sha(BASE/'TASK_CATALOG.csv'),source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        output_sha256={p.relative_to(BASE).as_posix():sha(p) for p in files},seconds=time.perf_counter()-started,
        proof_caveat='Declared identities are definition-derived; finite audits check implementation. D8 closure is not all mathematics and does not imply transferable representations.')
    write(HERE/'CATALOG_AUDIT.json',result)
    print(json.dumps({k:result[k] for k in ['status','catalog_rows','graph_edges','direct_definition_edges','seconds']}),flush=True)


if __name__=='__main__':main()
