"""CPU-only inventory of the task registries already present in this workspace.

No neural model, checkpoint, transfer result, or test input is read. The audit
checks exact identities on all S_n, 1 <= n <= 7, plus deterministic random n=10.
Run: .venv/Scripts/python.exe experiments/pilot_v3_multitarget/catalog/build_catalog.py
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import random
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = OUT.parent
sys.path.insert(0, str(ROOT / 'experiments/permworld_pilot/vendor'))
from neurips_permutations import math_ops as m
from neurips_permutations.passage import TASK_SPECS, REPRESENTATIONS


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform(p, action):
    if action == 'identity': return p
    if action == 'inverse': return m.inverse(p)
    if action == 'reverse': return tuple(reversed(p))
    if action == 'complement': return tuple(len(p) + 1 - v for v in p)
    raise ValueError(action)


def relations():
    registry = ROOT / 'reference/neurips/configs/property_task_geometry.toml'
    original = tomllib.loads(registry.read_text(encoding='utf-8'))['relation_pairs']
    out = [dict(relation_id=r['pair_id'], left=r['left'], right=r['right'],
                input_transform=r['input_transform'], answer_mapping=f"y+{r['right_label_offset']}",
                offset=r['right_label_offset'], origin='registered8', relation_type='cross_task_exact')
           for r in original]
    for tag, left, right, action in [
        ('fixed_reverse','fixed_points','anti_fixed_points','reverse'),
        ('subsequence_reverse','lis_length','lds_length','reverse'),
        ('run_reverse','longest_increasing_run','longest_decreasing_run','reverse'),
        ('record_max_reverse','left_to_right_maxima','right_to_left_maxima','reverse'),
        ('record_min_reverse','left_to_right_minima','right_to_left_minima','reverse'),
        ('record_inverse','left_to_right_maxima','right_to_left_minima','inverse'),
        ('double_complement','double_ascents','double_descents','complement'),
        ('double_reverse','double_ascents','double_descents','reverse'),
        ('rtl_record_complement','right_to_left_maxima','right_to_left_minima','complement')]:
        out.append(dict(relation_id=tag,left=left,right=right,input_transform=action,
                        answer_mapping='y+0',offset=0,origin='existing_definitions',relation_type='cross_task_exact'))
    for task in ['fixed_points','lis_length','lds_length','cycle_count','two_cycle_count',
                 'three_cycle_count','even_cycle_count','odd_cycle_count','longest_cycle',
                 'shortest_cycle','nontrivial_cycle_count','max_displacement','displacement_one_count']:
        out.append(dict(relation_id=task+'_inverse_self',left=task,right=task,
                        input_transform='inverse',answer_mapping='y+0',offset=0,
                        origin='existing_definitions',relation_type='self_invariance'))
    return out


FUNCTIONS = dict(m.PROPERTY_FUNCTIONS)
FUNCTIONS.update(to_cycle=m.canonical_cycles,to_lehmer=m.lehmer_code,
                 to_inversion_vector=m.inversion_vector,to_reduced_word=m.reduced_coxeter_word,
                 length=m.inversion_count,parity=m.parity,cycle_type=m.cycle_type,
                 rsk_shape=m.rsk_shape,pattern_avoidance=m.avoids_pattern,
                 inverse=m.inverse,compose=m.compose,power=m.power,
                 conjugate=m.conjugate,commutator=m.commutator,
                 right_multiply_simple=m.right_multiply_simple,bruhat_leq=m.bruhat_leq)

RANGES = {
    'to_cycle':'ordered disjoint cycles; every value 1..n exactly once',
    'to_lehmer':'length n vector; 0 <= L_i <= n-i',
    'to_inversion_vector':'value-indexed length n vector; 0 <= I_v <= n-v',
    'to_reduced_word':'sequence over 1..n-1 of length inv(p)',
    'length':'0..n(n-1)/2','parity':'{0,1}',
    'cycle_type':'integer partition of n in descending order',
    'rsk_shape':'integer partition of n in descending order',
    'pattern_avoidance':'{0,1}','bruhat_leq':'{0,1}',
    'descents':'0..n-1','recoils':'0..n-1','peaks':'0..floor((n-1)/2)',
    'valleys':'0..floor((n-1)/2)','double_ascents':'0..max(n-2,0)',
    'double_descents':'0..max(n-2,0)','successions':'0..n-1',
    'adjacencies':'0..n-1','fixed_points':'{0,..,n-2,n}; n=1 gives {1}',
    'anti_fixed_points':'{0,..,n-2,n}; n=1 gives {1}',
    'exceedances':'0..n-1','deficiencies':'0..n-1',
    'left_to_right_maxima':'1..n','left_to_right_minima':'1..n',
    'right_to_left_maxima':'1..n','right_to_left_minima':'1..n',
    'cycle_count':'1..n','two_cycle_count':'0..floor(n/2)',
    'three_cycle_count':'0..floor(n/3)','even_cycle_count':'0..floor(n/2)',
    'odd_cycle_count':'subset of 0..n; number of odd cycles has parity n mod 2',
    'longest_cycle':'1..n','shortest_cycle':'subset of 1..n',
    'nontrivial_cycle_count':'0..floor(n/2)','lis_length':'1..n','lds_length':'1..n',
    'longest_increasing_run':'1..n','longest_decreasing_run':'1..n',
    'global_descents':'0..n-1','components':'1..n','max_displacement':'0..n-1',
    'displacement_one_count':'subset of 0..n',
}

ADDITIONAL = {
    'length':'inv(p)=sum(Lehmer(p))=sum(inversion_vector(p))=len(reduced_word(p)); parity=inv mod 2; inv(Cp)=n(n-1)/2-inv(p)',
    'parity':'parity(p)=(n-cycle_count(p)) mod 2; parity(p inverse)=parity(p)',
    'cycle_type':'determines all cycle-length statistics; invariant under inverse and conjugation',
    'rsk_shape':'first row=LIS; number of rows=LDS; transpose shape under value complement',
    'fixed_points':'cycle_count(p)=fixed_points(p)+nontrivial_cycle_count(p); exceedances+deficiencies+fixed_points=n',
    'exceedances':'exceedances+deficiencies+fixed_points=n; same marginal Eulerian distribution as descents is not a pointwise identity',
    'deficiencies':'exceedances+deficiencies+fixed_points=n',
    'descents':'descents(Cp)=n-1-descents(p); ascents is not a registered standalone task',
    'peaks':'peaks+valleys+double_ascents+double_descents=max(n-2,0)',
    'valleys':'peaks+valleys+double_ascents+double_descents=max(n-2,0)',
    'double_ascents':'double_ascents is contiguous increasing triples, not the number of ascents; four triple types partition interior positions',
    'double_descents':'four triple types partition interior positions',
    'successions':'adjacencies(p)=successions(p)+successions(Cp); successions and adjacencies are not an exact identity-mapped dual pair',
    'adjacencies':'adjacencies(p)=successions(p)+successions(Cp); adjacency count invariant under complement and reverse',
    'cycle_count':'cycle_count=fixed_points+nontrivial_cycle_count=even_cycle_count+odd_cycle_count; these require joint source answers',
    'two_cycle_count':'cycle_length_count(p,k=2); same parameterized family as three_cycle_count; k=1 equals fixed_points',
    'three_cycle_count':'cycle_length_count(p,k=3); same parameterized family as two_cycle_count; k=1 equals fixed_points',
    'even_cycle_count':'cycle_count=even_cycle_count+odd_cycle_count; no identity-mapped cross-task swap asserted',
    'odd_cycle_count':'cycle_count=even_cycle_count+odd_cycle_count; no identity-mapped cross-task swap asserted',
    'nontrivial_cycle_count':'cycle_count=fixed_points+nontrivial_cycle_count; not an identity-mapped dual pair',
    'max_displacement':'no simple direct map from displacement_one_count is registered; shared displacement definition is not a proven duality',
    'displacement_one_count':'no simple direct map from max_displacement is registered; shared displacement definition is not a proven duality',
    'inverse':'involutive; preserves cycle type; descents(inverse(p))=recoils(p)',
    'compose':'compose(a,b)=a o b; inverse(a o b)=inverse(b) o inverse(a)',
    'power':'power(p,k) composes k copies; k=0 identity; k is a parameter, not a new task',
    'conjugate':'conjugate(g,p)=g o p o inverse(g); preserves cycle type',
    'commutator':'commutator(p,q)=p o q o inverse(p) o inverse(q)',
    'right_multiply_simple':'p o s_i swaps positions i and i+1; i is a parameter',
    'bruhat_leq':'strong Bruhat order is preserved by inversion; not weak order',
    'pattern_avoidance':'pattern argument is a parameter; avoidance is complement of pattern containment',
}


def perm_rows(rs):
    rows=[]
    names=list(TASK_SPECS)
    assert set(names)==set(FUNCTIONS)
    for task in names:
        fn=FUNCTIONS[task]; spec=TASK_SPECS[task]
        rel=[r for r in rs if task in (r['left'],r['right'])]
        formula=[f"{r['right']}({r['input_transform']}(p))={r['left']}(p)+{r['offset']}" for r in rel]
        if task in ADDITIONAL: formula.append(ADDITIONAL[task])
        kind='mathematical_task'; family=task; param='n (input length; not a new task)'
        if task.startswith('to_'):
            kind='output_encoding_variant'; family='permutation_identity'; param+='; output representation'
        if task in ('two_cycle_count','three_cycle_count'):
            kind='parameter_variant'; family='cycle_length_count'; param+='; k='+('2' if task=='two_cycle_count' else '3')
        if spec.operand_kind:
            param+='; '+str(spec.operand_kind)+' supplied as part of input'
        generator='uniform random permutation; exact deterministic function label; reject occupied transformation orbits for pilot partitions'
        if task in ('pattern_avoidance','bruhat_leq'):
            generator='upstream v3 balances labels by constructive sampling; v2 differs; not used in scalar pilot'
        elif spec.operand_kind:
            generator='upstream deterministic record generator; random permutation plus typed operand/parameter; not used in scalar pilot'
        rows.append(dict(world='PermWorld',task_id=task,identity_kind=kind,
            mathematical_family=family,definition=' '.join(inspect.getdoc(fn).split()),
            input_range='p in S_n, values 1..n, n>=1'+('; '+str(spec.operand_kind) if spec.operand_kind else ''),
            output_range=RANGES.get(task,'S_n'),parameter_variants=param,
            input_encoding_variants='one_line; canonical_cycle; upstream also lehmer and inversion_vector (same mathematical object)',
            output_encoding=spec.answer_kind,task_token=spec.token,generation=generator,
            known_relations=' | '.join(formula) or 'No specific direct cross-task relation annotated here; this does not assert mathematical independence',
            input_transforms=';'.join(sorted({r['input_transform'] for r in rel})),
            answer_mappings=';'.join(sorted({r['answer_mapping'] for r in rel})),
            registry_membership=';'.join(name for name,group in [('v2_20',m.V2_TASK_NAMES),('v3_20',m.V3_TASK_NAMES),('property32',m.PROPERTY32_TASK_NAMES)] if task in group),
            pilot_v3_eligible=task in m.PROPERTY32_TASK_NAMES,
            source=f"experiments/permworld_pilot/vendor/neurips_permutations/math_ops.py:{inspect.getsourcelines(fn)[1]}; reference/neurips/src/neurips_permutations/passage.py",
            caveat='Range assumes n>=1; intervals may be outer bounds, not full attainable support; registration/natural pairing alone is not a relationship certificate'))
    return rows


def other_rows():
    rows=[]
    def add(world,task,family,definition,inputs,outputs,params,encoding,generation,relations,actions,mappings,source,kind='mathematical_task'):
        rows.append(dict(world=world,task_id=task,identity_kind=kind,mathematical_family=family,
            definition=definition,input_range=inputs,output_range=outputs,parameter_variants=params,
            input_encoding_variants=encoding,output_encoding='scalar',task_token='see source; token reuse across worlds does not identify tasks',
            generation=generation,known_relations=relations,input_transforms=actions,answer_mappings=mappings,
            registry_membership='2026TMLR current source tree',pilot_v3_eligible=False,source=source,
            caveat='Catalog only; no new cross-domain experiment and not counted as PermWorld tasks'))
    graphsource='reference/2026TMLR/experiments/graphworld_pilot_20261010/core.py'
    for task,definition in [('max_out','max_i sum_j A_ij'),('min_out','min_i sum_j A_ij'),
                            ('max_in','max_j sum_i A_ij'),('min_in','min_j sum_i A_ij')]:
        add('GraphWorld',task,task,definition,'directed simple loopless graph, adjacency A in {0,1}^{n x n}','0..n-1',
            'n=6,7,8; edge probability=.25,.5,.75 (not new tasks)',
            'row-major adjacency excluding diagonal; same existing numeric vocabulary',
            'independent Bernoulli directed edges, zero diagonal; reject occupied <R,C,W> orbits',
            'max_in(A^T)=max_out(A); min_in(A^T)=min_out(A); min_out(CA)=n-1-max_out(A); min_in(CA)=n-1-max_in(A)',
            'R: transpose; C: complement off diagonal','identity; n-1-y',graphsource)
    for task,definition in [('F1','integer Hamming weight of column 1'),('F2','integer Hamming weight of row 1'),
                            ('F3','integer Hamming weight of column1 XOR column2'),('F4','integer Hamming weight of row1 XOR row2')]:
        add('MatrixGroupWorld',task,task,definition,'A in GL(n,F2)','1..n','n=6,8,10 (not new tasks)',
            'row-major binary matrix with row delimiters',
            'iid uniform binary matrices, reject singular, then reject occupied 72-candidate LAR/LA^TR orbits',
            'F2(A^T)=F1(A); F4(A^T)=F3(A); F3(AQ)=F1(A); F4(Q^T A)=F2(A), Q=I+E_21',
            'transpose; right_Q; left_Qt','identity',
            'reference/2026TMLR/experiments/matrixgroupworld_20261011/core.py')
    add('FiniteFieldLegacy','linear_form','finite_field_linear_form',
        'f_a(x)=a^T z mod p; z is an invertible linear change of input coordinates; source arrays use 3 or 4 latent coordinates',
        'x in F_p^d, odd prime p, d>=3','0..p-1',
        'coefficient a; field p; dimension d; basis/world seed; scalar-equivalent coefficients grouped projectively',
        'coordinate one-hot vectors','enumerate complete F_p^d; invertible random basis',
        'linear combinations of coefficient vectors give answer combinations modulo p; f_(c a)=c f_a mod p',
        'invertible linear input actions; field_symmetry swaps latent coordinates 3 and 4',
        'linear combinations mod p; nonzero scalar output relabeling',
        'reference/2026TMLR/experiments/algebra.py; reference/2026TMLR/experiments/field_symmetry.py',kind='parameterized_family_template')
    return rows


def audit(rs):
    started=time.perf_counter(); rng=random.Random(2026101803)
    perms=[p for n in range(1,8) for p in itertools.permutations(range(1,n+1))]
    exact_count=len(perms)
    perms += [tuple(rng.sample(range(1,11),10)) for _ in range(2048)]
    stats={r['relation_id']:dict(checks=0,failures=0) for r in rs}
    aggregates=dict(four_triples=0,cycle_sum=0,positional_sum=0,succession_adjacency=0,
                    descents_complement=0,inversion_encodings=0,cycle_parity=0)
    for p in perms:
        values={a:{t:int(f(transform(p,a))) for t,f in m.PROPERTY_FUNCTIONS.items()}
                for a in ('identity','inverse','reverse','complement')}
        for r in rs:
            st=stats[r['relation_id']]; st['checks']+=1
            ok=values[r['input_transform']][r['right']]==values['identity'][r['left']]+r['offset']
            st['failures']+=not ok
            assert ok,(r,p)
        v=values['identity']; n=len(p)
        assert sum(v[t] for t in ('peaks','valleys','double_ascents','double_descents'))==max(n-2,0)
        assert v['cycle_count']==v['fixed_points']+v['nontrivial_cycle_count']==v['even_cycle_count']+v['odd_cycle_count']
        assert v['fixed_points']+v['exceedances']+v['deficiencies']==n
        assert v['adjacencies']==v['successions']+values['complement']['successions']
        assert values['complement']['descents']==n-1-v['descents']
        assert m.inversion_count(p)==sum(m.lehmer_code(p))==sum(m.inversion_vector(p))==len(m.reduced_coxeter_word(p))
        assert m.parity(p)==(n-v['cycle_count'])%2
        for key in aggregates: aggregates[key]+=1
    audited=[]
    for r in rs:
        audited.append({**r,'formula':f"{r['right']}({r['input_transform']}(p)) = {r['left']}(p) + {r['offset']}",**stats[r['relation_id']]})
    write_csv(OUT/'RELATION_AUDIT.csv',audited)
    return dict(status='passed',exhaustive_n=list(range(1,8)),exhaustive_inputs=exact_count,
                random_n=10,random_inputs=2048,random_seed=2026101803,
                cross_task_relations=sum(r['relation_type']=='cross_task_exact' for r in rs),
                self_invariances=sum(r['relation_type']=='self_invariance' for r in rs),
                aggregate_identity_checks=aggregates,seconds=time.perf_counter()-started,
                proof_limit='Finite checks validate implementations; exact identities follow from definitions, not statistical fits',
                neural_results_read=False,neural_calls=0)


def candidate_mapping_audit():
    """Find counterexamples to any answer-only map under each D8 action.

    Two inputs with the same source answer but different transformed target
    answers disprove a deterministic map for that fixed action and length.
    This says nothing about more complicated mathematics or transforms.
    """
    candidates={
        'fixed_points':['successions'],
        'exceedances':['recoils'],
        'left_to_right_minima':['cycle_count','longest_increasing_run'],
        'longest_decreasing_run':['left_to_right_maxima','cycle_count'],
    }
    def act(p, name):
        for flag,op in zip(name,('inverse','reverse','complement')):
            if flag=='1': p=transform(p,op)
        return p
    actions=[''.join(bits) for bits in itertools.product('01',repeat=3)]
    rows=[]
    # At n=7, this disproves even arbitrary maps, rather than just y->y.
    ps=list(itertools.permutations(range(1,8)))
    for target,choices in candidates.items():
        for source in choices:
            for action in actions:
                seen={}; witness=None; equal=0
                for p in ps:
                    sy=int(m.PROPERTY_FUNCTIONS[source](p))
                    ty=int(m.PROPERTY_FUNCTIONS[target](act(p,action)))
                    equal+=sy==ty
                    if sy in seen and seen[sy][1]!=ty and witness is None:
                        witness=(seen[sy][0],p,sy,seen[sy][1],ty)
                    seen.setdefault(sy,(p,ty))
                assert witness is not None,(target,source,action)
                p,q,sy,py,qy=witness
                rows.append(dict(target=target,replacement_candidate=source,length=7,
                    action_bits_I_R_C=action,input_action='apply inverse then reverse then complement where bit=1',
                    inputs_checked=len(ps),identity_answer_fraction=equal/len(ps),
                    arbitrary_source_answer_to_transformed_target_map=False,
                    witness_input_1=json.dumps(p),witness_input_2=json.dumps(q),
                    common_source_answer=sy,target_answer_1=py,target_answer_2=qy,
                    interpretation='Only excludes answer-only maps under these eight actions; not mathematically unrelated'))
    write_csv(OUT/'REPLACEMENT_RELATION_COUNTEREXAMPLES.csv',rows)
    return dict(candidate_pairs=sum(map(len,candidates.values())),actions=8,length=7,
                counterexamples=len(rows),no_neural_or_test_results_used=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rs=relations(); rows=perm_rows(rs)+other_rows()
    write_csv(BASE/'TASK_CATALOG.csv',rows)
    result=audit(rs)
    result['replacement_relation_audit']=candidate_mapping_audit()
    sources=[ROOT/'experiments/permworld_pilot/vendor/neurips_permutations/math_ops.py',
             ROOT/'experiments/permworld_pilot/vendor/neurips_permutations/passage.py',
             ROOT/'reference/neurips/src/neurips_permutations/math_ops.py',
             ROOT/'reference/2026TMLR/external/neurips/src/neurips_permutations/math_ops.py',
             ROOT/'reference/neurips/configs/property_task_geometry.toml',
             ROOT/'reference/2026TMLR/experiments/permworld_encoding/core.py',
             ROOT/'reference/2026TMLR/experiments/graphworld_pilot_20261010/core.py',
             ROOT/'reference/2026TMLR/experiments/matrixgroupworld_20261011/core.py',
             ROOT/'reference/2026TMLR/experiments/algebra.py',
             ROOT/'reference/2026TMLR/experiments/field_symmetry.py', Path(__file__)]
    result.update(catalog_rows=len(rows),permworld_registered_endpoints=len(TASK_SPECS),
                  permworld_scalar_property32=len(m.PROPERTY32_TASK_NAMES),
                  permworld_encoding_conversion_endpoints=4,
                  permworld_input_representations=list(REPRESENTATIONS),
                  world_rows={w:sum(r['world']==w for r in rows) for w in sorted({r['world'] for r in rows})},
                  catalog_sha256=sha(BASE/'TASK_CATALOG.csv'),
                  source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sources})
    (OUT/'CATALOG_AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','catalog_rows','permworld_registered_endpoints','cross_task_relations','seconds')}),flush=True)


if __name__=='__main__': main()
