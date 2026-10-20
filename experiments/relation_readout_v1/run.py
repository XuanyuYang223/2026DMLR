"""Finite, inference-only registered relation readout experiment."""
import sys, json, time, subprocess, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments.task_count_v2 import train as old, design, data_pipeline as dp
from permworld_platform import calibration as platform
from permworld_platform.encoding import encode_example, decode_answer_ids
from permworld_platform.registry import label
from permworld_platform.objects import canonical_orbit_key
from permworld_platform.relations import list_relations
from experiments.relation_readout_v1.executor import query_inputs, apply_answers
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'experiments/task_count_v2'
RIDS = ['position_partition', 'fixed_from_two_deficiency_queries', 'fixed_complement', 'lis_length_complement']
dump = platform.write_json
sha = platform.sha


def freeze():
    assert not (OUT/'FROZEN_PROTOCOL.json').exists()
    protected = {str(p.relative_to(ROOT)): sha(p) for directory in ['experiments/task_count_v2', 'publication'] for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    dump(OUT/'PRESERVATION_BEFORE.json', protected)
    selections = platform.read_json(BASE/'FROZEN_SELECTIONS.json')['selections']
    edges = [next(e for e in list_relations() if e['relation_id']==rid) for rid in RIDS]
    dump(OUT/'routes.json', edges)
    inventory=[]
    for stage, rep, chain, k, budget, spec in old.formal_runs():
        rid = old.pretrain_id(rep,chain,k,budget) if stage=='pretraining' else old.adaptation_id(rep,chain,k,budget,spec)
        complete=platform.read_json(BASE/stage/rid/'COMPLETE.json')
        selection=next((s for s in selections if s['run_id']==rid), None)
        name='endpoint.pt' if stage=='pretraining' else 'best.pt'
        path=BASE/stage/rid/name
        expected=complete['artifact_sha256'][name]
        actual=sha(path) if path.exists() else None
        if selection: assert expected==selection['selected_checkpoint_sha256']
        sources=design.CHAINS[chain][:k] if chain!='baseline' else []
        inventory.append(dict(run_id=rid,stage=stage,seed=rep['initialization_seed'],chain=chain,k=k,budget=budget,target=spec['task_id'] if spec else None,path=str(path.relative_to(ROOT)),expected_sha256=expected,actual_sha256=actual,valid=actual==expected,sources=sources,selection=selection,completion=complete))
    assert len(inventory)==78
    dump(OUT/'checkpoints.json', inventory)
    availability=[]
    for cp in inventory:
        if cp['stage']!='pretraining': continue
        for edge in edges:
            available=all({'task_id':q['task_id'],'params':q['params']} in cp['sources'] for q in edge['queries'])
            availability.append(dict(run_id=cp['run_id'],route=edge['relation_id'],status='available' if available and cp['valid'] else 'NA',reason='' if available and cp['valid'] else 'missing source exposure or invalid checkpoint'))
    dump(OUT/'availability.json',availability)
    protocol=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),created_utc=old.utc(),routes_sha256=sha(OUT/'routes.json'),checkpoints_sha256=sha(OUT/'checkpoints.json'),seeds=[2026102001,2026102002,2026102003],lengths=[10,12],count_per_length=512,formal_gpu_wall_limit_seconds=7200,batch_size=64,statistics='Three original paired training repeats; per-repeat values, mean and sample SD (ddof=1). No input-level inference or route selection.',diagnostics='Complement -> identity; inverse -> identity; E+D -> D from cyclic successor index within length (fixed derangement).',failure_policy='No clipping, modulo, imputation or voting; missing required pretrained source is NA.',precision=old.CFG['precision'],max_new_tokens=old.CFG['max_new_tokens'],metrics=['accuracy','paired_delta_B','query_accuracy','all_sources_correct','error_cancellation','illegal','eos_failure','out_of_bounds','diagnostic_gap','queries','actual_forward_calls','seconds'],code_sha256={p.name:sha(p) for p in OUT.glob('*.py')})
    dump(OUT/'FROZEN_PROTOCOL.json',protocol)
    occupied,bindings=dp._history()
    for manifest in ['manifest.json','test_manifest.json']:
        m=platform.read_json(BASE/'data'/manifest)
        bindings[str((BASE/'data'/manifest).relative_to(ROOT))]=sha(BASE/'data'/manifest)
        for record in m['pools']:
            path=BASE/'data'/record['path']; assert sha(path)==record['sha256']
            bindings[str(path.relative_to(ROOT))]=sha(path)
            for p in dp._pool_array(BASE/'data',record): occupied.add(canonical_orbit_key(tuple(map(int,p))))
    pools=[]
    for seed,rngseed in zip([4103,5207,6301],protocol['seeds']):
        for n in [10,12]:
            rng=np.random.default_rng(np.random.SeedSequence([rngseed,n])); accepted=[]; proposals=0
            while len(accepted)<512:
                proposals+=1
                if proposals>2000000: raise RuntimeError('Proposal cap')
                p=tuple(map(int,rng.permutation(n)+1)); key=canonical_orbit_key(p)
                if key in occupied: continue
                occupied.add(key); accepted.append(p)
            path=OUT/f'data/s{seed}_n{n}.npz'; path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,p=np.array(accepted,dtype=np.int16))
            pools.append(dict(seed=seed,n=n,path=str(path.relative_to(OUT)),sha256=sha(path),proposals=proposals))
    dump(OUT/'data_manifest.json',dict(pools=pools,historical_bindings=bindings,protocol_sha256=sha(OUT/'FROZEN_PROTOCOL.json')))
    oracle=[]
    for pool in pools:
        ps=np.load(OUT/pool['path'])['p']
        for edge in edges:
            correct=wrong=0
            for i,p in enumerate(ps):
                p=tuple(map(int,p)); queries=query_inputs(edge,p)
                a=[label(t,q,params=params or None) for t,q,params in queries]
                target=label(edge['target']['task_id'],p)
                assert apply_answers(edge,pool['n'],a)==target
                correct+=1
                if edge['relation_id']=='position_partition':
                    a[1]=label('deficiencies',tuple(map(int,ps[(i+1)%len(ps)])))
                else:
                    t,_,params=queries[-1]; a[-1]=label(t,p,params=params or None)
                wrong+=apply_answers(edge,pool['n'],a)==target
            oracle.append(dict(seed=pool['seed'],n=pool['n'],route=edge['relation_id'],correct_oracle_accuracy=correct/512,wrong_oracle_accuracy=wrong/512))
    platform.write_csv(OUT/'oracle_checks.csv',oracle)
    print('Frozen; all correct oracle routes passed',flush=True)


class Provider:
    def __init__(self, model): self.model=model; self.calls=0
    def predict(self, task, ps, params):
        records=[encode_example(task,p,params,max_seq_len=old.CFG['model']['max_seq_len']) for p in ps]
        result=[]; started=time.monotonic(); calls=self.calls
        def hook(*args): self.calls+=1
        handle=self.model.register_forward_hook(hook)
        for start in range(0,len(records),64):
            result.extend(platform.greedy_answers(self.model,records[start:start+64],'cuda'))
        torch.cuda.synchronize(); handle.remove()
        elapsed=time.monotonic()-started
        rows=[]
        for record,ids in zip(records,result):
            eos=ids[-1:]==[platform.token_id('<EOS>')]
            try:
                value=decode_answer_ids(ids)
                legal=type(value) is int and 0<=value<=len(ps[0]) and (task not in ['lis_length','lds_length'] or value>=1)
            except (ValueError,TypeError): value=None; legal=False
            rows.append(dict(ids=ids,value=value if legal else None,correct=ids==record['answer_ids'],illegal=not legal,eos_failure=not eos))
        return rows,dict(task=task,count=len(ps),seconds=elapsed,actual_forward_calls=self.calls-calls)


def evaluate():
    protocol=platform.read_json(OUT/'FROZEN_PROTOCOL.json')
    for name,h in protocol['code_sha256'].items(): assert sha(OUT/name)==h
    inventory=platform.read_json(OUT/'checkpoints.json'); edges=platform.read_json(OUT/'routes.json')
    availability=platform.read_json(OUT/'availability.json'); manifest=platform.read_json(OUT/'data_manifest.json')
    device=platform.runtime(); model=platform.make_model(4103,device); model.eval()
    first=next(c for c in inventory if c['valid'])
    model.load_state_dict(platform.load_checkpoint(ROOT/first['path'])['model'])
    smoke_ps=[tuple(map(int,p)) for p in np.load(OUT/manifest['pools'][0]['path'])['p'][:16]]
    smoke, timing=Provider(model).predict('lds_length',smoke_ps,{})
    dump(OUT/'SMOKE.json',dict(timing=timing,count=len(smoke),purpose='Pipeline and throughput only; no route, checkpoint or budget changes.'))
    start=time.monotonic(); raw=[]; timings=[]; cache={}; completed=[]
    for cp in inventory:
        if not cp['valid']: continue
        if time.monotonic()-start>=7200: break
        assert sha(ROOT/cp['path'])==cp['expected_sha256']
        checkpoint=platform.load_checkpoint(ROOT/cp['path'])
        if 'state_sha256' in checkpoint: assert platform.state_hash(checkpoint['model'])==checkpoint['state_sha256']
        model.load_state_dict(checkpoint['model'],strict=True); model.eval(); provider=Provider(model)
        for pool in [x for x in manifest['pools'] if x['seed']==cp['seed']]:
            ps=[tuple(map(int,p)) for p in np.load(OUT/pool['path'])['p']]; n=pool['n']; local={}
            def predict(task, qs, params):
                key=(task,tuple(qs),json.dumps(params,sort_keys=True))
                if key not in local:
                    if time.monotonic()-start>=7200: raise TimeoutError('Frozen GPU wall budget exhausted')
                    local[key],timing=provider.predict(task,qs,params)
                    timings.append(dict(run_id=cp['run_id'],n=n,**timing))
                    with (OUT/'source_predictions.jsonl').open('a',encoding='utf8') as f:
                        for i,(q,row) in enumerate(zip(qs,local[key])): f.write(json.dumps(dict(run_id=cp['run_id'],n=n,task=task,p=q,params=params,**row))+'\n')
                return local[key]
            if cp['stage']=='adaptation':
                predictions=predict(cp['target'],ps,{})
                cache[(cp['run_id'],n)]=predictions
                for i,row in enumerate(predictions): raw.append(dict(run_id=cp['run_id'],seed=cp['seed'],chain=cp['chain'],k=cp['k'],budget=cp['budget'],target=cp['target'],n=n,index=i,route='direct',method='C' if cp['chain']=='baseline' else 'B',diagnostic=False,correct=row['correct'],all_sources_correct=row['correct'],error_cancellation=False,illegal=row['illegal'],eos_failure=row['eos_failure'],out_of_bounds=False,query_correct=[row['correct']],reconstructed=row['value'],queries=1))
            for edge in edges:
                if cp['chain']=='baseline': continue
                if cp['target'] and cp['target']!=edge['target']['task_id']: continue
                parent=old.pretrain_id({'initialization_seed':cp['seed']},cp['chain'],cp['k'],cp['budget'])
                if not next(a for a in availability if a['run_id']==parent and a['route']==edge['relation_id'])['status']=='available': continue
                for diagnostic in [False,True]:
                    query_rows=[]
                    for j,query in enumerate(edge['queries']):
                        qs=[query_inputs(edge,p)[j][1] for p in ps]
                        if diagnostic and j==len(edge['queries'])-1:
                            qs=ps[1:]+ps[:1] if edge['relation_id']=='position_partition' else ps
                        query_rows.append(predict(query['task_id'],qs,query['params']))
                    for i,p in enumerate(ps):
                        answers=[q[i] for q in query_rows]; value=apply_answers(edge,n,[a['value'] for a in answers]); gold=label(edge['target']['task_id'],p)
                        all_correct=all(a['correct'] for a in answers); bounds=value is not None and not ((1 if edge['target']['task_id']=='lis_length' else 0)<=value<=n)
                        correct=value==gold and not bounds
                        if all_correct and not diagnostic: assert correct
                        raw.append(dict(run_id=cp['run_id'],seed=cp['seed'],chain=cp['chain'],k=cp['k'],budget=cp['budget'],target=edge['target']['task_id'],n=n,index=i,route=edge['relation_id'],method='A' if cp['stage']=='pretraining' else 'D',diagnostic=diagnostic,correct=correct,all_sources_correct=all_correct,error_cancellation=correct and not all_correct,illegal=any(a['illegal'] for a in answers),eos_failure=any(a['eos_failure'] for a in answers),out_of_bounds=bounds,query_correct=[a['correct'] for a in answers],reconstructed=value,queries=len(answers)))
        completed.append(cp['run_id']); print('Evaluated',cp['run_id'],round(time.monotonic()-start,1),flush=True)
    with (OUT/'raw_predictions.jsonl').open('w',encoding='utf8') as f:
        for row in raw: f.write(json.dumps(row)+'\n')
    platform.write_csv(OUT/'compute.csv',timings)
    dump(OUT/'EVALUATION_COMPLETE.json',dict(completed=completed,expected=78,seconds=time.monotonic()-start,budget_seconds=7200,status='complete' if len(completed)==78 else 'partial'))


if __name__=='__main__':
    if sys.argv[1]=='freeze': freeze()
    elif sys.argv[1]=='evaluate': evaluate()
