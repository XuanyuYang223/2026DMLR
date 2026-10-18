"""Read-only artifact/budget audit. No test forwards or training."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
import torch
import pilot

root=pilot.HERE/'runs'/'pilot_v1'
pilot.verify(root)
assert json.loads((root/'COMPLETED.json').read_text())['rows']==18
records=[]; seconds=0.; tokens=0; supervised=0
for s in pilot.CFG['seeds']:
    unit=root/f'seed{s}'
    pre=[]
    for c in ['B','C']:
        m=json.loads((unit/f'pretrain_{c}.json').read_text());pre.append(m)
        assert m['step']==2400 and m['input_tokens']==4300800 and m['supervised_tokens']==307200
        assert m['tasks']==pilot.SOURCES[c]
        records.append(dict(seed=s,stage='pretrain',condition=c,n=8192,updates=m['step'],input_tokens=m['input_tokens'],supervised_tokens=m['supervised_tokens'],seconds=m['seconds'],peak_memory=m['peak_memory']))
    assert pre[0]['seed']==pre[1]['seed']
    start=torch.load(unit/'step0.pt',map_location='cpu',weights_only=False)['model']
    initial_row=start['token_embedding.weight'][pilot.V['<FIXED_POINTS>']]
    # Check the implemented adaptation initialization rule independently for B/C.
    for c in ['B','C']:
        state=torch.load(unit/f'pretrain_{c}.pt',map_location='cpu',weights_only=False)['model']
        state['token_embedding.weight'][pilot.V['<FIXED_POINTS>']].copy_(initial_row)
        assert torch.equal(state['lm_head.weight'][pilot.V['<FIXED_POINTS>']],initial_row)
    for n in pilot.CFG['sizes']:
        seeds=[]
        for c in ['A','B','C']:
            m=json.loads((unit/f'adapt_{c}_{n}.json').read_text());seeds.append(m['seed'])
            assert m['step']==800 and m['input_tokens']==1433600 and m['supervised_tokens']==102400 and m['tasks']==[0]
            records.append(dict(seed=s,stage='adapt',condition=c,n=n,updates=m['step'],input_tokens=m['input_tokens'],supervised_tokens=m['supervised_tokens'],seconds=m['seconds'],peak_memory=m['peak_memory']))
        assert len(set(seeds))==1
    rows=list(csv.DictReader((root/f'results_seed{s}.csv').open()))
    assert len(rows)==9
    for r in rows:assert 0<=float(r['test_accuracy'])<=1
for rec in records:seconds+=rec['seconds'];tokens+=rec['input_tokens'];supervised+=rec['supervised_tokens']
pilot.csvwrite(root/'actual_budgets.csv',records)
allrows=list(csv.DictReader((root/'results_all.csv').open()))
summary=[]
for n in pilot.CFG['sizes']:
    means={c:float(np.mean([float(r['test_accuracy']) for r in allrows if int(r['n'])==n and r['condition']==c])) for c in ['A','B','C']}
    summary.append(dict(n=n,**means,B_minus_A=means['B']-means['A'],C_minus_A=means['C']-means['A'],B_minus_C=means['B']-means['C']))
pilot.csvwrite(root/'summary.csv',summary)
sources={}
repo=pilot.HERE.parents[1]
for name,ref in [('neurips','74f0de2017115f06f11b7285366b66e7e8431d30'),('2026TMLR','d9a113d20d34db3c21cf69cf546cff4a0ab4904a')]:
    sources[name]=dict(url=f'https://github.com/XuanyuYang223/{name}',commit=ref)
for f in sorted((pilot.HERE/'vendor').rglob('*.py')):
    if f.name=='sdpa.py': source=repo/'reference/2026TMLR/experiments/longrun_attention.py'
    else:source=repo/'reference/neurips/src/neurips_permutations'/f.name
    assert pilot.sha(f)==pilot.sha(source)
pilot.write(root/'provenance.json',dict(sources=sources,vendor_matches_reference_worktree=True,python=__import__('sys').version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
pilot.write(root/'vocabulary.json',dict(pilot.V))
pilot.write(root/'final_audit.json',dict(status='passed',source_code_and_data_hashes_verified=True,formal_runs=len(records),formal_seconds=seconds,formal_input_tokens=tokens,formal_supervised_tokens=supervised,paired_pretrain_budgets_equal=True,paired_adaptation_budgets_equal=True,target_task_tied_row_reset_verified=True,test_forwards_in_this_audit=0))
files={str(f.relative_to(root)):dict(bytes=f.stat().st_size,sha256=pilot.sha(f)) for f in sorted(root.rglob('*')) if f.is_file() and f.name not in ['artifact_manifest.json','execution.log']}
pilot.write(root/'artifact_manifest.json',files)
print(json.dumps(json.loads((root/'final_audit.json').read_text()),indent=2))
