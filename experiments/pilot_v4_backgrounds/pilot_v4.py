"""Finite paired background study, reusing frozen v3 training primitives."""
import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch

from design import HERE, ROOT, V3, CFG, TARGETS, BACKGROUNDS, GROUPS, group, source_id, adapt_id, cells
sys.path.insert(0, str(HERE.parent/'pilot_v3_multitarget'))
import pilot_v3 as p

OUT=HERE/'artifacts'
read,write,sha,csvread,csvwrite,load,save=p.read,p.write,p.sha,p.csvread,p.csvwrite,p.load,p.save
statehash=p.statehash

def bind(out):
    global OUT
    OUT=out.resolve();p.OUT=OUT;p.CFG=CFG

def ref(path):return os.path.relpath(path,OUT).replace('\\','/')
def resolve(path):return (OUT/path).resolve()
def oldresolve(path):return (V3/path).resolve()
def identity(seed,target,background,condition):
    return dict(seed=seed,model_initialization_seed=seed,data_generation_seed=CFG['data_generation_seeds'][str(seed)],
                data_pool_id=f's{seed}',target=target,background=background,condition=condition)

def old_data():
    return p.old_data_paths()+sorted((V3/'data').glob('*.npz'))

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[f for folder in ['permworld_pilot','pilot_v2_diagnostics','pilot_v3_multitarget']
           for f in (HERE.parent/folder).rglob('*') if f.is_file() and '__pycache__' not in f.parts]
    paths.append(ROOT/'PILOT_REPORT.md')
    write(OUT/'preservation.json',{f.relative_to(ROOT).as_posix():sha(f) for f in sorted(paths)})
    ready=read(V3/'DATA_READY.json');pools=[]
    for row in ready['pools']:
        origin=V3/'data'/f's{row["seed"]}_{row["role"]}.npz';dest=OUT/'data'/origin.name
        assert sha(origin)==row['sha256'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(origin,dest)
        pools.append(dict(**row,data_generation_seed=CFG['data_generation_seeds'][str(row['seed'])],source_file=ref(origin)))
    for seed,digest in ready['initials'].items():
        origin=V3/'initial'/f'seed{seed}.pt';dest=OUT/'initial'/origin.name
        assert sha(origin)==digest;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(origin,dest)
    write(OUT/'DATA_READY.json',dict(unix=time.time(),pools=pools,initials=ready['initials'],exact_v3_reuse=True))
    write(OUT/'config.json',CFG)
    rows=[]
    for g in GROUPS:
        for c,tasks in g['sources'].items():
            assert len(set(tasks))==4 and not set(tasks).intersection(TARGETS)
            for slot,task in enumerate(tasks):rows.append(dict(target=g['target'],background=g['background'],condition=c,slot=slot,task=task,role='candidate' if slot==0 else 'common',fraction=.25))
    csvwrite(OUT/'compositions.csv',rows)
    write(OUT/'seed_provenance.json',dict(data_generation=CFG['data_generation_seeds'],model_initialization=CFG['model_initialization_seeds'],
          source_sampling_and_dropout=CFG['pretrain_sampler_and_dropout_seeds'],adapt_sampling_and_dropout=CFG['adapt_sampler_and_dropout_seeds'],
          source_note='v1 generated s17/s42 pools sequentially from RNG20261018; v3 generated s73 roles sequentially from RNG2026101803 after excluding all v1/v2 inputs.',
          new_test_generation=CFG['new_test_seed'],new_test_pool_order=CFG['seeds'],seed_interpretation='A paired seed block jointly indexes data and initialization; their effects are not independently varied.'))
    print('Copied exact v3 pools and initial weights; historical preservation snapshot complete.',flush=True)

def smoke():
    assert p.v1.DEVICE.type=='cuda'
    tasks=group('fixed_points','g1_recoils')['sources']['B'];pool=p.data(17,'source_train');tiny=(pool[0][:64],pool[1][:64])
    net,init=p.initial(17);opt=p.optmake(net,.0003);p.v1.seed(901);rng=np.random.default_rng(901)
    for st in range(4):p.step(net,opt,tiny,tasks,rng,.0003)
    save(OUT/'smoke'/'resume.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),**p.rngstates(rng)))
    for st in range(4):p.step(net,opt,tiny,tasks,rng,.0003)
    expected=statehash(net.state_dict());cp=load(OUT/'smoke'/'resume.pt');n2,_=p.initial(17);n2.load_state_dict(cp['model']);o2=p.optmake(n2,.0003);o2.load_state_dict(cp['optimizer']);r2=p.restore(cp)
    for st in range(4):p.step(n2,o2,tiny,tasks,r2,.0003)
    assert statehash(n2.state_dict())==expected
    before=torch.cuda.get_rng_state_all();p.evaluate(n2,tiny,'fixed_points');assert all(torch.equal(a,b) for a,b in zip(before,torch.cuda.get_rng_state_all()))
    ids,gold,eq=p.batch(tiny,p.T['fixed_points']);n2.eval()
    with torch.no_grad():
        a=n2(ids)[:,eq];alter=ids.clone();alter[:,eq+1]=(gold+1)%10;assert (a-n2(alter)[:,eq]).abs().max().item()==0
        token=p.v1.V['<FIXED_POINTS>'];n2.token_embedding.weight[token].copy_(init['model']['token_embedding.weight'][token].to(p.v1.DEVICE))
    fresh=p.optmake(n2,.0003);assert not fresh.state
    for st in range(4):p.step(n2,fresh,tiny,['fixed_points'],np.random.default_rng(910+st),.0003)
    metric=p.evaluate(n2,tiny,'fixed_points');assert np.isfinite(metric['answer_ce'])
    # At shared indices/slots, common-task token sequences and labels are bitwise equal.
    rng=np.random.default_rng(1017);idx=rng.choice(64,64,replace=False);slots=np.tile(np.arange(4),16);rng.shuffle(slots)
    for g in GROUPS:
        bt=np.array([p.T[t] for t in g['sources']['B']])[slots];ct=np.array([p.T[t] for t in g['sources']['C']])[slots]
        b,by,_=p.batch((tiny[0][idx],tiny[1][idx]),bt);c,cy,_=p.batch((tiny[0][idx],tiny[1][idx]),ct)
        assert torch.equal(b[slots!=0],c[slots!=0]) and torch.equal(by[slots!=0],cy[slots!=0])
        assert torch.equal(b[:,:-4],c[:,:-4])
    write(OUT/'SMOKE.json',dict(unix=time.time(),status='passed',resume_bitwise_equal=True,causal_no_answer_leak=True,evaluation_preserves_rng=True,
          common_task_sequences_equal=True,fresh_optimizer=True,full_pretrain_reset_adapt_evaluate=True))

def freeze():
    assert read(OUT/'SMOKE.json')['status']=='passed'
    assert (HERE/'catalog'/'CATALOG_AUDIT.json').exists() and (HERE/'references'/'REUSE_AUDIT.md').exists()
    assert read(HERE/'catalog'/'CATALOG_AUDIT.json')['status']=='passed'
    assert not list((OUT/'data').glob('*_test.npz')) and not (OUT/'PRETRAIN_COMPLETE.json').exists()
    catalog_design=read(HERE/'catalog'/'BACKGROUND_DESIGN.json')
    assert catalog_design['targets']==TARGETS
    for g in GROUPS:
        documented=next(r for r in catalog_design['groups'] if (r['target'],r['background'])==(g['target'],g['background']))
        assert documented['sources']==g['sources'] and documented['changed_slot']==g['changed_slot']
    files=[HERE/'pilot_v4.py',HERE/'design.py',HERE/'TASK_CATALOG.csv',HERE/'DATA_PLATFORM.md',
           *sorted((HERE/'catalog').rglob('*')),*sorted((HERE/'references').rglob('*')),
           HERE.parent/'pilot_v3_multitarget'/'pilot_v3.py',HERE.parent/'permworld_pilot'/'pilot.py',HERE.parent/'permworld_pilot'/'sdpa.py',
           *sorted((HERE.parent/'permworld_pilot'/'vendor').rglob('*.py'))]
    files=[f for f in files if f.is_file() and '__pycache__' not in f.parts]
    evidence=[V3/'FROZEN_PROTOCOL.json',V3/'FINAL_AUDIT.json',V3/'SELECTIONS_FROZEN.json']
    tasks=sorted({t for g in GROUPS for ts in g['sources'].values() for t in ts})
    evidence += [V3/'single'/f's{s}_{t}'/name for s in CFG['seeds'] for t in tasks for name in ['curve.csv','completed.json']]
    write(OUT/'FROZEN_PROTOCOL.json',dict(unix=time.time(),config=CFG,groups=GROUPS,code={f.relative_to(ROOT).as_posix():sha(f) for f in files},
          evidence={ref(f):sha(f) for f in evidence},data_sha256=sha(OUT/'DATA_READY.json'),compositions_sha256=sha(OUT/'compositions.csv'),
          historical_selection='Both cases and g0 anchor exploratory based on v3; additional backgrounds fixed before v4 training/test.',new_test_exists=False))
    print('All six target/background combinations frozen before new transfer training/test.',flush=True)

def verify():
    f=read(OUT/'FROZEN_PROTOCOL.json');assert f['config']==CFG==read(OUT/'config.json')
    assert f['data_sha256']==sha(OUT/'DATA_READY.json') and f['compositions_sha256']==sha(OUT/'compositions.csv')
    for name,digest in f['code'].items():assert sha(ROOT/name)==digest,name
    for name,digest in f['evidence'].items():assert sha(resolve(name))==digest,name
    ready=read(OUT/'DATA_READY.json')
    for r in ready['pools']:assert sha(OUT/'data'/f's{r["seed"]}_{r["role"]}.npz')==r['sha256']
    for s,digest in ready['initials'].items():assert sha(OUT/'initial'/f'seed{s}.pt')==digest

def source_folder(seed,target,background,c):return OUT/'multi'/source_id(seed,target,background,c)

def pretrain():
    verify();index={}
    for oldfolder in sorted((V3/'multi').iterdir()):
        m=read(oldfolder/'completed.json');index.setdefault((m['seed'],tuple(m['tasks'])),oldfolder)
    for seed in CFG['seeds']:
        for g in GROUPS:
            for c,tasks in g['sources'].items():
                folder=source_folder(seed,g['target'],g['background'],c)
                if (folder/'completed.json').exists():continue
                key=(seed,tuple(tasks));ident=identity(seed,g['target'],g['background'],c)
                if key in index:
                    oldfolder=V3/'multi'/f's{seed}_{g["target"]}_{c}' if g['background']=='g0_v3' else index[key]
                    m=read(oldfolder/'completed.json');path=oldresolve(m['checkpoint'])
                    assert m['tasks']==tasks
                    assert sha(path)==m['checkpoint_sha256'];cp=load(path)
                    assert cp['step']==4800 and statehash(cp['model'])==m['model_state_hash'] and cp['config']['model']==CFG['model']
                    assert m['train_sha256']==p.poolhash(seed,'source_train') and m['dev_sha256']==p.poolhash(seed,'source_dev')
                    curve=csvread(oldfolder/'curve.csv')
                    for row in curve:row.update(ident,run_id=folder.name)
                    csvwrite(folder/'curve.csv',curve)
                    m.update(ident,new_compute=False,checkpoint=ref(path),reused_from=ref(oldfolder),reuse_reason='exact ordered task tuple, initial state, input pools, schedule and budget')
                    write(folder/'completed.json',m)
                else:
                    p.train_sources(seed,tasks,folder,'multi');m=read(folder/'completed.json');m.update(ident)
                    write(folder/'completed.json',m)
    # Independently reconstruct per-slot streams; metadata schedule hash binds actual training.
    exposures=[];curves=[];slots_summary=[]
    for seed in CFG['seeds']:
        rng=np.random.default_rng(seed+1000);indices=[];assign=[];h=hashlib.sha256()
        for _ in range(4800):
            idx=rng.choice(8192,64,replace=False);slots=np.tile(np.arange(4),16);rng.shuffle(slots);h.update(idx.tobytes());h.update(slots.tobytes());indices.append(idx);assign.append(slots)
        indices=np.asarray(indices);assign=np.asarray(assign);d=np.load(OUT/'data'/f's{seed}_source_train.npz')
        for slot in range(4):
            stream=indices[assign==slot]
            slots_summary.append(dict(seed=seed,slot=slot,exposures=len(stream),unique_inputs=len(set(stream.tolist())),index_stream_sha256=hashlib.sha256(stream.tobytes()).hexdigest(),input_stream_sha256=hashlib.sha256(d['p'][stream].tobytes()).hexdigest()))
        for g in GROUPS:
            for c,tasks in g['sources'].items():
                folder=source_folder(seed,g['target'],g['background'],c);m=read(folder/'completed.json');assert m['schedule_sha256']==h.hexdigest()
                for row in csvread(folder/'curve.csv'):row.update(identity(seed,g['target'],g['background'],c));curves.append(row)
                for slot,task in enumerate(tasks):
                    sr=next(r for r in slots_summary if r['seed']==seed and r['slot']==slot);assert m['actual_unique_per_task'][task]==sr['unique_inputs']
                    stream=indices[assign==slot]
                    exposures.append(dict(**identity(seed,g['target'],g['background'],c),task=task,slot=slot,length=10,
                          exposures=76800,unique_inputs=sr['unique_inputs'],input_tokens=76800*28,supervised_tokens=76800*2,task_fraction=.25,
                          index_stream_sha256=sr['index_stream_sha256'],input_stream_sha256=sr['input_stream_sha256'],
                          label_stream_sha256=hashlib.sha256(d['y'][stream,p.T[task]].tobytes()).hexdigest(),new_compute=m['new_compute']))
    csvwrite(OUT/'source_exposures.csv',exposures);csvwrite(OUT/'sampling_slots.csv',slots_summary);csvwrite(OUT/'source_curves.csv',curves)
    write(OUT/'PRETRAIN_COMPLETE.json',dict(unix=time.time(),logical_runs=36,new_runs=24,reused_runs=12))

def source_path(seed,target,background,c):
    return OUT/'initial'/f'seed{seed}.pt' if c=='A' else resolve(read(source_folder(seed,target,background,c)/'completed.json')['checkpoint'])

def reuse_adaptation(seed,target,background,c,lr,folder):
    oldfolder=V3/'adapt'/f's{seed}_{target}_{c}_lr{lr:.0e}';m=read(oldfolder/'completed.json')
    assert m['updates']==800 and m['lr']==lr
    assert m['v3_train_sha256']==p.poolhash(seed,'target_train') and m['v3_dev_sha256']==p.poolhash(seed,'target_dev')
    for which in ['best','endpoint']:
        path=oldresolve(m[which+'_path']);assert sha(path)==m[which+'_sha256'];m[which+'_path']=ref(path)
    path=source_path(seed,target,background,c);cp=load(path)
    assert statehash(cp['model'])==m['initialization']['loaded_state_hash']==m['initialization']['source_state_hash']
    original_audit=copy.deepcopy(m['initialization']);audit=copy.deepcopy(original_audit)
    audit.update(source_path=ref(path),source_sha256=sha(path),train_sha256=p.poolhash(seed,'target_train'),dev_sha256=p.poolhash(seed,'target_dev'))
    ident=identity(seed,target,background,c);curve=csvread(oldfolder/'curve.csv')
    for row in curve:row.update(ident)
    csvwrite(folder/'curve.csv',curve)
    for which in ['best','endpoint']:m[which].update(ident)
    m.update(ident,run_id=folder.name,new_compute=False,reused_from=ref(oldfolder),reused_metadata_sha256=sha(oldfolder/'completed.json'),
             original_initialization_audit=original_audit,initialization=audit,v4_train_sha256=p.poolhash(seed,'target_train'),v4_dev_sha256=p.poolhash(seed,'target_dev'))
    write(folder/'completed.json',m)

def adapt_one(seed,target,background,c,lr):
    folder=OUT/'adapt'/adapt_id(seed,target,background,c,lr)
    if (folder/'completed.json').exists():return
    if c=='A' or background=='g0_v3':reuse_adaptation(seed,target,background,c,lr,folder);return
    ident=identity(seed,target,background,c);net,initial=p.initial(seed);path=source_path(seed,target,background,c);source=load(path)
    net.load_state_dict(source['model'],strict=True);loaded=statehash(net.state_dict());assert loaded==statehash(source['model'])
    token=p.v1.V[f'<{target.upper()}>']
    with torch.no_grad():net.token_embedding.weight[token].copy_(initial['model']['token_embedding.weight'][token].to(p.v1.DEVICE))
    assert net.lm_head.weight.data_ptr()==net.token_embedding.weight.data_ptr()
    assert torch.equal(net.token_embedding.weight[token].cpu(),initial['model']['token_embedding.weight'][token])
    rowhash=hashlib.sha256(net.token_embedding.weight[token].detach().cpu().numpy().tobytes()).hexdigest();opt=p.optmake(net,lr);assert not opt.state
    audit=dict(source_path=ref(path),source_sha256=sha(path),source_state_hash=loaded,loaded_state_hash=loaded,
               adapt_initial_state_hash=statehash(net.state_dict()),target_row_hash=rowhash,reset_row_ids=[token],
               optimizer_entries=0,optimizer_state_reset=True,new_output_head=False,train_sha256=p.poolhash(seed,'target_train'),dev_sha256=p.poolhash(seed,'target_dev'))
    train=p.data(seed,'target_train');dev=p.data(seed,'target_dev');p.v1.seed(seed+2000);rng=np.random.default_rng(seed+2000)
    mode=int(torch.bincount(train[1][:,p.T[target]]).argmax());curve=[];best=None;schedule=hashlib.sha256();start=time.monotonic()
    def measure(st,loss=None,norm=None):
        row=dict(**ident,lr=lr,step=st,last_batch_two_token_loss=loss,gradient_norm=norm)
        for split,d in [('train',train),('dev',dev)]:
            row.update({split+'_'+k:v for k,v in p.evaluate(net,d,target).items()})
            row[split+'_majority_accuracy']=float((d[1][:,p.T[target]]==mode).float().mean())
        return row
    curve.append(measure(0))
    for st in range(1,801):
        loss,norm,idx,slots=p.step(net,opt,train,[target],rng,p.adapt_lr(lr,st));schedule.update(idx.tobytes());schedule.update(slots.tobytes())
        if st%100==0:
            row=measure(st,loss,norm);curve.append(row)
            if best is None or (-row['dev_accuracy'],row['dev_answer_ce'],st)<(-best['dev_accuracy'],best['dev_answer_ce'],best['step']):
                best=row.copy();save(folder/'best.pt',dict(model=net.state_dict(),config=CFG,metrics=row,step=st,lr=lr,**ident))
            csvwrite(folder/'curve.csv',curve)
    assert {int(v['step']) for v in opt.state.values()}=={800}
    save(folder/'endpoint.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),config=CFG,metrics=curve[-1],step=800,lr=lr,**ident,**p.rngstates(rng)))
    m=dict(**ident,run_id=folder.name,lr=lr,best=best,endpoint=curve[-1],seconds=time.monotonic()-start,
           updates=800,input_tokens=1433600,supervised_tokens=102400,sample_exposures=51200,schedule_sha256=schedule.hexdigest(),
           initialization=audit,final_model_state_hash=statehash(net.state_dict()),best_path=ref(folder/'best.pt'),endpoint_path=ref(folder/'endpoint.pt'),
           best_sha256=sha(folder/'best.pt'),endpoint_sha256=sha(folder/'endpoint.pt'),new_compute=True,
           v4_train_sha256=p.poolhash(seed,'target_train'),v4_dev_sha256=p.poolhash(seed,'target_dev'))
    write(folder/'completed.json',m)
    print(dict(stage='adapt',run=folder.name,best_dev=best['dev_accuracy'],best_step=best['step'],endpoint_dev=curve[-1]['dev_accuracy'],seconds=m['seconds']),flush=True)

def adapt():
    verify();assert (OUT/'PRETRAIN_COMPLETE.json').exists()
    for seed,target,background,c in cells():
        for lr in CFG['learning_rates']:adapt_one(seed,target,background,c,lr)
    write(OUT/'ADAPT_COMPLETE.json',dict(unix=time.time(),unique_candidates=126,new_candidates=72,reused_candidates=54))

def select():
    verify();assert (OUT/'ADAPT_COMPLETE.json').exists();assert not list((OUT/'data').glob('*_test.npz'))
    selected=[];curves=[];candidates=[]
    for seed,target,background,c in cells():
        ms=[read(OUT/'adapt'/adapt_id(seed,target,background,c,lr)/'completed.json') for lr in CFG['learning_rates']]
        for m in ms:
            curves+=csvread(OUT/'adapt'/m['run_id']/'curve.csv')
            candidates.append(dict(**identity(seed,target,background,c),lr=m['lr'],best_step=m['best']['step'],best_dev=m['best']['dev_accuracy'],best_dev_ce=m['best']['dev_answer_ce'],endpoint_dev=m['endpoint']['dev_accuracy'],new_compute=m['new_compute']))
        m=min(ms,key=lambda m:(-m['best']['dev_accuracy'],m['best']['dev_answer_ce'],m['best']['step'],CFG['learning_rates'].index(m['lr'])))
        selected.append(dict(**identity(seed,target,background,c),lr=m['lr'],step=m['best']['step'],dev_accuracy=m['best']['dev_accuracy'],dev_ce=m['best']['dev_answer_ce'],**{k:m[k] for k in ['best_path','endpoint_path','best_sha256','endpoint_sha256']}))
    assert len(selected)==42 and len(candidates)==126
    csvwrite(OUT/'target_curves.csv',curves);csvwrite(OUT/'candidate_development_results.csv',candidates)
    write(OUT/'SELECTIONS_FROZEN.json',dict(unix=time.time(),protocol_sha256=sha(OUT/'FROZEN_PROTOCOL.json'),selections=selected))
    print('42 unique selections frozen; shared A is not independently selected for each background.',flush=True)

def generate_test():
    assert (OUT/'SELECTIONS_FROZEN.json').exists()
    paths=old_data()+[f for f in sorted((OUT/'data').glob('*.npz')) if not f.name.endswith('_test.npz')]
    seen=p.input_keys(paths);excluded=len(seen);rng=np.random.default_rng(CFG['new_test_seed']);records=[]
    for seed in CFG['seeds']:
        values=[]
        while len(values)<2048:
            q=tuple(map(int,rng.permutation(10)+1));key=p.v1.key(q)
            if key in seen:continue
            seen.add(key);values.append(q)
        path=OUT/'data'/f's{seed}_test.npz';np.savez_compressed(path,p=np.asarray(values,dtype=np.int16),y=np.asarray([p.labels(q) for q in values],dtype=np.int16))
        records.append(dict(seed=seed,data_generation_seed=CFG['new_test_seed'],count=2048,sha256=sha(path),unique_orbits=len({p.v1.key(q) for q in values})))
    # Expanded-orbit cross-check, independent of the rejection keys.
    expanded=set()
    for path in paths:
        for q in np.load(path)['p']:expanded.update(p.v1.orbit(q))
    for r in records:
        members=set().union(*(p.v1.orbit(q) for q in np.load(OUT/'data'/f's{r["seed"]}_test.npz')['p']))
        assert not members.intersection(expanded);expanded.update(members)
    write(OUT/'test_data_audit.json',dict(unix=time.time(),selection_sha256=sha(OUT/'SELECTIONS_FROZEN.json'),excluded_orbits=excluded,
          overlap=0,pools=records,shared_across_targets_backgrounds_conditions=True))

def test():
    verify();lock=read(OUT/'SELECTIONS_FROZEN.json');assert lock['protocol_sha256']==sha(OUT/'FROZEN_PROTOCOL.json')
    if not (OUT/'test_data_audit.json').exists():generate_test()
    audit=read(OUT/'test_data_audit.json');assert audit['selection_sha256']==sha(OUT/'SELECTIONS_FROZEN.json')
    for r in audit['pools']:assert sha(OUT/'data'/f's{r["seed"]}_test.npz')==r['sha256']
    rows=[];current_seed=None;pool=train=None
    for seed,target,background,c in cells():
        if seed!=current_seed:pool=p.data(seed,'test');train=p.data(seed,'target_train');current_seed=seed
        choice=next(r for r in lock['selections'] if (r['seed'],r['target'],r['background'],r['condition'])==(seed,target,background,c))
        mode=int(torch.bincount(train[1][:,p.T[target]]).argmax());baseline=float((pool[1][:,p.T[target]]==mode).float().mean())
        for lr in CFG['learning_rates']:
            m=read(OUT/'adapt'/adapt_id(seed,target,background,c,lr)/'completed.json')
            for which in ['best','endpoint']:
                if lr==choice['lr']:assert m[which+'_sha256']==choice[which+'_sha256'] and m[which+'_path']==choice[which+'_path']
                path=resolve(m[which+'_path']);assert sha(path)==m[which+'_sha256'];cp=load(path);net,_=p.initial(seed);net.load_state_dict(cp['model'],strict=True)
                metric=p.evaluate(net,pool,target)
                rows.append(dict(**identity(seed,target,background,c),lr=lr,checkpoint=which,adapt_step=cp['step'],selected_lr=lr==choice['lr'],selected_primary=lr==choice['lr'] and which=='best',
                      dev_accuracy=m[which]['dev_accuracy'],dev_ce=m[which]['dev_answer_ce'],train_accuracy=m[which]['train_accuracy'],
                      **{'test_'+k:v for k,v in metric.items()},majority_accuracy=baseline,checkpoint_sha256=sha(path),new_compute=m['new_compute']))
        print(dict(stage='test',seed=seed,target=target,background=background,condition=c,test_accuracy=next(r['test_accuracy'] for r in reversed(rows) if r['selected_primary'])),flush=True)
    assert len(rows)==252
    for seed in CFG['seeds']:csvwrite(OUT/f'results_seed{seed}.csv',[r for r in rows if r['seed']==seed])
    csvwrite(OUT/'results_all_candidates.csv',rows);csvwrite(OUT/'results_selected.csv',[r for r in rows if r['selected_primary']])
    write(OUT/'TEST_COMPLETE.json',dict(unix=time.time(),candidate_rows=252,selected_rows=42,selection_sha256=sha(OUT/'SELECTIONS_FROZEN.json')))

def preservation_check():
    for path,digest in read(OUT/'preservation.json').items():assert sha(ROOT/path)==digest,path

STAGES=[('prepare','DATA_READY.json'),('smoke','SMOKE.json'),('freeze','FROZEN_PROTOCOL.json'),('pretrain','PRETRAIN_COMPLETE.json'),
        ('adapt','ADAPT_COMPLETE.json'),('select','SELECTIONS_FROZEN.json'),('test','TEST_COMPLETE.json')]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['all',*[s for s,_ in STAGES]])
    parser.add_argument('--out',type=Path,default=OUT);a=parser.parse_args();bind(a.out)
    for stage,marker in STAGES:
        if a.stage not in ['all',stage]:continue
        if (OUT/marker).exists():print('Skip completed',stage,flush=True);continue
        start=time.monotonic();print('Start',stage,flush=True)
        try:globals()[stage]()
        except Exception:
            write(OUT/f'FAILURE_{stage}_{int(time.time())}.json',dict(stage=stage,traceback=traceback.format_exc()));raise
        write(OUT/f'timing_{stage}.json',dict(stage=stage,seconds=time.monotonic()-start));print('Finished',stage,time.monotonic()-start,flush=True)
    if (OUT/'TEST_COMPLETE.json').exists():preservation_check();print('Finite v4 complete; all v1/v2/v3 files preserved; no expansion scheduled.',flush=True)

if __name__=='__main__':main()
