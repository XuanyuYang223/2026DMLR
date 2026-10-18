"""Frozen, paired diagnostics of the existing PermWorld pilot; never writes v1."""
import argparse, copy, csv, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
V1_CODE=HERE.parent/'permworld_pilot'
sys.path.insert(0,str(V1_CODE))
import pilot as v1
OLD=V1_CODE/'runs'/'pilot_v1'
OUT=HERE/'artifacts'
CONFIG=dict(version='pilot_v2_diagnostics/v1',seeds=[17,42],target='fixed_points',target_n=256,
    permutation_length=10,model=v1.CFG['model'],vocab_size=len(v1.VOCABULARY),encoding='one_line',
    sources={k:[v1.TASKS[t] for t in ts] for k,ts in v1.SOURCES.items()},
    learning_rates=[.0003,.0001,.00003],adapt_updates=800,batch=64,evaluate_every=100,
    warmup=100,minimum_lr_ratio=.1,weight_decay=.01,betas=[.9,.999],epsilon=1e-8,clip=1.,
    optimizer='AdamW, fresh per adaptation',precision='FP32 weights, CUDA BF16 training, TF32 disabled',
    selection='max dev answer accuracy, min dev answer CE, earliest update, declared LR order; step0 excluded',
    source_budgets=[2400,4800],extension_lr=.00003,extension_updates=2400,
    extension='Continue original checkpoint including optimizer and RNG, hold original terminal LR; same 8192 inputs',
    new_test_count=2048,new_test_seed=2026101802,
    test_rule='Generate only after selections freeze; reject all v1 and cross-seed full inverse/reverse/complement orbits',
    development_reuse='Reuse v1 target train[:256] and dev; known historical dev results make this exploratory',
    scope='No task replacement, no target/task-count expansion, stop after diagnostics')

def write(p,x):v1.write(p,x)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def csvwrite(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    if rows:v1.csvwrite(p,rows)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def statehash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def rowhash(net):return hashlib.sha256(net.token_embedding.weight[v1.V['<FIXED_POINTS>']].detach().cpu().numpy().tobytes()).hexdigest()
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,p)
def pool(seed,kind,limit=None):
    x,y=v1.loadpool(OLD,f's{seed}_{kind}');return (x[:limit],y[:limit]) if limit else (x,y)
def optimizer(net,lr):
    return torch.optim.AdamW(net.parameters(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=.01,fused=v1.DEVICE.type=='cuda')
def rng_state(rng):
    return dict(numpy_rng=copy.deepcopy(rng.bit_generator.state),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if v1.DEVICE.type=='cuda' else [])
def restore_rng(cp):
    rng=np.random.default_rng();rng.bit_generator.state=cp['numpy_rng'];torch.set_rng_state(cp['torch_rng'].cpu())
    if v1.DEVICE.type=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in cp['cuda_rng']])
    return rng

@torch.inference_mode()
def evaluate(net,data,task):
    net.eval();x,y=data;correct=eos_correct=0;answer_ce=eos_ce=0.
    for i in range(0,len(x),128):
        ids,gold,eq=v1.batch(x[i:i+128],y[i:i+128],task,False)
        z=net(ids)[:,eq].float();correct+=int((z.argmax(-1)==gold).sum());answer_ce+=float(F.cross_entropy(z,gold,reduction='sum'))
        ez=net(torch.cat([ids,gold[:,None]],1))[:,-1].float();eos=torch.full_like(gold,v1.V['<EOS>'])
        eos_correct+=int((ez.argmax(-1)==eos).sum());eos_ce+=float(F.cross_entropy(ez,eos,reduction='sum'))
    return dict(accuracy=correct/len(x),answer_ce=answer_ce/len(x),eos_ce=eos_ce/len(x),two_token_ce=(answer_ce+eos_ce)/(2*len(x)),eos_accuracy=eos_correct/len(x),count=len(x))

def step(net,opt,data,tasks,rng,lr):
    net.train();x,y=data;idx=rng.choice(len(x),64,replace=False)
    slot=np.tile(np.arange(len(tasks)),64//len(tasks));rng.shuffle(slot)
    ids,gold,eq=v1.batch(x[idx],y[idx],np.asarray(tasks)[slot])
    for group in opt.param_groups:group['lr']=lr
    opt.zero_grad(set_to_none=True)
    with torch.autocast(v1.DEVICE.type,dtype=torch.bfloat16,enabled=v1.DEVICE.type=='cuda'):
        z=net(ids);loss=.5*(F.cross_entropy(z[:,eq],gold)+F.cross_entropy(z[:,eq+1],torch.full_like(gold,v1.V['<EOS>'])))
    assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
    return float(loss.detach()),float(norm),idx,slot

def from_checkpoint(path,seed):
    cp=load(path);net=v1.make(seed);net.load_state_dict(cp['model'],strict=True)
    assert statehash(net.state_dict())==statehash(cp['model'])
    return net,cp

def adapt_initial(seed,condition,budget):
    initial=load(OLD/f'seed{seed}'/'step0.pt')['model']
    source=OLD/f'seed{seed}'/'step0.pt' if condition=='A' else source_path(seed,condition,budget)
    net,cp=from_checkpoint(source,seed);loaded_hash=statehash(net.state_dict())
    with torch.no_grad():net.token_embedding.weight[v1.V['<FIXED_POINTS>']].copy_(initial['token_embedding.weight'][v1.V['<FIXED_POINTS>']].to(v1.DEVICE))
    assert torch.equal(net.token_embedding.weight[v1.V['<FIXED_POINTS>']].cpu(),initial['token_embedding.weight'][v1.V['<FIXED_POINTS>']])
    assert net.lm_head.weight.data_ptr()==net.token_embedding.weight.data_ptr()
    return net,dict(source_path=source.relative_to(HERE.parent).as_posix(),source_sha256=sha(source),loaded_state_hash=loaded_hash,
        source_state_hash=statehash(cp['model']),adapt_initial_state_hash=statehash(net.state_dict()),target_row_hash=rowhash(net),
        tied_output_row=True,reset_row_ids=[v1.V['<FIXED_POINTS>']],new_output_head=False)

def source_path(seed,condition,budget):
    return OLD/f'seed{seed}'/f'pretrain_{condition}.pt' if budget==2400 else OUT/'extended'/f'seed{seed}_{condition}_4800.pt'

def inventory():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'inventory.json').exists(),'Inventory already exists; use run_all to resume.'
    v1.verify(OLD)
    originals={str(p.relative_to(HERE.parent.parent)).replace('\\','/'):sha(p) for p in sorted(OLD.rglob('*')) if p.is_file()}
    for p in [V1_CODE/'pilot.py',HERE.parent.parent/'PILOT_REPORT.md']:
        originals[p.relative_to(HERE.parent.parent).as_posix()]=sha(p)
    write(OUT/'v1_preservation.json',originals)
    rows=[];budgets=[];target_curves=[];source_curves=[];checks=[];histograms=[]
    for seed in CONFIG['seeds']:
        train=pool(seed,'pretrain');dev=pool(seed,'source_dev');initial=load(OLD/f'seed{seed}'/'step0.pt')['model']
        for condition in ['B','C']:
            net,cp=from_checkpoint(source_path(seed,condition,2400),seed)
            for t in v1.SOURCES[condition]:
                mode=int(torch.bincount(train[1][:,t]).argmax())
                for split,data in [('train',train),('dev',dev)]:
                    rows.append(dict(seed=seed,condition=condition,budget=2400,step=2400,task=v1.TASKS[t],length=10,split=split,**evaluate(net,data,t),majority_label=mode,majority_accuracy=float((data[1][:,t]==mode).float().mean()),evidence='supplemented_from_original_checkpoint'))
            for r in cp['curve']:source_curves.append(dict(seed=seed,condition=condition,step=r['step'],last_batch_two_token_loss=r['loss']))
            opt=optimizer(net,.0003);assert not opt.state
            checks.append(dict(seed=seed,condition=condition,loaded_exactly=True,changed_state_tensors=sum(not torch.equal(v,initial[k]) for k,v in cp['model'].items()),saved_optimizer_steps=sorted({int(v['step']) for v in cp['optimizer']['state'].values()}),fresh_optimizer_entries=len(opt.state)))
            # Reconstruct exact historical schedule and prospective continuation, without model training.
            rng=np.random.default_rng(seed+1000);seen=[set() for _ in range(4)];schedule_hash=hashlib.sha256()
            for st in range(1,4801):
                idx=rng.choice(8192,64,replace=False);slot=np.tile(np.arange(4),16);rng.shuffle(slot)
                schedule_hash.update(idx.tobytes());schedule_hash.update(slot.tobytes())
                for j in range(4):seen[j].update(map(int,idx[slot==j]))
                if st in [2400,4800]:
                    if st==2400:assert rng.bit_generator.state==cp['numpy_rng']
                    for j,t in enumerate(v1.SOURCES[condition]):
                        budgets.append(dict(seed=seed,condition=condition,budget=st,task=v1.TASKS[t],pool_unique_inputs=8192,actual_unique_inputs=len(seen[j]),task_exposures=st*16,task_fraction=.25,total_exposures=st*64,input_tokens=st*64*28,supervised_tokens=st*64*2,updates=st,schedule_sha256=schedule_hash.hexdigest(),status='observed' if st==2400 else 'planned_then_verified_in_training'))
            del net,opt,cp
        for c in ['A','B','C']:
            cp=load(OLD/f'seed{seed}'/f'adapt_{c}_256.pt')
            assert {int(v['step']) for v in cp['optimizer']['state'].values()}=={800}
            for r in cp['curve']:
                target_curves.append(dict(seed=seed,condition=c,step=r['step'],last_batch_two_token_loss=r['loss'],dev_accuracy=r['accuracy'],dev_answer_ce=r['ce'],full_train_metrics='missing_historically',checkpoint_available=r['step']==800))
        for t in [1,2]:
            values=train[1][:,t].cpu().numpy();counts=np.bincount(values,minlength=11)
            for answer,count in enumerate(counts):histograms.append(dict(seed=seed,task=v1.TASKS[t],answer=answer,count=int(count),probability=float(count/len(values))))
    csvwrite(OUT/'source_learning_2400.csv',rows);csvwrite(OUT/'source_budgets.csv',budgets)
    csvwrite(OUT/'v1_observed_target_curves.csv',target_curves);csvwrite(OUT/'v1_observed_pretrain_loss.csv',source_curves);csvwrite(OUT/'replacement_distributions.csv',histograms)
    write(OUT/'v1_checkpoint_audit.json',checks)
    write(OUT/'inventory.json',dict(completed=True,extension_justified=any(r['split']=='dev' and r['accuracy']<.90 for r in rows),
        missing=['historical per-task source train CE and historical per-task source trajectories','historical full target train trajectories','historical intermediate model checkpoints'],
        supplemented=['source endpoint full train/dev answer CE, two-token CE, accuracy and length-10 majority baselines','exact sampling replay and checkpoint loading audit'],
        not_recoverable='Historical intermediate checkpoints/metrics cannot be recovered from final checkpoints; new replay curves will be labelled separately',
        inference='Under-learning and increasing dev CE are supported; LR mismatch and transfer mechanism remain hypotheses'))
    print('Inventory and source endpoint supplementation complete.',flush=True)

def smoke():
    net,cp=from_checkpoint(source_path(17,'B',2400),17);data=pool(17,'pretrain');opt=optimizer(net,.00003);opt.load_state_dict(cp['optimizer']);rng=restore_rng(cp)
    for _ in range(4):step(net,opt,data,v1.SOURCES['B'],rng,.00003)
    p=OUT/'smoke'/'resume.pt';save(p,dict(model=net.state_dict(),optimizer=opt.state_dict(),**rng_state(rng)))
    for _ in range(4):step(net,opt,data,v1.SOURCES['B'],rng,.00003)
    expected=statehash(net.state_dict());expected_rng=copy.deepcopy(rng.bit_generator.state)
    n2,saved=from_checkpoint(p,17);o2=optimizer(n2,.00003);o2.load_state_dict(saved['optimizer']);r2=restore_rng(saved)
    for _ in range(4):step(n2,o2,data,v1.SOURCES['B'],r2,.00003)
    assert statehash(n2.state_dict())==expected and r2.bit_generator.state==expected_rng
    # Evaluation must not perturb dropout RNG, allowing faithful v1 replays.
    before=torch.cuda.get_rng_state_all() if v1.DEVICE.type=='cuda' else [torch.get_rng_state()]
    evaluate(n2,pool(17,'target_train',256),0)
    after=torch.cuda.get_rng_state_all() if v1.DEVICE.type=='cuda' else [torch.get_rng_state()]
    assert all(torch.equal(a,b) for a,b in zip(before,after))
    rows=[]
    for c in ['A','B','C']:
        net,audit=adapt_initial(17,c,0 if c=='A' else 2400);audit['condition']=c;rows.append(audit)
    assert len({r['target_row_hash'] for r in rows})==1
    write(OUT/'smoke.json',dict(status='passed',resume_state_bitwise_equal=True,evaluation_preserves_rng=True,initializations=rows))
    print('Smoke passed, exact 4+4 continuation and initialization checks.',flush=True)

def freeze():
    assert read(OUT/'inventory.json')['extension_justified'];assert read(OUT/'smoke.json')['status']=='passed'
    files=[HERE/'diagnose.py',V1_CODE/'pilot.py',*sorted((V1_CODE/'vendor').rglob('*.py'))]
    write(OUT/'FROZEN_PROTOCOL.json',dict(config=CONFIG,created_unix=time.time(),code={p.relative_to(HERE.parent).as_posix():sha(p) for p in files},inventory_sha256=sha(OUT/'inventory.json')))

def verify():
    f=read(OUT/'FROZEN_PROTOCOL.json');assert f['config']==CONFIG
    for p,h in f['code'].items():assert sha(HERE.parent/p)==h,p

def extend(seed,condition):
    path=source_path(seed,condition,4800)
    if path.with_suffix('.json').exists():return
    train=pool(seed,'pretrain');dev=pool(seed,'source_dev');net,cp=from_checkpoint(source_path(seed,condition,2400),seed)
    opt=optimizer(net,CONFIG['extension_lr']);opt.load_state_dict(cp['optimizer'])
    assert {int(v['step']) for v in opt.state.values()}=={2400}
    assert all(abs(g['lr']-.00003)<1e-12 for g in opt.param_groups)
    rng=restore_rng(cp);start=time.monotonic();curve=[]
    for st in range(2401,4801):
        loss,norm,_,_=step(net,opt,train,v1.SOURCES[condition],rng,.00003)
        if st%600==0:
            for t in v1.SOURCES[condition]:
                mode=int(torch.bincount(train[1][:,t]).argmax())
                for split,data in [('train',train),('dev',dev)]:
                    curve.append(dict(seed=seed,condition=condition,budget=4800,step=st,task=v1.TASKS[t],length=10,split=split,**evaluate(net,data,t),majority_label=mode,majority_accuracy=float((data[1][:,t]==mode).float().mean()),evidence='new_continuation'))
            print(dict(stage='extend',seed=seed,condition=condition,step=st,loss=loss,seconds=time.monotonic()-start),flush=True)
    assert {int(v['step']) for v in opt.state.values()}=={4800}
    expected_rng=np.random.default_rng(seed+1000)
    for _ in range(4800):
        expected_rng.choice(8192,64,replace=False);expected_slots=np.tile(np.arange(4),16);expected_rng.shuffle(expected_slots)
    assert rng.bit_generator.state==expected_rng.bit_generator.state
    meta=dict(seed=seed,condition=condition,step=4800,additional_updates=2400,additional_input_tokens=4300800,additional_supervised_tokens=307200,seconds=time.monotonic()-start,
        source_checkpoint_sha256=sha(source_path(seed,condition,2400)),lr=.00003,optimizer_continued=True,rng_continued=True,final_rng_numpy=rng.bit_generator.state,
        model_state_hash=statehash(net.state_dict()),sampler_replay_4800_verified=True,protocol_sha256=sha(OUT/'FROZEN_PROTOCOL.json'))
    save(path,dict(model=net.state_dict(),optimizer=opt.state_dict(),config=CONFIG,**rng_state(rng),**meta));write(path.with_suffix('.json'),meta)
    csvwrite(path.with_suffix('.csv'),curve)

def runid(seed,condition,budget,lr):return f's{seed}_{condition}_p{budget}_lr{lr:.0e}'

def adapt(seed,condition,budget,lr):
    rid=runid(seed,condition,budget,lr);dest=OUT/'adapt'/rid
    if (dest/'completed.json').exists():return
    data=pool(seed,'target_train',256);dev=pool(seed,'target_dev');net,audit=adapt_initial(seed,condition,budget);opt=optimizer(net,lr);assert not opt.state
    audit.update(seed=seed,condition=condition,budget=budget,lr=lr,optimizer_entries=0,optimizer_state_reset=True,
        train_sha256=sha(OLD/'data'/f's{seed}_target_train.npz'),train_slice=[0,256],dev_sha256=sha(OLD/'data'/f's{seed}_target_dev.npz'))
    write(dest/'initialization.json',audit)
    v1.seed(seed+2000);rng=np.random.default_rng(seed+2000);curve=[];best_key=None;best_meta=None;start=time.monotonic()
    mode=int(torch.bincount(data[1][:,0]).argmax())
    def measure(st,loss=None,grad=None):
        row=dict(seed=seed,condition=condition,pretrain_updates=budget,lr=lr,step=st,last_batch_two_token_loss=loss,gradient_norm=grad)
        for split,d in [('train',data),('dev',dev)]:
            row.update({split+'_'+k:v for k,v in evaluate(net,d,0).items()})
            row[split+'_majority_accuracy']=float((d[1][:,0]==mode).float().mean())
        return row
    curve.append(measure(0));schedule_hash=hashlib.sha256()
    for st in range(1,801):
        factor=st/100 if st<=100 else .1+.9*.5*(1+math.cos(math.pi*(st-100)/700))
        loss,norm,idx,slot=step(net,opt,data,[0],rng,lr*factor);schedule_hash.update(idx.tobytes());schedule_hash.update(slot.tobytes())
        if st%100==0:
            row=measure(st,loss,norm);curve.append(row);key=(-row['dev_accuracy'],row['dev_answer_ce'],st)
            if best_key is None or key<best_key:
                best_key=key;best_meta=row.copy();save(dest/'best.pt',dict(model=net.state_dict(),config=CONFIG,seed=seed,step=st,lr=lr,condition=condition,budget=budget,metrics=row))
            csvwrite(dest/'curve.csv',curve)
            if st%400==0:print(dict(stage='adapt',run=rid,step=st,train=row['train_accuracy'],dev=row['dev_accuracy'],dev_ce=row['dev_answer_ce'],seconds=time.monotonic()-start),flush=True)
    assert {int(v['step']) for v in opt.state.values()}=={800}
    finalhash=statehash(net.state_dict());replay={}
    if lr==.0003 and budget in [0,2400]:
        old=load(OLD/f'seed{seed}'/f'adapt_{condition}_256.pt')['model']
        replay=dict(bitwise_same=statehash(old)==finalhash,parameter_max_abs_difference=max(float((v.detach().cpu()-old[k]).abs().max()) for k,v in net.state_dict().items() if v.dtype!=torch.bool))
    seconds=time.monotonic()-start
    save(dest/'endpoint.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),config=CONFIG,seed=seed,step=800,lr=lr,condition=condition,budget=budget,metrics=curve[-1],**rng_state(rng)))
    write(dest/'completed.json',dict(run_id=rid,seed=seed,condition=condition,budget=budget,lr=lr,best=best_meta,endpoint=curve[-1],seconds=seconds,
        input_tokens=1433600,supervised_tokens=102400,updates=800,sample_exposures=51200,schedule_sha256=schedule_hash.hexdigest(),
        initialization=audit,final_model_state_hash=finalhash,replay_v1=replay,best_sha256=sha(dest/'best.pt'),endpoint_sha256=sha(dest/'endpoint.pt')))

def train():
    verify();start=time.monotonic()
    for seed in CONFIG['seeds']:
        for c in ['B','C']:extend(seed,c)
        for c,b in [('A',0),('B',2400),('C',2400),('B',4800),('C',4800)]:
            for lr in CONFIG['learning_rates']:adapt(seed,c,b,lr)
    write(OUT/'TRAINING_COMPLETE.json',dict(completed=True,unix=time.time(),invocation_seconds=time.monotonic()-start,test_evaluated=False))

def select():
    verify();assert (OUT/'TRAINING_COMPLETE.json').exists();selections=[];candidates=[];curves=[];audits=[]
    for seed in CONFIG['seeds']:
        for c,b in [('A',0),('B',2400),('C',2400),('B',4800),('C',4800)]:
            group=[]
            for rank,lr in enumerate(CONFIG['learning_rates']):
                dest=OUT/'adapt'/runid(seed,c,b,lr);m=read(dest/'completed.json');group.append((rank,dest,m))
                candidates.append(dict(seed=seed,condition=c,budget=b,lr=lr,best_step=m['best']['step'],best_dev_accuracy=m['best']['dev_accuracy'],best_dev_ce=m['best']['dev_answer_ce'],endpoint_dev_accuracy=m['endpoint']['dev_accuracy'],endpoint_dev_ce=m['endpoint']['dev_answer_ce'],seconds=m['seconds']))
                with (dest/'curve.csv').open() as f:curves.extend(list(csv.DictReader(f)))
                audits.append(m['initialization'])
            rank,dest,m=min(group,key=lambda x:(-x[2]['best']['dev_accuracy'],x[2]['best']['dev_answer_ce'],x[2]['best']['step'],x[0]))
            selections.append(dict(seed=seed,condition=c,budget=b,lr=m['lr'],step=m['best']['step'],dev_accuracy=m['best']['dev_accuracy'],dev_ce=m['best']['dev_answer_ce'],best_path=dest.relative_to(OUT).as_posix()+'/best.pt',endpoint_path=dest.relative_to(OUT).as_posix()+'/endpoint.pt',best_sha256=m['best_sha256'],endpoint_sha256=m['endpoint_sha256']))
    for seed in CONFIG['seeds']:
        sub=[r for r in audits if r['seed']==seed]
        assert len({r['target_row_hash'] for r in sub})==1
        assert len({r['train_sha256'] for r in sub})==len({r['dev_sha256'] for r in sub})==1
        schedules=[read(p)['schedule_sha256'] for p in (OUT/'adapt').glob(f's{seed}_*/completed.json')]
        assert len(schedules)==15 and len(set(schedules))==1
    assert len(candidates)==30
    csvwrite(OUT/'candidate_development_results.csv',candidates);csvwrite(OUT/'target_curves.csv',curves)
    write(OUT/'SELECTIONS_FROZEN.json',dict(protocol_sha256=sha(OUT/'FROZEN_PROTOCOL.json'),unix=time.time(),rule=CONFIG['selection'],selections=selections))
    print('All 10 selections frozen before new test generation.',flush=True)

def generate_test():
    assert (OUT/'SELECTIONS_FROZEN.json').exists();seen=set()
    for path in sorted((OLD/'data').glob('*.npz')):
        for p in np.load(path)['p']:seen.add(v1.key(p))
    original_count=len(seen);rng=np.random.default_rng(CONFIG['new_test_seed']);meta=[]
    for seed in CONFIG['seeds']:
        ps=[]
        while len(ps)<2048:
            p=tuple(map(int,rng.permutation(10)+1));k=v1.key(p)
            if k in seen:continue
            seen.add(k);ps.append(p)
        path=OUT/'data'/f's{seed}_test.npz';path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,p=np.asarray(ps,dtype=np.int16),y=np.asarray([v1.labels(p) for p in ps],dtype=np.int16))
        meta.append(dict(seed=seed,count=len(ps),sha256=sha(path),unique_inputs=len(set(ps)),unique_orbits=len({v1.key(p) for p in ps})))
    # Explicit expanded-set overlap check, separate from canonical-key rejection.
    excluded=set()
    for path in (OLD/'data').glob('*.npz'):
        for p in np.load(path)['p']:excluded.update(v1.orbit(p))
    for rec in meta:
        ps=np.load(OUT/'data'/f's{rec["seed"]}_test.npz')['p'];members=set().union(*(v1.orbit(p) for p in ps))
        assert not members.intersection(excluded);excluded.update(members)
    write(OUT/'test_data_audit.json',dict(created_unix=time.time(),selection_sha256=sha(OUT/'SELECTIONS_FROZEN.json'),excluded_original_orbits=original_count,overlap=0,pools=meta))

def test():
    verify();selection=read(OUT/'SELECTIONS_FROZEN.json')
    assert selection['protocol_sha256']==sha(OUT/'FROZEN_PROTOCOL.json')
    for selected in selection['selections']:
        for which in ['best','endpoint']:assert sha(OUT/selected[which+'_path'])==selected[which+'_sha256']
    if not (OUT/'test_data_audit.json').exists():generate_test()
    data_audit=read(OUT/'test_data_audit.json');assert data_audit['selection_sha256']==sha(OUT/'SELECTIONS_FROZEN.json')
    for rec in data_audit['pools']:assert sha(OUT/'data'/f's{rec["seed"]}_test.npz')==rec['sha256']
    rows=[];comparison=[]
    for seed in CONFIG['seeds']:
        d=np.load(OUT/'data'/f's{seed}_test.npz');data=(v1.tensors(d['p']),torch.tensor(d['y'],device=v1.DEVICE,dtype=torch.long))
        mode=int(torch.bincount(pool(seed,'target_train',256)[1][:,0]).argmax());baseline=float((data[1][:,0]==mode).float().mean())
        for c,b in [('A',0),('B',2400),('C',2400),('B',4800),('C',4800)]:
            selected=next(r for r in selection['selections'] if r['seed']==seed and r['condition']==c and r['budget']==b)
            for lr in CONFIG['learning_rates']:
                dest=OUT/'adapt'/runid(seed,c,b,lr);m=read(dest/'completed.json')
                for which in ['best','endpoint']:
                    path=dest/f'{which}.pt';assert sha(path)==m[which+'_sha256'];net,cp=from_checkpoint(path,seed)
                    row=dict(seed=seed,condition=c,pretrain_updates=b,lr=lr,checkpoint=which,adapt_step=cp['step'],dev_accuracy=cp['metrics']['dev_accuracy'],dev_ce=cp['metrics']['dev_answer_ce'],train_accuracy=cp['metrics']['train_accuracy'],**{'test_'+k:v for k,v in evaluate(net,data,0).items()},majority_accuracy=baseline,checkpoint_sha256=sha(path))
                    rows.append(row)
                    if lr==selected['lr']:
                        comparison.append(dict(protocol=('dev_selected' if which=='best' else 'selected_lr_endpoint'),**row))
                    if lr==.0003 and which=='endpoint':comparison.append(dict(protocol='replay_fixed_lr_endpoint',**row))
            if b in [0,2400]:
                path=OLD/f'seed{seed}'/f'adapt_{c}_256.pt';net,cp=from_checkpoint(path,seed);metrics=evaluate(net,data,0);devm=evaluate(net,pool(seed,'target_dev'),0);trm=evaluate(net,pool(seed,'target_train',256),0)
                comparison.append(dict(protocol='original_v1_endpoint_new_test',seed=seed,condition=c,pretrain_updates=b,lr=.0003,checkpoint='v1_endpoint',adapt_step=800,dev_accuracy=devm['accuracy'],dev_ce=devm['answer_ce'],train_accuracy=trm['accuracy'],**{'test_'+k:v for k,v in metrics.items()},majority_accuracy=baseline,checkpoint_sha256=sha(path)))
        csvwrite(OUT/f'results_seed{seed}.csv',[r for r in rows if r['seed']==seed])
    csvwrite(OUT/'results_all_candidates.csv',rows);csvwrite(OUT/'protocol_comparison.csv',comparison)
    write(OUT/'TEST_COMPLETE.json',dict(unix=time.time(),rows=len(rows),comparison_rows=len(comparison),selection_sha256=sha(OUT/'SELECTIONS_FROZEN.json')))

def preservation_check():
    for name,h in read(OUT/'v1_preservation.json').items():assert sha(HERE.parent.parent/name)==h,name

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['inventory','smoke','freeze','train','select','test']);a=p.parse_args()
    print(dict(stage=a.stage,device=str(v1.DEVICE)),flush=True);globals()[a.stage]()
