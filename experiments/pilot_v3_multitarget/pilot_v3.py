"""Finite, development-selected, multi-target PermWorld composition experiment."""
import argparse, copy, csv, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(HERE.parent/'permworld_pilot'))
import pilot as v1
OLD=HERE.parent/'permworld_pilot'/'runs'/'pilot_v1'
V2=HERE.parent/'pilot_v2_diagnostics'/'artifacts'
OUT=HERE/'artifacts'
TASKS=list(v1.PF)
T={t:i for i,t in enumerate(TASKS)}
TARGETS=['fixed_points','exceedances','left_to_right_minima','longest_decreasing_run']
PLANS={
 'fixed_points':dict(related='anti_fixed_points',candidates=['successions'],common=['descents','peaks','left_to_right_maxima'],transform='complement',reason='retain v1/v2 composition as historical anchor'),
 'exceedances':dict(related='deficiencies',candidates=['recoils'],common=['descents','peaks','successions'],transform='inverse',reason='Eulerian answer distribution; no claim of mathematical unrelatedness'),
 'left_to_right_minima':dict(related='right_to_left_minima',candidates=['cycle_count','longest_increasing_run'],common=['descents','peaks','successions'],transform='reverse',reason='record/cycle distributions and run-count alternative, selected on development only'),
 'longest_decreasing_run':dict(related='longest_increasing_run',candidates=['left_to_right_maxima','cycle_count'],common=['descents','peaks','successions'],transform='complement',reason='count distribution and development learning-curve matching')}
DESIGN_TASKS=sorted(set(TARGETS+[x for p in PLANS.values() for x in [p['related'],*p['common'],*p['candidates']]]))
CFG=dict(version='pilot_v3_multitarget/v1',seeds=[17,42,73],n=10,targets=TARGETS,plans=PLANS,design_tasks=DESIGN_TASKS,
 model=v1.CFG['model'],vocab_size=len(v1.VOCABULARY),encoding='one_line',source_pool=8192,source_dev=1024,target_samples=256,target_dev=1024,test_count=2048,
 batch=64,pretrain_updates=4800,source_lr='3e-4; warmup100 and cosine to3e-5 at2400, then constant3e-5 through4800',
 adaptation_updates=800,learning_rates=[.0003,.0001,.00003],adapt_eval_every=100,source_eval_steps=[0,600,1200,2400,3600,4800],
 weight_decay=.01,betas=[.9,.999],epsilon=1e-8,clip=1.,precision='FP32 weights/evaluation; BF16 training; TF32 disabled',
 target_gate='seed17 single-task source-dev endpoint accuracy>=.90 and >=training-majority dev accuracy+.15; no transfer outcomes',
 replacement_rule='min TV + abs(endpoint dev accuracy gap) + mean absolute dev accuracy gap at600,1200,2400,3600,4800; candidate order breaks ties',
 matching_design_seed=17,single_task_reference='same per-seed initial model and source train/dev;4800 updates;also retain1200 equal-task-exposure checkpoint',
 selection='max dev answer accuracy; min dev answer CE; earliest update; declared LR order; step0 excluded',
 split_rule='full inverse/reverse/complement orbit exclusion across old and new roles/seeds; within-seed pools intentionally shared across targets and conditions',
 reuse='fixed_points seeds17/42 reuse exact v2 B/C4800 and A/B/C adaptation candidates with hashes and compatibility checks',
 new_data_seed=2026101803,new_test_seed=2026101804,scope='four targets,256 samples,three seeds; no sample-efficiency curve or scaling law; stop after report')

def write(p,o):v1.write(p,o)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def statehash(s):
 h=hashlib.sha256()
 for k,v in sorted(s.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def csvwrite(p,rows):
 rows=list(rows);p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def csvread(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def save(p,o):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');torch.save(o,q);os.replace(q,p)
def reference(p):return os.path.relpath(p,OUT).replace('\\','/')
def resolve(s):return (OUT/s).resolve()
def labels(p):return [int(v1.PF[t](tuple(map(int,p)))) for t in TASKS]
def data(seed,role):
 d=np.load(OUT/'data'/f's{seed}_{role}.npz');return v1.tensors(d['p']),torch.tensor(d['y'],device=v1.DEVICE,dtype=torch.long)
def poolhash(seed,role):return sha(OUT/'data'/f's{seed}_{role}.npz')
def batch(pool,tasks,answers=True):
 x,y=pool;b=len(x);ts=np.broadcast_to(tasks,(b,)).copy();eq=x.shape[1]+1
 ids=torch.full((b,x.shape[1]+(4 if answers else 2)),v1.V['<PAD>'],device=v1.DEVICE,dtype=torch.long);ids[:,:x.shape[1]]=x
 ids[:,x.shape[1]]=torch.tensor([v1.V[f'<{TASKS[t].upper()}>'] for t in ts],device=v1.DEVICE);ids[:,eq]=v1.V['=']
 gold=y[torch.arange(b,device=v1.DEVICE),torch.as_tensor(ts,device=v1.DEVICE)]
 if answers:ids[:,-2]=gold;ids[:,-1]=v1.V['<EOS>']
 return ids,gold,eq
@torch.inference_mode()
def evaluate(net,pool,task):
 net.eval();x,y=pool;correct=eosgood=0;ce=eosce=0.
 for i in range(0,len(x),128):
  ids,gold,eq=batch((x[i:i+128],y[i:i+128]),T[task],False);z=net(ids)[:,eq].float();correct+=int((z.argmax(-1)==gold).sum());ce+=float(F.cross_entropy(z,gold,reduction='sum'))
  ez=net(torch.cat([ids,gold[:,None]],1))[:,-1].float();eg=torch.full_like(gold,v1.V['<EOS>']);eosgood+=int((ez.argmax(-1)==eg).sum());eosce+=float(F.cross_entropy(ez,eg,reduction='sum'))
 return dict(accuracy=correct/len(x),answer_ce=ce/len(x),eos_ce=eosce/len(x),two_token_ce=(ce+eosce)/(2*len(x)),eos_accuracy=eosgood/len(x),count=len(x))
def optmake(net,lr):return torch.optim.AdamW(net.parameters(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=.01,fused=v1.DEVICE.type=='cuda')
def rngstates(rng):return dict(numpy_rng=copy.deepcopy(rng.bit_generator.state),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
def restore(cp):
 rng=np.random.default_rng();rng.bit_generator.state=cp['numpy_rng'];torch.set_rng_state(cp['torch_rng'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in cp['cuda_rng']]);return rng
def source_lr(st):return .0003*(st/100 if st<=100 else .1+.9*.5*(1+math.cos(math.pi*(st-100)/2300))) if st<=2400 else .00003
def adapt_lr(lr,st):return lr*(st/100 if st<=100 else .1+.9*.5*(1+math.cos(math.pi*(st-100)/700)))
def step(net,opt,pool,tasks,rng,lr):
 net.train();x,y=pool;idx=rng.choice(len(x),64,replace=False);slots=np.tile(np.arange(len(tasks)),64//len(tasks));rng.shuffle(slots)
 ids,gold,eq=batch((x[idx],y[idx]),np.asarray([T[t] for t in tasks])[slots])
 for g in opt.param_groups:g['lr']=lr
 opt.zero_grad(set_to_none=True)
 with torch.autocast(v1.DEVICE.type,dtype=torch.bfloat16,enabled=v1.DEVICE.type=='cuda'):
  z=net(ids);loss=.5*(F.cross_entropy(z[:,eq],gold)+F.cross_entropy(z[:,eq+1],torch.full_like(gold,v1.V['<EOS>'])))
 assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
 return float(loss.detach()),float(norm),idx,slots
def initial(seed):
 cp=load(OUT/'initial'/f'seed{seed}.pt');net=v1.make(seed);net.load_state_dict(cp['model'],strict=True);assert statehash(net.state_dict())==statehash(cp['model']);return net,cp
def old_data_paths():return sorted((OLD/'data').glob('*.npz'))+sorted((V2/'data').glob('*.npz'))
def input_keys(paths):
 keys=set()
 for p in paths:
  for q in np.load(p)['p']:keys.add(v1.key(q))
 return keys
def preserve():
 paths=[p for folder in [HERE.parent/'permworld_pilot',HERE.parent/'pilot_v2_diagnostics'] for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
 paths+=[ROOT/'PILOT_REPORT.md']
 write(OUT/'preservation.json',{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)})
def preservation_check():
 for p,h in read(OUT/'preservation.json').items():assert sha(ROOT/p)==h,p
def prepare():
 assert not (OUT/'DATA_READY.json').exists();OUT.mkdir(parents=True,exist_ok=True);preserve();seen=input_keys(old_data_paths());original=len(seen);rng=np.random.default_rng(CFG['new_data_seed']);records=[]
 for seed in CFG['seeds']:
  for role,count,oldrole in [('source_train',8192,'pretrain'),('source_dev',1024,'source_dev'),('target_train',256,'target_train'),('target_dev',1024,'target_dev')]:
   if seed in [17,42]:ps=np.load(OLD/'data'/f's{seed}_{oldrole}.npz')['p'][:count];provenance='reused v1 inputs; all32 labels independently regenerated'
   else:
    ps=[]
    while len(ps)<count:
     p=tuple(map(int,rng.permutation(10)+1));k=v1.key(p)
     if k in seen:continue
     seen.add(k);ps.append(p)
    ps=np.asarray(ps,dtype=np.int16);provenance='new; excludes all v1/v2 full orbits'
   path=OUT/'data'/f's{seed}_{role}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,p=ps,y=np.asarray([labels(p) for p in ps],dtype=np.int16))
   records.append(dict(seed=seed,role=role,length=10,count=count,sha256=sha(path),input_sha256=hashlib.sha256(ps.tobytes()).hexdigest(),provenance=provenance))
  if seed in [17,42]:cp=load(OLD/f'seed{seed}'/'step0.pt');origin=str(OLD/f'seed{seed}'/'step0.pt')
  else:net=v1.make(seed);cp=dict(model=net.state_dict());origin='new random initialization'
  save(OUT/'initial'/f'seed{seed}.pt',dict(model=cp['model'],seed=seed,origin=origin))
 # Independent expanded-orbit check across v3 roles and seeds; target reuse across task trials is intentional.
 expanded=set()
 for r in records:
  p=np.load(OUT/'data'/f's{r["seed"]}_{r["role"]}.npz')['p'];members=set().union(*(v1.orbit(q) for q in p));assert not expanded.intersection(members);expanded.update(members)
 write(OUT/'DATA_READY.json',dict(pools=records,initials={str(s):sha(OUT/'initial'/f'seed{s}.pt') for s in CFG['seeds']},role_seed_orbit_overlap=0,historical_excluded_orbits_for_new_data=original,task_condition_reuse=True));write(OUT/'config_plan.json',CFG)
 print('Prepared v3 data and preservation hashes.',flush=True)
def smoke():
 ready=read(OUT/'DATA_READY.json')
 if 'initials' not in ready:
  ready['initials']={str(s):sha(OUT/'initial'/f'seed{s}.pt') for s in CFG['seeds']};write(OUT/'DATA_READY.json',ready)
 assert v1.DEVICE.type=='cuda';net,_=initial(17);pool=data(17,'source_train');tiny=(pool[0][:64],pool[1][:64]);opt=optmake(net,.0003);v1.seed(901);rng=np.random.default_rng(901)
 for st in range(1,5):step(net,opt,tiny,PLANS['fixed_points']['common']+['anti_fixed_points'],rng,.0003)
 path=OUT/'smoke'/'resume.pt';save(path,dict(model=net.state_dict(),optimizer=opt.state_dict(),**rngstates(rng)))
 for st in range(5,9):step(net,opt,tiny,PLANS['fixed_points']['common']+['anti_fixed_points'],rng,.0003)
 expected=statehash(net.state_dict());cp=load(path);n2,_=initial(17);n2.load_state_dict(cp['model']);o2=optmake(n2,.0003);o2.load_state_dict(cp['optimizer']);r2=restore(cp)
 for st in range(5,9):step(n2,o2,tiny,PLANS['fixed_points']['common']+['anti_fixed_points'],r2,.0003)
 assert expected==statehash(n2.state_dict());before=torch.cuda.get_rng_state_all();evaluate(n2,tiny,'fixed_points');assert all(torch.equal(a,b) for a,b in zip(before,torch.cuda.get_rng_state_all()))
 ids,gold,eq=batch(tiny,T['fixed_points']);n2.eval()
 with torch.no_grad():
  z=n2(ids)[:,eq];alter=ids.clone();alter[:,eq+1]=(gold+1)%10;assert float((z-n2(alter)[:,eq]).abs().max())==0
 for target in TARGETS:
  p=(4,2,9,1,7,5,10,3,8,6);inverse=tuple(p.index(i)+1 for i in range(1,11));q=(tuple(11-v for v in p) if PLANS[target]['transform']=='complement' else p[::-1] if PLANS[target]['transform']=='reverse' else inverse)
  assert v1.PF[target](p)==v1.PF[PLANS[target]['related']](q)
 fresh=optmake(n2,.0003);assert len(fresh.state)==0;step(n2,fresh,tiny,['fixed_points'],np.random.default_rng(902),.0003)
 write(OUT/'SMOKE.json',dict(status='passed',resume_bitwise_equal=True,evaluation_preserves_rng=True,causal_no_answer_leak=True,fresh_optimizer=True))
 assert read(HERE/'catalog'/'CATALOG_AUDIT.json')
 files=[HERE/'pilot_v3.py',HERE/'TASK_CATALOG.csv',HERE/'catalog'/'RELATION_AUDIT.csv',HERE/'catalog'/'REPLACEMENT_RELATION_COUNTEREXAMPLES.csv',HERE/'catalog'/'build_catalog.py',HERE.parent/'permworld_pilot'/'pilot.py',*sorted((HERE.parent/'permworld_pilot'/'vendor').rglob('*.py'))]
 write(OUT/'DESIGN_FROZEN.json',dict(config=CFG,unix=time.time(),code={p.relative_to(ROOT).as_posix():sha(p) for p in files},data_sha256=sha(OUT/'DATA_READY.json')))
 print('Smoke passed; candidate/design plan frozen before single-task development.',flush=True)
def verify():
 f=read(OUT/'DESIGN_FROZEN.json');assert f['config']==CFG;assert f['data_sha256']==sha(OUT/'DATA_READY.json')
 for p,h in f['code'].items():assert sha(ROOT/p)==h,p
 ready=read(OUT/'DATA_READY.json')
 for r in ready['pools']:assert sha(OUT/'data'/f's{r["seed"]}_{r["role"]}.npz')==r['sha256']
 for s,h in ready['initials'].items():assert sha(OUT/'initial'/f'seed{s}.pt')==h
def source_measure(net,train,dev,tasks,seed,kind,step_id,tag):
 rows=[]
 for task in tasks:
  mode=int(torch.bincount(train[1][:,T[task]]).argmax())
  for split,d in [('train',train),('dev',dev)]:rows.append(dict(seed=seed,kind=kind,run_id=tag,task=task,step=step_id,length=10,split=split,**evaluate(net,d,task),majority_label=mode,majority_accuracy=float((d[1][:,T[task]]==mode).float().mean())))
 return rows
def train_sources(seed,tasks,folder,kind):
 if (folder/'completed.json').exists():return
 net,cp=initial(seed);train=data(seed,'source_train');dev=data(seed,'source_dev');opt=optmake(net,.0003);assert not opt.state
 v1.seed(seed+1000);rng=np.random.default_rng(seed+1000);curve=[];schedule=hashlib.sha256();seen=[set() for _ in tasks];start=time.monotonic()
 curve.extend(source_measure(net,train,dev,tasks,seed,kind,0,folder.name))
 for st in range(1,4801):
  loss,norm,idx,slots=step(net,opt,train,tasks,rng,source_lr(st));schedule.update(idx.tobytes());schedule.update(slots.tobytes())
  for j in range(len(tasks)):seen[j].update(map(int,idx[slots==j]))
  if st in CFG['source_eval_steps']:
   curve.extend(source_measure(net,train,dev,tasks,seed,kind,st,folder.name));csvwrite(folder/'curve.csv',curve)
   print(dict(stage=kind,run=folder.name,step=st,loss=loss,dev={r['task']:r['accuracy'] for r in curve if r['step']==st and r['split']=='dev'},seconds=time.monotonic()-start),flush=True)
  if st in [1200,2400,4800]:save(folder/f'step{st}.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),step=st,tasks=tasks,seed=seed,config=CFG,**rngstates(rng)))
 assert {int(v['step']) for v in opt.state.values()}=={4800}
 meta=dict(seed=seed,tasks=tasks,kind=kind,updates=4800,input_tokens=4800*64*28,supervised_tokens=4800*64*2,sample_exposures=4800*64,task_exposures=4800*64//len(tasks),actual_unique_per_task=dict(zip(tasks,map(len,seen))),task_fraction=1/len(tasks),schedule_sha256=schedule.hexdigest(),model_state_hash=statehash(net.state_dict()),seconds=time.monotonic()-start,new_compute=True,checkpoint=reference(folder/'step4800.pt'),checkpoint_sha256=sha(folder/'step4800.pt'),initial_state_hash=statehash(cp['model']),train_sha256=poolhash(seed,'source_train'),dev_sha256=poolhash(seed,'source_dev'))
 write(folder/'completed.json',meta)
def design_references():
 verify()
 for task in DESIGN_TASKS:train_sources(17,[task],OUT/'single'/f's17_{task}','single')
 write(OUT/'DESIGN_REFERENCES_COMPLETE.json',dict(unix=time.time(),tasks=DESIGN_TASKS))
def freeze():
 verify();assert (OUT/'DESIGN_REFERENCES_COMPLETE.json').exists();refs={task:csvread(OUT/'single'/f's17_{task}'/'curve.csv') for task in DESIGN_TASKS};checks=[];matching=[];groups=[]
 train=data(17,'source_train')[1].cpu().numpy()
 def metric(task,st):return next(r for r in refs[task] if int(r['step'])==st and r['split']=='dev')
 for target in TARGETS:
  r=metric(target,4800);passed=float(r['accuracy'])>=.9 and float(r['accuracy'])>=float(r['majority_accuracy'])+.15;checks.append(dict(target=target,accuracy=float(r['accuracy']),majority_accuracy=float(r['majority_accuracy']),pass_gate=passed))
  p=PLANS[target];a=p['related'];hist_a=np.bincount(train[:,T[a]],minlength=11)/len(train);scored=[]
  for rank,b in enumerate(p['candidates']):
   hist_b=np.bincount(train[:,T[b]],minlength=11)/len(train);tv=float(np.abs(hist_a-hist_b).sum()/2);endpoint=abs(float(metric(a,4800)['accuracy'])-float(metric(b,4800)['accuracy']));curve_gap=float(np.mean([abs(float(metric(a,st)['accuracy'])-float(metric(b,st)['accuracy'])) for st in CFG['source_eval_steps'] if st]));score=tv+endpoint+curve_gap
   row=dict(target=target,related=a,replacement=b,tv=tv,endpoint_dev_accuracy_gap=endpoint,mean_curve_accuracy_gap=curve_gap,score=score,candidate_order=rank,selected=False);matching.append(row);scored.append(row)
  chosen=min(scored,key=lambda x:(x['score'],x['candidate_order']));chosen['selected']=True
  sources={'B':[a,*p['common']],'C':[chosen['replacement'],*p['common']]}
  for ts in sources.values():assert len(set(ts))==4 and target not in ts and not set(ts).intersection(TARGETS)
  groups.append(dict(target=target,sources=sources,related=a,replacement=chosen['replacement'],transform=p['transform'],answer_map='identity',selection_reason=p['reason']))
 csvwrite(OUT/'target_development_gates.csv',checks);csvwrite(OUT/'replacement_matching.csv',matching)
 assert all(r['pass_gate'] for r in checks),'Prespecified target development gate failed; retain results, no transfer/test selection.'
 write(OUT/'FROZEN_PROTOCOL.json',dict(config=CFG,unix=time.time(),groups=groups,target_gates=checks,design_sha256=sha(OUT/'DESIGN_FROZEN.json'),matching_sha256=sha(OUT/'replacement_matching.csv'),single_refs={t:sha(OUT/'single'/f's17_{t}'/'completed.json') for t in DESIGN_TASKS}))
 print('Four targets and all compositions frozen using single-task development only.',flush=True)
def frozen():
 verify();f=read(OUT/'FROZEN_PROTOCOL.json');assert f['design_sha256']==sha(OUT/'DESIGN_FROZEN.json');assert f['matching_sha256']==sha(OUT/'replacement_matching.csv')
 for t,h in f['single_refs'].items():assert sha(OUT/'single'/f's17_{t}'/'completed.json')==h
 return f
def complete_references():
 f=frozen();tasks=sorted(set(t for g in f['groups'] for ts in g['sources'].values() for t in ts))
 for seed in [42,73]:
  for task in tasks:train_sources(seed,[task],OUT/'single'/f's{seed}_{task}','single')
 write(OUT/'SINGLE_REFERENCES_COMPLETE.json',dict(unix=time.time(),source_tasks=tasks,design_seed_target_references_only=True))
def reuse_source(seed,c,tasks,folder):
 path=V2/'extended'/f'seed{seed}_{c}_4800.pt';cp=load(path);meta=read(path.with_suffix('.json'));assert cp['step']==4800 and cp['config']['model']==CFG['model'];assert tasks==[v1.TASKS[t] for t in v1.SOURCES[c]]
 for role,oldrole in [('source_train','pretrain'),('source_dev','source_dev')]:assert np.array_equal(np.load(OUT/'data'/f's{seed}_{role}.npz')['p'],np.load(OLD/'data'/f's{seed}_{oldrole}.npz')['p'])
 rows=[]
 for r in csvread(V2/'source_learning_2400.csv')+csvread(path.with_suffix('.csv')):
  if int(r['seed'])==seed and r['condition']==c:rows.append(dict(seed=seed,kind='multi',run_id=folder.name,task=r['task'],step=r['step'],length=10,split=r['split'],**{k:r[k] for k in ['accuracy','answer_ce','eos_ce','two_token_ce','eos_accuracy','count','majority_label','majority_accuracy']}))
 csvwrite(folder/'curve.csv',rows)
 oldmeta=read(OLD/f'seed{seed}'/f'pretrain_{c}.json');schedule=hashlib.sha256();rng=np.random.default_rng(seed+1000);seen=[set() for _ in tasks]
 for _ in range(4800):
  idx=rng.choice(8192,64,replace=False);slots=np.tile(np.arange(4),16);rng.shuffle(slots);schedule.update(idx.tobytes());schedule.update(slots.tobytes())
  for j in range(4):seen[j].update(map(int,idx[slots==j]))
 assert rng.bit_generator.state==cp['numpy_rng'];assert {int(v['step']) for v in cp['optimizer']['state'].values()}=={4800}
 write(folder/'completed.json',dict(seed=seed,tasks=tasks,kind='multi',updates=4800,input_tokens=8601600,supervised_tokens=614400,sample_exposures=307200,task_exposures=76800,task_fraction=.25,actual_unique_per_task=dict(zip(tasks,map(len,seen))),schedule_sha256=schedule.hexdigest(),model_state_hash=statehash(cp['model']),seconds=oldmeta['seconds']+meta['seconds'],new_compute=False,checkpoint=reference(path),checkpoint_sha256=sha(path),initial_state_hash=statehash(load(OUT/'initial'/f'seed{seed}.pt')['model']),train_sha256=poolhash(seed,'source_train'),dev_sha256=poolhash(seed,'source_dev'),missing_historical_curves='before2400 full source metrics unavailable; original checkpoints retained'))
def pretrain():
 f=frozen()
 for seed in CFG['seeds']:
  for g in f['groups']:
   for c in ['B','C']:
    folder=OUT/'multi'/f's{seed}_{g["target"]}_{c}'
    if (folder/'completed.json').exists():continue
    if seed in [17,42] and g['target']=='fixed_points':reuse_source(seed,c,g['sources'][c],folder)
    else:train_sources(seed,g['sources'][c],folder,'multi')
 write(OUT/'PRETRAIN_COMPLETE.json',dict(unix=time.time()))
def runid(seed,target,c,lr):return f's{seed}_{target}_{c}_lr{lr:.0e}'
def reuse_adapt(seed,target,c,lr,folder):
 b=0 if c=='A' else 4800;oldfolder=V2/'adapt'/f's{seed}_{c}_p{b}_lr{lr:.0e}';m=read(oldfolder/'completed.json')
 for role in ['target_train','target_dev']:assert np.array_equal(np.load(OUT/'data'/f's{seed}_{role}.npz')['p'],np.load(OLD/'data'/f's{seed}_{role}.npz')['p'][:CFG['target_samples']] if role=='target_train' else np.load(OLD/'data'/f's{seed}_{role}.npz')['p'])
 rows=[]
 for r in csvread(oldfolder/'curve.csv'):r.update(target=target);rows.append(r)
 csvwrite(folder/'curve.csv',rows)
 for which in ['best','endpoint']:assert sha(oldfolder/f'{which}.pt')==m[which+'_sha256']
 # Original v2 adaptation reset precisely this task's existing tied row and used these source weights.
 m.update(target=target,new_compute=False,run_id=folder.name,best_path=reference(oldfolder/'best.pt'),endpoint_path=reference(oldfolder/'endpoint.pt'),reused_from=reference(oldfolder),v3_train_sha256=poolhash(seed,'target_train'),v3_dev_sha256=poolhash(seed,'target_dev'))
 write(folder/'completed.json',m)
def adapt_one(seed,target,c,lr):
 folder=OUT/'adapt'/runid(seed,target,c,lr)
 if (folder/'completed.json').exists():return
 if seed in [17,42] and target=='fixed_points':reuse_adapt(seed,target,c,lr,folder);return
 net,initial_cp=initial(seed);source_path=OUT/'initial'/f'seed{seed}.pt' if c=='A' else resolve(read(OUT/'multi'/f's{seed}_{target}_{c}'/'completed.json')['checkpoint']);source=load(source_path);net.load_state_dict(source['model'],strict=True);loaded=statehash(net.state_dict());assert loaded==statehash(source['model'])
 token=v1.V[f'<{target.upper()}>']
 with torch.no_grad():net.token_embedding.weight[token].copy_(initial_cp['model']['token_embedding.weight'][token].to(v1.DEVICE))
 assert net.lm_head.weight.data_ptr()==net.token_embedding.weight.data_ptr();assert torch.equal(net.token_embedding.weight[token].cpu(),initial_cp['model']['token_embedding.weight'][token])
 rowhash=hashlib.sha256(net.token_embedding.weight[token].detach().cpu().numpy().tobytes()).hexdigest();opt=optmake(net,lr);assert not opt.state
 audit=dict(source_path=reference(source_path),source_sha256=sha(source_path),source_state_hash=loaded,loaded_state_hash=loaded,adapt_initial_state_hash=statehash(net.state_dict()),target_row_hash=rowhash,reset_row_ids=[token],optimizer_entries=0,optimizer_state_reset=True,train_sha256=poolhash(seed,'target_train'),dev_sha256=poolhash(seed,'target_dev'),new_output_head=False)
 train=data(seed,'target_train');dev=data(seed,'target_dev');v1.seed(seed+2000);rng=np.random.default_rng(seed+2000);curve=[];best=None;schedule=hashlib.sha256();start=time.monotonic();mode=int(torch.bincount(train[1][:,T[target]]).argmax())
 def measure(st,loss=None,norm=None):
  r=dict(seed=seed,target=target,condition=c,lr=lr,step=st,last_batch_two_token_loss=loss,gradient_norm=norm)
  for split,d in [('train',train),('dev',dev)]:r.update({split+'_'+k:v for k,v in evaluate(net,d,target).items()});r[split+'_majority_accuracy']=float((d[1][:,T[target]]==mode).float().mean())
  return r
 curve.append(measure(0))
 for st in range(1,801):
  loss,norm,idx,slots=step(net,opt,train,[target],rng,adapt_lr(lr,st));schedule.update(idx.tobytes());schedule.update(slots.tobytes())
  if st%100==0:
   r=measure(st,loss,norm);curve.append(r)
   if best is None or (-r['dev_accuracy'],r['dev_answer_ce'],st)<(-best['dev_accuracy'],best['dev_answer_ce'],best['step']):
    best=r.copy();save(folder/'best.pt',dict(model=net.state_dict(),config=CFG,metrics=r,step=st,seed=seed,target=target,condition=c,lr=lr))
   csvwrite(folder/'curve.csv',curve)
 assert {int(v['step']) for v in opt.state.values()}=={800}
 save(folder/'endpoint.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),config=CFG,metrics=curve[-1],step=800,seed=seed,target=target,condition=c,lr=lr,**rngstates(rng)))
 meta=dict(run_id=folder.name,seed=seed,target=target,condition=c,lr=lr,best=best,endpoint=curve[-1],seconds=time.monotonic()-start,updates=800,input_tokens=1433600,supervised_tokens=102400,sample_exposures=51200,schedule_sha256=schedule.hexdigest(),initialization=audit,final_model_state_hash=statehash(net.state_dict()),best_path=reference(folder/'best.pt'),endpoint_path=reference(folder/'endpoint.pt'),best_sha256=sha(folder/'best.pt'),endpoint_sha256=sha(folder/'endpoint.pt'),new_compute=True,v3_train_sha256=poolhash(seed,'target_train'),v3_dev_sha256=poolhash(seed,'target_dev'))
 write(folder/'completed.json',meta);print(dict(stage='adapt',run=folder.name,best_dev=best['dev_accuracy'],best_step=best['step'],endpoint_dev=curve[-1]['dev_accuracy'],seconds=meta['seconds']),flush=True)
def adapt():
 frozen();assert (OUT/'PRETRAIN_COMPLETE.json').exists()
 for seed in CFG['seeds']:
  for target in TARGETS:
   for c in ['A','B','C']:
    for lr in CFG['learning_rates']:adapt_one(seed,target,c,lr)
 write(OUT/'ADAPT_COMPLETE.json',dict(unix=time.time()))
def select():
 frozen();assert (OUT/'ADAPT_COMPLETE.json').exists();selected=[];curves=[];candidates=[]
 for seed in CFG['seeds']:
  for target in TARGETS:
   for c in ['A','B','C']:
    group=[read(OUT/'adapt'/runid(seed,target,c,lr)/'completed.json') for lr in CFG['learning_rates']]
    for m in group:
     curves+=csvread(OUT/'adapt'/m['run_id']/'curve.csv');candidates.append(dict(seed=seed,target=target,condition=c,lr=m['lr'],best_step=m['best']['step'],best_dev=m['best']['dev_accuracy'],best_dev_ce=m['best']['dev_answer_ce'],endpoint_dev=m['endpoint']['dev_accuracy'],new_compute=m['new_compute']))
    m=min(group,key=lambda m:(-m['best']['dev_accuracy'],m['best']['dev_answer_ce'],m['best']['step'],CFG['learning_rates'].index(m['lr'])))
    selected.append(dict(seed=seed,target=target,condition=c,lr=m['lr'],step=m['best']['step'],dev_accuracy=m['best']['dev_accuracy'],dev_ce=m['best']['dev_answer_ce'],**{k:m[k] for k in ['best_path','endpoint_path','best_sha256','endpoint_sha256']}))
 assert len(selected)==36 and len(candidates)==108
 write(OUT/'SELECTIONS_FROZEN.json',dict(unix=time.time(),protocol_sha256=sha(OUT/'FROZEN_PROTOCOL.json'),selections=selected));csvwrite(OUT/'target_curves.csv',curves);csvwrite(OUT/'candidate_development_results.csv',candidates)
 print('All36 development selections frozen. No v3 test generated or evaluated.',flush=True)
def generate_test():
 assert (OUT/'SELECTIONS_FROZEN.json').exists();paths=old_data_paths()+sorted((OUT/'data').glob('*.npz'));seen=input_keys(paths);original=len(seen);rng=np.random.default_rng(CFG['new_test_seed']);records=[];expanded=set()
 for path in paths:
  for p in np.load(path)['p']:expanded.update(v1.orbit(p))
 for seed in CFG['seeds']:
  ps=[]
  while len(ps)<2048:
   p=tuple(map(int,rng.permutation(10)+1));k=v1.key(p)
   if k in seen:continue
   seen.add(k);ps.append(p)
  members=set().union(*(v1.orbit(p) for p in ps));assert not members.intersection(expanded);expanded.update(members)
  path=OUT/'data'/f's{seed}_test.npz';np.savez_compressed(path,p=np.asarray(ps,dtype=np.int16),y=np.asarray([labels(p) for p in ps],dtype=np.int16));records.append(dict(seed=seed,count=2048,sha256=sha(path),unique_orbits=len({v1.key(p) for p in ps})))
 write(OUT/'test_data_audit.json',dict(unix=time.time(),selection_sha256=sha(OUT/'SELECTIONS_FROZEN.json'),excluded_orbits=original,overlap=0,pools=records,shared_across_targets=True))
def test():
 f=frozen();s=read(OUT/'SELECTIONS_FROZEN.json');assert s['protocol_sha256']==sha(OUT/'FROZEN_PROTOCOL.json')
 for r in s['selections']:
  for which in ['best','endpoint']:assert sha(resolve(r[which+'_path']))==r[which+'_sha256']
 if not (OUT/'test_data_audit.json').exists():generate_test()
 audit=read(OUT/'test_data_audit.json');assert audit['selection_sha256']==sha(OUT/'SELECTIONS_FROZEN.json')
 for rec in audit['pools']:assert sha(OUT/'data'/f's{rec["seed"]}_test.npz')==rec['sha256']
 rows=[]
 for seed in CFG['seeds']:
  pool=data(seed,'test');train=data(seed,'target_train')
  for target in TARGETS:
   mode=int(torch.bincount(train[1][:,T[target]]).argmax());baseline=float((pool[1][:,T[target]]==mode).float().mean())
   for c in ['A','B','C']:
    choice=next(r for r in s['selections'] if r['seed']==seed and r['target']==target and r['condition']==c)
    for lr in CFG['learning_rates']:
     m=read(OUT/'adapt'/runid(seed,target,c,lr)/'completed.json')
     if lr==choice['lr']:
      for which in ['best','endpoint']:assert m[which+'_path']==choice[which+'_path'] and m[which+'_sha256']==choice[which+'_sha256']
     for which in ['best','endpoint']:
      p=resolve(m[which+'_path']);assert sha(p)==m[which+'_sha256'];cp=load(p);net,_=initial(seed);net.load_state_dict(cp['model'],strict=True);metric=evaluate(net,pool,target)
      rows.append(dict(seed=seed,target=target,condition=c,lr=lr,checkpoint=which,adapt_step=cp['step'],selected_lr=lr==choice['lr'],selected_primary=lr==choice['lr'] and which=='best',dev_accuracy=m[which]['dev_accuracy'],dev_ce=m[which]['dev_answer_ce'],train_accuracy=m[which]['train_accuracy'],**{'test_'+k:v for k,v in metric.items()},majority_accuracy=baseline,checkpoint_sha256=sha(p),new_compute=m['new_compute']))
   print(dict(stage='test',seed=seed,target=target,selected={r['condition']:r['test_accuracy'] for r in rows if r['seed']==seed and r['target']==target and r['selected_primary']}),flush=True)
  csvwrite(OUT/f'results_seed{seed}.csv',[r for r in rows if r['seed']==seed])
 csvwrite(OUT/'results_all_candidates.csv',rows);csvwrite(OUT/'results_selected.csv',[r for r in rows if r['selected_primary']]);write(OUT/'TEST_COMPLETE.json',dict(unix=time.time(),candidate_rows=len(rows),selected_rows=36,selection_sha256=sha(OUT/'SELECTIONS_FROZEN.json')))

STAGES=[('prepare','DATA_READY.json'),('smoke','SMOKE.json'),('design_references','DESIGN_REFERENCES_COMPLETE.json'),('freeze','FROZEN_PROTOCOL.json'),('complete_references','SINGLE_REFERENCES_COMPLETE.json'),('pretrain','PRETRAIN_COMPLETE.json'),('adapt','ADAPT_COMPLETE.json'),('select','SELECTIONS_FROZEN.json'),('test','TEST_COMPLETE.json')]
def main():
 global OUT
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['all',*[x[0] for x in STAGES]]);p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();OUT=a.out.resolve()
 for stage,marker in STAGES:
  if a.stage not in ['all',stage]:continue
  if (OUT/marker).exists():print('Skip completed',stage,flush=True);continue
  started=time.monotonic();print('Start',stage,flush=True)
  try:globals()[stage]()
  except Exception:
   import traceback
   write(OUT/f'FAILURE_{stage}_{int(time.time())}.json',dict(stage=stage,traceback=traceback.format_exc()));raise
  write(OUT/f'timing_{stage}.json',dict(stage=stage,seconds=time.monotonic()-started));print('Finished',stage,time.monotonic()-started,flush=True)
 if (OUT/'TEST_COMPLETE.json').exists():preservation_check();print('V1/v2 byte preservation passed.',flush=True)
if __name__=='__main__':main()
