"""Small, paired PermWorld transfer pilot. Run stages in README order."""
import argparse, copy, csv, hashlib, itertools, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
from neurips_permutations.math_ops import PROPERTY_FUNCTIONS as PF
from neurips_permutations.models import CausalTransformer
from neurips_permutations.passage import VOCABULARY, TOKEN_TO_ID as V, one_line_tokens
from sdpa import accelerate

TASKS = ['fixed_points', 'anti_fixed_points', 'successions', 'descents', 'peaks', 'left_to_right_maxima']
SOURCES = {'B': [1,3,4,5], 'C': [2,3,4,5]}
CFG = dict(n=10, seeds=[17,42], sizes=[64,256,1024], batch=64,
           model=dict(d_model=256,layers=4,n_heads=8,mlp_ratio=4,dropout=.1,tie_embeddings=True,max_seq_len=128),
           lr=.0003, weight_decay=.01, pre_steps=2400, adapt_steps=800, dev_steps=2400,
           train_count=8192, dev_count=1024, test_count=2048,
           split_rule='Reject overlap of complete inverse/reverse/complement orbits across all pools and seeds.',
           endpoint='fixed final update, no best checkpoint selection')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def make(s):
    seed(s)
    return accelerate(CausalTransformer(vocab_size=len(VOCABULARY),**CFG['model'])).to(DEVICE)

def orbit(p):
    p=tuple(p); inv=[0]*len(p)
    for i,v in enumerate(p,1): inv[v-1]=i
    ans=set()
    for q in [p,tuple(inv)]:
        for r in [q,q[::-1]]:
            ans.add(r); ans.add(tuple(len(p)+1-v for v in r))
    return ans

def key(p): return min(orbit(p))
def labels(p): return [int(PF[t](tuple(map(int,p)))) for t in TASKS]
def tensors(p):
    ids=[[V[t] for t in ('<BOS>','<SIZE>',f'{len(q):02d}',*one_line_tokens(tuple(map(int,q))))] for q in p]
    return torch.tensor(ids,dtype=torch.long,device=DEVICE)

def batch(prefix,y,tasks,answers=True):
    b=len(prefix); tasks=np.broadcast_to(tasks,(b,)).copy()
    ids=torch.full((b,prefix.shape[1]+(4 if answers else 2)),V['<PAD>'],device=DEVICE,dtype=torch.long)
    ids[:,:prefix.shape[1]]=prefix
    ids[:,prefix.shape[1]]=torch.tensor([V[f'<{TASKS[t].upper()}>'] for t in tasks],device=DEVICE)
    ids[:,prefix.shape[1]+1]=V['=']
    gold=y[torch.arange(b,device=DEVICE),torch.as_tensor(tasks,device=DEVICE)]
    if answers:
        ids[:,-2]=gold; ids[:,-1]=V['<EOS>']
    return ids,gold,prefix.shape[1]+1

def loadpool(root,name):
    d=np.load(root/'data'/f'{name}.npz')
    return tensors(d['p']),torch.tensor(d['y'],dtype=torch.long,device=DEVICE)

@torch.inference_mode()
def evaluate(net,pool,task):
    net.eval(); x,y=pool; good=eos=0; ce=0.
    for start in range(0,len(x),128):
        ids,gold,eq=batch(x[start:start+128],y[start:start+128],task,False)
        z=net(ids)[:,eq].float(); good+=int((z.argmax(-1)==gold).sum())
        ce+=float(F.cross_entropy(z,gold,reduction='sum'))
        ids=torch.cat([ids,gold[:,None]],1)
        eos+=int((net(ids)[:,-1].argmax(-1)==V['<EOS>']).sum())
    return dict(accuracy=good/len(x),ce=ce/len(x),eos_accuracy=eos/len(x),count=len(x))

def train(net,pool,tasks,steps,s,out,dev=None):
    seed(s); x,y=pool; rng=np.random.default_rng(s); opt=torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'],betas=(.9,.999),eps=1e-8,fused=DEVICE.type=='cuda')
    assert not opt.state
    start=time.monotonic(); curve=[]; last=0.; tokens=0
    if DEVICE.type=='cuda': torch.cuda.reset_peak_memory_stats()
    # Exact equal task counts every batch; identical shuffled slots for B/C.
    for step in range(1,steps+1):
        net.train(); idx=rng.choice(len(x),CFG['batch'],replace=False)
        slot=np.tile(np.arange(len(tasks)),CFG['batch']//len(tasks)); rng.shuffle(slot)
        ti=np.asarray(tasks)[slot]; ids,gold,eq=batch(x[idx],y[idx],ti)
        factor=min(step/100,1.) if step<=100 else .1+.9*.5*(1+math.cos(math.pi*(step-100)/max(1,steps-100)))
        for g in opt.param_groups:g['lr']=CFG['lr']*factor
        opt.zero_grad(set_to_none=True)
        with torch.autocast(DEVICE.type,dtype=torch.bfloat16,enabled=DEVICE.type=='cuda'):
            z=net(ids); loss=.5*(F.cross_entropy(z[:,eq],gold)+F.cross_entropy(z[:,eq+1],torch.full_like(gold,V['<EOS>'])))
        assert torch.isfinite(loss); loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True); opt.step()
        last=float(loss.detach()); tokens+=ids.numel()
        if step%400==0 or step==steps:
            row=dict(step=step,loss=last,elapsed_seconds=time.monotonic()-start)
            if dev is not None: row.update(evaluate(net,dev,tasks[0]))
            curve.append(row); print(out.name,row,flush=True)
    seconds=time.monotonic()-start
    out.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(model=net.state_dict(),optimizer=opt.state_dict(),config=CFG,tasks=tasks,step=steps,seed=s,
                 torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if DEVICE.type=='cuda' else [],
                 numpy_rng=rng.bit_generator.state,curve=curve,seconds=seconds,input_tokens=tokens,
                 supervised_tokens=2*steps*CFG['batch'],sample_exposures=steps*CFG['batch'],
                 peak_memory=torch.cuda.max_memory_allocated() if DEVICE.type=='cuda' else None)
    torch.save(payload,out); write(out.with_suffix('.json'),{k:v for k,v in payload.items() if k not in ['model','optimizer','torch_rng','cuda_rng']})
    return payload

def prepare(root):
    assert not (root/'data').exists(), 'Existing data are immutable; use another --out.'
    seen=set(); rng=np.random.default_rng(20261018); audit={}; pools={}
    specs=[('development_train',8192),('development_dev',1024)]
    for s in CFG['seeds']:
        specs += [(f's{s}_pretrain',CFG['train_count']),(f's{s}_source_dev',1024),(f's{s}_target_train',1024),(f's{s}_target_dev',CFG['dev_count']),(f's{s}_test',CFG['test_count'])]
    for name,count in specs:
        ps=[]
        while len(ps)<count:
            p=tuple(map(int,rng.permutation(CFG['n'])+1)); k=key(p)
            if k in seen: continue
            seen.add(k); ps.append(p)
        p=np.array(ps,dtype=np.int16); y=np.array([labels(q) for q in p],dtype=np.int16)
        path=root/'data'/f'{name}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,p=p,y=y)
        audit[name]=dict(count=count,sha256=sha(path),unique_inputs=len(set(ps)),unique_orbits=len({key(q) for q in ps}))
    # Independent exact overlap audit, including every transform rather than only keys.
    all_inputs=set(); full_orbits=set()
    for name,_ in specs:
        p=np.load(root/'data'/f'{name}.npz')['p']; members=set().union(*(orbit(q) for q in p))
        assert not full_orbits.intersection(members); full_orbits.update(members)
        assert len({tuple(q) for q in p})==len(p)
    write(root/'data_audit.json',dict(pools=audit,overlapping_orbits=0,duplicate_inputs=0,global_across_seeds=True,nested_target_subsets=True,
          historical_exclusion=False,historical_note='Fresh pilot data, old alignment weights/data never trained or loaded; no exhaustive old artifact exclusion claimed.'))
    write(root/'config.json',CFG)

def smoke(root):
    seed(900); net=make(900); pool=loadpool(root,'development_train'); tiny=(pool[0][:64],pool[1][:64]); initial=copy.deepcopy(net.state_dict())
    # Direct definitions, independent of upstream label functions, all n<=7.
    checked=0
    for n in range(2,8):
        for p in itertools.permutations(range(1,n+1)):
            expected=[sum(v==i for i,v in enumerate(p,1)),sum(v==n+1-i for i,v in enumerate(p,1)),sum(b==a+1 for a,b in zip(p,p[1:])),sum(a>b for a,b in zip(p,p[1:])),sum(p[i-1]<p[i]>p[i+1] for i in range(1,n-1)),sum(v>max(p[:i],default=0) for i,v in enumerate(p))]
            assert labels(p)==expected
            assert PF['fixed_points'](p)==PF['anti_fixed_points'](tuple(n+1-v for v in p));checked+=1
    ids,gold,eq=batch(*tiny,0);net.eval()
    with torch.no_grad():
        a=net(ids)[:,eq]; changed=ids.clone();changed[:,eq+1]=(gold+1)%10
        err=float((a-net(changed)[:,eq]).abs().max());assert err<1e-5
    train(net,tiny,[0],8,901,root/'smoke'/'trained.pt')
    net.eval()
    with torch.no_grad(): z=net(ids).cpu()
    net2=make(902);net2.load_state_dict(torch.load(root/'smoke'/'trained.pt',map_location=DEVICE,weights_only=False)['model']);net2.eval()
    with torch.no_grad(): loaderr=float((z-net2(ids).cpu()).abs().max())
    assert loaderr==0 and any(not torch.equal(initial[k],v) for k,v in net.state_dict().items())
    # Exercise the full multi-task pretrain -> fresh optimizer adaptation -> prompt eval path.
    train(net,tiny,SOURCES['B'],8,903,root/'smoke'/'source.pt')
    train(net,tiny,[0],8,904,root/'smoke'/'adapt.pt'); result=evaluate(net,tiny,0)
    write(root/'smoke_audit.json',dict(status='passed',label_cases=checked,causal_suffix_max_error=err,save_load_max_error=loaderr,
          optimizer_reset=True,multitask_and_adaptation_completed=True,example_prompt_evaluation=result,device=str(DEVICE),torch=torch.__version__,cuda=torch.version.cuda))

def develop(root):
    assert (root/'smoke_audit.json').exists()
    net=make(910); train(net,loadpool(root,'development_train'),[0],CFG['dev_steps'],911,root/'development'/'target.pt',loadpool(root,'development_dev'))
    result=evaluate(net,loadpool(root,'development_dev'),0); y=loadpool(root,'development_train')[1][:,0]; mode=int(torch.bincount(y).argmax()); baseline=float((loadpool(root,'development_dev')[1][:,0]==mode).float().mean())
    write(root/'development_result.json',dict(task=TASKS[0],result=result,train_majority_label=mode,majority_accuracy=baseline,pass_gate=result['accuracy']>=.90,
           choice='Single candidate: fixed points, known complement relation, historical learnability; accept only development accuracy >=0.90. No transfer or test results consulted.'))

def freeze(root):
    dev=json.loads((root/'development_result.json').read_text()); assert dev['pass_gate'], 'Development gate failed; retain result, do not run formal.'
    hashes={str(p.relative_to(HERE)):sha(p) for p in [HERE/'pilot.py',*sorted((HERE/'vendor').rglob('*.py'))]}
    write(root/'FROZEN_PROTOCOL.json',dict(config=CFG,code_hashes=hashes,config_sha256=sha(root/'config.json'),development_sha256=sha(root/'development_result.json'),data_audit_sha256=sha(root/'data_audit.json'),
          target=TASKS[0],sources={k:[TASKS[i] for i in v] for k,v in SOURCES.items()},related_identity='fixed_points(p)=anti_fixed_points(complement(p))',
          replacement='successions is not in the registered direct pair with fixed_points; not claimed mathematically unrelated',
          analysis='Report every seed, B-A, C-A, B-C and descriptive average only; no significance claim with two seeds.',frozen_unix=time.time()))

def verify(root):
    frozen=json.loads((root/'FROZEN_PROTOCOL.json').read_text())
    assert frozen['config']==CFG and frozen['config_sha256']==sha(root/'config.json')
    for p,h in frozen['code_hashes'].items():assert sha(HERE/p)==h,p
    for name,rec in json.loads((root/'data_audit.json').read_text())['pools'].items():assert sha(root/'data'/f'{name}.npz')==rec['sha256']

def csvwrite(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def formal(root):
    verify(root); learning=[]
    for s in CFG['seeds']:
        unit=root/f'seed{s}'; unit.mkdir(exist_ok=True); net=make(s);initial=copy.deepcopy(net.state_dict());torch.save(dict(model=initial,config=CFG,seed=s),unit/'step0.pt')
        pre=loadpool(root,f's{s}_pretrain'); srcdev=loadpool(root,f's{s}_source_dev')
        for cond in ['B','C']:
            path=unit/f'pretrain_{cond}.pt'
            if path.exists():continue
            net=make(s);net.load_state_dict(initial);train(net,pre,SOURCES[cond],CFG['pre_steps'],s+1000,path)
        # Source metrics do not gate or alter any later training choice.
        for cond in ['B','C']:
            net=make(s);net.load_state_dict(torch.load(unit/f'pretrain_{cond}.pt',map_location=DEVICE,weights_only=False)['model'])
            for t in SOURCES[cond]:
                mode=int(torch.bincount(pre[1][:,t]).argmax())
                learning.append(dict(seed=s,condition=cond,task=TASKS[t],train_accuracy=evaluate(net,pre,t)['accuracy'],**evaluate(net,srcdev,t),majority_label=mode,majority_accuracy=float((srcdev[1][:,t]==mode).float().mean())))
        target=loadpool(root,f's{s}_target_train'); dev=loadpool(root,f's{s}_target_dev')
        for n in CFG['sizes']:
            for cond in ['A','B','C']:
                path=unit/f'adapt_{cond}_{n}.pt'
                if path.exists():continue
                net=make(s);net.load_state_dict(initial if cond=='A' else torch.load(unit/f'pretrain_{cond}.pt',map_location=DEVICE,weights_only=False)['model'])
                # Tied embedding row also serves as output class: restore target task row identically.
                with torch.no_grad():net.token_embedding.weight[V['<FIXED_POINTS>']].copy_(initial['token_embedding.weight'][V['<FIXED_POINTS>']])
                train(net,(target[0][:n],target[1][:n]),[0],CFG['adapt_steps'],s+2000,path,dev)
    csvwrite(root/'source_learning.csv',learning)
    write(root/'TRAINING_COMPLETE.json',dict(completed=True,unix=time.time(),test_evaluated=False))

def results(root):
    verify(root);assert (root/'TRAINING_COMPLETE.json').exists()
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    allrows=[]
    for s in CFG['seeds']:
        rows=[]; test=loadpool(root,f's{s}_test');dev=loadpool(root,f's{s}_target_dev');target=loadpool(root,f's{s}_target_train')
        for n in CFG['sizes']:
            mode=int(torch.bincount(target[1][:n,0]).argmax())
            for cond in ['A','B','C']:
                cp=torch.load(root/f'seed{s}'/f'adapt_{cond}_{n}.pt',map_location=DEVICE,weights_only=False);net=make(s);net.load_state_dict(cp['model'])
                row=dict(seed=s,condition=cond,n=n,test_accuracy=evaluate(net,test,0)['accuracy'],dev_accuracy=evaluate(net,dev,0)['accuracy'],train_accuracy=evaluate(net,(target[0][:n],target[1][:n]),0)['accuracy'],majority_label=mode,test_majority_accuracy=float((test[1][:,0]==mode).float().mean()),adapt_updates=cp['step'],adapt_input_tokens=cp['input_tokens'],adapt_supervised_tokens=cp['supervised_tokens'],adapt_seconds=cp['seconds'],checkpoint_sha256=sha(root/f'seed{s}'/f'adapt_{cond}_{n}.pt'))
                rows.append(row); print(row,flush=True)
        csvwrite(root/f'results_seed{s}.csv',rows);allrows.extend(rows)
    csvwrite(root/'results_all.csv',allrows)
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,s in zip(axes,CFG['seeds']+[None]):
        for c in ['A','B','C']:
            vals=[np.mean([r['test_accuracy'] for r in allrows if r['n']==n and r['condition']==c and (s is None or r['seed']==s)]) for n in CFG['sizes']]
            ax.plot(CFG['sizes'],vals,'o-',label=c)
        ax.set_xscale('log',base=4);ax.set_xticks(CFG['sizes'],[str(n) for n in CFG['sizes']]);ax.set_ylim(0,1.02);ax.grid(alpha=.3);ax.legend();ax.set_title(f'Seed {s}' if s else 'Paired mean (2 seeds)');ax.set_xlabel('Unique target training examples')
    axes[0].set_ylabel('Held-out target accuracy');fig.tight_layout();fig.savefig(root/'sample_efficiency.png',dpi=180);fig.savefig(root/'sample_efficiency.svg');plt.close(fig)
    # Replacement matching audit uses source train/development only, never test.
    match=[]
    for s in CFG['seeds']:
        data=np.load(root/'data'/f's{s}_pretrain.npz')['y']; p=np.bincount(data[:,1],minlength=11)/len(data);q=np.bincount(data[:,2],minlength=11)/len(data)
        for t in [1,2]:
            yy=data[:,t];hist=np.bincount(yy,minlength=11)/len(yy)
            match.append(dict(seed=s,task=TASKS[t],mean=float(yy.mean()),std=float(yy.std()),entropy_bits=float(-(hist[hist>0]*np.log2(hist[hist>0])).sum()),histogram=hist.tolist(),replacement_total_variation=float(abs(p-q).sum()/2)))
    write(root/'replacement_matching.json',match)
    write(root/'COMPLETED.json',dict(status='completed',rows=len(allrows),test_used_only_after_training=True,unix=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','smoke','develop','freeze','formal','results']);p.add_argument('--out',type=Path,default=HERE/'runs'/'pilot_v1');a=p.parse_args()
    print(dict(stage=a.stage,device=str(DEVICE),torch=torch.__version__,cuda=torch.version.cuda),flush=True)
    globals()[a.stage](a.out.resolve())
