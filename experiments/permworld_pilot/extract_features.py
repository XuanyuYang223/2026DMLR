"""Small CKA bridge; extracts prompt input-end states, does not compute metrics."""
import argparse
from pathlib import Path
import numpy as np
import torch
import pilot

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--data',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--task',default='fixed_points',choices=pilot.TASKS)
p.add_argument('--limit',type=int,default=256)
a=p.parse_args()
cp=torch.load(a.checkpoint,map_location=pilot.DEVICE,weights_only=False)
assert cp['config']['model']==pilot.CFG['model']
net=pilot.make(cp['seed']);net.load_state_dict(cp['model']);net.eval()
d=np.load(a.data);prefix=pilot.tensors(d['p'][:a.limit]);y=torch.tensor(d['y'][:a.limit],dtype=torch.long,device=pilot.DEVICE)
features={}; current={}; handles=[]
end=prefix.shape[1]-1
for i,block in enumerate(net.blocks):
    def hook(module,args,output,i=i):current[f'layer{i+1}']=output[:,end].detach().float().cpu().numpy()
    handles.append(block.register_forward_hook(hook))
with torch.inference_mode():
    for start in range(0,len(prefix),128):
        ids,_,_=pilot.batch(prefix[start:start+128],y[start:start+128],pilot.TASKS.index(a.task),False)
        net(ids)
        for k,v in current.items():features.setdefault(k,[]).append(v)
for h in handles:h.remove()
features={k:np.concatenate(v) for k,v in features.items()}
# Existing CKA main state applies the final normalization at the input end.
with torch.inference_mode():features['final_norm']=net.final_norm(torch.tensor(features[f'layer{len(net.blocks)}'],device=pilot.DEVICE)).cpu().numpy()
a.output.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.output,**features,p=d['p'][:a.limit],labels=d['y'][:a.limit],input_end_index=end)
print({k:v.shape for k,v in features.items()})
