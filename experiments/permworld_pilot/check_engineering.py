"""Check equivalent attention and real saved feature interface; no test data."""
import json
import torch
import pilot

root=pilot.HERE/'runs'/'pilot_v1'
cp=torch.load(root/'seed17'/'adapt_B_256.pt',map_location=pilot.DEVICE,weights_only=False)
reference=pilot.CausalTransformer(vocab_size=len(pilot.VOCABULARY),**pilot.CFG['model']).to(pilot.DEVICE)
reference.load_state_dict(cp['model']); reference.eval()
fast=pilot.make(17);fast.load_state_dict(cp['model']);fast.eval()
x,y=pilot.loadpool(root,'s17_target_dev');ids,gold,eq=pilot.batch(x[:16],y[:16],0)
with torch.inference_mode():
    a=reference(ids);b=fast(ids);err=float((a-b).abs().max())
assert err<1e-4
explicit=.5*(torch.nn.functional.cross_entropy(b[:,eq],gold)+torch.nn.functional.cross_entropy(b[:,eq+1],torch.full_like(gold,pilot.V['<EOS>'])))
target=torch.full_like(ids,-100);target[:,-2]=gold;target[:,-1]=pilot.V['<EOS>']
shifted=torch.nn.functional.cross_entropy(b[:,:-1].reshape(-1,b.shape[-1]),target[:,1:].reshape(-1),ignore_index=-100)
loss_error=float(abs(explicit-shifted));assert loss_error<1e-6
old=cp['optimizer'];fresh=torch.optim.AdamW(fast.parameters(),lr=pilot.CFG['lr'])
assert len(old['state'])>0 and len(fresh.state)==0
pilot.write(root/'engineering_equivalence.json',dict(status='passed',sdpa_vs_original_fp32_max_error=err,two_token_vs_shifted_loss_error=loss_error,preexisting_optimizer_entries=len(old['state']),fresh_optimizer_entries=len(fresh.state),used_split='s17_target_dev',test_inputs_read=False))
print(json.loads((root/'engineering_equivalence.json').read_text()))
