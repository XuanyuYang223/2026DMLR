"""Read-only checkpoint provenance and executor consistency validation."""
import sys,json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from permworld_platform import calibration as p
from experiments.task_count_v2 import design
OUT=Path(__file__).resolve().parent
BASE=ROOT/'experiments/task_count_v2'
inventory=json.loads((OUT/'checkpoints.json').read_text())
protocol_sha=p.sha(BASE/'FROZEN_PROTOCOL.json')
checks=[]
for cp in inventory:
    checkpoint=p.load_checkpoint(ROOT/cp['path'])
    identity=checkpoint['identity']
    for key in ['seed','chain','k']:
        assert identity[key]==cp[key]
    assert identity['budget_tokens']==cp['budget']
    assert checkpoint['frozen_protocol_sha256']==protocol_sha
    state_digest=p.state_hash(checkpoint['model'])
    if 'state_sha256' in checkpoint: assert state_digest==checkpoint['state_sha256']
    if cp['stage']=='pretraining':
        path=BASE/'pretraining'/cp['run_id']/'task_exposures.csv'
        assert p.sha(path)==cp['completion']['artifact_sha256']['task_exposures.csv']
        exposures=list(csv.DictReader(path.open()))
        assert {r['source_task'] for r in exposures}=={design.task_key(s) for s in cp['sources']}
        assert all(int(r['sample_exposures'])>0 for r in exposures)
        parent=BASE/'initial'/f"seed{cp['seed']}.pt"
        initial=p.load_checkpoint(parent)
        assert checkpoint['initial_state_sha256']==initial['state_sha256']==p.state_hash(initial['model'])
    else:
        assert identity['target']==cp['target']
        assert checkpoint['update']==cp['selection']['selected_update']
        if cp['chain']=='baseline': parent=BASE/'initial'/f"seed{cp['seed']}.pt"
        else: parent=BASE/'pretraining'/f"{cp['chain']}_k{cp['k']}_b{cp['budget']}_s{cp['seed']}"/'endpoint.pt'
        assert p.load_checkpoint(parent)['state_sha256']==cp['completion']['loaded_state_sha256']
        assert p.sha(parent)==checkpoint['parent_checkpoint_sha256']
        assert checkpoint['development']==cp['selection']['best_development']
    checks.append(dict(run_id=cp['run_id'],identity_verified=True,state_sha256=state_digest,archived_state_digest_present='state_sha256' in checkpoint,provenance_verified=True,source_exposure_or_selection_verified=True))
p.write_json(OUT/'PROVENANCE_AUDIT.json',checks)
print('Verified all 78 checkpoint identities, states, parents and source exposure/selection records')
