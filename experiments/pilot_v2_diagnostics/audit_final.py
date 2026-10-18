"""Read-only final numerical/provenance checks, without test forwards."""
import argparse, csv, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import torch
import diagnose as d

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=d.OUT);a=p.parse_args();d.OUT=a.out.resolve();root=d.OUT
d.verify();d.preservation_check()
selection=d.read(root/'SELECTIONS_FROZEN.json');test=d.read(root/'test_data_audit.json');finished=d.read(root/'TEST_COMPLETE.json')
assert d.read(root/'TRAINING_COMPLETE.json')['unix']<=selection['unix']<=test['created_unix']<=finished['unix']
assert test['selection_sha256']==finished['selection_sha256']==d.sha(root/'SELECTIONS_FROZEN.json')
for row in selection['selections']:
    for which in ['best','endpoint']:assert d.sha(root/row[which+'_path'])==row[which+'_sha256']
for rec in test['pools']:assert d.sha(root/'data'/f's{rec["seed"]}_test.npz')==rec['sha256']
metas=[d.read(p) for p in sorted((root/'adapt').glob('*/completed.json'))]
assert len(metas)==30
for seed in [17,42]:
    rows=[m for m in metas if m['seed']==seed];assert len(rows)==15
    assert len({r['schedule_sha256'] for r in rows})==1
    assert len({r['initialization']['target_row_hash'] for r in rows})==1
    assert len({r['initialization']['train_sha256'] for r in rows})==1
    assert len({r['initialization']['dev_sha256'] for r in rows})==1
    for c,b in [('A',0),('B',2400),('C',2400),('B',4800),('C',4800)]:
        candidates=[r for r in rows if r['condition']==c and r['budget']==b]
        assert sorted(r['lr'] for r in candidates)==sorted(d.CONFIG['learning_rates'])
        chosen=min(candidates,key=lambda r:(-r['best']['dev_accuracy'],r['best']['dev_answer_ce'],r['best']['step'],d.CONFIG['learning_rates'].index(r['lr'])))
        sel=next(s for s in selection['selections'] if s['seed']==seed and s['condition']==c and s['budget']==b)
        assert chosen['lr']==sel['lr'] and chosen['best']['step']==sel['step']
for m in metas:
    assert m['updates']==800 and m['input_tokens']==1433600 and m['supervised_tokens']==102400
    assert m['initialization']['loaded_state_hash']==m['initialization']['source_state_hash']
    assert m['initialization']['optimizer_entries']==0 and m['initialization']['optimizer_state_reset']
    folder=root/'adapt'/m['run_id'];cp=d.load(folder/'endpoint.pt')
    assert {int(v['step']) for v in cp['optimizer']['state'].values()}=={800}
    assert d.statehash(cp['model'])==m['final_model_state_hash']
    for name in ['best','endpoint']:assert d.sha(folder/f'{name}.pt')==m[name+'_sha256']
for seed in [17,42]:
    for c in ['B','C']:
        path=d.source_path(seed,c,4800);meta=d.read(path.with_suffix('.json'));cp=d.load(path)
        assert cp['step']==4800 and meta['additional_updates']==2400
        assert meta['sampler_replay_4800_verified'] and meta['source_checkpoint_sha256']==d.sha(d.source_path(seed,c,2400))
        assert {int(v['step']) for v in cp['optimizer']['state'].values()}=={4800}
        assert d.statehash(cp['model'])==meta['model_state_hash']
with (root/'results_all_candidates.csv').open() as f:raw=list(csv.DictReader(f))
assert len(raw)==60
for seed in [17,42]:
    with (root/f'results_seed{seed}.csv').open() as f:assert len(list(csv.DictReader(f)))==30
assert all(0<=float(r['test_accuracy'])<=1 for r in raw)
replay=[m for m in metas if m['replay_v1']]
record=dict(status='passed',v1_files_byte_preserved=True,frozen_source_hashes_verified=True,adaptation_candidates=30,
    extended_source_runs=4,candidate_test_rows=60,selection_before_test_generation=True,paired_schedule_and_target_initialization_verified=True,
    candidate_budget_equality_verified=True,checkpoint_load_hashes_verified=True,adaptation_optimizer_reset_verified=True,
    source_optimizer_and_rng_continuation_verified=True,original_protocol_replay_bitwise_equal=sum(m['replay_v1']['bitwise_same'] for m in replay),original_protocol_replay_count=len(replay),
    added_pretrain_updates=9600,added_pretrain_input_tokens=17203200,added_pretrain_supervised_tokens=1228800,
    diagnostic_adaptation_updates=24000,diagnostic_adaptation_input_tokens=43008000,diagnostic_adaptation_supervised_tokens=3072000,
    tests_forwarded_by_this_audit=0,scope_status='completed; expansion paused pending user guidance')
d.write(root/'FINAL_AUDIT.json',record)
d.write(root/'environment.json',dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
    device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',threads=d.v1.torch.get_num_threads(),tf32_matmul=d.v1.torch.backends.cuda.matmul.allow_tf32))
manifest={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=d.sha(p)) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ['artifact_manifest.json','execution.log']}
d.write(root/'artifact_manifest.json',manifest)
print(json.dumps(record,indent=2))
