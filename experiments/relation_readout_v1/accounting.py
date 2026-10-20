"""Join measured query batches to routes without rerunning inference."""
import sys,json,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from permworld_platform import calibration as p
from experiments.relation_readout_v1.executor import query_inputs
OUT=Path(__file__).resolve().parent
timings=list(csv.DictReader((OUT/'compute.csv').open()))
source=[json.loads(line) for line in (OUT/'source_predictions.jsonl').open()]
lookup={}; cursor=0
for block_id,timing in enumerate(timings):
    count=int(timing['count']); block=source[cursor:cursor+count]; cursor+=count
    assert len(block)==count and all(r['run_id']==timing['run_id'] and r['task']==timing['task'] for r in block)
    key=(timing['run_id'],int(timing['n']),timing['task'],tuple(tuple(r['p']) for r in block))
    assert key not in lookup
    lookup[key]=(block_id,int(timing['actual_forward_calls']),float(timing['seconds']))
assert cursor==len(source)
edges={e['relation_id']:e for e in json.loads((OUT/'routes.json').read_text())}
manifest=json.loads((OUT/'data_manifest.json').read_text())
summary=list(csv.DictReader((OUT/'summary_per_repeat.csv').open()))
raw=[json.loads(line) for line in (OUT/'raw_predictions.jsonl').open()]
runids={(r['seed'],r['chain'],r['k'],r['budget'],r['target'],r['method']):r['run_id'] for r in raw}
rows=[]
for s in summary:
    seed,n,k,budget=map(int,[s['seed'],s['n'],s['k'],s['budget']])
    rid=runids[(seed,s['chain'],k,budget,s['target'],s['method'])]
    pool=next(x for x in manifest['pools'] if x['seed']==seed and x['n']==n)
    ps=[tuple(map(int,q)) for q in np.load(OUT/pool['path'])['p']]
    blocks=[]
    if s['route']=='direct': blocks=[lookup[(rid,n,s['target'],tuple(ps))]]
    else:
        edge=edges[s['route']]
        for j,q in enumerate(edge['queries']):
            qs=[query_inputs(edge,permutation)[j][1] for permutation in ps]
            if s['diagnostic']=='True' and j==len(edge['queries'])-1: qs=ps[1:]+ps[:1] if s['route']=='position_partition' else ps
            blocks.append(lookup[(rid,n,q['task_id'],tuple(qs))])
    unique={b[0]:b for b in blocks}
    rows.append({**{key:s[key] for key in ['seed','chain','k','budget','target','n','route','method','diagnostic']},'logical_queries_per_object':len(blocks),'objects':len(ps),'measured_batched_forward_calls_for_required_blocks':sum(b[1] for b in unique.values()),'measured_seconds_for_required_blocks':sum(b[2] for b in unique.values()),'shared_physical_block_ids':json.dumps(list(unique)),'accounting_note':'Blocks shared across routes/diagnostics are charged once globally in compute.csv; do not sum this table across methods.'})
p.write_csv(OUT/'route_compute.csv',rows)
p.write_json(OUT/'COMPUTE_TOTALS.json',dict(physical_batched_forward_calls=sum(int(t['actual_forward_calls']) for t in timings),physical_query_objects=len(source),synchronized_prediction_seconds=sum(float(t['seconds']) for t in timings),route_rows=len(rows)))
print('Validated measured query block accounting for',len(rows),'route/direct repeat-length conditions')
