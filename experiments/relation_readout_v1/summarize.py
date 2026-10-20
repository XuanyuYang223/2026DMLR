"""Generate repeat-level tables, figures and the English report from raw records."""
import json, csv, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from permworld_platform import calibration as platform
OUT=Path(__file__).resolve().parent


def main():
    rows=[json.loads(line) for line in (OUT/'raw_predictions.jsonl').open()]
    keys=['seed','chain','k','budget','target','n','route','method','diagnostic']
    groups=defaultdict(list)
    direct={}
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
        if r['method']=='B': direct[(r['seed'],r['chain'],r['k'],r['budget'],r['target'],r['n'],r['index'])]=r['correct']
    metrics=['correct','all_sources_correct','error_cancellation','illegal','eos_failure','out_of_bounds']
    summaries=[]
    for key,rs in groups.items():
        s=dict(zip(keys,key)); s['count']=len(rs)
        for metric in metrics: s[metric]=float(np.mean([r[metric] for r in rs]))
        s['query_0_accuracy']=float(np.mean([r['query_correct'][0] for r in rs]))
        s['query_1_accuracy']=float(np.mean([r['query_correct'][1] for r in rs])) if rs[0]['queries']==2 else ''
        s['queries']=rs[0]['queries']
        s['paired_delta_B']=float(np.mean([r['correct']-direct[(r['seed'],r['chain'],r['k'],r['budget'],r['target'],r['n'],r['index'])] for r in rs])) if s['method'] in ['A','D'] else ''
        summaries.append(s)
    platform.write_csv(OUT/'summary_per_repeat.csv',summaries)
    aggregate=defaultdict(list)
    for s in summaries: aggregate[tuple(s[k] for k in keys if k!='seed')].append(s)
    stats=[]
    for key,ss in aggregate.items():
        s=dict(zip([k for k in keys if k!='seed'],key)); s['training_repeats']=len(ss)
        for metric in metrics+['paired_delta_B','query_0_accuracy','query_1_accuracy']:
            values=[x[metric] for x in ss if x[metric]!='']
            s[metric+'_mean']=float(np.mean(values)) if values else ''
            s[metric+'_sample_sd']=float(np.std(values,ddof=1)) if len(values)>1 else ''
        stats.append(s)
    platform.write_csv(OUT/'summary_mean_sd.csv',stats)
    contrasts=[]
    for s in summaries:
        if s['method'] not in ['A','D'] or s['diagnostic']: continue
        wrong=next(x for x in summaries if all(x[k]==s[k] for k in keys if k!='diagnostic') and x['diagnostic'])
        contrasts.append({**{k:s[k] for k in keys},'accuracy':s['correct'],'wrong_query_accuracy':wrong['correct'],'correct_minus_wrong':s['correct']-wrong['correct']})
    platform.write_csv(OUT/'query_diagnostics.csv',contrasts)
    contrast_groups=defaultdict(list)
    for s in contrasts: contrast_groups[tuple(s[k] for k in keys if k!='seed')].append(s)
    contrast_stats=[]
    for key,ss in contrast_groups.items():
        entry=dict(zip([k for k in keys if k!='seed'],key))
        for metric in ['accuracy','wrong_query_accuracy','correct_minus_wrong']:
            entry[metric+'_mean']=float(np.mean([s[metric] for s in ss]))
            entry[metric+'_sample_sd']=float(np.std([s[metric] for s in ss],ddof=1))
        contrast_stats.append(entry)
    platform.write_csv(OUT/'query_diagnostics_mean_sd.csv',contrast_stats)
    # Each panel is one route; each point is a mean over the same three training repeats.
    routes=json.loads((OUT/'routes.json').read_text())
    for kind in ['reconstruction_vs_adaptation','source_before_after']:
        fig,axes=plt.subplots(2,2,figsize=(13,9))
        for ax,edge in zip(axes.flat,routes):
            entries=[s for s in stats if s['route']==edge['relation_id'] and s['method']=='A' and not s['diagnostic']]
            labels=[]
            for j,s in enumerate(entries):
                labels.append(f"{s['chain'].replace('chain_','')} k{s['k']} {s['budget']//1000000}M n{s['n']}")
                d=next(x for x in stats if all(x[k]==s[k] for k in ['chain','k','budget','target','n','route','diagnostic']) and x['method']=='D')
                if kind=='reconstruction_vs_adaptation':
                    b=next(x for x in stats if all(x[k]==s[k] for k in ['chain','k','budget','target','n']) and x['method']=='B')
                    series=[(s,'A: pretrained reconstruction'),(b,'B: direct adaptation'),(d,'D: adapted reconstruction')]; metric='correct'
                else: series=[(s,'Before adaptation'),(d,'After adaptation')]; metric='all_sources_correct'
                for z,(x,label_text) in enumerate(series): ax.errorbar(j+(z-1)*.15,x[metric+'_mean'],yerr=x[metric+'_sample_sd'],fmt='o',label=label_text if j==0 else None,capsize=2)
            ax.set_title(edge['relation_id']); ax.set_ylim(-.03,1.03); ax.set_xticks(range(len(labels)),labels,rotation=65,ha='right',fontsize=7); ax.set_ylabel('Accuracy (mean +/- sample SD)')
            if entries: ax.legend(fontsize=7)
            else: ax.text(.5,.5,'NA: source tasks not pretrained',ha='center')
        fig.tight_layout(); fig.savefig(OUT/(kind+'.png'),dpi=170); fig.savefig(OUT/(kind+'.svg')); plt.close(fig)
    preserved=json.loads((OUT/'PRESERVATION_BEFORE.json').read_text())
    changed=[p for p,h in preserved.items() if not (ROOT/p).exists() or platform.sha(ROOT/p)!=h]
    assert not changed,changed
    platform.write_json(OUT/'PRESERVATION_AFTER.json',dict(checked_files=len(preserved),changed=changed))
    cp=json.loads((OUT/'checkpoints.json').read_text()); completion=json.loads((OUT/'EVALUATION_COMPLETE.json').read_text())
    oracle=list(csv.DictReader((OUT/'oracle_checks.csv').open()))
    text=['# Registered relation readout on existing models','',f"Base commit: `{json.loads((OUT/'FROZEN_PROTOCOL.json').read_text())['base_commit']}`. Inference only; no new training repeats, training, checkpoint reselection or task-library expansion. Existing historical records and publication files remain byte-identical.",'',f"Checkpoint audit: {sum(c['valid'] for c in cp)}/78 file hashes match the archived completion/selection records (24 pretrained, 48 selected adaptations, six shared scratch-trained target baselines). Formal evaluation status: {completion['status']}; {len(completion['completed'])}/78 checkpoints, {completion['seconds']:.1f} seconds, fixed 7200-second wall limit. Scratch baselines are target models trained from shared random initialization, not untrained random heads.",'','The three units are the original training repeats (4103, 5207, 6301). Each has 512 new inputs at each of n=10 and n=12; every method within a repeat shares those inputs. Historical full inverse/reverse/complement orbits and all new test orbits are excluded globally. Transformed queries are declared queries of one object, never additional independent samples.','', '## Per-route comparisons','', 'Values below average the two lengths within each training repeat, then report the mean and sample SD across three original repeats. Detailed per-length/per-repeat values and all diagnostics are in the CSV files. No best-of-route aggregation is performed.','', '| Route | Condition | A reconstruction | B direct adaptation | C scratch | D adapted reconstruction | A minus B |','|---|---|---:|---:|---:|---:|---:|']
    for edge in routes:
        conditions=sorted({(s['chain'],s['k'],s['budget']) for s in summaries if s['route']==edge['relation_id'] and s['method']=='A'})
        if not conditions: text.append(f"| {edge['relation_id']} | NA | NA | NA | NA | NA | NA |")
        for chain,k,budget in conditions:
            vals={}
            for method in ['A','B','C','D']:
                vals[method]=[]
                for seed in [4103,5207,6301]:
                    selected=[s for s in summaries if s['seed']==seed and s['target']==edge['target']['task_id'] and s['method']==method and not s['diagnostic'] and ((method=='C') or (s['chain']==chain and s['k']==k and s['budget']==budget)) and (method in ['B','C'] or s['route']==edge['relation_id'])]
                    vals[method].append(float(np.mean([s['correct'] for s in selected])))
            fmt=lambda v:f'{np.mean(v)*100:.2f} +/- {np.std(v,ddof=1)*100:.2f}'
            text.append(f"| {edge['relation_id']} | {chain}, k{k}, {budget//1000000}M | {fmt(vals['A'])} | {fmt(vals['B'])} | {fmt(vals['C'])} | {fmt(vals['D'])} | {fmt(np.array(vals['A'])-vals['B'])} |")
    text+=['','The fixed 4M and 32M comparisons show why route annotation alone is insufficient: both alpha fixed-point formulas trail direct adaptation at 4M but exceed it at 32M. The beta complement route also changes from a deficit at 4M to an advantage at 32M. LIS reconstruction has its strongest accuracy in alpha k2 at 32M, while alpha k16 has substantially lower source readout accuracy. These are prespecified condition comparisons, not a selected route or an argument for changing the source set after this test. Adapted readout D often deteriorates sharply, particularly for the two fixed-point arithmetic routes; direct target success therefore does not imply preserved source-query competence.','','Availability is structural: alpha k16 supports both fixed-point arithmetic routes; beta k16 supports the fixed-point complement route. LIS complement is available for alpha k2, alpha k16 and beta k16. Other combinations are NA and are enumerated separately in availability.json; they are never counted as incorrect.','','## Query checks and research interpretation','','Every correct route recovered its target exactly under oracle source answers on all 3072 new objects. The executor also checks that every model case with all required source answers correct reconstructs correctly. Wrong-query oracle accuracies are retained in `oracle_checks.csv`, including accidental successes.','']
    for edge in routes:
        o=[float(r['wrong_oracle_accuracy']) for r in oracle if r['route']==edge['relation_id']]
        a=[s for s in summaries if s['route']==edge['relation_id'] and s['method']=='A' and not s['diagnostic']]
        d=[s for s in summaries if s['route']==edge['relation_id'] and s['method']=='D' and not s['diagnostic']]
        text.append(f"- {edge['relation_id']}: wrong-query oracle accuracy {np.mean(o)*100:.2f}%. Available pretrained conditions: {len(a)//6}."+(f" Across these fixed conditions, all-source correctness changes from {np.mean([s['all_sources_correct'] for s in a])*100:.2f}% before adaptation to {np.mean([s['all_sources_correct'] for s in d])*100:.2f}% after adaptation; A error-cancellation frequency is {np.mean([s['error_cancellation'] for s in a])*100:.2f}% of objects." if a else ' Unavailable routes remain NA.'))
    text+=['','These annotations support concrete decisions about source coverage, legal query construction, and whether learned source answers are accurate enough for a specific external readout. Differences between A and D measure retained source prediction ability after adaptation. Error cancellation separates mathematically correct source reasoning from an accidentally correct reconstructed target. The wrong-query controls diagnose query sensitivity; their nonzero oracle success rates show why they are not unrelated tasks or fair competing algorithms.','','Formula-assisted reconstruction supplies external mathematical information and one or two source queries. It has a different information and inference budget from direct target adaptation. These results do not show that a model discovered or internally used a formula, do not establish a mechanism, and do not imply superiority over data-selection methods. Conditions and routes are descriptive comparisons, not extra independent training repeats.','','## Artifacts and accounting','','`source_predictions.jsonl` records complete source/direct token generations, queried permutations, EOS and legality. `raw_predictions.jsonl` records each target reconstruction and per-query correctness. `summary_per_repeat.csv` and `summary_mean_sd.csv` contain the requested metrics; `query_diagnostics.csv` contains paired correct-minus-wrong contrasts. `compute.csv` records actual batched forward calls and synchronized elapsed time per checkpoint, length and source task. Identical query arrays within one checkpoint/length are cached; two-query routes still require two logical source answers per object. Shared predictions must not be charged again as physical forwards. Total formal wall time includes loading, encoding and raw-output writes; smoke timing is separate. No output is clipped, reduced modulo, filled in or replaced by a majority. Illegal source outputs propagate failure.','', '![Reconstruction comparison](reconstruction_vs_adaptation.png)','','![Source retention](source_before_after.png)']
    compute=json.loads((OUT/'COMPUTE_TOTALS.json').read_text()) if (OUT/'COMPUTE_TOTALS.json').exists() else None
    if compute:
        text+=['',f"Measured physical inference: {compute['physical_batched_forward_calls']} batched model forwards, {compute['physical_query_objects']} queried objects, and {compute['synchronized_prediction_seconds']:.3f} synchronized prediction seconds. `route_compute.csv` joins every route/direct repeat-length condition to its measured physical block IDs, forward counts and time. Blocks shared across routes or diagnostics must not be summed twice. Checkpoint-internal identities, parent hashes, initialization states, actual source exposures and development selections are verified in `PROVENANCE_AUDIT.json`. Adaptation checkpoints have no archived final-state digest field: their complete file hashes are verified and new state digests are recorded separately without rebinding archived hashes."]
    consistency=sum(r['all_sources_correct'] and not r['diagnostic'] and r['method'] in ['A','D'] for r in rows)
    violations=sum(r['all_sources_correct'] and not r['diagnostic'] and r['method'] in ['A','D'] and not r['correct'] for r in rows)
    assert violations==0
    failures={k:sum(r[k] for r in rows) for k in ['illegal','eos_failure','out_of_bounds']}
    platform.write_json(OUT/'RESULT_VALIDATION.json',dict(target_evaluation_rows=len(rows),executor_consistency_cases=consistency,consistency_violations=violations,failures_including_reused_predictions=failures))
    text+=['',f"Across {len(rows)} target evaluation records (including distinct routes and query diagnostics), there are {failures['illegal']} records with an illegal required output, {failures['eos_failure']} EOS failures and {failures['out_of_bounds']} out-of-range reconstructions. These are record counts with shared predictions, not independent samples or unique physical generation counts. All {consistency} correct-route cases with every required source answer correct passed executor consistency. Detailed failure rates appear by condition in the summary tables. `query_diagnostics_mean_sd.csv` reports the paired correct-minus-wrong differences across the original repeats."]
    text+=['','## New-test labeling and paired-repeat review','','The 89.52% direct-adaptation result above is measured on this experiment\'s newly generated test inputs. The historical 88.44% result uses a different test set and remains an archived result. Their difference must not be interpreted as model improvement: the selected checkpoint is unchanged. The two arithmetic routes share checkpoints and test objects and are parallel readouts, not independent replications. Source-query degradation after adaptation is descriptive and does not establish that it caused the target accuracy gap.','','The table below retains each original paired repeat for alpha k16, 32M. Each value averages n=10 and n=12 within that repeat; differences are in percentage points.','','| Route | Original seed | A new-test accuracy (%) | B new-test accuracy (%) | A minus B (pp) |','|---|---:|---:|---:|---:|']
    for route in ['position_partition','fixed_from_two_deficiency_queries']:
        for seed in [4103,5207,6301]:
            a=[s for s in summaries if s['route']==route and s['seed']==seed and s['method']=='A' and s['chain']=='chain_alpha' and s['k']==16 and s['budget']==32000000 and not s['diagnostic']]
            b=[s for s in summaries if s['seed']==seed and s['method']=='B' and s['chain']=='chain_alpha' and s['k']==16 and s['budget']==32000000 and s['target']=='fixed_points']
            av=float(np.mean([s['correct'] for s in a])); bv=float(np.mean([s['correct'] for s in b]))
            text.append(f'| {route} | {seed} | {100*av:.4f} | {100*bv:.4f} | {100*(av-bv):+.4f} |')
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    frozen=json.loads((OUT/'FROZEN_PROTOCOL.json').read_text())
    platform.write_json(OUT/'DOCUMENTATION_REVISION.json',dict(reason='Post-inference reporting and measured compute joins only; executor, inference code, routes, data and predictions unchanged.',original_summary_generator_sha256=frozen['code_sha256'].get('summarize.py'),current_summary_generator_sha256=platform.sha(OUT/'summarize.py'),inference_source_unchanged=all(platform.sha(OUT/name)==frozen['code_sha256'][name] for name in ['run.py','executor.py'])))
    for svg in OUT.glob('*.svg'):
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
    platform.write_json(OUT/'MANIFEST.json',{str(p.relative_to(OUT)):platform.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='MANIFEST.json' and '__pycache__' not in str(p)})


if __name__=='__main__': main()
