"""Fixed two-contrast confirmation plan, chosen from v4 exploration."""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
V3 = HERE.parent / 'pilot_v3_multitarget' / 'artifacts'
V4 = HERE.parent / 'pilot_v4_backgrounds' / 'artifacts'
BASE = json.loads((V4 / 'FROZEN_PROTOCOL.json').read_text(encoding='utf-8'))['config']
TARGETS = ['fixed_points', 'longest_decreasing_run']
BACKGROUND_BY_TARGET = {'fixed_points': ['g0_v3', 'g2_deficiencies'],
                        'longest_decreasing_run': ['g0_v3', 'g1_recoils']}
BACKGROUNDS = ['g0_v3', 'g2_deficiencies', 'g1_recoils']
GROUPS = [copy.deepcopy(g) for g in BASE['groups'] if g['background'] in BACKGROUND_BY_TARGET[g['target']]]
SEEDS = [101, 211, 307, 401, 503]
REPEATS = [dict(repeat_id=f'R{i+1}', seed=s, model_initialization_seed=s,
                data_generation_seed=2026101901+i, test_generation_seed=2026101951+i,
                source_sampling_seed=s+1000, source_dropout_seed=s+1000,
                adapt_sampling_seed=s+2000, adapt_dropout_seed=s+2000) for i,s in enumerate(SEEDS)]
KEYS = ['n','model','vocab_size','encoding','source_pool','source_dev','target_samples',
        'target_dev','test_count','batch','pretrain_updates','source_lr','adaptation_updates',
        'learning_rates','adapt_eval_every','source_eval_steps','weight_decay','betas','epsilon',
        'clip','precision','selection']
CFG = {k: copy.deepcopy(BASE[k]) for k in KEYS}
CFG.update(version='pilot_v5_confirmation/v1', seeds=SEEDS, repeats=REPEATS, targets=TARGETS,
           backgrounds=BACKGROUNDS, background_by_target=BACKGROUND_BY_TARGET, groups=GROUPS,
           hypotheses=[dict(id='H1',target='fixed_points',old_background='g0_v3',new_background='g2_deficiencies',expected_direction='positive'),
                       dict(id='H2',target='longest_decreasing_run',old_background='g0_v3',new_background='g1_recoils',expected_direction='negative')],
           hypothesis_origin='Two fixed contrasts selected after v4 exploratory results, explicitly requested by the user. V4 seeds17/42/73 are exploration only; no pooling with validation.',
           data_generation_seeds={str(r['seed']):r['data_generation_seed'] for r in REPEATS},
           model_initialization_seeds={str(s):s for s in SEEDS},
           test_generation_seeds={str(r['seed']):r['test_generation_seed'] for r in REPEATS},
           pretrain_sampler_and_dropout_seeds={str(s):s+1000 for s in SEEDS},
           adapt_sampler_and_dropout_seeds={str(s):s+2000 for s in SEEDS},
           pool_generation_order=['source_train','source_dev','target_train','target_dev'],
           repeat_order=SEEDS,
           data_reuse='None. Five distinct initialization seeds and five distinct input-generator streams; all old v1-v4 inputs excluded. All roles/blocks orbit-disjoint; within-block sharing is intentional.',
           pairing='same initialization,source input pool,index stream and task slot stream;candidate slot0,common slots1..3;one shared A per target/block;paired 256 target samples and dev/test inputs',
           isolation='Full inverse/reverse/complement orbit exclusion across old v1-v4 and all new roles/blocks. Test generators run only after all development selections freeze. Rejection without replacement creates a disjoint finite-population design, not a claim of mathematically independent IID pools.',
           primary_contrast='Per block: (B-C)_new_background-(B-C)_g0; primary mean over five blocks, not difference of unpaired standard deviations.',
           statistics_plan='stats_plan.json and paired_statistics.py frozen before formal training; two primary contrasts, auxiliary B-A/C-A/source learning; no optional stopping or enlargement',
           reuse='Frozen model/encoding/training primitives only. No old trained weights or old results counted as new validation repeats.',
           scope='2 fixed hypotheses;5 new paired units;4 target/background groups;40 source runs;150 adaptation LR candidates;50 selections;300 best/endpoint tests;no new single-task training,no extra LR,no extension,no further backgrounds.')

def group(target, background):
    return next(g for g in GROUPS if (g['target'],g['background'])==(target,background))

def source_id(seed,target,background,condition):
    return f's{seed}_{target}_{background}_{condition}'

def adapt_id(seed,target,background,condition,lr):
    return f'{source_id(seed,target,background,condition)}_lr{lr:.0e}'

def cells():
    for seed in SEEDS:
        for target in TARGETS:
            yield seed,target,'shared','A'
            for background in BACKGROUND_BY_TARGET[target]:
                for condition in ['B','C']:
                    yield seed,target,background,condition
