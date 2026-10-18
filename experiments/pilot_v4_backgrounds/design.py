"""Finite exploratory background experiment; configuration contains no outcomes."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
V3 = HERE.parent / 'pilot_v3_multitarget' / 'artifacts'
BASE = json.loads((V3 / 'FROZEN_PROTOCOL.json').read_text(encoding='utf-8'))['config']
TARGETS = ['fixed_points', 'longest_decreasing_run']
BACKGROUNDS = ['g0_v3', 'g1_recoils', 'g2_deficiencies']
PAIRS = {'fixed_points': ['anti_fixed_points', 'successions'],
         'longest_decreasing_run': ['longest_increasing_run', 'left_to_right_maxima']}
COMMON = {'fixed_points': ['descents', 'peaks', 'left_to_right_maxima'],
          'longest_decreasing_run': ['descents', 'peaks', 'successions']}
GROUPS = []
for target in TARGETS:
    for background in BACKGROUNDS:
        common = COMMON[target].copy()
        if background == 'g1_recoils': common[0] = 'recoils'
        if background == 'g2_deficiencies': common[1] = 'deficiencies'
        GROUPS.append(dict(target=target, background=background, common=common,
                           related=PAIRS[target][0], replacement=PAIRS[target][1],
                           sources={c: [task, *common] for c, task in zip(['B','C'], PAIRS[target])},
                           changed_slot=None if background == 'g0_v3' else 1 if background == 'g1_recoils' else 2))
KEYS = ['seeds','n','model','vocab_size','encoding','source_pool','source_dev','target_samples',
        'target_dev','test_count','batch','pretrain_updates','source_lr','adaptation_updates',
        'learning_rates','adapt_eval_every','source_eval_steps','weight_decay','betas','epsilon',
        'clip','precision','selection']
CFG = {k: BASE[k] for k in KEYS}
CFG.update(version='pilot_v4_backgrounds/v1', targets=TARGETS, backgrounds=BACKGROUNDS, groups=GROUPS,
           case_selection='Exploratory cases selected from pilot_v3 outcomes at user request; not representative of all tasks.',
           background_selection='Prespecified one-background-slot changes using definitions and existing single-task records, before new transfer outcomes; not claimed difficulty matched.',
           data_generation_seeds={'17':20261018,'42':20261018,'73':2026101803},
           model_initialization_seeds={'17':17,'42':42,'73':73},
           pretrain_sampler_and_dropout_seeds={'17':1017,'42':1042,'73':1073},
           adapt_sampler_and_dropout_seeds={'17':2017,'42':2042,'73':2073},
           new_test_seed=2026101805,
           data_reuse='Exact v3 source/target train/dev pools and step0 weights; block IDs17/42/73 are not data generation RNG seeds.',
           pairing='candidate slot0; three common slots1..3; same input indices and slot assignments across all conditions/backgrounds within seed.',
           test_isolation='Fresh tests exclude all v1/v2/v3 inputs and v4 train/dev under full inverse/reverse/complement orbits; new test seeds mutually isolated.',
           primary_contrast='B-C for each target/background/seed; interaction=(B-C)_background2-(B-C)_background1, paired on seed and shared test inputs.',
           reuse='Exact ordered source task tuples only; A once per target/seed; g0 reuses compatible v3 training, new test evaluation only.',
           scope='Two targets,three backgrounds,three paired seeds,256 samples; no added LR candidates,no budget extension,no additional experiment.')

def group(target, background):
    return next(g for g in GROUPS if g['target']==target and g['background']==background)

def source_id(seed, target, background, condition):
    return f's{seed}_{target}_{background}_{condition}'

def adapt_id(seed, target, background, condition, lr):
    return f'{source_id(seed,target,background,condition)}_lr{lr:.0e}'

def cells():
    for seed in CFG['seeds']:
        for target in TARGETS:
            yield seed,target,'shared','A'
            for background in BACKGROUNDS:
                for condition in ['B','C']:
                    yield seed,target,background,condition
