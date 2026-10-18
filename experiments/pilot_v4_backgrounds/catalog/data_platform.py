"""CPU-only PermWorld registry, representations, transformations and pool tools.

This module adds no mathematical tasks and never reads checkpoints or outcomes.
The training engine may retain its original NumPy generator for exact replay;
sample_pool below declares a separate Python Random proposal stream explicitly.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments/permworld_pilot/vendor'))
from neurips_permutations.math_ops import PROPERTY_FUNCTIONS, validate_permutation, inverse
from neurips_permutations.passage import TOKEN_TO_ID, ID_TO_TOKEN, one_line_tokens

TASK_ORDER = tuple(PROPERTY_FUNCTIONS)
ACTIONS = tuple(''.join(x) for x in itertools.product('01', repeat=3))


def transform(permutation, action='000'):
    """Bits denote inverse, reverse, complement, applied in that order."""
    p = validate_permutation(permutation)
    names = {'identity': '000', 'inverse': '100', 'reverse': '010', 'complement': '001'}
    action = names.get(action, action)
    if action not in ACTIONS:
        raise ValueError(f'Unknown action: {action}')
    if action[0] == '1': p = inverse(p)
    if action[1] == '1': p = tuple(reversed(p))
    if action[2] == '1': p = tuple(len(p) + 1 - v for v in p)
    return p


def orbit(permutation):
    return tuple(sorted({transform(permutation, a) for a in ACTIONS}))


def orbit_key(permutation):
    return orbit(permutation)[0]


def labels(permutation, tasks=TASK_ORDER):
    p = validate_permutation(permutation)
    unknown = set(tasks) - PROPERTY_FUNCTIONS.keys()
    if unknown: raise ValueError(f'Unregistered tasks: {unknown}')
    return tuple(int(PROPERTY_FUNCTIONS[t](p)) for t in tasks)


def encode_one_line(permutation):
    p = validate_permutation(permutation)
    if not 1 <= len(p) <= 30: raise ValueError('Scalar pilot grammar supports 1 <= n <= 30')
    tokens = ('<BOS>', '<SIZE>', f'{len(p):02d}', *one_line_tokens(p))
    return tuple(TOKEN_TO_ID[t] for t in tokens)


def decode_one_line(ids):
    tokens = tuple(ID_TO_TOKEN[int(i)] for i in ids)
    if len(tokens) < 6 or tokens[:2] != ('<BOS>', '<SIZE>'):
        raise ValueError('Invalid prefix')
    n = int(tokens[2])
    if not 1 <= n <= 30 or len(tokens) != 2*n + 4:
        raise ValueError('Invalid length')
    if tokens[3] != '<ONE_START>' or tokens[-1] != '<ONE_END>':
        raise ValueError('Invalid boundaries')
    body = tokens[4:-1]
    if any(t != ',' for t in body[1::2]): raise ValueError('Invalid separators')
    p = validate_permutation(tuple(int(t) for t in body[::2]))
    if encode_one_line(p) != tuple(ids): raise ValueError('Noncanonical encoding')
    return p


def sample_pool(n, count, seed, excluded_orbit_keys=()):
    """Uniform permutation proposals; accept one representative per unseen orbit.

    The returned pool is not claimed to be uniform over orbits. The proposal
    stream is Python Random.sample, distinct from the engine's frozen NumPy RNG.
    Caller must union returned keys into the global exclusion set across roles.
    """
    if not 1 <= n <= 30 or count < 0: raise ValueError('Invalid n or count')
    rng = random.Random(seed)
    seen = {tuple(k) for k in excluded_orbit_keys}
    output = []
    proposed = 0
    limit = max(10000, 1000 * count)
    while len(output) < count:
        proposed += 1
        if proposed > limit: raise RuntimeError('Pool capacity or rejection limit reached; no partial pool returned')
        p = tuple(rng.sample(range(1, n + 1), n))
        key = orbit_key(p)
        if key in seen: continue
        seen.add(key); output.append(p)
    return output, dict(n=n, count=count, seed=seed, proposals=proposed,
        generator='python.Random.sample; unseen inverse/reverse/complement orbit rejection',
        input_sha256=hashlib.sha256(json.dumps(output, separators=(',', ':')).encode()).hexdigest())


def validate_pools(pools):
    """Exact full-orbit role disjointness; same-role task reuse is intentional."""
    expanded = set()
    result = {}
    for role, permutations in pools.items():
        values = [validate_permutation(p) for p in permutations]
        keys = {orbit_key(p) for p in values}
        if len(keys) != len(values): raise ValueError(f'Duplicate orbit within {role}')
        members = {q for p in values for q in orbit(p)}
        if expanded & members: raise ValueError(f'Cross-role orbit overlap at {role}')
        expanded |= members
        result[role] = dict(inputs=len(values), unique_orbits=len(keys), expanded_members=len(members))
    return dict(status='passed', overlap=0, roles=result)


def action_composition_table():
    # A generic eight-member orbit makes the action representation faithful.
    witness = next(p for p in itertools.permutations(range(1, 6)) if len(orbit(p)) == 8)
    identify = {transform(witness, action): action for action in ACTIONS}
    return {(first, second): identify[transform(transform(witness, first), second)]
            for first in ACTIONS for second in ACTIONS}


def unit_checks():
    checks = 0
    table = action_composition_table()
    for n in range(1, 7):
        for p in itertools.permutations(range(1, n + 1)):
            assert decode_one_line(encode_one_line(p)) == p
            assert len(encode_one_line(p)) == 2*n + 4
            key = orbit_key(p)
            for action in ACTIONS:
                q = transform(p, action)
                assert orbit_key(q) == key
                assert sorted(q) == list(range(1, n+1))
                for second in ACTIONS:
                    assert transform(q, second) == transform(p, table[action, second])
            checks += 1
    seen = set(); pools = {}
    for i, role in enumerate(['source', 'adapt', 'test_fixture']):
        ps, meta = sample_pool(6, 16, 2026102100+i, seen)
        same, same_meta = sample_pool(6, 16, 2026102100+i, seen)
        assert ps == same and meta == same_meta
        seen.update(map(orbit_key, ps)); pools[role] = ps
    split_audit = validate_pools(pools)
    try: validate_pools({'a': pools['source'], 'b': [transform(pools['source'][0], 'inverse')]})
    except ValueError: pass
    else: raise AssertionError('Leakage witness was not rejected')
    try: decode_one_line((*encode_one_line((1,2,3))[:-1], TOKEN_TO_ID['<EOS>']))
    except ValueError: pass
    else: raise AssertionError('Malformed boundary was not rejected')
    return dict(status='passed', exhaustive_roundtrip_and_group_inputs=checks,
                action_composition_pairs=64, split_fixture=split_audit,
                malformed_input_rejected=True, orbit_leakage_rejected=True,
                model_imported='torch' in sys.modules, actual_study_test_inputs_generated=False)
