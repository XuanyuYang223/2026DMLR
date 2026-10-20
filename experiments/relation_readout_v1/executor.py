"""Registered-query executor with an injectable, non-oracle prediction provider."""
from permworld_platform.objects import transform
from permworld_platform.relations import eval_expression


def query_inputs(edge, permutation):
    result = []
    for query in edge['queries']:
        p = tuple(permutation)
        for action in query['transforms']:
            p = transform(action, p)
        result.append((query['task_id'], p, query['params']))
    return result


def reconstruct(edge, permutation, provider):
    answers = [provider(task, p, params) for task, p, params in query_inputs(edge, permutation)]
    return apply_answers(edge, len(permutation), answers)


def apply_answers(edge, n, answers):
    if any(value is None for value in answers):
        return None
    return eval_expression(edge['answer_map'], answers, {'n': n})
