"""Versioned, lossless language-model interface; historical token IDs stay fixed.

Integers >=100 use the existing delimited base-100 number syntax. Structured
answers use explicitly new list delimiters. No answer is clipped or reduced.
The historical pilot's one-answer-token evaluator must not read this protocol.
"""
from __future__ import annotations

from experiments.permworld_pilot.vendor.neurips_permutations import passage as old
from .objects import validate_permutation
from .tasks import TASK_IDS, label, validate_parameters

ENCODING_VERSION = "permworld_sequence_v1"
INPUT_ENCODING = "one_line_v1"
_EXTRA = tuple(f"<{t.upper()}>" for t in TASK_IDS) + (
    "<LIST_START>", "<LIST_END>", "<PARAM_START>", "<PARAM_END>",
    "<PARAM_K>", "<PARAM_PATTERN>", "<PARAM_EXPONENT>", "<PARAM_INDEX>", "<OPERAND_Q>",
)
VOCABULARY = tuple(old.VOCABULARY) + tuple(t for t in _EXTRA if t not in old.VOCABULARY)
assert len(set(VOCABULARY)) == len(VOCABULARY)
TOKEN_TO_ID = {t: i for i, t in enumerate(VOCABULARY)}
ID_TO_TOKEN = dict(enumerate(VOCABULARY))


def encode_value(value) -> tuple[str, ...]:
    if isinstance(value, bool):
        value = int(value)
    if type(value) is int:
        return old.encode_number(value)
    if isinstance(value, (list, tuple)):
        result = ["<LIST_START>"]
        for i, element in enumerate(value):
            if i:
                result.append(",")
            result.extend(encode_value(element))
        return (*result, "<LIST_END>")
    raise TypeError(f"unsupported exact answer type: {type(value).__name__}")


def _decode(tokens, start=0):
    if start >= len(tokens):
        raise ValueError("missing value")
    if tokens[start] != "<LIST_START>":
        return old.decode_number(tokens, start)
    i, result = start + 1, []
    if i < len(tokens) and tokens[i] == "<LIST_END>":
        return tuple(result), i + 1
    while True:
        value, i = _decode(tokens, i)
        result.append(value)
        if i >= len(tokens):
            raise ValueError("unterminated list")
        if tokens[i] == "<LIST_END>":
            return tuple(result), i + 1
        if tokens[i] != ",":
            raise ValueError("missing list separator")
        i += 1


def decode_value(tokens):
    tokens = tuple(tokens)
    value, end = _decode(tokens)
    if end != len(tokens):
        raise ValueError("trailing tokens after exact answer")
    return value


def decode_answer_ids(ids, *, boolean=False):
    try:
        tokens = [ID_TO_TOKEN[int(i)] for i in ids]
    except (KeyError, ValueError) as exc:
        raise ValueError("invalid vocabulary ID") from exc
    if not tokens or tokens[-1] != "<EOS>":
        raise ValueError("answer must terminate with EOS")
    value = decode_value(tokens[:-1])
    if boolean:
        if value not in (0, 1) or not isinstance(value, int):
            raise ValueError("boolean answer must be zero or one")
        return bool(value)
    return value


def encode_object(permutation, encoding=INPUT_ENCODING):
    if encoding != INPUT_ENCODING:
        raise ValueError(f"v0.1 exposes one input encoding: {INPUT_ENCODING}")
    return old.one_line_tokens(validate_permutation(permutation))


def encode_example(task_id, permutation, params=None, operands=None, *, max_seq_len=None):
    """Produce a complete exact sample. answer_ids includes EOS.

    Causal next-token supervision starts at logits[prompt_length-1]; the
    original prompt ends with '=' and contains no answer-derived fields.
    """
    p = validate_permutation(permutation)
    params = validate_parameters(task_id, params)
    answer = label(task_id, p, params, operands)
    prompt = ["<BOS>", "<SIZE>", *old.encode_number(len(p)), *encode_object(p)]
    if operands:
        prompt.extend(("<OPERAND_Q>", *encode_object(operands["q"])))
    prompt.append(f"<{task_id.upper()}>")
    if params:
        prompt.append("<PARAM_START>")
        for key, value in sorted(params.items()):
            prompt.append(f"<PARAM_{key.upper()}>")
            prompt.extend(encode_value(value))
        prompt.append("<PARAM_END>")
    prompt.append("=")
    answer_tokens = (*encode_value(answer), "<EOS>")
    result = dict(task_id=task_id, params=params, p=p, n=len(p), answer_value=answer,
                  encoding_version=ENCODING_VERSION, input_encoding=INPUT_ENCODING,
                  prompt_ids=[TOKEN_TO_ID[t] for t in prompt], prompt_length=len(prompt),
                  answer_ids=[TOKEN_TO_ID[t] for t in answer_tokens])
    result["input_ids"] = result["prompt_ids"] + result["answer_ids"]
    if max_seq_len is not None and len(result["input_ids"]) > max_seq_len:
        raise ValueError(f"sample requires {len(result['input_ids'])} tokens, context={max_seq_len}; no truncation permitted")
    assert decode_answer_ids(result["answer_ids"], boolean=isinstance(answer, bool)) == answer
    return result


def vocabulary_manifest():
    return dict(encoding_version=ENCODING_VERSION, input_encodings=[INPUT_ENCODING],
                mathematical_task_count_added_by_encoding=0, tokens=list(VOCABULARY),
                legacy_prefix_tokens=len(old.VOCABULARY), legacy_token_ids_unchanged=True,
                integer_encoding="canonical base100; >=100 delimited NUM_START/NUM_END",
                structured_encoding="recursive LIST_START/comma/LIST_END; bool is0/1 with task type",
                output_evaluation="autoregressive exact full answer including EOS",
                historical_single_token_evaluator_compatible=False,
                excessive_context_policy="raise ValueError; never truncate")
