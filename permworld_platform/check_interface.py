"""Boundary and round-trip checks for the new exact sequence contract."""
import json
from pathlib import Path

from .encoding import (VOCABULARY, encode_value, decode_value, encode_example,
                       decode_answer_ids, old, ENCODING_VERSION)
from .registry import list_task_instances


def main():
    assert VOCABULARY[:188] == old.VOCABULARY
    values = [0, 1, 99, 100, 101, 220, 9999, 10000, 155117520, 10**80,
              (), (1, 2, 100), ((1, 3), (2,)), ((), (10000,))]
    for value in values:
        assert decode_value(encode_value(value)) == value
    invalid = [("<NUM_START>", "00", "01", "<NUM_END>"),
               ("<LIST_START>", "01", ",", "<LIST_END>"), ("01", "02")]
    for tokens in invalid:
        try:
            decode_value(tokens)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted noncanonical answer {tokens}")
    count = 0
    for inst in list_task_instances(include_aliases=True, include_binary=True):
        p = (3, 1, 4, 2, 6, 5, 8, 7, 10, 9)
        kw = {"operands": {"q": p[::-1]}} if inst["input_arity"] == 2 else {}
        sample = encode_example(inst["task_id"], p, inst["params"], **kw)
        assert decode_answer_ids(sample["answer_ids"], boolean=isinstance(sample["answer_value"], bool)) == sample["answer_value"]
        assert sample["input_ids"][:sample["prompt_length"]] == sample["prompt_ids"]
        count += 1
    big = encode_example("increasing_subsequence_count", tuple(range(1, 31)), {"k": 15})
    assert big["answer_value"] == 155117520
    try:
        encode_example("to_reduced_word", tuple(range(30, 0, -1)), max_seq_len=128)
    except ValueError as exc:
        assert "no truncation" in str(exc)
    else:
        raise AssertionError("over-context answer was silently accepted")
    audit = dict(status="passed", encoding=ENCODING_VERSION, vocabulary_size=len(VOCABULARY),
                 historical_ids_preserved=188, endpoint_instance_roundtrips=count,
                 integer_or_structure_roundtrips=len(values), malformed_decodings_rejected=len(invalid),
                 large_answer=big["answer_value"], large_answer_token_count=len(big["answer_ids"]),
                 excessive_context_explicitly_rejected=True)
    path = Path("experiments/platform_v01/interface_audit.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
