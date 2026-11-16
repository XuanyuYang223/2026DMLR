"""Read immutable train/dev pools and inventory complete answer/context lengths.

No model is constructed, no test pool is opened, and no input pool is generated.
New output files are exclusive-create; existing records are never overwritten.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from permworld_platform.encoding import encode_example


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    platform = ROOT / "experiments/platform_v01"
    manifest_path = platform / "dataset/manifest.json"
    index_path = platform / "LEARNABILITY_INDEX.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    with index_path.open(encoding="utf8", newline="") as handle:
        index = {row["instance_id"]: row for row in csv.DictReader(handle)}
    specs = [spec for spec in manifest["instances"]
             if index[spec["instance_id"]]["status"] == "not_run_in_v0.1"]
    assert len(specs) == 50 and all(spec["input_arity"] == 1 for spec in specs)
    pools = {}
    bindings = {str(path.relative_to(ROOT).as_posix()): sha(path)
                for path in (manifest_path, index_path, Path(__file__),
                             ROOT / "permworld_platform/encoding.py", ROOT / "permworld_platform/tasks.py")}
    for seed in (617, 829):
        for role in ("train", "dev"):
            for length in (8, 10, 12):
                path = platform / "dataset/data" / f"s{seed}_{role}_n{length}.npz"
                with np.load(path, allow_pickle=False) as archive:
                    pools[(seed, role, length)] = archive["p"].copy()
                bindings[path.relative_to(ROOT).as_posix()] = sha(path)
    rows, failures = [], []
    rejected = set()
    for spec in specs:
        for (seed, role, length), objects in pools.items():
            answers, sequences, overflow = [], [], []
            for input_index, permutation in enumerate(objects):
                record = encode_example(spec["task_id"], tuple(map(int, permutation)),
                                        spec["params"], max_seq_len=None)
                answers.append(len(record["answer_ids"]))
                sequences.append(len(record["input_ids"]))
                if sequences[-1] > 128:
                    # Exercise the actual reader rejection, without truncation.
                    try:
                        encode_example(spec["task_id"], tuple(map(int, permutation)),
                                       spec["params"], max_seq_len=128)
                    except ValueError as exc:
                        message = str(exc)
                    else:
                        raise AssertionError("Expected context rejection did not occur")
                    overflow.append(input_index)
                    failures.append(dict(instance_id=spec["instance_id"], seed=seed,
                                         split=role, length=length, input_index=input_index,
                                         required_sequence_tokens=sequences[-1], reader_error=message))
            if overflow:
                rejected.add(spec["instance_id"])
            within_cap = sum(value <= 8 for value in answers)
            rows.append(dict(instance_id=spec["instance_id"], family=spec["family"],
                             seed=seed, split=role, length=length, count=len(objects),
                             answer_tokens_min=min(answers), answer_tokens_max=max(answers),
                             sequence_tokens_max=max(sequences), context_rejected_count=len(overflow),
                             generation_cap_tokens=8, cap_compatible_count=within_cap,
                             cap_impossible_count=len(objects)-within_cap,
                             exact_accuracy_upper_bound=within_cap/len(objects)))
    accepted = [spec for spec in specs if spec["instance_id"] not in rejected]
    rejected_specs = [spec for spec in specs if spec["instance_id"] in rejected]
    for row in rows:
        row["instance_training_status"] = ("reader_rejected_entire_instance" if row["instance_id"] in rejected
                                           else "eligible_unchanged_generation_budget")
    result = dict(status="passed_with_explicit_reader_rejection_and_generation_limits",
                  created_at_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-started,
                  command=".venv/Scripts/python.exe -B experiments/platform_calibration_v02/reuse_authorized_v1/encoding_preflight.py",
                  scoped_instance_count=len(specs), accepted_instance_count=len(accepted),
                  rejected_instance_count=len(rejected_specs), accepted_instances=accepted,
                  rejected_instances=rejected_specs, context_failures=failures,
                  input_and_source_sha256=bindings, test_pool_opened=False, model_constructed=False,
                  new_inputs_generated=False, labels_truncated=False,
                  context_policy="Any required train/dev sample exceeding128 rejects the entire instance across both seeds and all lengths. No samples are filtered or replaced.",
                  generation_policy="Keep eight generation tokens including EOS. Full gold answers are retained for training/loss and exact matching; longer gold answers cannot be exactly generated under this cap.",
                  interpretation="Generation-capped exact accuracy is protocol-limited, not evidence of intrinsic task unlearnability.")
    csv_path = HERE / "ENCODING_PREFLIGHT.csv"
    json_path = HERE / "ENCODING_PREFLIGHT.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("Preflight records already exist; preserve them")
    with csv_path.open("x", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result["table_sha256"] = sha(csv_path)
    with json_path.open("x", encoding="utf8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "seconds", "scoped_instance_count",
                                                  "accepted_instance_count", "rejected_instance_count")}))


if __name__ == "__main__":
    main()
