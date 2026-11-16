"""Run disjoint formal units without concurrent shared-table aggregation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.fixed_points_deficiencies_arch_v1 import train
from experiments.fixed_points_deficiencies_arch_v1.design import REPEATS, ROUTES, SIZES


def exclusive_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pretrain", "adapt"])
    parser.add_argument("--units", nargs="+", required=True, choices=[r["repeat_id"] for r in REPEATS])
    args = parser.parse_args()
    assert len(set(args.units)) == len(args.units)
    protocol_hash = train.verify_protocol()
    train.require_smoke(protocol_hash)
    if args.stage == "adapt":
        for repeat in REPEATS:
            for condition in train.CONDITIONS:
                assert train.valid_completed(train.source_dir(repeat["repeat_id"], condition))
    started = datetime.now(timezone.utc).isoformat()
    begin = time.perf_counter()
    worker_id = "_".join(args.units)
    prefix = "SOURCE" if args.stage == "pretrain" else "ADAPT"
    record_path = train.OUT / f"{prefix}_WORKER_{worker_id}.json"
    if record_path.exists():
        raise FileExistsError("Preserve the existing completed worker record")
    locks = [train.OUT / f"{prefix}_{unit}.lock" for unit in args.units]
    for lock, unit in zip(locks, args.units):
        exclusive_json(lock, dict(unit=unit, stage=args.stage, process_id=os.getpid(),
                                  started_at_utc=started, protocol_sha256=protocol_hash))
    print(json.dumps(dict(worker=worker_id, stage=args.stage, units=args.units, started_at_utc=started)), flush=True)
    # On an exception the ownership records remain for explicit inspection;
    # another process cannot silently restart the same formal units.
    for unit in args.units:
        if args.stage == "pretrain":
            for condition in train.CONDITIONS:
                train.pretrain_one(unit, condition)
        else:
            for condition in ROUTES:
                for size in SIZES:
                    for lr_index in range(3):
                        train.adapt_one(unit, condition, size, lr_index)
    train.verify_protocol()
    record = dict(status="complete", stage=args.stage, units=args.units,
                  started_at_utc=started, finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.perf_counter()-begin, wrapper_sha256=train.sha(__file__),
                  protocol_sha256=protocol_hash,
                  scheduling="Disjoint unit ownership; aggregate only after both workers finish",
                  timing_scope="Worker elapsed envelope; concurrent timers overlap and are not dedicated GPU-hours")
    record["source_models" if args.stage == "pretrain" else "candidate_runs"] = len(args.units) * (4 if args.stage == "pretrain" else 45)
    exclusive_json(record_path, record)
    for lock in locks:
        lock.unlink()
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
