"""Execute disjoint paired units without concurrent shared-table aggregation.

This scheduling wrapper changes no frozen training, selection or random stream.
Use two workers on disjoint units, then aggregate only after both have finished.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import time

import train
from design import REPEATS, ROUTES, SIZES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", nargs="+", required=True,
                        choices=[r["repeat_id"] for r in REPEATS])
    args = parser.parse_args()
    assert len(set(args.units)) == len(args.units)
    train.verify_protocol()
    # Both workers start only after the complete source inventory exists.
    for repeat in REPEATS:
        for condition in train.CONDITIONS:
            assert train.valid_completed(train.source_dir(repeat["repeat_id"], condition))
    worker_id = "_".join(args.units)
    started = datetime.now(timezone.utc).isoformat()
    begin = time.perf_counter()
    print(json.dumps(dict(worker=worker_id, started_at_utc=started, units=args.units)), flush=True)
    for repeat_id in args.units:
        for condition in ROUTES:
            for size in SIZES:
                for lr_index in range(3):
                    train.adapt_one(repeat_id, condition, size, lr_index)
    record = dict(status="complete", units=args.units, candidate_runs=len(args.units)*45,
        started_at_utc=started, finished_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-begin,
        wrapper_sha256=train.sha(__file__),
        protocol_sha256=train.sha(train.OUT / "FROZEN_PROTOCOL.json"),
        scheduling="Separate process per disjoint unit group; no concurrent aggregate writes")
    train.write(train.OUT / f"ADAPT_WORKER_{worker_id}.json", record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
