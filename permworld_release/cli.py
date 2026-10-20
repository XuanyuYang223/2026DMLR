"""Explicit-output CLI for an installed PermWorld data package.

Historical research generators remain in the checkout. The wheel contains exact
oracles and mathematical certificates, not the historical experiment input pools.
It therefore cannot certify exclusion against those pools.
"""
from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path


def _demo_scope(parser, config):
    from .data import require_disposable_config
    try:
        require_disposable_config(config)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    registry = sub.add_parser("registry", help="Export definitions and certificate references")
    registry.add_argument("--out", type=Path, required=True)
    generate = sub.add_parser("generate", help="Generate explicitly scoped disposable data")
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--config", type=Path, required=True)
    verify = sub.add_parser("verify-data", help="Replay and verify disposable generated data")
    verify.add_argument("--out", type=Path, required=True)
    sample = sub.add_parser("sample", help="Read complete model-ready samples")
    sample.add_argument("--dataset", type=Path, required=True)
    sample.add_argument("--seed", type=int, required=True)
    sample.add_argument("--role", choices=["train", "dev", "test"], default="train")
    sample.add_argument("--task", required=True)
    sample.add_argument("--params", default="{}")
    sample.add_argument("--count", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "registry":
        from permworld_platform.registry import export_registry
        result = export_registry(args.out)
    elif args.command == "generate":
        from .data import generate_dataset
        config = json.loads(args.config.read_text(encoding="utf8"))
        _demo_scope(parser, config)
        manifest = generate_dataset(args.out, config)
        result = {
            "status": "generated_and_verified", "out": str(args.out),
            "pools": len(manifest["pools"]), "instances": len(manifest["instances"]),
            "historical_exclusion": False,
            "scope": "Disposable data; isolated across its own splits and repeats only",
        }
    elif args.command == "verify-data":
        from .data import verify_dataset
        config = json.loads((args.out / "manifest.json").read_text(encoding="utf8"))["config"]
        _demo_scope(parser, config)
        result = verify_dataset(args.out)
        result = {**result, "historical_exclusion": False,
                  "historical_overlap": "not checked; historical pools are not part of the wheel"}
    else:
        from permworld_platform.data import read_samples
        if args.count < 1:
            parser.error("--count must be positive")
        result = list(islice(read_samples(args.dataset, args.seed, args.role,
                                         args.task, json.loads(args.params)), args.count))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
