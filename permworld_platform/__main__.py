"""CLI for registry inspection, fixed data generation, and model-ready samples."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    registry = sub.add_parser("registry")
    registry.add_argument("--out", type=Path, default=Path("experiments/platform_v01/registry"))
    generate = sub.add_parser("generate")
    generate.add_argument("--out", type=Path, default=Path("experiments/platform_v01/dataset"))
    generate.add_argument("--config", type=Path)
    verify = sub.add_parser("verify-data")
    verify.add_argument("--out", type=Path, default=Path("experiments/platform_v01/dataset"))
    sample = sub.add_parser("sample")
    sample.add_argument("--dataset", type=Path, default=Path("experiments/platform_v01/dataset"))
    sample.add_argument("--seed", type=int, default=617)
    sample.add_argument("--role", choices=["train", "dev", "test"], default="train")
    sample.add_argument("--task", default="increasing_subsequence_count")
    sample.add_argument("--params", default='{"k":3}')
    sample.add_argument("--count", type=int, default=2)
    args = parser.parse_args()
    if args.command == "registry":
        from .registry import export_registry
        result = export_registry(args.out)
    elif args.command == "generate":
        from .data import generate_dataset, DEFAULT_CONFIG
        config = json.loads(args.config.read_text(encoding="utf8")) if args.config else DEFAULT_CONFIG
        result = generate_dataset(args.out, config)
        result = dict(status="generated_and_verified", pools=len(result["pools"]), instances=len(result["instances"]), out=str(args.out))
    elif args.command == "verify-data":
        from .data import verify_dataset
        result = verify_dataset(args.out)
    else:
        from itertools import islice
        from .data import read_samples
        if args.count < 1:
            parser.error("--count must be positive")
        result = list(islice(read_samples(args.dataset, args.seed, args.role, args.task, json.loads(args.params)), args.count))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
