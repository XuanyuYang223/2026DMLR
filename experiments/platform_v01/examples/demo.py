"""Small CPU-only end-to-end example: objects -> exact labels -> LM samples."""
import argparse
import json
import sys
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from permworld_platform.data import DEFAULT_CONFIG, generate_dataset, read_samples, write_json
from permworld_platform.encoding import decode_answer_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/platform_v01/examples/small_dataset")
    args = parser.parse_args()
    config = {**DEFAULT_CONFIG,
              "replicates": [{"model_seed": 9001, "data_seed": 9002}],
              "per_length_counts": {"train": 16, "dev": 4, "test": 4},
              "historical_exclusion": False,
              "task_instances": [{"task_id": "cycle_count", "params": {}},
                                 {"task_id": "increasing_subsequence_count", "params": {"k": 4}}]}
    manifest = generate_dataset(args.out, config)
    samples = list(islice(read_samples(args.out, 9001, "train", "increasing_subsequence_count", {"k": 4}), 2))
    assert all(decode_answer_ids(s["answer_ids"]) == s["answer_value"] for s in samples)
    result = dict(status="passed", purpose="CPU integration demo only; never calibration training data",
                  historical_exclusion=False, pools=len(manifest["pools"]), samples=samples)
    write_json(args.out.parent / "small_config.json", config)
    write_json(args.out.parent / "DEMO_RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
