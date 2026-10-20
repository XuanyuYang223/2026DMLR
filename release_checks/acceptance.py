"""Accept an installed wheel using disposable CPU-only installation fixtures.

Run with an isolated Python interpreter from outside the repository:
  python -I /path/to/checkout/release_checks/acceptance.py \
      --repo /path/to/checkout --out /path/to/new/output

This does not train a model, load weights, or evaluate an experimental test set.
The tiny generated pools are installation fixtures, not research datasets.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stderr
import csv
from importlib import metadata
from io import StringIO
import json
import math
import os
from pathlib import Path
import platform
import shutil
import site
import subprocess
import sys
import time


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rejects(function, exceptions=(ValueError, TypeError, KeyError)):
    try:
        function()
    except exceptions as exc:
        return {"type": type(exc).__name__, "message": str(exc)}
    raise AssertionError("Invalid input was accepted")


def run(repo: Path, out: Path):
    started = time.perf_counter()
    repo, out = repo.resolve(), out.resolve()
    require(not out.exists(), "Use a new output directory; existing evidence is never overwritten")
    require(not out.is_relative_to(repo), "Place disposable output outside the checkout")
    require(not Path.cwd().resolve().is_relative_to(repo), "Run from outside the checkout")
    require(sys.prefix != sys.base_prefix, "A newly created virtual environment is required")
    configuration = Path(sys.prefix) / "pyvenv.cfg"
    require(configuration.exists(), "Virtual-environment metadata is missing")
    cfg = configuration.read_text(encoding="utf8").lower()
    require("include-system-site-packages = false" in cfg, "System third-party packages must be isolated")
    require(sys.flags.isolated == 1, "Run Python with -I to ignore PYTHONPATH and user packages")
    require(site.ENABLE_USER_SITE is False, "User site packages must be disabled")

    import permworld_platform
    from permworld_platform import label
    from permworld_platform.data import (
        DATA_VERSION, read_samples,
    )
    from permworld_platform.encoding import (
        VOCABULARY, decode_answer_ids, decode_value, encode_example, encode_value,
    )
    from permworld_platform.registry import export_registry, list_task_specs
    from permworld_platform.relations import historical_references, list_relations
    from permworld_release.cli import _demo_scope
    from permworld_release import generate_dataset, verify_dataset
    import numpy

    installed = Path(permworld_platform.__file__).resolve()
    require(not installed.is_relative_to(repo), "Import resolved to checkout rather than installed wheel")
    require(installed.is_relative_to(Path(sys.prefix).resolve()), "Package is outside the new environment")
    dist = metadata.distribution("permworld-platform")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    require(not direct.get("dir_info", {}).get("editable", False), "Editable installation is not accepted")
    out.mkdir(parents=True)

    p = (3, 1, 4, 2)
    fixtures = [
        ("cycle_count", p, {}, None, 1),
        ("lis_length", p, {}, None, 2),
        ("components", p, {}, None, 1),
        ("fixed_points", p, {}, None, 0),
        ("increasing_subsequence_count", p, {"k": 2}, None, 3),
        ("increasing_subsequence_count", p, {"k": 5}, None, 0),
        ("cycle_length_count", p, {"k": 4}, None, 1),
        ("component_length_count", (2, 1, 4, 3), {"k": 2}, None, 2),
        ("to_cycle", (2, 1, 4, 3), {}, None, ((1, 2), (3, 4))),
        ("compose", (2, 1, 3), {}, {"q": (1, 3, 2)}, (2, 3, 1)),
        ("power", (2, 3, 1), {"exponent": 2}, None, (3, 1, 2)),
        ("pattern_avoidance", (3, 2, 1), {"pattern": (1, 2, 3)}, None, True),
    ]
    labels = []
    for task, permutation, params, operands, expected in fixtures:
        actual = label(task, permutation, params, operands)
        require(actual == expected, f"Incorrect exact fixture: {task}")
        sample = encode_example(task, permutation, params, operands, max_seq_len=128)
        require(sample["input_ids"][sample["prompt_length"]:] == sample["answer_ids"],
                f"Prompt/answer split failed: {task}")
        require(decode_answer_ids(sample["answer_ids"], boolean=isinstance(expected, bool)) == expected,
                f"Full-answer round trip failed: {task}")
        labels.append({"task_id": task, "params": params, "answer": actual,
                       "answer_tokens_including_eos": len(sample["answer_ids"])})

    wide = encode_example("increasing_subsequence_count", tuple(range(1, 13)), {"k": 3})
    require(wide["answer_value"] == math.comb(12, 3) == 220, "Incorrect multi-token count")
    require(len(wide["answer_ids"]) > 2, "The answer must exercise the multi-token protocol")
    require(decode_answer_ids(wide["answer_ids"]) == 220, "Multi-token answer was truncated")
    huge = math.comb(30, 15)
    require(decode_value(encode_value(huge)) == huge, "Large integer encoding was truncated")
    rejected = {
        "missing_parameter": rejects(lambda: label("cycle_length_count", p)),
        "zero_parameter": rejects(lambda: label("cycle_length_count", p, {"k": 0})),
        "boolean_parameter": rejects(lambda: label("cycle_length_count", p, {"k": True})),
        "duplicate_input_value": rejects(lambda: label("cycle_count", (1, 1, 3))),
        "missing_binary_operand": rejects(lambda: label("compose", p)),
        "answer_without_eos": rejects(lambda: decode_answer_ids(wide["answer_ids"][:-1])),
        "insufficient_context": rejects(
            lambda: encode_example("increasing_subsequence_count", tuple(range(1, 13)), {"k": 3},
                                   max_seq_len=4)),
    }
    # The supported installed CLI must not interpret a missing history as a
    # certificate of historical exclusion.
    diagnostics = StringIO()
    with redirect_stderr(diagnostics):
        rejected["wheel_cli_history_policy"] = rejects(
            lambda: _demo_scope(argparse.ArgumentParser(), {"historical_exclusion": True}),
            exceptions=(SystemExit,))
    require("not bundled in the wheel" in diagnostics.getvalue(), "Missing history guard diagnostic")
    rejected["safe_api_missing_config"] = rejects(lambda: generate_dataset(out / "must_not_exist", None))
    rejected["safe_api_missing_policy"] = rejects(lambda: generate_dataset(out / "must_not_exist", {}))
    rejected["safe_api_history_policy"] = rejects(
        lambda: generate_dataset(out / "must_not_exist", {"historical_exclusion": True}))
    require(not (out / "must_not_exist").exists(), "Rejected data policy created output")

    # Reuse byte-verified mathematical certificates; no historical formula rerun.
    history = historical_references()
    relations = list_relations()
    registry = export_registry(out / "registry", benchmark_samples=2)
    require(len(list_task_specs()) == 54, "Registered endpoint count changed")
    require(history["historical_rechecks_run"] == 0, "Historical checks were unexpectedly rerun")
    require(len(relations) == 74, "Historical plus platform relation count changed")

    config = {
        "version": DATA_VERSION, "lengths": [8, 10],
        "replicates": [
            {"model_seed": 19001, "data_seed": 2026101901},
            {"model_seed": 19002, "data_seed": 2026101902},
        ],
        "per_length_counts": {"train": 8, "dev": 4, "test": 4},
        "proposal": "uniform permutation proposals; reject occupied I/R/C orbits",
        "split_order": ["train", "dev", "test"],
        "historical_exclusion": False,
        "max_proposals_per_pool": 10000,
        "test_use": "disposable installation fixture only; no model evaluation or research use",
        "task_instances": [
            {"task_id": "cycle_count"}, {"task_id": "lis_length"},
            {"task_id": "components"}, {"task_id": "fixed_points"},
            {"task_id": "increasing_subsequence_count", "params": {"k": 3}},
            {"task_id": "cycle_length_count", "params": {"k": 2}},
        ],
    }
    (out / "demo_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf8")
    generated = generate_dataset(out / "dataset", config)
    quality = verify_dataset(out / "dataset", recompute_labels=True)
    require(quality["objects_checked"] == 64 and quality["labels_recomputed"] == 384,
            "Tiny data acceptance covered an unexpected scope")
    require(quality["historical_orbit_overlap"] is None and quality["historical_exclusion"] is False,
            "Unavailable historical overlap must be unknown, not a zero-overlap result")
    # Repeated calls preserve an immutable same-config dataset.
    require(generate_dataset(out / "dataset", config) == generated, "Same-config reuse changed metadata")
    replay = generate_dataset(out / "dataset_replay", config)
    fields = ("canonical_object_content_sha256", "label_sha256", "count", "proposals")
    require([[r[f] for f in fields] for r in generated["pools"]] ==
            [[r[f] for f in fields] for r in replay["pools"]], "Seeded regeneration differed")
    rows = list(read_samples(out / "dataset", 19001, "train",
                             "increasing_subsequence_count", {"k": 3}))
    require(len(rows) == 16 and Counter(row["n"] for row in rows) == {8: 8, 10: 8},
            "Reader returned the wrong split/length counts")
    for row in rows:
        require(decode_answer_ids(row["answer_ids"]) == row["answer_value"], "Reader answer mismatch")
        require(row["input_ids"][row["prompt_length"]:] == row["answer_ids"], "Reader label mask boundary")
    with (out / "dataset/answer_distributions.csv").open(encoding="utf8", newline="") as handle:
        distributions = list(csv.DictReader(handle))
    require(len(distributions) == 72, "Missing per-length task distributions")
    require(all(0 <= float(row["train_majority_accuracy"]) <= 1 for row in distributions),
            "Invalid train-derived majority accuracy")
    corrupt = out / "corrupt_dataset"
    shutil.copytree(out / "dataset", corrupt)
    corrupt_label = corrupt / generated["pools"][0]["label_path"]
    with corrupt_label.open("a", encoding="utf8") as handle:
        handle.write("\n")
    rejected["corrupt_dataset"] = rejects(lambda: next(read_samples(
        corrupt, 19001, "train", "increasing_subsequence_count", {"k": 3})))
    require("torch" not in sys.modules, "CPU interface acceptance imported a training runtime")
    # Exercise the actual installed console script as well as the Python API.
    console = Path(sys.executable).parent / ("permworld-data.exe" if sys.platform == "win32" else "permworld-data")
    require(console.is_file(), "Installed console entry point is missing")
    cli_commands = [
        [str(console), "--help"],
        [str(console), "generate", "--out", str(out / "dataset_cli"),
         "--config", str(out / "demo_config.json")],
        [str(console), "verify-data", "--out", str(out / "dataset_cli")],
        [str(console), "sample", "--dataset", str(out / "dataset_cli"),
         "--seed", "19001", "--role", "train", "--task", "increasing_subsequence_count",
         "--params", '{"k":3}', "--count", "1"],
    ]
    cli_records = []
    cli_env = dict(os.environ)
    cli_env.pop("PYTHONPATH", None)
    cli_env.pop("PYTHONHOME", None)
    cli_env["PYTHONNOUSERSITE"] = "1"
    for command in cli_commands:
        process = subprocess.run(command, text=True, capture_output=True, check=False, env=cli_env)
        cli_records.append({"command": command, "returncode": process.returncode,
                            "stdout": process.stdout, "stderr": process.stderr})
        require(process.returncode == 0, f"Installed CLI failed: {command}: {process.stderr}")
    (out / "CLI.json").write_text(json.dumps(cli_records, indent=2) + "\n", encoding="utf8")
    result = {
        "status": "passed",
        "scope": "installed-wheel CPU interfaces and disposable generated fixtures",
        "python": sys.version, "platform": platform.platform(), "executable": sys.executable,
        "environment_prefix": sys.prefix, "base_prefix": sys.base_prefix,
        "isolated_python": bool(sys.flags.isolated), "system_site_packages": False,
        "user_site_packages": site.ENABLE_USER_SITE, "package_file": str(installed),
        "distribution_version": dist.version, "numpy_version": numpy.__version__,
        "editable_install": False, "outside_checkout_import": True,
        "representative_exact_fixtures": labels, "rejections": rejected,
        "vocabulary_size": len(VOCABULARY),
        "multi_token_value": wide["answer_value"],
        "multi_token_answer_length": len(wide["answer_ids"]),
        "large_integer_roundtrip": huge,
        "registry_export_endpoints": registry["task_endpoints"],
        "relation_entries": len(relations), "historical_certificate_reuse": history,
        "generated_objects": quality["objects_checked"],
        "exact_labels_recomputed": quality["labels_recomputed"],
        "quality": quality, "reader_samples": len(rows), "distribution_rows": len(distributions),
        "seeded_regeneration": "same object content and label hashes",
        "installed_console_commands_passed": len(cli_commands),
        "historical_exclusion_enabled": False,
        "fixture_warning": "Demo inputs are isolated within this fixture only, not from historical experiments. The frozen generator's generic relation_isolation prose must be interpreted with config.historical_exclusion=false. Do not use these pools for scientific comparisons.",
        "checkpoint_files_opened": 0, "training_runs": 0, "model_test_forwards": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (out / "ACCEPTANCE.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.repo, args.out)
