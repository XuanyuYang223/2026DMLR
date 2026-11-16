"""New calibration driver reusing immutable v0.1 functions and train/dev pools.

The numeric configuration, optimizer, full-answer loss and eight-token generation
budget are unchanged. Only the original scalar-representative loader eligibility
assertion is expanded: long gold answers remain intact and their capped exact
accuracy is identified by ENCODING_PREFLIGHT.csv. No historical file is written.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from permworld_platform import calibration as legacy

OLD_CONFIG = copy.deepcopy(legacy.CFG)
OLD_TASKS = copy.deepcopy(legacy.TASKS)
OLD_MAKE_MODEL = legacy.make_model
DATA_DIR = ROOT / "experiments/platform_v01/dataset/data"
OLD_FREEZE = ROOT / "experiments/platform_v01/calibration/FROZEN_CALIBRATION.json"
PROTOCOL = HERE / "FROZEN_PROTOCOL.json"
_protocol = None
_initial_hashes = {}
_engineering = False


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def git(*args):
    return subprocess.check_output(["git", "--no-optional-locks", *args], cwd=ROOT,
                                   text=True).strip()


def require_committed(path):
    relative = Path(path).resolve().relative_to(ROOT).as_posix()
    committed = git("rev-parse", f"HEAD:{relative}")
    working = git("hash-object", relative)
    if committed != working:
        raise RuntimeError(f"Protocol/source is not committed in its current form: {relative}")


def verify_protocol():
    global _protocol
    frozen = read_json(PROTOCOL)
    require_committed(PROTOCOL)
    if sha(OLD_FREEZE) != frozen["inherited_config_sha256"]:
        raise ValueError("Inherited calibration configuration changed")
    allowed_metadata = {"version", "representatives", "representative_selection", "scope", "reporting", "pairing"}
    for key, value in OLD_CONFIG.items():
        if key not in allowed_metadata and frozen["config"].get(key) != value:
            raise ValueError(f"Numeric or scientific configuration changed: {key}")
    for key in ("code_sha256", "train_dev_sha256"):
        for relative, digest in frozen[key].items():
            path = ROOT / relative
            if sha(path) != digest:
                raise ValueError(f"Bound {key} bytes changed: {relative}")
            require_committed(path)
    expected_pools = {str((DATA_DIR / f"s{seed}_{role}_n{n}.npz").relative_to(ROOT).as_posix())
                      for seed in (617, 829) for role in ("train", "dev") for n in (8, 10, 12)}
    if set(frozen["train_dev_sha256"]) != expected_pools:
        raise ValueError("Protocol must bind exactly the inherited twelve train/dev pools")
    preflight = read_json(HERE / "ENCODING_PREFLIGHT.json")
    if frozen["accepted_instances"] != preflight["accepted_instances"]:
        raise ValueError("Accepted instances differ from the complete encoding preflight")
    if frozen["rejected_instances"] != preflight["rejected_instances"]:
        raise ValueError("Reader-rejected instances differ from the preflight")
    if len(frozen["accepted_instances"]) != 49 or len(frozen["rejected_instances"]) != 1:
        raise ValueError("Unexpected scope; no task can be silently removed")
    for seed in (617, 829):
        binding = frozen["reused_initial_checkpoint_sha256"][str(seed)]
        if sha(ROOT / binding["path"]) != binding["sha256"]:
            raise ValueError(f"Historical initialization checkpoint changed for seed{seed}")
    _protocol = frozen
    legacy.CFG = copy.deepcopy(frozen["config"])
    legacy.TASKS = copy.deepcopy(frozen["accepted_instances"])
    legacy.load_pools = load_pools
    legacy.task_slug = task_slug
    legacy.make_model = make_model
    return frozen


def task_slug(spec):
    value = spec.get("instance_id") or spec["task_id"]
    if spec.get("params") and "instance_id" not in spec:
        value += "_" + json.dumps(spec["params"], sort_keys=True, separators=(",", ":"))
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    if not slug or len(slug) > 100:
        slug = spec["task_id"] + "_" + hashlib.sha256(value.encode()).hexdigest()[:16]
    return slug


def load_pools(data_dir, spec, repeat):
    if _protocol is None:
        raise RuntimeError("Verify the committed protocol before reading calibration data")
    if Path(data_dir).resolve() != DATA_DIR.resolve():
        raise ValueError("Only the author-approved canonical train/dev directory is allowed")
    if not _engineering and spec not in _protocol["accepted_instances"]:
        raise ValueError("Requested task is not in the accepted frozen scope")
    result = {}
    for role in ("train", "dev"):
        result[role] = {}
        for length in legacy.CFG["lengths"]:
            path = legacy.pool_path(DATA_DIR, repeat, role, length)
            relative = path.relative_to(ROOT).as_posix()
            if sha(path) != _protocol["train_dev_sha256"][relative]:
                raise ValueError(f"Canonical input bytes changed: {relative}")
            with np.load(path, allow_pickle=False) as archive:
                objects = archive["p"].copy()
            if objects.shape != (legacy.CFG[f"{role}_per_length"], length):
                raise ValueError("Canonical input shape differs from the inherited configuration")
            records = [legacy.encoding().encode_example(spec["task_id"], tuple(map(int, p)),
                       spec["params"], max_seq_len=legacy.CFG["model"]["max_seq_len"]) for p in objects]
            for record in records:
                if record["input_ids"][record["prompt_length"]:] != record["answer_ids"]:
                    raise AssertionError("Full answer-token supervision was altered")
                if record["answer_ids"][-1] != legacy.token_id("<EOS>"):
                    raise AssertionError("The gold answer must include EOS")
                # Deliberately retain long gold answers. Generation remains capped
                # at the inherited eight tokens; no answer is truncated for loss.
            result[role][length] = records
    return result


def make_model(seed, device):
    model = OLD_MAKE_MODEL(seed, device)
    if seed in (617, 829):
        if seed not in _initial_hashes:
            binding = _protocol["reused_initial_checkpoint_sha256"][str(seed)]
            original = legacy.load_checkpoint(ROOT / binding["path"])
            _initial_hashes[seed] = legacy.state_hash(original["model"])
            del original
        if legacy.state_hash(model.state_dict()) != _initial_hashes[seed]:
            raise ValueError(f"Fresh seed{seed} construction differs from the v0.1 step-zero state")
    return model


def smoke(out, device):
    global _engineering
    destination = out / "engineering"
    if destination.exists():
        raise FileExistsError("Preserve the existing engineering run; choose a fresh output directory")
    _engineering = True
    tasks = legacy.TASKS
    try:
        # The inherited engineering check uses I3, variable numeric answers and
        # a distinct engineering initialization; it is not a formal task run.
        legacy.TASKS = OLD_TASKS
        legacy.smoke(destination, DATA_DIR, device)
        repeat = legacy.CFG["repeats"][0]
        spec = next(s for s in _protocol["accepted_instances"] if s["task_id"] == "to_cycle")
        records = load_pools(DATA_DIR, spec, repeat)["train"][8][:4]
        ids, valid, targets, answer_mask, eos_mask = legacy.collate(records, device)
        for i, record in enumerate(records):
            if int((targets[i] != -100).sum()) != len(record["answer_ids"]):
                raise AssertionError("Long structured gold supervision was truncated")
            if int(answer_mask[i].sum()) != len(record["answer_ids"]) - 1 or int(eos_mask[i].sum()) != 1:
                raise AssertionError("Long structured answer/EOS masks are wrong")

        class GoldFollowingStub:
            """Engineering stub demonstrates the unchanged generation cap only."""
            def __call__(self, prefix):
                import torch
                offset = prefix.shape[1] - records[0]["prompt_length"]
                logits = torch.full((len(records), prefix.shape[1], len(legacy.vocabulary())),
                                    -1e6, device=device)
                for i, record in enumerate(records):
                    logits[i, -1, record["answer_ids"][offset]] = 1e6
                return logits

        generated = legacy.greedy_answers(GoldFollowingStub(), records, device)
        assert all(len(tokens) == 8 for tokens in generated)
        assert all(tokens == record["answer_ids"][:8] for tokens, record in zip(generated, records))
        assert all(tokens != record["answer_ids"] for tokens, record in zip(generated, records))
        write_new_json(destination / "STRUCTURED_CAP_SMOKE.json", dict(
            passed=True, created_at_utc=utc(), full_gold_supervision=True,
            generation_limit_unchanged=8, cap_limited_exact_match_demonstrated=True,
            formal_model_evaluated=False, test_pool_opened=False,
            frozen_protocol_sha256=sha(PROTOCOL), new_optimizer_updates_for_this_extension=0))
    finally:
        legacy.TASKS = tasks
        _engineering = False


def train(out, seeds, device):
    alias = out / "FROZEN_CALIBRATION.json"
    if not alias.exists() or alias.read_bytes() != PROTOCOL.read_bytes():
        raise ValueError("Before workers start, copy the committed protocol unchanged to artifacts/FROZEN_CALIBRATION.json")
    if not (out / "engineering/SMOKE.json").exists() or not read_json(out / "engineering/SMOKE.json")["passed"]:
        raise ValueError("Complete the committed engineering smoke before formal training")
    if not (out / "engineering/STRUCTURED_CAP_SMOKE.json").exists():
        raise ValueError("Complete the structured-answer wiring check before formal training")
    for seed in seeds:
        lock = out / f"WORKER_SEED_{seed}.lock"
        started_at = utc()
        write_new_json(lock, dict(seed=seed, process_id=os.getpid(), created_at_utc=started_at,
                                  frozen_protocol_sha256=sha(PROTOCOL)))
        # An exception intentionally leaves this ownership record for inspection;
        # a second process cannot silently restart or overlap this seed.
        started = time.monotonic()
        repeat = next(r for r in legacy.CFG["repeats"] if r["initialization_seed"] == seed)
        for spec in legacy.TASKS:
            run_dir = out / "runs" / legacy.run_id(spec, repeat)
            if run_dir.exists() and not (run_dir / "COMPLETE.json").exists():
                raise RuntimeError(f"Incomplete run retained for diagnosis: {run_dir}")
            legacy.train_one(out, DATA_DIR, spec, repeat, device)
        verify_protocol()
        finished_at = utc()
        elapsed = time.monotonic() - started
        worker = dict(
            status="complete", seed=seed, repeat=repeat, process_id=os.getpid(),
            started_at_utc=started_at, finished_at_utc=finished_at,
            completed_at_utc=finished_at, seconds=elapsed, elapsed_seconds=elapsed,
            run_ids=[legacy.run_id(spec, repeat) for spec in legacy.TASKS],
            runs=len(legacy.TASKS), test_pool_opened=False, new_inputs_generated=False,
            timer_scope="This worker's elapsed run span; concurrent workers overlap and must not be summed as wall time.",
            frozen_protocol_sha256=sha(PROTOCOL), training_commit=git("rev-parse", "HEAD"))
        write_new_json(out / f"WORKER_{seed}.json", worker)
        write_new_json(out / f"WORKER_SEED_{seed}_COMPLETE.json", worker)
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "train"))
    parser.add_argument("--out", type=Path, default=HERE / "artifacts")
    parser.add_argument("--seeds", type=int, nargs="+", default=[617, 829])
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(HERE.resolve()):
        raise ValueError("All new output must stay inside reuse_authorized_v1")
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= {617, 829}:
        raise ValueError("Only the two inherited seed-to-data pairings are allowed")
    verify_protocol()
    device = legacy.runtime("cuda")
    if args.stage == "smoke":
        smoke(args.out, device)
    else:
        train(args.out, args.seeds, device)


if __name__ == "__main__":
    main()
