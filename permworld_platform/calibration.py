"""Finite, preselected single-task calibration for PermWorld platform v0.1.

No transfer conditions, task search, learning-rate search, or budget extension.
Only training and development data are opened; the platform test split stays unused.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "experiments" / "permworld_pilot" / "vendor"
sys.path.insert(0, str(VENDOR))
from neurips_permutations.models import CausalTransformer
from sdpa import accelerate

DEFAULT_OUT = ROOT / "experiments" / "platform_v01" / "calibration"
TASKS = [
    {"task_id": "cycle_count", "params": {}, "family": "cycle_decomposition"},
    {"task_id": "longest_cycle", "params": {}, "family": "cycle_decomposition"},
    {"task_id": "lis_length", "params": {}, "family": "global_subsequence"},
    {"task_id": "increasing_subsequence_count", "params": {"k": 3}, "family": "global_subsequence"},
    {"task_id": "components", "params": {}, "family": "permutation_blocks"},
    {"task_id": "largest_component", "params": {}, "family": "permutation_blocks"},
]
CFG = {
    "version": "platform_v0.1/single_task_calibration_v1",
    "representatives": TASKS,
    "representative_selection": "Two representatives per family fixed before observing calibration training outputs; retained regardless of success.",
    "repeats": [
        {"initialization_seed": 617, "data_seed": 2026102001, "sampling_and_dropout_seed": 1617},
        {"initialization_seed": 829, "data_seed": 2026102002, "sampling_and_dropout_seed": 1829},
    ],
    "lengths": [8, 10, 12],
    "train_per_length": 1024,
    "dev_per_length": 256,
    "updates": 2400,
    "batch_size": 64,
    "eval_every": 200,
    "evaluation_batch_size": 128,
    "learning_rate": 3e-4,
    "warmup": 100,
    "minimum_lr": 3e-5,
    "weight_decay": .01,
    "betas": [.9, .999],
    "epsilon": 1e-8,
    "gradient_clip": 1.,
    "model": {"d_model": 256, "layers": 4, "n_heads": 8, "mlp_ratio": 4,
              "dropout": .1, "tie_embeddings": True, "max_seq_len": 128},
    "precision": "FP32 weights/evaluation; BF16 training on CUDA; TF32 disabled",
    "loss": "Mean over examples of mean teacher-forced CE over answer tokens and EOS; prompt tokens ignored",
    "accuracy": "Unconstrained greedy generation from prompt alone; exact complete answer-token sequence including EOS",
    "max_new_tokens": 8,
    "selection": "max development exact-sequence accuracy, min development answer CE, earliest step; step zero excluded",
    "reporting": "Fixed-budget endpoint is the primary calibration reference; dev-best is a descriptive selected checkpoint, not an independently tested generalization estimate",
    "majority": "For each task/length, the smallest most frequent training answer; applied unchanged to training/development",
    "length_schedule": "Each consecutive block of three updates visits a seeded random permutation of lengths8,10,12; exactly800updates per length",
    "pairing": "Same initialization and input pools per repeat, identical length/index schedule and dropout seed across representatives; variable answer lengths can change RNG consumption",
    "test_policy": "Never load or evaluate the platform test split in calibration",
    "scope": "12 single-task runs; no adaptation/transfer, hyperparameter search, automatic training extension, or task removal",
}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(path: Path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def runtime(device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Frozen formal calibration requires CUDA; do not silently change precision/device")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device(device)


def encoding():
    from permworld_platform import encoding as enc
    return enc


def vocabulary():
    return list(encoding().VOCABULARY)


def token_id(token):
    return encoding().TOKEN_TO_ID[token]


def make_model(seed, device):
    seed_all(seed)
    return accelerate(CausalTransformer(vocab_size=len(vocabulary()), **CFG["model"])).to(device)


def save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def task_slug(spec):
    params = "".join(f"_{k}{v}" for k, v in sorted(spec["params"].items()))
    return spec["task_id"] + params


def run_id(spec, repeat):
    return f"s{repeat['initialization_seed']}_{task_slug(spec)}"


def pool_path(data_dir, repeat, role, length):
    """The platform stores task-independent permutations in immutable NPZ pools."""
    return data_dir / f"s{repeat['initialization_seed']}_{role}_n{length}.npz"


def load_pools(data_dir, spec, repeat):
    enc = encoding()
    result = {}
    for role in ("train", "dev"):
        result[role] = {}
        for length in CFG["lengths"]:
            path = pool_path(data_dir, repeat, role, length)
            permutations = np.load(path, allow_pickle=False)["p"]
            expected = CFG[f"{role}_per_length"]
            if len(permutations) != expected or permutations.shape[1] != length:
                raise ValueError(f"Wrong frozen data shape: {path}: {permutations.shape}")
            records = [enc.encode_example(spec["task_id"], tuple(map(int, p)), spec["params"], max_seq_len=CFG["model"]["max_seq_len"])
                       for p in permutations]
            for rec in records:
                assert rec["input_ids"][rec["prompt_length"]:] == rec["answer_ids"]
                assert rec["answer_ids"][-1] == token_id("<EOS>")
                assert len(rec["answer_ids"]) <= CFG["max_new_tokens"]
            result[role][length] = records
    return result


def collate(records, device):
    """Right padding with an explicit attention mask; label mask excludes the prompt."""
    width = max(len(r["input_ids"]) for r in records)
    ids = torch.full((len(records), width), token_id("<PAD>"), dtype=torch.long, device=device)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    targets = torch.full_like(ids, -100)
    answer_mask = torch.zeros_like(ids, dtype=torch.bool)
    eos_mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(records):
        values = torch.as_tensor(row["input_ids"], dtype=torch.long, device=device)
        ids[i, :len(values)] = values
        valid[i, :len(values)] = True
        start = row["prompt_length"] - 1
        stop = len(values) - 1
        targets[i, start:stop] = values[start + 1:stop + 1]
        answer_mask[i, start:stop - 1] = True
        eos_mask[i, stop - 1] = True
    return ids, valid, targets, answer_mask, eos_mask


def token_loss(logits, targets):
    return F.cross_entropy(logits.float().transpose(1, 2), targets, ignore_index=-100, reduction="none")


def step(model, optimizer, records, lr, device):
    model.train()
    ids, valid, targets, answer_mask, eos_mask = collate(records, device)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(ids, attention_mask=valid)
        losses = token_loss(logits, targets)
        mask = targets != -100
        loss = (losses.sum(1) / mask.sum(1)).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite calibration loss; retain the failed run")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    return {"loss": float(loss.detach()), "gradient_norm": float(norm),
            "input_tokens": int(valid.sum()), "padded_input_tokens": ids.numel(),
            "supervised_tokens": int(mask.sum()), "answer_tokens": int(answer_mask.sum())}


def lr_at(step_number):
    if step_number <= CFG["warmup"]:
        return CFG["learning_rate"] * step_number / CFG["warmup"]
    progress = (step_number - CFG["warmup"]) / (CFG["updates"] - CFG["warmup"])
    return CFG["minimum_lr"] + .5 * (CFG["learning_rate"] - CFG["minimum_lr"]) * (1 + math.cos(math.pi * progress))


def optimizer_for(model, device):
    return torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"],
                             betas=tuple(CFG["betas"]), eps=CFG["epsilon"], fused=device.type == "cuda")


@torch.inference_mode()
def greedy_answers(model, records, device):
    """Generate from prompts only; no gold token is ever appended to this stream."""
    lengths = {r["prompt_length"] for r in records}
    if len(lengths) != 1:
        raise ValueError("Autoregressive evaluation batches must share a prompt length")
    prompt_length = next(iter(lengths))
    ids = torch.tensor([r["input_ids"][:prompt_length] for r in records], device=device)
    generated = [[] for _ in records]
    finished = [False] * len(records)
    for _ in range(CFG["max_new_tokens"]):
        predicted = model(ids)[:, -1].argmax(-1)
        for i, value in enumerate(predicted.tolist()):
            if not finished[i]:
                generated[i].append(value)
                finished[i] = value == token_id("<EOS>")
        if all(finished):
            break
        ids = torch.cat([ids, predicted[:, None]], dim=1)
    return generated


def majority_answers(train_by_length):
    from collections import Counter
    result = {}
    for length, records in train_by_length.items():
        counts = Counter(tuple(row["answer_ids"]) for row in records)
        most = max(counts.values())
        # The exact numeric value is used for tie breaking, not token ordering.
        tied = [list(answer) for answer, count in counts.items() if count == most]
        values = {tuple(row["answer_ids"]): row["answer_value"] for row in records}
        result[length] = {"tokens": min(tied, key=lambda tokens: values[tuple(tokens)])}
        result[length]["value"] = values[tuple(result[length]["tokens"])]
    return result


@torch.inference_mode()
def evaluate(model, by_length, majority, device):
    model.eval()
    rows = []
    for length, records in by_length.items():
        counts = {"correct": 0, "eos_generated": 0, "majority_correct": 0, "count": len(records)}
        answer_ce = eos_ce = sequence_ce = 0.
        for start in range(0, len(records), CFG["evaluation_batch_size"]):
            batch = records[start:start + CFG["evaluation_batch_size"]]
            ids, valid, targets, answer_mask, eos_mask = collate(batch, device)
            logits = model(ids, attention_mask=valid).float()
            losses = token_loss(logits, targets)
            answer_ce += float((losses.mul(answer_mask).sum(1) / answer_mask.sum(1)).sum())
            eos_ce += float(losses.mul(eos_mask).sum())
            sequence_ce += float((losses.sum(1) / (targets != -100).sum(1)).sum())
            generated = greedy_answers(model, batch, device)
            for row, prediction in zip(batch, generated):
                counts["correct"] += int(prediction == row["answer_ids"])
                counts["eos_generated"] += int(prediction[-1:] == [token_id("<EOS>")])
                counts["majority_correct"] += int(row["answer_ids"] == majority[length]["tokens"])
        count = len(records)
        rows.append({"length": length, **counts, "accuracy": counts["correct"] / count,
                     "answer_ce": answer_ce / count, "eos_ce": eos_ce / count,
                     "sequence_ce": sequence_ce / count, "eos_accuracy": counts["eos_generated"] / count,
                     "majority_accuracy": counts["majority_correct"] / count,
                     "majority_answer": majority[length]["value"]})
    total = sum(r["count"] for r in rows)
    aggregate = {"length": "all", "count": total}
    for key in ("correct", "eos_generated", "majority_correct"):
        aggregate[key] = sum(r[key] for r in rows)
    for key in ("accuracy", "answer_ce", "eos_ce", "sequence_ce", "eos_accuracy", "majority_accuracy"):
        aggregate[key] = sum(r[key] * r["count"] for r in rows) / total
    aggregate["majority_answer"] = "training-derived per-length values"
    return rows + [aggregate]


def rng_state(rng):
    return {"numpy_rng": copy.deepcopy(rng.bit_generator.state), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(payload):
    rng = np.random.default_rng()
    rng.bit_generator.state = payload["numpy_rng"]
    torch.set_rng_state(payload["torch_rng"].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in payload["cuda_rng"]])
    return rng


def freeze(out, data_dir):
    if (out / "FROZEN_CALIBRATION.json").exists():
        verify_freeze(out, data_dir)
        return
    if not (out / "SMOKE.json").exists() or not read_json(out / "SMOKE.json")["passed"]:
        raise RuntimeError("Run and pass the smoke test before freezing")
    files = sorted((ROOT / "permworld_platform").glob("*.py"))
    files += sorted((ROOT / "permworld_platform").rglob("*.json"))
    files += [VENDOR / "neurips_permutations" / "models.py", VENDOR / "sdpa.py"]
    platform_out = ROOT / "experiments" / "platform_v01"
    registry_path = platform_out / "registry" / "TASK_REGISTRY.json"
    certificate_path = platform_out / "mathematics" / "CERTIFICATE.json"
    quality_path = data_dir.parent / "quality_full.json"
    interface_path = platform_out / "interface_audit.json"
    for certificate in (certificate_path, quality_path, interface_path):
        assert read_json(certificate)["status"] == "passed", f"Required pretraining validation incomplete: {certificate}"
    evidence_files = [registry_path, certificate_path, quality_path, interface_path,
                      data_dir.parent / "manifest.json", data_dir.parent / "vocabulary.json", out / "SMOKE.json"]
    data_files = [pool_path(data_dir, repeat, role, length) for repeat in CFG["repeats"]
                  for role in ("train", "dev") for length in CFG["lengths"]]
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "FROZEN_CALIBRATION.json", {
        "unix": time.time(), "config": CFG, "vocabulary": vocabulary(),
        "code_and_registry_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        "registry_data_certificate_evidence_sha256": {os.path.relpath(p, out).replace("\\", "/"): sha(p) for p in evidence_files},
        "training_development_data_sha256": {os.path.relpath(p, out).replace("\\", "/"): sha(p) for p in data_files},
        "test_split_opened": False, "formal_training_started": False,
    })


def verify_freeze(out, data_dir):
    frozen = read_json(out / "FROZEN_CALIBRATION.json")
    assert frozen["config"] == CFG, "Frozen configuration changed"
    assert frozen["vocabulary"] == vocabulary(), "Frozen vocabulary changed"
    for rel, digest in frozen["code_and_registry_sha256"].items():
        assert sha(ROOT / rel) == digest, f"Frozen code/registry changed: {rel}"
    for rel, digest in frozen["training_development_data_sha256"].items():
        assert sha(out / rel) == digest, f"Frozen data changed: {rel}"
    for rel, digest in frozen["registry_data_certificate_evidence_sha256"].items():
        assert sha(out / rel) == digest, f"Frozen registry/data/certificate evidence changed: {rel}"
    expected = {str(pool_path(data_dir, rep, role, n).resolve()) for rep in CFG["repeats"]
                for role in ("train", "dev") for n in CFG["lengths"]}
    assert expected == {str((out / rel).resolve()) for rel in frozen["training_development_data_sha256"]}
    return frozen


def smoke(out, data_dir, device):
    """Two updates, exact save/resume, answer isolation, padding, and evaluation RNG."""
    started = time.monotonic()
    repeat = CFG["repeats"][0]
    spec = TASKS[3]  # count(k=3) deliberately exercises multi-token answers.
    pools = load_pools(data_dir, spec, repeat)
    # Guarantee variable output lengths, including220 (three decimal digits),
    # rather than relying on a random batch containing an answer >=100.
    sample = [encoding().encode_example(spec["task_id"], tuple(range(1, 13)), spec["params"]),
              encoding().encode_example(spec["task_id"], tuple(range(12, 0, -1)), spec["params"]),
              *pools["train"][12][:6]]
    assert sample[0]["answer_value"] == 220 and sample[1]["answer_value"] == 0
    assert len({len(row["answer_ids"]) for row in sample}) > 1
    model = make_model(900617, device)
    optimizer = optimizer_for(model, device)
    assert not optimizer.state
    seed_all(731)
    rng = np.random.default_rng(731)
    step(model, optimizer, sample, 3e-4, device)
    cp = out / "smoke" / "resume.pt"
    save_checkpoint(cp, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), **rng_state(rng)})
    step(model, optimizer, sample, 3e-4, device)
    expected = state_hash(model.state_dict())
    restored = make_model(900617, device)
    payload = load_checkpoint(cp)
    restored.load_state_dict(payload["model"], strict=True)
    restored_optimizer = optimizer_for(restored, device)
    restored_optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload)
    step(restored, restored_optimizer, sample, 3e-4, device)
    assert state_hash(restored.state_dict()) == expected, "Save/resume mismatch"
    restored.eval()
    ids, valid, _, _, _ = collate(sample, device)
    corrupt = ids.clone()
    for i, row in enumerate(sample):
        corrupt[i, row["prompt_length"]:] = token_id("<BOS>")
    with torch.inference_mode():
        original_logits = restored(ids, attention_mask=valid)
        changed_logits = restored(corrupt, attention_mask=valid)
        for i, row in enumerate(sample):
            end = row["prompt_length"]
            assert torch.equal(original_logits[i, :end], changed_logits[i, :end]), "Answer leakage into prompt"
        # Explicit right padding must leave valid logits unchanged, within FP32 rounding.
        padded = torch.cat([ids, torch.full((len(ids), 3), token_id("<PAD>"), device=device)], dim=1)
        padded_mask = torch.cat([valid, torch.zeros((len(ids), 3), device=device, dtype=torch.bool)], dim=1)
        extended = restored(padded, attention_mask=padded_mask)
        assert torch.allclose(original_logits, extended[:, :ids.shape[1]], atol=1e-5, rtol=1e-5)
    before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    tiny = {n: pools["dev"][n][:8] for n in CFG["lengths"]}
    metrics = evaluate(restored, tiny, majority_answers(pools["train"]), device)
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))
    contaminated = copy.deepcopy(sample)
    for row in contaminated:
        row["input_ids"][row["prompt_length"]:] = [token_id("<BOS>")] * len(row["answer_ids"])
        row["answer_ids"] = [token_id("<BOS>")] * len(row["answer_ids"])
    assert greedy_answers(restored, sample, device) == greedy_answers(restored, contaminated, device)
    write_json(out / "SMOKE.json", {
        "passed": True, "unix": time.time(), "seconds": time.monotonic() - started,
        "save_resume_bitwise": True, "causal_answer_leakage_absent": True,
        "greedy_generation_independent_of_gold": True, "explicit_padding_checked": True,
        "evaluation_preserves_rng": True, "optimizer_initially_empty": True,
        "two_updates": True, "test_split_opened": False, "metrics": metrics,
        "smoke_answer_values": [r["answer_value"] for r in sample],
        "smoke_answer_token_lengths": [len(r["answer_ids"]) for r in sample],
    })
    print("Calibration smoke passed", flush=True)


def train_one(out, data_dir, spec, repeat, device):
    identity = {"run_id": run_id(spec, repeat), **repeat, **spec,
                "params": json.dumps(spec["params"], sort_keys=True)}
    run_dir = out / "runs" / identity["run_id"]
    complete = run_dir / "COMPLETE.json"
    if complete.exists():
        record = read_json(complete)
        assert record["frozen_protocol_sha256"] == sha(out / "FROZEN_CALIBRATION.json")
        for name, digest in record["checkpoint_sha256"].items():
            assert sha(run_dir / name) == digest
        print("Reusing completed calibration", identity["run_id"], flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    pools = load_pools(data_dir, spec, repeat)
    majority = majority_answers(pools["train"])
    model = make_model(repeat["initialization_seed"], device)
    initial_hash = state_hash(model.state_dict())
    initial_path = out / "initial" / f"seed{repeat['initialization_seed']}.pt"
    if initial_path.exists():
        initial = load_checkpoint(initial_path)
        assert state_hash(initial["model"]) == initial_hash
    else:
        save_checkpoint(initial_path, {"model": model.state_dict(), "initialization_seed": repeat["initialization_seed"],
                                       "config": CFG, "vocabulary": vocabulary()})
    optimizer = optimizer_for(model, device)
    assert not optimizer.state
    seed_all(repeat["sampling_and_dropout_seed"])
    rng = np.random.default_rng(repeat["sampling_and_dropout_seed"])
    curves = []
    exposure = {n: np.zeros(CFG["train_per_length"], dtype=np.int64) for n in CFG["lengths"]}
    length_updates = {n: 0 for n in CFG["lengths"]}
    totals = {"input_tokens": 0, "padded_input_tokens": 0, "supervised_tokens": 0, "answer_tokens": 0}
    totals_by_length = {n: totals.copy() for n in CFG["lengths"]}
    best_key = None
    best_step = None
    best_dev = None
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    length_order = []
    sample_schedule_hash = hashlib.sha256()
    last = {}

    def record_metrics(step_number):
        selected = None
        for role in ("train", "dev"):
            for metrics in evaluate(model, pools[role], majority, device):
                row = {**identity, "step": step_number, "split": role, **metrics,
                       "elapsed_seconds": time.monotonic() - started}
                curves.append(row)
                if role == "dev" and metrics["length"] == "all":
                    selected = metrics
        write_csv(run_dir / "curves.csv", curves)
        return selected

    record_metrics(0)
    for update in range(1, CFG["updates"] + 1):
        if not length_order:
            length_order = list(map(int, rng.permutation(CFG["lengths"])))
        length = length_order.pop()
        indices = rng.choice(CFG["train_per_length"], CFG["batch_size"], replace=False)
        sample_schedule_hash.update(np.asarray([length, *indices], dtype="<i8").tobytes())
        batch = [pools["train"][length][i] for i in indices]
        last = step(model, optimizer, batch, lr_at(update), device)
        exposure[length][indices] += 1
        length_updates[length] += 1
        for key in totals:
            totals[key] += last[key]
            totals_by_length[length][key] += last[key]
        if update % CFG["eval_every"] == 0 or update == CFG["updates"]:
            dev = record_metrics(update)
            key = (-dev["accuracy"], dev["answer_ce"], update)
            if best_key is None or key < best_key:
                best_key, best_step, best_dev = key, update, dev
                save_checkpoint(run_dir / "best.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": update, "identity": identity, "config": CFG, "development": dev,
                    "initial_state_sha256": initial_hash, "tokens": totals.copy(), **rng_state(rng)})
            print(json.dumps({"run": identity["run_id"], "step": update,
                              "dev_exact": round(dev["accuracy"], 6), "dev_answer_ce": round(dev["answer_ce"], 6),
                              "best_step": best_step, "seconds": round(time.monotonic() - started, 2)}), flush=True)
    save_checkpoint(run_dir / "endpoint.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "step": CFG["updates"], "identity": identity, "config": CFG, "development": dev,
        "initial_state_sha256": initial_hash, "tokens": totals.copy(), **rng_state(rng)})
    seconds = time.monotonic() - started
    assert set(length_updates.values()) == {CFG["updates"] // len(CFG["lengths"])}
    write_csv(run_dir / "input_exposures.csv", [
        {**identity, "length": n, "input_index": i, "exposures": int(count)}
        for n, values in exposure.items() for i, count in enumerate(values)])
    write_json(complete, {**identity, "completed_unix": time.time(), "seconds": seconds,
        "updates": CFG["updates"], "sample_exposures": CFG["updates"] * CFG["batch_size"], **totals,
        "length_updates": length_updates, "tokens_by_length": totals_by_length,
        "best_step": best_step, "best_dev": best_dev, "endpoint_dev": dev,
        "initial_state_sha256": initial_hash, "parameter_count": sum(p.numel() for p in model.parameters()),
        "length_and_index_schedule_sha256": sample_schedule_hash.hexdigest(),
        "peak_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "frozen_protocol_sha256": sha(out / "FROZEN_CALIBRATION.json"),
        "checkpoint_sha256": {name: sha(run_dir / name) for name in ("best.pt", "endpoint.pt")},
        "test_evaluated": False, "optimizer_started_empty": True,
        "training_timer_scope": "Training, all full train/dev evaluations, and checkpoints; excludes data encoding, summary and final audits",
    })
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


def train_all(out, data_dir, device):
    verify_freeze(out, data_dir)
    started = time.monotonic()
    write_json(out / "TRAINING_STARTED.json", {"unix": time.time(), "torch": torch.__version__,
        "numpy": np.__version__, "python": sys.version, "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "cuda": torch.version.cuda, "frozen_protocol_sha256": sha(out / "FROZEN_CALIBRATION.json")})
    for repeat in CFG["repeats"]:
        for spec in TASKS:
            train_one(out, data_dir, spec, repeat, device)
    verify_freeze(out, data_dir)
    write_json(out / "CALIBRATION_COMPLETE.json", {"unix": time.time(), "runs": len(TASKS) * len(CFG["repeats"]),
        "stage_seconds": time.monotonic() - started, "test_evaluated": False,
        "no_transfer_or_search": True, "frozen_protocol_sha256": sha(out / "FROZEN_CALIBRATION.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["smoke", "freeze", "train", "verify"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "experiments" / "platform_v01" / "dataset" / "data")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    out, data_dir = args.out.resolve(), args.data_dir.resolve()
    if args.stage == "freeze":
        freeze(out, data_dir)
    elif args.stage == "verify":
        verify_freeze(out, data_dir)
        print("Frozen calibration protocol verified")
    else:
        device = runtime(args.device)
        if args.stage == "smoke":
            smoke(out, data_dir, device)
        elif args.stage == "train":
            if args.device != "cuda":
                raise RuntimeError("Formal frozen calibration requires CUDA")
            train_all(out, data_dir, device)


if __name__ == "__main__":
    main()
