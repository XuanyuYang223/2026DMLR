"""Run only the committed, endpoint-only GRU development gate.

No test loader or formal replication writer is called here. The two gate tasks
are independently optimized from the same new initialization and input streams.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.fixed_points_deficiencies_arch_v1.plan import BASE_CFG, GATE, GRU_MODEL, TRANSFORMER_MODEL
from experiments.fixed_points_deficiencies_arch_v1.models import make_model, factory_metadata, seed_all, device_setup
from experiments.fixed_points_deficiencies_arch_v1 import scalar_gate_helpers as h

OUT = HERE / "artifacts/gate"
FROZEN = HERE / "GATE_FROZEN.json"
RESULT = HERE / "artifacts/GATE_RESULT.json"


def utc():
    return datetime.now(timezone.utc).isoformat()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def committed(path):
    rel = relative(path)
    actual = git("hash-object", rel)
    expected = git("rev-parse", "HEAD:" + rel)
    if actual != expected:
        raise RuntimeError(f"Uncommitted gate binding: {rel}")


def verify_protocol():
    frozen = h.read(FROZEN)
    assert frozen["config"] == GATE, "Gate scientific configuration differs from the committed freeze"
    committed(FROZEN)
    for rel, expected in frozen["code_sha256"].items():
        path = ROOT / rel
        assert h.sha(path) == expected, f"Frozen code/plan changed: {rel}"
        committed(path)
    manifest = ROOT / frozen["gate_data_manifest_path"]
    assert h.sha(manifest) == frozen["gate_data_manifest_sha256"]
    committed(manifest)
    for rel, expected in frozen["gate_input_sha256"].items():
        assert h.sha(ROOT / rel) == expected, rel
        committed(ROOT / rel)
    assert len(frozen["gate_input_sha256"]) == 2
    assert all("test" not in Path(rel).name.lower() for rel in frozen["gate_input_sha256"])
    return frozen, h.sha(FROZEN)


def environment():
    return dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
        numpy=np.__version__, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        cpu_threads=torch.get_num_threads(), tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32, cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def raw_pools():
    from experiments.fixed_points_deficiencies_arch_v1.data_pipeline import load_gate_pool
    pools = {role: np.asarray(load_gate_pool(role)) for role in ("train", "dev")}
    assert pools["train"].shape == (GATE["train_count"], GATE["n"])
    assert pools["dev"].shape == (GATE["dev_count"], GATE["n"])
    return pools


def input_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<i2").tobytes()).hexdigest()


def initial(frozen_hash, device):
    path = OUT / "initial.pt"
    net = make_model(GATE["model_seed"], "gru", GRU_MODEL, device)
    weights = h.state(net)
    digest = h.statehash(weights)
    if path.exists():
        old = h.load(path)
        assert old["config"] == GATE and old["frozen_gate_protocol_sha256"] == frozen_hash
        assert old["model_sha256"] == h.statehash(old["model"]) == digest
        net.load_state_dict(old["model"], strict=True)
    else:
        h.save(path, dict(model=weights, model_sha256=digest, config=GATE,
            frozen_gate_protocol_sha256=frozen_hash, factory=factory_metadata("gru", GRU_MODEL),
            parameter_count=sum(v.numel() for v in net.parameters()), created_at_utc=utc()))
    assert net.token_embedding.weight.data_ptr() == net.lm_head.weight.data_ptr()
    return net, digest, path


def checkpoint(path, net, opt, rng, **record):
    weights = h.state(net)
    h.save(path, dict(model=weights, model_sha256=h.statehash(weights),
        optimizer=opt.state_dict(), config=GATE, factory=factory_metadata("gru", GRU_MODEL),
        **h.rng_state(rng), **record))


def verify_complete(folder, frozen_hash):
    path = folder / "COMPLETE.json"
    if not path.exists():
        return None
    record = h.read(path)
    assert record["status"] == "complete" and record["frozen_gate_protocol_sha256"] == frozen_hash
    assert record["updates"] == GATE["updates"]
    for name, digest in record["file_sha256"].items():
        assert h.sha(folder / name) == digest, folder / name
    return record


def smoke():
    _, frozen_hash = verify_protocol()
    marker = OUT / "SMOKE.json"
    if marker.exists():
        record = h.read(marker)
        assert record["status"] == "passed" and record["frozen_gate_protocol_sha256"] == frozen_hash
        print(record, flush=True)
        return
    device = device_setup()
    raw = raw_pools()["train"][:256]
    task = GATE["tasks"][0]
    tiny = h.encode(raw, task, device)
    net = make_model(GATE["model_seed"], "gru", GRU_MODEL, device)
    initial_weights = h.state(net)
    opt = h.optimizer(net, BASE_CFG)
    assert not opt.state
    seed_all(GATE["dropout_seed"])
    rng = np.random.default_rng(GATE["sampling_seed"])
    for step in range(1, 5):
        h.train_step(net, opt, tiny, task, rng, step, BASE_CFG)
    resume = OUT / "engineering/resume.pt"
    checkpoint(resume, net, opt, rng, step=4, frozen_gate_protocol_sha256=frozen_hash)
    for step in range(5, 9):
        h.train_step(net, opt, tiny, task, rng, step, BASE_CFG)
    expected = h.statehash(net.state_dict())
    cp = h.load(resume)
    resumed = make_model(GATE["model_seed"], "gru", GRU_MODEL, device)
    resumed.load_state_dict(cp["model"], strict=True)
    restored_opt = h.optimizer(resumed, BASE_CFG)
    restored_opt.load_state_dict(cp["optimizer"])
    resumed_rng = h.restore_rng(cp)
    for step in range(5, 9):
        h.train_step(resumed, restored_opt, tiny, task, resumed_rng, step, BASE_CFG)
    assert expected == h.statehash(resumed.state_dict()), "GRU resume is not bitwise equivalent"
    saved_rng = h.rng_state(resumed_rng)
    baseline = int(torch.bincount(tiny[1]).argmax())
    metrics = h.evaluate(resumed, tiny, task, baseline)
    assert torch.equal(saved_rng["torch_rng"], torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(saved_rng["cuda_rng"], torch.cuda.get_rng_state_all()))
    one = (tiny[0][:1], tiny[1][:1])

    class FixedOutput(torch.nn.Module):
        """Engineering fixture: a correct first token alone must not pass."""
        def __init__(self, correct_eos):
            super().__init__()
            self.correct_eos = correct_eos

        def forward(self, tokens):
            token = int(one[1][0])
            if tokens.shape[1] == one[0].shape[1] + 3:
                token = h.EOS if self.correct_eos else h.V["<PAD>"]
            logits = torch.zeros((*tokens.shape, len(h.V)), device=tokens.device)
            logits[..., token] = 1
            return logits

    answer_only = h.evaluate(FixedOutput(False), one, task, int(one[1][0]))
    answer_eos = h.evaluate(FixedOutput(True), one, task, int(one[1][0]))
    assert answer_only["legacy_accuracy"] == 1 and answer_only["accuracy"] == 0
    assert answer_eos["accuracy"] == 1
    ids, gold, eq = h.batch((tiny[0][:64], tiny[1][:64]), task)
    resumed.eval()
    with torch.inference_mode():
        logits = resumed(ids)[:, eq]
        changed = ids.clone()
        changed[:, eq + 1] = (gold + 1) % 10
        assert torch.equal(logits, resumed(changed)[:, eq]), "Causal answer leakage"
    before = h.state(resumed)
    row = h.V["<FIXED_POINTS>"]
    with torch.no_grad():
        resumed.token_embedding.weight[row].copy_(initial_weights["token_embedding.weight"][row].to(device))
    for name, tensor in resumed.state_dict().items():
        expected_tensor = before[name]
        if name in ("token_embedding.weight", "lm_head.weight"):
            expected_tensor[row].copy_(initial_weights["token_embedding.weight"][row])
        assert torch.equal(expected_tensor, tensor.cpu())
    assert resumed.token_embedding.weight.data_ptr() == resumed.lm_head.weight.data_ptr()
    fresh = h.optimizer(resumed, BASE_CFG)
    assert not fresh.state
    sync(device)
    start = time.perf_counter()
    for step in range(1, 33):
        h.train_step(resumed, fresh, tiny, task, resumed_rng, step, BASE_CFG)
    sync(device)
    seconds = time.perf_counter() - start
    # Prepare the shared untouched state before separately owned task workers
    # may start. This prevents concurrent creation of the same checkpoint.
    prepared, initial_digest, initial_path = initial(frozen_hash, device)
    del prepared
    h.write(marker, dict(status="passed", completed_at_utc=utc(), environment=environment(),
        frozen_gate_protocol_sha256=frozen_hash, commit=git("rev-parse", "HEAD"),
        strict_save_load=True, resume_bitwise_equal=True, causal_no_answer_leak=True,
        evaluation_preserves_rng=True, target_only_reset=True, tied_weights=True, fresh_optimizer=True,
        complete_answer_eos_required=True,
        fixture_sha256=input_hash(raw), fixture="First 256 gate-training inputs; discarded engineering models",
        smoke_metrics=metrics, timing_updates=32, timing_seconds=seconds,
        seconds_per_update=seconds / 32, initial_model_sha256=initial_digest,
        initial_checkpoint_sha256=h.sha(initial_path), test_evaluated=False))
    print({"status": "passed", "smoke": relative(marker), "seconds_per_update": seconds / 32}, flush=True)


def expected_stream(step_count, pool_size):
    rng = np.random.default_rng(GATE["sampling_seed"])
    digest = hashlib.sha256()
    for _ in range(step_count):
        indices = rng.choice(pool_size, GATE["batch"], replace=False)
        slots = np.zeros(GATE["batch"], dtype=np.int64)
        rng.shuffle(slots)
        digest.update(np.asarray(indices, dtype="<i8").tobytes())
        digest.update(slots.astype("<i8").tobytes())
    return digest


def run_task(task):
    _, frozen_hash = verify_protocol()
    smoke_record = h.read(OUT / "SMOKE.json")
    assert smoke_record["status"] == "passed" and smoke_record["frozen_gate_protocol_sha256"] == frozen_hash
    assert h.sha(OUT / "initial.pt") == smoke_record["initial_checkpoint_sha256"]
    if task not in GATE["tasks"]:
        raise ValueError(task)
    folder = OUT / task
    complete = verify_complete(folder, frozen_hash)
    if complete:
        return complete
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "worker.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()} started={utc()}\n")
    try:
        device = device_setup()
        raw = raw_pools()
        train, dev = (h.encode(raw[role], task, device) for role in ("train", "dev"))
        net, initial_digest, initial_path = initial(frozen_hash, device)
        opt = h.optimizer(net, BASE_CFG)
        assert len(opt.state) == 0
        seed_all(GATE["dropout_seed"])
        rng = np.random.default_rng(GATE["sampling_seed"])
        baseline = int(torch.bincount(train[1]).argmax())
        exposures = np.zeros(len(train[0]), dtype=np.int64)
        start_step, curves, losses, elapsed, resumes = 0, [], [], 0.0, []
        started_at = utc()
        last = folder / "last.pt"
        if last.exists():
            cp = h.load(last)
            assert cp["config"] == GATE and cp["task"] == task
            assert cp["frozen_gate_protocol_sha256"] == frozen_hash and cp["initial_model_sha256"] == initial_digest
            assert cp["model_sha256"] == h.statehash(cp["model"])
            net.load_state_dict(cp["model"], strict=True)
            opt.load_state_dict(cp["optimizer"])
            rng = h.restore_rng(cp)
            start_step, curves, losses = cp["step"], cp["curves"], cp["losses"]
            elapsed, exposures, started_at = cp["elapsed_seconds"], cp["exposures"], cp["started_at_utc"]
            resumes = cp["resumes"] + [dict(at_step=start_step, resumed_at_utc=utc())]
        stream = expected_stream(start_step, len(train[0]))
        sync(device)
        begin = time.perf_counter()

        def measure(step):
            for split, data in (("train", train), ("dev", dev)):
                curves.append(dict(task=task, architecture="gru", step=step, split=split, length=GATE["n"],
                    elapsed_seconds=elapsed + time.perf_counter() - begin,
                    lr=0.0 if step == 0 else h.source_lr(step), **h.evaluate(net, data, task, baseline)))
            h.csvwrite(folder / "curves.csv", curves)

        if not curves:
            measure(0)
        for step in range(start_step + 1, GATE["updates"] + 1):
            loss, grad, indices, slots = h.train_step(net, opt, train, task, rng, step, BASE_CFG)
            exposures[indices] += 1
            stream.update(np.asarray(indices, dtype="<i8").tobytes())
            stream.update(slots.astype("<i8").tobytes())
            losses.append(dict(task=task, step=step, loss=loss, gradient_norm=grad, lr=h.source_lr(step)))
            if step in GATE["eval_steps"]:
                measure(step)
                print({"gate_task": task, "step": step, "dev_exact": curves[-1]["accuracy"],
                       "dev_answer_ce": curves[-1]["answer_ce"]}, flush=True)
            if step % 100 == 0 or step == GATE["updates"]:
                sync(device)
                checkpoint(last, net, opt, rng, task=task, step=step, curves=curves, losses=losses,
                    exposures=exposures, elapsed_seconds=elapsed + time.perf_counter() - begin,
                    started_at_utc=started_at, resumes=resumes, initial_model_sha256=initial_digest,
                    frozen_gate_protocol_sha256=frozen_hash)
        sync(device)
        seconds = elapsed + time.perf_counter() - begin
        assert stream.hexdigest() == expected_stream(GATE["updates"], len(train[0])).hexdigest()
        assert int(exposures.sum()) == GATE["updates"] * GATE["batch"]
        h.csvwrite(folder / "losses.csv", losses)
        h.csvwrite(folder / "input_exposures.csv", [dict(task=task, input_index=i, exposures=int(count))
                                                     for i, count in enumerate(exposures)])
        endpoint = folder / "endpoint.pt"
        checkpoint(endpoint, net, opt, rng, task=task, step=GATE["updates"],
            initial_model_sha256=initial_digest, frozen_gate_protocol_sha256=frozen_hash,
            input_stream_sha256=stream.hexdigest(), input_hashes={k: input_hash(v) for k, v in raw.items()})
        endpoint_train, endpoint_dev = curves[-2:]
        assert endpoint_train["step"] == endpoint_dev["step"] == GATE["updates"]
        sequence_length = train[0].shape[1] + 4
        sample_exposures = int(exposures.sum())
        record = dict(status="complete", task=task, architecture="gru", model=GRU_MODEL,
            factory=factory_metadata("gru", GRU_MODEL), parameter_count=sum(p.numel() for p in net.parameters()),
            model_seed=GATE["model_seed"], data_seed=GATE["data_seed"], sampling_seed=GATE["sampling_seed"],
            dropout_seed=GATE["dropout_seed"], updates=GATE["updates"], batch=GATE["batch"],
            initial_model_sha256=initial_digest, initial_checkpoint_sha256=h.sha(initial_path),
            endpoint_model_sha256=h.statehash(net.state_dict()), endpoint_train=endpoint_train, endpoint_dev=endpoint_dev,
            criterion_passed=endpoint_dev["accuracy"] >= GATE["threshold"],
            input_hashes={k: input_hash(v) for k, v in raw.items()}, input_stream_sha256=stream.hexdigest(),
            sample_exposures=sample_exposures, unique_training_inputs=int(np.count_nonzero(exposures)),
            prompt_tokens=sample_exposures * (sequence_length - 2),
            training_sequence_tokens=sample_exposures * sequence_length, supervised_tokens=sample_exposures * 2,
            minimum_exposures=int(exposures.min()), maximum_exposures=int(exposures.max()),
            elapsed_seconds=seconds, started_at_utc=started_at, finished_at_utc=utc(), resumes=resumes,
            optimizer_started_empty=True, full_parameter_training=all(p.requires_grad for p in net.parameters()),
            test_evaluated=False, checkpoint_selection="Fixed endpoint only; no best-step or test selection",
            frozen_gate_protocol_sha256=frozen_hash, commit=git("rev-parse", "HEAD"),
            timer_scope="Training, source train/dev evaluation and periodic checkpoints; final export excluded",
            file_sha256={name: h.sha(folder / name) for name in ("endpoint.pt", "curves.csv", "losses.csv", "input_exposures.csv")})
        h.write(folder / "COMPLETE.json", record)
        return record
    finally:
        lock.unlink()


def decide():
    frozen, frozen_hash = verify_protocol()
    records = [verify_complete(OUT / task, frozen_hash) for task in GATE["tasks"]]
    assert all(records), "Both gate tasks must complete before any architecture selection"
    assert len({r["initial_model_sha256"] for r in records}) == 1
    assert len({r["input_stream_sha256"] for r in records}) == 1
    assert records[0]["input_hashes"] == records[1]["input_hashes"]
    passed = all(r["endpoint_dev"]["accuracy"] >= GATE["threshold"] for r in records)
    selection = "gru" if passed else "transformer"
    payload = dict(status="complete", passed=passed, selected_architecture=selection,
        selected_model_config=GRU_MODEL if passed else TRANSFORMER_MODEL,
        frozen_gate_protocol_sha256=frozen_hash, gate_data_manifest_sha256=frozen["gate_data_manifest_sha256"],
        criterion=GATE["criterion"], threshold=GATE["threshold"], runs=records,
        gate_weights_reused_for_formal_runs=False, test_evaluated=False,
        gate_is_confirmatory_unit=False, completed_at_utc=utc())
    # Task workers may finish together. An exclusive decision lock prevents
    # competing writes, without counting either process as a new repeat.
    decision_lock = RESULT.with_suffix(".lock")
    try:
        with decision_lock.open("x", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()} started={utc()}\n")
    except FileExistsError:
        print({"status": "decision_owned_by_another_worker", "path": relative(decision_lock)}, flush=True)
        return None
    try:
        if RESULT.exists():
            existing = h.read(RESULT)
            payload["completed_at_utc"] = existing["completed_at_utc"]
            assert existing == payload, "Preserve the existing architecture decision"
        else:
            h.write(RESULT, payload)
    finally:
        decision_lock.unlink()
    print({"status": "complete", "passed": passed, "selected_architecture": selection,
           "endpoint_dev": {r["task"]: r["endpoint_dev"]["accuracy"] for r in records}}, flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "run", "verify"))
    parser.add_argument("--task", choices=GATE["tasks"])
    args = parser.parse_args()
    verify_protocol()
    if args.command == "smoke":
        smoke()
    elif args.command == "run":
        for task in ([args.task] if args.task else GATE["tasks"]):
            run_task(task)
        if all((OUT / task / "COMPLETE.json").exists() for task in GATE["tasks"]):
            decide()
    else:
        decide()


if __name__ == "__main__":
    main()
