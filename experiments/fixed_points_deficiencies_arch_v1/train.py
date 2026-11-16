"""Frozen legacy-v5 training with complete-answer evaluation and paired streams.

This module calls immutable historical tensor/model/training primitives, never
their experiment writers. All new output is confined to the requested output
directory. A test command is gated by the frozen development selections.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.pilot_v3_multitarget import pilot_v3 as legacy
from experiments.fixed_points_deficiencies_arch_v1.models import make_model
from experiments.fixed_points_deficiencies_arch_v1.design import CFG, CONDITIONS, REPEATS, ROUTES, SIZES, SOURCES, TARGET
from experiments.fixed_points_deficiencies_arch_v1.plan import FORMAL_ENGINEERING

OUT = HERE / "artifacts"
ALL_TASKS = sorted({TARGET, *(task for tasks in SOURCES.values() for task in tasks)})
V = legacy.v1.V
EOS = V["<EOS>"]
DEVICE = legacy.v1.DEVICE
TARGET_ROW = V[f"<{TARGET.upper()}>"]
POOL_CACHE = {}
assert len(legacy.v1.VOCABULARY) == CFG["vocab_size"] == 188


def make(seed):
    return make_model(seed, CFG["architecture"], CFG["model"], device=DEVICE)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write(path, obj):
    legacy.write(Path(path), obj)


def read(path):
    return legacy.read(Path(path))


def sha(path):
    return legacy.sha(Path(path))


def relative(path):
    return Path(path).resolve().relative_to(OUT.resolve()).as_posix()


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def state(net):
    return {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}


def environment():
    return dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                numpy=np.__version__, cuda=torch.version.cuda,
                device=str(DEVICE), gpu=torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
                cpu_threads=torch.get_num_threads(), tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32)


def rep(repeat_id):
    return next(r for r in REPEATS if r["repeat_id"] == repeat_id)


def initial_path(repeat_id):
    return OUT / "initial" / f"{repeat_id}.pt"


def prepare_initials():
    entries = []
    for repeat in REPEATS:
        path = initial_path(repeat["repeat_id"])
        if not path.exists():
            net = make(repeat["model_seed"])
            weights = state(net)
            legacy.save(path, dict(model=weights, repeat=repeat, model_sha256=legacy.statehash(weights), config=CFG))
            del net
        cp = legacy.load(path)
        assert cp["repeat"] == repeat and cp["config"] == CFG
        assert cp["model_sha256"] == legacy.statehash(cp["model"])
        entries.append(dict(**repeat, path=relative(path), sha256=sha(path), model_sha256=cp["model_sha256"]))
    payload = dict(status="prepared", initials=entries, environment=environment())
    manifest = OUT / "INITIALS.json"
    if manifest.exists():
        assert read(manifest)["initials"] == entries
    else:
        write(manifest, payload)
    return entries


def verify_protocol():
    from experiments.fixed_points_deficiencies_arch_v1.protocol import verify as verify_frozen
    assert OUT.resolve() == (HERE / "artifacts").resolve()
    verify_frozen()
    frozen = read(OUT / "FROZEN_PROTOCOL.json")
    assert frozen.get("status", "frozen") == "frozen"
    if "config" in frozen:
        assert frozen["config"] == CFG
    code = frozen["code_sha256"]
    for name, expected in code.items():
        assert sha(ROOT / name) == expected, f"Frozen source changed: {name}"
    for field, filename in [("data_manifest_sha256", "data/manifest.json"),
                            ("initial_manifest_sha256", "INITIALS.json")]:
        if field in frozen:
            assert sha(OUT / filename) == frozen[field], filename
    for row in read(OUT / "INITIALS.json")["initials"]:
        assert sha(OUT / row["path"]) == row["sha256"]
    return sha(OUT / "FROZEN_PROTOCOL.json")


def require_smoke(protocol_hash):
    record = read(OUT / "SMOKE.json")
    assert record["status"] == "passed" and record["protocol_sha256"] == protocol_hash
    assert record["architecture"] == CFG["architecture"] and record["model_config"] == CFG["model"]
    assert record["resume_bitwise_equal"] and record["target_only_reset"] and record["fresh_optimizer"]


def load_raw(repeat_id, role):
    from experiments.fixed_points_deficiencies_arch_v1.data_pipeline import load_pool
    return np.asarray(load_pool(OUT / "data", repeat_id, role))


def encode(permutations):
    labels = np.zeros((len(permutations), len(legacy.TASKS)), dtype=np.int64)
    for task in ALL_TASKS:
        labels[:, legacy.T[task]] = [int(legacy.v1.PF[task](tuple(map(int, q)))) for q in permutations]
    assert all(V[f"{i:02d}"] == i for i in range(11))
    return legacy.v1.tensors(permutations), torch.tensor(labels, device=DEVICE, dtype=torch.long)


def pool(repeat_id, role, size=None):
    key = (repeat_id, role)
    if key not in POOL_CACHE:
        POOL_CACHE.clear() if len(POOL_CACHE) > 5 else None
        POOL_CACHE[key] = encode(load_raw(repeat_id, role))
    x, y = POOL_CACHE[key]
    return (x, y) if size is None else (x[:size], y[:size])


def input_hash(repeat_id, role, size=None):
    values = load_raw(repeat_id, role)
    if size is not None:
        values = values[:size]
    return hashlib.sha256(np.asarray(values, dtype="<i2").tobytes()).hexdigest()


def mode(train, task):
    return int(torch.bincount(train[1][:, legacy.T[task]]).argmax())


@torch.inference_mode()
def evaluate(net, dataset, task, majority=None, raw=False):
    """Greedy exact answer plus EOS, with no token masking or answer constraints.

    Legal outputs have exactly two tokens. Generation stops at EOS or the fixed
    two-token horizon. Any missing EOS or premature EOS fails exact match. The
    teacher-forced EOS pass also provides the historical bridge and loss.
    """
    net.eval()
    x, y = dataset
    correct = legacy_correct = eos_correct = majority_correct = 0
    answer_loss = eos_loss = 0.0
    predictions = []
    for offset in range(0, len(x), 128):
        ids, gold, eq = legacy.batch((x[offset:offset+128], y[offset:offset+128]), legacy.T[task], False)
        answer_logits = net(ids)[:, eq].float()
        first = answer_logits.argmax(-1)
        nll = F.cross_entropy(answer_logits, gold, reduction="none")
        answer_loss += float(nll.sum())
        teacher_logits = net(torch.cat((ids, gold[:, None]), 1))[:, -1].float()
        teacher_second = teacher_logits.argmax(-1)
        second = teacher_second.clone()
        wrong = first != gold
        # The correct-first subset already has exactly the greedy second pass.
        if bool(wrong.any()):
            second[wrong] = net(torch.cat((ids[wrong], first[wrong, None]), 1))[:, -1].float().argmax(-1)
        first_correct = first == gold
        full_correct = first_correct & (second == EOS)
        correct += int(full_correct.sum())
        legacy_correct += int(first_correct.sum())
        eos_correct += int((teacher_second == EOS).sum())
        eos_loss += float(F.cross_entropy(teacher_logits, torch.full_like(gold, EOS), reduction="sum"))
        if majority is not None:
            majority_correct += int((gold == majority).sum())
        if raw:
            for j in range(len(gold)):
                a, b = int(first[j]), int(second[j])
                generated = [a] if a == EOS else [a, b]
                predictions.append(dict(example_index=offset+j, gold_answer_ids=[int(gold[j]), EOS],
                    predicted_ids=generated, correct=bool(full_correct[j]), legacy_correct=bool(first_correct[j]),
                    answer_nll=float(nll[j]), legacy_answer_prediction_id=a,
                    terminated_by_eos=EOS in generated, truncated=EOS not in generated))
    count = len(x)
    metrics = dict(count=count, correct=correct, accuracy=correct/count,
                   legacy_correct=legacy_correct, legacy_accuracy=legacy_correct/count,
                   answer_ce=answer_loss/count, eos_ce=eos_loss/count,
                   two_token_ce=(answer_loss+eos_loss)/(2*count), teacher_eos_accuracy=eos_correct/count)
    if majority is not None:
        metrics.update(majority_label=majority, majority_correct=majority_correct, majority_accuracy=majority_correct/count)
    return (metrics, predictions) if raw else metrics


def run_id(repeat_id, condition, size=None, lr_index=None):
    name = f"{repeat_id}_{condition}"
    return name if size is None else f"{name}_n{size}_lr{lr_index}"


def source_dir(repeat_id, condition):
    return OUT / "pretrain" / run_id(repeat_id, condition)


def adapt_dir(repeat_id, condition, size, lr_index):
    return OUT / "adapt" / run_id(repeat_id, condition, size, lr_index)


def checkpoint(path, net, ident, **extra):
    weights = state(net)
    legacy.save(path, dict(model=weights, model_sha256=legacy.statehash(weights), config=CFG,
                           identity=ident, **extra))


def initial_net(repeat_id):
    repeat = rep(repeat_id)
    cp = legacy.load(initial_path(repeat_id))
    assert cp["model_sha256"] == legacy.statehash(cp["model"])
    net = make(repeat["model_seed"])
    net.load_state_dict(cp["model"], strict=True)
    assert legacy.statehash(net.state_dict()) == cp["model_sha256"]
    assert net.token_embedding.weight.data_ptr() == net.lm_head.weight.data_ptr()
    return net, cp


def source_measure(net, train, dev, tasks, ident, step_id, elapsed):
    rows = []
    for task in tasks:
        baseline = mode(train, task)
        for split, values in [("train", train), ("dev", dev)]:
            rows.append(dict(**ident, task=task, step=step_id, split=split, length=CFG["n"],
                elapsed_seconds=elapsed, lr=0.0 if step_id == 0 else legacy.source_lr(step_id),
                **evaluate(net, values, task, baseline)))
    return rows


def restore_last(path, net, opt):
    cp = legacy.load(path)
    net.load_state_dict(cp["model"], strict=True)
    assert legacy.statehash(net.state_dict()) == cp["model_sha256"]
    opt.load_state_dict(cp["optimizer"])
    rng = legacy.restore(cp)
    return cp, rng


def save_last(path, net, opt, rng, ident, **extra):
    checkpoint(path, net, ident, optimizer=opt.state_dict(), **legacy.rngstates(rng), **extra)


def valid_completed(folder):
    path = folder / "completed.json"
    if not path.exists():
        return None
    row = read(path)
    for kind in ["endpoint", "best"]:
        if f"{kind}_sha256" in row:
            assert sha(folder / f"{kind}.pt") == row[f"{kind}_sha256"]
    assert row["protocol_sha256"] == sha(OUT / "FROZEN_PROTOCOL.json")
    return row


def stream_digest(seed, updates, pool_size, task_count):
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    for _ in range(updates):
        indices = rng.choice(pool_size, 64, replace=False)
        slots = np.tile(np.arange(task_count), 64//task_count)
        rng.shuffle(slots)
        digest.update(np.asarray(indices, dtype="<i4").tobytes())
        digest.update(np.asarray(slots, dtype="<i1").tobytes())
    return digest


def stream_hash(seed, updates, pool_size, task_count):
    return stream_digest(seed, updates, pool_size, task_count).hexdigest()


def pretrain_one(repeat_id, condition):
    protocol_hash = verify_protocol()
    require_smoke(protocol_hash)
    folder = source_dir(repeat_id, condition)
    if valid_completed(folder):
        return
    repeat, tasks = rep(repeat_id), SOURCES[condition]
    ident = dict(repeat_id=repeat_id, condition=condition, run_id=run_id(repeat_id, condition),
                 model_seed=repeat["model_seed"], data_seed=repeat["data_seed"],
                 sampling_seed=repeat["source_sampling_seed"], dropout_seed=repeat["source_dropout_seed"])
    train, dev = pool(repeat_id, "source_train"), pool(repeat_id, "source_dev")
    net, initial = initial_net(repeat_id)
    opt = legacy.optmake(net, 3e-4)
    legacy.v1.seed(repeat["source_dropout_seed"])
    rng = np.random.default_rng(repeat["source_sampling_seed"])
    start, curves, elapsed, resumes = 0, [], 0.0, []
    exposures = np.zeros((len(tasks), len(train[0])), dtype=np.int64)
    last = folder / "last.pt"
    if last.exists():
        cp, rng = restore_last(last, net, opt)
        assert cp["identity"] == ident and cp["protocol_sha256"] == protocol_hash
        start, curves, exposures, elapsed = cp["step"], cp["curves"], cp["exposures"], cp["elapsed_seconds"]
        resumes = cp.get("resumes", []) + [dict(at_step=start, at_utc=utc(), restored_optimizer=True)]
    begin = time.perf_counter()
    observed_stream = stream_digest(repeat["source_sampling_seed"], start, len(train[0]), 4)
    if not curves:
        curves = source_measure(net, train, dev, tasks, ident, 0, elapsed)
        legacy.csvwrite(folder / "curves.csv", curves)
    sequence, _, _ = legacy.batch((train[0][:64], train[1][:64]), legacy.T[tasks[0]])
    sequence_tokens, prompt_tokens = sequence.shape[1], sequence.shape[1]-2
    for step_id in range(start+1, CFG["pretrain_updates"]+1):
        loss, grad, indices, slots = legacy.step(net, opt, train, tasks, rng, legacy.source_lr(step_id))
        observed_stream.update(np.asarray(indices, dtype="<i4").tobytes())
        observed_stream.update(np.asarray(slots, dtype="<i1").tobytes())
        for slot in range(len(tasks)):
            np.add.at(exposures[slot], indices[slots == slot], 1)
        if step_id in CFG["source_eval_steps"]:
            sync()
            spent = elapsed + time.perf_counter()-begin
            rows = source_measure(net, train, dev, tasks, ident, step_id, spent)
            for row in rows:
                row.update(last_batch_loss=loss, last_grad_norm=grad)
            curves.extend(rows)
            legacy.csvwrite(folder / "curves.csv", curves)
            save_last(last, net, opt, rng, ident, step=step_id, curves=curves, exposures=exposures,
                      elapsed_seconds=elapsed+time.perf_counter()-begin, resumes=resumes, protocol_sha256=protocol_hash)
            print(f"{ident['run_id']} source {step_id}/{CFG['pretrain_updates']}", flush=True)
    sync()
    spent = elapsed+time.perf_counter()-begin
    assert observed_stream.hexdigest() == stream_hash(repeat["source_sampling_seed"], CFG["pretrain_updates"], len(train[0]), 4)
    checkpoint(folder / "endpoint.pt", net, ident, step=CFG["pretrain_updates"],
               protocol_sha256=protocol_hash, initial_sha256=sha(initial_path(repeat_id)))
    exposure_rows = []
    for slot, task in enumerate(tasks):
        count = int(exposures[slot].sum())
        assert count == CFG["pretrain_updates"]*16
        exposure_rows.append(dict(**ident, task=task, task_slot=slot, sample_exposures=count,
            unique_inputs=int((exposures[slot] > 0).sum()), pool_size=len(train[0]),
            prompt_tokens=count*prompt_tokens, training_sequence_tokens=count*sequence_tokens,
            supervised_tokens=count*2, updates=CFG["pretrain_updates"],
            min_input_exposures=int(exposures[slot].min()), max_input_exposures=int(exposures[slot].max())))
    legacy.csvwrite(folder / "exposures.csv", exposure_rows)
    np.savez_compressed(folder / "input_exposure_counts.npz", counts=exposures)
    write(folder / "completed.json", dict(**ident, status="complete", finished_at_utc=utc(), tasks=tasks,
        updates=CFG["pretrain_updates"], elapsed_seconds=spent, protocol_sha256=protocol_hash,
        initial_sha256=sha(initial_path(repeat_id)), initial_model_sha256=initial["model_sha256"],
        endpoint_sha256=sha(folder / "endpoint.pt"), endpoint_model_sha256=legacy.statehash(net.state_dict()),
        source_input_sha256=input_hash(repeat_id, "source_train"), source_dev_input_sha256=input_hash(repeat_id, "source_dev"),
        stream_sha256=observed_stream.hexdigest(),
        sample_exposures=int(exposures.sum()), prompt_tokens=int(exposures.sum())*prompt_tokens,
        training_sequence_tokens=int(exposures.sum())*sequence_tokens, supervised_tokens=int(exposures.sum())*2,
        independent_inputs=len(train[0]), resumes=resumes))
    if last.exists():
        last.unlink()
    del net, opt


def adapted_initial(repeat_id, condition):
    net, initial = initial_net(repeat_id)
    source = initial_path(repeat_id) if condition == "A" else source_dir(repeat_id, condition) / "endpoint.pt"
    cp = legacy.load(source)
    net.load_state_dict(cp["model"], strict=True)
    loaded_hash = legacy.statehash(net.state_dict())
    assert loaded_hash == cp["model_sha256"]
    before = state(net)
    with torch.no_grad():
        net.token_embedding.weight[TARGET_ROW].copy_(initial["model"]["token_embedding.weight"][TARGET_ROW].to(DEVICE))
    after = state(net)
    for name in before:
        if name in ["token_embedding.weight", "lm_head.weight"]:
            before[name][TARGET_ROW].copy_(initial["model"]["token_embedding.weight"][TARGET_ROW])
        assert torch.equal(before[name], after[name]), name
    assert net.token_embedding.weight.data_ptr() == net.lm_head.weight.data_ptr()
    audit = dict(source_checkpoint=relative(source), source_checkpoint_sha256=sha(source), source_model_sha256=loaded_hash,
        loaded_model_sha256=loaded_hash, strict_complete_load=True, target_row=TARGET_ROW,
        target_row_sha256=hashlib.sha256(after["token_embedding.weight"][TARGET_ROW].numpy().tobytes()).hexdigest(),
        target_row_matches_step0=True, only_target_tied_row_reset=True, tied_weights=True,
        adaptation_initial_model_sha256=legacy.statehash(after), initial_sha256=sha(initial_path(repeat_id)),
        full_parameter_training=all(parameter.requires_grad for parameter in net.parameters()))
    return net, audit


def adapt_one(repeat_id, condition, size, lr_index):
    protocol_hash = verify_protocol()
    require_smoke(protocol_hash)
    folder = adapt_dir(repeat_id, condition, size, lr_index)
    if valid_completed(folder):
        return
    if condition != "A":
        assert valid_completed(source_dir(repeat_id, condition))
    repeat, lr = rep(repeat_id), CFG["learning_rates"][lr_index]
    ident = dict(repeat_id=repeat_id, condition=condition, target_samples=size, lr=lr,
                 lr_index=lr_index, run_id=run_id(repeat_id, condition, size, lr_index),
                 model_seed=repeat["model_seed"], data_seed=repeat["data_seed"], sampling_seed=repeat["adapt_sampling_seed"],
                 dropout_seed=repeat["adapt_dropout_seed"])
    train, dev = pool(repeat_id, "target_train", size), pool(repeat_id, "target_dev")
    net, audit = adapted_initial(repeat_id, condition)
    opt = legacy.optmake(net, lr)
    audit["fresh_optimizer_state_entries"] = len(opt.state)
    assert len(opt.state) == 0
    baseline = mode(train, TARGET)
    legacy.v1.seed(repeat["adapt_dropout_seed"])
    rng = np.random.default_rng(repeat["adapt_sampling_seed"])
    start, curves, elapsed, best, resumes = 0, [], 0.0, None, []
    exposures = np.zeros(size, dtype=np.int64)
    last = folder / "last.pt"
    if last.exists():
        cp, rng = restore_last(last, net, opt)
        assert cp["identity"] == ident and cp["protocol_sha256"] == protocol_hash
        start, curves, elapsed, best = cp["step"], cp["curves"], cp["elapsed_seconds"], cp["best"]
        exposures, audit = cp["exposures"], cp["load_audit"]
        resumes = cp.get("resumes", []) + [dict(at_step=start, at_utc=utc(), restored_optimizer=True)]
    write(folder / "load_audit.json", audit)
    begin = time.perf_counter()
    observed_stream = stream_digest(repeat["adapt_sampling_seed"], start, size, 1)
    if not curves:
        for split, values in [("train", train), ("dev", dev)]:
            curves.append(dict(**ident, task=TARGET, step=0, split=split, length=10,
                               elapsed_seconds=elapsed, **evaluate(net, values, TARGET, baseline)))
        legacy.csvwrite(folder / "curves.csv", curves)
    for step_id in range(start+1, CFG["adaptation_updates"]+1):
        loss, grad, indices, slots = legacy.step(net, opt, train, [TARGET], rng, legacy.adapt_lr(lr, step_id))
        observed_stream.update(np.asarray(indices, dtype="<i4").tobytes())
        observed_stream.update(np.asarray(slots, dtype="<i1").tobytes())
        np.add.at(exposures, indices, 1)
        if step_id % CFG["adapt_eval_every"] == 0:
            sync()
            rows = []
            for split, values in [("train", train), ("dev", dev)]:
                rows.append(dict(**ident, task=TARGET, step=step_id, split=split, length=10,
                    elapsed_seconds=elapsed+time.perf_counter()-begin, scheduled_lr=legacy.adapt_lr(lr, step_id),
                    last_batch_loss=loss, last_grad_norm=grad, **evaluate(net, values, TARGET, baseline)))
            curves.extend(rows)
            development = rows[1]
            key = (-development["accuracy"], development["answer_ce"], step_id)
            if best is None or key < tuple(best["selection_key"]):
                best = dict(step=step_id, selection_key=list(key), metrics=development)
                checkpoint(folder / "best.pt", net, ident, step=step_id, development=development,
                           protocol_sha256=protocol_hash, load_audit=audit)
            legacy.csvwrite(folder / "curves.csv", curves)
            save_last(last, net, opt, rng, ident, step=step_id, curves=curves, exposures=exposures,
                      elapsed_seconds=elapsed+time.perf_counter()-begin, best=best, load_audit=audit,
                      resumes=resumes, protocol_sha256=protocol_hash)
    sync()
    checkpoint(folder / "endpoint.pt", net, ident, step=CFG["adaptation_updates"],
               protocol_sha256=protocol_hash, development=curves[-1], load_audit=audit)
    np.savez_compressed(folder / "input_exposure_counts.npz", counts=exposures)
    sequence, _, _ = legacy.batch((train[0][:64], train[1][:64]), legacy.T[TARGET])
    count = int(exposures.sum())
    assert count == CFG["adaptation_updates"]*64
    assert observed_stream.hexdigest() == stream_hash(repeat["adapt_sampling_seed"], CFG["adaptation_updates"], size, 1)
    write(folder / "completed.json", dict(**ident, status="complete", finished_at_utc=utc(),
        updates=CFG["adaptation_updates"], elapsed_seconds=elapsed+time.perf_counter()-begin,
        protocol_sha256=protocol_hash, best=best, endpoint=curves[-1],
        best_sha256=sha(folder / "best.pt"), endpoint_sha256=sha(folder / "endpoint.pt"),
        train_input_sha256=input_hash(repeat_id, "target_train", size), dev_input_sha256=input_hash(repeat_id, "target_dev"),
        stream_sha256=observed_stream.hexdigest(),
        sample_exposures=count, prompt_tokens=count*(sequence.shape[1]-2),
        training_sequence_tokens=count*sequence.shape[1], supervised_tokens=count*2,
        independent_inputs=int((exposures > 0).sum()), target_development_labels=len(dev[0]),
        load_audit=audit, resumes=resumes))
    if last.exists():
        last.unlink()
    print(f"{ident['run_id']} adaptation complete; best step {best['step']}", flush=True)
    del net, opt


def select():
    protocol_hash = verify_protocol()
    rows = []
    for repeat in REPEATS:
        for condition in ROUTES:
            for size in SIZES:
                candidates = [valid_completed(adapt_dir(repeat["repeat_id"], condition, size, i))
                              for i in range(len(CFG["learning_rates"]))]
                assert all(candidates), "All 360 candidates must complete before selection freezes."
                chosen = min(candidates, key=lambda r: (*r["best"]["selection_key"], r["lr_index"]))
                folder = adapt_dir(repeat["repeat_id"], condition, size, chosen["lr_index"])
                rows.append(dict(repeat_id=repeat["repeat_id"], condition=condition, target_samples=size,
                    run_id=chosen["run_id"], lr=chosen["lr"], lr_index=chosen["lr_index"],
                    step=chosen["best"]["step"], dev_accuracy=chosen["best"]["metrics"]["accuracy"],
                    dev_answer_ce=chosen["best"]["metrics"]["answer_ce"],
                    selected_path=relative(folder / "best.pt"), selected_sha256=chosen["best_sha256"],
                    endpoint_path=relative(folder / "endpoint.pt"), endpoint_sha256=chosen["endpoint_sha256"]))
    assert len(rows) == 120
    path = OUT / "SELECTIONS_FROZEN.json"
    if path.exists():
        assert read(path)["selections"] == rows
    else:
        assert not (OUT / "data" / "test_manifest.json").exists(), "Test must not exist before development selection."
        write(path, dict(status="frozen", training_complete=True, frozen_at_utc=utc(),
                         protocol_sha256=protocol_hash, selections=rows, candidate_runs=360))
    legacy.csvwrite(OUT / "selected_development.csv", rows)
    return path


def test():
    protocol_hash = verify_protocol()
    selection_path = OUT / "SELECTIONS_FROZEN.json"
    selected = read(selection_path)
    assert selected["status"] == "frozen" and selected["training_complete"]
    assert selected["protocol_sha256"] == protocol_hash and len(selected["selections"]) == 120
    from experiments.fixed_points_deficiencies_arch_v1.data_pipeline import verify as verify_data
    verify_data(OUT / "data", include_test=True, replay=False)
    rows = []
    for row in selected["selections"]:
        repeat_id, condition, size = row["repeat_id"], row["condition"], row["target_samples"]
        train = pool(repeat_id, "target_train", size)
        testing = pool(repeat_id, "test")
        permutations = load_raw(repeat_id, "test")
        for kind in ["selected", "endpoint"]:
            path = OUT / row[f"{kind}_path"]
            assert sha(path) == row[f"{kind}_sha256"]
            prefix = f"{repeat_id}_{condition}_n{size}_{kind}"
            result_path = OUT / "test_metrics" / f"{prefix}.json"
            prediction_path = OUT / "predictions" / f"{prefix}.jsonl"
            if result_path.exists():
                result = read(result_path)
                assert result["checkpoint_sha256"] == sha(path)
                assert result["predictions_sha256"] == sha(prediction_path)
                rows.append(result)
                continue
            begin = time.perf_counter()
            net, _ = initial_net(repeat_id)
            cp = legacy.load(path)
            net.load_state_dict(cp["model"], strict=True)
            assert legacy.statehash(net.state_dict()) == cp["model_sha256"]
            metrics, predictions = evaluate(net, testing, TARGET, mode(train, TARGET), raw=True)
            ident = dict(repeat_id=repeat_id, condition=condition, target_samples=size,
                         checkpoint_kind=kind, run_id=row["run_id"], lr=row["lr"], step=cp["step"])
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
                for prediction, permutation in zip(predictions, permutations):
                    handle.write(json.dumps(dict(**ident, **prediction, checkpoint_sha256=row[f"{kind}_sha256"],
                                                 permutation=list(map(int, permutation))), sort_keys=True)+"\n")
            result = dict(**ident, **metrics, checkpoint_sha256=sha(path),
                          predictions_sha256=sha(prediction_path), test_input_sha256=input_hash(repeat_id, "test"),
                          selection_manifest_sha256=sha(selection_path), elapsed_seconds=time.perf_counter()-begin)
            write(result_path, result)
            rows.append(result)
            legacy.csvwrite(OUT / "results_test.csv", rows)
            del net
    assert len(rows) == 240
    legacy.csvwrite(OUT / "results_test.csv", rows)
    write(OUT / "TEST_COMPLETE.json", dict(status="complete", evaluations=240, selected_results=120,
          selection_manifest_sha256=sha(selection_path), results_sha256=sha(OUT / "results_test.csv"), finished_at_utc=utc()))


def aggregate():
    sources, adaptations, exposures, accounting = [], [], [], []
    for stage, collection in [("pretrain", sources), ("adapt", adaptations)]:
        for folder in sorted((OUT / stage).glob("*")):
            if (folder / "curves.csv").exists():
                collection.extend(legacy.csvread(folder / "curves.csv"))
            if (folder / "exposures.csv").exists():
                exposures.extend(legacy.csvread(folder / "exposures.csv"))
            if (folder / "completed.json").exists():
                metadata = read(folder / "completed.json")
                keys = ["run_id", "repeat_id", "condition", "target_samples", "lr", "model_seed", "data_seed",
                        "sampling_seed", "dropout_seed", "updates", "elapsed_seconds", "sample_exposures", "prompt_tokens",
                        "training_sequence_tokens", "supervised_tokens", "independent_inputs", "stream_sha256"]
                accounting.append(dict(stage=stage, **{key: metadata[key] for key in keys if key in metadata}))
    legacy.csvwrite(OUT / "source_curves.csv", sources)
    legacy.csvwrite(OUT / "adaptation_curves.csv", adaptations)
    legacy.csvwrite(OUT / "exposures.csv", exposures)
    legacy.csvwrite(OUT / "accounting.csv", accounting)
    return sources, adaptations, exposures, accounting


def verify():
    verify_protocol()
    _, _, exposures, accounting = aggregate()
    complete = [row for row in accounting if row["stage"] == "pretrain"]
    for repeat in REPEATS:
        rows = [row for row in complete if row["repeat_id"] == repeat["repeat_id"]]
        if rows:
            assert len({row["stream_sha256"] for row in rows}) == 1
        for size in SIZES:
            rows = [row for row in accounting if row["stage"] == "adapt" and row["repeat_id"] == repeat["repeat_id"] and row["target_samples"] == size]
            if rows:
                assert len({row["stream_sha256"] for row in rows}) == 1
            audits = [read(adapt_dir(repeat["repeat_id"], c, size, i) / "load_audit.json")
                      for c in ROUTES for i in range(3) if (adapt_dir(repeat["repeat_id"], c, size, i) / "load_audit.json").exists()]
            if audits:
                assert len({r["target_row_sha256"] for r in audits}) == 1
                assert all(r["fresh_optimizer_state_entries"] == 0 and r["only_target_tied_row_reset"] for r in audits)
    report = dict(pretraining_complete=len(complete), adaptation_complete=len(accounting)-len(complete),
                  source_exposure_rows=len(exposures), pairing_checks="passed", verified_at_utc=utc())
    write(OUT / "TRAINING_AUDIT.json", report)
    print(json.dumps(report), flush=True)


def smoke():
    """Small fixture, exact resume and timing; never a formal repeated unit."""
    protocol_hash = verify_protocol()
    folder = OUT / "smoke"
    if folder.exists() or (OUT / "SMOKE.json").exists():
        raise FileExistsError("Preserve the existing engineering attempt; no silent rerun or overwrite")
    fixture = load_raw(REPEATS[0]["repeat_id"], "source_train")[:256]
    tiny = encode(fixture)
    net = make(REPEATS[0]["model_seed"])
    init = state(net)
    tasks = SOURCES["D"]
    opt = legacy.optmake(net, 3e-4)
    legacy.v1.seed(FORMAL_ENGINEERING["dropout_seed"])
    rng = np.random.default_rng(FORMAL_ENGINEERING["sampling_seed"])
    indices_seen, slots_seen = [], []
    for step_id in range(1, 5):
        _, _, indices, slots = legacy.step(net, opt, tiny, tasks, rng, legacy.source_lr(step_id))
        indices_seen.append(indices.tolist()); slots_seen.append(slots.tolist())
        assert np.bincount(slots).tolist() == [16]*4
    save_last(folder / "resume.pt", net, opt, rng, dict(smoke=True), step=4)
    for step_id in range(5, 9):
        legacy.step(net, opt, tiny, tasks, rng, legacy.source_lr(step_id))
    expected = legacy.statehash(net.state_dict())
    resumed = make(REPEATS[0]["model_seed"])
    resumed_opt = legacy.optmake(resumed, 3e-4)
    _, resumed_rng = restore_last(folder / "resume.pt", resumed, resumed_opt)
    for step_id in range(5, 9):
        legacy.step(resumed, resumed_opt, tiny, tasks, resumed_rng, legacy.source_lr(step_id))
    assert expected == legacy.statehash(resumed.state_dict()), "Resume must reproduce bitwise weights."
    saved_rng = legacy.rngstates(resumed_rng)
    metrics, predictions = evaluate(resumed, tiny, TARGET, mode(tiny, TARGET), raw=True)
    assert metrics["correct"] == sum(r["correct"] for r in predictions)
    assert all(r["correct"] == (r["predicted_ids"] == r["gold_answer_ids"]) for r in predictions)
    assert torch.equal(saved_rng["torch_rng"], torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(saved_rng["cuda_rng"], torch.cuda.get_rng_state_all()))
    ids, gold, eq = legacy.batch((tiny[0][:64], tiny[1][:64]), legacy.T[TARGET])
    resumed.eval()
    with torch.inference_mode():
        logits = resumed(ids)[:, eq]
        changed = ids.clone(); changed[:, eq+1] = (gold+1)%10
        assert torch.equal(logits, resumed(changed)[:, eq]), "Causal answer leakage."
    before = state(resumed)
    with torch.no_grad():
        resumed.token_embedding.weight[TARGET_ROW].copy_(init["token_embedding.weight"][TARGET_ROW].to(DEVICE))
    for name, value in resumed.state_dict().items():
        expected_value = before[name]
        if name in ["token_embedding.weight", "lm_head.weight"]:
            expected_value[TARGET_ROW].copy_(init["token_embedding.weight"][TARGET_ROW])
        assert torch.equal(expected_value, value.cpu())
    fresh = legacy.optmake(resumed, 3e-4)
    assert len(fresh.state) == 0
    sync(); begin = time.perf_counter()
    for step_id in range(1, 101):
        legacy.step(resumed, fresh, tiny, tasks, resumed_rng, legacy.source_lr(step_id))
    sync(); seconds = time.perf_counter()-begin
    evaluation_begin = time.perf_counter()
    evaluate(resumed, tiny, TARGET, mode(tiny, TARGET))
    sync(); evaluation_seconds = time.perf_counter()-evaluation_begin
    # Exact declared selection priorities, including LR last and no step zero.
    candidates = [(0.7, 0.6, 200, 1), (0.7, 0.5, 300, 2), (0.7, 0.5, 200, 1), (0.7, 0.5, 200, 0)]
    assert min(candidates, key=lambda r: (-r[0], r[1], r[2], r[3])) == candidates[-1]
    sequence_tokens = int(ids.shape[1])
    payload = dict(status="passed", at_utc=utc(), environment=environment(),
        protocol_sha256=protocol_hash, architecture=CFG["architecture"], model_config=CFG["model"],
        engineering_optimizer_updates=FORMAL_ENGINEERING["optimizer_updates"],
        engineering_sequence_tokens=FORMAL_ENGINEERING["optimizer_updates"]*64*28,
        engineering_supervised_tokens=FORMAL_ENGINEERING["optimizer_updates"]*64*2,
        resume_bitwise_equal=True, strict_save_load=True, target_only_reset=True, fresh_optimizer=True,
        causal_no_answer_leak=True, greedy_exact_matches_prediction_records=True,
        evaluation_preserves_rng=True, balanced_slots=[16]*4, selection_priorities_checked=True,
        timing_updates=100, timing_seconds=seconds, seconds_per_update=seconds/100,
        evaluation_256_examples_seconds=evaluation_seconds, sequence_tokens=sequence_tokens,
        prompt_tokens=sequence_tokens-2, supervised_tokens=2,
        formal_updates=32*4800+360*800, update_only_estimate_hours=(32*4800+360*800)*seconds/100/3600,
        fixture_sha256=hashlib.sha256(fixture.tobytes()).hexdigest(),
        fixture_origin="First 256 isolated R1 source_train inputs; discarded engineering model",
        initial_smoke_note="This separately recorded engineering check is not a formal paired unit")
    write(OUT / "SMOKE.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare-initials", "smoke", "pretrain", "adapt", "select", "test", "verify", "aggregate"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--repeat", choices=[r["repeat_id"] for r in REPEATS])
    parser.add_argument("--condition", choices=ROUTES)
    parser.add_argument("--samples", type=int, choices=SIZES)
    parser.add_argument("--lr-index", type=int, choices=range(3))
    args = parser.parse_args()
    OUT = args.out.resolve()
    assert OUT == (HERE / "artifacts").resolve(), "Use the single protocol-bound artifact directory."
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare-initials":
        prepare_initials()
    elif args.command == "smoke":
        smoke()
    elif args.command == "pretrain":
        for repeat in REPEATS:
            if args.repeat and repeat["repeat_id"] != args.repeat:
                continue
            for condition in CONDITIONS:
                if args.condition and condition != args.condition:
                    continue
                pretrain_one(repeat["repeat_id"], condition)
        aggregate()
    elif args.command == "adapt":
        for repeat in REPEATS:
            if args.repeat and repeat["repeat_id"] != args.repeat:
                continue
            for condition in ROUTES:
                if args.condition and condition != args.condition:
                    continue
                for size in SIZES:
                    if args.samples and size != args.samples:
                        continue
                    for i in range(3):
                        if args.lr_index is not None and i != args.lr_index:
                            continue
                        adapt_one(repeat["repeat_id"], condition, size, i)
        aggregate()
    else:
        globals()[args.command]()


if __name__ == "__main__":
    main()
