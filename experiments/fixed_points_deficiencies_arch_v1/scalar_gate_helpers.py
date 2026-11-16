"""Legacy scalar protocol primitives with explicit device and no old writers."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from experiments.permworld_pilot.vendor.neurips_permutations.math_ops import PROPERTY_FUNCTIONS
from experiments.permworld_pilot.vendor.neurips_permutations.passage import TOKEN_TO_ID as V, one_line_tokens

HERE = Path(__file__).resolve().parent
EOS = V["<EOS>"]


def confined(path):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE):
        raise ValueError("All new writes must remain in the architecture replication directory")
    return path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, obj):
    path = confined(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def csvwrite(path, rows):
    path = confined(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def statehash(weights):
    h = hashlib.sha256()
    for name, value in sorted(weights.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save(path, payload):
    path = confined(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def rng_state(rng):
    return dict(numpy_rng=copy.deepcopy(rng.bit_generator.state), python_rng=random.getstate(),
        numpy_global_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(payload):
    rng = np.random.default_rng()
    rng.bit_generator.state = payload["numpy_rng"]
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_global_rng"])
    torch.set_rng_state(payload["torch_rng"].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in payload["cuda_rng"]])
    return rng


def encode(permutations, task, device):
    prefixes = [[V[t] for t in ("<BOS>", "<SIZE>", f"{len(q):02d}", *one_line_tokens(tuple(map(int, q))))]
                for q in permutations]
    answers = [int(PROPERTY_FUNCTIONS[task](tuple(map(int, q)))) for q in permutations]
    assert all(V[f"{i:02d}"] == i for i in range(11))
    return (torch.tensor(prefixes, dtype=torch.long, device=device),
            torch.tensor(answers, dtype=torch.long, device=device))


def batch(dataset, task, answers=True):
    x, gold = dataset
    eq = x.shape[1] + 1
    ids = torch.full((len(x), x.shape[1] + (4 if answers else 2)), V["<PAD>"], dtype=torch.long, device=x.device)
    ids[:, :x.shape[1]] = x
    ids[:, x.shape[1]] = V[f"<{task.upper()}>"]
    ids[:, eq] = V["="]
    if answers:
        ids[:, -2] = gold
        ids[:, -1] = EOS
    return ids, gold, eq


def optimizer(net, config):
    return torch.optim.AdamW(net.parameters(), lr=3e-4, betas=tuple(config["betas"]),
        eps=config["epsilon"], weight_decay=config["weight_decay"],
        fused=next(net.parameters()).device.type == "cuda")


def source_lr(step):
    if step > 2400:
        return .00003
    return .0003 * (step / 100 if step <= 100 else .1 + .9 * .5 * (1 + math.cos(math.pi * (step - 100) / 2300)))


def train_step(net, opt, dataset, task, rng, step, config):
    net.train()
    x, y = dataset
    indices = rng.choice(len(x), 64, replace=False)
    # The original one-task path also shuffles a 64-slot all-zero array. Keep
    # that RNG consumption so the archived sampling primitive is reproduced.
    slots = np.zeros(64, dtype=np.int64)
    rng.shuffle(slots)
    ids, gold, eq = batch((x[indices], y[indices]), task)
    for group in opt.param_groups:
        group["lr"] = source_lr(step)
    opt.zero_grad(set_to_none=True)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        logits = net(ids)
        loss = .5 * (F.cross_entropy(logits[:, eq], gold) + F.cross_entropy(logits[:, eq + 1], torch.full_like(gold, EOS)))
    assert torch.isfinite(loss)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), config["clip"], error_if_nonfinite=True)
    opt.step()
    return float(loss.detach()), float(norm), indices, slots


@torch.inference_mode()
def evaluate(net, dataset, task, majority):
    net.eval()
    x, y = dataset
    correct = legacy_correct = eos_correct = majority_correct = 0
    answer_loss = eos_loss = 0.0
    for offset in range(0, len(x), 128):
        ids, gold, eq = batch((x[offset:offset + 128], y[offset:offset + 128]), task, False)
        logits = net(ids)[:, eq].float()
        first = logits.argmax(-1)
        answer_loss += float(F.cross_entropy(logits, gold, reduction="none").sum())
        teacher_logits = net(torch.cat((ids, gold[:, None]), 1))[:, -1].float()
        teacher_second = teacher_logits.argmax(-1)
        second = teacher_second.clone()
        wrong = first != gold
        if bool(wrong.any()):
            second[wrong] = net(torch.cat((ids[wrong], first[wrong, None]), 1))[:, -1].float().argmax(-1)
        correct += int(((first == gold) & (second == EOS)).sum())
        legacy_correct += int((first == gold).sum())
        eos_correct += int((teacher_second == EOS).sum())
        majority_correct += int((gold == majority).sum())
        eos_loss += float(F.cross_entropy(teacher_logits, torch.full_like(gold, EOS), reduction="sum"))
    count = len(x)
    return dict(count=count, correct=correct, accuracy=correct / count,
        legacy_correct=legacy_correct, legacy_accuracy=legacy_correct / count,
        answer_ce=answer_loss / count, eos_ce=eos_loss / count,
        two_token_ce=(answer_loss + eos_loss) / (2 * count),
        teacher_eos_accuracy=eos_correct / count, majority_label=majority,
        majority_correct=majority_correct, majority_accuracy=majority_correct / count)
