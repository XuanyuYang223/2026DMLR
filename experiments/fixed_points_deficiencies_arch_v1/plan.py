"""Prespecified model-change replication and its separate development gate."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fixed_points_deficiencies_v1.design import CFG as ORIGINAL_CONFIG

ORIGINAL = ROOT / "experiments/fixed_points_deficiencies_v1"
BASE_CFG = copy.deepcopy(ORIGINAL_CONFIG)
TARGET = "fixed_points"
CONDITIONS = ["P", "D", "T", "U"]
ROUTES = [*CONDITIONS, "A"]
SOURCES = copy.deepcopy(BASE_CFG["sources"])
SIZES = [256, 512, 1024]
COUNTS = dict(source_train=8192, source_dev=1024, target_train=1024,
              target_dev=1024, test=2048)
MODEL_SEEDS = [31001, 32003, 33013, 34019, 35023, 36007, 37003, 38011]
REPEATS = [dict(repeat_id=f"R{i+1}", repeat=i+1, model_seed=seed,
    data_seed=2026111701+i, test_seed=2026111751+i, subset_seed=2026111801+i,
    source_sampling_seed=2026111901+i, source_dropout_seed=2026112001+i,
    adapt_sampling_seed=2026112101+i, adapt_dropout_seed=2026112201+i)
    for i, seed in enumerate(MODEL_SEEDS)]

GRU_MODEL = {key: copy.deepcopy(BASE_CFG["model"][key])
             for key in ("d_model", "layers", "dropout", "tie_embeddings", "max_seq_len")}
TRANSFORMER_MODEL = copy.deepcopy(BASE_CFG["model"])
TRANSFORMER_MODEL["layers"] = 8
assert GRU_MODEL["layers"] == 4 and GRU_MODEL["d_model"] == 256

GATE = dict(
    version="architecture_gate_v1", n=10, tasks=["fixed_points", "deficiencies"],
    model_seed=381001, data_seed=2026112601, sampling_seed=2026112602,
    dropout_seed=2026112603, train_count=8192, dev_count=1024,
    updates=4800, batch=64, source_lr=BASE_CFG["source_lr"],
    eval_steps=copy.deepcopy(BASE_CFG["source_eval_steps"]),
    architecture="gru", model=GRU_MODEL, vocab_size=188,
    threshold=0.90, criterion="Both fixed-update-4800 complete-answer-plus-EOS development accuracies must be at least 0.90",
    sampling="Shared gate inputs, initialization, sampling and dropout seeds across the two independent single-task runs",
    schedule="Original source schedule: warmup100; cosine to3e-5 by2400; constant through4800",
    no_test_pool=True, no_hyperparameter_search=True,
    fallback="Eight-layer width256 Transformer if either gate endpoint fails the accuracy threshold; no tuning",
    independence="Gate inputs are excluded from all formal units; gate weights are discarded; the gate is not a confirmatory unit")

FORMAL_ENGINEERING = dict(dropout_seed=2026112701, sampling_seed=2026112702,
                         optimizer_updates=112, formal_repeat=False)

BASE_CFG.update(version="fixed_points_deficiencies_architecture_replication_v1",
    repeats=REPEATS, sources=SOURCES, target_sizes=SIZES, counts=COUNTS,
    output_policy="New experiment directory only; commit/push new artifacts on the case-integration branch; checkpoints remain local",
    unit_policy="Eight new paired units; report separately and never pool with the original eight-unit confirmation",
    architecture_selection="Separate GRU development gate with a prespecified eight-layer Transformer fallback")

assert BASE_CFG["n"] == 10 and BASE_CFG["vocab_size"] == 188
assert BASE_CFG["pretrain_updates"] == GATE["updates"] == 4800
assert BASE_CFG["adaptation_updates"] == 800
assert BASE_CFG["learning_rates"] == [3e-4, 1e-4, 3e-5]


def all_new_seeds():
    keys = ("model_seed", "data_seed", "test_seed", "subset_seed", "source_sampling_seed",
            "source_dropout_seed", "adapt_sampling_seed", "adapt_dropout_seed")
    return ([r[k] for r in REPEATS for k in keys] + [GATE[k] for k in
        ("model_seed", "data_seed", "sampling_seed", "dropout_seed")] +
        [FORMAL_ENGINEERING[k] for k in ("dropout_seed", "sampling_seed")])


assert len(all_new_seeds()) == len(set(all_new_seeds()))
