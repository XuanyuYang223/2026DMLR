"""Fixed eight-unit deficiencies confirmation; scientific choices precede training."""
from __future__ import annotations
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_COMMIT = "1cd90d54d2737c98660216cd75ebe25d1740860f"
V5_CONFIG = ROOT / "experiments/pilot_v5_confirmation/artifacts/config.json"
V5 = json.loads(V5_CONFIG.read_text(encoding="utf-8-sig"))
TARGET = "fixed_points"
CONDITIONS = ["P", "D", "T", "U"]
ROUTES = [*CONDITIONS, "A"]
COMMON = ["descents", "successions", "left_to_right_maxima"]
CANDIDATES = {"P": "peaks", "D": "deficiencies", "T": "anti_fixed_points", "U": "right_to_left_maxima"}
SOURCES = {c: [*COMMON, CANDIDATES[c]] for c in CONDITIONS}
SIZES = [256, 512, 1024]
MODEL_SEEDS = [17011, 18013, 19001, 20011, 21001, 22003, 23003, 24001]
REPEATS = [dict(repeat_id=f"R{i+1}", repeat=i+1, model_seed=seed,
    data_seed=2026111001+i, test_seed=2026111051+i, subset_seed=2026111101+i,
    source_sampling_seed=2026111201+i, source_dropout_seed=2026111301+i,
    adapt_sampling_seed=2026111401+i, adapt_dropout_seed=2026111501+i)
    for i, seed in enumerate(MODEL_SEEDS)]
INHERITED_KEYS = ["n", "model", "vocab_size", "encoding", "batch", "pretrain_updates",
    "source_lr", "adaptation_updates", "learning_rates", "adapt_eval_every",
    "source_eval_steps", "weight_decay", "betas", "epsilon", "clip", "precision"]
CFG = {key: copy.deepcopy(V5[key]) for key in INHERITED_KEYS}
CFG.update(version="fixed_points_deficiencies_confirmation_v1", target=TARGET,
    conditions=CONDITIONS, routes=ROUTES, sources=SOURCES, target_sizes=SIZES,
    repeats=REPEATS, source_pool=8192, source_dev=1024, target_pool=1024,
    target_dev=1024, test_count=2048,
    counts=dict(source_train=8192, source_dev=1024, target_train=1024, target_dev=1024, test=2048),
    source_checkpoint="Fixed 4800-update endpoint; no target-based selection",
    adaptation="Full-parameter independent runs at every size and learning rate; no continuation across sizes",
    target_reset="Only fixed_points tied embedding/output row restored to this unit's step-0 row",
    selection="Max complete-answer-plus-EOS dev accuracy; min answer CE; earliest positive step; declared LR order",
    primary_accuracy="Unconstrained greedy answer followed by EOS; both tokens must exactly match",
    legacy_bridge="First-answer-token accuracy plus teacher-forced EOS accuracy, reported separately",
    loss="Inherited v5: 0.5*(answer-token mean CE + teacher-forced EOS mean CE)",
    supervision="256/512/1024 target training labels plus 1024 separately disclosed target development labels",
    primary="Within-unit S_D-S_U; S=.25*A256+.5*A512+.25*A1024",
    test_policy="Generate all new tests only after all 120 development selections freeze; no test-driven choice",
    input_policy="No explicit paired-transform queries, formula supervision or local-position labels",
    stopping="Exactly 8 units, 32 source models, 360 LR candidates, 120 selected target results; no extensions",
    output_policy="Local delivery only; no automatic upload or publication")
assert CFG["n"] == 10 and CFG["batch"] == 64 and CFG["vocab_size"] == 188
assert CFG["pretrain_updates"] == 4800 and CFG["adaptation_updates"] == 800
assert CFG["learning_rates"] == [3e-4, 1e-4, 3e-5]
