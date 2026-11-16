"""Instantiate the unchanged confirmation design after the declared model gate."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.fixed_points_deficiencies_arch_v1.plan import (
    BASE_CFG, CONDITIONS, GRU_MODEL, HERE, REPEATS, ROUTES, SIZES, SOURCES,
    TARGET, TRANSFORMER_MODEL,
)

DECISION_PATH = HERE / "artifacts/GATE_RESULT.json"
if not DECISION_PATH.is_file():
    raise RuntimeError("Run the committed development gate before constructing the formal architecture design")
DECISION = json.loads(DECISION_PATH.read_text(encoding="utf-8-sig"))
assert DECISION["status"] == "complete"
expected_architecture = "gru" if DECISION["passed"] else "transformer"
expected_model = GRU_MODEL if DECISION["passed"] else TRANSFORMER_MODEL
assert DECISION["selected_architecture"] == expected_architecture
assert DECISION["selected_model_config"] == expected_model
CFG = copy.deepcopy(BASE_CFG)
CFG.update(architecture=expected_architecture, model=copy.deepcopy(expected_model),
           model_gate_passed=bool(DECISION["passed"]),
           model_gate_result="artifacts/GATE_RESULT.json")
