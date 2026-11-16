"""Isolated architecture factories for the prespecified replication.

The GRU follows the audited reference class rather than importing the reference
experiment module, which contains unrelated paths and import-time work.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from experiments.permworld_pilot.vendor.neurips_permutations.models import CausalTransformer
from experiments.permworld_pilot.vendor.neurips_permutations.passage import TOKEN_TO_ID
from experiments.permworld_pilot.vendor.sdpa import accelerate

VOCAB_SIZE = 188
assert len(TOKEN_TO_ID) == VOCAB_SIZE
REFERENCE_GRU = "reference/2026TMLR/experiments/permworld_encoding_archdev_20261009/common.py:CausalGRU"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_setup(device=None):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.device(device if device is not None else "cuda" if torch.cuda.is_available() else "cpu")


class CausalGRU(nn.Module):
    """Reference GRU with vocabulary size supplied explicitly.

    The only constructor adaptation replaces len(reference TOKEN_TO_ID) by the
    explicit, checked vocabulary size. Module construction, initialization order,
    recurrent cells, positional embeddings, dropout and readout are retained.
    Padding must follow a contiguous valid prefix.
    """

    def __init__(self, config, vocab_size=VOCAB_SIZE):
        super().__init__()
        if vocab_size != VOCAB_SIZE:
            raise ValueError("This frozen replication uses the 188-token legacy vocabulary")
        self.config = SimpleNamespace(**config)
        d = config["d_model"]
        self.token_embedding = nn.Embedding(vocab_size, d)
        self.position_embedding = nn.Embedding(config["max_seq_len"], d)
        self.embedding_dropout = nn.Dropout(config["dropout"])
        # Explicit dropout keeps all stochastic state in checkpointed Torch RNGs.
        self.recurrent_layers = nn.ModuleList([
            nn.GRU(d, d, 1, batch_first=True, dropout=0.) for _ in range(config["layers"])])
        self.interlayer_dropout = nn.ModuleList([
            nn.Dropout(config["dropout"]) for _ in range(config["layers"] - 1)])
        self.final_norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        nn.init.normal_(self.token_embedding.weight, std=.02)
        nn.init.normal_(self.position_embedding.weight, std=.02)
        if config["tie_embeddings"]:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, ids, mask=None):
        if mask is None:
            mask = torch.ones_like(ids, dtype=torch.bool)
        positions = torch.arange(ids.shape[1], device=ids.device)
        h = self.token_embedding(ids) + self.position_embedding(positions)[None]
        h = self.embedding_dropout(h) * mask[..., None]
        for i, recurrent in enumerate(self.recurrent_layers):
            h, _ = recurrent(h)
            if i < len(self.interlayer_dropout):
                h = self.interlayer_dropout[i](h)
        h = self.final_norm(h) * mask[..., None]
        return self.lm_head(h)


def factory_metadata(architecture, model_config):
    if architecture not in ("gru", "transformer"):
        raise ValueError(architecture)
    ordered_tokens = sorted(TOKEN_TO_ID, key=TOKEN_TO_ID.get)
    return dict(architecture=architecture, model=copy.deepcopy(model_config), vocab_size=VOCAB_SIZE,
        vocabulary_sha256=hashlib.sha256(json.dumps(ordered_tokens, separators=(",", ":")).encode()).hexdigest(),
        constructor="CausalGRU" if architecture == "gru" else "CausalTransformer with inherited SDPA adapter",
        reference=REFERENCE_GRU if architecture == "gru" else "experiments/permworld_pilot/vendor/neurips_permutations/models.py:CausalTransformer",
        model_factory_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        recurrent_dropout="Four one-layer GRUs with native dropout zero and explicit interlayer Torch dropout" if architecture == "gru" else None,
        backend_policy="TF32 disabled; cuDNN deterministic; cuDNN benchmark disabled",
        comparison="Architecture/depth change, not parameter-count matching")


def make_model(seed, architecture, model_config, device=None):
    device = device_setup(device)
    seed_all(seed)
    if architecture == "gru":
        result = CausalGRU(copy.deepcopy(model_config), vocab_size=VOCAB_SIZE)
    elif architecture == "transformer":
        result = accelerate(CausalTransformer(vocab_size=VOCAB_SIZE, **copy.deepcopy(model_config)))
    else:
        raise ValueError(architecture)
    result = result.to(device)
    if model_config["tie_embeddings"]:
        assert result.token_embedding.weight.data_ptr() == result.lm_head.weight.data_ptr()
    return result
