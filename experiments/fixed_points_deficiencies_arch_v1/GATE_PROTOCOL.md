# Prespecified development gate for the architecture replication

This gate belongs to the new architecture replication. It does not alter the
original confirmation, the platform calibration, or any historical report.
`plan.py:GATE` is the machine-readable specification. The gate is development
work and is not one of the eight formal paired units.

## Fixed architecture and data

The candidate is the reference four-layer, width-256 causal GRU, with learned
positions, dropout 0.1, layer normalization and tied token embedding/readout.
Four single-layer GRUs have native dropout zero; three explicit Torch dropout
modules preserve the reference resumable RNG design. The constructor is adapted
only to accept the checked legacy vocabulary size of 188. It does not import the
reference experiment module or its external paths. Vocabulary IDs, one-line
encoding and context 128 remain those of the original confirmation.

The two independent single-task runs are fixed_points and deficiencies. They
share one newly generated, historically orbit-isolated n=10 training pool of
8192 permutations and development pool of 1024 permutations. Train and
development pools are mutually orbit-isolated. The gate has no test pool. Its
inputs are subsequently excluded from every formal unit. The data pipeline
preserves generation order, hashes, exclusion provenance and capacity checks.

Both tasks start from the same new initialization (seed 381001). Data,
sampling and dropout seeds are respectively 2026112601, 2026112602 and
2026112603. Each task gets a fresh optimizer and the same sample-index and
dropout streams; no weights transfer between the two tasks. The existing
single-task sampling primitive's all-zero task-slot shuffle is retained.

## Fixed optimization and evaluation

Each task receives 4800 updates of batch 64. AdamW betas (0.9, 0.999), epsilon
1e-8, weight decay 0.01, gradient clipping 1.0 and the original source learning
rate schedule are unchanged: 100-update warmup to 3e-4, cosine decay to 3e-5 at
update 2400, then constant 3e-5 through 4800. FP32 weights/evaluation and CUDA
BF16 training are retained; TF32 is disabled. cuDNN deterministic execution is
enabled and benchmark selection is disabled for the reference GRU restart
check. Unsupported numerical/backend behavior is an engineering failure to
record, not permission to change the scientific configuration.

The loss is the original equally weighted answer-token and teacher-forced EOS
cross-entropy. Train and development evaluation occurs at updates 0, 600,
1200, 2400, 3600 and 4800. Main accuracy requires the unconstrained greedy
answer and EOS both to match. Answer-token-only accuracy and teacher-forced EOS
accuracy are separate bridge diagnostics. The two-token generation horizon
matches these scalar answer types. Majority labels come only from training.

The architecture passes exactly when **both** update-4800 development exact
accuracies are at least 0.90. Earlier checkpoints, training accuracy and
majority baselines do not select the architecture. If either endpoint fails,
the formal experiment uses the already specified eight-layer, width-256
Transformer with eight heads and FF multiplier four; no additional GRU search
or second gate is permitted. The fallback retains context 128, vocabulary 188,
the inherited optimizer, data protocol and all other formal controls. Gate
weights are never reused for formal training.

## Commitment, engineering checks and artifacts

`GATE_FROZEN.json` binds the exact gate configuration, source and protocol
hashes, gate manifest and both input files. It and all bound files must match
Git HEAD before `smoke` or `run` can optimize any model. No gate optimizer is
executed while implementing this runner. The frozen protocol is committed
before the first engineering update.

The subsequent smoke uses the first 256 gate-training inputs and discarded
models. It checks strict save/load, bitwise interrupted-versus-uninterrupted
continuation, causal answer isolation, evaluation RNG preservation, tied
target-row reset and a fresh optimizer; 32 additional engineering updates
provide timing. It is not a gate score or a formal repeat. Formal gate execution
requires its passed, protocol-bound record.

Each task retains the fixed endpoint, curves, every update loss, input exposure
counts, initialization binding, actual prompt/sequence/supervision tokens,
elapsed time, environment and any resume history. Periodic checkpoints preserve
model, optimizer, Python/NumPy/Torch RNGs and accumulated records. Exclusive
task locks prevent concurrent duplicate ownership. A resumed run uses the same
seed and resumes its latest saved state; no failed attempt is a new unit.

`artifacts/GATE_RESULT.json` records both endpoints and the deterministic
architecture decision. It is a binding input to the later formal freeze.
The gate does not load formal testing data or run a test forward.

## Reproduction commands

After gate data preparation and the gate freeze have been committed:

```powershell
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/gate_runner.py smoke
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/gate_runner.py run
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_arch_v1/gate_runner.py verify
```

`run --task fixed_points` and `run --task deficiencies` support separately
owned workers on the same frozen gate. Both must finish before a decision.
These commands are instructions, not a claim that the gate has already run.
