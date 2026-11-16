# Development-gate analysis plan

This plan is frozen before any gate or engineering optimizer update.

1. Audit the committed gate configuration, data manifest, both pools and source
   hashes before execution. Require the complete-answer smoke to pass.
2. Retain the two independent task endpoints at update 4800. Evaluate full
   training and development pools at the fixed checkpoints. Do not choose a
   checkpoint by accuracy for the gate decision.
3. For each task, report exact answer-plus-EOS development accuracy, answer
   loss, training accuracy, training-derived majority baseline, answer-only
   bridge accuracy, and learning curves. These are descriptive gate records,
   not a new transfer comparison or an additional confirmatory unit.
4. Compute the predeclared Boolean decision: fixed_points endpoint accuracy
   >=0.90 AND deficiencies endpoint accuracy >=0.90. Record both raw fractions
   and counts. A failure is retained and triggers only the predefined
   eight-layer Transformer fallback. No extra seed, architecture candidate,
   training extension, parameter search or alternate threshold is introduced.
5. Verify shared initialization/input/stream hashes and equal exposure counts.
   Report prompt tokens, complete training-sequence tokens and supervised
   tokens separately. Record wall times and any resumes; concurrent durations
   must not be summed and called wall time or dedicated GPU-hours.
6. Write the immutable gate result to `artifacts/GATE_RESULT.json`, including
   endpoint records, selected architecture/model configuration and frozen gate
   and data-manifest hashes. The formal protocol binds this decision before
   any formal training. Gate weights are discarded and gate inputs remain
   excluded from every formal unit.

Passing this gate establishes only fixed-budget single-task learnability on
this development pool. It does not establish transfer, formula use, robustness
across seeds, or equivalence of architecture capacity. Gate failure is not a
proof that the GRU cannot learn the mathematical tasks under other protocols.
The gate contains no test set and yields no test-based scientific claim.
