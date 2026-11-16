# Baseline and reproduction interpretation

This clarification records a code-review observation during the first formal runs. It does not change the committed protocol, data, model, optimizer, predictions, baseline labels or analysis implementation.

The inherited majority baseline predicts the complete answer that is most frequent in the corresponding training pool, separately by length. It is an uncapped constant-answer baseline: unlike the model's greedy decoder, it does not have the eight-token generation limit. Some structured majority answers therefore exceed the model's output budget. The reported baseline and gains retain the original definition. For generation-limited instances, a negative gain may reflect this protocol asymmetry rather than optimization failure or inability to compute the mathematical function. Report the structural exact-accuracy ceiling alongside the gain.

The calibration driver verifies both the file and state hashes of the original local initialization checkpoints `experiments/platform_v01/calibration/initial/seed617.pt` and `seed829.pt`. These weights are ignored by Git. A clone without those assets cannot run the hash-bound driver until the matching initializations are restored. This is explicit local provenance, not a claim of checkpoint-free training reproduction. The later paper's checkpoint hosting and access policy remains an author decision.

The scalar v0.1 representatives fit the generation budget, so this note does not revise their recorded baseline or accuracy. It qualifies cross-task comparisons involving the newly included structured outputs. All test pools remain outside this calibration.
