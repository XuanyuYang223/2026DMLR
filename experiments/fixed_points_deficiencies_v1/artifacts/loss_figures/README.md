# Saved learning-loss trajectories

These figures read stored training/development curves only. They add no training,
model evaluation, checkpoint selection or statistical test. All panels use n=10.
Thick lines are means of the eight paired units; thin lines retain each unit.
Solid lines show development, and dashed lines show training. Y axes explicitly
use symlog scaling, with a linear region below 0.01, to preserve large initial
losses and readable later losses, including exact zero. All saved loss values
are plotted without clipping or alteration. Panels autoscale independently.

Source panels include every condition/task at all six prespecified evaluations.
Target panels show the entire 0-to-800 trajectory of each frozen selected-LR
candidate, including updates after the selected checkpoint. They are
selection-dependent descriptions, not independent evidence for a new checkpoint
rule. The raw adaptation table retains all three LR candidates for every route,
sample size and unit: 360 runs, not only the 120 displayed candidates.

`answer_ce` is teacher-forced answer-token cross-entropy. `two_token_ce` is the
mean of answer-token and teacher-forced EOS cross-entropies. These losses are
distinct from the complete generated answer/EOS exact-match accuracy.

| Stage | Answer loss | Mean answer + EOS loss |
| --- | --- | --- |
| Source | [PNG](source_answer_ce.png), [SVG](source_answer_ce.svg), [PDF](source_answer_ce.pdf) | [PNG](source_two_token_ce.png), [SVG](source_two_token_ce.svg), [PDF](source_two_token_ce.pdf) |
| Adaptation | [PNG](adaptation_answer_ce.png), [SVG](adaptation_answer_ce.svg), [PDF](adaptation_answer_ce.pdf) | [PNG](adaptation_two_token_ce.png), [SVG](adaptation_two_token_ce.svg), [PDF](adaptation_two_token_ce.pdf) |

The separate manifest records source-table/selection hashes and this plotting
script hash. Frozen `analysis.py` outputs remain unchanged.
