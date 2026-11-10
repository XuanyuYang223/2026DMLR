# Proposed DMLR result paragraph: canonical completion draft

Motivated by earlier V5 and task-count observations, we conducted a new
prespecified fixed-points source-replacement comparison in eight paired
data/initialization units. Three sources (descents, successions and
left-to-right maxima) were fixed; the fourth was peaks (P), deficiencies (D),
anti-fixed points (T), or right-to-left maxima (U). The legacy one-line n=10
Transformer and 4800-update source budget were fixed. Full-parameter adaptation
used independent fits on nested 256/512/1024 training subsets, 800 updates per
learning-rate candidate, and a separate 1024-example labeled development set.
All 120 development selections froze before independent test generation.
The sole primary outcome was the paired difference in
`S=.25*A256+.5*A512+.25*A1024`, with accuracy defined as complete greedy
answer-plus-EOS exact match. D-U was **+8.76 percentage points**
(sample SD 4.88; 95% paired-t interval
[+4.68, +12.84], df=7), positive in 8/8 units.
The primary interval is entirely positive. This supports additional transfer value of deficiencies over the specified U task in this background, budget and adaptation procedure.

The prespecified secondary gain over random initialization was smaller:
**D-A=+1.77 pp** (sample SD 1.35;
7 positive units and 1 tie).
P, T and U retained negative area gains relative to A in all eight units.
The descriptive identity `D-U=(D-A)-(U-A)` includes U's
-6.99 pp mean negative transfer; it is not a mechanism decomposition.
Pointwise D-U means narrowed from +24.1333 / +5.1514 / +0.6042 pp across
256/512/1024 training examples. These secondary observations do not replace
the weighted-area primary comparison or establish a scaling law.

D and U were not matched in source difficulty, answer distribution or
redundancy; U itself has a reversal relation to a common source. Numeric answer
parameters were retained at adaptation, while the target-specific task row
was restored. Neither the positive comparison, source competence, nor answer
derivability establishes formula use or a transplantable lemma module.
Small-sample t assumptions and finite-space orbit exclusion limit interval
coverage. The finding concerns one background, model, budget and length; the
additional development labels are part of target supervision. Historical V5
failed confirmation and task-count negative outcomes remain separate evidence.

This documentation-only draft adds baseline and metric context to the
[original frozen-analysis draft](artifacts/analysis/PAPER_RESULT_DRAFT.md),
which remains unchanged. See [the canonical result report](RESULT_REPORT.md),
[raw metrics](artifacts/results_test.csv), [paired summaries](artifacts/analysis/paired_summary.csv)
and [documentation amendment](REPORT_AMENDMENT.json). No new evaluation or
scientific selection was performed to write this paragraph.
