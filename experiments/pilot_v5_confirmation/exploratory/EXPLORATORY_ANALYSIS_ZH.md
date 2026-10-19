# Per-seed paired recomputation of existing v4 results

This file reads only existing `pilot_v4_backgrounds/artifacts/results_selected.csv`, without new training or reading v5 confirmation results. Old seeds 17/42/73 provide exploratory evidence only. All results are accuracies of development-selected checkpoints on the same new v4 test pools; A is shared across backgrounds.

For each seed, compute `Δ(B−A)=(B_new−A)−(B_original−A)` and `Δ(C−A)=(C_new−A)−(C_original−A)`, then the primary interaction `Δ(B−C)=(B_new−C_new)−(B_original−C_original)`. Shared A cancels, so the first two terms equal the background changes in B and C themselves; the interaction is their difference. Values below are percentage points; CSV files retain raw accuracy proportions.

| Target | Background change | seed | Δ(B−A) pp | Δ(C−A) pp | Δ(B−C) pp |
| --- | --- | --- | --- | --- | --- |
| fixed_points | g1_recoils−g0 | 17 | -16.2109 | -10.4492 | -5.7617 |
| fixed_points | g1_recoils−g0 | 42 | -16.6016 | -20.1172 | +3.5156 |
| fixed_points | g1_recoils−g0 | 73 | -4.1504 | -13.6230 | +9.4727 |
| fixed_points | g2_deficiencies−g0 | 17 | +16.5039 | +15.0879 | +1.4160 |
| fixed_points | g2_deficiencies−g0 | 42 | +12.2559 | +11.9141 | +0.3418 |
| fixed_points | g2_deficiencies−g0 | 73 | +20.9961 | +9.6680 | +11.3281 |
| longest_decreasing_run | g1_recoils−g0 | 17 | -18.0176 | +2.0508 | -20.0684 |
| longest_decreasing_run | g1_recoils−g0 | 42 | +1.0742 | -0.9277 | +2.0020 |
| longest_decreasing_run | g1_recoils−g0 | 73 | -17.7246 | +0.1953 | -17.9199 |
| longest_decreasing_run | g2_deficiencies−g0 | 17 | -7.7148 | -0.3418 | -7.3730 |
| longest_decreasing_run | g2_deficiencies−g0 | 42 | +3.5645 | +2.1484 | +1.4160 |
| longest_decreasing_run | g2_deficiencies−g0 | 73 | -11.7676 | -0.6836 | -11.0840 |

| Target | Background change | Mean Δ(B−A) pp | Mean Δ(C−A) pp | Mean Δ(B−C) pp | Per-seed interaction range pp |
| --- | --- | --- | --- | --- | --- |
| fixed_points | g1_recoils−g0 | -12.3210 | -14.7298 | +2.4089 | [-5.7617, +9.4727] |
| fixed_points | g2_deficiencies−g0 | +16.5853 | +12.2233 | +4.3620 | [+0.3418, +11.3281] |
| longest_decreasing_run | g1_recoils−g0 | -11.5560 | +0.4395 | -11.9954 | [-20.0684, +2.0020] |
| longest_decreasing_run | g2_deficiencies−g0 | -5.3060 | +0.3743 | -5.6803 | [-11.0840, +1.4160] |

For fixed_points, the deficiencies background versus the original increases mean B−A by 16.59pp and C−A by 12.22pp, leaving a primary interaction of only +4.36pp. All three seeds are positive, ranging from about +0.34 to +11.33pp. Improvement in B alone cannot all be attributed to the advantage of the related source over its replacement.

For longest_decreasing_run, the recoils background versus the original has a mean primary interaction of −12.00pp, negative in two seeds and positive in one. This mainly reflects change in B, while C changes less on average. The other two exploratory compositions are retained without adding confirmation contrasts from their results.

These are direct within-seed differences; separate SDs for the two groups are not used to judge background-effect reliability. Three exploratory blocks and contrasts chosen from the same results cannot provide independent confirmation; no significance claim is made for those blocks. The next stage fixes the positive FP g2−g0 and negative LDR g1−g0 directions, using five new paired data/initialization units; see `../stats_plan.json`. Exploration and confirmation are reported separately, not pooled as eight repeats.

![Background differences by seed](exploratory_paired_background_changes.png)

Recomputed raw data: [exploratory_paired_differences.csv](exploratory_paired_differences.csv). All 12 interactions were independently checked against the old v4 interaction CSV and agree within 1e−12. That old table was not used in recomputation; see the [audit](exploratory_statistics_audit.json).
