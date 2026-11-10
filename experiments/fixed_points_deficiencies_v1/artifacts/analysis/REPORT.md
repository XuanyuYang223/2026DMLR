# Fixed-points source-replacement confirmation

Status: all 32 pretraining models, 360 learning-rate candidates and 120 selected
adaptations are represented by complete test records. Selected and fixed-endpoint
results remain separate. This is a new eight-unit confirmation, motivated by v5
and task-count v1/v2; historical failed confirmation and negative transfer are
not rewritten or pooled with this study.

## Primary result

The prespecified score is S = 0.25 A256 + 0.5 A512 + 0.25 A1024, using complete
generated answer plus EOS exact match. The sole primary comparison S_D-S_U is
**+8.7601 percentage points**, sample SD 4.8776,
range [+3.6621, +17.1509], with a paired 95% t interval
**[+4.6823, +12.8378]**, df=7. 8 of eight differences
are positive, 0 negative and 0 zero.

The primary interval is positive, supporting additional transfer value of deficiencies over the specified U source in this background and budget.

![Each paired primary difference](primary_paired_difference.png)

## All prespecified area contrasts

All differences are computed within unit before aggregation. Secondary contrasts
are descriptive and cannot replace the primary contrast.

| Contrast | Mean (pp) | Sample SD | Unit range (pp) | Positive units |
| --- | ---: | ---: | --- | ---: |
| D_minus_U | +8.7601 | 4.8776 | [+3.6621, +17.1509] | 8/8 |
| D_minus_P | +8.1131 | 4.0236 | [+4.0039, +15.6006] | 8/8 |
| U_minus_P | -0.6470 | 6.1354 | [-13.1470, +5.3711] | 6/8 |
| D_minus_T | +7.9773 | 4.3632 | [+2.0020, +12.7319] | 8/8 |
| P_minus_A | -6.3461 | 2.9339 | [-11.7432, -2.0630] | 0/8 |
| D_minus_A | +1.7670 | 1.3512 | [+0.0000, +3.8574] | 7/8 |
| T_minus_A | -6.2103 | 3.5444 | [-10.7910, -1.8921] | 0/8 |
| U_minus_A | -6.9931 | 4.5911 | [-15.2100, -2.0874] | 0/8 |

The per-size table contains all ten pairwise differences at each of the three
sample sizes. It retains every unit and both selected/endpoint diagnostics;
`metric=legacy_answer_token` is an explicitly separate evaluation bridge.
It is not pooled with the complete-answer metric.

![Target sample-size curves and paired gains](target_learning_curves.png)

## Protocol and supervision

Common sources are descents, successions and left_to_right_maxima; the fourth
slot is P=peaks, D=deficiencies, T=anti_fixed_points or U=right_to_left_maxima.
A is one shared scratch adaptation per unit/sample-size/LR candidate. P/D match
the old v5 C task sets but use the newly declared common-first slot order. T is
a new composition, not a direct replay of old B.

The model/tokenizer/input/loss/optimizer and schedules inherit v5: one-line
n=10, vocabulary 188, four Transformer layers, width 256, eight heads,
4800 pretraining updates, batch 64 with 16 samples per task. Each adaptation
starts independently from the source endpoint or paired step-0 state, resets
the fixed_points tied token row to its step-0 value, and uses a fresh optimizer.
All parameters adapt for 800 updates at each of three fixed learning rates.
Development selection uses full-answer accuracy, answer CE, earliest positive
step, then fixed LR order. This full-answer selection/evaluation differs from
the legacy v5 answer-token metric and is disclosed rather than merged.

Training subsets are nested 256/512/1024. **Each unit additionally uses 1024
labeled target development examples** for selection. A 256-example adaptation
is therefore not a procedure with only 256 target labels in total. All 120
selections freeze before independent new tests are generated. No test-driven
task, checkpoint, LR, sample-size or budget choice occurs.

| Target training examples | Training-derived majority test accuracy (%) | SD across units |
| --- | ---: | ---: |
| 256 | 36.9324 | 1.0068 |
| 512 | 36.6943 | 1.0384 |
| 1024 | 36.9934 | 1.0202 |

## Source learning and accounting

All source training/development curves, full-answer accuracy, loss and
training-derived majority baselines are retained. `source_endpoint_summary.csv`
separates common-source performance by composition and split; the original
per-unit records remain in `../source_curves.csv`. Training/exposure records
report actual tokens, supervised tokens, samples, unique inputs, updates and
runtime in `../exposures.csv` and `../accounting.csv`. Equal update counts alone
are not asserted to match all resources. Engineering checks/retries are
separate from the eight scientific units.

| Stage | Runs | Updates | Samples | Sequence tokens | Supervised tokens | Runtime (hours) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pretrain | 32 | 153600 | 9830400 | 275251200 | 19660800 | 0.6091 |
| adapt | 360 | 288000 | 18432000 | 516096000 | 36864000 | 2.6309 |

Runtime is the sum of recorded scientific training-run times, not total elapsed
wall time; source/development evaluation, final tests, data checks and engineering
checks have separate records. Unique input counts are per run; summing reused
inputs across conditions would not measure independent data.

## Interpretation limits

The paired t interval assumes approximately independent normal unit differences;
eight units provide limited diagnostics. Global inverse/reverse/complement orbit
rejection draws from a finite universe and introduces dependence, despite new
RNG streams. The interval has assumption-sensitive coverage, and eight units
are a fixed compute budget rather than a power guarantee. Conditions, samples,
models reused across sample sizes and shared baselines do not enlarge n.

FP+E+D=n and E(p)=D(inverse(p)) are mathematical identities. D(p) alone is not
enough for that reconstruction. Anti-fixed points are connected through
complement/reversal and can also use positional comparison. U is related to
the common left-to-right-record task by reversal; it is not mathematically
unrelated, nor matched to D in difficulty, distribution or redundancy.
Training supplies no paired-transform queries, formula supervision or local
position labels. Positive transfer cannot prove formula use or an independently
transplantable lemma module. Source-learning correlations are descriptive,
not mediator-causal evidence.

If D and U both improve over P without D exceeding U, improvement over the old
background is not specific evidence for deficiencies. D-versus-T concerns
these two sources only, not a general hierarchy of shared computation and
transformation relations. U does not fully separate adding a task from removing
peaks. No outcomes trigger another target, architecture, budget or background.

## Artifacts and reproduction

See `scores_per_repeat.csv`, `paired_differences.csv`, `paired_summary.csv`,
`condition_summary.csv`, `primary_summary.json`, `../results_test.csv`, and
`../predictions/`. Independent raw-prediction recounting is implemented in
`../../independent_recompute.py`, separately from the analysis calculations.

```powershell
.venv/Scripts/python.exe experiments/fixed_points_deficiencies_v1/analysis.py --artifacts experiments/fixed_points_deficiencies_v1/artifacts
.venv/Scripts/python.exe experiments/fixed_points_deficiencies_v1/independent_recompute.py --artifacts experiments/fixed_points_deficiencies_v1/artifacts
```

No upload, publication or additional experiment is performed by analysis.
