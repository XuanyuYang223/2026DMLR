# Completed source-learning records

This descriptive audit covers the 32 fixed-endpoint source models only: eight paired units and four compositions. It does not inspect target adaptation outcomes or test data. The table entries are eight-unit means followed by minimum and maximum; accuracy and majority entries are percentages. All tasks and units are retained.

## Completeness and exposure pairing

All 32 models completed 4,800 updates. There are 128 task-exposure rows and 1,536 source curve rows (four tasks, train/development, six recorded steps per model). The evaluation steps are 0, 600, 1,200, 2,400, 3,600, and 4,800. Direct comparison of all `(4,8192)` exposure-count arrays confirms exact equality across P/D/T/U for every unit, including the changed fourth slot. All four conditions also share the recorded initialization, train/development input hashes, and observed sampler-stream hash within each unit.

Each task receives exactly 76,800 sample exposures and 2,150,400 nonpadding sequence tokens (1,996,800 prompt tokens and 153,600 supervised answer/EOS tokens). Actual unique inputs per task range from 8,188 to 8,192; the common pool has 8,192 inputs. Across all eight units there are 65,536 distinct source-training permutations, shared across conditions and tasks rather than multiplied by those counts.

| Accounting quantity | All 32 source models |
|---|---:|
| updates | 153,600 |
| sample exposures | 9,830,400 |
| prompt tokens | 255,590,400 |
| training sequence tokens | 275,251,200 |
| supervised tokens | 19,660,800 |
| Sum of recorded run times | 2192.65 seconds (36.54 minutes) |

The input format is 26 prompt tokens plus one answer and EOS, giving 28 forwarded nonpadding sequence tokens per sample. Equal source sample counts here therefore also give equal source sequence and supervised tokens. The recorded time sum includes source evaluations and periodic checkpoint work but is not total setup/process overhead.

## Endpoint performance

Answer CE is the answer-token cross-entropy. Majority labels are determined on each source training pool with the smallest-answer tie rule and applied unchanged to development inputs. Full-answer accuracy requires the answer token and generated EOS to match. The machine-readable audit also retains two-token loss and the legacy first-answer metric.

| Condition | Task | Split | Full accuracy mean [min, max] (%) | Answer CE mean [min, max] | Majority mean [min, max] (%) |
|---|---|---|---:|---:|---:|
| P | descents | train | 67.85 [53.96, 96.96] | 0.7448 [0.1160, 1.0263] | 36.56 [35.84, 37.15] |
| P | descents | dev | 66.11 [53.91, 96.09] | 0.7722 [0.1302, 1.0305] | 36.07 [33.59, 39.75] |
| P | successions | train | 55.07 [45.73, 75.73] | 1.0081 [0.6191, 1.1678] | 40.57 [39.98, 41.42] |
| P | successions | dev | 53.33 [45.21, 73.83] | 1.0525 [0.7001, 1.2296] | 39.45 [37.40, 40.62] |
| P | left_to_right_maxima | train | 97.59 [96.40, 98.71] | 0.0745 [0.0456, 0.1010] | 32.28 [31.35, 32.84] |
| P | left_to_right_maxima | dev | 97.36 [95.51, 98.24] | 0.0824 [0.0562, 0.1254] | 32.54 [31.35, 33.20] |
| P | peaks | train | 67.65 [62.44, 78.58] | 0.7178 [0.4967, 0.8142] | 50.77 [50.29, 51.29] |
| P | peaks | dev | 66.82 [60.25, 77.34] | 0.7356 [0.5248, 0.8766] | 50.68 [49.22, 51.76] |
| D | descents | train | 57.22 [50.26, 65.64] | 0.9603 [0.7770, 1.0998] | 36.56 [35.84, 37.15] |
| D | descents | dev | 55.07 [47.17, 63.87] | 1.0014 [0.8252, 1.1639] | 36.07 [33.59, 39.75] |
| D | successions | train | 48.33 [45.51, 54.52] | 1.1227 [1.0029, 1.1826] | 40.57 [39.98, 41.42] |
| D | successions | dev | 46.47 [45.21, 49.80] | 1.1660 [1.0889, 1.2046] | 39.45 [37.40, 40.62] |
| D | left_to_right_maxima | train | 98.05 [95.74, 99.13] | 0.0569 [0.0308, 0.1200] | 32.28 [31.35, 32.84] |
| D | left_to_right_maxima | dev | 97.63 [95.51, 99.02] | 0.0665 [0.0387, 0.1229] | 32.54 [31.35, 33.20] |
| D | deficiencies | train | 99.87 [99.71, 99.98] | 0.0188 [0.0110, 0.0283] | 36.40 [35.79, 36.72] |
| D | deficiencies | dev | 99.82 [99.41, 100.00] | 0.0201 [0.0101, 0.0297] | 35.61 [33.89, 38.18] |
| T | descents | train | 50.69 [47.13, 58.61] | 1.0939 [0.9398, 1.1657] | 36.56 [35.84, 37.15] |
| T | descents | dev | 49.85 [44.82, 56.15] | 1.1204 [0.9776, 1.1895] | 36.07 [33.59, 39.75] |
| T | successions | train | 46.46 [42.77, 52.20] | 1.1635 [1.0786, 1.2274] | 40.57 [39.98, 41.42] |
| T | successions | dev | 45.09 [42.38, 48.44] | 1.1982 [1.1451, 1.2687] | 39.45 [37.40, 40.62] |
| T | left_to_right_maxima | train | 97.53 [94.82, 99.15] | 0.0746 [0.0403, 0.1388] | 32.28 [31.35, 32.84] |
| T | left_to_right_maxima | dev | 97.42 [95.21, 99.22] | 0.0793 [0.0403, 0.1414] | 32.54 [31.35, 33.20] |
| T | anti_fixed_points | train | 93.42 [82.43, 99.08] | 0.1949 [0.0398, 0.4665] | 37.20 [36.57, 37.81] |
| T | anti_fixed_points | dev | 93.31 [81.35, 99.22] | 0.1950 [0.0396, 0.4587] | 37.12 [35.74, 39.16] |
| U | descents | train | 59.98 [53.91, 75.37] | 0.8960 [0.6188, 1.0104] | 36.56 [35.84, 37.15] |
| U | descents | dev | 58.00 [50.78, 72.75] | 0.9383 [0.6684, 1.0625] | 36.07 [33.59, 39.75] |
| U | successions | train | 47.55 [44.74, 50.85] | 1.1411 [1.0833, 1.1738] | 40.57 [39.98, 41.42] |
| U | successions | dev | 45.78 [43.95, 47.56] | 1.1787 [1.1470, 1.2127] | 39.45 [37.40, 40.62] |
| U | left_to_right_maxima | train | 97.13 [95.17, 98.85] | 0.0844 [0.0434, 0.1270] | 32.28 [31.35, 32.84] |
| U | left_to_right_maxima | dev | 96.41 [94.24, 98.83] | 0.0974 [0.0486, 0.1468] | 32.54 [31.35, 33.20] |
| U | right_to_left_maxima | train | 94.86 [92.49, 98.47] | 0.1314 [0.0499, 0.1869] | 32.38 [31.09, 33.29] |
| U | right_to_left_maxima | dev | 94.54 [92.38, 97.95] | 0.1389 [0.0598, 0.1983] | 32.63 [30.47, 35.06] |

## Common-source differences

These are within-unit differences at the fixed endpoint; averaging comes after pairing. They describe how source learning differs under the frozen compositions. They do not identify interference, formula use, causal mediation, or a downstream transfer mechanism.

| Common task | Development contrast | Accuracy difference mean [min, max] (pp) | Answer CE difference mean [min, max] |
|---|---|---:|---:|
| descents | D-U | -2.93 [-11.23, 4.30] | 0.0630 [-0.0967, 0.2469] |
| descents | D-P | -11.05 [-39.55, 7.23] | 0.2291 [-0.1728, 0.8923] |
| descents | U-P | -8.12 [-39.94, 16.11] | 0.1661 [-0.3296, 0.8347] |
| descents | D-T | 5.21 [-0.68, 10.25] | -0.1190 [-0.2362, -0.0244] |
| successions | D-U | 0.70 [-2.05, 3.22] | -0.0127 [-0.0756, 0.0215] |
| successions | D-P | -6.86 [-28.32, 4.20] | 0.1135 [-0.1408, 0.5045] |
| successions | U-P | -7.56 [-26.37, 2.34] | 0.1262 [-0.0652, 0.4829] |
| successions | D-T | 1.38 [-2.44, 4.79] | -0.0322 [-0.0685, 0.0207] |
| left_to_right_maxima | D-U | 1.22 [-0.39, 3.12] | -0.0309 [-0.0691, 0.0058] |
| left_to_right_maxima | D-P | 0.27 [-2.64, 2.54] | -0.0160 [-0.0617, 0.0542] |
| left_to_right_maxima | U-P | -0.95 [-4.00, 2.64] | 0.0149 [-0.0601, 0.0819] |
| left_to_right_maxima | D-T | 0.21 [-1.07, 1.56] | -0.0128 [-0.0413, 0.0134] |

The candidate tasks are not equally mastered at this budget. Development mean accuracy is 99.82% for deficiencies, 93.31% for anti-fixed points, 94.54% for right-to-left maxima, and 66.82% for peaks. The common left-to-right-maxima task reaches 96.41?97.63% mean development accuracy across conditions, while descents and successions remain much less accurate. Moderate training accuracy for those tasks is evidence of incomplete source learning under this protocol; the small train/development gaps do not establish a unique optimization or representation explanation.

Across the eight units, the D composition has lower mean common-descents accuracy than P and U, while D-versus-U successions differences are small and vary in sign. The precise paired values and their ranges are retained above and in the JSON. These observations must remain separate from the target transfer comparison. No source-learning result was used to exclude a unit, change the candidate, or extend training.

## Reusable evidence

- `source_curves.csv`: all 1,536 train/development learning records.
- `exposures.csv`: all 128 task exposure and token records.
- `SOURCE_ONLY_AUDIT.json`: endpoint summaries, every paired common-source difference, per-unit equality checks, timing, and input-file hashes.
- `pretrain/<unit>_<condition>/input_exposure_counts.npz`: actual per-task/per-input exposure arrays checked in this audit.

The raw source records remain unchanged. The shared accounting table may later gain adaptation rows; this audit binds the individual source completion records and its original source-only accounting snapshot separately.
