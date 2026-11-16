# Remaining platform-instance calibration results

The fixed grid is complete: 50 requested instances, 49 trained with both inherited units (98 runs), and one explicit reader rejection. All runs use the endpoint at update 2400 as the primary development reference. No model test forward or new data pool was produced. The rejected `to_reduced_word` instance has no invented accuracy; no input was truncated, filtered or replaced.

## Data reuse and comparability

The training/development inputs and seed-to-unit pairing are the same canonical pools used by the original six-task calibration. The pool-reuse audit verifies their hashes, archived run/checkpoint bindings and index streams. These results are directly comparable in data split, initialization pairing, training configuration and decoding budget, and they are not independent data units. Different tasks share inputs intentionally. Reserved test inputs remain unopened and no model forward uses them.

The original capacity failure remains an immutable historical record. The author resolved it for this calibration by permitting read-only reuse; the separate transfer experiment still requires fresh historical-orbit exclusion. This stage does not produce a new transfer result.

## Endpoint results and interpretation

The table below includes every requested instance at each trained length and pooled across lengths. Means summarize the same two units; no confidence interval or family ranking is claimed. There are 8 completed instances with a pooled development generation ceiling below one and 8 completed instances with a mean endpoint development score below their training-derived majority baseline. These are descriptive counts, not independent statistical trials or task-removal rules.

The model still generates at most eight tokens including EOS. Complete gold answers remain the loss and exact-match reference. A ceiling below one therefore identifies protocol-limited exact accuracy: a zero score on an impossible-length answer does not establish that the underlying mathematical computation is unlearnable. Context-valid structured tasks remain in the results. Loss curves supervise their entire gold answer. All three lengths occur in training; per-length development differences are not unseen-length extrapolation.

The majority answer is determined from the corresponding training pool separately for each length. `endpoint_by_unit.csv` includes both train/development baselines, all losses, exact counts and per-unit cap fractions. `endpoint_summary.csv` provides the unrounded table values. `LEARNABILITY_INDEX_v02.csv` preserves the original order and marks the six original calibrations as inherited, the new completions as shared-unit calibration, the context rejection explicitly, and nonmaterialized binary/alias exclusions without changing the old index.

| Instance | Family | Length | Dev617 (%) | Dev829 (%) | Mean dev (%) | Mean train (%) | Dev majority (%) | Gain (pp) | Dev ceiling (%) | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| to_cycle | permutation_identity | 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| to_cycle | permutation_identity | 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| to_cycle | permutation_identity | 12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| to_cycle | permutation_identity | all | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| to_lehmer | permutation_identity | 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| to_lehmer | permutation_identity | 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| to_lehmer | permutation_identity | 12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| to_lehmer | permutation_identity | all | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| to_inversion_vector | permutation_identity | 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| to_inversion_vector | permutation_identity | 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| to_inversion_vector | permutation_identity | 12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| to_inversion_vector | permutation_identity | all | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| to_reduced_word | permutation_identity | 8 | not measured | not measured | not measured | not measured | not measured | not measured | not measured | reader-rejected: context |
| to_reduced_word | permutation_identity | 10 | not measured | not measured | not measured | not measured | not measured | not measured | not measured | reader-rejected: context |
| to_reduced_word | permutation_identity | 12 | not measured | not measured | not measured | not measured | not measured | not measured | not measured | reader-rejected: context |
| to_reduced_word | permutation_identity | all | not measured | not measured | not measured | not measured | not measured | not measured | not measured | reader-rejected: context |
| length | length | 8 | 69.92 | 77.73 | 73.83 | 82.18 | 8.40 | +65.43 | 100.00 | completed |
| length | length | 10 | 61.72 | 66.02 | 63.87 | 75.10 | 7.42 | +56.45 | 100.00 | completed |
| length | length | 12 | 37.50 | 33.20 | 35.35 | 53.27 | 5.08 | +30.27 | 100.00 | completed |
| length | length | all | 56.38 | 58.98 | 57.68 | 70.18 | 6.97 | +50.72 | 100.00 | completed |
| descents | descents | 8 | 100.00 | 99.61 | 99.80 | 99.95 | 40.62 | +59.18 | 100.00 | completed |
| descents | descents | 10 | 99.61 | 100.00 | 99.80 | 100.00 | 38.09 | +61.72 | 100.00 | completed |
| descents | descents | 12 | 97.66 | 100.00 | 98.83 | 99.41 | 33.59 | +65.23 | 100.00 | completed |
| descents | descents | all | 99.09 | 99.87 | 99.48 | 99.79 | 37.43 | +62.04 | 100.00 | completed |
| fixed_points | fixed_points | 8 | 100.00 | 100.00 | 100.00 | 99.71 | 36.52 | +63.48 | 100.00 | completed |
| fixed_points | fixed_points | 10 | 99.61 | 100.00 | 99.80 | 99.90 | 33.40 | +66.41 | 100.00 | completed |
| fixed_points | fixed_points | 12 | 100.00 | 100.00 | 100.00 | 99.80 | 38.28 | +61.72 | 100.00 | completed |
| fixed_points | fixed_points | all | 99.87 | 100.00 | 99.93 | 99.80 | 36.07 | +63.87 | 100.00 | completed |
| parity | parity | 8 | 53.52 | 48.44 | 50.98 | 72.07 | 48.24 | +2.73 | 100.00 | completed |
| parity | parity | 10 | 50.78 | 51.17 | 50.98 | 77.49 | 47.46 | +3.52 | 100.00 | completed |
| parity | parity | 12 | 48.05 | 48.83 | 48.44 | 78.66 | 49.02 | -0.59 | 100.00 | completed |
| parity | parity | all | 50.78 | 49.48 | 50.13 | 76.07 | 48.24 | +1.89 | 100.00 | completed |
| cycle_type | cycle_decomposition | 8 | 34.38 | 33.98 | 34.18 | 42.77 | 13.87 | +20.31 | 76.37 | completed; generation-capped |
| cycle_type | cycle_decomposition | 10 | 18.75 | 23.44 | 21.09 | 38.23 | 10.74 | +10.35 | 69.14 | completed; generation-capped |
| cycle_type | cycle_decomposition | 12 | 21.09 | 12.89 | 16.99 | 35.69 | 10.94 | +6.05 | 70.12 | completed; generation-capped |
| cycle_type | cycle_decomposition | all | 24.74 | 23.44 | 24.09 | 38.90 | 11.85 | +12.24 | 71.88 | completed; generation-capped |
| rsk_shape | global_subsequence | 8 | 33.98 | 28.91 | 31.45 | 38.62 | 24.22 | +7.23 | 36.33 | completed; generation-capped |
| rsk_shape | global_subsequence | 10 | 14.84 | 13.67 | 14.26 | 12.45 | 14.84 | -0.59 | 20.90 | completed; generation-capped |
| rsk_shape | global_subsequence | 12 | 1.56 | 1.56 | 1.56 | 3.08 | 9.96 | -8.40 | 5.27 | completed; generation-capped |
| rsk_shape | global_subsequence | all | 16.80 | 14.71 | 15.76 | 18.05 | 16.34 | -0.59 | 20.83 | completed; generation-capped |
| lds_length | global_subsequence | 8 | 98.83 | 98.83 | 98.83 | 99.95 | 46.09 | +52.73 | 100.00 | completed |
| lds_length | global_subsequence | 10 | 96.88 | 88.28 | 92.58 | 99.22 | 40.43 | +52.15 | 100.00 | completed |
| lds_length | global_subsequence | 12 | 80.86 | 76.95 | 78.91 | 96.83 | 41.60 | +37.30 | 100.00 | completed |
| lds_length | global_subsequence | all | 92.19 | 88.02 | 90.10 | 98.67 | 42.71 | +47.40 | 100.00 | completed |
| pattern_avoidance[pattern=1,2,3] | pattern_avoidance | 8 | 96.88 | 96.88 | 96.88 | 99.90 | 96.29 | +0.59 | 100.00 | completed |
| pattern_avoidance[pattern=1,2,3] | pattern_avoidance | 10 | 99.61 | 100.00 | 99.80 | 100.00 | 99.80 | +0.00 | 100.00 | completed |
| pattern_avoidance[pattern=1,2,3] | pattern_avoidance | 12 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | 100.00 | completed |
| pattern_avoidance[pattern=1,2,3] | pattern_avoidance | all | 98.83 | 98.96 | 98.89 | 99.97 | 98.70 | +0.20 | 100.00 | completed |
| inverse | inverse | 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| inverse | inverse | 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| inverse | inverse | 12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| inverse | inverse | all | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| power[exponent=2] | power | 8 | 0.00 | 0.00 | 0.00 | 0.00 | 2.34 | -2.34 | 0.00 | completed; generation-capped |
| power[exponent=2] | power | 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| power[exponent=2] | power | 12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| power[exponent=2] | power | all | 0.00 | 0.00 | 0.00 | 0.00 | 0.78 | -0.78 | 0.00 | completed; generation-capped |
| right_multiply_simple[index=1] | right_multiply_simple | 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| right_multiply_simple[index=1] | right_multiply_simple | 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| right_multiply_simple[index=1] | right_multiply_simple | 12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| right_multiply_simple[index=1] | right_multiply_simple | all | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.00 | completed; generation-capped |
| peaks | peaks | 8 | 96.48 | 90.23 | 93.36 | 98.78 | 57.81 | +35.55 | 100.00 | completed |
| peaks | peaks | 10 | 92.58 | 76.17 | 84.38 | 96.39 | 50.59 | +33.79 | 100.00 | completed |
| peaks | peaks | 12 | 73.05 | 65.23 | 69.14 | 91.06 | 48.63 | +20.51 | 100.00 | completed |
| peaks | peaks | all | 87.37 | 77.21 | 82.29 | 95.41 | 52.34 | +29.95 | 100.00 | completed |
| exceedances | exceedances | 8 | 100.00 | 100.00 | 100.00 | 100.00 | 36.52 | +63.48 | 100.00 | completed |
| exceedances | exceedances | 10 | 100.00 | 100.00 | 100.00 | 100.00 | 36.33 | +63.67 | 100.00 | completed |
| exceedances | exceedances | 12 | 99.22 | 99.61 | 99.41 | 99.95 | 32.62 | +66.80 | 100.00 | completed |
| exceedances | exceedances | all | 99.74 | 99.87 | 99.80 | 99.98 | 35.16 | +64.65 | 100.00 | completed |
| recoils | recoils | 8 | 76.17 | 98.83 | 87.50 | 95.21 | 34.38 | +53.12 | 100.00 | completed |
| recoils | recoils | 10 | 66.02 | 77.73 | 71.88 | 90.48 | 34.57 | +37.30 | 100.00 | completed |
| recoils | recoils | 12 | 53.91 | 53.52 | 53.71 | 80.47 | 32.03 | +21.68 | 100.00 | completed |
| recoils | recoils | all | 65.36 | 76.69 | 71.03 | 88.72 | 33.66 | +37.37 | 100.00 | completed |
| valleys | valleys | 8 | 98.44 | 98.44 | 98.44 | 99.66 | 58.79 | +39.65 | 100.00 | completed |
| valleys | valleys | 10 | 89.45 | 94.53 | 91.99 | 98.83 | 51.37 | +40.62 | 100.00 | completed |
| valleys | valleys | 12 | 77.34 | 81.25 | 79.30 | 96.68 | 50.00 | +29.30 | 100.00 | completed |
| valleys | valleys | all | 88.41 | 91.41 | 89.91 | 98.39 | 53.39 | +36.52 | 100.00 | completed |
| double_ascents | double_ascents | 8 | 100.00 | 99.61 | 99.80 | 99.85 | 43.95 | +55.86 | 100.00 | completed |
| double_ascents | double_ascents | 10 | 96.48 | 99.22 | 97.85 | 99.61 | 36.13 | +61.72 | 100.00 | completed |
| double_ascents | double_ascents | 12 | 77.73 | 88.28 | 83.01 | 95.21 | 33.79 | +49.22 | 100.00 | completed |
| double_ascents | double_ascents | all | 91.41 | 95.70 | 93.55 | 98.23 | 37.96 | +55.60 | 100.00 | completed |
| double_descents | double_descents | 8 | 100.00 | 100.00 | 100.00 | 100.00 | 39.45 | +60.55 | 100.00 | completed |
| double_descents | double_descents | 10 | 99.61 | 99.61 | 99.61 | 99.76 | 41.21 | +58.40 | 100.00 | completed |
| double_descents | double_descents | 12 | 94.14 | 96.09 | 95.12 | 98.73 | 30.86 | +64.26 | 100.00 | completed |
| double_descents | double_descents | all | 97.92 | 98.57 | 98.24 | 99.50 | 37.17 | +61.07 | 100.00 | completed |
| successions | successions | 8 | 43.75 | 41.41 | 42.58 | 73.44 | 40.82 | +1.76 | 100.00 | completed |
| successions | successions | 10 | 33.98 | 44.92 | 39.45 | 69.82 | 39.65 | -0.20 | 100.00 | completed |
| successions | successions | 12 | 33.20 | 34.77 | 33.98 | 71.29 | 36.52 | -2.54 | 100.00 | completed |
| successions | successions | all | 36.98 | 40.36 | 38.67 | 71.52 | 39.00 | -0.33 | 100.00 | completed |
| adjacencies | adjacencies | 8 | 39.45 | 43.36 | 41.41 | 71.19 | 32.42 | +8.98 | 100.00 | completed |
| adjacencies | adjacencies | 10 | 38.28 | 34.77 | 36.52 | 67.24 | 29.49 | +7.03 | 100.00 | completed |
| adjacencies | adjacencies | 12 | 27.73 | 28.52 | 28.12 | 67.19 | 27.54 | +0.59 | 100.00 | completed |
| adjacencies | adjacencies | all | 35.16 | 35.55 | 35.35 | 68.54 | 29.82 | +5.53 | 100.00 | completed |
| anti_fixed_points | anti_fixed_points | 8 | 100.00 | 99.22 | 99.61 | 99.76 | 31.64 | +67.97 | 100.00 | completed |
| anti_fixed_points | anti_fixed_points | 10 | 99.61 | 98.83 | 99.22 | 99.76 | 39.65 | +59.57 | 100.00 | completed |
| anti_fixed_points | anti_fixed_points | 12 | 96.88 | 96.48 | 96.68 | 99.56 | 37.11 | +59.57 | 100.00 | completed |
| anti_fixed_points | anti_fixed_points | all | 98.83 | 98.18 | 98.50 | 99.69 | 36.13 | +62.37 | 100.00 | completed |
| deficiencies | deficiencies | 8 | 100.00 | 100.00 | 100.00 | 100.00 | 41.21 | +58.79 | 100.00 | completed |
| deficiencies | deficiencies | 10 | 100.00 | 100.00 | 100.00 | 100.00 | 36.52 | +63.48 | 100.00 | completed |
| deficiencies | deficiencies | 12 | 100.00 | 100.00 | 100.00 | 99.85 | 35.74 | +64.26 | 100.00 | completed |
| deficiencies | deficiencies | all | 100.00 | 100.00 | 100.00 | 99.95 | 37.83 | +62.17 | 100.00 | completed |
| left_to_right_maxima | left_to_right_maxima | 8 | 99.61 | 100.00 | 99.80 | 99.95 | 31.64 | +68.16 | 100.00 | completed |
| left_to_right_maxima | left_to_right_maxima | 10 | 99.61 | 98.83 | 99.22 | 100.00 | 29.69 | +69.53 | 100.00 | completed |
| left_to_right_maxima | left_to_right_maxima | 12 | 96.48 | 98.83 | 97.66 | 99.76 | 34.38 | +63.28 | 100.00 | completed |
| left_to_right_maxima | left_to_right_maxima | all | 98.57 | 99.22 | 98.89 | 99.90 | 31.90 | +66.99 | 100.00 | completed |
| left_to_right_minima | left_to_right_minima | 8 | 100.00 | 100.00 | 100.00 | 100.00 | 30.66 | +69.34 | 100.00 | completed |
| left_to_right_minima | left_to_right_minima | 10 | 100.00 | 100.00 | 100.00 | 99.90 | 33.40 | +66.60 | 100.00 | completed |
| left_to_right_minima | left_to_right_minima | 12 | 99.22 | 99.61 | 99.41 | 99.90 | 33.20 | +66.21 | 100.00 | completed |
| left_to_right_minima | left_to_right_minima | all | 99.74 | 99.87 | 99.80 | 99.93 | 32.42 | +67.38 | 100.00 | completed |
| right_to_left_maxima | right_to_left_maxima | 8 | 99.61 | 100.00 | 99.80 | 100.00 | 31.64 | +68.16 | 100.00 | completed |
| right_to_left_maxima | right_to_left_maxima | 10 | 98.83 | 98.83 | 98.83 | 99.80 | 31.45 | +67.38 | 100.00 | completed |
| right_to_left_maxima | right_to_left_maxima | 12 | 95.31 | 97.27 | 96.29 | 99.07 | 33.20 | +63.09 | 100.00 | completed |
| right_to_left_maxima | right_to_left_maxima | all | 97.92 | 98.70 | 98.31 | 99.63 | 32.10 | +66.21 | 100.00 | completed |
| right_to_left_minima | right_to_left_minima | 8 | 100.00 | 100.00 | 100.00 | 99.90 | 31.84 | +68.16 | 100.00 | completed |
| right_to_left_minima | right_to_left_minima | 10 | 100.00 | 99.22 | 99.61 | 99.85 | 36.33 | +63.28 | 100.00 | completed |
| right_to_left_minima | right_to_left_minima | 12 | 98.05 | 97.66 | 97.85 | 99.90 | 34.77 | +63.09 | 100.00 | completed |
| right_to_left_minima | right_to_left_minima | all | 99.35 | 98.96 | 99.15 | 99.89 | 34.31 | +64.84 | 100.00 | completed |
| even_cycle_count | cycle_decomposition | 8 | 43.75 | 41.80 | 42.77 | 64.89 | 46.68 | -3.91 | 100.00 | completed |
| even_cycle_count | cycle_decomposition | 10 | 35.94 | 35.94 | 35.94 | 64.89 | 42.58 | -6.64 | 100.00 | completed |
| even_cycle_count | cycle_decomposition | 12 | 41.80 | 37.89 | 39.84 | 67.29 | 46.29 | -6.45 | 100.00 | completed |
| even_cycle_count | cycle_decomposition | all | 40.49 | 38.54 | 39.52 | 65.69 | 45.18 | -5.66 | 100.00 | completed |
| odd_cycle_count | cycle_decomposition | 8 | 87.50 | 87.11 | 87.30 | 91.80 | 58.59 | +28.71 | 100.00 | completed |
| odd_cycle_count | cycle_decomposition | 10 | 77.34 | 77.73 | 77.54 | 93.21 | 61.72 | +15.82 | 100.00 | completed |
| odd_cycle_count | cycle_decomposition | 12 | 75.39 | 77.73 | 76.56 | 93.75 | 64.26 | +12.30 | 100.00 | completed |
| odd_cycle_count | cycle_decomposition | all | 80.08 | 80.86 | 80.47 | 92.92 | 61.52 | +18.95 | 100.00 | completed |
| shortest_cycle | cycle_decomposition | 8 | 80.86 | 78.52 | 79.69 | 86.47 | 63.48 | +16.21 | 100.00 | completed |
| shortest_cycle | cycle_decomposition | 10 | 78.52 | 78.12 | 78.32 | 87.84 | 66.60 | +11.72 | 100.00 | completed |
| shortest_cycle | cycle_decomposition | 12 | 75.78 | 69.92 | 72.85 | 90.97 | 62.30 | +10.55 | 100.00 | completed |
| shortest_cycle | cycle_decomposition | all | 78.39 | 75.52 | 76.95 | 88.43 | 64.13 | +12.83 | 100.00 | completed |
| nontrivial_cycle_count | cycle_decomposition | 8 | 48.44 | 47.66 | 48.05 | 71.92 | 45.12 | +2.93 | 100.00 | completed |
| nontrivial_cycle_count | cycle_decomposition | 10 | 37.89 | 46.88 | 42.38 | 68.70 | 46.88 | -4.49 | 100.00 | completed |
| nontrivial_cycle_count | cycle_decomposition | 12 | 36.33 | 35.16 | 35.74 | 66.89 | 42.77 | -7.03 | 100.00 | completed |
| nontrivial_cycle_count | cycle_decomposition | all | 40.89 | 43.23 | 42.06 | 69.17 | 44.92 | -2.86 | 100.00 | completed |
| longest_increasing_run | longest_increasing_run | 8 | 97.27 | 99.22 | 98.24 | 99.66 | 52.15 | +46.09 | 100.00 | completed |
| longest_increasing_run | longest_increasing_run | 10 | 85.55 | 94.14 | 89.84 | 98.58 | 54.49 | +35.35 | 100.00 | completed |
| longest_increasing_run | longest_increasing_run | 12 | 66.80 | 69.53 | 68.16 | 91.60 | 58.59 | +9.57 | 100.00 | completed |
| longest_increasing_run | longest_increasing_run | all | 83.20 | 87.63 | 85.42 | 96.61 | 55.08 | +30.34 | 100.00 | completed |
| longest_decreasing_run | longest_decreasing_run | 8 | 93.36 | 96.09 | 94.73 | 99.32 | 46.68 | +48.05 | 100.00 | completed |
| longest_decreasing_run | longest_decreasing_run | 10 | 82.03 | 88.67 | 85.35 | 97.51 | 56.45 | +28.91 | 100.00 | completed |
| longest_decreasing_run | longest_decreasing_run | 12 | 63.67 | 77.73 | 70.70 | 94.38 | 49.41 | +21.29 | 100.00 | completed |
| longest_decreasing_run | longest_decreasing_run | all | 79.69 | 87.50 | 83.59 | 97.07 | 50.85 | +32.75 | 100.00 | completed |
| global_descents | permutation_blocks | 8 | 97.66 | 96.88 | 97.27 | 99.85 | 72.46 | +24.80 | 100.00 | completed |
| global_descents | permutation_blocks | 10 | 96.48 | 98.83 | 97.66 | 99.90 | 79.30 | +18.36 | 100.00 | completed |
| global_descents | permutation_blocks | 12 | 97.66 | 98.83 | 98.24 | 99.90 | 82.23 | +16.02 | 100.00 | completed |
| global_descents | permutation_blocks | all | 97.27 | 98.18 | 97.72 | 99.89 | 77.99 | +19.73 | 100.00 | completed |
| max_displacement | max_displacement | 8 | 99.61 | 100.00 | 99.80 | 99.61 | 35.16 | +64.65 | 100.00 | completed |
| max_displacement | max_displacement | 10 | 98.83 | 99.22 | 99.02 | 100.00 | 29.10 | +69.92 | 100.00 | completed |
| max_displacement | max_displacement | 12 | 100.00 | 98.83 | 99.41 | 100.00 | 25.78 | +73.63 | 100.00 | completed |
| max_displacement | max_displacement | all | 99.48 | 99.35 | 99.41 | 99.87 | 30.01 | +69.40 | 100.00 | completed |
| displacement_one_count | displacement_one_count | 8 | 100.00 | 100.00 | 100.00 | 99.90 | 26.95 | +73.05 | 100.00 | completed |
| displacement_one_count | displacement_one_count | 10 | 100.00 | 100.00 | 100.00 | 99.85 | 27.15 | +72.85 | 100.00 | completed |
| displacement_one_count | displacement_one_count | 12 | 100.00 | 98.44 | 99.22 | 99.80 | 26.37 | +72.85 | 100.00 | completed |
| displacement_one_count | displacement_one_count | all | 100.00 | 99.48 | 99.74 | 99.85 | 26.82 | +72.92 | 100.00 | completed |
| cycle_length_count[k=1] | cycle_decomposition | 8 | 100.00 | 100.00 | 100.00 | 99.95 | 36.52 | +63.48 | 100.00 | completed |
| cycle_length_count[k=1] | cycle_decomposition | 10 | 98.83 | 100.00 | 99.41 | 99.95 | 33.40 | +66.02 | 100.00 | completed |
| cycle_length_count[k=1] | cycle_decomposition | 12 | 99.61 | 100.00 | 99.80 | 99.90 | 38.28 | +61.52 | 100.00 | completed |
| cycle_length_count[k=1] | cycle_decomposition | all | 99.48 | 100.00 | 99.74 | 99.93 | 36.07 | +63.67 | 100.00 | completed |
| cycle_length_count[k=2] | cycle_decomposition | 8 | 59.38 | 71.48 | 65.43 | 79.98 | 60.94 | +4.49 | 100.00 | completed |
| cycle_length_count[k=2] | cycle_decomposition | 10 | 56.25 | 63.67 | 59.96 | 79.93 | 60.74 | -0.78 | 100.00 | completed |
| cycle_length_count[k=2] | cycle_decomposition | 12 | 53.12 | 57.42 | 55.27 | 77.49 | 62.70 | -7.42 | 100.00 | completed |
| cycle_length_count[k=2] | cycle_decomposition | all | 56.25 | 64.19 | 60.22 | 79.13 | 61.46 | -1.24 | 100.00 | completed |
| cycle_length_count[k=3] | cycle_decomposition | 8 | 65.62 | 62.89 | 64.26 | 78.12 | 72.66 | -8.40 | 100.00 | completed |
| cycle_length_count[k=3] | cycle_decomposition | 10 | 63.67 | 69.92 | 66.80 | 80.52 | 74.80 | -8.01 | 100.00 | completed |
| cycle_length_count[k=3] | cycle_decomposition | 12 | 67.97 | 64.84 | 66.41 | 83.84 | 72.07 | -5.66 | 100.00 | completed |
| cycle_length_count[k=3] | cycle_decomposition | all | 65.76 | 65.89 | 65.82 | 80.83 | 73.18 | -7.36 | 100.00 | completed |
| increasing_subsequence_count[k=1] | global_subsequence | 8 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | 100.00 | completed |
| increasing_subsequence_count[k=1] | global_subsequence | 10 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | 100.00 | completed |
| increasing_subsequence_count[k=1] | global_subsequence | 12 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | 100.00 | completed |
| increasing_subsequence_count[k=1] | global_subsequence | all | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | 100.00 | completed |
| increasing_subsequence_count[k=2] | global_subsequence | 8 | 82.03 | 85.55 | 83.79 | 89.99 | 8.40 | +75.39 | 100.00 | completed |
| increasing_subsequence_count[k=2] | global_subsequence | 10 | 56.25 | 55.86 | 56.05 | 68.70 | 7.42 | +48.63 | 100.00 | completed |
| increasing_subsequence_count[k=2] | global_subsequence | 12 | 41.80 | 36.72 | 39.26 | 53.37 | 5.08 | +34.18 | 100.00 | completed |
| increasing_subsequence_count[k=2] | global_subsequence | all | 60.03 | 59.38 | 59.70 | 70.69 | 6.97 | +52.73 | 100.00 | completed |
| decreasing_subsequence_count[k=1] | global_subsequence | 8 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | 100.00 | completed |
| decreasing_subsequence_count[k=1] | global_subsequence | 10 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | 100.00 | completed |
| decreasing_subsequence_count[k=1] | global_subsequence | 12 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | 100.00 | completed |
| decreasing_subsequence_count[k=1] | global_subsequence | all | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 | 100.00 | completed |
| decreasing_subsequence_count[k=2] | global_subsequence | 8 | 77.34 | 84.77 | 81.05 | 87.26 | 8.40 | +72.66 | 100.00 | completed |
| decreasing_subsequence_count[k=2] | global_subsequence | 10 | 67.97 | 69.92 | 68.95 | 77.64 | 7.42 | +61.52 | 100.00 | completed |
| decreasing_subsequence_count[k=2] | global_subsequence | 12 | 27.73 | 36.33 | 32.03 | 51.51 | 5.08 | +26.95 | 100.00 | completed |
| decreasing_subsequence_count[k=2] | global_subsequence | all | 57.68 | 63.67 | 60.68 | 72.14 | 6.97 | +53.71 | 100.00 | completed |
| decreasing_subsequence_count[k=3] | global_subsequence | 8 | 53.52 | 48.05 | 50.78 | 66.06 | 7.03 | +43.75 | 100.00 | completed |
| decreasing_subsequence_count[k=3] | global_subsequence | 10 | 31.25 | 28.52 | 29.88 | 47.85 | 2.93 | +26.95 | 100.00 | completed |
| decreasing_subsequence_count[k=3] | global_subsequence | 12 | 10.16 | 11.72 | 10.94 | 36.96 | 1.76 | +9.18 | 100.00 | completed |
| decreasing_subsequence_count[k=3] | global_subsequence | all | 31.64 | 29.43 | 30.53 | 50.29 | 3.91 | +26.63 | 100.00 | completed |
| smallest_component | permutation_blocks | 8 | 99.61 | 99.61 | 99.61 | 100.00 | 72.27 | +27.34 | 100.00 | completed |
| smallest_component | permutation_blocks | 10 | 98.83 | 100.00 | 99.41 | 100.00 | 73.44 | +25.98 | 100.00 | completed |
| smallest_component | permutation_blocks | 12 | 100.00 | 99.22 | 99.61 | 100.00 | 85.55 | +14.06 | 100.00 | completed |
| smallest_component | permutation_blocks | all | 99.48 | 99.61 | 99.54 | 100.00 | 77.08 | +22.46 | 100.00 | completed |
| component_length_count[k=1] | permutation_blocks | 8 | 98.83 | 98.44 | 98.63 | 99.85 | 75.00 | +23.63 | 100.00 | completed |
| component_length_count[k=1] | permutation_blocks | 10 | 99.22 | 98.83 | 99.02 | 99.90 | 76.76 | +22.27 | 100.00 | completed |
| component_length_count[k=1] | permutation_blocks | 12 | 100.00 | 99.61 | 99.80 | 99.85 | 87.30 | +12.50 | 100.00 | completed |
| component_length_count[k=1] | permutation_blocks | all | 99.35 | 98.96 | 99.15 | 99.87 | 79.69 | +19.47 | 100.00 | completed |
| component_length_count[k=2] | permutation_blocks | 8 | 99.22 | 98.83 | 99.02 | 99.95 | 96.09 | +2.93 | 100.00 | completed |
| component_length_count[k=2] | permutation_blocks | 10 | 98.83 | 99.22 | 99.02 | 100.00 | 95.90 | +3.12 | 100.00 | completed |
| component_length_count[k=2] | permutation_blocks | 12 | 99.22 | 99.61 | 99.41 | 99.95 | 98.63 | +0.78 | 100.00 | completed |
| component_length_count[k=2] | permutation_blocks | all | 99.09 | 99.22 | 99.15 | 99.97 | 96.88 | +2.28 | 100.00 | completed |
| component_length_count[k=3] | permutation_blocks | 8 | 98.44 | 99.22 | 98.83 | 99.90 | 99.41 | -0.59 | 100.00 | completed |
| component_length_count[k=3] | permutation_blocks | 10 | 100.00 | 99.22 | 99.61 | 100.00 | 99.61 | +0.00 | 100.00 | completed |
| component_length_count[k=3] | permutation_blocks | 12 | 99.61 | 99.61 | 99.61 | 100.00 | 99.61 | +0.00 | 100.00 | completed |
| component_length_count[k=3] | permutation_blocks | all | 99.35 | 99.35 | 99.35 | 99.97 | 99.54 | -0.20 | 100.00 | completed |

## Curves, checkpoint diagnostics and actual cost

`learning_curves.csv` retains every training/development record. The pooled curve pages in `figures/` retain both units and show the generation ceiling. `checkpoint_diagnostics.csv` reports the selected development-best and fixed endpoint; the selected checkpoint never replaces the endpoint primary reference.

Actual formal totals from `accounting.csv`: 235200 optimizer updates, 15052800 example exposures, 483639226 nonpadding sequence tokens, 488783424 padded sequence tokens, 73220026 supervised tokens and 58167226 non-EOS answer tokens. Repeated exposures are not additional independent inputs. `exposure_summary.csv` gives per-task/unit/length unique input counts, exposure extrema and actual token counts; individual run directories retain each input-index count.

Accumulated run elapsed time is 11131.37 seconds. Concurrent runs overlap, so this sum is not wall time or dedicated GPU-hours. Worker start/finish envelope: 5591.73 seconds. Per-run timing includes training, training/development evaluation and checkpoint saving, and excludes data encoding and this report/audit.

## Audit and limitations

`SOURCE_ONLY_AUDIT.json` passed: the complete fixed inventory, frozen source/data hashes, original initialization states, all index/exposure streams, checkpoint hashes, endpoint optimizer counters, curve arithmetic and development selections were checked. No model forward was required for this audit. Statistical evidence remains limited to two shared units, one architecture, one fixed optimization/decoding configuration and the trained lengths. The generation cap and context rejection constrain the calibrated interface; incomplete fitting, generalization gaps and negative gains remain visible and are not reasons to delete runs.

No extra task, repeat, decoding-budget search, training extension or model test evaluation follows this report. The next separately requested transfer stage has its own fresh-unit and test-gating requirements.
