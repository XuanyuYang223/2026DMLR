# Case-integration stage status

This work follows the four-stage request received on 2026-11-16. Existing experiment, publication and scientific records are immutable. All outputs of this continuation use new paths. The pre-existing untracked `LOCAL_STATE_SUMMARY.md` is preserved and excluded from these commits.

## Stage 1 - Completed

- Starting branch: `experiments/fixed-points-deficiencies-v1`; starting commit: `36da394d60c051e13a80bfb84e51936b9afaf4ac`.
- Ran the normal push `git push --set-upstream origin experiments/fixed-points-deficiencies-v1` successfully. It published commits `1cd90d5`, `519133c`, and `36da394` without rewriting history.
- `git ls-remote --heads origin experiments/fixed-points-deficiencies-v1` returned `36da394d60c051e13a80bfb84e51936b9afaf4ac`, matching the local tip.
- No tracked `.pt`, `.pth`, or `.ckpt` files were present. Ignored checkpoints were not added or pushed.
- Created the new continuation branch `publication/permworld-dmlr-case-integration` from that tip. No author decision is needed for Stage 1.

## Stage 2 - Capacity and protocol preflight in progress

No new training has started. The first check is whether new paired input pools can satisfy the inherited lengths and sizes while excluding every historical inverse/reverse/complement orbit. A failure without an authorized fallback will stop the continuation.

## Stage 2 - Stopped at the capacity gate

- Scope inspection identifies 50 remaining materialized unary non-alias instances. No model, training pool, development pool, test pool, or calibration result was produced.
- Exact enumeration gives 5,282 full inverse/reverse/complement orbits at n=8. An independent Burnside fixed-point count agrees.
- The six canonical v0.1 n=8 pools already occupy 3,072 distinct orbits, leaving at most 2,210. The requested two new units need 2,560 fresh orbits for training and development alone: `2 * (1024 + 256)`. This fails even before reserving any new tests. Including the archived small-example pools leaves only 2,196.
- Audit evidence and its standalone reproducer are saved under `experiments/platform_calibration_v02/`. The canonical-pool subset alone proves infeasibility; no claim of a complete upstream-history scan is needed or made. Additional historical exclusions can only decrease availability.
- No fallback for this failure was provided. Reusing historical inputs, sharing orbits across new units/roles, changing lengths, or shrinking pools would alter the request. None was done. Resumption requires an explicit author decision about the conflicting data requirements.
- The inherited eight-token generation limit is separately documented as an unresolved structured-answer compatibility issue. It was not changed, and no task was silently excluded.

## Stages 3 and 4 - Not started under the stop rule

The second-architecture gate/replication and paper integration were not started. No GRU gate result, Transformer fallback result, new paper draft, new main figure, or claim ledger is asserted. The existing experiment and publication records remain unchanged.

## Delivery

Only the new stage-status and failed-preflight evidence are to be committed and pushed on `publication/permworld-dmlr-case-integration`, by a normal push. No merge, force-push, history rewrite, checkpoint upload, or existing-file revision is part of this delivery. `LOCAL_STATE_SUMMARY.md` remains an untracked pre-existing file.

## Stage 2 - Author-approved canonical-pool reuse

The author resolved the calibration data conflict by permitting read-only reuse of the original platform_v01 train/development pools and seed pairing. New-unit historical exclusion continues to apply to Stage 3 transfer. The old capacity audit and all prior stopped-preflight files are unchanged.

New continuation outputs are in `experiments/platform_calibration_v02/reuse_authorized_v1/`. `POOL_REUSE_AUDIT.json` passed: the twelve canonical train/development hashes match both the original calibration freeze and dataset manifest; archived completions, initializations, checkpoint metadata and exact exposure streams support their actual historical use. No reserved test NPZ was opened and no model forward was run by that audit.

The complete CPU encoding scan retains all fifty requested instances. `to_reduced_word` is explicitly reader-rejected because required samples exceed context 128; no samples are filtered or replaced. The remaining forty-nine instances keep the original eight-token generation budget and full-answer training loss. Generation-impossible answers are flagged with their structural accuracy ceiling, not interpreted as failure to learn the mathematical function. The protocol and analysis plan are recorded before any new optimizer update and will be committed before smoke or formal training.

## Stage 2 - Completed after the explicit reuse decision

- Protocol and implementation commit `b95a3f3f66dd5aa99db0397618b332e7b936e5d4` preceded smoke and formal optimization. The unchanged canonical train/development pools and original seed pairing were verified against archived run, checkpoint and exposure evidence before training.
- All 98 prescribed runs completed: 49 accepted instances with both original units. The remaining requested instance, `to_reduced_word`, is explicitly reader-rejected. No reserved test input was opened by the calibration and no model test forward was performed.
- The source-only final audit returned exit code zero. Independent read-only review found no discrepancy in the 400 endpoint rows, 200 summary rows, 62-row index or output hashes. Completion of the workers is supported by their records and the audit; their expired execution-session handles do not supply OS exit codes.
- Outputs: `experiments/platform_calibration_v02/reuse_authorized_v1/README.md` and `artifacts/final/RESULT_REPORT.md`, CSVs, curves, accounting, checkpoint diagnostics and `SOURCE_ONLY_AUDIT.json`. The new index retains the original six calibrations and all nonmaterialized exclusions.
- These are shared-input, directly comparable extensions of v0.1, not independent new data units. Generation ceilings, the uncapped-majority caveat, constants, equivalent functions and weak fits remain explicit. Recorded worker-envelope wall time is 5591.73 seconds; no additional calibration or decoding-budget search is performed.
- The earlier `CAPACITY_AUDIT.json` and all stopped-preflight records are unchanged. No further author decision is required for this stage. Stage 3 may now proceed with its separate gate, new n=10 units and preflight capacity reservation; Stage 4 must retain the n=8 capacity limitation.

## Stage 3 - Capacity passed; gate registered before optimization

New work is confined to `experiments/fixed_points_deficiencies_arch_v1/`. The n=10 capacity check includes the original confirmation's input pools and explicitly distinguishes accounting archives from permutation inputs. Its conservative available bound is 185551 orbits, exceeding the 115712 required for the separate gate and eight formal units including future tests. No isolation rule changed.

The separate gate has only 8192 training and 1024 development inputs. Full historical/role orbit checks and exact generation replay passed. `GATE_FROZEN.json` records the endpoint-only GRU threshold, the prespecified Transformer fallback, all seed/config choices and bound code/data before any optimizer update. The gate registration will be committed before smoke or training. No formal pool or new test pool has been generated at this point; no model choice or replication result is claimed yet.

## Stage 3 - GRU gate passed; formal experiment registered

The gate registration commit `27f88ad4fb413f788519006aa590aacb2bfc18ee` preceded smoke and training. Both single-task fixed-4800 endpoints achieved 1024/1024 complete-answer development accuracy, selecting the four-layer width-256 GRU under the prespecified rule. Gate verification passed; its weights and data are separate from the formal units.

Formal preparation and exact replay passed for all eight new units: 32 train/development pools, 90112 objects, zero within-pool, between-pool or historical orbit overlap. Eight fresh initialization states were bound to the registered protocol. The formal registration hash is `c2ef7262ec63cc32d9b1895982ca3acbde1fed0f796fc1e0415aa78d3e3fa9ab`. It will be committed before formal smoke or optimizer updates. Tests remain ungenerated. No formal transfer result is asserted at this stage.

## Stage 3 - Complete; GRU primary comparison remains uncertain

The formal registration was committed as `5fde30e` before smoke or formal optimizer updates. All 32 source models and 360 adaptation candidates completed; both source workers and both adaptation workers returned OS exit code zero. The post-training controller completed all twelve stages with exit code zero, including the independent training audit, development-choice freeze, deferred test generation/replay, 240 evaluations and independent reconstruction of 491520 prediction records.

`experiments/fixed_points_deficiencies_arch_v1/RESULT_REPORT.md` reports the sole primary D-U score difference as +3.1235 pp, sample SD 5.2604, paired 95% t interval [-1.2743, +7.5213] pp, with six positive and two negative unit differences. The interval includes zero: additional transfer value remains uncertain, not proven absent. Scratch-relative mean differences are D +4.3427, P +1.1658, T +1.2939 and U +1.2192 pp. These eight GRU units are reported separately from the original eight Transformer units; historical negative/uncertain results are unchanged.

Outputs include the new README, canonical report, paper paragraph, raw predictions/CSV, source/target curves, actual exposure/time records and audit/manifests. The separate GRU gate passed both tasks at 100.00% endpoint development accuracy; no fallback was used. The experiment ends at the prescribed scope. No further training or test evaluation is added. Stage 4 can now integrate the new evidence without changing existing publication files.
