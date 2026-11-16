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
