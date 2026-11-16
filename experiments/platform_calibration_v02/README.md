# Platform calibration v0.2: preflight blocked

**No calibration was run.** The requested fresh paired data cannot fit the inherited n=8 train/development sizes under full inverse/reverse/complement orbit exclusion. The continuation stops here, as required by the user-specified gate policy.

## Scope inspected

The intended scope is the intersection of the materialized instance list in
`../platform_v01/dataset/manifest.json` and the rows marked `not_run_in_v0.1`
in `../platform_v01/LEARNABILITY_INDEX.csv`: 50 instances. The four binary
endpoints and the two non-materialized aliases are outside that intersection.
Materialized representation outputs, parameter instances and constants remain
in scope. No instance was removed based on performance.

The requested calibration inherits the configuration in
`../platform_v01/calibration/FROZEN_CALIBRATION.json`: two new paired units,
lengths 8/10/12, 1,024 training and 256 development inputs per unit and length,
2,400 updates, one learning rate, and endpoint development accuracy. No model
test evaluation was requested. No feasible experimental protocol was frozen,
and no training began.

## Exact capacity obstruction

| Quantity at n=8 | Value |
| --- | ---: |
| Permutations | 40,320 |
| Complete inverse/reverse/complement orbits | 5,282 |
| Distinct orbits already used by the canonical v0.1 pools | 3,072 |
| Maximum remaining after those canonical pools alone | 2,210 |
| Required new train/development orbits | 2,560 |
| Minimum shortfall from the canonical subset alone | 350 |
| Occupied after also including archived small examples | 3,086 |
| Remaining after that additional exclusion | 2,196 |

The required count is `2 * (1024 + 256)`, with no new test reservation.
Historical reserved test inputs still count as historical inputs. Reusing them
would violate the requested isolation. Additional exclusions, including any
upstream reference inputs, cannot repair this obstruction.

See [CAPACITY_AUDIT.json](CAPACITY_AUDIT.json) for enumeration, independent
Burnside verification and input-file hashes, and
[capacity_preflight.py](capacity_preflight.py) for the standalone reproducer.
This is a data-capacity preflight, not a model evaluation or a new scientific
transfer result. It does not read historical target performance to select tasks.

## Separate unresolved interface constraint

The inherited calibration configuration has `max_new_tokens=8`.
`../../permworld_platform/calibration.py` asserts that every gold answer,
including EOS, fits that limit. The newly requested structured outputs are not
all scalar answers. The reader's explicit 128-token context rejection is a
different restriction; it does not authorize increasing the generation limit
or truncating answers. This compatibility question was identified during code
inspection and left unresolved after the decisive capacity failure.

## Status and resumption boundary

No new data, initialization, calibration curves, endpoint accuracy table, or
updated learnability result index exists. The original index remains the source
of completed calibration status. There are no new test inputs or model forwards.

The architecture replication and paper integration were not started. The author
must resolve the conflicting data constraints before resumption; this directory
does not propose or apply a replacement protocol. Existing records are preserved.
Stage-level execution and push status are in [STAGE_STATUS.md](../../STAGE_STATUS.md).
