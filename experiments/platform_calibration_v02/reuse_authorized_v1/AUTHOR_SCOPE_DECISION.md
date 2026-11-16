# Author scope decision: shared canonical calibration pools

The author resolved the Stage 2 data-capacity conflict on 2026-11-16:

- Reuse the canonical platform_v01 training and development pools, unchanged, at lengths 8, 10 and 12. Preserve the original seed-to-data-unit pairing.
- First verify by hash and archived run records that the original six-task calibration actually used those pools. Stop on an inconsistency rather than infer missing provenance.
- The fresh-unit historical-orbit exclusion requirement applies to the subsequent transfer replication, not to this development-only single-task calibration. Different tasks share the same inputs and unchanged train/development split. Reserved test inputs are not opened or evaluated here.
- Report that the calibration shares data units with platform_v01; tasks and model runs do not provide independent additional data units.
- Preserve the earlier CAPACITY_AUDIT.json. The later paper must explain that the remaining n=8 orbit capacity cannot accommodate another calibration of the original size under historical exclusion.

All other scientific calibration settings remain inherited. Stage 3 still requires new paired n=10 transfer units and a separate capacity check. Existing reports, frozen files and the earlier stopped-preflight directory contents are preserved. New continuation files are isolated in this subdirectory.
