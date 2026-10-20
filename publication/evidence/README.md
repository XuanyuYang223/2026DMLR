# Evidence scopes

The default DMLR release uses `dmlr_claims.csv`, `dmlr_evidence.md` and
`dmlr_recompute_checks.json`, rebuilt by `build_dmlr_evidence.py` from the retained
DMLR experiment tables. Figure inputs are listed in `../figures/SOURCE_MAP.json`.

The following files are unchanged archival records of the earlier integration:

- `original_permworld_audit.md`, `source_inventory.csv`, `UPSTREAM_AUDIT.json`
  and `upstream_resource_inventory.json`.
- `upstream_alignment_final_norm.csv`, `upstream_alignment_by_seed.csv` and
  `upstream_alignment_by_relation.csv`.
- `audit_original_permworld.py`, the optional historical re-extraction tool.

Their narrative describes the earlier combined-paper scope. The current DMLR
paper boundary supersedes that framing: original workshop alignment is cited
background only, and newer TMLR representation results belong to the separate
paper. These files are neither main-figure inputs nor default verification or
installation dependencies. They are retained for provenance, not as DMLR results.

The original PermWorld task definitions, exact generators, transformations and
mathematical relations remain credited inherited components of the DMLR platform.
No historical repository or experiment is rewritten by this separation.
