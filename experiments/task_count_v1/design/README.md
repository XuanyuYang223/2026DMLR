# Frozen task-count design

The task lists were specified from mathematical definitions before training this
experiment. This module reads no transfer metrics or calibration performance.
Both chains are fixed prefixes, with different membership at every k=2,4,8,16.
They use the same broad family counts: (1 cycle, 1 subsequence) at k=2;
(1,1,1 block,1 legacy) at k=4; (2,2,2,2) at k=8; (4,3,4,5) at k=16.
Thus k=2 does not span all families. At k=16 their intersection has 8 instances
and their union has all 24 pool instances. Family strata do not match difficulty.

`configuration.json` contains the complete ordered lists and selection rule;
`task_sets.csv` gives every membership and its position. `source_pool.csv` binds
instances to registry templates, mathematical families and exact definitions.
There are 24 pool instances from 22 templates; each chain prefix happens to use
distinct templates, so its template count equals k. There is one input encoding.
The term k is not a count of independent mathematical information.

All four held-out targets are excluded from every source set. The fixed-point
alias cycle_length_count(k=1), old two/three-cycle aliases, D_2=inversion-count,
and conditional-on-length constants are excluded. Non-scalar tasks are outside
this scalar experiment. Registry tasks omitted for scope are not called aliases
or unrelated. Exclusions and their reasons are explicitly listed in the config.

## Retained relationships and interpretation

`target_relationships.csv` distinguishes same-input reconstruction from transformed
queries and incomplete histogram availability. In particular, alpha at k=16 can
reconstruct cycle_count=even_cycle_count+odd_cycle_count and fixed_points via
either n-exceedances-deficiencies or even+odd-nontrivial_cycle_count. The latter
is a symbolic composition of two previously certified partition identities.
Beta does not contain those complete same-input query sets, but it retains
fixed_points(p)=anti_fixed_points(complement(p)). Both chains eventually contain
LDS(complement(p))=LIS(p) and global_descents(complement(p))+1=components(p).
Some relations require labels at different transformed inputs; the metadata
does not imply these inputs are jointly presented during training. Legal task
relationships are not automatically input leakage. Additional joint derivations
are unclassified, and no mechanism claim is made about a trained model.

`relation_inventory.json` reuses the platform's full 74-entry inventory with its
proof, finite-check and historical observation distinctions. Its old empirical
references are retained as metadata and are not read to select the lists.
`relation_reuse.json` records byte-hash reuse of existing evidence: no unchanged
historical formula is rechecked. Source-source links, including inverse
invariance, monotone subsequence complement/reversal and block identities, remain
available in that inventory; missing edges mean unclassified.

## New validation, without model selection

`inequivalence_audit.json` records exact witnesses at both n=10 and n=12 showing
each selected source is nonconstant and no source has the same answer function
as any other source or target. Fixtures are identity/reversal, shifts, adjacent
swaps and 1,024 seeded uniform proposals per length. They are not train/dev/test
data. One unequal answer disproves identical functions; this is not a proof of
independence, matched difficulty or absence of joint derivability.
`encoding_bounds.csv` checks exact output bounds and complete sequence encoding.
The I_3 and D_3 sources include multi-token integer answers up to 220 at n=12;
eight generation tokens are sufficient for all selected scalar outputs.

Increasing k also changes composition, available relations, answer-length
distribution and exposure per task. Results establish only finite curves for
these two lists, not a pure causal effect of count or a scaling law. The chains
share models, targets and baselines and do not create extra independent repeats.

Recreate into a fresh output directory from the repository root:

```powershell
.venv/Scripts/python.exe experiments/task_count_v1/design.py --out experiments/task_count_v1_reproduction/design
```

Reusing an existing output verifies its frozen configuration, source and artifact
hashes. It never silently changes the design.
