# Frozen task-count by budget design

This is a follow-up hypothesis motivated by v1 exploration, frozen before v2
transfer outcomes. The primary target is LIS length; fixed points is auxiliary.
All four v1 held-out task IDs remain excluded from every source set and are reset
at adaptation. The exact v1 chains are retained only at k=2 and k=16. Each source
set is crossed with 4M and 32M nonpadding sequence tokens. There are three new
paired repeat units, 24 pretrained endpoints and 54 target fits including six
shared scratch baselines.

`configuration.json` records the hypotheses and complete ordered chains.
`task_sets.csv` records the memberships. Both k2 sets contain one cycle and one
subsequence instance; k16 sets contain four cycle, three subsequence, four block
and five legacy instances. The pool has 24 instances and 22 templates. Each
individual source set has k distinct templates. There is one encoding.

`v1_evidence_reuse.json` verifies the v1 frozen design and all original evidence
by SHA256. Existing inequivalence witnesses and output-encoding bounds are reused,
not rerun. `target_relationships.csv` retains all four original held-out targets
at the selected k values. `relation_inventory.json` retains the proof, finite
verification, empirical association and unchecked distinctions from the platform.
No missing relation is relabeled as mathematical unrelatedness. Legal transformed
input and joint answer relations are not automatically data leakage.

Alpha k16 can derive fixed points from n minus exceedances minus deficiencies or
even plus odd minus nontrivial cycle counts. Beta retains the transformed-input
anti-fixed-point relation. LDS relates to LIS under complement or reversal, so
its presence is retained and annotated. Having source labels in a set does not
show that the model learned or used a formula.

The primary contrast is the within-repeat difference between the 32M and 4M
k16-minus-k2 effects. Baseline gains, endpoints and source learning are auxiliary.
Chains, shared baselines and target reuse are not extra independent repeats.
Three-repeat sample SD is descriptive; differences are computed before their
uncertainty is summarized. This study cannot establish a scaling law or a pure
composition-free count effect.

Recreate into a fresh directory:

```powershell
.venv/Scripts/python.exe experiments/task_count_v2/design.py --out experiments/task_count_v2_reproduction/design
```

An existing frozen directory is only verified, never silently updated.
