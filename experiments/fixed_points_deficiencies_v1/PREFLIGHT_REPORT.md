# Fixed-points source-replacement confirmation: preflight

Status: mathematical and historical-protocol preflight completed; the formal
training protocol is not frozen and no training or model evaluation has started.
The received request ends midway through the full-composition relation-audit
requirement. Budget, repeats, lengths, pool sizes, adaptation rules and the
statistical plan await the missing continuation; historical values are not
silently adopted.

The starting HEAD is `1ea08df6b867e719bdbc4110d971fa0161304a39` on
`experiments/relation-readout-v1`, with a clean worktree. This includes the
later relation-readout study and all previous publication/experiment work.
The new branch is `experiments/fixed-points-deficiencies-v1`.
Only this new experiment directory is changed. Repository content is English.

## Specified task sets and historical correspondence

The target is `fixed_points`. Source slots 0,1,2 are `descents`, `successions`,
and `left_to_right_maxima`; slot 3 is P=`peaks`, D=`deficiencies`,
T=`anti_fixed_points`, or U=`right_to_left_maxima`. A is target-only learning
from random initialization. These choices come from the user and will not be
selected using new transfer performance.

P and D match v5's `fixed_points/g0_v3/C` and
`fixed_points/g2_deficiencies/C` task sets. Their old slot order was
`[successions, descents, peaks-or-deficiencies, left_to_right_maxima]`.
The new order therefore permutes old slots as `[1,0,3,2]`; it is not an exact
replay of the historical sampler. T is neither old B set: old B omitted
`successions` and retained `peaks` or `deficiencies`. U remains the specified
comparison regardless of its future performance.

See [COMPOSITIONS.csv](preflight_audit/COMPOSITIONS.csv) and the unchanged
[v5 configuration](../pilot_v5_confirmation/artifacts/config.json).

## Definitions and boundaries

The platform dispatches these functions to unchanged
[`math_ops.py`](../permworld_pilot/vendor/neurips_permutations/math_ops.py).
Inputs are one-based permutations of length 1 through 30; these tasks have no
parameters or second operand. Let positions i run from 1 to n.

| Task | Exact definition | Range or support |
| --- | --- | --- |
| fixed_points | Count p_i=i | 0 through n, excluding n-1 for n>=2; n=1 gives 1 |
| deficiencies | Count p_i<i | 0 through n-1 |
| anti_fixed_points | Count p_i=n+1-i | Same attainable support as fixed_points |
| descents | Count p_i>p_(i+1) | 0 through n-1 |
| successions | Count p_(i+1)-p_i=+1; not absolute difference | 0 through n-1 |
| peaks | Interior indices with p_(i-1)<p_i>p_(i+1); endpoints excluded | 0 through floor((n-1)/2) |
| left_to_right_maxima | Count strict prefix records, including first position | 1 through n |
| right_to_left_maxima | Count strict suffix records, including last position | 1 through n |

At n=1 the adjacency/peak/deficiency counts are zero, both record counts and
both fixed/anti-fixed counts are one. Registry-safe bounds should not be
misread as exact attainable support.

## Three types of relation claims

Let I, R and C denote inverse, reversal and value complement. These are input
queries, not extra independent observations. FP, E, D, AF, L and U denote fixed
points, exceedances, deficiencies, anti-fixed points, prefix and suffix maxima.

| Type | Statement | Evidence or limitation |
| --- | --- | --- |
| Proved answer identity | FP(p)+E(p)+D(p)=n | Every position is equal to, above or below its index; v5 `position_partition`. |
| Proved transformed identity | E(p)=D(Ip), hence FP(p)=n-D(p)-D(Ip) | Inversion transposes permutation graph points; v5 `inverse_exceedance_deficiency` and `fixed_from_two_deficiency_queries`. D(p) alone does not provide the second answer. |
| Proved transformed identity | FP(p)=AF(Cp)=AF(Rp) | Complement or reversal reindexes the anti-diagonal equality; v5 `fixed_complement` and `fixed_reverse`. |
| Proved candidate/common relation | U(p)=L(Rp); U(Ip)=U(p); U(Cp)=RTLmin(p) | v4 record-relation definitions/closure. Reversal exchanges scan direction; graph transposition preserves north-east records; complement exchanges maxima and minima. U is explicitly related to a common source. |
| Proved common-source bound | successions(p)+descents(p)<=n-1 | Successions are ascents; v5 `successions_descents_bound`. |
| Proved transformation relations | descents(Rp)=descents(Cp)=n-1-descents(p); peaks(Rp)=peaks(p); successions(Ip)=successions(CRp)=successions(p) | Existing v5 certificates; these do not determine FP. |
| Proved combination bounds | L<=n-descents; U<=descents+1; L+U<=n+1 | Every prefix record after the first requires an ascent; every suffix record before the last requires a descent. Newly checked on all permutations through n=6. |
| Available common subcomputation | FP/D can compare a value with its position; FP/AF can compare with direct or reversed positional references | Algorithmic possibility, not a claim about a trained model. T does not have only a transform relation with no shared computation. |
| Available common subcomputation | Peaks/descents can share adjacent-comparison bits; L/U can use directional running maxima; tasks can share counting reductions | Possible implementations. Their presence does not predict which transfer contrast will be positive. |
| Learned-mechanism hypothesis | A trained model reuses positional comparisons, executes a registered formula, or benefits because of that formula | Untested in this preflight; source accuracy and transfer accuracy alone would not prove it. |

The existing [v5 inventory](../pilot_v5_confirmation/mathematics/audit/RELATION_INVENTORY.csv)
and [v4 record-relation graph](../pilot_v4_backgrounds/catalog/RELATION_GRAPH.csv)
are retained and bound by hashes. No original full certificate enumeration was
rerun. Reversal is a bijection, so U and L also have the same answer distribution
under uniform permutations; this does not make their same-input labels identical.
No difficulty, answer-distribution or redundancy match between U and D is assumed.

## Complete-source answer sufficiency

| Condition | Four answers on p, plus n | Four tasks on every ordered I/R/C transform, plus n |
| --- | --- | --- |
| P | Insufficient: explicit n=10 collision | Insufficient: inherited full-query collision |
| D | Insufficient: explicit n=10 collision | Sufficient via D(p) and D(Ip) |
| T | Insufficient: explicit n=10 collision | Sufficient via AF(Cp) or AF(Rp) |
| U | Insufficient: explicit n=10 collision | Insufficient: inherited P collision plus U=L after reversal |

These are statements about specified scalar answer vectors. The full raw
permutation is not included in those vectors; FP is of course computable from
that raw input. Counterexamples disprove a map on all S_10, without asserting
the same finite-length result at every other n or mathematical independence.
The allowed transformations form a closed eight-element group, so U supplies
no additional information beyond L when *all eight* transformed L queries are
already available. This redundancy statement does not apply to L(p) alone.

The small audit re-evaluated the historical collision inputs using the v5
independent oracle and a separate suffix-record oracle. It saved six explicit
collisions in [COUNTEREXAMPLES.csv](preflight_audit/COUNTEREXAMPLES.csv), with
160 source-label cross-checks. The new suffix-record oracle and combination
bounds passed on all 873 permutations at lengths 1 through 6.
[AUDIT.json](preflight_audit/AUDIT.json) records hashes and actual execution time.
These are mathematical fixtures, not research training or model-test samples.

## Training and isolation requirements identified

The historical protocols are materially different. V5 uses n=10, vocabulary
188, 4,800 source updates, three adaptation learning rates and first-answer-token
accuracy with a separate teacher-forced EOS check. Task-count v2 uses n=10,12,
vocabulary 203, token-budget scheduling, one adaptation learning rate and greedy
complete-answer/EOS accuracy. Both use 800 adaptation updates, but that does not
make their protocols interchangeable. No choice between them is frozen here.

The old task-count history helper includes pilots v1-v5, platform data/examples
and task-count v1, but omits the subsequent task-count v2 and relation-readout
inputs. The new generator must verify and include their manifests and pools in
historical orbit exclusion, then recompute capacity using the completed request.
It must not invoke old output-writing generation/freeze stages. New random
initializations cannot be replaced by historical trained checkpoints.

All previous failed confirmation, negative transfer and explanatory limits
remain unchanged. This study's hypothesis is motivated by historical results,
and its future new-repeat results must be reported separately.

## Reproduce this preflight

From the repository with the existing platform environment:

```powershell
.venv/Scripts/python.exe -B experiments/fixed_points_deficiencies_v1/preflight.py --out ../fixed-points-preflight-rerun
```

The output directory must not exist. This command imports no model, opens no
checkpoint, generates no research split and starts no training. Complete the
missing request before freezing or executing the formal experiment.
