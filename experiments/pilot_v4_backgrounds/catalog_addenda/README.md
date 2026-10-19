# Post-freeze definition-level relation review addendum

This addendum was completed after v4 compositions, configuration, and the original catalog were frozen, during formal pretraining. It reads only task definitions, the original relation graph, and the frozen protocol, without transfer results, checkpoints, or training performance. It changes no compositions, data, budgets, selection rules, or frozen catalog. The original catalog explicitly closes only declared relations; this addendum describes uncovered joint relations, self-symmetries, and algorithmic structure. Missing graph edges do not mean mathematical unrelatedness.

## Multi-input joint relations that must be disclosed for g2

Let `F=fixed_points`,`AF=anti_fixed_points`,`D=deficiencies`, and`E=exceedances`. The position partition gives `F(p)+D(p)+E(p)=n`. Inversion maps each `p_i>i` to a below-diagonal position in the inverse permutation, so `E(p)=D(I(p))`. Combining them:

```text
F(p) = n - D(p) - D(I(p)).
```

Since `AF(p)=F(C(p))`, substitution gives:

```text
AF(p) = n - D(C(p)) - D(I(C(p)))
      = n - D(C(p)) - D(R(I(p))).
```

I denotes inversion, R one-line reversal, and C complement. The frozen platform's bit order executes I, then R, then C; thus `100=I`,`001=C`,`110=R∘I=I∘C`, and`111=C∘R∘I`. The formulas hold for all permutations with n≥1.

Thus in fixed_points g2, background deficiencies has exact relations to both the target and B candidate through joint use of two transformed inputs. The original table's "missing exceedances" correctly describes named tasks on the same input, but does not exclude joint reconstruction across transformed inputs. C in g2 also contains deficiencies and therefore is not a control entirely without target relations.

**One deficiencies label on one input does not determine the target.** At n=10,`p=(1,3,2,4,5,6,7,8,9,10)` and`q=(2,3,1,4,5,6,7,8,9,10)` both have D=1, but F is 8 and 7 respectively. Formal training supplies no such paired queries, joint answers, or relation constraints. Within-pool orbit deduplication also does not retain distinct transformation-equivalent inputs simultaneously (self-symmetric permutations can map back to themselves). This is an implicit mathematical relation, not single-label leakage or proof that the model used it.

## How the six background groups change computational structure

B candidates for both targets retain identity-answer relations under complement/reverse in all backgrounds. g1 removes descents and replaces it with recoils; it does not add a coexisting descents/recoils source pair. Their population answer distributions match, but adjacent-position descents and inverse-position descents of adjacent values require different one-line computations. Existing early single-task learning differences are in frozen `BACKGROUND_MATCHING.csv`.

For LDR, let `s_i=1[p_i>p_(i+1)]` and`d=sum(s_i)`. descents counts negative comparisons; LDR is one plus the longest consecutive 1 segment; LIR is one plus the longest consecutive 0 segment; peaks counts local 0-to 1 transitions. These tasks share the local comparison sequence, but one scalar answer does not determine the others. Cutting at ascending or descending edges yields n−d decreasing runs or d+1 increasing runs. The longest segment is at least the average segment length, so:

```text
ceil(n/(n-d)) <= LDR(p) <= d+1
ceil(n/(d+1)) <= LIR(p) <= n-d.
```

g1 removes this same-input descents statistic while retaining peaks. g2 retains descents but replaces peaks with a position-inequality count. These changes affect LDR's local computational structure differently and cannot be summarized only by relation-edge counts.

successions is a subset of local ascending edges, so `successions(p)+descents(p)<=n−1`. Since successions is invariant under inversion,`successions(p)+recoils(p)<=n−1` also holds. This common-source relation has corresponding forms in g0/g2 and g1; replacement does not remove all local-structure links. For fixed_points, the g2 position-comparison structure and the multi-input joint identities above especially require disclosure.

## Three self-symmetries absent from the original graph

The following follow directly from definitions and passed the same CPU checks; frozen `RELATION_GRAPH.csv` was not expanded in place:

| Relation | Definition-level justification |
| --- | --- |
| `peaks(R(p))=peaks(p)` | Reversal swaps a peak's left/right neighbors; its center remains larger than both. |
| `successions(C(R(p)))=successions(p)` | Combined reversal/complement maps each adjacent `(a,a+1)` to another adjacent pair with the same directed difference 1. |
| `deficiencies(C(R(I(p))))=deficiencies(p)` | RC swaps below-/above-diagonal positions; inversion then supplies the D/E bijection. |

These edges mainly complete existing nodes' self-symmetries; they are not new independent tasks or training evidence. They also distinguish complete closure of declared relations from complete classification of all mathematical relations.

## Supported interpretations and limitations

The design tests whether B−C changes with replacement of one common task for two post-selected exploratory targets, three frozen backgrounds, and a limited number of paired blocks. Source curves describe which common tasks improve or worsen together.

It does not isolate the causal effect of mathematical relatedness itself or prove gradient conflict, representation sharing, or identity use from correlations. g1 preserves population marginal distributions but changes learning speed and computational structure. g2 changes local/position structure, answer distributions, learning dynamics, and fixed_points multi-input relations together. Both targets share complement-type candidate relations, data/initialization blocks, and many sources; they are not independently sampled task-family examples. Positive or negative outcomes establish neither sample-efficiency curves nor task-count scaling.

Future task libraries should prioritize relation categories rather than simply more tasks: invertible input transforms with identity/affine answer maps; equal distributions without a current direct transform correspondence; multi-source/multi-input joint identities; one-way recoverability and inequalities; shared local computational structure. Existing cycle_count/parity/nontrivial-cycle, position-partition, and local-run families offer candidates, but new targets, paired-input interventions, or training require separate freezing and subsequent authorization. This addendum triggers no new experiments.

## Verification and reproduction

The audit exhausts S1…S7(5913 permutations) and uses fixed seed=2026101806 to generate 2048 n=10 definition-check fixtures. These are not formal test inputs and are not written into experiment pools. Nine identities/bounds each receive 7961 checks, totaling 71649; see `RELATION_ADDENDUM_AUDIT.json`. General validity rests on the definition-level proofs above; finite checks validate implementations.

```powershell
.venv/Scripts/python.exe -B experiments/pilot_v4_backgrounds/catalog_addenda/audit_postfreeze_relations.py
# Existing audit results cannot be overwritten; use a new file for rechecking:
.venv/Scripts/python.exe -B experiments/pilot_v4_backgrounds/catalog_addenda/audit_postfreeze_relations.py --out experiments/pilot_v4_backgrounds/catalog_addenda/RECHECK.json
```

The audit records before/after SHA256 values for all original catalog files, TASK_CATALOG, DATA_PLATFORM, design, and FROZEN_PROTOCOL, and asserts no changes. It also records addendum code/documentation and original mathematical-implementation hashes. The script imports no Torch, loads no model, and reads no transfer results.
