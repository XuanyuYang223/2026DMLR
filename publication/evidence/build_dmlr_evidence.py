"""Build a read-only, hash-bound synthesis of archived DMLR evidence.

No training, inference, historical rewriting, or statistical pooling across
stages is performed. Run from any directory with the repository's Python.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
P1 = "experiments/permworld_pilot/runs/pilot_v1/"
P2 = "experiments/pilot_v2_diagnostics/"
P3 = "experiments/pilot_v3_multitarget/"
P4 = "experiments/pilot_v4_backgrounds/"
P5 = "experiments/pilot_v5_confirmation/"
PL = "experiments/platform_v01/"
T1 = "experiments/task_count_v1/"
T2 = "experiments/task_count_v2/"


def rows(path):
    with (ROOT / path).open(encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def close(a, b):
    assert abs(float(a) - float(b)) < 1e-12, (a, b)


def mean(values):
    return statistics.mean(map(float, values))


def raw_checks():
    checks = []
    original = rows(P1 + "results_all.csv")
    for row in rows(P1 + "summary.csv"):
        n = row["n"]
        means = {c: mean(r["test_accuracy"] for r in original if r["n"] == n and r["condition"] == c)
                 for c in "ABC"}
        for c, value in means.items():
            close(row[c], value)
        for key, a, b in (("B_minus_A", "B", "A"), ("C_minus_A", "C", "A"), ("B_minus_C", "B", "C")):
            close(row[key], means[a] - means[b])
    checks.append({"stage": "pilot_v1", "check": "all three sample-size A/B/C means and differences from 18 raw rows", "rows": len(original)})

    original = rows(P2 + "artifacts/protocol_comparison.csv")
    for row in rows(P2 + "artifacts/protocol_comparison_mean.csv"):
        if row["protocol"] != "dev_selected":
            continue
        matches = [r for r in original if r["protocol"] == "dev_selected" and r["condition"] == row["condition"]
                   and r["pretrain_updates"] == row["actual_pretrain_updates"]]
        assert len(matches) == 2
        close(row["test_accuracy_mean"], mean(r["test_accuracy"] for r in matches))
    checks.append({"stage": "pilot_v2", "check": "all development-selected condition means from original protocol rows"})

    selected = rows(P3 + "artifacts/results_selected.csv")
    lookup = {(r["target"], r["seed"], r["condition"]): float(r["test_accuracy"]) for r in selected}
    assert len(lookup) == 36
    for row in rows(P3 + "artifacts/paired_deltas.csv"):
        a, b = row["contrast"].split("-")
        close(row["test_accuracy_delta"], lookup[row["target"], row["seed"], a] - lookup[row["target"], row["seed"], b])
    checks.append({"stage": "pilot_v3", "check": "all 36 paired target contrasts from 36 unique selected models"})

    for stage, raw_path, summary_path in (
        ("pilot_v4_recomputed_before_v5", P5 + "exploratory/exploratory_paired_differences.csv", P5 + "exploratory/exploratory_summary.csv"),
        ("pilot_v5", P5 + "artifacts/confirmation_paired_differences.csv", P5 + "artifacts/confirmation_summary.csv")):
        source = rows(raw_path)
        for row in source:
            for contrast, a, b in (("B_minus_A", "B", "A"), ("C_minus_A", "C", "A"), ("B_minus_C", "B", "C")):
                old_a, new_a = (float(row[f"{a}_original"]), float(row[f"{a}_new"]))
                old_b = float(row["A_shared"] if b == "A" else row[f"{b}_original"])
                new_b = float(row["A_shared"] if b == "A" else row[f"{b}_new"])
                close(row[f"delta_{contrast}"], (new_a - new_b) - (old_a - old_b))
        for summary in rows(summary_path):
            values = [float(r[summary["metric"]]) for r in source if r["target"] == summary["target"] and
                      r["original_background"] == summary["original_background"] and r["new_background"] == summary["new_background"]]
            assert len(values) == int(summary["n"])
            close(summary["mean"], mean(values))
            close(summary["sample_sd"], statistics.stdev(values))
            if stage == "pilot_v5" and summary["primary"] == "True":
                se = statistics.stdev(values) / len(values)**.5
                for field, critical, sign in (("ci95_lower", "t_95_critical", -1), ("ci95_upper", "t_95_critical", 1),
                                              ("family95_lower", "t_family_critical", -1), ("family95_upper", "t_family_critical", 1)):
                    close(summary[field], mean(values) + sign * float(summary[critical]) * se)
        checks.append({"stage": stage, "check": "all per-repeat background differences, means and sample SDs; v5 primary paired-t and simultaneous intervals", "raw_rows": len(source)})

    calibration = rows(PL + "calibration/results_by_seed.csv")
    for summary in rows(PL + "calibration/task_summary.csv"):
        values = [r for r in calibration if r["task_id"] == summary["task_id"] and r["params"] == summary["params"]
                  and r["length"] == summary["length"] and r["checkpoint"] == "endpoint" and r["split"] == "dev"]
        assert len(values) == 2
        close(summary["endpoint_dev_accuracy_mean"], mean(r["accuracy"] for r in values))
        close(summary["majority_dev_accuracy_mean"], mean(r["majority_accuracy"] for r in values))
    checks.append({"stage": "platform_v01", "check": "all 24 task-by-length endpoint development means and majority baselines from two raw seeds"})

    for stage, directory in (("task_count_v1", T1), ("task_count_v2", T2)):
        raw = rows(directory + "raw_test_results.csv")
        lookup = {(r["run_id"], r["checkpoint"], r["length"]): float(r["accuracy"]) for r in raw}
        assert len(lookup) == len(raw)
        if stage == "task_count_v1":
            for row in rows(directory + "paired_k_contrasts.csv"):
                close(row["paired_accuracy_change_pp"], 100 * (lookup[row["to_run_id"], row["checkpoint"], row["length"]]
                      - lookup[row["from_run_id"], row["checkpoint"], row["length"]]))
        else:
            for row in rows(directory + "paired_budget_interactions.csv"):
                values = {cell: lookup[row[cell + "_run_id"], row["checkpoint"], row["length"]]
                          for cell in ("k2_b4000000", "k2_b32000000", "k16_b4000000", "k16_b32000000")}
                d4 = 100 * (values["k16_b4000000"] - values["k2_b4000000"])
                d32 = 100 * (values["k16_b32000000"] - values["k2_b32000000"])
                close(row["d4_k16_minus_k2_pp"], d4)
                close(row["d32_k16_minus_k2_pp"], d32)
                close(row["paired_budget_interaction_pp"], d32 - d4)
        checks.append({"stage": stage, "check": "all archived paired count/budget contrasts reconstructed directly from unique raw test rows", "raw_rows": len(raw)})
    return checks


def build_claims():
    claims = []

    def add(cid, stage, status, claim, units, scope, anchor, artifacts, support, limits, paper_use):
        assert len(set(artifacts)) == len(artifacts)
        claims.append({"claim_id": cid, "stage": stage, "evidence_status": status, "claim": claim,
            "independent_units": units, "protocol_scope": scope, "quantitative_anchor": anchor,
            "artifact_paths": json.dumps(artifacts), "artifact_sha256": json.dumps({p: sha(p) for p in artifacts}, sort_keys=True),
            "support": support, "limitations": limits, "suggested_paper_use": paper_use})

    add("D01", "pilot_v1", "exploratory", "The original held-out fixed-point pilot did not show uniformly beneficial related-source pretraining.",
        "Two paired units (17,42); sample sizes and A/B/C share their unit's inputs/initialization.",
        "n=10; 188-token legacy evaluator; one target; 64/256/1024 target samples; fixed 800 adaptation updates.",
        "At 256 samples: A 84.01%, B 68.26%, C 75.51%; B-A=-15.75 pp, C-A=-8.50 pp. At 1024: A 98.85%, B 98.36%, C 97.92%.",
        [P1+"results_all.csv",P1+"summary.csv","PILOT_REPORT.md"],
        "The raw two-seed sample-size curves retain negative and near-zero average gains.",
        "One target and two seeds; a negative average does not mean every seed is negative. Fixed update budget causes unequal reuse as sample count changes.",
        "Motivation and exploratory sample-size appendix; not a general sample-efficiency law.")
    add("D02", "pilot_v2", "exploratory_diagnostic", "Development selection and the historical doubled source budget did not remove mean negative transfer.",
        "Same two historical units 17/42; no new independent repeats.", "fixed_points, 256 samples; equal three-LR adaptation searches; new isolated test; source 2400 versus 4800 steps.",
        "Development-selected 2400: B-A=-17.26 pp, C-A=-8.74 pp, B-C=-8.52 pp; 4800: B-A=-10.55 pp, C-A=-12.87 pp, B-C=+2.32 pp.",
        [P2+"artifacts/protocol_comparison.csv",P2+"artifacts/protocol_comparison_mean.csv",P2+"DIAGNOSTIC_REPORT.md"],
        "Failure to load/reset weights is not supported by engineering audits; negative means persist under the selected protocol.",
        "Historical continuation uses low terminal LR and is not the fresh relative-budget comparison in task-count v2; reused development inputs and hypothesis remain exploratory.",
        "Protocol diagnostic; distinguish clearly from task-count v2's budget intervention.")
    add("D03", "pilot_v2", "exploratory_diagnostic", "The nominally matched candidate pair differed markedly in learnability under multitask training.",
        "Two paired source-model units; source records within units are not independent experiments.", "anti_fixed_points versus successions, identical n=10 source inputs and 25% example shares.",
        "Mean source development accuracy at 2400: 77.98% versus 45.12%; at 4800: 89.31% versus 51.86%; source-histogram TV 0.0443/0.0520.",
        [P2+"artifacts/source_learning_table.csv",P2+"artifacts/replacement_matching_summary.csv",P2+"DIAGNOSTIC_REPORT.md"],
        "Similar marginal answer distributions do not ensure similar learning curves.", "These are mixture-learning differences, not intrinsic difficulty or a causal isolation of relation effects.", "Design limitation motivating source calibration.")
    add("D04", "pilot_v3", "exploratory", "Transfer varies across held-out targets, and the candidate replacement contrast changes sign across seeds for all four targets.",
        "Three paired units 17/42/73; first two fixed-point units reuse historical training; four targets are not 12 independent repeats.",
        "Four targets, 256 samples; four sources; 4800 source steps; equal three-LR development selection; new tests.",
        "Mean B-A: fixed_points -10.12 pp; exceedances +2.59 pp; left_to_right_minima -1.53 pp; longest_decreasing_run +11.20 pp. B-C signs are mixed for all four targets.",
        [P3+"artifacts/results_selected.csv",P3+"artifacts/paired_deltas.csv",P3+"PILOT_V3_REPORT.md"],
        "The fixed-point negative effect is not a universal outcome across the tested mathematical targets.",
        "LDR A/C are near majority; eligibility at 8192 single-task examples does not establish learnability at 256. All tested direct candidate relations have identity answer maps.",
        "Main broad-target exploratory panel; retain all four targets.")
    add("D05", "pilot_v3", "exploratory_diagnostic", "Single-task and mixture references show both degradation and improvement, with exposure confounding.",
        "Three units; 84 deduplicated seed-task-model-state records share inputs/tasks and are not 84 independent repeats.",
        "Single-task 4800 and mixture 4800 share total updates, but single-task exposure is four times larger; single-task 1200 matches exposure only.",
        "84 deduplicated source records: 60 declines >=5 pp, 6 improvements >=5 pp, 18 smaller gaps; successions seed 73 improvement +43.55 pp retained.",
        [P3+"artifacts/source_learning_effects_unique.csv",P3+"artifacts/single_vs_multi_sources.csv",P3+"PILOT_V3_REPORT.md"],
        "Composition accompanies different source learning; low performance cannot be summarized only by relatedness labels.",
        "The references do not isolate interference or synergy because exposure, alternation and LR histories differ.", "Source-learning diagnostic and limitation.")
    add("D06", "pilot_v4", "exploratory", "The same candidate-source replacement has different observed effects across background sets.",
        "Three paired blocks 17/42/73; reuse from v3; 42 unique selected models, not 54 independent display rows.",
        "Two targets selected from v3, three backgrounds, 256 samples; fixed source candidates and paired slots.",
        "FP deficiencies-v3 interaction +4.36 pp (seeds +1.42,+0.34,+11.33); LDR recoils-v3 -12.00 pp (seeds -20.07,+2.00,-17.92).",
        [P4+"artifacts/results_selected.csv",P4+"artifacts/background_interactions.csv",P5+"exploratory/exploratory_paired_differences.csv"],
        "Observed background-dependent variation motivated the two frozen v5 hypotheses.",
        "Case/background selection is exploratory; v3/v4 reused units must not be counted as independent replications. Must present v5 follow-up beside these effects.", "Exploration-to-confirmation narrative, never standalone confirmation.")
    add("D07", "pilot_v5", "confirmatory_nonreplication", "The prespecified positive fixed-point background interaction did not reproduce in mean direction.",
        "Five new paired units 101/211/307/401/503; old three exploratory units are not pooled.", "Primary difference (B-C)_deficiencies-(B-C)_original, n=10, 256 samples, frozen budget and selection.",
        "Mean -3.02 pp, SD 15.75; 2/5 direction matches; 95% paired-t CI [-22.57,+16.54]; two-contrast simultaneous CI [-27.63,+21.60].",
        [P5+"artifacts/confirmation_paired_differences.csv",P5+"artifacts/confirmation_summary.csv",P5+"stats_plan.json"],
        "The exploration's positive replacement interaction is not corroborated by the new point estimate.",
        "Wide intervals include both directions; failure of directional replication is not proof of zero background effect. Five units and finite-pool dependence limit interval interpretation.", "Mandatory main-text negative confirmation result.")
    add("D08", "pilot_v5", "confirmatory_uncertain", "The LDR confirmation point estimate matches the exploratory direction but does not establish a stable interaction.",
        "Five new paired units, shared across targets; not 10 independent target replicates.", "Primary difference (B-C)_recoils-(B-C)_original with a two-contrast interval family.",
        "Mean -13.15 pp, SD 12.67; 4/5 direction matches; 95% CI [-28.89,+2.58]; simultaneous CI [-32.97,+6.66].",
        [P5+"artifacts/confirmation_paired_differences.csv",P5+"artifacts/confirmation_summary.csv",P5+"PILOT_V5_REPORT.md"],
        "A large negative point estimate remains uncertain under the prespecified paired analysis.",
        "Both interval types include zero; LDR A 53.56% and recoils B/C 53.50%/52.38% are near 53.67% majority. No claim of reliable universal background dependence.", "Mandatory uncertainty alongside v5 fixed-point non-replication.")
    add("D09", "pilot_v5_math", "resource_verified_relations", "Same-input joint derivability and transformed-input joint derivability are distinct design facts.",
        "Deterministic mathematical audits, not learning replicates.", "Position partition and inverse identity; four-source scalar query models audited for 12 v4 groups.",
        "fixed_points+exceedances+deficiencies=n; exceedances(p)=deficiencies(p^-1). All 12 same-input groups have counterexamples; 7/12 groups recover targets under 32 transformed queries, and five have insufficiency witnesses.",
        [P5+"mathematics/MATH_AUDIT.json",P5+"mathematics/audit/TARGET_JOINT_DERIVABILITY.csv",P5+"mathematics/audit/RELATION_INVENTORY.csv"],
        "Both B/C fixed-point deficiencies groups can recover FP from D(p),D(p^-1), but the position partition alone cannot recover FP from one D label.",
        "Valid relations are not instance leakage; no explicit paired-query training or mechanism demonstration was performed. Counterexamples restrict only the declared query model.", "Relation-schema methodological contribution.")
    add("R01", "platform_v0.1", "resource", "The unified platform distinguishes endpoints, canonical templates, instances and encoding variants.",
        "Deterministic registry inventory, not statistical replicates.", "Legacy unary/binary interfaces remain callable; the canonical dataset is unary only.",
        "48 legacy + 6 new = 54 endpoints; 48 canonical mathematical templates; 62 declared instances; 56 materialized unary non-alias instances; one input encoding.",
        [PL+"registry/TASK_COUNTS.json",PL+"registry/TASK_REGISTRY.json",PL+"registry/TASK_INSTANCES.csv"],
        "Machine-readable definitions, parameters, ranges, complexities, encoding and relation status are available under one interface.",
        "48 templates are not 48 mathematically independent capabilities; binary pools are not generated or certified by this release.", "Primary resource contribution with explicit counting conventions.")
    add("R02", "platform_v0.1", "resource", "The three focus families provide differentiated tasks while explicitly retaining constants and aliases.",
        "Deterministic function classes under named relations.", "Cycle decomposition, global subsequences and permutation blocks at n=8/10/12 and default parameters.",
        "Focus: 16 templates / 24 instances; 22 nonconstant instances; 23 same-input classes; 20 registered answer-preserving transformation classes (19 nonconstant).",
        [PL+"registry/TASK_COUNTS.json",PL+"mathematics/EQUIVALENCE_AUDIT.json"],
        "Cycle family: 6 templates / 8 instances; subsequences: 5/9; blocks: 5/7. I1=D1=n constants remain visible.",
        "Other lengths/parameters can be degenerate; known transformation classes do not classify arbitrary answer maps or all joint relations.", "Resource counts and protection against count inflation.")
    add("R03", "platform_v0.1", "resource_verified_correctness", "Exact generators have independent algorithm checks and explicit relation evidence.",
        "6105 deterministic fixtures, not 6105 training or statistical units.", "Exhaustive S1...S7 plus 192 larger-n fixtures; historical v5 certificates reused.",
        "272741 independent label comparisons; 445581 new formula/bound pointwise checks; 45 new relations + 29 historical relations reused; zero historical formula rechecks.",
        [PL+"mathematics/CERTIFICATE.json",PL+"mathematics/RELATIONS.json",PL+"mathematics/HISTORICAL_REUSE.json"],
        "Independent cycle, subsequence, RSK and block algorithms verify oracle implementation; proof and finite-check status remain separate.",
        "Finite verification is not a general proof; mathematical correctness does not imply model learnability.", "Exact-oracle and relationship-certificate contribution.")
    add("R04", "platform_v0.1", "resource_protocol", "The versioned sequence protocol supports large and structured answers without silent truncation.",
        "Engineering and data audits, not statistical replicates.", "permworld_sequence_v1, one_line_v1, current 203-token vocabulary; 188 legacy IDs retained.",
        "9216 fixed objects / 516096 exact labels; n=30,k=15 integer 155117520 round trip; 32 canonical I3 labels >=100; all focus instances fit a 128-token context.",
        [PL+"PLATFORM_REPORT.md",PL+"dataset/manifest.json",PL+"registry/MODEL_INTERFACE_COMPATIBILITY.csv"],
        "Delimited base-100 integers and nested lists are explicitly encoded; oversized legacy examples are visibly rejected.",
        "The six scalar representatives do not calibrate structured-output learning; legacy and platform accuracies use different protocols.", "Resource and measurement-protocol contribution.")
    add("R05", "platform_v0.1_calibration", "resource_development_calibration", "Six frozen single-task representatives span sharply different attainable learning levels under one budget.",
        "Two paired seed/data units 617/829; six tasks sharing this protocol are not 12 independent replicates.", "2400 updates/task; n=8/10/12 all trained; complete-answer-plus-EOS development accuracy; no inference on reserved tests.",
        "Mean endpoint development: cycle_count 42.84%; longest_cycle 22.53%; LIS 86.26%; I3 count 25.26%; components 99.74%; largest_component 99.02%.",
        [PL+"calibration/results_by_seed.csv",PL+"calibration/task_summary.csv",PL+"calibration/FROZEN_CALIBRATION.json"],
        "All six retained; block majority is 77.08%, so high accuracy needs baseline context; cycle_count at n=12 has development accuracy 35.74% versus 36.72% majority.",
        "Development only: no unseen-length extrapolation, transfer or entire-family ranking; 12 runs are six tasks x two units.", "Single-task resource calibration panel, explicitly labeled development.")
    add("T01", "task_count_v1", "exploratory_frozen_comparison", "At 4M tokens, the tested count curves do not show universally monotonic transfer improvement.",
        "Three new paired units 1103/2207/3301; chains, targets and shared baselines are not extra replicates.", "Two fixed nested chains; k=2/4/8/16; four targets; 256 samples; n=10/12; full answer plus EOS.",
        "24 pretrained models / 108 target fits; LIS k16-k2 mean -5.47 pp alpha / -6.02 pp beta, negative in all three repeats of each chain; all eight cycle-count means are at or below A.",
        [T1+"raw_test_results.csv",T1+"paired_k_contrasts.csv",T1+"REPORT.md"],
        "LIS retains positive mean gains over A while losing benefit as k increases; modest component gains and fixed-point chain/seed variation are preserved.",
        "Changing k changes source composition, relations and exposure; three paired units cannot establish a pure task-count causal effect or scaling law.", "Finite task-count results across all four targets.")
    add("T02", "task_count_v1", "exploratory_diagnostic", "The fixed-point average peak is unstable, and many source endpoints remain underlearned.",
        "Three paired units; 180 source-task/model records are not 180 independent replicates.", "Development-selected primary checkpoint; fixed 4M budget; no result-triggered extension.",
        "FP alpha-k8 gains +26.17,+12.01,-10.94 pp (mean +9.08, SD 18.73); 61/180 source endpoints are at or below their training-derived majority baseline.",
        [T1+"paired_deltas.csv",T1+"source_learning.csv",T1+"REPORT.md"],
        "The mean peak is not a stable optimum; limited source learning motivates a frozen budget follow-up.",
        "Selecting the best k after inspecting outcomes would overstate this exploration; source-accuracy/transfer association does not establish a mechanism.", "Retained negative and uncertain evidence beside the count curve.")
    add("T03", "task_count_v2", "prospectively_frozen_followup", "A larger relative-progress budget reverses the LIS count disadvantage in alpha and reduces it in beta.",
        "Three new paired units 4103/5207/6301; two chains share these units, not six independent interactions.", "Primary LIS; fixed chain composition at each k; 2x2 comparison of k=2/16 and 4M/32M; fresh paired initialization; 256 samples.",
        "Alpha D4=-6.18 pp, D32=+11.75, interaction +17.94 (SD 7.89); beta D4=-5.79, D32=+0.75, interaction +6.54 (SD 5.96); both interactions are positive in all three repeats.",
        [T2+"raw_test_results.csv",T2+"paired_budget_interactions.csv",T2+"budget_interaction_summary.csv",T2+"FROZEN_PROTOCOL.json"],
        "The direction of k16-k2 depends on budget in the prespecified alpha case; beta shrinks the disadvantage without positive D32 in all three repeats.",
        "The hypothesis comes from v1, with limited new repeats and composition coverage; different absolute-token LR paths mean budget and trajectory are not separately identified.", "Primary fixed-composition budget interaction, conditional on the tested design rather than universal.")
    add("T04", "task_count_v2", "prospectively_frozen_auxiliary", "The auxiliary fixed-point budget interaction has opposite signs across source chains.",
        "Three new paired units, shared with LIS; the auxiliary target does not double the independent repeat count.", "Same 2x2 protocol; fixed_points is the auxiliary target.",
        "Alpha interaction +41.41 pp (SD 3.27), beta -9.08 (SD 11.55); alpha-k16-32M test accuracy 88.44%, +23.63 pp over A; beta-k16-32M 43.39%, -21.42 pp over A.",
        [T2+"raw_test_results.csv",T2+"paired_budget_interactions.csv",T2+"budget_interaction_summary.csv"],
        "A larger budget does not generally eliminate negative transfer; seven of eight fixed-point conditions retain negative mean gains.",
        "Relation availability and improved source learning do not prove formula use; the auxiliary outcome must not replace the primary LIS question.", "Auxiliary counterexample to universal budget benefits.")
    add("T05", "task_count_v2", "resource_execution_audit", "The budget comparison increases repeated training on a fixed pool, with audited pairing and budget-specific LR paths.",
        "24 source models and 54 target fits across three repeat units; not 78 independent repetitions.", "12 fresh-initialization budget pairs; the same 8192 source objects per unit; uniform example shares; whole-example token stopping.",
        "431999672 actual pretraining tokens; 82944000 adaptation tokens; 12 pairs verified; 135 formal checkpoints checked; 54 development selections; 324 test CSV rows; zero new test forwards in the audit.",
        [T2+"RUN_AUDIT.json",T2+"audit_budget_pairing.csv",T2+"task_exposures.csv",T2+"learning_rate_schedule.csv"],
        "32M extends the sampled-example prefix but starts from random weights, not 4M weights; per-task coverage and repeated exposure counts are explicit.",
        "Eightfold total tokens are not eightfold independent data or strict per-task exposure matching; absolute learning rates differ from the first update.", "Reproducibility and qualification of causal scope.")
    return claims


def write_markdown(claims, checks):
    lines = ["# DMLR evidence synthesis for the integrated PermWorld paper", "",
        "This document synthesizes archived evidence across the entire DMLR line. It does not run models or pool stages into a larger statistical sample. The accompanying `dmlr_claims.csv` is the claim ledger: each row names its evidential status, unit of repetition, quantitative anchor, limitations, and exact current artifact hashes. `dmlr_recompute_checks.json` records independent arithmetic checks from raw selected results.", "",
        "## Defensible contribution", "",
        "PermWorld turns mathematical source-task composition into an auditable learning experiment: exact task definitions and answer generators, explicitly typed single-task and joint relations, transformation-orbit input isolation, paired source/target training, and source-learning diagnostics. The empirical contribution is conditional evidence that task composition and training budget affect transfer; the archive also establishes limits on that evidence. It is not a theorem that mathematical relatedness improves transfer, a general ranking of task families, or a scaling law.", "",
        "The broad line matters. The original fixed-point pilot exposed negative means; protocol and source diagnostics did not remove them; four-target exploration showed heterogeneous effects; background exploration motivated two fixed hypotheses; the five-unit confirmation did not replicate the fixed-point direction and left the LDR effect uncertain. The platform then made richer task families and full-answer supervision usable, followed by four-target task-count curves and a new fixed-composition budget comparison. A paper centered only on the latest positive LIS interaction would omit both the resource contribution and important negative evidence.", "",
        "## Evidence stages and units", "",
        "| Stage | Status | Actual unit / scope | Critical qualification |", "|---|---|---|---|",
        "| pilot_v1 | Exploratory | 2 paired units; 1 target; 64/256/1024 examples | Fixed-endpoint, legacy single-answer-token accuracy |",
        "| pilot_v2 | Diagnostic exploration | Same 2 units; 256 examples; adaptation selection and source continuation | Reused development data; not a new independent replication |",
        "| pilot_v3 | Multi-target exploration | 3 paired units; 4 targets; 256 examples | Historical weights reused where compatible; targets are not extra units |",
        "| pilot_v4 | Background exploration | Same 3 blocks; 2 selected targets; 3 backgrounds | Shared A and reused weights do not create extra repeats |",
        "| pilot_v5 | Frozen confirmation | 5 new paired units; 2 fixed background contrasts | Report separately from old 3 units; retain non-replication and uncertainty |",
        "| platform v0.1 | Resource and development calibration | Registry/certificates; 6 representatives x 2 units | Calibration has no model test forwards and no transfer experiment |",
        "| task-count v1 | Frozen finite comparison motivated by exploration | 3 new units; 4 targets; 2 chains; k=2/4/8/16 | 24 source models reused for108 target fits; composition changes with k |",
        "| task-count v2 | Prospectively frozen budget follow-up | 3 new units; primary LIS, auxiliary FP; 2 chains; k=2/16; 4M/32M | 24 source models,54 target fits; shared models and baselines are not repetitions |", "",
        "Pilot stages use the legacy 188-token vocabulary and answer-token accuracy at n=10. The platform and task-count stages use a 203-token versioned sequence protocol, per-example answer-plus-EOS mean cross-entropy, and complete generated-answer-plus-EOS accuracy; the task-count lengths are10/12. These protocols and datasets are different. Do not concatenate their accuracy values into a single longitudinal performance curve, interpret rerun checkpoints on new tests as new trained repeats, or sum all stage counts as an independent sample size.", "",
        "## Findings that must survive paper compression", "",
        "1. **Original and multi-target exploration.** At256 examples, pilot_v1 averages A/B/C=84.01/68.26/75.51%. In v3, mean B-A is-10.12pp for fixed_points,+2.59 for exceedances,-1.53 for left_to_right_minima,+11.20 for longest_decreasing_run. Every target's B-C contrast changes sign across the three seeds. Source learning changes with composition; exposure and optimization differences prevent interpreting these as pure interference or synergy.", "",
        "2. **Exploration did not become blanket confirmation.** V4's fixed-point deficiencies interaction was+4.36pp and its LDR recoils interaction-12.00pp. In five new v5 units, FP is-3.02pp with95% paired-t interval[-22.57,+16.54], so the prespecified positive mean direction did not replicate. LDR is-13.15pp with interval[-28.89,+2.58], matching direction but remaining uncertain. Two-contrast simultaneous intervals are wider. Both results must accompany any v4 background-effect figure; they are not evidence of absence of all background effects.", "",
        "3. **Resource contribution has auditable counting and correctness.** The registry retains48 legacy endpoints and adds6, giving54 endpoints and48 canonical mathematical templates. It declares62 instances and materializes56 unary non-alias instances. The three focus families contain16 templates/24 declared instances/22 nonconstant instances at the default lengths and parameters, with one input encoding. These are not independent mathematical capabilities. The certificate records272,741 independent label comparisons and445,581 new relation checks on6,105 fixtures;29 historical relation certificates are reused without rerunning old formulas. Explicit formulas, finite verification, empirical association and unchecked claims remain distinct.", "",
        "4. **Learnability is uneven and measurement-specific.** Fixed-budget two-seed endpoint development means are42.84% cycle_count,22.53% longest_cycle,86.26% LIS,25.26% length-three increasing-subsequence count,99.74% components and99.02% largest_component. Block majority is77.08%; cycle_count atn12 is35.74% against36.72% majority. All lengths occur in training; these are development calibration, not held-out-length generalization or new test results.", "",
        "5. **The finite task-count curve depends on what is trained and for how long.** At4M tokens in task-countv1, LIS k16-k2 is negative in all three repeats of both chains, despite positive mean transfer over A. The four-target curves retain cycle-count nonbenefit, small component gains and unstable fixed-point gains. In the new task-countv2 units, the LIS interaction[(k16-k2)_32M-(k16-k2)_4M] is+17.94pp(alpha) and+6.54pp(beta), positive in all three repeats of each chain; only alpha establishes positive k16-k2 in every32M repeat. Auxiliary FP interactions have opposite signs,+41.41pp and-9.08pp. This is a conditional budget/composition finding, not monotonic universal benefit.", "",
        "6. **A larger budget is not proportionally more independent data.** Both task-countv2 budgets use the same8192 source objects per repeat. More steps can increase a task's coverage of that fixed pool and repeat previously seen task-input pairs. Uniform sample shares are not equal token shares for different answer lengths. Both budgets start from paired random weights with fresh optimizers, but the absolute-token LR trajectories differ. Thus updates, learning-rate trajectory and fixed-pool exposure are not independently identified.", "",
        "## Mathematical relations and mechanism boundaries", "",
        "The exact position partition is fixed_points(p)+exceedances(p)+deficiencies(p)=n. Together with exceedances(p)=deficiencies(p^-1), it gives fixed_points(p)=n-deficiencies(p)-deficiencies(p^-1). The first form needs two same-input answers; the second needs two transformed-input queries. The v5 fixed-point deficiencies groups do not contain exceedances, and both B and C have the second relation. Declared relationships are legitimate properties of tasks, not automatically data leakage. No explicit transformed-query supervision or mediator intervention demonstrated that a Transformer implemented these formulas.", "",
        "Single-task accuracy is not a sufficient proxy for target benefit. V5's related fixed-point candidate is near99% development in both backgrounds while its primary replacement interaction fails directional replication. In task-countv2, alpha-k2 LDS improves from84.57% to99.09% development while mean LIS target accuracy declines. Source-learning records constrain simple explanations, but do not distinguish formula use, optimization, representation changes, answer-distribution effects, or adaptation plasticity.", "",
        "## Claim ledger", ""]
    for claim in claims:
        lines += [f"### {claim['claim_id']} — {claim['claim']}", "", f"**Status:** {claim['evidence_status']}. **Units:** {claim['independent_units']}", "",
                  claim["quantitative_anchor"], "", f"**Supported:** {claim['support']}", "", f"**Limit:** {claim['limitations']}", "",
                  "Evidence: " + "; ".join(f"[{p}](../../{p})" for p in json.loads(claim["artifact_paths"])) + ". Exact SHA-256 values are in the CSV.", ""]
    lines += ["## Recompute", "", "```powershell", ".venv/Scripts/python.exe publication/evidence/build_dmlr_evidence.py", "```", "",
        f"This rebuild performs {len(checks)} stage-level arithmetic checks using archived raw rows, writes only these publication evidence files, and performs no model inference, training, task selection, or historical-file updates.", ""]
    (OUT / "dmlr_evidence.md").write_text("\n".join(lines), encoding="utf8")


def main():
    checks = raw_checks()
    claims = build_claims()
    with (OUT / "dmlr_claims.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(claims[0]))
        writer.writeheader()
        writer.writerows(claims)
    write_markdown(claims, checks)
    audit = {"status": "pass", "claim_count": len(claims), "checks": checks,
             "training_runs": 0, "neural_forwards": 0, "historical_files_written": 0,
             "independent_units_pooled_across_stages": False,
             "source_artifact_sha256": {path: sha(path) for path in sorted({p for c in claims for p in json.loads(c["artifact_paths"])})}}
    (OUT / "dmlr_recompute_checks.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": "pass", "claims": len(claims), "stage_checks": len(checks), "source_artifacts": len(audit["source_artifact_sha256"])}))


if __name__ == "__main__":
    main()
