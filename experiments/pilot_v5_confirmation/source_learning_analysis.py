"""CPU-only source-learning descriptions, independent of target test results."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import paired_statistics as ps

HERE = Path(__file__).resolve().parent


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def pp(value):
    return f"{100 * value:+.2f}"


def pct(value):
    return f"{100 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "artifacts")
    parser.add_argument("--analysis-out", type=Path, default=HERE / "source_analysis")
    args = parser.parse_args()
    out, analysis_out = args.out.resolve(), args.analysis_out.resolve()
    assert (out / "PRETRAIN_COMPLETE.json").exists()
    analysis_out.mkdir(parents=True, exist_ok=True)
    curves_path, exposures_path = out / "source_curves.csv", out / "source_exposures.csv"
    curves, exposures = ps.read_csv(curves_path), ps.read_csv(exposures_path)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert len(exposures) == 160
    index = {(r["target"], int(r["seed"]), r["background"], r["condition"], r["task"], int(r["step"]), r["split"]): r for r in curves}
    learning = []
    for exposure in exposures:
        row = {**exposure, "role": "candidate" if int(exposure["slot"]) == 0 else "common"}
        for split in ("train", "dev"):
            key = (row["target"], int(row["seed"]), row["background"], row["condition"], row["task"])
            end = index[*key, 4800, split]
            mid = index[*key, 2400, split]
            for metric in ("accuracy", "answer_ce", "majority_accuracy", "count"):
                row[split + "_" + metric] = float(end[metric])
            row[split + "_accuracy_above_majority"] = float(end["accuracy"]) - float(end["majority_accuracy"])
            row[split + "_accuracy_2400"] = float(mid["accuracy"])
            row[split + "_accuracy_4800_minus_2400"] = float(end["accuracy"]) - float(mid["accuracy"])
            row[split + "_answer_ce_4800_minus_2400"] = float(end["answer_ce"]) - float(mid["answer_ce"])
        row["train_minus_dev_accuracy"] = row["train_accuracy"] - row["dev_accuracy"]
        learning.append(row)
    lookup = {(r["target"], int(r["seed"]), r["background"], r["condition"], int(r["slot"])): r for r in learning}
    differences, candidate_rows, common_bc = [], [], []
    for hypothesis in config["hypotheses"]:
        target, original, new = hypothesis["target"], hypothesis["old_background"], hypothesis["new_background"]
        for seed in config["seeds"]:
            candidate = dict(target=target, seed=seed, original_background=original, new_background=new)
            for condition in ("B", "C"):
                for slot in range(4):
                    old, fresh = lookup[target, seed, original, condition, slot], lookup[target, seed, new, condition, slot]
                    same = old["task"] == fresh["task"]
                    row = dict(target=target, seed=seed, condition=condition, slot=slot,
                               original_background=original, new_background=new,
                               old_task=old["task"], new_task=fresh["task"], same_task=same,
                               old_train_accuracy=old["train_accuracy"], new_train_accuracy=fresh["train_accuracy"],
                               old_dev_accuracy=old["dev_accuracy"], new_dev_accuracy=fresh["dev_accuracy"],
                               old_dev_majority_accuracy=old["dev_majority_accuracy"], new_dev_majority_accuracy=fresh["dev_majority_accuracy"],
                               old_dev_answer_ce=old["dev_answer_ce"], new_dev_answer_ce=fresh["dev_answer_ce"],
                               train_accuracy_change=fresh["train_accuracy"] - old["train_accuracy"] if same else "",
                               dev_accuracy_change=fresh["dev_accuracy"] - old["dev_accuracy"] if same else "",
                               dev_answer_ce_change=fresh["dev_answer_ce"] - old["dev_answer_ce"] if same else "")
                    differences.append(row)
                    if slot == 0:
                        candidate[condition + "_task"] = old["task"]
                        for field in ("old_train_accuracy", "new_train_accuracy", "old_dev_accuracy", "new_dev_accuracy", "dev_accuracy_change", "train_accuracy_change", "old_dev_answer_ce", "new_dev_answer_ce"):
                            candidate[condition + "_" + field] = row[field]
            candidate_rows.append(candidate)
            for background in (original, new):
                for slot in (1, 2, 3):
                    b, c = lookup[target, seed, background, "B", slot], lookup[target, seed, background, "C", slot]
                    assert b["task"] == c["task"]
                    common_bc.append(dict(target=target, seed=seed, background=background, task=b["task"], slot=slot,
                                          B_train_accuracy=b["train_accuracy"], C_train_accuracy=c["train_accuracy"],
                                          B_dev_accuracy=b["dev_accuracy"], C_dev_accuracy=c["dev_accuracy"],
                                          B_minus_C_train_accuracy=b["train_accuracy"] - c["train_accuracy"],
                                          B_minus_C_dev_accuracy=b["dev_accuracy"] - c["dev_accuracy"]))
    grouped = defaultdict(list)
    for row in learning:
        grouped[row["target"], row["background"], row["condition"], row["task"]].append(row)
    summaries = []
    for (target, background, condition, task), rows in grouped.items():
        summary = dict(target=target, background=background, condition=condition, task=task, n=len(rows))
        for metric in ("train_accuracy", "dev_accuracy", "dev_answer_ce", "dev_majority_accuracy", "dev_accuracy_above_majority", "train_minus_dev_accuracy", "dev_accuracy_4800_minus_2400"):
            values = [r[metric] for r in rows]
            summary.update({metric + "_mean": ps.average(values), metric + "_min": min(values), metric + "_max": max(values)})
        summary["dev_below_own_majority_count"] = sum(r["dev_accuracy"] < r["dev_majority_accuracy"] for r in rows)
        summary["dev_decrease_2400_to_4800_count"] = sum(r["dev_accuracy_4800_minus_2400"] < 0 for r in rows)
        summaries.append(summary)
    ps.write_csv(analysis_out / "source_final_by_repeat.csv", learning)
    ps.write_csv(analysis_out / "source_final_summary.csv", summaries)
    ps.write_csv(analysis_out / "source_background_differences.csv", differences)
    ps.write_csv(analysis_out / "candidate_source_background_differences.csv", candidate_rows)
    ps.write_csv(analysis_out / "common_source_BC.csv", common_bc)
    result = dict(unix=time.time(), status="source_only_complete", target_test_results_read=False,
                  test_complete_exists_at_analysis=(out / "TEST_COMPLETE.json").exists(),
                  model_imports_or_forward=False, source_runs=40, final_task_rows=len(learning),
                  same_task_background_comparisons=sum(r["same_task"] for r in differences),
                  changed_task_background_comparisons=sum(not r["same_task"] for r in differences),
                  candidate_comparison_rows=len(candidate_rows), common_BC_rows=len(common_bc),
                  source_curves_sha256=ps.sha256(curves_path), source_exposures_sha256=ps.sha256(exposures_path),
                  config_sha256=ps.sha256(out / "config.json"), script_sha256=ps.sha256(Path(__file__)),
                  note="Descriptive source-only analysis. No association with new confirmation transfer interactions has been assessed.")
    (analysis_out / "SOURCE_ANALYSIS_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    def summary(target, background, condition, task):
        return next(r for r in summaries if (r["target"], r["background"], r["condition"], r["task"]) == (target, background, condition, task))

    lir0 = summary("longest_decreasing_run", "g0_v3", "B", "longest_increasing_run")
    lir1 = summary("longest_decreasing_run", "g1_recoils", "B", "longest_increasing_run")
    af0 = summary("fixed_points", "g0_v3", "B", "anti_fixed_points")
    af2 = summary("fixed_points", "g2_deficiencies", "B", "anti_fixed_points")
    lir_changes = [r["B_dev_accuracy_change"] for r in candidate_rows if r["target"] == "longest_decreasing_run"]
    af_changes = [r["B_dev_accuracy_change"] for r in candidate_rows if r["target"] == "fixed_points"]
    unchanged = [r for r in differences if r["same_task"] and r["slot"] != 0]
    peak = [r for r in unchanged if r["target"] == "longest_decreasing_run" and r["old_task"] == "peaks"]
    candidate_table = table(["Target", "seed", "B dev original→new", "B dev change pp", "C dev original→new", "C dev change pp"],
                            [[r["target"], r["seed"], f"{pct(r['B_old_dev_accuracy'])}→{pct(r['B_new_dev_accuracy'])}", pp(r["B_dev_accuracy_change"]),
                              f"{pct(r['C_old_dev_accuracy'])}→{pct(r['C_new_dev_accuracy'])}", pp(r["C_dev_accuracy_change"])] for r in candidate_rows])
    common_table = table(["Target", "seed", "Condition", "Unchanged common source", "Dev background difference pp"],
                         [[r["target"], r["seed"], r["condition"], r["old_task"], pp(r["dev_accuracy_change"])] for r in unchanged])
    full_table = table(["Target", "Background", "Condition", "Task", "Mean train", "Mean dev", "Dev range", "Mean majority baseline", "2400→4800 mean dev change pp"],
                       [[r["target"], r["background"], r["condition"], r["task"], pct(r["train_accuracy_mean"]), pct(r["dev_accuracy_mean"]),
                         f"[{pct(r['dev_accuracy_min'])}, {pct(r['dev_accuracy_max'])}]", pct(r["dev_majority_accuracy_mean"]), pp(r["dev_accuracy_4800_minus_2400_mean"])] for r in summaries])
    text = f"""# v5 source-task learning records: description before confirmation testing

This note reads only `source_curves.csv`, `source_exposures.csv`, the frozen configuration, and pretraining completion markers. It imports no model, performs no new training/forward passes, and reads no new target-test results. TEST_COMPLETE exists at analysis time: {result['test_complete_exists_at_analysis']}. **These source changes have not yet been linked to the new confirmation interactions.** The following describes only directly observed source learning under the fixed4800-step budget.

All 40 new pretraining runs are complete. The 160 task×run records all use n=10 and 76800 exposures per task. Every source run completed 4800 steps under the original configuration; insufficient source learning does not trigger extensions. Metrics use 8192 training and 1024 development examples per task. The majority class is determined from that task's training pool and evaluated on its development pool. Learning must be interpreted relative to each task's own baseline, not by directly comparing accuracies across tasks.

## Paired candidate-source differences for all five repeats

{candidate_table}

For the FP-related candidate anti_fixed_points, mean development accuracy changes from {pct (af0['dev_accuracy_mean'])} to {pct (af2['dev_accuracy_mean'])}, a mean change of {pp (ps.average (af_changes))}pp; {sum (x<0 for x in af_changes)} of five repeats decrease and {sum (x>0 for x in af_changes)} increase. Both backgrounds are near 99%, so FP background differences cannot generally be explained by complete failure to learn the related candidate in the original background. This observation does not yet establish how target transfer changes.

For the LDR-related candidate longest_increasing_run, mean development accuracy changes from {pct (lir0['dev_accuracy_mean'])} to {pct (lir1['dev_accuracy_mean'])}, declining in all five repeats: {', '.join (pp (v) for v in lir_changes)}pp. The new-background mean development majority baseline is {pct (lir1['dev_majority_accuracy_mean'])}; mean accuracy above baseline is only {pp (lir1['dev_accuracy_above_majority_mean'])}pp, with {lir1['dev_below_own_majority_count']}/5 below their own baseline and seed211 exactly at its baseline. New-background mean training accuracy is also only {pct (lir1['train_accuracy_mean'])}; insufficient source learning is therefore not limited to a development gap after successful training fit. Optimization speed and representational limitations remain indistinguishable. Original-background development accuracy ranges from {pct (lir0['dev_accuracy_min'])} to {pct (lir0['dev_accuracy_max'])}, with substantial initialization/data-repeat variation. This supports the limited-repeat observation that the recoils background accompanies lower LIR source learning, not a causal claim that this decline removes target benefits.

## Unchanged common tasks and inconsistent changes

{common_table}

For peaks across the two LDR conditions, the 10 same-task background differences include {sum (r['dev_accuracy_change']<0 for r in peak)} negative and {sum (r['dev_accuracy_change']>0 for r in peak)} positive changes. The v4 observation that peaks decreased in every block therefore does not reproduce entry by entry. These positive changes must be retained, rather than claiming that all common sources are consistently harmed. successions also varies by seed/condition. All 40 unchanged-common-source comparisons are retained;20 replaced-slot entries show original/new task levels without computing a same-task learning difference.

Within-background B/C common-source differences are in `source_analysis/common_source_BC.csv`. These tasks have identical definitions and input streams but different compositions, so learning changes are substantive composition results. Accuracy tables alone cannot determine whether optimization speed, gradient competition, shared structure, or other causes explain them. Candidate and common sources need not improve together; one candidate accuracy cannot proxy all target outcomes.

## Training, development, and second-half budget changes for all sources

{full_table}

`source_final_by_repeat.csv` retains train/dev CE, differences from majority baselines, generalization gaps,2400→4800 changes, exposures, and input-stream hashes for every entry. Many descents/recoils/peaks/successions conditions remain far below perfect accuracy at 4800, with some still improving in the second half. Source learning therefore differs under this budget; this is not a pure composition comparison after all sources have been fully learned. No new learning threshold or curve-based budget extension is introduced. Historical single-task records use different seeds/pools and cannot replace same-budget single-task causal controls for the new repeats.

The current evidence supports background- and repeat-dependent source learning and five consistent decreases for the LDR-related candidate in the recoils background. Whether the prespecified positive/negative target interactions reproduce must be answered under the frozen statistical plan after all new adaptation, development selection, and independent testing finish. This note reads no new target curves or tests and changes no compositions, budgets, selection rules, or statistical methods.
"""
    path = HERE / "SOURCE_LEARNING_NOTES.md"
    path.write_text(text, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
