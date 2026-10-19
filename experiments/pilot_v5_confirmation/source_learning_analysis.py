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
    candidate_table = table(["目标", "seed", "B dev原→新", "B dev变化pp", "C dev原→新", "C dev变化pp"],
                            [[r["target"], r["seed"], f"{pct(r['B_old_dev_accuracy'])}→{pct(r['B_new_dev_accuracy'])}", pp(r["B_dev_accuracy_change"]),
                              f"{pct(r['C_old_dev_accuracy'])}→{pct(r['C_new_dev_accuracy'])}", pp(r["C_dev_accuracy_change"])] for r in candidate_rows])
    common_table = table(["目标", "seed", "条件", "未换公共源", "dev背景差pp"],
                         [[r["target"], r["seed"], r["condition"], r["old_task"], pp(r["dev_accuracy_change"])] for r in unchanged])
    full_table = table(["目标", "背景", "条件", "任务", "train均值", "dev均值", "dev范围", "多数基线均值", "2400→4800 dev均值变化pp"],
                       [[r["target"], r["background"], r["condition"], r["task"], pct(r["train_accuracy_mean"]), pct(r["dev_accuracy_mean"]),
                         f"[{pct(r['dev_accuracy_min'])}, {pct(r['dev_accuracy_max'])}]", pct(r["dev_majority_accuracy_mean"]), pp(r["dev_accuracy_4800_minus_2400_mean"])] for r in summaries])
    text = f"""# v5 源任务学习记录：在验证测试之前的描述

本说明只读 `source_curves.csv`、`source_exposures.csv`、冻结config和预训练完成标记。没有导入模型、没有新增训练/forward，没有读取任何新目标测试结果。分析时TEST_COMPLETE存在：{result['test_complete_exists_at_analysis']}。**尚未将这些源变化与新验证交互联系起来**；下面只描述在固定4800步预算下能直接看到的源学习现象。

40个新预训练run全部完成，160条任务×run记录均为n=10、每任务76800次暴露。每个源run均按原配置完成4800步，源训练不足不触发延长。结果由各任务的8192条train和1024条dev计算；多数类别由该任务训练池决定，并在相应dev上评估。学习程度须结合任务自己的基线，而不是跨任务直接比较准确率。

## 候选源的全部五重复配对差

{candidate_table}

FP相关候选anti_fixed_points的dev均值由{pct(af0['dev_accuracy_mean'])}变为{pct(af2['dev_accuracy_mean'])}，平均变化{pp(ps.average(af_changes))}pp；五次中{sum(x<0 for x in af_changes)}次降低、{sum(x>0 for x in af_changes)}次提高。两个背景下均接近99%，因此不能把FP背景差泛化解释为“原背景完全没学会相关候选”。这一观察尚不说明目标迁移如何变化。

LDR相关候选longest_increasing_run的dev均值由{pct(lir0['dev_accuracy_mean'])}变为{pct(lir1['dev_accuracy_mean'])}，五次均下降：{', '.join(pp(v) for v in lir_changes)}pp。新背景dev多数基线均值为{pct(lir1['dev_majority_accuracy_mean'])}，平均超基线仅{pp(lir1['dev_accuracy_above_majority_mean'])}pp，其中{lir1['dev_below_own_majority_count']}/5次低于各自基线，seed211恰好等于自己的基线。其新背景train均值也仅{pct(lir1['train_accuracy_mean'])}，所以这种源学习不足并非只表现为训练集学会后的开发集落差；优化速度与表示限制仍不能据此区分。原背景dev从{pct(lir0['dev_accuracy_min'])}到{pct(lir0['dev_accuracy_max'])}，仍有明显初始化/数据重复差异。这支持“recoils背景伴随LIR源学习下降”的有限重复描述，尚不证明源下降导致目标收益消失。

## 未换公共任务及不一致现象

{common_table}

LDR两个条件的peaks共有10条同任务跨背景差，其中{sum(r['dev_accuracy_change']<0 for r in peak)}条为负、{sum(r['dev_accuracy_change']>0 for r in peak)}条为正。因此旧v4中peaks各块均下降的描述在新源记录上并非逐项完全复现；必须保留这些正向变化，不能写成所有公共源一致受损。successions同样具有seed/条件差异，完整40条未换公共源对照保留；20条被换slot只列不同任务的原/新水平，不计算其“同任务学习差”。

同一背景内B/C公共源差另见 `source_analysis/common_source_BC.csv`。这些task定义及输入流相同而组合不同，所以它们的学习改变是组合研究的实际结果；仅凭准确率表仍不能判断是优化速度、梯度竞争、共享结构还是其他原因。候选与公共源可能不同步改善，也不能用一个候选准确率作所有目标效果的代理。

## 所有源的训练、开发与后半预算变化

{full_table}

`source_final_by_repeat.csv`逐条保留train/dev CE、与多数基线之差、泛化差、2400→4800变化、暴露量及输入流hash。许多descents/recoils/peaks/successions条件在4800步仍远未接近满分，且部分仍有后半程改善；这说明当前预算的源学习状态不同，不能视为所有源充分学会后的纯任务组成比较。这里不设新“学会”门槛，也不据曲线延长预算；已有历史单任务数据使用不同seed和池，不能替代同预算新重复的单任务因果对照。

至此能支持的是源学习存在背景及重复依赖，以及LDR相关候选在recoils背景下的五次一致下降。是否复现预指定的正/负目标交互，必须等待全部新适配、开发选择和独立测试完成，再按冻结统计方案回答。本说明不读取新目标曲线或测试，也不改变任何组合、训练预算、选择规则或统计方法。
"""
    path = HERE / "SOURCE_LEARNING_NOTES.md"
    path.write_text(text, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
