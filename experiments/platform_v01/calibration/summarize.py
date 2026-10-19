"""Summarize the finite v0.1 calibration, retaining every preselected task."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from permworld_platform.calibration import CFG, TASKS, read_json, run_id, task_slug, write_csv, write_json


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out = args.out.resolve()
    all_curves, selected, compute, exposures, per_length_exposures = [], [], [], [], []
    completed = []
    for repeat in CFG["repeats"]:
        for task in TASKS:
            rid = run_id(task, repeat)
            run_dir = out / "runs" / rid
            record = read_json(run_dir / "COMPLETE.json")
            curve = rows(run_dir / "curves.csv")
            all_curves.extend(curve)
            completed.append(record)
            dev_rows = [r for r in curve if r["split"] == "dev" and r["length"] == "all" and int(r["step"]) > 0]
            chosen = min(dev_rows, key=lambda r: (-float(r["accuracy"]), float(r["answer_ce"]), int(r["step"])))
            assert int(chosen["step"]) == record["best_step"]
            for checkpoint, update in [("endpoint", CFG["updates"]), ("dev_best", record["best_step"])]:
                for row in curve:
                    if int(row["step"]) == update:
                        selected.append({"checkpoint": checkpoint, **row,
                            "above_majority_pp": 100 * (float(row["accuracy"]) - float(row["majority_accuracy"])),
                            "role": "primary fixed-budget reference" if checkpoint == "endpoint" else "descriptive development-selected reference"})
            compute.append({k: record[k] for k in ["run_id", "initialization_seed", "data_seed", "task_id", "family", "params",
                "updates", "sample_exposures", "input_tokens", "padded_input_tokens", "supervised_tokens", "answer_tokens", "seconds",
                "parameter_count", "peak_memory_bytes", "initial_state_sha256", "best_step"]})
            exposures.extend(rows(run_dir / "input_exposures.csv"))
            for length in CFG["lengths"]:
                n = str(length)
                values = [int(r["exposures"]) for r in exposures if r["run_id"] == rid and int(r["length"]) == length]
                per_length_exposures.append({"run_id": rid, "task_id": task["task_id"], "params": json.dumps(task["params"]),
                    "initialization_seed": repeat["initialization_seed"], "data_seed": repeat["data_seed"], "length": length,
                    "updates": record["length_updates"][n], "distinct_train_inputs": len(values),
                    "exposures": sum(values), "min_exposure_per_input": min(values), "max_exposure_per_input": max(values),
                    **record["tokens_by_length"][n]})
    write_csv(out / "learning_curves.csv", all_curves)
    write_csv(out / "results_by_seed.csv", selected)
    for repeat in CFG["repeats"]:
        seed = repeat["initialization_seed"]
        write_csv(out / f"results_seed{seed}.csv", [r for r in selected if int(r["initialization_seed"]) == seed])
    write_csv(out / "compute.csv", compute)
    write_csv(out / "input_exposures.csv", exposures)
    write_csv(out / "length_exposures.csv", per_length_exposures)
    task_summary = []
    for task in TASKS:
        for length in [*CFG["lengths"], "all"]:
            task_rows = [r for r in selected if r["task_id"] == task["task_id"] and r["split"] == "dev"
                         and r["checkpoint"] == "endpoint" and str(r["length"]) == str(length)]
            accuracy = np.asarray([float(r["accuracy"]) for r in task_rows])
            baseline = np.asarray([float(r["majority_accuracy"]) for r in task_rows])
            train_rows = [r for r in selected if r["task_id"] == task["task_id"] and r["split"] == "train"
                         and r["checkpoint"] == "endpoint" and str(r["length"]) == str(length)]
            train_accuracy = np.asarray([float(r["accuracy"]) for r in train_rows])
            task_summary.append({"task_id": task["task_id"], "params": json.dumps(task["params"]), "family": task["family"],
                "length": length, "repeats": len(task_rows), "endpoint_train_accuracy_mean": train_accuracy.mean(),
                "endpoint_dev_accuracy_mean": accuracy.mean(), "endpoint_dev_accuracy_min": accuracy.min(),
                "endpoint_dev_accuracy_max": accuracy.max(), "majority_dev_accuracy_mean": baseline.mean(),
                "endpoint_above_majority_pp_mean": 100 * (accuracy - baseline).mean(),
                "status": "retained; limited fixed-budget development evidence, not a task exclusion criterion"})
    write_csv(out / "task_summary.csv", task_summary)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    colors = {CFG["repeats"][0]["initialization_seed"]: "#1767a6", CFG["repeats"][1]["initialization_seed"]: "#d76623"}
    for family in dict.fromkeys(t["family"] for t in TASKS):
        tasks = [t for t in TASKS if t["family"] == family]
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        for col, task in enumerate(tasks):
            for repeat in CFG["repeats"]:
                seed = repeat["initialization_seed"]
                for split, style in [("train", "--"), ("dev", "-")]:
                    cr = [r for r in all_curves if r["task_id"] == task["task_id"] and int(r["initialization_seed"]) == seed
                          and r["length"] == "all" and r["split"] == split]
                    cr.sort(key=lambda r: int(r["step"]))
                    xs = [int(r["step"]) for r in cr]
                    axes[0, col].plot(xs, [100 * float(r["accuracy"]) for r in cr], style, color=colors[seed], label=f"{seed} {split}")
                    axes[1, col].plot(xs, [float(r["answer_ce"]) for r in cr], style, color=colors[seed], label=f"{seed} {split}")
                baseline = next(float(r["majority_accuracy"]) for r in cr)
                axes[0, col].axhline(100 * baseline, color=colors[seed], linestyle=":", alpha=.65)
            axes[0, col].set_title(task_slug(task))
            axes[0, col].set_ylim(0, 102)
            for row in range(2):
                axes[row, col].set_xlabel("Updates")
                axes[row, col].grid(alpha=.2)
                axes[row, col].legend(fontsize=8)
            axes[0, col].set_ylabel("Exact answer + EOS accuracy (%)")
            axes[1, col].set_ylabel("Teacher-forced answer token CE")
        fig.suptitle(f"{family}: fixed-budget single-task calibration; dotted = train-derived dev majority")
        for suffix in ("png", "svg"):
            fig.savefig(figures / f"{family}_curves.{suffix}", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, task in zip(axes.flat, TASKS):
        for repeat in CFG["repeats"]:
            seed = repeat["initialization_seed"]
            task_rows = [r for r in selected if r["task_id"] == task["task_id"] and int(r["initialization_seed"]) == seed
                         and r["checkpoint"] == "endpoint" and r["split"] == "dev" and r["length"] != "all"]
            task_rows.sort(key=lambda r: int(r["length"]))
            axis.plot([int(r["length"]) for r in task_rows], [100 * float(r["accuracy"]) for r in task_rows],
                      "o-", color=colors[seed], label=f"{seed} endpoint")
            axis.plot([int(r["length"]) for r in task_rows], [100 * float(r["majority_accuracy"]) for r in task_rows],
                      ":", color=colors[seed], label=f"{seed} majority")
        axis.set_title(task_slug(task))
        axis.set_xlabel("Permutation length")
        axis.set_ylabel("Dev exact accuracy (%)")
        axis.set_ylim(0, 102)
        axis.set_xticks(CFG["lengths"])
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    for suffix in ("png", "svg"):
        fig.savefig(figures / f"endpoint_by_length.{suffix}", dpi=150)
    plt.close(fig)
    timings = {"runs": len(compute), "updates": sum(r["updates"] for r in compute),
        "input_tokens": sum(r["input_tokens"] for r in compute), "padded_input_tokens": sum(r["padded_input_tokens"] for r in compute),
        "supervised_tokens": sum(r["supervised_tokens"] for r in compute),
        "sample_exposures": sum(r["sample_exposures"] for r in compute),
        "run_seconds": sum(r["seconds"] for r in compute), "test_evaluated": False,
        "training_timer_scope": "Full train/dev evaluations and checkpoint saves included; data encoding, summaries/audits excluded"}
    write_json(out / "SUMMARY.json", timings)
    text = ["# PermWorld v0.1 单任务标定\n\n", "本轮预先固定六个任务、两个种子和统一 2400 次更新；不进行迁移组合搜索，也不按表现删除任务。终点是主要固定预算参照，开发集最佳 checkpoint 仅作描述性结果；平台测试集没有用于标定。\n\n",
            f"模型沿用 4 层、宽度 256、8 个注意力头的 Transformer 结构。为显式版本化的新任务与结构化答案接口，词表从历史 188 扩展到 203，参数量为 {compute[0]['parameter_count']:,}；它不是与历史 checkpoint 完全相同的参数空间。本轮每个种子从同一份新初始化开始，六个任务分别建立空状态 AdamW，没有加载旧迁移权重。\n\n",
            "训练使用长度 8、10、12，各长度 1024 个训练输入、256 个开发输入；每三次更新按固定种子打乱长度顺序，因此每长度恰好 800 次更新。所有任务共用每个重复的数据与模型初始权重。三个长度都出现在训练中，按长度结果不构成未见长度的外推测试。\n\n",
            "准确率要求从提示独立生成全部答案及 EOS 完全正确；交叉熵使用 teacher forcing，两者不同。输出编码已显式版本化，可容纳 increasing_subsequence_count(k=3) 超过原单 token 范围的答案。这六个代表均输出标量；平台支持结构化答案的正确编码，但本轮没有标定所有结构化输出任务的可学习性。\n\n",
            "| 任务 | 终点训练准确率均值 | 终点开发准确率均值 | 两种子开发范围 | 开发多数基线 | 超多数基线 |\n|---|---:|---:|---:|---:|---:|\n"]
    for row in task_summary:
        if row["length"] == "all":
            text.append(f"| {row['task_id']}{row['params'] if row['params'] != '{}' else ''} | {100 * row['endpoint_train_accuracy_mean']:.2f}% | {100 * row['endpoint_dev_accuracy_mean']:.2f}% | {100 * row['endpoint_dev_accuracy_min']:.2f}–{100 * row['endpoint_dev_accuracy_max']:.2f}% | {100 * row['majority_dev_accuracy_mean']:.2f}% | {row['endpoint_above_majority_pp_mean']:+.2f} pp |\n")
    cy12 = next(r for r in task_summary if r["task_id"] == "cycle_count" and r["length"] == 12)
    longest = next(r for r in task_summary if r["task_id"] == "longest_cycle" and r["length"] == "all")
    subseq = next(r for r in task_summary if r["task_id"] == "increasing_subsequence_count" and r["length"] == "all")
    text.extend([f"\n循环任务的困难没有被总体均值掩盖：cycle_count 在长度 12 的平均开发准确率只有 {100 * cy12['endpoint_dev_accuracy_mean']:.2f}%，低于该长度多数基线 {100 * cy12['majority_dev_accuracy_mean']:.2f}%。longest_cycle 的平均训练准确率为 {100 * longest['endpoint_train_accuracy_mean']:.2f}%，开发准确率为 {100 * longest['endpoint_dev_accuracy_mean']:.2f}%，同时存在未充分拟合与泛化落差；这两者不能归结为单一机制。\n",
        f"\nincreasing_subsequence_count(k=3) 的开发准确率为 {100 * subseq['endpoint_dev_accuracy_mean']:.2f}%，虽然比多数基线高 {subseq['endpoint_above_majority_pp_mean']:.2f} 个百分点，训练准确率也只有 {100 * subseq['endpoint_train_accuracy_mean']:.2f}%，仍保留为当前配置下未充分学会的任务。相比之下，两个排列块代表的开发准确率都接近 99%，且明确高于约 77% 的多数基线。上述是有限预算下的开发观察，不是对整个任务族的保证。\n",
        "\n多数基线按任务及长度只从训练集选择，再应用于开发集；高原始准确率可能源于答案不平衡，需同时读取超多数基线差值和按长度曲线。仅两个种子，开发集参与 checkpoint 选择，不能据此估计稳定迁移能力或形成测试性能结论。未学会的任务全部保留，预算不足与编码/优化难度仍不能分离。本轮的输入长度、训练样本量及编码接口已有显式变化，准确率不能与 v5 直接作为控制对照。\n",
        f"\n共 {timings['runs']} 次训练，{timings['updates']:,} 次更新，{timings['sample_exposures']:,} 次样本暴露，{timings['input_tokens']:,} 个实际输入 tokens；右侧 padding 计算量另列。累计训练计时 {timings['run_seconds'] / 60:.2f} 分钟，包含完整训练/开发评估和保存。\n",
        "\n- [逐种子与按长度结果](results_by_seed.csv)\n- [训练/开发完整曲线](learning_curves.csv)\n- [每个输入实际暴露量](input_exposures.csv)\n- [按长度预算](length_exposures.csv)\n- [运行成本](compute.csv)\n- [按长度图](figures/endpoint_by_length.png)\n- [完整开发前向与 checkpoint 审计](CALIBRATION_AUDIT.json)\n",
        "\n复现命令从仓库根目录运行，先按[平台复现说明](../README.md)在 `experiments/platform_v01_reproduction/dataset` 生成并核验新数据目录。下面所有输出写入新目录，freeze 绑定其中的数据及当时的代码哈希。已有完整运行按哈希检查后跳过；不完整运行从相同初始模型重新开始，不承诺逐更新断点恢复。归档目录的 SMOKE 也属于冻结证据，重跑 smoke 应使用新目录。\n\n```powershell\n.venv/Scripts/python.exe -m permworld_platform.calibration smoke --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data\n.venv/Scripts/python.exe -m permworld_platform.calibration freeze --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data\n.venv/Scripts/python.exe -m permworld_platform.calibration train --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data\n.venv/Scripts/python.exe experiments/platform_v01/calibration/summarize.py --out experiments/platform_v01_reproduction/calibration\n.venv/Scripts/python.exe experiments/platform_v01/calibration/audit.py --forwards --out experiments/platform_v01_reproduction/calibration --data-dir experiments/platform_v01_reproduction/dataset/data\n```\n"])
    (out / "CALIBRATION_REPORT.md").write_text("".join(text), encoding="utf-8")
    print(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
