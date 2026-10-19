"""Build the release report from saved evidence; no model import or training."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def rows(path):
    with path.open(encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    registry = read(HERE / "registry/TASK_REGISTRY.json")
    counts = registry["counts"]
    certificate = read(HERE / "mathematics/CERTIFICATE.json")
    quality = read(HERE / "dataset/quality_full.json")
    data = read(HERE / "dataset/manifest.json")
    compute = rows(HERE / "calibration/compute.csv")
    selected = rows(HERE / "calibration/results_by_seed.csv")
    summary = rows(HERE / "calibration/task_summary.csv")
    calibration_audit = read(HERE / "calibration/CALIBRATION_AUDIT.json")
    audit = read(HERE / "PLATFORM_AUDIT.json")
    assert audit["status"] == calibration_audit["status"] == "passed"
    all_length = [r for r in summary if r["length"] == "all"]
    lines = ["# PermWorld Data Platform v0.1 Delivery Report\n\n",
        "The unified interface, registry, exact generators, versioned sequence protocol, fixed data, correctness certificates, and two-seed single-task calibration of six preselected representatives are complete for three task families. Additional training for the two previous background-interaction comparisons has stopped. pilot_v5's non-replication and uncertain conclusions remain unchanged; this round performed no transfer-combination search.\n\n",
        "## Platform deliverables and effective counts\n\n",
        "| Family | Focus templates | Default instances | Nonconstant instances at fixed length |\n| --- | ---: | ---: | ---: |\n",
        "| Cycle decomposition | 6 | 8 | 8 |\n| Global subsequences | 5 | 9 | 7 |\n| Permutation blocks | 5 | 7 | 7 |\n\n",
        f"All {counts['legacy_endpoints_preserved']} legacy endpoints are retained, with {counts['new_registry_endpoints']} additions, for {counts['total_registry_endpoints']} in total. Excluding two parameter aliases and four representation-output endpoints leaves {counts['canonical_mathematical_templates']} canonical mathematical templates, not the same number of independent mathematical capabilities. The full catalog declares {counts['all_declared_endpoint_instances_including_aliases_and_binary']} instances; canonical data materialize {counts['default_unary_manifest_instances']} unary non-alias instances. One input encoding, one_line_v1, is exposed.\n\n",
        "The focus families contain 16 templates/24 declared instances. I₁=D₁=n; both are retained and marked as constants at fixed length, leaving 23 same-input function classes and 22 nonconstant classes. Merging under registered answer-preserving transformations gives 20 classes/19 nonconstant classes; additionally allowing the components/global_descents add-one mapping gives 19/18. These counting conventions use explicit relations and do not claim mathematical independence. Witness inputs with different answers are retained for the other 275 instance pairs; distinct same-input functions can still have joint relations.\n\n",
        "cycle_length_count(k=1,2,3) already corresponds to fixed_points/two_cycle_count/three_cycle_count; D₂=length and I₂=binom(n,2)−length. New endpoints and genuinely new computational capabilities must be counted separately. Existing tasks such as shortest_cycle are fully retained. The CPU example with k=4 is a runtime configuration and does not increase the template count.\n\n",
        "The 22 nonconstant instances apply to this release's n=8/10/12 and default k=1/2/3. Small n and k>n introduce further degeneracies, so nonconstancy is not claimed at every legal length.\n\n",
        "The [task registry](registry/TASK_REGISTRY.json) contains definitions, legal lengths, parameters, exact algorithms, output ranges/types, families, complexity, measured label-generation costs, and relation evidence. The [instance table](registry/TASK_INSTANCES.csv) and [effective counts](registry/TASK_COUNTS.json) are saved separately.\n\n",
        "## Data, output protocol, and correctness\n\n",
        f"Fixed data cover n=8/10/12 and two data units, with {quality['objects_checked']:,} distinct permutations and {quality['labels_recomputed']:,} exact labels. Each unit has 1024 training, 256 development, and 256 reserved test inputs per length. Full inverse/reverse/complement orbit overlaps with v1–v5 and across this round's roles/units are rejected consistently; RNG replay, duplicate, and leakage checks passed. Proposals are uniform permutations, but the pools after orbit rejection are not claimed to sample uniform orbit classes or be strictly IID. Generation took {data['generation_seconds']:.2f} seconds, excluding subsequent independent verification.\n\n",
        "The explicitly versioned answer protocol is permworld_sequence_v1. All 188 legacy token IDs are preserved, with the vocabulary extended to 203 entries. Large integers use delimited base-100 and structured answers use nested lists, without modulo or truncation. The n=30,k=15 count 155117520 has passed round-trip verification. All focus instances fit a 128-token context; some legacy structured/binary tasks exceed it at large n and are explicitly rejected by the reader. Training six scalar representatives does not substitute for calibrating all structured-output tasks.\n\n",
        "Data-quality tables retain actual answer distributions, observed constants, and majority baselines. The canonical data contain 32 increasing-subsequence counts at k=3 that are ≥100, exercising the multi-token path. The smoke test additionally uses the explicit answer 220 to check variable-length labels, causal masks, padding, full autoregressive evaluation, save/resume, and a fresh optimizer. Development/test majority baselines derive from training labels. The data platform and calibration use different valid tie-breaks for tied structured classes; the six calibration representatives are unaffected.\n\n",
        f"The independent mathematical certificate exhausts S₁…S₇ and adds 192 fixed large-n inputs, totaling {certificate['total_fixtures']:,} fixtures. All {certificate['independent_label_comparisons']:,} label comparisons and {certificate['new_relation_pointwise_checks']:,} pointwise checks of new formulas passed, taking about {certificate['elapsed_seconds']:.2f} CPU seconds. Independent checks use union-find for cycles, subset enumeration for subsequences, growth diagrams for RSK, and set-based cuts for permutation blocks. Finite checks and general formula proofs are registered separately. The 29 v5 relations reuse definition/certificate hashes only; historical formulas were re-verified 0 times.\n\n",
        "The [relation list](mathematics/RELATIONS.json) records single-task mappings, joint queries, and input transformations separately, explicitly stating whether answers on different inputs are needed. Proofs/theorems, finite checks, empirical associations, and unchecked items have separate statuses. An unregistered relation is not evidence of being unrelated, and valid relations are not automatically leakage. Legacy binary-task interfaces remain callable, but this release does not generate or certify binary data pools.\n\n",
        "## Limited single-task calibration results\n\n",
        "Representatives were frozen before formal training: cycle count/longest cycle, LIS/length-3 subsequence count, and block count/largest block. Each task uses seeds 617 and 829, corresponding to data seeds 2026102001 and 2026102002. Each run has 2400 updates, batch size 64, 800 updates at each of three lengths, and one learning rate of 3e-4. The fixed endpoint is the primary reference; the development-best checkpoint is saved separately and is not presented as independent test performance. No task, seed, or budget was changed in response to performance.\n\n",
        "The table reports fixed-endpoint development accuracy for the complete answer plus EOS. Baselines apply each length's training-majority answer to the development set. Lengths are pooled by sample count before averaging over the two seeds.\n\n",
        "| Task | seed617 | seed829 | Two-seed mean | Mean majority baseline | Mean training accuracy | Above baseline |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"]
    for row in all_length:
        matches = [r for r in selected if r["task_id"] == row["task_id"] and r["split"] == "dev" and r["length"] == "all" and r["checkpoint"] == "endpoint"]
        by_seed = {int(r["initialization_seed"]): float(r["accuracy"]) for r in matches}
        name = row["task_id"] + ("(k=3)" if row["task_id"] == "increasing_subsequence_count" else "")
        lines.append(f"| {name} | {100*by_seed[617]:.2f}% | {100*by_seed[829]:.2f}% | {100*float(row['endpoint_dev_accuracy_mean']):.2f}% | {100*float(row['majority_dev_accuracy_mean']):.2f}% | {100*float(row['endpoint_train_accuracy_mean']):.2f}% | {float(row['endpoint_above_majority_pp_mean']):+.2f} pp |\n")
    lines += ["\nSee the [calibration report](calibration/CALIBRATION_REPORT.md), [raw per-seed CSV](calibration/results_by_seed.csv), and [curve CSV](calibration/learning_curves.csv) for per-length performance, full train/dev curves, and endpoint versus development-best comparisons. All representatives are retained. Block tasks have high majority baselines, so their gains above baseline must be considered. For other tasks, moderate accuracy does not establish that a task is fully learned, even when well above baseline. Low training performance supports insufficient learning under the current configuration; train/dev gaps may additionally reflect generalization issues. These results cannot separate algorithmic difficulty, optimization, representation, and budget effects.\n\n",
        "Specific observations: cycle_count and longest_cycle have train/dev gaps of about 31.04 and 34.88 percentage points, respectively, which cannot be attributed solely to failure to fit the training data. At n=12, cycle_count's mean development accuracy is 35.74%, slightly below the 36.72% majority baseline. LIS reaches 98.44% at n=8 and 73.24% at n=12. The length-3 subsequence count reaches only 10.94% at n=12, with training accuracy of just 33.15%, indicating substantial remaining learning limitations. Both block representatives reach roughly 99% and exceed baseline by about 22 percentage points, showing that this configuration can learn these two specific statistics, not that the whole family is easy. All three lengths occur in training, so these differences do not establish unseen-length extrapolation.\n\n",
        "Train/dev curve figures: [cycle decomposition](calibration/figures/cycle_decomposition_curves.png), [global subsequences](calibration/figures/global_subsequence_curves.png), and [permutation blocks](calibration/figures/permutation_blocks_curves.png). Raw per-seed tables: [617](calibration/results_seed617.csv) and [829](calibration/results_seed829.csv).\n\n",
        "![Endpoint and majority baseline by length](calibration/figures/endpoint_by_length.png)\n\n",
        "Structured rsk_shape and other tasks not selected as representatives have no learning calibration in this round. [LEARNABILITY_INDEX.csv](LEARNABILITY_INDEX.csv) distinguishes calibrated/not-run instances explicitly. Two seeds are insufficient to rank entire task families. Reserved test objects received quality checks, but no model test forwards were performed. Development curves therefore do not establish new test performance, unseen-length extrapolation, transfer, or scaling conclusions.\n\n"]
    total_seconds = sum(float(r["seconds"]) for r in compute)
    total_tokens = sum(int(r["input_tokens"]) for r in compute)
    total_updates = sum(int(r["updates"]) for r in compute)
    parameters = sorted({int(r["parameter_count"]) for r in compute})
    lines += ["## Costs, preservation, and closure\n\n",
        f"All 12 runs completed: {total_updates:,} updates and {total_tokens:,} nonpadding training-input tokens. Total run time was {total_seconds/60:.2f} minutes, including full train/dev evaluation and checkpoint saving every 200 steps. The RTX5070 ran a 4-layer/width 256/8-head model with {parameters[0]:,} parameters under the 203-token vocabulary. The architecture matches the legacy 188-token model, but its parameter space is not identical. Equal update budgets do not imply equal token costs. Actual per-run values are in [compute.csv](calibration/compute.csv); software details are in [ENVIRONMENT.json](ENVIRONMENT.json).\n\n",
        f"Final audits passed: {audit['historical_files_byte_preserved']:,} historical v1–v5 files were byte-preserved; 26 calibration checkpoints, including 2 initializations, were loaded and checked; 24 best/endpoint development forwards reproduced saved results; and 1248 curve records and 36864 input-exposure records were verified. Forward replay uses the frozen model and evaluator; independent mathematical algorithms separately establish label correctness. Full evidence is retained in the [platform audit](PLATFORM_AUDIT.json), [calibration audit](calibration/CALIBRATION_AUDIT.json), and [integration review](INTEGRATION_REVIEW.json).\n\n",
        "Historical-file auditing ran in parallel with training. The platform audit's seconds include wall-clock time waiting for calibration to finish and must not be added to training time as a total. No v5 result was retrained or rewritten.\n\n",
        "See the [README](README.md) for complete runnable commands and examples. Source, data, configurations, CSVs, figures, certificates, and environment records are archived. By project convention, checkpoints stay local and are excluded from Git. Regenerate registries/certificates and repeat training in new directories to avoid overwriting frozen evidence.\n\n",
        "The next step should use fixed calibration to distinguish specific templates that are usable from those requiring separately frozen training-protocol diagnostics. Family names alone do not determine difficulty or independence. Future combination selection should exclude count inflation from constants/aliases and inspect the input conditions of single-task and joint relations. This round stops at platform v0.1 and limited calibration, without automatically adding background variants for the two previous targets or new transfer searches.\n"]
    (HERE / "PLATFORM_REPORT.md").write_text("".join(lines), encoding="utf8")
    index = []
    for inst in registry["instances"]:
        matched = [r for r in all_length if r["task_id"] == inst["task_id"] and json.loads(r["params"]) == inst["params"]]
        row = dict(instance_id=inst["instance_id"], task_id=inst["task_id"], params=json.dumps(inst["params"]),
                   status="calibrated_fixed_budget_dev_only" if matched else "not_run_in_v0.1",
                   seeds="617;829" if matched else "", lengths="8;10;12" if matched else "",
                   endpoint_dev_mean=matched[0]["endpoint_dev_accuracy_mean"] if matched else "",
                   majority_dev_mean=matched[0]["majority_dev_accuracy_mean"] if matched else "",
                   retained=True, model_test_evaluated=False)
        index.append(row)
    with (HERE / "LEARNABILITY_INDEX.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index[0]))
        writer.writeheader(); writer.writerows(index)
    print(json.dumps(dict(status="report_complete", calibrated_instances=6, declared_instances=len(index), updates=total_updates,
                          training_input_tokens=total_tokens, training_minutes=total_seconds/60)))


if __name__ == "__main__":
    main()
