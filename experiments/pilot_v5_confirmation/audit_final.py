"""Independent CPU audit of the five-repeat fixed-hypothesis validation.

Never imports the training engine, queries CUDA, or executes model forwards.
Run after TEST_COMPLETE and summarization. For later report/figure edits only,
use --refresh-manifest; it checks hashes but does not repeat the full audit.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
HISTORICAL = [HERE.parent / "permworld_pilot/runs/pilot_v1",
              HERE.parent / "pilot_v2_diagnostics/artifacts",
              HERE.parent / "pilot_v3_multitarget/artifacts",
              HERE.parent / "pilot_v4_backgrounds/artifacts"]
sys.path.insert(0, str(HERE.parent / "permworld_pilot/vendor"))
from neurips_permutations.math_ops import PROPERTY_FUNCTIONS
from neurips_permutations.models import CausalTransformer
from neurips_permutations.passage import TOKEN_TO_ID, VOCABULARY

torch.set_num_threads(2)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def state_hash(state, token=None, initial_row=None):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if token is not None and name in ["token_embedding.weight", "lm_head.weight"]:
            value = value.clone()
            value[token].copy_(initial_row)
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message):
    check(math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-12), message)


def truth(value):
    check(value in [True, False, "True", "False"], f"Invalid Boolean {value!r}")
    return value is True or value == "True"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def csvwrite(path, values):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(dict.fromkeys(k for r in values for k in r)))
        writer.writeheader()
        writer.writerows(values)


def artifact_files(out):
    """Retain the original manifest scope; execution logs are separate records."""
    return {p.resolve() for directory in {HERE, out} for p in directory.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
            and p.name != "artifact_manifest.json" and p.suffix != ".log"}


def artifact_manifest_entries(out):
    return {os.path.relpath(path, out).replace("\\", "/"):
            dict(bytes=path.stat().st_size, sha256=sha(path)) for path in sorted(artifact_files(out))}


def write_artifact_manifest(out, entries=None, refresh=None):
    """Write inventory; this function does not perform or claim an audit."""
    manifest = dict(path_base=str(out), files=artifact_manifest_entries(out) if entries is None else entries,
                    final_audit_sha256=sha(out / "FINAL_AUDIT.json"),
                    test_complete_sha256=sha(out / "TEST_COMPLETE.json"))
    if refresh is not None:
        manifest["presentation_only_refresh"] = refresh
    write(out / "artifact_manifest.json", manifest)


def validate_manifest_refresh(previous, current, frozen_relative_paths):
    """Only nonfrozen Markdown/PNG/SVG may be changed or added by refresh."""
    check(set(previous).issubset(current), "Manifest refresh refuses removal of previously listed artifacts")
    changed = [name for name in previous if previous[name] != current[name]]
    added = sorted(set(current) - set(previous))
    for name in changed + added:
        check(name not in frozen_relative_paths and Path(name).suffix.lower() in {".md", ".png", ".svg"},
              f"Manifest refresh refuses changed/new protected artifact: {name}; rerun the full audit")
    return dict(changed_presentation_files=sorted(changed), new_presentation_files=added,
                unchanged_previously_listed_files=len(previous) - len(changed))


def refresh_artifact_manifest(out):
    """Hash-check protected artifacts and refresh delivery files, without re-audit."""
    check((out / "FINAL_AUDIT.json").exists() and (out / "TEST_COMPLETE.json").exists(),
          "Manifest refresh requires existing FINAL_AUDIT and TEST_COMPLETE")
    check(read(out / "FINAL_AUDIT.json")["status"] == "passed", "Manifest refresh requires a passed full audit")
    previous_path = out / "artifact_manifest.json"
    check(previous_path.exists(), "Manifest refresh requires the prior full-audit artifact manifest")
    previous_sha, previous = sha(previous_path), read(previous_path)
    check(Path(previous["path_base"]).resolve() == out.resolve(), "Manifest output-directory binding")
    check(previous["final_audit_sha256"] == sha(out / "FINAL_AUDIT.json")
          and previous["test_complete_sha256"] == sha(out / "TEST_COMPLETE.json"),
          "Audit/test completion changed since the previous manifest")
    frozen = read(out / "FROZEN_PROTOCOL.json")
    frozen_paths = set()
    for name, digest in frozen["code"].items():
        path = (ROOT / name).resolve()
        check(sha(path) == digest, f"Frozen code/artifact changed: {name}")
        frozen_paths.add(os.path.relpath(path, out).replace("\\", "/"))
    for name, digest in frozen["evidence"].items():
        path = (out / name).resolve()
        check(sha(path) == digest, f"Frozen historical evidence changed: {name}")
        frozen_paths.add(os.path.relpath(path, out).replace("\\", "/"))
    current = artifact_manifest_entries(out)
    changes = validate_manifest_refresh(previous["files"], current, frozen_paths)
    refresh = dict(unix=time.time(), previous_manifest_sha256=previous_sha,
                   action="delivery_manifest_refresh_only", full_audit_repeated=False,
                   checkpoint_loads=0, model_forward_calls=0,
                   protected_artifacts_hash_rechecked=True, **changes)
    write_artifact_manifest(out, current, refresh)
    print(json.dumps(refresh, indent=2, ensure_ascii=False))


def orbit_key(values):
    p = tuple(map(int, values))
    inverse = tuple(p.index(v) + 1 for v in range(1, len(p) + 1))
    orbit = []
    for base in [p, inverse]:
        for q in [base, base[::-1]]:
            orbit.extend([q, tuple(len(q) + 1 - v for v in q)])
    return min(orbit)


def best_key(row):
    return (-float(row["dev_accuracy"]), float(row["dev_answer_ce"]), int(row["step"]))


def source_lr(step):
    return .00003 if step > 2400 else .0003 * (step / 100 if step <= 100 else
        .1 + .9 * .5 * (1 + math.cos(math.pi * (step - 100) / 2300)))


def audit_freeze_review(out):
    """Small pretraining review; records only evidence available before tests."""
    frozen = read(out / "FROZEN_PROTOCOL.json")
    ready, smoke = read(out / "DATA_READY.json"), read(out / "SMOKE.json")
    math_audit = read(HERE / "mathematics/MATH_AUDIT.json")
    exploratory = read(HERE / "exploratory/exploratory_statistics_audit.json")
    plan, cfg = read(HERE / "stats_plan.json"), frozen["config"]
    check(cfg == read(out / "config.json"), "Configuration binding")
    check(frozen["data_sha256"] == sha(out / "DATA_READY.json"), "Data-ready binding")
    check(frozen["compositions_sha256"] == sha(out / "compositions.csv"), "Composition binding")
    for path, digest in frozen["code"].items():
        check(sha(ROOT / path) == digest, f"Frozen code/artifact changed: {path}")
    for path, digest in frozen["evidence"].items():
        check(sha((out / path).resolve()) == digest, f"Exploratory evidence changed: {path}")
    for path, digest in math_audit["sha256"].items():
        check(sha(HERE / Path(path.replace("\\", "/"))) == digest, "Nested mathematics audit binding")
    check(frozen["statistics_plan_sha256"] == exploratory["plan_sha256"] == sha(HERE / "stats_plan.json"), "Plan binding")
    check(exploratory["statistics_sha256"] == sha(HERE / "paired_statistics.py"), "Analysis code binding")
    check(exploratory["input_sha256"] == sha(HISTORICAL[3] / "results_selected.csv"), "Exploration source binding")
    check(frozen["mathematics_audit_sha256"] == sha(HERE / "mathematics/MATH_AUDIT.json"), "Mathematics binding")
    math_unix = datetime.fromisoformat(math_audit["generated_utc"]).timestamp()
    check(ready["unix"] < smoke["unix"] < frozen["unix"] and math_unix < frozen["unix"], "Freeze evidence chronology")
    check(smoke["status"] == math_audit["status"] == "passed" and smoke["expanded_historical_role_block_orbits_disjoint"],
          "Required smoke/math/orbit check")
    check(not list((out / "data").glob("*_test.npz")) and not (out / "TEST_COMPLETE.json").exists(),
          "Freeze review must precede validation tests")
    seeds = cfg["seeds"]
    check(seeds == [101, 211, 307, 401, 503], "New repeats")
    check([r["seed"] for r in plan["repetitions"]] == seeds, "Plan repeat binding")
    old_config = read(HISTORICAL[3] / "FROZEN_PROTOCOL.json")["config"]
    check(set(seeds).isdisjoint(old_config["seeds"]), "Initialization seed novelty")
    v2_cfg = read(HISTORICAL[1] / "FROZEN_PROTOCOL.json")["config"]
    v3_cfg = read(HISTORICAL[2] / "FROZEN_PROTOCOL.json")["config"]
    old_generator_seeds = set(old_config["data_generation_seeds"].values())
    old_generator_seeds.update([v2_cfg["new_test_seed"], v3_cfg["new_data_seed"],
                                v3_cfg["new_test_seed"], old_config["new_test_seed"]])
    new_generator_seeds = set(cfg["data_generation_seeds"].values()) | set(cfg["test_generation_seeds"].values())
    check(len(new_generator_seeds) == 10 and new_generator_seeds.isdisjoint(old_generator_seeds), "Generator seed novelty")
    old_initial_hashes = set()
    for path in sorted((HISTORICAL[2] / "initial").glob("*.pt")):
        cp = torch.load(path, map_location="cpu", weights_only=False)
        old_initial_hashes.add(state_hash(cp["model"]))
    new_initial_hashes = set()
    for seed in seeds:
        path = out / "initial" / f"seed{seed}.pt"
        check(sha(path) == ready["initials"][str(seed)], "Initial file binding")
        cp = torch.load(path, map_location="cpu", weights_only=False)
        digest = state_hash(cp["model"])
        torch.random.default_generator.manual_seed(seed)
        model = CausalTransformer(vocab_size=188, **cfg["model"])
        check(state_hash(model.state_dict()) == digest, "Fresh initialization independent replay")
        new_initial_hashes.add(digest)
        del model
    check(len(new_initial_hashes) == 5 and new_initial_hashes.isdisjoint(old_initial_hashes), "No old initial weights reused")
    old_pool_hashes = {sha(path) for root in HISTORICAL for path in (root / "data").glob("*.npz")}
    check(all(r["sha256"] not in old_pool_hashes for r in ready["pools"]), "No exact old data reuse")
    old_source_metadata = [read(path) for root in HISTORICAL[2:] for path in (root / "multi").glob("*/completed.json")]
    check(all(m["initial_state_hash"] not in new_initial_hashes for m in old_source_metadata), "No old source initialization compatibility")
    check(all(m["train_sha256"] not in {r["sha256"] for r in ready["pools"]} for m in old_source_metadata),
          "No old source training-data compatibility")
    checkpoint_paths = list((out / "multi").glob("*/*.pt"))
    check(all(path.stat().st_mtime >= frozen["unix"] for path in checkpoint_paths), "Saved formal-training checkpoints precede freeze")
    record = dict(status="passed", reviewed_utc=datetime.now(timezone.utc).isoformat(),
                  freeze_utc=datetime.fromtimestamp(frozen["unix"], timezone.utc).isoformat(),
                  data_ready_unix=ready["unix"], smoke_unix=smoke["unix"], mathematics_unix=math_unix,
                  freeze_unix=frozen["unix"], frozen_file_hashes_checked=len(frozen["code"]),
                  exploratory_evidence_hashes_checked=len(frozen["evidence"]),
                  mathematics_nested_hashes_checked=len(math_audit["sha256"]),
                  statistical_plan_code_exploration_bindings_verified=True,
                  old_initial_state_hashes_compared=len(old_initial_hashes), new_initializations_replayed=5,
                  old_source_metadata_checked=len(old_source_metadata),
                  old_generator_seed_set=sorted(old_generator_seeds), new_generator_seed_set=sorted(new_generator_seeds),
                  new_data_and_initialization_seed_sets_disjoint_from_old=True,
                  saved_formal_checkpoints_after_freeze=len(checkpoint_paths), no_new_test_inputs_or_results_yet=True,
                  model_forward_calls=0, gpu_query_or_execution=False,
                  old_checkpoint_reuse_incompatibility=[
                      "Architecture/encoding and nominal budgets are reusable, but the confirmation requires fresh initializations and fresh input pools.",
                      "Every old v3/v4 source record has an initial parameter hash different from all five confirmation initializations.",
                      "Every old v3/v4 source train-data hash differs from every new confirmation input pool.",
                      "Old adaptations likewise cannot count as independent new repetitions; their initializations and target examples belong to exploration.",
                      "V1 source checkpoints also used the earlier2400-update budget; even budget-compatible later checkpoints fail the fresh-data/fresh-initialization requirement."],
                  scope="Pretraining evidence review only; final checkpoint/result audit awaits all tests and summaries.")
    write(out / "FREEZE_REVIEW.json", record)
    print(json.dumps(record, indent=2, ensure_ascii=False))


def verify_statistical_outputs(out, selected, plan):
    """Recompute contrasts/uncertainty independently of paired_statistics.py."""
    path = out / "confirmation_paired_differences.csv"
    if not path.exists():
        return dict(confirmation_statistics_present=False)
    differences = rows(path)
    summary = rows(out / "confirmation_summary.csv")
    audit = read(out / "confirmation_statistics_audit.json")
    check(audit["input_sha256"] == sha(out / "results_selected.csv")
          and audit["plan_sha256"] == sha(HERE / "stats_plan.json")
          and audit["statistics_sha256"] == sha(HERE / "paired_statistics.py"), "Statistical input/source bindings")
    check(len(differences) == 10 and len(summary) == 6, "Statistical output scope")
    index = {(r["target"], int(r["seed"]), r["background"], r["condition"]): float(r["test_accuracy"])
             for r in selected}
    verified = {}
    seeds = [r["seed"] for r in plan["repetitions"]]
    for contrast in plan["contrasts"]:
        target, old, new = contrast["target"], contrast["original_background"], contrast["new_background"]
        for seed in seeds:
            a = index[(target, seed, "shared", "A")]
            b0, c0 = [index[(target, seed, old, c)] for c in ["B", "C"]]
            b1, c1 = [index[(target, seed, new, c)] for c in ["B", "C"]]
            expected = dict(A_shared=a, B_original=b0, C_original=c0, B_new=b1, C_new=c1,
                            B_minus_A_original=b0-a, B_minus_A_new=b1-a,
                            C_minus_A_original=c0-a, C_minus_A_new=c1-a,
                            B_minus_C_original=b0-c0, B_minus_C_new=b1-c1,
                            delta_B_minus_A=b1-b0, delta_C_minus_A=c1-c0,
                            delta_B_minus_C=b1-c1-b0+c0)
            found = [r for r in differences if r["target"] == target and int(r["seed"]) == seed
                     and r["original_background"] == old and r["new_background"] == new]
            check(len(found) == 1, "Paired interaction coverage")
            for field, value in expected.items():
                close(found[0][field], value, f"Independent paired contrast: {target}/{seed}/{field}")
            verified[(target, seed)] = expected
        for metric in ["delta_B_minus_A", "delta_C_minus_A", "delta_B_minus_C"]:
            values = np.asarray([verified[(target, s)][metric] for s in seeds])
            mean, sd = float(values.mean()), float(values.std(ddof=1))
            row = next(r for r in summary if r["target"] == target and r["metric"] == metric)
            check(int(row["n"]) == 5, "Repeat-level statistical denominator")
            for field, value in [("mean", mean), ("sample_sd", sd), ("min", values.min()), ("max", values.max())]:
                close(row[field], value, "Independent paired summary")
            if metric != "delta_B_minus_C":
                check(not truth(row["primary"]), "Auxiliary statistic mislabeled")
                continue
            check(truth(row["primary"]) and int(row["df"]) == 4, "Primary t-model degrees of freedom")
            se = sd / math.sqrt(5)
            close(row["standard_error"], se, "Primary paired standard error")
            close(row["t_95_critical"], 2.7764451051977987, "95% t critical value")
            q = float(row["t_family_critical"])
            # For df=4, evaluate the CDF by an independently written
            # trigonometric antiderivative of (3/8)*(1+x*x/4)^(-5/2).
            theta = math.atan(q / 2)
            close(.5 + .75 * (math.sin(theta) - math.sin(theta) ** 3 / 3), .9875, "Family t critical value")
            for prefix, critical in [("ci95", 2.7764451051977987), ("family95", q)]:
                close(row[prefix + "_lower"], mean - critical * se, "Paired interval lower limit")
                close(row[prefix + "_upper"], mean + critical * se, "Paired interval upper limit")
            direction = contrast["expected_direction"]
            check(int(row["direction_matching_count"]) == int(np.sum(values * direction > 0)), "Direction replication count")
            loo = [(values.sum() - v) / 4 for v in values]
            close(row["leave_one_mean_min"], min(loo), "Leave-one-repeat-out lower mean")
            close(row["leave_one_mean_max"], max(loo), "Leave-one-repeat-out upper mean")
            signed_sums = [sum((1 if (mask >> i) & 1 else -1) * values[i] for i in range(5)) for mask in range(32)]
            p = sum(abs(s) >= abs(values.sum()) - 1e-12 for s in signed_sums) / 32
            close(row["sign_flip_two_sided_p"], p, "Sign-flip diagnostic")
            nonzero = values[values != 0]
            pos = int(np.sum(nonzero > 0))
            n = len(nonzero)
            p = min(1, 2 * sum(math.comb(n, k) for k in range(min(pos, n-pos)+1)) / (2**n))
            close(row["sign_test_two_sided_p"], p, "Sign diagnostic")
    return dict(confirmation_statistics_present=True, independently_recomputed_paired_interactions=10,
                independently_recomputed_auxiliary_differences=20, independent_pair_level_summaries=6,
                independent_primary_t_intervals_and_sign_diagnostics=2,
                exploratory_and_confirmation_not_pooled=True)


def audit(out):
    started = time.monotonic()
    frozen = read(out / "FROZEN_PROTOCOL.json")
    cfg, groups = frozen["config"], frozen["groups"]
    ready, smoke = read(out / "DATA_READY.json"), read(out / "SMOKE.json")
    selection = read(out / "SELECTIONS_FROZEN.json")
    test_audit, finish = read(out / "test_data_audit.json"), read(out / "TEST_COMPLETE.json")
    seeds, targets = cfg["seeds"], cfg["targets"]
    expected_backgrounds = {"fixed_points": ["g0_v3", "g2_deficiencies"],
                            "longest_decreasing_run": ["g0_v3", "g1_recoils"]}
    lrs = cfg["learning_rates"]
    check(seeds == [101, 211, 307, 401, 503], "Unexpected repeat scope")
    check(targets == ["fixed_points", "longest_decreasing_run"], "Unexpected targets")
    check(lrs == [.0003, .0001, .00003] and cfg["target_samples"] == 256, "Adaptation scope drift")
    check(cfg == read(out / "config.json"), "Config disagreement")
    check(cfg["vocab_size"] == len(VOCABULARY) == 188 and cfg["n"] == 10, "Vocab/length mismatch")
    check(cfg["pretrain_updates"] == 4800 and cfg["adaptation_updates"] == 800 and cfg["batch"] == 64,
          "Training budget drift")
    check(smoke["status"] == "passed" and smoke["common_task_sequences_equal"], "Smoke incomplete")
    check(smoke["expanded_historical_role_block_orbits_disjoint"], "Expanded orbit smoke check")
    check(read(HERE / "mathematics/MATH_AUDIT.json")["status"] == "passed", "Mathematical checks incomplete")
    check(frozen["mathematics_audit_sha256"] == sha(HERE / "mathematics/MATH_AUDIT.json"), "Mathematical audit binding")
    plan = read(HERE / "stats_plan.json")
    check(frozen["statistics_plan_sha256"] == sha(HERE / "stats_plan.json"), "Statistical plan binding")
    check([r["seed"] for r in plan["repetitions"]] == seeds, "Statistical repetition scope")
    for i, unit in enumerate(plan["repetitions"]):
        check(unit["model_initialization_seed"] == seeds[i]
              and unit["data_generation_seed"] == 2026101901 + i
              and unit["test_data_seed"] == 2026101951 + i, "Statistical generator/initialization seeds")
    expected_contrasts = [("fixed_points", "g0_v3", "g2_deficiencies", 1),
                         ("longest_decreasing_run", "g0_v3", "g1_recoils", -1)]
    check([(r["target"], r["original_background"], r["new_background"], r["expected_direction"])
           for r in plan["contrasts"]] == expected_contrasts, "Statistical primary contrast scope")
    check(frozen["data_sha256"] == sha(out / "DATA_READY.json"), "Data manifest changed")
    check(frozen["compositions_sha256"] == sha(out / "compositions.csv"), "Composition table changed")
    for path, digest in frozen["code"].items():
        check(sha(ROOT / path) == digest, f"Frozen source changed: {path}")
    for path, digest in frozen["evidence"].items():
        check(sha((out / path).resolve()) == digest, f"Frozen evidence changed: {path}")
    check(selection["protocol_sha256"] == sha(out / "FROZEN_PROTOCOL.json"), "Selection protocol binding")
    check(test_audit["selection_sha256"] == finish["selection_sha256"] == sha(out / "SELECTIONS_FROZEN.json"),
          "Test selection binding")
    chronology = [ready["unix"], smoke["unix"], frozen["unix"], read(out / "PRETRAIN_COMPLETE.json")["unix"],
                  read(out / "ADAPT_COMPLETE.json")["unix"], selection["unix"], test_audit["unix"], finish["unix"]]
    check(chronology == sorted(chronology), "Freeze/train/select/test chronology")
    preserved = read(out / "preservation.json")
    expected_preserved = {p.relative_to(ROOT).as_posix() for name in
                          ["permworld_pilot", "pilot_v2_diagnostics", "pilot_v3_multitarget", "pilot_v4_backgrounds"]
                          for p in (HERE.parent / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    expected_preserved.add("PILOT_REPORT.md")
    check(set(preserved) == expected_preserved, "Historical preservation snapshot completeness")
    for path, digest in preserved.items():
        check(sha(ROOT / path) == digest, f"Historical artifact changed: {path}")
    print("[audit] Frozen protocol and all historical bytes verified.", flush=True)

    def resolve(path):
        return (out / path).resolve()

    def reference(path):
        return os.path.relpath(path.resolve(), out).replace("\\", "/")

    group_map = {(g["target"], g["background"]): g for g in groups}
    check(len(groups) == len(group_map) == 4, "Composition coverage")
    v4_groups = {(g["target"], g["background"]): g for g in read(HISTORICAL[3] / "FROZEN_PROTOCOL.json")["groups"]}
    expected_cells = {(seed, target, "shared", "A") for seed in seeds for target in targets}
    expected_cells |= {(seed, target, background, condition) for seed in seeds for target in targets
                       for background in expected_backgrounds[target] for condition in ["B", "C"]}
    expected_source_cells = {cell for cell in expected_cells if cell[-1] != "A"}
    for target in targets:
        for background in expected_backgrounds[target]:
            group = group_map[(target, background)]
            check(group["sources"] == v4_groups[(target, background)]["sources"], "Frozen v4 composition changed")
            for tasks in group["sources"].values():
                check(len(set(tasks)) == 4 and not set(tasks).intersection(targets), "Task count or target leakage")
            check(group["sources"]["B"][1:] == group["sources"]["C"][1:] == group["common"],
                  "B/C common tasks differ")
    task_index = {t: i for i, t in enumerate(PROPERTY_FUNCTIONS)}
    label_checks = 0

    def verify_arrays(p, y, count, label):
        nonlocal label_checks
        check(p.shape == (count, 10) and y.shape == (count, 32), f"Data shape: {label}")
        check(np.all(np.sort(p, axis=1) == np.arange(1, 11)), f"Invalid permutation: {label}")
        for task in PROPERTY_FUNCTIONS:
            answers = np.asarray([PROPERTY_FUNCTIONS[task](tuple(map(int, q))) for q in p], dtype=np.int16)
            check(np.array_equal(answers, y[:, task_index[task]]), f"Label mismatch: {label}/{task}")
            label_checks += count

    historical_keys = set()
    historical_paths = []
    for root in HISTORICAL:
        for path in sorted((root / "data").glob("*.npz")):
            historical_paths.append(path)
            with np.load(path, allow_pickle=False) as archive:
                historical_keys.update(orbit_key(q) for q in archive["p"])
    check({reference(path): sha(path) for path in historical_paths} == ready["historical_data_files"],
          "Historical exclusion-file manifest")
    check(len(historical_keys) == ready["historical_excluded_orbits"] and ready["all_pools_fresh"]
          and not ready["exact_v3_reuse"], "Fresh input isolation declaration")
    pool_data, pool_hashes, current_keys = {}, {}, set()
    generation_seen = historical_keys.copy()

    def replay_pool(rng, count):
        accepted = []
        while len(accepted) < count:
            value = tuple(map(int, rng.permutation(10) + 1))
            key = orbit_key(value)
            if key in generation_seen:
                continue
            generation_seen.add(key)
            accepted.append(value)
        return np.asarray(accepted, dtype=np.int16)

    expected_pool_order = [(seed, role) for seed in seeds for role in
                           ["source_train", "source_dev", "target_train", "target_dev"]]
    check([(r["seed"], r["role"]) for r in ready["pools"]] == expected_pool_order, "Data generation order")
    rng = None
    previous_seed = None
    for row in ready["pools"]:
        seed, role = row["seed"], row["role"]
        expected_data_seed = 2026101901 + seeds.index(seed)
        check(row["data_generation_seed"] == cfg["data_generation_seeds"][str(seed)] == expected_data_seed,
              "Data generation seed")
        if seed != previous_seed:
            rng = np.random.default_rng(expected_data_seed)
            previous_seed = seed
        path = out / "data" / f"s{seed}_{role}.npz"
        check(sha(path) == row["sha256"], f"Data SHA: {path}")
        count = dict(source_train=8192, source_dev=1024, target_train=256, target_dev=1024)[role]
        check(row["count"] == count, "Role sample count")
        with np.load(path, allow_pickle=False) as archive:
            p, y = archive["p"], archive["y"]
        verify_arrays(p, y, count, path.name)
        check(hashlib.sha256(p.tobytes()).hexdigest() == row["input_sha256"], "Input tensor SHA")
        keys = {orbit_key(q) for q in p}
        check(len(keys) == count and keys.isdisjoint(historical_keys | current_keys), "Train/dev historical or role/seed orbit leakage")
        check(np.array_equal(p, replay_pool(rng, count)), "Independent data RNG replay mismatch")
        current_keys.update(keys)
        pool_data[(seed, role)], pool_hashes[(seed, role)] = (p, y), row["sha256"]
    check(len(pool_data) == 20, "Train/development pool count")
    blocked = historical_keys | current_keys
    check(ready["total_excluded_after_generation"] == len(blocked), "Train/development exclusion count")
    check(test_audit["excluded_orbits"] == len(blocked), "Historical exclusion count")
    check([r["seed"] for r in test_audit["pools"]] == seeds, "Test seed/order coverage")
    for row in test_audit["pools"]:
        seed = row["seed"]
        path = out / "data" / f"s{seed}_test.npz"
        check(sha(path) == row["sha256"], "Test SHA changed")
        expected_test_seed = 2026101951 + seeds.index(seed)
        check(row["data_generation_seed"] == cfg["test_generation_seeds"][str(seed)] == expected_test_seed,
              "Independent new test generation seed")
        with np.load(path, allow_pickle=False) as archive:
            p, y = archive["p"], archive["y"]
        verify_arrays(p, y, 2048, path.name)
        keys = {orbit_key(q) for q in p}
        check(len(keys) == row["unique_orbits"] == row["count"] == 2048 and keys.isdisjoint(blocked), "New test orbit isolation")
        check(np.array_equal(p, replay_pool(np.random.default_rng(expected_test_seed), 2048)), "Independent test RNG replay mismatch")
        blocked.update(keys)
        pool_data[(seed, "test")] = (p, y)
    print("[audit] All25 input pools, mathematical labels, orbit exclusions and RNG replay verified.", flush=True)

    checkpoints, expected_shapes = {}, None

    def checkpoint(path, expected_sha=None, step=None, optimizer_step=None):
        nonlocal expected_shapes
        path = path.resolve()
        digest = sha(path)
        if expected_sha:
            check(digest == expected_sha, f"Checkpoint SHA: {path}")
        cp = torch.load(path, map_location="cpu", weights_only=False)
        state = cp["model"]
        shapes = {k: tuple(v.shape) for k, v in state.items()}
        if expected_shapes is None:
            expected_shapes = shapes
        check(shapes == expected_shapes and shapes["token_embedding.weight"] == (188, 256), "Model shape mismatch")
        check(torch.equal(state["token_embedding.weight"], state["lm_head.weight"]), "Tied output mismatch")
        check(all(not v.is_floating_point() or bool(torch.isfinite(v).all()) for v in state.values()), "Nonfinite model parameters")
        if step is not None:
            check(cp.get("step", 0) == step, f"Checkpoint step: {path}")
        steps = sorted({int(s["step"]) for s in cp.get("optimizer", {}).get("state", {}).values()})
        if optimizer_step is not None:
            check(steps == [optimizer_step], f"Optimizer counter: {path}")
            for group in cp["optimizer"]["param_groups"]:
                check(group["weight_decay"] == .01 and tuple(group["betas"]) == (.9, .999)
                      and group["eps"] == 1e-8, "Optimizer configuration")
        parameter_hash = state_hash(state)
        record = dict(path=reference(path), sha256=digest, state_sha256=parameter_hash,
                      step=cp.get("step", 0), optimizer_steps=";".join(map(str, steps)), cpu_loaded=True,
                      tied_embedding_values_equal=True)
        check(path not in checkpoints or checkpoints[path] == record, "Checkpoint changed during audit")
        checkpoints[path] = record
        return cp, parameter_hash

    initial_states, initial_hashes = {}, {}
    for seed in seeds:
        path = out / "initial" / f"seed{seed}.pt"
        cp, initial_hashes[seed] = checkpoint(path, ready["initials"][str(seed)])
        check(cp["seed"] == cfg["model_initialization_seeds"][str(seed)] == seed, "Model initialization seed")
        torch.random.default_generator.manual_seed(seed)
        independent = CausalTransformer(vocab_size=188, **cfg["model"])
        check(state_hash(independent.state_dict()) == initial_hashes[seed], "Independent CPU initialization replay")
        check(sum(v.numel() for v in independent.parameters()) == 3240448, "Parameter count changed")
        initial_states[seed] = cp["model"]
        del independent
    check(len(set(initial_hashes.values())) == 5, "Distinct repeat initializations")
    print("[audit] Five fresh initializations independently reconstructed on CPU.", flush=True)

    def schedule(seed, task_count, updates, count, offset):
        rng, digest = np.random.default_rng(seed + offset), hashlib.sha256()
        indices, slots = [], []
        for _ in range(updates):
            index = rng.choice(count, 64, replace=False)
            slot = np.tile(np.arange(task_count), 64 // task_count)
            rng.shuffle(slot)
            digest.update(index.tobytes()); digest.update(slot.tobytes())
            indices.append(index); slots.append(slot)
        return digest.hexdigest(), rng.bit_generator.state, np.asarray(indices), np.asarray(slots)

    source_schedules = {s: schedule(s, 4, 4800, 8192, 1000) for s in seeds}
    adaptation_schedules = {s: schedule(s, 1, 800, 256, 2000) for s in seeds}
    slot_rows, exposure_rows = rows(out / "sampling_slots.csv"), rows(out / "source_exposures.csv")
    check(len(slot_rows) == 20 and len(exposure_rows) == 160, "Slot/exposure table size")
    slot_evidence = {}
    for seed in seeds:
        _, _, indices, slots = source_schedules[seed]
        p, y = pool_data[(seed, "source_train")]
        for slot in range(4):
            stream = indices[slots == slot]
            evidence = dict(exposures=len(stream), unique_inputs=len(set(stream.tolist())),
                            index_stream_sha256=hashlib.sha256(stream.tobytes()).hexdigest(),
                            input_stream_sha256=hashlib.sha256(p[stream].tobytes()).hexdigest())
            saved = next(r for r in slot_rows if int(r["seed"]) == seed and int(r["slot"]) == slot)
            for field, value in evidence.items():
                check(str(saved[field]) == str(value), f"Slot stream mismatch: {seed}/{slot}/{field}")
            slot_evidence[(seed, slot)] = (stream, evidence)

    source_metas, source_states = {}, {}
    for meta_path in sorted((out / "multi").glob("*/completed.json")):
        meta = read(meta_path)
        cell = (meta["seed"], meta["target"], meta["background"], meta["condition"])
        seed, target, background, condition = cell
        check(cell in expected_source_cells and cell not in source_metas, "Source run coverage")
        tasks = group_map[(target, background)]["sources"][condition]
        check(meta["tasks"] == tasks and meta["kind"] == "multi" and meta["new_compute"], "Source task/new training tuple")
        check(meta["updates"] == 4800 and meta["input_tokens"] == 8601600 and meta["supervised_tokens"] == 614400
              and meta["sample_exposures"] == 307200 and meta["task_exposures"] == 76800
              and meta["task_fraction"] == .25, "Source budget equality")
        check(meta["initial_state_hash"] == initial_hashes[seed], "Source initial weights")
        check(meta["train_sha256"] == pool_hashes[(seed, "source_train")]
              and meta["dev_sha256"] == pool_hashes[(seed, "source_dev")], "Source data binding")
        check(meta["schedule_sha256"] == source_schedules[seed][0], "Paired source sampler")
        cp, final_hash = checkpoint(resolve(meta["checkpoint"]), meta["checkpoint_sha256"], 4800, 4800)
        check(final_hash == meta["model_state_hash"] and cp["numpy_rng"] == source_schedules[seed][1], "Source parameter/RNG hash")
        check(cp["seed"] == seed and cp["config"] == cfg and cp["tasks"] == tasks, "Source checkpoint identity/config")
        close(cp["optimizer"]["param_groups"][0]["lr"], source_lr(4800), "Source final LR")
        for step in [1200, 2400]:
            partial, _ = checkpoint(meta_path.parent / f"step{step}.pt", step=step, optimizer_step=step)
            check(partial["tasks"] == tasks and partial["seed"] == seed and partial["config"] == cfg, "Partial source identity")
            close(partial["optimizer"]["param_groups"][0]["lr"], source_lr(step), "Source LR trajectory")
            del partial
        curve = rows(meta_path.parent / "curve.csv")
        check(len(curve) == 48, "Six evaluation points times four sources times two splits")
        for slot, task in enumerate(tasks):
            stream, evidence = slot_evidence[(seed, slot)]
            check(meta["actual_unique_per_task"][task] == evidence["unique_inputs"], "Source unique count")
            exposure = next(r for r in exposure_rows if (int(r["seed"]), r["target"], r["background"], r["condition"], int(r["slot"])) == (*cell, slot))
            check(exposure["task"] == task and int(exposure["exposures"]) == 76800
                  and int(exposure["input_tokens"]) == 76800 * 28 and int(exposure["supervised_tokens"]) == 153600
                  and float(exposure["task_fraction"]) == .25 and truth(exposure["new_compute"]), "Source exposure budget")
            for field in ["index_stream_sha256", "input_stream_sha256"]:
                check(exposure[field] == evidence[field], "Unpaired task slot exposure")
            label_digest = hashlib.sha256(pool_data[(seed, "source_train")][1][stream, task_index[task]].tobytes()).hexdigest()
            check(exposure["label_stream_sha256"] == label_digest, "Task label stream binding")
            train_y = pool_data[(seed, "source_train")][1][:, task_index[task]]
            mode = int(np.bincount(train_y).argmax())
            for split, count in [("train", 8192), ("dev", 1024)]:
                task_curve = [r for r in curve if r["task"] == task and r["split"] == split]
                check([int(r["step"]) for r in task_curve] == [0, 600, 1200, 2400, 3600, 4800], "Source curve schedule")
                eval_y = pool_data[(seed, "source_train" if split == "train" else "source_dev")][1][:, task_index[task]]
                for point in task_curve:
                    check(int(point["count"]) == count and int(point["length"]) == 10, "Source curve count/length")
                    close(point["majority_accuracy"], np.mean(eval_y == mode), "Source majority baseline")
                    check(0 <= float(point["accuracy"]) <= 1, "Source accuracy range")
                    for field in ["answer_ce", "eos_ce", "two_token_ce"]:
                        check(math.isfinite(float(point[field])) and float(point[field]) >= 0, "Invalid source loss")
        source_metas[cell], source_states[cell] = meta, cp["model"]
        del cp
    check(set(source_metas) == expected_source_cells and len(source_metas) == 40, "Expected40 source cells")
    print("[audit] Forty source runs, checkpoints, curves and paired slot streams verified.", flush=True)

    metas, initial_reconstructions = {}, {}
    selections = selection["selections"]
    check(len(selections) == 50, "Unique selection count")
    check({(r["seed"], r["target"], r["background"], r["condition"]) for r in selections} == expected_cells,
          "A must be selected once in shared background")
    for meta_path in sorted((out / "adapt").glob("*/completed.json")):
        meta = read(meta_path)
        cell = (meta["seed"], meta["target"], meta["background"], meta["condition"])
        seed, target, background, condition = cell
        lr = meta["lr"]
        check(cell in expected_cells and lr in lrs and (*cell, lr) not in metas, "Adaptation candidate coverage")
        check(meta["updates"] == 800 and meta["input_tokens"] == 1433600 and meta["supervised_tokens"] == 102400
              and meta["sample_exposures"] == 51200 and meta["new_compute"], "Adaptation budget/new training equality")
        check(meta["schedule_sha256"] == adaptation_schedules[seed][0], "Adaptation paired sampler")
        check(meta["v5_train_sha256"] == pool_hashes[(seed, "target_train")]
              and meta["v5_dev_sha256"] == pool_hashes[(seed, "target_dev")], "Adaptation data binding")
        init = meta["initialization"]
        check(init["optimizer_entries"] == 0 and init["optimizer_state_reset"] and init["new_output_head"] is False,
              "Fresh adaptation optimizer/head")
        token = TOKEN_TO_ID[f"<{target.upper()}>"]
        initial_row = initial_states[seed]["token_embedding.weight"][token]
        check(init["reset_row_ids"] == [token] and init["target_row_hash"] == tensor_hash(initial_row), "Paired target token initialization")
        expected_source = out / "initial" / f"seed{seed}.pt" if condition == "A" else resolve(source_metas[cell]["checkpoint"])
        check(resolve(init["source_path"]) == expected_source.resolve() and sha(expected_source) == init["source_sha256"],
              "Source path/hash")
        check(init["train_sha256"] == pool_hashes[(seed, "target_train")]
              and init["dev_sha256"] == pool_hashes[(seed, "target_dev")], "Adaptation data hashes")
        if cell not in initial_reconstructions:
            state = initial_states[seed] if condition == "A" else source_states[cell]
            initial_reconstructions[cell] = (state_hash(state), state_hash(state, token, initial_row))
        loaded, restored = initial_reconstructions[cell]
        check(init["loaded_state_hash"] == init["source_state_hash"] == loaded
              and init["adapt_initial_state_hash"] == restored, "Loaded/reset source weights")
        curve = rows(meta_path.parent / "curve.csv")
        check([int(r["step"]) for r in curve] == list(range(0, 801, 100)), "Equal adaptation evaluation budget")
        mode = int(np.bincount(pool_data[(seed, "target_train")][1][:, task_index[target]]).argmax())
        for row in curve:
            check((int(row["seed"]), row["target"], row["background"], row["condition"]) == cell, "Curve identity")
            close(row["lr"], lr, "Curve learning rate")
            for split, count in [("train", 256), ("dev", 1024)]:
                check(int(row[split + "_count"]) == count, "Curve sample count")
                yy = pool_data[(seed, "target_train" if split == "train" else "target_dev")][1][:, task_index[target]]
                close(row[split + "_majority_accuracy"], np.mean(yy == mode), "Curve majority baseline")
                for field in ["answer_ce", "eos_ce", "two_token_ce"]:
                    check(math.isfinite(float(row[split + "_" + field])) and float(row[split + "_" + field]) >= 0,
                          "Invalid development/training loss")
        best = min((r for r in curve if int(r["step"]) > 0), key=best_key)
        endpoint = curve[-1]
        for which, expected in [("best", best), ("endpoint", endpoint)]:
            for field in ["step", "train_accuracy", "dev_accuracy", "dev_answer_ce"]:
                close(meta[which][field], expected[field], "Best/endpoint curve mismatch")
            cp, digest = checkpoint(resolve(meta[which + "_path"]), meta[which + "_sha256"],
                                    int(expected["step"]), 800 if which == "endpoint" else None)
            check(cp["seed"] == seed and cp["condition"] == condition and cp["lr"] == lr, "Adaptation checkpoint identity")
            check(cp["config"] == cfg and cp["target"] == target and cp["background"] == background, "Adaptation configuration")
            for field in ["step", "train_accuracy", "dev_accuracy", "dev_answer_ce"]:
                close(cp["metrics"][field], meta[which][field], "Saved checkpoint metrics")
            if which == "endpoint":
                check(digest == meta["final_model_state_hash"] and cp["numpy_rng"] == adaptation_schedules[seed][1],
                      "Adaptation final parameter/RNG hash")
                close(cp["optimizer"]["param_groups"][0]["lr"], .1 * lr, "Adaptation final LR")
            del cp
        metas[(*cell, lr)] = meta
    check(len(metas) == 150 and sum(key[3] == "A" for key in metas) == 30, "Unique150 candidates/shared30 A candidates")
    print("[audit] All150 adaptation candidates, weights and reset-optimizer evidence verified.", flush=True)
    for cell in expected_cells:
        candidates = [metas[(*cell, lr)] for lr in lrs]
        chosen = min(candidates, key=lambda m: (*best_key(m["best"]), lrs.index(m["lr"])))
        saved = next(r for r in selections if (r["seed"], r["target"], r["background"], r["condition"]) == cell)
        check(chosen["lr"] == saved["lr"] and chosen["best"]["step"] == saved["step"], "Independent development selection mismatch")
        close(chosen["best"]["dev_accuracy"], saved["dev_accuracy"], "Selected development accuracy")
        close(chosen["best"]["dev_answer_ce"], saved["dev_ce"], "Selected development CE")
        for which in ["best", "endpoint"]:
            check(chosen[which + "_path"] == saved[which + "_path"] and chosen[which + "_sha256"] == saved[which + "_sha256"],
                  "Selection checkpoint hash binding")

    raw, selected = rows(out / "results_all_candidates.csv"), rows(out / "results_selected.csv")
    check(len(raw) == finish["candidate_rows"] == 300 and len(selected) == finish["selected_rows"] == 50, "Test row totals")
    check(selected == [r for r in raw if truth(r["selected_primary"])], "Selected raw rows mismatch")
    expected_result_keys = {(*cell, lr, which) for cell in expected_cells for lr in lrs for which in ["best", "endpoint"]}
    actual_result_keys = {(int(r["seed"]), r["target"], r["background"], r["condition"], float(r["lr"]), r["checkpoint"]) for r in raw}
    check(actual_result_keys == expected_result_keys and len(actual_result_keys) == len(raw), "Full test candidate grid")
    for seed in seeds:
        check(rows(out / f"results_seed{seed}.csv") == [r for r in raw if int(r["seed"]) == seed], "Per-repeat raw result mismatch")
    for row in raw:
        cell = (int(row["seed"]), row["target"], row["background"], row["condition"])
        seed, target, background, condition = cell
        lr, which = float(row["lr"]), row["checkpoint"]
        meta = metas[(*cell, lr)]
        choice = next(r for r in selections if (r["seed"], r["target"], r["background"], r["condition"]) == cell)
        check(row["checkpoint_sha256"] == meta[which + "_sha256"], "Test checkpoint provenance")
        check(truth(row["selected_lr"]) == (lr == choice["lr"])
              and truth(row["selected_primary"]) == (lr == choice["lr"] and which == "best"), "Result selection flags")
        check(int(row["adapt_step"]) == meta[which]["step"] and int(row["test_count"]) == 2048, "Test counts")
        check(truth(row["new_compute"]), "All validation models must be new training")
        check(int(row["model_initialization_seed"]) == seed
              and int(row["data_generation_seed"]) == 2026101901 + seeds.index(seed), "Result repeat seed provenance")
        for field in ["dev_accuracy", "train_accuracy"]:
            close(row[field], meta[which][field], "Result development/train metric binding")
        close(row["dev_ce"], meta[which]["dev_answer_ce"], "Result development CE binding")
        accuracy = float(row["test_accuracy"])
        check(0 <= accuracy <= 1 and abs(accuracy * 2048 - round(accuracy * 2048)) < 1e-8, "Invalid test accuracy")
        for field in ["test_answer_ce", "test_eos_ce", "test_two_token_ce"]:
            check(math.isfinite(float(row[field])) and float(row[field]) >= 0, "Invalid test loss")
        close(row["test_two_token_ce"], (float(row["test_answer_ce"]) + float(row["test_eos_ce"])) / 2, "Test CE aggregation")
        mode = int(np.bincount(pool_data[(seed, "target_train")][1][:, task_index[target]]).argmax())
        close(row["majority_accuracy"], np.mean(pool_data[(seed, "test")][1][:, task_index[target]] == mode), "Test majority baseline")

    statistical_audit = verify_statistical_outputs(out, selected, plan)
    record = dict(status="passed", audit_unix=time.time(), audit_seconds=time.monotonic() - started,
        **statistical_audit,
        cpu_only_checkpoint_loads=True, model_forward_calls=0, tests_forwarded_by_this_audit=0,
        historical_files_byte_preserved=len(preserved), frozen_code_and_evidence_verified=True,
        chronological_freeze_before_new_training_and_test=True, independent_train_dev_test_rng_replays=25,
        independent_cpu_initialization_replays=5, full_orbit_historical_and_cross_repeat_isolation_verified=True,
        mathematical_labels_recomputed=label_checks, targets=targets, backgrounds=expected_backgrounds,
        paired_repeat_blocks=seeds, source_runs=40, source_runs_new=40, source_runs_reused=0,
        source_per_slot_input_and_label_streams_verified=160, adaptation_candidates=150,
        adaptation_candidates_new=150, adaptation_candidates_reused=0, shared_A_candidates=30,
        independently_recomputed_selections=50, selected_test_rows=50, candidate_test_rows=300,
        logical_ABC_display_rows=60, shared_A_not_independent_background_observations=True,
        checkpoints_cpu_loaded=len(checkpoints), source_loaded_parameter_and_target_row_reset_hashes_verified=True,
        fresh_adaptation_optimizer_records_and_saved_counters_verified=True,
        new_pretrain_updates=sum(m["updates"] for m in source_metas.values()),
        new_adaptation_updates=sum(m["updates"] for m in metas.values()),
        new_pretrain_input_tokens=sum(m["input_tokens"] for m in source_metas.values()),
        new_adaptation_input_tokens=sum(m["input_tokens"] for m in metas.values()),
        new_pretrain_seconds=sum(m["seconds"] for m in source_metas.values()),
        new_adaptation_seconds=sum(m["seconds"] for m in metas.values()),
        limitations=["Hypotheses and cases were selected after v4 exploratory outcomes; new repeats form the validation evidence.",
            "Repeat blocks jointly vary data and initialization. Their separate variance components cannot be estimated.",
            "Full-orbit exclusion couples rejection sampling across blocks; repeat RNG streams and initialization seeds are distinct.",
            "A is shared across backgrounds. Display duplication does not add independent observations.",
            "No new single-task references were trained for these repeat pools.",
            "This CPU audit checks saved evidence and weights; it does not independently rerun neural inference."])
    write(out / "FINAL_AUDIT.json", record)
    write(out / "audit_environment.json", dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
        numpy=np.__version__, cuda_build=torch.version.cuda, cpu_audit_threads=torch.get_num_threads(),
        training_precision=cfg["precision"], gpu_queried_or_executed=False))
    csvwrite(out / "checkpoint_audit.csv", list(checkpoints.values()))
    write_artifact_manifest(out)
    print(json.dumps(record, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "artifacts")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze-review", action="store_true", help="Review already frozen pretraining evidence before validation tests")
    mode.add_argument("--refresh-manifest", action="store_true",
                      help="After a passed full audit, hash-check protected artifacts and refresh nonfrozen Markdown/PNG/SVG inventory only; does not repeat audit")
    args = parser.parse_args()
    action = audit_freeze_review if args.freeze_review else refresh_artifact_manifest if args.refresh_manifest else audit
    action(args.out.resolve())


if __name__ == "__main__":
    main()
