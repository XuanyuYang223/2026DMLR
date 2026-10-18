"""Read-only inventory of previous validation evidence; no model training or test reads."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def rows(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def js(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    records = []
    base = dict(architecture="transformer", encoding="one_line", d_model=256,
                layers=4, heads=8, vocab_size=188, dropout=0.1,
                train_accuracy="", dev_accuracy="", dev_answer_ce="", majority_accuracy="",
                checkpoint_local=False, checkpoint_path="", checkpoint_sha256="",
                training_compatible=False, model_definition_compatible=True)

    def add(**values):
        records.append(base | values)

    path = "reference/neurips/results/property-task-geometry/cka/run_provenance.csv"
    for r in rows(path):
        if r["kind"] != "specialist":
            continue
        add(reference_id=r["logical_run_id"], task=r["tasks"], record_type="single_task",
            source="neurips_geometry", source_file=path, model_seed=r["seed"],
            train_lengths="2..30", metric_length="mixed_2..30", train_unique=490000,
            dev_count=160, updates=20000, exposures=1280000, lr=0.0003, warmup=1000,
            dev_accuracy=r["seen_task_sequence_accuracy_macro"],
            metric="teacher_forced_answer_and_EOS_exact", evaluation_split="validation",
            checkpoint_sha256=r["checkpoint_sha256"], curves_source="not_present_in_checkout",
            reuse_policy="background_validation_evidence_only",
            caveat="Same architecture/labels; different lengths, train pool, update and warmup budgets; checkpoint bytes absent; exact metric includes EOS.")

    prefix = "reference/2026TMLR/diagnostics/permworld_eight_relations_20261010"
    manifests = {r["run_id"]: r for r in rows(prefix + "/completed_checkpoint_manifest.csv")}
    for r in rows(prefix + "/source_learning.csv"):
        if not (r["architecture"] == "transformer" and r["encoding"] == "one_line"
                and r["condition"] == "trained" and r["split"] == "validation"
                and r["action"] == "identity" and r["length"] == "10"):
            continue
        m = manifests[r["run_id"]]
        add(reference_id=r["run_id"], task=r["task"], record_type="single_task",
            source="TMLR_eight_relations", source_file=prefix + "/source_learning.csv",
            model_seed=m["model_seed"], train_lengths="10..30", metric_length=10,
            train_unique=21 * 4096, dev_count=r["samples"], updates=20000,
            exposures=m["exposures"], lr=0.0003, warmup=1000,
            dev_accuracy=r["answer_accuracy"], dev_answer_ce=r["answer_ce"],
            majority_accuracy=r["length_majority_accuracy"], metric="full_vocab_answer",
            evaluation_split="validation", checkpoint_path=m["checkpoint_path"],
            checkpoint_sha256=m["checkpoint_sha256"],
            curves_source="reference/2026TMLR/diagnostics/permworld_learnability_20261010/old_learning_diagnosis_curves.csv (original four tasks only)",
            reuse_policy="background_n10_validation_evidence_only",
            caveat="Same model including dropout; mixed-length training and 20k updates; only one initializer; checkpoint bytes absent; not a matched pilot-budget control.")

    prefix = "reference/2026TMLR/diagnostics/permworld_learnability_20261010"
    summaries = {r["run_id"]: r for r in rows(prefix + "/model_learning_summary.csv")}
    for r in rows(prefix + "/validation_length.csv"):
        if not (r["architecture"] == "transformer" and r["encoding"] == "one_line"
                and r["step"] == "8000" and r["length"] == "10"):
            continue
        m = summaries[r["run_id"]]
        add(reference_id=r["run_id"], task=r["task"], record_type="single_task",
            source="TMLR_learning_development", source_file=prefix + "/validation_length.csv",
            model_seed=m["model_seed"], train_lengths="10,11,12", metric_length=10,
            train_unique=6144, dev_count=r["samples"], updates=8000, exposures=m["exposures"],
            lr=0.001, warmup=400, dropout=0, model_definition_compatible=False,
            dev_accuracy=r["answer_accuracy"], dev_answer_ce=r["answer_ce"],
            majority_accuracy=r["length_baseline_accuracy"], metric="full_vocab_answer",
            evaluation_split="validation", curves_source=prefix + "/validation_curves.csv",
            reuse_policy="background_protocol_sensitivity_only",
            caveat="Dropout, learning rate, length window, train pool, warmup and budget differ simultaneously; no matched causal comparison.")

    prefix = "experiments/permworld_pilot/runs/pilot_v1"
    r = js(prefix + "/development_result.json")
    add(reference_id="pilot_v1_fixed_points_development", task="fixed_points", record_type="single_task",
        source="pilot_v1", source_file=prefix + "/development_result.json", model_seed=910,
        train_lengths="10", metric_length=10, train_unique=8192, dev_count=1024,
        updates=2400, exposures=153600, lr=0.0003, warmup=100,
        dev_accuracy=r["result"]["accuracy"], dev_answer_ce=r["result"]["ce"],
        majority_accuracy=r["majority_accuracy"], metric="full_vocab_answer", evaluation_split="dev",
        checkpoint_local=True, checkpoint_path=prefix + "/development/target.pt",
        checkpoint_sha256=hashlib.sha256((ROOT / prefix / "development/target.pt").read_bytes()).hexdigest(),
        curves_source=prefix + "/development/target.json", training_compatible="partially_2400_steps",
        reuse_policy="reuse_as_original_target_learnability_reference",
        caveat="Fully compatible architecture and fixed n10; development seed910/optimizer seed911; 2400 updates rather than 4800; not a source-task control.")

    for path in sorted((ROOT / "experiments/pilot_v2_diagnostics/artifacts/adapt").glob("s*_A_p0_*/completed.json")):
        r = json.loads(path.read_text())
        e = r["endpoint"]
        cp = path.parent / "endpoint.pt"
        add(reference_id="pilot_v2_" + r["run_id"], task="fixed_points", record_type="single_task",
            source="pilot_v2", source_file=path.relative_to(ROOT).as_posix(), model_seed=r["seed"],
            train_lengths="10", metric_length=10, train_unique=256, dev_count=e["dev_count"],
            updates=800, exposures=51200, lr=r["lr"], warmup=100,
            train_accuracy=e["train_accuracy"], dev_accuracy=e["dev_accuracy"],
            dev_answer_ce=e["dev_answer_ce"], majority_accuracy=e["dev_majority_accuracy"],
            metric="full_vocab_answer", evaluation_split="dev", checkpoint_local=cp.exists(),
            checkpoint_path=cp.relative_to(ROOT).as_posix(),
            checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
            curves_source=(path.parent / "curve.csv").relative_to(ROOT).as_posix(),
            training_compatible="adaptation_only", reuse_policy="reuse_as_previous_256_sample_target_reference",
            caveat="Compatible previous adaptation; 256-example small-sample run cannot replace an 8192-example source learnability control; v3 new data require new runs.")

    path = "experiments/pilot_v2_diagnostics/artifacts/source_learning_endpoints.csv"
    for r in rows(path):
        if r["split"] != "dev":
            continue
        add(reference_id=f"pilot_v2_s{r['seed']}_{r['condition']}_p{r['budget']}_{r['task']}",
            task=r["task"], record_type="four_task", source="pilot_v2", source_file=path,
            model_seed=r["seed"], train_lengths="10", metric_length=10, train_unique=8192,
            dev_count=r["count"], updates=r["budget"], exposures=int(r["budget"]) * 64,
            lr="3e-4 cosine to2400; constant3e-5 continuation", warmup=100,
            dev_accuracy=r["accuracy"], dev_answer_ce=r["answer_ce"],
            majority_accuracy=r["majority_accuracy"], metric="full_vocab_answer", evaluation_split="dev",
            checkpoint_local=True, curves_source="experiments/pilot_v2_diagnostics/artifacts/source_learning_all.csv",
            training_compatible="previous_multitask_only", reuse_policy="reuse_as_previous_composition_reference",
            caveat="Four-task record, not single-task evidence; 16 samples/task/update, shared8192 source pool; aggregate exposures shown are over all four tasks.")

    fields = list(dict.fromkeys(k for r in records for k in r))
    with (OUT / "candidates_existing.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(records)
    local_checkpoints = [str(p.relative_to(ROOT)) for p in (ROOT / "reference").rglob("*")
                         if p.is_file() and p.suffix in {".pt", ".pth", ".safetensors"}]
    identity = {}
    for name in ["math_ops.py", "models.py", "passage.py"]:
        a = ROOT / "experiments/permworld_pilot/vendor/neurips_permutations" / name
        b = ROOT / "reference/neurips/src/neurips_permutations" / name
        identity[name] = dict(bytes_identical=a.read_bytes() == b.read_bytes(),
                              sha256=hashlib.sha256(a.read_bytes()).hexdigest())
    audit = dict(inventory_rows=len(records), reference_local_checkpoints=local_checkpoints,
                 upstream_vendor_identity=identity, test_metrics_read=False,
                 matching_budget_source_single_task_checkpoints_available=False,
                 note="Only validation/development fields are extracted. Existing source bytes and results are not modified.")
    (OUT / "inventory_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    # A reuse manifest is separate from scientific single-task evidence. Source
    # and adaptation weights may be reused only for an unchanged fixed-points
    # anchor; none constitutes a new source single-task reference.
    reuse = []
    import torch
    v1 = ROOT / "experiments/permworld_pilot/runs/pilot_v1"
    v2 = ROOT / "experiments/pilot_v2_diagnostics/artifacts"
    for seed in [17, 42]:
        pre_data = v1 / "data" / f"s{seed}_pretrain.npz"
        source_dev = v1 / "data" / f"s{seed}_source_dev.npz"
        target_train = v1 / "data" / f"s{seed}_target_train.npz"
        target_dev = v1 / "data" / f"s{seed}_target_dev.npz"
        paths = [(v1 / f"seed{seed}/step0.pt", "initialization", "all", "", 0)]
        paths += [(v2 / "extended" / f"seed{seed}_{condition}_4800.pt", "pretrain", condition, "", 4800)
                  for condition in ["B", "C"]]
        for condition in ["A", "B", "C"]:
            budget = 0 if condition == "A" else 4800
            for lr in ["3e-04", "1e-04", "3e-05"]:
                folder = v2 / "adapt" / f"s{seed}_{condition}_p{budget}_lr{lr}"
                paths += [(folder / f"{kind}.pt", f"adapt_{kind}", condition, lr, budget)
                          for kind in ["best", "endpoint"]]
        for path, kind, condition, lr, budget in paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = payload.get("model", payload)
            assert tuple(state["token_embedding.weight"].shape) == (188, 256)
            assert torch.equal(state["token_embedding.weight"], state["lm_head.weight"])
            optimizer_steps = sorted({int(s["step"]) for s in payload.get("optimizer", {}).get("state", {}).values()})
            reuse.append(dict(path=path.relative_to(ROOT).as_posix(), kind=kind, seed=seed,
                target="fixed_points" if kind != "initialization" else "any_target_with_same_seed_init",
                condition=condition, learning_rate=lr, pretrain_updates=budget,
                checkpoint_step=payload.get("step", 0), optimizer_steps=";".join(map(str, optimizer_steps)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cpu_load_verified=True,
                embedding_shape="188x256", tied_embedding_values_equal=True,
                source_train_sha256=hashlib.sha256(pre_data.read_bytes()).hexdigest(),
                source_dev_sha256=hashlib.sha256(source_dev.read_bytes()).hexdigest(),
                target_train_sha256=hashlib.sha256(target_train.read_bytes()).hexdigest(),
                target_train_slice="0:256", target_dev_sha256=hashlib.sha256(target_dev.read_bytes()).hexdigest(),
                reuse_condition="Exact original task order, datasets, optimizer schedules and dev selection; evaluate on fresh v3 test only after freeze"))
            del payload, state
    with (OUT / "reusable_local_checkpoints.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(reuse[0]))
        writer.writeheader()
        writer.writerows(reuse)
    audit["reusable_local_checkpoint_rows"] = len(reuse)
    audit["all_reuse_checkpoints_loaded_on_cpu"] = True
    (OUT / "inventory_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
