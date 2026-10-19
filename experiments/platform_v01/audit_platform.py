"""Read-only evidence checks; write a new platform audit, never historical artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def preserve_snapshot():
    dest = HERE / "preservation.json"
    if dest.exists():
        return read(dest)
    old = ROOT / "experiments/pilot_v5_confirmation/artifacts"
    expected = read(old / "preservation.json")
    manifest = read(old / "artifact_manifest.json")
    for rel, item in manifest["files"].items():
        path = (old / rel).resolve()
        expected[path.relative_to(ROOT).as_posix()] = item["sha256"]
    old_dirs = ["permworld_pilot", "pilot_v2_diagnostics", "pilot_v3_multitarget",
                "pilot_v4_backgrounds", "pilot_v5_confirmation"]
    existing = {p.relative_to(ROOT).as_posix(): p for name in old_dirs
                for p in (ROOT / "experiments" / name).rglob("*")
                if p.is_file() and "__pycache__" not in p.parts}
    existing["PILOT_REPORT.md"] = ROOT / "PILOT_REPORT.md"
    missing_manifest_entries = sorted(set(existing) - set(expected))
    for name in missing_manifest_entries:
        expected[name] = sha(existing[name])
    assert set(expected) == set(existing), "historical snapshot scope differs"
    result = dict(files=expected, previous_v5_preservation_sha256=sha(old / "preservation.json"),
                  previous_v5_manifest_sha256=sha(old / "artifact_manifest.json"),
                  expected_hash_provenance="v5 preservation and completion manifests; only omitted logs/manifest hashed now",
                  newly_hashed_files=missing_manifest_entries)
    write(dest, result)
    return result


def audit(wait_for_calibration=False):
    started = time.perf_counter()
    snapshot = preserve_snapshot()
    for i, (rel, digest) in enumerate(snapshot["files"].items(), 1):
        path = ROOT / rel
        assert path.is_file() and sha(path) == digest, f"Historical artifact modified or missing: {rel}"
        if i % 500 == 0:
            print(f"[preservation] {i}/{len(snapshot['files'])} historical files verified", flush=True)
    math_dir = HERE / "mathematics"
    cert = read(math_dir / "CERTIFICATE.json")
    assert cert["status"] == "passed"
    for rel, digest in cert["source_sha256"].items():
        assert sha(ROOT / rel) == digest, f"Math source changed: {rel}"
    for rel, digest in cert["artifact_sha256"].items():
        assert sha(math_dir / rel) == digest, f"Math evidence changed: {rel}"
    assert cert["historical_formula_rechecks_run"] == 0
    quality = read(HERE / "dataset/quality_full.json")
    interface = read(HERE / "interface_audit.json")
    core = read(HERE / "registry/CORE_INTERFACE_AUDIT.json")
    assert quality["status"] == interface["status"] == core["status"] == "passed"
    meta = read(HERE / "dataset/manifest.json")
    for pool in meta["pools"]:
        for key in ("object", "label"):
            assert sha(HERE / "dataset" / pool[f"{key}_path"]) == pool[f"{key}_sha256"]
    if wait_for_calibration:
        deadline = time.monotonic() + 1800
        while not ((HERE / "calibration/CALIBRATION_COMPLETE.json").exists()
                   and (HERE / "calibration/CALIBRATION_AUDIT.json").exists()):
            if time.monotonic() > deadline:
                raise TimeoutError("calibration/audit not completed; no platform success claimed")
            time.sleep(10)
    frozen = read(HERE / "calibration/FROZEN_CALIBRATION.json")
    complete = read(HERE / "calibration/CALIBRATION_COMPLETE.json")
    assert complete["runs"] == 12 and complete["test_evaluated"] is False
    assert complete["unix"] > frozen["unix"]
    assert complete["frozen_protocol_sha256"] == sha(HERE / "calibration/FROZEN_CALIBRATION.json")
    for rel, digest in frozen["code_and_registry_sha256"].items():
        assert sha(ROOT / rel) == digest, f"Frozen platform code changed: {rel}"
    for field in ("registry_data_certificate_evidence_sha256", "training_development_data_sha256"):
        for rel, digest in frozen[field].items():
            assert sha((HERE / "calibration" / rel).resolve()) == digest, f"Frozen evidence changed: {rel}"
    calibration_audit = read(HERE / "calibration/CALIBRATION_AUDIT.json")
    assert calibration_audit["status"] == "passed" and calibration_audit["development_forwards_repeated"] is True
    assert calibration_audit["test_forward_calls"] == 0
    result = dict(status="passed", seconds=time.perf_counter()-started,
                  historical_files_byte_preserved=len(snapshot["files"]),
                  mathematical_certificate_source_and_artifacts_bound=True,
                  old_formulas_rechecked=0, data_pool_pairs_hash_verified=len(meta["pools"]),
                  input_orbits_and_rng_replay=quality, encoding_boundary_audit=interface,
                  formal_calibration_runs=12, test_evaluated=False,
                  no_transfer_or_background_variant_training=True,
                  calibration_checkpoints_independently_forwarded=calibration_audit["checkpoint_audits"],
                  limitation="Use calibration/CALIBRATION_AUDIT.json for repeat neural forward checks; this audit itself only checks bytes and saved evidence.")
    write(HERE / "PLATFORM_AUDIT.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--wait-for-calibration", action="store_true")
    args = parser.parse_args()
    if args.snapshot_only:
        result = preserve_snapshot()
        print(json.dumps(dict(files=len(result["files"]), newly_hashed_files=result["newly_hashed_files"])))
    else:
        audit(args.wait_for_calibration)


if __name__ == "__main__":
    main()
