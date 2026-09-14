import numpy as np
import pytest
import csv
import json
import subprocess
import sys
from pathlib import Path

from tactile_audit.analysis import analyze_run, digest, file_sha, paired_bootstrap


def test_paired_zero_and_constant_intervals():
    differences = np.column_stack([np.zeros(12), np.ones(12) * 0.25])
    result = paired_bootstrap("braille", differences, None, np.arange(12) % 3, resamples=150)
    means, bounds = result["sample_micro_class_stratified"]
    np.testing.assert_array_equal(means, [0, 0.25])
    np.testing.assert_allclose(bounds, [[0, 0.25], [0, 0.25]])


def test_cluster_micro_and_macro_are_distinct():
    differences = np.asarray([0, 0, 0, 1.0])[:, None]
    result = paired_bootstrap("stemnist", differences, np.asarray(["a", "a", "a", "b"]), np.zeros(4), resamples=500)
    assert result["sample_micro_cluster_ratio"][0][0] == 0.25
    assert result["participant_macro"][0][0] == 0.5


def test_bootstrap_repeatability_and_pairing():
    values = np.linspace(-1, 1, 30)
    differences = np.column_stack([values, -values])
    args = ("braille", differences, None, np.arange(30) % 3)
    first = paired_bootstrap(*args, resamples=257, seed=42)["sample_micro_class_stratified"]
    second = paired_bootstrap(*args, resamples=257, seed=42)["sample_micro_class_stratified"]
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_allclose(first[1][:, 0], -first[1][::-1, 1])


def test_reject_missing_groups_and_nonfinite_inputs():
    with pytest.raises(ValueError):
        paired_bootstrap("stemnist", np.ones((4, 1)), ["", "a", "b", "b"], [0] * 4)
    with pytest.raises(ValueError):
        paired_bootstrap("braille", [[float("nan")]], None, [0])


def test_historical_bootstrap_synthetic_golden():
    fixture = json.loads((Path(__file__).parent / "fixtures/historical_bootstrap_synthetic.json").read_text())
    for case in fixture["cases"]:
        actual = paired_bootstrap(case["dataset"], case["differences"], case["groups"], case["labels"])
        for estimand, expected in case["expected"].items():
            np.testing.assert_allclose(actual[estimand][0], expected[0], rtol=0, atol=1e-14)
            np.testing.assert_allclose(actual[estimand][1], expected[1], rtol=0, atol=1e-14)


@pytest.fixture
def synthetic_run(tmp_path):
    model = {"probe_id": "synthetic_tcn", "dataset": "braille", "train_arms": ["clean"], "training_profile": "braille_stage_b"}
    config = {"models": [model], "split_seeds": [42, 123], "folds": [1, 2, 3, 4, 5],
              "evaluation": {"conditions": ["clean", "pbj", "cf_matched"], "severities_ms": [25], "realizations": 1}}
    manifest = {"configuration": config, "configuration_id": digest(config), "release_version": "0.2.0",
                "dataset_layout": {"braille": {"samples": 10, "classes": 2, "participants": 0}},
                "dataset_identity": {"braille": "synthetic_not_paper_data"}}
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    for seed in config["split_seeds"]:
        for fold in config["folds"]:
            job = f"synthetic_tcn__clean__seed{seed}__fold{fold}"
            checkpoint = tmp_path / "checkpoints" / digest(job)[:24]
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.pt").write_bytes(b"synthetic checksum placeholder, never deserialized")
            checkpoint_sha = file_sha(checkpoint / "model.pt")
            (checkpoint / "commit.json").write_text(json.dumps({"checkpoint_sha256": checkpoint_sha,
                "configuration_id": manifest["configuration_id"], "job_id": job, "dataset_identity": "synthetic_not_paper_data", "run_kind": "portable_new_run"}))
            for condition in config["evaluation"]["conditions"]:
                severity = 0 if condition == "clean" else 25
                unit = f"{condition}_ms{severity}_r0"
                directory = tmp_path / "evaluation" / digest(job)[:24] / digest(unit)[:16]
                directory.mkdir(parents=True)
                identity = {"configuration_id": manifest["configuration_id"], "job_id": job, "run_kind": "portable_new_run",
                            "condition": condition, "severity_ms": severity, "realization": 0}
                rows = []
                for index in range((fold - 1) * 2, fold * 2):
                    correct = 0 if condition == "pbj" else 1
                    label = index % 2
                    rows.append({**identity, "dataset": "braille", "probe_id": model["probe_id"], "training_profile": "braille_stage_b",
                                 "train_arm": "clean", "split_seed": seed, "fold": fold, "sample_index": index,
                                 "sample_id": f"synthetic_{index}", "participant": "", "label": label,
                                 "prediction": label if correct else 1 - label, "correct": correct,
                                 "active_before": 10, "active_after": 10 if condition == "clean" else 8})
                path = directory / "predictions.csv"
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                marker = {**identity, "dataset_identity": "synthetic_not_paper_data", "checkpoint_sha256": checkpoint_sha,
                          "samples": 2, "correct": sum(row["correct"] for row in rows), "active_before": 20,
                          "active_after": sum(row["active_after"] for row in rows), "predictions_sha256": file_sha(path)}
                (directory / "commit.json").write_text(json.dumps(marker))
    return tmp_path


def test_complete_oof_aggregation_and_bootstrap(synthetic_run):
    conditions, contrasts, provenance = analyze_run(synthetic_run, resamples=200)
    assert len(conditions) == 3 and len(contrasts) == 1 and len(provenance) == 30
    assert contrasts[0]["difference_pp"] == -100
    assert contrasts[0]["ci95_low_pp"] == contrasts[0]["ci95_high_pp"] == -100


def test_missing_unit_rejected(synthetic_run):
    next((synthetic_run / "evaluation").rglob("commit.json")).unlink()
    with pytest.raises(FileNotFoundError):
        analyze_run(synthetic_run, resamples=10)


def test_duplicate_rows_rejected_even_with_updated_hash(synthetic_run):
    path = next((synthetic_run / "evaluation").rglob("predictions.csv"))
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines + [lines[1]]) + "\n")
    marker_path = path.parent / "commit.json"
    marker = json.loads(marker_path.read_text())
    marker["predictions_sha256"] = file_sha(path)
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="Duplicate"):
        analyze_run(synthetic_run, resamples=10)


def test_prediction_tampering_rejected(synthetic_run):
    path = next((synthetic_run / "evaluation").rglob("predictions.csv"))
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        analyze_run(synthetic_run, resamples=10)


def test_analysis_cli_report_and_no_overwrite(synthetic_run):
    script = Path(__file__).resolve().parents[1] / "scripts/analyze_run.py"
    output = synthetic_run / "analysis"
    command = [sys.executable, str(script), "--run", str(synthetic_run), "--output", str(output), "--resamples", "10", "--execute"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert (output / "report.md").is_file()
    assert (output / "paired_ci.csv").is_file()
    assert (output / "provenance.json").is_file()
    assert subprocess.run(command, capture_output=True).returncode != 0
