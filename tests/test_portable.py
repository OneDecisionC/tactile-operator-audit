import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import icassp_stage_b_runner as stage
import run_icassp_jitter_screening as screen
from tactile_audit.workflow import CONDITIONS, TrainingDataset, generate_condition, operator_seed

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("portable", ROOT / "examples" / "run_portable.py")
portable = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(portable)


def configuration():
    return portable.read_configuration(ROOT / "examples" / "paper_probes.json")


def sample_bundle(dataset="braille", count=10):
    channels, steps, classes = (24, 45, 27) if dataset == "braille" else (512, 80, 35)
    clean = np.zeros((count, steps, channels), dtype=np.uint8)
    clean[:, [0, 1, 2, 4, 8, steps - 1], :3] = 1
    raw = [screen.RawEvents(np.nonzero(array)[0] * 0.025 + 0.001, np.nonzero(array)[1]) for array in clean]
    return screen.DatasetBundle(
        name=dataset, clean=clean if dataset == "braille" else np.packbits(clean, axis=-1, bitorder="little"),
        labels=np.arange(count) % 2, sample_ids=tuple(f"synthetic_{index:03}" for index in range(count)),
        raw_getter=lambda index: raw[index], classes=tuple(str(index) for index in range(classes)),
        duration_seconds=steps * 0.025, time_steps=steps, channels=channels, metadata={},
        groups=None if dataset == "braille" else np.asarray([f"group_{index}" for index in range(count)]),
    )


def test_configuration_and_rng_identity():
    config = configuration()
    models = {model["probe_id"]: model for model in config["models"]}
    assert "braille_conv1d_h83_l2_k5" in models
    registry = portable.rng_registry(config)
    seed = stage.model_initialization_seed(registry, SimpleNamespace(probe_id="braille_conv1d_h83_l2_k5", split_seed=42, fold=1))
    assert seed == 4865500485794623649
    jobs = portable.selected_jobs(config, SimpleNamespace(probe="", arm="", seed=None, fold=None))
    assert len(jobs) == 75
    grid = portable.condition_grid(config)
    assert len(grid) == len(set(grid)) == 136
    assert {condition for condition, _, _ in grid} == CONDITIONS
    transfer = portable.read_configuration(ROOT / "examples" / "braille_transfer.json")
    assert len(portable.selected_jobs(transfer, SimpleNamespace(probe="", arm="", seed=None, fold=None))) == 90


@pytest.mark.parametrize("dataset", ["braille", "stemnist"])
def test_all_conditions_preserve_expected_counts(dataset):
    config = configuration()
    registry = portable.rng_registry(config)
    bundle = sample_bundle(dataset)
    adapter = screen.OperatorAdapter()
    clean = screen.unpack_clean_sample(bundle, 0)
    results = {name: generate_condition(registry, adapter, bundle, 0, name, 75 if not name.startswith("retention_") else 0, 0)[0] for name in CONDITIONS}
    assert all(array.shape == clean.shape and set(np.unique(array)) <= {0, 1} for array in results.values())
    for name in ("cf", "ipbj_movement"):
        assert np.array_equal(results[name].sum(axis=0), clean.sum(axis=0))
    for name in CONDITIONS - {"clean", "prebin", "cf", "ipbj_movement", "retention_95", "retention_90", "retention_85"}:
        assert np.array_equal(results[name].sum(axis=0), results["pbj"].sum(axis=0))
    for percent in (95, 90, 85):
        assert results[f"retention_{percent}"].sum() == (percent * int(clean.sum()) + 50) // 100
    assert np.all(results["retention_85"] <= results["retention_90"])
    assert np.all(results["retention_90"] <= results["retention_95"])
    if dataset == "stemnist":
        assert operator_seed(bundle, 0, registry, retention=True) != operator_seed(bundle, 0, registry)


@pytest.mark.parametrize("arm,condition", [("matched_aug", "matched"), ("cf_matched_aug", "cf_matched")])
def test_matched_augmentation_reuses_historical_draws(arm, condition):
    registry = portable.rng_registry(configuration())
    bundle = sample_bundle()
    adapter = screen.OperatorAdapter()
    job = SimpleNamespace(dataset="braille", train_arm=arm, split_seed=42, fold=1)
    for epoch in range(1, 4):
        dataset = TrainingDataset(screen, bundle, range(10), registry, job, adapter, epoch)
        for index in range(10):
            augmented, severity, seed = stage.augmentation_decision(registry, job, epoch, index)
            clean = screen.unpack_clean_sample(bundle, index)
            expected = adapter.postbin_bundle(clean, severity // 25, seed, 0)[0][condition] if augmented else clean
            actual, label, sample_index = dataset[index]
            assert np.array_equal(actual, expected)
            assert label == bundle.labels[index] and sample_index == index


@pytest.mark.parametrize("dataset", ["braille", "stemnist"])
def test_training_checkpoint_evaluation_resume(tmp_path, monkeypatch, dataset):
    pytest.importorskip("torch")
    pytest.importorskip("snntorch")
    config = configuration()
    config["device"] = "cpu"
    selected = "braille_tcn_small_h64_b2_k3" if dataset == "braille" else "stemnist_tcn_large_h128_b3_k3"
    config["models"] = [model for model in config["models"] if model["probe_id"] == selected]
    config["split_seeds"] = [42]
    config["folds"] = [1]
    config["evaluation"]["conditions"] = ["clean", "pbj", "ipbj_movement", "retention_90"]
    config["evaluation"]["severities_ms"] = [25]
    config["evaluation"]["realizations"] = 1
    profile = config["training_profiles"][config["models"][0]["training_profile"]]
    profile["epochs"], profile["batch_size"] = 1, 4
    _, _, deps = portable.load_runtime()
    deps.torch.set_num_threads(2)
    bundle = sample_bundle(dataset)
    monkeypatch.setattr(screen, "exact_folds", lambda *_: [(np.arange(8), np.arange(8, 10))])
    model, record = portable.selected_jobs(config, SimpleNamespace(probe="", arm="", seed=None, fold=None))[0]
    job = stage.TrainJob(**record)
    portable.train_job(config, stage, screen, deps, bundle, model, job, tmp_path, "synthetic")
    checkpoint = tmp_path / record["checkpoint_relpath"]
    before = checkpoint.read_bytes()
    portable.train_job(config, stage, screen, deps, bundle, model, job, tmp_path, "synthetic")
    assert checkpoint.read_bytes() == before
    portable.evaluate_job(config, stage, screen, deps, bundle, model, job, tmp_path, "synthetic")
    markers = list((tmp_path / "evaluation").rglob("commit.json"))
    assert len(markers) == 4
    portable.evaluate_job(config, stage, screen, deps, bundle, model, job, tmp_path, "synthetic")
    predictions = next((tmp_path / "evaluation").rglob("predictions.csv"))
    predictions.write_bytes(predictions.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="checksum"):
        portable.evaluate_job(config, stage, screen, deps, bundle, model, job, tmp_path, "synthetic")


@pytest.mark.parametrize("model_index", range(5))
def test_all_paper_models_forward(model_index):
    pytest.importorskip("torch")
    pytest.importorskip("snntorch")
    _, _, deps = portable.load_runtime()
    deps.torch.set_num_threads(2)
    entry = configuration()["models"][model_index]
    bundle = sample_bundle(entry["dataset"])
    model = stage.make_registered_model(deps, entry, bundle)
    model.eval()
    with deps.torch.inference_mode():
        logits = model(deps.torch.zeros(2, bundle.time_steps, bundle.channels))
    assert tuple(logits.shape) == (2, len(bundle.classes))
    assert deps.torch.isfinite(logits).all()
