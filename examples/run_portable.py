"""Portable NEW-RUN orchestration over the preserved scientific implementations.

No training, evaluation, cache generation or data inspection happens without
--execute. Training additionally requires --allow-training. This file never
relaxes the historical Stage-B formal-environment gates or imports its registry.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from tactile_audit.workflow import CONDITIONS, RETENTION_CONDITIONS, TRAIN_ARMS, TrainingDataset, generate_condition

REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA = "tactile_operator_audit_portable_v1"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY / path).resolve()


def read_configuration(path):
    configuration = json.loads(path.read_text(encoding="utf-8-sig"))
    if configuration.get("schema") != SCHEMA or configuration.get("run_kind") != "portable_new_run":
        raise ValueError("Expected a portable_new_run configuration, not a historical registry.")
    if not configuration.get("models"):
        raise ValueError("At least one model is required.")
    names = [model["probe_id"] for model in configuration["models"]]
    if len(names) != len(set(names)):
        raise ValueError("Probe IDs must be unique.")
    for name in names:
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("Probe IDs may contain only ASCII letters, numbers, _ and -.")
    for field in ("split_seeds", "folds"):
        values = configuration[field]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{field} must be nonempty and unique.")
    if any(not isinstance(seed, int) or seed < 0 for seed in configuration["split_seeds"]):
        raise ValueError("Split seeds must be nonnegative integers.")
    if any(fold not in (1, 2, 3, 4, 5) for fold in configuration["folds"]):
        raise ValueError("folds selects from the original five-fold split, not a new fold count.")
    for model in configuration["models"]:
        profile_name = model["training_profile"]
        expected = "braille_stage_b" if model["dataset"] == "braille" else "stemnist_stage_a"
        if model["dataset"] not in ("braille", "stemnist") or profile_name != expected:
            raise ValueError("Do not interchange the Braille Stage-B and STEMNIST Stage-A profiles.")
        profile = configuration["training_profiles"][profile_name]
        expected_rng = "icassp_stage_b_formal_v1" if profile_name == "braille_stage_b" else "binary80_groupcv_v1"
        if profile["rng_protocol_id"] != expected_rng:
            raise ValueError("The historical RNG protocol ID must not be renamed.")
        if model["probe_id"] == "braille_conv1d_h83":
            raise ValueError("Use braille_conv1d_h83_l2_k5 to retain the historical initialization key.")
        optimizer = "AdamW" if profile_name == "braille_stage_b" else "Adam"
        clipping = 1.0 if profile_name == "braille_stage_b" else None
        if profile["optimizer"] != optimizer or profile["gradient_clip_norm"] != clipping:
            raise ValueError(f"{profile_name} requires {optimizer} and gradient_clip_norm={clipping}.")
        if profile["checkpoint_rule"] != "final_epoch_only":
            raise ValueError("Only final-epoch checkpoints are supported.")
        if profile["epochs"] < 1 or profile["batch_size"] < 1:
            raise ValueError("Epoch and batch-size settings must be positive.")
        arms = model["train_arms"]
        if not arms or len(arms) != len(set(arms)) or not set(arms) <= TRAIN_ARMS:
            raise ValueError("Invalid or duplicate training arms.")
        if profile_name == "stemnist_stage_a" and arms != ["clean"]:
            raise ValueError("The paper STEMNIST Stage-A probes were clean-trained; this launcher blocks other arms.")
    augmentation = configuration["augmentation"]
    if augmentation != {"augmentation_probability": 0.75, "severities_ms": [25, 50, 75]}:
        raise ValueError("The preserved Stage-B augmentation implementation fixes probability .75 and severities 25/50/75 ms.")
    evaluation = configuration["evaluation"]
    if not evaluation["conditions"] or not set(evaluation["conditions"]) <= CONDITIONS:
        raise ValueError("Unknown evaluation condition.")
    if len(evaluation["conditions"]) != len(set(evaluation["conditions"])):
        raise ValueError("Duplicate evaluation conditions.")
    if not evaluation["severities_ms"] or not set(evaluation["severities_ms"]) <= {25, 50, 75}:
        raise ValueError("Evaluation severities must be selected from 25/50/75 ms.")
    if len(evaluation["severities_ms"]) != len(set(evaluation["severities_ms"])):
        raise ValueError("Duplicate evaluation severities.")
    if evaluation["realizations"] < 1:
        raise ValueError("At least one realization is required.")
    if any(int(evaluation["batch_size_by_dataset"][name]) < 1 for name in ("braille", "stemnist")):
        raise ValueError("Evaluation batch sizes must be positive.")
    output = resolve_path(configuration["output_root"])
    if output == REPOSITORY / "outputs" or not output.is_relative_to((REPOSITORY / "outputs").resolve()):
        raise ValueError("output_root must be a child directory of this repository's outputs/.")
    return configuration


def selected_jobs(configuration, args):
    jobs = []
    for model in configuration["models"]:
        if args.probe and model["probe_id"] != args.probe:
            continue
        for arm in model["train_arms"]:
            if getattr(args, "arm", "") and arm != args.arm:
                continue
            for seed in configuration["split_seeds"]:
                if args.seed is not None and seed != args.seed:
                    continue
                for fold in configuration["folds"]:
                    if args.fold is not None and fold != args.fold:
                        continue
                    job_id = f"{model['probe_id']}__{arm}__seed{seed}__fold{fold}"
                    jobs.append((model, {
                        "job_id": job_id, "probe_id": model["probe_id"],
                        "dataset": model["dataset"], "family": model["family"],
                        "phase": model["phase"], "train_arm": arm,
                        "split_seed": seed, "fold": fold,
                        "checkpoint_relpath": f"checkpoints/{digest(job_id)[:24]}/model.pt",
                        "manifest_relpath": f"checkpoints/{digest(job_id)[:24]}/commit.json",
                    }))
    if not jobs:
        raise ValueError("No jobs match the configuration and selectors.")
    return jobs


def condition_grid(configuration):
    settings = configuration["evaluation"]
    result = [("clean", 0, 0)] if "clean" in settings["conditions"] else []
    for realization in range(settings["realizations"]):
        result.extend((condition, 0, realization) for condition in settings["conditions"] if condition in RETENTION_CONDITIONS)
    for severity in settings["severities_ms"]:
        for realization in range(settings["realizations"]):
            result.extend(
                (condition, severity, realization)
                for condition in settings["conditions"] if condition != "clean" and condition not in RETENTION_CONDITIONS
            )
    return result


def describe(configuration, jobs, command):
    config_id = digest(configuration)
    print(json.dumps({
        "run_kind": "portable_new_run", "historical_formal_replay": False,
        "command": command, "execution_requested": False,
        "configuration_id": config_id,
        "output_directory": str(resolve_path(configuration["output_root"]) / config_id[:16]),
        "selected_jobs": len(jobs), "evaluation_units": len(jobs) * len(condition_grid(configuration)),
        "probes": sorted({model["probe_id"] for model, _ in jobs}),
        "training_profiles": {
            name: configuration["training_profiles"][name]
            for name in sorted({model["training_profile"] for model, _ in jobs})
        },
        "data_note": "Paths resolve from the repository root. This plan has not read data or imported PyTorch.",
    }, indent=2))


def prepare_cache(configuration):
    import run_icassp_jitter_screening as screen

    screen.build_stemnist_cache(SimpleNamespace(
        data_root=str(resolve_path(configuration["data"]["stemnist_data_root"])),
        cache_dir=str(resolve_path(configuration["data"]["stemnist_cache"])),
        rebuild_cache=False,
    ))


def load_runtime():
    # Stage-B sets CUBLAS_WORKSPACE_CONFIG before the first torch import.
    import icassp_stage_b_runner as stage_b
    import run_icassp_jitter_screening as screen

    return stage_b, screen, screen.import_runtime_dependencies()


def load_bundle(configuration, screen, dataset):
    if dataset == "braille":
        return screen.load_braille_bundle(resolve_path(configuration["data"]["braille_root"]))
    return screen.load_stemnist_bundle(
        resolve_path(configuration["data"]["stemnist_cache"]), verify_hash=True
    )


def bundle_identity(screen, bundle):
    source = {
        "dataset": bundle.name, "samples": len(bundle.labels),
        "classes": list(bundle.classes), "channels": bundle.channels,
        "time_steps": bundle.time_steps, "labels": bundle.labels.tolist(),
        "sample_ids": list(bundle.sample_ids),
        "groups": None if bundle.groups is None else bundle.groups.tolist(),
        "source_sha256": bundle.metadata.get("source_sha256"),
        "clean_sha256": bundle.metadata.get("clean_sha256"),
        "packed_sha256": bundle.metadata.get("packed_sha256"),
        "cache_index_sha256": bundle.metadata.get("cache_index_sha256"),
    }
    return screen.sha256_json(source)


def source_inventory(stage_b):
    names = (
        "run_braille_ablation_baselines", "run_stemnist_robustness",
        "run_icassp_jitter_screening", "icassp_stage_b_runner",
        "icassp_stage_b_operator_cache", "icassp_jitter_operators",
        "injective_pbj_operator", "survivor_rule_operator", "retention_matched_operator",
        "tactile_audit.workflow",
    )
    inventory = {
        name + ".py": stage_b.sha256_file(Path(importlib.import_module(name).__file__))
        for name in names
    }
    inventory["examples/run_portable.py"] = stage_b.sha256_file(Path(__file__))
    return inventory


def bind_run(configuration, stage_b, deps, identities, layouts=None):
    config_id = digest(configuration)
    output = resolve_path(configuration["output_root"]) / config_id[:16]
    output.mkdir(parents=True, exist_ok=True)
    environment = stage_b.execution_environment(deps.torch, configuration["device"], formal=False)
    environment["numpy_version"] = importlib.import_module("numpy").__version__
    environment["scipy_version"] = importlib.import_module("scipy").__version__
    manifest_path = output / "run_manifest.json"
    content = {
        "schema": SCHEMA, "run_kind": "portable_new_run", "formal": False,
        "configuration_id": config_id, "configuration": configuration,
        "environment": environment, "source_sha256": source_inventory(stage_b),
        "dataset_identity": identities,
        "dataset_layout": layouts or {},
        "release_version": "0.2.0",
        "release_validation": "See docs/VALIDATION.md; smoke tests are not historical reproduction.",
        "results_scope": "New checkpoints and per-unit predictions; not historical paper-result replay.",
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field in ("configuration_id", "environment", "source_sha256"):
            if previous.get(field) != content[field]:
                raise RuntimeError(f"Existing run differs in {field}; use a new output_root.")
        for dataset, identity in identities.items():
            if dataset in previous["dataset_identity"] and previous["dataset_identity"][dataset] != identity:
                raise RuntimeError(f"Existing run has a different {dataset} dataset identity.")
        previous["dataset_identity"].update(identities)
        for dataset, layout in (layouts or {}).items():
            if dataset in previous.get("dataset_layout", {}) and previous["dataset_layout"][dataset] != layout:
                raise RuntimeError(f"Dataset layout changed: {dataset}")
        previous.setdefault("dataset_layout", {}).update(layouts or {})
        content = previous
    stage_b.write_json_atomic(manifest_path, content)
    return output, environment


def commit_matches(stage_b, directory, expected):
    marker = directory / "commit.json"
    if not marker.exists():
        if (directory / "model.pt").exists() or (directory / "model.pt.tmp").exists():
            raise RuntimeError(f"Uncommitted checkpoint preserved at {directory}; choose a new output_root.")
        return False
    committed = json.loads(marker.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if committed.get(key) != value:
            raise RuntimeError(f"Checkpoint commit mismatch in {key}: {directory}")
    if stage_b.sha256_file(directory / "model.pt") != committed["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint checksum mismatch: {directory}")
    return True


def rng_registry(configuration):
    return {
        "protocol_id": configuration["training_profiles"]["braille_stage_b"]["rng_protocol_id"],
        "augmentation": {"augmented_arm_sampling": {
            "augmentation_probability": 0.75,
            "severities_ms": configuration["augmentation"]["severities_ms"],
        }},
    }


def train_job(configuration, stage_b, screen, deps, bundle, model_entry, job, output, data_identity):
    import numpy as np

    torch = deps.torch
    profile_name = model_entry["training_profile"]
    profile = configuration["training_profiles"][profile_name]
    device = screen.resolve_device(torch, configuration["device"])
    folds = screen.exact_folds(deps, bundle, job.split_seed)
    train_indices, validation_indices = folds[job.fold - 1]
    expected = {
        "configuration_id": digest(configuration), "job_id": job.job_id,
        "dataset_identity": data_identity, "run_kind": "portable_new_run",
    }
    directory = output / "checkpoints" / digest(job.job_id)[:24]
    directory.mkdir(parents=True, exist_ok=True)
    with stage_b.exclusive_writer_lock(directory / "writer.lock", expected):
        if commit_matches(stage_b, directory, expected):
            print(f"resume training: {job.job_id}", flush=True)
            return
        registry = rng_registry(configuration)
        if profile_name == "braille_stage_b":
            stage_b.configure_runtime_determinism(torch)
            seed = stage_b.model_initialization_seed(registry, job)
            stage_b.seed_model_initialization_rngs(torch, seed)
        else:
            seed = job.split_seed + job.fold - 1
            deps.stem_module.seed_everything(seed, True)
        model = stage_b.make_registered_model(deps, model_entry, bundle).to(device)
        backend = stage_b.concrete_model_backend(model, model_entry, formal=False)
        if model_entry["family"] in ("RateSNN", "paper_scnn") and backend != "snntorch.Leaky":
            raise RuntimeError("Portable paper probes require snntorch.Leaky; local surrogate fallback is not allowed.")
        optimizer_class = torch.optim.AdamW if profile_name == "braille_stage_b" else torch.optim.Adam
        optimizer = optimizer_class(
            model.parameters(), lr=float(profile["learning_rate"]),
            weight_decay=float(profile["weight_decay"]),
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        adapter = screen.OperatorAdapter()
        history = []
        stem_args = SimpleNamespace(
            batch_size=int(profile["batch_size"]), num_workers=0, pin_memory=True,
            train_aug_mode="none",
        )
        for epoch in range(1, int(profile["epochs"]) + 1):
            if profile_name == "stemnist_stage_a":
                # Same per-fold/epoch RNG scheme, loader and epoch implementation
                # as run_stemnist_robustness.run_experiment; Adam, no clipping.
                torch.manual_seed(seed + 10_000 + epoch)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed + 10_000 + epoch)
                loader = deps.stem_module.make_loader(
                    bundle.clean, bundle.labels, train_indices, stem_args,
                    shuffle=True, seed=seed + epoch,
                )
                loss, accuracy = deps.stem_module.train_one_epoch(
                    model, loader, optimizer, loss_fn, device, stem_args, seed, epoch,
                )
            else:
                # Training orchestration follows Stage-B train_one_job.
                # Its dataset, augmentation and RNG implementations are reused.
                dataset = TrainingDataset(
                    screen, bundle, train_indices, registry, job, adapter, epoch,
                )
                permutation_seed = stage_b.stable_seed(
                    registry["protocol_id"], job.dataset, job.split_seed, job.fold,
                    epoch, "epoch_permutation",
                )
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=int(profile["batch_size"]), shuffle=True,
                    generator=torch.Generator().manual_seed(permutation_seed),
                    num_workers=0, collate_fn=stage_b._collate(torch),
                )
                model.train()
                total_loss, correct, seen = 0.0, 0, 0
                for inputs, labels, _ in loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(inputs)
                    batch_loss = torch.nn.functional.cross_entropy(logits, labels)
                    if not torch.isfinite(batch_loss):
                        raise RuntimeError("Nonfinite training loss; no checkpoint will be committed.")
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += float(batch_loss.item()) * len(labels)
                    correct += int((logits.argmax(1) == labels).sum().item())
                    seen += len(labels)
                loss, accuracy = total_loss / max(seen, 1), correct / max(seen, 1)
            history.append({"epoch": epoch, "train_loss": loss, "train_accuracy": accuracy})
            print(f"{job.job_id} epoch={epoch}/{profile['epochs']} loss={loss:.6f} accuracy={accuracy:.6f}", flush=True)
        payload = {
            **expected, "formal": False, "training_profile": profile_name,
            "profile": profile, "job": job.as_dict(), "model": model_entry,
            "epoch": int(profile["epochs"]), "checkpoint_rule": "final_epoch_only",
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "training_seed": seed, "model_backend": backend, "history": history,
            "train_indices_sha256": screen.sha256_array(np.asarray(train_indices, dtype=np.int64)),
            "validation_indices": np.asarray(validation_indices, dtype=np.int64).tolist(),
        }
        temporary = directory / "model.pt.tmp"
        if temporary.exists():
            raise RuntimeError(f"Interrupted checkpoint file preserved: {temporary}")
        torch.save(payload, temporary)
        os.replace(temporary, directory / "model.pt")
        stage_b.write_csv_atomic(directory / "history.csv", history)
        stage_b.write_json_atomic(directory / "commit.json", {
            **expected, "formal": False, "checkpoint_rule": "final_epoch_only",
            "final_epoch": int(profile["epochs"]),
            "checkpoint_sha256": stage_b.sha256_file(directory / "model.pt"),
        })


def evaluate_job(configuration, stage_b, screen, deps, bundle, model_entry, job, output, data_identity):
    import numpy as np

    torch = deps.torch
    profile_name = model_entry["training_profile"]
    device = screen.resolve_device(torch, configuration["device"])
    if profile_name == "braille_stage_b":
        stage_b.configure_runtime_determinism(torch)
    else:
        deps.stem_module.configure_determinism(True)
    checkpoint_directory = output / "checkpoints" / digest(job.job_id)[:24]
    expected = {
        "configuration_id": digest(configuration), "job_id": job.job_id,
        "dataset_identity": data_identity, "run_kind": "portable_new_run",
    }
    if not commit_matches(stage_b, checkpoint_directory, expected):
        raise FileNotFoundError(f"No committed new-run checkpoint: {checkpoint_directory}")
    checkpoint_path = checkpoint_directory / "model.pt"
    checkpoint_sha = stage_b.sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validation_indices = screen.exact_folds(deps, bundle, job.split_seed)[job.fold - 1][1]
    if checkpoint["validation_indices"] != np.asarray(validation_indices, dtype=np.int64).tolist():
        raise RuntimeError("Checkpoint validation indices differ from the reconstructed fold.")
    model = stage_b.make_registered_model(deps, model_entry, bundle).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    adapter = screen.OperatorAdapter()
    batch_size = int(configuration["evaluation"]["batch_size_by_dataset"][bundle.name])
    registry = rng_registry(configuration)
    for condition, severity, realization in condition_grid(configuration):
        unit = f"{condition}_ms{severity}_r{realization}"
        unit_directory = output / "evaluation" / digest(job.job_id)[:24] / digest(unit)[:16]
        unit_directory.mkdir(parents=True, exist_ok=True)
        marker = unit_directory / "commit.json"
        csv_path = unit_directory / "predictions.csv"
        unit_identity = {
            **expected, "checkpoint_sha256": checkpoint_sha, "condition": condition,
            "severity_ms": severity, "realization": realization,
        }
        with stage_b.exclusive_writer_lock(unit_directory / "writer.lock", unit_identity):
            if marker.exists():
                committed = json.loads(marker.read_text(encoding="utf-8"))
                if any(committed.get(key) != value for key, value in unit_identity.items()):
                    raise RuntimeError(f"Evaluation identity mismatch: {unit_directory}")
                if stage_b.sha256_file(csv_path) != committed["predictions_sha256"]:
                    raise RuntimeError(f"Prediction checksum mismatch: {unit_directory}")
                print(f"resume evaluation: {job.job_id}/{unit}", flush=True)
                continue
            temporary = unit_directory / "predictions.csv.tmp"
            if csv_path.exists() or temporary.exists():
                raise RuntimeError(f"Uncommitted prediction output preserved: {unit_directory}")
            fields = [
                "run_kind", "configuration_id", "job_id", "probe_id", "dataset",
                "training_profile", "train_arm", "split_seed", "fold", "condition",
                "severity_ms", "realization", "sample_index", "sample_id", "participant",
                "label", "prediction", "correct", "active_before", "active_after",
            ]
            correct_total = active_before_total = active_after_total = samples_total = 0
            with temporary.open("x", encoding="utf-8", newline="") as handle, torch.inference_mode():
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for start in range(0, len(validation_indices), batch_size):
                    indices = validation_indices[start:start + batch_size]
                    arrays, before_counts, after_counts = [], [], []
                    for raw_index in indices:
                        index = int(raw_index)
                        array, _ = generate_condition(registry, adapter, bundle, index, condition, severity, realization)
                        arrays.append(np.asarray(array, dtype=np.float32))
                        before_counts.append(int(np.count_nonzero(screen.unpack_clean_sample(bundle, index))))
                        after_counts.append(int(np.count_nonzero(array)))
                    inputs = torch.as_tensor(np.stack(arrays)).to(device)
                    predictions = model(inputs).argmax(1).detach().cpu().tolist()
                    for position, raw_index in enumerate(indices):
                        index = int(raw_index)
                        label, prediction = int(bundle.labels[index]), int(predictions[position])
                        correct = int(label == prediction)
                        writer.writerow({
                            "run_kind": "portable_new_run", "configuration_id": digest(configuration),
                            "job_id": job.job_id, "probe_id": job.probe_id, "dataset": bundle.name,
                            "training_profile": profile_name, "train_arm": job.train_arm,
                            "split_seed": job.split_seed, "fold": job.fold, "condition": condition,
                            "severity_ms": severity, "realization": realization,
                            "sample_index": index, "sample_id": bundle.sample_ids[index],
                            "participant": "" if bundle.groups is None else str(bundle.groups[index]),
                            "label": label, "prediction": prediction, "correct": correct,
                            "active_before": before_counts[position], "active_after": after_counts[position],
                        })
                        correct_total += correct
                        active_before_total += before_counts[position]
                        active_after_total += after_counts[position]
                        samples_total += 1
            os.replace(temporary, csv_path)
            summary = {
                **unit_identity, "samples": samples_total, "correct": correct_total,
                "accuracy": correct_total / samples_total,
                "active_before": active_before_total, "active_after": active_after_total,
                "active_retention": active_after_total / active_before_total if active_before_total else 1.0,
                "summary_scope": "Descriptive pooled counts for this single fold/condition/realization.",
                "predictions_sha256": stage_b.sha256_file(csv_path),
            }
            stage_b.write_json_atomic(marker, summary)
            print(f"{job.job_id}/{unit} accuracy={summary['accuracy']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY / "examples" / "paper_probes.json")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "cache", "train", "eval"):
        command = subparsers.add_parser(name)
        command.add_argument("--execute", action="store_true")
        command.add_argument("--allow-training", action="store_true")
        command.add_argument("--probe", default="")
        command.add_argument("--arm", default="")
        command.add_argument("--seed", type=int, default=None)
        command.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()
    configuration = read_configuration(args.config.resolve())
    jobs = selected_jobs(configuration, args)
    if args.command == "plan" or not args.execute:
        describe(configuration, jobs, args.command)
        return
    if args.command == "train" and not args.allow_training:
        parser.error("Training requires both --execute and --allow-training.")
    if args.command == "cache":
        prepare_cache(configuration)
        return
    stage_b, screen, deps = load_runtime()
    bundles = {
        dataset: load_bundle(configuration, screen, dataset)
        for dataset in sorted({model["dataset"] for model, _ in jobs})
    }
    identities = {name: bundle_identity(screen, bundle) for name, bundle in bundles.items()}
    layouts = {name: {"samples": len(bundle.labels), "classes": len(bundle.classes),
                     "participants": 0 if bundle.groups is None else len(set(bundle.groups.tolist()))}
               for name, bundle in bundles.items()}
    output, _ = bind_run(configuration, stage_b, deps, identities, layouts)
    for model_entry, job_record in jobs:
        job = stage_b.TrainJob(**job_record)
        function = train_job if args.command == "train" else evaluate_job
        function(
            configuration, stage_b, screen, deps, bundles[job.dataset],
            model_entry, job, output, identities[job.dataset],
        )


if __name__ == "__main__":
    main()
