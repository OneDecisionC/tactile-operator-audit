"""Validated OOF aggregation and paired bootstrap, without model imports."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from tactile_audit.workflow import RETENTION_CONDITIONS


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def paired_bootstrap(dataset, differences, groups, labels, *, resamples=10000, seed=20260907):
    """Adapt the historical shared multinomial draws; input is sample x contrast."""
    differences = np.asarray(differences, dtype=np.float64)
    labels = np.asarray(labels)
    if differences.ndim != 2 or len(differences) != len(labels) or not len(labels):
        raise ValueError("Expected nonempty sample-by-contrast differences and aligned labels")
    if not np.isfinite(differences).all() or resamples < 2:
        raise ValueError("Finite differences and at least two resamples are required")
    rng = np.random.default_rng(seed)
    if dataset == "stemnist":
        groups = np.asarray(groups)
        if groups.shape != labels.shape or np.any(groups == ""):
            raise ValueError("STEMNIST requires one participant identifier per sample")
        unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        if len(unique) < 2:
            raise ValueError("Cluster bootstrap requires at least two participants")
        means = np.stack([differences[inverse == group].mean(axis=0) for group in range(len(counts))])
        weights = rng.multinomial(len(unique), np.full(len(unique), 1.0 / len(unique)), size=resamples)
        macro = weights @ means / len(unique)
        micro = (weights @ (means * counts[:, None])) / (weights @ counts)[:, None]
        return {
            "participant_macro": (means.mean(axis=0), np.quantile(macro, (0.025, 0.975), axis=0)),
            "sample_micro_cluster_ratio": (differences.mean(axis=0), np.quantile(micro, (0.025, 0.975), axis=0)),
        }
    if dataset != "braille":
        raise ValueError("Unknown dataset")
    samples = np.zeros((resamples, differences.shape[1]), dtype=np.float64)
    for label in np.unique(labels):
        class_values = differences[labels == label]
        count = len(class_values)
        for start in range(0, resamples, 128):
            stop = min(start + 128, resamples)
            weights = rng.multinomial(count, np.full(count, 1.0 / count), size=stop - start)
            samples[start:stop] += weights @ class_values
    samples /= len(labels)
    return {"sample_micro_class_stratified": (differences.mean(axis=0), np.quantile(samples, (0.025, 0.975), axis=0))}


def condition_cells(settings, condition):
    if condition == "clean":
        return [(0, 0)]
    if condition in RETENTION_CONDITIONS:
        return [(0, realization) for realization in range(settings["realizations"])]
    return [(severity, realization) for severity in settings["severities_ms"] for realization in range(settings["realizations"])]


def load_oof(run, manifest, model, arm):
    config = manifest["configuration"]
    if sorted(config["folds"]) != [1, 2, 3, 4, 5]:
        raise ValueError("OOF analysis requires all five folds, not a selected subset")
    dataset = model["dataset"]
    layout = manifest["dataset_layout"][dataset]
    count = layout["samples"]
    seeds = config["split_seeds"]
    settings = config["evaluation"]
    if "clean" not in settings["conditions"]:
        raise ValueError("Include clean evaluation to validate the complete OOF population")
    values = {condition: np.full((len(seeds), len(condition_cells(settings, condition)), count), np.nan)
              for condition in settings["conditions"]}
    after = {condition: np.full(array.shape, -1, dtype=np.int32) for condition, array in values.items()}
    labels = np.full(count, -1, dtype=np.int64)
    groups = np.full(count, "", dtype=object)
    sample_ids = np.full(count, "", dtype=object)
    before = np.full(count, -1, dtype=np.int64)
    folds = np.full((len(seeds), count), -1, dtype=np.int8)
    sources = []
    for seed_index, seed in enumerate(seeds):
        for fold in config["folds"]:
            job = f"{model['probe_id']}__{arm}__seed{seed}__fold{fold}"
            checkpoint_dir = run / "checkpoints" / digest(job)[:24]
            checkpoint_commit = json.loads((checkpoint_dir / "commit.json").read_text())
            checkpoint_identity = {"configuration_id": manifest["configuration_id"], "job_id": job,
                                   "dataset_identity": manifest["dataset_identity"][dataset], "run_kind": "portable_new_run"}
            if any(checkpoint_commit.get(key) != value for key, value in checkpoint_identity.items()):
                raise ValueError(f"Checkpoint commit identity mismatch: {job}")
            if file_sha(checkpoint_dir / "model.pt") != checkpoint_commit["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {job}")
            for condition in settings["conditions"]:
                for cell, (severity, realization) in enumerate(condition_cells(settings, condition)):
                    unit = f"{condition}_ms{severity}_r{realization}"
                    directory = run / "evaluation" / digest(job)[:24] / digest(unit)[:16]
                    marker = json.loads((directory / "commit.json").read_text())
                    expected = {"configuration_id": manifest["configuration_id"], "job_id": job,
                                "dataset_identity": manifest["dataset_identity"][dataset],
                                "checkpoint_sha256": checkpoint_commit["checkpoint_sha256"],
                                "condition": condition, "severity_ms": severity, "realization": realization,
                                "run_kind": "portable_new_run"}
                    if any(marker.get(key) != value for key, value in expected.items()):
                        raise ValueError(f"Evaluation identity mismatch: {job}/{unit}")
                    path = directory / "predictions.csv"
                    if file_sha(path) != marker["predictions_sha256"]:
                        raise ValueError(f"Prediction hash mismatch: {path}")
                    row_count = correct_sum = before_sum = after_sum = 0
                    with path.open(encoding="utf-8", newline="") as handle:
                        for row in csv.DictReader(handle):
                            row_expected = {**expected, "dataset": dataset, "probe_id": model["probe_id"], "train_arm": arm,
                                            "split_seed": seed, "fold": fold, "training_profile": model["training_profile"]}
                            for key, value in row_expected.items():
                                if key not in {"dataset_identity", "checkpoint_sha256"} and row.get(key) != str(value):
                                    raise ValueError(f"Prediction row identity mismatch in {key}: {path}")
                            index = int(row["sample_index"])
                            label, prediction, correct = int(row["label"]), int(row["prediction"]), int(row["correct"])
                            active_before, active_after = int(row["active_before"]), int(row["active_after"])
                            if not 0 <= index < count or not 0 <= label < layout["classes"] or not 0 <= prediction < layout["classes"]:
                                raise ValueError("Sample index, label, or prediction outside the declared dataset")
                            if correct != int(label == prediction) or min(active_before, active_after) < 0:
                                raise ValueError("Incorrect correctness/occupancy fields")
                            if not np.isnan(values[condition][seed_index, cell, index]):
                                raise ValueError("Duplicate OOF prediction")
                            if labels[index] >= 0 and (labels[index] != label or groups[index] != row["participant"] or sample_ids[index] != row["sample_id"] or before[index] != active_before):
                                raise ValueError("Unpaired sample identity, label, participant or clean occupancy")
                            if folds[seed_index, index] not in (-1, fold):
                                raise ValueError("Fold assignment changed across conditions")
                            labels[index], groups[index], sample_ids[index], before[index] = label, row["participant"], row["sample_id"], active_before
                            folds[seed_index, index] = fold
                            values[condition][seed_index, cell, index] = correct
                            after[condition][seed_index, cell, index] = active_after
                            row_count += 1
                            correct_sum += correct
                            before_sum += active_before
                            after_sum += active_after
                    if (row_count, correct_sum, before_sum, after_sum) != (marker["samples"], marker["correct"], marker["active_before"], marker["active_after"]):
                        raise ValueError("Evaluation summary differs from its prediction rows")
                    sources.append({"path": str(path.relative_to(run)), "sha256": marker["predictions_sha256"]})
    if len(set(sample_ids)) != count or any(not np.isfinite(array).all() for array in values.values()):
        raise ValueError("Missing OOF predictions or nonunique sample identities")
    if dataset == "stemnist":
        if "" in groups or len(set(groups)) != layout["participants"]:
            raise ValueError("Missing or inconsistent participant groups")
        for seed_index in range(len(seeds)):
            if any(len(set(folds[seed_index, groups == group])) != 1 for group in set(groups)):
                raise ValueError("A participant appears in multiple held-out folds")
    for condition in ("matched", "cf_matched", "pbj_identity_loss", "ipbj_plus_identity_loss",
                      "fixed_index_loss", "fixed_index_combined", "minimum_proposed_displacement_loss",
                      "minimum_proposed_displacement_combined", "fixed_random_priority_loss", "fixed_random_priority_combined"):
        if condition in after and "pbj" in after and not np.array_equal(after[condition], after["pbj"]):
            raise ValueError(f"Per-sample/channel-budget aggregate check failed: {condition} vs PBJ")
    return values, labels, groups, before, after, sources, sample_ids


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else None


def analyze_run(run, *, resamples=10000, bootstrap_seed=20260907):
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    config = manifest["configuration"]
    if digest(config) != manifest["configuration_id"] or manifest.get("release_version") != "0.2.0":
        raise ValueError("Expected a valid v0.2.0 portable run manifest")
    conditions, contrasts, provenance = [], [], []
    for model in config["models"]:
        arm_data = {}
        for arm in model["train_arms"]:
            values, labels, groups, before, after, sources, sample_ids = load_oof(run, manifest, model, arm)
            provenance.extend(sources)
            pooled = {name: array.mean(axis=1) for name, array in values.items()}
            arm_data[arm] = (pooled.get("pbj"), labels.copy(), groups.copy(), sample_ids.copy(), after.get("pbj"))
            for condition, samples in pooled.items():
                per_seed = samples.mean(axis=1) * 100
                mean, sd = summarize(per_seed)
                conditions.append({"dataset": model["dataset"], "probe_id": model["probe_id"], "train_arm": arm,
                                   "condition": condition, "estimand": "sample_micro", "mean_pct": mean, "sample_sd_pct": sd,
                                   "seed_values_pct": json.dumps(per_seed.tolist()),
                                   "pooled_active_retention_pct": float(100 * after[condition].sum() / (before.sum() * after[condition].shape[0] * after[condition].shape[1])) if before.sum() else 100.0})
                if model["dataset"] == "stemnist":
                    per_seed = np.stack([samples[:, groups == group].mean(axis=1) for group in np.unique(groups)]).mean(axis=0) * 100
                    mean, sd = summarize(per_seed)
                    conditions.append({**conditions[-1], "estimand": "participant_macro", "mean_pct": mean, "sample_sd_pct": sd,
                                       "seed_values_pct": json.dumps(per_seed.tolist()), "pooled_active_retention_pct": ""})
            if arm == "clean":
                formulas = {}
                if {"pbj", "cf_matched"} <= pooled.keys():
                    formulas["PBJ_minus_CF+M"] = pooled["pbj"] - pooled["cf_matched"]
                if "pbj_identity_loss" in pooled:
                    formulas["Clean_minus_identity_loss"] = pooled["clean"] - pooled["pbj_identity_loss"]
                for rule, loss, combined in [
                    ("assignment_preserving", "pbj_identity_loss", "ipbj_plus_identity_loss"),
                    ("fixed_index", "fixed_index_loss", "fixed_index_combined"),
                    ("minimum_proposed_displacement", "minimum_proposed_displacement_loss", "minimum_proposed_displacement_combined"),
                    ("fixed_random_priority", "fixed_random_priority_loss", "fixed_random_priority_combined")]:
                    if {"clean", "ipbj_movement", "pbj", loss, combined} <= pooled.keys():
                        formulas[f"{rule}_interaction"] = pooled[combined] - pooled["ipbj_movement"] - pooled[loss] + pooled["clean"]
                        formulas[f"{rule}_residual"] = pooled["pbj"] - pooled[combined]
                contrasts.extend(contrast_rows(model, arm, formulas, labels, groups, resamples, bootstrap_seed))
        transfer = {}
        if "pbj_aug" in arm_data:
            left, labels, groups, sample_ids, left_after = arm_data["pbj_aug"]
            for right_arm in ("matched_aug", "cf_matched_aug"):
                if right_arm in arm_data and left is not None:
                    right, right_labels, right_groups, right_ids, right_after = arm_data[right_arm]
                    if not np.array_equal(labels, right_labels) or not np.array_equal(groups, right_groups) or not np.array_equal(sample_ids, right_ids) or not np.array_equal(left_after, right_after):
                        raise ValueError("Training arms have inconsistent sample alignment")
                    transfer[f"PBJ_aug_minus_{right_arm}_on_PBJ"] = left - right
            contrasts.extend(contrast_rows(model, "between_arms", transfer, labels, groups, resamples, bootstrap_seed))
    return conditions, contrasts, provenance


def contrast_rows(model, arm, formulas, labels, groups, resamples, seed):
    if not formulas:
        return []
    names = list(formulas)
    differences = np.stack([formulas[name].mean(axis=0) for name in names], axis=1)
    intervals = paired_bootstrap(model["dataset"], differences, groups, labels, resamples=resamples, seed=seed)
    result = []
    for estimand, (means, bounds) in intervals.items():
        for index, name in enumerate(names):
            per_seed = formulas[name].mean(axis=1)
            if estimand == "participant_macro":
                per_seed = np.stack([formulas[name][:, groups == group].mean(axis=1) for group in np.unique(groups)]).mean(axis=0)
            _, sd = summarize(per_seed * 100)
            result.append({"dataset": model["dataset"], "probe_id": model["probe_id"], "train_arm": arm,
                           "contrast": name, "severity_ms": "pooled", "estimand": estimand,
                           "difference_pp": float(100 * means[index]), "sample_sd_pp": sd,
                           "ci95_low_pp": float(100 * bounds[0, index]), "ci95_high_pp": float(100 * bounds[1, index]),
                           "bootstrap_resamples": resamples, "bootstrap_seed": seed, "samples": len(labels),
                           "clusters": len(set(groups)) if model["dataset"] == "stemnist" else len(labels)})
    return result
