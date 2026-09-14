"""Independent Stage-B training/evaluation runner for the ICASSP study.

This file consumes the frozen JSON registry and deliberately does not inspect any
Stage-A result, manifest, log, or intermediate output.  A Stage-B training job is
identified only by ``(protocol, probe, arm, split seed, fold)``.  Its final-epoch
checkpoint and manifest form a transaction: the manifest is written last and
contains the checkpoint SHA-256.  Resume accepts a job only after re-hashing it.

The module keeps PyTorch imports lazy so plan/registry/integrity tests work in the
base Python environment.  Formal and smoke configurations have different hashes
and disjoint output roots; a smoke override can never be labelled formal.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import json
import os
import platform
import random
import socket
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# Must be configured before the lazy torch import initializes a CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

RUNNER_SCHEMA_VERSION = 1
DEFAULT_REGISTRY = Path("ICASSP20260813/icassp_stage_b_registry_v1.json")
EXPECTED_PROTOCOL = "icassp_stage_b_formal_v1"
TRAIN_ARMS = ("clean", "pbj_aug", "cf_aug", "prebin_aug")
TEST_CONDITIONS = ("clean", "prebin", "pbj", "cf", "matched", "cf_matched")
REQUIRED_TOP_LEVEL = {
    "protocol_id", "freeze", "project_root", "source_files", "data", "operators",
    "models", "training", "augmentation", "job_matrix", "phase_order", "evaluation",
    "primary_transfer_analysis", "execution_environment", "operator_cache",
    "scale_assertions",
}


class RegistryError(ValueError):
    """Raised for a clear, path-qualified frozen-registry schema violation."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    digest = hashlib.blake2b(digest_size=8, person=b"icasstgb")
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little") & 0x7FFF_FFFF_FFFF_FFFF


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, path)


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryError(f"{location} must be an object")
    return value


def _sequence(value: Any, location: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise RegistryError(f"{location} must be an array")
    return value


def load_registry(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise RegistryError(f"registry not found: {path}") from exc
    try:
        registry = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RegistryError(f"registry is not valid UTF-8 JSON: {path}: {exc}") from exc
    if not isinstance(registry, dict):
        raise RegistryError("registry root must be an object")
    return registry, sha256_bytes(raw)


def _expected_eval_conditions(registry: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    operators = ("prebin", "pbj", "cf", "matched", "cf_matched")
    specs = [("clean", 0, 0)]
    evaluation = _mapping(registry["evaluation"], "evaluation")
    for severity in evaluation["severities_ms"]:
        for realization in evaluation["realizations"]:
            specs.extend((operator, int(severity), int(realization)) for operator in operators)
    return specs


def validate_registry(
    registry: Mapping[str, Any], registry_path: Path | None = None, *, verify_files: bool = False
) -> dict[str, Any]:
    missing = sorted(REQUIRED_TOP_LEVEL - set(registry))
    if missing:
        raise RegistryError(f"registry missing required top-level fields: {missing}")
    schema = _mapping(registry.get("schema"), "schema")
    if schema.get("name") != "icassp_stage_b_train_registry" or schema.get("version") != 1:
        raise RegistryError("schema must be icassp_stage_b_train_registry version 1")
    required_declared = set(_sequence(schema.get("required_top_level"), "schema.required_top_level"))
    if required_declared != REQUIRED_TOP_LEVEL:
        raise RegistryError("schema.required_top_level differs from the runner contract")
    if registry.get("protocol_id") != EXPECTED_PROTOCOL:
        raise RegistryError(f"protocol_id must be {EXPECTED_PROTOCOL!r}")
    freeze = _mapping(registry["freeze"], "freeze")
    if freeze.get("stage_a_intermediate_performance_consulted") is not False:
        raise RegistryError("freeze.stage_a_intermediate_performance_consulted must be false")
    if freeze.get("checkpoint_selection") != "final_epoch_only_no_validation_selection":
        raise RegistryError("freeze.checkpoint_selection must freeze final epoch only")
    formal_environment = _mapping(registry["execution_environment"], "execution_environment")
    expected_environment_fields = {
        "formal_device", "python_version", "torch_version", "torch_cuda_version",
        "cudnn_version", "snntorch_version",
        "cuda_device_name", "cuda_compute_capability", "cuda_total_memory_bytes",
        "required_lif_backends", "resume_environment_rule",
    }
    if set(formal_environment) != expected_environment_fields:
        raise RegistryError("execution_environment fields differ from the frozen runner contract")
    environment_literals = {
        "formal_device": "cuda:0", "python_version": "3.11.14",
        "torch_version": "2.11.0+cu128", "torch_cuda_version": "12.8",
        "cudnn_version": 91900, "snntorch_version": "0.9.4",
        "cuda_device_name": "NVIDIA GeForce RTX 5060 Ti",
        "cuda_compute_capability": [12, 0], "cuda_total_memory_bytes": 17102864384,
    }
    for key, value in environment_literals.items():
        if formal_environment.get(key) != value:
            raise RegistryError(f"execution_environment.{key} must be frozen as {value!r}")
    if formal_environment.get("required_lif_backends") != {
        "braille_ratesnn_h192_l3": "snntorch.Leaky",
        "stemnist_paper_scnn": "snntorch.Leaky",
    }:
        raise RegistryError("execution_environment.required_lif_backends differs from freeze")
    if formal_environment.get("resume_environment_rule") != (
        "Formal checkpoint and evaluation resume must fail closed unless the resolved "
        "execution environment and LIF backend identity exactly match this registry."
    ):
        raise RegistryError("execution_environment.resume_environment_rule differs from freeze")
    sources = _mapping(registry["source_files"], "source_files")
    required_sources = {"braille_model_module", "stemnist_model_module", "stage_b_runtime_module"}
    if not required_sources.issubset(sources) or not set(sources).issubset(
        required_sources | {"stage_b_runner", "stage_b_operator_cache_builder"}
    ):
        raise RegistryError("source_files must freeze the three runtime modules and optional final stage_b_runner")
    for source_name, source_value in sources.items():
        source = _mapping(source_value, f"source_files.{source_name}")
        if not isinstance(source.get("path"), str) or len(str(source.get("sha256", ""))) != 64:
            raise RegistryError(f"source_files.{source_name} requires path and SHA-256")

    models = _sequence(registry["models"], "models")
    if len(models) != 6:
        raise RegistryError(f"models must contain exactly 6 probes, found {len(models)}")
    probes: dict[str, Mapping[str, Any]] = {}
    phase_counts = {"p0_core": 0, "p1_dense": 0}
    datasets = _mapping(registry["data"], "data")
    if set(datasets) != {"braille", "stemnist"}:
        raise RegistryError("data must contain exactly braille and stemnist")
    for index, raw_model in enumerate(models):
        model = _mapping(raw_model, f"models[{index}]")
        probe_id = model.get("probe_id")
        if not isinstance(probe_id, str) or not probe_id:
            raise RegistryError(f"models[{index}].probe_id must be a non-empty string")
        if probe_id in probes:
            raise RegistryError(f"duplicate probe_id: {probe_id}")
        if model.get("dataset") not in datasets:
            raise RegistryError(f"models[{index}].dataset is unknown")
        if model.get("phase") not in phase_counts:
            raise RegistryError(f"models[{index}].phase must be p0_core or p1_dense")
        if not model.get("parameter_verification", {}).get("verified"):
            raise RegistryError(f"models[{index}] parameter count is not verified")
        if int(model.get("parameter_count", -1)) <= 0:
            raise RegistryError(f"models[{index}].parameter_count must be positive")
        effective = _mapping(model.get("effective_config"), f"models[{index}].effective_config")
        if model.get("family") == "RateSNN" and int(effective.get("layers", 0)) < 2:
            raise RegistryError(f"models[{index}] RateSNN requires layers >= 2")
        if probe_id == "braille_ratesnn_h192_l3" and float(effective.get("dropout", -1)) != 0.2:
            raise RegistryError("models[0] frozen Braille RateSNN dropout must be 0.2")
        probes[probe_id] = model
        phase_counts[str(model["phase"])] += 1

    matrix = _mapping(_mapping(registry["job_matrix"], "job_matrix")["dimensions"], "job_matrix.dimensions")
    matrix_probes = list(_sequence(matrix.get("probe_id"), "job_matrix.dimensions.probe_id"))
    arms = list(_sequence(matrix.get("train_arm"), "job_matrix.dimensions.train_arm"))
    seeds = list(_sequence(matrix.get("split_seed"), "job_matrix.dimensions.split_seed"))
    folds = list(_sequence(matrix.get("fold"), "job_matrix.dimensions.fold"))
    if matrix_probes != [str(model["probe_id"]) for model in models]:
        raise RegistryError("job_matrix probe order/content must exactly match models")
    if arms != list(TRAIN_ARMS):
        raise RegistryError(f"job_matrix train_arm must be {list(TRAIN_ARMS)}")
    if seeds != [42, 123, 202] or folds != [1, 2, 3, 4, 5]:
        raise RegistryError("job_matrix must freeze seeds [42,123,202] and folds [1..5]")
    augmentation = _mapping(registry["augmentation"], "augmentation")
    if list(augmentation.get("train_arms", [])) != arms:
        raise RegistryError("augmentation.train_arms differs from job matrix")
    sampling = _mapping(augmentation.get("augmented_arm_sampling"), "augmentation.augmented_arm_sampling")
    if float(sampling.get("augmentation_probability", -1)) != 0.75:
        raise RegistryError("augmentation_probability must be frozen at 0.75")
    if list(sampling.get("severities_ms", [])) != [25, 50, 75]:
        raise RegistryError("training severities must be [25,50,75] ms")
    shared = _mapping(augmentation.get("shared_randomness"), "augmentation.shared_randomness")
    if any(field in shared.get("key_fields", []) for field in ("probe_id", "train_arm")):
        raise RegistryError("shared RNG key must exclude probe_id and train_arm")
    if not {"probe_id", "train_arm"}.issubset(set(shared.get("excluded_key_fields", []))):
        raise RegistryError("shared RNG exclusion must include probe_id and train_arm")

    training = _mapping(registry["training"], "training")
    loss = _mapping(training.get("loss"), "training.loss")
    scheduler = _mapping(training.get("lr_scheduler"), "training.lr_scheduler")
    clipping = _mapping(training.get("gradient_clipping"), "training.gradient_clipping")
    deterministic = _mapping(training.get("deterministic_execution"),
                             "training.deterministic_execution")
    if training.get("optimizer") != "AdamW" or float(training.get("learning_rate", -1)) != 0.002:
        raise RegistryError("training optimizer/lr must be AdamW at 0.002")
    if loss != {"name": "CrossEntropyLoss", "class_weights": None,
                "label_smoothing": 0.0, "reduction": "mean"}:
        raise RegistryError("training.loss must freeze unweighted mean CrossEntropyLoss")
    if scheduler.get("name") != "none":
        raise RegistryError("training.lr_scheduler.name must be none")
    if clipping != {"enabled": True, "type": "global_norm", "max_norm": 1.0}:
        raise RegistryError("training.gradient_clipping must freeze global norm 1.0")
    deterministic_expected = {
        "torch_deterministic_algorithms": True, "cudnn_deterministic": True,
        "cudnn_benchmark": False, "cublas_workspace_config": ":4096:8",
        "fail_closed_on_nondeterministic_operation": True,
    }
    if deterministic != deterministic_expected:
        raise RegistryError("training.deterministic_execution differs from frozen contract")
    if training.get("model_initialization_seed_key") != [
        "protocol_id", "weights", "probe_id", "split_seed", "fold"
    ]:
        raise RegistryError("training.model_initialization_seed_key differs from freeze")
    if training.get("epoch_permutation_seed_key") != [
        "protocol_id", "dataset", "split_seed", "fold", "epoch", "epoch_permutation"
    ]:
        raise RegistryError("training.epoch_permutation_seed_key differs from freeze")

    evaluation = _mapping(registry["evaluation"], "evaluation")
    definitions = _mapping(evaluation.get("metric_definitions"), "evaluation.metric_definitions")
    if set(definitions) != {"accuracy", "mAcc", "Retention", "MeanDrop", "ActiveRetention",
                           "fold_aggregation", "seed_summary"}:
        raise RegistryError("evaluation.metric_definitions is incomplete or has unknown metrics")
    transaction = _mapping(evaluation.get("prediction_transaction"),
                           "evaluation.prediction_transaction")
    expected_columns = ["unit_id", "job_id", "dataset", "probe_id", "train_arm",
                        "split_seed", "fold", "condition", "severity_ms", "realization",
                        "sample_index", "sample_id", "label", "prediction", "correct",
                        "active_before", "active_after"]
    if transaction.get("prediction_columns") != expected_columns:
        raise RegistryError("evaluation.prediction_transaction.prediction_columns differs from runner")
    if int(transaction.get("expected_units", -1)) != 16560 or int(
        transaction.get("expected_prediction_rows", -1)) != 21693600:
        raise RegistryError("evaluation prediction transaction scale differs from freeze")
    cache = _mapping(registry["operator_cache"], "operator_cache")
    if cache.get("schema") != {"name": "icassp_stage_b_operator_cache", "version": 1}:
        raise RegistryError("operator_cache.schema must freeze version 1")
    if cache.get("builder_source_key") != "stage_b_operator_cache_builder":
        raise RegistryError("operator_cache.builder_source_key differs from runner")
    if "stage_b_operator_cache_builder" not in sources:
        raise RegistryError("source_files.stage_b_operator_cache_builder is required")
    if cache.get("formal_cache_parent_relpath") != (
        "ICASSP20260813/experiments/stage_b_v1/operator_cache"
    ):
        raise RegistryError("operator_cache.formal_cache_parent_relpath differs from runner")
    if cache.get("condition_order") != ["prebin", "pbj", "cf", "matched", "cf_matched"]:
        raise RegistryError("operator_cache.condition_order differs from runner")
    if cache.get("severities_ms") != [25, 50, 75] or cache.get("realizations") != [0, 1, 2]:
        raise RegistryError("operator_cache severity/realization grid differs from runner")
    if int(cache.get("cached_conditions", -1)) != 45 or int(cache.get("inventory_files", -1)) != 90:
        raise RegistryError("operator_cache scale differs from runner")
    if cache.get("storage") != {"format": "numpy_npy_v2", "dtype": "uint8",
                                "layout": "NTP", "bitorder": "little",
                                "allow_pickle": False}:
        raise RegistryError("operator_cache.storage differs from runner")
    if cache.get("cache_id_rule") != (
        "sha256(canonical_json(schema, protocol_id, registry_sha256, condition_table, storage))[0:24]"
    ):
        raise RegistryError("operator_cache.cache_id_rule differs from builder API")
    if cache.get("spot_validation") != {
        "rule_id": "operator_cache_spot_v1", "samples_per_dataset": 64,
        "anchors": ["first", "last"], "regenerate_all_conditions": True,
    }:
        raise RegistryError("operator_cache.spot_validation differs from builder API")
    if cache.get("publication_transaction") != (
        "O_EXCL single-writer lock; versioned staging; flush/fsync/hash all NPY files; "
        "cache_manifest.json; commit.json written last; atomic same-volume directory publication."
    ):
        raise RegistryError("operator_cache.publication_transaction differs from builder API")
    policy = _mapping(cache.get("formal_eval_policy"), "operator_cache.formal_eval_policy")
    required_policy = {"cache_required": True, "deep_validate_before_first_formal_eval": True,
                       "live_corruption_fallback": False,
                       "bind_cache_content_and_manifest_hashes_to_each_eval_commit": True,
                       "smoke_may_use_live_generation": True}
    if policy != required_policy:
        raise RegistryError("operator_cache.formal_eval_policy differs from fail-closed contract")
    if "batch_size_by_dataset" in evaluation:
        batch_sizes = _mapping(evaluation["batch_size_by_dataset"],
                               "evaluation.batch_size_by_dataset")
        if batch_sizes != {"braille": 128, "stemnist": 32}:
            raise RegistryError("evaluation.batch_size_by_dataset must freeze braille=128, stemnist=32")

    scale = _mapping(registry["scale_assertions"], "scale_assertions")
    expected = {
        "probe_count": 6, "train_arm_count": 4, "seed_count": 3, "fold_count": 5,
        "train_jobs": 360, "checkpoints": 360, "p0_core_jobs": 240,
        "p1_dense_jobs": 120, "formal_evaluation_units": 16560,
        "prediction_rows_estimate": 21693600,
    }
    for key, value in expected.items():
        if int(scale.get(key, -1)) != value:
            raise RegistryError(f"scale_assertions.{key} must be {value}")
    jobs = len(matrix_probes) * len(arms) * len(seeds) * len(folds)
    if jobs != 360 or phase_counts != {"p0_core": 4, "p1_dense": 2}:
        raise RegistryError(f"derived plan scale mismatch: jobs={jobs}, phase probes={phase_counts}")
    if len(_expected_eval_conditions(registry)) != 46:
        raise RegistryError("evaluation must expand to clean + 5*3*3 = 46 conditions")
    primary = registry.get("primary_transfer_analysis", {})
    if primary.get("preregistered_representative_probe_id") != "braille_tcn_small_h64_b2_k3":
        raise RegistryError("primary transfer probe must be braille_tcn_small_h64_b2_k3")

    project_root = Path(str(registry.get("project_root", registry_path.parent if registry_path else "."))).resolve()
    verified_files = 0
    if verify_files:
        sources: list[tuple[Path, str, str]] = []
        for name, entry in _mapping(registry.get("source_files"), "source_files").items():
            item = _mapping(entry, f"source_files.{name}")
            sources.append((project_root / str(item["path"]), str(item["sha256"]), f"source_files.{name}"))
        operators = _mapping(registry["operators"], "operators")
        sources.append((project_root / str(operators["module_path"]), str(operators["module_sha256"]), "operators"))
        braille = _mapping(datasets["braille"], "data.braille")
        sources.append((project_root / str(braille["source_path"]), str(braille["source_sha256"]), "data.braille"))
        stem = _mapping(datasets["stemnist"], "data.stemnist")
        index = _mapping(stem["cache_index"], "data.stemnist.cache_index")
        sources.append((project_root / str(index["path"]), str(index["sha256"]), "data.stemnist.cache_index"))
        for index_number, item in enumerate(stem["cache_files"]):
            sources.append((project_root / str(item["path"]), str(item["sha256"]), f"data.stemnist.cache_files[{index_number}]"))
        for path, expected_sha, location in sources:
            if not path.is_file():
                raise RegistryError(f"{location} missing frozen file: {path}")
            actual = sha256_file(path)
            if actual != expected_sha:
                raise RegistryError(f"{location} SHA-256 mismatch: expected {expected_sha}, got {actual}")
            verified_files += 1
    return {"jobs": jobs, "eval_units": jobs * 46, "prediction_rows": 21693600,
            "phase_probe_counts": phase_counts, "verified_files": verified_files,
            "project_root": str(project_root)}


@dataclass(frozen=True)
class TrainJob:
    job_id: str
    probe_id: str
    dataset: str
    family: str
    phase: str
    train_arm: str
    split_seed: int
    fold: int
    checkpoint_relpath: str
    manifest_relpath: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def materialize_plan(registry: Mapping[str, Any]) -> list[TrainJob]:
    validate_registry(registry)
    by_probe = {str(model["probe_id"]): model for model in registry["models"]}
    dimensions = registry["job_matrix"]["dimensions"]
    template = registry["job_matrix"]["job_id_template"]
    ckpt_template = registry["job_matrix"]["checkpoint_relpath_template"]
    manifest_template = registry["job_matrix"]["manifest_relpath_template"]
    jobs: list[TrainJob] = []
    for probe_id in dimensions["probe_id"]:
        model = by_probe[probe_id]
        for arm in dimensions["train_arm"]:
            for seed in dimensions["split_seed"]:
                for fold in dimensions["fold"]:
                    values = {"protocol_id": registry["protocol_id"], "probe_id": probe_id,
                              "train_arm": arm, "split_seed": int(seed), "fold": int(fold),
                              "phase": model["phase"], "dataset": model["dataset"]}
                    jobs.append(TrainJob(
                        template.format(**values), probe_id, model["dataset"], model["family"],
                        model["phase"], arm, int(seed), int(fold),
                        ckpt_template.format(**values), manifest_template.format(**values)))
    jobs.sort(key=lambda job: (0 if job.phase == "p0_core" else 1,
                               job.probe_id, TRAIN_ARMS.index(job.train_arm),
                               job.split_seed, job.fold))
    if len(jobs) != 360 or len({job.job_id for job in jobs}) != 360:
        raise AssertionError("formal plan must contain 360 unique jobs")
    counts = {phase: sum(job.phase == phase for job in jobs) for phase in ("p0_core", "p1_dense")}
    if counts != {"p0_core": 240, "p1_dense": 120}:
        raise AssertionError(f"phase job counts differ from freeze: {counts}")
    if any(job.phase != "p0_core" for job in jobs[:240]) or any(
        job.phase != "p1_dense" for job in jobs[240:]
    ):
        raise AssertionError("formal scheduler must materialize every P0 job before P1")
    return jobs


def run_identity(registry_sha256: str, *, smoke_epochs: int = 0, smoke_max_samples: int = 0) -> dict[str, Any]:
    smoke = bool(smoke_epochs or smoke_max_samples)
    payload = {"runner_schema_version": RUNNER_SCHEMA_VERSION, "registry_sha256": registry_sha256,
               "mode": "smoke" if smoke else "formal", "formal": not smoke,
               "smoke_epochs": int(smoke_epochs), "smoke_max_samples": int(smoke_max_samples)}
    payload["config_hash"] = sha256_bytes(canonical_json(payload).encode("utf-8"))
    return payload


def execution_environment(torch: Any, requested_device: str, *, formal: bool) -> dict[str, Any]:
    requested = str(requested_device)
    if requested == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = str(torch.device(requested))
    device_type = torch.device(resolved).type
    if formal and device_type != "cuda":
        raise RuntimeError("formal Stage-B execution is frozen to CUDA; CPU is smoke-only")
    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    try:
        snntorch_module = importlib.import_module("snntorch")
        snntorch_version = str(getattr(snntorch_module, "__version__", "unknown"))
    except Exception:
        snntorch_version = "unavailable"
    if device_type == "cuda":
        index = torch.device(resolved).index
        index = torch.cuda.current_device() if index is None else index
        device_name = str(torch.cuda.get_device_name(index))
        capability = list(torch.cuda.get_device_capability(index))
        total_memory = int(torch.cuda.get_device_properties(index).total_memory)
        resolved_device = f"cuda:{index}"
    else:
        device_name, capability, total_memory, resolved_device = "cpu", [], 0, "cpu"
    return {
        "resolved_device": resolved_device, "device_type": device_type, "device_name": device_name,
        "cuda_compute_capability": capability, "cuda_total_memory_bytes": total_memory,
        "python_version": platform.python_version(), "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda or "unavailable"),
        "cudnn_version": (int(torch.backends.cudnn.version())
                          if torch.backends.cudnn.is_available() else 0),
        "snntorch_version": snntorch_version,
    }


def validate_formal_execution_environment(registry: Mapping[str, Any],
                                          environment: Mapping[str, Any]) -> None:
    frozen = registry.get("execution_environment")
    if frozen is None:
        return
    expected = _mapping(frozen, "execution_environment")
    comparisons = {
        "resolved_device": expected.get("formal_device"),
        "python_version": expected.get("python_version"),
        "torch_version": expected.get("torch_version"),
        "torch_cuda_version": expected.get("torch_cuda_version"),
        "cudnn_version": expected.get("cudnn_version"),
        "snntorch_version": expected.get("snntorch_version"),
        "device_name": expected.get("cuda_device_name"),
        "cuda_compute_capability": expected.get("cuda_compute_capability"),
        "cuda_total_memory_bytes": expected.get("cuda_total_memory_bytes"),
    }
    mismatches = {key: (environment.get(key), value) for key, value in comparisons.items()
                  if value is not None and environment.get(key) != value}
    if mismatches:
        raise RuntimeError(f"formal execution environment differs from frozen registry: {mismatches}")


def bind_execution_identity(identity: Mapping[str, Any], environment: Mapping[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in identity.items() if key != "config_hash"}
    payload["execution_environment"] = dict(environment)
    payload["resolved_device"] = environment["resolved_device"]
    payload["config_hash"] = sha256_bytes(canonical_json(payload).encode("utf-8"))
    return payload


@contextlib.contextmanager
def exclusive_writer_lock(lock_path: Path, metadata: Mapping[str, Any]):
    """Acquire one fail-closed O_EXCL writer lock; never remove another lock."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"pid": os.getpid(), "hostname": socket.gethostname(),
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              **dict(metadata)}
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(str(lock_path), flags, 0o600)
    except FileExistsError as exc:
        try:
            existing = lock_path.read_text(encoding="utf-8")
        except Exception:
            existing = "<unreadable>"
        raise RuntimeError(
            f"writer lock already exists (active or stale); manual audit/removal required: "
            f"{lock_path}; metadata={existing}"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
            handle.flush(); os.fsync(handle.fileno())
        yield record
    finally:
        # Only the process that successfully created this lock reaches here.
        try:
            lock_path.unlink()
        except FileNotFoundError as exc:
            raise RuntimeError(f"owned writer lock disappeared unexpectedly: {lock_path}") from exc


def shared_draw_seed(registry: Mapping[str, Any], job: TrainJob, epoch: int,
                     global_sample_index: int, draw_kind: str) -> int:
    """Counter key intentionally excludes probe/model/arm for paired training."""
    return stable_seed(registry["protocol_id"], job.dataset, job.split_seed, job.fold,
                       int(epoch), int(global_sample_index), draw_kind)


def augmentation_decision(registry: Mapping[str, Any], job: TrainJob, epoch: int,
                          global_sample_index: int) -> tuple[bool, int, int]:
    gate_rng = np.random.Generator(np.random.PCG64(shared_draw_seed(
        registry, job, epoch, global_sample_index, "augmentation_gate")))
    severity_rng = np.random.Generator(np.random.PCG64(shared_draw_seed(
        registry, job, epoch, global_sample_index, "severity_choice")))
    severity_values = registry["augmentation"]["augmented_arm_sampling"]["severities_ms"]
    augmented = bool(gate_rng.random() < 0.75)
    severity = int(severity_values[int(severity_rng.integers(0, len(severity_values)))])
    operator_seed = shared_draw_seed(registry, job, epoch, global_sample_index,
                                     "operator_base_randomness")
    return augmented, severity, operator_seed


def model_initialization_seed(registry: Mapping[str, Any], job: TrainJob) -> int:
    """Frozen initialization key: identical across the four arms of one fold."""
    return stable_seed(registry["protocol_id"], "weights", job.probe_id,
                       job.split_seed, job.fold)


def seed_model_initialization_rngs(torch: Any, initialization_seed: int) -> dict[str, int]:
    """Seed every conventional runtime RNG before constructing a model."""
    python_seed = int(initialization_seed)
    numpy_seed = int(initialization_seed) % (2**32)
    random.seed(python_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(python_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(python_seed)
    return {"python_random_seed": python_seed, "numpy_random_seed": numpy_seed,
            "torch_seed": python_seed, "torch_cuda_seed_all": python_seed}


def _resolve_artifacts(project_root: Path, job: TrainJob, identity: Mapping[str, Any]) -> tuple[Path, Path]:
    if identity["formal"]:
        return project_root / job.checkpoint_relpath, project_root / job.manifest_relpath
    root = project_root / "ICASSP20260813" / "experiments" / "stage_b_smoke" / str(identity["config_hash"])
    relative = Path(job.phase) / job.dataset / job.probe_id / job.train_arm / f"seed{job.split_seed}" / f"fold{job.fold}"
    return root / "checkpoints" / relative / "final.pt", root / "manifests" / relative.with_suffix(".json")


def validate_job_commit(checkpoint_path: Path, manifest_path: Path, job: TrainJob,
                        registry_sha256: str, config_hash: str,
                        execution_environment_expected: Mapping[str, Any] | None = None
                        ) -> Mapping[str, Any] | None:
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"invalid committed job manifest {manifest_path}: {exc}") from exc
    for key, expected in (("job_id", job.job_id), ("registry_sha256", registry_sha256),
                          ("config_hash", config_hash)):
        if manifest.get(key) != expected:
            raise RuntimeError(f"committed job identity mismatch for {job.job_id}: {key}")
    if not checkpoint_path.is_file():
        raise RuntimeError(f"committed checkpoint is missing: {checkpoint_path}")
    actual = sha256_file(checkpoint_path)
    if actual != manifest.get("checkpoint_sha256"):
        raise RuntimeError(f"checkpoint tamper/hash mismatch for {job.job_id}: {actual}")
    if manifest.get("checkpoint_rule") != "final_epoch_only_no_validation_selection":
        raise RuntimeError(f"checkpoint selection rule mismatch for {job.job_id}")
    if execution_environment_expected is not None and manifest.get(
        "execution_environment"
    ) != dict(execution_environment_expected):
        raise RuntimeError(f"checkpoint execution environment mismatch for {job.job_id}")
    return manifest


def require_committed_jobs(registry: Mapping[str, Any], registry_sha256: str,
                           project_root: Path, identity: Mapping[str, Any],
                           jobs: Sequence[TrainJob], purpose: str) -> None:
    missing: list[str] = []
    for job in jobs:
        checkpoint, manifest = _resolve_artifacts(project_root, job, identity)
        try:
            committed = validate_job_commit(checkpoint, manifest, job, registry_sha256,
                                            str(identity["config_hash"]),
                                            identity.get("execution_environment"))
        except RuntimeError:
            raise
        if committed is None:
            missing.append(job.job_id)
    if missing:
        raise RuntimeError(f"{purpose} requires {len(jobs)} verified checkpoint commits; "
                           f"missing {len(missing)} (first: {missing[0]})")


def commit_checkpoint_transaction(checkpoint_path: Path, manifest_path: Path, checkpoint_writer: Any,
                                  manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Write checkpoint first and atomic manifest last (the commit marker)."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp.{os.getpid()}")
    checkpoint_writer(temporary)
    os.replace(temporary, checkpoint_path)
    committed = dict(manifest)
    committed["checkpoint_sha256"] = sha256_file(checkpoint_path)
    committed["checkpoint_bytes"] = checkpoint_path.stat().st_size
    committed["committed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json_atomic(manifest_path, committed)
    return committed


def _load_runtime(project_root: Path):
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    screen = importlib.import_module("run_icassp_jitter_screening")
    deps = screen.import_runtime_dependencies()
    configure_runtime_determinism(deps.torch)
    return screen, deps


def configure_runtime_determinism(torch: Any) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be set before CUDA initialization")
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _load_bundle(screen: Any, registry: Mapping[str, Any], dataset: str, project_root: Path):
    if dataset == "braille":
        bundle = screen.load_braille_bundle(project_root)
    else:
        cache = project_root / registry["data"]["stemnist"]["cache_root"]
        bundle = screen.load_stemnist_bundle(cache, verify_hash=True)
    frozen = registry["data"][dataset]
    if (len(bundle.labels), len(bundle.classes), bundle.channels, bundle.time_steps) != (
        int(frozen["samples"]), int(frozen["classes"]), int(frozen["channels"]), int(frozen["time_bins"])):
        raise RuntimeError(f"loaded {dataset} identity differs from frozen registry")
    if dataset == "stemnist" and len(set(bundle.groups.tolist())) != 34:
        raise RuntimeError("STEMNIST unique participant count must be 34")
    return bundle


def _model_args(model: Mapping[str, Any]) -> SimpleNamespace:
    cfg = model["effective_config"]
    return SimpleNamespace(
        hidden=int(cfg.get("hidden", 0)),
        layers=int(cfg.get("layers", cfg.get("blocks_via_args_layers", 2))),
        beta=float(cfg.get("beta", 0.9)), threshold_v=float(cfg.get("threshold_v", 1.0)),
        dropout=float(cfg.get("dropout", 0.3)), snn_readout=str(cfg.get("readout", "spikes")),
    )


def make_registered_model(deps: Any, model: Mapping[str, Any], bundle: Any):
    args = _model_args(model)
    family, dataset = model["family"], model["dataset"]
    if dataset == "braille":
        name = {"RateSNN": "snn", "TCN": "tcn", "Conv1D": "conv1d"}[family]
        instance = deps.braille_module.make_model(name, (len(bundle.labels), bundle.time_steps, bundle.channels),
                                                  len(bundle.classes), args)
    elif family == "paper_scnn":
        instance = deps.stem_module.make_model("paper_scnn", args)
    elif family == "TCN":
        instance = deps.stem_module.make_model("tcn", args)
    elif family == "Conv1D":
        instance = deps.braille_module.Conv1DClassifier(bundle.channels, args.hidden,
                                                        len(bundle.classes), args.dropout)
    else:
        raise RuntimeError(f"unsupported registered model family: {dataset}/{family}")
    actual = sum(parameter.numel() for parameter in instance.parameters())
    if actual != int(model["parameter_count"]):
        raise RuntimeError(f"{model['probe_id']} parameter count {actual} != frozen {model['parameter_count']}")
    return instance


def concrete_model_backend(model: Any, model_entry: Mapping[str, Any], *, formal: bool) -> str:
    if model_entry["family"] not in ("paper_scnn", "RateSNN"):
        return f"pytorch.{model_entry['family']}"
    if model_entry["family"] == "paper_scnn":
        neuron = getattr(model, "lif1", None)
    else:
        lif_layers = getattr(model, "lif_layers", [])
        neuron = lif_layers[0] if len(lif_layers) else None
    neuron_class = type(neuron)
    is_snntorch = neuron_class.__name__ == "Leaky" and neuron_class.__module__.startswith("snntorch")
    backend = "snntorch.Leaky" if is_snntorch else f"{neuron_class.__module__}.{neuron_class.__name__}"
    if formal and not is_snntorch:
        raise RuntimeError(
            f"formal {model_entry['probe_id']} requires concrete backend snntorch.Leaky; got {backend}"
        )
    return backend


class StageBTrainingDataset:
    def __init__(self, screen: Any, bundle: Any, indices: Sequence[int], registry: Mapping[str, Any],
                 job: TrainJob, adapter: Any, epoch: int):
        self.screen, self.bundle = screen, bundle
        self.indices = np.asarray(indices, dtype=np.int64)
        self.registry, self.job, self.adapter, self.epoch = registry, job, adapter, int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        clean = self.screen.unpack_clean_sample(self.bundle, index)
        augmented, severity, operator_seed = augmentation_decision(self.registry, self.job, self.epoch, index)
        output = clean
        if self.job.train_arm != "clean" and augmented:
            if self.job.train_arm == "prebin_aug":
                output, _ = self.adapter.prebin_binary(
                    self.bundle.raw_getter(index), severity, self.bundle.duration_seconds, operator_seed,
                    n_steps=self.bundle.time_steps, n_channels=self.bundle.channels, realization=0)
            else:
                arrays, _ = self.adapter.postbin_bundle(clean, severity // 25, operator_seed, 0)
                output = arrays["pbj" if self.job.train_arm == "pbj_aug" else "cf"]
        return np.asarray(output, dtype=np.float32), int(self.bundle.labels[index]), index


def _collate(torch: Any):
    def collate(batch: Sequence[tuple[np.ndarray, int, int]]):
        return (torch.as_tensor(np.stack([x[0] for x in batch])),
                torch.as_tensor([x[1] for x in batch], dtype=torch.long),
                torch.as_tensor([x[2] for x in batch], dtype=torch.long))
    return collate


def train_one_job(registry: Mapping[str, Any], registry_sha256: str, project_root: Path,
                  job: TrainJob, identity: Mapping[str, Any], device_name: str = "auto",
                  *, frozen_files_verified: bool = False,
                  phase_gate_verified: bool = False) -> str:
    if not frozen_files_verified:
        validate_registry(registry, verify_files=True)
    screen, deps = _load_runtime(project_root); torch = deps.torch
    current_environment = execution_environment(torch, device_name, formal=bool(identity["formal"]))
    if identity["formal"]:
        validate_formal_execution_environment(registry, current_environment)
    if "execution_environment" in identity:
        if identity["execution_environment"] != current_environment:
            raise RuntimeError("requested/runtime execution environment differs from bound identity")
        identity = dict(identity)
    else:
        identity = bind_execution_identity(identity, current_environment)
    if identity["formal"] and job.phase == "p1_dense" and not phase_gate_verified:
        p0_jobs = [candidate for candidate in materialize_plan(registry)
                   if candidate.phase == "p0_core"]
        require_committed_jobs(registry, registry_sha256, project_root, identity,
                               p0_jobs, "formal P1 start gate")
    checkpoint_path, manifest_path = _resolve_artifacts(project_root, job, identity)
    if validate_job_commit(checkpoint_path, manifest_path, job, registry_sha256,
                           identity["config_hash"], current_environment):
        return "resumed"
    lock_path = manifest_path.with_name(f"{manifest_path.name}.writer.lock")
    lock_metadata = {"scope": "training_job", "job_id": job.job_id,
                     "config_hash": identity["config_hash"],
                     "resolved_device": current_environment["resolved_device"]}
    with exclusive_writer_lock(lock_path, lock_metadata):
        job_started = time.monotonic()
        if validate_job_commit(checkpoint_path, manifest_path, job, registry_sha256,
                               identity["config_hash"], current_environment):
            return "resumed"
        device = torch.device(current_environment["resolved_device"])
        bundle = _load_bundle(screen, registry, job.dataset, project_root)
        folds = screen.exact_folds(deps, bundle, job.split_seed)
        train_indices, validation_indices = folds[job.fold - 1]
        max_samples = int(identity["smoke_max_samples"])
        if max_samples:
            train_indices = train_indices[:max_samples]
            validation_indices = validation_indices[:max_samples]
        model_entry = next(model for model in registry["models"] if model["probe_id"] == job.probe_id)
        initialization_seed = model_initialization_seed(registry, job)
        initialization_rngs = seed_model_initialization_rngs(torch, initialization_seed)
        model = make_registered_model(deps, model_entry, bundle).to(device)
        model_backend = concrete_model_backend(model, model_entry, formal=bool(identity["formal"]))
        training = registry["training"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                      weight_decay=float(training["weight_decay"]))
        epochs = int(identity["smoke_epochs"] or training["dataset_settings"][job.dataset]["epochs"])
        batch_size = int(training["dataset_settings"][job.dataset]["batch_size"])
        adapter = screen.OperatorAdapter(); history: list[dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            dataset = StageBTrainingDataset(screen, bundle, train_indices, registry, job, adapter, epoch)
            permutation_seed = stable_seed(registry["protocol_id"], job.dataset, job.split_seed,
                                           job.fold, epoch, "epoch_permutation")
            generator = torch.Generator().manual_seed(permutation_seed)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                 generator=generator, num_workers=0,
                                                 collate_fn=_collate(torch))
            model.train(); total_loss = 0.0; correct = 0; seen = 0
            for inputs, labels, _ in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True); logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(logits, labels); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                total_loss += float(loss.item()) * len(labels)
                correct += int((logits.argmax(1) == labels).sum().item()); seen += len(labels)
            history.append({"epoch": epoch, "train_loss": total_loss / max(seen, 1),
                            "train_accuracy": correct / max(seen, 1)})
            print(f"train-progress job={job.job_id} epoch={epoch}/{epochs} "
                  f"loss={history[-1]['train_loss']:.8f} "
                  f"accuracy={history[-1]['train_accuracy']:.8f} "
                  f"elapsed_seconds={time.monotonic() - job_started:.3f}", flush=True)

        common = {"runner_schema_version": RUNNER_SCHEMA_VERSION,
                  "protocol_id": registry["protocol_id"], "registry_sha256": registry_sha256,
                  "config_hash": identity["config_hash"], "formal": identity["formal"],
                  "job": job.as_dict(), "execution_environment": current_environment,
                  "resolved_device": current_environment["resolved_device"],
                  "model_backend": model_backend, "snntorch_version": current_environment["snntorch_version"],
                  "checkpoint_rule": "final_epoch_only_no_validation_selection"}
        payload = {**common, "epoch": epochs, "state_dict": model.state_dict(),
                   "optimizer_state_dict": optimizer.state_dict(), "history": history,
                   "train_indices_sha256": sha256_bytes(np.asarray(train_indices, dtype="<i8").tobytes()),
                   "validation_indices": np.asarray(validation_indices, dtype=np.int64).tolist(),
                   "parameter_count": int(model_entry["parameter_count"]),
                   "model_initialization_seed": initialization_seed,
                   "model_initialization_rngs": initialization_rngs}
        manifest = {**common, "job_id": job.job_id, "final_epoch": epochs,
                    "train_samples": len(train_indices), "validation_samples": len(validation_indices),
                    "model_initialization_seed": initialization_seed,
                    "model_initialization_rngs": initialization_rngs, "history": history}
        commit_checkpoint_transaction(checkpoint_path, manifest_path,
                                      lambda path: torch.save(payload, path), manifest)
        return "trained"


def evaluation_unit_id(job: TrainJob, condition: str, severity: int, realization: int) -> str:
    return f"{job.job_id}__{condition}__ms{severity}__r{realization}"


def _eval_unit_dir(project_root: Path, identity: Mapping[str, Any], unit_id: str) -> Path:
    mode = "stage_b_v1" if identity["formal"] else f"stage_b_smoke/{identity['config_hash']}"
    # Keep Windows paths well below MAX_PATH while the commit marker retains
    # the full human-readable unit identity.
    shard = sha256_bytes(unit_id.encode("utf-8"))[:24]
    return project_root / "ICASSP20260813" / "experiments" / mode / "evaluation_units" / shard[:2] / shard


def validate_eval_commit(unit_dir: Path, unit_id: str, config_hash: str,
                         execution_environment_expected: Mapping[str, Any] | None = None,
                         cache_binding_expected: Mapping[str, Any] | None = None
                         ) -> Mapping[str, Any] | None:
    commit = unit_dir / "commit.json"
    if not commit.exists():
        return None
    marker = json.loads(commit.read_text(encoding="utf-8"))
    if marker.get("unit_id") != unit_id or marker.get("config_hash") != config_hash:
        raise RuntimeError(f"evaluation commit identity mismatch: {unit_id}")
    if execution_environment_expected is not None and marker.get(
        "execution_environment"
    ) != dict(execution_environment_expected):
        raise RuntimeError(f"evaluation execution environment mismatch: {unit_id}")
    if cache_binding_expected is not None and marker.get("operator_cache") != dict(
        cache_binding_expected
    ):
        raise RuntimeError(f"evaluation operator-cache binding mismatch: {unit_id}")
    for name, digest in marker.get("artifacts", {}).items():
        path = unit_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"evaluation shard tamper/hash mismatch: {path}")
    return marker


def commit_eval_unit(unit_dir: Path, unit_id: str, config_hash: str,
                     result: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]],
                     execution_environment_record: Mapping[str, Any] | None = None,
                     cache_binding_record: Mapping[str, Any] | None = None) -> None:
    unit_dir.mkdir(parents=True, exist_ok=True)
    result_path, prediction_path = unit_dir / "result.json", unit_dir / "predictions.csv"
    write_json_atomic(result_path, dict(result)); write_csv_atomic(prediction_path, predictions)
    write_json_atomic(unit_dir / "commit.json", {
        "runner_schema_version": RUNNER_SCHEMA_VERSION, "unit_id": unit_id,
        "config_hash": config_hash, "prediction_rows": len(predictions),
        "execution_environment": (dict(execution_environment_record)
                                  if execution_environment_record is not None else None),
        "operator_cache": (dict(cache_binding_record)
                           if cache_binding_record is not None else None),
        "artifacts": {"result.json": sha256_file(result_path),
                      "predictions.csv": sha256_file(prediction_path)}})


def generate_stage_b_condition_sample(
    registry: Mapping[str, Any], screen: Any, adapter: Any, bundle: Any,
    sample_index: int, condition: str, severity_ms: int, realization: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate evaluation corruption from the Stage-B registry protocol root."""
    clean = screen.unpack_clean_sample(bundle, sample_index)
    if condition == "clean":
        return clean.copy(), {"identity": True, "operator_root_protocol": registry["protocol_id"]}
    operator_seed = stable_seed(registry["protocol_id"], "operator_root_v1", bundle.name,
                                bundle.sample_ids[sample_index])
    if condition == "prebin":
        output, audit = adapter.prebin_binary(
            bundle.raw_getter(sample_index), severity_ms, bundle.duration_seconds,
            operator_seed, n_steps=bundle.time_steps, n_channels=bundle.channels,
            realization=realization)
    elif condition in ("pbj", "cf", "matched", "cf_matched"):
        arrays, audit = adapter.postbin_bundle(clean, severity_ms // 25,
                                               operator_seed, realization)
        output = arrays[condition]
    else:
        raise ValueError(f"unknown Stage-B evaluation condition: {condition}")
    audit = dict(audit)
    audit.update({"operator_root_protocol": registry["protocol_id"],
                  "operator_root_seed": operator_seed})
    return output, audit


def evaluation_batch_size(registry: Mapping[str, Any], dataset: str) -> int:
    frozen = registry["evaluation"].get(
        "batch_size_by_dataset", {"braille": 128, "stemnist": 32})
    try:
        value = int(frozen[dataset])
    except Exception as exc:
        raise RegistryError(f"evaluation batch size is missing for {dataset}") from exc
    if value <= 0:
        raise RegistryError(f"evaluation batch size must be positive for {dataset}")
    return value


def load_formal_operator_cache(registry: Mapping[str, Any], registry_sha256: str,
                               registry_path: Path, project_root: Path):
    """Deep-validate and open the only cache allowed for formal evaluation."""
    module = importlib.import_module("icassp_stage_b_operator_cache")
    parent = project_root / str(registry["operator_cache"]["formal_cache_parent_relpath"])
    identity = module.cache_id(registry, registry_sha256)
    cache_root = parent.resolve() / identity
    reader = module.PackedOperatorCache(cache_root, registry_path, deep_validate=True)
    binding = {
        "cache_id": identity,
        "cache_content_sha256": str(reader.content_sha256),
        "cache_manifest_sha256": sha256_file(cache_root / "cache_manifest.json"),
    }
    if reader.manifest.get("cache_identity", {}).get("cache_id") != identity:
        raise RuntimeError("formal operator cache reader identity mismatch")
    return reader, binding


def cached_or_live_batch(
    registry: Mapping[str, Any], identity: Mapping[str, Any], screen: Any,
    adapter: Any, bundle: Any, condition: str, severity: int, realization: int,
    batch_indices: Sequence[int], operator_cache_reader: Any | None,
    cache_binding: Mapping[str, Any] | None,
) -> np.ndarray:
    if condition == "clean":
        return np.stack([screen.unpack_clean_sample(bundle, int(index))
                         for index in batch_indices]).astype(np.uint8, copy=False)
    if identity["formal"]:
        if operator_cache_reader is None or cache_binding is None:
            raise RuntimeError("formal corrupted evaluation requires the frozen operator cache")
        return np.asarray(operator_cache_reader.get_batch(
            bundle.name, condition, severity, realization, batch_indices, dtype=np.uint8),
            dtype=np.uint8)
    return np.stack([generate_stage_b_condition_sample(
        registry, screen, adapter, bundle, int(index), condition, severity, realization)[0]
        for index in batch_indices]).astype(np.uint8, copy=False)


def _evaluate_unit_locked(
    registry: Mapping[str, Any], screen: Any, adapter: Any, bundle: Any, model: Any,
    torch: Any, device: Any, validation_indices: np.ndarray, job: TrainJob,
    identity: Mapping[str, Any], checkpoint_path: Path, unit_id: str, unit_dir: Path,
    condition: str, severity: int, realization: int, eval_batch_size: int,
    model_backend: str, operator_cache_reader: Any | None,
    cache_binding: Mapping[str, Any] | None,
) -> int:
    environment = identity["execution_environment"]
    existing = validate_eval_commit(unit_dir, unit_id, identity["config_hash"], environment,
                                    cache_binding)
    if existing:
        return int(existing["prediction_rows"])
    predictions: list[dict[str, Any]] = []; active_before = active_after = correct = 0
    with torch.no_grad():
        for start in range(0, len(validation_indices), eval_batch_size):
            batch_indices = [int(value) for value in
                             validation_indices[start:start + eval_batch_size]]
            batch_outputs: list[np.ndarray] = []; batch_counts: list[tuple[int, int]] = []
            condition_outputs = cached_or_live_batch(
                registry, identity, screen, adapter, bundle, condition, severity,
                realization, batch_indices, operator_cache_reader, cache_binding)
            for offset, index in enumerate(batch_indices):
                clean = screen.unpack_clean_sample(bundle, index)
                output = np.asarray(condition_outputs[offset], dtype=np.uint8)
                before = int(np.count_nonzero(clean)); after = int(np.count_nonzero(output))
                batch_outputs.append(output); batch_counts.append((before, after))
            logits = model(torch.as_tensor(np.stack(batch_outputs), dtype=torch.float32, device=device))
            batch_predictions = logits.argmax(1).detach().cpu().tolist()
            for index, prediction_value, (before, after) in zip(
                batch_indices, batch_predictions, batch_counts
            ):
                prediction = int(prediction_value); label = int(bundle.labels[index])
                is_correct = int(prediction == label)
                correct += is_correct; active_before += before; active_after += after
                predictions.append({"unit_id": unit_id, "job_id": job.job_id,
                                    "dataset": job.dataset, "probe_id": job.probe_id,
                                    "train_arm": job.train_arm, "split_seed": job.split_seed,
                                    "fold": job.fold, "condition": condition,
                                    "severity_ms": severity, "realization": realization,
                                    "sample_index": index, "sample_id": bundle.sample_ids[index],
                                    "label": label, "prediction": prediction, "correct": is_correct,
                                    "active_before": before, "active_after": after})
    result = {"unit_id": unit_id, "job_id": job.job_id,
              "config_hash": identity["config_hash"],
              "checkpoint_sha256": sha256_file(checkpoint_path), "condition": condition,
              "severity_ms": severity, "realization": realization, "samples": len(predictions),
              "correct": correct, "accuracy": correct / max(len(predictions), 1),
              "active_before": active_before, "active_after": active_after,
              "active_retention": active_after / max(active_before, 1),
              "evaluation_batch_size": eval_batch_size,
              "execution_environment": environment,
              "resolved_device": environment["resolved_device"],
              "model_backend": model_backend,
              "snntorch_version": environment["snntorch_version"],
              "operator_cache": (dict(cache_binding) if cache_binding is not None else None)}
    commit_eval_unit(unit_dir, unit_id, identity["config_hash"], result, predictions,
                     environment, cache_binding)
    return len(predictions)


def evaluate_one_job(registry: Mapping[str, Any], registry_sha256: str, project_root: Path,
                     job: TrainJob, identity: Mapping[str, Any], device_name: str = "auto",
                     max_units: int = 0, *, frozen_files_verified: bool = False,
                     formal_gate_verified: bool = False,
                     operator_cache_reader: Any | None = None,
                     cache_binding: Mapping[str, Any] | None = None) -> tuple[int, int]:
    if not frozen_files_verified:
        validate_registry(registry, verify_files=True)
    screen, deps = _load_runtime(project_root); torch = deps.torch
    current_environment = execution_environment(torch, device_name, formal=bool(identity["formal"]))
    if identity["formal"]:
        validate_formal_execution_environment(registry, current_environment)
    if "execution_environment" in identity:
        if identity["execution_environment"] != current_environment:
            raise RuntimeError("requested/runtime execution environment differs from bound identity")
        identity = dict(identity)
    else:
        identity = bind_execution_identity(identity, current_environment)
    if identity["formal"] and not formal_gate_verified:
        require_committed_jobs(registry, registry_sha256, project_root, identity,
                               materialize_plan(registry), "formal evaluation gate")
    if identity["formal"] and (operator_cache_reader is None or cache_binding is None):
        raise RuntimeError("formal evaluation requires one deep-validated frozen operator cache reader")
    checkpoint_path, manifest_path = _resolve_artifacts(project_root, job, identity)
    validate_job_commit(checkpoint_path, manifest_path, job, registry_sha256,
                        identity["config_hash"], current_environment)
    device = torch.device(current_environment["resolved_device"])
    bundle = _load_bundle(screen, registry, job.dataset, project_root)
    validation_indices = screen.exact_folds(deps, bundle, job.split_seed)[job.fold - 1][1]
    if identity["smoke_max_samples"]:
        validation_indices = validation_indices[:int(identity["smoke_max_samples"])]
    try:
        checkpoint = screen.load_checkpoint_safely(torch, checkpoint_path)
    except ValueError:
        # Schema-v1 development smoke checkpoints briefly used this key.  They
        # remain isolated by their smoke config hash and can still be audited.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_entry = next(model for model in registry["models"] if model["probe_id"] == job.probe_id)
    model = make_registered_model(deps, model_entry, bundle)
    model_backend = concrete_model_backend(model, model_entry, formal=bool(identity["formal"]))
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
    if not isinstance(state_dict, Mapping):
        raise RuntimeError(f"Stage-B checkpoint lacks a state_dict: {checkpoint_path}")
    model.load_state_dict(state_dict, strict=True); model.to(device).eval()
    adapter = screen.OperatorAdapter(); completed = rows_written = 0
    eval_batch_size = evaluation_batch_size(registry, job.dataset)
    for condition, severity, realization in _expected_eval_conditions(registry):
        if max_units and completed >= max_units:
            break
        unit_id = evaluation_unit_id(job, condition, severity, realization)
        unit_dir = _eval_unit_dir(project_root, identity, unit_id)
        existing = validate_eval_commit(unit_dir, unit_id, identity["config_hash"],
                                        current_environment, cache_binding)
        if existing:
            completed += 1; rows_written += int(existing["prediction_rows"]); continue
        lock_path = unit_dir / "writer.lock"
        with exclusive_writer_lock(
            lock_path, {"scope": "evaluation_unit", "unit_id": unit_id,
                        "config_hash": identity["config_hash"],
                        "resolved_device": current_environment["resolved_device"]}
        ):
            written = _evaluate_unit_locked(
                registry, screen, adapter, bundle, model, torch, device,
                validation_indices, job, identity, checkpoint_path, unit_id, unit_dir,
                condition, severity, realization, eval_batch_size, model_backend,
                operator_cache_reader, cache_binding)
        completed += 1; rows_written += written
        print(f"eval-progress job={job.job_id} condition={condition} "
              f"severity_ms={severity} realization={realization} "
              f"unit={completed}/46 rows={written}", flush=True)
    return completed, rows_written


FROZEN_SEED_METRICS = "frozen_primary_metrics_by_seed.csv"
FROZEN_AGGREGATE_METRICS = "frozen_primary_metrics_3seed.csv"
FROZEN_METRIC_COMMIT = "frozen_primary_metrics_commit.json"


def compute_frozen_metric_tables(
    condition_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute the frozen metrics from five-fold-combined OOF conditions.

    ``condition_rows`` contains one row per dataset/probe/arm/seed/test condition;
    therefore every corrupted operator contributes exactly 3 severities times 3
    realizations.  Folds and samples are never treated as independent repeats.
    """
    grouped: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = {}
    for row in condition_rows:
        key = (str(row["dataset"]), str(row["probe_id"]), str(row["train_arm"]),
               int(row["split_seed"]))
        grouped.setdefault(key, []).append(row)
    seed_rows: list[dict[str, Any]] = []
    operators = ("prebin", "pbj", "cf", "matched", "cf_matched")
    for (dataset, probe_id, arm, seed), rows in sorted(grouped.items()):
        clean_rows = [row for row in rows if row["condition"] == "clean"]
        if len(clean_rows) != 1:
            raise RuntimeError(f"expected one clean OOF condition for {(probe_id, arm, seed)}")
        clean = clean_rows[0]
        clean_accuracy = float(clean["accuracy"])
        if clean_accuracy <= 0.0:
            raise RuntimeError(f"Retention is undefined for zero clean accuracy: {(probe_id, arm, seed)}")
        clean_samples = int(clean["samples"])
        for operator in operators:
            selected = [row for row in rows if row["condition"] == operator]
            identities = {(int(row["severity_ms"]), int(row["realization"])) for row in selected}
            expected = {(severity, realization) for severity in (25, 50, 75)
                        for realization in (0, 1, 2)}
            if len(selected) != 9 or identities != expected:
                raise RuntimeError(
                    f"{operator} must contain exactly 3 severities x 3 realizations: "
                    f"{(probe_id, arm, seed)}"
                )
            if any(int(row["samples"]) != clean_samples for row in selected):
                raise RuntimeError(f"OOF sample count changes across conditions: {(probe_id, arm, seed)}")
            accuracies = [float(row["accuracy"]) for row in selected]
            active_retentions = [float(row["active_after"]) / float(row["active_before"])
                                 for row in selected]
            seed_rows.append({
                "dataset": dataset, "probe_id": probe_id, "train_arm": arm,
                "split_seed": seed, "operator": operator, "oof_samples": clean_samples,
                "condition_count": 9, "clean_accuracy": clean_accuracy,
                "mAcc": statistics.fmean(accuracies),
                "Retention": statistics.fmean(value / clean_accuracy for value in accuracies),
                "MeanDrop": statistics.fmean(clean_accuracy - value for value in accuracies),
                "ActiveRetention": statistics.fmean(active_retentions),
            })

    aggregate_groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in seed_rows:
        key = (str(row["dataset"]), str(row["probe_id"]), str(row["train_arm"]),
               str(row["operator"]))
        aggregate_groups.setdefault(key, []).append(row)
    aggregate_rows: list[dict[str, Any]] = []
    metric_names = ("clean_accuracy", "mAcc", "Retention", "MeanDrop", "ActiveRetention")
    for (dataset, probe_id, arm, operator), rows in sorted(aggregate_groups.items()):
        seeds = sorted(int(row["split_seed"]) for row in rows)
        if seeds != [42, 123, 202]:
            raise RuntimeError(f"aggregate requires exactly frozen seeds 42/123/202: {probe_id}/{arm}")
        aggregate: dict[str, Any] = {
            "dataset": dataset, "probe_id": probe_id, "train_arm": arm,
            "operator": operator, "seed_count": 3, "seeds": "42|123|202",
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in rows]
            aggregate[f"{metric}_mean"] = statistics.fmean(values)
            aggregate[f"{metric}_sample_sd"] = statistics.stdev(values)
        aggregate_rows.append(aggregate)
    return seed_rows, aggregate_rows


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return value


def collect_oof_condition_rows(
    registry: Mapping[str, Any], project_root: Path, jobs: Sequence[TrainJob],
    identity: Mapping[str, Any], *, verify_prediction_oof: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    """Combine five fold units and optionally stream-check exact OOF coverage."""
    grouped_jobs: dict[tuple[str, str, str, int], list[TrainJob]] = {}
    for job in jobs:
        key = (job.dataset, job.probe_id, job.train_arm, job.split_seed)
        grouped_jobs.setdefault(key, []).append(job)
    condition_rows: list[dict[str, Any]] = []
    source_commits: list[str] = []
    for key, fold_jobs in sorted(grouped_jobs.items()):
        dataset, probe_id, arm, seed = key
        if sorted(job.fold for job in fold_jobs) != [1, 2, 3, 4, 5]:
            raise RuntimeError(f"five-fold job set is incomplete: {key}")
        expected_samples = int(registry["data"][dataset]["samples"])
        for condition, severity, realization in _expected_eval_conditions(registry):
            samples = correct = active_before = active_after = 0
            seen_samples: set[int] = set()
            for job in sorted(fold_jobs, key=lambda value: value.fold):
                unit_id = evaluation_unit_id(job, condition, severity, realization)
                unit_dir = _eval_unit_dir(project_root, identity, unit_id)
                marker = validate_eval_commit(unit_dir, unit_id, identity["config_hash"],
                                              identity.get("execution_environment"),
                                              identity.get("operator_cache"))
                if marker is None:
                    raise RuntimeError(f"missing evaluation unit for frozen metrics: {unit_id}")
                source_commits.append(sha256_file(unit_dir / "commit.json"))
                result = _read_json(unit_dir / "result.json")
                for field, expected_value in (("unit_id", unit_id), ("condition", condition),
                                               ("severity_ms", severity), ("realization", realization)):
                    if result.get(field) != expected_value:
                        raise RuntimeError(f"evaluation result field mismatch {field}: {unit_id}")
                samples += int(result["samples"]); correct += int(result["correct"])
                active_before += int(result["active_before"]); active_after += int(result["active_after"])
                if verify_prediction_oof:
                    with (unit_dir / "predictions.csv").open("r", newline="", encoding="utf-8") as handle:
                        rows_read = 0
                        for prediction in csv.DictReader(handle):
                            rows_read += 1
                            if prediction.get("unit_id") != unit_id:
                                raise RuntimeError(f"prediction unit identity mismatch: {unit_id}")
                            index = int(prediction["sample_index"])
                            if index in seen_samples:
                                raise RuntimeError(f"duplicate OOF sample {index}: {key}/{condition}")
                            seen_samples.add(index)
                        if rows_read != int(marker["prediction_rows"]):
                            raise RuntimeError(f"prediction row count mismatch: {unit_id}")
            if samples != expected_samples:
                raise RuntimeError(f"OOF sample cardinality {samples} != {expected_samples}: {key}/{condition}")
            if verify_prediction_oof and seen_samples != set(range(expected_samples)):
                raise RuntimeError(f"OOF samples do not cover the dataset exactly once: {key}/{condition}")
            condition_rows.append({
                "dataset": dataset, "probe_id": probe_id, "train_arm": arm,
                "split_seed": seed, "condition": condition, "severity_ms": severity,
                "realization": realization, "samples": samples, "correct": correct,
                "accuracy": correct / samples, "active_before": active_before,
                "active_after": active_after,
            })
    source_digest = sha256_bytes(canonical_json(sorted(source_commits)).encode("utf-8"))
    return condition_rows, source_digest


def commit_frozen_metric_tables(
    output_dir: Path, registry_sha256: str, config_hash: str,
    source_evaluation_digest: str, seed_rows: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Transactionally publish both frozen tables, with commit.json written last."""
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = output_dir / FROZEN_SEED_METRICS
    aggregate_path = output_dir / FROZEN_AGGREGATE_METRICS
    write_csv_atomic(seed_path, seed_rows); write_csv_atomic(aggregate_path, aggregate_rows)
    marker = {
        "runner_schema_version": RUNNER_SCHEMA_VERSION, "registry_sha256": registry_sha256,
        "config_hash": config_hash, "source_evaluation_commit_digest": source_evaluation_digest,
        "seed_rows": len(seed_rows), "aggregate_rows": len(aggregate_rows),
        "statistical_unit": "seed_after_five_fold_oof_concatenation",
        "standard_deviation": "sample_sd_ddof_1_across_three_seeds",
        "artifacts": {FROZEN_SEED_METRICS: sha256_file(seed_path),
                      FROZEN_AGGREGATE_METRICS: sha256_file(aggregate_path)},
    }
    write_json_atomic(output_dir / FROZEN_METRIC_COMMIT, marker)
    return marker


def validate_frozen_metric_commit(output_dir: Path, registry_sha256: str,
                                  config_hash: str) -> Mapping[str, Any] | None:
    path = output_dir / FROZEN_METRIC_COMMIT
    if not path.exists():
        return None
    marker = _read_json(path)
    if marker.get("registry_sha256") != registry_sha256 or marker.get("config_hash") != config_hash:
        raise RuntimeError("frozen metric commit identity mismatch")
    for name, digest in marker.get("artifacts", {}).items():
        artifact = output_dir / name
        if not artifact.is_file() or sha256_file(artifact) != digest:
            raise RuntimeError(f"frozen metric artifact tamper/hash mismatch: {artifact}")
    return marker


def finalize(registry: Mapping[str, Any], registry_sha256: str, project_root: Path,
             identity: Mapping[str, Any], require_evaluation: bool = True,
             cache_binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if identity["formal"] and not require_evaluation:
        raise RuntimeError("formal finalize cannot bypass evaluation or write complete without it")
    if identity["formal"] and require_evaluation and cache_binding is None:
        raise RuntimeError("formal finalize requires a validated operator-cache binding")
    if cache_binding is not None:
        identity = {**dict(identity), "operator_cache": dict(cache_binding)}
    jobs = materialize_plan(registry); committed = eval_units = prediction_rows = 0
    for job in jobs:
        checkpoint, manifest = _resolve_artifacts(project_root, job, identity)
        if validate_job_commit(checkpoint, manifest, job, registry_sha256,
                               identity["config_hash"], identity.get("execution_environment")):
            committed += 1
        if require_evaluation:
            for condition, severity, realization in _expected_eval_conditions(registry):
                unit_id = evaluation_unit_id(job, condition, severity, realization)
                marker = validate_eval_commit(_eval_unit_dir(project_root, identity, unit_id), unit_id,
                                              identity["config_hash"],
                                              identity.get("execution_environment"),
                                              identity.get("operator_cache"))
                if marker:
                    eval_units += 1; prediction_rows += int(marker["prediction_rows"])
    expected_jobs = 360 if identity["formal"] else len(jobs)
    integrity = {"formal": identity["formal"], "config_hash": identity["config_hash"],
                 "registry_sha256": registry_sha256, "checkpoints": committed,
                 "expected_checkpoints": expected_jobs, "evaluation_units": eval_units,
                 "expected_evaluation_units": 16560 if require_evaluation else 0,
                 "prediction_rows": prediction_rows,
                 "expected_prediction_rows": 21693600 if require_evaluation and identity["formal"] else None}
    integrity["complete"] = (committed == expected_jobs and
                             (not require_evaluation or (eval_units == 16560 and
                              (not identity["formal"] or prediction_rows == 21693600))))
    output = project_root / "ICASSP20260813" / "experiments" / (
        "stage_b_v1/completion_integrity.json" if identity["formal"] else
        f"stage_b_smoke/{identity['config_hash']}/completion_integrity.json")
    if not integrity["complete"]:
        write_json_atomic(output, integrity)
        raise RuntimeError(f"Stage B integrity is incomplete: {integrity}")
    if require_evaluation:
        output_dir = output.parent
        condition_rows, source_digest = collect_oof_condition_rows(
            registry, project_root, jobs, identity, verify_prediction_oof=True)
        seed_rows, aggregate_rows = compute_frozen_metric_tables(condition_rows)
        marker = commit_frozen_metric_tables(output_dir, registry_sha256,
                                             identity["config_hash"], source_digest,
                                             seed_rows, aggregate_rows)
        if len(seed_rows) != 360 or len(aggregate_rows) != 120:
            raise RuntimeError(f"frozen metric table scale mismatch: {len(seed_rows)}/{len(aggregate_rows)}")
        integrity["frozen_metrics"] = {
            "seed_rows": len(seed_rows), "aggregate_rows": len(aggregate_rows),
            "commit_sha256": sha256_file(output_dir / FROZEN_METRIC_COMMIT),
            "source_evaluation_commit_digest": marker["source_evaluation_commit_digest"],
        }
    write_json_atomic(output, integrity)
    return integrity


def _select_jobs(jobs: Sequence[TrainJob], args: argparse.Namespace) -> list[TrainJob]:
    selected = [job for job in jobs if (not args.phase or job.phase == args.phase)
                and (not args.job_id or job.job_id == args.job_id)]
    if args.limit_jobs:
        selected = selected[:args.limit_jobs]
    if args.job_id and not selected:
        raise ValueError(f"unknown/unselected job id: {args.job_id}")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate"); validate.add_argument("--verify-files", action="store_true")
    materialize = sub.add_parser("materialize-plan"); materialize.add_argument("--output", type=Path, required=True)
    for name in ("train", "eval"):
        command = sub.add_parser(name)
        command.add_argument("--phase", choices=("p0_core", "p1_dense"), default="")
        command.add_argument("--job-id", default=""); command.add_argument("--limit-jobs", type=int, default=0)
        command.add_argument("--device", default="auto")
        command.add_argument("--smoke-epochs", type=int, default=0)
        command.add_argument("--smoke-max-samples", type=int, default=0)
        if name == "eval": command.add_argument("--max-units-per-job", type=int, default=0)
    final = sub.add_parser("finalize"); final.add_argument("--without-evaluation", action="store_true")
    final.add_argument("--device", default="auto")
    final.add_argument("--smoke-epochs", type=int, default=0); final.add_argument("--smoke-max-samples", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry_path = args.registry.resolve(); registry, registry_sha = load_registry(registry_path)
    force_verify = args.command in ("train", "eval")
    summary = validate_registry(
        registry, registry_path,
        verify_files=bool(getattr(args, "verify_files", False) or force_verify))
    project_root = Path(str(registry["project_root"])).resolve(); jobs = materialize_plan(registry)
    if args.command == "validate":
        print(json.dumps({**summary, "registry_sha256": registry_sha}, indent=2)); return 0
    if args.command == "materialize-plan":
        payload = {"registry_sha256": registry_sha, "formal": True, "jobs": [job.as_dict() for job in jobs],
                   "scale": {"checkpoints": 360, "p0_core": 240, "p1_dense": 120,
                             "evaluation_units": 16560, "prediction_rows": 21693600}}
        write_json_atomic(args.output.resolve(), payload); print(f"materialized 360 jobs: {args.output.resolve()}"); return 0
    identity = run_identity(registry_sha, smoke_epochs=getattr(args, "smoke_epochs", 0),
                            smoke_max_samples=getattr(args, "smoke_max_samples", 0))
    if args.command in ("train", "eval", "finalize"):
        _, runtime_deps = _load_runtime(project_root)
        environment = execution_environment(runtime_deps.torch, getattr(args, "device", "auto"),
                                            formal=bool(identity["formal"]))
        if identity["formal"]:
            validate_formal_execution_environment(registry, environment)
        identity = bind_execution_identity(identity, environment)
    if args.command == "train":
        selected = _select_jobs(jobs, args)
        p0_jobs = [job for job in jobs if job.phase == "p0_core"]
        p1_gate_checked = False
        for number, job in enumerate(selected, 1):
            if identity["formal"] and job.phase == "p1_dense" and not p1_gate_checked:
                require_committed_jobs(registry, registry_sha, project_root, identity,
                                       p0_jobs, "formal P1 start gate")
                p1_gate_checked = True
            status = train_one_job(
                registry, registry_sha, project_root, job, identity, args.device,
                frozen_files_verified=True,
                phase_gate_verified=(job.phase == "p0_core" or p1_gate_checked))
            print(f"[{number}/{len(selected)}] {status}: {job.job_id}", flush=True)
        return 0
    if args.command == "eval":
        selected = _select_jobs(jobs, args)
        cache_reader = None; cache_binding = None
        if identity["formal"]:
            require_committed_jobs(registry, registry_sha, project_root, identity,
                                   jobs, "formal evaluation gate")
            # Exactly once per CLI invocation; reused across every selected job.
            cache_reader, cache_binding = load_formal_operator_cache(
                registry, registry_sha, registry_path, project_root)
        for number, job in enumerate(selected, 1):
            units, rows = evaluate_one_job(registry, registry_sha, project_root, job, identity,
                                           args.device, args.max_units_per_job,
                                           frozen_files_verified=True,
                                           formal_gate_verified=identity["formal"],
                                           operator_cache_reader=cache_reader,
                                           cache_binding=cache_binding)
            print(f"[{number}/{len(selected)}] {job.job_id}: units={units} rows={rows}", flush=True)
        return 0
    if identity["formal"] and args.without_evaluation:
        raise RuntimeError("--without-evaluation is forbidden for formal finalize")
    final_cache_binding = None
    if identity["formal"]:
        _, final_cache_binding = load_formal_operator_cache(
            registry, registry_sha, registry_path, project_root)
    integrity = finalize(registry, registry_sha, project_root, identity,
                         require_evaluation=not args.without_evaluation,
                         cache_binding=final_cache_binding)
    print(json.dumps(integrity, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
