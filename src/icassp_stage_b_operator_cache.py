"""Build and validate the frozen Stage-B packed operator cache.

The cache is an evaluation acceleration artifact, never a scientific input to
training.  Formal builds are fail-closed: the frozen registry must name and
hash this builder, all frozen inputs are verified, and a complete commit marker
is published last.  Evaluation code should bind its result commits to
``cache_identity.content_sha256`` and must not fall back to live corruption.

This module deliberately never reads Stage-A outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import icassp_stage_b_runner as runner


CACHE_SCHEMA_NAME = "icassp_stage_b_operator_cache"
CACHE_SCHEMA_VERSION = 1
BUILDER_VERSION = 1
BUILDER_SOURCE_KEY = "stage_b_operator_cache_builder"
SPOT_RULE_ID = "operator_cache_spot_v1"
SPOT_COUNT = 64
CONDITIONS = ("prebin", "pbj", "cf", "matched", "cf_matched")
SEVERITIES_MS = (25, 50, 75)
REALIZATIONS = (0, 1, 2)
EXPECTED_DATASETS = ("braille", "stemnist")


class CacheError(RuntimeError):
    """A frozen cache identity, integrity, or protocol violation."""


@dataclass(frozen=True)
class Condition:
    index: int
    condition: str
    severity_ms: int
    realization: int

    @property
    def filename(self) -> str:
        return (
            f"c{self.index:02d}_{self.condition}_ms{self.severity_ms:03d}"
            f"_r{self.realization}.npy"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "condition": self.condition,
            "severity_ms": self.severity_ms,
            "realization": self.realization,
        }


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_int64(values: Sequence[int]) -> str:
    return sha256_bytes(np.asarray(values, dtype="<i8").tobytes(order="C"))


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def condition_table() -> list[Condition]:
    result: list[Condition] = []
    for severity_index, severity in enumerate(SEVERITIES_MS):
        for realization in REALIZATIONS:
            for condition_index, name in enumerate(CONDITIONS):
                index = ((severity_index * len(REALIZATIONS) + realization)
                         * len(CONDITIONS) + condition_index)
                result.append(Condition(index, name, severity, realization))
    if [item.index for item in result] != list(range(45)):
        raise AssertionError("condition indexing is not contiguous")
    return result


def condition_lookup() -> dict[tuple[str, int, int], Condition]:
    return {
        (item.condition, item.severity_ms, item.realization): item
        for item in condition_table()
    }


def cache_id(registry: Mapping[str, Any], registry_sha256: str) -> str:
    identity = {
        "schema": [CACHE_SCHEMA_NAME, CACHE_SCHEMA_VERSION],
        "protocol_id": registry["protocol_id"],
        "registry_sha256": registry_sha256,
        "conditions": [item.as_dict() for item in condition_table()],
        "storage": {"dtype": "uint8", "bitorder": "little", "layout": "NTP"},
    }
    return sha256_bytes(canonical_json(identity).encode("utf-8"))[:24]


def default_cache_parent(project_root: Path) -> Path:
    return (
        project_root / "ICASSP20260813" / "experiments" / "stage_b_v1"
        / "operator_cache"
    )


def _builder_source_entry(registry: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = registry.get("source_files", {}).get(BUILDER_SOURCE_KEY)
    return value if isinstance(value, Mapping) else None


def verify_builder_registration(
    registry: Mapping[str, Any], project_root: Path, *, allow_unregistered: bool = False
) -> dict[str, str]:
    actual_path = Path(__file__).resolve()
    actual_sha = sha256_file(actual_path)
    entry = _builder_source_entry(registry)
    if entry is None:
        if not allow_unregistered:
            raise CacheError(
                f"formal registry must freeze source_files.{BUILDER_SOURCE_KEY}; "
                "update and re-freeze the registry before formal checkpoints"
            )
        return {"path": str(actual_path), "sha256": actual_sha, "registration": "development"}
    expected_path = (project_root / str(entry.get("path", ""))).resolve()
    if expected_path != actual_path:
        raise CacheError(f"registered builder path {expected_path} != running builder {actual_path}")
    if str(entry.get("sha256", "")).lower() != actual_sha:
        raise CacheError(
            f"builder SHA-256 mismatch: expected {entry.get('sha256')}, got {actual_sha}"
        )
    return {"path": str(entry["path"]), "sha256": actual_sha, "registration": "frozen"}


def sample_identity_records(bundle: Any) -> list[dict[str, Any]]:
    groups = getattr(bundle, "groups", None)
    records: list[dict[str, Any]] = []
    for index, sample_id in enumerate(bundle.sample_ids):
        group = None if groups is None else str(groups[index])
        records.append({
            "index": index,
            "sample_id": str(sample_id),
            "label": int(bundle.labels[index]),
            "group": group,
        })
    return records


def sample_identity_sha256(bundle: Any) -> str:
    return sha256_bytes(canonical_json(sample_identity_records(bundle)).encode("utf-8"))


def select_spot_samples(
    protocol_id: str, dataset: str, sample_ids: Sequence[str], count: int = SPOT_COUNT
) -> list[dict[str, Any]]:
    total = len(sample_ids)
    if total == 0:
        return []
    target = min(int(count), total)
    anchors = [0] if total == 1 else [0, total - 1]
    ranked: list[tuple[str, int]] = []
    for index, sample_id in enumerate(sample_ids):
        payload = [protocol_id, SPOT_RULE_ID, dataset, index, str(sample_id)]
        ranked.append((sha256_bytes(canonical_json(payload).encode("utf-8")), index))
    ranked.sort(key=lambda item: (item[0], item[1]))
    selected = set(anchors)
    for _, index in ranked:
        if len(selected) >= target:
            break
        selected.add(index)
    digest_by_index = {index: digest for digest, index in ranked}
    return [
        {"index": index, "sample_id": str(sample_ids[index]),
         "rank_digest": digest_by_index[index]}
        for index in sorted(selected)
    ]


def spot_selection_payload(registry: Mapping[str, Any], bundle: Any) -> dict[str, Any]:
    samples = select_spot_samples(
        str(registry["protocol_id"]), str(bundle.name), bundle.sample_ids
    )
    return {
        "rule_id": SPOT_RULE_ID,
        "requested_count": SPOT_COUNT,
        "count": len(samples),
        "samples": samples,
        "selection_sha256": sha256_bytes(canonical_json(samples).encode("utf-8")),
    }


def _logical_clean_sha256(screen: Any, bundle: Any) -> str:
    digest = hashlib.sha256()
    for index in range(len(bundle.labels)):
        clean = np.asarray(screen.unpack_clean_sample(bundle, index), dtype=np.uint8)
        digest.update(np.ascontiguousarray(clean).tobytes(order="C"))
    return digest.hexdigest()


def _dataset_manifest(registry: Mapping[str, Any], screen: Any, bundle: Any) -> dict[str, Any]:
    frozen = registry["data"][bundle.name]
    packed_channels = (int(bundle.channels) + 7) // 8
    input_hashes: dict[str, Any]
    if bundle.name == "braille":
        input_hashes = {"source_sha256": frozen["source_sha256"]}
    else:
        input_hashes = {
            "cache_index_sha256": frozen["cache_index"]["sha256"],
            "cache_files": {item["role"]: item["sha256"] for item in frozen["cache_files"]},
        }
    return {
        "dataset_id": frozen["dataset_id"],
        "input_hashes": input_hashes,
        "samples": len(bundle.labels),
        "time_steps": int(bundle.time_steps),
        "channels": int(bundle.channels),
        "packed_channels": packed_channels,
        "sample_identity_sha256": sample_identity_sha256(bundle),
        "clean_representation_sha256": _logical_clean_sha256(screen, bundle),
    }


def _source_manifest(registry: Mapping[str, Any]) -> dict[str, Any]:
    sources = registry["source_files"]
    result: dict[str, Any] = {}
    for output_name, source_name in (
        ("runner", "stage_b_runner"),
        ("runtime", "stage_b_runtime_module"),
    ):
        entry = sources.get(source_name)
        if not isinstance(entry, Mapping):
            raise CacheError(f"registry source_files.{source_name} is required")
        result[output_name] = {"path": entry["path"], "sha256": entry["sha256"]}
    result["operator"] = {
        "path": registry["operators"]["module_path"],
        "sha256": registry["operators"]["module_sha256"],
    }
    return result


def _operator_seed(registry: Mapping[str, Any], bundle: Any, sample_index: int) -> int:
    return runner.stable_seed(
        registry["protocol_id"], "operator_root_v1", bundle.name,
        bundle.sample_ids[sample_index],
    )


def _pack(array: np.ndarray, channels: int) -> np.ndarray:
    source = np.asarray(array, dtype=np.uint8)
    if source.ndim != 2 or source.shape[1] != channels:
        raise CacheError(f"logical operator array has invalid shape {source.shape}")
    if np.any((source != 0) & (source != 1)):
        raise CacheError("logical operator array is not binary")
    return np.packbits(source, axis=-1, bitorder="little")


def _sync_memmap(value: np.memmap) -> None:
    value.flush()
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.flush()


def _fsync_file(path: Path) -> None:
    # Windows maps ``os.fsync`` to ``_commit``, which rejects a read-only
    # descriptor with EBADF.  Open read/write without changing the contents so
    # the durability barrier works on both Windows and POSIX.
    with path.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _condition_relative_path(dataset: str, item: Condition) -> Path:
    return Path(dataset) / "conditions" / item.filename


def _build_dataset_files(
    registry: Mapping[str, Any], screen: Any, bundle: Any, adapter: Any,
    staging: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lookup = condition_lookup()
    samples = len(bundle.labels)
    packed_channels = (int(bundle.channels) + 7) // 8
    shape = (samples, int(bundle.time_steps), packed_channels)
    inventory: list[dict[str, Any]] = []
    active_counts: dict[int, np.ndarray] = {
        item.index: np.zeros(samples, dtype=np.int64) for item in condition_table()
    }
    raw_before = raw_after = 0
    max_prebin_displacement = 0.0

    for severity in SEVERITIES_MS:
        for realization in REALIZATIONS:
            group = [lookup[(name, severity, realization)] for name in CONDITIONS]
            maps: dict[str, np.memmap] = {}
            for item in group:
                path = staging / _condition_relative_path(bundle.name, item)
                path.parent.mkdir(parents=True, exist_ok=True)
                maps[item.condition] = np.lib.format.open_memmap(
                    path, mode="w+", dtype=np.uint8, shape=shape,
                    fortran_order=False, version=(2, 0),
                )
            for index in range(samples):
                clean = screen.unpack_clean_sample(bundle, index)
                seed = _operator_seed(registry, bundle, index)
                prebin, pre_audit = adapter.prebin_binary(
                    bundle.raw_getter(index), severity, bundle.duration_seconds, seed,
                    n_steps=bundle.time_steps, n_channels=bundle.channels,
                    realization=realization,
                )
                arrays, _ = adapter.postbin_bundle(
                    clean, severity // 25, seed, realization
                )
                outputs = {"prebin": prebin, **arrays}
                raw_before += int(pre_audit["raw_events_before"])
                raw_after += int(pre_audit["raw_events_after"])
                max_prebin_displacement = max(
                    max_prebin_displacement,
                    float(pre_audit["max_abs_displacement_seconds"]),
                )
                for item in group:
                    output = np.asarray(outputs[item.condition], dtype=np.uint8)
                    maps[item.condition][index] = _pack(output, bundle.channels)
                    active_counts[item.index][index] = int(np.count_nonzero(output))
            for item in group:
                mapped = maps.pop(item.condition)
                _sync_memmap(mapped)
                del mapped
                path = staging / _condition_relative_path(bundle.name, item)
                _fsync_file(path)
                file_bytes = path.stat().st_size
                payload_bytes = int(np.prod(shape, dtype=np.int64))
                inventory.append({
                    "dataset": bundle.name,
                    "condition_index": item.index,
                    "condition": item.condition,
                    "severity_ms": item.severity_ms,
                    "realization": item.realization,
                    "relative_path": _condition_relative_path(bundle.name, item).as_posix(),
                    "npy_format": "numpy_v2",
                    "dtype": "uint8",
                    "shape": list(shape),
                    "c_order": True,
                    "bitorder": "little",
                    "logical_channels": int(bundle.channels),
                    "payload_bytes": payload_bytes,
                    "file_bytes": file_bytes,
                    "sha256": sha256_file(path),
                    "active_after_total": int(active_counts[item.index].sum()),
                    "active_after_per_sample_sha256": sha256_int64(active_counts[item.index]),
                })
    inventory.sort(key=lambda row: int(row["condition_index"]))
    audit = {
        "samples_audited": samples,
        "severity_realization_groups": 9,
        "condition_files": 45,
        "prebin_raw_events_before_accumulated": raw_before,
        "prebin_raw_events_after_accumulated": raw_after,
        "prebin_raw_event_count_conserved": raw_before == raw_after,
        "prebin_max_abs_displacement_seconds": max_prebin_displacement,
        "postbin_bundle_calls": samples * 9,
        "prebin_calls": samples * 9,
        "adapter_invariants_passed": True,
    }
    if raw_before != raw_after:
        raise CacheError(f"{bundle.name}: pre-bin raw event count changed")
    return inventory, audit


def _cache_content_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": manifest["schema"],
        "protocol_id": manifest["protocol_id"],
        "registry": manifest["registry"],
        "builder": manifest["builder"],
        "frozen_sources": manifest["frozen_sources"],
        "randomness": manifest["randomness"],
        "datasets": manifest["datasets"],
        "conditions": manifest["conditions"],
        "inventory": manifest["inventory"],
        "full_audit": manifest["full_audit"],
        "spot_audit": manifest["spot_audit"],
    }


def compute_content_sha256(manifest: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(_cache_content_payload(manifest)).encode("utf-8"))


class ExclusiveBuildLock:
    def __init__(self, path: Path, registry_sha256: str):
        self.path = path
        self.registry_sha256 = registry_sha256
        self.acquired = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json({
            "pid": os.getpid(), "host": socket.gethostname(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "registry_sha256": self.registry_sha256,
        }).encode("utf-8")
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise CacheError(f"operator-cache build lock already exists: {self.path}") from exc
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.acquired:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def _load_manifest(cache_root: Path) -> dict[str, Any]:
    path = cache_root / "cache_manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CacheError(f"invalid or missing cache manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CacheError("cache manifest root must be an object")
    return value


def _expected_inventory_keys() -> set[tuple[str, int]]:
    return {(dataset, item.index) for dataset in EXPECTED_DATASETS for item in condition_table()}


def _validate_manifest_contract(
    manifest: Mapping[str, Any], registry: Mapping[str, Any], registry_sha256: str,
    *, allow_development: bool,
) -> None:
    allowed_top = {
        "schema", "protocol_id", "registry", "cache_identity", "builder",
        "frozen_sources", "randomness", "datasets", "conditions", "inventory",
        "full_audit", "spot_audit", "created_utc",
    }
    if set(manifest) != allowed_top:
        raise CacheError(
            f"operator-cache manifest fields differ from schema: "
            f"missing={sorted(allowed_top - set(manifest))}, "
            f"unknown={sorted(set(manifest) - allowed_top)}"
        )
    if manifest.get("schema") != {"name": CACHE_SCHEMA_NAME, "version": CACHE_SCHEMA_VERSION}:
        raise CacheError("operator-cache schema mismatch")
    if manifest.get("protocol_id") != registry["protocol_id"]:
        raise CacheError("operator-cache protocol mismatch")
    if manifest.get("registry", {}).get("sha256") != registry_sha256:
        raise CacheError("operator-cache registry SHA mismatch")
    if manifest.get("registry", {}).get("registry_id") != registry["registry_id"]:
        raise CacheError("operator-cache registry ID mismatch")
    if manifest.get("cache_identity", {}).get("cache_id") != cache_id(
        registry, registry_sha256
    ):
        raise CacheError("operator-cache ID mismatch")
    if not allow_development and manifest.get("cache_identity", {}).get("formal") is not True:
        raise CacheError("development cache cannot satisfy the formal cache gate")
    registered_builder = _builder_source_entry(registry)
    if registered_builder is not None:
        if manifest.get("builder", {}).get("path") != registered_builder.get("path") or str(
            manifest.get("builder", {}).get("sha256", "")
        ).lower() != str(registered_builder.get("sha256", "")).lower():
            raise CacheError("operator-cache builder identity differs from registry")
    if manifest.get("frozen_sources") != _source_manifest(registry):
        raise CacheError("operator-cache frozen source identities differ from registry")
    expected_randomness = {
        "root_rule_id": "operator_root_v1",
        "root_key_fields": ["protocol_id", "operator_root_v1", "dataset", "sample_id"],
        "condition_order": list(CONDITIONS),
        "severity_order_ms": list(SEVERITIES_MS),
        "realization_order": list(REALIZATIONS),
    }
    if manifest.get("randomness") != expected_randomness:
        raise CacheError("operator-cache randomness contract mismatch")
    if manifest.get("conditions") != [item.as_dict() for item in condition_table()]:
        raise CacheError("operator-cache condition table mismatch")
    if set(manifest.get("datasets", {})) != set(EXPECTED_DATASETS):
        raise CacheError("operator-cache datasets must be exactly braille and stemnist")
    if set(manifest.get("full_audit", {})) != set(EXPECTED_DATASETS):
        raise CacheError("operator-cache full audit must cover both datasets")
    if set(manifest.get("spot_audit", {})) != set(EXPECTED_DATASETS):
        raise CacheError("operator-cache spot audit must cover both datasets")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, list) or len(inventory) != 90:
        raise CacheError("formal operator cache must inventory exactly 90 files")
    keys = {(str(row.get("dataset")), int(row.get("condition_index", -1))) for row in inventory}
    if keys != _expected_inventory_keys():
        raise CacheError("operator-cache inventory is incomplete or duplicated")
    if manifest.get("cache_identity", {}).get("content_sha256") != compute_content_sha256(manifest):
        raise CacheError("operator-cache scientific content hash mismatch")


def _validate_inventory_file(cache_root: Path, row: Mapping[str, Any], *, deep: bool) -> None:
    allowed_fields = {
        "dataset", "condition_index", "condition", "severity_ms", "realization",
        "relative_path", "npy_format", "dtype", "shape", "c_order", "bitorder",
        "logical_channels", "payload_bytes", "file_bytes", "sha256",
        "active_after_total", "active_after_per_sample_sha256",
    }
    if set(row) != allowed_fields:
        raise CacheError("cache inventory entry has missing or unknown fields")
    relative = Path(str(row["relative_path"]))
    path = (cache_root / relative).resolve()
    try:
        path.relative_to(cache_root.resolve())
    except ValueError as exc:
        raise CacheError(f"inventory path escapes cache root: {relative}") from exc
    expected = Condition(
        int(row["condition_index"]), str(row["condition"]),
        int(row["severity_ms"]), int(row["realization"]),
    )
    canonical_relative = _condition_relative_path(str(row["dataset"]), expected)
    if relative.as_posix() != canonical_relative.as_posix():
        raise CacheError(f"non-canonical condition filename: {relative}")
    if not path.is_file() or path.stat().st_size != int(row["file_bytes"]):
        raise CacheError(f"missing or size-mismatched cache file: {path}")
    if sha256_file(path) != row["sha256"]:
        raise CacheError(f"cache file SHA-256 mismatch: {path}")
    with path.open("rb") as handle:
        if np.lib.format.read_magic(handle) != (2, 0):
            raise CacheError(f"cache .npy format must be version 2.0: {path}")
    if row.get("npy_format") != "numpy_v2" or row.get("dtype") != "uint8":
        raise CacheError(f"cache .npy metadata mismatch: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.uint8 or list(array.shape) != list(row["shape"]):
        raise CacheError(f"cache array dtype/shape mismatch: {path}")
    if bool(row.get("c_order")) is not True or row.get("bitorder") != "little":
        raise CacheError(f"cache storage contract mismatch: {path}")
    if int(array.nbytes) != int(row["payload_bytes"]):
        raise CacheError(f"cache payload byte count mismatch: {path}")
    if deep:
        channels = int(row["logical_channels"])
        counts = np.empty(array.shape[0], dtype=np.int64)
        for start in range(0, array.shape[0], 128):
            packed = np.asarray(array[start:start + 128])
            logical = np.unpackbits(packed, axis=-1, count=channels, bitorder="little")
            if not np.array_equal(
                np.packbits(logical, axis=-1, bitorder="little"), packed
            ):
                raise CacheError(f"pack/unpack roundtrip failed: {path}")
            counts[start:start + len(logical)] = np.count_nonzero(logical, axis=(1, 2))
        if int(counts.sum()) != int(row["active_after_total"]):
            raise CacheError(f"cache active total mismatch: {path}")
        if sha256_int64(counts) != row["active_after_per_sample_sha256"]:
            raise CacheError(f"cache per-sample active digest mismatch: {path}")


def _validate_spots(
    cache_root: Path, manifest: Mapping[str, Any], registry: Mapping[str, Any],
    project_root: Path,
) -> None:
    screen, _ = runner._load_runtime(project_root)
    adapter = screen.OperatorAdapter()
    inventory = {
        (str(row["dataset"]), int(row["condition_index"])): row
        for row in manifest["inventory"]
    }
    for dataset in EXPECTED_DATASETS:
        bundle = runner._load_bundle(screen, registry, dataset, project_root)
        expected_dataset = _dataset_manifest(registry, screen, bundle)
        if manifest["datasets"].get(dataset) != expected_dataset:
            raise CacheError(f"{dataset}: cache dataset/sample identity mismatch")
        expected_spot = spot_selection_payload(registry, bundle)
        actual_spot = manifest["spot_audit"][dataset]
        if actual_spot != expected_spot:
            raise CacheError(f"{dataset}: deterministic spot selection mismatch")
        arrays = {
            item.index: np.load(
                cache_root / inventory[(dataset, item.index)]["relative_path"],
                mmap_mode="r", allow_pickle=False,
            )
            for item in condition_table()
        }
        lookup = condition_lookup()
        for selected in expected_spot["samples"]:
            index = int(selected["index"])
            clean = screen.unpack_clean_sample(bundle, index)
            seed = _operator_seed(registry, bundle, index)
            for severity in SEVERITIES_MS:
                for realization in REALIZATIONS:
                    prebin, _ = adapter.prebin_binary(
                        bundle.raw_getter(index), severity, bundle.duration_seconds, seed,
                        n_steps=bundle.time_steps, n_channels=bundle.channels,
                        realization=realization,
                    )
                    post, _ = adapter.postbin_bundle(
                        clean, severity // 25, seed, realization
                    )
                    outputs = {"prebin": prebin, **post}
                    for name in CONDITIONS:
                        item = lookup[(name, severity, realization)]
                        cached = np.asarray(arrays[item.index][index])
                        if not np.array_equal(cached, _pack(outputs[name], bundle.channels)):
                            raise CacheError(
                                f"spot regeneration mismatch: {dataset}/{index}/{item.filename}"
                            )


def validate_cache(
    cache_root: Path, registry_path: Path, *, deep: bool = False,
    spot: bool = True, require_commit: bool = True,
    allow_unregistered_builder: bool = False,
) -> dict[str, Any]:
    registry, registry_sha256 = runner.load_registry(registry_path)
    summary = runner.validate_registry(registry, registry_path, verify_files=True)
    project_root = Path(str(registry["project_root"])).resolve()
    verify_builder_registration(
        registry, project_root, allow_unregistered=allow_unregistered_builder
    )
    cache_root = cache_root.resolve()
    manifest = _load_manifest(cache_root)
    _validate_manifest_contract(
        manifest, registry, registry_sha256,
        allow_development=allow_unregistered_builder,
    )
    if require_commit:
        commit_path = cache_root / "commit.json"
        try:
            commit = json.loads(commit_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CacheError(f"invalid or missing cache commit {commit_path}: {exc}") from exc
        required_commit_fields = {
            "schema", "protocol_id", "registry_sha256", "cache_id",
            "cache_content_sha256", "manifest_sha256", "inventory_count",
            "complete", "committed_utc",
        }
        if set(commit) != required_commit_fields:
            raise CacheError("operator-cache commit has missing or unknown fields")
        if commit.get("schema") != {
            "name": "icassp_stage_b_operator_cache_commit", "version": 1
        } or commit.get("protocol_id") != registry["protocol_id"]:
            raise CacheError("operator-cache commit schema/protocol mismatch")
        if commit.get("complete") is not True or int(commit.get("inventory_count", -1)) != 90:
            raise CacheError("operator-cache commit is incomplete")
        if commit.get("registry_sha256") != registry_sha256:
            raise CacheError("operator-cache commit registry mismatch")
        if commit.get("cache_content_sha256") != manifest["cache_identity"]["content_sha256"]:
            raise CacheError("operator-cache commit content mismatch")
        if commit.get("cache_id") != manifest["cache_identity"]["cache_id"]:
            raise CacheError("operator-cache commit ID mismatch")
        if commit.get("manifest_sha256") != sha256_file(cache_root / "cache_manifest.json"):
            raise CacheError("operator-cache manifest hash mismatch")
    for row in manifest["inventory"]:
        _validate_inventory_file(cache_root, row, deep=deep)
    npy_files = {path.resolve() for path in cache_root.rglob("*.npy") if path.is_file()}
    declared = {
        (cache_root / str(row["relative_path"])).resolve() for row in manifest["inventory"]
    }
    if npy_files != declared:
        raise CacheError("cache directory has missing or undeclared .npy files")
    allowed_files = declared | {(cache_root / "cache_manifest.json").resolve()}
    if require_commit:
        allowed_files.add((cache_root / "commit.json").resolve())
    actual_files = {path.resolve() for path in cache_root.rglob("*") if path.is_file()}
    if actual_files != allowed_files:
        raise CacheError("cache directory has missing or undeclared files")
    if spot:
        _validate_spots(cache_root, manifest, registry, project_root)
    return {
        **summary,
        "cache_root": str(cache_root),
        "cache_id": manifest["cache_identity"]["cache_id"],
        "cache_content_sha256": manifest["cache_identity"]["content_sha256"],
        "inventory_count": len(manifest["inventory"]),
        "deep": bool(deep),
        "spot": bool(spot),
        "complete": True,
    }


def build_cache(
    registry_path: Path, cache_parent: Path | None = None, *,
    allow_unregistered_builder: bool = False,
) -> Path:
    registry_path = registry_path.resolve()
    registry, registry_sha256 = runner.load_registry(registry_path)
    runner.validate_registry(registry, registry_path, verify_files=True)
    project_root = Path(str(registry["project_root"])).resolve()
    builder = verify_builder_registration(
        registry, project_root, allow_unregistered=allow_unregistered_builder
    )
    if builder["registration"] != "frozen" and cache_parent is None:
        raise CacheError(
            "development cache builds require an explicit --cache-parent and may not "
            "write beneath the formal default root"
        )
    parent = (cache_parent or default_cache_parent(project_root)).resolve()
    identity = cache_id(registry, registry_sha256)
    final_root = parent / identity
    if final_root.exists():
        validate_cache(
            final_root, registry_path, deep=False, spot=True, require_commit=True,
            allow_unregistered_builder=allow_unregistered_builder,
        )
        return final_root
    lock_path = parent / f"{identity}.build.lock"
    with ExclusiveBuildLock(lock_path, registry_sha256):
        if final_root.exists():
            validate_cache(
                final_root, registry_path, deep=False, spot=True, require_commit=True,
                allow_unregistered_builder=allow_unregistered_builder,
            )
            return final_root
        staging = parent / f".{identity}.building.{uuid.uuid4().hex}"
        staging.mkdir(parents=True, exist_ok=False)
        screen, _ = runner._load_runtime(project_root)
        adapter = screen.OperatorAdapter()
        inventory: list[dict[str, Any]] = []
        datasets: dict[str, Any] = {}
        full_audit: dict[str, Any] = {}
        spot_audit: dict[str, Any] = {}
        for dataset in EXPECTED_DATASETS:
            bundle = runner._load_bundle(screen, registry, dataset, project_root)
            datasets[dataset] = _dataset_manifest(registry, screen, bundle)
            spot_audit[dataset] = spot_selection_payload(registry, bundle)
            rows, audit = _build_dataset_files(
                registry, screen, bundle, adapter, staging
            )
            inventory.extend(rows)
            full_audit[dataset] = audit
        inventory.sort(key=lambda row: (EXPECTED_DATASETS.index(row["dataset"]),
                                        int(row["condition_index"])))
        manifest: dict[str, Any] = {
            "schema": {"name": CACHE_SCHEMA_NAME, "version": CACHE_SCHEMA_VERSION},
            "protocol_id": registry["protocol_id"],
            "registry": {"registry_id": registry["registry_id"],
                         "sha256": registry_sha256},
            "cache_identity": {
                "cache_id": identity, "content_sha256": "",
                "formal": builder["registration"] == "frozen",
            },
            "builder": {"path": builder["path"], "sha256": builder["sha256"],
                        "version": BUILDER_VERSION},
            "frozen_sources": _source_manifest(registry),
            "randomness": {
                "root_rule_id": "operator_root_v1",
                "root_key_fields": ["protocol_id", "operator_root_v1", "dataset", "sample_id"],
                "condition_order": list(CONDITIONS),
                "severity_order_ms": list(SEVERITIES_MS),
                "realization_order": list(REALIZATIONS),
            },
            "datasets": datasets,
            "conditions": [item.as_dict() for item in condition_table()],
            "inventory": inventory,
            "full_audit": full_audit,
            "spot_audit": spot_audit,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        manifest["cache_identity"]["content_sha256"] = compute_content_sha256(manifest)
        write_json_atomic(staging / "cache_manifest.json", manifest)
        validate_cache(
            staging, registry_path, deep=False, spot=True, require_commit=False,
            allow_unregistered_builder=allow_unregistered_builder,
        )
        commit = {
            "schema": {"name": "icassp_stage_b_operator_cache_commit", "version": 1},
            "protocol_id": registry["protocol_id"],
            "registry_sha256": registry_sha256,
            "cache_id": identity,
            "cache_content_sha256": manifest["cache_identity"]["content_sha256"],
            "manifest_sha256": sha256_file(staging / "cache_manifest.json"),
            "inventory_count": 90,
            "complete": True,
            "committed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        write_json_atomic(staging / "commit.json", commit)
        parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final_root)
        validate_cache(
            final_root, registry_path, deep=False, spot=True, require_commit=True,
            allow_unregistered_builder=allow_unregistered_builder,
        )
    return final_root


class PackedOperatorCache:
    """Read-only packed condition memmaps bound to one committed cache."""

    def __init__(
        self, cache_root: Path, registry_path: Path, *, deep_validate: bool = False,
        allow_unregistered_builder: bool = False,
    ):
        self.cache_root = cache_root.resolve()
        validate_cache(
            self.cache_root, registry_path, deep=deep_validate, spot=True,
            require_commit=True, allow_unregistered_builder=allow_unregistered_builder,
        )
        self.manifest = _load_manifest(self.cache_root)
        self._rows = {
            (str(row["dataset"]), str(row["condition"]), int(row["severity_ms"]),
             int(row["realization"])): row
            for row in self.manifest["inventory"]
        }
        self._arrays: dict[tuple[str, str, int, int], np.ndarray] = {}

    @property
    def content_sha256(self) -> str:
        return str(self.manifest["cache_identity"]["content_sha256"])

    def get_batch(
        self, dataset: str, condition: str, severity_ms: int, realization: int,
        sample_indices: Sequence[int], *, dtype: Any = np.float32,
    ) -> np.ndarray:
        key = (str(dataset), str(condition), int(severity_ms), int(realization))
        if key not in self._rows:
            raise CacheError(f"unknown cached condition key: {key}")
        row = self._rows[key]
        array = self._arrays.get(key)
        if array is None:
            array = np.load(
                self.cache_root / row["relative_path"], mmap_mode="r", allow_pickle=False
            )
            self._arrays[key] = array
        indices = np.asarray(sample_indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= array.shape[0]):
            raise IndexError("operator-cache sample index is out of range")
        packed = np.asarray(array[indices])
        logical = np.unpackbits(
            packed, axis=-1, count=int(row["logical_channels"]), bitorder="little"
        )
        return logical.astype(dtype, copy=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=runner.DEFAULT_REGISTRY)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--cache-parent", type=Path, default=None)
    build.add_argument("--allow-unregistered-builder", action="store_true",
                       help="Development only; never valid for a formal cache freeze.")
    validate = sub.add_parser("validate")
    validate.add_argument("--cache-root", type=Path, required=True)
    validate.add_argument("--deep", action="store_true")
    validate.add_argument("--no-spot", action="store_true")
    validate.add_argument("--allow-unregistered-builder", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        root = build_cache(
            args.registry, args.cache_parent,
            allow_unregistered_builder=args.allow_unregistered_builder,
        )
        print(json.dumps({"cache_root": str(root), "complete": True}, indent=2))
        return 0
    summary = validate_cache(
        args.cache_root, args.registry, deep=args.deep, spot=not args.no_spot,
        allow_unregistered_builder=args.allow_unregistered_builder,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
