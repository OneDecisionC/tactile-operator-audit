"""Participant-disjoint robustness evaluation for the STEMNIST tactile dataset.

The script deliberately remains independent from ``run_braille_robustness.py``.
It has two phases:

1. Read the released HDF5 spike files once and build a packed-bit ``.npy``
   cache.  The cache stores 80 x 512 binary values per sample in 5,120 bytes,
   rather than materialising a 1.47 GiB float32 tensor for the full dataset.
2. Train/evaluate a paper-topology convolutional SNN and a temporal
   convolutional network (TCN) with participant-disjoint folds.  Packed samples
   are unpacked only for the current mini-batch, and corruptions are likewise
   generated per mini-batch.

STEMNIST event convention used here
------------------------------------
* 2 seconds, 25 ms bins -> 80 time bins.
* channels 0..255: positive/ON polarity for physical taxels 1..256.
* channels 256..511: negative/OFF polarity for the same taxels.
* Multiple raw events landing in one (time-bin, channel) cell are collapsed to
  one binary event.  Post-bin time jitter can introduce additional collisions.
* A taxel fault always disables both polarity channels of a physical taxel.

This is an explicitly new ``binary80_groupcv_v1`` robustness protocol.  It is
not the paper's 20-bin/count-input/sample-split reproduction protocol, so its
clean accuracy must not be compared directly with the paper's 89.16% number.

The released Windows HDF5 files expose a compound dtype.  On this machine the
STAtten environment currently has an h5py/NumPy ABI mismatch, while the base
Anaconda Python reads the files correctly.  Cache preparation therefore has no
PyTorch dependency and can be run once with base Python; training then uses the
STAtten Python on the portable packed cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import string
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Sequence

import numpy as np


try:  # Cache preparation intentionally works in a Python without PyTorch.
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ModuleNotFoundError:  # pragma: no cover - exercised by base cache builder
    torch = None
    nn = None
    DataLoader = None
    snn = None
else:
    try:
        import snntorch as snn
    except ModuleNotFoundError:  # A local surrogate-LIF fallback remains available.
        snn = None


CLASS_NAMES = tuple(string.ascii_uppercase) + tuple("123456789")
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASS_NAMES)}
N_TAXELS = 256
N_POLARITIES = 2
N_CHANNELS = N_TAXELS * N_POLARITIES
SOURCE_ARCHIVE_MD5 = "6ca4638b2f95bf34f59873ab62399bd8"
CACHE_SCHEMA_VERSION = 3
MANIFEST_SCHEMA_VERSION = 2
CONFIG_SCHEMA_VERSION = 1
PROTOCOL_ID = "binary80_groupcv_v1"
SPLITTER_ID = "deterministic_balanced_group_v2"
FILENAME_RE = re.compile(
    r"^(?P<participant>[A-Z]{2})_(?P<label>[A-Z1-9])_(?P<repetition>[0-9]+)_spikes[.]h5$"
)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _critical_cache_index_payload(
    metadata: dict[str, Any], *, include_dataset_root: bool
) -> dict[str, Any]:
    """Return cache metadata that affects sample/label/group interpretation.

    Absolute extraction paths are provenance annotations, not data identity.
    Omitting them makes an already verified packed cache portable across hosts.
    """
    payload = {
        "cache_schema_version": metadata.get("cache_schema_version"),
        "source": metadata.get("source"),
        "class_names": metadata.get("class_names"),
        "participants": metadata.get("participants"),
        "participant_counts": metadata.get("participant_counts"),
        "label_counts": metadata.get("label_counts"),
        "encoding": metadata.get("encoding"),
        "event_audit": metadata.get("event_audit"),
        "packed_bytes": metadata.get("packed_bytes"),
        "packed_sha256": metadata.get("packed_sha256"),
        "samples": metadata.get("samples"),
    }
    if include_dataset_root:
        payload["dataset_root"] = metadata.get("dataset_root")
    return payload


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def critical_cache_index_sha256(metadata: dict[str, Any]) -> str:
    """Portable digest of all metadata affecting cache interpretation."""
    return _sha256_json(
        _critical_cache_index_payload(metadata, include_dataset_root=False)
    )


def legacy_critical_cache_index_sha256(metadata: dict[str, Any]) -> str:
    """Schema-v3 digest used before absolute dataset_root was made non-critical."""
    return _sha256_json(
        _critical_cache_index_payload(metadata, include_dataset_root=True)
    )


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_archive(data_root: Path) -> dict[str, Any]:
    """Verify the released ZIP when it accompanies the extracted tree."""
    expected_names = ("STEMNIST_Dataset.zip", "STEMNIST Dataset.zip")
    search_roots = [data_root, data_root.parent, data_root.parent.parent]
    candidates: list[Path] = []
    for root in search_roots:
        for name in expected_names:
            candidate = root / name
            if candidate.is_file() and candidate.resolve() not in {
                path.resolve() for path in candidates
            }:
                candidates.append(candidate)
        if root.is_dir():
            for candidate in root.glob("*.zip"):
                if "stemnist" in candidate.name.lower() and candidate.resolve() not in {
                    path.resolve() for path in candidates
                }:
                    candidates.append(candidate)
    if not candidates:
        return {
            "status": "not_present",
            "expected_md5": SOURCE_ARCHIVE_MD5,
            "path": None,
            "actual_md5": None,
        }
    if len(candidates) > 1:
        raise ValueError(
            "Multiple candidate STEMNIST source archives found: "
            + ", ".join(str(path) for path in candidates)
        )
    archive = candidates[0]
    actual = md5_file(archive)
    if actual.lower() != SOURCE_ARCHIVE_MD5.lower():
        raise ValueError(
            f"STEMNIST source ZIP MD5 mismatch: {actual} != {SOURCE_ARCHIVE_MD5} "
            f"for {archive}"
        )
    return {
        "status": "verified",
        "expected_md5": SOURCE_ARCHIVE_MD5,
        "path": str(archive.resolve()),
        "actual_md5": actual,
        "bytes": archive.stat().st_size,
    }


def resolved_time_geometry(event_dt: float, duration: float) -> tuple[int, int]:
    if event_dt <= 0 or duration <= 0:
        raise ValueError("event_dt and duration must be positive")
    bins_float = duration / event_dt
    bins = int(round(bins_float))
    if not math.isclose(bins_float, bins, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"duration/event_dt must be integral, got {duration}/{event_dt}={bins_float}"
        )
    packed_channels = math.ceil(N_CHANNELS / 8)
    return bins, packed_channels


def default_cache_paths(cache_dir: Path, event_dt: float, duration: float) -> tuple[Path, Path]:
    bins, _ = resolved_time_geometry(event_dt, duration)
    milliseconds = int(round(event_dt * 1000))
    stem = f"stemnist_binary_{milliseconds}ms_{bins}bins_packed"
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}_index.json"


def enumerate_spike_records(data_root: Path) -> list[dict[str, Any]]:
    processed_root = data_root / "ProcessedSpikes"
    raw_root = data_root / "RawCharacters"
    if not processed_root.is_dir():
        raise FileNotFoundError(f"Missing ProcessedSpikes directory: {processed_root}")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Missing RawCharacters directory: {raw_root}")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for label in CLASS_NAMES:
        class_dir = processed_root / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for path in class_dir.glob("*.h5"):
            match = FILENAME_RE.fullmatch(path.name)
            if match is None:
                raise ValueError(f"Unexpected spike filename: {path}")
            parsed = match.groupdict()
            if parsed["label"] != label:
                raise ValueError(
                    f"Directory/filename label mismatch: directory={label}, file={path.name}"
                )
            repetition = int(parsed["repetition"])
            sample_id = f"{parsed['participant']}_{label}_{repetition}"
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample id: {sample_id}")
            seen_ids.add(sample_id)
            raw_path = raw_root / f"{sample_id}.h5"
            if not raw_path.is_file():
                raise FileNotFoundError(f"Missing raw counterpart for {path}: {raw_path}")
            records.append(
                {
                    "sample_id": sample_id,
                    "participant": parsed["participant"],
                    "label": label,
                    "label_index": CLASS_TO_INDEX[label],
                    "repetition": repetition,
                    "processed_path": path.relative_to(data_root).as_posix(),
                    "raw_path": raw_path.relative_to(data_root).as_posix(),
                    "processed_bytes": path.stat().st_size,
                    "raw_bytes": raw_path.stat().st_size,
                }
            )

    records.sort(
        key=lambda record: (
            record["label_index"],
            record["participant"],
            record["repetition"],
        )
    )
    for index, record in enumerate(records):
        record["sample_index"] = index
    processed_discovered = {
        path.resolve() for path in processed_root.rglob("*.h5") if path.is_file()
    }
    processed_indexed = {
        (data_root / record["processed_path"]).resolve() for record in records
    }
    raw_discovered = {
        path.resolve() for path in raw_root.glob("*.h5") if path.is_file()
    }
    raw_indexed = {(data_root / record["raw_path"]).resolve() for record in records}
    if processed_discovered != processed_indexed:
        raise ValueError("ProcessedSpikes contains unindexed/missing HDF5 files")
    if raw_discovered != raw_indexed:
        raise ValueError("RawCharacters contains unpaired/missing HDF5 files")
    participant_counts = Counter(record["participant"] for record in records)
    label_counts = Counter(record["label"] for record in records)
    if len(records) != 7700 or len(participant_counts) != 34:
        raise ValueError(
            "Unexpected official STEMNIST cardinality: "
            f"samples={len(records)}, participants={len(participant_counts)}"
        )
    if any(label_counts[label] != 220 for label in CLASS_NAMES):
        raise ValueError(f"Unexpected per-label counts: {dict(sorted(label_counts.items()))}")
    return records


def _friendly_hdf5_error(exc: Exception) -> RuntimeError:
    return RuntimeError(
        "Failed to read the STEMNIST compound HDF5 dtype.  On this computer, "
        "prepare the packed cache with a Python installation with working h5py "
        "and run training with the STAtten Python.  Original error: "
        f"{type(exc).__name__}: {exc}"
    )


def build_packed_cache(
    data_root: Path,
    packed_path: Path,
    index_path: Path,
    event_dt: float,
    duration: float,
    rebuild: bool,
) -> dict[str, Any]:
    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("h5py is required only for cache preparation") from exc

    if packed_path.exists() or index_path.exists():
        if not rebuild:
            if packed_path.exists() and index_path.exists():
                existing = load_cache_metadata(index_path)
                if existing.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
                    raise ValueError(
                        "Packed cache was created with an obsolete binning schema; "
                        "rebuild it explicitly with --rebuild_cache. "
                        f"Found={existing.get('cache_schema_version')}, "
                        f"required={CACHE_SCHEMA_VERSION}"
                    )
                _, existing = load_packed_cache(
                    packed_path,
                    index_path,
                    event_dt,
                    duration,
                    verify_hash=True,
                )
                return existing
            raise FileExistsError(
                "Only one cache component exists. Remove/rebuild it explicitly with "
                f"--rebuild_cache: {packed_path}, {index_path}"
            )

    archive_provenance = verify_source_archive(data_root)
    records = enumerate_spike_records(data_root)
    bins, packed_channels = resolved_time_geometry(event_dt, duration)
    packed_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_packed = packed_path.with_name(packed_path.name + ".tmp.npy")
    if temporary_packed.exists():
        temporary_packed.unlink()

    packed = np.lib.format.open_memmap(
        temporary_packed,
        mode="w+",
        dtype=np.uint8,
        shape=(len(records), bins, packed_channels),
    )

    total_raw_spikes = 0
    total_binary_events = 0
    boundary_clips = 0
    maximum_tick_error = 0.0
    participants = Counter()
    labels = Counter()
    root_attr_schema: dict[str, str] | None = None
    spike_dtype: list[tuple[str, str]] | None = None
    started = time.time()

    try:
        for index, record in enumerate(records):
            path = data_root / record["processed_path"]
            try:
                with h5py.File(path, "r") as handle:
                    if "spikes" not in handle:
                        raise KeyError(f"Missing 'spikes' dataset in {path}")
                    spikes = handle["spikes"][:]
                    attributes = {key: handle.attrs[key] for key in handle.attrs.keys()}
            except Exception as exc:
                if "precision" in str(exc).lower() or "dtype" in str(exc).lower():
                    raise _friendly_hdf5_error(exc) from exc
                raise

            expected_fields = ("timestamp", "taxel_id", "polarity")
            if spikes.dtype.names != expected_fields:
                raise ValueError(
                    f"Unexpected spike dtype in {path}: {spikes.dtype}; expected {expected_fields}"
                )
            if root_attr_schema is None:
                root_attr_schema = {
                    key: str(np.asarray(value).dtype) for key, value in attributes.items()
                }
                spike_dtype = [(name, str(spikes.dtype[name])) for name in expected_fields]

            if int(attributes.get("num_spikes", len(spikes))) != len(spikes):
                raise ValueError(f"num_spikes attribute mismatch in {path}")
            sampling_rate = int(attributes.get("sampling_rate", 120))
            if sampling_rate != 120:
                raise ValueError(f"Unexpected sampling rate in {path}")
            if not math.isclose(
                float(attributes.get("duration", duration)),
                duration,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(f"Unexpected duration in {path}")
            original_file = str(attributes.get("original_file", ""))
            if original_file:
                released_name = PureWindowsPath(original_file).name
                if released_name != Path(record["raw_path"]).name:
                    raise ValueError(
                        f"original_file mismatch in {path}: {released_name} versus "
                        f"{Path(record['raw_path']).name}"
                    )

            timestamps = np.asarray(spikes["timestamp"], dtype=np.float64)
            taxel_ids = np.asarray(spikes["taxel_id"], dtype=np.int64)
            polarities = np.asarray(spikes["polarity"], dtype=np.int64)
            if np.any(~np.isfinite(timestamps)):
                raise ValueError(f"Non-finite timestamp in {path}")
            if np.any(timestamps < -1e-7) or np.any(timestamps > duration + 1e-6):
                raise ValueError(
                    f"Timestamp outside [0,{duration}] in {path}: "
                    f"min={timestamps.min()}, max={timestamps.max()}"
                )
            if np.any((taxel_ids < 1) | (taxel_ids > N_TAXELS)):
                raise ValueError(f"taxel_id outside 1..{N_TAXELS} in {path}")
            if np.any(~np.isin(polarities, (-1, 1))):
                raise ValueError(f"Polarity outside -1/+1 in {path}")

            # Released timestamps are float32 encodings of exact 120 Hz frame
            # ticks.  Direct floor(timestamp / 0.025) puts values such as
            # 0.524999976 (tick 63, exactly 0.525 s) in the preceding bin.
            # Recover/validate integer source ticks before constructing bins.
            tick_positions = timestamps * sampling_rate
            tick_indices = np.rint(tick_positions).astype(np.int64)
            tick_error = float(np.max(np.abs(tick_positions - tick_indices), initial=0.0))
            maximum_tick_error = max(maximum_tick_error, tick_error)
            if tick_error > 1e-3:
                raise ValueError(
                    f"Timestamp is not on the {sampling_rate} Hz grid in {path}: "
                    f"maximum tick error={tick_error}"
                )
            ticks_per_bin_float = event_dt * sampling_rate
            ticks_per_bin = int(round(ticks_per_bin_float))
            if math.isclose(
                ticks_per_bin_float, ticks_per_bin, rel_tol=0.0, abs_tol=1e-9
            ):
                bin_indices = tick_indices // ticks_per_bin
            else:
                bin_indices = np.floor(
                    tick_indices / ticks_per_bin_float + 1e-12
                ).astype(np.int64)
            boundary_clips += int(np.count_nonzero(bin_indices == bins))
            bin_indices = np.clip(bin_indices, 0, bins - 1)
            polarity_offsets = (polarities < 0).astype(np.int64) * N_TAXELS
            channels = polarity_offsets + taxel_ids - 1
            binary = np.zeros((bins, N_CHANNELS), dtype=np.uint8)
            binary[bin_indices, channels] = 1
            packed[index] = np.packbits(binary, axis=-1, bitorder="little")

            raw_count = int(len(spikes))
            binary_count = int(binary.sum())
            total_raw_spikes += raw_count
            total_binary_events += binary_count
            record["raw_spike_count"] = raw_count
            record["binary_event_count"] = binary_count
            record["bin_collision_count"] = raw_count - binary_count
            participants[record["participant"]] += 1
            labels[record["label"]] += 1

            if (index + 1) % 500 == 0 or index + 1 == len(records):
                elapsed = time.time() - started
                print(
                    f"Cache {index + 1}/{len(records)} samples, "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
        packed.flush()
    except Exception:
        del packed
        if temporary_packed.exists():
            temporary_packed.unlink()
        raise
    del packed
    os.replace(temporary_packed, packed_path)

    packed_sha256 = sha256_file(packed_path)
    total_processed_bytes = sum(record["processed_bytes"] for record in records)
    total_raw_bytes = sum(record["raw_bytes"] for record in records)
    metadata = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "STEMNIST",
        "dataset_root": str(data_root.resolve()),
        "source_archive_md5": SOURCE_ARCHIVE_MD5,
        "source_archive_md5_status": archive_provenance["status"],
        "source_archive_provenance": archive_provenance,
        "source": {
            "processed_files": len(records),
            "raw_files": len(records),
            "processed_bytes": total_processed_bytes,
            "raw_bytes": total_raw_bytes,
            "root_attribute_dtypes": root_attr_schema,
            "spike_dtype": spike_dtype,
            "sample_identity_source": (
                "processed filename validated against class directory and raw basename; "
                "raw experimenter/experiment_number attributes are not used"
            ),
        },
        "class_names": list(CLASS_NAMES),
        "participants": sorted(participants),
        "participant_counts": dict(sorted(participants.items())),
        "label_counts": {label: labels[label] for label in CLASS_NAMES},
        "encoding": {
            "duration_seconds": duration,
            "event_dt_seconds": event_dt,
            "time_bins": bins,
            "physical_taxels": N_TAXELS,
            "polarities": N_POLARITIES,
            "channels": N_CHANNELS,
            "channel_mapping": {
                "0:255": "ON/+1 for taxel_id 1:256",
                "256:511": "OFF/-1 for taxel_id 1:256",
            },
            "representation": "post-bin binary occupancy",
            "binning": (
                "recover nearest 120-Hz integer tick, validate <=1e-3 tick error, "
                "then integer-divide by ticks_per_bin when event_dt*120 is integral; "
                "exact-duration events clipped to final bin"
            ),
            "binning_implementation": "source_tick_quantized_v2",
            "packbits_axis": -1,
            "packbits_bitorder": "little",
            "packed_shape": [len(records), bins, packed_channels],
            "packed_dtype": "uint8",
        },
        "event_audit": {
            "raw_spikes": total_raw_spikes,
            "binary_events": total_binary_events,
            "bin_collisions": total_raw_spikes - total_binary_events,
            "binary_retention": (
                total_binary_events / total_raw_spikes if total_raw_spikes else 1.0
            ),
            "exact_duration_boundary_clips": boundary_clips,
            "maximum_source_tick_error": maximum_tick_error,
        },
        "packed_file": str(packed_path.resolve()),
        "packed_bytes": packed_path.stat().st_size,
        "packed_sha256": packed_sha256,
        "samples": records,
        "critical_index_hash_scheme": "portable_without_dataset_root_v2",
    }
    metadata["critical_index_sha256"] = critical_cache_index_sha256(metadata)
    write_json_atomic(index_path, metadata)
    return metadata


def load_cache_metadata(index_path: Path) -> dict[str, Any]:
    return json.loads(index_path.read_text(encoding="utf-8"))


def load_packed_cache(
    packed_path: Path,
    index_path: Path,
    event_dt: float,
    duration: float,
    verify_hash: bool,
) -> tuple[np.memmap, dict[str, Any]]:
    if not packed_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            f"Packed cache is incomplete: {packed_path}, {index_path}. "
            "Run --prepare_cache_only first."
        )
    metadata = load_cache_metadata(index_path)
    if metadata.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported cache schema: {metadata.get('cache_schema_version')}")
    expected_index_sha = metadata.get("critical_index_sha256")
    actual_index_sha = critical_cache_index_sha256(metadata)
    hash_scheme = metadata.get("critical_index_hash_scheme")
    legacy_hash_accepted = False
    if expected_index_sha != actual_index_sha:
        legacy_index_sha = legacy_critical_cache_index_sha256(metadata)
        if hash_scheme is None and expected_index_sha == legacy_index_sha:
            # Existing schema-v3 caches remain readable without rewriting a
            # provenance file while it may be used by another process.
            legacy_hash_accepted = True
        else:
            raise ValueError(
                "Critical cache-index SHA-256 mismatch; labels/groups/encoding may be "
                f"modified or mismatched: {actual_index_sha} != {expected_index_sha}"
            )
    encoding = metadata["encoding"]
    if not math.isclose(
        float(encoding["event_dt_seconds"]), event_dt, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Cache event_dt does not match requested event_dt")
    if not math.isclose(
        float(encoding["duration_seconds"]), duration, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Cache duration does not match requested duration")
    if verify_hash:
        actual_sha = sha256_file(packed_path)
        if actual_sha != metadata["packed_sha256"]:
            raise ValueError(
                f"Packed cache SHA-256 mismatch: {actual_sha} != {metadata['packed_sha256']}"
            )
    packed = np.load(packed_path, mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(int(value) for value in encoding["packed_shape"])
    if packed.shape != expected_shape or packed.dtype != np.uint8:
        raise ValueError(
            f"Packed cache shape/dtype mismatch: {packed.shape}/{packed.dtype}, "
            f"expected {expected_shape}/uint8"
        )
    if len(metadata["samples"]) != packed.shape[0]:
        raise ValueError("Cache sample metadata length does not match packed array")
    if metadata.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Cache class order does not match the runner")
    if int(encoding.get("channels", -1)) != N_CHANNELS:
        raise ValueError("Cache channel count does not match the runner")
    if int(encoding.get("time_bins", -1)) != expected_shape[1]:
        raise ValueError("Cache time-bin metadata does not match its array shape")
    records = metadata["samples"]
    seen_sample_ids: set[str] = set()
    for expected_index, record in enumerate(records):
        if int(record.get("sample_index", -1)) != expected_index:
            raise ValueError(f"Non-consecutive sample_index at record {expected_index}")
        label = record.get("label")
        if label not in CLASS_TO_INDEX or int(record.get("label_index", -1)) != CLASS_TO_INDEX[label]:
            raise ValueError(f"Invalid label mapping at record {expected_index}")
        sample_id = str(record.get("sample_id", ""))
        if not sample_id or sample_id in seen_sample_ids:
            raise ValueError(f"Missing/duplicate sample_id at record {expected_index}")
        seen_sample_ids.add(sample_id)
        expected_id = f"{record.get('participant')}_{label}_{int(record.get('repetition', -1))}"
        if sample_id != expected_id:
            raise ValueError(f"Sample identity fields disagree at record {expected_index}")
    participants_from_records = sorted({record["participant"] for record in records})
    if participants_from_records != metadata.get("participants"):
        raise ValueError("Cache participant list does not match sample records")
    metadata["portable_critical_index_sha256"] = actual_index_sha
    metadata["critical_index_hash_validation"] = (
        "accepted_legacy_schema3_with_absolute_root"
        if legacy_hash_accepted
        else "portable_hash_verified"
    )
    return packed, metadata


def select_sample_indices(
    metadata: dict[str, Any],
    seed: int,
    max_participants: int,
    max_samples_per_participant_class: int,
) -> np.ndarray:
    records = metadata["samples"]
    indices = np.arange(len(records), dtype=np.int64)
    participants = sorted({record["participant"] for record in records})
    if max_participants > 0:
        if max_participants < 2:
            raise ValueError("max_participants must be zero or at least two")
        rng = np.random.default_rng(seed)
        shuffled = np.asarray(participants, dtype=object)
        rng.shuffle(shuffled)
        selected_participants = set(shuffled[:max_participants].tolist())
        indices = np.asarray(
            [
                index
                for index in indices
                if records[int(index)]["participant"] in selected_participants
            ],
            dtype=np.int64,
        )
    if max_samples_per_participant_class > 0:
        grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index in indices:
            record = records[int(index)]
            grouped[(record["participant"], record["label"])].append(int(index))
        selected: list[int] = []
        for key in sorted(grouped):
            members = sorted(
                grouped[key], key=lambda idx: records[idx]["repetition"]
            )
            selected.extend(members[:max_samples_per_participant_class])
        indices = np.asarray(sorted(selected), dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("Sample filtering selected no samples")
    return indices


def _deterministic_group_folds(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} participant groups, found {len(unique_groups)}"
        )
    n_classes = int(labels.max()) + 1
    group_counts: dict[str, np.ndarray] = {}
    group_sizes: dict[str, int] = {}
    for group in unique_groups:
        member_labels = labels[groups == group]
        group_counts[group] = np.bincount(member_labels, minlength=n_classes)
        group_sizes[group] = int(len(member_labels))
    rng = np.random.default_rng(seed)
    random_ties = {group: float(rng.random()) for group in unique_groups}
    ordered_groups = sorted(
        unique_groups,
        key=lambda group: (-group_sizes[group], random_ties[group], group),
    )
    total_class = np.bincount(labels, minlength=n_classes).astype(np.float64)
    target_class = total_class / n_splits
    target_size = len(labels) / n_splits
    fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
    fold_counts = np.zeros((n_splits, n_classes), dtype=np.float64)
    fold_sizes = np.zeros(n_splits, dtype=np.float64)

    for order_index, group in enumerate(ordered_groups):
        if order_index < n_splits:
            candidate_folds = [order_index]
        else:
            candidate_folds = list(range(n_splits))
        def placement_score(fold: int) -> tuple[float, float, float, int]:
            trial_counts = fold_counts.copy()
            trial_sizes = fold_sizes.copy()
            trial_counts[fold] += group_counts[group]
            trial_sizes[fold] += group_sizes[group]
            # Score the global fold configuration, not only how close the one
            # candidate fold is to its target.  The previous local score could
            # fill early folds and leave one-participant validation folds.
            class_fraction = trial_counts / np.maximum(total_class[None, :], 1.0)
            class_imbalance = float(
                np.mean((class_fraction - (1.0 / n_splits)) ** 2)
            )
            size_fraction = trial_sizes / max(float(len(labels)), 1.0)
            size_imbalance = float(
                np.mean((size_fraction - (1.0 / n_splits)) ** 2)
            )
            return (
                class_imbalance + size_imbalance,
                float(trial_sizes.max() - trial_sizes.min()),
                float(fold_sizes[fold]),
                fold,
            )

        best_fold = min(candidate_folds, key=placement_score)
        fold_groups[best_fold].append(group)
        fold_counts[best_fold] += group_counts[group]
        fold_sizes[best_fold] += group_sizes[group]

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for validation_groups in fold_groups:
        validation_mask = np.isin(groups, validation_groups)
        validation = np.flatnonzero(validation_mask)
        train = np.flatnonzero(~validation_mask)
        folds.append((train, validation))
    return folds


def make_group_folds(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    if len(set(groups.tolist())) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} participants, found {len(set(groups.tolist()))}"
        )
    # Always use the repository-owned implementation.  Selecting sklearn when
    # importable and a fallback otherwise made the same seed environment-
    # dependent, even though both branches were individually deterministic.
    folds = _deterministic_group_folds(labels, groups, n_splits, seed)
    validation_sizes = [len(validation) for _, validation in folds]
    largest_group = max(Counter(groups.tolist()).values())
    if max(validation_sizes) - min(validation_sizes) > largest_group:
        raise AssertionError(
            "Built-in group split is insufficiently balanced: "
            f"validation sizes={validation_sizes}, largest group={largest_group}"
        )
    validation_group_counts = [
        len(set(groups[validation].tolist())) for _, validation in folds
    ]
    if len(set(groups.tolist())) >= 2 * n_splits and min(validation_group_counts) < 2:
        raise AssertionError(
            "Built-in splitter produced a one-participant validation fold despite "
            "enough groups"
        )
    return [(np.asarray(a), np.asarray(b)) for a, b in folds], SPLITTER_ID


def validate_group_folds(
    folds: list[tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
    groups: np.ndarray,
) -> list[dict[str, Any]]:
    validation_seen = np.zeros(len(labels), dtype=np.int64)
    audits: list[dict[str, Any]] = []
    for fold_index, (train, validation) in enumerate(folds, start=1):
        train_groups = set(groups[train].tolist())
        validation_groups = set(groups[validation].tolist())
        overlap = sorted(train_groups & validation_groups)
        if overlap:
            raise AssertionError(f"Participant leakage in fold {fold_index}: {overlap}")
        validation_seen[validation] += 1
        total_label_counts = np.bincount(labels, minlength=len(CLASS_NAMES))
        validation_label_counts = np.bincount(
            labels[validation], minlength=len(CLASS_NAMES)
        )
        missing_feasible = np.flatnonzero(
            (total_label_counts >= len(folds)) & (validation_label_counts == 0)
        )
        if len(missing_feasible):
            raise AssertionError(
                f"Fold {fold_index} misses feasible labels: {missing_feasible.tolist()}"
            )
        audits.append(
            {
                "fold": fold_index,
                "train_samples": int(len(train)),
                "validation_samples": int(len(validation)),
                "train_participants": sorted(train_groups),
                "validation_participants": sorted(validation_groups),
                "participant_overlap": overlap,
                "train_label_counts": np.bincount(
                    labels[train], minlength=len(CLASS_NAMES)
                ).tolist(),
                "validation_label_counts": validation_label_counts.tolist(),
            }
        )
    if not np.all(validation_seen == 1):
        raise AssertionError(
            "Every selected sample must appear in exactly one validation fold; "
            f"counts={Counter(validation_seen.tolist())}"
        )
    return audits


def condition_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "corruption": "clean",
            "amount_name": "none",
            "amount": 0,
            "jitter_max_ms": 0.0,
            "taxel_fraction": 0.0,
            "seed_offset": 0,
            "implementation": "identity_v1",
        }
    ]
    specs.extend(
        {
            "corruption": "event_dropout",
            "amount_name": "rate",
            "amount": float(rate),
            "jitter_max_ms": 0.0,
            "taxel_fraction": 0.0,
            "seed_offset": int(round(float(rate) * 1000)),
            "implementation": "per_sample_active_event_bernoulli_v1",
        }
        for rate in args.event_dropout
    )
    specs.extend(
        {
            "corruption": "time_jitter",
            "amount_name": "steps",
            "amount": int(radius),
            "jitter_max_ms": round(
                float(radius) * float(args.event_dt) * 1000.0, 6
            ),
            "taxel_fraction": 0.0,
            "seed_offset": 2000 + int(radius),
            "implementation": "postbin_binary_shift_clip_collision_v1",
        }
        for radius in args.time_jitter
    )
    specs.extend(
        {
            "corruption": "taxel_dropout",
            "amount_name": "taxels",
            "amount": int(count),
            "jitter_max_ms": 0.0,
            "taxel_fraction": float(count) / N_TAXELS,
            "seed_offset": 3000 + int(count),
            "implementation": "per_sample_physical_taxel_mask_both_polarities_v1",
        }
        for count in args.taxel_dropout
    )
    return specs


def sample_rng(base_seed: int, sample_index: int, stream: int = 0) -> np.random.Generator:
    entropy = [
        int(base_seed) & 0xFFFFFFFF,
        int(sample_index) & 0xFFFFFFFF,
        int(stream) & 0xFFFFFFFF,
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def stable_torch_seed(base_seed: int, sample_index: int, stream: int = 0) -> int:
    """Derive a process-independent positive int64 seed for one sample."""
    payload = f"{int(base_seed)}:{int(sample_index)}:{int(stream)}".encode("ascii")
    digest = hashlib.blake2b(payload, digest_size=8, person=b"stemnist").digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF_FFFF_FFFF


def sample_torch_generator(
    base_seed: int, sample_index: int, stream: int = 0
) -> "torch.Generator":
    generator = torch.Generator()
    generator.manual_seed(stable_torch_seed(base_seed, sample_index, stream))
    return generator


def apply_corruption_array(
    sample: np.ndarray,
    corruption: str,
    amount: float | int,
    rng: np.random.Generator,
) -> np.ndarray:
    if corruption == "clean":
        return sample.copy()
    if corruption == "event_dropout":
        rate = float(amount)
        if not 0 <= rate <= 1:
            raise ValueError(f"event dropout rate must lie in [0,1], got {rate}")
        output = sample.copy()
        flat = output.reshape(-1)
        active = np.flatnonzero(flat)
        if len(active):
            removed = rng.random(len(active)) < rate
            flat[active[removed]] = 0
        return output
    if corruption == "time_jitter":
        radius = int(amount)
        if radius <= 0:
            return sample.copy()
        output = np.zeros_like(sample)
        times, channels = np.nonzero(sample)
        if len(times):
            shifts = rng.integers(-radius, radius + 1, size=len(times))
            shifted = np.clip(times + shifts, 0, sample.shape[0] - 1)
            # Assignment intentionally collapses events landing on the same cell.
            output[shifted, channels] = 1
        return output
    if corruption == "taxel_dropout":
        count = min(max(int(amount), 0), N_TAXELS)
        output = sample.copy()
        if count:
            dropped = rng.choice(N_TAXELS, size=count, replace=False)
            output[:, dropped] = 0
            output[:, dropped + N_TAXELS] = 0
        return output
    raise ValueError(f"Unknown corruption: {corruption}")


def apply_corruption_tensor(
    sample: "torch.Tensor",
    corruption: str,
    amount: float | int,
    generator: "torch.Generator",
) -> "torch.Tensor":
    """CPU-tensor counterpart used by training/evaluation mini-batches."""
    if sample.device.type != "cpu":
        raise ValueError("Per-sample corruption must run before transfer to the accelerator")
    if corruption == "clean":
        return sample.clone()
    if corruption == "event_dropout":
        rate = float(amount)
        if not 0 <= rate <= 1:
            raise ValueError(f"event dropout rate must lie in [0,1], got {rate}")
        output = sample.clone()
        flat = output.reshape(-1)
        active = torch.nonzero(flat != 0, as_tuple=False).flatten()
        if active.numel():
            removed = torch.rand(active.numel(), generator=generator) < rate
            flat[active[removed]] = 0
        return output
    if corruption == "time_jitter":
        radius = int(amount)
        if radius <= 0:
            return sample.clone()
        output = torch.zeros_like(sample)
        coordinates = torch.nonzero(sample != 0, as_tuple=False)
        if coordinates.numel():
            shifts = torch.randint(
                -radius,
                radius + 1,
                (coordinates.shape[0],),
                generator=generator,
            )
            shifted = (coordinates[:, 0] + shifts).clamp_(0, sample.shape[0] - 1)
            # Advanced assignment intentionally collapses events landing on one cell.
            output[shifted, coordinates[:, 1]] = 1
        return output
    if corruption == "taxel_dropout":
        count = min(max(int(amount), 0), N_TAXELS)
        output = sample.clone()
        if count:
            dropped = torch.randperm(N_TAXELS, generator=generator)[:count]
            output[:, dropped] = 0
            output[:, dropped + N_TAXELS] = 0
        return output
    raise ValueError(f"Unknown corruption: {corruption}")


def corrupt_batch(
    inputs: "torch.Tensor",
    sample_indices: "torch.Tensor",
    spec: dict[str, Any],
    fold_seed: int,
) -> tuple["torch.Tensor", list[int], list[int]]:
    before = inputs.sum(dim=(1, 2), dtype=torch.int64).tolist()
    if spec["corruption"] == "clean":
        return inputs, before, list(before)
    output = torch.empty_like(inputs)
    condition_seed = fold_seed + int(spec["seed_offset"])
    for batch_index, sample_index in enumerate(sample_indices.tolist()):
        output[batch_index] = apply_corruption_tensor(
            inputs[batch_index],
            spec["corruption"],
            spec["amount"],
            sample_torch_generator(condition_seed, int(sample_index)),
        )
    after = output.sum(dim=(1, 2), dtype=torch.int64).tolist()
    return output, before, after


def augment_training_batch(
    inputs: "torch.Tensor",
    sample_indices: "torch.Tensor",
    args: argparse.Namespace,
    fold_seed: int,
    epoch: int,
) -> "torch.Tensor":
    if args.train_aug_mode == "none":
        return inputs
    if args.train_aug_mode != "sample_one_sample":
        raise ValueError(f"Unknown train_aug_mode: {args.train_aug_mode}")
    kinds: list[tuple[str, float | int, float]] = []
    if args.train_event_weight > 0 and args.train_event_dropout > 0:
        kinds.append(("event_dropout", args.train_event_dropout, args.train_event_weight))
    if args.train_time_weight > 0 and args.train_time_jitter > 0:
        kinds.append(("time_jitter", args.train_time_jitter, args.train_time_weight))
    if args.train_taxel_weight > 0 and args.train_taxel_dropout > 0:
        kinds.append(("taxel_dropout", args.train_taxel_dropout, args.train_taxel_weight))
    if not kinds:
        return inputs
    weights = [float(entry[2]) for entry in kinds]
    weight_total = sum(weights)
    cumulative: list[float] = []
    running = 0.0
    for weight in weights:
        running += weight / weight_total
        cumulative.append(running)
    output = inputs.clone()
    base_seed = fold_seed + 100_000 + epoch
    for batch_index, sample_index in enumerate(sample_indices.tolist()):
        generator = sample_torch_generator(base_seed, int(sample_index), stream=17)
        if float(torch.rand((), generator=generator).item()) < args.train_aug_clean_prob:
            continue
        choice = float(torch.rand((), generator=generator).item())
        kind_index = next(
            (index for index, boundary in enumerate(cumulative) if choice < boundary),
            len(kinds) - 1,
        )
        corruption, amount, _ = kinds[kind_index]
        output[batch_index] = apply_corruption_tensor(
            inputs[batch_index], corruption, amount, generator
        )
    return output


class PackedSubset:
    def __init__(
        self,
        packed: np.memmap,
        labels: np.ndarray,
        global_indices: np.ndarray,
    ) -> None:
        self.packed = packed
        self.labels = labels
        self.global_indices = np.asarray(global_indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.global_indices))

    def __getitem__(self, item: int) -> tuple[np.ndarray, int, int]:
        global_index = int(self.global_indices[item])
        return (
            np.asarray(self.packed[global_index]),
            int(self.labels[global_index]),
            global_index,
        )


def packed_collate(
    batch: Sequence[tuple[np.ndarray, int, int]],
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    packed = np.stack([item[0] for item in batch], axis=0)
    unpacked = np.unpackbits(
        packed, axis=-1, count=N_CHANNELS, bitorder="little"
    ).astype(np.float32, copy=False)
    labels = np.asarray([item[1] for item in batch], dtype=np.int64)
    indices = np.asarray([item[2] for item in batch], dtype=np.int64)
    # torch.from_numpy is broken by the currently installed mixed NumPy ABI in
    # STAtten, while as_tensor remains a zero-copy conversion for these arrays.
    return torch.as_tensor(unpacked), torch.as_tensor(labels), torch.as_tensor(indices)


if nn is not None:

    class _SurrogateSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, membrane_minus_threshold: "torch.Tensor") -> "torch.Tensor":
            ctx.save_for_backward(membrane_minus_threshold)
            return (membrane_minus_threshold > 0).to(membrane_minus_threshold.dtype)

        @staticmethod
        def backward(ctx: Any, gradient: "torch.Tensor") -> tuple["torch.Tensor"]:
            (distance,) = ctx.saved_tensors
            slope = 25.0
            surrogate = 1.0 / (1.0 + slope * distance.abs()).pow(2)
            return (gradient * surrogate,)


    class LocalLeaky(nn.Module):
        """Reset-by-subtraction LIF used only when snnTorch is unavailable."""

        def __init__(self, beta: float, threshold: float) -> None:
            super().__init__()
            self.beta = float(beta)
            self.threshold = float(threshold)

        def init_leaky(self) -> None:
            return None

        def forward(
            self, current: "torch.Tensor", membrane: "torch.Tensor | None"
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            if membrane is None:
                membrane = torch.zeros_like(current)
            membrane = self.beta * membrane + current
            spike = _SurrogateSpike.apply(membrane - self.threshold)
            membrane = membrane - spike.detach() * self.threshold
            return spike, membrane


    def make_leaky(beta: float, threshold: float) -> "nn.Module":
        if snn is not None:
            return snn.Leaky(beta=beta, threshold=threshold)
        return LocalLeaky(beta=beta, threshold=threshold)

    class PaperConvSNN(nn.Module):
        """The 2-layer 2D convolutional SNN topology described by STEMNIST."""

        def __init__(
            self,
            n_output: int,
            beta: float,
            threshold: float,
            dropout: float,
            readout: str,
        ) -> None:
            super().__init__()
            self.readout = readout
            self.conv1 = nn.Conv2d(2, 8, kernel_size=4, stride=1, padding=0)
            self.lif1 = make_leaky(beta=beta, threshold=threshold)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
            self.lif2 = make_leaky(beta=beta, threshold=threshold)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.drop = nn.Dropout(dropout)
            self.fc_out = nn.Linear(16 * 3 * 3, n_output)
            self.lif_out = make_leaky(beta=beta, threshold=threshold)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            batch, steps, channels = x.shape
            if channels != N_CHANNELS:
                raise ValueError(f"PaperConvSNN expects {N_CHANNELS} channels")
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem_out = self.lif_out.init_leaky()
            spk_sum = torch.zeros(batch, self.fc_out.out_features, device=x.device)
            mem_sum = torch.zeros_like(spk_sum)
            for step in range(steps):
                frame = x[:, step].reshape(batch, 2, 16, 16)
                spk1, mem1 = self.lif1(self.conv1(frame), mem1)
                spk1 = self.pool1(spk1)
                spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
                spk2 = self.pool2(spk2)
                current = self.fc_out(self.drop(spk2.flatten(1)))
                spk_out, mem_out = self.lif_out(current, mem_out)
                spk_sum += spk_out
                mem_sum += mem_out
            if self.readout == "spikes":
                return spk_sum
            if self.readout == "membrane_sum":
                return mem_sum
            if self.readout == "membrane_final":
                return mem_out
            raise ValueError(f"Unknown SNN readout: {self.readout}")


    class CausalChomp1d(nn.Module):
        def __init__(self, size: int) -> None:
            super().__init__()
            self.size = int(size)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            if self.size == 0:
                return x
            return x[:, :, : -self.size].contiguous()


    class TCNBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int,
            dropout: float,
        ) -> None:
            super().__init__()
            padding = (kernel_size - 1) * dilation
            self.net = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                ),
                CausalChomp1d(padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                ),
                CausalChomp1d(padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.residual = (
                nn.Conv1d(in_channels, out_channels, kernel_size=1)
                if in_channels != out_channels
                else nn.Identity()
            )
            self.activation = nn.ReLU()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.activation(self.net(x) + self.residual(x))


    class TCNClassifier(nn.Module):
        def __init__(
            self,
            n_output: int,
            hidden: int,
            blocks: int,
            dropout: float,
        ) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            in_channels = N_CHANNELS
            for block in range(blocks):
                layers.append(
                    TCNBlock(
                        in_channels,
                        hidden,
                        kernel_size=3,
                        dilation=2**block,
                        dropout=dropout,
                    )
                )
                in_channels = hidden
            self.features = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(hidden, n_output)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            features = self.features(x.transpose(1, 2))
            return self.head(self.pool(features).squeeze(-1))


def make_model(model_name: str, args: argparse.Namespace) -> "nn.Module":
    if nn is None:
        raise RuntimeError("PyTorch is required for training")
    if model_name == "paper_scnn":
        return PaperConvSNN(
            n_output=len(CLASS_NAMES),
            beta=args.beta,
            threshold=args.threshold_v,
            dropout=args.dropout,
            readout=args.snn_readout,
        )
    if model_name == "tcn":
        return TCNClassifier(
            n_output=len(CLASS_NAMES),
            hidden=args.hidden,
            blocks=args.layers,
            dropout=args.dropout,
        )
    raise ValueError(f"Unknown model: {model_name}")


def model_backend(model_name: str) -> str:
    if model_name == "paper_scnn":
        return "snntorch.Leaky" if snn is not None else "local_surrogate_lif"
    if model_name == "tcn":
        return "pytorch_tcn"
    raise ValueError(f"Unknown model: {model_name}")


def validate_model_backends(args: argparse.Namespace) -> dict[str, str]:
    if "paper_scnn" in args.models and snn is None and not args.allow_local_lif:
        raise RuntimeError(
            "paper_scnn requires snnTorch for a formal run. Install snntorch or "
            "explicitly pass --allow_local_lif to use the non-paper fallback."
        )
    return {name: model_backend(name) for name in args.models}


def config_payload(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    model_backends: dict[str, str],
) -> dict[str, Any]:
    """Scientifically relevant, seed-independent experiment identity."""
    return {
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "data": {
            "dataset": "STEMNIST",
            "cache_schema_version": metadata["cache_schema_version"],
            "packed_sha256": metadata["packed_sha256"],
            "portable_critical_index_sha256": metadata[
                "portable_critical_index_sha256"
            ],
            "event_dt": args.event_dt,
            "duration": args.duration,
            "representation": metadata["encoding"]["representation"],
        },
        "split": {
            "splitter_id": SPLITTER_ID,
            "folds": args.folds,
            "max_folds": args.max_folds,
            "max_participants": args.max_participants,
            "max_samples_per_participant_class": (
                args.max_samples_per_participant_class
            ),
        },
        "models": {
            "names": list(args.models),
            "backends": model_backends,
            "backend_versions": {
                "torch": torch.__version__,
                "snntorch": getattr(snn, "__version__", None),
            },
            "paper_scnn": {
                "beta": args.beta,
                "threshold": args.threshold_v,
                "dropout": args.dropout,
                "readout": args.snn_readout,
            },
            "tcn": {
                "hidden": args.hidden,
                "layers": args.layers,
                "dropout": args.dropout,
            },
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "deterministic": args.deterministic,
            "augmentation": {
                "mode": args.train_aug_mode,
                "clean_probability": args.train_aug_clean_prob,
                "event_dropout": args.train_event_dropout,
                "time_jitter": args.train_time_jitter,
                "taxel_dropout": args.train_taxel_dropout,
                "weights": [
                    args.train_event_weight,
                    args.train_time_weight,
                    args.train_taxel_weight,
                ],
            },
        },
        "corruption": {
            "event_dropout": list(args.event_dropout),
            "time_jitter": list(args.time_jitter),
            "taxel_dropout": list(args.taxel_dropout),
            "rng": "blake2b64_per_sample_fold_condition_v1",
            "severity_streams": "independent_non_nested_v1",
        },
    }


def make_config_id(payload: dict[str, Any]) -> str:
    return "cfg_" + _sha256_json(payload)[:16]


def configure_determinism(deterministic: bool) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(bool(deterministic))
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic)


def seed_everything(seed: int, deterministic: bool) -> None:
    configure_determinism(deterministic)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> "torch.device":
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def make_loader(
    packed: np.memmap,
    labels: np.ndarray,
    indices: np.ndarray,
    args: argparse.Namespace,
    shuffle: bool,
    seed: int,
) -> "DataLoader":
    dataset = PackedSubset(packed, labels, indices)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        collate_fn=packed_collate,
        generator=generator,
        drop_last=False,
    )


def train_one_epoch(
    model: "nn.Module",
    loader: "DataLoader",
    optimizer: "torch.optim.Optimizer",
    loss_fn: "nn.Module",
    device: "torch.device",
    args: argparse.Namespace,
    fold_seed: int,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for inputs, targets, sample_indices in loader:
        inputs = augment_training_batch(inputs, sample_indices, args, fold_seed, epoch)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(targets)
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += int(len(targets))
    return total_loss / total, total_correct / total


PREDICTION_FIELDS = [
    "config_id",
    "protocol_id",
    "model",
    "backend",
    "split_seed",
    "fold",
    "fold_seed",
    "sample_index",
    "sample_id",
    "participant",
    "label",
    "target",
    "prediction",
    "correct",
    "true_logit",
    "predicted_logit",
    "margin",
    "active_before",
    "active_after",
    "corruption",
    "amount_name",
    "amount",
    "jitter_max_ms",
    "taxel_fraction",
]


def evaluate_condition(
    model: "nn.Module",
    loader: "DataLoader",
    loss_fn: "nn.Module",
    device: "torch.device",
    spec: dict[str, Any],
    fold_seed: int,
    model_name: str,
    backend: str,
    config_id: str,
    split_seed: int,
    fold: int,
    records: list[dict[str, Any]],
    prediction_writer: csv.DictWriter | None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    active_before_total = 0
    active_after_total = 0
    participant_correct: Counter[str] = Counter()
    participant_total: Counter[str] = Counter()
    with torch.no_grad():
        for inputs, targets, sample_indices in loader:
            corrupted, active_before, active_after = corrupt_batch(
                inputs, sample_indices, spec, fold_seed
            )
            corrupted = corrupted.to(device, non_blocking=True)
            targets_device = targets.to(device, non_blocking=True)
            logits = model(corrupted)
            loss = loss_fn(logits, targets_device)
            predictions = logits.argmax(dim=1)
            top2 = logits.topk(k=min(2, logits.shape[1]), dim=1).values
            margins = top2[:, 0] - top2[:, -1]
            true_logits = logits.gather(1, targets_device[:, None]).squeeze(1)
            predicted_logits = logits.gather(1, predictions[:, None]).squeeze(1)

            total_loss += float(loss.item()) * len(targets)
            total_correct += int((predictions == targets_device).sum().item())
            total += int(len(targets))
            active_before_total += int(sum(active_before))
            active_after_total += int(sum(active_after))

            predictions_cpu = predictions.cpu().tolist()
            targets_cpu = targets.tolist()
            global_indices = sample_indices.tolist()
            for batch_index, global_index in enumerate(global_indices):
                participant = records[int(global_index)]["participant"]
                participant_total[participant] += 1
                participant_correct[participant] += int(
                    int(predictions_cpu[batch_index]) == int(targets_cpu[batch_index])
                )

            if prediction_writer is not None:
                # tolist avoids torch.from/to-numpy ABI coupling in the STAtten env.
                true_logits_cpu = true_logits.cpu().tolist()
                predicted_logits_cpu = predicted_logits.cpu().tolist()
                margins_cpu = margins.cpu().tolist()
                for batch_index, global_index in enumerate(global_indices):
                    record = records[int(global_index)]
                    target = int(targets[batch_index].item())
                    prediction = int(predictions_cpu[batch_index])
                    prediction_writer.writerow(
                        {
                            "config_id": config_id,
                            "protocol_id": PROTOCOL_ID,
                            "model": model_name,
                            "backend": backend,
                            "split_seed": split_seed,
                            "fold": fold,
                            "fold_seed": fold_seed,
                            "sample_index": global_index,
                            "sample_id": record["sample_id"],
                            "participant": record["participant"],
                            "label": record["label"],
                            "target": target,
                            "prediction": prediction,
                            "correct": int(target == prediction),
                            "true_logit": float(true_logits_cpu[batch_index]),
                            "predicted_logit": float(predicted_logits_cpu[batch_index]),
                            "margin": float(margins_cpu[batch_index]),
                            "active_before": int(active_before[batch_index]),
                            "active_after": int(active_after[batch_index]),
                            "corruption": spec["corruption"],
                            "amount_name": spec["amount_name"],
                            "amount": spec["amount"],
                            "jitter_max_ms": spec["jitter_max_ms"],
                            "taxel_fraction": spec["taxel_fraction"],
                        }
                    )
    return {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
        "participant_macro_accuracy": statistics.mean(
            participant_correct[participant] / count
            for participant, count in participant_total.items()
        ),
        "active_before": active_before_total,
        "active_after": active_after_total,
        "activity_retention": (
            active_after_total / active_before_total if active_before_total else 1.0
        ),
        "samples": total,
    }


RESULT_FIELDS = [
    "config_id",
    "protocol_id",
    "model",
    "backend",
    "split_seed",
    "fold",
    "fold_seed",
    "splitter",
    "train_samples",
    "validation_samples",
    "train_participants",
    "validation_participants",
    "epochs",
    "parameters",
    "corruption",
    "amount_name",
    "amount",
    "jitter_max_ms",
    "taxel_fraction",
    "loss",
    "accuracy",
    "participant_macro_accuracy",
    "active_before",
    "active_after",
    "activity_retention",
]


SUMMARY_FIELDS = [
    "config_id",
    "protocol_id",
    "model",
    "backend",
    "split_seed",
    "folds",
    "corruption",
    "amount_name",
    "amount",
    "jitter_max_ms",
    "taxel_fraction",
    "loss_mean",
    "loss_std",
    "loss_pooled",
    "accuracy_mean",
    "accuracy_std",
    "accuracy_pooled",
    "participant_macro_accuracy_fold_mean",
    "participant_macro_accuracy_pooled",
    "clean_accuracy_drop_pp_mean",
    "clean_accuracy_drop_pp_pooled",
    "activity_retention_mean",
    "activity_retention_std",
    "activity_retention_pooled",
]


HISTORY_FIELDS = [
    "config_id",
    "protocol_id",
    "model",
    "backend",
    "split_seed",
    "fold",
    "fold_seed",
    "epoch",
    "train_loss",
    "train_accuracy",
]


def summarize_fold_results(rows: list[dict[str, Any]], split_seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["model"]),
            str(row["corruption"]),
            str(row["amount_name"]),
            str(row["amount"]),
        )
        grouped[key].append(row)
    summary: list[dict[str, Any]] = []
    clean_by_model_fold = {
        (str(row["model"]), int(row["fold"])): row
        for row in rows
        if row["corruption"] == "clean"
    }
    for key, members in sorted(grouped.items()):
        losses = [float(member["loss"]) for member in members]
        accuracies = [float(member["accuracy"]) for member in members]
        participant_macros = [
            float(member["participant_macro_accuracy"]) for member in members
        ]
        retentions = [float(member["activity_retention"]) for member in members]
        sample_weights = [int(member["validation_samples"]) for member in members]
        weight_total = sum(sample_weights)
        pooled_loss = sum(
            loss * weight for loss, weight in zip(losses, sample_weights)
        ) / weight_total
        pooled_accuracy = sum(
            accuracy * weight for accuracy, weight in zip(accuracies, sample_weights)
        ) / weight_total
        drops = [
            100.0
            * (
                float(clean_by_model_fold[(str(member["model"]), int(member["fold"]))]["accuracy"])
                - float(member["accuracy"])
            )
            for member in members
        ]
        summary.append(
            {
                "config_id": members[0]["config_id"],
                "protocol_id": members[0]["protocol_id"],
                "model": key[0],
                "backend": members[0]["backend"],
                "split_seed": split_seed,
                "folds": len(members),
                "corruption": key[1],
                "amount_name": key[2],
                "amount": key[3],
                "jitter_max_ms": members[0].get("jitter_max_ms", 0.0),
                "taxel_fraction": members[0].get("taxel_fraction", 0.0),
                "loss_mean": statistics.mean(losses),
                "loss_std": statistics.stdev(losses) if len(losses) > 1 else "",
                "loss_pooled": pooled_loss,
                "accuracy_mean": statistics.mean(accuracies),
                "accuracy_std": (
                    statistics.stdev(accuracies) if len(accuracies) > 1 else ""
                ),
                "accuracy_pooled": pooled_accuracy,
                "participant_macro_accuracy_fold_mean": statistics.mean(
                    participant_macros
                ),
                "participant_macro_accuracy_pooled": sum(
                    value * int(member["validation_participants"])
                    for value, member in zip(participant_macros, members)
                )
                / sum(int(member["validation_participants"]) for member in members),
                "clean_accuracy_drop_pp_mean": statistics.mean(drops),
                "clean_accuracy_drop_pp_pooled": sum(
                    drop * weight for drop, weight in zip(drops, sample_weights)
                ) / weight_total,
                "activity_retention_mean": statistics.mean(retentions),
                "activity_retention_std": (
                    statistics.stdev(retentions) if len(retentions) > 1 else ""
                ),
                "activity_retention_pooled": (
                    sum(int(member["active_after"]) for member in members)
                    / sum(int(member["active_before"]) for member in members)
                    if sum(int(member["active_before"]) for member in members)
                    else 1.0
                ),
            }
        )
    return summary


def make_manifest(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    selected_indices: np.ndarray,
    labels_selected: np.ndarray,
    groups_selected: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    audits: list[dict[str, Any]],
    splitter_source: str,
    specs: list[dict[str, Any]],
    code_sha256: str,
    config_id: str,
    config_definition: dict[str, Any],
    model_backends: dict[str, str],
    environment: dict[str, Any],
    index_file_sha256: str,
    archive_provenance: dict[str, Any],
) -> dict[str, Any]:
    records = metadata["samples"]
    fold_entries: list[dict[str, Any]] = []
    for fold_index, ((train_local, validation_local), audit) in enumerate(
        zip(folds, audits), start=1
    ):
        train_global = selected_indices[train_local]
        validation_global = selected_indices[validation_local]
        fold_seed = args.seed + fold_index - 1
        fold_entries.append(
            {
                **audit,
                "fold_seed": fold_seed,
                "train_sample_indices": train_global.tolist(),
                "validation_sample_indices": validation_global.tolist(),
                "validation_sample_order": [
                    records[int(index)]["sample_id"] for index in validation_global
                ],
                "corruption_condition_seeds": {
                    f"{spec['corruption']}:{spec['amount']}": fold_seed
                    + int(spec["seed_offset"])
                    for spec in specs
                },
            }
        )
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "config_id": config_id,
        "config_definition": config_definition,
        "protocol_id": PROTOCOL_ID,
        "protocol_note": (
            "80-bin binary participant-group-CV robustness protocol; not the "
            "paper's 20-bin/count-input/sample-split reproduction protocol"
        ),
        "dataset": "STEMNIST",
        "dataset_root": metadata["dataset_root"],
        "source_archive_md5": metadata["source_archive_md5"],
        "source_archive_md5_status": metadata["source_archive_md5_status"],
        "source_archive_runtime_verification": archive_provenance,
        "args": vars(args),
        "environment": environment,
        "model_backends": model_backends,
        "cache": {
            "packed_file": metadata["packed_file"],
            "packed_sha256": metadata["packed_sha256"],
            "index_file_sha256": index_file_sha256,
            "cache_schema_version": metadata["cache_schema_version"],
            "critical_index_sha256_stored": metadata["critical_index_sha256"],
            "portable_critical_index_sha256": metadata[
                "portable_critical_index_sha256"
            ],
            "critical_index_hash_validation": metadata[
                "critical_index_hash_validation"
            ],
            "source": metadata["source"],
            "encoding": metadata["encoding"],
            "event_audit": metadata["event_audit"],
        },
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": code_sha256,
        },
        "split": {
            "splitter": splitter_source,
            "participant_disjoint": True,
            "split_seed": args.seed,
            "folds": args.folds,
            "selected_samples": int(len(selected_indices)),
            "selected_sample_indices": selected_indices.tolist(),
            "selected_participants": sorted(set(groups_selected.tolist())),
            "selected_label_counts": np.bincount(
                labels_selected, minlength=len(CLASS_NAMES)
            ).tolist(),
            "max_participants": args.max_participants,
            "max_samples_per_participant_class": args.max_samples_per_participant_class,
        },
        "corruption_protocol": {
            "conditions": specs,
            "sample_rng": (
                "per-sample CPU torch.Generator seeded by BLAKE2b64"
                "(condition_seed:sample_index:stream)"
            ),
            "severity_streams": "independent, non-nested",
            "time_jitter": (
                "each post-bin active occupancy cell receives an independent discrete "
                "uniform integer shift in [-steps,+steps], including zero; boundaries "
                "clip and collisions collapse to one active cell"
            ),
            "event_dropout": (
                "Bernoulli deletion of post-bin active occupancy cells, not raw HDF5 "
                "events; source binning collisions are recorded in cache.event_audit"
            ),
            "taxel_dropout": (
                "per-sample independently drawn physical taxel mask; the same taxel "
                "index is removed in both polarity channels"
            ),
            "activity_retention": "active_after / active_before, not accuracy retention",
        },
        "training_augmentation": {
            "mode": args.train_aug_mode,
            "clean_probability": args.train_aug_clean_prob,
            "event_dropout": args.train_event_dropout,
            "time_jitter_steps": args.train_time_jitter,
            "taxel_dropout": args.train_taxel_dropout,
            "weights": {
                "event": args.train_event_weight,
                "time": args.train_time_weight,
                "taxel": args.train_taxel_weight,
            },
        },
        "fold_entries": fold_entries,
    }


def environment_metadata(device: "torch.device", deterministic: bool) -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "snntorch": getattr(snn, "__version__", None),
        "snn_backend": "snntorch.Leaky" if snn is not None else "local_surrogate_lif",
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "deterministic_requested": deterministic,
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
    }


def prepare_cache_via_subprocess(args: argparse.Namespace) -> None:
    command = [
        str(Path(args.cache_builder_python).resolve()),
        str(Path(__file__).resolve()),
        "--prepare_cache_only",
        "--data_root",
        args.data_root,
        "--cache_dir",
        args.cache_dir,
        "--event_dt",
        str(args.event_dt),
        "--duration",
        str(args.duration),
    ]
    if args.rebuild_cache:
        command.append("--rebuild_cache")
    subprocess.run(command, check=True)


def prepare_runtime(args: argparse.Namespace) -> dict[str, Any]:
    if torch is None or nn is None:
        raise RuntimeError("Training requires PyTorch")
    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_dir)
    packed_path, index_path = default_cache_paths(
        cache_dir, args.event_dt, args.duration
    )
    if args.rebuild_cache or not packed_path.exists() or not index_path.exists():
        if args.cache_builder_python:
            prepare_cache_via_subprocess(args)
        else:
            build_packed_cache(
                data_root,
                packed_path,
                index_path,
                args.event_dt,
                args.duration,
                rebuild=args.rebuild_cache,
            )
    packed, metadata = load_packed_cache(
        packed_path,
        index_path,
        args.event_dt,
        args.duration,
        # Formal outputs are never based on an unchecked packed array.  The
        # CLI validation also rejects --no-verify_cache_hash for training.
        verify_hash=True,
    )
    archive_provenance = verify_source_archive(data_root)
    model_backends = validate_model_backends(args)
    # Configure cuBLAS/cuDNN before resolve_device/environment metadata can
    # initialize a CUDA context.
    configure_determinism(args.deterministic)
    device = resolve_device(args.device)
    environment = environment_metadata(device, args.deterministic)
    environment["model_backends"] = model_backends
    definition = config_payload(args, metadata, model_backends)
    config_id = make_config_id(definition)
    return {
        "data_root": data_root,
        "packed_path": packed_path,
        "index_path": index_path,
        "packed": packed,
        "metadata": metadata,
        "archive_provenance": archive_provenance,
        "index_file_sha256": sha256_file(index_path),
        "model_backends": model_backends,
        "device": device,
        "environment": environment,
        "config_definition": definition,
        "config_id": config_id,
        "code_sha256": sha256_file(Path(__file__)),
    }


def _format_seed_file(raw: str, config_id: str, seed: int) -> Path:
    try:
        formatted = raw.format(config_id=config_id, seed=seed)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Invalid output template {raw!r}; only {{config_id}} and {{seed}} are allowed"
        ) from exc
    path = Path(formatted)
    suffix_parts: list[str] = []
    if "{config_id}" not in raw:
        suffix_parts.append(config_id)
    if "{seed}" not in raw:
        suffix_parts.append(f"seed{seed}")
    if suffix_parts:
        suffix = "_" + "_".join(suffix_parts)
        path = path.with_name(path.stem + suffix + path.suffix)
    return path


def _format_checkpoint_dir(raw: str, config_id: str) -> Path:
    try:
        formatted = raw.format(config_id=config_id, seed="all")
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Invalid checkpoint template {raw!r}; only {{config_id}} and {{seed}} are allowed"
        ) from exc
    path = Path(formatted)
    return path if "{config_id}" in raw else path / config_id


def seed_output_paths(
    args: argparse.Namespace, config_id: str, seed: int
) -> dict[str, Path]:
    paths = {
        "results": _format_seed_file(args.results_csv, config_id, seed),
        "summary": _format_seed_file(args.summary_csv, config_id, seed),
        "predictions": _format_seed_file(args.predictions_csv, config_id, seed),
        "history": _format_seed_file(args.history_csv, config_id, seed),
        "manifest": _format_seed_file(args.manifest_json, config_id, seed),
        "json": _format_seed_file(args.json, config_id, seed),
        "checkpoint_dir": _format_checkpoint_dir(args.checkpoint_dir, config_id),
    }
    file_keys = ["results", "summary", "predictions", "history", "manifest", "json"]
    normalized = [os.path.normcase(str(paths[key].resolve())) for key in file_keys]
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "Resolved result/summary/prediction/history/manifest/JSON output paths "
            "must all be distinct"
        )
    return paths


def run_experiment(
    args: argparse.Namespace,
    runtime: dict[str, Any],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    packed = runtime["packed"]
    metadata = runtime["metadata"]
    packed_path = runtime["packed_path"]
    index_path = runtime["index_path"]
    model_backends = runtime["model_backends"]
    device = runtime["device"]
    config_id = runtime["config_id"]
    records = metadata["samples"]
    labels_global = np.asarray(
        [record["label_index"] for record in records], dtype=np.int64
    )
    groups_global = np.asarray(
        [record["participant"] for record in records], dtype=object
    )
    selected_indices = select_sample_indices(
        metadata,
        args.seed,
        args.max_participants,
        args.max_samples_per_participant_class,
    )
    labels_selected = labels_global[selected_indices]
    groups_selected = groups_global[selected_indices]
    folds, splitter_source = make_group_folds(
        labels_selected, groups_selected, args.folds, args.seed
    )
    audits = validate_group_folds(folds, labels_selected, groups_selected)
    executed_folds = len(folds) if args.max_folds <= 0 else min(args.max_folds, len(folds))
    executed_fold_pairs = folds[:executed_folds]
    executed_audits = audits[:executed_folds]
    specs = condition_specs(args)
    code_sha256 = runtime["code_sha256"]
    manifest = make_manifest(
        args,
        metadata,
        selected_indices,
        labels_selected,
        groups_selected,
        executed_fold_pairs,
        executed_audits,
        splitter_source,
        specs,
        code_sha256,
        config_id,
        runtime["config_definition"],
        model_backends,
        runtime["environment"],
        runtime["index_file_sha256"],
        runtime["archive_provenance"],
    )
    manifest["execution"] = {
        "status": "planned",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configured_folds": len(folds),
        "executed_folds": executed_folds,
        "models": list(args.models),
    }

    owned_files = [
        output_paths["results"],
        output_paths["summary"],
        output_paths["history"],
        output_paths["manifest"],
        output_paths["json"],
    ]
    if args.save_predictions:
        owned_files.append(output_paths["predictions"])
    checkpoint_paths = [
        output_paths["checkpoint_dir"]
        / f"{model_name}_splitseed{args.seed}_fold{fold_index}.pt"
        for fold_index in range(1, executed_folds + 1)
        for model_name in args.models
    ] if args.save_checkpoints else []
    preflight_files = owned_files + checkpoint_paths
    if not args.overwrite:
        existing = [str(path) for path in preflight_files if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing outputs; use --overwrite: "
                + ", ".join(existing)
            )
    else:
        # --overwrite authorizes replacing only the explicitly configured files.
        # Removing old terminal outputs up front prevents a failed rerun from
        # masquerading as the previous complete run.
        for path in dict.fromkeys(preflight_files):
            if path.is_file():
                path.unlink()
    write_json_atomic(output_paths["manifest"], manifest)
    if args.save_checkpoints:
        output_paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed, args.deterministic)
    print(
        f"Loaded packed STEMNIST: samples={len(selected_indices)}, "
        f"participants={len(set(groups_selected.tolist()))}, classes={len(CLASS_NAMES)}, "
        f"shape=({metadata['encoding']['time_bins']},{N_CHANNELS}), "
        f"device={device}, splitter={splitter_source}, config_id={config_id}"
    )

    prediction_handle = None
    prediction_writer = None
    prediction_temporary = output_paths["predictions"].with_name(
        output_paths["predictions"].name + ".tmp"
    )
    if args.save_predictions:
        output_paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
        if prediction_temporary.exists():
            if args.overwrite:
                prediction_temporary.unlink()
            else:
                raise FileExistsError(f"Prediction temp file exists: {prediction_temporary}")
        prediction_handle = prediction_temporary.open(
            "w", newline="", encoding="utf-8"
        )
        prediction_writer = csv.DictWriter(
            prediction_handle, fieldnames=PREDICTION_FIELDS
        )
        prediction_writer.writeheader()

    results: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    loss_fn = nn.CrossEntropyLoss()
    try:
        for fold_index, (train_local, validation_local) in enumerate(
            folds[:executed_folds], start=1
        ):
            fold_seed = args.seed + fold_index - 1
            train_global = selected_indices[train_local]
            validation_global = selected_indices[validation_local]
            audit = audits[fold_index - 1]
            for model_name in args.models:
                backend = model_backends[model_name]
                seed_everything(fold_seed, args.deterministic)
                model = make_model(model_name, args).to(device)
                parameter_count = sum(parameter.numel() for parameter in model.parameters())
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
                )
                started = time.time()
                for epoch in range(1, args.epochs + 1):
                    torch.manual_seed(fold_seed + 10_000 + epoch)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(fold_seed + 10_000 + epoch)
                    train_loader = make_loader(
                        packed,
                        labels_global,
                        train_global,
                        args,
                        shuffle=True,
                        seed=fold_seed + epoch,
                    )
                    train_loss, train_accuracy = train_one_epoch(
                        model,
                        train_loader,
                        optimizer,
                        loss_fn,
                        device,
                        args,
                        fold_seed,
                        epoch,
                    )
                    history.append(
                        {
                            "config_id": config_id,
                            "protocol_id": PROTOCOL_ID,
                            "model": model_name,
                            "backend": backend,
                            "split_seed": args.seed,
                            "fold": fold_index,
                            "fold_seed": fold_seed,
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "train_accuracy": train_accuracy,
                        }
                    )
                    if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
                        print(
                            f"{model_name} fold={fold_index}/{args.folds} "
                            f"epoch={epoch}/{args.epochs} loss={train_loss:.5f} "
                            f"acc={train_accuracy:.4f}",
                            flush=True,
                        )

                if args.save_checkpoints:
                    checkpoint_path = output_paths["checkpoint_dir"] / (
                        f"{model_name}_splitseed{args.seed}_fold{fold_index}.pt"
                    )
                    if checkpoint_path.exists() and not args.overwrite:
                        raise FileExistsError(
                            f"Refusing to overwrite checkpoint: {checkpoint_path}"
                        )
                    checkpoint = {
                        "config_id": config_id,
                        "config_definition": runtime["config_definition"],
                        "protocol_id": PROTOCOL_ID,
                        "model_name": model_name,
                        "backend": backend,
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                        "split_seed": args.seed,
                        "fold": fold_index,
                        "fold_seed": fold_seed,
                        "args": vars(args),
                        "meta": {
                            "dataset": "STEMNIST",
                            "samples": int(len(selected_indices)),
                            "time_steps": metadata["encoding"]["time_bins"],
                            "channels": N_CHANNELS,
                            "physical_taxels": N_TAXELS,
                            "classes": len(CLASS_NAMES),
                            "class_names": list(CLASS_NAMES),
                            "participant_disjoint": True,
                            "train_participants": audit["train_participants"],
                            "validation_participants": audit[
                                "validation_participants"
                            ],
                            "packed_sha256": metadata["packed_sha256"],
                            "portable_critical_index_sha256": metadata[
                                "portable_critical_index_sha256"
                            ],
                            "index_file_sha256": runtime["index_file_sha256"],
                            "code_sha256": code_sha256,
                        },
                    }
                    checkpoint_temporary = checkpoint_path.with_name(
                        checkpoint_path.name + ".tmp"
                    )
                    torch.save(checkpoint, checkpoint_temporary)
                    os.replace(checkpoint_temporary, checkpoint_path)
                    checkpoint_records.append(
                        {
                            "model": model_name,
                            "backend": backend,
                            "split_seed": args.seed,
                            "fold": fold_index,
                            "path": str(checkpoint_path.resolve()),
                            "sha256": sha256_file(checkpoint_path),
                        }
                    )

                condition_results: list[dict[str, Any]] = []
                for spec in specs:
                    validation_loader = make_loader(
                        packed,
                        labels_global,
                        validation_global,
                        args,
                        shuffle=False,
                        seed=fold_seed,
                    )
                    metrics = evaluate_condition(
                        model,
                        validation_loader,
                        loss_fn,
                        device,
                        spec,
                        fold_seed,
                        model_name,
                        backend,
                        config_id,
                        args.seed,
                        fold_index,
                        records,
                        prediction_writer,
                    )
                    row = {
                        "config_id": config_id,
                        "protocol_id": PROTOCOL_ID,
                        "model": model_name,
                        "backend": backend,
                        "split_seed": args.seed,
                        "fold": fold_index,
                        "fold_seed": fold_seed,
                        "splitter": splitter_source,
                        "train_samples": len(train_global),
                        "validation_samples": len(validation_global),
                        "train_participants": len(audit["train_participants"]),
                        "validation_participants": len(
                            audit["validation_participants"]
                        ),
                        "epochs": args.epochs,
                        "parameters": parameter_count,
                        "corruption": spec["corruption"],
                        "amount_name": spec["amount_name"],
                        "amount": spec["amount"],
                        "jitter_max_ms": spec["jitter_max_ms"],
                        "taxel_fraction": spec["taxel_fraction"],
                        **metrics,
                    }
                    results.append(row)
                    condition_results.append(row)
                    print(
                        f"{model_name} fold={fold_index} {spec['corruption']}="
                        f"{spec['amount']} accuracy={metrics['accuracy']:.4f} "
                        f"activity_retention={metrics['activity_retention']:.4f}",
                        flush=True,
                    )
                    if prediction_handle is not None:
                        prediction_handle.flush()

                corrupted_rows = [
                    row for row in condition_results if row["corruption"] != "clean"
                ]
                robust_rows = [
                    row
                    for row in corrupted_rows
                    if (
                        (
                            row["corruption"] == "event_dropout"
                            and math.isclose(
                                float(row["amount"]),
                                0.3,
                                rel_tol=0.0,
                                abs_tol=1e-12,
                            )
                        )
                        or (row["corruption"] == "time_jitter" and int(row["amount"]) == 3)
                        or (row["corruption"] == "taxel_dropout" and int(row["amount"]) == 43)
                    )
                ]
                if len(robust_rows) != 3:
                    raise AssertionError(
                        "RobustAvg requires exactly event_dropout=0.3, time_jitter=3, "
                        "and taxel_dropout=43"
                    )
                derived_rows: dict[str, dict[str, Any]] = {}
                for derived_name, amount_name, members in (
                    (
                        "RobustAvg",
                        "fixed_event0.3_jitter3_taxel43",
                        robust_rows,
                    ),
                    ("AllCorruptAvg", "mean_of_all_corruptions", corrupted_rows),
                ):
                    derived_row = {
                        "config_id": config_id,
                        "protocol_id": PROTOCOL_ID,
                        "model": model_name,
                        "backend": backend,
                        "split_seed": args.seed,
                        "fold": fold_index,
                        "fold_seed": fold_seed,
                        "splitter": splitter_source,
                        "train_samples": len(train_global),
                        "validation_samples": len(validation_global),
                        "train_participants": len(audit["train_participants"]),
                        "validation_participants": len(audit["validation_participants"]),
                        "epochs": args.epochs,
                        "parameters": parameter_count,
                        "corruption": derived_name,
                        "amount_name": amount_name,
                        "amount": len(members),
                        "jitter_max_ms": "",
                        "taxel_fraction": "",
                        "loss": statistics.mean(float(row["loss"]) for row in members),
                        "accuracy": statistics.mean(
                            float(row["accuracy"]) for row in members
                        ),
                        "participant_macro_accuracy": statistics.mean(
                            float(row["participant_macro_accuracy"]) for row in members
                        ),
                        "active_before": sum(int(row["active_before"]) for row in members),
                        "active_after": sum(int(row["active_after"]) for row in members),
                        "activity_retention": statistics.mean(
                            float(row["activity_retention"]) for row in members
                        ),
                    }
                    results.append(derived_row)
                    derived_rows[derived_name] = derived_row
                print(
                    f"Completed {model_name} fold={fold_index} in "
                    f"{time.time() - started:.1f}s, RobustAvg="
                    f"{derived_rows['RobustAvg']['accuracy']:.4f}, AllCorruptAvg="
                    f"{derived_rows['AllCorruptAvg']['accuracy']:.4f}",
                    flush=True,
                )
                write_csv_atomic(output_paths["results"], results, RESULT_FIELDS)
                write_csv_atomic(output_paths["history"], history, HISTORY_FIELDS)
    finally:
        if prediction_handle is not None:
            prediction_handle.close()

    if args.save_predictions:
        os.replace(prediction_temporary, output_paths["predictions"])
    summary = summarize_fold_results(results, args.seed)
    write_csv_atomic(output_paths["summary"], summary, SUMMARY_FIELDS)
    write_csv_atomic(output_paths["results"], results, RESULT_FIELDS)
    write_csv_atomic(output_paths["history"], history, HISTORY_FIELDS)
    reported_outputs: dict[str, Any] = {
        key: str(output_paths[key].resolve())
        for key in ("results", "summary", "history", "manifest", "json")
    }
    if args.save_predictions:
        reported_outputs["predictions"] = str(output_paths["predictions"].resolve())
    if args.save_checkpoints:
        reported_outputs["checkpoints"] = checkpoint_records
    final_payload = {
        "schema_version": 2,
        "status": "complete",
        "config_id": config_id,
        "config_definition": runtime["config_definition"],
        "protocol_id": PROTOCOL_ID,
        "args": vars(args),
        "environment": runtime["environment"],
        "source_archive_runtime_verification": runtime["archive_provenance"],
        "cache": {
            "packed_file": str(packed_path.resolve()),
            "index_file": str(index_path.resolve()),
            "packed_sha256": metadata["packed_sha256"],
            "critical_index_sha256_stored": metadata["critical_index_sha256"],
            "portable_critical_index_sha256": metadata[
                "portable_critical_index_sha256"
            ],
            "index_file_sha256": runtime["index_file_sha256"],
            "event_audit": metadata["event_audit"],
        },
        "split": {
            "splitter": splitter_source,
            "participant_disjoint": True,
            "selected_samples": len(selected_indices),
            "selected_participants": len(set(groups_selected.tolist())),
            "configured_folds": args.folds,
            "executed_folds": executed_folds,
        },
        "outputs": reported_outputs,
        "summary": summary,
    }
    write_json_atomic(output_paths["json"], final_payload)
    manifest["execution"].update(
        {
            "status": "complete",
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "result_rows": len(results),
            "history_rows": len(history),
            "prediction_rows": (
                sum(
                    int(row["validation_samples"])
                    for row in results
                    if row["corruption"] not in {"RobustAvg", "AllCorruptAvg"}
                )
                if args.save_predictions
                else 0
            ),
            "outputs": reported_outputs,
        }
    )
    write_json_atomic(output_paths["manifest"], manifest)
    return {
        "config_id": config_id,
        "split_seed": args.seed,
        "results": results,
        "summary": summary,
        "outputs": reported_outputs,
    }


CROSSSEED_SUMMARY_FIELDS = [
    "config_id",
    "protocol_id",
    "model",
    "backend",
    "seeds",
    "seed_count",
    "folds_per_seed",
    "corruption",
    "amount_name",
    "amount",
    "jitter_max_ms",
    "taxel_fraction",
    "loss_seed_mean",
    "loss_seed_std",
    "accuracy_seed_mean",
    "accuracy_seed_std",
    "participant_macro_accuracy_seed_mean",
    "participant_macro_accuracy_seed_std",
    "clean_accuracy_drop_pp_seed_mean",
    "clean_accuracy_drop_pp_seed_std",
    "activity_retention_seed_mean",
    "activity_retention_seed_std",
]


def _sample_std_or_blank(values: list[float]) -> float | str:
    return statistics.stdev(values) if len(values) > 1 else ""


def summarize_across_seeds(
    per_seed_summaries: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for seed_summary in per_seed_summaries:
        for row in seed_summary:
            key = (
                str(row["config_id"]),
                str(row["protocol_id"]),
                str(row["model"]),
                str(row["backend"]),
                str(row["corruption"]),
                str(row["amount_name"]),
                str(row["amount"]),
            )
            grouped[key].append(row)
    output: list[dict[str, Any]] = []
    for key, members in sorted(grouped.items()):
        seeds = [int(member["split_seed"]) for member in members]
        if len(set(seeds)) != len(seeds):
            raise AssertionError(f"Duplicate split seed in cross-seed summary: {seeds}")
        losses = [float(member["loss_pooled"]) for member in members]
        accuracies = [float(member["accuracy_pooled"]) for member in members]
        participant_macros = [
            float(member["participant_macro_accuracy_pooled"]) for member in members
        ]
        drops = [float(member["clean_accuracy_drop_pp_pooled"]) for member in members]
        retentions = [float(member["activity_retention_pooled"]) for member in members]
        output.append(
            {
                "config_id": key[0],
                "protocol_id": key[1],
                "model": key[2],
                "backend": key[3],
                "seeds": ";".join(str(seed) for seed in sorted(seeds)),
                "seed_count": len(seeds),
                "folds_per_seed": ";".join(
                    str(int(member["folds"])) for member in members
                ),
                "corruption": key[4],
                "amount_name": key[5],
                "amount": key[6],
                "jitter_max_ms": members[0]["jitter_max_ms"],
                "taxel_fraction": members[0]["taxel_fraction"],
                "loss_seed_mean": statistics.mean(losses),
                "loss_seed_std": _sample_std_or_blank(losses),
                "accuracy_seed_mean": statistics.mean(accuracies),
                "accuracy_seed_std": _sample_std_or_blank(accuracies),
                "participant_macro_accuracy_seed_mean": statistics.mean(
                    participant_macros
                ),
                "participant_macro_accuracy_seed_std": _sample_std_or_blank(
                    participant_macros
                ),
                "clean_accuracy_drop_pp_seed_mean": statistics.mean(drops),
                "clean_accuracy_drop_pp_seed_std": _sample_std_or_blank(drops),
                "activity_retention_seed_mean": statistics.mean(retentions),
                "activity_retention_seed_std": _sample_std_or_blank(retentions),
            }
        )
    return output


def _format_crossseed_file(raw: str, config_id: str, seeds: Sequence[int]) -> Path:
    seed_set_id = "seeds" + "-".join(str(seed) for seed in sorted(seeds))
    try:
        formatted = raw.format(
            config_id=config_id, seed=f"{seed_set_id}_crossseed"
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Invalid output template {raw!r}; only {{config_id}} and {{seed}} are allowed"
        ) from exc
    path = Path(formatted)
    suffix_parts: list[str] = []
    if "{config_id}" not in raw:
        suffix_parts.append(config_id)
    if "{seed}" not in raw:
        suffix_parts.extend((seed_set_id, "crossseed"))
    if suffix_parts:
        path = path.with_name(
            path.stem + "_" + "_".join(suffix_parts) + path.suffix
        )
    return path


def run_experiments(args: argparse.Namespace) -> dict[str, Any]:
    runtime = prepare_runtime(args)
    config_id = runtime["config_id"]
    seed_jobs: list[tuple[argparse.Namespace, dict[str, Path]]] = []
    all_targets: list[Path] = []
    for seed in args.seeds:
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = int(seed)
        paths = seed_output_paths(seed_args, config_id, int(seed))
        seed_jobs.append((seed_args, paths))
        for key in ("results", "summary", "history", "manifest", "json"):
            all_targets.append(paths[key])
        if args.save_predictions:
            all_targets.append(paths["predictions"])
        if args.save_checkpoints:
            executed_folds = args.folds if args.max_folds <= 0 else min(
                args.max_folds, args.folds
            )
            all_targets.extend(
                paths["checkpoint_dir"]
                / f"{model_name}_splitseed{seed}_fold{fold}.pt"
                for fold in range(1, executed_folds + 1)
                for model_name in args.models
            )

    crossseed_summary_path = _format_crossseed_file(
        args.summary_csv, config_id, args.seeds
    )
    crossseed_json_path = _format_crossseed_file(args.json, config_id, args.seeds)
    all_targets.extend((crossseed_summary_path, crossseed_json_path))
    normalized = [os.path.normcase(str(path.resolve())) for path in all_targets]
    if len(set(normalized)) != len(normalized):
        duplicates = [
            path
            for path, count in Counter(normalized).items()
            if count > 1
        ]
        raise ValueError(f"Output/checkpoint paths collide: {duplicates}")
    existing = [str(path) for path in all_targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite any seed/config output; use --overwrite only to "
            "replace this exact config/seed set: " + ", ".join(existing)
        )
    if args.overwrite:
        for path in (crossseed_summary_path, crossseed_json_path):
            if path.is_file():
                path.unlink()

    run_outputs: list[dict[str, Any]] = []
    for seed_args, paths in seed_jobs:
        run_outputs.append(run_experiment(seed_args, runtime, paths))
    crossseed_summary = summarize_across_seeds(
        [entry["summary"] for entry in run_outputs]
    )
    write_csv_atomic(
        crossseed_summary_path, crossseed_summary, CROSSSEED_SUMMARY_FIELDS
    )
    crossseed_payload = {
        "schema_version": 1,
        "status": "complete",
        "config_id": config_id,
        "protocol_id": PROTOCOL_ID,
        "seeds": list(args.seeds),
        "config_definition": runtime["config_definition"],
        "environment": runtime["environment"],
        "source_archive_runtime_verification": runtime["archive_provenance"],
        "cache": {
            "packed_sha256": runtime["metadata"]["packed_sha256"],
            "portable_critical_index_sha256": runtime["metadata"][
                "portable_critical_index_sha256"
            ],
            "index_file_sha256": runtime["index_file_sha256"],
        },
        "seed_runs": [entry["outputs"] for entry in run_outputs],
        "crossseed_summary_csv": str(crossseed_summary_path.resolve()),
        "summary": crossseed_summary,
    }
    write_json_atomic(crossseed_json_path, crossseed_payload)
    print(
        f"Completed config_id={config_id}, seeds={list(args.seeds)}; "
        f"cross-seed summary={crossseed_summary_path}",
        flush=True,
    )
    return crossseed_payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    default_data_root = (
        project_root / "STEMNIST_data" / "extracted" / "STEMNIST Dataset"
    )
    default_cache_dir = project_root / "STEMNIST_data" / "cache"
    parser = argparse.ArgumentParser(
        description="Participant-disjoint STEMNIST corruption robustness runner."
    )
    parser.add_argument("--data_root", default=str(default_data_root))
    parser.add_argument("--cache_dir", default=str(default_cache_dir))
    parser.add_argument("--prepare_cache_only", action="store_true")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument(
        "--cache_builder_python",
        default="",
        help="Optional compatible Python used to prepare a missing HDF5 cache.",
    )
    parser.add_argument("--verify_cache_hash", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--event_dt", type=float, default=0.025)
    parser.add_argument("--duration", type=float, default=2.0)

    parser.add_argument("--models", nargs="+", choices=["paper_scnn", "tcn"], default=["paper_scnn", "tcn"])
    parser.add_argument(
        "--allow_local_lif",
        action="store_true",
        help=(
            "Explicitly permit the non-paper local surrogate LIF when snnTorch is "
            "unavailable. Formal paper_scnn runs otherwise fail fast."
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Backward-compatible single split seed.",
    )
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="One or more independently split experiment seeds, e.g. 42 123 202.",
    )
    parser.add_argument("--max_folds", type=int, default=0)
    parser.add_argument("--max_participants", type=int, default=0)
    parser.add_argument("--max_samples_per_participant_class", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--threshold_v", type=float, default=1.0)
    parser.add_argument(
        "--snn_readout",
        choices=["spikes", "membrane_sum", "membrane_final"],
        default="spikes",
    )

    parser.add_argument("--event_dropout", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--time_jitter", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--taxel_dropout", type=int, nargs="+", default=[21, 43, 85])
    parser.add_argument(
        "--train_aug_mode",
        choices=["none", "sample_one_sample"],
        default="none",
    )
    parser.add_argument("--train_aug_clean_prob", type=float, default=0.25)
    parser.add_argument("--train_event_dropout", type=float, default=0.3)
    parser.add_argument("--train_time_jitter", type=int, default=1)
    parser.add_argument("--train_taxel_dropout", type=int, default=43)
    parser.add_argument("--train_event_weight", type=float, default=1.0)
    parser.add_argument("--train_time_weight", type=float, default=3.0)
    parser.add_argument("--train_taxel_weight", type=float, default=1.0)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--results_csv", default=str(project_root / "stemnist_robustness_results.csv"))
    parser.add_argument("--summary_csv", default=str(project_root / "stemnist_robustness_summary.csv"))
    parser.add_argument("--predictions_csv", default=str(project_root / "stemnist_robustness_predictions.csv"))
    parser.add_argument("--history_csv", default=str(project_root / "stemnist_robustness_history.csv"))
    parser.add_argument("--checkpoint_dir", default=str(project_root / "stemnist_robustness_checkpoints"))
    parser.add_argument("--manifest_json", default=str(project_root / "stemnist_robustness_manifest.json"))
    parser.add_argument("--json", default=str(project_root / "stemnist_robustness_results.json"))
    args = parser.parse_args(argv)

    if args.seeds is None:
        args.seeds = [42 if args.seed is None else int(args.seed)]
    args.seed = int(args.seeds[0])
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not contain duplicates")

    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if not math.isclose(args.event_dt, 0.025, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(
        args.duration, 2.0, rel_tol=0.0, abs_tol=1e-12
    ):
        parser.error(
            f"{PROTOCOL_ID} requires --event_dt 0.025 and --duration 2.0; "
            "use a separately named protocol for other temporal geometry"
        )
    if args.max_folds < 0:
        parser.error("--max_folds must be non-negative")
    if args.max_participants == 1 or args.max_participants < 0:
        parser.error("--max_participants must be zero or at least 2")
    if args.max_participants and args.max_participants < args.folds:
        parser.error("--max_participants must be at least --folds")
    if args.max_samples_per_participant_class < 0:
        parser.error("--max_samples_per_participant_class must be non-negative")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.weight_decay < 0:
        parser.error("--weight_decay must be non-negative")
    if args.hidden < 1 or args.layers < 1:
        parser.error("--hidden and --layers must be positive")
    if not 0 <= args.dropout <= 1:
        parser.error("--dropout must lie in [0,1]")
    if not 0 <= args.beta <= 1:
        parser.error("--beta must lie in [0,1]")
    if args.threshold_v <= 0:
        parser.error("--threshold_v must be positive")
    if args.num_workers < 0:
        parser.error("--num_workers must be non-negative")
    if args.log_every < 1:
        parser.error("--log_every must be positive")
    if not 0 <= args.train_aug_clean_prob <= 1:
        parser.error("--train_aug_clean_prob must lie in [0,1]")
    if not 0 <= args.train_event_dropout <= 1:
        parser.error("--train_event_dropout must lie in [0,1]")
    if args.train_time_jitter < 0:
        parser.error("--train_time_jitter must be non-negative")
    if not 0 <= args.train_taxel_dropout <= N_TAXELS:
        parser.error(f"--train_taxel_dropout must lie in [0,{N_TAXELS}]")
    if any(
        weight < 0
        for weight in (
            args.train_event_weight,
            args.train_time_weight,
            args.train_taxel_weight,
        )
    ):
        parser.error("training augmentation weights must be non-negative")
    if args.train_aug_mode == "sample_one_sample" and not any(
        weight > 0
        for weight in (
            args.train_event_weight,
            args.train_time_weight,
            args.train_taxel_weight,
        )
    ):
        parser.error("sample_one_sample needs at least one positive augmentation weight")
    if any(not 0 <= value <= 1 for value in args.event_dropout):
        parser.error("all --event_dropout values must lie in [0,1]")
    if any(value < 0 for value in args.time_jitter):
        parser.error("all --time_jitter values must be non-negative")
    if any(not 0 <= value <= N_TAXELS for value in args.taxel_dropout):
        parser.error(f"all --taxel_dropout values must lie in [0,{N_TAXELS}]")
    if len(set(args.event_dropout)) != len(args.event_dropout):
        parser.error("--event_dropout must not contain duplicate severities")
    if len(set(args.time_jitter)) != len(args.time_jitter):
        parser.error("--time_jitter must not contain duplicate severities")
    if len(set(args.taxel_dropout)) != len(args.taxel_dropout):
        parser.error("--taxel_dropout must not contain duplicate severities")
    robust_event_matches = sum(
        math.isclose(float(value), 0.3, rel_tol=0.0, abs_tol=1e-12)
        for value in args.event_dropout
    )
    if robust_event_matches != 1:
        parser.error(
            "--event_dropout must include exactly one 0.3 condition for fixed RobustAvg"
        )
    if 3 not in args.time_jitter:
        parser.error("--time_jitter must include 3 for fixed RobustAvg")
    if 43 not in args.taxel_dropout:
        parser.error("--taxel_dropout must include 43 for fixed RobustAvg")
    if args.train_aug_mode == "sample_one_sample":
        enabled = (
            args.train_event_weight > 0 and args.train_event_dropout > 0,
            args.train_time_weight > 0 and args.train_time_jitter > 0,
            args.train_taxel_weight > 0 and args.train_taxel_dropout > 0,
        )
        if not any(enabled):
            parser.error(
                "sample_one_sample needs a positive weight attached to a non-zero "
                "augmentation amount"
            )
    raw_output_paths = [
        args.results_csv,
        args.summary_csv,
        args.predictions_csv,
        args.history_csv,
        args.manifest_json,
        args.json,
    ]
    normalized_output_paths = {
        os.path.normcase(str(Path(path))) for path in raw_output_paths
    }
    if len(normalized_output_paths) != len(raw_output_paths):
        parser.error("all configured output file paths must be distinct")
    if not args.prepare_cache_only and not args.verify_cache_hash:
        parser.error("training/evaluation requires --verify_cache_hash")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_dir)
    packed_path, index_path = default_cache_paths(
        cache_dir, args.event_dt, args.duration
    )
    if args.prepare_cache_only:
        metadata = build_packed_cache(
            data_root,
            packed_path,
            index_path,
            args.event_dt,
            args.duration,
            rebuild=args.rebuild_cache,
        )
        print(
            json.dumps(
                {
                    "status": "cache_ready",
                    "packed_file": metadata["packed_file"],
                    "index_file": str(index_path.resolve()),
                    "packed_sha256": metadata["packed_sha256"],
                    "portable_critical_index_sha256": metadata.get(
                        "portable_critical_index_sha256",
                        critical_cache_index_sha256(metadata),
                    ),
                    "index_file_sha256": sha256_file(index_path),
                    "samples": len(metadata["samples"]),
                    "participants": len(metadata["participants"]),
                    "event_audit": metadata["event_audit"],
                },
                indent=2,
            )
        )
        return
    run_experiments(args)


if __name__ == "__main__":
    main()
