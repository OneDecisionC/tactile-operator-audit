"""Evaluation-only ICASSP jitter/collision screening on frozen checkpoints.

This runner is intentionally independent from all training runners and writes only
to a new, explicitly selected output directory.  It evaluates the existing formal
Braille and STEMNIST checkpoints under six conditions:

``clean``, raw/pre-bin jitter, collision-prone post-bin jitter (PBJ),
collision-free jitter (CF), PBJ-loss-matched dropout, and CF plus the same matched
dropout.  Matched controls are required to reproduce the PBJ active-cell count for
every sample and channel, not merely in aggregate.

The ``cache-build`` subcommand deliberately imports neither torch nor snnTorch.
It can therefore run with the base conda Python that has a working h5py build; the
portable NumPy cache can subsequently be evaluated in the STAtten environment.

The operator module is loaded lazily at evaluation time.  Its supported contract
is documented in :class:`OperatorAdapter`; no operator implementation is copied
into this screening runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import pickle
import platform
import re
import shutil
import string
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


SCRIPT_SCHEMA_VERSION = 2
PERSISTENCE_SCHEMA_VERSION = 2
PROTOCOL_ID = "icassp_jitter_deconfounding_screen_v1"
STEM_CACHE_SCHEMA_VERSION = 1
CLASS_NAMES_STEM = tuple(string.ascii_uppercase) + tuple("123456789")
CLASS_TO_INDEX_STEM = {name: index for index, name in enumerate(CLASS_NAMES_STEM)}
STEM_FILENAME_RE = re.compile(
    r"^(?P<participant>[A-Z]{2})_(?P<label>[A-Z1-9])_(?P<repetition>[0-9]+)_spikes[.]h5$"
)
STEM_CHANNELS = 512
STEM_TAXELS = 256
STEM_DURATION_SECONDS = 2.0
STEM_SAMPLING_HZ = 120
EVENT_DT_SECONDS = 0.025
BRAILLE_DURATION_SECONDS = 1.3
BRAILLE_CROP_START = 7
BRAILLE_CROP_END = -1
DEFAULT_SEEDS = (42, 123, 202)
DEFAULT_SEVERITIES_MS = (25, 50, 75)
DEFAULT_CONDITIONS = ("clean", "prebin", "pbj", "cf", "matched", "cf_matched")
DEFAULT_MODEL_IDS = (
    "braille_snn_clean",
    "braille_tcn_samplemix",
    "stemnist_paper_scnn_clean",
    "stemnist_tcn_large_clean",
)
FROZEN_OPERATORS = ("prebin", "pbj", "cf", "matched", "cf_matched")
POSTBIN_CONDITIONS = frozenset(("pbj", "cf", "matched", "cf_matched"))


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, path)


def write_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8, person=b"icasjitr").digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF_FFFF_FFFF


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "packed": cache_dir / "stemnist_icassp_clean25ms_packed.npy",
        "offsets": cache_dir / "stemnist_icassp_raw_offsets.npy",
        "ticks": cache_dir / "stemnist_icassp_raw_ticks120hz.npy",
        "channels": cache_dir / "stemnist_icassp_raw_channels.npy",
        "index": cache_dir / "stemnist_icassp_event_cache_index.json",
    }


def enumerate_stemnist_records(data_root: Path) -> list[dict[str, Any]]:
    processed_root = data_root / "ProcessedSpikes"
    if not processed_root.is_dir():
        raise FileNotFoundError(f"Missing ProcessedSpikes: {processed_root}")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for label in CLASS_NAMES_STEM:
        class_dir = processed_root / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for path in class_dir.glob("*.h5"):
            match = STEM_FILENAME_RE.fullmatch(path.name)
            if match is None:
                raise ValueError(f"Unexpected STEMNIST filename: {path}")
            fields = match.groupdict()
            if fields["label"] != label:
                raise ValueError(f"Label/path mismatch: {path}")
            repetition = int(fields["repetition"])
            sample_id = f"{fields['participant']}_{label}_{repetition}"
            if sample_id in seen:
                raise ValueError(f"Duplicate sample: {sample_id}")
            seen.add(sample_id)
            records.append(
                {
                    "sample_id": sample_id,
                    "participant": fields["participant"],
                    "label": label,
                    "label_index": CLASS_TO_INDEX_STEM[label],
                    "repetition": repetition,
                    "processed_path": path.relative_to(data_root).as_posix(),
                    "processed_bytes": path.stat().st_size,
                }
            )
    records.sort(
        key=lambda row: (row["label_index"], row["participant"], row["repetition"])
    )
    for index, record in enumerate(records):
        record["sample_index"] = index
    discovered = {path.resolve() for path in processed_root.rglob("*.h5")}
    indexed = {(data_root / row["processed_path"]).resolve() for row in records}
    if discovered != indexed:
        raise ValueError("ProcessedSpikes contains unindexed or missing files")
    counts = Counter(row["label"] for row in records)
    if len(records) != 7700 or any(counts[name] != 220 for name in CLASS_NAMES_STEM):
        raise ValueError(f"Unexpected official STEMNIST cardinality: {len(records)}")
    return records


def build_stemnist_cache(args: argparse.Namespace) -> None:
    """Build raw-event and clean-occupancy cache without importing torch."""
    try:
        import h5py
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "cache-build requires a working h5py; use the base conda Python"
        ) from exc

    data_root = Path(args.data_root).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    paths = cache_paths(cache_dir)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.rebuild_cache:
        if len(existing) == len(paths):
            metadata = validate_stemnist_cache(cache_dir, verify_hash=True)
            print(
                f"Cache already complete and verified: {cache_dir} "
                f"({metadata['samples']} samples)",
                flush=True,
            )
            return
        raise FileExistsError(
            "Incomplete cache exists; pass --rebuild-cache to replace only this "
            f"dedicated cache: {[str(path) for path in existing]}"
        )
    if args.rebuild_cache:
        for path in existing:
            path.unlink()

    records = enumerate_stemnist_records(data_root)
    packed = np.zeros((len(records), 80, STEM_CHANNELS // 8), dtype=np.uint8)
    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    tick_chunks: list[np.ndarray] = []
    channel_chunks: list[np.ndarray] = []
    total_binary = 0
    maximum_tick_error = 0.0
    started = time.time()
    for sample_index, record in enumerate(records):
        path = data_root / record["processed_path"]
        with h5py.File(path, "r") as handle:
            if "spikes" not in handle:
                raise KeyError(f"Missing spikes dataset: {path}")
            spikes = handle["spikes"][:]
            sampling_hz = int(handle.attrs.get("sampling_rate", STEM_SAMPLING_HZ))
            duration = float(handle.attrs.get("duration", STEM_DURATION_SECONDS))
        if spikes.dtype.names != ("timestamp", "taxel_id", "polarity"):
            raise ValueError(f"Unexpected compound dtype in {path}: {spikes.dtype}")
        if sampling_hz != STEM_SAMPLING_HZ or not math.isclose(
            duration, STEM_DURATION_SECONDS, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(f"Unexpected temporal geometry in {path}")
        timestamps = np.asarray(spikes["timestamp"], dtype=np.float64)
        ticks_float = timestamps * STEM_SAMPLING_HZ
        ticks = np.rint(ticks_float).astype(np.int16)
        tick_error = float(np.max(np.abs(ticks_float - ticks), initial=0.0))
        maximum_tick_error = max(maximum_tick_error, tick_error)
        if tick_error > 1e-3 or np.any(ticks < 0) or np.any(ticks > 240):
            raise ValueError(f"Invalid 120-Hz timestamp grid in {path}")
        taxels = np.asarray(spikes["taxel_id"], dtype=np.int64)
        polarities = np.asarray(spikes["polarity"], dtype=np.int64)
        if np.any((taxels < 1) | (taxels > 256)) or np.any(
            ~np.isin(polarities, (-1, 1))
        ):
            raise ValueError(f"Invalid channel identity in {path}")
        channels = (taxels - 1 + (polarities < 0) * STEM_TAXELS).astype(np.uint16)
        bins = np.clip(ticks.astype(np.int64) // 3, 0, 79)
        binary = np.zeros((80, STEM_CHANNELS), dtype=np.uint8)
        binary[bins, channels] = 1
        packed[sample_index] = np.packbits(binary, axis=-1, bitorder="little")
        offsets[sample_index + 1] = offsets[sample_index] + len(ticks)
        tick_chunks.append(ticks)
        channel_chunks.append(channels)
        binary_count = int(binary.sum())
        total_binary += binary_count
        record["raw_spike_count"] = int(len(ticks))
        record["binary_event_count"] = binary_count
        record["bin_collision_count"] = int(len(ticks) - binary_count)
        if (sample_index + 1) % 500 == 0 or sample_index + 1 == len(records):
            print(
                f"cache-build {sample_index + 1}/{len(records)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    ticks_all = np.concatenate(tick_chunks).astype(np.int16, copy=False)
    channels_all = np.concatenate(channel_chunks).astype(np.uint16, copy=False)
    if int(offsets[-1]) != len(ticks_all) or len(ticks_all) != len(channels_all):
        raise AssertionError("Ragged event cache offsets are inconsistent")

    cache_dir.mkdir(parents=True, exist_ok=True)
    write_npy_atomic(paths["packed"], packed)
    write_npy_atomic(paths["offsets"], offsets)
    write_npy_atomic(paths["ticks"], ticks_all)
    write_npy_atomic(paths["channels"], channels_all)
    file_hashes = {
        name: sha256_file(path)
        for name, path in paths.items()
        if name != "index"
    }
    metadata = {
        "cache_schema_version": STEM_CACHE_SCHEMA_VERSION,
        "created_utc": utc_now(),
        "dataset": "stemnist",
        "dataset_root": str(data_root),
        "samples": len(records),
        "raw_events": int(len(ticks_all)),
        "binary_events": int(total_binary),
        "maximum_source_tick_error": maximum_tick_error,
        "temporal_geometry": {
            "source_sampling_hz": STEM_SAMPLING_HZ,
            "duration_seconds": STEM_DURATION_SECONDS,
            "event_dt_seconds": EVENT_DT_SECONDS,
            "time_bins": 80,
            "channels": STEM_CHANNELS,
            "tick_to_clean_bin": "clip(tick // 3, 0, 79)",
        },
        "array_shapes": {
            "packed": list(packed.shape),
            "offsets": list(offsets.shape),
            "ticks": list(ticks_all.shape),
            "channels": list(channels_all.shape),
        },
        "array_dtypes": {
            "packed": str(packed.dtype),
            "offsets": str(offsets.dtype),
            "ticks": str(ticks_all.dtype),
            "channels": str(channels_all.dtype),
        },
        "files": {name: path.name for name, path in paths.items()},
        "sha256": file_hashes,
        "records": records,
    }
    metadata["critical_index_sha256"] = sha256_json(
        {key: value for key, value in metadata.items() if key != "dataset_root"}
    )
    write_json_atomic(paths["index"], metadata)
    print(
        f"Portable STEMNIST event cache written: {cache_dir}; "
        f"events={len(ticks_all)}, packed_shape={packed.shape}",
        flush=True,
    )


def validate_stemnist_cache(
    cache_dir: Path, verify_hash: bool
) -> dict[str, Any]:
    paths = cache_paths(cache_dir)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete STEMNIST ICASSP cache: {missing}")
    metadata = json.loads(paths["index"].read_text(encoding="utf-8"))
    if metadata.get("cache_schema_version") != STEM_CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported STEMNIST ICASSP cache schema")
    if metadata.get("samples") != 7700:
        raise ValueError("STEMNIST cache sample count mismatch")
    critical_expected = metadata.get("critical_index_sha256")
    critical_payload = {
        key: value
        for key, value in metadata.items()
        if key not in {"dataset_root", "critical_index_sha256"}
    }
    critical_actual = sha256_json(critical_payload)
    if critical_expected != critical_actual:
        raise ValueError(
            "STEMNIST cache critical index SHA-256 mismatch: "
            f"{critical_actual} != {critical_expected}"
        )
    if verify_hash:
        for name in ("packed", "offsets", "ticks", "channels"):
            actual = sha256_file(paths[name])
            expected = metadata["sha256"][name]
            if actual != expected:
                raise ValueError(f"Cache SHA-256 mismatch for {name}: {actual} != {expected}")
    arrays = {
        name: np.load(paths[name], mmap_mode="r", allow_pickle=False)
        for name in ("packed", "offsets", "ticks", "channels")
    }
    expected_shapes = {
        "packed": (7700, 80, STEM_CHANNELS // 8),
        "offsets": (7701,),
        "ticks": (int(metadata.get("raw_events", -1)),),
        "channels": (int(metadata.get("raw_events", -1)),),
    }
    expected_dtypes = {
        "packed": np.dtype(np.uint8),
        "offsets": np.dtype(np.int64),
        "ticks": np.dtype(np.int16),
        "channels": np.dtype(np.uint16),
    }
    for name, array in arrays.items():
        if tuple(array.shape) != expected_shapes[name]:
            raise ValueError(
                f"STEMNIST cache shape mismatch for {name}: "
                f"{tuple(array.shape)} != {expected_shapes[name]}"
            )
        if array.dtype != expected_dtypes[name]:
            raise ValueError(
                f"STEMNIST cache dtype mismatch for {name}: "
                f"{array.dtype} != {expected_dtypes[name]}"
            )
        if metadata.get("array_shapes", {}).get(name) != list(array.shape):
            raise ValueError(f"STEMNIST cache index shape mismatch for {name}")
        if metadata.get("array_dtypes", {}).get(name) != str(array.dtype):
            raise ValueError(f"STEMNIST cache index dtype mismatch for {name}")
    offsets = arrays["offsets"]
    ticks = arrays["ticks"]
    channels = arrays["channels"]
    packed = arrays["packed"]
    if int(offsets[0]) != 0 or int(offsets[-1]) != len(ticks):
        raise ValueError("STEMNIST cache offset endpoints are inconsistent")
    if np.any(np.diff(offsets) < 0):
        raise ValueError("STEMNIST cache offsets are not monotone")
    if len(ticks) and (int(ticks.min()) < 0 or int(ticks.max()) > 240):
        raise ValueError("STEMNIST cache ticks lie outside 0..240")
    if len(channels) and (
        int(channels.min()) < 0 or int(channels.max()) >= STEM_CHANNELS
    ):
        raise ValueError("STEMNIST cache channels lie outside 0..511")
    records = metadata.get("records", [])
    if len(records) != 7700:
        raise ValueError("STEMNIST cache record count mismatch")
    reconstructed_binary_events = 0
    for sample_index, record in enumerate(records):
        if int(record.get("sample_index", -1)) != sample_index:
            raise ValueError("STEMNIST cache record indices are not canonical")
        start, stop = int(offsets[sample_index]), int(offsets[sample_index + 1])
        if int(record.get("raw_spike_count", -1)) != stop - start:
            raise ValueError(
                f"STEMNIST cache raw count mismatch at sample {sample_index}"
            )
        binary = np.zeros((80, STEM_CHANNELS), dtype=np.uint8)
        sample_ticks = np.asarray(ticks[start:stop], dtype=np.int64)
        sample_channels = np.asarray(channels[start:stop], dtype=np.int64)
        binary[np.clip(sample_ticks // 3, 0, 79), sample_channels] = 1
        reconstructed = np.packbits(binary, axis=-1, bitorder="little")
        if not np.array_equal(reconstructed, packed[sample_index]):
            raise ValueError(
                f"STEMNIST cache raw-to-packed reconstruction mismatch at sample "
                f"{sample_index}"
            )
        binary_count = int(np.count_nonzero(binary))
        reconstructed_binary_events += binary_count
        if int(record.get("binary_event_count", -1)) != binary_count:
            raise ValueError(
                f"STEMNIST cache binary count mismatch at sample {sample_index}"
            )
    if reconstructed_binary_events != int(metadata.get("binary_events", -1)):
        raise ValueError("STEMNIST cache reconstructed binary total mismatch")
    return metadata


@dataclass(frozen=True)
class RawEvents:
    times_seconds: np.ndarray
    channels: np.ndarray


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    clean: Any
    labels: np.ndarray
    sample_ids: tuple[str, ...]
    raw_getter: Any
    classes: tuple[str, ...]
    duration_seconds: float
    time_steps: int
    channels: int
    metadata: dict[str, Any]
    groups: np.ndarray | None = None


def load_braille_bundle(project_root: Path) -> DatasetBundle:
    path = (
        project_root
        / "braille_letters_dataset"
        / "data"
        / "data_braille_letters_th1"
    )
    with path.open("rb") as handle:
        samples = pickle.load(handle, encoding="latin1")
    classes = tuple(sorted({str(sample["letter"]) for sample in samples}))
    label_to_index = {name: index for index, name in enumerate(classes)}
    labels = np.asarray(
        [label_to_index[str(sample["letter"])] for sample in samples], dtype=np.int64
    )
    sample_ids = tuple(f"braille_{index:05d}" for index in range(len(samples)))
    raw_samples: list[RawEvents] = []
    clean = np.zeros((len(samples), 45, 24), dtype=np.uint8)
    for sample_index, sample in enumerate(samples):
        time_parts: list[np.ndarray] = []
        channel_parts: list[np.ndarray] = []
        for taxel, (on_times, off_times) in enumerate(sample["events"]):
            on = np.asarray(on_times, dtype=np.float64)
            off = np.asarray(off_times, dtype=np.float64)
            if len(on):
                time_parts.append(on)
                channel_parts.append(np.full(len(on), taxel, dtype=np.int32))
            if len(off):
                time_parts.append(off)
                channel_parts.append(np.full(len(off), taxel + 12, dtype=np.int32))
        times = np.concatenate(time_parts) if time_parts else np.empty(0, np.float64)
        channels = (
            np.concatenate(channel_parts) if channel_parts else np.empty(0, np.int32)
        )
        order = np.lexsort((channels, times))
        raw = RawEvents(times[order], channels[order])
        # The frozen checkpoints observe full 25-ms bins 7..51 only.  Pre-bin
        # jitter must operate inside that 1.125-s observation window (not map
        # 1.3 s into 45 bins, which would silently change bin width).
        observation_start = BRAILLE_CROP_START * EVENT_DT_SECONDS
        in_window = (raw.times_seconds >= observation_start) & (
            raw.times_seconds <= BRAILLE_DURATION_SECONDS
        )
        observed_times = raw.times_seconds[in_window] - observation_start
        legacy_bins = np.clip(
            (raw.times_seconds[in_window] / EVENT_DT_SECONDS).astype(np.int64)
            - BRAILLE_CROP_START,
            0,
            44,
        )
        shifted_bins = np.clip(
            (observed_times / EVENT_DT_SECONDS).astype(np.int64), 0, 44
        )
        # Subtraction at a decimal bin boundary can yield e.g. 1.075 / .025
        # == 42.999999999.  Nudge only those boundary cases by 2.5 ps so the
        # zero-jitter reconstruction is bit-identical to the legacy bin/crop.
        boundary_roundoff = shifted_bins != legacy_bins
        observed_times[boundary_roundoff] = (
            legacy_bins[boundary_roundoff].astype(np.float64) + 1e-10
        ) * EVENT_DT_SECONDS
        observed_raw = RawEvents(observed_times, raw.channels[in_window])
        raw_samples.append(observed_raw)
        clean[sample_index] = bin_events(
            raw,
            duration_seconds=BRAILLE_DURATION_SECONDS,
            time_steps_full=52,
            channels=24,
            crop_start=BRAILLE_CROP_START,
            crop_end=52,
        )
        observed_reconstruction = bin_events(
            observed_raw,
            duration_seconds=BRAILLE_DURATION_SECONDS - observation_start,
            time_steps_full=45,
            channels=24,
        )
        if not np.array_equal(observed_reconstruction, clean[sample_index]):
            raise AssertionError(
                f"Braille observation-window reconstruction mismatch at sample {sample_index}"
            )
    source_sha = sha256_file(path)
    return DatasetBundle(
        name="braille",
        clean=clean,
        labels=labels,
        sample_ids=sample_ids,
        raw_getter=lambda index: raw_samples[int(index)],
        classes=classes,
        duration_seconds=BRAILLE_DURATION_SECONDS
        - BRAILLE_CROP_START * EVENT_DT_SECONDS,
        time_steps=45,
        channels=24,
        metadata={
            "source_path": str(path.resolve()),
            "source_sha256": source_sha,
            "samples": len(samples),
            "classes": len(classes),
            "class_names": list(classes),
            "event_dt_seconds": EVENT_DT_SECONDS,
            "crop_start": BRAILLE_CROP_START,
            "crop_end": BRAILLE_CROP_END,
            "observation_start_seconds": BRAILLE_CROP_START * EVENT_DT_SECONDS,
            "observation_duration_seconds": BRAILLE_DURATION_SECONDS
            - BRAILLE_CROP_START * EVENT_DT_SECONDS,
            "clean_reconstruction_exact": True,
            "clean_sha256": sha256_array(clean),
        },
    )


def load_stemnist_bundle(cache_dir: Path, verify_hash: bool) -> DatasetBundle:
    metadata = validate_stemnist_cache(cache_dir, verify_hash=verify_hash)
    paths = cache_paths(cache_dir)
    packed = np.load(paths["packed"], mmap_mode="r", allow_pickle=False)
    offsets = np.load(paths["offsets"], mmap_mode="r", allow_pickle=False)
    ticks = np.load(paths["ticks"], mmap_mode="r", allow_pickle=False)
    channels = np.load(paths["channels"], mmap_mode="r", allow_pickle=False)
    records = metadata["records"]
    labels = np.asarray([row["label_index"] for row in records], dtype=np.int64)
    groups = np.asarray([row["participant"] for row in records], dtype=object)
    sample_ids = tuple(str(row["sample_id"]) for row in records)

    def raw_getter(index: int) -> RawEvents:
        start, stop = int(offsets[index]), int(offsets[index + 1])
        return RawEvents(
            np.asarray(ticks[start:stop], dtype=np.float64) / STEM_SAMPLING_HZ,
            np.asarray(channels[start:stop], dtype=np.int32),
        )

    return DatasetBundle(
        name="stemnist",
        clean=packed,
        labels=labels,
        sample_ids=sample_ids,
        raw_getter=raw_getter,
        classes=CLASS_NAMES_STEM,
        duration_seconds=STEM_DURATION_SECONDS,
        time_steps=80,
        channels=STEM_CHANNELS,
        metadata={
            "cache_dir": str(cache_dir.resolve()),
            "cache_index_sha256": sha256_file(paths["index"]),
            "packed_sha256": metadata["sha256"]["packed"],
            "raw_events": metadata["raw_events"],
            "binary_events": metadata["binary_events"],
            "samples": metadata["samples"],
            "classes": len(CLASS_NAMES_STEM),
            "class_names": list(CLASS_NAMES_STEM),
            "event_dt_seconds": EVENT_DT_SECONDS,
        },
        groups=groups,
    )


def bin_events(
    events: RawEvents,
    *,
    duration_seconds: float,
    time_steps_full: int,
    channels: int,
    crop_start: int = 0,
    crop_end: int | None = None,
) -> np.ndarray:
    times = np.asarray(events.times_seconds, dtype=np.float64)
    event_channels = np.asarray(events.channels, dtype=np.int64)
    if times.shape != event_channels.shape:
        raise ValueError("Raw event time/channel arrays differ in shape")
    if np.any(~np.isfinite(times)) or np.any(times < 0) or np.any(times > duration_seconds):
        raise ValueError("Raw/pre-bin jitter produced an out-of-domain timestamp")
    if np.any((event_channels < 0) | (event_channels >= channels)):
        raise ValueError("Raw event channel is outside the representation")
    output = np.zeros((time_steps_full, channels), dtype=np.uint8)
    if len(times):
        bins = np.minimum((times / EVENT_DT_SECONDS).astype(np.int64), time_steps_full - 1)
        output[bins, event_channels] = 1
    stop = time_steps_full if crop_end is None else crop_end
    return output[crop_start:stop]


def unpack_clean_sample(bundle: DatasetBundle, sample_index: int) -> np.ndarray:
    if bundle.name == "braille":
        return np.asarray(bundle.clean[sample_index], dtype=np.uint8)
    packed = np.asarray(bundle.clean[sample_index])
    return np.unpackbits(
        packed, axis=-1, count=bundle.channels, bitorder="little"
    ).astype(np.uint8, copy=False)


class OperatorAdapter:
    """Thin adapter for ``icassp_jitter_operators``.

    Preferred API (keyword arguments are intentional)::

        prebin_jitter(timestamps, channels, severity_seconds, duration_seconds,
                      seed, *, n_steps, n_channels, realization,
                      return_timestamps=True)
            -> (binary, jittered_timestamps, audit)

        make_deconfounded_bundle(clean, radius_bins, seed, *, realization)
            -> mapping containing pbj, cf, matched, cf_matched and optional audit

    Aliases ``prebin_jitter`` and ``postbin_deconfounded_bundle`` are accepted.
    Every returned post-bin tensor must have the same [time, channel] shape as
    ``clean``.  This runner independently enforces binary values, exact CF count
    conservation, and exact PBJ/matched/CF+matched sample-by-channel counts.
    """

    def __init__(self) -> None:
        try:
            self.module = importlib.import_module("icassp_jitter_operators")
        except Exception as exc:
            raise RuntimeError(
                "Evaluation requires sibling module icassp_jitter_operators.py"
            ) from exc
        self.module_sha256 = sha256_file(Path(self.module.__file__).resolve())

    def prebin_binary(
        self,
        events: RawEvents,
        severity_ms: int,
        duration_seconds: float,
        seed: int,
        *,
        n_steps: int,
        n_channels: int,
        realization: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        function = getattr(self.module, "prebin_jitter", None)
        if function is None:
            raise AttributeError(
                "icassp_jitter_operators must expose prebin_jitter"
            )
        result = function(
            timestamps=np.asarray(events.times_seconds, dtype=np.float64),
            channels=np.asarray(events.channels, dtype=np.int32),
            severity_seconds=float(severity_ms) / 1000.0,
            duration_seconds=float(duration_seconds),
            seed=int(seed),
            n_steps=int(n_steps),
            n_channels=int(n_channels),
            realization=int(realization),
            return_timestamps=True,
        )
        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError(
                "prebin_jitter(..., return_timestamps=True) must return "
                "(binary, jittered_timestamps, audit)"
            )
        binary, jittered_times, raw_audit = result
        binary = np.asarray(binary, dtype=np.uint8)
        jittered_times = np.asarray(jittered_times, dtype=np.float64)
        audit = dict(raw_audit)
        if binary.shape != (n_steps, n_channels):
            raise ValueError(
                f"Pre-bin binary shape {binary.shape} != {(n_steps, n_channels)}"
            )
        if np.any((binary != 0) & (binary != 1)):
            raise ValueError("Pre-bin output is not binary")
        if len(jittered_times) != len(events.times_seconds):
            raise AssertionError("Pre-bin jitter must not delete raw events")
        if np.any(jittered_times < 0) or np.any(
            jittered_times > duration_seconds
        ):
            raise AssertionError("Pre-bin jitter violated conditional valid boundaries")
        displacement = np.abs(jittered_times - events.times_seconds)
        if np.any(displacement > severity_ms / 1000.0 + 1e-12):
            raise AssertionError("Pre-bin displacement exceeds requested severity")
        audit.update(
            {
                "raw_events_before": int(len(events.times_seconds)),
                "raw_events_after": int(len(jittered_times)),
                "raw_event_count_conserved": True,
                "max_abs_displacement_seconds": float(displacement.max(initial=0.0)),
                "boundary_policy_valid": True,
            }
        )
        return binary, audit

    def postbin_bundle(
        self, clean: np.ndarray, radius_bins: int, seed: int, realization: int
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        function = getattr(self.module, "make_deconfounded_bundle", None)
        if function is None:
            raise AttributeError(
                "icassp_jitter_operators must expose make_deconfounded_bundle"
            )
        result = function(
            clean=np.asarray(clean, dtype=np.uint8),
            radius_bins=int(radius_bins),
            seed=int(seed),
            realization=int(realization),
        )
        audit: dict[str, Any] = {}
        if isinstance(result, Mapping):
            arrays_source = result.get("arrays", result)
            audit = dict(result.get("audit", {}))
        elif hasattr(result, "arrays"):
            arrays_source = result.arrays
            audit = dict(getattr(result, "audit", {}))
        else:
            raise TypeError("Post-bin bundle must be a mapping or expose .arrays")
        aliases = {
            "pbj": ("pbj", "postbin_jitter"),
            "cf": ("cf", "cf_jitter", "collision_free"),
            "matched": ("matched", "matched_dropout", "event_loss_matched"),
            "cf_matched": (
                "cf_matched",
                "cf_plus_matched",
                "cf_matched_dropout",
            ),
        }
        arrays: dict[str, np.ndarray] = {}
        for canonical, names in aliases.items():
            value = next(
                (arrays_source[name] for name in names if name in arrays_source), None
            )
            if value is None:
                raise KeyError(f"Post-bin operator bundle is missing {canonical}")
            array = np.asarray(value, dtype=np.uint8)
            if array.shape != clean.shape:
                raise ValueError(f"{canonical} shape {array.shape} != {clean.shape}")
            if np.any((array != 0) & (array != 1)):
                raise ValueError(f"{canonical} output is not binary")
            arrays[canonical] = array
        clean_counts = np.count_nonzero(clean, axis=0)
        counts = {name: np.count_nonzero(array, axis=0) for name, array in arrays.items()}
        if not np.array_equal(counts["cf"], clean_counts):
            raise AssertionError("CF failed exact per-sample/channel conservation")
        if not np.array_equal(counts["matched"], counts["pbj"]):
            raise AssertionError("matched dropout does not exactly match PBJ by channel")
        if not np.array_equal(counts["cf_matched"], counts["pbj"]):
            raise AssertionError("CF+matched does not exactly match PBJ by channel")
        if np.any(counts["pbj"] > clean_counts):
            raise AssertionError("PBJ unexpectedly increased a channel's occupancy")
        audit.update(
            {
                "exact_cf_per_channel_conservation": True,
                "exact_matched_pbj_per_channel_counts": True,
                "exact_cf_matched_pbj_per_channel_counts": True,
                "clean_channel_counts_sha256": sha256_array(clean_counts),
                "pbj_channel_counts_sha256": sha256_array(counts["pbj"]),
                "matched_channel_counts_sha256": sha256_array(counts["matched"]),
                "cf_matched_channel_counts_sha256": sha256_array(
                    counts["cf_matched"]
                ),
            }
        )
        return arrays, audit


@dataclass(frozen=True)
class ModelRegistryEntry:
    registry_id: str
    dataset: str
    checkpoint_model_name: str
    training_distribution: str
    checkpoint_root: Path
    checkpoint_pattern: str
    config_id: str = ""
    comparison_note: str = ""

    def checkpoint_path(self, split_seed: int, fold: int) -> Path:
        return self.checkpoint_root / self.checkpoint_pattern.format(
            split_seed=split_seed, fold=fold
        )


def model_registry(args: argparse.Namespace, project_root: Path) -> dict[str, ModelRegistryEntry]:
    braille_clean_root = Path(args.braille_clean_snn_root)
    braille_tcn_root = Path(args.braille_tcn_root)
    stem_root = Path(args.stem_checkpoint_root)
    entries = (
        ModelRegistryEntry(
            "braille_snn_clean",
            "braille",
            "snn",
            "clean",
            braille_clean_root,
            "seed{split_seed}/checkpoints/clean_ce_adaptive0_seed{split_seed}_fold{fold}.pt",
            comparison_note="Frozen clean-CE RateSNN; primary clean-trained Braille probe.",
        ),
        ModelRegistryEntry(
            "braille_tcn_samplemix",
            "braille",
            "tcn",
            "samplemix",
            braille_tcn_root,
            "tcn_seed{split_seed}_fold{fold}.pt",
            comparison_note=(
                "Frozen taxel-enumeration TCN was trained with samplemix. It is an "
                "operator-screening probe and must not be interpreted as a controlled "
                "clean-trained SNN-versus-TCN architecture comparison."
            ),
        ),
        ModelRegistryEntry(
            "stemnist_paper_scnn_clean",
            "stemnist",
            "paper_scnn",
            "clean",
            stem_root / "cfg_2f4811e3ec848ef7",
            "paper_scnn_splitseed{split_seed}_fold{fold}.pt",
            config_id="cfg_2f4811e3ec848ef7",
            comparison_note="Frozen clean-trained STEMNIST paper-topology Conv-SNN.",
        ),
        ModelRegistryEntry(
            "stemnist_tcn_large_clean",
            "stemnist",
            "tcn",
            "clean",
            stem_root / "cfg_b7bc07c654cd7667",
            "tcn_splitseed{split_seed}_fold{fold}.pt",
            config_id="cfg_b7bc07c654cd7667",
            comparison_note=(
                "Frozen clean-trained 514k-parameter STEMNIST TCN; capacity is not "
                "matched to the paper Conv-SNN."
            ),
        ),
    )
    return {entry.registry_id: entry for entry in entries}


@dataclass
class RuntimeDeps:
    torch: Any
    nn: Any
    functional: Any
    braille_module: Any
    stem_module: Any


def import_runtime_dependencies() -> RuntimeDeps:
    """Import GPU/runtime dependencies only after any cache-builder handoff."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except Exception as exc:
        raise RuntimeError(
            "Evaluation requires PyTorch; run eval with the STAtten Python"
        ) from exc
    try:
        braille_module = importlib.import_module("run_braille_ablation_baselines")
        stem_module = importlib.import_module("run_stemnist_robustness")
    except Exception as exc:
        raise RuntimeError("Failed to import the frozen model/split definitions") from exc
    return RuntimeDeps(torch, nn, functional, braille_module, stem_module)


def load_checkpoint_safely(torch: Any, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen checkpoint: {path}")
    try:
        from torch.torch_version import TorchVersion

        with torch.serialization.safe_globals([TorchVersion]):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"Could not safely load checkpoint {path}") from exc
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise ValueError(f"Invalid checkpoint payload: {path}")
    return checkpoint


def namespace_from_checkpoint(
    checkpoint: Mapping[str, Any], entry: ModelRegistryEntry
) -> argparse.Namespace:
    values = dict(checkpoint.get("args", {}))
    if entry.dataset == "braille":
        if entry.checkpoint_model_name == "snn":
            values["hidden"] = int(values.get("hidden", values.get("snn_hidden", 192)))
            values["dropout"] = float(
                values.get("snn_dropout", values.get("dropout", 0.2))
            )
        else:
            values["hidden"] = int(values.get("dense_hidden", values.get("hidden", 128)))
            effective = values.get("effective_model_dropout", {})
            values["dropout"] = float(
                effective.get(
                    entry.checkpoint_model_name,
                    values.get("dense_dropout", values.get("dropout", 0.3)),
                )
            )
        values.setdefault("layers", 3)
        values.setdefault("beta", 0.8)
        values.setdefault("threshold_v", 0.5)
        values.setdefault("snn_readout", "spikes")
    return argparse.Namespace(**values)


def validate_checkpoint(
    checkpoint: Mapping[str, Any],
    path: Path,
    entry: ModelRegistryEntry,
    bundle: DatasetBundle,
    split_seed: int,
    fold: int,
    validation_indices: np.ndarray,
) -> None:
    mismatches: list[str] = []

    def compare(name: str, expected: Any, actual: Any) -> None:
        if expected != actual:
            mismatches.append(f"{name}: expected {expected!r}, got {actual!r}")

    compare("split_seed", split_seed, int(checkpoint.get("split_seed", -1)))
    compare("fold", fold, int(checkpoint.get("fold", -1)))
    expected_fold_seed = split_seed + fold - 1
    compare("fold_seed", expected_fold_seed, int(checkpoint.get("fold_seed", -1)))
    if "model_name" in checkpoint:
        compare("model_name", entry.checkpoint_model_name, checkpoint["model_name"])
    if entry.config_id:
        compare("config_id", entry.config_id, checkpoint.get("config_id"))
    meta = checkpoint.get("meta", {})
    compare("meta.samples", len(bundle.labels), int(meta.get("samples", -1)))
    compare("meta.time_steps", bundle.time_steps, int(meta.get("time_steps", -1)))
    compare("meta.channels", bundle.channels, int(meta.get("channels", -1)))
    compare("meta.classes", len(bundle.classes), int(meta.get("classes", -1)))
    checkpoint_args = checkpoint.get("args", {})
    compare("args.folds", 5, int(checkpoint_args.get("folds", -1)))
    if bundle.name == "braille":
        compare("args.threshold", "th1", checkpoint_args.get("threshold"))
        compare("args.event_dt", EVENT_DT_SECONDS, float(checkpoint_args.get("event_dt", -1)))
        compare("args.event_encoding", "polarity_binary", checkpoint_args.get("event_encoding"))
        compare("args.crop_start", BRAILLE_CROP_START, int(checkpoint_args.get("crop_start", -1)))
        compare("args.crop_end", BRAILLE_CROP_END, int(checkpoint_args.get("crop_end", 0)))
    else:
        expected_participants = sorted(set(bundle.groups[validation_indices].tolist()))
        actual_participants = sorted(meta.get("validation_participants", []))
        compare("meta.validation_participants", expected_participants, actual_participants)
        compare("args.event_dt", EVENT_DT_SECONDS, float(checkpoint_args.get("event_dt", -1)))
        compare("args.duration", STEM_DURATION_SECONDS, float(checkpoint_args.get("duration", -1)))
    if mismatches:
        raise ValueError(
            f"Checkpoint/protocol mismatch for {path}:\n  - " + "\n  - ".join(mismatches)
        )


def make_model_from_checkpoint(
    deps: RuntimeDeps,
    checkpoint: Mapping[str, Any],
    entry: ModelRegistryEntry,
    bundle: DatasetBundle,
    device: Any,
) -> Any:
    namespace = namespace_from_checkpoint(checkpoint, entry)
    if entry.dataset == "braille":
        model = deps.braille_module.make_model(
            entry.checkpoint_model_name,
            (len(bundle.labels), bundle.time_steps, bundle.channels),
            len(bundle.classes),
            namespace,
        )
    else:
        model = deps.stem_module.make_model(entry.checkpoint_model_name, namespace)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if missing or unexpected:
        raise ValueError(f"State mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model


def exact_folds(
    deps: RuntimeDeps, bundle: DatasetBundle, split_seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    if bundle.name == "braille":
        folds = list(
            deps.braille_module.stratified_kfold(
                bundle.labels, n_splits=5, seed=split_seed
            )
        )
    else:
        folds, splitter = deps.stem_module.make_group_folds(
            bundle.labels, bundle.groups, n_splits=5, seed=split_seed
        )
        if splitter != deps.stem_module.SPLITTER_ID:
            raise AssertionError(f"Unexpected STEMNIST splitter: {splitter}")
        deps.stem_module.validate_group_folds(folds, bundle.labels, bundle.groups)
    validation_seen = np.zeros(len(bundle.labels), dtype=np.int8)
    for train_indices, validation_indices in folds:
        if np.intersect1d(train_indices, validation_indices).size:
            raise AssertionError("Train/validation index overlap")
        validation_seen[validation_indices] += 1
    if not np.all(validation_seen == 1):
        raise AssertionError("Exact fold reconstruction did not cover every sample once")
    return [(np.asarray(a), np.asarray(b)) for a, b in folds]


@dataclass(frozen=True)
class ConditionSpec:
    condition: str
    severity_ms: int
    realization: int

    @property
    def radius_bins(self) -> int:
        if self.severity_ms % 25:
            raise ValueError("Post-bin severity must be an integer multiple of 25 ms")
        return self.severity_ms // 25

    @property
    def condition_id(self) -> str:
        return f"{self.condition}_ms{self.severity_ms}_r{self.realization}"


def condition_specs(args: argparse.Namespace) -> list[ConditionSpec]:
    specs: list[ConditionSpec] = []
    if "clean" in args.conditions:
        specs.append(ConditionSpec("clean", 0, 0))
    for severity in args.severities_ms:
        for realization in range(args.realizations):
            for condition in args.conditions:
                if condition != "clean":
                    specs.append(ConditionSpec(condition, int(severity), realization))
    return specs


def generate_condition_sample(
    adapter: OperatorAdapter,
    bundle: DatasetBundle,
    sample_index: int,
    spec: ConditionSpec,
    split_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    clean = unpack_clean_sample(bundle, sample_index)
    clean_counts = np.count_nonzero(clean, axis=0)
    # Corruption identity is fixed by dataset/sample and realization only.  It
    # is deliberately shared across models, checkpoints, folds/split seeds,
    # and severities (operators scale/prefix one normalized realization).
    common_seed = stable_seed(
        PROTOCOL_ID, "operator_root_v1", bundle.name, bundle.sample_ids[sample_index]
    )
    if spec.condition == "clean":
        output = clean.copy()
        audit: dict[str, Any] = {"identity": True}
    elif spec.condition == "prebin":
        output, audit = adapter.prebin_binary(
            bundle.raw_getter(sample_index),
            spec.severity_ms,
            bundle.duration_seconds,
            common_seed,
            n_steps=bundle.time_steps,
            n_channels=bundle.channels,
            realization=spec.realization,
        )
    elif spec.condition in POSTBIN_CONDITIONS:
        arrays, audit = adapter.postbin_bundle(
            clean, spec.radius_bins, common_seed, spec.realization
        )
        output = arrays[spec.condition]
    else:
        raise ValueError(f"Unknown condition: {spec.condition}")
    if output.shape != clean.shape:
        raise AssertionError(f"Corrupted sample shape changed: {output.shape} != {clean.shape}")
    output_counts = np.count_nonzero(output, axis=0)
    audit.update(
        {
            "active_before": int(clean_counts.sum()),
            "active_after": int(output_counts.sum()),
            "active_delta": int(output_counts.sum() - clean_counts.sum()),
            "active_retention": (
                float(output_counts.sum() / clean_counts.sum())
                if clean_counts.sum()
                else 1.0
            ),
            "before_channel_counts_sha256": sha256_array(clean_counts),
            "after_channel_counts_sha256": sha256_array(output_counts),
        }
    )
    return output.astype(np.float32, copy=False), audit


def resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def unit_identifier(
    entry: ModelRegistryEntry,
    split_seed: int,
    fold: int,
    spec: ConditionSpec,
) -> str:
    return (
        f"{entry.registry_id}__seed{split_seed}__fold{fold}__"
        f"{spec.condition_id}"
    )


def evaluate_unit(
    deps: RuntimeDeps,
    adapter: OperatorAdapter,
    bundle: DatasetBundle,
    entry: ModelRegistryEntry,
    split_seed: int,
    fold: int,
    validation_indices: np.ndarray,
    full_validation_indices: np.ndarray,
    spec: ConditionSpec,
    args: argparse.Namespace,
    device: Any,
    config_hash: str,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    checkpoint_path = entry.checkpoint_path(split_seed, fold).resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != expected_checkpoint_sha256:
        raise RuntimeError(
            "Frozen checkpoint changed after startup inventory: "
            f"{checkpoint_path}; {checkpoint_sha} != {expected_checkpoint_sha256}"
        )
    checkpoint = load_checkpoint_safely(deps.torch, checkpoint_path)
    validate_checkpoint(
        checkpoint,
        checkpoint_path,
        entry,
        bundle,
        split_seed,
        fold,
        full_validation_indices,
    )
    model = make_model_from_checkpoint(deps, checkpoint, entry, bundle, device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    unit_id = unit_identifier(entry, split_seed, fold, spec)
    predictions: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    total_loss = 0.0
    total_correct = 0
    total = 0
    active_before_total = 0
    active_after_total = 0
    model.eval()
    with deps.torch.no_grad():
        for start in range(0, len(validation_indices), args.batch_size):
            batch_indices = validation_indices[start : start + args.batch_size]
            batch_arrays: list[np.ndarray] = []
            batch_audits: list[dict[str, Any]] = []
            for global_index in batch_indices.tolist():
                array, audit = generate_condition_sample(
                    adapter, bundle, int(global_index), spec, split_seed
                )
                batch_arrays.append(array)
                batch_audits.append(audit)
            inputs_np = np.stack(batch_arrays, axis=0).astype(np.float32, copy=False)
            targets_np = bundle.labels[batch_indices]
            inputs = deps.torch.as_tensor(inputs_np, dtype=deps.torch.float32).to(
                device, non_blocking=True
            )
            targets = deps.torch.as_tensor(targets_np, dtype=deps.torch.long).to(
                device, non_blocking=True
            )
            logits = model(inputs)
            loss_vector = deps.functional.cross_entropy(logits, targets, reduction="none")
            predicted = logits.argmax(dim=1)
            top2 = logits.topk(k=min(2, logits.shape[1]), dim=1).values
            margins = top2[:, 0] - top2[:, -1]
            true_logits = logits.gather(1, targets[:, None]).squeeze(1)
            predicted_logits = logits.gather(1, predicted[:, None]).squeeze(1)
            predicted_cpu = predicted.cpu().tolist()
            targets_cpu = targets.cpu().tolist()
            loss_cpu = loss_vector.cpu().tolist()
            true_logits_cpu = true_logits.cpu().tolist()
            predicted_logits_cpu = predicted_logits.cpu().tolist()
            margins_cpu = margins.cpu().tolist()
            total_loss += float(sum(loss_cpu))
            total_correct += sum(
                int(prediction == target)
                for prediction, target in zip(predicted_cpu, targets_cpu)
            )
            total += len(batch_indices)
            for local_index, global_index in enumerate(batch_indices.tolist()):
                audit = batch_audits[local_index]
                active_before_total += int(audit["active_before"])
                active_after_total += int(audit["active_after"])
                if args.save_predictions:
                    row = {
                        "protocol_id": PROTOCOL_ID,
                        "config_hash": config_hash,
                        "unit_id": unit_id,
                        "dataset": bundle.name,
                        "model_registry_id": entry.registry_id,
                        "checkpoint_model_name": entry.checkpoint_model_name,
                        "training_distribution": entry.training_distribution,
                        "split_seed": split_seed,
                        "fold": fold,
                        "sample_index": global_index,
                        "sample_id": bundle.sample_ids[global_index],
                        "target": int(targets_cpu[local_index]),
                        "prediction": int(predicted_cpu[local_index]),
                        "correct": int(predicted_cpu[local_index] == targets_cpu[local_index]),
                        "loss": float(loss_cpu[local_index]),
                        "true_logit": float(true_logits_cpu[local_index]),
                        "predicted_logit": float(predicted_logits_cpu[local_index]),
                        "margin": float(margins_cpu[local_index]),
                        "condition": spec.condition,
                        "severity_ms": spec.severity_ms,
                        "radius_bins": spec.radius_bins if spec.severity_ms else 0,
                        "realization": spec.realization,
                        "active_before": audit["active_before"],
                        "active_after": audit["active_after"],
                        "active_retention": audit["active_retention"],
                        "before_channel_counts_sha256": audit[
                            "before_channel_counts_sha256"
                        ],
                        "after_channel_counts_sha256": audit[
                            "after_channel_counts_sha256"
                        ],
                        "exact_cf_per_channel_conservation": audit.get(
                            "exact_cf_per_channel_conservation", ""
                        ),
                        "exact_matched_pbj_per_channel_counts": audit.get(
                            "exact_matched_pbj_per_channel_counts", ""
                        ),
                        "exact_cf_matched_pbj_per_channel_counts": audit.get(
                            "exact_cf_matched_pbj_per_channel_counts", ""
                        ),
                        "raw_event_count_conserved": audit.get(
                            "raw_event_count_conserved", ""
                        ),
                        "raw_events_before": audit.get("input_event_count", ""),
                        "raw_events_after": audit.get("output_event_count", ""),
                        "boundary_constrained_events": audit.get(
                            "boundary_constrained_events", ""
                        ),
                        "events_exactly_at_boundary_after_jitter": audit.get(
                            "events_exactly_at_boundary_after_jitter", ""
                        ),
                        "prebin_binary_collision_count": audit.get(
                            "binary_collision_count", ""
                        ),
                    }
                    if bundle.groups is not None:
                        row["participant"] = str(bundle.groups[global_index])
                    predictions.append(row)
                audit_records.append(audit)
    del model
    if device.type == "cuda":
        deps.torch.cuda.empty_cache()
    if total == 0:
        raise AssertionError("Evaluation unit contains no validation samples")

    def every_true(field: str) -> bool | str:
        values = [record[field] for record in audit_records if field in record]
        return bool(values and all(bool(value) for value in values)) if values else ""

    def audit_sum(field: str) -> int | str:
        values = [record[field] for record in audit_records if field in record]
        return int(sum(int(value) for value in values)) if values else ""

    def nested_audit_values(*path: str) -> list[Any]:
        values: list[Any] = []
        for record in audit_records:
            value: Any = record
            for component in path:
                if not isinstance(value, Mapping) or component not in value:
                    value = None
                    break
                value = value[component]
            if value is not None:
                values.append(value)
        return values

    def nested_audit_sum(*path: str) -> int | str:
        values = nested_audit_values(*path)
        return int(sum(int(value) for value in values)) if values else ""

    def primitive_digest(*path: str) -> str:
        values = nested_audit_values(*path)
        return sha256_json(values) if values else ""

    pbj_displacement_digest = primitive_digest(
        "pbj", "normalized_displacement_digest"
    )
    cf_pass_digest = primitive_digest("cf", "cf_pass_digests")
    matched_rank_digest = primitive_digest("matched", "matched_rank_digest")
    prebin_quantile_digest = primitive_digest("uniform_quantile_digest")
    if spec.condition in POSTBIN_CONDITIONS:
        operator_primitive_digest = sha256_json(
            {
                "pbj_normalized_displacement": pbj_displacement_digest,
                "cf_passes": cf_pass_digest,
                "matched_rank": matched_rank_digest,
            }
        )
    elif spec.condition == "prebin":
        operator_primitive_digest = prebin_quantile_digest
    else:
        operator_primitive_digest = ""

    result = {
        "protocol_id": PROTOCOL_ID,
        "config_hash": config_hash,
        "unit_id": unit_id,
        "dataset": bundle.name,
        "model_registry_id": entry.registry_id,
        "checkpoint_model_name": entry.checkpoint_model_name,
        "training_distribution": entry.training_distribution,
        "comparison_note": entry.comparison_note,
        "split_seed": split_seed,
        "fold": fold,
        "fold_seed": split_seed + fold - 1,
        "condition": spec.condition,
        "severity_ms": spec.severity_ms,
        "radius_bins": spec.radius_bins if spec.severity_ms else 0,
        "realization": spec.realization,
        "samples": total,
        "loss": total_loss / total,
        "accuracy": total_correct / total,
        "correct": total_correct,
        "active_before": active_before_total,
        "active_after": active_after_total,
        "active_retention": (
            active_after_total / active_before_total if active_before_total else 1.0
        ),
        "parameters": parameter_count,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_sha256_expected": expected_checkpoint_sha256,
        "validation_indices_sha256": sha256_array(full_validation_indices),
        "evaluated_indices_sha256": sha256_array(validation_indices),
        "full_validation_samples": int(len(full_validation_indices)),
        "exact_cf_per_channel_conservation": every_true(
            "exact_cf_per_channel_conservation"
        ),
        "exact_matched_pbj_per_channel_counts": every_true(
            "exact_matched_pbj_per_channel_counts"
        ),
        "exact_cf_matched_pbj_per_channel_counts": every_true(
            "exact_cf_matched_pbj_per_channel_counts"
        ),
        "raw_event_count_conserved": every_true("raw_event_count_conserved"),
        "raw_events_before": audit_sum("input_event_count"),
        "raw_events_after": audit_sum("output_event_count"),
        "boundary_constrained_events": audit_sum("boundary_constrained_events"),
        "events_exactly_at_boundary_after_jitter": audit_sum(
            "events_exactly_at_boundary_after_jitter"
        ),
        "prebin_binary_collision_count": audit_sum("binary_collision_count"),
        "pbj_boundary_clipped_active_cells": nested_audit_sum(
            "pbj", "boundary_clipped_active_cells"
        ),
        "pbj_active_cells_moved": nested_audit_sum(
            "pbj", "active_cells_moved"
        ),
        "pbj_binary_collision_loss": nested_audit_sum(
            "pbj", "binary_collision_loss"
        ),
        "prebin_uniform_quantile_primitives_sha256": prebin_quantile_digest,
        "pbj_normalized_displacement_primitives_sha256": pbj_displacement_digest,
        "cf_pass_primitives_sha256": cf_pass_digest,
        "matched_rank_primitives_sha256": matched_rank_digest,
        "operator_primitive_digest_sha256": operator_primitive_digest,
    }
    if spec.condition == "prebin" and result["raw_events_before"]:
        result["boundary_constrained_fraction"] = float(
            int(result["boundary_constrained_events"])
            / int(result["raw_events_before"])
        )
    else:
        result["boundary_constrained_fraction"] = ""
    audit_summary = {
        **result,
        "active_delta": active_after_total - active_before_total,
        "minimum_sample_active_retention": min(
            float(record["active_retention"]) for record in audit_records
        ),
        "maximum_sample_active_retention": max(
            float(record["active_retention"]) for record in audit_records
        ),
    }
    return {"result": result, "audit": audit_summary, "predictions": predictions}


RESULT_BASENAME = "results.csv"
AUDIT_BASENAME = "active_audits.csv"
# Schema-v1 compatibility sentinel only.  Schema v2 never creates this file:
# predictions live in one atomically committed shard per evaluation unit.
PREDICTION_BASENAME = "predictions.csv"
MANIFEST_BASENAME = "manifest.json"
STATE_BASENAME = "resume_state.json"
SUMMARY_BASENAME = "summary.csv"
INTEGRITY_BASENAME = "completion_integrity.json"
FROZEN_SEED_METRICS_BASENAME = "frozen_primary_metrics_by_seed.csv"
FROZEN_AGGREGATE_METRICS_BASENAME = "frozen_primary_metrics_3seed.csv"
UNIT_ROOT_BASENAME = "units"
UNIT_RESULT_BASENAME = "result.json"
UNIT_AUDIT_BASENAME = "audit.json"
UNIT_PREDICTIONS_BASENAME = "predictions.csv"
UNIT_COMMIT_BASENAME = "commit.json"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_config_payload(
    args: argparse.Namespace,
    bundles: Mapping[str, DatasetBundle],
    entries: Sequence[ModelRegistryEntry],
    checkpoint_inventory: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "script_schema_version": SCRIPT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "datasets": list(args.datasets),
        "models": [entry.registry_id for entry in entries],
        "seeds": list(args.seeds),
        "folds": 5,
        "severities_ms": list(args.severities_ms),
        "realizations": args.realizations,
        "conditions": list(args.conditions),
        "save_predictions": bool(args.save_predictions),
        "dataset_identity": {
            name: bundle.metadata for name, bundle in bundles.items()
        },
        "checkpoint_registry": [
            {
                **entry.__dict__,
                "checkpoint_root": str(entry.checkpoint_root.resolve()),
            }
            for entry in entries
        ],
        "checkpoint_inventory": list(checkpoint_inventory or []),
        "smoke": {
            "max_folds": args.max_folds,
            "max_validation_samples": args.max_validation_samples,
        },
    }


def checkpoint_inventory_key(
    registry_id: str, split_seed: int, fold: int
) -> str:
    return f"{registry_id}__seed{int(split_seed)}__fold{int(fold)}"


def build_checkpoint_inventory(
    entries: Sequence[ModelRegistryEntry],
    seeds: Sequence[int],
    max_folds: int = 0,
) -> list[dict[str, Any]]:
    """Hash every checkpoint selected for this run before config identity exists."""
    fold_count = int(max_folds) if max_folds else 5
    inventory: list[dict[str, Any]] = []
    for entry in entries:
        for split_seed in seeds:
            for fold in range(1, fold_count + 1):
                path = entry.checkpoint_path(int(split_seed), fold).resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"Missing frozen checkpoint: {path}")
                inventory.append(
                    {
                        "inventory_key": checkpoint_inventory_key(
                            entry.registry_id, int(split_seed), fold
                        ),
                        "model_registry_id": entry.registry_id,
                        "dataset": entry.dataset,
                        "split_seed": int(split_seed),
                        "fold": fold,
                        "checkpoint_path": str(path),
                        "checkpoint_sha256": sha256_file(path),
                    }
                )
    keys = [str(row["inventory_key"]) for row in inventory]
    if len(keys) != len(set(keys)):
        raise AssertionError("Checkpoint inventory contains duplicate keys")
    return inventory


SUMMARY_INTEGER_FIELDS = frozenset(("split_seed", "severity_ms", "realization"))


def normalize_summary_group_value(field: str, value: Any) -> Any:
    """Normalize native rows and csv.DictReader rows to one grouping identity."""
    if field in SUMMARY_INTEGER_FIELDS:
        if isinstance(value, bool):
            raise ValueError(f"Boolean is invalid for integer summary field {field}")
        if isinstance(value, (int, np.integer)):
            return int(value)
        text = str(value).strip()
        if not text:
            raise ValueError(f"Empty integer summary field: {field}")
        numeric = float(text)
        if not numeric.is_integer():
            raise ValueError(f"Non-integral summary field {field}: {value!r}")
        return int(numeric)
    return str(value)


def summarize_results(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    fields = (
        "dataset",
        "model_registry_id",
        "checkpoint_model_name",
        "training_distribution",
        "split_seed",
        "condition",
        "severity_ms",
        "realization",
    )
    for row in rows:
        key = tuple(
            normalize_summary_group_value(field, row[field]) for field in fields
        )
        grouped[key].append(row)
    summary: list[dict[str, Any]] = []
    for key, members in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        accuracies = np.asarray([float(member["accuracy"]) for member in members])
        sample_counts = np.asarray([int(member["samples"]) for member in members])
        total_samples = int(sample_counts.sum())
        active_before = sum(int(member["active_before"]) for member in members)
        active_after = sum(int(member["active_after"]) for member in members)
        row = {field: value for field, value in zip(fields, key)}
        row.update(
            {
                "folds": len(members),
                "samples": total_samples,
                "accuracy_fold_mean": float(accuracies.mean()),
                "accuracy_fold_std": (
                    float(accuracies.std(ddof=1)) if len(accuracies) > 1 else ""
                ),
                "accuracy_pooled": float(
                    sum(int(member["correct"]) for member in members) / total_samples
                ),
                "loss_pooled": float(
                    sum(float(member["loss"]) * int(member["samples"]) for member in members)
                    / total_samples
                ),
                "active_before": active_before,
                "active_after": active_after,
                "active_retention": (
                    active_after / active_before if active_before else 1.0
                ),
            }
        )
        summary.append(row)
    return summary


def _unit_directory(output_dir: Path, unit_id: str) -> Path:
    allowed = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    if not unit_id or any(character not in allowed for character in unit_id):
        raise ValueError(f"Unsafe unit identifier for shard path: {unit_id!r}")
    return output_dir / UNIT_ROOT_BASENAME / unit_id


def _read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def commit_unit_transaction(
    output_dir: Path,
    config_hash: str,
    result: Mapping[str, Any],
    audit: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    save_predictions: bool,
) -> dict[str, Any]:
    """Atomically commit one unit, with ``commit.json`` as the final marker."""
    unit_id = str(result["unit_id"])
    if str(audit.get("unit_id", "")) != unit_id:
        raise ValueError("Unit result/audit identifiers differ")
    unit_dir = _unit_directory(output_dir, unit_id)
    unit_dir.mkdir(parents=True, exist_ok=True)
    commit_path = unit_dir / UNIT_COMMIT_BASENAME
    if commit_path.exists():
        raise FileExistsError(f"Evaluation unit is already committed: {unit_id}")

    result_path = unit_dir / UNIT_RESULT_BASENAME
    audit_path = unit_dir / UNIT_AUDIT_BASENAME
    prediction_path = unit_dir / UNIT_PREDICTIONS_BASENAME
    write_json_atomic(result_path, dict(result))
    write_json_atomic(audit_path, dict(audit))
    if save_predictions:
        if any(str(row.get("unit_id", "")) != unit_id for row in predictions):
            raise ValueError(f"Prediction shard mixes unit identifiers: {unit_id}")
        write_csv_atomic(prediction_path, predictions)
    elif prediction_path.exists():
        raise ValueError(f"Unexpected prediction shard in no-predictions run: {prediction_path}")

    marker = {
        "persistence_schema_version": PERSISTENCE_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "config_hash": config_hash,
        "unit_id": unit_id,
        "committed_utc": utc_now(),
        "save_predictions": bool(save_predictions),
        "prediction_rows": len(predictions) if save_predictions else 0,
        "artifacts": {
            UNIT_RESULT_BASENAME: sha256_file(result_path),
            UNIT_AUDIT_BASENAME: sha256_file(audit_path),
            UNIT_PREDICTIONS_BASENAME: (
                sha256_file(prediction_path) if save_predictions else None
            ),
        },
    }
    # This replace is the transaction commit point.  Files without this marker
    # are an uncommitted tail and are safely overwritten by a resumed unit.
    write_json_atomic(commit_path, marker)
    return marker


def load_unit_transaction(
    output_dir: Path,
    unit_id: str,
    config_hash: str,
    save_predictions: bool,
    *,
    stream_predictions: bool,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Validate one committed unit without retaining prediction rows in memory."""
    unit_dir = _unit_directory(output_dir, unit_id)
    marker = _read_json_mapping(unit_dir / UNIT_COMMIT_BASENAME)
    if marker.get("persistence_schema_version") != PERSISTENCE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported unit persistence schema: {unit_id}")
    if marker.get("config_hash") != config_hash or marker.get("unit_id") != unit_id:
        raise ValueError(f"Unit commit identity mismatch: {unit_id}")
    if bool(marker.get("save_predictions")) != bool(save_predictions):
        raise ValueError(f"Unit prediction policy mismatch: {unit_id}")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"Unit commit lacks artifact hashes: {unit_id}")

    result_path = unit_dir / UNIT_RESULT_BASENAME
    audit_path = unit_dir / UNIT_AUDIT_BASENAME
    prediction_path = unit_dir / UNIT_PREDICTIONS_BASENAME
    for path in (result_path, audit_path):
        if not path.is_file() or sha256_file(path) != artifacts.get(path.name):
            raise ValueError(f"Committed unit artifact hash mismatch: {path}")
    result = _read_json_mapping(result_path)
    audit = _read_json_mapping(audit_path)
    if str(result.get("unit_id", "")) != unit_id or str(audit.get("unit_id", "")) != unit_id:
        raise ValueError(f"Committed result/audit identity mismatch: {unit_id}")

    prediction_rows = 0
    if save_predictions:
        if not prediction_path.is_file() or sha256_file(prediction_path) != artifacts.get(
            UNIT_PREDICTIONS_BASENAME
        ):
            raise ValueError(f"Committed prediction artifact hash mismatch: {prediction_path}")
        if stream_predictions:
            with prediction_path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if str(row.get("unit_id", "")) != unit_id:
                        raise ValueError(f"Prediction shard mixes unit identifiers: {unit_id}")
                    prediction_rows += 1
        else:
            prediction_rows = int(marker.get("prediction_rows", -1))
        if prediction_rows != int(marker.get("prediction_rows", -1)):
            raise ValueError(f"Prediction shard row-count mismatch: {unit_id}")
    elif prediction_path.exists():
        raise ValueError(f"No-predictions commit owns a prediction shard: {unit_id}")
    return result, audit, prediction_rows


def verify_resume_state(
    output_dir: Path,
    config_hash: str,
    args: argparse.Namespace,
) -> tuple[set[str], list[dict[str, Any]], list[dict[str, Any]], int]:
    state_path = output_dir / STATE_BASENAME
    manifest_path = output_dir / MANIFEST_BASENAME
    unit_root = output_dir / UNIT_ROOT_BASENAME
    artifact_paths = [
        output_dir / RESULT_BASENAME,
        output_dir / AUDIT_BASENAME,
        output_dir / PREDICTION_BASENAME,
        manifest_path,
        state_path,
        output_dir / SUMMARY_BASENAME,
        output_dir / INTEGRITY_BASENAME,
        unit_root,
    ]
    existing = [path for path in artifact_paths if path.exists()]
    if not existing:
        return set(), [], [], 0
    if not args.resume:
        raise FileExistsError(
            f"Output directory already owns artifacts; use --resume or select a new "
            f"--output-dir: {output_dir}"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Cannot safely resume without {manifest_path}")
    manifest = _read_json_mapping(manifest_path)
    if manifest.get("persistence_schema_version") != PERSISTENCE_SCHEMA_VERSION:
        raise ValueError(
            "Legacy/unsharded output cannot be resumed under persistence schema v2; "
            "choose a new output directory"
        )
    if manifest.get("config_hash") != config_hash:
        raise ValueError(
            "Resume configuration mismatch; existing outputs will not be mixed with "
            "a different protocol/configuration"
        )
    if (output_dir / PREDICTION_BASENAME).exists():
        raise ValueError("Schema v2 must not contain a top-level predictions.csv")
    if state_path.is_file():
        state = _read_json_mapping(state_path)
        if state.get("config_hash") != config_hash:
            raise ValueError("Resume state configuration mismatch")

    completed: set[str] = set()
    results: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    prediction_rows = 0
    if unit_root.is_dir():
        for unit_dir in sorted(path for path in unit_root.iterdir() if path.is_dir()):
            commit_path = unit_dir / UNIT_COMMIT_BASENAME
            if not commit_path.is_file():
                # Uncommitted tail from a killed process: not scientific state.
                continue
            unit_id = unit_dir.name
            result, audit, rows = load_unit_transaction(
                output_dir,
                unit_id,
                config_hash,
                args.save_predictions,
                stream_predictions=True,
            )
            completed.add(unit_id)
            results.append(result)
            audits.append(audit)
            prediction_rows += rows
    return completed, results, audits, prediction_rows


def persist_progress(
    output_dir: Path,
    config_hash: str,
    completed_count: int,
    prediction_rows: int,
) -> None:
    """Write a lightweight progress hint; unit commit markers remain authoritative."""
    write_json_atomic(
        output_dir / STATE_BASENAME,
        {
            "persistence_schema_version": PERSISTENCE_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "config_hash": config_hash,
            "updated_utc": utc_now(),
            "committed_units_count": int(completed_count),
            "prediction_rows": int(prediction_rows),
            "authoritative_state": f"{UNIT_ROOT_BASENAME}/*/{UNIT_COMMIT_BASENAME}",
        },
    )


def materialize_derived_tables(
    output_dir: Path,
    results: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
) -> None:
    """Create compact, regenerable indexes once; predictions remain sharded."""
    write_csv_atomic(output_dir / RESULT_BASENAME, results)
    write_csv_atomic(output_dir / AUDIT_BASENAME, audits)
    write_csv_atomic(output_dir / SUMMARY_BASENAME, summarize_results(results))


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _integer_row_value(row: Mapping[str, Any], field: str) -> int:
    return int(normalize_summary_group_value(field, row[field]))


def _pooled_accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
    samples = sum(int(row["samples"]) for row in rows)
    if samples <= 0:
        raise ValueError("Cannot pool accuracy over zero samples")
    return float(sum(int(row["correct"]) for row in rows) / samples)


def _complete_fold_cell(
    members: Sequence[Mapping[str, Any]], expected_folds: set[int]
) -> bool:
    folds = [_integer_row_value(row, "fold") for row in members]
    return len(folds) == len(expected_folds) and set(folds) == expected_folds


def frozen_protocol_scope(args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    expected_datasets = {"braille", "stemnist"}
    if set(args.datasets) != expected_datasets:
        reasons.append("datasets must be exactly braille+stemnist")
    if set(args.models) != set(DEFAULT_MODEL_IDS):
        reasons.append("models must be the four frozen Stage-A registry entries")
    if set(args.seeds) != set(DEFAULT_SEEDS) or len(args.seeds) != len(DEFAULT_SEEDS):
        reasons.append("seeds must be exactly 42,123,202")
    if set(args.severities_ms) != set(DEFAULT_SEVERITIES_MS):
        reasons.append("severities must be exactly 25,50,75 ms")
    if int(args.realizations) != 3:
        reasons.append("realizations must equal 3")
    if set(args.conditions) != set(DEFAULT_CONDITIONS):
        reasons.append("conditions must be the six frozen conditions")
    if int(args.max_folds) != 0:
        reasons.append("max_folds must be zero (all five folds)")
    if int(args.max_validation_samples) != 0:
        reasons.append("max_validation_samples must be zero (full folds)")
    return not reasons, reasons


def build_frozen_primary_metric_tables(
    rows: Sequence[Mapping[str, Any]], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build the preregistered per-seed and three-seed primary metric tables.

    A seed/model/operator entry is publishable only when clean has five folds and
    every one of the operator's 3 severities x 3 realizations has five folds.
    Incomplete rows are retained and visibly blank rather than silently averaged.
    """
    grouped: dict[tuple[str, int, str, int, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        key = (
            str(row["model_registry_id"]),
            _integer_row_value(row, "split_seed"),
            str(row["condition"]),
            _integer_row_value(row, "severity_ms"),
            _integer_row_value(row, "realization"),
        )
        grouped[key].append(row)

    expected_folds = set(range(1, 6))
    expected_cells = [
        (severity, realization)
        for severity in DEFAULT_SEVERITIES_MS
        for realization in range(3)
    ]
    per_seed: list[dict[str, Any]] = []
    for model_id in args.models:
        for split_seed in DEFAULT_SEEDS:
            clean_members = grouped.get((model_id, split_seed, "clean", 0, 0), [])
            clean_complete = _complete_fold_cell(clean_members, expected_folds)
            clean_accuracy = _pooled_accuracy(clean_members) if clean_complete else None
            for operator in FROZEN_OPERATORS:
                cell_members = [
                    grouped.get(
                        (model_id, split_seed, operator, severity, realization), []
                    )
                    for severity, realization in expected_cells
                ]
                cells_complete = [
                    _complete_fold_cell(members, expected_folds)
                    for members in cell_members
                ]
                complete = bool(clean_complete and all(cells_complete))
                condition_accuracies = (
                    [_pooled_accuracy(members) for members in cell_members]
                    if complete
                    else []
                )
                if complete:
                    macc = float(np.mean(condition_accuracies))
                    retention = (
                        float(macc / clean_accuracy) if clean_accuracy else math.nan
                    )
                    mean_drop = float(clean_accuracy - macc)
                    active_before = sum(
                        int(row["active_before"])
                        for members in cell_members
                        for row in members
                    )
                    active_after = sum(
                        int(row["active_after"])
                        for members in cell_members
                        for row in members
                    )
                    active_retention = (
                        float(active_after / active_before)
                        if active_before
                        else 1.0
                    )
                else:
                    macc = retention = mean_drop = active_retention = ""
                missing_cells = [
                    f"{severity}ms_r{realization}"
                    for (severity, realization), present in zip(
                        expected_cells, cells_complete
                    )
                    if not present
                ]
                per_seed.append(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "model_registry_id": model_id,
                        "split_seed": split_seed,
                        "operator": operator,
                        "status": "complete" if complete else "incomplete",
                        "clean_5fold_complete": clean_complete,
                        "condition_cells_expected": 9,
                        "condition_cells_complete": int(sum(cells_complete)),
                        "folds_per_condition_expected": 5,
                        "missing_condition_cells": ";".join(missing_cells),
                        "clean_accuracy": (
                            float(clean_accuracy) if clean_accuracy is not None else ""
                        ),
                        "mAcc": macc,
                        "Retention": retention,
                        "MeanDrop": mean_drop,
                        "ActiveRetention": active_retention,
                        "metric_definition": (
                            "mAcc=mean(9 pooled-5-fold accuracies); "
                            "Retention=mAcc/clean pooled-5-fold accuracy; "
                            "MeanDrop=clean-mAcc; ActiveRetention=pooled active_after/"
                            "active_before over 9x5 operator units"
                        ),
                    }
                )

    by_model_operator: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_seed:
        by_model_operator[
            (str(row["model_registry_id"]), str(row["operator"]))
        ].append(row)
    aggregate: list[dict[str, Any]] = []
    metric_names = ("mAcc", "Retention", "MeanDrop", "ActiveRetention")
    for model_id in args.models:
        for operator in FROZEN_OPERATORS:
            members = sorted(
                by_model_operator.get((model_id, operator), []),
                key=lambda row: int(row["split_seed"]),
            )
            complete_members = [row for row in members if row["status"] == "complete"]
            seeds_present = {int(row["split_seed"]) for row in complete_members}
            complete = seeds_present == set(DEFAULT_SEEDS) and len(complete_members) == 3
            aggregate_row: dict[str, Any] = {
                "protocol_id": PROTOCOL_ID,
                "model_registry_id": model_id,
                "operator": operator,
                "status": "complete" if complete else "incomplete",
                "seeds_expected": "42;123;202",
                "seeds_complete": ";".join(map(str, sorted(seeds_present))),
                "seed_count_complete": len(complete_members),
                "sample_sd_ddof": 1,
            }
            for metric in metric_names:
                if complete:
                    values = np.asarray(
                        [float(row[metric]) for row in complete_members],
                        dtype=np.float64,
                    )
                    aggregate_row[f"{metric}_mean"] = float(values.mean())
                    aggregate_row[f"{metric}_sample_sd"] = float(
                        values.std(ddof=1)
                    )
                else:
                    aggregate_row[f"{metric}_mean"] = ""
                    aggregate_row[f"{metric}_sample_sd"] = ""
            aggregate.append(aggregate_row)

    expected_seed_rows = len(args.models) * len(FROZEN_OPERATORS) * len(DEFAULT_SEEDS)
    expected_aggregate_rows = len(args.models) * len(FROZEN_OPERATORS)
    status = {
        "per_seed_rows": len(per_seed),
        "per_seed_rows_expected": expected_seed_rows,
        "per_seed_complete": sum(row["status"] == "complete" for row in per_seed),
        "aggregate_rows": len(aggregate),
        "aggregate_rows_expected": expected_aggregate_rows,
        "aggregate_complete": sum(row["status"] == "complete" for row in aggregate),
    }
    status["complete"] = bool(
        status["per_seed_complete"] == expected_seed_rows
        and status["aggregate_complete"] == expected_aggregate_rows
    )
    return per_seed, aggregate, status


def write_completion_integrity(
    output_dir: Path,
    config_hash: str,
    expected_units: set[str],
    completed: set[str],
    results: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    checkpoint_inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result_ids = [str(row["unit_id"]) for row in results]
    audit_ids = [str(row["unit_id"]) for row in audits]
    failures: list[str] = []
    if len(result_ids) != len(set(result_ids)):
        failures.append("duplicate result unit_id")
    if len(audit_ids) != len(set(audit_ids)):
        failures.append("duplicate audit unit_id")
    if set(result_ids) != expected_units:
        failures.append("result units differ from expected units")
    if set(audit_ids) != expected_units:
        failures.append("audit units differ from expected units")
    if completed != expected_units:
        failures.append("resume completed units differ from expected units")

    inventory_by_key = {
        str(row["inventory_key"]): row for row in checkpoint_inventory
    }
    if len(inventory_by_key) != len(checkpoint_inventory):
        failures.append("duplicate checkpoint inventory key")
    observed_checkpoint_keys: set[str] = set()
    observed_checkpoint_hashes: dict[str, set[str]] = defaultdict(set)
    for row in results:
        key = checkpoint_inventory_key(
            str(row["model_registry_id"]),
            _integer_row_value(row, "split_seed"),
            _integer_row_value(row, "fold"),
        )
        observed_checkpoint_keys.add(key)
        actual_hash = str(row.get("checkpoint_sha256", ""))
        observed_checkpoint_hashes[key].add(actual_hash)
        inventory_row = inventory_by_key.get(key)
        if inventory_row is None:
            failures.append(f"result references un-inventoried checkpoint: {key}")
            continue
        expected_hash = str(inventory_row["checkpoint_sha256"])
        if actual_hash != expected_hash:
            failures.append(f"checkpoint SHA mismatch: {key}")
        if str(row.get("checkpoint_sha256_expected", "")) != expected_hash:
            failures.append(f"checkpoint expected-SHA field mismatch: {key}")
        if str(row.get("checkpoint_path", "")) != str(
            inventory_row["checkpoint_path"]
        ):
            failures.append(f"checkpoint path mismatch: {key}")
    if observed_checkpoint_keys != set(inventory_by_key):
        failures.append("observed checkpoints differ from startup inventory")
    for key, hashes in observed_checkpoint_hashes.items():
        if len(hashes) != 1:
            failures.append(f"mixed checkpoint hashes within run: {key}")

    result_by_id = {str(row["unit_id"]): row for row in results}
    for unit_id in sorted(expected_units):
        row = result_by_id.get(unit_id)
        if row is None:
            continue
        condition = str(row["condition"])
        if condition == "cf" and not truthy(row["exact_cf_per_channel_conservation"]):
            failures.append(f"CF conservation failed: {unit_id}")
        if condition == "matched" and not truthy(
            row["exact_matched_pbj_per_channel_counts"]
        ):
            failures.append(f"matched/PBJ channel-count equality failed: {unit_id}")
        if condition == "cf_matched" and not truthy(
            row["exact_cf_matched_pbj_per_channel_counts"]
        ):
            failures.append(f"CF+matched/PBJ channel-count equality failed: {unit_id}")
        if condition == "prebin":
            if not truthy(row["raw_event_count_conserved"]):
                failures.append(f"pre-bin raw event conservation failed: {unit_id}")
            if int(row["raw_events_before"]) != int(row["raw_events_after"]):
                failures.append(f"pre-bin raw event totals differ: {unit_id}")
            if not str(row.get("prebin_uniform_quantile_primitives_sha256", "")):
                failures.append(f"missing pre-bin primitive digest: {unit_id}")
        if condition in POSTBIN_CONDITIONS:
            for field in (
                "pbj_boundary_clipped_active_cells",
                "pbj_active_cells_moved",
                "pbj_binary_collision_loss",
                "pbj_normalized_displacement_primitives_sha256",
                "cf_pass_primitives_sha256",
                "matched_rank_primitives_sha256",
                "operator_primitive_digest_sha256",
            ):
                if str(row.get(field, "")) == "":
                    failures.append(f"missing PBJ/primitive audit {field}: {unit_id}")

    prediction_counts: Counter[str] = Counter()
    if args.save_predictions:
        # Prediction shards are deliberately never concatenated or retained in
        # memory.  Stream every committed shard and re-check its marker/hash.
        for unit_id in sorted(completed):
            _, _, row_count = load_unit_transaction(
                output_dir,
                unit_id,
                config_hash,
                args.save_predictions,
                stream_predictions=True,
            )
            prediction_counts[unit_id] = row_count
        if set(prediction_counts) != expected_units:
            failures.append("prediction units differ from expected units")
        for unit_id, row in result_by_id.items():
            if prediction_counts[unit_id] != int(row["samples"]):
                failures.append(f"prediction row count mismatch: {unit_id}")

    per_seed_metrics, aggregate_metrics, metric_status = (
        build_frozen_primary_metric_tables(results, args)
    )
    write_csv_atomic(
        output_dir / FROZEN_SEED_METRICS_BASENAME, per_seed_metrics
    )
    write_csv_atomic(
        output_dir / FROZEN_AGGREGATE_METRICS_BASENAME, aggregate_metrics
    )
    is_partial_smoke = bool(args.max_folds or args.max_validation_samples)
    is_frozen_scope, formal_incomplete_reasons = frozen_protocol_scope(args)
    if not metric_status["complete"]:
        formal_incomplete_reasons.append("frozen primary metric matrix is incomplete")
    formal_complete = bool(
        not failures
        and not is_partial_smoke
        and is_frozen_scope
        and metric_status["complete"]
    )
    payload = {
        "protocol_id": PROTOCOL_ID,
        "config_hash": config_hash,
        "checked_utc": utc_now(),
        "status": (
            "failed"
            if failures
            else (
                "formal_complete"
                if formal_complete
                else (
                    "partial_smoke_complete"
                    if is_partial_smoke
                    else "configured_scope_complete_nonformal"
                )
            )
        ),
        "is_partial_smoke": is_partial_smoke,
        "frozen_protocol_scope": is_frozen_scope,
        "formal_complete": formal_complete,
        "formal_incomplete_reasons": formal_incomplete_reasons,
        "frozen_primary_metrics": {
            **metric_status,
            "per_seed_path": str(
                (output_dir / FROZEN_SEED_METRICS_BASENAME).resolve()
            ),
            "three_seed_path": str(
                (output_dir / FROZEN_AGGREGATE_METRICS_BASENAME).resolve()
            ),
        },
        "expected_units": len(expected_units),
        "completed_units": len(completed),
        "result_units": len(set(result_ids)),
        "audit_units": len(set(audit_ids)),
        "prediction_units": len(prediction_counts) if args.save_predictions else 0,
        "checks": {
            "all_expected_units_present": set(result_ids) == expected_units,
            "all_audit_units_present": set(audit_ids) == expected_units,
            "cf_exact_conservation": not any(
                "CF conservation" in failure for failure in failures
            ),
            "matched_exact_pbj_channel_counts": not any(
                "matched/PBJ" in failure for failure in failures
            ),
            "cf_matched_exact_pbj_channel_counts": not any(
                "CF+matched/PBJ" in failure for failure in failures
            ),
            "prebin_raw_event_conservation": not any(
                "pre-bin raw event" in failure for failure in failures
            ),
            "checkpoint_inventory_exact": not any(
                "checkpoint" in failure for failure in failures
            ),
            "pbj_and_primitive_audits_present": not any(
                "primitive audit" in failure or "primitive digest" in failure
                for failure in failures
            ),
            "frozen_primary_metrics_complete": metric_status["complete"],
        },
        "failures": failures,
    }
    write_json_atomic(output_dir / INTEGRITY_BASENAME, payload)
    if failures:
        raise AssertionError(
            "Completion integrity failed:\n  - " + "\n  - ".join(failures[:20])
        )
    return payload


def maybe_build_cache_via_subprocess(args: argparse.Namespace) -> None:
    cache_dir = Path(args.stem_cache_dir)
    try:
        validate_stemnist_cache(cache_dir, verify_hash=args.verify_cache_hash)
        return
    except FileNotFoundError:
        pass
    if not args.cache_builder_python:
        raise FileNotFoundError(
            "STEMNIST ICASSP event cache is missing. Build it with base Python:\n"
            f"  python {Path(__file__).name} "
            f"cache-build --data-root \"{args.stem_data_root}\" "
            f"--cache-dir \"{args.stem_cache_dir}\""
        )
    command = [
        str(Path(args.cache_builder_python).resolve()),
        str(Path(__file__).resolve()),
        "cache-build",
        "--data-root",
        args.stem_data_root,
        "--cache-dir",
        args.stem_cache_dir,
    ]
    subprocess.run(command, check=True)


def configure_determinism(deps: RuntimeDeps, deterministic: bool) -> None:
    if deterministic:
        deps.torch.use_deterministic_algorithms(True, warn_only=True)
        if deps.torch.backends.cudnn.is_available():
            deps.torch.backends.cudnn.benchmark = False
            deps.torch.backends.cudnn.deterministic = True


def run_evaluation(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parent
    if "stemnist" in args.datasets:
        # This runs before torch/model imports so a base-Python subprocess can
        # prepare the HDF5-derived portable cache without ABI contamination.
        maybe_build_cache_via_subprocess(args)
    deps = import_runtime_dependencies()
    configure_determinism(deps, args.deterministic)
    device = resolve_device(deps.torch, args.device)
    adapter = OperatorAdapter()
    bundles: dict[str, DatasetBundle] = {}
    if "braille" in args.datasets:
        bundles["braille"] = load_braille_bundle(project_root)
    if "stemnist" in args.datasets:
        bundles["stemnist"] = load_stemnist_bundle(
            Path(args.stem_cache_dir), verify_hash=args.verify_cache_hash
        )
    registry = model_registry(args, project_root)
    selected_entries = [registry[name] for name in args.models]
    for entry in selected_entries:
        if entry.dataset not in args.datasets:
            raise ValueError(
                f"Selected model {entry.registry_id} belongs to excluded dataset "
                f"{entry.dataset}"
            )
    checkpoint_inventory = build_checkpoint_inventory(
        selected_entries, args.seeds, args.max_folds
    )
    checkpoint_inventory_by_key = {
        str(row["inventory_key"]): row for row in checkpoint_inventory
    }
    payload = make_config_payload(
        args, bundles, selected_entries, checkpoint_inventory
    )
    payload["operator_module_sha256"] = adapter.module_sha256
    payload["script_sha256"] = sha256_file(Path(__file__).resolve())
    config_hash = sha256_json(payload)[:20]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completed, results, audits, prediction_rows = verify_resume_state(
        output_dir, config_hash, args
    )
    manifest_path = output_dir / MANIFEST_BASENAME
    if not manifest_path.exists():
        manifest = {
            **payload,
            "config_hash": config_hash,
            "created_utc": utc_now(),
            "output_dir": str(output_dir),
            "environment": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": deps.torch.__version__,
                "snntorch": getattr(deps.stem_module.snn, "__version__", None),
                "device": str(device),
                "device_name": (
                    deps.torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "CPU"
                ),
                "deterministic_algorithms": deps.torch.are_deterministic_algorithms_enabled(),
            },
            "matching_requirement": (
                "PBJ, matched dropout, and CF+matched have exactly equal active-cell "
                "counts for every evaluated sample and channel"
            ),
            "operator_seed_root": (
                "BLAKE2b64(protocol_id, operator_root_v1, dataset, sample_id); "
                "shared across models, folds, split seeds and severities; realization "
                "is passed separately to each operator"
            ),
            "prebin_note": (
                "Raw/pre-bin jitter conserves raw event count and may increase or "
                "decrease post-bin binary occupancy; it is deliberately not loss-matched"
            ),
            "persistence_schema_version": PERSISTENCE_SCHEMA_VERSION,
            "prediction_storage": (
                f"per-unit CSV shards under {UNIT_ROOT_BASENAME}/; each unit is committed "
                f"by an atomic {UNIT_COMMIT_BASENAME} marker; no top-level predictions.csv"
            ),
        }
        write_json_atomic(manifest_path, manifest)
    if not (output_dir / STATE_BASENAME).exists():
        # Establish an empty, resumable transaction before the first expensive
        # unit; a failure in unit one then cannot leave an ambiguous manifest.
        persist_progress(
            output_dir,
            config_hash,
            len(completed),
            prediction_rows,
        )
    specs = condition_specs(args)
    expected_units: set[str] = set()
    for entry in selected_entries:
        bundle = bundles[entry.dataset]
        for split_seed in args.seeds:
            folds = exact_folds(deps, bundle, split_seed)
            if args.max_folds:
                folds = folds[: args.max_folds]
            for fold_zero in range(len(folds)):
                for spec in specs:
                    expected_units.add(
                        unit_identifier(entry, split_seed, fold_zero + 1, spec)
                    )
    planned_units = len(expected_units)
    print(
        f"Protocol={PROTOCOL_ID} config={config_hash} device={device}; "
        f"planned_units={planned_units}, already_complete={len(completed)}",
        flush=True,
    )
    for entry in selected_entries:
        bundle = bundles[entry.dataset]
        for split_seed in args.seeds:
            folds = exact_folds(deps, bundle, split_seed)
            if args.max_folds:
                folds = folds[: args.max_folds]
            for fold_zero, (_, validation_indices) in enumerate(folds):
                fold = fold_zero + 1
                full_validation_indices = validation_indices.copy()
                if args.max_validation_samples:
                    validation_indices = validation_indices[
                        : args.max_validation_samples
                    ]
                for spec in specs:
                    unit_id = unit_identifier(entry, split_seed, fold, spec)
                    if unit_id in completed:
                        print(f"resume skip {unit_id}", flush=True)
                        continue
                    print(f"evaluate {unit_id} n={len(validation_indices)}", flush=True)
                    outputs = evaluate_unit(
                        deps,
                        adapter,
                        bundle,
                        entry,
                        split_seed,
                        fold,
                        validation_indices,
                        full_validation_indices,
                        spec,
                        args,
                        device,
                        config_hash,
                        str(
                            checkpoint_inventory_by_key[
                                checkpoint_inventory_key(
                                    entry.registry_id, split_seed, fold
                                )
                            ]["checkpoint_sha256"]
                        ),
                    )
                    marker = commit_unit_transaction(
                        output_dir,
                        config_hash,
                        outputs["result"],
                        outputs["audit"],
                        outputs["predictions"],
                        args.save_predictions,
                    )
                    results.append(outputs["result"])
                    audits.append(outputs["audit"])
                    prediction_rows += int(marker["prediction_rows"])
                    completed.add(unit_id)
                    persist_progress(
                        output_dir,
                        config_hash,
                        len(completed),
                        prediction_rows,
                    )
    materialize_derived_tables(output_dir, results, audits)
    integrity = write_completion_integrity(
        output_dir,
        config_hash,
        expected_units,
        completed,
        results,
        audits,
        args,
        checkpoint_inventory,
    )
    print(
        f"Screening complete: {len(completed)}/{planned_units} units; "
        f"integrity={integrity['status']}; outputs={output_dir}",
        flush=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Evaluation-only ICASSP jitter/collision screening on frozen formal "
            "Braille and STEMNIST checkpoints."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    cache_parser = subparsers.add_parser(
        "cache-build",
        help="Build portable STEMNIST raw-event/clean packed cache (no torch import).",
    )
    cache_parser.add_argument(
        "--data-root",
        default=str(
            project_root / "STEMNIST_data" / "extracted" / "STEMNIST Dataset"
        ),
    )
    cache_parser.add_argument(
        "--cache-dir", default=str(project_root / "STEMNIST_data" / "icassp_cache")
    )
    cache_parser.add_argument("--rebuild-cache", action="store_true")

    eval_parser = subparsers.add_parser(
        "eval", help="Evaluate frozen checkpoints; never trains or writes checkpoints."
    )
    eval_parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["braille", "stemnist"],
        default=["braille", "stemnist"],
    )
    all_model_ids = list(DEFAULT_MODEL_IDS)
    eval_parser.add_argument("--models", nargs="+", choices=all_model_ids, default=all_model_ids)
    eval_parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    eval_parser.add_argument(
        "--severities-ms", nargs="+", type=int, default=list(DEFAULT_SEVERITIES_MS)
    )
    eval_parser.add_argument("--realizations", type=int, default=3)
    eval_parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(DEFAULT_CONDITIONS),
        default=list(DEFAULT_CONDITIONS),
    )
    eval_parser.add_argument("--batch-size", type=int, default=64)
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    eval_parser.add_argument(
        "--save-predictions", action=argparse.BooleanOptionalAction, default=True
    )
    eval_parser.add_argument("--resume", action="store_true")
    eval_parser.add_argument(
        "--output-dir",
        default=str(
            project_root
            / "ICASSP20260813"
            / "icassp_jitter_screening_stage_a_v1"
        ),
        help="Dedicated new directory; existing artifacts require --resume.",
    )
    eval_parser.add_argument(
        "--max-folds",
        type=int,
        default=0,
        help="Smoke only: evaluate the first N folds; zero means all 5.",
    )
    eval_parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=0,
        help="Smoke only: truncate each validation fold; zero means full fold.",
    )
    eval_parser.add_argument(
        "--stem-data-root",
        default=str(
            project_root / "STEMNIST_data" / "extracted" / "STEMNIST Dataset"
        ),
    )
    eval_parser.add_argument(
        "--stem-cache-dir",
        default=str(project_root / "STEMNIST_data" / "icassp_cache"),
    )
    eval_parser.add_argument(
        "--cache-builder-python",
        default="",
        help="Optional base Python executable used to build a missing STEMNIST cache.",
    )
    eval_parser.add_argument(
        "--verify-cache-hash", action=argparse.BooleanOptionalAction, default=True
    )
    eval_parser.add_argument(
        "--braille-clean-snn-root",
        default=str(project_root / "ICASSP20260808" / "consistency_formal_v1" / "clean_ce"),
    )
    eval_parser.add_argument(
        "--braille-tcn-root",
        default=str(project_root / "ICASSP20260808" / "taxel_enum_3seed_v2_checkpoints"),
    )
    eval_parser.add_argument(
        "--stem-checkpoint-root",
        default=str(project_root / "stemnist_robustness_checkpoints"),
    )
    args = parser.parse_args(argv)
    if args.command == "eval":
        if len(set(args.datasets)) != len(args.datasets):
            parser.error("--datasets contains duplicates")
        if len(set(args.models)) != len(args.models):
            parser.error("--models contains duplicates")
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("--seeds contains duplicates")
        if len(set(args.conditions)) != len(args.conditions):
            parser.error("--conditions contains duplicates")
        if any(seed < 0 for seed in args.seeds):
            parser.error("--seeds must be non-negative")
        if any(value not in DEFAULT_SEVERITIES_MS for value in args.severities_ms):
            parser.error("--severities-ms must be selected from 25 50 75")
        if len(set(args.severities_ms)) != len(args.severities_ms):
            parser.error("--severities-ms contains duplicates")
        if args.realizations < 1:
            parser.error("--realizations must be positive")
        if args.batch_size < 1:
            parser.error("--batch-size must be positive")
        if args.max_folds < 0 or args.max_folds > 5:
            parser.error("--max-folds must lie in 0..5")
        if args.max_validation_samples < 0:
            parser.error("--max-validation-samples must be non-negative")
        if args.resume and not Path(args.output_dir).exists():
            parser.error("--resume requires an existing --output-dir")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "cache-build":
        build_stemnist_cache(args)
    elif args.command == "eval":
        run_evaluation(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
