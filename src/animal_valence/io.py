"""Portable, pickle-free feature caches and explicit prediction artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def index_rows(rows, require_labels=False):
    result = {}
    for row in rows:
        key = row.get("id")
        if not isinstance(key, str) or not key or key in result:
            raise ValueError("Rows require unique, nonempty string IDs")
        if not isinstance(row.get("domain"), str) or not row["domain"]:
            raise ValueError("Rows require a nonempty domain")
        if require_labels and (isinstance(row.get("y_true"), bool) or
                               not isinstance(row.get("y_true"), int)):
            raise ValueError("Labelled rows require integer y_true")
        result[key] = row
    if not result:
        raise ValueError("No records supplied")
    return result


class FeatureCache:
    """Load fixed vectors (X) or concatenated frames (X_cat, offsets)."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as archive:
            self.ids = archive["ids"].astype(str).tolist()
            if not self.ids or len(set(self.ids)) != len(self.ids) or any(not key for key in self.ids):
                raise ValueError("Cache IDs must be unique and nonempty")
            self.index = {key: i for i, key in enumerate(self.ids)}
            self.hashes = archive["audio_sha256"].astype(str).tolist() if "audio_sha256" in archive else None
            if self.hashes is not None and (len(self.hashes) != len(self.ids) or any(
                    len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower())
                    for value in self.hashes)):
                raise ValueError("audio_sha256 must contain one SHA256 digest per ID")
            self.sequence = "X_cat" in archive
            if self.sequence:
                data = np.asarray(archive["X_cat"], dtype=np.float32)
                offsets = archive["offsets"]
                if (offsets.ndim != 1 or len(offsets) != len(self.ids) + 1 or
                    not np.issubdtype(offsets.dtype, np.integer) or offsets[0] != 0 or
                    offsets[-1] != len(data) or np.any(np.diff(offsets) <= 0)):
                    raise ValueError("Frame offsets must describe nonempty, contiguous recordings")
                self.offsets = offsets.astype(np.int64)
                if not np.isfinite(data).all():
                    raise ValueError("Sequence features must be finite")
            else:
                data = np.asarray(archive["X"], dtype=np.float32)
                if len(data) != len(self.ids) or np.isinf(data).any():
                    raise ValueError("Vector cache requires one row per ID and no infinite values")
            if data.ndim != 2 or data.shape[1] == 0:
                raise ValueError("Feature data must be a nonempty two-dimensional matrix")
            self.data = data

    def take(self, ids):
        if any(key not in self.index for key in ids):
            raise ValueError("Requested IDs are absent from the feature cache")
        positions = [self.index[key] for key in ids]
        if self.sequence:
            return [self.data[self.offsets[i]:self.offsets[i + 1]].copy() for i in positions]
        return self.data[positions].copy()

    def verify(self, rows):
        """Check supplied audio digests when present; never infer sample labels."""
        for row in rows:
            if row["id"] not in self.index:
                raise ValueError(f"Missing feature ID: {row['id']}")
            if row.get("audio_sha256"):
                if self.hashes is None or row["audio_sha256"].lower() != self.hashes[self.index[row["id"]]].lower():
                    raise ValueError(f"Audio digest mismatch: {row['id']}")


def load_features(paths, rows):
    """Join vector blocks, or aligned sequence streams, by recording ID."""
    caches = [FeatureCache(path) for path in paths]
    if not caches or len({cache.sequence for cache in caches}) != 1:
        raise ValueError("Supply vector caches or sequence caches, without mixing their types")
    for cache in caches:
        cache.verify(rows)
    # Cross-cache content identity is checked whenever both sides supply hashes.
    for row in rows:
        hashes = {cache.hashes[cache.index[row["id"]]] for cache in caches if cache.hashes is not None}
        if len(hashes) > 1:
            raise ValueError(f"Feature blocks refer to different audio: {row['id']}")
    arrays = [cache.take([row["id"] for row in rows]) for cache in caches]
    if not caches[0].sequence:
        return np.concatenate(arrays, axis=1)
    output = []
    for streams in zip(*arrays):
        if len({len(stream) for stream in streams}) != 1:
            raise ValueError("Dual-stream MIL requires aligned frame grids; resample before caching")
        output.append(np.concatenate(streams, axis=1))
    return output


def check_split(train_rows, validation_rows):
    """Reject shared IDs, known audio digests, and known groups across L/V."""
    left, right = index_rows(train_rows, True), index_rows(validation_rows, True)
    if left.keys() & right.keys():
        raise ValueError("Training and validation IDs overlap")
    for key in ("audio_sha256", "group"):
        a = {r[key] for r in train_rows if r.get(key)}
        b = {r[key] for r in validation_rows if r.get(key)}
        if a & b:
            raise ValueError(f"Training and validation share {key}")
