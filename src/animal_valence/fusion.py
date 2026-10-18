"""Calibration-free, per-class ordinal rank fusion."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def rank_fusion(score_maps: Sequence[Mapping[str, Sequence[float]]], ids=None):
    """Fuse class scores on shared IDs, returning sorted IDs and an N x C array.

    Each member ranks records independently within each class. Ordinal ranks
    run from zero to one, are averaged across members, and are normalized
    within each record. Equal scores follow sorted ID order (stable ordinal
    ties), matching Python's stable sort. All members must use the same
    declared class order. These batch-dependent scores are not calibrated
    posterior probabilities.

    ``ids`` optionally selects a shared subset; it is also sorted before
    ranking. At least two records and two classes are required. A record with
    zero total rank has no defined normalized score and raises ValueError.
    """
    maps = list(score_maps)
    if not maps or any(not isinstance(scores, Mapping) or not scores for scores in maps):
        raise ValueError("score_maps must contain nonempty member mappings")
    n_classes = None
    checked = []
    for scores in maps:
        member = {}
        for record_id, values in scores.items():
            if not isinstance(record_id, str) or not record_id:
                raise ValueError("Record IDs must be nonempty strings")
            try:
                vector = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError):
                raise ValueError("Class scores must be numeric sequences") from None
            if vector.ndim != 1 or len(vector) < 2 or not np.isfinite(vector).all():
                raise ValueError("Class scores must be finite vectors with at least two classes")
            if n_classes is None:
                n_classes = len(vector)
            if len(vector) != n_classes:
                raise ValueError("All score vectors must have the same class count and order")
            member[record_id] = vector
        checked.append(member)
    shared = set.intersection(*(set(scores) for scores in checked))
    if ids is None:
        selected = sorted(shared)
    else:
        requested = list(ids)
        if any(not isinstance(item, str) or not item for item in requested):
            raise ValueError("Record IDs must be nonempty strings")
        if len(set(requested)) != len(requested):
            raise ValueError("Requested IDs must be unique")
        if not set(requested) <= shared:
            raise ValueError("Requested IDs must occur in every member")
        selected = sorted(requested)
    if len(selected) < 2:
        raise ValueError("Rank fusion requires at least two shared records")
    fused = np.zeros((len(selected), n_classes), dtype=np.float64)
    for scores in checked:
        matrix = np.asarray([scores[key] for key in selected])
        # Stable sorting assigns ordinal positions in sorted-ID order for ties.
        order = np.argsort(matrix, axis=0, kind="stable")
        ranks = np.empty_like(matrix)
        np.put_along_axis(ranks, order, np.arange(len(selected))[:, None], axis=0)
        fused += ranks / (len(selected) - 1)
    fused /= len(checked)
    totals = fused.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        bad = [selected[index] for index in np.flatnonzero(totals[:, 0] <= 0)]
        raise ValueError(f"Zero total rank for records: {bad}")
    return selected, fused / totals
