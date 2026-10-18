"""Select a small rank-fusion member set using visible validation only."""
from __future__ import annotations

import numpy as np

from .fusion import rank_fusion
from .metrics import label_metrics


def _index(rows):
    indexed = {}
    for row in rows:
        key = row.get("id")
        if not isinstance(key, str) or not key:
            raise ValueError("Validation rows require nonempty string IDs")
        if key in indexed:
            raise ValueError(f"Duplicate validation ID: {key}")
        indexed[key] = row
    if not indexed:
        raise ValueError("Empty validation member rows")
    return indexed


def _score(maps, ids, label_names):
    base = maps[0]
    for key in ids:
        reference = (base[key]["y_true"], base[key]["domain"])
        if any((member[key]["y_true"], member[key]["domain"]) != reference for member in maps[1:]):
            raise ValueError(f"Validation labels/domains disagree for ID: {key}")
    score_maps = [{key: member[key]["probabilities"] for key in ids} for member in maps]
    selected, scores = rank_fusion(score_maps, ids=ids)
    if scores.shape[1] != len(label_names):
        raise ValueError("Validation probability class count does not match label_names")
    metric = label_metrics([base[key]["y_true"] for key in selected], scores.argmax(axis=1),
                           [base[key]["domain"] for key in selected], label_names, scores)
    return metric["primary"]


def select_members(validation_units, label_names, candidate_sizes=(3, 4), min_shared=10, exclude_repeat=None):
    """Rank members individually, compare top-k candidates, and choose on V.

    Each unit is ``{repeat, fold, members: {name: validation_rows}}``. A member's
    score is the mean of its single-member rank-fusion primary across available
    units. Candidate sets contain the top-ranked members and are compared on
    units with all candidate members and at least ``min_shared`` shared IDs.
    Scores average units equally, matching the selection rule. Ties prefer
    fewer members and then lexical member names. If no requested size fits,
    the available member set is used. Units from ``exclude_repeat`` are never
    used, including their labels. There is deliberately no test-data input.
    """
    if isinstance(min_shared, bool) or not isinstance(min_shared, int) or min_shared < 2:
        raise ValueError("min_shared must be an integer of at least two")
    sizes = list(candidate_sizes)
    if not sizes or any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in sizes):
        raise ValueError("candidate_sizes must contain positive integers")
    if label_names is None or isinstance(label_names, str):
        raise ValueError("Explicit ordered label_names are required")
    names = list(label_names)
    units, seen = [], set()
    for raw in validation_units:
        if raw["repeat"] == exclude_repeat:
            continue
        key = (raw["repeat"], raw["fold"])
        if any(isinstance(value, bool) or not isinstance(value, int) for value in key):
            raise ValueError("Validation repeat and fold must be integers")
        if key in seen:
            raise ValueError(f"Duplicate validation unit: {key}")
        seen.add(key)
        members = raw["members"]
        if not members or any(not isinstance(name, str) or not name for name in members):
            raise ValueError("Validation units require named members")
        units.append((key, {name: _index(rows) for name, rows in members.items()}))
    if not units:
        raise ValueError("No eligible validation units")
    units.sort(key=lambda item: item[0])
    all_members = sorted({name for _, members in units for name in members})
    individual = {}
    for name in all_members:
        values = [_score([members[name]], sorted(members[name]), names)
                  for _, members in units if name in members]
        individual[name] = float(np.mean(values))
    ranked = sorted(all_members, key=lambda name: (-individual[name], name))
    candidates = {tuple(sorted(ranked[:size])) for size in sizes if size <= len(ranked)}
    if not candidates:
        candidates = {tuple(sorted(ranked))}
    candidate_scores = []
    for candidate in sorted(candidates, key=lambda item: (len(item), item)):
        values = []
        for _, members in units:
            if not set(candidate) <= set(members):
                continue
            maps = [members[name] for name in candidate]
            shared = sorted(set.intersection(*(set(member) for member in maps)))
            if len(shared) >= min_shared:
                values.append(_score(maps, shared, names))
        candidate_scores.append({"members": list(candidate),
                                 "score": float(np.mean(values)) if values else None,
                                 "n_units": len(values)})
    eligible = [item for item in candidate_scores if item["score"] is not None]
    if not eligible:
        raise ValueError("No candidate has enough shared validation records")
    chosen = min(eligible, key=lambda item: (-item["score"], len(item["members"]), item["members"]))
    return {"chosen_members": chosen["members"], "validation_score": chosen["score"],
            "individual_scores": individual, "candidate_scores": candidate_scores}
