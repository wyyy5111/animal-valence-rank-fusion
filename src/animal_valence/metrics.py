"""Declared-class recording metrics and complete-repeat aggregation."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _indices(values, name, n_classes):
    raw = np.asarray(values)
    try:
        result = raw.astype(np.int64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be integer class indices") from None
    if (raw.ndim != 1 or not len(raw) or not np.array_equal(raw, result)
            or np.any((result < 0) | (result >= n_classes))):
        raise ValueError(f"{name} must be nonempty integer indices within declared classes")
    return result


def _basic(y, pred, names):
    count = len(names)
    confusion = np.zeros((count, count), dtype=np.int64)
    np.add.at(confusion, (y, pred), 1)
    support = confusion.sum(axis=1)
    true_positive = np.diag(confusion).astype(float)
    recall = np.divide(true_positive, support, out=np.zeros(count), where=support > 0)
    denominator = support + confusion.sum(axis=0)
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros(count), where=denominator > 0)
    return {
        "n": len(y), "accuracy": float(np.mean(y == pred)),
        "uar": float(recall.mean()), "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()), "confusion": confusion.tolist(),
        "label_names": list(names), "support": support.tolist(),
        "per_class_recall": dict(zip(names, recall.tolist())),
        "missing_classes": [names[index] for index in range(count) if support[index] == 0],
        "absent_class_policy": "zero recall/F1; retain all declared classes in macro denominator",
    }


def label_metrics(y_true, y_pred, domains=None, label_names=None, probabilities=None):
    """Report explicit decisions, keeping absent classes in macro denominators.

    ``primary`` is macro-domain UAR: compute UAR with all declared classes
    inside each domain, then average domains equally. Optional probabilities
    provide calibration diagnostics; they never replace ``y_pred``.
    """
    if label_names is None or isinstance(label_names, str):
        raise ValueError("Explicit ordered label_names are required")
    names = [str(name) for name in label_names]
    if len(names) < 2 or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("label_names must contain at least two unique nonempty classes")
    y = _indices(y_true, "y_true", len(names))
    pred = _indices(y_pred, "y_pred", len(names))
    if pred.shape != y.shape:
        raise ValueError("Prediction and label vectors must align")
    d = np.asarray([str(domain) for domain in domains] if domains is not None else ["all"] * len(y))
    if d.shape != y.shape:
        raise ValueError("domains must align with labels")
    result = _basic(y, pred, names)
    result["per_domain"] = {
        domain: _basic(y[d == domain], pred[d == domain], names) for domain in sorted(set(d))
    }
    result["macro_domain_uar"] = float(np.mean([m["uar"] for m in result["per_domain"].values()]))
    result["primary"] = result["macro_domain_uar"]
    if probabilities is not None:
        try:
            p = np.asarray(probabilities, dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("Probabilities must be a numeric matrix") from None
        if (p.shape != (len(y), len(names)) or not np.isfinite(p).all()
                or np.any(p < 0) or np.any(p > 1)
                or not np.allclose(p.sum(axis=1), 1.0, atol=1e-5, rtol=0)):
            raise ValueError("Probabilities must be finite N x C values in [0, 1] with row sums one")
        probability_pred = p.argmax(axis=1)
        argmax_metrics = _basic(y, probability_pred, names)
        result["proba_argmax_accuracy"] = argmax_metrics["accuracy"]
        result["proba_argmax_uar"] = argmax_metrics["uar"]
        result["cross_entropy"] = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
        confidence = p.max(axis=1)
        correct = probability_pred == y
        ece = 0.0
        edges = np.linspace(0, 1, 11)
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (confidence >= low) & ((confidence < high) if high < 1 else (confidence <= high))
            if mask.any():
                ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
        result["ece_10_bins"] = ece
    return result


def combine_oof(fold_rows, expected_ids, expected_folds=3):
    """Join one repeat's folds only after exact, duplicate-free ID coverage."""
    if isinstance(expected_folds, bool) or not isinstance(expected_folds, int) or expected_folds < 1:
        raise ValueError("expected_folds must be a positive integer")
    folds = list(fold_rows)
    if len(folds) != expected_folds:
        raise ValueError(f"Expected {expected_folds} folds, got {len(folds)}")
    expected_list = list(expected_ids)
    if not expected_list or any(not isinstance(key, str) or not key for key in expected_list):
        raise ValueError("expected_ids must contain nonempty string IDs")
    expected = set(expected_list)
    if len(expected) != len(expected_list):
        raise ValueError("Duplicate expected IDs")
    rows, seen = [], set()
    for fold in folds:
        items = fold.get("predictions", []) if isinstance(fold, Mapping) else fold
        for raw in items:
            row = dict(raw)
            record_id = row.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError("OOF rows require nonempty string IDs")
            if record_id in seen:
                raise ValueError(f"OOF duplicate ID: {record_id}")
            seen.add(record_id)
            rows.append(row)
    if seen != expected:
        raise ValueError(f"OOF coverage mismatch: missing={len(expected - seen)}, extra={len(seen - expected)}")
    return sorted(rows, key=lambda row: row["id"])


def summarize_repeats(repeat_metrics):
    """Mean and sample standard deviation across repeat-level metrics."""
    repeats = list(repeat_metrics)
    if not repeats:
        raise ValueError("No repeats")
    result = {"n_repeats": len(repeats), "scope": "variation across complete repeats; not independent folds"}
    for key in ("primary", "macro_domain_uar", "uar", "balanced_accuracy", "macro_f1", "accuracy"):
        values = [float(item[key]) for item in repeats if item.get(key) is not None]
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite repeat metric: {key}")
        result[key] = ({"mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        "n": len(values)} if values else None)
    return result
