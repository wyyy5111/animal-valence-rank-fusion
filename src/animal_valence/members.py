"""Supervised vector members fitted on L and selected on visible V.

This module does not import PyTorch. SVM native decisions select the member;
its probability output is retained separately for per-class rank fusion.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np

from .metrics import label_metrics


def _matrix(values, *, dimension=None, name="features"):
    x = np.asarray(values, dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] == 0 or x.shape[0] == 0
            or (dimension is not None and x.shape[1] != dimension)
            or np.isinf(x).any()):
        raise ValueError(f"{name} must be a nonempty N x D matrix without infinite values")
    return x


@dataclass
class LOnlyNormalizer:
    """Column medians, means and population standard deviations fitted on L."""

    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values):
        x = _matrix(values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(x, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        filled = np.where(np.isnan(x), median, x)
        std = filled.std(axis=0)
        return cls(median, filled.mean(axis=0), np.where(std > 1e-8, std, 1.0))

    def transform(self, values):
        x = _matrix(values, dimension=len(self.mean)).astype(self.mean.dtype, copy=False)
        result = (np.where(np.isnan(x), self.median, x) - self.mean) / self.std
        if not np.isfinite(result).all():
            raise ValueError("Normalization produced nonfinite features")
        return result


def _labels(values, n, count, name):
    raw = np.asarray(values)
    try:
        y = raw.astype(np.int64)
    except (ValueError, TypeError, OverflowError):
        raise ValueError(f"{name} must contain integer class indices") from None
    if (raw.shape != (n,) or not np.array_equal(raw, y)
            or np.any((y < 0) | (y >= count))):
        raise ValueError(f"{name} must align with features and declared classes")
    return y


def prepare_labels(y_train, n_train, y_val, n_val, label_names):
    """Validate a shared, explicit integer class order for L and V."""
    raw = np.asarray(y_train)
    if label_names is None:
        if raw.ndim != 1 or len(raw) != n_train or not len(raw):
            raise ValueError("Training labels must align with features")
        try:
            numeric = raw.astype(np.int64)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Training labels must be integer class indices") from None
        if not np.array_equal(raw, numeric) or numeric.min() < 0:
            raise ValueError("Training labels must be nonnegative integer class indices")
        names = tuple(str(i) for i in range(int(numeric.max()) + 1))
    else:
        if isinstance(label_names, str):
            raise ValueError("label_names must be an ordered sequence")
        names = tuple(str(value) for value in label_names)
    if len(names) < 2 or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("At least two unique nonempty label names are required")
    train = _labels(y_train, n_train, len(names), "y_train")
    val = _labels(y_val, n_val, len(names), "y_val")
    if set(train.tolist()) != set(range(len(names))):
        raise ValueError("Every declared class needs at least one training example")
    return train, val, names


def _domains(values, n, name):
    if values is None:
        return np.asarray(["all"] * n)
    if isinstance(values, str) or len(values) != n:
        raise ValueError(f"{name} must align with examples")
    return np.asarray([str(value) for value in values])


def training_weights(y, domains=None, groups=None):
    """Equal observed domain/class cells, then known groups within each cell.

    Missing group IDs denote independent recordings, never invented animals.
    """
    labels = np.asarray(y)
    d = _domains(domains, len(labels), "train_domains")
    if groups is not None and (isinstance(groups, str) or len(groups) != len(labels)):
        raise ValueError("train_groups must align with examples")
    cells = defaultdict(set)
    pairs = []
    for i, (domain, label) in enumerate(zip(d, labels)):
        group = None if groups is None else groups[i]
        group_key = ("recording", i) if group in (None, "") else ("group", str(group))
        cell = (str(domain), int(label))
        cells[cell].add(group_key)
        pairs.append((cell, group_key))
    counts = Counter(pairs)
    weights = np.asarray([1 / (len(cells[cell]) * counts[(cell, group)])
                          for cell, group in pairs], dtype=np.float64)
    return weights / weights.mean()


@dataclass
class VectorMember:
    estimator: object
    normalizer: LOnlyNormalizer
    label_names: tuple[str, ...]
    metadata: dict

    def predict_proba(self, features):
        probabilities = self.estimator.predict_proba(self.normalizer.transform(features))
        result = np.zeros((len(probabilities), len(self.label_names)), dtype=np.float64)
        result[:, self.estimator.classes_.astype(int)] = probabilities
        return result

    def predict(self, features):
        """Keep native SVM decisions distinct from probability argmax."""
        return np.asarray(self.estimator.predict(self.normalizer.transform(features)), dtype=np.int64)

    def save(self, path):
        """Serialize the fitted member with joblib; load only trusted files."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


def load_vector(path):
    """Load a trusted joblib artifact (joblib files can execute Python code)."""
    import joblib
    member = joblib.load(path)
    if not isinstance(member, VectorMember):
        raise ValueError("Artifact is not a VectorMember")
    return member


def fit_vector(x_train, y_train, x_val, y_val, *, train_domains=None,
               val_domains=None, train_groups=None, label_names=None,
               kind="svm", c_values=(0.1, 1.0, 10.0), seed=2026):
    """Fit L-only candidates; select by V domain-equal UAR, then macro-F1.

    C ties prefer the earlier, smaller C. Hidden examples and test labels are
    absent from this interface. Supported kinds are RBF ``svm`` and ``linear``
    logistic regression (``logistic`` is an alias for ``linear``).
    """
    if kind == "logistic":
        kind = "linear"
    if kind not in {"svm", "linear"}:
        raise ValueError("kind must be svm, linear or logistic")
    train_x = _matrix(x_train, name="x_train")
    val_x = _matrix(x_val, dimension=train_x.shape[1], name="x_val")
    train_y, val_y, names = prepare_labels(y_train, len(train_x), y_val, len(val_x), label_names)
    domains = _domains(val_domains, len(val_x), "val_domains")
    weights = training_weights(train_y, train_domains, train_groups)
    cs = sorted(set(float(value) for value in c_values))
    if not cs or any(not np.isfinite(value) or value <= 0 for value in cs):
        raise ValueError("c_values must contain finite positive values")
    normalizer = LOnlyNormalizer.fit(train_x)
    train_scaled = normalizer.transform(train_x)
    val_scaled = normalizer.transform(val_x)
    trials, best = [], None
    for rank, c in enumerate(cs):
        if kind == "svm":
            from sklearn.svm import SVC
            estimator = SVC(C=c, kernel="rbf", gamma="scale", probability=True, random_state=seed)
        else:
            from sklearn.linear_model import LogisticRegression
            estimator = LogisticRegression(C=c, solver="lbfgs", max_iter=2000, random_state=seed)
        estimator.fit(train_scaled, train_y, sample_weight=weights)
        predictions = estimator.predict(val_scaled).astype(np.int64)
        probabilities = estimator.predict_proba(val_scaled)
        score = label_metrics(val_y, predictions, domains=domains, label_names=names,
                              probabilities=probabilities)
        trials.append({"C": c, "validation_primary": score["primary"],
                       "validation_macro_f1": score["macro_f1"],
                       "validation_cross_entropy": score["cross_entropy"]})
        key = (score["primary"], score["macro_f1"], -rank)
        if best is None or key > best[0]:
            best = (key, estimator, c)
    metadata = {"kind": kind, "seed": int(seed), "selected_C": best[2],
                "selection_trials": trials,
                "selection_rule": "visible-V domain-equal UAR, macro-F1, then smaller C",
                "decision_rule": "native_predict" if kind == "svm" else "probability_argmax",
                "preprocessing_fit": "L only: median imputation and standardization",
                "sample_weights": "equal domain-class cells, then groups within each cell",
                "label_names": list(names), "n_train": len(train_x), "n_validation": len(val_x),
                "input_dim": int(train_x.shape[1])}
    return VectorMember(best[1], normalizer, names, metadata)
