"""Binary, per-domain quantile decisions from validation label priors."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def estimate_priors(validation_rows):
    """Estimate P(positive) per domain from visible binary validation labels."""
    labels = defaultdict(list)
    for row in validation_rows:
        domain = row.get("domain")
        value = row.get("y_true")
        if not isinstance(domain, str) or not domain:
            raise ValueError("Validation rows require nonempty string domains")
        if value not in (0, 1):
            raise ValueError("Prior estimation requires binary y_true labels 0 or 1")
        labels[domain].append(int(value))
    if not labels:
        raise ValueError("No validation rows")
    return {domain: float(np.mean(values)) for domain, values in sorted(labels.items())}


def apply_prior(rows, priors):
    """Apply per-domain threshold quantile(positive_scores, 1 - prior).

    Positive is column 1. Equal-to-threshold scores predict positive, so ties
    and finite batch sizes can prevent an exact positive fraction. Unseen
    domains use the original neutral prior of 0.5. Rates zero and one produce
    all-negative and all-positive decisions respectively. Target labels are
    never accessed. Input row order and other fields are preserved.
    """
    validated_priors = {}
    for domain, value in priors.items():
        try:
            rate = float(value)
        except (TypeError, ValueError):
            raise ValueError("Priors must be finite rates in [0, 1]") from None
        if not np.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("Priors must be finite rates in [0, 1]")
        validated_priors[domain] = rate
    output = [dict(row) for row in rows]
    if not output:
        raise ValueError("No target rows")
    domains = defaultdict(list)
    for index, row in enumerate(output):
        if not isinstance(row.get("domain"), str) or not row["domain"]:
            raise ValueError("Target rows require nonempty string domains")
        try:
            probabilities = np.asarray(row["probabilities"], dtype=float)
        except (KeyError, TypeError, ValueError):
            raise ValueError("Target rows require binary probabilities") from None
        if (probabilities.shape != (2,) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or np.any(probabilities > 1)
                or not np.isclose(probabilities.sum(), 1.0, atol=1e-5, rtol=0)):
            raise ValueError("Target probabilities must be two finite values summing to one")
        domains[row["domain"]].append((index, float(probabilities[1])))
    for domain, items in domains.items():
        rate = validated_priors.get(domain, 0.5)
        threshold = float(np.quantile([score for _, score in items], 1 - rate)) if 0 < rate < 1 else None
        for index, score in items:
            output[index]["y_pred"] = int(rate == 1 or (rate > 0 and score >= threshold))
    return output
