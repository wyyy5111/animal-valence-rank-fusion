"""Core methods for low-label animal vocalization valence recognition."""

from .fusion import rank_fusion
from .metrics import label_metrics, combine_oof, summarize_repeats
from .selection import select_members
from .prior import estimate_priors, apply_prior

__version__ = "1.0.0"
__all__ = ["rank_fusion", "label_metrics", "combine_oof", "summarize_repeats",
           "select_members", "estimate_priors", "apply_prior"]
