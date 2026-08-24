"""Condition-informed FMEA decision-support prototype.

The predictive layer estimates an independently defined near-term degradation
event.  It never predicts or reconstructs FMEA risk-priority numbers (RPNs).
"""

from .target import PREDICTION_HORIZON

RANDOM_SEED = 20260824
__version__ = "0.1.0"

__all__ = ["PREDICTION_HORIZON", "RANDOM_SEED"]
