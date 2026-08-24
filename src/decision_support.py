"""Transparent condition alert bands and non-mutating FMEA escalation logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .fmea import (
    RATING_COLUMNS,
    STATIC_RPN_COLUMN,
    static_priority_category,
    validate_static_fmea,
)


@dataclass(frozen=True)
class AlertBands:
    """Predeclared probability cutoffs; replace with validated policy if needed."""

    moderate: float = 0.20
    high: float = 0.50
    critical: float = 0.80

    def __post_init__(self) -> None:
        values = (self.moderate, self.high, self.critical)
        if not all(np.isfinite(values)) or not (0 < self.moderate < self.high < self.critical <= 1):
            raise ValueError(
                "Alert cutoffs must satisfy 0 < moderate < high < critical <= 1"
            )


def condition_alert_level(probability: float, bands: AlertBands | None = None) -> str:
    """Map the final retained near-term event probability to a reviewable band."""

    thresholds = bands or AlertBands()
    value = float(probability)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Condition probability must lie in [0, 1]")
    if value >= thresholds.critical:
        return "CRITICAL"
    if value >= thresholds.high:
        return "HIGH"
    if value >= thresholds.moderate:
        return "MODERATE"
    return "LOW"


def _align_probabilities(
    fmea: pd.DataFrame,
    probabilities: float | Sequence[float] | Mapping[str, float] | pd.Series,
) -> np.ndarray:
    if np.isscalar(probabilities):
        values = np.full(len(fmea), float(probabilities), dtype=float)
    elif isinstance(probabilities, Mapping):
        values = fmea["Failure_Mode"].map(probabilities).to_numpy(dtype=float)
    elif isinstance(probabilities, pd.Series) and set(fmea["Failure_Mode"]).issubset(
        probabilities.index
    ):
        values = fmea["Failure_Mode"].map(probabilities).to_numpy(dtype=float)
    else:
        values = np.asarray(probabilities, dtype=float).reshape(-1)
    if values.size != len(fmea):
        raise ValueError("Provide one probability per FMEA row, or one shared probability")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Condition probabilities must be finite and lie in [0, 1]")
    return values


def _urgency(static_priority: str, severity: int, alert: str) -> tuple[str, bool, str]:
    if alert == "CRITICAL" and severity >= 9:
        return (
            "IMMEDIATE_ENGINEERING_REVIEW",
            True,
            "Critical condition evidence combined with high engineering severity",
        )
    if alert == "CRITICAL":
        return (
            "PROMPT_INSPECTION",
            True,
            "Critical near-term condition probability",
        )
    if alert == "HIGH" and severity >= 7:
        return (
            "ACCELERATED_INSPECTION",
            True,
            "High condition alert on a severity-rated failure-mode scenario",
        )
    if alert == "HIGH":
        return "ENGINEERING_REVIEW", True, "High near-term condition probability"
    if alert == "MODERATE":
        return (
            "ENHANCED_MONITORING",
            False,
            "Moderate condition evidence; retain static scores and increase observation",
        )
    return (
        f"{static_priority}_STATIC_PLAN",
        False,
        "Low condition evidence; retain static engineering priority",
    )


def condition_informed_prioritization(
    static_fmea: pd.DataFrame,
    probabilities: float | Sequence[float] | Mapping[str, float] | pd.Series,
    *,
    bands: AlertBands | None = None,
) -> pd.DataFrame:
    """Add condition urgency to a copy without changing S/O/D or Static_RPN.

    The result deliberately contains no dynamic RPN.  Condition information
    changes timing/urgency, while engineering severity and all static ratings
    remain intact and separately visible.
    """

    validate_static_fmea(static_fmea)
    thresholds = bands or AlertBands()
    protected_columns = [*RATING_COLUMNS, STATIC_RPN_COLUMN]
    protected_before = static_fmea[protected_columns].copy(deep=True)
    output = static_fmea.copy(deep=True)
    condition_probability = _align_probabilities(output, probabilities)
    output["Condition_Probability"] = condition_probability
    output["Condition_Alert"] = [
        condition_alert_level(value, thresholds) for value in condition_probability
    ]
    if "Static_Priority" not in output.columns:
        output["Static_Priority"] = [
            static_priority_category(severity, rpn)
            for severity, rpn in zip(output["Severity"], output[STATIC_RPN_COLUMN])
        ]

    decisions = [
        _urgency(str(priority), int(severity), str(alert))
        for priority, severity, alert in zip(
            output["Static_Priority"], output["Severity"], output["Condition_Alert"]
        )
    ]
    output["Recommended_Urgency"] = [decision[0] for decision in decisions]
    output["Condition_Escalated"] = [decision[1] for decision in decisions]
    output["Escalation_Reason"] = [decision[2] for decision in decisions]

    if not static_fmea[protected_columns].equals(protected_before):
        raise AssertionError("Static input FMEA was mutated")
    if not output[protected_columns].reset_index(drop=True).equals(
        protected_before.reset_index(drop=True)
    ):
        raise AssertionError("Condition integration changed protected static ratings")
    if any("dynamic_rpn" in column.lower() for column in output.columns):
        raise AssertionError("Dynamic RPN construction is prohibited")
    return output


def condition_policy_change_summary(prioritized_fmea: pd.DataFrame) -> dict[str, int]:
    """Count transparent priority changes relative to the frozen static policy."""

    required = {"Condition_Alert", "Condition_Escalated", "Recommended_Urgency"}
    missing = sorted(required.difference(prioritized_fmea.columns))
    if missing:
        raise ValueError(f"Missing condition-priority columns: {missing}")
    return {
        "fmea_items": int(len(prioritized_fmea)),
        "escalated_items": int(prioritized_fmea["Condition_Escalated"].sum()),
        "high_or_critical_alert_items": int(
            prioritized_fmea["Condition_Alert"].isin(["HIGH", "CRITICAL"]).sum()
        ),
    }


__all__ = [
    "AlertBands",
    "condition_alert_level",
    "condition_informed_prioritization",
    "condition_policy_change_summary",
]
