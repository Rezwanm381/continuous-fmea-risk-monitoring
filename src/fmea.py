"""Narrow illustrative engineering FMEA kept separate from predictive outcomes.

C-MAPSS FD001 supplies degradation trajectories and end-of-life timing, not
ground-truth failure-mode labels.  The table below is therefore explicitly an
``ENGINEERING_SCENARIO_ASSUMPTION``.  Static RPN is a secondary prioritization
context only and is never an ML target or model input.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


SCENARIO_BASIS = "ENGINEERING_SCENARIO_ASSUMPTION"
RATING_COLUMNS = ("Severity", "Occurrence", "Detection")
STATIC_RPN_COLUMN = "Static_RPN"
REQUIRED_FMEA_COLUMNS = (
    "Failure_Mode",
    "Effect",
    "Potential_Cause",
    "Current_Control",
    *RATING_COLUMNS,
    STATIC_RPN_COLUMN,
    "Engineering_Action",
    "Scenario_Basis",
)


def _validated_rating(values: Sequence[int] | pd.Series, name: str) -> np.ndarray:
    numeric = np.asarray(values, dtype=float)
    if numeric.ndim != 1 or numeric.size == 0 or not np.isfinite(numeric).all():
        raise ValueError(f"{name} ratings must be a non-empty finite sequence")
    if ((numeric < 1) | (numeric > 10)).any() or not np.equal(
        numeric, np.floor(numeric)
    ).all():
        raise ValueError(f"{name} ratings must be whole numbers from 1 to 10")
    return numeric.astype(int)


def calculate_static_rpn(
    severity: Sequence[int] | pd.Series,
    occurrence: Sequence[int] | pd.Series,
    detection: Sequence[int] | pd.Series,
) -> np.ndarray:
    """Calculate the conventional secondary S × O × D field."""

    severity_values = _validated_rating(severity, "Severity")
    occurrence_values = _validated_rating(occurrence, "Occurrence")
    detection_values = _validated_rating(detection, "Detection")
    if not (
        severity_values.size == occurrence_values.size == detection_values.size
    ):
        raise ValueError("Severity, Occurrence, and Detection must align")
    return severity_values * occurrence_values * detection_values


def add_static_rpn(fmea: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with Static_RPN; input ratings are never modified."""

    missing = sorted(set(RATING_COLUMNS).difference(fmea.columns))
    if missing:
        raise ValueError(f"Missing FMEA rating columns: {missing}")
    output = fmea.copy(deep=True)
    output[STATIC_RPN_COLUMN] = calculate_static_rpn(
        output["Severity"], output["Occurrence"], output["Detection"]
    )
    return output


def static_priority_category(severity: int, static_rpn: int) -> str:
    """Transparent illustrative priority tier; not an industry-standard cutoff."""

    if not 1 <= int(severity) <= 10 or not 1 <= int(static_rpn) <= 1_000:
        raise ValueError("severity/static_rpn are outside valid FMEA ranges")
    if severity >= 9 or static_rpn >= 250:
        return "HIGH"
    if severity >= 7 or static_rpn >= 120:
        return "MODERATE"
    return "ROUTINE"


def build_static_fmea_scenario() -> pd.DataFrame:
    """Create a three-mode illustrative turbofan degradation FMEA scenario."""

    frame = pd.DataFrame.from_records(
        [
            {
                "Failure_Mode": "HPC flow-capacity degradation",
                "Effect": "Reduced core-flow capability and available operating margin",
                "Potential_Cause": "Illustrative high-pressure-compressor flow-capacity loss",
                "Current_Control": "Scheduled HPC inspection and multichannel trend review",
                "Severity": 8,
                "Occurrence": 5,
                "Detection": 5,
                "Engineering_Action": "Review condition trend and plan focused inspection",
                "Scenario_Basis": SCENARIO_BASIS,
            },
            {
                "Failure_Mode": "HPC efficiency degradation",
                "Effect": "Reduced compression efficiency with elevated operating burden",
                "Potential_Cause": "Illustrative high-pressure-compressor efficiency loss",
                "Current_Control": "Temperature/pressure trending and scheduled HPC inspection",
                "Severity": 9,
                "Occurrence": 4,
                "Detection": 5,
                "Engineering_Action": "Prioritize engineering review when condition alert rises",
                "Scenario_Basis": SCENARIO_BASIS,
            },
            {
                "Failure_Mode": "Combined HPC performance-margin degradation",
                "Effect": "Progressive loss of simulated engine health margin",
                "Potential_Cause": "Illustrative coupled HPC flow and efficiency deterioration",
                "Current_Control": "Scheduled review of correlated condition trajectories",
                "Severity": 7,
                "Occurrence": 4,
                "Detection": 4,
                "Engineering_Action": "Increase inspection urgency if condition evidence persists",
                "Scenario_Basis": SCENARIO_BASIS,
            },
        ]
    )
    frame = add_static_rpn(frame)
    frame["Static_Priority"] = [
        static_priority_category(severity, rpn)
        for severity, rpn in zip(frame["Severity"], frame[STATIC_RPN_COLUMN])
    ]
    validate_static_fmea(frame)
    return frame


def validate_static_fmea(fmea: pd.DataFrame) -> None:
    """Validate static scores and reject any dynamic-RPN construction."""

    missing = sorted(set(REQUIRED_FMEA_COLUMNS).difference(fmea.columns))
    if missing:
        raise ValueError(f"Missing required FMEA columns: {missing}")
    forbidden = [column for column in fmea.columns if "dynamic_rpn" in column.lower()]
    if forbidden:
        raise ValueError(f"Dynamic RPN fields are prohibited: {forbidden}")
    expected = calculate_static_rpn(
        fmea["Severity"], fmea["Occurrence"], fmea["Detection"]
    )
    actual = pd.to_numeric(fmea[STATIC_RPN_COLUMN], errors="raise").to_numpy()
    if not np.array_equal(expected, actual):
        raise ValueError("Static_RPN must equal Severity × Occurrence × Detection")
    if not (fmea["Scenario_Basis"] == SCENARIO_BASIS).all():
        raise ValueError("Illustrative FMEA rows must disclose their scenario basis")


__all__ = [
    "REQUIRED_FMEA_COLUMNS",
    "RATING_COLUMNS",
    "SCENARIO_BASIS",
    "STATIC_RPN_COLUMN",
    "add_static_rpn",
    "build_static_fmea_scenario",
    "calculate_static_rpn",
    "static_priority_category",
    "validate_static_fmea",
]
