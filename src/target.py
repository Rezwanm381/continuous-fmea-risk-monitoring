"""Independent future-event target construction for C-MAPSS.

The primary target is one when an engine reaches its observed or reported
end-of-life cycle within the next 30 cycles (inclusive), and zero otherwise.
Remaining useful life is used only to construct/evaluate this outcome; it is
explicitly prohibited from the predictor matrix.
"""

from __future__ import annotations

from collections.abc import Iterable
import re

import numpy as np
import pandas as pd

from .data_loader import CYCLE_COLUMN, UNIT_COLUMN


PREDICTION_HORIZON = 30
RUL_COLUMN = "remaining_useful_life"
EVENT_CYCLE_COLUMN = "event_cycle"
TARGET_COLUMN = "event_within_horizon"

_FORBIDDEN_EXACT = {
    UNIT_COLUMN,
    RUL_COLUMN,
    "rul",
    "final_rul",
    EVENT_CYCLE_COLUMN,
    TARGET_COLUMN,
    "target",
    "label",
    "failure_flag",
    "future_failure",
    "severity",
    "occurrence",
    "detection",
    "static_rpn",
    "rpn",
}
_FORBIDDEN_TOKENS = {
    "s",
    "o",
    "d",
    "sev",
    "occ",
    "det",
    "sod",
    "rul",
    "target",
    "label",
    "event",
    "failure",
    "future",
    "severity",
    "occurrence",
    "detection",
    "rpn",
}
_FORBIDDEN_COMPACT_PHRASES = {
    "rul",
    "target",
    "label",
    "event",
    "failure",
    "future",
    "severity",
    "occurrence",
    "detection",
    "rpn",
    "remainingusefullife",
    "finalrul",
    "eventcycle",
    "eventwithinhorizon",
    "failureflag",
    "futurefailure",
    "staticrpn",
}


def _validate_horizon(horizon: int) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise TypeError("horizon must be an integer number of cycles")
    if int(horizon) <= 0:
        raise ValueError("horizon must be positive")
    return int(horizon)


def _validate_trajectory_keys(frame: pd.DataFrame) -> None:
    required = {UNIT_COLUMN, CYCLE_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing trajectory columns: {missing}")
    if frame.empty:
        raise ValueError("Trajectory frame cannot be empty")
    if frame[[UNIT_COLUMN, CYCLE_COLUMN]].isna().any().any():
        raise ValueError("Trajectory keys cannot be missing")
    if frame.duplicated([UNIT_COLUMN, CYCLE_COLUMN]).any():
        raise ValueError("Duplicate (unit_id, cycle) rows are not allowed")


def _attach_target(
    frame: pd.DataFrame, event_cycles: pd.Series, horizon: int
) -> pd.DataFrame:
    output = frame.copy(deep=True)
    output[EVENT_CYCLE_COLUMN] = output[UNIT_COLUMN].map(event_cycles)
    if output[EVENT_CYCLE_COLUMN].isna().any():
        missing_units = output.loc[
            output[EVENT_CYCLE_COLUMN].isna(), UNIT_COLUMN
        ].unique()
        raise ValueError(f"Missing event-cycle information for assets: {missing_units.tolist()}")
    output[EVENT_CYCLE_COLUMN] = output[EVENT_CYCLE_COLUMN].astype(int)
    output[RUL_COLUMN] = output[EVENT_CYCLE_COLUMN] - output[CYCLE_COLUMN]
    if (output[RUL_COLUMN] < 0).any():
        raise ValueError("Constructed remaining useful life cannot be negative")
    output[TARGET_COLUMN] = (output[RUL_COLUMN] <= horizon).astype(np.int8)
    return output


def construct_train_targets(
    train: pd.DataFrame, horizon: int = PREDICTION_HORIZON
) -> pd.DataFrame:
    """Reconstruct row RUL and the future-event target for run-to-failure train data."""

    _validate_trajectory_keys(train)
    horizon_value = _validate_horizon(horizon)
    event_cycles = train.groupby(UNIT_COLUMN, sort=False)[CYCLE_COLUMN].max()
    return _attach_target(train, event_cycles, horizon_value)


def construct_test_targets(
    test: pd.DataFrame,
    final_rul: pd.DataFrame | pd.Series,
    horizon: int = PREDICTION_HORIZON,
) -> pd.DataFrame:
    """Reconstruct row RUL/target from official test trajectories and RUL offsets.

    For each asset, ``event_cycle = last_observed_cycle + official_final_rul``.
    The official offset is never exposed as a predictor.
    """

    _validate_trajectory_keys(test)
    horizon_value = _validate_horizon(horizon)
    observed_ends = test.groupby(UNIT_COLUMN, sort=False)[CYCLE_COLUMN].max()

    if isinstance(final_rul, pd.Series):
        if final_rul.index.equals(pd.RangeIndex(len(final_rul))):
            offsets = pd.Series(
                final_rul.to_numpy(), index=sorted(test[UNIT_COLUMN].unique())
            )
        else:
            offsets = final_rul.copy()
    else:
        required = {UNIT_COLUMN, "final_rul"}
        missing = sorted(required.difference(final_rul.columns))
        if missing:
            raise ValueError(f"Missing final-RUL columns: {missing}")
        if final_rul[UNIT_COLUMN].duplicated().any():
            raise ValueError("final_rul must have exactly one row per asset")
        offsets = final_rul.set_index(UNIT_COLUMN)["final_rul"]

    offsets = pd.to_numeric(offsets, errors="raise")
    expected = set(observed_ends.index.tolist())
    supplied = set(offsets.index.tolist())
    if expected != supplied:
        raise ValueError(
            "Official final-RUL asset IDs must match test assets exactly; "
            f"missing={sorted(expected - supplied)}, extra={sorted(supplied - expected)}"
        )
    if offsets.isna().any() or (offsets < 0).any() or not np.isfinite(offsets).all():
        raise ValueError("Official final-RUL offsets must be finite and non-negative")
    if not np.all(np.equal(offsets, np.floor(offsets))):
        raise ValueError("Official final-RUL offsets must be whole cycles")

    event_cycles = observed_ends + offsets.astype(int).reindex(observed_ends.index)
    return _attach_target(test, event_cycles, horizon_value)


def assert_predictors_are_leakage_safe(columns: Iterable[str]) -> None:
    """Reject outcome, future, and RUL fields from a proposed predictor set."""

    unsafe: list[str] = []
    for column in columns:
        normalized = str(column).strip().lower()
        tokens = set(re.findall(r"[a-z]+|\d+", normalized))
        compact = "".join(re.findall(r"[a-z0-9]+", normalized))
        is_forbidden = normalized in _FORBIDDEN_EXACT
        is_forbidden = is_forbidden or bool(tokens.intersection(_FORBIDDEN_TOKENS))
        is_forbidden = is_forbidden or any(
            phrase in compact for phrase in _FORBIDDEN_COMPACT_PHRASES
        )
        if is_forbidden:
            unsafe.append(str(column))
    if unsafe:
        raise ValueError(
            "Predictor leakage firewall rejected outcome-derived columns: "
            f"{sorted(unsafe)}"
        )


__all__ = [
    "EVENT_CYCLE_COLUMN",
    "PREDICTION_HORIZON",
    "RUL_COLUMN",
    "TARGET_COLUMN",
    "assert_predictors_are_leakage_safe",
    "construct_test_targets",
    "construct_train_targets",
]
