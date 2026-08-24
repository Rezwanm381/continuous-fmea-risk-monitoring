"""Causal, trailing-only features for C-MAPSS FD001.

The default current-sensor set excludes six exactly constant FD001 channels
(1, 5, 10, 16, 18, and 19) plus the deliberately omitted two-valued sensor 6.
Rolling summaries are deliberately limited
to sensors 2, 4, 7, 11, 15, and 21: a compact selection spanning temperature,
pressure/speed, flow/ratio, and bleed/efficiency-related degradation signals.
This is an engineering modeling choice, not a claim that those channels are
causal.  Every rolling statistic uses rows at or before the prediction cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .data_loader import CYCLE_COLUMN, SETTING_COLUMNS, UNIT_COLUMN
from .target import assert_predictors_are_leakage_safe


CURRENT_SENSOR_COLUMNS = tuple(
    f"sensor_{index}" for index in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)
)
ROLLING_SENSOR_COLUMNS = tuple(f"sensor_{index}" for index in (2, 4, 7, 11, 15, 21))
ROLLING_SENSOR_RATIONALE = {
    "sensor_2": "temperature trajectory",
    "sensor_4": "temperature trajectory",
    "sensor_7": "pressure/condition trajectory",
    "sensor_11": "pressure/condition trajectory",
    "sensor_15": "ratio/efficiency trajectory",
    "sensor_21": "flow/condition trajectory",
}


@dataclass(frozen=True)
class FeatureSpec:
    """Explicit, reviewable definition of the predictor families."""

    current_sensors: tuple[str, ...] = CURRENT_SENSOR_COLUMNS
    rolling_sensors: tuple[str, ...] = ROLLING_SENSOR_COLUMNS
    rolling_windows: tuple[int, ...] = (5, 10)
    include_operating_settings: bool = True
    include_age: bool = True
    include_initial_deviation: bool = True

    def __post_init__(self) -> None:
        if not self.current_sensors:
            raise ValueError("At least one current sensor is required")
        if len(set(self.current_sensors)) != len(self.current_sensors):
            raise ValueError("current_sensors cannot contain duplicates")
        if len(set(self.rolling_sensors)) != len(self.rolling_sensors):
            raise ValueError("rolling_sensors cannot contain duplicates")
        if not set(self.rolling_sensors).issubset(self.current_sensors):
            raise ValueError("rolling_sensors must be included in current_sensors")
        if any(
            isinstance(window, bool)
            or not isinstance(window, (int, np.integer))
            or int(window) < 2
            for window in self.rolling_windows
        ):
            raise ValueError("rolling windows must be integers of at least two cycles")
        if len(set(self.rolling_windows)) != len(self.rolling_windows):
            raise ValueError("rolling_windows cannot contain duplicates")


@dataclass(frozen=True)
class PerturbationCheck:
    compared_rows: int
    perturbed_rows: int
    max_feature_delta: float
    max_prediction_delta: float | None


def _trailing_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(x_centered, values - values.mean()) / denominator)


def _validate_source(frame: pd.DataFrame, spec: FeatureSpec) -> None:
    required = {UNIT_COLUMN, CYCLE_COLUMN, *spec.current_sensors}
    if spec.include_operating_settings:
        required.update(SETTING_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing feature source columns: {missing}")
    if frame.empty:
        raise ValueError("Feature source frame cannot be empty")
    if frame.duplicated([UNIT_COLUMN, CYCLE_COLUMN]).any():
        raise ValueError("Duplicate (unit_id, cycle) rows are not allowed")
    if frame[list(required)].isna().any().any():
        raise ValueError("Raw feature source columns cannot contain missing values")
    if not np.isfinite(frame[list(required)].to_numpy(dtype=float)).all():
        raise ValueError("Raw feature source columns must be finite")


def build_feature_frame(
    frame: pd.DataFrame, spec: FeatureSpec | None = None
) -> pd.DataFrame:
    """Build current and trailing-only predictors, preserving input row order.

    Learned imputation/scaling is intentionally absent here and belongs inside
    model pipelines, where it is fit separately within each development fold.
    """

    definition = spec or FeatureSpec()
    _validate_source(frame, definition)
    working = frame.copy(deep=False).assign(_row_order=np.arange(len(frame)))
    working = working.sort_values(
        [UNIT_COLUMN, CYCLE_COLUMN], kind="mergesort"
    ).reset_index(drop=True)
    result = working[[UNIT_COLUMN, CYCLE_COLUMN, "_row_order"]].copy()

    if definition.include_age:
        result["asset_age"] = working[CYCLE_COLUMN].to_numpy(dtype=float)
    if definition.include_operating_settings:
        for column in SETTING_COLUMNS:
            result[f"current__{column}"] = working[column].to_numpy(dtype=float)
    for sensor in definition.current_sensors:
        result[f"current__{sensor}"] = working[sensor].to_numpy(dtype=float)

    grouped = working.groupby(UNIT_COLUMN, sort=False, group_keys=False)
    for sensor in definition.rolling_sensors:
        if definition.include_initial_deviation:
            initial = grouped[sensor].transform("first")
            result[f"{sensor}__from_initial"] = (
                working[sensor].to_numpy(dtype=float) - initial.to_numpy(dtype=float)
            )
        for window in definition.rolling_windows:
            rolling = grouped[sensor].rolling(window=int(window), min_periods=1)
            mean = rolling.mean().reset_index(level=0, drop=True).sort_index()
            std = rolling.std(ddof=0).reset_index(level=0, drop=True).sort_index()
            slope = (
                rolling.apply(_trailing_slope, raw=True)
                .reset_index(level=0, drop=True)
                .sort_index()
            )
            result[f"{sensor}__roll_mean_{window}"] = mean.to_numpy(dtype=float)
            result[f"{sensor}__roll_std_{window}"] = std.to_numpy(dtype=float)
            result[f"{sensor}__roll_slope_{window}"] = slope.to_numpy(dtype=float)

    result = result.sort_values("_row_order", kind="mergesort").drop(columns="_row_order")
    result = result.reset_index(drop=True)
    assert_predictors_are_leakage_safe(feature_columns(result))
    if result.drop(columns=[UNIT_COLUMN, CYCLE_COLUMN]).isna().any().any():
        raise AssertionError("Trailing feature construction unexpectedly produced missing values")
    return result


def feature_columns(feature_frame: pd.DataFrame) -> list[str]:
    """Return model predictors, excluding asset/cycle identifiers."""

    columns = [
        column
        for column in feature_frame.columns
        if column not in {UNIT_COLUMN, CYCLE_COLUMN}
    ]
    assert_predictors_are_leakage_safe(columns)
    return columns


def _positive_probability(estimator: object, predictors: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(predictors), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Estimator must return two-column binary probabilities")
    classes = np.asarray(getattr(estimator, "classes_", [0, 1]))
    positive = np.flatnonzero(classes == 1)
    if positive.size != 1:
        raise ValueError("Estimator must expose class 1")
    return probabilities[:, int(positive[0])]


def assert_future_perturbation_invariance(
    frame: pd.DataFrame,
    *,
    asset_id: object,
    after_cycle: int,
    spec: FeatureSpec | None = None,
    perturb_columns: Sequence[str] | None = None,
    magnitude: float = 1_000_000.0,
    estimator: object | None = None,
    prediction_columns: Sequence[str] | None = None,
    atol: float = 1e-12,
) -> PerturbationCheck:
    """Stress-test that changing future raw values cannot affect earlier rows.

    If a fitted estimator is supplied, earlier predictions are compared as well
    as features.  The input frame is never modified.
    """

    definition = spec or FeatureSpec()
    columns = tuple(perturb_columns or definition.current_sensors)
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Perturbation columns not found: {missing}")
    future_mask = (frame[UNIT_COLUMN] == asset_id) & (frame[CYCLE_COLUMN] > after_cycle)
    earlier_mask = (frame[UNIT_COLUMN] == asset_id) & (frame[CYCLE_COLUMN] <= after_cycle)
    if not future_mask.any():
        raise ValueError("No future rows exist after the requested cutoff")
    if not earlier_mask.any():
        raise ValueError("No earlier rows exist at or before the requested cutoff")

    perturbed = frame.copy(deep=True)
    for column in columns:
        perturbed.loc[future_mask, column] = (
            perturbed.loc[future_mask, column].astype(float) + float(magnitude)
        )
    original_features = build_feature_frame(frame, definition)
    perturbed_features = build_feature_frame(perturbed, definition)
    predictor_names = feature_columns(original_features)

    comparison_mask = earlier_mask.to_numpy()
    original_values = original_features.loc[comparison_mask, predictor_names].to_numpy()
    perturbed_values = perturbed_features.loc[comparison_mask, predictor_names].to_numpy()
    maximum_feature_delta = float(np.max(np.abs(original_values - perturbed_values)))
    if not np.allclose(original_values, perturbed_values, rtol=0.0, atol=atol):
        raise AssertionError(
            "Future-value perturbation changed one or more earlier-time features"
        )

    maximum_prediction_delta: float | None = None
    if estimator is not None:
        model_columns = list(prediction_columns or predictor_names)
        original_probability = _positive_probability(
            estimator, original_features.loc[comparison_mask, model_columns]
        )
        perturbed_probability = _positive_probability(
            estimator, perturbed_features.loc[comparison_mask, model_columns]
        )
        maximum_prediction_delta = float(
            np.max(np.abs(original_probability - perturbed_probability))
        )
        if not np.allclose(
            original_probability, perturbed_probability, rtol=0.0, atol=atol
        ):
            raise AssertionError("Future-value perturbation changed earlier predictions")

    return PerturbationCheck(
        compared_rows=int(earlier_mask.sum()),
        perturbed_rows=int(future_mask.sum()),
        max_feature_delta=maximum_feature_delta,
        max_prediction_delta=maximum_prediction_delta,
    )


__all__ = [
    "CURRENT_SENSOR_COLUMNS",
    "FeatureSpec",
    "PerturbationCheck",
    "ROLLING_SENSOR_COLUMNS",
    "ROLLING_SENSOR_RATIONALE",
    "assert_future_perturbation_invariance",
    "build_feature_frame",
    "feature_columns",
]
