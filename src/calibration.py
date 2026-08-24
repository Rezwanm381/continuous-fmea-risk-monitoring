"""Training-only sigmoid calibration for grouped out-of-fold probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression

from . import RANDOM_SEED
from .data_validation import make_group_kfold_splits
from .models import oof_grouped_probabilities, positive_class_probability


def _validate_probabilities(probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Probabilities must be a non-empty finite sequence")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("Probabilities must lie in [0, 1]")
    return values


def _logit(probabilities: np.ndarray, epsilon: float) -> np.ndarray:
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


@dataclass
class SigmoidProbabilityCalibrator:
    """Platt-style mapping fitted only to development OOF probabilities."""

    epsilon: float = 1e-6
    random_seed: int = RANDOM_SEED

    def fit(
        self,
        raw_probabilities: Sequence[float] | np.ndarray,
        y_true: Sequence[int] | np.ndarray,
    ) -> "SigmoidProbabilityCalibrator":
        probability = _validate_probabilities(raw_probabilities)
        target = np.asarray(y_true).reshape(-1)
        if target.size != probability.size:
            raise ValueError("raw_probabilities and y_true must align")
        if set(np.unique(target)) != {0, 1}:
            raise ValueError("Calibration requires both binary outcome classes")
        if not 0.0 < self.epsilon < 0.5:
            raise ValueError("epsilon must lie between 0 and 0.5")
        self.model_ = LogisticRegression(
            solver="lbfgs", max_iter=1_000, random_state=self.random_seed
        )
        self.model_.fit(_logit(probability, self.epsilon), target.astype(int))
        return self

    @property
    def classes_(self) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Calibrator must be fitted first")
        return self.model_.classes_

    def predict_proba(
        self, raw_probabilities: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Calibrator must be fitted first")
        probability = _validate_probabilities(raw_probabilities)
        return self.model_.predict_proba(_logit(probability, self.epsilon))

    def transform(
        self, raw_probabilities: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        return self.predict_proba(raw_probabilities)[:, 1]


def fit_sigmoid_calibrator(
    oof_raw_probabilities: Sequence[float] | np.ndarray,
    y_true: Sequence[int] | np.ndarray,
    *,
    random_seed: int = RANDOM_SEED,
) -> SigmoidProbabilityCalibrator:
    """Fit sigmoid calibration on development OOF predictions, never locked test."""

    return SigmoidProbabilityCalibrator(random_seed=random_seed).fit(
        oof_raw_probabilities, y_true
    )


@dataclass(frozen=True)
class NestedCalibrationResult:
    """Raw and calibrated probabilities from leakage-isolated outer folds."""

    raw_probabilities: np.ndarray
    calibrated_probabilities: np.ndarray


def _take_rows(values: object, indices: np.ndarray) -> object:
    if isinstance(values, (pd.DataFrame, pd.Series)):
        return values.iloc[indices]
    return np.asarray(values)[indices]


def nested_grouped_oof_calibration(
    estimator: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
    y_true: Sequence[int] | np.ndarray,
    groups: Sequence[object] | np.ndarray,
    *,
    n_splits: int = 5,
    random_seed: int = RANDOM_SEED,
) -> NestedCalibrationResult:
    """Evaluate sigmoid calibration with genuinely nested grouped folds.

    For each outer validation fold, inner grouped OOF probabilities are created
    using only the outer-development assets.  The sigmoid is fitted to those
    inner scores, while a separate base estimator fitted to all outer-development
    rows supplies raw probabilities for the outer validation assets.  Therefore
    an outer asset's labels cannot influence its own raw or calibrated score.

    The returned probabilities are for development diagnostics and the
    calibration-retention decision.  A final calibrator for locked-test use must
    still be fitted separately to the complete development OOF probabilities.
    """

    target = np.asarray(y_true).reshape(-1)
    group_values = np.asarray(groups).reshape(-1)
    if len(X) != target.size or target.size != group_values.size:
        raise ValueError("X, y_true, and groups must have the same row count")
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("Nested calibration requires both binary outcome classes")

    raw = np.full(target.size, np.nan, dtype=float)
    calibrated = np.full(target.size, np.nan, dtype=float)
    outer_splits = make_group_kfold_splits(group_values, n_splits=n_splits)
    for outer_train, outer_validation in outer_splits:
        outer_train_X = _take_rows(X, outer_train)
        outer_train_y = target[outer_train]
        outer_train_groups = group_values[outer_train]
        inner_splits = min(n_splits, int(np.unique(outer_train_groups).size))
        if inner_splits < 2:
            raise ValueError("Nested calibration requires at least two inner asset groups")

        inner_oof = oof_grouped_probabilities(
            clone(estimator),
            outer_train_X,
            outer_train_y,
            outer_train_groups,
            n_splits=inner_splits,
        )
        calibrator = fit_sigmoid_calibrator(
            inner_oof, outer_train_y, random_seed=random_seed
        )

        outer_estimator = clone(estimator)
        outer_estimator.fit(outer_train_X, outer_train_y)
        outer_raw = positive_class_probability(
            outer_estimator, _take_rows(X, outer_validation)
        )
        raw[outer_validation] = outer_raw
        calibrated[outer_validation] = calibrator.transform(outer_raw)

    if not np.isfinite(raw).all() or not np.isfinite(calibrated).all():
        raise AssertionError("Nested grouped calibration did not cover every row")
    return NestedCalibrationResult(
        raw_probabilities=raw,
        calibrated_probabilities=calibrated,
    )


__all__ = [
    "NestedCalibrationResult",
    "SigmoidProbabilityCalibrator",
    "fit_sigmoid_calibrator",
    "nested_grouped_oof_calibration",
]
