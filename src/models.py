"""Restrained baseline and candidate models with asset-grouped OOF prediction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import RANDOM_SEED
from .data_validation import make_group_kfold_splits


class ConstantProbabilityClassifier(ClassifierMixin, BaseEstimator):
    """Intercept-only baseline that predicts development-set prevalence."""

    def fit(
        self,
        X: object,
        y: Sequence[int] | np.ndarray,
        sample_weight: Sequence[float] | np.ndarray | None = None,
    ) -> "ConstantProbabilityClassifier":
        target = np.asarray(y).reshape(-1)
        if target.size == 0 or not set(np.unique(target)).issubset({0, 1}):
            raise ValueError("y must be a non-empty binary target")
        if len(X) != target.size:  # type: ignore[arg-type]
            raise ValueError("X and y must contain the same number of rows")
        if sample_weight is None:
            prevalence = float(target.mean())
        else:
            weights = np.asarray(sample_weight, dtype=float).reshape(-1)
            if weights.size != target.size or (weights < 0).any() or weights.sum() <= 0:
                raise ValueError("sample_weight must be non-negative and align with y")
            prevalence = float(np.average(target, weights=weights))
        self.prevalence_ = prevalence
        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_ = int(np.asarray(X).shape[1])
        return self

    def predict_proba(self, X: object) -> np.ndarray:
        if not hasattr(self, "prevalence_"):
            raise RuntimeError("ConstantProbabilityClassifier must be fitted first")
        count = len(X)  # type: ignore[arg-type]
        positive = np.full(count, self.prevalence_, dtype=float)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: object) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_constant_baseline() -> ConstantProbabilityClassifier:
    return ConstantProbabilityClassifier()


def build_age_only_baseline(
    *, age_column: str = "asset_age", random_seed: int = RANDOM_SEED
) -> Pipeline:
    """Logistic age-only baseline; all learned transforms remain in-fold."""

    age_preprocessor = ColumnTransformer(
        transformers=[
            (
                "age",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                [age_column],
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("preprocess", age_preprocessor),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear", max_iter=1_000, random_state=random_seed
                ),
            ),
        ]
    )


def build_logistic_model(*, random_seed: int = RANDOM_SEED) -> Pipeline:
    """Regularized logistic model with in-pipeline median imputation/scaling."""

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear", max_iter=1_000, random_state=random_seed
                ),
            ),
        ]
    )


def build_random_forest_model(
    *,
    random_seed: int = RANDOM_SEED,
    n_estimators: int = 200,
    max_depth: int | None = 10,
    min_samples_leaf: int = 2,
    class_weight: str | dict[int, float] | None = None,
) -> Pipeline:
    """Modest deterministic random-forest candidate with in-fold imputation."""

    if n_estimators <= 0 or min_samples_leaf <= 0:
        raise ValueError("n_estimators and min_samples_leaf must be positive")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=int(n_estimators),
                    max_depth=max_depth,
                    min_samples_leaf=int(min_samples_leaf),
                    class_weight=class_weight,
                    random_state=random_seed,
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_model_registry(
    *, random_seed: int = RANDOM_SEED, forest_estimators: int = 200
) -> Mapping[str, BaseEstimator]:
    """Return the predeclared baselines and two primary candidates."""

    return {
        "constant_prevalence": build_constant_baseline(),
        "age_only": build_age_only_baseline(random_seed=random_seed),
        "logistic_regression": build_logistic_model(random_seed=random_seed),
        "random_forest": build_random_forest_model(
            random_seed=random_seed, n_estimators=forest_estimators
        ),
    }


def _take_rows(values: object, indices: np.ndarray) -> object:
    if isinstance(values, (pd.DataFrame, pd.Series)):
        return values.iloc[indices]
    return np.asarray(values)[indices]


def positive_class_probability(estimator: BaseEstimator, X: object) -> np.ndarray:
    """Extract class-1 probability robustly from a fitted binary estimator."""

    if not hasattr(estimator, "predict_proba"):
        raise TypeError("Estimator must implement predict_proba")
    probabilities = np.asarray(estimator.predict_proba(X), dtype=float)  # type: ignore[attr-defined]
    classes = np.asarray(getattr(estimator, "classes_", []))
    positive = np.flatnonzero(classes == 1)
    if probabilities.ndim != 2 or positive.size != 1:
        raise ValueError("Estimator must be a fitted binary classifier with class 1")
    result = probabilities[:, int(positive[0])]
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError("Estimator produced invalid probabilities")
    return result


def oof_grouped_probabilities(
    estimator: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
    y: Sequence[int] | np.ndarray,
    groups: Sequence[object] | np.ndarray,
    *,
    n_splits: int = 5,
) -> np.ndarray:
    """Generate one out-of-fold probability per row using asset-separated folds."""

    target = np.asarray(y).reshape(-1)
    group_values = np.asarray(groups).reshape(-1)
    if len(X) != target.size or target.size != group_values.size:
        raise ValueError("X, y, and groups must have the same row count")
    if not set(np.unique(target)).issubset({0, 1}):
        raise ValueError("y must be binary")
    splits = make_group_kfold_splits(group_values, n_splits=n_splits)
    oof = np.full(target.size, np.nan, dtype=float)
    for train_index, validation_index in splits:
        train_target = target[train_index]
        if np.unique(train_target).size < 2 and not isinstance(
            estimator, ConstantProbabilityClassifier
        ):
            raise ValueError("A development fold contains only one outcome class")
        fold_estimator = clone(estimator)
        fold_estimator.fit(_take_rows(X, train_index), train_target)
        oof[validation_index] = positive_class_probability(
            fold_estimator, _take_rows(X, validation_index)
        )
    if not np.isfinite(oof).all():
        raise AssertionError("Grouped OOF generation did not predict every row exactly once")
    return oof


__all__ = [
    "ConstantProbabilityClassifier",
    "build_age_only_baseline",
    "build_constant_baseline",
    "build_logistic_model",
    "build_model_registry",
    "build_random_forest_model",
    "oof_grouped_probabilities",
    "positive_class_probability",
]
