"""Row-level probability, calibration, threshold, and asset-bootstrap metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import RANDOM_SEED


@dataclass(frozen=True)
class ProbabilityMetrics:
    n_observations: int
    prevalence: float
    pr_auc: float
    roc_auc: float
    brier_score: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    predicted_positive_rate: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    target_recall: float
    achieved_recall: float
    validation_precision: float


def _validate_binary_inputs(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(y_true).reshape(-1)
    probability = np.asarray(probabilities, dtype=float).reshape(-1)
    if target.size == 0 or target.size != probability.size:
        raise ValueError("y_true and probabilities must be non-empty and aligned")
    if not set(np.unique(target)).issubset({0, 1}):
        raise ValueError("y_true must be binary")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    return target.astype(int), probability


def evaluate_probabilities(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> ProbabilityMetrics:
    """Evaluate discrimination and calibration without using row accuracy."""

    target, probability = _validate_binary_inputs(y_true, probabilities)
    positive_count = int(target.sum())
    pr_auc = (
        float(average_precision_score(target, probability))
        if positive_count > 0
        else float("nan")
    )
    roc_auc = (
        float(roc_auc_score(target, probability))
        if np.unique(target).size == 2
        else float("nan")
    )
    return ProbabilityMetrics(
        n_observations=int(target.size),
        prevalence=float(target.mean()),
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        brier_score=float(brier_score_loss(target, probability)),
    )


def evaluate_at_threshold(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float,
) -> ThresholdMetrics:
    """Compute secondary operating-point metrics at a predeclared threshold."""

    target, probability = _validate_binary_inputs(y_true, probabilities)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    prediction = (probability >= float(threshold)).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        target, prediction, labels=[0, 1]
    ).ravel()
    return ThresholdMetrics(
        threshold=float(threshold),
        precision=float(precision_score(target, prediction, zero_division=0)),
        recall=float(recall_score(target, prediction, zero_division=0)),
        f1=float(f1_score(target, prediction, zero_division=0)),
        true_positive=int(true_positive),
        false_positive=int(false_positive),
        true_negative=int(true_negative),
        false_negative=int(false_negative),
        predicted_positive_rate=float(prediction.mean()),
    )


def select_threshold_policy(
    y_validation: Sequence[int] | np.ndarray,
    validation_probabilities: Sequence[float] | np.ndarray,
    *,
    recall_target: float = 0.80,
) -> ThresholdSelection:
    """Choose the highest validation threshold that meets a recall target.

    The rule is deterministic and must be applied before locked-test evaluation.
    It does not optimize test-set F1.
    """

    target, probability = _validate_binary_inputs(y_validation, validation_probabilities)
    if not 0.0 < recall_target <= 1.0:
        raise ValueError("recall_target must lie in (0, 1]")
    if target.sum() == 0:
        raise ValueError("Recall threshold selection requires validation positives")

    selected: ThresholdMetrics | None = None
    for candidate in np.sort(np.unique(probability))[::-1]:
        metrics = evaluate_at_threshold(target, probability, float(candidate))
        if metrics.recall + 1e-15 >= recall_target:
            selected = metrics
            break
    if selected is None:
        raise AssertionError("No threshold met recall despite validation positives")
    return ThresholdSelection(
        threshold=selected.threshold,
        target_recall=float(recall_target),
        achieved_recall=selected.recall,
        validation_precision=selected.precision,
    )


def select_recall_threshold(
    y_validation: Sequence[int] | np.ndarray,
    validation_probabilities: Sequence[float] | np.ndarray,
    *,
    recall_target: float = 0.80,
) -> float:
    """Convenience wrapper returning only the validation-selected threshold."""

    return select_threshold_policy(
        y_validation, validation_probabilities, recall_target=recall_target
    ).threshold


def calibration_table(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Return observed and predicted bin probabilities for calibration plots."""

    target, probability = _validate_binary_inputs(y_true, probabilities)
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")
    observed, predicted = calibration_curve(
        target, probability, n_bins=n_bins, strategy=strategy
    )
    return pd.DataFrame(
        {
            "mean_predicted_probability": predicted,
            "observed_event_rate": observed,
        }
    )


def asset_bootstrap_probability_metrics(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    groups: Sequence[object] | np.ndarray,
    *,
    n_bootstrap: int = 500,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Bootstrap complete assets, preserving within-asset row dependence."""

    target, probability = _validate_binary_inputs(y_true, probabilities)
    group_values = np.asarray(groups).reshape(-1)
    if group_values.size != target.size:
        raise ValueError("groups must align with y_true")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    assets = np.unique(group_values)
    if assets.size < 2:
        raise ValueError("Asset bootstrap requires at least two assets")
    row_indices = {asset: np.flatnonzero(group_values == asset) for asset in assets}
    generator = np.random.default_rng(random_seed)
    records: list[dict[str, float | int]] = []
    for replicate in range(int(n_bootstrap)):
        sampled_assets = generator.choice(assets, size=assets.size, replace=True)
        sampled_rows = np.concatenate([row_indices[asset] for asset in sampled_assets])
        metrics = evaluate_probabilities(target[sampled_rows], probability[sampled_rows])
        records.append({"replicate": replicate, **metrics.to_dict()})
    return pd.DataFrame.from_records(records)


def bootstrap_interval(
    bootstrap_metrics: pd.DataFrame,
    metric: str,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile interval for a metric produced by the asset bootstrap."""

    if metric not in bootstrap_metrics.columns:
        raise ValueError(f"Bootstrap metric not found: {metric}")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    values = pd.to_numeric(bootstrap_metrics[metric], errors="coerce").dropna()
    if values.empty:
        return float("nan"), float("nan")
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(values.to_numpy(), [tail, 1.0 - tail])
    return float(lower), float(upper)


__all__ = [
    "ProbabilityMetrics",
    "ThresholdMetrics",
    "ThresholdSelection",
    "asset_bootstrap_probability_metrics",
    "bootstrap_interval",
    "calibration_table",
    "evaluate_at_threshold",
    "evaluate_probabilities",
    "select_recall_threshold",
    "select_threshold_policy",
]
