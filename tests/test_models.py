"""Fast deterministic tests for baselines, candidates, calibration, and metrics."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.calibration import (
    fit_sigmoid_calibrator,
    nested_grouped_oof_calibration,
)
from src.data_validation import make_group_kfold_splits
from src.evaluation import (
    asset_bootstrap_probability_metrics,
    evaluate_at_threshold,
    evaluate_probabilities,
    select_recall_threshold,
    select_threshold_policy,
)
from src.models import (
    build_age_only_baseline,
    build_constant_baseline,
    build_logistic_model,
    build_random_forest_model,
    oof_grouped_probabilities,
    positive_class_probability,
)


def model_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    groups = np.repeat(np.arange(1, 7), 4)
    age = np.tile(np.arange(1, 5), 6)
    target = np.tile([0, 0, 1, 1], 6)
    frame = pd.DataFrame(
        {
            "asset_age": age.astype(float),
            "current__sensor_2": age + groups * 0.05,
            "sensor_2__roll_mean_3": age * 0.8 + groups * 0.02,
        }
    )
    return frame, target, groups


class BaselineTests(unittest.TestCase):
    def test_constant_baseline_equals_training_prevalence(self) -> None:
        X = np.zeros((5, 2))
        y = np.array([0, 0, 0, 1, 1])
        model = build_constant_baseline().fit(X, y)
        probabilities = model.predict_proba(np.ones((3, 2)))
        np.testing.assert_allclose(probabilities[:, 1], 0.4)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_age_only_baseline_ignores_other_columns(self) -> None:
        X, y, _ = model_data()
        model = build_age_only_baseline().fit(X, y)
        changed = X.copy()
        changed["current__sensor_2"] += 1_000_000
        np.testing.assert_allclose(
            positive_class_probability(model, X),
            positive_class_probability(model, changed),
        )

    def test_age_only_probabilities_are_valid(self) -> None:
        X, y, _ = model_data()
        probabilities = positive_class_probability(build_age_only_baseline().fit(X, y), X)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())


class CandidateModelTests(unittest.TestCase):
    def test_logistic_pipeline_imputer_is_fit_on_training_data(self) -> None:
        X = pd.DataFrame({"a": [1.0, np.nan, 9.0, 10.0], "b": [2.0, 4.0, 6.0, 8.0]})
        y = np.array([0, 0, 1, 1])
        model = build_logistic_model().fit(X, y)
        np.testing.assert_allclose(model.named_steps["imputer"].statistics_, [9.0, 5.0])
        probabilities = positive_class_probability(model, X)
        self.assertFalse(np.isnan(probabilities).any())

    def test_logistic_is_deterministic(self) -> None:
        X, y, _ = model_data()
        first = build_logistic_model(random_seed=17).fit(X, y)
        second = build_logistic_model(random_seed=17).fit(X, y)
        np.testing.assert_allclose(
            positive_class_probability(first, X), positive_class_probability(second, X)
        )

    def test_random_forest_is_deterministic(self) -> None:
        X, y, _ = model_data()
        first = build_random_forest_model(random_seed=17, n_estimators=20).fit(X, y)
        second = build_random_forest_model(random_seed=17, n_estimators=20).fit(X, y)
        np.testing.assert_allclose(
            positive_class_probability(first, X), positive_class_probability(second, X)
        )

    def test_grouped_oof_probabilities_cover_every_row(self) -> None:
        X, y, groups = model_data()
        model = build_logistic_model(random_seed=8)
        first = oof_grouped_probabilities(model, X, y, groups, n_splits=3)
        second = oof_grouped_probabilities(model, X, y, groups, n_splits=3)
        self.assertEqual(len(first), len(y))
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(((first >= 0) & (first <= 1)).all())
        np.testing.assert_allclose(first, second)

    def test_constant_grouped_oof_uses_fold_prevalence(self) -> None:
        X, y, groups = model_data()
        oof = oof_grouped_probabilities(
            build_constant_baseline(), X, y, groups, n_splits=3
        )
        np.testing.assert_allclose(oof, 0.5)


class CalibrationAndMetricTests(unittest.TestCase):
    def test_sigmoid_calibrator_returns_valid_probabilities(self) -> None:
        raw = np.array([0.02, 0.1, 0.2, 0.35, 0.6, 0.9, 0.98, 1.0])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        calibrator = fit_sigmoid_calibrator(raw, y)
        calibrated = calibrator.transform(raw)
        self.assertTrue(np.isfinite(calibrated).all())
        self.assertTrue(((calibrated >= 0) & (calibrated <= 1)).all())
        self.assertTrue(np.all(np.diff(calibrated) >= 0))

    def test_nested_calibration_is_isolated_from_validation_labels(self) -> None:
        groups = np.repeat(np.arange(10), 6)
        age = np.tile(np.arange(6, dtype=float), 10)
        X = pd.DataFrame(
            {
                "asset_age": age,
                "condition_signal": age + groups * 0.15,
            }
        )
        y = np.tile([0, 0, 0, 1, 1, 1], 10)
        first_outer_validation = make_group_kfold_splits(groups, n_splits=5)[0][1]

        original = nested_grouped_oof_calibration(
            build_logistic_model(random_seed=19),
            X,
            y,
            groups,
            n_splits=5,
            random_seed=19,
        )
        perturbed_y = y.copy()
        perturbed_y[first_outer_validation] = 1 - perturbed_y[first_outer_validation]
        perturbed = nested_grouped_oof_calibration(
            build_logistic_model(random_seed=19),
            X,
            perturbed_y,
            groups,
            n_splits=5,
            random_seed=19,
        )

        np.testing.assert_allclose(
            original.raw_probabilities[first_outer_validation],
            perturbed.raw_probabilities[first_outer_validation],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            original.calibrated_probabilities[first_outer_validation],
            perturbed.calibrated_probabilities[first_outer_validation],
            rtol=0.0,
            atol=0.0,
        )

    def test_probability_metrics_match_manual_brier(self) -> None:
        y = np.array([0, 0, 1, 1])
        probability = np.array([0.1, 0.4, 0.35, 0.8])
        metrics = evaluate_probabilities(y, probability)
        self.assertAlmostEqual(metrics.brier_score, np.mean((probability - y) ** 2))
        self.assertAlmostEqual(metrics.prevalence, 0.5)
        self.assertGreater(metrics.pr_auc, metrics.prevalence)
        self.assertGreater(metrics.roc_auc, 0.5)

    def test_highest_threshold_meeting_recall_is_selected(self) -> None:
        y = np.array([0, 1, 1, 0])
        probability = np.array([0.95, 0.9, 0.8, 0.1])
        self.assertEqual(select_recall_threshold(y, probability, recall_target=0.5), 0.9)
        self.assertEqual(select_recall_threshold(y, probability, recall_target=1.0), 0.8)
        policy = select_threshold_policy(y, probability, recall_target=0.5)
        self.assertEqual(policy.achieved_recall, 0.5)
        self.assertEqual(policy.validation_precision, 0.5)

    def test_threshold_metrics_have_manual_confusion_counts(self) -> None:
        metrics = evaluate_at_threshold([0, 1, 1, 0], [0.95, 0.9, 0.8, 0.1], 0.9)
        self.assertEqual(metrics.true_positive, 1)
        self.assertEqual(metrics.false_positive, 1)
        self.assertEqual(metrics.true_negative, 1)
        self.assertEqual(metrics.false_negative, 1)

    def test_asset_bootstrap_is_seed_deterministic(self) -> None:
        X, y, groups = model_data()
        probability = np.linspace(0.05, 0.95, len(y))
        first = asset_bootstrap_probability_metrics(
            y, probability, groups, n_bootstrap=12, random_seed=31
        )
        second = asset_bootstrap_probability_metrics(
            y, probability, groups, n_bootstrap=12, random_seed=31
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 12)


if __name__ == "__main__":
    unittest.main()
