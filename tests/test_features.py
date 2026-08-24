"""Tests proving features and predictions depend only on present/past rows."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features import (
    FeatureSpec,
    assert_future_perturbation_invariance,
    build_feature_frame,
    feature_columns,
)
from src.models import build_logistic_model


def make_sensor_frame(asset_count: int = 2, cycle_count: int = 6) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    for asset in range(1, asset_count + 1):
        for cycle in range(1, cycle_count + 1):
            record: dict[str, float | int] = {
                "unit_id": asset,
                "cycle": cycle,
                "op_setting_1": 0.1 * asset,
                "op_setting_2": 0.2 * asset,
                "op_setting_3": 0.3 * asset,
            }
            for sensor in range(1, 22):
                record[f"sensor_{sensor}"] = asset * 100 + cycle + sensor / 100
            records.append(record)
    return pd.DataFrame.from_records(records)


SMALL_SPEC = FeatureSpec(
    current_sensors=("sensor_2",),
    rolling_sensors=("sensor_2",),
    rolling_windows=(3,),
    include_operating_settings=False,
)


class TrailingFeatureTests(unittest.TestCase):
    def test_manual_trailing_mean_std_and_slope(self) -> None:
        frame = make_sensor_frame(asset_count=1, cycle_count=5)
        features = build_feature_frame(frame, SMALL_SPEC)
        expected_means = [101.02, 101.52, 102.02, 103.02, 104.02]
        expected_stds = [0.0, 0.5, np.sqrt(2 / 3), np.sqrt(2 / 3), np.sqrt(2 / 3)]
        np.testing.assert_allclose(features["sensor_2__roll_mean_3"], expected_means)
        np.testing.assert_allclose(features["sensor_2__roll_std_3"], expected_stds)
        np.testing.assert_allclose(features["sensor_2__roll_slope_3"], [0, 1, 1, 1, 1])
        np.testing.assert_allclose(features["sensor_2__from_initial"], [0, 1, 2, 3, 4])

    def test_asset_history_resets_between_assets(self) -> None:
        frame = make_sensor_frame(asset_count=2, cycle_count=3)
        features = build_feature_frame(frame, SMALL_SPEC)
        first_asset_two = features.loc[features["unit_id"] == 2].iloc[0]
        self.assertAlmostEqual(first_asset_two["sensor_2__roll_mean_3"], 201.02)
        self.assertEqual(first_asset_two["sensor_2__roll_std_3"], 0.0)
        self.assertEqual(first_asset_two["sensor_2__from_initial"], 0.0)

    def test_cycle_order_not_input_row_order_controls_history(self) -> None:
        frame = make_sensor_frame(asset_count=1, cycle_count=4).iloc[[2, 0, 3, 1]].reset_index(drop=True)
        features = build_feature_frame(frame, SMALL_SPEC)
        self.assertEqual(features["cycle"].tolist(), [3, 1, 4, 2])
        cycle_three = features.loc[features["cycle"] == 3].iloc[0]
        self.assertAlmostEqual(cycle_three["sensor_2__roll_mean_3"], 102.02)

    def test_feature_builder_does_not_mutate_input(self) -> None:
        frame = make_sensor_frame(asset_count=1, cycle_count=4)
        before = frame.copy(deep=True)
        build_feature_frame(frame, SMALL_SPEC)
        pd.testing.assert_frame_equal(frame, before)

    def test_identifiers_are_not_predictors(self) -> None:
        features = build_feature_frame(make_sensor_frame(1, 3), SMALL_SPEC)
        predictors = feature_columns(features)
        self.assertNotIn("unit_id", predictors)
        self.assertNotIn("cycle", predictors)
        self.assertIn("asset_age", predictors)


class FuturePerturbationTests(unittest.TestCase):
    def test_future_sensor_changes_leave_earlier_features_unchanged(self) -> None:
        frame = make_sensor_frame(asset_count=2, cycle_count=6)
        result = assert_future_perturbation_invariance(
            frame,
            asset_id=1,
            after_cycle=3,
            spec=SMALL_SPEC,
            perturb_columns=["sensor_2"],
        )
        self.assertEqual(result.compared_rows, 3)
        self.assertEqual(result.perturbed_rows, 3)
        self.assertEqual(result.max_feature_delta, 0.0)
        self.assertIsNone(result.max_prediction_delta)

    def test_future_sensor_changes_leave_earlier_predictions_unchanged(self) -> None:
        frame = make_sensor_frame(asset_count=2, cycle_count=6)
        features = build_feature_frame(frame, SMALL_SPEC)
        predictors = feature_columns(features)
        target = np.tile([0, 0, 0, 1, 1, 1], 2)
        model = build_logistic_model().fit(features[predictors], target)
        result = assert_future_perturbation_invariance(
            frame,
            asset_id=1,
            after_cycle=3,
            spec=SMALL_SPEC,
            perturb_columns=["sensor_2"],
            estimator=model,
            prediction_columns=predictors,
        )
        self.assertEqual(result.max_feature_delta, 0.0)
        self.assertEqual(result.max_prediction_delta, 0.0)

    def test_current_perturbation_can_change_current_but_not_prior_feature(self) -> None:
        frame = make_sensor_frame(asset_count=1, cycle_count=5)
        original = build_feature_frame(frame, SMALL_SPEC)
        changed = frame.copy()
        changed.loc[changed["cycle"] == 4, "sensor_2"] += 10_000
        rebuilt = build_feature_frame(changed, SMALL_SPEC)
        prior = original["cycle"] <= 3
        pd.testing.assert_frame_equal(original.loc[prior], rebuilt.loc[prior])
        self.assertNotEqual(
            original.loc[original["cycle"] == 4, "current__sensor_2"].iloc[0],
            rebuilt.loc[rebuilt["cycle"] == 4, "current__sensor_2"].iloc[0],
        )

    def test_cutoff_requires_both_earlier_and_future_rows(self) -> None:
        frame = make_sensor_frame(asset_count=1, cycle_count=3)
        with self.assertRaisesRegex(ValueError, "No future rows"):
            assert_future_perturbation_invariance(
                frame, asset_id=1, after_cycle=3, spec=SMALL_SPEC
            )


class FeatureSpecificationTests(unittest.TestCase):
    def test_rolling_sensor_must_be_current(self) -> None:
        with self.assertRaises(ValueError):
            FeatureSpec(
                current_sensors=("sensor_2",),
                rolling_sensors=("sensor_3",),
                rolling_windows=(3,),
            )

    def test_centered_or_single_cycle_window_cannot_be_requested(self) -> None:
        with self.assertRaises(ValueError):
            FeatureSpec(
                current_sensors=("sensor_2",),
                rolling_sensors=("sensor_2",),
                rolling_windows=(1,),
            )


if __name__ == "__main__":
    unittest.main()
