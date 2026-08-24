"""Tests for independent H=30 target and official test-RUL reconstruction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import CMAPSS_COLUMNS, load_cmapss
from src.target import (
    EVENT_CYCLE_COLUMN,
    PREDICTION_HORIZON,
    RUL_COLUMN,
    TARGET_COLUMN,
    assert_predictors_are_leakage_safe,
    construct_test_targets,
    construct_train_targets,
)


class TrainTargetTests(unittest.TestCase):
    def test_horizon_boundary_is_inclusive(self) -> None:
        frame = pd.DataFrame({"unit_id": [1] * 5, "cycle": [1, 2, 3, 4, 5]})
        labeled = construct_train_targets(frame, horizon=2)
        self.assertEqual(labeled[RUL_COLUMN].tolist(), [4, 3, 2, 1, 0])
        self.assertEqual(labeled[TARGET_COLUMN].tolist(), [0, 0, 1, 1, 1])
        self.assertEqual(labeled[EVENT_CYCLE_COLUMN].tolist(), [5] * 5)

    def test_default_horizon_is_thirty_with_exact_boundary(self) -> None:
        self.assertEqual(PREDICTION_HORIZON, 30)
        frame = pd.DataFrame(
            {"unit_id": [1] * 32, "cycle": np.arange(1, 33, dtype=int)}
        )
        labeled = construct_train_targets(frame)
        self.assertEqual(labeled.loc[labeled[RUL_COLUMN] == 31, TARGET_COLUMN].item(), 0)
        self.assertEqual(labeled.loc[labeled[RUL_COLUMN] == 30, TARGET_COLUMN].item(), 1)
        self.assertEqual(labeled.loc[labeled[RUL_COLUMN] == 0, TARGET_COLUMN].item(), 1)
        self.assertEqual(int(labeled[TARGET_COLUMN].sum()), 31)

    def test_each_asset_has_its_own_event_cycle(self) -> None:
        frame = pd.DataFrame(
            {
                "unit_id": [1, 1, 1, 2, 2],
                "cycle": [1, 2, 3, 1, 2],
            }
        )
        labeled = construct_train_targets(frame, horizon=1)
        self.assertEqual(labeled[RUL_COLUMN].tolist(), [2, 1, 0, 1, 0])
        self.assertEqual(labeled[TARGET_COLUMN].tolist(), [0, 1, 1, 1, 1])
        self.assertGreaterEqual(int(labeled[RUL_COLUMN].min()), 0)

    def test_target_construction_does_not_mutate_source(self) -> None:
        frame = pd.DataFrame({"unit_id": [1, 1], "cycle": [1, 2]})
        before = frame.copy(deep=True)
        construct_train_targets(frame, horizon=1)
        pd.testing.assert_frame_equal(frame, before)

    def test_duplicate_asset_cycle_is_rejected(self) -> None:
        frame = pd.DataFrame({"unit_id": [1, 1], "cycle": [1, 1]})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            construct_train_targets(frame)

    def test_invalid_horizon_is_rejected(self) -> None:
        frame = pd.DataFrame({"unit_id": [1], "cycle": [1]})
        for bad_horizon in (0, -1):
            with self.subTest(horizon=bad_horizon), self.assertRaises(ValueError):
                construct_train_targets(frame, horizon=bad_horizon)
        with self.assertRaises(TypeError):
            construct_train_targets(frame, horizon=2.5)  # type: ignore[arg-type]


class OfficialTestTargetTests(unittest.TestCase):
    def test_official_final_rul_offsets_reconstruct_all_rows(self) -> None:
        test = pd.DataFrame(
            {
                "unit_id": [1, 1, 1, 2, 2],
                "cycle": [1, 2, 3, 1, 2],
            }
        )
        final_rul = pd.DataFrame({"unit_id": [1, 2], "final_rul": [2, 0]})
        labeled = construct_test_targets(test, final_rul, horizon=2)
        self.assertEqual(labeled[EVENT_CYCLE_COLUMN].tolist(), [5, 5, 5, 2, 2])
        self.assertEqual(labeled[RUL_COLUMN].tolist(), [4, 3, 2, 1, 0])
        self.assertEqual(labeled[TARGET_COLUMN].tolist(), [0, 0, 1, 1, 1])

    def test_final_rul_asset_mismatch_is_rejected(self) -> None:
        test = pd.DataFrame({"unit_id": [1, 1], "cycle": [1, 2]})
        final_rul = pd.DataFrame({"unit_id": [2], "final_rul": [3]})
        with self.assertRaisesRegex(ValueError, "match test assets"):
            construct_test_targets(test, final_rul)

    def test_negative_or_fractional_final_rul_is_rejected(self) -> None:
        test = pd.DataFrame({"unit_id": [1], "cycle": [1]})
        for value in (-1, 1.5):
            final_rul = pd.DataFrame({"unit_id": [1], "final_rul": [value]})
            with self.subTest(value=value), self.assertRaises(ValueError):
                construct_test_targets(test, final_rul)


class LeakageFirewallTests(unittest.TestCase):
    def test_outcome_columns_are_rejected_as_predictors(self) -> None:
        for forbidden in (
            RUL_COLUMN,
            "rul",
            "final_rul",
            EVENT_CYCLE_COLUMN,
            TARGET_COLUMN,
            "future_sensor_2",
            "failure_target",
        ):
            with self.subTest(column=forbidden), self.assertRaises(ValueError):
                assert_predictors_are_leakage_safe(["asset_age", forbidden])

    def test_trailing_features_are_accepted(self) -> None:
        assert_predictors_are_leakage_safe(
            ["asset_age", "current__sensor_2", "sensor_2__roll_mean_5"]
        )

    def test_derived_outcome_and_fmea_names_are_rejected(self) -> None:
        derived_names = (
            "rul_squared",
            "RULScaled",
            "remaining_useful_life_roll_mean",
            "event_cycle_lag",
            "eventWithinHorizonCopy",
            "target_probability",
            "failure_flag_rolling",
            "static_rpn_normalized",
            "severity_score",
            "occurrence_rank",
            "detection_scaled",
            "s_score",
            "o_rank",
            "d_value",
        )
        for column in derived_names:
            with self.subTest(column=column), self.assertRaises(ValueError):
                assert_predictors_are_leakage_safe(["asset_age", column])


class LoaderTests(unittest.TestCase):
    def test_loads_official_three_file_shape(self) -> None:
        row = np.arange(1, len(CMAPSS_COLUMNS) + 1, dtype=float)
        row[0], row[1] = 1, 1
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            text = " ".join(str(value) for value in row) + "\n"
            (directory / "train_FD001.txt").write_text(text, encoding="utf-8")
            (directory / "test_FD001.txt").write_text(text, encoding="utf-8")
            (directory / "RUL_FD001.txt").write_text("7\n", encoding="utf-8")
            loaded = load_cmapss(directory)
        self.assertEqual(loaded.subset, "FD001")
        self.assertEqual(loaded.train.columns.tolist(), list(CMAPSS_COLUMNS))
        self.assertEqual(loaded.test_final_rul.to_dict("records"), [{"unit_id": 1, "final_rul": 7}])


if __name__ == "__main__":
    unittest.main()
