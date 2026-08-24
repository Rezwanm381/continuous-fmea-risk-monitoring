"""Tests for asset-wise split leakage controls and trajectory validation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data_loader import CMAPSS_COLUMNS
from src.data_validation import (
    DataValidationError,
    assert_no_asset_overlap,
    make_group_kfold_splits,
    validate_cmapss_frame,
)


def make_valid_cmapss() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for asset in (1, 2, 3):
        for cycle in (1, 2, 3):
            record = {column: 1.0 for column in CMAPSS_COLUMNS}
            record["unit_id"] = asset
            record["cycle"] = cycle
            record["sensor_2"] = asset * 10 + cycle
            rows.append(record)
    return pd.DataFrame.from_records(rows, columns=CMAPSS_COLUMNS)


class GroupSplitTests(unittest.TestCase):
    def test_group_kfold_has_no_asset_overlap(self) -> None:
        groups = np.repeat(np.arange(1, 7), 3)
        splits = make_group_kfold_splits(groups, n_splits=3)
        validation_counts = np.zeros(len(groups), dtype=int)
        for train_index, validation_index in splits:
            train_assets = set(groups[train_index])
            validation_assets = set(groups[validation_index])
            self.assertTrue(train_assets.isdisjoint(validation_assets))
            validation_counts[validation_index] += 1
        np.testing.assert_array_equal(validation_counts, np.ones(len(groups), dtype=int))

    def test_overlap_assertion_fails_loudly(self) -> None:
        with self.assertRaisesRegex(DataValidationError, "Asset leakage"):
            assert_no_asset_overlap([1, 2, 3], [3, 4])

    def test_disjoint_official_asset_namespaces_pass(self) -> None:
        assert_no_asset_overlap(["train-1", "train-2"], ["test-1", "test-2"])

    def test_split_count_cannot_exceed_assets(self) -> None:
        with self.assertRaisesRegex(DataValidationError, "unique assets"):
            make_group_kfold_splits([1, 1, 2, 2], n_splits=3)

    def test_group_values_cannot_be_missing(self) -> None:
        with self.assertRaises(DataValidationError):
            make_group_kfold_splits([1, 1, np.nan, np.nan], n_splits=2)


class DataValidationTests(unittest.TestCase):
    def test_quality_summary_counts_assets_rows_and_prevalence(self) -> None:
        frame = make_valid_cmapss()
        frame["event_within_horizon"] = [0, 0, 1] * 3
        summary = validate_cmapss_frame(
            frame, target_column="event_within_horizon"
        )
        self.assertEqual(summary.number_of_assets, 3)
        self.assertEqual(summary.number_of_rows, 9)
        self.assertAlmostEqual(summary.target_prevalence or 0.0, 1 / 3)
        self.assertIn("sensor_1", summary.constant_columns)
        self.assertNotIn("sensor_2", summary.constant_columns)

    def test_duplicate_asset_cycle_is_rejected(self) -> None:
        frame = pd.concat([make_valid_cmapss(), make_valid_cmapss().iloc[[0]]])
        with self.assertRaisesRegex(DataValidationError, "duplicate"):
            validate_cmapss_frame(frame)

    def test_out_of_order_cycles_are_rejected(self) -> None:
        frame = make_valid_cmapss()
        first_asset = frame["unit_id"] == 1
        frame.loc[first_asset, "cycle"] = [1, 3, 2]
        with self.assertRaisesRegex(DataValidationError, "strictly increasing"):
            validate_cmapss_frame(frame)

    def test_missing_required_column_is_rejected(self) -> None:
        frame = make_valid_cmapss().drop(columns="sensor_21")
        with self.assertRaisesRegex(DataValidationError, "Missing required"):
            validate_cmapss_frame(frame)

    def test_non_binary_target_is_rejected(self) -> None:
        frame = make_valid_cmapss()
        frame["event_within_horizon"] = 2
        with self.assertRaisesRegex(DataValidationError, "binary"):
            validate_cmapss_frame(frame, target_column="event_within_horizon")


if __name__ == "__main__":
    unittest.main()
