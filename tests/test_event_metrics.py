"""Manual event-level checks for warnings, lead time, and false episodes."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.event_metrics import (
    asset_event_metrics,
    count_alert_episodes,
    select_representative_asset,
    summarize_event_metrics,
)


def event_frame() -> pd.DataFrame:
    probabilities = {
        1: [0.6, 0.7, 0.1, 0.2, 0.8, 0.9],
        2: [0.1, 0.2, 0.3, 0.4, 0.45, 0.7],
        3: [0.1, 0.2, 0.1, 0.2, 0.3, 0.4],
    }
    rows: list[dict[str, float | int]] = []
    for asset, scores in probabilities.items():
        for cycle, score in enumerate(scores, start=1):
            rows.append(
                {
                    "unit_id": asset,
                    "cycle": cycle,
                    "event_cycle": 6,
                    "event_within_horizon": int(cycle >= 4),
                    "condition_probability": score,
                }
            )
    return pd.DataFrame.from_records(rows)


class EpisodeTests(unittest.TestCase):
    def test_contiguous_false_alerts_form_one_episode(self) -> None:
        self.assertEqual(count_alert_episodes([True, True, False, True, True]), 2)

    def test_cycle_gap_starts_new_episode(self) -> None:
        self.assertEqual(count_alert_episodes([True, True], cycles=[1, 3]), 2)

    def test_non_increasing_cycles_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            count_alert_episodes([True, False], cycles=[2, 1])


class AssetEventMetricTests(unittest.TestCase):
    def test_first_warning_lead_and_false_episode_are_manual(self) -> None:
        per_asset = asset_event_metrics(event_frame(), threshold=0.5)
        asset_one = per_asset.set_index("unit_id").loc[1]
        self.assertEqual(asset_one["first_warning_cycle"], 1)
        self.assertEqual(asset_one["event_cycle"], 6)
        self.assertEqual(asset_one["lead_time"], 5)
        self.assertTrue(asset_one["warned_before_event"])
        self.assertEqual(asset_one["false_alert_rows"], 2)
        self.assertEqual(asset_one["false_alert_episodes"], 1)
        self.assertEqual(asset_one["alert_episodes"], 2)

    def test_warning_at_event_has_zero_lead_but_no_coverage(self) -> None:
        per_asset = asset_event_metrics(event_frame(), threshold=0.5)
        asset_two = per_asset.set_index("unit_id").loc[2]
        self.assertEqual(asset_two["first_warning_cycle"], 6)
        self.assertEqual(asset_two["lead_time"], 0)
        self.assertFalse(asset_two["warned_before_event"])
        self.assertTrue(asset_two["warning_at_event_only"])
        self.assertFalse(asset_two["never_warned"])

    def test_unwarned_asset_is_explicit(self) -> None:
        per_asset = asset_event_metrics(event_frame(), threshold=0.5)
        asset_three = per_asset.set_index("unit_id").loc[3]
        self.assertTrue(asset_three["never_warned"])
        self.assertTrue(np.isnan(asset_three["first_warning_cycle"]))
        self.assertTrue(np.isnan(asset_three["lead_time"]))

    def test_summary_uses_assets_not_rows(self) -> None:
        summary = summarize_event_metrics(asset_event_metrics(event_frame(), threshold=0.5))
        self.assertEqual(summary.number_of_assets, 3)
        self.assertAlmostEqual(summary.warning_coverage, 1 / 3)
        self.assertEqual(summary.median_lead_time, 5)
        self.assertAlmostEqual(summary.fraction_never_warned, 1 / 3)
        self.assertAlmostEqual(summary.fraction_warning_at_event_only, 1 / 3)
        self.assertEqual(summary.total_false_alert_episodes, 1)
        self.assertAlmostEqual(summary.mean_false_alert_episodes_per_asset, 1 / 3)

    def test_false_alert_fallback_uses_event_cycle_and_horizon(self) -> None:
        frame = pd.DataFrame(
            {
                "unit_id": [1] * 6,
                "cycle": np.arange(1, 7),
                "event_cycle": [6] * 6,
                "condition_probability": [0.1, 0.8, 0.1, 0.1, 0.8, 0.8],
            }
        )
        result = asset_event_metrics(
            frame,
            threshold=0.5,
            target_column=None,
            horizon=2,
        ).iloc[0]
        self.assertEqual(result["false_alert_rows"], 1)
        self.assertEqual(result["false_alert_episodes"], 1)

    def test_event_cycle_cannot_precede_observations(self) -> None:
        frame = event_frame().loc[lambda value: value["unit_id"] == 1].copy()
        frame["event_cycle"] = 5
        with self.assertRaisesRegex(ValueError, "precedes"):
            asset_event_metrics(frame, threshold=0.5)

    def test_analysis_separates_eligible_warnings_from_all_asset_false_alerts(self) -> None:
        from src.event_metrics import event_outputs

        frame = pd.DataFrame.from_records(
            [
                # Eligible asset: early false episode, then a qualifying warning.
                *(
                    {
                        "unit_id": 1,
                        "cycle": cycle,
                        "event_cycle": 5,
                        "event_within_horizon": int(cycle >= 3),
                    }
                    for cycle in range(1, 6)
                ),
                # Ineligible asset: released history never enters the event window.
                *(
                    {
                        "unit_id": 2,
                        "cycle": cycle,
                        "event_cycle": 10,
                        "event_within_horizon": 0,
                    }
                    for cycle in range(1, 4)
                ),
                # Eligible asset with no warning.
                *(
                    {
                        "unit_id": 3,
                        "cycle": cycle,
                        "event_cycle": 5,
                        "event_within_horizon": int(cycle >= 3),
                    }
                    for cycle in range(1, 6)
                ),
            ]
        )
        probability = np.array(
            [0.8, 0.8, 0.1, 0.8, 0.9, 0.1, 0.8, 0.1, *([0.1] * 5)]
        )
        qualifying, all_assets, summary = event_outputs(frame, probability, 0.5)

        self.assertEqual(qualifying["unit_id"].tolist(), [1, 3])
        asset_one = qualifying.set_index("unit_id").loc[1]
        self.assertEqual(asset_one["first_warning_cycle"], 4)
        self.assertEqual(asset_one["lead_time"], 1)
        self.assertEqual(summary["eligible_assets"], 2)
        self.assertEqual(summary["all_test_assets"], 3)
        self.assertEqual(summary["warning_coverage"], 0.5)
        self.assertEqual(summary["total_false_alert_rows"], 3)
        self.assertEqual(summary["total_false_alert_episodes_all_assets"], 2)
        self.assertAlmostEqual(summary["fraction_all_assets_with_false_alerts"], 2 / 3)
        self.assertEqual(
            all_assets.set_index("unit_id").loc[2, "false_alert_episodes"], 1
        )


class RepresentativeAssetTests(unittest.TestCase):
    def test_median_lead_asset_selection_is_deterministic(self) -> None:
        per_asset = pd.DataFrame(
            {
                "unit_id": [3, 1, 2],
                "lead_time": [8.0, 2.0, 5.0],
                "warned_before_event": [True, True, True],
            }
        )
        self.assertEqual(select_representative_asset(per_asset), 2)

    def test_representative_asset_requires_warning(self) -> None:
        per_asset = pd.DataFrame(
            {"unit_id": [1], "lead_time": [np.nan], "warned_before_event": [False]}
        )
        with self.assertRaises(ValueError):
            select_representative_asset(per_asset)


if __name__ == "__main__":
    unittest.main()
