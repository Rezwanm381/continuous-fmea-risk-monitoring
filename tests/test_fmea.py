"""Tests for static FMEA context and transparent condition escalation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.decision_support import (
    AlertBands,
    condition_alert_level,
    condition_informed_prioritization,
    condition_policy_change_summary,
)
from src.fmea import (
    SCENARIO_BASIS,
    STATIC_RPN_COLUMN,
    add_static_rpn,
    build_static_fmea_scenario,
    calculate_static_rpn,
    validate_static_fmea,
)


class StaticFmeaTests(unittest.TestCase):
    def test_static_rpn_calculation(self) -> None:
        calculated = calculate_static_rpn([8, 9], [5, 4], [5, 5])
        np.testing.assert_array_equal(calculated, [200, 180])

    def test_static_scenario_is_narrow_and_explicitly_illustrative(self) -> None:
        scenario = build_static_fmea_scenario()
        self.assertEqual(len(scenario), 3)
        self.assertTrue((scenario["Scenario_Basis"] == SCENARIO_BASIS).all())
        np.testing.assert_array_equal(
            scenario[STATIC_RPN_COLUMN],
            scenario["Severity"] * scenario["Occurrence"] * scenario["Detection"],
        )
        self.assertFalse(any("dynamic" in column.lower() for column in scenario.columns))

    def test_add_static_rpn_returns_copy(self) -> None:
        frame = pd.DataFrame(
            {"Severity": [8], "Occurrence": [5], "Detection": [4]}
        )
        before = frame.copy(deep=True)
        output = add_static_rpn(frame)
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(output[STATIC_RPN_COLUMN].iloc[0], 160)

    def test_invalid_rating_is_rejected(self) -> None:
        for severity in (0, 11, 4.5):
            with self.subTest(severity=severity), self.assertRaises(ValueError):
                calculate_static_rpn([severity], [5], [5])

    def test_static_rpn_mismatch_is_rejected(self) -> None:
        scenario = build_static_fmea_scenario()
        scenario.loc[0, STATIC_RPN_COLUMN] += 1
        with self.assertRaisesRegex(ValueError, "must equal"):
            validate_static_fmea(scenario)

    def test_dynamic_rpn_field_is_rejected(self) -> None:
        scenario = build_static_fmea_scenario()
        scenario["Dynamic_RPN"] = scenario[STATIC_RPN_COLUMN]
        with self.assertRaisesRegex(ValueError, "prohibited"):
            validate_static_fmea(scenario)


class AlertBandTests(unittest.TestCase):
    def test_default_alert_boundaries_are_inclusive(self) -> None:
        expected = {
            0.0: "LOW",
            0.199: "LOW",
            0.2: "MODERATE",
            0.5: "HIGH",
            0.8: "CRITICAL",
            1.0: "CRITICAL",
        }
        for probability, alert in expected.items():
            with self.subTest(probability=probability):
                self.assertEqual(condition_alert_level(probability), alert)

    def test_band_thresholds_must_be_ordered(self) -> None:
        with self.assertRaises(ValueError):
            AlertBands(moderate=0.5, high=0.4, critical=0.8)

    def test_probability_outside_unit_interval_is_rejected(self) -> None:
        for probability in (-0.1, 1.1):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                condition_alert_level(probability)


class DecisionSupportTests(unittest.TestCase):
    def test_condition_logic_preserves_all_static_scores_and_input(self) -> None:
        static = build_static_fmea_scenario()
        before = static.copy(deep=True)
        prioritized = condition_informed_prioritization(static, [0.1, 0.55, 0.85])
        pd.testing.assert_frame_equal(static, before)
        pd.testing.assert_frame_equal(
            prioritized[["Severity", "Occurrence", "Detection", STATIC_RPN_COLUMN]],
            static[["Severity", "Occurrence", "Detection", STATIC_RPN_COLUMN]],
        )
        self.assertEqual(
            prioritized["Condition_Alert"].tolist(), ["LOW", "HIGH", "CRITICAL"]
        )
        self.assertEqual(
            prioritized["Condition_Escalated"].tolist(), [False, True, True]
        )
        self.assertFalse(any("dynamic_rpn" in column.lower() for column in prioritized.columns))

    def test_high_severity_critical_alert_has_explicit_urgency(self) -> None:
        static = build_static_fmea_scenario()
        prioritized = condition_informed_prioritization(static, 0.9)
        high_severity = prioritized.loc[prioritized["Severity"] >= 9].iloc[0]
        self.assertEqual(
            high_severity["Recommended_Urgency"], "IMMEDIATE_ENGINEERING_REVIEW"
        )
        self.assertEqual(high_severity["Severity"], 9)

    def test_failure_mode_mapping_aligns_probabilities(self) -> None:
        static = build_static_fmea_scenario()
        mapping = {
            failure_mode: probability
            for failure_mode, probability in zip(static["Failure_Mode"], [0.1, 0.3, 0.9])
        }
        prioritized = condition_informed_prioritization(static, mapping)
        np.testing.assert_allclose(prioritized["Condition_Probability"], [0.1, 0.3, 0.9])

    def test_missing_failure_mode_probability_is_rejected(self) -> None:
        static = build_static_fmea_scenario()
        with self.assertRaises(ValueError):
            condition_informed_prioritization(
                static, {static["Failure_Mode"].iloc[0]: 0.5}
            )

    def test_policy_change_summary_counts_escalations(self) -> None:
        prioritized = condition_informed_prioritization(
            build_static_fmea_scenario(), [0.1, 0.55, 0.85]
        )
        summary = condition_policy_change_summary(prioritized)
        self.assertEqual(summary["fmea_items"], 3)
        self.assertEqual(summary["escalated_items"], 2)
        self.assertEqual(summary["high_or_critical_alert_items"], 2)


if __name__ == "__main__":
    unittest.main()
