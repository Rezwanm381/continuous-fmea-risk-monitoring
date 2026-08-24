"""Asset/event-level warning coverage, lead time, and false-alert burden."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .data_loader import CYCLE_COLUMN, UNIT_COLUMN
from .target import EVENT_CYCLE_COLUMN, PREDICTION_HORIZON, TARGET_COLUMN


@dataclass(frozen=True)
class EventMetricSummary:
    number_of_assets: int
    warning_coverage: float
    median_lead_time: float
    fraction_never_warned: float
    fraction_warning_at_event_only: float
    total_false_alert_episodes: int
    mean_false_alert_episodes_per_asset: float
    fraction_assets_with_false_alerts: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @property
    def assets_with_false_alerts(self) -> float:
        """Backward-readable alias; the value is a fraction, not an asset count."""

        return self.fraction_assets_with_false_alerts


def count_alert_episodes(
    alert_mask: Sequence[bool] | np.ndarray,
    cycles: Sequence[int] | np.ndarray | None = None,
) -> int:
    """Count contiguous alert runs; a cycle gap starts a new episode."""

    alerts = np.asarray(alert_mask, dtype=bool).reshape(-1)
    if alerts.size == 0:
        return 0
    if cycles is None:
        cycle_values = np.arange(alerts.size)
    else:
        cycle_values = np.asarray(cycles).reshape(-1)
        if cycle_values.size != alerts.size:
            raise ValueError("cycles must align with alert_mask")
        if np.any(np.diff(cycle_values.astype(float)) <= 0):
            raise ValueError("cycles must be strictly increasing")
    starts = alerts.copy()
    starts[1:] &= (~alerts[:-1]) | (np.diff(cycle_values.astype(float)) > 1)
    return int(starts.sum())


def _validate_event_frame(
    frame: pd.DataFrame,
    probability_column: str,
    event_cycle_column: str | None,
    target_column: str | None,
) -> None:
    required = {UNIT_COLUMN, CYCLE_COLUMN, probability_column}
    if event_cycle_column is not None:
        required.add(event_cycle_column)
    if target_column is not None:
        required.add(target_column)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing event metric columns: {missing}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Event metric input must be non-empty and complete")
    probability = frame[probability_column].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Condition probabilities must be finite and lie in [0, 1]")
    if frame.duplicated([UNIT_COLUMN, CYCLE_COLUMN]).any():
        raise ValueError("Duplicate asset-cycle observations are not allowed")
    if target_column is not None:
        target_values = set(frame[target_column].unique().tolist())
        if not target_values.issubset({0, 1, False, True}):
            raise ValueError("Target column must be binary")


def asset_event_metrics(
    frame: pd.DataFrame,
    threshold: float,
    *,
    probability_column: str = "condition_probability",
    event_cycle_column: str | None = EVENT_CYCLE_COLUMN,
    target_column: str | None = TARGET_COLUMN,
    horizon: int = PREDICTION_HORIZON,
) -> pd.DataFrame:
    """Compute one transparent warning record per asset.

    ``warning_coverage`` later counts only warnings strictly before the event.
    When the caller needs qualifying-window coverage, it must pass only rows
    inside that window; full-history calls intentionally retain earlier alerts
    for false-alert accounting.
    A threshold crossing at the event is retained as ``warning_at_event_only``
    but has zero actionable lead time.  A false alert is a threshold crossing
    while the independently constructed future-event target is zero.  When no
    target column is supplied, cycles earlier than ``event_cycle - horizon``
    provide the equivalent definition.
    """

    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    if isinstance(horizon, bool) or int(horizon) <= 0:
        raise ValueError("horizon must be positive")
    resolved_event_column = (
        event_cycle_column if event_cycle_column in frame.columns else None
    )
    resolved_target_column = target_column if target_column in frame.columns else None
    _validate_event_frame(
        frame, probability_column, resolved_event_column, resolved_target_column
    )

    records: list[dict[str, object]] = []
    ordered = frame.sort_values([UNIT_COLUMN, CYCLE_COLUMN], kind="mergesort")
    for asset, asset_frame in ordered.groupby(UNIT_COLUMN, sort=True):
        cycles = asset_frame[CYCLE_COLUMN].to_numpy(dtype=int)
        if np.any(np.diff(cycles) <= 0):
            raise ValueError(f"Cycles are not strictly increasing for asset {asset}")
        if resolved_event_column is None:
            event_cycle = int(cycles.max())
        else:
            event_values = asset_frame[resolved_event_column].unique()
            if len(event_values) != 1:
                raise ValueError(f"Asset {asset} has inconsistent event cycles")
            numeric_event = float(event_values[0])
            if not np.isfinite(numeric_event) or numeric_event != np.floor(numeric_event):
                raise ValueError(f"Asset {asset} event cycle must be a finite whole cycle")
            event_cycle = int(numeric_event)
        if event_cycle < int(cycles.max()):
            raise ValueError(f"Event cycle precedes observed data for asset {asset}")

        probabilities = asset_frame[probability_column].to_numpy(dtype=float)
        warning_mask = (probabilities >= threshold) & (cycles <= event_cycle)
        warning_cycles = cycles[warning_mask]
        first_warning = int(warning_cycles.min()) if warning_cycles.size else None
        lead_time = event_cycle - first_warning if first_warning is not None else np.nan
        warned_before = bool(first_warning is not None and first_warning < event_cycle)
        warned_at_event_only = bool(first_warning == event_cycle)

        if resolved_target_column is not None:
            non_event_window = asset_frame[resolved_target_column].to_numpy(dtype=int) == 0
        else:
            non_event_window = cycles < (event_cycle - int(horizon))
        false_alert_mask = warning_mask & non_event_window
        records.append(
            {
                UNIT_COLUMN: asset,
                "event_cycle": event_cycle,
                "first_warning_cycle": first_warning,
                "lead_time": float(lead_time),
                "warned_at_or_before_event": bool(first_warning is not None),
                "warned_before_event": warned_before,
                "warning_at_event_only": warned_at_event_only,
                "never_warned": bool(first_warning is None),
                "alert_rows": int(warning_mask.sum()),
                "alert_episodes": count_alert_episodes(warning_mask, cycles),
                "false_alert_rows": int(false_alert_mask.sum()),
                "false_alert_episodes": count_alert_episodes(false_alert_mask, cycles),
            }
        )
    return pd.DataFrame.from_records(records)


def summarize_event_metrics(per_asset: pd.DataFrame) -> EventMetricSummary:
    """Summarize the per-asset event table without treating rows as independent."""

    required = {
        "warned_before_event",
        "warning_at_event_only",
        "never_warned",
        "lead_time",
        "false_alert_episodes",
    }
    missing = sorted(required.difference(per_asset.columns))
    if missing or per_asset.empty:
        raise ValueError(f"Invalid per-asset event table; missing={missing}")
    actionable_leads = pd.to_numeric(
        per_asset.loc[per_asset["warned_before_event"], "lead_time"], errors="coerce"
    ).dropna()
    median_lead = float(actionable_leads.median()) if not actionable_leads.empty else np.nan
    false_episodes = pd.to_numeric(per_asset["false_alert_episodes"], errors="raise")
    return EventMetricSummary(
        number_of_assets=int(len(per_asset)),
        warning_coverage=float(per_asset["warned_before_event"].mean()),
        median_lead_time=median_lead,
        fraction_never_warned=float(per_asset["never_warned"].mean()),
        fraction_warning_at_event_only=float(per_asset["warning_at_event_only"].mean()),
        total_false_alert_episodes=int(false_episodes.sum()),
        mean_false_alert_episodes_per_asset=float(false_episodes.mean()),
        fraction_assets_with_false_alerts=float((false_episodes > 0).mean()),
    )


def event_outputs(
    target_frame: pd.DataFrame,
    probability: Sequence[float] | np.ndarray,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """Build qualifying-warning and all-exposure event outputs.

    Coverage and lead time use only assets with released rows inside the target
    window. False-alert burden uses every supplied asset and every observed
    target-negative row. Keeping this orchestration beside the lower-level
    event metrics makes the eligibility/exposure boundary reusable and testable.
    """

    event_frame = target_frame[
        [UNIT_COLUMN, CYCLE_COLUMN, EVENT_CYCLE_COLUMN, TARGET_COLUMN]
    ].copy()
    event_frame["condition_probability"] = np.asarray(probability, dtype=float)
    full_asset = asset_event_metrics(event_frame, threshold)
    eligible_window = event_frame.loc[event_frame[TARGET_COLUMN] == 1].copy()
    qualifying = asset_event_metrics(eligible_window, threshold)
    qualifying = qualifying.drop(
        columns=["false_alert_rows", "false_alert_episodes"]
    ).merge(
        full_asset[[UNIT_COLUMN, "false_alert_rows", "false_alert_episodes"]],
        on=UNIT_COLUMN,
        how="left",
        validate="one_to_one",
    )
    warning_summary = summarize_event_metrics(qualifying)
    summary: dict[str, float | int] = warning_summary.to_dict()
    summary.update(
        {
            "eligible_assets": int(len(qualifying)),
            "all_test_assets": int(len(full_asset)),
            "total_false_alert_rows": int(full_asset["false_alert_rows"].sum()),
            "total_false_alert_episodes_all_assets": int(
                full_asset["false_alert_episodes"].sum()
            ),
            "mean_false_alert_rows_per_asset": float(
                full_asset["false_alert_rows"].mean()
            ),
            "false_alert_rows_per_1000_negative_rows": float(
                1000.0
                * full_asset["false_alert_rows"].sum()
                / max(1, int((event_frame[TARGET_COLUMN] == 0).sum()))
            ),
            "mean_false_alert_episodes_per_asset_all": float(
                full_asset["false_alert_episodes"].mean()
            ),
            "fraction_all_assets_with_false_alerts": float(
                (full_asset["false_alert_episodes"] > 0).mean()
            ),
        }
    )
    return qualifying, full_asset, summary


def select_representative_asset(per_asset: pd.DataFrame) -> object:
    """Select the deterministic median-lead warned asset after model finalization."""

    required = {UNIT_COLUMN, "lead_time", "warned_before_event"}
    missing = sorted(required.difference(per_asset.columns))
    if missing:
        raise ValueError(f"per_asset missing representative-selection columns: {missing}")
    candidates = per_asset.loc[
        per_asset["warned_before_event"].astype(bool)
    ].copy()
    if candidates.empty:
        raise ValueError("No asset received a pre-event warning")
    median = float(candidates["lead_time"].median())
    candidates["_distance"] = (candidates["lead_time"] - median).abs()
    candidates["_asset_key"] = candidates[UNIT_COLUMN].astype(str)
    selected = candidates.sort_values(
        ["_distance", "_asset_key"], kind="mergesort"
    ).iloc[0]
    return selected[UNIT_COLUMN]


__all__ = [
    "EventMetricSummary",
    "asset_event_metrics",
    "count_alert_episodes",
    "event_outputs",
    "select_representative_asset",
    "summarize_event_metrics",
]
