"""Compact, static matplotlib figures for the technical prototype."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from .data_loader import CYCLE_COLUMN, UNIT_COLUMN


FIGURE_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.22,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def plot_precision_recall_curves(
    curves: Mapping[str, tuple[Sequence[int], Sequence[float]]],
    *,
    title: str = "Near-term event precision–recall",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot stepwise PR curves and render a constant score as prevalence."""

    if not curves:
        raise ValueError("At least one model curve is required")
    with plt.rc_context(FIGURE_STYLE):
        figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        for name, (y_true, probability) in curves.items():
            target = np.asarray(y_true).reshape(-1)
            scores = np.asarray(probability, dtype=float).reshape(-1)
            precision, recall, _ = precision_recall_curve(target, scores)
            average_precision = average_precision_score(target, scores)
            label = f"{name} (AP={average_precision:.3f})"
            if np.unique(scores).size == 1:
                axis.axhline(
                    average_precision,
                    linewidth=1.7,
                    linestyle=":",
                    color="#64748b",
                    label=f"{name} / prevalence ({average_precision:.3f})",
                )
            else:
                axis.step(recall, precision, where="post", linewidth=2, label=label)
        axis.set(xlabel="Recall", ylabel="Precision", title=title)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.legend(frameon=False)
    return figure, axis


def plot_calibration_curves(
    calibration_tables: Mapping[str, pd.DataFrame],
    *,
    policy_threshold: float | None = None,
    sample_count: int | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot full-scale reliability evidence with explicit support limitations."""

    required = {"mean_predicted_probability", "observed_event_rate"}
    if not calibration_tables:
        raise ValueError("At least one calibration table is required")
    with plt.rc_context(FIGURE_STYLE):
        figure, (axis, support_axis) = plt.subplots(
            1,
            2,
            figsize=(10.4, 4.8),
            gridspec_kw={"width_ratios": [1.35, 1.0]},
            constrained_layout=True,
        )
        axis.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", label="Ideal reference")
        support_lines: list[str] = []
        for name, table in calibration_tables.items():
            missing = required.difference(table.columns)
            if missing:
                raise ValueError(f"Calibration table {name!r} missing {sorted(missing)}")
            axis.plot(
                table["mean_predicted_probability"],
                table["observed_event_rate"],
                marker="o",
                linewidth=2,
                label=name,
            )
            positive_bins = int((table["observed_event_rate"] > 0).sum())
            if sample_count is not None and sample_count > 0:
                smallest_bin, remainder = divmod(sample_count, len(table))
                largest_bin = smallest_bin + int(remainder > 0)
                count_text = f"• rows per quantile bin: {smallest_bin}–{largest_bin}"
            else:
                count_text = "• row counts: see calibration diagnostics"
            support_lines.extend(
                [
                    name,
                    f"• quantile bins: {len(table)}",
                    count_text,
                    f"• bins containing events: {positive_bins}/{len(table)}",
                    "",
                ]
            )
        if policy_threshold is not None:
            if not 0 <= policy_threshold <= 1:
                raise ValueError("policy_threshold must lie in [0, 1]")
            axis.axvline(
                policy_threshold,
                color="#d97706",
                linestyle=":",
                linewidth=1.7,
                label=f"Policy threshold = {policy_threshold:.3f}",
            )
        axis.set(
            xlabel="Mean predicted probability",
            ylabel="Observed event rate",
            title="Locked-test reliability (diagnostic only)",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis.legend(frameon=False)
        support_axis.axis("off")
        support_axis.text(
            0,
            1,
            "\n".join(support_lines).rstrip()
            + "\n\nMost quantile bins are near zero.\n"
            + "This plot does not establish fleet calibration\n"
            + "or reliability near the action threshold.",
            ha="left",
            va="top",
            fontsize=9.5,
            linespacing=1.35,
            color="#334155",
        )
    return figure, axis


def plot_asset_risk_trajectory(
    frame: pd.DataFrame,
    asset_id: object,
    threshold: float,
    *,
    probability_column: str = "condition_probability",
    event_cycle_column: str = "event_cycle",
) -> tuple[plt.Figure, plt.Axes]:
    """Show one deterministically selected asset's risk, warning, and event cycles."""

    required = {UNIT_COLUMN, CYCLE_COLUMN, probability_column, event_cycle_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing trajectory plot columns: {missing}")
    asset = frame.loc[frame[UNIT_COLUMN] == asset_id].sort_values(CYCLE_COLUMN)
    if asset.empty:
        raise ValueError(f"Asset not found: {asset_id}")
    events = asset[event_cycle_column].unique()
    if len(events) != 1:
        raise ValueError("Asset must have one event cycle")
    warning_rows = asset.loc[asset[probability_column] >= threshold, CYCLE_COLUMN]
    first_warning = int(warning_rows.iloc[0]) if not warning_rows.empty else None
    event_cycle = int(events[0])
    with plt.rc_context(FIGURE_STYLE):
        figure, axis = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
        axis.plot(
            asset[CYCLE_COLUMN],
            asset[probability_column],
            color="#2563eb",
            linewidth=2,
            label="Condition probability",
        )
        axis.axhline(threshold, color="#d97706", linestyle="--", label="Policy threshold")
        axis.axvline(event_cycle, color="#991b1b", linestyle=":", label="Event cycle")
        if first_warning is not None:
            axis.axvline(first_warning, color="#047857", linestyle="--", label="First warning")
        axis.set(
            xlabel="Cycle",
            ylabel="Near-term event probability",
            title=f"Asset {asset_id}: condition trajectory",
            ylim=(0, 1.02),
        )
        axis.legend(frameon=False)
    return figure, axis


def plot_lead_time_distribution(
    per_asset: pd.DataFrame,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot actionable first-warning lead times across warned assets."""

    if "lead_time" not in per_asset.columns:
        raise ValueError("per_asset must contain lead_time")
    lead_times = pd.to_numeric(per_asset["lead_time"], errors="coerce").dropna()
    lead_times = lead_times.loc[lead_times > 0]
    if lead_times.empty:
        raise ValueError("No positive lead times are available")
    bins = min(12, max(3, int(np.sqrt(len(lead_times))) + 1))
    with plt.rc_context(FIGURE_STYLE):
        figure, axis = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
        axis.hist(lead_times, bins=bins, color="#2563eb", edgecolor="white")
        axis.axvline(
            float(lead_times.median()),
            color="#991b1b",
            linestyle="--",
            label=f"Median = {lead_times.median():.1f}",
        )
        axis.set(
            xlabel="First-warning lead time (cycles)",
            ylabel="Assets",
            title="Pre-event warning lead time",
        )
        axis.legend(frameon=False)
    return figure, axis


def plot_condition_informed_fmea(
    prioritized_fmea: pd.DataFrame,
) -> tuple[plt.Figure, plt.Axes]:
    """Separate static scenario ratings from one shared asset probability."""

    required = {"Failure_Mode", "Static_RPN", "Condition_Probability"}
    missing = sorted(required.difference(prioritized_fmea.columns))
    if missing:
        raise ValueError(f"Missing condition-informed FMEA columns: {missing}")
    labels = prioritized_fmea["Failure_Mode"].astype(str).tolist()
    positions = np.arange(len(labels))
    probabilities = prioritized_fmea["Condition_Probability"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Condition probabilities must lie in [0, 1]")
    if not np.allclose(probabilities, probabilities[0]):
        raise ValueError("This scenario figure expects one shared asset-level probability")
    shared_probability = float(probabilities[0])
    with plt.rc_context(FIGURE_STYLE):
        figure, (static_axis, condition_axis) = plt.subplots(
            1,
            2,
            figsize=(11.0, 4.8),
            gridspec_kw={"width_ratios": [1.55, 0.75]},
            constrained_layout=True,
        )
        static_axis.barh(
            positions,
            prioritized_fmea["Static_RPN"],
            color="#94a3b8",
            label="Static RPN",
        )
        static_axis.set_yticks(positions, labels)
        static_axis.invert_yaxis()
        static_axis.set_xlabel("Static RPN (scenario context only)")
        static_axis.set_title("Frozen engineering scenario\nS/O/D are assumptions")
        static_axis.grid(axis="x", alpha=0.22)

        condition_axis.axhspan(0.0, 0.2, color="#dcfce7", alpha=0.65)
        condition_axis.axhspan(0.2, 0.5, color="#fef3c7", alpha=0.65)
        condition_axis.axhspan(0.5, 0.8, color="#fed7aa", alpha=0.65)
        condition_axis.axhspan(0.8, 1.0, color="#fecaca", alpha=0.65)
        condition_axis.bar(
            [0],
            [shared_probability],
            width=0.55,
            color="#2563eb",
            label="Shared engine-level probability",
        )
        condition_axis.text(
            0,
            min(0.97, shared_probability + 0.04),
            f"{shared_probability:.3f}",
            ha="center",
            va="bottom",
            weight="bold",
        )
        condition_axis.set_xticks([0], ["Selected\nengine"])
        condition_axis.set_ylabel("30-cycle event probability")
        condition_axis.set_ylim(0, 1)
        condition_axis.set_title("One shared asset signal\nnot mode-specific")
        condition_axis.grid(axis="y", alpha=0.22)
        figure.suptitle(
            "Illustrative FMEA scenario + condition-informed urgency\n"
            "Policies are not operationally workload-matched; no dynamic RPN",
            fontsize=12.5,
            weight="bold",
        )
    return figure, static_axis


__all__ = [
    "plot_asset_risk_trajectory",
    "plot_calibration_curves",
    "plot_condition_informed_fmea",
    "plot_lead_time_distribution",
    "plot_precision_recall_curves",
]
