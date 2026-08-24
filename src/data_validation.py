"""Data-quality checks and asset-separated validation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .data_loader import CMAPSS_COLUMNS, CYCLE_COLUMN, UNIT_COLUMN


class DataValidationError(ValueError):
    """Raised when C-MAPSS structure or leakage controls are violated."""


@dataclass(frozen=True)
class DataQualitySummary:
    number_of_assets: int
    number_of_rows: int
    missing_cells: int
    duplicate_asset_cycles: int
    out_of_order_assets: int
    cycle_gap_count: int
    constant_columns: tuple[str, ...]
    near_constant_columns: tuple[str, ...]
    target_prevalence: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _as_1d_groups(groups: Iterable[object]) -> np.ndarray:
    values = np.asarray(list(groups) if not isinstance(groups, np.ndarray) else groups)
    if values.ndim != 1 or values.size == 0:
        raise DataValidationError("groups must be a non-empty one-dimensional sequence")
    if pd.isna(values).any():
        raise DataValidationError("asset groups cannot contain missing values")
    return values


def assert_no_asset_overlap(
    development_assets: Iterable[object], validation_assets: Iterable[object]
) -> None:
    """Fail loudly if an asset is present on both sides of a split."""

    development = set(_as_1d_groups(development_assets).tolist())
    validation = set(_as_1d_groups(validation_assets).tolist())
    overlap = development.intersection(validation)
    if overlap:
        preview = sorted(overlap, key=str)[:10]
        raise DataValidationError(f"Asset leakage detected; overlapping assets: {preview}")


def make_group_kfold_splits(
    groups: Sequence[object] | np.ndarray, n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic GroupKFold indices and verify every fold."""

    group_values = _as_1d_groups(groups)
    unique_group_count = np.unique(group_values).size
    if not 2 <= n_splits <= unique_group_count:
        raise DataValidationError(
            f"n_splits must be between 2 and the {unique_group_count} unique assets"
        )
    dummy = np.zeros((group_values.size, 1), dtype=float)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    validation_counts = np.zeros(group_values.size, dtype=int)
    for train_index, validation_index in GroupKFold(n_splits=n_splits).split(
        dummy, groups=group_values
    ):
        assert_no_asset_overlap(group_values[train_index], group_values[validation_index])
        validation_counts[validation_index] += 1
        splits.append((train_index, validation_index))
    if not np.all(validation_counts == 1):
        raise DataValidationError("Each row must occur in exactly one validation fold")
    return splits


def validate_cmapss_frame(
    frame: pd.DataFrame,
    *,
    required_columns: Sequence[str] = CMAPSS_COLUMNS,
    target_column: str | None = None,
) -> DataQualitySummary:
    """Validate trajectory keys, timing, numeric values, and summarize quality."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataValidationError("C-MAPSS trajectory frame must be a non-empty DataFrame")
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")
    if target_column is not None and target_column not in frame.columns:
        raise DataValidationError(f"Target column not found: {target_column}")

    required = list(dict.fromkeys([*required_columns, *([target_column] if target_column else [])]))
    non_numeric = [column for column in required if not pd.api.types.is_numeric_dtype(frame[column])]
    if non_numeric:
        raise DataValidationError(f"Expected numeric columns: {non_numeric}")
    numeric_values = frame[required].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise DataValidationError("Required data contain missing or non-finite values")
    if (frame[UNIT_COLUMN] <= 0).any() or (frame[CYCLE_COLUMN] <= 0).any():
        raise DataValidationError("unit_id and cycle must be positive")

    duplicate_count = int(frame.duplicated([UNIT_COLUMN, CYCLE_COLUMN]).sum())
    if duplicate_count:
        raise DataValidationError(
            f"Found {duplicate_count} duplicate (unit_id, cycle) observations"
        )

    out_of_order = 0
    cycle_gap_count = 0
    for _, asset_frame in frame.groupby(UNIT_COLUMN, sort=False):
        cycles = asset_frame[CYCLE_COLUMN].to_numpy()
        differences = np.diff(cycles)
        if np.any(differences <= 0):
            out_of_order += 1
        cycle_gap_count += int(np.sum(differences > 1))
    if out_of_order:
        raise DataValidationError(
            f"Cycles must be strictly increasing within each asset; failed assets: {out_of_order}"
        )

    candidate_columns = [
        column
        for column in required_columns
        if column not in {UNIT_COLUMN, CYCLE_COLUMN}
    ]
    unique_counts = frame[candidate_columns].nunique(dropna=False)
    constant_columns = tuple(unique_counts[unique_counts <= 1].index.tolist())
    near_constant_columns: list[str] = []
    for column in candidate_columns:
        frequencies = frame[column].value_counts(normalize=True, dropna=False)
        if len(frequencies) > 1 and float(frequencies.iloc[0]) >= 0.99:
            near_constant_columns.append(column)

    prevalence: float | None = None
    if target_column is not None:
        target_values = set(frame[target_column].unique().tolist())
        if not target_values.issubset({0, 1, False, True}):
            raise DataValidationError("Target must be binary")
        prevalence = float(frame[target_column].mean())

    return DataQualitySummary(
        number_of_assets=int(frame[UNIT_COLUMN].nunique()),
        number_of_rows=int(len(frame)),
        missing_cells=int(frame[required].isna().sum().sum()),
        duplicate_asset_cycles=duplicate_count,
        out_of_order_assets=out_of_order,
        cycle_gap_count=cycle_gap_count,
        constant_columns=constant_columns,
        near_constant_columns=tuple(near_constant_columns),
        target_prevalence=prevalence,
    )


__all__ = [
    "DataQualitySummary",
    "DataValidationError",
    "assert_no_asset_overlap",
    "make_group_kfold_splits",
    "validate_cmapss_frame",
]
