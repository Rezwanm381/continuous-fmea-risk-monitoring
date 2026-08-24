"""Read NASA C-MAPSS text files without downloading or altering source data.

Official C-MAPSS subsets use whitespace-delimited, headerless files.  Each
trajectory row has an engine identifier, cycle, three operating settings, and
21 sensor channels.  The official test RUL file has one final-RUL value per
test engine, in engine-id order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


UNIT_COLUMN = "unit_id"
CYCLE_COLUMN = "cycle"
SETTING_COLUMNS = tuple(f"op_setting_{index}" for index in range(1, 4))
SENSOR_COLUMNS = tuple(f"sensor_{index}" for index in range(1, 22))
CMAPSS_COLUMNS = (UNIT_COLUMN, CYCLE_COLUMN, *SETTING_COLUMNS, *SENSOR_COLUMNS)


@dataclass(frozen=True)
class CMapssData:
    """The three files forming an official C-MAPSS subset."""

    train: pd.DataFrame
    test: pd.DataFrame
    test_final_rul: pd.DataFrame
    subset: str
    source_directory: Path


def _normalise_subset(subset: str) -> str:
    value = str(subset).strip().upper()
    if not re.fullmatch(r"FD\d{3}", value):
        raise ValueError("subset must have the form 'FD001'")
    return value


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required C-MAPSS file not found: {path}")
    return path


def load_trajectory_file(path: str | Path) -> pd.DataFrame:
    """Load an official ``train_*.txt`` or ``test_*.txt`` trajectory file."""

    source = _require_file(Path(path))
    frame = pd.read_csv(source, sep=r"\s+", header=None, engine="python")
    if frame.shape[1] != len(CMAPSS_COLUMNS):
        raise ValueError(
            f"Expected {len(CMAPSS_COLUMNS)} C-MAPSS columns in {source}, "
            f"found {frame.shape[1]}"
        )
    frame.columns = list(CMAPSS_COLUMNS)
    unit_values = pd.to_numeric(frame[UNIT_COLUMN], errors="raise")
    cycle_values = pd.to_numeric(frame[CYCLE_COLUMN], errors="raise")
    if (
        unit_values.isna().any()
        or cycle_values.isna().any()
        or not np.isfinite(unit_values).all()
        or not np.isfinite(cycle_values).all()
        or not np.equal(unit_values, np.floor(unit_values)).all()
        or not np.equal(cycle_values, np.floor(cycle_values)).all()
    ):
        raise ValueError("C-MAPSS unit identifiers and cycles must be finite whole numbers")
    frame[UNIT_COLUMN] = unit_values.astype(int)
    frame[CYCLE_COLUMN] = cycle_values.astype(int)
    for column in (*SETTING_COLUMNS, *SENSOR_COLUMNS):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    return frame


def load_test_rul_file(path: str | Path) -> pd.DataFrame:
    """Load official per-engine final RUL values for a test subset."""

    source = _require_file(Path(path))
    frame = pd.read_csv(source, sep=r"\s+", header=None, engine="python")
    if frame.shape[1] != 1:
        raise ValueError(f"Expected one RUL column in {source}, found {frame.shape[1]}")
    values = pd.to_numeric(frame.iloc[:, 0], errors="raise")
    if (
        values.isna().any()
        or (values < 0).any()
        or not np.isfinite(values).all()
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError(
            "Official test final-RUL values must be finite non-negative whole cycles"
        )
    return pd.DataFrame(
        {
            UNIT_COLUMN: range(1, len(values) + 1),
            "final_rul": values.to_numpy(dtype=int),
        }
    )


def load_cmapss(directory: str | Path, subset: str = "FD001") -> CMapssData:
    """Load the official train, test, and RUL files for one C-MAPSS subset.

    This function performs local reads only.  It intentionally has no download
    behavior so data-rights and locked-test controls remain explicit.
    """

    data_directory = Path(directory)
    subset_name = _normalise_subset(subset)
    train = load_trajectory_file(data_directory / f"train_{subset_name}.txt")
    test = load_trajectory_file(data_directory / f"test_{subset_name}.txt")
    final_rul = load_test_rul_file(data_directory / f"RUL_{subset_name}.txt")

    test_units = sorted(test[UNIT_COLUMN].unique().tolist())
    expected_units = final_rul[UNIT_COLUMN].tolist()
    if test_units != expected_units:
        raise ValueError(
            "Test engine identifiers do not match the row order/count in the "
            "official RUL file"
        )
    return CMapssData(
        train=train,
        test=test,
        test_final_rul=final_rul,
        subset=subset_name,
        source_directory=data_directory.resolve(),
    )


__all__ = [
    "CMAPSS_COLUMNS",
    "CYCLE_COLUMN",
    "CMapssData",
    "SENSOR_COLUMNS",
    "SETTING_COLUMNS",
    "UNIT_COLUMN",
    "load_cmapss",
    "load_test_rul_file",
    "load_trajectory_file",
]
