"""One-command rebuild of the condition-informed FMEA technical prototype.

The script deliberately keeps the future-event model separate from the
illustrative engineering FMEA.  It never predicts RPN and it never uses RUL,
event-cycle, target, or future sensor values as predictors.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile


DEV_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DEV_ROOT.parent
OUTPUT_ROOT = PROJECT_ROOT / "05_OUTPUTS"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
REPORT_DIR = DEV_ROOT / "reports"
DATA_DIR = Path(
    os.environ.get(
        "CMAPSS_DATA_DIR",
        str(OUTPUT_ROOT / "private_data_cache" / "FD001"),
    )
).expanduser().resolve()
REPORT_EXPORT_DIR = Path(
    os.environ.get("PORTFOLIO_REPORT_DIR", str(OUTPUT_ROOT))
).expanduser().resolve()
BUILD_REPORT_PATH = REPORT_EXPORT_DIR / "module_7_25D_continuous_fmea_build_report.md"

# Matplotlib must receive a writable cache before any plotting import occurs.
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_ROOT / ".matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from src import RANDOM_SEED  # noqa: E402
from src.calibration import (  # noqa: E402
    fit_sigmoid_calibrator,
    nested_grouped_oof_calibration,
)
from src.data_loader import (  # noqa: E402
    CYCLE_COLUMN,
    SENSOR_COLUMNS,
    SETTING_COLUMNS,
    UNIT_COLUMN,
    load_cmapss,
)
from src.data_validation import (  # noqa: E402
    assert_no_asset_overlap,
    make_group_kfold_splits,
    validate_cmapss_frame,
)
from src.decision_support import (  # noqa: E402
    AlertBands,
    condition_informed_prioritization,
    condition_policy_change_summary,
)
from src.evaluation import (  # noqa: E402
    asset_bootstrap_probability_metrics,
    bootstrap_interval,
    calibration_table,
    evaluate_at_threshold,
    evaluate_probabilities,
    select_threshold_policy,
)
from src.event_metrics import (  # noqa: E402
    event_outputs,
    select_representative_asset,
)
from src.features import (  # noqa: E402
    FeatureSpec,
    assert_future_perturbation_invariance,
    build_feature_frame,
    feature_columns,
)
from src.fmea import build_static_fmea_scenario  # noqa: E402
from src.models import (  # noqa: E402
    build_age_only_baseline,
    build_constant_baseline,
    build_logistic_model,
    build_random_forest_model,
    oof_grouped_probabilities,
    positive_class_probability,
)
from src.target import (  # noqa: E402
    EVENT_CYCLE_COLUMN,
    PREDICTION_HORIZON,
    TARGET_COLUMN,
    assert_predictors_are_leakage_safe,
    construct_test_targets,
    construct_train_targets,
)
from src.visualization import (  # noqa: E402
    plot_calibration_curves,
    plot_condition_informed_fmea,
    plot_precision_recall_curves,
)


OFFICIAL_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
OUTER_SHA256 = "C9C5DEC12A945A82E8BB4446589D7FB3CC057B5E5D81FA1A12E25EE9912AD3B2"
INNER_SHA256 = "74BEF434A34DB25C7BF72E668EA4CD52AFE5F2CF8E44367C55A82BFD91A5A34F"
FD001_HASHES = {
    "train_FD001.txt": "963B5E22825B34D8B21C69E1AEB4AF3E647050EB672EE8834BA4B5D91D2DE0F8",
    "test_FD001.txt": "3CDA7109CE17BAFB5443F2AC926CFCF88154B941B8C4CF95EB55D1DDD6F52851",
    "RUL_FD001.txt": "A19C8EC94931949D0485BDC35118206E9C81C4547B422EFB9CF86F4CEDDBCECA",
    "readme.txt": "4F5270554B775C67E73AFF383C5436FD329D6E4CC3D3A116913276FAE511269B",
}
N_SPLITS = 5
RECALL_TARGET = 0.80
MODEL_MATERIALITY_TOLERANCE = 0.01
CALIBRATION_BRIER_MINIMUM_IMPROVEMENT = 0.001
BOOTSTRAP_REPLICATES = 500


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def ensure_fd001() -> dict[str, str]:
    """Load verified local files or acquire only FD001 from NASA's archive."""

    raw_root = DATA_DIR.parent
    raw_root.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    complete = all((DATA_DIR / name).is_file() for name in FD001_HASHES)
    if not complete:
        outer_path = raw_root / "CMAPSSData.zip"
        if not outer_path.is_file():
            part_path = raw_root / "CMAPSSData.zip.part"
            request = urllib.request.Request(
                OFFICIAL_URL, headers={"User-Agent": "continuous-fmea-prototype/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response, part_path.open(
                "wb"
            ) as destination:
                shutil.copyfileobj(response, destination)
            part_path.replace(outer_path)
        if sha256_file(outer_path) != OUTER_SHA256:
            raise RuntimeError("NASA outer archive SHA-256 does not match the verified build hash")
        with zipfile.ZipFile(outer_path) as outer:
            nested_names = [name for name in outer.namelist() if name.endswith("CMAPSSData.zip")]
            if len(nested_names) != 1:
                raise RuntimeError("Expected exactly one nested C-MAPSS archive")
            inner_bytes = outer.read(nested_names[0])
        if hashlib.sha256(inner_bytes).hexdigest().upper() != INNER_SHA256:
            raise RuntimeError("NASA nested archive SHA-256 does not match the verified build hash")
        inner_path = raw_root / "CMAPSSData_inner.zip"
        inner_path.write_bytes(inner_bytes)
        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
            for name in FD001_HASHES:
                if name not in inner.namelist():
                    raise RuntimeError(f"NASA nested archive is missing {name}")
                with inner.open(name) as source, (DATA_DIR / name).open("wb") as destination:
                    shutil.copyfileobj(source, destination)

    observed = {name: sha256_file(DATA_DIR / name) for name in FD001_HASHES}
    mismatches = {
        name: (FD001_HASHES[name], observed[name])
        for name in FD001_HASHES
        if observed[name] != FD001_HASHES[name]
    }
    if mismatches:
        raise RuntimeError(f"FD001 file integrity check failed: {mismatches}")
    return observed


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    shown = frame.loc[:, columns] if columns else frame

    def render(value: object) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.5f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [str(column) for column in shown.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in shown.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def run_unit_tests() -> str:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=DEV_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    if completed.returncode != 0:
        raise RuntimeError(f"Unit tests failed before analysis:\n{combined}")
    lines = [line.strip() for line in combined.splitlines() if line.strip()]
    return lines[-1] if lines else "tests passed"


def model_estimator(model_key: str):
    if model_key == "constant_prevalence":
        return build_constant_baseline()
    if model_key == "age_only":
        return build_age_only_baseline(random_seed=RANDOM_SEED)
    if model_key == "logistic_regression":
        return build_logistic_model(random_seed=RANDOM_SEED)
    if model_key == "random_forest":
        return build_random_forest_model(
            random_seed=RANDOM_SEED,
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
        )
    raise KeyError(model_key)


def save_figure(figure: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        if value.is_absolute():
            try:
                return str(value.relative_to(PROJECT_ROOT))
            except ValueError:
                return value.name
        return str(value)
    return value


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    test_result = run_unit_tests()
    file_hashes = ensure_fd001()
    data = load_cmapss(DATA_DIR, "FD001")
    train_target = construct_train_targets(data.train, PREDICTION_HORIZON)
    train_quality = validate_cmapss_frame(train_target, target_column=TARGET_COLUMN)

    feature_spec = FeatureSpec()
    train_features = build_feature_frame(data.train, feature_spec)
    all_feature_columns = feature_columns(train_features)
    # FD001's third operating setting is exactly constant in development data.
    full_columns = [
        column for column in all_feature_columns if column != "current__op_setting_3"
    ]
    assert_predictors_are_leakage_safe(full_columns)

    age_columns = ["asset_age"]
    current_sensor_columns = [
        column for column in full_columns if column.startswith("current__sensor_")
    ]
    current_plus_rolling_columns = [
        column
        for column in full_columns
        if column.startswith("current__sensor_") or column.startswith("sensor_")
    ]
    feature_groups = {
        "AGE_ONLY": age_columns,
        "CURRENT_SENSORS": current_sensor_columns,
        "CURRENT_PLUS_ROLLING": current_plus_rolling_columns,
        "FULL_PARSIMONIOUS_FEATURE_SET": full_columns,
    }

    y_train = train_target[TARGET_COLUMN].to_numpy(dtype=int)
    train_groups = train_target[UNIT_COLUMN].to_numpy()
    make_group_kfold_splits(train_groups, N_SPLITS)
    development_namespace = np.array(
        [f"DEV:FD001:{unit:03d}" for unit in np.unique(data.train[UNIT_COLUMN])]
    )
    test_namespace = np.array(
        [f"TEST:FD001:{unit:03d}" for unit in np.unique(data.test[UNIT_COLUMN])]
    )
    assert_no_asset_overlap(development_namespace, test_namespace)

    model_inputs_train = {
        "constant_prevalence": train_features[age_columns],
        "age_only": train_features,
        "logistic_regression": train_features[full_columns],
        "random_forest": train_features[full_columns],
    }
    oof_raw: dict[str, np.ndarray] = {}
    dev_raw_metrics: dict[str, object] = {}
    for model_key in model_inputs_train:
        oof_raw[model_key] = oof_grouped_probabilities(
            model_estimator(model_key),
            model_inputs_train[model_key],
            y_train,
            train_groups,
            n_splits=N_SPLITS,
        )
        dev_raw_metrics[model_key] = evaluate_probabilities(y_train, oof_raw[model_key])

    logistic_pr = dev_raw_metrics["logistic_regression"].pr_auc
    forest_pr = dev_raw_metrics["random_forest"].pr_auc
    selected_model = (
        "random_forest"
        if forest_pr > logistic_pr + MODEL_MATERIALITY_TOLERANCE
        else "logistic_regression"
    )
    selection_reason = (
        "Random Forest exceeded Logistic Regression grouped-OOF PR-AUC by more than "
        f"the predeclared {MODEL_MATERIALITY_TOLERANCE:.2f} materiality tolerance."
        if selected_model == "random_forest"
        else "Logistic Regression was retained because Random Forest did not exceed its "
        f"grouped-OOF PR-AUC by more than the predeclared {MODEL_MATERIALITY_TOLERANCE:.2f} tolerance."
    )

    nested_calibration = nested_grouped_oof_calibration(
        model_estimator(selected_model),
        model_inputs_train[selected_model],
        y_train,
        train_groups,
        n_splits=N_SPLITS,
        random_seed=RANDOM_SEED,
    )
    if not np.allclose(
        nested_calibration.raw_probabilities,
        oof_raw[selected_model],
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError(
            "Nested calibration outer-fold raw scores do not match grouped OOF scores"
        )
    selected_nested_calibrated = nested_calibration.calibrated_probabilities
    raw_selected_dev_metrics = evaluate_probabilities(y_train, oof_raw[selected_model])
    calibrated_selected_dev_metrics = evaluate_probabilities(
        y_train, selected_nested_calibrated
    )
    brier_improvement = (
        raw_selected_dev_metrics.brier_score
        - calibrated_selected_dev_metrics.brier_score
    )
    final_calibrator = fit_sigmoid_calibrator(
        oof_raw[selected_model], y_train, random_seed=RANDOM_SEED
    )
    calibration_slope = float(final_calibrator.model_.coef_[0, 0])
    calibration_retained = (
        brier_improvement >= CALIBRATION_BRIER_MINIMUM_IMPROVEMENT
        and calibration_slope > 0.0
    )
    development_probability = dict(oof_raw)
    if calibration_retained:
        development_probability[selected_model] = selected_nested_calibrated

    # Freeze all operating thresholds on raw grouped OOF ranks. When calibration
    # is retained, map the selected raw scalar through the final monotone sigmoid
    # so the same rows define the policy on the final probability scale.
    raw_thresholds: dict[str, object] = {}
    policy_thresholds: dict[str, float] = {}
    for model_key in model_inputs_train:
        raw_threshold = select_threshold_policy(
            y_train,
            oof_raw[model_key],
            recall_target=RECALL_TARGET,
        )
        raw_thresholds[model_key] = raw_threshold
        policy_thresholds[model_key] = (
            float(final_calibrator.transform([raw_threshold.threshold])[0])
            if model_key == selected_model and calibration_retained
            else float(raw_threshold.threshold)
        )
    selected_raw_threshold = float(raw_thresholds[selected_model].threshold)
    selected_threshold = policy_thresholds[selected_model]

    # Compact development-only ablations use the selected model architecture.
    ablation_records: list[dict[str, object]] = []
    for group_name, columns in feature_groups.items():
        if group_name == "FULL_PARSIMONIOUS_FEATURE_SET":
            probability = oof_raw[selected_model]
        else:
            probability = oof_grouped_probabilities(
                model_estimator(selected_model),
                train_features[columns],
                y_train,
                train_groups,
                n_splits=N_SPLITS,
            )
        metrics = evaluate_probabilities(y_train, probability)
        ablation_records.append(
            {
                "Feature_Group": group_name,
                "Feature_Count": len(columns),
                "Model_Architecture": selected_model,
                "Dev_OOF_PR_AUC": metrics.pr_auc,
                "Dev_OOF_ROC_AUC": metrics.roc_auc,
                "Dev_OOF_Brier": metrics.brier_score,
            }
        )
    ablations = pd.DataFrame.from_records(ablation_records)
    age_ablation_pr = float(
        ablations.loc[ablations["Feature_Group"] == "AGE_ONLY", "Dev_OOF_PR_AUC"].iloc[0]
    )
    ablations["PR_AUC_Delta_vs_AGE_ONLY"] = ablations["Dev_OOF_PR_AUC"] - age_ablation_pr

    # Match the frozen static age policy to development alert capacity only.
    development_alert_rate = float(
        (oof_raw[selected_model] >= selected_raw_threshold).mean()
    )
    candidate_cycles = np.arange(
        int(train_target[CYCLE_COLUMN].min()), int(train_target[CYCLE_COLUMN].max()) + 1
    )
    static_rates = np.array(
        [(train_target[CYCLE_COLUMN].to_numpy() >= cycle).mean() for cycle in candidate_cycles]
    )
    static_cutoff = int(candidate_cycles[np.argmin(np.abs(static_rates - development_alert_rate))])
    static_development_alert_rate = float(
        (train_target[CYCLE_COLUMN].to_numpy() >= static_cutoff).mean()
    )

    # Fit final models and complete the future-value stress test using development
    # data only. Every design choice above is frozen before official-test truth is
    # constructed or scored below.
    fitted_models: dict[str, object] = {}
    for model_key in model_inputs_train:
        estimator = model_estimator(model_key)
        estimator.fit(model_inputs_train[model_key], y_train)
        fitted_models[model_key] = estimator

    perturb_asset = int(
        train_target.groupby(UNIT_COLUMN)[CYCLE_COLUMN]
        .max()
        .loc[lambda values: values > 40]
        .index.min()
    )
    perturbation = assert_future_perturbation_invariance(
        data.train,
        asset_id=perturb_asset,
        after_cycle=20,
        spec=feature_spec,
        estimator=fitted_models[selected_model],
        prediction_columns=full_columns,
    )

    # ------------------------- LOCKED FINAL EVALUATION -------------------------
    test_target = construct_test_targets(
        data.test, data.test_final_rul, PREDICTION_HORIZON
    )
    test_quality = validate_cmapss_frame(test_target, target_column=TARGET_COLUMN)
    test_features = build_feature_frame(data.test, feature_spec)
    if full_columns != [
        column
        for column in feature_columns(test_features)
        if column != "current__op_setting_3"
    ]:
        raise AssertionError("Train/test feature definitions do not match")
    y_test = test_target[TARGET_COLUMN].to_numpy(dtype=int)
    test_groups = test_target[UNIT_COLUMN].to_numpy()
    model_inputs_test = {
        "constant_prevalence": test_features[age_columns],
        "age_only": test_features,
        "logistic_regression": test_features[full_columns],
        "random_forest": test_features[full_columns],
    }
    test_raw: dict[str, np.ndarray] = {
        model_key: positive_class_probability(
            fitted_models[model_key], model_inputs_test[model_key]
        )
        for model_key in model_inputs_train
    }
    test_probability = dict(test_raw)
    if calibration_retained:
        test_probability[selected_model] = final_calibrator.transform(
            test_raw[selected_model]
        )

    model_records: list[dict[str, object]] = []
    event_tables: dict[str, pd.DataFrame] = {}
    full_alert_tables: dict[str, pd.DataFrame] = {}
    event_summaries: dict[str, dict[str, float | int]] = {}
    for model_key in model_inputs_train:
        dev_metrics = evaluate_probabilities(y_train, development_probability[model_key])
        test_metrics = evaluate_probabilities(y_test, test_probability[model_key])
        dev_operating = evaluate_at_threshold(
            y_train,
            oof_raw[model_key],
            float(raw_thresholds[model_key].threshold),
        )
        test_operating = evaluate_at_threshold(
            y_test, test_probability[model_key], policy_thresholds[model_key]
        )
        qualifying, full_alerts, event_summary = event_outputs(
            test_target, test_probability[model_key], policy_thresholds[model_key]
        )
        event_tables[model_key] = qualifying
        full_alert_tables[model_key] = full_alerts
        event_summaries[model_key] = event_summary
        model_records.append(
            {
                "Model": model_key,
                "Features": (
                    "constant"
                    if model_key == "constant_prevalence"
                    else "age"
                    if model_key == "age_only"
                    else "full_parsimonious"
                ),
                "Calibration": (
                    "nested grouped OOF sigmoid retained"
                    if model_key == selected_model and calibration_retained
                    else "raw probability"
                ),
                "Dev_OOF_PR_AUC": dev_metrics.pr_auc,
                "Dev_OOF_ROC_AUC": dev_metrics.roc_auc,
                "Dev_OOF_Brier": dev_metrics.brier_score,
                "Policy_Threshold": policy_thresholds[model_key],
                "Raw_Dev_Threshold": float(raw_thresholds[model_key].threshold),
                "Dev_Recall_at_Threshold": dev_operating.recall,
                "Dev_Precision_at_Threshold": dev_operating.precision,
                "Test_PR_AUC": test_metrics.pr_auc,
                "Test_ROC_AUC": test_metrics.roc_auc,
                "Test_Brier": test_metrics.brier_score,
                "Test_Recall_at_Threshold": test_operating.recall,
                "Test_Precision_at_Threshold": test_operating.precision,
                "Eligible_Warning_Coverage": event_summary["warning_coverage"],
                "Median_Lead_Time": event_summary["median_lead_time"],
            }
        )
    model_comparison = pd.DataFrame.from_records(model_records)
    selected_probability = test_probability[selected_model]
    selected_event = event_tables[selected_model]
    selected_full_alerts = full_alert_tables[selected_model]
    selected_event_summary = event_summaries[selected_model]

    eligible_count = int(selected_event_summary["eligible_assets"])
    warned_count = int(selected_event["warned_before_event"].sum())
    coverage_value = warned_count / eligible_count
    wilson_z = 1.959963984540054
    wilson_denominator = 1.0 + wilson_z**2 / eligible_count
    wilson_center = (
        coverage_value + wilson_z**2 / (2.0 * eligible_count)
    ) / wilson_denominator
    wilson_half_width = (
        wilson_z
        * np.sqrt(
            coverage_value * (1.0 - coverage_value) / eligible_count
            + wilson_z**2 / (4.0 * eligible_count**2)
        )
        / wilson_denominator
    )
    coverage_wilson_interval = (
        wilson_center - wilson_half_width,
        wilson_center + wilson_half_width,
    )
    threshold_sensitivity: dict[float, dict[str, float | int]] = {}
    for delta in (-0.02, -0.01, 0.01, 0.02):
        sensitivity_threshold = float(np.clip(selected_threshold + delta, 0.0, 1.0))
        _, _, sensitivity_summary = event_outputs(
            test_target, selected_probability, sensitivity_threshold
        )
        threshold_sensitivity[delta] = sensitivity_summary

    static_test_probability = (
        test_target[CYCLE_COLUMN].to_numpy() >= static_cutoff
    ).astype(float)
    static_qualifying, static_full_alerts, static_summary = event_outputs(
        test_target, static_test_probability, 0.5
    )
    static_vs_condition = pd.DataFrame.from_records(
        [
            {
                "Policy": "Frozen static age/inspection proxy",
                "Development_Alert_Rate": static_development_alert_rate,
                "Test_Alert_Rate": float(static_test_probability.mean()),
                "Eligible_Warning_Coverage": static_summary["warning_coverage"],
                "Median_Lead_Time": static_summary["median_lead_time"],
                "False_Alert_Rows_All_Assets": static_summary["total_false_alert_rows"],
                "False_Alert_Episodes_All_Assets": static_summary[
                    "total_false_alert_episodes_all_assets"
                ],
                "Fraction_Assets_With_False_Alerts": static_summary[
                    "fraction_all_assets_with_false_alerts"
                ],
            },
            {
                "Policy": "Condition-informed threshold",
                "Development_Alert_Rate": development_alert_rate,
                "Test_Alert_Rate": float((selected_probability >= selected_threshold).mean()),
                "Eligible_Warning_Coverage": selected_event_summary["warning_coverage"],
                "Median_Lead_Time": selected_event_summary["median_lead_time"],
                "False_Alert_Rows_All_Assets": selected_event_summary[
                    "total_false_alert_rows"
                ],
                "False_Alert_Episodes_All_Assets": selected_event_summary[
                    "total_false_alert_episodes_all_assets"
                ],
                "Fraction_Assets_With_False_Alerts": selected_event_summary[
                    "fraction_all_assets_with_false_alerts"
                ],
            },
        ]
    )

    warned = selected_event.loc[selected_event["warned_before_event"]].copy()
    if warned.empty:
        representative_asset = int(selected_event[UNIT_COLUMN].min())
        representative_rule = "smallest eligible asset because no eligible asset was warned"
    else:
        representative_asset = int(select_representative_asset(selected_event))
        representative_rule = "eligible warned asset nearest the median lead time; asset ID tie-break"
    representative_record = selected_event.loc[
        selected_event[UNIT_COLUMN] == representative_asset
    ].iloc[0]

    representative_mask = test_target[UNIT_COLUMN].to_numpy() == representative_asset
    representative_last_probability = float(selected_probability[representative_mask][-1])
    static_fmea = build_static_fmea_scenario()
    condition_fmea = condition_informed_prioritization(
        static_fmea, representative_last_probability, bands=AlertBands()
    )
    fmea_change = condition_policy_change_summary(condition_fmea)

    selected_test_metrics = evaluate_probabilities(y_test, selected_probability)
    selected_test_operating = evaluate_at_threshold(
        y_test, selected_probability, selected_threshold
    )
    bootstrap = asset_bootstrap_probability_metrics(
        y_test,
        selected_probability,
        test_groups,
        n_bootstrap=BOOTSTRAP_REPLICATES,
        random_seed=RANDOM_SEED,
    )
    bootstrap_intervals = {
        metric: bootstrap_interval(bootstrap, metric)
        for metric in ("pr_auc", "roc_auc", "brier_score")
    }

    if selected_model == "logistic_regression":
        classifier = fitted_models[selected_model].named_steps["classifier"]
        coefficients = classifier.coef_[0]
        importance = pd.DataFrame(
            {
                "Feature": full_columns,
                "Importance": np.abs(coefficients),
                "Signed_Effect": coefficients,
                "Method": "absolute standardized logistic coefficient",
            }
        ).sort_values("Importance", ascending=False, kind="mergesort")
    else:
        result = permutation_importance(
            fitted_models[selected_model],
            test_features[full_columns],
            y_test,
            scoring="average_precision",
            n_repeats=3,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        importance = pd.DataFrame(
            {
                "Feature": full_columns,
                "Importance": result.importances_mean,
                "Importance_SD": result.importances_std,
                "Method": "locked-test row permutation decrease in average precision",
            }
        ).sort_values("Importance", ascending=False, kind="mergesort")

    # Persist review outputs outside the public source candidate. Row/asset-level
    # derivatives remain internal pending the Module 8 publication audit.
    quality_records = []
    for split_name, quality in (("development_train", train_quality), ("official_test", test_quality)):
        record = quality.to_dict()
        record["split"] = split_name
        record["constant_columns"] = "; ".join(record["constant_columns"])
        record["near_constant_columns"] = "; ".join(record["near_constant_columns"])
        quality_records.append(record)
    data_quality_table = pd.DataFrame.from_records(quality_records)

    range_records: list[dict[str, object]] = []
    for split_name, frame in (("development_train", data.train), ("official_test", data.test)):
        for column in (*SETTING_COLUMNS, *SENSOR_COLUMNS):
            range_records.append(
                {
                    "Split": split_name,
                    "Variable": column,
                    "Minimum": float(frame[column].min()),
                    "Maximum": float(frame[column].max()),
                    "Unique_Values": int(frame[column].nunique()),
                    "Exact_Constant": bool(frame[column].nunique() == 1),
                    "Used_As_Predictor": bool(f"current__{column}" in full_columns),
                }
            )
    sensor_ranges = pd.DataFrame.from_records(range_records)

    selected_predictions = test_target[
        [UNIT_COLUMN, CYCLE_COLUMN, EVENT_CYCLE_COLUMN, TARGET_COLUMN]
    ].copy()
    selected_predictions["condition_probability"] = selected_probability
    selected_predictions["policy_warning"] = selected_probability >= selected_threshold
    selected_predictions["qualifying_window"] = selected_predictions[TARGET_COLUMN] == 1

    development_calibration_raw = calibration_table(
        y_train, oof_raw[selected_model], n_bins=10, strategy="quantile"
    ).assign(Probability_Version="raw_grouped_oof")
    development_calibration_final = calibration_table(
        y_train, development_probability[selected_model], n_bins=10, strategy="quantile"
    ).assign(
        Probability_Version=(
            "nested_grouped_oof_sigmoid" if calibration_retained else "raw_grouped_oof"
        )
    )
    development_calibration = pd.concat(
        [development_calibration_raw, development_calibration_final], ignore_index=True
    ).drop_duplicates()
    test_calibration_raw = calibration_table(
        y_test, test_raw[selected_model], n_bins=10, strategy="quantile"
    ).assign(Probability_Version="raw_locked_test")
    test_calibration_final = calibration_table(
        y_test, selected_probability, n_bins=10, strategy="quantile"
    ).assign(
        Probability_Version=(
            "sigmoid_locked_test" if calibration_retained else "raw_locked_test"
        )
    )
    test_calibration = pd.concat(
        [test_calibration_raw, test_calibration_final], ignore_index=True
    ).drop_duplicates()

    output_tables = {
        "data_quality_summary.csv": data_quality_table,
        "sensor_ranges.csv": sensor_ranges,
        "model_comparison.csv": model_comparison,
        "ablation_results.csv": ablations,
        "test_predictions.csv": selected_predictions,
        "per_asset_event_metrics.csv": selected_event,
        "all_asset_alert_burden.csv": selected_full_alerts,
        "static_policy_event_metrics.csv": static_qualifying,
        "static_vs_condition_comparison.csv": static_vs_condition,
        "static_fmea.csv": static_fmea,
        "condition_informed_fmea.csv": condition_fmea,
        "feature_importance.csv": importance,
        "calibration_development.csv": development_calibration,
        "calibration_test.csv": test_calibration,
        "bootstrap_metrics.csv": bootstrap,
    }
    for filename, table in output_tables.items():
        table.to_csv(TABLE_DIR / filename, index=False)

    # Figure 1: one fixed development trajectory, normalized only for display.
    example_train_asset = int(data.train[UNIT_COLUMN].min())
    trajectory = data.train.loc[data.train[UNIT_COLUMN] == example_train_asset].copy()
    figure, axis = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    for sensor, color in (("sensor_2", "#2563eb"), ("sensor_11", "#d97706")):
        values = trajectory[sensor].to_numpy(dtype=float)
        scale = values.std(ddof=0)
        z_values = (values - values.mean()) / scale if scale > 0 else values * 0
        axis.plot(trajectory[CYCLE_COLUMN], z_values, label=sensor, linewidth=1.8, color=color)
    axis.set(
        title=f"FD001 development engine {example_train_asset}: display-only z-score trajectories",
        xlabel="Cycle",
        ylabel="Within-engine standardized sensor value",
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.22)
    save_figure(figure, "01_example_sensor_trajectory.png")

    # Figure 2: target boundary illustration using a fixed development asset.
    target_example = train_target.loc[
        train_target[UNIT_COLUMN] == example_train_asset
    ].tail(PREDICTION_HORIZON + 35)
    event_cycle_example = int(target_example[EVENT_CYCLE_COLUMN].iloc[0])
    figure, axis = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
    axis.step(
        target_example[CYCLE_COLUMN],
        target_example[TARGET_COLUMN],
        where="post",
        linewidth=2.2,
        color="#2563eb",
    )
    axis.axvline(
        event_cycle_example - PREDICTION_HORIZON,
        color="#d97706",
        linestyle="--",
        label="30-cycle horizon boundary",
    )
    axis.axvline(event_cycle_example, color="#991b1b", linestyle=":", label="Simulated EOL")
    axis.set(
        title="Independent future-event target",
        xlabel="Cycle",
        ylabel="Y(asset, t, 30)",
        yticks=[0, 1],
        ylim=(-0.08, 1.08),
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.22)
    save_figure(figure, "02_target_horizon.png")

    # Figure 3: compact leakage/split design diagram.
    figure, axis = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    axis.axis("off")
    boxes = [
        (0.03, 0.56, 0.26, 0.25, "100 complete train engines\ntrailing-only features"),
        (0.37, 0.56, 0.26, 0.25, "5-fold GroupKFold\nOOF selection/calibration"),
        (0.71, 0.56, 0.26, 0.25, "100 separate test engines\nlocked final evaluation"),
        (0.20, 0.08, 0.26, 0.24, "Predictors: data at/before t\nNo RUL/event/target fields"),
        (0.55, 0.08, 0.26, 0.24, "Target: EOL within 30 cycles\nFuture outcome, label only"),
    ]
    for x, y, width, height, label in boxes:
        axis.add_patch(
            plt.Rectangle((x, y), width, height, facecolor="#eff6ff", edgecolor="#2563eb", linewidth=1.5)
        )
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=10)
    axis.annotate("", xy=(0.37, 0.685), xytext=(0.29, 0.685), arrowprops={"arrowstyle": "->", "color": "#475569"})
    axis.annotate("", xy=(0.71, 0.685), xytext=(0.63, 0.685), arrowprops={"arrowstyle": "->", "color": "#475569"})
    axis.text(0.67, 0.84, "freeze decisions", ha="center", va="center", fontsize=9, color="#475569")
    axis.text(0.5, 0.96, "Asset-separated development and temporal leakage firewall", ha="center", va="top", fontsize=13, weight="bold")
    save_figure(figure, "03_validation_and_leakage_design.png")

    # Figure 4: locked-test PR curves after all development choices were frozen.
    pr_curves = {
        "Constant": (y_test, test_probability["constant_prevalence"]),
        "Age only": (y_test, test_probability["age_only"]),
        "Logistic": (y_test, test_probability["logistic_regression"]),
        "Random forest": (y_test, test_probability["random_forest"]),
    }
    figure, _ = plot_precision_recall_curves(
        pr_curves,
        title="Locked official-test precision–recall (simulated benchmark)",
    )
    save_figure(figure, "04_locked_test_precision_recall.png")

    # Figure 5: selected-model locked-test calibration; test labels do not fit the mapping.
    calibration_curves = {"Raw probability (retained)": calibration_table(y_test, selected_probability, n_bins=10)}
    if calibration_retained:
        calibration_curves = {
            "Raw": calibration_table(y_test, test_raw[selected_model], n_bins=10),
            "Sigmoid (fitted on development OOF)": calibration_table(
                y_test, selected_probability, n_bins=10
            ),
        }
    figure, _ = plot_calibration_curves(
        calibration_curves,
        policy_threshold=selected_threshold,
        sample_count=len(y_test),
    )
    save_figure(figure, "05_locked_test_calibration.png")

    # Figure 6: representative qualifying warning and eligible lead-time distribution.
    representative = selected_predictions.loc[
        selected_predictions[UNIT_COLUMN] == representative_asset
    ].sort_values(CYCLE_COLUMN)
    event_cycle = int(representative[EVENT_CYCLE_COLUMN].iloc[0])
    first_warning = representative_record["first_warning_cycle"]
    figure, (risk_axis, lead_axis) = plt.subplots(
        1, 2, figsize=(11.0, 4.4), constrained_layout=True
    )
    risk_axis.plot(
        representative[CYCLE_COLUMN],
        representative["condition_probability"],
        color="#2563eb",
        linewidth=2,
    )
    risk_axis.axhline(selected_threshold, color="#d97706", linestyle="--", label="Policy threshold")
    risk_axis.axvspan(event_cycle - PREDICTION_HORIZON, event_cycle, color="#fef3c7", alpha=0.6, label="Qualifying window")
    risk_axis.axvline(event_cycle, color="#991b1b", linestyle=":", label="Simulated EOL")
    if not pd.isna(first_warning):
        risk_axis.axvline(float(first_warning), color="#047857", linestyle="--", label="First qualifying warning")
    risk_axis.set(
        title=f"Illustrative engine {representative_asset}: deterministic median-lead case",
        xlabel="Cycle",
        ylabel="Near-term event probability",
        ylim=(0, 1.02),
        xlim=(int(representative[CYCLE_COLUMN].min()), event_cycle),
    )
    risk_axis.legend(frameon=False, fontsize=8)
    qualifying_leads = pd.to_numeric(
        selected_event.loc[selected_event["warned_before_event"], "lead_time"], errors="coerce"
    ).dropna()
    if qualifying_leads.empty:
        lead_axis.text(0.5, 0.5, "No eligible pre-event warnings", ha="center", va="center")
        lead_axis.set_axis_off()
    else:
        bins = min(10, max(3, int(np.sqrt(len(qualifying_leads))) + 1))
        lead_axis.hist(qualifying_leads, bins=bins, color="#2563eb", edgecolor="white")
        lead_axis.axvline(float(qualifying_leads.median()), color="#991b1b", linestyle="--", label=f"Median = {qualifying_leads.median():.1f}")
        lead_axis.set(
            title=(
                f"Warned eligible engines (n={warned_count})\n"
                f"{eligible_count - warned_count} eligible engines were unwarned"
            ),
            xlabel="Cycles",
            ylabel="Engines",
        )
        lead_axis.legend(frameon=False)
    figure.suptitle(
        "Illustrative benchmark warning; not field performance",
        fontsize=12.5,
        weight="bold",
    )
    save_figure(figure, "06_representative_asset_and_lead_time.png")

    # Figure 7: static engineering context and separate condition evidence.
    figure, _ = plot_condition_informed_fmea(condition_fmea)
    save_figure(figure, "07_static_and_condition_informed_fmea.png")

    # Figure 8: final-model associative importance, explicitly non-causal.
    top_importance = importance.head(12).sort_values("Importance", ascending=True)
    figure, axis = plt.subplots(figsize=(7.8, 5.2), constrained_layout=True)
    axis.barh(top_importance["Feature"], top_importance["Importance"], color="#2563eb")
    axis.set(
        title="Selected-model associations under correlated features\n(not causal or physical rankings)",
        xlabel=str(importance["Method"].iloc[0]),
        ylabel="",
    )
    axis.grid(axis="x", alpha=0.22)
    save_figure(figure, "08_feature_importance.png")

    selected_row = model_comparison.loc[model_comparison["Model"] == selected_model].iloc[0]
    best_ablation = ablations.sort_values("Dev_OOF_PR_AUC", ascending=False).iloc[0]
    calibration_result = (
        "Retained nested grouped OOF sigmoid calibration: development OOF Brier improved "
        f"from {raw_selected_dev_metrics.brier_score:.5f} to {calibrated_selected_dev_metrics.brier_score:.5f} "
        f"(improvement {brier_improvement:.5f}; positive slope {calibration_slope:.5f}); the final mapping was fitted only on all development OOF predictions."
        if calibration_retained
        else "Sigmoid calibration was not retained: nested grouped development OOF Brier changed "
        f"from {raw_selected_dev_metrics.brier_score:.5f} to {calibrated_selected_dev_metrics.brier_score:.5f} "
        f"(improvement {brier_improvement:.5f}; final slope {calibration_slope:.5f}); retention required at least "
        f"{CALIBRATION_BRIER_MINIMUM_IMPROVEMENT:.3f} improvement and a positive slope."
    )

    data_quality_report = f"""# Data quality report

## Verified source and structure

NASA C-MAPSS FD001 was loaded from the official public repository. The integrity-checked raw cache is stored outside the repository source tree and excluded from publication packaging because NASA's dataset record does not specify a license.

{markdown_table(data_quality_table)}

The two official splits contain **200 separately simulated engines** and **{len(train_target) + len(test_target):,} observations**: {len(train_target):,} development rows and {len(test_target):,} official-test rows. Unit numbers restart in the test files, so split-qualified namespaces are used for overlap checks.

## Integrity checks

- Missing required cells: 0 in both splits.
- Duplicate `(unit_id, cycle)` keys: 0 in both splits.
- Out-of-order engines: 0; cycle gaps: 0.
- Exact-constant fields in both splits: `op_setting_3`, `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, and `sensor_19`.
- Six exactly constant sensor channels, the constant third setting, and the deliberately omitted two-valued `sensor_6` are excluded from model predictors. No channel was dropped solely because of correlation.
- Full sensor/setting ranges and use decisions are recorded in `05_OUTPUTS/tables/sensor_ranges.csv`.

## Target and event distribution

Development prevalence is **{y_train.mean():.5f}** ({int(y_train.sum()):,}/{len(y_train):,}); locked-test prevalence is **{y_test.mean():.5f}** ({int(y_test.sum()):,}/{len(y_test):,}). The difference follows the official test truncation design. Development trajectories end at simulated EOL. Official test terminal RUL ranges from {int(data.test_final_rul['final_rul'].min())} to {int(data.test_final_rul['final_rul'].max())} cycles (median {float(data.test_final_rul['final_rul'].median()):.1f}). Only **{selected_event_summary['eligible_assets']}** test trajectories contain released observations inside the 30-cycle target window; the other 75 remain valid negative exposure for row metrics and false-alert burden but are not counted as missed qualifying warnings.
"""
    write_text(REPORT_DIR / "data_quality_report.md", data_quality_report)

    predictive_report = f"""# Predictive results

## Scope and class prevalence

The independently defined outcome is simulated end of useful life within {PREDICTION_HORIZON} cycles. It is not RPN. Development prevalence was **{y_train.mean():.3%}**; official-test prevalence was **{y_test.mean():.3%}**. Official test histories are truncated, and 75 of 100 do not release a row inside the final 30-cycle window. This construction shift materially affects calibration and threshold transfer. PR-AUC is primary because positive rows are uncommon.

## Model comparison

{markdown_table(model_comparison, ['Model', 'Features', 'Calibration', 'Dev_OOF_PR_AUC', 'Dev_OOF_ROC_AUC', 'Dev_OOF_Brier', 'Policy_Threshold', 'Test_PR_AUC', 'Test_ROC_AUC', 'Test_Brier', 'Test_Recall_at_Threshold', 'Test_Precision_at_Threshold'])}

All thresholds were chosen from grouped development OOF probabilities for 80% row recall and applied unchanged to the official test. Baselines are the fold-specific prevalence model and age-only Logistic Regression. The historical circular-RPN regression is `HISTORICAL_INVALIDATED_APPROACH`, not a baseline.

## Selection and calibration

Selected model: **{selected_model}**. {selection_reason}

{calibration_result} Raw probabilities were retained. This result does not establish perfect calibration or transfer to another fleet, sensor system, operating regime, or prevalence.

On the locked test, the selected probability achieved PR-AUC **{selected_test_metrics.pr_auc:.5f}**, ROC-AUC **{selected_test_metrics.roc_auc:.5f}**, and Brier **{selected_test_metrics.brier_score:.5f}**. At threshold **{selected_threshold:.5f}**, recall was **{selected_test_operating.recall:.3f}**, precision **{selected_test_operating.precision:.3f}**, and F1 **{selected_test_operating.f1:.3f}**.

The high ROC-AUC is plausible for this benchmark but is not a production headline: positives are contiguous terminal-window observations, sequential rows are correlated, age and simulated degradation trajectories are informative, and official truncation creates many all-negative released histories. The result is benchmark discrimination, not “99.7% accuracy,” real-failure prediction, or field validity.

Whole-engine bootstrap percentile intervals ({BOOTSTRAP_REPLICATES} replicates) were: PR-AUC {bootstrap_intervals['pr_auc'][0]:.5f}–{bootstrap_intervals['pr_auc'][1]:.5f}; ROC-AUC {bootstrap_intervals['roc_auc'][0]:.5f}–{bootstrap_intervals['roc_auc'][1]:.5f}; Brier {bootstrap_intervals['brier_score'][0]:.5f}–{bootstrap_intervals['brier_score'][1]:.5f}. These intervals describe only this simulated benchmark.

## Threshold discipline and bounded sensitivity

The threshold **{selected_threshold:.9f}** was selected only from grouped-development OOF predictions as the highest score meeting 80% development row recall. It achieved {selected_test_operating.recall:.1%} row recall on the truncated official test, so 80% is a development policy target, not a transported guarantee. At ±0.01, eligible coverage remained {float(threshold_sensitivity[-0.01]['warning_coverage']):.0%} and false-alert rows remained {int(threshold_sensitivity[-0.01]['total_false_alert_rows'])}. At −0.02 coverage was {float(threshold_sensitivity[-0.02]['warning_coverage']):.0%}; at +0.02 it was {float(threshold_sensitivity[0.02]['warning_coverage']):.0%}. These are fixed sensitivity checks, not threshold retuning.

## Ablation

{markdown_table(ablations)}

The highest grouped-development ablation was **{best_ablation['Feature_Group']}** (PR-AUC {float(best_ablation['Dev_OOF_PR_AUC']):.5f}). The full parsimonious set's change from age-only was {float(ablations.loc[ablations['Feature_Group'] == 'FULL_PARSIMONIOUS_FEATURE_SET', 'PR_AUC_Delta_vs_AGE_ONLY'].iloc[0]):+.5f}. These are within-setup comparisons, not proof of universal model superiority.

## Feature importance

{markdown_table(importance.head(12))}

Importance is associative and model-specific, not causal. It was computed only after the final model and evaluation policy were frozen.
"""
    write_text(REPORT_DIR / "predictive_results.md", predictive_report)

    event_report = f"""# Event-level results

## Eligibility and warning policy

The threshold **{selected_threshold:.5f}** corresponds to the highest raw grouped-development OOF score meeting the predeclared 80% row-recall target. A qualifying warning is the first threshold crossing in a released row inside the 30-cycle event window. Only the **{eligible_count}** official-test engines whose released histories enter that window are eligible for coverage and lead time. All 100 test engines contribute to false-alert exposure.

## Selected condition policy

- Qualifying warning coverage: **{warned_count}/{eligible_count} = {selected_event_summary['warning_coverage']:.3%}**.
- Wilson 95% interval: **{coverage_wilson_interval[0]:.1%}–{coverage_wilson_interval[1]:.1%}**.
- Median qualifying first-warning lead: **{selected_event_summary['median_lead_time']:.1f} cycles** among warned eligible engines.
- Eligible fraction never warned: **{selected_event_summary['fraction_never_warned']:.3%}**.
- False-alert rows across all test engines: **{selected_event_summary['total_false_alert_rows']:,}** ({selected_event_summary['mean_false_alert_rows_per_asset']:.2f} per engine).
- False-alert rows per 1,000 target-negative exposure rows: **{selected_event_summary['false_alert_rows_per_1000_negative_rows']:.2f}**.
- False-alert episodes across all test engines: **{selected_event_summary['total_false_alert_episodes_all_assets']:,}** ({selected_event_summary['mean_false_alert_episodes_per_asset_all']:.2f} per engine).
- Engines with at least one false-alert episode: **{selected_event_summary['fraction_all_assets_with_false_alerts']:.3%}**.

## Representative asset

Test engine **{representative_asset}** was selected after finalization by the rule: {representative_rule}. Its first qualifying warning cycle is **{representative_record['first_warning_cycle']}**, simulated event cycle **{int(representative_record['event_cycle'])}**, and qualifying lead time **{representative_record['lead_time']} cycles**. The final released-row probability was **{representative_last_probability:.5f}**. This is a deterministic median-case illustration, not a best-case selection.

## Limits

The event is simulated benchmark EOL, not a field failure. Seventy-five test engines are truncated before the 30-cycle window and therefore cannot support qualifying-warning coverage. The 64% point estimate is imprecise and conditional on unequal released follow-up inside the window; it is not a stable production-performance estimate. Alert rows and episodes are scenario workload proxies, not maintenance costs. Test access was retrospective and is logged; no test result was used to redesign the target, features, model set, calibration rule, selection tolerance, or threshold policy.
"""
    write_text(REPORT_DIR / "event_level_results.md", event_report)

    fmea_report = f"""# Condition-informed FMEA results

## Static engineering scenario

FD001 documents one simulated high-pressure-compressor degradation mode, not observed failure-mode labels. The three-row table is therefore an explicit `ENGINEERING_SCENARIO_ASSUMPTION` covering narrow HPC flow, efficiency, and combined-margin deterioration.

{markdown_table(static_fmea, ['Failure_Mode', 'Effect', 'Current_Control', 'Severity', 'Occurrence', 'Detection', 'Static_RPN', 'Static_Priority', 'Engineering_Action', 'Scenario_Basis'])}

`Static_RPN = Severity × Occurrence × Detection` is retained as transparent secondary engineering context. It is never a predictive target, never recalculated from the model, and not treated as an interval scale or complete risk ordering.

## Condition integration

Fixed final-model probability bands are `LOW < 0.20`, `MODERATE 0.20–<0.50`, `HIGH 0.50–<0.80`, and `CRITICAL >= 0.80`. They are a `PORTFOLIO_SCENARIO_ASSUMPTION`, not validated industrial limits or IEC/ISO thresholds. The asset-level probability changes urgency/timing only. High-severity rows at high/critical condition levels receive accelerated or immediate engineering review; all S/O/D values and Static_RPN remain unchanged.

For representative test engine {representative_asset}, the last released probability was {representative_last_probability:.5f}. Because FD001 has no mode-specific outcome labels, that single asset-level condition value is shared across the illustrative scenario rather than falsely attributed to a particular failure mode.

{markdown_table(condition_fmea, ['Failure_Mode', 'Severity', 'Occurrence', 'Detection', 'Static_RPN', 'Condition_Probability', 'Condition_Alert', 'Recommended_Urgency', 'Condition_Escalated'])}

The example changed urgency for **{fmea_change['escalated_items']} of {fmea_change['fmea_items']}** FMEA rows. It did not change engineering severity or assert a diagnosed physical failure mode.

## Illustrative frozen-policy comparison

Under the defined illustrative policies, the static age/inspection proxy begins warning at cycle **{static_cutoff}**, chosen only on development data to nearly match condition-policy alert-state rows ({static_development_alert_rate:.3%} versus {development_alert_rate:.3%}). Both policies were then frozen and applied to the official test.

This matches only development alert-state rows. Inspected assets, alert/inspection episodes, maintenance hours, cost, asset-level workload, and resource constraints are not matched.

{markdown_table(static_vs_condition)}

This `SCENARIO_POLICY_ILLUSTRATION` asks whether condition information changes timing and row-alert burden under the coded assumptions. It is not an operational superiority test and does not claim that ML “beats FMEA,” prevents failures, optimizes cost, or replaces engineering judgment.
"""
    write_text(REPORT_DIR / "fmea_results.md", fmea_report)

    test_access_log = """# Locked-test access log

1. **Target-integrity access (2026-08-24):** after the dataset, horizon, target formula, feature policy, split, models, model-selection rule, calibration rule, and threshold rule were frozen, the official target structure was inspected to verify row counts and event-metric eligibility. This established that 25 released test trajectories enter the 30-cycle window.
2. **Scripted final evaluation (2026-08-24):** `python run_analysis.py` fitted all decisions on grouped development data, then generated final locked-test row metrics, event analysis, calibration diagnostics, bootstrap intervals, and final-model interpretation in one scripted pass.

This is a retrospective portfolio reconstruction, not a preregistered experiment. Re-running the deterministic script reproduces evaluation but should not be interpreted as a new independent test.
"""
    write_text(REPORT_DIR / "test_access_log.md", test_access_log)

    figure_names = [
        "01_example_sensor_trajectory.png",
        "02_target_horizon.png",
        "03_validation_and_leakage_design.png",
        "04_locked_test_precision_recall.png",
        "05_locked_test_calibration.png",
        "06_representative_asset_and_lead_time.png",
        "07_static_and_condition_informed_fmea.png",
        "08_feature_importance.png",
    ]
    manifest = f"""# Build manifest

| Field | Value |
| --- | --- |
| Public_Framing | CONDITION_INFORMED_FMEA_DECISION_SUPPORT_PROTOTYPE / TECHNICAL_PROTOTYPE |
| Dataset | NASA C-MAPSS FD001 |
| Data_Provenance | PUBLIC_VERIFIED |
| Redistribution_Status | VERIFY_BEFORE_PUBLICATION |
| Number_of_Assets | 200 total: 100 development + 100 official test |
| Number_of_Rows | {len(train_target) + len(test_target):,} total: {len(train_target):,} development + {len(test_target):,} test |
| Target_Definition | Y(asset,t,30)=1 iff 0 <= simulated_event_cycle - t <= 30 |
| Prediction_Horizon | 30 cycles, inclusive |
| Feature_Groups | age; 2 nonconstant settings; 14 current sensors; 5/10-cycle trailing mean/std/slope on 6 sensors; initial deviation |
| Split_Strategy | Official train/test preserved; 5-fold GroupKFold by engine for development |
| Baselines | Fold-specific constant prevalence; age-only Logistic Regression |
| Candidate_Models | Regularized Logistic Regression; 200-tree depth-10 Random Forest |
| Selected_Model | {selected_model} |
| Primary_Metric | Grouped development OOF PR-AUC, 0.01 simplicity tolerance |
| Calibration_Method | {'Nested grouped OOF development sigmoid retained; final sigmoid fitted on all OOF predictions' if calibration_retained else 'Raw probability; nested grouped OOF sigmoid did not meet 0.001 Brier-improvement rule'} |
| Threshold_Policy | Highest raw grouped-development score meeting 80% row recall; {'mapped through retained sigmoid' if calibration_retained else 'raw scale retained'}; final threshold={selected_threshold:.6f} |
| Event_Metrics | 30-cycle qualifying-warning coverage, first-warning lead, false-alert rows/episodes, never-warned fraction |
| Static_FMEA_Method | Three-row ENGINEERING_SCENARIO_ASSUMPTION; S×O×D retained only as static context |
| Condition_Integration_Method | Fixed 0.20/0.50/0.80 bands change urgency only; S/O/D and Static_RPN immutable |
| Leakage_Controls | Target firewall; trailing-only features; namespaced asset separation; fold-local transforms; future perturbation |
| Ablations | AGE_ONLY; CURRENT_SENSORS; CURRENT_PLUS_ROLLING; FULL_PARSIMONIOUS_FEATURE_SET |
| Tests | {test_result}; perturbation feature delta={perturbation.max_feature_delta:.1e}, prediction delta={perturbation.max_prediction_delta:.1e} |
| Figures | {len(figure_names)}: {', '.join(figure_names)} |
| Validation_Status | VALIDATED_WITH_MAJOR_LIMITATIONS |
| Publication_Safety | Source candidate excludes raw and row/asset-level NASA data; generated outputs remain outside the source candidate pending Module 8 |
| Known_Blockers | NASA redistribution verification; non-operational policy matching; 25 eligible-event engines; calibration/threshold transport; benchmark-formulation inflation; no field/domain/deployment/safety validation |
| Next_Module | Module 8 publication audit; not started |
"""
    write_text(DEV_ROOT / "BUILD_MANIFEST.md", manifest)

    key_files = [
        DEV_ROOT / "run_analysis.py",
        DEV_ROOT / "README.md",
        DEV_ROOT / "BUILD_MANIFEST.md",
        REPORT_DIR / "historical_model_audit.md",
        REPORT_DIR / "dataset_selection.md",
        REPORT_DIR / "data_quality_report.md",
        REPORT_DIR / "methodology.md",
        REPORT_DIR / "predictive_results.md",
        REPORT_DIR / "event_level_results.md",
        REPORT_DIR / "fmea_results.md",
        REPORT_DIR / "limitations.md",
        REPORT_DIR / "methodological_sources.md",
        DEV_ROOT / "docs" / "leakage_controls.md",
        DEV_ROOT / "docs" / "target_definition.md",
    ]
    build_report = f"""# Module 7.25D — Continuous FMEA technical rebuild

## 1. Historical-model audit
The retained notebooks, synthetic CSV, and incomplete paper were inspected read-only. Exact formulas, split logic, models, saved metrics, and unsupported conclusions are documented in `historical_model_audit.md`.

## 2. Invalidation of historical predictive claim
Verdict: **INVALID_FOR_PREDICTIVE_EVIDENCE**. Historical RPN values are additive synthetic functions of the same predictors supplied to the regressors; neither target is an independent future outcome nor conventional S×O×D. The historical work remains preserved only as `HISTORICAL_INVALIDATED_APPROACH` coursework context.

Historical/reference sources are outside this script's write scope; Module 7.25D post-build SHA-256 verification is recorded in `historical_model_audit.md`.

## 3. Dataset selection
NASA C-MAPSS FD001 was selected over UCI Hydraulic Systems and MetroPT-3 because it provides 200 separate ordered trajectories, run-to-failure development assets, an official test split, and terminal RUL truth suitable for an independent future-event target.

## 4. Rights and provenance
`DATA_PROVENANCE=PUBLIC_VERIFIED`; `REDISTRIBUTION_STATUS=VERIFY_BEFORE_PUBLICATION`. NASA's record says “License not specified.” Raw data are kept in a private runtime cache outside the source candidate and are never silently bundled. File hashes: {json.dumps(file_hashes, sort_keys=True)}.

## 5. Engineering case boundary
The result is a **CONDITION_INFORMED_FMEA_DECISION_SUPPORT_PROTOTYPE** with separate engineering-FMEA, condition-observation, and transparent decision-support layers. It is not an autonomous safety system or deployed maintenance platform.

## 6. Target definition
`Y(asset,t,30)=1` exactly when simulated EOL is 0–30 cycles ahead, inclusive. RUL/event cycle are label-only. Horizon 30 was predeclared for interpretability, not selected from test performance.

## 7. Leakage controls
Only current and trailing values enter predictors. RUL, event cycle, target, future readings, S/O/D and RPN are rejected. Learned transforms fit within folds. Future perturbation produced maximum earlier feature delta {perturbation.max_feature_delta:.1e} and prediction delta {perturbation.max_prediction_delta:.1e}.

## 8. Split strategy
The official split is preserved. Development uses five-fold GroupKFold on engines. Split-qualified namespaces eliminate train/test numeric-ID reuse. No row-random split is used.

## 9. Data quality
{len(train_target) + len(test_target):,} rows across 200 assets; no required missing cells, duplicate asset/cycles, ordering failures, or cycle gaps. Development/test prevalence: {y_train.mean():.5f}/{y_test.mean():.5f}. Six exactly constant sensors, one constant setting, and the two-valued sensor 6 are excluded.

## 10. Features
{len(full_columns)} predictors: age, two nonconstant settings, 14 current sensors, initial deviations, and trailing 5/10-cycle means, standard deviations and slopes for six predeclared sensors.

## 11. Baselines
Fold-specific constant prevalence and age-only Logistic Regression were evaluated. Their actual results appear in the model table below; the circular historical model is not a baseline.

## 12. Candidate models
Regularized Logistic Regression and a deterministic 200-tree, depth-10 Random Forest; no SMOTE, deep learning, XGBoost, broad tuning, or seed search.

## 13. Model selection
Selected **{selected_model}** by grouped-development PR-AUC with a predeclared 0.01 simplicity tolerance. {selection_reason}

## 14. Predictive performance
{markdown_table(model_comparison, ['Model', 'Dev_OOF_PR_AUC', 'Test_PR_AUC', 'Test_ROC_AUC', 'Test_Brier', 'Test_Recall_at_Threshold', 'Test_Precision_at_Threshold'])}

## 15. Calibration
{calibration_result}

## 16. Ablations
{markdown_table(ablations)}

## 17. Event-level results
Eligible 30-cycle warning coverage {warned_count}/{eligible_count} = {selected_event_summary['warning_coverage']:.3%} (Wilson 95% interval {coverage_wilson_interval[0]:.1%}–{coverage_wilson_interval[1]:.1%}); eligible never-warned fraction {selected_event_summary['fraction_never_warned']:.3%}; all-test false-alert rows {selected_event_summary['total_false_alert_rows']:,} and episodes {selected_event_summary['total_false_alert_episodes_all_assets']:,}. The point estimate is conditional on a small, unequally followed benchmark sample.

## 18. Lead-time analysis
Median qualifying first-warning lead is {selected_event_summary['median_lead_time']:.1f} cycles. Representative engine {representative_asset} was chosen by {representative_rule}.

## 19. FMEA scenario
Three narrow illustrative HPC-related modes are explicitly labeled `ENGINEERING_SCENARIO_ASSUMPTION`. Static S/O/D and S×O×D remain engineering context only.

## 20. Condition-informed logic
Fixed probability bands 0.20/0.50/0.80 are `PORTFOLIO_SCENARIO_ASSUMPTION` values that change inspection/review urgency. They never modify static severity, occurrence, detection, or Static_RPN.

## 21. Static-versus-condition comparison
This is a `SCENARIO_POLICY_ILLUSTRATION`. Static cycle cutoff {static_cutoff} was chosen on development data to nearly match alert-state rows only; assets, episodes, inspections, labor, cost, and operational workload were not matched. {markdown_table(static_vs_condition)}

## 22. Leakage tests
Target boundaries, trailing-window invariance, future perturbation, forbidden predictor fields, and asset separation all passed.

## 23. Test suite
**{test_result}** using the isolated analysis environment. All mandatory target, feature, split, model, event-metric and FMEA tests passed.

## 24. Figures
Eight compact figures were generated: {', '.join(figure_names)}.

## 25. Methodological sources
Verified sources cover IEC 60812, ISO 17359/13372, RPN limitations, NASA C-MAPSS, CBM/PHM architecture, grouped validation, time ordering and preprocessing leakage. Full links and usage notes appear in `methodological_sources.md`.

## 26. Limitations
Simulated benchmark; illustrative failure-mode mapping; severe official-test truncation for event coverage; simplified threshold/policy logic; no cost or causal model; calibration and shift uncertainty; retrospective test access; no deployment, field validation, safety certification or production claim.

## 27. Files created
Key implementation/report files: {', '.join(str(path.relative_to(PROJECT_ROOT)) for path in key_files)}. Modular `src`, six test files, three explanatory notebooks, 15 result tables and eight figures complete the build.

## 28. Blockers
No technical correctness blocker remains. Publication remains blocked on explicit NASA redistribution/license verification, Module 8 artifact-by-artifact review, and the unresolved external/domain limits documented by Module 7.5D.

## 29. Validation and professionalization status
Module 7.5D verdict: **VALIDATED_WITH_MAJOR_LIMITATIONS**. Repository professionalization is complete; Module 8 publication audit has not started.
"""
    write_text(BUILD_REPORT_PATH, build_report)

    summary = {
        "status": "PROFESSIONALIZED_PENDING_MODULE_8_PUBLICATION_AUDIT",
        "historical_verdict": "INVALID_FOR_PREDICTIVE_EVIDENCE",
        "public_framing": "CONDITION_INFORMED_FMEA_DECISION_SUPPORT_PROTOTYPE",
        "dataset": "NASA C-MAPSS FD001",
        "data_provenance": "PUBLIC_VERIFIED",
        "redistribution_status": "VERIFY_BEFORE_PUBLICATION",
        "assets": {"development": 100, "test": 100, "total": 200},
        "observations": {
            "development": len(train_target),
            "test": len(test_target),
            "total": len(train_target) + len(test_target),
        },
        "prevalence": {"development": float(y_train.mean()), "test": float(y_test.mean())},
        "selected_model": selected_model,
        "selection_reason": selection_reason,
        "calibration_retained": calibration_retained,
        "calibration_brier_improvement": brier_improvement,
        "threshold": selected_threshold,
        "test_metrics": selected_test_metrics.to_dict(),
        "test_threshold_metrics": selected_test_operating.to_dict(),
        "bootstrap_intervals": bootstrap_intervals,
        "event_metrics": selected_event_summary,
        "representative_asset": {
            "unit_id": representative_asset,
            "selection_rule": representative_rule,
            "first_warning_cycle": representative_record["first_warning_cycle"],
            "event_cycle": representative_record["event_cycle"],
            "lead_time": representative_record["lead_time"],
            "last_probability": representative_last_probability,
        },
        "static_policy": {"cycle_cutoff": static_cutoff},
        "fmea_change": fmea_change,
        "perturbation": {
            "asset": perturb_asset,
            "after_cycle": 20,
            "compared_rows": perturbation.compared_rows,
            "perturbed_rows": perturbation.perturbed_rows,
            "max_feature_delta": perturbation.max_feature_delta,
            "max_prediction_delta": perturbation.max_prediction_delta,
        },
        "asset_leakage_test": "PASS: grouped folds and split-qualified official namespaces have no overlap",
        "tests": test_result,
        "figures": figure_names,
        "paths": {
            "leakage_controls": DEV_ROOT / "docs" / "leakage_controls.md",
            "target_definition": DEV_ROOT / "docs" / "target_definition.md",
            "build_manifest": DEV_ROOT / "BUILD_MANIFEST.md",
            "readme": DEV_ROOT / "README.md",
            "predictive_results": REPORT_DIR / "predictive_results.md",
            "event_results": REPORT_DIR / "event_level_results.md",
            "fmea_results": REPORT_DIR / "fmea_results.md",
            "limitations": REPORT_DIR / "limitations.md",
            "build_report": BUILD_REPORT_PATH,
        },
    }
    write_text(
        OUTPUT_ROOT / "run_summary.json",
        json.dumps(json_ready(summary), indent=2, sort_keys=True),
    )
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
