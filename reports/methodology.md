# Methodology

## Project architecture

The implementation is a `CONDITION_INFORMED_FMEA_DECISION_SUPPORT_PROTOTYPE` with three deliberately separate components:

1. a governed, frozen engineering FMEA scenario;
2. a condition model estimating a future simulated end-of-life event;
3. a transparent decision rule that changes review/inspection urgency without rewriting FMEA ratings.

The historical circular RPN regression appears only as `HISTORICAL_INVALIDATED_APPROACH` and is not a comparator.

## Dataset and rights

The primary and only modeling dataset is NASA C-MAPSS FD001: simulated run-to-failure turbofan trajectories with one operating condition and one documented high-pressure-compressor degradation mode. Development has 100 complete engines; the official test has 100 separately simulated, truncated engines plus terminal RUL truth. Raw access is public and provenance is verified, but NASA's record states `License not specified`, so redistribution remains `VERIFY_BEFORE_PUBLICATION`.

Development target prevalence is 15.026%; released official-test prevalence is 2.535%. Official truncation explains most of this difference: 75 test histories end before the 30-cycle window. The shift materially affects calibration, threshold transport, and interpretation.

## Target and timing

At engine cycle `t`, the target is one when simulated end of useful life occurs within 30 cycles (`0 <= event_cycle - t <= 30`). The 30-cycle horizon is a predeclared scenario assumption, not a tuned value or calendar duration. RUL and event cycle are used only to construct the label and event metrics.

Predictors use current observations and right-aligned trailing history. Rolling windows never center or look forward. Features are constructed within cycle-sorted, namespaced assets; train/test unit-number reuse cannot merge engines accidentally.

## Features

The parsimonious feature families are:

- age/usage: current cycle;
- operating context: the two nonconstant supplied settings;
- current condition: selected sensor values available at `t`;
- trailing condition: 5- and 10-cycle mean, standard deviation, and slope for a small predeclared sensor subset;
- optional trailing deviation/baseline terms implemented only from past/current rows.

Exact and near-constant fields are reported. The fixed predictor policy excludes six exactly constant sensor channels, the constant third operating setting, and the deliberately omitted two-valued sensor 6. Learned imputation and scaling live inside the fitted pipeline; no learned variance filter is used.

## Development and locked test

The official NASA split is preserved. Development uses five-fold `GroupKFold` on complete engines. No engine is shared across paired model-development folds. Out-of-fold development probabilities drive model comparison, calibration choice, threshold selection, and ablations. The official test is evaluated after those decisions are frozen.

Because each engine has a relative lifecycle clock, a global row-wise random split or global `TimeSeriesSplit` is not used.

## Models and preprocessing

Required baselines are:

- fold-specific constant development prevalence;
- age/cycle-only Logistic Regression.

Candidate models are regularized Logistic Regression and one restrained Random Forest. Logistic Regression uses fold-local imputation and scaling. Random Forest uses fold-local imputation and a fixed 200-tree, depth-10 configuration. No SMOTE, deep learning, seed search, or large hyperparameter grid is used. Fixed seed: 20260824.

## Calibration and model selection

Primary selection criterion is grouped development PR-AUC. Calibration quality and simplicity support the decision: Random Forest is selected only when its PR-AUC exceeds Logistic Regression by more than the predeclared 0.01 materiality tolerance. Sigmoid recalibration is evaluated with nested group-cross-fitted mappings: each outer fold's calibrator is trained on inner grouped OOF scores from outer-training assets only. It is retained only when it lowers development OOF Brier by at least 0.001 with a positive slope. The final mapping is fitted only on all development OOF predictions. Test labels never fit the calibrator.

The validated nested sigmoid improvement was approximately 0.000125, so calibration was rejected and raw Logistic probabilities were retained. This is not evidence of perfect calibration or external probability transfer.

## Metrics and uncertainty

Primary predictive evidence is PR-AUC (average precision). ROC-AUC and Brier score are also reported. Precision, recall, and F1 are secondary operating-point metrics. Calibration is shown with a reliability curve and bin diagnostics.

ROC-AUC is not used as an accuracy headline. Positives are contiguous terminal-window rows, sequential observations are correlated, age/degradation trajectories are informative, and official truncation creates many all-negative released histories. These properties make benchmark discrimination easier than real field failure prediction.

Asset-level bootstrap intervals resample whole engines rather than correlated rows. They describe uncertainty within this benchmark only.

## Threshold policy

The warning threshold is the highest raw grouped-development OOF score attaining the predeclared row-level recall target (80%). When sigmoid calibration is retained, that raw scalar is mapped through the final monotone development-fitted sigmoid so the same decision boundary is used on the calibrated probability scale. It is not selected by maximizing test F1. The warning threshold and separate fixed condition bands are frozen before official-test predictions are scored.

The final threshold is 0.871048942. Bounded checks do not retune it: at ±0.01, eligible coverage remains 64% and false-alert rows remain 4; at −0.02 coverage is 76%; at +0.02 it is 60%. Official-test row recall is 50.9%, so the 80% value is a development policy target rather than a transported guarantee.

## Event-level evaluation

For each eligible official-test engine, the first threshold crossing inside the observed 30-cycle target window is the first *qualifying* warning and `lead_time = event_cycle - first_qualifying_warning_cycle`. Earlier target-negative crossings remain false alerts and do not qualify the asset as warned. Warning coverage is the fraction of engines whose released history actually enters the 30-cycle event window and receives a qualifying warning. Assets truncated before the event window are not counted as missed; they remain valid exposure for false-alert evaluation. False-alert rows, episodes, time in warning, unwarned eligible assets, and alert workload are reported.

A representative case is selected only after finalization by a deterministic rule: the correctly warned eligible test asset with lead time closest to the median (ties resolved by asset ID). This prevents best-case cherry-picking.

Only 25 official-test engines are eligible for coverage/lead evaluation. Sixteen are warned: 64% with a Wilson 95% interval of approximately 44.5%–79.8%. Unequal released follow-up inside the window makes the result conditional and descriptive. Any resampling uncertainty uses complete assets, never individual rows.

## Ablations and importance

Compact grouped-development ablations compare `AGE_ONLY`, `CURRENT_SENSORS`, `CURRENT_PLUS_ROLLING`, and `FULL_PARSIMONIOUS`. The analysis asks whether condition history adds signal beyond simple aging. A selected Random Forest uses held-out permutation importance; a selected Logistic Regression uses standardized coefficients. Importance is associative, not causal.

## Static versus condition-informed policy

The static policy is a frozen cycle/inspection-age proxy tied to the same static FMEA scenario. Its development alert-state-row exposure is nearly matched to the selected condition-policy exposure. Inspected assets, episodes, maintenance hours, cost, asset-level workload, and resource constraints are not matched. Both policies are then applied unchanged to the official test and compared for eligible-event coverage, lead time, false-alert behavior, and row-alert burden. This `SCENARIO_POLICY_ILLUSTRATION` is not “ML versus FMEA” and cannot establish operational superiority, avoided failure, cost, downtime, or safety benefit.

## Leakage controls

Automated controls cover target boundaries, trailing-only feature invariance, asset overlap, finite/deterministic probabilities, future-value perturbation, event metrics, and non-mutating FMEA escalation. The complete firewall is in `docs/leakage_controls.md`.

## Reproducibility

`python run_analysis.py` acquires/loads permitted data, validates hashes/schema/order, constructs the target/features, executes grouped development, fits/calibrates the locked model, evaluates once on the official test, runs event/FMEA/policy analyses, performs leakage checks, and writes tables, figures, and reports. Notebooks are explanatory clients of the modules and are never required for the build.
