# Leakage controls

## Verdict criterion

The rebuild passes the leakage gate only when every predictor for asset `i` at cycle `t` is computable from information available no later than `t`, the target is a future simulated end-of-life outcome, and development/evaluation assets do not overlap.

Independent Module 7.5D verdicts: `TARGET_INDEPENDENCE_PASS`, temporal leakage pass, grouped asset split pass, and fold-local preprocessing pass after nested calibration remediation.

## Target firewall

- Primary target: `Y(i,t;30) = 1` when simulated end of useful life is no more than 30 cycles after cycle `t`; otherwise `0`.
- `RUL`, `event_cycle`, terminal test RUL, the binary target, and any value derived from them are label/evaluation fields only.
- Historical `RPN`, `Severity`, `Occurrence`, `Detection`, alarm flags derived from RPN, and the old synthetic thresholds are prohibited model features and targets.
- Future sensor rows, future failure flags, post-event status, future downtime, full-lifetime length, and per-asset maximum cycle are prohibited predictors.
- Asset IDs are identifiers/groups, not predictors.
- The official test terminal-RUL vector is loaded only to construct locked evaluation labels and event metrics.

Compact flow:

```text
rows through t → leakage-safe feature frame → grouped-development model → 30-cycle event risk
```

Explicitly forbidden inputs are future RUL, event cycle, future sensor rows, future rolling statistics, whole-lifetime normalization, final asset length, target derivatives, and FMEA score fields.

The word *independent* means independent of the predictor-construction rule. It does not mean that a useful sensor must be statistically independent of degradation.

## Temporal feature rules

- Build each official split separately and sort by raw asset ID and cycle before feature construction; split-qualified namespaces are used only for cross-split overlap assertions.
- Current features use only the row at `t`.
- Rolling means, standard deviations, and slopes use a right-aligned trailing window ending at `t`.
- No centered window, backward fill, future smoothing, or cross-asset window is permitted.
- Baseline deviations use expanding/trailing information only; a full-lifetime asset mean is forbidden.
- Missing-value imputation, scaling, and any other learned transform are fitted inside each development training fold. Exact-constant exclusions are a fixed, documented policy; no learned variance filter is used.

## Asset and test controls

- Namespace IDs as `DEV:<unit>` and `TEST:<unit>` because raw unit numbers restart in the official files.
- Use complete-engine `GroupKFold` partitions for model development.
- Assert that no group appears in both sides of a fold.
- Preserve the official NASA train/test split. The test set is not used for feature choice, model choice, calibration, or threshold selection.
- A global `TimeSeriesSplit` is inappropriate because engine cycles are relative clocks, not one shared calendar.
- If a future analysis introduces a within-engine boundary, it must be labeled as a separate within-asset forecasting design and purge at least the feature lookback plus target horizon.

## Test-set access policy

This is a retrospective portfolio reconstruction, not a preregistered study. Target definition, horizon, feature families, candidate models, calibration rule, model-selection rule, and threshold policy are frozen before predictive results are inspected. The analysis records one target-integrity/eligibility inspection and one scripted final evaluation after development decisions are locked.

## Automated evidence

- Target boundary tests cover early, horizon-boundary, terminal, and invalid negative-RUL cases.
- Feature tests alter future sensor values by extreme amounts and require all earlier features to remain bitwise/numerically unchanged.
- Split tests fail loudly on any asset overlap.
- Pipeline tests require finite probabilities in `[0,1]`, deterministic fixed-seed behavior, and training-fold-only transformations.
- The future-value perturbation stress test fits a fixed model, changes only future rows in memory, and requires both earlier features and earlier predictions to remain unchanged.

## Calibration isolation

Sigmoid diagnostics use nested grouped folds: for each outer calibration fold, base-model meta-scores for the calibrator are generated only from inner grouped folds within the outer-training assets. The corrected diagnostic remains development-only and rejects calibration because Brier improvement is approximately 0.000125, below the 0.001 gate.

## Interpretation boundary

Passing leakage controls establishes implementation safety for this benchmark formulation. It does not establish field validity, threshold transport, fleet calibration, maintenance benefit, or safety performance.

No persisted raw or historical file is modified by these tests.
