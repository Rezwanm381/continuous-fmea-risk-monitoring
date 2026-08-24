# Publication artifact policy

This policy classifies the current `05_OUTPUTS` tree and records the Module 8A interim decision. It does not authorize publication or replace final Module 8B review.

## `PUBLIC_SAFE_AGGREGATE` candidates

These artifacts contain aggregate or illustrative results and remain candidates for final Module 8B review only with their captions, limitations, attribution, and rights status intact:

- `figures/*.png`, except that only the data-independent validation/leakage schematic is interim-allowlisted;
- `tables/model_comparison.csv`;
- `tables/ablation_results.csv`;
- `tables/data_quality_summary.csv`;
- `tables/sensor_ranges.csv`;
- `tables/static_vs_condition_comparison.csv`;
- `tables/static_fmea.csv`;
- `tables/condition_informed_fmea.csv`;
- `tables/feature_importance.csv`;
- `tables/calibration_development.csv`;
- `tables/calibration_test.csv`;
- `tables/bootstrap_metrics.csv`; and
- `run_summary.json` after path/status review.

`PUBLIC_SAFE_AGGREGATE` means “eligible for Module 8B review,” not “cleared for release.” NASA-data-derived figures are `HOLD_FOR_FINAL_8B`; public access does not establish derivative redistribution rights.

## `INTERNAL_ROW_LEVEL_DERIVED`

Exclude from the public candidate pending rights review:

- `tables/test_predictions.csv`;
- `tables/per_asset_event_metrics.csv`;
- `tables/all_asset_alert_burden.csv`; and
- `tables/static_policy_event_metrics.csv`.

These files expose row- or engine-level trajectory/event structure derived from NASA data.

## `PRIVATE_RAW_DATA`

Exclude unconditionally unless Module 8 records explicit redistribution clearance:

- `private_data_cache/**`;
- NASA archives; and
- extracted FD001 raw text files.

## Internal runtime and validation material

Never export:

- `.analysis_env/**`;
- `.matplotlib*/**`;
- `module_7_5D_validation_work/**`;
- `module_7_5D_clean_reproduction/**`;
- `m75d_repro/**`;
- Python/pytest caches; and
- internal placeholders or local scratch copies.

## Allowlist rule

Module 9 may build an interim export only from the explicit Module 8A allowlist rooted at the repository root. It may not publish `05_OUTPUTS` wholesale. Final artifact decisions remain subject to Module 8B.
