# Data quality report

## Verified source and structure

NASA C-MAPSS FD001 was loaded from the official public repository. The integrity-checked raw cache is stored outside the repository source tree and excluded from publication packaging because NASA's dataset record does not specify a license.

| number_of_assets | number_of_rows | missing_cells | duplicate_asset_cycles | out_of_order_assets | cycle_gap_count | constant_columns | near_constant_columns | target_prevalence | split |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 20631 | 0 | 0 | 0 | 0 | op_setting_3; sensor_1; sensor_5; sensor_10; sensor_16; sensor_18; sensor_19 |  | 0.15026 | development_train |
| 100 | 13096 | 0 | 0 | 0 | 0 | op_setting_3; sensor_1; sensor_5; sensor_10; sensor_16; sensor_18; sensor_19 |  | 0.02535 | official_test |

The two official splits contain **200 separately simulated engines** and **33,727 observations**: 20,631 development rows and 13,096 official-test rows. Unit numbers restart in the test files, so split-qualified namespaces are used for overlap checks.

## Integrity checks

- Missing required cells: 0 in both splits.
- Duplicate `(unit_id, cycle)` keys: 0 in both splits.
- Out-of-order engines: 0; cycle gaps: 0.
- Exact-constant fields in both splits: `op_setting_3`, `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, and `sensor_19`.
- Six exactly constant sensor channels, the constant third setting, and the deliberately omitted two-valued `sensor_6` are excluded from model predictors. No channel was dropped solely because of correlation.
- Full sensor/setting ranges and use decisions are recorded in `05_OUTPUTS/tables/sensor_ranges.csv`.

## Target and event distribution

Development prevalence is **0.15026** (3,100/20,631); locked-test prevalence is **0.02535** (332/13,096). The difference follows the official test truncation design. Development trajectories end at simulated EOL. Official test terminal RUL ranges from 7 to 145 cycles (median 86.0). Only **25** test trajectories contain released observations inside the 30-cycle target window; the other 75 remain valid negative exposure for row metrics and false-alert burden but are not counted as missed qualifying warnings.
