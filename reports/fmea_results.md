# Condition-informed FMEA results

## Static engineering scenario

FD001 documents one simulated high-pressure-compressor degradation mode, not observed failure-mode labels. The three-row table is therefore an explicit `ENGINEERING_SCENARIO_ASSUMPTION` covering narrow HPC flow, efficiency, and combined-margin deterioration.

| Failure_Mode | Effect | Current_Control | Severity | Occurrence | Detection | Static_RPN | Static_Priority | Engineering_Action | Scenario_Basis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HPC flow-capacity degradation | Reduced core-flow capability and available operating margin | Scheduled HPC inspection and multichannel trend review | 8 | 5 | 5 | 200 | MODERATE | Review condition trend and plan focused inspection | ENGINEERING_SCENARIO_ASSUMPTION |
| HPC efficiency degradation | Reduced compression efficiency with elevated operating burden | Temperature/pressure trending and scheduled HPC inspection | 9 | 4 | 5 | 180 | HIGH | Prioritize engineering review when condition alert rises | ENGINEERING_SCENARIO_ASSUMPTION |
| Combined HPC performance-margin degradation | Progressive loss of simulated engine health margin | Scheduled review of correlated condition trajectories | 7 | 4 | 4 | 112 | MODERATE | Increase inspection urgency if condition evidence persists | ENGINEERING_SCENARIO_ASSUMPTION |

`Static_RPN = Severity × Occurrence × Detection` is retained as transparent secondary engineering context. It is never a predictive target, never recalculated from the model, and not treated as an interval scale or complete risk ordering.

## Condition integration

Fixed final-model probability bands are `LOW < 0.20`, `MODERATE 0.20–<0.50`, `HIGH 0.50–<0.80`, and `CRITICAL >= 0.80`. They are a `PORTFOLIO_SCENARIO_ASSUMPTION`, not validated industrial limits or IEC/ISO thresholds. The asset-level probability changes urgency/timing only. High-severity rows at high/critical condition levels receive accelerated or immediate engineering review; all S/O/D values and Static_RPN remain unchanged.

For representative test engine 24, the last released probability was 0.97151. Because FD001 has no mode-specific outcome labels, that single asset-level condition value is shared across the illustrative scenario rather than falsely attributed to a particular failure mode.

| Failure_Mode | Severity | Occurrence | Detection | Static_RPN | Condition_Probability | Condition_Alert | Recommended_Urgency | Condition_Escalated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HPC flow-capacity degradation | 8 | 5 | 5 | 200 | 0.97151 | CRITICAL | PROMPT_INSPECTION | True |
| HPC efficiency degradation | 9 | 4 | 5 | 180 | 0.97151 | CRITICAL | IMMEDIATE_ENGINEERING_REVIEW | True |
| Combined HPC performance-margin degradation | 7 | 4 | 4 | 112 | 0.97151 | CRITICAL | PROMPT_INSPECTION | True |

The example changed urgency for **3 of 3** FMEA rows. It did not change engineering severity or assert a diagnosed physical failure mode.

## Illustrative frozen-policy comparison

Under the defined illustrative policies, the static age/inspection proxy begins warning at cycle **191**, chosen only on development data to nearly match condition-policy alert-state rows (12.413% versus 12.273%). Both policies were then frozen and applied to the official test.

This matches only development alert-state rows. Inspected assets, alert/inspection episodes, maintenance hours, cost, asset-level workload, and resource constraints are not matched.

| Policy | Development_Alert_Rate | Test_Alert_Rate | Eligible_Warning_Coverage | Median_Lead_Time | False_Alert_Rows_All_Assets | False_Alert_Episodes_All_Assets | Fraction_Assets_With_False_Alerts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen static age/inspection proxy | 0.12413 | 0.02749 | 0.28000 | 24.00000 | 277 | 7 | 0.07000 |
| Condition-informed threshold | 0.12273 | 0.01321 | 0.64000 | 22.00000 | 4 | 2 | 0.02000 |

This `SCENARIO_POLICY_ILLUSTRATION` asks whether condition information changes timing and row-alert burden under the coded assumptions. It is not an operational superiority test and does not claim that ML “beats FMEA,” prevents failures, optimizes cost, or replaces engineering judgment.
