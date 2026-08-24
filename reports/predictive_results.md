# Predictive results

## Scope and class prevalence

The independently defined outcome is simulated end of useful life within 30 cycles. It is not RPN. Development prevalence was **15.026%**; official-test prevalence was **2.535%**. Official test histories are truncated, and 75 of 100 do not release a row inside the final 30-cycle window. This construction shift materially affects calibration and threshold transfer. PR-AUC is primary because positive rows are uncommon.

## Model comparison

| Model | Features | Calibration | Dev_OOF_PR_AUC | Dev_OOF_ROC_AUC | Dev_OOF_Brier | Policy_Threshold | Test_PR_AUC | Test_ROC_AUC | Test_Brier | Test_Recall_at_Threshold | Test_Precision_at_Threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant_prevalence | constant | raw probability | 0.15015 | 0.49968 | 0.12768 | 0.15025 | 0.02535 | 0.50000 | 0.04031 | 1.00000 | 0.02535 |
| age_only | age | raw probability | 0.50841 | 0.89537 | 0.09526 | 0.18830 | 0.21034 | 0.92987 | 0.02750 | 0.62952 | 0.17757 |
| logistic_regression | full_parsimonious | raw probability | 0.97997 | 0.99601 | 0.01796 | 0.87105 | 0.90513 | 0.99689 | 0.00701 | 0.50904 | 0.97688 |
| random_forest | full_parsimonious | raw probability | 0.97011 | 0.99414 | 0.02203 | 0.67853 | 0.88932 | 0.99636 | 0.00759 | 0.52108 | 0.95580 |

All thresholds were chosen from grouped development OOF probabilities for 80% row recall and applied unchanged to the official test. Baselines are the fold-specific prevalence model and age-only Logistic Regression. The historical circular-RPN regression is `HISTORICAL_INVALIDATED_APPROACH`, not a baseline.

## Selection and calibration

Selected model: **logistic_regression**. Logistic Regression was retained because Random Forest did not exceed its grouped-OOF PR-AUC by more than the predeclared 0.01 tolerance.

Sigmoid calibration was not retained: nested grouped development OOF Brier changed from 0.01796 to 0.01784 (improvement 0.00012; final slope 0.93672); retention required at least 0.001 improvement and a positive slope. Raw probabilities were retained. This result does not establish perfect calibration or transfer to another fleet, sensor system, operating regime, or prevalence.

On the locked test, the selected probability achieved PR-AUC **0.90513**, ROC-AUC **0.99689**, and Brier **0.00701**. At threshold **0.87105**, recall was **0.509**, precision **0.977**, and F1 **0.669**.

The high ROC-AUC is plausible for this benchmark but is not a production headline: positives are contiguous terminal-window observations, sequential rows are correlated, age and simulated degradation trajectories are informative, and official truncation creates many all-negative released histories. The result is benchmark discrimination, not “99.7% accuracy,” real-failure prediction, or field validity.

Whole-engine bootstrap percentile intervals (500 replicates) were: PR-AUC 0.85860–0.93858; ROC-AUC 0.99564–0.99806; Brier 0.00477–0.00941. These intervals describe only this simulated benchmark.

## Threshold discipline and bounded sensitivity

The threshold **0.871048942** was selected only from grouped-development OOF predictions as the highest score meeting 80% development row recall. It achieved 50.9% row recall on the truncated official test, so 80% is a development policy target, not a transported guarantee. At ±0.01, eligible coverage remained 64% and false-alert rows remained 4. At −0.02 coverage was 76%; at +0.02 it was 60%. These are fixed sensitivity checks, not threshold retuning.

## Ablation

| Feature_Group | Feature_Count | Model_Architecture | Dev_OOF_PR_AUC | Dev_OOF_ROC_AUC | Dev_OOF_Brier | PR_AUC_Delta_vs_AGE_ONLY |
| --- | --- | --- | --- | --- | --- | --- |
| AGE_ONLY | 1 | logistic_regression | 0.50841 | 0.89537 | 0.09526 | 0.00000 |
| CURRENT_SENSORS | 14 | logistic_regression | 0.94781 | 0.98883 | 0.02915 | 0.43941 |
| CURRENT_PLUS_ROLLING | 56 | logistic_regression | 0.96195 | 0.99253 | 0.02481 | 0.45354 |
| FULL_PARSIMONIOUS_FEATURE_SET | 59 | logistic_regression | 0.97997 | 0.99601 | 0.01796 | 0.47156 |

The highest grouped-development ablation was **FULL_PARSIMONIOUS_FEATURE_SET** (PR-AUC 0.97997). The full parsimonious set's change from age-only was +0.47156. These are within-setup comparisons, not proof of universal model superiority.

## Feature importance

| Feature | Importance | Signed_Effect | Method |
| --- | --- | --- | --- |
| asset_age | 3.56196 | -3.56196 | absolute standardized logistic coefficient |
| sensor_11__from_initial | 1.59500 | 1.59500 | absolute standardized logistic coefficient |
| sensor_11__roll_mean_5 | 1.56881 | 1.56881 | absolute standardized logistic coefficient |
| sensor_11__roll_mean_10 | 1.50674 | 1.50674 | absolute standardized logistic coefficient |
| current__sensor_11 | 1.33534 | -1.33534 | absolute standardized logistic coefficient |
| sensor_4__roll_mean_5 | 1.16677 | 1.16677 | absolute standardized logistic coefficient |
| sensor_15__roll_mean_10 | 1.12709 | 1.12709 | absolute standardized logistic coefficient |
| current__sensor_14 | 0.98500 | 0.98500 | absolute standardized logistic coefficient |
| sensor_4__from_initial | 0.93983 | 0.93983 | absolute standardized logistic coefficient |
| current__sensor_4 | 0.92069 | -0.92069 | absolute standardized logistic coefficient |
| current__sensor_9 | 0.80142 | 0.80142 | absolute standardized logistic coefficient |
| current__sensor_8 | 0.78683 | 0.78683 | absolute standardized logistic coefficient |

Importance is associative and model-specific, not causal. It was computed only after the final model and evaluation policy were frozen.
