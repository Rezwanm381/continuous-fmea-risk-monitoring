# Independent validation summary

## Verdict

`VALIDATED_WITH_MAJOR_LIMITATIONS`

The internal Module 7.5D reproduction audit separately ran the technical workflow, audited target construction and leakage controls, recomputed all headline metrics, reviewed the code/tests/notebooks/figures, and corrected one development-calibration validation defect. No critical issue remains.

## Strong validated items

- Historical circular RPN prediction: `INVALID_FOR_PREDICTIVE_EVIDENCE`.
- NASA C-MAPSS FD001 identity and official-test event-cycle construction: pass.
- Target independence: `TARGET_INDEPENDENCE_PASS`.
- Temporal feature timing and future-value perturbation: pass; maximum earlier delta `0.0`.
- Five engine-grouped development folds and official split separation: pass.
- Fold-local preprocessing: pass.
- Nested calibration isolation: corrected and regression-tested.
- Logistic Regression selection: grouped-development OOF PR-AUC `0.979969253`.
- Locked-test metrics reproduced: PR-AUC `0.905130191`, ROC-AUC `0.996893324`, Brier `0.007012987`.
- Threshold `0.871048942`: selected on development OOF only.
- Event metrics reproduced: 16/25 coverage, 22-cycle median lead, 4 false rows, 2 episodes.
- Static FMEA arithmetic and separation from condition urgency: pass.
- Test suite: 74 tests plus 36 subtests pass.
- Three new notebooks execute top-to-bottom.
- Critical issues: zero.

## Corrective action

The original sigmoid diagnostic did not fully isolate each outer calibration fold. Proper nested grouped OOF calibration changed development Brier from `0.017961356` to `0.017836767`, an improvement of `0.000124589`. This remained below the predefined `0.001` retention gate, so calibration stayed rejected. Raw Logistic probabilities, threshold, locked metrics, event results, FMEA scenario results, tables, and figures were unchanged.

## Major limitations that remain

1. NASA raw and row/asset-level derived redistribution remains `VERIFY_BEFORE_PUBLICATION`.
2. Static-versus-condition policies are nearly matched only on development alert-state rows, not operational workload.
3. Event evidence uses only 25 eligible official-test engines and depends on unequal released follow-up; 16/25 has Wilson 95% interval approximately 44.5%–79.8%.
4. Calibration and threshold transport are uncertain; the development 80% threshold achieved 50.9% official-test row recall.
5. Terminal-window labels, correlated sequential rows, age/degradation structure, and official truncation can inflate apparent performance relative to a field failure task.
6. No field, domain, deployment, causal, maintenance-cost, cybersecurity, safety, certification, or standards-compliance validation exists.

The full independent evidence is in the portfolio-level `08_REPORTS/module_7_5D_continuous_fmea_validation_report.md` and its reproduction/issue registers.
