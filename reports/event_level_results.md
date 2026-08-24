# Event-level results

## Eligibility and warning policy

The threshold **0.87105** corresponds to the highest raw grouped-development OOF score meeting the predeclared 80% row-recall target. A qualifying warning is the first threshold crossing in a released row inside the 30-cycle event window. Only the **25** official-test engines whose released histories enter that window are eligible for coverage and lead time. All 100 test engines contribute to false-alert exposure.

## Selected condition policy

- Qualifying warning coverage: **16/25 = 64.000%**.
- Wilson 95% interval: **44.5%–79.8%**.
- Median qualifying first-warning lead: **22.0 cycles** among warned eligible engines.
- Eligible fraction never warned: **36.000%**.
- False-alert rows across all test engines: **4** (0.04 per engine).
- False-alert rows per 1,000 target-negative exposure rows: **0.31**.
- False-alert episodes across all test engines: **2** (0.02 per engine).
- Engines with at least one false-alert episode: **2.000%**.

## Representative asset

Test engine **24** was selected after finalization by the rule: eligible warned asset nearest the median lead time; asset ID tie-break. Its first qualifying warning cycle is **184.0**, simulated event cycle **206**, and qualifying lead time **22.0 cycles**. The final released-row probability was **0.97151**. This is a deterministic median-case illustration, not a best-case selection.

## Limits

The event is simulated benchmark EOL, not a field failure. Seventy-five test engines are truncated before the 30-cycle window and therefore cannot support qualifying-warning coverage. The 64% point estimate is imprecise and conditional on unequal released follow-up inside the window; it is not a stable production-performance estimate. Alert rows and episodes are scenario workload proxies, not maintenance costs. Test access was retrospective and is logged; no test result was used to redesign the target, features, model set, calibration rule, selection tolerance, or threshold policy.
