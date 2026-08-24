# Locked-test access log

1. **Target-integrity access (2026-08-24):** after the dataset, horizon, target formula, feature policy, split, models, model-selection rule, calibration rule, and threshold rule were frozen, the official target structure was inspected to verify row counts and event-metric eligibility. This established that 25 released test trajectories enter the 30-cycle window.
2. **Scripted final evaluation (2026-08-24):** `python run_analysis.py` fitted all decisions on grouped development data, then generated final locked-test row metrics, event analysis, calibration diagnostics, bootstrap intervals, and final-model interpretation in one scripted pass.

This is a retrospective portfolio reconstruction, not a preregistered experiment. Re-running the deterministic script reproduces evaluation but should not be interpreted as a new independent test.
