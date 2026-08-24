# Project background

## The reconstruction problem

The historical coursework explored an appealing idea: use monitored condition signals to support FMEA review. During portfolio reconstruction, however, the earlier predictive target was found to be circular. It was an RPN-like score built directly from predictor-side sensor thresholds plus noise. A model trained on the same sensors could recover that recipe without predicting a later event.

Circular targets produce convincing-looking metrics while answering the wrong question. The portfolio rebuild therefore preserves the earlier work as provenance, invalidates its predictive claims, and replaces the target with a separate future simulated degradation event.

## Why grouped asset validation matters

Rows from one engine are strongly related. A random row split would allow nearby states from the same engine to appear in both training and validation and would overstate generalization. This project keeps complete engines together in five grouped development folds and preserves NASA's separate official-test engines.

Engine cycles are relative lifecycle clocks, not a shared calendar. For that reason, a single global time-series split is not an adequate substitute for asset grouping.

## Why temporal construction matters

A forecast at cycle `t` may use current and prior measurements only. Centered rolling windows, future backfill, final lifetime, event cycle, future RUL, and whole-trajectory normalization would reveal information unavailable at prediction time. The feature code uses right-aligned trailing windows and verifies timing with an extreme future-value perturbation test.

```text
observations through t → trailing feature state → risk estimate → event in cycles t…t+30
```

## Why PR-AUC is primary

Only 2.535% of released official-test rows are positive. ROC-AUC can remain high when the many negatives are easy to rank, especially when positives form a terminal window. PR-AUC places greater emphasis on how concentrated positive predictions are and is therefore the headline discrimination metric. ROC-AUC and Brier score remain useful secondary diagnostics.

## Why probability calibration matters

Risk decisions use probabilities differently from rankings. A model can rank rows well while overstating or understating absolute risk. The project evaluates a nested development-only sigmoid mapping and retains it only if Brier score improves by at least 0.001. The observed improvement was approximately 0.000125, so raw probabilities were retained. This disciplined rejection is more informative than automatically applying calibration.

## Why event metrics matter

Row metrics do not show whether an engine receives a usable warning, how early it arrives, or how much alert burden accumulates. The project therefore reports eligible-engine coverage, first qualifying warning lead, false-alert rows, and false-alert episodes. Only 25 official-test engines enter the observable 30-cycle window, so uncertainty and follow-up limitations remain prominent.

## How FMEA and condition evidence differ

Static FMEA captures engineering judgment about failure modes, effects, causes, controls, Severity, Occurrence, and Detection. The condition model estimates a separate dataset-internal event probability. The model may change review urgency under an illustrative policy, but it does not rewrite S/O/D, diagnose a specific illustrative mode, or create a dynamic RPN.

The static-versus-condition comparison is likewise a scenario illustration. It is nearly matched on development alert-state rows only—not inspections, episodes, cost, labor, or operational workload—so it cannot establish operational superiority.

