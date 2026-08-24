# Target definition

## Engineering question

At engine cycle `t`, using only condition history available through `t`, does the NASA C-MAPSS FD001 engine reach its simulated end-of-useful-life condition within the next 30 cycles?

```text
information available through cycle t
        → current/trailing features
        → condition probability
        → simulated event within cycles t through t+30
```

## Primary outcome

For asset `i`, cycle `t`, event cycle `E_i`, and horizon `H = 30`:

`RUL(i,t) = E_i - t`

`Y(i,t;30) = 1` when `0 <= RUL(i,t) <= 30`, otherwise `0`.

The terminal run-to-failure training row is included as positive (`RUL = 0`) because C-MAPSS contains pre/post-cycle condition snapshots through the simulated failure cycle and contains no post-failure rows. Negative RUL is invalid and causes a validation failure.

## Event-cycle construction

- Development file: `E_i` is the final recorded cycle of each complete run-to-failure engine.
- Official test file: `E_i = last_observed_cycle_i + terminal_RUL_i`, using the supplied NASA test truth.
- The two splits are processed separately; split-qualified namespaces are used for cross-split overlap assertions because raw unit numbers restart at 1.
- `E_i`, RUL, and the target are retained only in the label/evaluation frame; they are excluded from the predictor matrix.

## Why 30 cycles

Thirty cycles is a predeclared engineering-scenario horizon within the 20/30/40-cycle range identified in the prior architecture. It is long enough to represent a meaningful inspection-planning window on this benchmark while remaining near-term relative to the observed engine lives. It is not an industry standard, a calendar duration, or a value selected to maximize model performance.

## Follow-up and event eligibility

Training engines run to the simulated event, so all rows have known follow-up. Official test trajectories are truncated but have terminal RUL truth, so row labels are known. For event-level *warning coverage*, an official-test asset is eligible only when its released trajectory contains at least one row inside the 30-cycle event window. Assets whose final observed RUL exceeds 30 are not counted as missed warnings; they remain valid negative-horizon exposure for false-alert evaluation.

Exactly 25 of 100 official-test engines are eligible under this observability rule. Sixteen receive a qualifying warning, so the 64% estimate has a Wilson 95% interval of approximately 44.5%–79.8% and must not be presented as precise production performance.

## Benchmark-structure implication

Development prevalence is 15.026%, while released official-test prevalence is 2.535%. Official truncation creates 75 all-negative released histories and explains most of the difference. Terminal-window labels and correlated sequential rows also make ranking easier than a real field failure task. This is a limitation of interpretation, not target leakage and not a reason to tune a different horizon.

## Scope of the outcome

The label means **simulated end of useful life due to the single documented FD001 high-pressure-compressor degradation mode within 30 cycles**. It is not a field failure, a maintenance work order, a safety incident, an FMEA rating, or a transferable production-fleet probability.
