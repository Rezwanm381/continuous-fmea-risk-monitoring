# Condition-informed FMEA framework

## Case boundary

This prototype supports human review of a narrow turbofan degradation scenario. NASA C-MAPSS FD001 supplies simulated condition histories and one documented high-pressure-compressor degradation/end-of-life outcome. The FMEA wording and ratings are a separate `ENGINEERING_SCENARIO_ASSUMPTION`; the dataset does not provide row-level labels for invented submodes or certified aerospace ratings.

## Layer 1: governed engineering FMEA

The static register records failure mode, effect, potential cause, current controls, Severity, Occurrence, Detection, static RPN, severity-led static priority, engineering action, and explicit scenario basis.

- `Static_RPN = Severity x Occurrence x Detection` is a secondary conventional reference only.
- Severity remains expert governed.
- Occurrence is not automatically replaced by model probability.
- Detection is an illustrative expert-governed detectability/control-effectiveness rating pending domain validation; it is not model confidence or demonstrated field effectiveness.
- Static scores are frozen during condition escalation.

## Layer 2: condition observation

The model estimates dataset-internal probability of the independently defined 30-cycle event from current and trailing condition history. Probability, trend, threshold, data-quality status, and model version remain visible as separate evidence.

## Decision-support layer

The rule changes urgency, not the FMEA facts:

1. Static priority establishes baseline review importance.
2. A validation-selected warning threshold implements a predeclared recall policy.
3. Fixed condition bands remain separate from that warning threshold: `LOW < 0.20`, `MODERATE = [0.20,0.50)`, `HIGH = [0.50,0.80)`, and `CRITICAL >= 0.80`. These are `PORTFOLIO_SCENARIO_ASSUMPTION` values, not validated industrial limits or IEC/ISO thresholds.
4. A high-severity item at `HIGH` or `CRITICAL` condition level is escalated to prompt inspection/review.
5. Other items may be advanced one urgency tier for review, but Severity/Occurrence/Detection and Static_RPN remain unchanged.
6. A human records accept, defer, reject, or request-more-data disposition.

This is a transparent prototype policy, not a standardized "dynamic RPN" method and not an autonomous maintenance command.

## Static comparison

The comparison is a `SCENARIO_POLICY_ILLUSTRATION` between a frozen age/periodic inspection proxy and the same governed FMEA augmented with condition evidence. The policies are nearly matched only on development alert-state rows, then compared on the locked test set for eligible-asset warning coverage, lead time, false-alert rows/episodes, and row-alert burden.

They are not matched on inspected assets, alert/inspection episodes, maintenance hours, cost, asset-level workload, or resource constraints. The question is whether condition history changes timing under the coded scenario—not whether machine learning “beats FMEA” or establishes operational superiority.
