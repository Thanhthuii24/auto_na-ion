# Research and Diagnostic Guide

Use this file to understand run diagnostics and formulate the next falsifiable hypothesis.
Policy and gates live in `program.md`. Benchmark rules live in `protocol.md`.

## 1. Diagnostic Thought Process
Always follow this strict reasoning sequence:
```text
Measured Facts (diagnostics.json) -> Symptom -> Competing Causes
-> Prioritize One Cause -> Falsifiable Hypothesis -> Single Mechanism Intervention
```

## 2. Separate Facts From Hypotheses
- Fact (Observation): Measured value directly from artifacts (e.g., "Validation MAPE reached 0.25 at epoch 9", "Update ratio of inter-cycle block is 0").
- Hypothesis (Idea): Unproven explanation (e.g., "Model overfits after epoch 9", "Block is disconnected from loss graph").
- Rule: Always label hypotheses explicitly; never state an assumption as a measured fact.

## 3. Reading Learning Dynamics
- Healthy: Smooth train loss decrease; validation improves stably; predictions maintain healthy variance.
- Underfit: Train loss plateaus high or drops very slowly; healthy gradients exist. (Next test: useful capacity, better feature representation).
- Overfit: Train loss drops continuously while validation error diverges. Check multiple seeds/epochs before concluding.
- Optimization Failure: NaN/Inf loss, exploding/vanishing gradients, zero updates, or activation collapse. Repair optimization before judging the architecture.

## 4. Reading Gradients and Updates
| Observed Pattern | Probable Cause | Recommended Next Step |
|---|---|---|
| Zero gradient & zero update | Disconnected from graph or missing optimizer group | Reconnect or remove block |
| Small gradient, non-zero update | Normal for deep stable layers | Retain; do not alter yet |
| Large/unstable gradient (>100 ratio) | Bad scaling or missing normalization | Add LayerNorm or stable residual scale |
| Early layers learn, late layers stall | Vanishing signal with depth | Reduce depth or add skip connections |
| Healthy gradients, poor validation | Representation/hypothesis limit | Change model mechanism (not LR/optimizer) |

## 5. Reading Activations and Predictions
- Activation Collapse: Near-zero standard deviation or high zero-fraction across deep blocks indicates representation collapse. Inspect norm/residual paths.
- Prediction Spread: Compare prediction distribution vs. target distribution.

    + Constant predictions or variance << target variance indicates mean-collapse.
    + Extreme predictions pulling to the mean indicates capacity or calibration limits.

## 6. Reading Prefix and Domain Subgroups
- Prefix Groups:
    + High error on short prefixes (1–10 cycles): Model lacks global initial health representation.
    + High error on long prefixes: Model fails to capture temporal degradation dynamics.
    + Abrupt error jumps between adjacent prefix lengths: Position-dependence artifact.
- Domain Behavior:
    + Domains with < 10 cells (e.g., MICH_EXP) are stress-tests; never use them alone to justify promotion.
    + Flag domain regressions when MAPE worsens by > 0.03 or Acc@15 drops by > 0.05.
    + Stable signed bias on ISU-ILCC reflects known domain calibration shifts.

## 7. Reading Mask & Padding Behavior
- Verify no sample has an empty observed mask.
- Attention pooling must mask out future padded cycles.
- Normalization and pooling must not leak unobserved (zero-filled) cycle values into observed states.

## 8. Identifying the Main Limit & Formulating Proposals
- Good Limit Statement: Specific, grounded in facts (e.g., "Short-prefix MAPE is high across all seeds because cycle-flattening overweights unobserved future slots").
- Weak Limit Statement: Vague, ungrounded (e.g., "CPMLP is weak", "Try attention", "Model needs more layers").
- Proposal Formulation: For the chosen limit, compare at least two plausible causes in your analysis before committing to one intervention.

## 9. Handling Regressions & Failed Mechanisms
- If a screen fails the hard floor or gradient gate: Mark Run: VALID, Decision: DISCARD.
- Development room (A2): a promising direction gets up to 3 evidence-led follow-ups and closes after 4 consecutive DISCARDs. A direction with no promising signal gets one targeted repair, then closes. See `program.md §7`.
- Read saved validation artifacts before proposing new compute-heavy inference.

## 10. Structured Analysis Output
Every reviewed experiment MUST synthesize its findings directly into analysis.json conforming to analysis.schema.json:
- Map facts directly to JSON evidence paths in diagnostics.json.
- Fill the Cause Ledger with at least two competing causes.
- Assign each major block an explicit action: RETAIN, REPAIR, REPLACE, or REMOVE.
- Validate locally via `python validate_analysis.py <ID>` before committing.
