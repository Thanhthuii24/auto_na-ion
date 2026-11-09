# v5 (Na-ion) — Gradient checking and diagnostics summary (thesis contribution 2)

All physics-loss runs use L = Relative Huber(δ = 0.15) + λ·L_phys, λ calibrated before step 1 as
λ = ρ · median_b ‖∇L_main‖ / ‖∇L_cal‖ over 3 fixed training batches, all parameters, float64 (loss.md §5.2).
Balance band (loss.md §5.3): weighted gradient ratio R = λ‖∇L_phys‖/‖∇L_main‖, median in [0.05, 2.0]; cosine and loss-value ratio are reported.

## 1. Per-run gradient record (split = paper split; val MAPE of the selected checkpoint)

| Run | Physics loss | Split | λ | median R (§5.3) | cosine samples | active fraction | §5.3 balance | val MAPE | decision |
|---|---|---|---|---|---|---|---|---|---|
| NA-E021 | P2 (anchor both, unmatched dropout — non-compliant) | 2021 | 0.057 | 0.15 | [-0.21, 0.81] | 0.48→0.52 | PASS | 0.0819 | DISCARD |
| NA-E022 | P2 compliant | 2021 | 0.118 | 1.88 | [0.79, -1.0] | 0.43→0.41 | PASS | 0.1760 | DISCARD |
| NA-E023 | prefix drift two-sided | 2021 | 0.018 | 0.35 | [0.38, -0.94] | 0.3→0.27 | PASS | 0.1019 | DISCARD |
| NA-E024 | prefix drift one-way | 2021 | 0.017 | 0.51 | [0.75, -0.97] | 0.3→0.3 | PASS | 0.1157 | DISCARD |
| NA-E025 | degradation law, learned b, rho 0.5 | 2021 | 0.089 | 0.59 | [0.58, 0.98] | 0.95→0.69 | PASS | 0.1017 | DISCARD |
| NA-E026 | degradation law, learned b, rho 1.0 | 2021 | 0.178 | 1.16 | [0.58, 0.98] | 0.95→0.69 | PASS | 0.0908 | DISCARD |
| NA-E027 | degradation law, learned b, rho 1.0 | 42 | 0.165 | 0.56 | [0.92, -0.11, -0.13, -0.76, 0.28, 0.96] | 0.97→0.23 | PASS | 0.1121 | KEEP |
| NA-E028 | degradation law, learned b, rho 1.0 | 2024 | 0.126 | 0.38 | [0.75, 0.72, 0.15, 0.84] | 0.94→0.23 | PASS | 0.1116 | DISCARD |
| NA-E032 | N1 fixed-b law, rho 1.0 (FROZEN) | 2021 | 0.481 | 1.55 | [0.09, 0.87] | 0.77→0.6 | PASS | 0.0767 | KEEP |
| NA-E033 | N1 fixed-b law, rho 1.0 (FROZEN) | 42 | 0.241 | 0.81 | [0.12, -0.64, -0.71, -0.8, -0.41, 0.09] | 0.73→0.15 | PASS | 0.1076 | KEEP |
| NA-E034 | N1 fixed-b law, rho 1.0 (FROZEN) | 2024 | 0.337 | 0.61 | [0.41, 0.63, 0.34, 0.95] | 0.73→0.15 | PASS | 0.1085 | KEEP |

NA-E027 is KEEP versus its split-matched parent, but the learned-b law as a whole failed its 3-split physics gate (A5-NA).

## 2. Findings

1. **Balance PASS is necessary, not sufficient.** Every physics run sat inside the §5.3 band, yet NA-E022 collapsed (MAPE 0.176,
   best epoch 1). Gradient balance and accuracy must be reported as two separate axes (tools/health_v2.py).
2. **The late-training cosine separates useful from harmful physics terms.**
   - P2 and prefix drift end strongly opposed to the main loss (cosine −0.94 … −1.00) → accuracy loss or no gain.
   - The degradation laws end aligned (+0.84 … +0.98) → accuracy gains (learned b: 3-split 0.1049; fixed b N1: 0.0976).
   - Exception kept in the record: N1 on split 42 stays negative through mid-training (−0.64 … −0.80) before recovering (+0.09),
     so gradient alignment is NOT uniform across splits.
3. **Zero-perturbation test exposed a silent implementation fault.** With f(x) and f(x̃) on independent dropout masks, the
   one-sided hinge was already 0.04–0.06 at β = 0 (P2) and the prefix-drift "drift" was 0.125 on an identity view versus a real
   drift of 0.083: the term measured dropout noise, not physics. Fix: `physics_dropout_matched` (both views on the same
   dropout-free network, anchor sg[f(x)]); after the fix the β = 0 hinge is exactly 0.
4. **Sensitivity pre-check predicts whether a what-if loss can work.** P2 at the data range moved the parent by 0.002–0.005 cycles
   (sub-resolution) → the compliant term degenerates into a constant "predict shorter" push (cosine −0.996 with the main loss).
   Capacity what-ifs (P0/P0c/P4b) move it by −13 … −70 cycles with 0 violations → already satisfied by the architecture.
   The same pre-check rejected the CE and hysteresis what-ifs before any run (loss.md §4.8).
5. **λ calibration range.** §5.2 produced λ from 0.017 (prefix drift) to 0.48 (N1, split 2021); the same ρ gives different λ per
   split because the gradient norms differ, so λ must be re-calibrated per split, never copied.
6. **Diagnostic tool fixes (reviewed by Codex).** (a) zero-initialised parameters (ext_da, ext_dc) were flagged as EXPLODING
   (grad/param ratio 6·10¹⁰ with grad 0.06–0.38) → denominator floor 1e-2·√n; (b) the legacy physics_balance thresholds (R ≤ 1,
   loss-value ratio ≤ 1, cosine ≥ −0.3 as gates) contradicted loss.md §5.3 → aligned; (c) branch-imbalance no longer compares a
   scalar calibration parameter with whole blocks. The remaining WARNING (gate vs LayerNorm, ratio ~1.3·10³) is by design
   (the gate is reparameterised ×50 so it can learn under the locked lr; without it the gate stayed at its init, NA-E006).

## 3. What the thesis may claim
- A gradient-based protocol (calibrated λ, balance band, cosine, active fraction, zero-perturbation and sensitivity checks)
  that caught two silent faults and predicted which physics terms help.
- Not: that every kept physics term is gradient-aligned on every split (N1 split 42), or that balance PASS implies accuracy.
