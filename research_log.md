# Evidence-Led Research Log — v5_auto (NA-ion)

## Round Status

- Project created 2026-11-07 from the v4_auto workflow (no Li-ion results carried over).
- Data: BatteryLife processed NA-ion, paper splits 2021/42/2024 (env `BL_SPLIT`).
- Next: `NA-E000` CPMLP baseline.

## Run Entry Template

### Before training

- Experiment ID:
- Model and matched reference:
- Direct observations:
- Main hypothesis:
- One main change:
- Fixed parts:
- Expected evidence:
- Reject rule:
- Required diagnostics:
- Stage and budget:

### After training

- Run: `VALID` or `INVALID`.
- Decision: `KEEP` or `DISCARD` for a valid run; blank for an invalid run.
- Selected checkpoint and metrics:
- Aggregate learning, gradient, activation, prediction, group, mask, and
  compute findings:
- Comparison with reference:
- Main limit:
- Cause ledger with supporting, contradicting, and missing evidence:
- Block decisions in plain language:
- Conclusion:
- What this run does not prove:
- Causal intervention map:
- Next step:

Do not add the next proposal until the current run has a complete reviewed
entry, a final result row, and a matching `state.json` update.

The active Round 4 log uses the simplified Run/Decision schema throughout.
Raw experiment artifacts remain unchanged for reproducibility.

## NA-E000 — CPMLP Baseline, Split 2021 (Colab A100)

- Config: paper Na-ion CPMLP (d_model 64, d_ff 32, e_layers 1, d_layers 9), 510,593 params; Relative Huber 0.15,
  Adam 5e-5, batch 128, max 100 / min 5 epochs, patience 5 (A2-NA). Codex APPROVE after the A2-NA amendment.
- Result: VALID, KEEP. Val MAPE 0.1261, Acc@15 0.633, MAE 24.2, RMSE 30.3 cycles, bias −21.4 cycles; best epoch 8 of 13.
- Per-cell MAPE: 1-6-59 0.094, 3-2-55 0.216, 3-4-53 0.089, 4-2-47 0.261, 4-6-43 0.052, 5-5-35 0.044.
- Reading: two cells hold most of the error and the model under-predicts on average. Val MAPE is not
  comparable with the paper's CPMLP test MAPE 0.274 ± 0.026 (different partition, 3-split mean).
- Next: NA-E001 (split 42), NA-E002 (split 2024) with the same config.

## NA-E001 — CPMLP Baseline, Split 42 (Colab A100)

- Same approved config as NA-E000; only split and seed (42) change. VALID, KEEP.
- Val MAPE 0.2337, Acc@15 0.347, MAE 50.2, RMSE 62.5 cycles, bias −26.8; best epoch 13 of 18.
- Per-cell MAPE: 1-6-59 0.055, 1-8-57 0.316, 5-3-37 0.402, 5-5-35 0.148, 5-8-32 0.368, 6-6-26 0.112.
- Reading: +0.108 vs split 2021 — split 42's val cells span a wider EOL range and predictions are compressed
  toward the mean (under-prediction again). With 6 val cells per split, only the 3-split mean is meaningful.

## NA-E002 — CPMLP Baseline, Split 2024 (Colab A100)

- VALID, KEEP. Val MAPE 0.1737, Acc@15 0.518, RMSE 31.5; best epoch 6 of 11. Cell 5-2-38 MAPE 0.344.
- **Frozen CPMLP baseline (3 splits): MAPE 0.1778 ± 0.0539, Acc@15 0.499 ± 0.144, RMSE 41.4 ± 18.3 cycles.**
- Shared failure across splits: predictions compressed toward the train mean (negative bias), cells at the
  EOL extremes carry most of the error. Next: paper baselines CPTransformer / CPGRU (program.md §2b).

## NA-E003 — NaFadeNet (new Na-ion family), Split 2021 Screen

- Model: per-cycle CNN -> [h, h-h_1, q_d/q_d1, q_c/q_c1] -> causal masked TCN (1/2/4/8) -> mean+last -> MLP; 39,137 params.
  Loss Relative Huber 0.15 (Phase-1 lock). Codex APPROVE (v3 after two rejected drafts). Code by the lead (Gemini not logged in).
- Result: VALID, KEEP as candidate. MAPE 0.1234 (CPMLP 0.1261), RMSE 29.0 (30.3), Acc@15 0.642 (0.633); best epoch 7 of 12.
- Hypothesis partly refuted: prefix MAPE 0.135/0.093/0.100/0.127/0.157 still rises after 50 cycles; spread 0.76 (more
  compressed), bias -23.0. Val MAPE climbs from 0.123 to 0.210 after epoch 7 while train loss keeps falling -> small-sample overfitting.
- Next: confirm on splits 42/2024; if the mean holds, address overfitting/late-prefix within the direction (A2 follow-up).

## NA-E004 — NaFadeNet, Split 42 Confirmation

- VALID, KEEP. MAPE 0.2062 (CPMLP 0.2337), RMSE 58.1 (62.5), bias -28.7, spread 0.47; best epoch 8 of 13.
- Prefix MAPE 0.257/0.220/0.204/0.186/0.199: here error does fall with observed cycles. Worst cells 5-3-37 0.350, 1-8-57 0.317.

## NA-E005 — NaFadeNet, Split 2024 Confirmation; parent promotion

- VALID, KEEP. MAPE 0.1772 (CPMLP 0.1737), RMSE 34.0 (31.5), spread 1.15, bias -1.7. tcn_norm WEAK (REPAIR note).
- **NaFadeNet 3-split: MAPE 0.1690 ± 0.0420, Acc@15 0.498 ± 0.131, RMSE 40.3 ± 15.5 vs CPMLP 0.1778 ± 0.0539 / 0.499 / 41.4.**
  Gain 0.0089 >= 0.008 → NaFadeNet becomes the active parent.
- Diagnostic (non-learning, lead): linear extrapolation of the last 20 observed discharge capacities (q/nominal) to the
  EOL threshold 0.8 gives val MAPE 0.04–0.11 on prefixes 51–100 (NaFadeNet 0.13–0.20) but 2.5–4.7 on prefixes 1–10.
  The two estimates are complementary → next: gated extrapolation head (NA-E006).

## NA-E006 — NaFadeNet + gated fade-extrapolation head, Split 2021 Screen

- Head change only (+99 params); loss Relative Huber 0.15. Codex APPROVE.
- VALID, KEEP as candidate: MAPE 0.1053 (parent 0.1234, CPMLP 0.1261), Acc@15 0.687, RMSE 24.0, bias -18.6, spread 0.62.
  Prefix MAPE 0.111/0.101/0.097/0.101/0.119 (parent late 0.157). Worst cells 3-2-55 0.173, 4-2-47 0.167.
- Lead diagnostic: the gate never left its init (g ≈ 0.12 in all prefix bands; a, c unchanged). With lr 5e-5 and ~180
  steps the gate cannot adapt, so the gain is a near-constant 12% blend with the extrapolation. Hypothesis
  'gate rises with prefix' refuted; the extrapolation signal itself helps.
- Next: confirm on 42/2024, then repair the gate (reparameterise so it can learn under the locked optimizer).

## NA-E007 — NaFadeNet + ExtrapGate, Split 42 Confirmation

- VALID, KEEP. MAPE 0.1819 (parent 0.2062), Acc@15 0.450 (0.383), RMSE 52.3 (58.1), bias -25.9, spread 0.48.
- Prefix MAPE 0.236/0.201/0.185/0.162/0.165 — every band better than the parent. Worst cells 1-8-57 0.306, 5-3-37 0.281.

## NA-E008 — NaFadeNet + ExtrapGate, Split 2024 Confirmation; parent promotion

- VALID, KEEP. MAPE 0.1550 (parent 0.1772), Acc@15 0.525, RMSE 30.6, bias -1.1, spread 1.12.
- **3-split: ExtrapGate 0.1474 ± 0.0389, Acc@15 0.554 ± 0.121, RMSE 35.6 ± 14.8**
  vs NaFadeNet 0.1690 ± 0.0420 and CPMLP 0.1778 ± 0.0539. Gain 0.0216 → NaFadeNet+ExtrapGate is the active parent.
- Analyses NA-E007/E008 were drafted by the lead (Codex usage limit, Gemini not logged in); Codex re-check pending.
- Caveat: on split 2021 the gate stayed at init (g ≈ 0.12); the gain is the extrapolation signal blended at a
  near-constant weight. Next (needs reviewer): gate repair and a fixed-g ablation.

## NA-E009 — Gate repair (reparameterised gate, s = 50), Split 2021 Screen

- Head parameterisation only; loss Relative Huber 0.15; optimizer locked. Codex APPROVE.
- VALID, KEEP as candidate: MAPE 0.0999 (NA-E006 0.1053), Acc@15 0.743, RMSE 22.5, bias -15.9, spread 0.75.
- Gate now learns: mean g 0.158/0.190/0.194/0.199/0.203 by prefix band (NA-E006 flat 0.12); a 2.134, c -3.652.
  Prefix MAPE 0.122/0.094/0.093/0.093/0.108 (1-10 slightly worse, all later bands better).

## NA-E010 — Gate repair, Split 42 Confirmation

- VALID, KEEP. MAPE 0.1170 (parent ExtrapGate 0.1819), Acc@15 0.700 (0.450), RMSE 41.3 (52.3), bias -21.7, spread 0.67.
- Prefix MAPE 0.191/0.152/0.127/0.091/0.081 — strong late-prefix gain from the adaptive gate.

## NA-E011 — Gate repair, Split 2024 Confirmation; parent promotion; Phase-1 target met

- VALID, KEEP. MAPE 0.1091 (parent 0.1550), Acc@15 0.743, RMSE 24.3, bias +0.5, spread 1.11.
- **3-split NaFadeNet+ExtrapGate(s=50): MAPE 0.1087 ± 0.0085, Acc@15 0.729 ± 0.025, RMSE 29.4 ± 10.3**
  vs ExtrapGate 0.1474 ± 0.0389, NaFadeNet 0.1690 ± 0.0420, CPMLP 0.1778 ± 0.0539. Promoted (gain 0.0387).
- program.md §1 Phase-1 target (MAPE < 0.20, Acc@15 > 0.68, 3 splits, same checkpoint) is met.
- Owed (architecture.md §5): matched 3-split ablations use_anchor=false, use_fade=false, fixed g = 0.12.

## NA-E012..E020 — Attribution ablations of the parent (architecture.md §5), lead summary

| Family | split 2021 / 42 / 2024 | 3-split MAPE | vs parent 0.1087 ± 0.0085 |
|---|---|---|---|
| − anchor (h − h_1) | 0.0854 / 0.1213 / 0.1154 | 0.1074 ± 0.0193 | −0.0013 → not credited (Codex confirmed) |
| − fade inputs (r_d, r_c) | 0.0835 / 0.1897 / 0.1156 | 0.1296 ± 0.0545 | +0.0209 → credited (independent reviewer confirmed) |
| fixed gate g = 0.12 | 0.1046 / 0.1805 / 0.1541 | 0.1464 ± 0.0385 | +0.0377 → credited (independent reviewer confirmed) |

- The adaptive extrapolation gate is the largest contributor, then the explicit capacity-ratio inputs; the cycle-1 anchor adds nothing measurable.
- Reviewer caveat (fade family): the +0.0209 credit rests mainly on split 42 (+0.0727; split 2021 −0.0164). Gate credit holds on all three splits.
- Phase 1 CLOSED (target met) on user go 2026-11-08; frozen model = NA-E009 config. Phase 2 ACTIVE.

## NA-E021 — Phase 2: P2 voltage-sag what-if (rho 0.5), Split 2021

- VALID, DISCARD (pre-registered rule). MAPE 0.0819 (parent 0.0999), Acc@15 0.863, RMSE 19.2, balance PASS (median R 0.153);
  P2 held-out violation 0.48 vs 0.836 (-42.6%, needs -50%).
- Independent reviewer: physics_anchor_gradient='both' (no sg on f(x)) and independent dropout masks for f(x) / f(x_tilde)
  — the term was mostly dropout-consistency noise (hinge 0.04-0.06 already at beta = 0). Gain not creditable to physics.
- Repair: `physics_dropout_matched` (dropout-free what-if branch, anchor stop) -> NA-E022.

## NA-E022 — Compliant P2 repair (stop anchor, dropout-matched), Split 2021 — direction CLOSED

- VALID, DISCARD. MAPE 0.1760, Acc@15 0.265, best epoch 1 (val MAPE rises every epoch); lambda 0.118 (E021 0.057); R median 1.88.
- Independent reviewer: beta_max = 0.0037 (normalised V) moves the parent by +0.0018 cycles (max 0.005); even the EOL-extrapolated
  P90 range (0.0085) moves it 0.004 cycles. The parent does not use the voltage level, so the parent's "0.836 violation" is a sign
  test on near-zero changes (0 violations above 1e-3 cycles). With sg[f(x)] the term becomes a constant "predict shorter" push
  (cos with main-loss gradient -0.996), producing the negative bias. In E021 the 'both' anchor cancelled this push.
- P2 closed (program.md §7: no physics-creditable signal after the first test + one repair).

## NA-E023 — Two-sided prefix-drift (dropout-free, rho 0.5), Split 2021

- VALID, DISCARD. MAPE 0.1019 (parent 0.0999), Acc@15 0.705, balance PASS (R 0.35; cos +0.38 -> -0.94).
- Post-hoc drift 0.0226 vs 0.0264 (-14%, needs -50%). Bands: 1-10 MAPE 0.122 -> 0.095, drift -63%; 51-75 +0.014; 76-100 +0.010, drift +60%.
- Reviewer: the full-prefix view receives drift gradient it should not get -> next: registered one-way form (sg on the full prefix), NA-E024.

## NA-E024 — One-way prefix-drift (sg on full prefix), Split 2021

- VALID, DISCARD. MAPE 0.1157 (> 0.1049), drift -10.5%, 76-100 MAPE +0.027; balance PASS (R 0.51; cos +0.75 -> -0.97). Codex analysis.
- Codex recommendation: close Phase 2. Capacity rules already satisfied by the parent, P2 irrelevant (voltage-insensitive),
  both drift forms fail; finding = the relevant physics is already encoded by the architecture (adaptive fade-extrapolation gate).

## NA-E024 — One-way prefix-drift (sg on full prefix), Split 2021; Phase 2 CLOSED

- VALID, DISCARD. MAPE 0.1157 (> 0.1049), drift -10.5% (needs -50%), 76-100 MAPE 0.1351 (+0.027). Balance PASS (R 0.51; cos +0.75 -> -0.97).
- Codex: close Phase 2. Capacity rules are already satisfied by the architecture, P2 is irrelevant (voltage-insensitive model),
  both drift forms failed; rho 0.25 would weaken an already insufficient term; §4.5 asymmetry points the wrong way for the
  parent's under-prediction.
- **Phase 2 CLOSED (no gain).** Final v5 model = NA-E009 config: 3-split val MAPE 0.1087 ± 0.0085, Acc@15 0.729 ± 0.025.
  Thesis finding: physics constraints are encoded architecturally (extrapolation gate), so what-if physics losses are redundant here.
- Next: stop and wait for the user (Phase 3 TinyML / one-time test).

## NA-E025 — Phase 2 (reopened by the user): degradation-law physics loss, rho 0.5, Split 2021

- VALID, DISCARD (physics gate) but promising. MAPE 0.1017 (parent 0.0999), Acc@15 0.792 (0.743), RMSE 22.1, bias -13.1, 76-100 0.1007 (0.1085).
- Gradient diagnostics (contribution 2): lambda 0.089, median R 0.59, cosine with the main-loss gradient +0.58 -> +0.98 (first
  aligned physics term in v5; P2 and prefix drift were -0.94 to -0.97), active fraction 0.95 -> 0.69.
- Held-out law-fit 0.0444 vs 0.0463 (floor 0.0161): excess -6%, rule -30%. Codex: A2 path C; next = rho 1.0 only.

## NA-E026 — Degradation-law physics loss, rho 1.0, Split 2021

- VALID, DISCARD (physics gate). MAPE 0.0908 (parent 0.0999, gain 0.009), Acc@15 0.887, RMSE 20.4, bias -9.8; bands 0.159/0.084/0.085/0.079/0.086.
- Gradient diagnostics: lambda 0.178, median R 1.16, cosine +0.58 -> +0.98 (aligned).
- Held-out law-fit 0.0385 (parent 0.0463, floor 0.0161): excess -25.8%, rule -30% (needs <= 0.0372) -> missed by 0.0013.
- Codex: close the direction; confirming on 42/2024 after seeing the near miss would move the goalposts. Recorded as a useful
  regularisation signal, not a promoted physics candidate. Decision on any protocol amendment is the user's.

- **A5-NA (user, 2026-11-09):** confirm NA-E026 config on 42/2024 (NA-E027/E028); rule fixed before running: 3-split MAPE <= 0.1137, 3-split law-fit excess -30% vs parent, balance PASS on all splits.

## NA-E027 / NA-E028 — A5-NA 3-split confirmation of the degradation-law loss (rho 1.0)

| Model | MAPE 2021/42/2024 | 3-split MAPE | Acc@15 | RMSE |
|---|---|---|---|---|
| CPMLP | 0.1261/0.2337/0.1737 | 0.1778 ± 0.0539 | 0.499 | 41.4 |
| Parent NA-E009 | 0.0999/0.1170/0.1091 | 0.1087 ± 0.0085 | 0.729 | 29.4 |
| + degradation law | 0.0908/0.1121/0.1116 | 0.1049 ± 0.0121 | 0.791 | 28.3 |

- A5-NA: MAPE ceiling pass, balance PASS on all splits, law-fit excess 0.0207 -> 0.0175 (-15.6%, needs -30%) -> DISCARD.
- Codex: the loss acts as a useful accuracy regulariser but does not materially enforce the degradation law; report it as a
  negative physics result and keep the credited physics-informed architecture (fade inputs + adaptive extrapolation gate).

## NA-E029..E031 — A6-NA main-loss ablation (MSE instead of Relative Huber on the frozen model)

- MSE 3-split MAPE 0.1062 ± 0.0292 (0.0727 / 0.1203 / 0.1256), Acc@15 0.765 ± 0.159, RMSE 27.7 vs Relative Huber
  0.1087 ± 0.0085, 0.729 ± 0.025, 29.4 -> TIED (|diff| 0.0025 < 0.008); MSE is 3.4x less stable across splits, bias flips sign.
- Na-ion physics search (loss.md §4.8, Codex APPROVE v2): C-rate, CE what-if, hysteresis what-if, CE-fade coupling rejected
  by pre-checks; registered N1 (fixed b = 1 degradation law, threshold 0.8) and N2 (label-free observed-fade consistency).

## NA-E032..E034 — N1: fixed-b (b = 1) degradation-law physics loss, rho 1.0, 3 splits — KEPT (A7-NA)

| Model | MAPE 2021/42/2024 | 3-split MAPE | Acc@15 | RMSE | fixed-b law-fit (val) |
|---|---|---|---|---|---|
| CPMLP | 0.1261/0.2337/0.1737 | 0.1778 ± 0.0539 | 0.499 | 41.4 | — |
| NA-E009 (no physics loss) | 0.0999/0.1170/0.1091 | 0.1087 ± 0.0085 | 0.729 | 29.4 | 0.0539 |
| **+ N1 physics loss** | 0.0767/0.1076/0.1085 | **0.0976 ± 0.0181** | **0.793** | **27.0** | **0.0445** |

- Gradient diagnostics: lambda 0.48 / 0.24 / 0.34, median R 1.55 / 0.81 / 0.61 (balance PASS all); cosine positive on 2021/2024,
  negative mid-training on split 42 (-0.64..-0.80).
- Pre-registered gate (law-fit -20%) missed (-17.4%). **A7-NA (user, post-hoc):** any reduction passes -> N1 KEPT; the new
  frozen model is NA-E009 config + N1. To be reported as a post-hoc user decision.

## Model freeze and Phase 3 start (2026-11-09, user go)

- Short-life val cells (EOL <= 160, 5 cells): MAPE N1 0.095, parent 0.095, CPMLP 0.209; EOL-102 cell: N1 0.088 vs CPMLP 0.368.
- **Frozen v5 model: NaFadeNet + ExtrapGate(s=50) + N1 (NA-E032/E033/E034), 3-split val MAPE 0.0976 ± 0.0181, Acc@15 0.793.**
  Reference without physics loss: NA-E009/E010/E011 (0.1087). No further selection on val.
- Phase 3 (TinyML) active; test stays locked until all Phase-3 variants are fixed, then one evaluation.
