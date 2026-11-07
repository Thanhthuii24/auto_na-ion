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
