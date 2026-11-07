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
