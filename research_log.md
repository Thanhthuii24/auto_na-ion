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
