# v5 (Na-ion) — one-time TEST results (2026-11-09)

| Model | MAPE (2021 / 42 / 2024) | MAPE TB | Acc@15 | RMSE | Paper |
|---|---|---|---|---|---|
| Dummy (train mean) | 0.445 / 0.380 / 0.387 | **0.404 ± 0.036** | 0.067 | 64.3 | paper Dummy 0.404 ± 0.029 |
| CPMLP (local, Relative Huber) | 0.308 / 0.254 / 0.235 | **0.266 ± 0.038** | 0.275 | 65.7 | paper CPMLP 0.274 ± 0.026 |
| CPTransformer (paper number only) | — | 0.255 ± 0.036 | 0.406 | — | from the BatteryLife paper |
| NaFadeNet+ExtrapGate(s=50), no physics loss | 0.254 / 0.171 / 0.148 | **0.191 ± 0.055** | 0.394 | 47.8 |  |
| FROZEN: NaFadeNet+ExtrapGate(s=50) + N1 physics loss (fp32) | 0.245 / 0.147 / 0.133 | **0.175 ± 0.061** | 0.515 | 45.3 |  |
| FROZEN N1, INT8 weight-only (ONNX dynamic) | 0.245 / 0.147 / 0.135 | **0.176 ± 0.061** | 0.513 | 45.4 |  |
| FROZEN N1, fp16 weights (fp32 compute) | 0.245 / 0.147 / 0.133 | **0.175 ± 0.061** | 0.515 | 45.3 |  |

- Benchmark matched: Dummy 0.404 and local CPMLP 0.266 reproduce the paper (0.404; 0.274 ± 0.026).
- Frozen model (N1) beats CPMLP by 0.091 test MAPE (-34%) and the paper's best Na-ion model (CPTransformer 0.255).
- Physics loss N1 improves the same architecture on every split (0.191 -> 0.175). INT8 weight-only costs +0.0007.
- Caveats: N1 kept under the post-hoc user amendment A7-NA; on-board STM32 measurement deferred; test split used once.
