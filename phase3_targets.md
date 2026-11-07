# Phase 3 (TinyML): hardware targets and budgets (draft, 2026-11-06)

Status: **ACTIVE (unlocked by the user 2026-11-06).** P0 and P1 done for R4-E076 (`phase3/R4-E076/`). Decisions: E076 first, then E079; STM32Cube.AI; no physical board; no NAS; KD only if pruning costs > 0.01 val MAPE; CPMLP not rerun.
- Phase 3 stays locked until the user unlocks it (`program.md` §309).
- Final-test data stays locked.
- Every number marked *(datasheet)* comes from the STM32 MCU datasheets, not from the board manuals. Re-check them before use.

## 1. Boards supplied by the user

| Board (manual) | MCU (variants) | Core / clock | Flash | SRAM | Measurement aid |
|---|---|---|---|---|---|
| MB1312 (UM2179) | STM32L496ZG, L4A6ZG, L4P5ZG, L4R5ZI | Cortex-M4F, 80 MHz (L496) / 120 MHz (L4R5) *(datasheet)* | 1 MB (G) / 2 MB (I) | 320 KB (L496) / 640 KB (L4R5) *(datasheet)* | JP5 IDD jumper |
| MB1364 (UM2407) | STM32H723ZG, H743ZI2, H753ZI | Cortex-M7, 550 MHz (H723) / 480 MHz (H743) *(datasheet)* | 1 MB (G) / 2 MB (I) | 564 KB (H723) / 1 MB (H743) *(datasheet)* | JP4 IDD jumper (MB1364-E01: the Ethernet PHY is not on VDD_MCU) |
| MB1940 (UM3417) | STM32N657X0 | Cortex-M55, 800 MHz + Neural-ART NPU *(datasheet)* | none internal; 512-Mbit Octo-SPI | 4.2 MB | CN12 shunts per rail (VDDCORE 50 mΩ, VDDIO 750 mΩ, …) |

## 2. Model footprint (measured 2026-11-06, CPU, one 100-cycle sample)

| Model | Parameters | Weights fp32 / int8 | MACs per full sample | Largest layer output (all 100 cycles at once) | Input |
|---|---|---|---|---|---|
| R4-E079 (main candidate) | 161,409 | 631 KB / 158 KB | 74.5 M | 243,200 values (950 KB fp32 / 238 KB int8) | 120,000 values (469 KB fp32) |
| R4-E076 | 161,025 | 629 KB / 157 KB | 74.5 M | same | same |
| CPMLP R4-E000 (baseline) | 449,281 | 1,755 KB / 439 KB | 22.8 M | 25,600 values | 90,000 values |

- The largest layer output is the per-cycle phase encoder run on all 100 cycles in one batch.
- Encoding **one cycle at a time** (streaming) needs only about 2.4 K values per cycle (about 10 KB fp32).
- The temporal TCN and summary then work on a [100, 32] state (12.5 KB fp32).
- The 12 fingerprint features (charge and discharge sides) are closed-form statistics that are easy to compute in C.

### 2.1 Where parameters and compute sit (R4-E079)

| Block | Params | Share | Compute note |
|---|---|---|---|
| `inter_mlp` (7 MLP blocks, d_model 32, d_ff 256) | 117,152 | 72.6% | runs once per prediction, so its MACs are small |
| `charge_encoder` + `discharge_encoder` (Conv1d 4→16→24→16, per phase) | 23,200 | 14.4% | runs on every cycle (×100), so it carries **most of the 74.5 M MACs** |
| fingerprint stream (`fp_mlp`, `fp_projection`) | 11,616 | 7.2% | 18 closed-form features plus a tiny MLP |
| TCN, `anchor_projection`, `inter_summary`, meta, head | 9,441 | 5.8% | small |

Pruning `inter_mlp` mainly saves flash. Pruning the curve encoders, or making them stream, mainly saves latency, energy and RAM.

## 3. Compression method catalogue

Reference course: MIT 6.5940 *TinyML and Efficient AI Computing* (Fall 2026, https://hanlab.mit.edu/courses/2026-fall-65940). Lecture (Lec) and lab numbers follow that page; videos are at https://live.efficientml.ai/.

| # | Method | Course | What it means for this model | Priority |
|---|---|---|---|---|
| M1 | **Structured pruning** (channel / neuron / block) | Lec 3–4, Lab 2 | Lower `d_ff` 256 → 64/128 and drop `inter_mlp` blocks (7 → 2–4); prune conv channels of the curve encoders (16/24/16). Rank by layer-wise sensitivity (val MAPE drop per layer at fixed sparsity) and L1/L2 channel norms. Fine-tune after each step. Output is a smaller **dense** model, so it speeds up any MCU toolchain. | **High** |
| M2 | **Unstructured pruning** (fine-grained magnitude, 50–90% sparsity) | Lec 3–4, Lab 2 | Saves storage only with compressed or sparse kernels; on Cortex-M the dense kernels do not speed up. Report as an ablation (sparsity vs MAPE curve), not as the deployed model. | Low |
| M3 | **INT8 post-training quantization** (PTQ) | Lec 5–6, Lab 4 | Per-channel symmetric INT8 weights and per-tensor INT8 activations, calibrated on **train** samples only. Watch-outs: LayerNorm (keep in fp32 or use a toolchain-supported fused op); the 18 fingerprint features contain `log`/`var` (compute in fp32 on the MCU, then quantize the standardised vector); the output is de-standardised in fp32. | **High** |
| M4 | **Quantization-aware training** (QAT) | Lec 5–6, Lab 4 | Fake-quant fine-tuning, used only if PTQ raises val MAPE by more than 0.01. | Conditional |
| M5 | **Knowledge distillation** (KD) | Lec 9 | Teacher = frozen fp32 main model; student = pruned or narrower model; loss = main loss + α·distance to teacher predictions. **Needs user approval**: teacher–student was rejected in Phase 2 for baseline fairness. In Phase 3 it would only compress our own model. | Pending approval |
| M6 | **NAS** (neural architecture search) | Lec 7–8, Lab 3 | Small search space only: d_model {16, 24, 32}, d_ff {32, 64, 128, 256}, MLP depth {1–7}, encoder channels, TCN dilations. Constraints: flash ≤ 256 KB, peak RAM ≤ 256 KB, MACs. Use a random or evolutionary search with short screens, not a full once-for-all supernet. | Medium (optional) |
| M7 | **MCUNet** (TinyNAS + TinyEngine co-design; patch-based inference to cut peak memory) | Lec 10 | The useful idea here is **memory-aware execution**. Our analogue of patch-based inference is **per-cycle streaming**: encode each new cycle once, cache its 32-d state, and run the TCN and head on the cached [≤100, 32] states. This takes peak activation from about 950 KB to about 10–20 KB. Also use the MCUNet idea of searching under the board's SRAM/flash limits (M6). | **High** (streaming) |
| M8 | **TinyEngine** (memory-scheduled code generation, in-place ops; open-source `mit-han-lab/tinyengine`, Cortex-M7) | Lec 11 | Compare one engine against the vendor toolchain on the same INT8 model: STM32Cube.AI (X-CUBE-AI) or TFLite Micro + CMSIS-NN versus TinyEngine. Report latency, peak RAM and flash. Check operator coverage first (Conv1d, LayerNorm, the masked TCN). | Medium |
| M9 | **Operator optimization** | Lec 11 (closest; no dedicated lecture) | Fuse Conv+ReLU and Linear+ReLU; fold the zero-init or identity projections (`anchor_projection` starts as identity plus a difference term); use CMSIS-NN SIMD kernels and im2col for Conv1d; make the TCN incremental (causal, dilations 1/2/4, so only the last receptive field needs recomputing per new cycle); compute fingerprint statistics as running sums. | **High** |

### 3.1 Planned order (each step checked against the previous one on val, one change at a time)
1. **P0, fp32 reference**: export the frozen model and run it on the board as is (or on the host if it does not fit). Record MAPE, flash, RAM and latency.
2. **P1, streaming + operator optimization** (M7, M9): must reproduce fp32 predictions within 1e-4 relative error.
3. **P2, structured pruning** (M1) with fine-tuning; KD (M5) only if approved.
4. **P3, INT8 PTQ** (M3); QAT (M4) only if the MAPE drop exceeds 0.01.
5. **P4, deployment**: X-CUBE-AI or TFLite Micro versus TinyEngine (M8); measure on the board(s).
6. Optional: NAS (M6); unstructured-pruning ablation (M2).

### 3.2 Phase 3 comparison rules (user decision 2026-11-06)
- **CPMLP is NOT put through Phase 3.** The baseline comparison uses the reported CPMLP results: our three-seed reproduction (val) and the BatteryLife paper (reference only).
- Project contributions (user): (1) the physics-informed loss, (2) gradient checking / diagnostics, (3) deployment on edge devices. Phase 3 serves contribution 3.
- Fine-tuning inside Phase 3 is part of compression and starts only from the frozen Phase-2 checkpoint. Data, split, metrics and the main loss stay locked.
- Every step reports fp32 versus compressed MAPE / Acc@15 / RMSE plus flash, peak RAM, latency and energy.

## 4. Proposed budgets (to be confirmed by the user before Phase 3)

| Item | Proposal |
|---|---|
| Primary target | NUCLEO-H743ZI2. It has enough flash and RAM for INT8, and an M7 with a DSP extension (CMSIS-NN / X-CUBE-AI). |
| Accelerated target | NUCLEO-N657X0-Q (NPU), for latency and energy comparison. |
| Stretch target | NUCLEO-L4R5ZI or L496ZG, to show the model runs on the smallest board (needs streaming). |
| Weights | ≤ 256 KB flash (INT8) |
| Peak RAM | ≤ 256 KB (streaming per cycle; input buffer stored per cycle, not as all 100 cycles) |
| Accuracy | INT8 val MAPE ≤ fp32 val MAPE + 0.01, same checkpoint, same cells |
| Quantisation | post-training INT8, per-channel weights, calibrated on train samples only; quantisation-aware training only if the drop exceeds 0.01 |
| Reported | flash, peak RAM, latency per new cycle and per full 100-cycle prediction, energy per inference (IDD jumper / CN12 shunts), MAPE / Acc@15 / RMSE fp32 vs INT8 |
| Baseline | Reported CPMLP results only; CPMLP is not deployed (user, 2026-11-06). |

## 5. Open questions for the user

1. Which of the three boards do you physically have?
2. Toolchain: STM32Cube.AI (X-CUBE-AI) or TFLite Micro + CMSIS-NN?
3. Is distillation (M5) allowed in Phase 3? `program.md` lists it, but teacher-student was rejected in Phase 2.
4. Is NAS (M6) in scope, or only pruning + quantization + streaming?
