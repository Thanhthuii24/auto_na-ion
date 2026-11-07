# Architecture Design Guide

Use this guide when proposing model architecture changes.
Workflow policies live in `program.md`. Baseline protocol lives in `protocol.md`.

## 1. Input/Output Specification
- **Input Tensors:**
  - Cycle Curves: `[B, 100, 3, 300]` (Channels: Voltage, Current, Capacity).
  - Observed Mask: `[B, 100]` (Binary: 1 for observed cycle, 0 for padded future cycle).
- **Output:** Scalar standardized EOL cycle prediction `[B, 1]`.
- **Deployment (later):** the final model must run alone on an STM32 (TinyML) without LLM connectivity.
  - This is handled in **Phase 3** (quantization, pruning, distillation).
  - Phase 1 searches for the **most accurate** model and does not limit size.

## 2. Inductive Bias & Component Roles
Every architecture must strictly decompose into these sequential interfaces:

```text
Intra-cycle Curve [B, 3, 300]
  -> [Curve Encoder] -> Cycle Features [B, 100, d_model]
  -> [Degradation Encoder with Mask] -> Aggregated Health State [B, d_state]
  -> [Prediction Head] -> EOL [B, 1]
```

### 2.1 Curve Encoder (Intra-cycle: 300 points)
Transforms raw $V$, $I$, $Q$ curve points of a single cycle into a compact electrochemical feature vector:

- **MLP Baseline:** Flatten points $\rightarrow$ Linear layers (simple, ignores local curve topology).
- **1D CNN / Multi-scale Conv1D:** Extracts local inflection points (e.g., $\frac{dQ}{dV}$ peaks and phase transitions).
- **Depthwise Separable Conv1D:** MobileNet-style channel-separable 1D conv (70–80% fewer FLOPs). Useful for Phase-3 compression; in Phase 1 use it only for an accuracy reason.
- **Residual Curve Block:** Conv1D with skip connections for stable deep gradient flow.

### 2.2 Degradation Encoder (Inter-cycle: 100 cycles)
Models battery aging dynamics across observed cycles while respecting the padding mask:

* **Anchor / Reference Comparator:** Compares every cycle $t$ directly to pristine Cycle 1 ($h'_t = [h_t, h_t - h_1]$). Fundamental battery inductive bias: aging is relative fade from initial state.
* **Dilated Causal TCN (Temporal Conv):** Dilated 1D causal convolutions over time. Receptive field spans 100 cycles without recurrency. Hardware-friendly, low latency on MCUs via CMSIS-NN/TFLite Micro.
* **Cycle Attention Pooling (Additive $\mathcal{O}(T)$):** Computes scalar relevance weights per observed cycle $\alpha_t = \text{Softmax}(w^T h_t)$ and pools via $\sum \alpha_t h_t$. Lightweight linear complexity; replaces heavy $\mathcal{O}(T^2)$ Multi-Head Self-Attention.
* **Dual-Branch Architecture:**
  * **Static Branch:** Encodes Cycle 1 for factory capacity / initial quality.
  * **Dynamic Branch:** Encodes fade trajectory over observed cycles.
  * **Fusion:** Concatenates both representations before the prediction head.
* **Masked Summary Pooling:** Parameter-free or lightweight pooling (mean, last observed, difference of last minus first).
* **Gated Recurrent Unit (GRU):** Compact sequential tracking; state must be extracted at the last observed index.
* **Masked Self-Attention / SSM:** Standard Transformer or State-Space blocks; allowed only if parameter/memory budget permits.

### 2.3 Prediction Head
Projects the aggregated degradation state [B, d_state] to scalar standardized EOL. Keep as a lightweight 1–2 layer MLP.

## 3. Evidence-to-Architecture Decision Table
Select interventions strictly based on observed diagnostic symptoms in `analysis.json`:

| Measured Diagnostic Symptom | Probable Bottleneck | Priority Architectural Intervention |
| :--- | :--- | :--- |
| Poor short-prefix MAPE (1–10 cycles) | Initial state ambiguity | Anchor difference against Cycle 1 ($h_t - h_1$) or Dual-Branch |
| Poor long-prefix MAPE (>50 cycles) | Missing temporal trend | Dilated Causal TCN or Last-minus-First trend pooling |
| Large error jumps across adjacent prefixes | Absolute position artifact | Replace static cycle-flatten with Causal TCN or Masked Pooling |
| High within-cycle feature loss | Curve inflection loss | Depthwise Separable Conv1D over 300-point curve |
| High RAM / FLOPs on microcontroller | TinyML hardware limits | **Phase 3 only** (not a Phase-1 reason): distillation, depthwise conv, $\mathcal{O}(T)$ pooling |
| Deep gradient vanishing | Optimization failure | Pre-LN or scaled residual skip paths |
| Flat/dead deep layers (>50% dead neurons) | Redundant capacity | Reduce layer depth, slim channel width, or share weights |
| Cross-domain calibration shift | Cell variance bias | Dual-Branch (isolate initial cell state from fade rate) |

## 4. Strict Mask Contract (Padding Safety)
Padded future cycles ($\text{mask} == 0$) MUST NEVER contaminate the observed state representations:

* **Masked Pooling:** Only sum/average over cycles where $\text{mask} == 1$.
* **Causal TCN:** Use causal padding so past cycles never attend to future padded cycles.
* **Attention:** Mask out padded keys/values with large negative values ($-\infty$) before softmax.
* **Recurrent / SSM:** State must be sampled from index `last_observed_idx = mask.sum() - 1`, never fixed index 99.
* **Normalization:** LayerNorm / BatchNorm must NOT compute statistics over zero-padded future cycle tensors.
* **Linear Flattening:** Forbidden to pass $[B, 100 \times d]$ into a fixed Linear layer that assigns static weights to unobserved cycle slots.

## 5. Valid Staged Mixing vs. Franken-Models
Test exactly one primary mechanism per experiment. When combining components, stage them sequentially:
* **Valid Staging:**
  * Run A: Test Causal TCN Degradation Encoder (keep baseline cycle MLP fixed).
  * Run B (if A is kept): Retain Causal TCN; replace cycle MLP with Depthwise Separable Conv1D.
  * Run C (Ablation): Verify that both mechanisms contribute independently.
* **Invalid Mix (Forbidden):**
  * Proposing 1D CNN + Multi-Head Attention + Dual-Branch + New Head in a single run.

## 6. Proposal Architecture Addendum
When proposing an architecture change, the proposal MUST specify:

* **Responsible stage:** Curve Encoder, Degradation Encoder, or Head.
* **Chosen candidate mechanism** and why the diagnostic evidence supports it.
* **Tensor input/output shapes** across all component interfaces.
* **Plain-text or pseudocode description** of the padding mask path.
* **Expected accuracy evidence:** MAPE, RMSE, Acc@15 and DB-MAPE versus the matched reference.

## 7. BatteryLife paper model families (reference set)

The paper's baselines (`BatteryLife/models/`) map onto the three blocks:

| Paper model | Curve encoder | Degradation encoder | Note |
|---|---|---|---|
| CPMLP | per-cycle MLP | flatten + MLP | paper's best MAPE; local baseline R4-E000..E002 |
| CPTransformer | per-cycle MLP | Transformer over cycles | paper 2nd best |
| CPGRU / CPBiGRU / CPLSTM / CPBiLSTM | per-cycle MLP | (Bi)GRU / (Bi)LSTM | paper 0.189–0.196 |
| iTransformer, PatchTST, Autoformer, DLinear | time-series models on flattened curves | — | weaker in the paper |
| CNN, MICN | convolution | convolution | weaker in the paper |

Local reproductions follow `program.md §2b`. New designs should be compared
with the best **local** reproduction, not with the paper numbers.
