# BatteryLife-Compatible Benchmark Protocol — v5_auto (NA-ion)

Status: **LOCKED FOR v5_auto ROUND 1** (created 2026-11-07, user-approved project).

## v5_auto dataset override (takes precedence over every MIX_large / Li-ion statement below)

- Task: BatteryLife **Na-ion** (`NA-ion` processed pickles, `Life labels/NA-ion_labels.json`).
- Splits: the three paper splits from `data_split_recorder.py` (seeds 2021 / 42 / 2024),
  selected by env `BL_SPLIT`; one locked cache per split (`protocol_lock.split_caches`).
- Preprocessing: identical to v4_auto / BatteryLife `data_loader.py` (100 early cycles,
  300 points per cycle, cells with EOL missing or ≤ 100 excluded).
- Training lock (paper `Selected_hyperparameters.md`, effective batch = 2 GPUs × 64):
  Adam lr 5e-05, batch 128. Paper CPMLP: d_model 64, d_ff 32, e_layers 1, d_layers 9, dropout 0.
- Reporting: every result is the mean ± std over the 3 splits (as in the paper).
  Screens may use split 2021 alone, but KEEP needs all 3 splits.
- Experiment IDs: `NA-E###`. Test split: locked until the user opens it.
- The Li-ion sections below (domains, MIX_large counts, ISU-ILCC, A-amendments that
  name R4 runs) are workflow reference only.

---

# BatteryLife-Compatible Benchmark Protocol

Status: **LOCKED FOR ROUND 4** (with user-approved amendment A1, 2026-11-02)

**Amendment A1** (user-approved, 2026-11-02). It does **not** change data,
split, preprocessing, labels, loss or the checkpoint rule:
- RMSE becomes a co-primary decision metric.
- Domain-balanced MAPE (DB-MAPE) is added as a guardrail.
- The BatteryLife paper baselines become the reference set (section
  "BatteryLife Paper Baselines").
- Phase 1 searches for the **most accurate** model; size is not a Phase-1 goal.

This protocol reproduces the Li-ion `MIX_large` task implemented in the local
`BatteryLife/` repository. The local loader, recorded split lists, CPMLP model,
training tutorial, and selected-hyperparameter table are the reference sources.
The audited cache is reused. No Round 3 experiment result is admissible as a
Round 4 baseline or candidate result.

## Reference Sources and Precedence

Use these local sources in order:

1. `BatteryLife/data_provider/data_loader.py` for sample construction and
   preprocessing.
2. `BatteryLife/data_provider/data_split_recorder.py` for exact cell membership.
3. `BatteryLife/models/CPMLP.py` for the reproduced baseline architecture.
4. `BatteryLife/assets/Selected_hyperparameters.md` and
   `BatteryLife/assets/Model_training.md` for Li-ion training settings and seeds.
5. `BatteryLife/run_main.py` and `BatteryLife/utils/tools.py` for optimization,
   checkpoint, and metric behavior.

If reference files disagree, record the discrepancy before implementation. Do
not silently combine incompatible variants.

## Dataset and Split

- Benchmark dataset: Li-ion `MIX_large` only.
- Use the exact `MIX_large_train_files`, `MIX_large_val_files`, and
  `MIX_large_test_files` lists from `data_split_recorder.py`.
- Expected list sizes before eligibility filtering are 515 train, 165 validation,
  and 163 test cells.
- Do not regenerate a random 60/20/20 split.
- The lists combine the Li-ion domains used by the local BatteryLife benchmark,
  including UL-PUR, RWTH, HUST, MATR, Stanford, Tongji, ISU-ILCC, CALCE, HNEI,
  SNL, MICH/MICH_EXP, and XJTU as present in the recorded lists.
- Do not include the separate Zn-ion, Na-ion, or CALB benchmark tasks.
- Verify that every listed file resolves to exactly one processed cell and that
  no physical cell or duplicate crosses partitions.

The split is fixed. Training seed changes model initialization, stochastic
training, and loader order; it does not create a new Li-ion split.

## Labels and Cell Eligibility

- Load EOL only from the official JSON life-label files using the same prefix
  and filename alias rules as the BatteryLife loader.
- Do not substitute recorded-cycle count when an official label is missing.
- Skip a cell when its official EOL label is missing.
- Skip a cell when `EOL <= early_cycle_threshold`.
- Set `early_cycle_threshold = 100`.
- Generate prefix samples by the exact rule in **Prefix Samples and Masks**.
- Record every skipped or failed cell with a reason and source domain.

## Cycle Preprocessing

For each accepted cell, reproduce `Dataset_original` behavior:

- Read current, voltage, charge capacity, discharge capacity, and time.
- Replace rows containing negative charge or discharge capacity with missing
  values and backward-fill as in the reference loader.
- Use nominal capacity from the processed cell except for the explicit RWTH and
  SNL filename overrides implemented by BatteryLife.
- Identify charge/discharge regions using the reference `±0.01 C` current rules
  and preserve its dataset-specific charge-first/discharge-first handling.
- Resample charge and discharge separately to 150 points each with the reference
  interpolation rule, then concatenate charge followed by discharge.
- Normalize voltage by the maximum concatenated voltage of that cycle.
- Normalize current and capacity by nominal capacity.
- Store each cycle in channel-first form `[3, 300]` with channels `[V, I, Q]`.
- Zero-fill unavailable cycle slots to produce one cell tensor `[100, 3, 300]`.
- Reject non-finite processed values; do not silently repair them after caching.

Implementation must match the local reference even where its interpolation
endpoint is unconventional. Any correction defines a different benchmark round.

### Amendment A6 — SOC-window-normalized capacity input (user-approved 2026-11-03)

- Reason: the EOL label uses `Qd / (nominal × SOC-window width) ≤ 0.8`, but the
  v1 capacity channel divides by nominal only. For cells with a narrow SOC window
  (139/503 train cells, width down to 0.16) the model cannot see the label threshold.
  Train-cell check: Spearman(physics EOL estimate, EOL) on ISU-ILCC is −0.65 with
  the v1 channel and +0.49 with the SOC-normalized channel.
- Rule: optional input `input_normalization = "soc_window"` uses the derived cache
  `.cache/batterylife_mix_large_v2_socnorm.pt`, where channel Q is divided by the
  cell's SOC-window width (metadata `SOC_interval`). Voltage and current channels,
  splits, labels, label scaler, prefixes and metrics are unchanged.
- The v2 cache is derived from the hash-verified v1 cache and is hash-locked in
  `protocol_lock.json` (`cache_v2_sha256`). The v1 cache and all earlier results
  stay valid. Runs on v2 are compared with v2 baselines (E025 re-trained on v2).
- Test cells are transformed with the same rule; final-test access stays locked.

### Amendment A7 — Test-condition metadata input (user-approved 2026-11-03)

- Reason: on ISU-ILCC train cells, Spearman(log EOL, SOC-window width) = −0.87 and four
  test settings explain R² = 0.79 of log EOL (Wöhler-type life ∝ DoD^−k). The v2 input removes
  the window width that the v1 input carried implicitly.
- Rule: optional `input_normalization = "soc_window_meta"` uses `.cache/batterylife_mix_large_v3_meta.pt`,
  derived from the hash-verified v2 cache. Each tensor gets a 4th channel, zero except
  `tensor[0, 3, 0:4] = [log SOC width, SOC start, log1p charge C-rate, log1p discharge C-rate]`
  (missing rate → 0), read from BatteryLife metadata (experimental settings known before prediction).
- Model `MetaConditionedAnchorTCN` = E025 + a zero-initialized linear map (480 parameters) that
  adds the 4 values to the cycle summary; at initialization it equals E025.
- Labels, splits, scaler, prefixes and metrics are unchanged; final-test access stays locked.

### Amendment A8 — Cosine learning-rate decay (user-approved 2026-11-05)

- Reason: with a constant learning rate of 5e-5, validation MAPE moves by 0.01–0.025 between
  neighbouring epochs. Examples:
  - R4-E055 reads 0.191 → 0.172 → 0.197;
  - R4-E056 reads 0.190 → 0.178 → 0.189.
  - So the selected epoch is a noisy dip, not a settled optimum.
- Rule: Phase-2 runs may set `scheduler = "cosine"`.
  - The rate starts at the protocol value 5e-5 and falls to `min_lr = 1e-6` with
    `CosineAnnealingLR` over `max_epochs × steps_per_epoch` optimizer steps.
  - Adam, batch 32, main loss, epochs and patience are unchanged.
- Fairness: any baseline compared with a cosine-schedule model in the report must be re-run
  with the same schedule. A constant-rate model is never compared with a cosine-rate model as if
  the setups matched.
- Labels, splits, caches, prefixes and metrics are unchanged; final-test access stays locked.

### Amendment A9 — MAPE is the deciding metric (user-approved 2026-11-05)

- Reason: the user set val MAPE as the priority. R4-E076 lowered MAPE 0.1784 → 0.1693 but
  failed only the RMSE guard (382.7 > 346.5 + 21).
- Rule: from R4-E076 on, KEEP/DISCARD is decided on val MAPE alone (candidate MAPE < reference MAPE).
  - RMSE, Acc@15, DB-MAPE and EOL-group MAPE stay mandatory in every report, but they no longer
    gate the decision.
  - This replaces the RMSE co-primary gate of A1 for decisions; A1 reporting duties are unchanged.
- Gradient-health INVALID rule, fairness rules (A8), labels, splits, caches and metrics are
  unchanged; final-test access stays locked.

### Amendment A10 — Phase 2 may extend the frozen trunk (user-approved 2026-11-05)

- Reason: the user asked for a new architecture after loss-only Phase-2 runs plateaued.
  - R4-E072..R4-E080 extended the E025 trunk: fingerprint stream, metadata, meta dropout, charge
    features and head variants.
  - Codex review R4-UNDERFIT §3 flagged that a verbal request does not amend the freeze in this section.
- Rule: in Phase 2, the E025 trunk may be extended.
  - Each extension changes one declared component against a matched reference run, is registered
    in its proposal, and is decided by A9.
  - Physics-loss changes still follow loss.md (registered before use).
  - R4-E072 onward are admissible as main-model candidates.
- Fairness: every extension trains from scratch with the same data, compute box and checkpoint rule.
  No fine-tuning, ensembling or teacher/distillation.
- Labels, splits, caches and metrics are unchanged; final-test access stays locked.

### Amendment A11 — Warmup + cosine learning-rate schedule (user-approved 2026-11-05)

- Reason (Claude/Codex review R4-UNDERFIT):
  - Long-life train cells are fitted poorly (train EOL > 2000 MAPE about 0.38).
  - Relative Huber gives them a gradient of about delta·scale/EOL, roughly 10x weaker than for short-life cells.
  - Note: BatteryLife's selected LR for Li-ion CPMLP is 5e-5. The higher peak is a diagnostic,
    not a paper setting.
- Rule: `scheduler = "warmup_cosine"`.
  - Adam starts at `learning_rate = 5e-5` and rises linearly over `warmup_epochs = 1` epoch to
    `peak_learning_rate = 5e-4`.
  - It then decays by cosine to `min_lr = 1e-6` at `max_epochs × steps_per_epoch`.
  - Batch, loss, epochs and checkpoint rule are unchanged.
- Test design: one matched run that changes only the schedule on the current best recipe, and a
  CPMLP run with the identical schedule, which A8's fairness rule requires for reporting.
- Mandatory extra diagnostics:
  - train-tail metrics (prefix-weighted and cell-balanced train MAPE, EOL > 2000 MAPE and bias,
    maximum prediction);
  - non-finite loss or an UNHEALTHY gradient makes the run INVALID.

## Prefix Samples and Masks

Prefix settings:

- Use `seq_len = 1`.
- `seq_len = 1` means prefix generation starts at one cycle.
- It does not mean that every sample observes only one cycle.

Prefix generation:

- Check prefix lengths `i = 1, ..., 100`.
- Keep prefix `i` only when the cell has at least `i` recorded cycles.
- Keep prefix `i` only when `i < EOL`.
- The largest valid prefix is
  `min(100, observed_cycles, ceil(EOL) - 1)`.
- This matches the current implementation exactly.

Each sample contains:

- cycle tensor: the cell-level [100, 3, 300] tensor
- observed mask: length 100
- first i mask entries: 1
- remaining mask entries: 0
- target: the cell's official total EOL
- cell ID, domain ID, prefix length, split

Before model input:

- Multiply the cycle tensor by the observed mask.
- Set every cycle after prefix `i` to zero.
- Keep the mask available to the model and diagnostics.
- Split cells before creating prefixes.
- Never place prefixes from one cell in different data splits.

## Label Scaling and Loading

- Fit `StandardScaler` on the unique training-cell EOL labels, matching the
  BatteryLife loader rather than weighting a cell by its number of prefixes.
- Apply that fitted scaler to repeated train-prefix labels, validation labels,
  and final-test labels.
- Train loader: shuffled, with deterministic experiment-seeded ordering and the
  declared drop-last behavior.
- Validation loader: not shuffled and not drop-last.
- Ordinary training APIs must return train and validation data only.
- Final-test data is exposed only by a separate locked evaluation entry point.

The dedicated loader generator is a reproducibility strengthening of the local
implementation; it does not alter samples, splits, labels, or metrics.

## Round 4 CPMLP Baseline

Use the local `CPMLP.py` architecture and Li-ion settings from
`Selected_hyperparameters.md`, except for the explicitly approved Relative
Huber loss and checkpoint rule:

```text
dataset = MIX_large
early_cycle_threshold = 100
seq_len = 1
charge_discharge_length = 300
d_model = 32
d_ff = 256
e_layers = 12
d_layers = 7
dropout = 0
learning_rate = 5e-5
loss = Relative Huber on unscaled relative EOL error
relative_huber_delta = 0.15
optimizer = Adam
maximum train epochs in Round 4 = 15
patience = 5
minimum epochs before patience can stop = 5
training seeds = 2021, 42, 2024
```

The table records batch size 16 per process for the original two-process setup,
equivalent to effective batch size 32. A local single-process reproduction must
use effective batch size 32 or explicitly document and validate another matched
configuration.

Seed 2021 is used for architecture screening. Seeds 42 and 2024 are added only
for confirmation of leading candidates and of the two best reproduced paper
baselines (amendment A1).

The Phase-1 comparison configuration is fixed to Relative Huber with delta
`0.15`, Adam, learning rate `5e-5`, batch size 32, standardized model output,
and zero physics-loss weight. The loss is computed after differentiably
converting the model output back to EOL cycles. Main-loss and physics changes
are not allowed during Phase 1.

- Phase 1 — Model: change architecture only.
- Phase 2 — Loss: freeze architecture and test one declared loss addition at a
  time, including physics loss.
- Phase 3 — TinyML: freeze architecture and loss before compression.

## Metrics and Selection

- Primary metric: sample-level MAPE on unscaled EOL.
- Co-primary metric (A1): sample-level RMSE in cycles. MAPE is dominated by
  short-life cells, RMSE by long-life cells; both must be judged.
- Guardrail metric (A1): **DB-MAPE**. It is the unweighted mean of per-domain
  MAPE over domains with at least 10 validation cells, taken from the same
  selected checkpoint. It shows whether the model is good on every battery
  type, not just the large ones. Computed by `tools/rescore.py` from
  `record.json`.
- Reported metric (amendment A4): **seen / unseen aging-condition MAPE and
  Acc@15 on validation**. A validation cell is *seen* when its BatteryLife
  aging condition (`name2agingConditionID.json`) occurs among the MIX_large
  training cells. This is BatteryLife's own seen/unseen split, applied to
  validation only (the test stays locked). Computed by `tools/rescore.py`.
- Secondary metric: sample-level 15%-accuracy, the fraction with absolute
  relative error `<= 0.15`. Store it canonically on `[0, 1]` for compatibility
  with the paper table; convert the local evaluator's percentage output by
  dividing by 100 and label any percentage presentation explicitly.
- Also report MAE for compatibility, plus per-domain and per-prefix
  diagnostics with sample and unique-cell counts.
- Let `m*` be the lowest validation MAPE reached by a run.
- Eligible checkpoints satisfy `MAPE <= m* + 0.005`.
- Select the eligible checkpoint with the highest Acc@15.
- Break an Acc@15 tie by lower MAPE, then by the earlier epoch.
- Reload one selected checkpoint and report every metric from it. Never combine
  MAPE from one checkpoint with Acc@15 from another.
- Report mean and standard deviation across seeds 2021, 42, and 2024 for the
  confirmed configuration.
- A single-seed result is screening evidence only.
- For every domain, report sample count and unique-cell count. A subgroup with
  fewer than 10 validation cells, including MICH_EXP, cannot alone justify a
  model change or promotion.
- For domains with at least 10 validation cells, flag a guardrail regression
  when MAPE worsens by more than `0.03` or Acc@15 falls by more than `0.05`
  against the matched reference. Promotion requires an explicit review of each
  flagged regression; headline metrics alone are insufficient.
- Architecture screening requires at least 5 complete epochs. A promoted
  candidate may train for at most 15 epochs with patience 5, applied only after
  the fifth complete epoch. Model selection must not be based on a universal
  300-second cutoff.
- A proposal may declare a safety compute limit. It is checked only after a
  completed epoch once the minimum-epoch requirement is met. A safety-limited
  run that has not completed 5 epochs is `inconclusive` and cannot be used as
  negative evidence about the architecture.
- Training compute counts forward, loss, backward, optimizer step, and in-step
  diagnostic hooks. Loading data/cache, waiting for a batch, device transfer,
  initialization, validation, and file I/O are excluded. Record excluded wall
  time separately.
- Every row in `results.tsv` must include:
  - `runner_version`: a short label for the training runner code version
    (e.g., `train.py` git hash or date stamp). Change this when the training
    loop, data loading, or timing logic changes.
  - `steps_per_sec`: optimizer steps divided by training compute seconds.
    Use this to detect unfair compute comparisons between experiments.

## BatteryLife Paper Baselines (reference set, A1)

The BatteryLife benchmark table (`BatteryLife/README.md`, Li-ion, 3 seeds) is
the literature reference:

| Group | Model | MAPE | 15%-Acc |
|---|---|---:|---:|
| — | Dummy | 0.831 | 0.296 |
| MLP | DLinear | 0.586 | 0.275 |
| MLP | MLP | 0.233 | 0.503 |
| MLP | **CPMLP** | **0.179** | **0.620** |
| Transformer | PatchTST | 0.288 | 0.430 |
| Transformer | Autoformer | 0.437 | 0.287 |
| Transformer | iTransformer | 0.209 | 0.516 |
| Transformer | CPTransformer | 0.184 | 0.573 |
| CNN | CNN | 0.337 | 0.371 |
| CNN | MICN | 0.249 | 0.494 |
| RNN | CPGRU | 0.189 | 0.585 |
| RNN | CPBiGRU | 0.190 | 0.566 |
| RNN | CPLSTM | 0.196 | 0.585 |
| RNN | CPBiLSTM | 0.191 | 0.421 |

- These numbers used the paper's loss (MSE), two-GPU batch sizes and
  early stopping. They are **literature context**, not local results.
- Local comparisons need **local reproductions** under this protocol
  (Relative Huber, fixed checkpoint rule, same split and cache, locked Phase-1
  optimizer). Architecture hyperparameters follow `Selected_hyperparameters.md`.
  The procedure and priority are in `program.md §2b`.
- The paper also reports **seen/unseen aging-condition** MAPE on the test set.
  The final test remains locked, so this split is not used during discovery.

The paper reports CPMLP Li-ion MAPE `0.179 ± 0.003` and 15%-accuracy
`0.620 ± 0.004`. These are reproduction references, not automatic pass/fail
thresholds, because the local dataset version, hardware, loss, and checkpoint
rule may differ. Round 4 is not an exact optimization reproduction of the
paper.

## Test-Set Compatibility Rule

The local `run_main.py` computes test metrics every epoch, but selection is based
on validation MAPE. This autoresearch round preserves the benchmark task, data,
split, preprocessing, and metrics while enforcing a stricter test lock:

- Do not construct or evaluate the final-test loader during discovery,
  confirmation, or deployment optimization.
- Freeze candidate, configuration, seed policy, and checkpoint-selection rule
  before final-test evaluation.
- Evaluate the final test once and do not tune from its result.

This intentional governance difference prevents adaptive test leakage without
changing the BatteryLife benchmark inputs or validation selection rule.

## Benchmark Lock Gate

Before E000 can run, audit and record:

- Reference-source hashes.
- Data and metric implementation hashes.
- Resolved file inventory and eligibility decisions.
- Exact split membership and cross-split duplicate audit.
- Prefix counts and unique-cell counts for each split and domain.
- Tensor, mask, label-scaler, and metric unit tests.
- Cache path and SHA-256.
- Protocol fingerprint.

- This protocol is `LOCKED` only when all fields above exist in
  `protocol_lock.json`.
- Any later behavior change starts a new benchmark round and a new baseline.
