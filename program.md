# Evidence-Led Research Program — Phase 1 Active

## 1. Targets and authority

- Phase-1 success: one frozen architecture has three-seed mean MAPE `< 0.20`
  and Acc@15 `> 0.68` from the same selected checkpoint for seeds 2021, 42,
  and 2024.
- One-seed evidence is preliminary; it may promote a candidate but cannot close
  Phase 1.
- Rule priority: `protocol.md` > `program.md` > `state.json` >
  `research_guide.md` / `architecture.md` / `loss.md`.
- Standard proposal, diagnostic, run, review, commit, and push attempts are
  authorized without repeated confirmation.
- Destructive reset, protocol/data changes, final-test access, and work outside
  this repository require separate authorization.

## 2. Active scope and locks

- Allowed: architecture changes, local reproductions of the BatteryLife paper
  baselines (§2b), CPMLP controls, and non-learning diagnostics.
- Phase 1 looks for the **most accurate** model. Model size, latency and
  parameter count are **not** Phase-1 goals: no efficiency-first track.
  Making the model small for STM32 happens in Phase 3 (quantization, pruning,
  distillation from the Phase-1/2 model as teacher).
- Locked: Phase-1 Relative Huber setup, optimizer policy, data, split,
  preprocessing, labels, masks, metrics, and checkpoint rule.
- Forbidden: auxiliary/physics loss, another main loss, pruning, quantization,
  distillation, final-test access, or changing a model during its run.
- The checked cache is read-only. `protocol_lock.json` and cache hashes MUST
  pass preflight.
- Only one training run may be active.

## 2b. BatteryLife paper baseline reproductions (amendment A1)

The paper's baselines are reproduced locally under this protocol, so that every
new model can be compared with them on equal terms. Rules:
- Model code comes unchanged from `BatteryLife/models/<Model>.py`, through one
  generic wrapper in `train.py`. The wrapper only permutes `[B,100,3,300]` to
  BatteryLife's `[B,100,300,3]` and squeezes the output.
- **Architecture** hyperparameters come from `assets/Selected_hyperparameters.md`
  (Li-ion row). Models without a row use the BatteryLife `run_main.py` defaults,
  written in the proposal.
- **Optimizer** stays at the Phase-1 lock (Adam, lr 5e-5, batch 32; enforced by
  `workflow_preflight.py`), so the comparison is matched. Reports must say
  that the paper's batch size differs.
- Loss, checkpoint rule, data and metrics are the Phase-1 ones (Relative Huber).
- Each reproduction uses proposal `type = "baseline"` (the value accepted by
  `workflow_preflight.py`) and `model.variant = "batterylife"`. It does **not**
  count as an architecture experiment and does not need a mechanism hypothesis.
- Seed 2021 first. The two best (by MAPE, then RMSE) also run seeds 42 and 2024.

| Priority | Models | Why |
|---|---|---|
| 0 (done) | CPMLP ×3 seeds (R4-E000..E002) | paper's best MAPE |
| 1 | CPTransformer, CPGRU | reference only (amendment A4): the paper's next best transformer and RNN |
| 2 (only if budget remains) | CPBiGRU, CPBiLSTM, CPLSTM, iTransformer, MLP, MICN | completeness |
| 3 (optional) | Dummy, DLinear, CNN, PatchTST, Autoformer | completeness |

After priority 1, the best local model (CPMLP, E006, E020/E025 family, or a
reproduced baseline) becomes `active_architecture_parent` if it wins under §7.

## 2c. Phase-1 focus: generalization to unseen aging conditions (amendment A4)

Evidence: `diagnostics/phase1_breakthrough_diagnosis.md`.
- Seen-condition cells are already near target (MAPE 0.137, Acc@15 0.678).
- Unseen-condition cells hold 62% of the error.
- One partial-SOC-window cell holds 20% of MAPE.
- Eight architectures share the same per-cell errors.

So Phase-1 proposals must target **cross-condition generalization**, not how
cycles are mixed. Ranked directions (each one mechanism; development room per §7):
1. **D1** SOC-window-aware view: add the per-cycle trajectory normalized by the
   cell's own early capacity (matches the label definition), next to the
   absolute view. Matched control: the same extra branch fed with the
   absolute view.
2. **D2** reference-pair (BatLiNet-style) inter-cell learning.
3. **D3** learned-shape degradation head. A fixed linear extrapolation is
   refuted (median APE 0.42).

Every review reports **unseen-condition MAPE and Acc@15**
(`tools/rescore.py`). A promising signal (§7) also includes an unseen-condition
MAPE improvement of at least 0.008 with no seen-condition regression above 0.005.

## 3. Experiment Contract

Before training, every proposal MUST define exactly:

1. One falsifiable question and one primary hypothesis.
2. One changed mechanism and its responsible block.
3. The matched reference and fixed components.
4. Expected metric, subgroup, and diagnostic evidence.
5. One quantitative reject rule.
6. Stage, seed, device, and compute budget.

- Multiple new mechanisms require separate experiments.
- Implementation-fault repairs keep the same hypothesis (at most 2 per
  mechanism, §7).
- Proposal ID MUST equal `state.json.next_experiment_id`.
- `workflow_preflight.py` MUST pass before training.

## 4. Execution state machine

```text
read state + matched evidence
-> proposal
-> workflow_preflight.py
-> train.py
-> analysis.json
-> validate_analysis.py
-> results.tsv
-> research_log.md
-> state.json
-> local Git commit
-> bounded push attempt
-> next contract or STOP
```

- Each step waits for the previous step; speculative run queues are forbidden.
- `state.json` is updated last and is the sole source for current ID, active
  parent, approval, budget, and next action.
- After interruption, reconcile `state.json`, `results.tsv`, experiment folders,
  temporary folders, and the latest research-log entry before continuing.
- Never create a second result row or process for the same experiment ID.

## 5. Runner artifacts and diagnostics

- `train.py` owns hooks and automatically writes `record.json` and
  `diagnostics.json`; the agent MUST NOT reimplement diagnostic logging per run.
- Expensive gradient/update/activation snapshots default to early, middle, and
  late windows, not every fixed number of steps.
- Every valid completed run keeps proposal, code snapshot, log, runner JSON,
  selected checkpoint, and selected-checkpoint validation predictions.
- Validation predictions contain prediction, target, cell, domain, prefix,
  signed/absolute relative error, and Acc@15 margin; reuse them before rerunning
  inference.
- Missing measurements are `NOT_AVAILABLE`, never guessed.

The runner evaluates:

- learning curve and validation MAPE, Acc@15, MAE, RMSE;
- gradients, parameter/update ratios, activations, and prediction spread;
- prefix, domain, and EOL groups with sample and cell counts;
- mask/padding health, parameters, bytes, memory, time, and throughput.

Gradient gate:

- Gradient/parameter ratio MUST remain within `1e-6` to `100` after allowed
  zero-init transients.
- NaN, Inf, disconnected required blocks, collapsed activations, or mask faults
  make the run `INVALID`.
- Late update ratio `< 1e-8` means frozen; the run is `INVALID` when the block is
  required.
- Late near-zero gradient fraction `> 0.50` marks a dead block; its layer action
  MUST be `REPAIR` or `REMOVE`.
- Branch imbalance `> 1000x` is a warning requiring an explained scale or one
  matched repair.

## 6. Training gates

- Smoke: 50–100 optimizer steps on the required accelerator; verify graph,
  shapes, masks, finite values, updates, memory, and speed. It is not quality
  evidence.
- Screen: seed 2021, exactly 5 complete epochs before KEEP/DISCARD eligibility.
- Screen hard floor against the matched reference:
  - checkpoint MAPE regression MUST be `<= 0.10`; and
  - `first_to_last_train_loss_ratio` MUST be `<= 0.70`, unless the same
    checkpoint already meets both Phase-1 targets.
- Hard-floor failure forces `VALID` + `DISCARD`; good gradients or interesting
  subgroups cannot override it.
- Full-run promotion requires floor pass, gradient gate pass, hypothesis-
  consistent evidence, and either useful metric progress or a meaningful
  guardrail-safe subgroup gain with improvement continuing at the boundary.
- **Phase-2 screen rule (user, 2026-11-03):**
  - Every new loss first runs a 5-epoch screen against the 5-epoch reference
    (R4-E025 for the frozen model).
  - Physics losses converge later (best epochs 7–12 seen). So a screen is
    **promoted to full** instead of discarded when MAPE is still improving at
    epoch 5, or when the violation rate falls clearly.
  - It is discarded at screen only for a balance failure, a DEAD/WEAK block,
    or a clearly worse MAPE that is no longer improving.
- Full run extends the identical configuration to at most 15 total epochs,
  never 5 plus 15. Patience 5 activates only after epoch 5.
- Three-seed confirmation is reserved for leading candidates.

## 7. Comparison, parent, and branch rules

- Compare the same seed, data/batches, loss, optimizer, checkpoint rule, and
  fair training opportunity. Never combine metrics from different checkpoints.
- Meaningful gain (amendment A1; thresholds about one seed-SD of the CPMLP
  baseline: MAPE 0.008, RMSE 21 cycles, DB-MAPE 0.007) requires at least one:
  - **A (MAPE):** MAPE improvement `>= 0.008` with RMSE regression `<= 10` cycles;
  - **B (RMSE):** RMSE improvement `>= 21` cycles with MAPE regression `<= 0.003`;
  - **C (Acc@15):** Acc@15 improvement `>= 0.02` with MAPE regression `<= 0.003`
    and RMSE regression `<= 10` cycles.
- Every gain must also pass:
  - **DB-MAPE** regression `<= 0.010`;
  - Acc@15 regression `<= 0.03` for paths A and B;
  - the domain guardrails below.
- Parameter or latency gains are **not** a Phase-1 reason to keep a model.
- **Baselines can never be frozen or used as the Phase-1/2 model** (user rule,
  2026-11-03). This covers CPMLP and every BatteryLife paper reproduction; they
  are references only. The frozen model must be one of this project's own
  architectures, chosen by three-seed results.
- Every review runs `python tools/layer_report.py <ID>` (per-block gradient/param
  ratio early and late, update ratio, near-zero fraction → OK / WEAK / DEAD /
  FROZEN / EXPLODING). Any non-OK block must get a REPAIR or REMOVE action in
  `analysis.json`.
- Metrics come from one selected checkpoint. Compute them with
  `python tools/rescore.py <ID>`; `python tools/rescore.py --write` fills the
  evaluation columns of `results.tsv`: RMSE, MAE, DB-MAPE, seen/unseen MAPE,
  unseen Acc@15, physics violation rate, λ, median balance ratio R, and block health.
- For domains with at least 10 validation cells, flag MAPE regression `> 0.03`
  or Acc@15 regression `> 0.05`. Unexplained flags block promotion.
- `state.json.active_architecture_parent` is the sole current parent pointer;
  operational text MUST NOT hard-code an experiment ID.
- Switch parent only after meaningful gain, healthy diagnostics and passed
  domain guardrails. Compute cost is reported but is not a reason to block a
  more accurate parent (amendment A1); it must fit the budget in `state.json`.
- **Development room (amendment A2, user, 2026-11-02):** a direction (model
  family, block idea, or Phase-2 loss) gets a fair chance to develop.
  - **Promising direction:** at least one of these, versus its matched reference:
    - a meaningful-gain path (A/B/C) is met; or
    - MAPE or RMSE improves by at least half a gain threshold (0.004 / 10 cycles); or
    - a guardrail-safe gain in a prefix band, EOL band, or a domain with
      ≥ 10 cells; or
    - an unseen-condition MAPE gain of at least 0.008 with seen-condition
      regression no more than 0.005 (A4).
  - A promising direction gets **up to 3 evidence-led follow-ups** after its
    first test. Each follow-up changes **one** mechanism and must cite the saved
    diagnostic that motivates it. Width/depth/dropout micro-tuning only counts
    when a diagnostic points to it.
  - A direction **closes** after **4 consecutive DISCARDs**. A KEEP resets its
    counter.
  - A direction with **no promising signal** at its first test gets one
    targeted repair (the old v4 rule), then closes.
  - **Implementation faults** (INVALID runs) do not use the follow-up budget:
    at most 2 implementation repairs per mechanism.
  - Track every direction in `state.json.directions`: name, tests, consecutive
    DISCARDs, promising (yes/no), status open/closed.
  - Closed directions from before A2 stay closed, unless a later diagnostic
    gives a new reason. Reopening one counts as a new direction and needs the
    user's OK.

## 8. Structured review and decision

- New runs MUST write `analysis.json` conforming to `analysis.schema.json`.
- `validate_analysis.py <ID>` is a hard gate before results/state updates and
  commit. Legacy `analysis.md` files remain valid and require no backfill.
- The structured review MUST answer the Experiment Contract, record only
  evidence-backed causes, cite JSON evidence paths, and assign each major block
  exactly one action: `RETAIN`, `REPAIR`, `REPLACE`, or `REMOVE`.
- If evidence cannot prioritize a cause, choose a capped diagnostic or matched
  ablation; do not invent a model to keep the loop moving.
- Loss/metric mismatch is deferred to Phase 2 while Phase 1 loss is locked.

Decision fields:

- `Run: VALID` means complete, protocol-compliant, finite, reproducible, and
  interpretable.
- `Run: INVALID` means implementation, numerical, protocol, schema, checkpoint,
  or completeness failure; it receives no KEEP/DISCARD decision.
- `Decision: KEEP` means a valid screen is promoted or a valid full run is
  retained as a candidate/parent.
- `Decision: DISCARD` stops that exact valid configuration; artifacts and the
  scientific lesson remain.

## 9. Compute, communication, and Git

- Read saved artifacts before new inference. Cache reusable predictions,
  residuals, features, and subgroup assignments once.
- Read-only diagnostics default to at most 2,000 validation samples and five
  prefixes per cell, with a two-minute wall-time cap.
- Full-train inference requires written proof that saved validation evidence
  cannot answer the question.
- User updates: announce run start, hypothesis, and device; report material
  faults only; do not narrate epochs or waiting. After every reviewed run,
  report one concise aggregate containing: `KEEP` or `DISCARD` and the exact
  gates behind that decision; what headline, prefix, domain, and gradient
  evidence improved or regressed; which cause the evidence supports or
  refutes; and the next experiment's single changed block, falsifiable
  hypothesis, and evidence-based reason for selecting it. End with Git commit
  and push status.
- Before commit: run `validate_analysis.py <ID>` for a new run,
  `git diff --check`, and inspect `git status`.
- Local commit is a hard gate after every reviewed experiment. Inspect and
  report its hash and subject.
- Push is a soft gate with at most two attempts per experiment. On network,
  authentication, rate-limit, or remote-conflict failure, record the exact
  error and commit in `state.json.git_sync`, then continue locally.
- Retry pending commits before the next successful push and at handoff. Never
  claim an unsynced commit exists upstream.

## 10. Stop and phase transition

- Success: the three-seed target in Section 1 closes Phase 1 and unlocks Phase
  2; detailed future-phase rules stay in `loss.md` until then.
- Ordinary no-gain stop: at least 20 valid architecture runs, four healthy model
  families, leading-candidate confirmation, and no meaningful gain in the last
  15 valid runs.
- Hard limits: 50 valid architecture experiments (baseline reproductions not
  counted), 150 total recorded runs, and the compute budget in `state.json`.
- **Phase transition without the target (amendment A3, user, 2026-11-02):**
  - When Phase 1 ends by the ordinary no-gain stop or a hard limit without
    reaching the target, record `TARGET_NOT_MET`.
  - Freeze the best model, chosen by the §7 metrics. It must have a
    **three-seed** confirmation; run seeds 42/2024 first if needed.
  - Then **unlock Phase 2**. The 0.20 / 0.68 target stays the yardstick that is
    reported, not a blocker.
  - Phase 3 unlocks the same way after Phase 2.
  - Final-test data stays locked.
- **Early Phase-1 closure (amendment A5, user, 2026-11-03):** Phase 1 may also
  close before the no-gain window is full when all of these hold:
  - the diagnosed error is shown to be a shared, physics-inconsistent bias that
    architecture changes do not move (`diagnostics/phase1_breakthrough_diagnosis.md`);
  - the A4 directions have been tested;
  - the user agrees.

  The best candidate still needs a three-seed confirmation before it is frozen.
- Handoff MUST leave synchronized records, active parent, pending Git state,
  one supported next action, physics lock, and final-test lock explicit.
