# Loss Scope Lock & Mathematical Definitions

This document governs the training objectives across all three research phases.
Workflow rules live in `program.md`. Benchmark rules live in `protocol.md`.

## 1. Active Objective (Phase 1 Locked)
The ONLY active training objective during Phase 1 architecture search is sample-level unscaled Relative Huber:

### Mathematical Definition:
$$\text{rel\_err} = \frac{\hat{y} - y}{\max(y, 1)}$$

$$\mathcal{L}_{\text{main}} = \begin{cases}
0.5 \cdot (\text{rel\_err})^2 & \text{if } |\text{rel\_err}| \le \delta \\
\delta \cdot (|\text{rel\_err}| - 0.5 \cdot \delta) & \text{otherwise}
\end{cases} \quad (\text{with } \delta = 0.15)$$

### PyTorch Implementation:
```python
def relative_huber_loss(pred_eol, target_eol, delta=0.15):
    target_safe = torch.clamp(target_eol, min=1.0)
    rel_err = (pred_eol - target_safe) / target_safe
    return F.huber_loss(rel_err, torch.zeros_like(rel_err), delta=delta, reduction='mean')
```

Locked values for Phase 1: `delta = 0.15`, `physics_weight = 0.0`, and
`extra_weight = 0.0`.

## 2. Forbidden Objectives in Phase 1
Strictly forbidden to add, weight, or tune:
- Auxiliary physics, monotonicity, or capacity-fade losses.
- Domain-alignment or group-robustness penalties.
- Metric-surrogate, quantile, ranking, or uncertainty losses.
- Distillation loss (reserved exclusively for Phase 3).
- Any trainable auxiliary head whose loss weight is zero.
## 3. Representations vs. Objectives & Diagnostics
* **Allowed:** Architectural representations (e.g., Anchor difference $h_t - h_1$, Causal TCN) are permitted as long as they backpropagate solely from Relative Huber.

* **Diagnostics $\neq$ Losses:** Measuring monotonicity violations, prefix variance, or domain biases as read-only diagnostics is permitted. They MUST NOT be backpropagated during Phase 1.
## 4. Phase 2 — Mathematical Definitions of Candidate Losses (Gated)
Phase 2 unlocks when Phase 1 closes (`program.md §10`): either the 3-seed target is met (mean MAPE < 0.20, Acc@15 > 0.68), or Phase 1 stops without it (amendment A3) and the best three-seed-confirmed model is frozen with `TARGET_NOT_MET` recorded.
When unlocked, test these pre-defined objectives **one at a time** (one loss
addition per experiment). The development-room rule (`program.md §7`, A2)
applies to each loss direction. A new objective must be added here, with its
formula and code, before it can be used:
### 4.1 Metric-Aligned: $\epsilon$-Insensitive Relative Huber (Acc@15 Surrogate)

Creates a deadzone for errors $\le 15\%$ ($\epsilon = 0.15$). Only penalizes samples violating Acc@15.

$$err_\epsilon = \max\left(0, \frac{|\hat{y} - y|}{\max(y, 1)} - \epsilon\right)$$

$$L_\epsilon = \text{Huber}(err_\epsilon, \delta = 0.05)$$

```python
def epsilon_insensitive_relative_loss(pred_eol, target_eol, epsilon=0.15, delta=0.05):
    target_safe = torch.clamp(target_eol, min=1.0)
    rel_err = torch.abs(pred_eol - target_safe) / target_safe
    excess_err = F.relu(rel_err - epsilon)
    return F.huber_loss(excess_err, torch.zeros_like(excess_err), delta=delta, reduction='mean')
```
### 4.2 Physics & Temporal Invariant: Prefix-Drift Regularization

Penalizes variance across different prefix predictions for the same physical cell in a batch:

$$L_{\text{drift}} = \frac{1}{|P|} \sum_{(i,j) \in \text{SameCell}} \frac{|\hat{y}_i - \hat{y}_j|}{\max(y_i, 1)}$$

```python
def prefix_drift_loss(pred_eol, target_eol, cell_ids):
    # cell_ids: 1D Tensor of physical cell IDs in the batch
    loss = torch.tensor(0.0, device=pred_eol.device)
    pairs = 0
    unique_cells = torch.unique(cell_ids)
    for c in unique_cells:
        mask = (cell_ids == c)
        if mask.sum() > 1:
            preds = pred_eol[mask]
            targets = target_eol[mask]
            diffs = torch.abs(preds.unsqueeze(0) - preds.unsqueeze(1))
            norm_diffs = diffs / torch.clamp(targets.unsqueeze(0), min=1.0)
            loss += norm_diffs.triu(diagonal=1).sum()
            pairs += (mask.sum() * (mask.sum() - 1)) / 2
    return loss / max(pairs, 1)
```
### 4.3 Domain Alignment: Deep CORAL Loss

Aligns the second-order statistics (covariance matrices $C_S, C_T$) of latent features $Z \in \mathbb{R}^{B \times d}$ across lab domains:

$$
L_{\text{CORAL}} = \frac{1}{4d^2} \|C_S - C_T\|_F^2
$$

```python
def coral_loss(source_features, target_features):
    d = source_features.size(1)
    # Covariance source
    ns = source_features.size(0)
    cs = torch.cov(source_features.T) if ns > 1 else torch.zeros(d, d, device=source_features.device)
    # Covariance target
    nt = target_features.size(0)
    ct = torch.cov(target_features.T) if nt > 1 else torch.zeros(d, d, device=target_features.device)
    return torch.sum((cs - ct) ** 2) / (4 * d * d)
```

### 4.4 Domain Robustness: Group-DRO Loss

Optimizes the worst-performing domain loss to prevent guardrail regressions on domains with $\ge 10$ cells:

$$
L_{\text{DRO}} = \max_{g \in \text{Domains}} L_g
$$

```python
def group_dro_loss(per_sample_losses, domain_ids):
    # Returns the maximum mean loss among present domains in batch
    unique_domains = torch.unique(domain_ids)
    domain_losses = [per_sample_losses[domain_ids == g].mean() for g in unique_domains]
    return torch.stack(domain_losses).max()
```
### 4.5 Safety Asymmetry: Asymmetric Relative Huber

Penalizes over-prediction ($\hat{y} > y$) by weight $\tau > 0.5$ more severely than under-prediction:

```python
def asymmetric_relative_huber(pred_eol, target_eol, delta=0.15, tau=0.75):
    target_safe = torch.clamp(target_eol, min=1.0)
    rel_err = (pred_eol - target_safe) / target_safe
    base_loss = F.huber_loss(rel_err, torch.zeros_like(rel_err), delta=delta, reduction='none')
    weight = torch.where(rel_err > 0, tau, 1.0 - tau)
    return (weight * base_loss).mean()
```
### 4.6 Physics losses (what-if, label-free) — exact definitions

**Notation**
- Sample = one cell prefix: curves $x\in\mathbb{R}^{100\times3\times300}$ (channels V, I, Q;
  points 0–149 charge, 150–299 discharge), mask $m\in\{0,1\}^{100}$, prefix length
  $i=\sum_n m_n$, label $y$ (cycles).
- $f(x)$ = model output in **standardized** units; cycles $\hat y=\mu+s\,f(x)$, with
  $\mu,s$ the training-label mean and SD. $\operatorname{sg}[\cdot]$ = stop-gradient.
- Cycle index $n=1..100$ (tensor index $t=n-1$). Ramp:
  $r_n=\operatorname{clip}\big((n-10)/90,\,0,\,1\big)$, so cycles 1–10 are never changed.
- $Q^{dis}_n=\max_k x_{n,2,150+k}$ (cycle discharge capacity).
- For a domain $d$, using **training cells only**:
  $\phi_c=(Q^{dis}_{10}-Q^{dis}_{100})/Q^{dis}_{10}$ (relative fade, cells with ≥ 100 cycles);
  $\alpha_{\max}(d)=P_{90}\{\phi_c\}_{c\in d}$. Domains with < 5 cells use the global $P_{90}$.
- Every what-if is applied **only to observed cycles** ($m_n=1$). Masked cycles stay 0.
- $\alpha\sim U(0,\alpha_{\max}(d))$ is drawn per sample from a dedicated RNG
  (seed + 10000), so training batches stay identical to the matched reference.

**Generic one-sided what-if loss** (rule: "worse battery → life not longer"):
$$L_{P}=\frac1B\sum_b \mathrm{ReLU}\big(f(\tilde x_b)-\operatorname{sg}[f(x_b)]+\mu_m\big)$$
- $\mu_m\ge0$ is an optional margin in standardized units (default 0).
- The labeled prediction $f(x)$ gets **no** gradient from $L_P$, so the label is never pulled.
- The loss is in standardized units, i.e. cycles divided by $s$: dimensionless (§5.1).

| ID | What-if $\tilde x$ (only the listed entries change) | Range |
|---|---|---|
| **P0** uniform fade | $\tilde x_{n,2,150:300}=x_{n,2,150:300}\,(1-\alpha r_n)$ | $\alpha\le\alpha_{\max}(d)$ |
| **P4** own-trend acceleration | $\tilde x_{n,2,150:300}=x_{10,2,150:300}+(1+\kappa r_n)\,(x_{n,2,150:300}-x_{10,2,150:300})$ for $n\ge11$ | $\kappa\sim U(0,1)$; only samples with $i\ge 20$ |
| **P0c** knee | $\tilde x_{n,2,150:300}=x_{n,2,150:300}\,(1-\alpha r_n^2)$ | $\alpha\le\alpha_{\max}(d)$ |
| **P2** voltage sag | $\tilde x_{n,0,150:300}=x_{n,0,150:300}-\beta r_n$ | $\beta\sim U(0,\beta_{\max}(d))$; $\beta_{\max}$ = $P_{90}$ of the training drop in mean discharge V (normalized), cycle 10 → 100 |

**P4b falling-trend acceleration** (added 2026-11-03 after R4-E064):
- P4 amplifies $x_n-x_{10}$. In 79 of 503 training cells (HUST 17/47, MATR 33/102,
  XJTU 14/15, ISU 14/143) capacity rises over the prefix (break-in).
  - For those cells P4 makes the battery better, so the rule "not longer life" is false.
  - R4-E064 failed for this reason: gradient cosine median −0.71, PHYSICS_GRADIENT_CONFLICT,
    HUST MAPE 0.18 → 0.68.
- P4b accelerates only the loss part of the trend:
  $\Delta_n=x_{n,2,150:300}-x_{10,2,150:300}$,
  $\tilde x_{n,2,150:300}=x_{10,2,150:300}+\Delta_n+\kappa r_n\min(\Delta_n,0)$ for $n\ge11$.
  - $\kappa\sim U(0,1)$; only samples with $i\ge20$. Rising parts are left unchanged.
- Config: `whatif_trend_p4b`.

**P5 horizon validity** (no what-if; true for every sample because $i<y$):
$$L_{P5}=\frac1B\sum_b\mathrm{ReLU}\Big(\frac{i_b-\hat y_b}{s}\Big).$$

**C-flip control** (same what-if, reversed rule):
$L_{flip}=\frac1B\sum_b\mathrm{ReLU}\big(\operatorname{sg}[f(x_b)]-f(\tilde x_b)\big)$.
If C-flip helps as much as P0, the gain is regularization, not physics.

**Held-out physics metric (always reported):**
- Violation rate = share of validation samples with $f(\tilde x)>f(x)$, pooled over
  $\alpha\in\{0.25,0.5,1.0\}\cdot\alpha_{\max}(d)$.
- Report it for the candidate **and** the matched reference.
- Claim "physics-informed" only if the violation rate falls **and** accuracy does
  not get worse.

```python
def ramp(device, dtype):
    n = torch.arange(1, 101, device=device, dtype=dtype)
    return ((n - 10) / 90).clamp(0, 1)                                   # [100]

def p0_whatif(x, mask, alpha):                                           # x [B,100,3,300]
    s = 1 - alpha[:, None] * ramp(x.device, x.dtype)[None] * mask        # [B,100]
    xt = x.clone(); xt[:, :, 2, 150:] = x[:, :, 2, 150:] * s[..., None]
    return xt

def one_sided(f_cf, f_x, margin=0.0):                                    # standardized outputs [B]
    return F.relu(f_cf - f_x.detach() + margin).mean()
```

**Tested and rejected (2026-11-03): linear-extrapolation upper bound**
($y\le k\cdot 20/\phi$).
- It holds for MATR/MICH, but not for cells whose fade slows after a fast start.
- Share of cells with $y\le$ bound: ISU-ILCC 18% (k = 1) and 70% (k = 2); SNL 10% / 79%;
  HUST 33% / 93%.
- As a loss it would contradict labels. Do not use it.

**Re-tested and rejected again (2026-11-03): prefix-only fade-extrapolation bound**
($\hat y\le k\cdot EOL_{lin}$, slope between mid-prefix and end-prefix, $k=4.6$ = train $P_{98}$).
- Built for R4-E065, removed before running: same family as the bound above, contradicts
  ~2% of training labels. Do not use it.

**Order to test:** P0 → P4 → P0c → P2. P5 may be added to any of them.

### 4.7 Objectives used in Phase 2 that were not pre-registered here (recorded 2026-11-03)

These were run before being written here, which broke the rule above. They are recorded
now with their exact form and outcome. Physics = a rule about Li-ion degradation that holds
without labels; statistical = everything else.

| Name (config) | Formula | Kind | Runs → outcome |
|---|---|---|---|
| `prefix_drift` | $\lambda\,\mathrm{mean}|f(x_j)-f(x_i)|$, $x_j$ = same sample truncated to $j=\lfloor u i\rfloor$, $u\sim U(0.5,1)$ | consistency (EOL is one number per cell); weak physics | E046–E057, E063 (v2 3-seed mean 0.1872) |
| `prefix_drift` + `physics_anchor_gradient="stop"` (one-way drift, registered 2026-11-05) | $\lambda\,\mathrm{mean}|f(x_j)-\mathrm{sg}[f(x_i)]|$: only the shorter-prefix branch $x_j$ is pulled toward the full-prefix prediction | consistency with information direction (a longer prefix carries strictly more ageing evidence, so the shorter-prefix prediction is the one that should move). Motivation: across 30 two-sided prefix-drift runs the max val prediction is 1968 (median) vs 2707 without physics, and EOL>2000 MAPE is 0.331 vs 0.227. Two-sided drift pulls informed long-prefix predictions toward uninformed short-prefix ones. | R4-E082 screen DISCARD (0.2025 at epoch 5, λ 0.061). **Correction 2026-11-06:** the motivating 30-vs-31 comparison was confounded by input version. Matched pairs show prefix drift does not cause the long-life error (E055 vs E056 EOL>2000 MAPE 0.367 vs 0.360; E025 vs E053 0.225 vs 0.248). The driver is input/metadata (v1 0.225; v2 0.367; v2+meta 0.257–0.276; meta dropout 0.5: 0.335–0.372). |
| `prefix_drift` + `physics_drift_space="log"` (scale-free drift, registered 2026-11-06) | $\lambda\,\mathrm{mean}|\log\hat y(x_j)-\log\hat y(x_i)|$, two-sided, $\hat y = s(f+\mu/s)$ | consistency measured in relative terms, like the Relative Huber main loss and MAPE. Standard-unit drift charges a long-life cell ~EOL/μ times more for the same relative inconsistency, which pulls high predictions toward the mean (two-sided runs: median max prediction 1968 vs 2707 without physics). One-way drift (R4-E082) removed that pull but weakened λ (0.061) and lost the bulk gain (screen 0.2025 vs 0.1752). | R4-E083 screen DISCARD (0.1891 > 0.1852; max prediction 1938, EOL>2000 0.335 unchanged). Motivation was confounded; see the R4-E082 correction. |

*v5 note (2026-11-08):* in v5 (Na-ion) the one-way row above is re-tested as NA-E024 after the two-sided NA-E023 helped short prefixes but moved the long view; in v4 R4-E082 this form lost the bulk gain (λ 0.061).
| `degradation_law`, `degradation_law2` | aux head fits $q(n)$ with a 1- or 2-exponent power law $1-a n^{z}$ on the train Q trajectory | physics (empirical fade law) | E047 INVALID, E048–E049 DISCARD |
| `timewarp_aug` | $x_\kappa(m)=x(\lfloor\kappa m\rfloor)$, label $EOL/\kappa$ | physics-motivated augmentation | E052 DISCARD |
| `budget_bound` | $\mathrm{ReLU}(\hat y-2\,EOL_{phys})/s$, $EOL_{phys}=i+(q_i-0.8)/rate$, active if $q_i-0.8<0.05$ | physics bound (same family as the rejected bound) | E061 DISCARD (RMSE 435) |
| `rarity_alpha` | main loss weighted by $w=\min(5,\hat p(\log EOL)^{-\alpha}/Z)$, KDE bw 0.25 | statistical | E060 (α=1) DISCARD, E064 (α=0.5) |
| `ema_decay`, `target_transform=log` | weight EMA; regression on $\log EOL$ | statistical / training | E058, E059 DISCARD |
| metadata (A7, model input) | $[\log w_{SOC}, SOC_0, \log(1+C_{ch}), \log(1+C_{dis})]$ added to the summary | input, not a loss | E062 DISCARD |

**`stat_tail_balance` (statistical; registered 2026-11-05 from the Argon report
`handoff/reports/phase2_breakthrough_argon.md`, rank 1).**
- Problem it targets:
  - Relative Huber (δ = 0.15) clips the gradient for |rel err| > 0.15.
  - $\partial L/\partial f$ scales with $s/y$, so long-life cells get about 9× less pull than
    short-life cells.
- Definitions, with $r=(\hat y-y)/\max(y,1)$ and $\mu$ = train label mean:
  - $e^+=\mathrm{ReLU}(r-0.15)$;
  - $e^-=\mathrm{ReLU}(-r-0.15)$;
  - $w=\mathrm{clip}(y/\mu,1,3)$.
- Loss:
  $L_{tail}=\frac1B\sum_b\big[\tau\,\mathrm{Huber}_{0.5}(e^+_b)+(1-\tau)\,w_b\,\mathrm{Huber}_{0.25}(e^-_b)\big]$, with $\tau=0.55$.
- Config: `physics2_type="stat_tail_balance"`, `tail_tau`, `tail_scale_max`.
- λ2 = ρ2 · median gradient ratio (ρ2 = 0.25), on 3 train batches.
- No extra forward pass. Uses labels, so it is **statistical, not physics**.

**Rejected before running: `phys_pair_monotone`** (Argon rank 2).
- Rule: for two cells $a,b$ in the same domain with $w_a\ge w_b$, late fade
  $\phi_a\ge\phi_b+\epsilon$ and voltage sag $\Delta V_a\ge\Delta V_b-0.002$, require $\hat y_a\le\hat y_b$.
- Checked on train labels: it holds for only 87–93% of pairs at ε = 0.008 (prefix 30–100).
  It reaches ≥ 98% only at prefix 100 with large gaps (few pairs).
- It is not a strict Li-ion law. As a loss it would contradict labels.

**Physics check for Li-ion (2026-11-03).**
- P0, P4, P0c and P2 are sound for Li-ion. EOL is defined by a capacity threshold, so with
  everything else fixed, faster fade, a knee or a voltage sag (resistance growth) cannot
  lengthen life. They are label-free and never contradict a label.
- P5 is always true but weak.
- Stress-factor rules from the data (train cells):
  - Depth of discharge (SOC width) is supported: Spearman(log EOL, log width) = −0.87 on ISU.
  - C-rate monotonicity is **not** supported: +0.04 / +0.22 on ISU, +0.13 on MATR. Do not use
    it as a rule.
  - The DoD rule holds across cells but not with the normalized fade trajectory held fixed,
    so it is not a valid what-if on v2 input.
Statistical comparators (label them as statistical): §4.5 asymmetric Huber and
§4.1 ε-insensitive.

## 5. Normalization, balancing and gradient checks (mandatory in Phase 2)

### 5.1 Normalization: every term is dimensionless and O(0.01–1)

| Term | Unit | Typical size |
|---|---|---|
| $L_{main}$ Relative Huber | relative error | ≈ 0.01–0.05 |
| $L_{P0},L_{P4},L_{P0c},L_{P2},L_{flip}$ | standardized output = cycles / $s$ | ≈ 0.001–0.1 |
| $L_{P5}$ | cycles / $s$ | ≈ 0 (rarely active) |

- Never add raw cycles, Ah or volts to the objective.
- Never divide by a tiny number to inflate a term.
- If a new term cannot be written in these units, it is not allowed.

### 5.2 Balancing: λ from gradients, not by hand

$L=L_{main}+\sum_j\lambda_j L_j$. Before step 1, on 3 fixed training batches:
$$\lambda_j=\rho\cdot\operatorname{median}_b\frac{\lVert\nabla_\theta L_{main,b}\rVert}{\lVert\nabla_\theta L^{cal}_{j,b}\rVert},\qquad \rho=0.5 .$$
- $\nabla_\theta$ is taken over **all trainable parameters**, in float64, with a
  denominator floor of $10^{-12}$.
- For one-sided terms, use the **unsigned** calibration loss
  $L^{cal}_j=\frac1B\sum|f(\tilde x)-\operatorname{sg}[f(x)]|$. At step 0 the sign is
  random, so the one-sided loss can be 0 and give a wrong λ.
- Changing $\rho$ is a new experiment. λ is logged and is never tuned on validation.

### 5.3 Balance band (checked during training)

**Goal:** neither loss may dominate, and the smaller loss must still be used.
- The **loss value** ratio `λL_j / L_main` is reported but is not a gate. One-sided
  physics terms are often tiny in value (e.g. about 2e-4 in R4-E040) while
  their gradient is not.
- What decides learning is the **gradient** ratio `R_j`. It is the gated quantity.
- A term is only "used" if R_j is inside the band **and** its active fraction is > 0.

Every 250 steps, for each term $j$ with active samples, log:
- $L_{main}$ and $\lambda_jL_j$;
- the weighted gradient ratio $R_j=\lambda_j\lVert\nabla L_j\rVert/\lVert\nabla L_{main}\rVert$;
- the cosine $\cos(\nabla L_{main},\nabla L_j)$;
- the active fraction.

| Check | Rule | Outcome |
|---|---|---|
| Too small | median $R_j<0.05$ over active samples after warm-up | `BALANCE_FAIL` (term has no effect) → DISCARD |
| Too large | median $R_j>2.0$, or any two consecutive samples $>5$ | `BALANCE_FAIL` (term dominates) → DISCARD |
| Inactive | active fraction = 0 on all samples | `NOT_ASSESSED`; the term did nothing; report it, do not claim physics |
| Satisfied (added after R4-E042) | R in band in the first samples, then median active fraction < 0.10, **and** held-out violation rate falls ≥ 50% vs the reference | `SATISFIED`: the term worked and is now inactive; not a balance failure |
| Non-finite / disconnected | NaN/Inf, or $\nabla L_j$ never reaches the encoder | `INVALID` |
| Cosine | log only. For one-sided physics terms a negative cosine is expected (they push against memorization) | flag, not a gate |

- One evidence-based repair may change **one** of: $\rho$, the margin, or the
  what-if range. Never change several together.
- These checks are written by `train.py` into `diagnostics.json`
  (`summary.loss_balance`). The reviewer checks them before any KEEP.

### 5.3b Adaptive λ (augmented Lagrangian), optional balancing mode

- `physics_lambda_mode = "lagrangian"`: start from the calibrated λ (§5.2).
- After every 250 steps, update
  `λ ← max(0, λ + η·(v − v*))`, where `v` = training active fraction and
  `v*` = 0.05 target (default `η = λ₀`).
- λ grows while the rule is violated and shrinks once it is satisfied. This
  avoids both a dominant term and the "R → 0 when satisfied" artefact (R4-E042).
- Log the λ trajectory. The balance band (§5.3) is still reported; the
  `SATISFIED` state applies.

### 5.4 Other auxiliary losses (§4.1–4.5)

- They use the same λ calibration (signed loss, no unsigned version needed),
  the same band and the same cosine rule.
- Exception: if a term is a full replacement of the main loss (e.g. §4.5),
  it replaces $L_{main}$ and needs no λ.

## 6. Preflight Enforcement (Fail-Closed)

`workflow_preflight.py` MUST reject the run if:

* `physics_loss_weight != 0.0` or `extra_loss_weight != 0.0` while in Phase 1.
* `main_loss != relative_huber` or `delta != 0.15`.
* In Phase 2, a proposed loss is not one of the registered implementations above.

## 4.8 v5 Na-ion physics candidates (registered 2026-11-09, user request; pre-checks on the locked caches)

**Context.** The frozen parent (NA-E009, NaFadeNet + adaptive extrapolation gate) already satisfies the capacity what-if rules
(P0/P0c/P4b: 0 violations > 1e-3 cycles, correct shifts −13…−70 cycles). P2 (NA-E021/E022), prefix drift (NA-E023/E024) and the
learned-exponent degradation law (NA-E025–E028, A5-NA) failed their physics gates. Pre-checks below decide which Na-ion physics
can still constrain the model. All statistics use training cells only unless stated (val only for held-out metrics).

**Pre-check results (rejected as losses — do not run):**

| Candidate | Physics | Pre-check (splits 2021 / 42 / 2024) | Verdict |
|---|---|---|---|
| C-rate monotonicity | higher current → faster ageing | corr(log EOL, peak charge current) = +0.55 (all cells, wrong sign; every cell is logged as a 2.5C protocol, the current spread is a tester artefact) | contradicts data |
| CE what-if (P-CE) | lower Coulombic efficiency (more side reactions) → shorter life | corr(log EOL, 1−CE) = −0.55…−0.74 (right sign), but the training CE spread is only 7–9e-4; at that range the parent moves 0.02–0.05 cycles (0 shifts > 1 cycle) | sub-resolution → would act as a constant push (cf. NA-E022) |
| Hysteresis what-if (P-H) | faster polarisation growth → shorter life | corr(log EOL, hysteresis growth) = −0.32…−0.70, but the parent moves ≤ 0.00 cycles at the training range (≤ 2.2e-3 normalised V) | sub-resolution (normalised voltage) |
| CE–fade coupling | Na inventory lost to side reactions sets the fade rate: dq/dn ∝ (1−CE) | corr(1−CE, true fade rate) = 0.24 / 0.45 / 0.21; fitted k negative; EOL implied by CE has train MAPE 4–37 | law not supported by the data |

**Supported physics:** near-linear capacity fade. Law exponent with the true EOL: b median 1.05 (0.85–1.25), fit error 1.2 % of the
fade budget; corr(observed fade rate cycles 10–100, true fade rate (q1 − 0.8)/EOL) = 0.95 / 0.96 / 0.96.

**Registered candidates (each needs a proposal, review and a decision rule fixed before running, 3 splits) — v2 after Codex review:**
- **N1 `degradation_law_fixed_b`** — training-time physics supervision with the measured Na-ion law (b = 1, no law head):
  $L_{N1}=\frac1{|S|}\sum_{s\in S}\frac1K\sum_k\frac{|q_{ref,s}-(q_{ref,s}-0.8)\,n_k/\hat y_s-q_s(n_k)|}{q_{ref,s}-0.8}$
  - $S$ = batch samples whose cell is a TRAINING cell; $\hat y_s$ = final model output de-standardised to cycles, clamped ≥ 20 (gradient flows).
  - $q_s(n)$ = per-cycle discharge capacity / (nominal × SOC width) from the raw train pickle, median-9 smoothed (tools/build_qtraj.py);
    $q_{ref}$ = median of cycles 1–10; threshold fixed at **0.8** (BatteryLife EOL definition — not $q$ at the label EOL).
  - $n_k$: K = 32 evenly spaced cycles on $[1, n_{0.8}]$, $n_{0.8}$ = first smoothed cycle with $q\le 0.8$ (train cells only).
  - Status of the trajectory: it is training-time supervision with the same standing as the EOL label (train cells only, never
    val/test, never an input). Restricting it to the observed prefix would make N1 identical to the gate's linear extrapolation.
  - Units: dimensionless, O(0.01–0.1) (fixed-b true-EOL scale 0.034 on split 2021).
- **N2 `observed_fade_consistency`** (label-free) — the pre-gate network branch must agree with the cell's own observed fade:
  $L_{N2}=\mathrm{mean}_{s\in V}\,\mathrm{ReLU}(|\log y^{cyc}_{net,s}-\log \mathrm{sg}[\hat y_{lin,s}]|-\tau)$, τ = 0.10,
  $y^{cyc}_{net}=\mu+s\,y_{net}$ clamped ≥ 20 cycles; $\hat y_{lin}$ = the gate's masked last-20-cycle line crossing 0.8 (stop-gradient);
  $V$ = samples with prefix i ≥ 50 whose line has negative slope and an unclamped crossing (n+1 < ŷ_lin < cap). Units: dimensionless.
  Caveat (Codex): overlaps the gate and may erase the useful complementarity of the two branches — tested only after N1.
- Held-out physics metrics: N1 → tools/law_fit_eval.py with b fixed to 1 and threshold 0.8 (excess over the true-EOL floor); N2 → share of val samples (i ≥ 50) with
  |log y_net − log ŷ_lin| > τ.
