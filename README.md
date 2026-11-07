# Battery EOL Autoresearch — Project Briefing

This document serves as the high-level architecture map and operational guide for the autonomous research loop.

## 1. Authority & Governance
Read files in strict priority order:
1. `protocol.md` (Locked benchmark, data, split, metrics, and test rules)
2. `program.md` (Active workflow, gates, diagnostics, budgets, and stop rules)
3. `state.json` (Live approval, active parent, budget, current run, and next ID)
4. `research_guide.md` | `architecture.md` | `loss.md` (Domain and design guides)
   - `phase3_targets.md` (Phase 3 TinyML draft: boards, model footprint, compression method catalogue M1–M9 mapped to MIT 6.5940 lectures; Phase 3 stays locked until the user unlocks it)
5. `research_log.md` (Completed experiment evidence and lessons)

*Team layer:* `multiagent.md` explains how Claude, Codex and Gemini share this workflow.

*Amendments:*
- A1 (metrics, BatteryLife baselines, most-accurate model): `protocol.md` header.
- A2 (development room): `program.md §7`.
- A3 (Phase 2/3 unlock after a no-gain stop, best confirmed model frozen): `program.md §10`.
- A4 (seen/unseen-condition metric; Phase-1 focus on unseen-condition generalization): `program.md §2c`.

*Note: This briefing is an architectural overview. If any rule appears to conflict with `protocol.md` or `program.md`, those files take precedence.*

## 2. Live State Reference
Do NOT hardcode run summaries in this file. At the start of any session:
- Inspect `state.json` for `current_experiment`, `next_experiment_id`, `active_architecture_parent`, `phase`, and remaining budget.
- Read the latest reviewed entry in `research_log.md` for recent lessons and closed branches.
- Active targets (Phase 1): Mean 3-seed MAPE strictly `< 0.20` and Acc@15 strictly `> 0.68` from the selected checkpoint.

## 3. Where the LLM Agent Is
The LLM agent operates strictly as an autonomous research controller. It is not part of embedded inference and does not run inside the battery predictor.

```text
protocol + saved experiment evidence
                 |
                 v
            LLM agent
  analyze -> form one hypothesis -> write one proposal
                 |
                 v
        preflight checks + GPU training
                 |
                 v
 metrics + group errors + loss/gradient diagnostics
                 |
                 v
            LLM agent
       VALID / INVALID, then KEEP / DISCARD
                 |
                 v
       next justified experiment or final TinyML model
```

## 4. How the Files Work Together
```
protocol.md + protocol_lock.json
              |
              v
      workflow_preflight.py <----- state.json
              ^                       |
              |                       |
      experiment proposal             |
              |                       |
              +-----------+-----------+
                          |
                          v
                       train.py <----- prepare.py <----- locked cache
                          |
             +------------+-------------+
             |            |             |
             v            v             v
     experiments/<ID>/  results.tsv   live.log
             |            |             |
             +------+-----+             +--> monitor only
                    |
                    v
         agent reads full evidence
                     |
          +----------+-----------+
          |                      |
          v                      v
    research_log.md          state.json
          |                      |
          +----------+-----------+
                     |
                     v
             next experiment proposal
```
File roles:
- protocol.md: Benchmark rules and locked definitions.
- protocol_lock.json: Hash verification for cache and protocol integrity.
- state.json: Live state machine and single source of truth for runtime pointers.
- workflow_preflight.py: Automated sanity check before launching runs; fails closed.
- prepare.py: Read-only access to locked preprocessed cache.
- train.py: Executes run, collects hooks-based diagnostics, saves outputs.
- experiments/<ID>/: Raw artifacts, code snapshot, and validated analysis.json.
- results.tsv: High-level metric summary table.
- live.log: Transient run monitoring only; not final evidence.
- research_log.md: Chronological narrative of validated lessons and next questions.
- tools/rescore.py: MAPE, Acc@15, RMSE, DB-MAPE from `record.json`; `--write` fills the evaluation columns of `results.tsv`.
- multiagent.md + claude/, codex/, gemini/, handoff/, tools/: the agent team layer.

## 5. Operational Boundaries & Safety Gates
- Phase Boundary: Phase 1 is strictly for architecture search. Loss components (Relative Huber fixed), physics loss, TinyML quantization, and final-test data are strictly LOCKED.
- Hypothesis Discipline: Exactly one main change and one falsifiable research question per proposal.
- Branch Limit (amendment A2): a promising direction gets up to 3 evidence-led follow-ups and closes after 4 consecutive DISCARDs; a direction without a promising signal gets one targeted repair, then closes (`program.md §7`).
- Automated Diagnostics: Layer gradients, update ratios, activations, and masks are computed via train.py hooks automatically. The agent reads diagnostics.json, never re-implements logging logic.
- Decision Contract: Every run must be classified as Run: VALID | INVALID. Only valid runs receive Decision: KEEP | DISCARD.

## 6. Session Handoff Checklist
Before ending a session or passing context to another agent instance, verify:
1. Is state.json updated with the latest ID, budget, and active parent?
2. Is the latest run recorded in results.tsv and reviewed in research_log.md?
3. Has the local Git checkpoint been committed?
4. What is the single supported next change, hypothesis, and reject rule?
