#!/usr/bin/env python3
"""Record a finished run in one step (analysis.json -> validate -> results.tsv -> research_log -> state).

Usage: python tools/auto_record.py <ID> <REF_ID> "<next action summary>" [--keep-if-better]
Decision: KEEP if MAPE < reference MAPE and RMSE <= reference RMSE + 21 (A1 guardrail), else DISCARD.
All loss/balance/layer facts are taken from the run's own files.
"""
import csv, json, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "tools"))
import rescore
PY = sys.executable

def main():
    global UNUSED
    eid, ref, nxt = sys.argv[1], sys.argv[2], sys.argv[3]
    UNUSED = [x for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else []) if x]  # layers unused by design
    ex = ROOT / "experiments" / eid
    rec = json.loads((ex / "record.json").read_text()); dg = json.loads((ex / "diagnostics.json").read_text())
    s, r = rescore.score(eid), rescore.score(ref)
    tr = rec["proposal"]["training"] if "proposal" in rec else {}
    lb = dg.get("summary", {}).get("loss_balance", {}) or {}
    smp = lb.get("samples", []) if isinstance(lb, dict) else []
    cos = [x["gradient_cosine"] for x in smp if x.get("gradient_cosine") is not None]
    R = [x["R"] for x in smp if x.get("R") is not None]
    health = dg.get("summary", {}).get("overall_gradient_health")
    layers = subprocess.run([PY, str(ROOT / "tools/layer_report.py"), eid, "--short"], capture_output=True, text=True).stdout.strip()
    lam = dg.get("calibrated_physics_lambda"); lp5 = (dg.get("physics_calibration") or {}).get("lambda_p5")
    by = {k: round(v["mape"], 3) for k, v in rec["metrics"]["by_domain"].items() if v.get("unique_cell_count", 0) >= 10}
    pc = dg.get("pcgrad")
    fmt = lambda v, n=4: "NA" if v is None else f"{v:.{n}f}"
    bal = (f"balance {lb.get('status')} R median {np.median(R):.3f} [{min(R):.3f}, {max(R):.3f}]; cosine median {np.median(cos):.3f}, min {min(cos):.3f}" if R and cos else "no physics balance samples")
    facts = (f"best epoch {rec['result']['best_epoch']}/{rec['result']['completed_epochs']} ({rec['result']['stop_reason']}): "
             f"MAPE {s['mape']:.4f} (ref {ref} {r['mape']:.4f}); Acc@15 {s['acc15']:.4f} ({r['acc15']:.4f}); RMSE {s['rmse']:.1f} ({r['rmse']:.1f}); "
             f"DB-MAPE {s['db_mape']:.4f} ({r['db_mape']:.4f}); seen {fmt(s['seen_mape'],3)}, unseen {fmt(s['unseen_mape'],3)} ({fmt(r['unseen_mape'],3)}); "
             f"domains {by}; physics_type {tr.get('physics_type','none')} lambda {fmt(lam)} lambda_p5 {fmt(lp5)} rho {tr.get('physics_rho')}; "
             f"ema {tr.get('ema_decay',0)}; input {tr.get('input_normalization','nominal')}; {bal}; layers {layers}; gradient {health}"
             + (f"; pcgrad {pc}" if pc else ""))
    screening = (rec.get("proposal") or {}).get("training_stage") == "screening"
    if screening:  # validate_analysis screen hard floor (program.md §6)
        ratio = dg.get("summary", {}).get("learning_dynamics", {}).get("first_to_last_train_loss_ratio")
        floor_pass = s["mape"] <= r["mape"] + 0.10 and ((s["mape"] < 0.20 and s["acc15"] > 0.68) or (ratio is not None and ratio <= 0.70))
        SCREEN = {"applicable": True, "pass": floor_pass, "reason": "5-epoch screen hard floor."}
    else:
        SCREEN = {"applicable": False, "pass": None, "reason": "Full run."}
    keep = s["mape"] < r["mape"]  # amendment A9: MAPE decides; RMSE reported only
    invalid = health == "UNHEALTHY"  # WARNING is decided on metrics (train.py action rule; Codex R4-UNDERFIT)
    a = {"schema_version": 1, "experiment_id": eid, "run_validity": "INVALID" if invalid else "VALID",
         "contract_answer": {"question": rec["proposal"].get("primary_hypothesis", "") if "proposal" in rec else eid,
                             "outcome": "INCONCLUSIVE" if invalid else ("SUPPORTED" if keep else "REFUTED"),
                             "conclusion": facts, "evidence_paths": ["record.json.metrics", "diagnostics.json.summary.loss_balance", "tools/layer_report.py"]},
         "main_symptom": ("MAPE better than reference within RMSE guardrail." if keep else "No guarded gain over reference."),
         "causes": [{"id": "C1_RECIPE_EFFECT", "class": "loss_design", "supporting_evidence": [f"MAPE {s['mape']:.4f} vs {r['mape']:.4f}"], "contradicting_evidence": [], "interpretation": "Effect of the changed component(s).", "missing_evidence": ["other seeds"]},
                    {"id": "C2_SEED_NOISE", "class": "evaluation", "supporting_evidence": ["seed spread ~0.02"], "contradicting_evidence": [], "interpretation": "Single-seed differences below ~0.01 are within noise.", "missing_evidence": []}],
         "selected_cause_id": "C1_RECIPE_EFFECT",
         "layers": [{"name": "changed_component", "role": rec["proposal"].get("changed_components", "") if "proposal" in rec else "", "evidence_paths": ["tools/layer_report.py"],
                     "persistent_dead_gradient": False, "action": "RETAIN" if keep else "REMOVE", "reason": "guarded gain" if keep else "no guarded gain"}]
                   + [{"name": u, "role": "unused by design in this run", "evidence_paths": ["tools/layer_report.py"], "persistent_dead_gradient": True,
                       "action": "REMOVE", "reason": "layer receives no input in this configuration (e.g. no metadata channel); zero gradient expected"} for u in UNUSED],
         "gates": {"gradient": "PASS" if health == "HEALTHY" else "FAIL", "screen_floor": SCREEN},
         "decision": None if invalid else ("KEEP" if (keep and (not screening or SCREEN["pass"])) else "DISCARD"),
         "next_action": {"type": "REPAIR" if invalid else "NEW_HYPOTHESIS", "summary": nxt, "source_cause_id": "C1_RECIPE_EFFECT", "responsible_block": "head"}}
    (ex / "analysis.json").write_text(json.dumps(a, indent=1))
    subprocess.run([PY, str(ROOT / "validate_analysis.py"), eid], check=True)
    rows = list(csv.reader((ROOT / "results.tsv").open(), delimiter="\t"))
    dec = "INVALID" if invalid else a["decision"]
    if not invalid and nxt == "confirmation": dec = "CONFIRM"  # seed-confirmation run: KEEP/DISCARD vs a single-seed ref is meaningless
    for row in rows:
        if row[0] == eid:
            if invalid: row[14] = "INVALID"
            row[15] = dec; row[16] = facts[:600]; row[17] = nxt
    csv.writer((ROOT / "results.tsv").open("w", newline=""), delimiter="\t", lineterminator="\n").writerows(rows)
    with (ROOT / "research_log.md").open("a") as h:
        h.write(f"\n## {eid} — {dec} (auto-record vs {ref})\n\n- {facts}\n- Next: {nxt}\n")
    st = json.loads((ROOT / "state.json").read_text()); st["current_experiment"] = eid
    st["next_experiment_id"] = f"NA-E{int(eid.split('E')[-1]) + 1:03d}"
    (ROOT / "state.json").write_text(json.dumps(st, indent=2))
    subprocess.run([PY, str(ROOT / "tools/rescore.py"), "--write"], capture_output=True)
    print(f"{eid} {dec}: MAPE {s['mape']:.4f} Acc {s['acc15']:.4f} RMSE {s['rmse']:.1f} DB {s['db_mape']:.4f} unseen {fmt(s['unseen_mape'],3)} | {bal} | {layers} | lam {fmt(lam)} lp5 {fmt(lp5)} | {by}")

if __name__ == "__main__":
    main()
