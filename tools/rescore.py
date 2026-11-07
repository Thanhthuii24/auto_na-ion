#!/usr/bin/env python3
"""Recompute the decision metrics for every run from saved record.json (no training).

Metrics (Round 4 amendment A1, user-approved 2026-11-02):
- MAPE, Acc@15 (BatteryLife paper metrics), RMSE (cycles, co-primary), MAE;
- DB-MAPE: domain-balanced MAPE = unweighted mean of per-domain MAPE over
  domains with >= 10 validation cells, all from the SAME selected checkpoint.
- Seen/unseen MAPE (amendment A4): validation cells whose BatteryLife aging
  condition (name2agingConditionID.json) does / does not occur among the
  MIX_large training cells. Needs validation_predictions.npz (else NA).

Usage:  python tools/rescore.py            # table for all runs
        python tools/rescore.py R4-E032    # one run, JSON
        python tools/rescore.py --write    # also fill the evaluation columns of results.tsv
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_CELLS = 10
BATTERYLIFE = ROOT.parent / "BatteryLife"
_TRAIN_CONDITIONS = None


def train_conditions():
    global _TRAIN_CONDITIONS
    if _TRAIN_CONDITIONS is None:
        import importlib.util
        cond = json.loads((BATTERYLIFE / "name2agingConditionID.json").read_text())
        spec = importlib.util.spec_from_file_location("sr", BATTERYLIFE / "data_provider" / "data_split_recorder.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        import sys as _s; _s.path.insert(0, str(ROOT)); import prepare as _p
        train = _p.load_split_lists()["train"]
        _TRAIN_CONDITIONS = (cond, {cond[f] for f in train if f in cond})
    return _TRAIN_CONDITIONS


def seen_unseen(run_id):
    path = ROOT / "experiments" / run_id / "validation_predictions.npz"
    if not path.exists():
        return None, None, None, None
    import numpy as np
    d = np.load(path, allow_pickle=True)
    cond, seen_set = train_conditions()
    y, p, cell = d["target"], d["prediction"], d["cell_id"]
    ae = np.abs(p - y) / y
    known = np.array([c in cond for c in cell])
    seen = np.array([cond.get(c) in seen_set for c in cell]) & known
    unseen = known & ~seen
    f = lambda m: float(ae[m].mean()) if m.any() else None
    g = lambda m: float((ae[m] <= 0.15).mean()) if m.any() else None
    return f(seen), f(unseen), g(seen), g(unseen)


def score(run_id):
    rec = json.loads((ROOT / "experiments" / run_id / "record.json").read_text())
    m = rec["metrics"]
    doms = {k: v for k, v in m["by_domain"].items() if v.get("unique_cell_count", 0) >= MIN_CELLS}
    db = sum(v["mape"] for v in doms.values()) / len(doms) if doms else None
    sm, um, sa, ua = seen_unseen(run_id)
    return {
        "run": run_id,
        "seen_mape": sm, "unseen_mape": um, "seen_acc15": sa, "unseen_acc15": ua,
        "mape": m.get("mape"),
        "acc15": m.get("acc15"),
        "rmse": m.get("rmse"),
        "mae": m.get("mae"),
        "db_mape": db,
        "db_domains": sorted(doms),
        "worst_domain": max(doms, key=lambda k: doms[k]["mape"]) if doms else None,
    }


def write_results_tsv(scored):
    """Fill the extra evaluation columns of results.tsv (amendments A1/A4)."""
    import csv, subprocess
    extra = ["val_rmse", "val_mae", "val_db_mape", "val_seen_mape", "val_unseen_mape",
             "val_unseen_acc15", "physics_violation_rate", "physics_lambda",
             "balance_R_median", "block_health"]
    path = ROOT / "results.tsv"
    csv_rows = list(csv.reader(path.open(), delimiter="\t"))
    by_run = {s["run"]: s for s in scored}
    fm = lambda v, n=4: (f"{v:.{n}f}" if isinstance(v, (int, float)) else "NA")
    out = [csv_rows[0][:18] + extra]
    for r in csv_rows[1:]:
        base = (r + [""] * 18)[:18]
        run_id = base[0]
        vals = ["NA"] * len(extra)
        sc = by_run.get(run_id)
        if sc:
            vals[0:6] = [fm(sc["rmse"], 1), fm(sc["mae"], 1), fm(sc["db_mape"]),
                         fm(sc["seen_mape"]), fm(sc["unseen_mape"]), fm(sc["unseen_acc15"])]
        rec_p = ROOT / "experiments" / run_id / "record.json"
        dia_p = ROOT / "experiments" / run_id / "diagnostics.json"
        if rec_p.exists():
            vals[6] = fm(json.loads(rec_p.read_text()).get("metrics", {}).get("physics_violation_rate"))
        if dia_p.exists():
            dg = json.loads(dia_p.read_text())
            vals[7] = fm(dg.get("calibrated_physics_lambda"))
            lb = dg.get("summary", {}).get("loss_balance", {})
            vals[8] = fm(lb.get("median_R") if isinstance(lb, dict) else None, 3)
            try:
                vals[9] = subprocess.run([sys.executable, str(ROOT / "tools" / "layer_report.py"), run_id, "--short"],
                                         capture_output=True, text=True, timeout=60).stdout.strip() or "NA"
            except Exception:
                vals[9] = "NA"
        out.append(base + vals)
    csv.writer(path.open("w", newline=""), delimiter="\t", lineterminator="\n").writerows(out)
    print(f"updated {path} ({len(extra)} evaluation columns)")


def main():
    write = "--write" in sys.argv
    if write:
        sys.argv.remove("--write")
    runs = sys.argv[1:] or sorted(p.name for p in (ROOT / "experiments").glob("NA-E*") if (p / "record.json").exists())
    if len(sys.argv) == 2:
        print(json.dumps(score(runs[0]), indent=2)); return
    na = lambda v, w=6: f"{v:{w}.3f}" if v is not None else f"{'NA':>{w}}"
    print(f"{'run':9} {'MAPE':>7} {'Acc15':>7} {'RMSE':>8} {'DB-MAPE':>8} {'seen':>6} {'unseen':>7}  worst")
    scored = []
    for r in runs:
        try:
            s = score(r)
            print(f"{r:9} {s['mape']:7.4f} {s['acc15']:7.4f} {s['rmse']:8.1f} {s['db_mape']:8.4f} {na(s['seen_mape'])} {na(s['unseen_mape'],7)}  {s['worst_domain']}")
            scored.append(s)
        except Exception as exc:
            print(f"{r:9} ERROR {exc}")
    if write:
        write_results_tsv(scored)


if __name__ == "__main__":
    main()
