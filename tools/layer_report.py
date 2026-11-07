#!/usr/bin/env python3
"""Per-layer/block learning report from diagnostics.json (no training).

For each monitored block: grad/param ratio early (first 3 samples) and late
(last 3), late update ratio, late near-zero-gradient fraction, and a status:
  DEAD      late near-zero gradient fraction > 0.50       (program.md §5)
  FROZEN    late update ratio < 1e-8
  WEAK      late grad/param ratio < 0.05 x median of all blocks (learns much less)
  EXPLODING late grad/param ratio > 100
  OK        otherwise
Usage: python tools/layer_report.py R4-E040 [--short]
"""
import json, sys, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def report(run_id):
    d = json.loads((ROOT / "experiments" / run_id / "diagnostics.json").read_text())
    samples = [s for s in d.get("samples", []) if s.get("gradients")]
    if not samples:
        return []
    blocks = list(samples[0]["gradients"])
    def avg(rows, block, key, sub="gradients"):
        vals = [r.get(sub, {}).get(block, {}) for r in rows]
        vals = [v.get(key) if isinstance(v, dict) else v for v in vals]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else None
    early, late = samples[:3], samples[-3:]
    rows = []
    for b in blocks:
        rows.append({
            "block": b,
            "ratio_early": avg(early, b, "grad_to_parameter_ratio"),
            "ratio_late": avg(late, b, "grad_to_parameter_ratio"),
            "near_zero_late": avg(late, b, "near_zero_gradient_fraction"),
            "update_late": avg(late, b, "update_ratio", sub="update_ratios") if isinstance(late[-1].get("update_ratios", {}).get(b), dict) else avg(late, b, None, sub="update_ratios"),
        })
    med = statistics.median([r["ratio_late"] for r in rows if r["ratio_late"]] or [0])
    for r in rows:
        s = "OK"
        if r["near_zero_late"] is not None and r["near_zero_late"] > 0.5: s = "DEAD"
        elif r["update_late"] is not None and r["update_late"] < 1e-8: s = "FROZEN"
        elif r["ratio_late"] is not None and r["ratio_late"] > 100: s = "EXPLODING"
        elif r["ratio_late"] is not None and med and r["ratio_late"] < 0.05 * med: s = "WEAK"
        r["status"] = s
    return rows


def main():
    run = sys.argv[1]
    rows = report(run)
    if "--short" in sys.argv:
        bad = [f"{r['block']}:{r['status']}" for r in rows if r["status"] != "OK"]
        print(";".join(bad) if bad else "all_OK"); return
    f = lambda v: f"{v:9.2e}" if isinstance(v, (int, float)) else "       NA"
    print(f"{'block':28} {'g/p early':>9} {'g/p late':>9} {'upd late':>9} {'zero late':>9}  status")
    for r in rows:
        print(f"{r['block'][:28]:28} {f(r['ratio_early'])} {f(r['ratio_late'])} {f(r['update_late'])} {f(r['near_zero_late'])}  {r['status']}")


if __name__ == "__main__":
    main()
