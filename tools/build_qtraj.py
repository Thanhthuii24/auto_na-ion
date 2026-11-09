#!/usr/bin/env python3
"""Sidecar for the Phase-2 degradation-law loss (user-approved 2026-11-03).

TRAIN cells only (v5: train list of the paper split selected by BL_SPLIT). For every recorded cycle n:
    q(n) = max discharge capacity in cycle n / (nominal * SOC_interval width)
the same normalisation as the BatteryLife EOL label (q <= 0.8 -> end of life).
Writes .cache/qtraj_train_v1.pt = {filename: {"q": float32[N], "eol": float, "cross": first n with q<=0.8 or None}}.
Validation/test cells are never read. The locked cache is untouched.
"""
import pickle, sys, math
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import prepare

def main():
    train = prepare.load_split_lists()["train"] if isinstance(prepare.load_split_lists(), dict) else None
    if train is None:
        train = prepare._load_split_recorder().MIX_large_train_files
    out, skipped = {}, []
    for k, fn in enumerate(train):
        path = prepare.resolve_cell_path(fn); eol = prepare.load_eol(fn)
        if not path.exists() or eol is None or eol <= prepare.EARLY_CYCLES:
            skipped.append(fn); continue
        cell = pickle.load(path.open("rb"))
        cap = prepare.nominal_capacity(fn, cell)
        soc = cell.get("SOC_interval") or [0, 1]
        width = float(soc[1]) - float(soc[0]) or 1.0
        q = []
        for cyc in cell["cycle_data"]:
            d = np.asarray(cyc["discharge_capacity_in_Ah"], dtype=np.float64)
            d = d[np.isfinite(d) & (d >= 0)]
            q.append(d.max() / (cap * width) if d.size else np.nan)
        q = np.asarray(q, dtype=np.float32)
        below = np.where(q <= 0.8)[0]
        out[fn] = {"q": torch.from_numpy(q), "eol": float(eol), "cross": int(below[0]) + 1 if below.size else None}
        print(f"{k+1}/{len(train)} {fn} N={len(q)} eol={eol:.0f} cross={out[fn]['cross']}", flush=True)
    torch.save(out, ROOT / ".cache" / f"qtraj_train_split{prepare.SPLIT}_v1.pt")  # v5: one file per paper split (env BL_SPLIT)
    print(f"saved {len(out)} cells, skipped {len(skipped)}")

if __name__ == "__main__":
    main()
