"""A11 diagnostic: train-set fit of a run's best checkpoint (diagnostic only; never a reported result).
Usage: python tools/train_tail.py R4-E0xx  -> writes experiments/<ID>/train_tail.json"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import prepare, train

def main(eid, n=4000, seed=3):
    prop = json.loads((ROOT / "experiments" / f".{eid}_proposal.json").read_text())
    norm = prop["training"].get("input_normalization", "nominal")
    lock = json.loads((ROOT / "protocol_lock.json").read_text())
    cache = {"soc_window": lock.get("cache_v2_path"), "soc_window_meta": lock.get("cache_v3_path")}.get(norm)
    tl, _, sc = prepare.load_train_val_data(32, prop["training"]["seed"], cache_path=cache)
    m = train.build_model(prop["model"])
    if hasattr(m, "set_reference_bank"): m.set_reference_bank(tl.dataset)
    ck = torch.load(ROOT / "experiments" / eid / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("model_state_dict", ck.get("state_dict", ck)), strict=False); m.eval()
    ds = tl.dataset; idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:n].tolist()
    c = torch.stack([ds[i]["cycle_curve_data"] for i in idx]).float(); k = torch.stack([ds[i]["curve_attn_mask"] for i in idx])
    with torch.no_grad(): z = torch.cat([m(c[j:j+500], k[j:j+500]) for j in range(0, len(idx), 500)]).numpy()
    p = z * sc.scale + sc.mean; y = np.array([float(ds[i]["raw_eol"]) for i in idx]); cid = np.array([ds[i]["cell_id"] for i in idx])
    ape = np.abs(p - y) / y; tail = y > 2000
    cells = np.unique(cid); cell_bal = float(np.mean([ape[cid == c_].mean() for c_ in cells]))
    out = {"note": "diagnostic only (A11); not a reported metric", "samples": len(idx), "cells": int(len(cells)), "seed": seed,
           "train_mape_prefix_weighted": float(ape.mean()), "train_mape_cell_balanced": cell_bal,
           "train_acc15": float((ape <= 0.15).mean()), "eol_gt2000_mape": float(ape[tail].mean()),
           "eol_gt2000_bias": float(((p - y) / y)[tail].mean()), "eol_gt2000_samples": int(tail.sum()), "max_prediction": float(p.max())}
    (ROOT / "experiments" / eid / "train_tail.json").write_text(json.dumps(out, indent=1)); print(eid, json.dumps(out))

if __name__ == "__main__":
    main(sys.argv[1])
