#!/usr/bin/env python3
"""Amendment A7 (user-approved 2026-11-03): v2 input + cell test-condition metadata.

Derived from the hash-verified v2 cache. Each cell tensor gets a 4th channel that is zero except
tensor[0, 3, 0:4] = [log(SOC width), SOC start, log1p(charge C-rate), log1p(discharge C-rate)]
(missing C-rate -> 0). These are experimental settings (BatteryLife metadata) known before prediction.
Physics motivation: Woehler-type cycle-life law, life ~ DoD^-k, plus rate stress.
"""
import json, math, pickle, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import prepare
OUT = ROOT / ".cache" / "batterylife_mix_large_v3_meta.pt"

def rate(proto):
    try:
        v = float(proto[0].get("rate_in_C"))
        return math.log1p(v) if math.isfinite(v) and v > 0 else 0.0
    except Exception:
        return 0.0

def main():
    lock = json.loads((ROOT / "protocol_lock.json").read_text())
    payload = prepare._load_locked_cache(lock["cache_v2_path"])
    meta_all = {}
    for split, cells in payload["cells"].items():
        for cell in cells:
            d = pickle.load(prepare.resolve_cell_path(cell["filename"]).open("rb"))
            soc = d.get("SOC_interval") or [0, 1]
            w = float(soc[1]) - float(soc[0])
            m = [math.log(w), float(soc[0]), rate(d.get("charge_protocol")), rate(d.get("discharge_protocol"))]
            t = cell["tensor"]
            extra = torch.zeros(t.shape[0], 1, t.shape[2], dtype=t.dtype)
            extra[0, 0, :4] = torch.tensor(m, dtype=t.dtype)
            cell["tensor"] = torch.cat([t, extra], dim=1)
            cell["meta"] = m; meta_all[cell["filename"]] = m
        print(f"[{split}] {len(cells)} cells", flush=True)
    payload["protocol_version"] = "BATTERYLIFE-MIX-LARGE-V3-META"; payload["derived_from"] = "batterylife_mix_large_v2_socnorm.pt"
    torch.save(payload, OUT)
    OUT.with_suffix(".audit.json").write_text(json.dumps({"meta": meta_all, "fields": ["log_soc_width", "soc_start", "log1p_charge_C", "log1p_discharge_C"]}, indent=1))
    print(f"saved {OUT.name} sha256={prepare.sha256(OUT)}")

if __name__ == "__main__":
    main()
