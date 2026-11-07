#!/usr/bin/env python3
"""Amendment A6 (user-approved 2026-11-03): SOC-window-normalized capacity channel.

Derived from the locked v1 cache (hash-verified). For every cell:
    channel 2 (capacity / nominal)  ->  capacity / (nominal * SOC-window width)
the same normalization as the BatteryLife EOL label. SOC width comes from cell
metadata (SOC_interval), an experimental setting known at prediction time.
No label, trajectory or outcome is read; test cells are transformed only so
that a later authorized final evaluation sees the same input definition.
"""
import json, pickle, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import prepare

OUT = ROOT / ".cache" / "batterylife_mix_large_v2_socnorm.pt"

def main():
    t = time.perf_counter()
    payload = prepare._load_locked_cache()
    widths = {}
    for split, cells in payload["cells"].items():
        for cell in cells:
            meta = pickle.load(prepare.resolve_cell_path(cell["filename"]).open("rb"))
            soc = meta.get("SOC_interval") or [0, 1]
            w = float(soc[1]) - float(soc[0])
            if not (0.05 < w <= 1.0):
                raise ValueError(f"bad SOC width {w} for {cell['filename']}")
            cell["tensor"] = cell["tensor"].clone()
            cell["tensor"][:, 2, :] = cell["tensor"][:, 2, :] / w
            cell["soc_width"] = w
            widths[cell["filename"]] = w
        print(f"[{split}] {len(cells)} cells", flush=True)
    payload["protocol_version"] = "BATTERYLIFE-MIX-LARGE-V2-SOCNORM"
    payload["derived_from"] = "batterylife_mix_large_v1.pt"
    torch.save(payload, OUT)
    audit = {"soc_widths": widths, "n_narrow": sum(w < 0.999 for w in widths.values()),
             "wall_seconds": round(time.perf_counter() - t, 1)}
    OUT.with_suffix(".audit.json").write_text(json.dumps(audit, indent=1))
    print(f"saved {OUT.name} sha256={prepare.sha256(OUT)} narrow={audit['n_narrow']}/{len(widths)}")

if __name__ == "__main__":
    main()
