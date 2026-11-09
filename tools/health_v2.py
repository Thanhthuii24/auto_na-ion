"""Re-score saved runs with the v5 health rules (loss.md §5.3), without touching diagnostics.json.
Physics balance: severe only if median weighted gradient ratio > 2.0 (or < 0.05 -> ineffective, WARNING);
loss-value ratio and negative cosine are reported flags. Layers: grad/param ratio with the denominator floored at
1e-2*sqrt(n) (zero-init parameters), parameter norms taken from best_checkpoint.pt. Usage: python tools/health_v2.py ID..."""
import json, math, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
for eid in sys.argv[1:]:
    d = json.loads((ROOT / f"experiments/{eid}/diagnostics.json").read_text())["summary"]
    rec = json.loads((ROOT / f"experiments/{eid}/record.json").read_text())
    met = rec.get("metrics", {})
    pb = d.get("physics_balance", {}); out = {}
    # axis 1 — accuracy (reported separately; never turned into a health failure)
    out["accuracy"] = f"val MAPE {met.get('mape', float('nan')):.4f}, Acc@15 {met.get('acc15', float('nan')):.3f} (compare with the matched reference)"
    # axis 2 — canonical loss.md §5.3 balance gate (loss_balance) + its full rules from the saved samples
    lb = d.get("loss_balance", {})
    if lb.get("samples"):
        R = [s["R"] for s in lb["samples"] if s.get("R") is not None]
        act = [s.get("active_fraction", 0) for s in lb["samples"]]
        consec5 = any(R[i] > 5 and R[i + 1] > 5 for i in range(len(R) - 1))
        inactive = all(a == 0 for a in act)
        out["loss_balance_5_3"] = f"{lb.get('status')} (median R {lb.get('median_R', float('nan')):.2f}; two consecutive R>5: {consec5}; inactive term: {inactive}; samples {len(R)})"
    else:
        out["loss_balance_5_3"] = lb.get("status", "NOT_AVAILABLE")
    if "weighted_to_main_gradient_ratio" in pb:
        gr = pb["weighted_to_main_gradient_ratio"]["median"]; lr = pb["weighted_to_main_loss_ratio"]["median"]
        cs = pb.get("gradient_cosine", {}).get("median")
        flags = (["GRADIENT_DOMINATES(>2.0)"] if gr > 2.0 else []) + (["GRADIENT_INEFFECTIVE(<0.05)"] if gr < 0.05 else []) \
            + (["loss_value_ratio>1 (report)"] if lr > 1 else []) + ([f"cosine {cs:.2f} < -0.3 (report)"] if cs is not None and cs < -0.3 else [])
        st = "UNHEALTHY" if gr > 2.0 else ("WARNING" if flags else "HEALTHY")
        out["physics"] = f"{st} (grad ratio {gr:.2f}, loss ratio {lr:.2f}, cosine {cs if cs is None else round(cs, 2)}) {flags}"
    sd = torch.load(ROOT / f"experiments/{eid}/best_checkpoint.pt", weights_only=False, map_location="cpu")["state_dict"]
    bad = []
    for name, lh in d.get("layer_health", {}).items():
        if isinstance(lh, dict) and "EXPLODING_GRADIENT_CANDIDATE" in lh.get("flags", []):
            ps = [v for k, v in sd.items() if k == name or k.startswith(name + ".")]
            n = sum(v.numel() for v in ps); pn = math.sqrt(sum(float((v.double() ** 2).sum()) for v in ps)) if ps else 0.0
            r = lh["grad_norm"]["last"] / max(pn, 1e-2 * math.sqrt(max(1, n)))
            bad.append(f"{name}: ratio v2 {r:.1f} -> {'EXPLODING' if r > 100 else 'OK'}")
    out["layers_flagged_before"] = bad or "none"
    print(eid, d.get("overall_gradient_health"), "->", out)
