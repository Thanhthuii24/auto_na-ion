"""Post-hoc held-out prefix drift (NA-E023 reject-rule metric, reviewer-specified).
Eval mode, truncate_prefix_view with torch.Generator().manual_seed(0), mean |f(x) - f(x_trunc)| * scale / raw_eol,
plus the same quantity per prefix band. Usage: BL_SPLIT=<s> python tools/posthoc_drift.py <ID>"""
import sys
import numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import prepare, train
eid = sys.argv[1]
ck = torch.load(f"experiments/{eid}/best_checkpoint.pt", weights_only=False, map_location="cpu")
m = train.build_model(ck["model"]); m.load_state_dict(ck["state_dict"]); m.eval()
_, va, sc = prepare.load_train_val_data(128, 2021)
g = torch.Generator().manual_seed(0); R, N = [], []
with torch.no_grad():
    for b in va:
        x, mk, y = b["cycle_curve_data"], b["curve_attn_mask"], b["raw_eol"]
        xt, mt = train.truncate_prefix_view(x, mk, g)
        R.append(((m(x, mk) - m(xt, mt)).abs() * sc.scale / y).numpy()); N.append(mk.sum(1).numpy())
R, N = np.concatenate(R), np.concatenate(N)
bands = {f"{a}-{c}": round(float(R[(N >= a) & (N <= c)].mean()), 4) for a, c in [(1, 10), (11, 25), (26, 50), (51, 75), (76, 100)]}
print(f"{eid} split {prepare.SPLIT}: mean relative drift {R.mean():.4f}  bands {bands}")
