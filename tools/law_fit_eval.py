"""Held-out degradation-law consistency (NA-E025 reject-rule metric). EVALUATION ONLY — never used in training.
For each VAL cell: real capacity trajectory q(n) up to its label EOL (raw pickle, same smoothing/normalisation as
train.load_law_targets); predicted EOL y_hat from the checkpoint at prefixes 20/50/100 (eval mode); law error =
min over b in [0.3, 4] of mean|q_ref - (q_ref - q_eol)(n/y_hat)^b - q(n)| / (q_ref - q_eol). Also the true-EOL floor.
Usage: BL_SPLIT=<s> python tools/law_fit_eval.py <ID>"""
import math, pickle, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import prepare, train

def val_traj(fn, k=train.LAW_K):
    cell = pickle.load(prepare.resolve_cell_path(fn).open("rb")); eol = prepare.load_eol(fn)
    cap = prepare.nominal_capacity(fn, cell); soc = cell.get("SOC_interval") or [0, 1]; w = float(soc[1]) - float(soc[0]) or 1.0
    q = []
    for cyc in cell["cycle_data"]:
        d = np.asarray(cyc["discharge_capacity_in_Ah"], dtype=np.float64); d = d[np.isfinite(d) & (d >= 0)]
        q.append(d.max() / (cap * w) if d.size else np.nan)
    q = np.asarray(q); n_end = int(min(len(q), math.ceil(eol))); qs = q[:n_end].copy(); g = np.isfinite(qs)
    qs = np.interp(np.arange(n_end), np.where(g)[0], qs[g]); pad = np.pad(qs, 4, mode="edge")
    qs = np.median(np.lib.stride_tricks.sliding_window_view(pad, 9), axis=1)
    n = np.resize(np.unique(np.round(np.linspace(1, n_end, k)).astype(int)), k)
    return n.astype(float), qs[n - 1], float(np.median(qs[:10])), float(qs[n_end - 1])

BS = np.linspace(0.3, 4, 75)
def law_err(n, q, qr, qe, y):
    return min(np.abs(qr - (qr - qe) * (n / y) ** b - q).mean() / (qr - qe) for b in BS)

eid = sys.argv[1]
ck = torch.load(ROOT / f"experiments/{eid}/best_checkpoint.pt", weights_only=False, map_location="cpu")
m = train.build_model(ck["model"]); m.load_state_dict(ck["state_dict"], strict=False); m.eval()
d = torch.load(ROOT / f".cache/batterylife_naion_split{prepare.SPLIT}_v1.pt", weights_only=False); sc = d["label_scaler"]
rows, floor = [], []
for c in d["cells"]["val"]:
    n, q, qr, qe = val_traj(c["filename"]); floor.append(law_err(n, q, qr, qe, c["eol"]))
    for P in (20, 50, 100):
        mk = torch.zeros(1, 100); mk[0, :P] = 1
        with torch.no_grad():
            y = float(m(c["tensor"][None] * mk[:, :, None, None], mk)) * sc["scale"] + sc["mean"]
        rows.append(law_err(n, q, qr, qe, max(y, 20.0)))
print(f"{eid} split {prepare.SPLIT}: val law-fit error {np.mean(rows):.4f} (true-EOL floor {np.mean(floor):.4f})")
