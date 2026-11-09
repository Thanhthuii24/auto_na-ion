"""Phase 3 P1 (v5): per-cycle streaming inference of the frozen NaFadeNet+ExtrapGate model (reference for the MCU C code).
Each new cycle: encode its curve once (CNN), build its state, run each causal TCN layer only at the new position using
cached past layer outputs (kernel 3, dilation d -> needs positions t, t-d, t-2d), update running sums for the masked mean,
update the q_d history for the gate's line fit. Checks equality with the full-window model at EVERY prefix of every val cell.
Reports the per-model streaming RAM (cached states) and the max relative difference (target <= 1e-4)."""
import os, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import prepare, train

class Streamer:
    def __init__(self, m):
        self.m = m; self.reset()
    def reset(self):
        self.layers = [[] for _ in range(len(self.m.tcn) + 1)]   # cached outputs per level: level 0 = state_proj output
        self.h1 = None; self.qd = []; self.qc = []; self.sum_last = None
    @torch.no_grad()
    def step(self, curve):                                          # curve [3,300] -> prediction (standardised) for prefix n
        m = self.m
        h = m.curve_enc(curve[None]).reshape(1, -1)[0]              # [C]
        if self.h1 is None: self.h1 = h
        qd, qc = float(curve[2, 150:].max()), float(curve[2, :150].max())
        self.qd.append(qd); self.qc.append(qc)
        parts = [h]
        if m.use_anchor: parts.append(h - self.h1)
        if m.use_fade:
            rd = (qd / max(self.qd[0], 1e-3) - 1.0) * m.fade_scale; rc = (qc / max(self.qc[0], 1e-3) - 1.0) * m.fade_scale
            parts.append(torch.tensor([rd, rc], dtype=h.dtype))
        x = m.state_proj(torch.cat(parts)); self.layers[0].append(x)
        t = len(self.layers[0]) - 1
        for l, (conv, norm) in enumerate(zip(m.tcn, m.tcn_norm)):
            d = conv.dilation[0]; prev = self.layers[l]
            taps = [prev[t - 2 * d] if t - 2 * d >= 0 else torch.zeros_like(x),
                    prev[t - d] if t - d >= 0 else torch.zeros_like(x), prev[t]]
            y = sum(F.linear(tap, conv.weight[:, :, k]) for k, tap in enumerate(taps)) + conv.bias
            x = prev[t] + F.gelu(norm(y)); self.layers[l + 1].append(x)
        n = t + 1; top = torch.stack(self.layers[-1])
        z = torch.cat([top.mean(0), top[-1], torch.tensor([n / 100.0], dtype=x.dtype)])
        out = m.head(z[None])[0, 0]
        if not m.extrap_gate: return out
        q = torch.tensor(self.qd + [0.0] * (100 - n), dtype=x.dtype)[None]; msk = torch.zeros(1, 100); msk[0, :n] = 1
        est, slope, resid = m._extrapolate(q * msk, msk)
        feats = torch.stack([torch.tensor([n / 100.0]), est / 100.0, (slope * 1000.0).clamp(-10, 10), (resid * 100.0).clamp(0, 10)], -1)
        s = m.gate_param_scale
        g = torch.sigmoid(-2.0 + s * m.gate(feats)).squeeze(-1) if m.fixed_gate is None else torch.full_like(est, m.fixed_gate)
        z_ext = (2.0 + s * m.ext_da) * est / 100.0 + (-3.6 + s * m.ext_dc)
        return ((1 - g) * out + g * z_ext)[0]

for split, eid in {"2021": "NA-E032", "42": "NA-E033", "2024": "NA-E034"}.items():
    os.environ["BL_SPLIT"] = split
    import importlib; importlib.reload(prepare)
    ck = torch.load(ROOT / f"experiments/{eid}/best_checkpoint.pt", weights_only=False, map_location="cpu")
    m = train.build_model(ck["model"]); m.load_state_dict(ck["state_dict"], strict=False); m.eval()
    d = torch.load(ROOT / f".cache/batterylife_naion_split{split}_v1.pt", weights_only=False)
    worst = 0.0; worst_std = 0.0; sc = d["label_scaler"]; S = Streamer(m)
    for c in d["cells"]["val"]:
        S.reset(); n_max = min(100, int(np.ceil(c["eol"])) - 1)
        for n in range(1, n_max + 1):
            ys = S.step(c["tensor"][n - 1])
            mk = torch.zeros(1, 100); mk[0, :n] = 1
            with torch.no_grad(): yf = m(c["tensor"][None] * mk[:, :, None, None], mk)[0]
            eol_s, eol_f = float(ys) * sc["scale"] + sc["mean"], float(yf) * sc["scale"] + sc["mean"]
            worst = max(worst, abs(eol_s - eol_f) / abs(eol_f))          # relative difference of the predicted EOL (cycles)
            worst_std = max(worst_std, float((ys - yf).abs()))
    D = m.state_proj.out_features; L = len(m.tcn)
    ram_kb = (100 * D * (L + 1) * 4 + 100 * 2 * 4 + 3 * 300 * 4) / 1024
    print(f"split {split} {eid}: max relative EOL diff streaming vs full window {worst:.2e} (max abs standardised diff {worst_std:.1e}) ({'PASS' if worst <= 1e-4 else 'FAIL'} <=1e-4) | streaming state RAM ~{ram_kb:.1f} KB (vs full-window input {100*3*300*4/1024:.0f} KB)")
