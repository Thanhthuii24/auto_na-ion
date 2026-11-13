"""Phase 3 board, step A2 (v5): split the frozen N1 model for STM32 (ST Edge AI) streaming deployment.
  E (ONNX): one cycle curve [1,3,300] -> h [1,32]                      (per new cycle on the MCU)
  H (ONNX): states [1,100,66] + mask [1,100] -> y_net [1,1]            (state_proj, causal TCN, masked mean, last, head)
  C (plain code, here a NumPy reference): states = [h, h - h_1, r_d, r_c]; masked line fit on the last 20 q_d values
     -> extrapolated EOL; gate MLP 4->16->1 (GELU); y = (1-g) y_net + g z_ext.
Checks: E + H + C reproduces the PyTorch model on every val prefix (relative EOL difference). Writes gate weights to JSON."""
import json, math, os, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import onnx, onnxsim, onnxruntime as ort
RUNS = {"2021": "NA-E032", "42": "NA-E033", "2024": "NA-E034"}

class Enc(nn.Module):
    def __init__(s, m): super().__init__(); s.enc = m.curve_enc
    def forward(s, curve): return s.enc(curve).reshape(1, -1)              # [1,3,300] -> [1,32]

class Head(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, states, mk):                                               # [1,100,66], [1,100]
        m = s.m; x = m.state_proj(states) * mk[..., None]
        for conv, norm in zip(m.tcn, m.tcn_norm):
            pad = (conv.kernel_size[0] - 1) * conv.dilation[0]
            y = conv(F.pad(x.transpose(1, 2), (pad, 0))).transpose(1, 2)
            x = (x + F.gelu(norm(y))) * mk[..., None]
        n = mk.sum(1).clamp_min(1.0)
        mean = x.sum(1) / n[:, None]
        onehot = mk - torch.cat([mk[:, 1:], torch.zeros_like(mk[:, :1])], 1)   # 1 at the last observed cycle
        last = (x * onehot[..., None]).sum(1)
        return m.head(torch.cat([mean, last, (n / mk.shape[1])[:, None]], -1))

def gelu(v): return 0.5 * v * (1.0 + np.vectorize(math.erf)(v / math.sqrt(2.0)))

def c_part(qd, n, y_net, G):
    """NumPy reference of the C code: masked line fit on the last 20 observed q_d, gate, mixing (float64)."""
    t = np.arange(1, n + 1, dtype=np.float64); q = np.asarray(qd[:n], np.float64)
    w = (t > n - G["window"]).astype(np.float64); sw = max(w.sum(), 1.0)
    mt, mq = (w * t).sum() / sw, (w * q).sum() / sw
    var = (w * (t - mt) ** 2).sum(); slope = (w * (t - mt) * (q - mq)).sum() / max(var, 1e-6)
    inter = mq - slope * mt; resid = math.sqrt((w * (q - inter - slope * t) ** 2).sum() / sw)
    est = (G["thr"] - inter) / min(slope, -1e-8) if (slope < 0 and n >= 3) else G["cap"]
    est = max(min(est, G["cap"]), n + 1)
    f = np.array([n / 100.0, est / 100.0, min(max(slope * 1000.0, -10), 10), min(max(resid * 100.0, 0), 10)])
    hdn = gelu(np.array(G["W1"]) @ f + np.array(G["b1"])); logit = float(np.array(G["W2"]) @ hdn + G["b2"])
    s = G["s"]; g = 1.0 / (1.0 + math.exp(-(-2.0 + s * logit)))
    z = (2.0 + s * G["da"]) * est / 100.0 + (-3.6 + s * G["dc"])
    return (1 - g) * y_net + g * z

def export(mod, args, names, path):
    torch.onnx.export(mod.eval(), args, str(path), input_names=names, output_names=["y"], opset_version=17, dynamic_axes=None)
    s, ok = onnxsim.simplify(onnx.load(str(path))); assert ok; onnx.save(s, str(path)); return sorted({n.op_type for n in s.graph.node})

if __name__ == "__main__":
    import train, prepare
    split = os.environ.get("BL_SPLIT", "2021"); eid = RUNS[split]; out = ROOT / "phase3" / "board"
    ck = torch.load(ROOT / f"experiments/{eid}/best_checkpoint.pt", weights_only=False, map_location="cpu")
    m = train.build_model(ck["model"]); m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.requires_grad_(False)
    opsE = export(Enc(m), (torch.rand(1, 3, 300),), ["curve"], out / f"{eid}_E.onnx")
    opsH = export(Head(m), (torch.rand(1, 100, 66), torch.ones(1, 100)), ["states", "mask"], out / f"{eid}_H.onnx")
    G = {"window": m.extrap_window, "thr": m.extrap_threshold, "cap": m.extrap_cap, "s": m.gate_param_scale,
         "W1": m.gate[0].weight.tolist(), "b1": m.gate[0].bias.tolist(), "W2": m.gate[2].weight[0].tolist(),
         "b2": float(m.gate[2].bias[0]), "da": float(m.ext_da), "dc": float(m.ext_dc), "fade_scale": m.fade_scale}
    (out / f"{eid}_C_params.json").write_text(json.dumps(G))
    sE = ort.InferenceSession(str(out / f"{eid}_E.onnx"), providers=["CPUExecutionProvider"])
    sH = ort.InferenceSession(str(out / f"{eid}_H.onnx"), providers=["CPUExecutionProvider"])
    d = torch.load(ROOT / f".cache/batterylife_naion_split{split}_v1.pt", weights_only=False); sc = d["label_scaler"]
    worst, n_eval = 0.0, 0
    for c in d["cells"]["val"]:
        X = c["tensor"].numpy(); nmax = min(100, int(np.ceil(c["eol"])) - 1)
        h = np.stack([sE.run(None, {"curve": X[i:i + 1]})[0][0] for i in range(nmax)])          # per-cycle encoder
        qd, qc = X[:, 2, 150:].max(1), X[:, 2, :150].max(1)
        for n in range(1, nmax + 1):
            mk = np.zeros((1, 100), np.float32); mk[0, :n] = 1
            st = np.zeros((1, 100, 66), np.float32)
            st[0, :n, :32] = h[:n]; st[0, :n, 32:64] = h[:n] - h[0]
            st[0, :n, 64] = (qd[:n] / max(qd[0], 1e-3) - 1.0) * G["fade_scale"]; st[0, :n, 65] = (qc[:n] / max(qc[0], 1e-3) - 1.0) * G["fade_scale"]
            y_net = float(sH.run(None, {"states": st, "mask": mk})[0][0, 0])
            y = c_part(qd, n, y_net, G)
            with torch.no_grad():
                yt = float(m(torch.from_numpy(X[None] * mk[0][None, :, None, None]), torch.from_numpy(mk)))
            e, et = y * sc["scale"] + sc["mean"], yt * sc["scale"] + sc["mean"]
            worst = max(worst, abs(e - et) / abs(et)); n_eval += 1
    print(f"{eid} split {split}: E+H (ONNX) + C reference vs PyTorch on {n_eval} val prefixes: max rel EOL diff {worst:.2e} "
          f"({'PASS' if worst <= 1e-4 else 'FAIL'} <= 1e-4)\n  E ops {opsE}\n  H ops {opsH}")
