"""Export the frozen v5 N1 models (NA-E032/E033/E034) as ST-Edge-AI-compatible ONNX (batch 1, static shapes).
Change vs train.NaFadeNet (math identical): the per-cycle Conv1d encoder applied to B*T=100 cycles as a batch is
rewritten as Conv2d with kernel (1,k) on [1,3,100,300], because ST does not support Conv with batch > 1.
Checks onnxruntime(simplified ONNX) vs PyTorch on every val prefix (relative EOL difference)."""
import os, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import onnx, onnxsim, onnxruntime as ort

class STNaFadeNet(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def encode(self, curves):                                   # [1,T,3,300] -> [1,T,C]
        x = curves.permute(0, 2, 1, 3)                          # [1,3,T,300]
        for layer in self.m.curve_enc:
            if isinstance(layer, nn.Conv1d):
                x = F.conv2d(x, layer.weight.unsqueeze(2), layer.bias, stride=(1, layer.stride[0]), padding=(0, layer.padding[0]))
            elif isinstance(layer, nn.AdaptiveAvgPool1d):
                x = x.mean(-1)                                  # [1,C,T]
            else:
                x = layer(x)
        return x.permute(0, 2, 1)
    def forward(self, curves, mask):
        m = self.m; T = curves.shape[1]; mk = mask
        h = self.encode(curves); parts = [h]
        if m.use_anchor: parts.append(h - h[:, :1])
        if m.use_fade:
            qd = curves[:, :, 2, 150:].amax(-1); qc = curves[:, :, 2, :150].amax(-1)
            parts.append(torch.stack([(qd / qd[:, :1].clamp_min(1e-3) - 1.0) * m.fade_scale,
                                      (qc / qc[:, :1].clamp_min(1e-3) - 1.0) * m.fade_scale], -1))
        x = m.state_proj(torch.cat(parts, -1)) * mk[..., None]
        for conv, norm in zip(m.tcn, m.tcn_norm):
            pad = (conv.kernel_size[0] - 1) * conv.dilation[0]
            y = conv(F.pad(x.transpose(1, 2), (pad, 0))).transpose(1, 2)
            x = (x + F.gelu(norm(y))) * mk[..., None]
        n = mk.sum(1).clamp_min(1.0)
        mean = x.sum(1) / n[:, None]
        last = x.gather(1, (n - 1).long()[:, None, None].expand(-1, 1, x.size(-1))).reshape(1, -1)
        out = m.head(torch.cat([mean, last, (n / T)[:, None]], -1)).reshape(1)
        est, slope, resid = m._extrapolate(curves[:, :, 2, 150:].amax(-1) * mk, mk)
        feats = torch.stack([n / T, est / 100.0, (slope * 1000.0).clamp(-10, 10), (resid * 100.0).clamp(0, 10)], -1)
        s = m.gate_param_scale
        g = torch.sigmoid(-2.0 + s * m.gate(feats)).reshape(1)
        z = (2.0 + s * m.ext_da) * est / 100.0 + (-3.6 + s * m.ext_dc)
        return (1 - g) * out + g * z

RUNS = {"2021": "NA-E032", "42": "NA-E033", "2024": "NA-E034"}
if __name__ == "__main__":
    import train, prepare
    split = os.environ.get("BL_SPLIT", "2021"); eid = RUNS[split]
    ck = torch.load(ROOT / f"experiments/{eid}/best_checkpoint.pt", weights_only=False, map_location="cpu")
    base = train.build_model(ck["model"]); base.load_state_dict(ck["state_dict"], strict=False); base.eval()
    w = STNaFadeNet(base).eval()
    out = ROOT / "phase3" / "board" / f"{eid}_st.onnx"
    x0, m0 = torch.rand(1, 100, 3, 300), torch.ones(1, 100)
    torch.onnx.export(w, (x0, m0), str(out), input_names=["curves", "mask"], output_names=["y"], opset_version=17, dynamic_axes=None)
    sim, ok = onnxsim.simplify(onnx.load(str(out))); assert ok; onnx.save(sim, str(out))
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    _, va, sc = prepare.load_train_val_data(1, 2021); worst = 0.0; X = []; M = []; Y = []
    with torch.no_grad():
        for b in va:
            x, mk = b["cycle_curve_data"], b["curve_attn_mask"].float()
            yt = float(base(x, mk)); yo = float(sess.run(None, {"curves": x.numpy(), "mask": mk.numpy()})[0][0])
            e_t, e_o = yt * sc.scale + sc.mean, yo * sc.scale + sc.mean
            worst = max(worst, abs(e_o - e_t) / abs(e_t)); X.append(x.numpy()); M.append(mk.numpy()); Y.append(yt)
    np.savez(ROOT / "phase3" / "board" / f"{eid}_val_io.npz", curves=np.concatenate(X), mask=np.concatenate(M), y_torch=np.array(Y, dtype=np.float32),
             eol=np.array([float(b["raw_eol"][0]) for b in va]), mean=sc.mean, scale=sc.scale)
    ops = sorted({n.op_type for n in sim.graph.node})
    print(f"{eid} split {split}: {len(Y)} val prefixes, max rel EOL diff ONNX vs PyTorch {worst:.2e} ({'PASS' if worst <= 1e-4 else 'FAIL'}); ops {ops}")
