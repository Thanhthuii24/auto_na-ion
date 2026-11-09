"""Phase 3 P3b (v5): fp16 weights and mixed static INT8 (only the network body: curve_enc Conv, state_proj Gemm/MatMul,
TCN Conv, head Gemm are quantized; the extrapolation arithmetic, capacity ratios, gate and output stay fp32).
Calibration: 256 TRAIN samples. Same val rule d_mape <= +0.001."""
import json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RUNS = {"2021": "NA-E032", "42": "NA-E033", "2024": "NA-E034"}
CODE = r'''
import json, sys, os, numpy as np, torch, onnx, onnxruntime as ort
sys.path.insert(0, ROOT)
import prepare
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader
tr, va, sc = prepare.load_train_val_data(128, 2021)
X, M, T = [], [], []
for b in va: X.append(b['cycle_curve_data'].numpy()); M.append(b['curve_attn_mask'].float().numpy()); T.append(b['raw_eol'].numpy())
t = np.concatenate(T)
def mape(path):
    s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    p = np.concatenate([s.run(None, {'curves': x, 'mask': m})[0] for x, m in zip(X, M)]) * sc.scale + sc.mean
    a = np.abs(p - t) / t
    return float(a.mean()), float((a <= .15).mean()), float(np.sqrt(((p - t) ** 2).mean())), os.path.getsize(path) / 1024
base = f'{ROOT}/phase3/{EID}_fp32.onnx'; out = {'fp32': mape(base)}
# fp16 weight storage: onnxconverter_common fails on this graph and NaFadeNet casts the mask to fp32, so weights are rounded to fp16 and computed in fp32
import train
ck = torch.load(f'{ROOT}/experiments/{EID}/best_checkpoint.pt', weights_only=False, map_location='cpu')
mh = train.build_model(ck['model']); mh.load_state_dict({k: (v.half().float() if v.is_floating_point() else v) for k, v in ck['state_dict'].items()}, strict=False); mh.eval()  # fp16-stored weights, fp32 compute
with torch.no_grad():
    ph = np.concatenate([mh(torch.from_numpy(x), torch.from_numpy(m)).numpy() for x, m in zip(X, M)]) * sc.scale + sc.mean
ah = np.abs(ph - t) / t
out['fp16 weights (fp32 compute)'] = (float(ah.mean()), float((ah <= .15).mean()), float(np.sqrt(((ph - t) ** 2).mean())), sum(q.numel() for q in mh.parameters()) * 2 / 1024)
g = onnx.load(base).graph
body = [n.name for n in g.node if n.op_type in ('Conv', 'Gemm', 'MatMul') and any(k in n.name for k in ('curve_enc', 'state_proj', '/tcn', '/head'))]
class Cal(CalibrationDataReader):
    def __init__(self):
        self.items = []
        for b in tr:
            for i in range(b['cycle_curve_data'].shape[0]):
                self.items.append({'curves': b['cycle_curve_data'][i:i+1].numpy(), 'mask': b['curve_attn_mask'][i:i+1].float().numpy()})
            if len(self.items) >= 256: break
        self.it = iter(self.items)
    def get_next(self): return next(self.it, None)
q2 = f'{ROOT}/phase3/{EID}_int8_static_mixed.onnx'
quantize_static(base, q2, Cal(), quant_format=QuantFormat.QDQ, per_channel=True, weight_type=QuantType.QInt8,
                activation_type=QuantType.QInt8, nodes_to_quantize=body)
out['int8 static mixed (body Conv/Gemm, gate+extrapolation fp32)'] = mape(q2) + (len(body),)
print(json.dumps(out))
'''
rows = []
for split, eid in RUNS.items():
    r = subprocess.run([sys.executable, "-c", f"ROOT={str(ROOT)!r}; EID={eid!r}\n" + CODE], env=dict(os.environ, BL_SPLIT=split), capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-2500:]); sys.exit(f"failed {split}")
    res = json.loads(r.stdout.strip().splitlines()[-1]); rows.append((split, eid, res)); print(split, res)
with (ROOT / "phase3" / "results_tinyml.tsv").open("a") as fh:
    for split, eid, res in rows:
        f32 = res["fp32"][0]
        for k, v in res.items():
            if k == "fp32": continue
            d = v[0] - f32
            fh.write(f"P3\t{split}\t{eid}\t{k}\t{v[0]:.6f}\t{v[1]:.4f}\t{v[2]:.2f}\t{d:+.6f}\t{v[3]:.1f}\t{'PASS' if d <= 0.001 else 'FAIL'} (rule d_mape<=+0.001)\n")
