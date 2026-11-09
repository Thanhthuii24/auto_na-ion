"""Phase 3 P3 (v5): post-training quantization variants of the frozen ONNX models, judged on val (rule d_mape <= +0.001).
Variants: fp16 weights (onnxconverter-common if available, else torch half export), INT8 dynamic (weight-only MatMul/Gemm),
INT8 static QDQ (per-channel weights, activations calibrated on 256 TRAIN samples only). Writes phase3/results_tinyml.tsv rows."""
import json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RUNS = {"2021": "NA-E032", "42": "NA-E033", "2024": "NA-E034"}
CODE = r'''
import json, sys, os, numpy as np, torch, onnx, onnxruntime as ort
sys.path.insert(0, ROOT)
import prepare
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, QuantFormat, CalibrationDataReader
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
q = f'{ROOT}/phase3/{EID}_int8_dynamic.onnx'
quantize_dynamic(base, q, weight_type=QuantType.QInt8); out['int8 dynamic (weights)'] = mape(q)
class Cal(CalibrationDataReader):
    def __init__(self):
        g = torch.Generator().manual_seed(0); self.items = []
        for b in tr:
            for i in range(b['cycle_curve_data'].shape[0]):
                self.items.append({'curves': b['cycle_curve_data'][i:i+1].numpy(), 'mask': b['curve_attn_mask'][i:i+1].float().numpy()})
            if len(self.items) >= 256: break
        self.it = iter(self.items)
    def get_next(self): return next(self.it, None)
q2 = f'{ROOT}/phase3/{EID}_int8_static.onnx'
quantize_static(base, q2, Cal(), quant_format=QuantFormat.QDQ, per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QInt8)
out['int8 static QDQ (train-calibrated)'] = mape(q2)
try:
    from onnxconverter_common import float16
    mdl = float16.convert_float_to_float16(onnx.load(base), keep_io_types=True)
    q3 = f'{ROOT}/phase3/{EID}_fp16.onnx'; onnx.save(mdl, q3); out['fp16 weights'] = mape(q3)
except Exception as e:
    out['fp16 weights'] = ('n/a', str(e)[:80])
print(json.dumps(out))
'''
rows = []
for split, eid in RUNS.items():
    env = dict(os.environ, BL_SPLIT=split)
    code = f"ROOT={str(ROOT)!r}; EID={eid!r}\n" + CODE
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-2500:]); sys.exit(f"failed {split}")
    res = json.loads(r.stdout.strip().splitlines()[-1]); rows.append((split, eid, res)); print(split, res)
with (ROOT / "phase3" / "results_tinyml.tsv").open("a") as fh:
    for split, eid, res in rows:
        f32 = res["fp32"][0]
        for k, v in res.items():
            if k == "fp32" or isinstance(v[0], str): continue
            d = v[0] - f32
            fh.write(f"P3\t{split}\t{eid}\t{k}\t{v[0]:.6f}\t{v[1]:.4f}\t{v[2]:.2f}\t{d:+.6f}\t{v[3]:.1f}\t{'PASS' if d <= 0.001 else 'FAIL'} (rule d_mape<=+0.001)\n")
