"""Phase 3 P0 (v5): fp32 reference of the frozen model on the host CPU + ONNX export check.
For each split: rebuild the frozen checkpoint, recompute val MAPE/Acc@15/RMSE on CPU (must match record.json),
export ONNX (opset 17), compare onnxruntime vs PyTorch on all val samples, and report parameter bytes.
Usage: python phase3/p0_reference.py   (writes phase3/results_tinyml.tsv rows)"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
RUNS = {"2021": "NA-E032", "42": "NA-E033", "2024": "NA-E034"}

def run_split(split, eid):
    code = f"""
import json, sys, numpy as np, torch, onnxruntime as ort
sys.path.insert(0, '{ROOT}')
import prepare, train
ck = torch.load('{ROOT}/experiments/{eid}/best_checkpoint.pt', weights_only=False, map_location='cpu')
m = train.build_model(ck['model']); m.load_state_dict(ck['state_dict'], strict=False); m.eval()
_, va, sc = prepare.load_train_val_data(128, 2021)
P, T, X, M = [], [], [], []
with torch.no_grad():
    for b in va:
        x, mk = b['cycle_curve_data'], b['curve_attn_mask'].float()
        P.append(m(x, mk) * sc.scale + sc.mean); T.append(b['raw_eol']); X.append(x); M.append(mk)
p, t = torch.cat(P).numpy(), torch.cat(T).numpy(); ape = np.abs(p - t) / t
rec = json.load(open('{ROOT}/experiments/{eid}/record.json'))['metrics']
onnx_path = '{ROOT}/phase3/{eid}_fp32.onnx'
torch.onnx.export(m, (X[0][:1], M[0][:1]), onnx_path, input_names=['curves', 'mask'], output_names=['y'],
                  dynamic_axes={{'curves': {{0: 'B'}}, 'mask': {{0: 'B'}}, 'y': {{0: 'B'}}}}, opset_version=17)
s = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
po = np.concatenate([s.run(None, {{'curves': x.numpy(), 'mask': mk.numpy()}})[0] for x, mk in zip(X, M)]) * sc.scale + sc.mean
ape_o = np.abs(po - t) / t
nparam = sum(q.numel() for q in m.parameters())
print(json.dumps({{'split': '{split}', 'eid': '{eid}', 'mape_cpu': float(ape.mean()), 'mape_record': rec['mape'], 'acc15': float((ape <= .15).mean()),
  'rmse': float(np.sqrt(((p - t) ** 2).mean())), 'mape_onnx': float(ape_o.mean()), 'max_rel_diff_onnx': float(np.max(np.abs(po - p) / np.abs(p))),
  'params': nparam, 'fp32_kb': nparam * 4 / 1024, 'onnx_kb': __import__('os').path.getsize(onnx_path) / 1024}}))
"""
    env = dict(os.environ, BL_SPLIT=split)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-3000:]); sys.exit(f"P0 failed on split {split}")
    return json.loads(r.stdout.strip().splitlines()[-1])

rows = [run_split(s, e) for s, e in RUNS.items()]
for r in rows: print(r)
out = ROOT / "phase3" / "results_tinyml.tsv"
hdr = "step\tsplit\trun\tmethod\tval_mape\tval_acc15\tval_rmse\td_mape_vs_fp32\tweights_kb\tnote\n"
if not out.exists(): out.write_text(hdr)
with out.open("a") as fh:
    for r in rows:
        fh.write(f"P0\t{r['split']}\t{r['eid']}\tfp32 PyTorch CPU\t{r['mape_cpu']:.6f}\t{r['acc15']:.4f}\t{r['rmse']:.2f}\t0\t{r['fp32_kb']:.1f}\trecord {r['mape_record']:.6f}\n")
        fh.write(f"P0\t{r['split']}\t{r['eid']}\tfp32 ONNX (onnxruntime)\t{r['mape_onnx']:.6f}\t\t\t{r['mape_onnx']-r['mape_cpu']:+.6f}\t{r['onnx_kb']:.1f}\tmax rel diff {r['max_rel_diff_onnx']:.2e}\n")
