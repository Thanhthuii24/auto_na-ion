"""Phase 3 board, step B3 (v5): static INT8 (QDQ, per-channel) of the ST split models E and H; C part stays fp32.
Calibration uses TRAIN cells only. Evaluates (1) onnxruntime INT8 chain, (2) ST C-model chain on host
(stedgeai validate E_int8 -> its C outputs feed H_int8), on all 600 val prefixes; reports val MAPE and weights size."""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np, torch, onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader, CalibrationMethod
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(Path(__file__).parent))
from st_split_export import c_part, RUNS
ST = "/Applications/ST/STEdgeAI/4.1/Utilities/macarm/stedgeai"; B = ROOT / "phase3" / "board"


def const_to_init(src, dst):
    """Move Constant-node tensors into initializers so the quantizer sees them as weights (per-channel, symmetric)."""
    import onnx
    m = onnx.load(src); keep = []
    for n in m.graph.node:
        if n.op_type == "Constant" and n.attribute[0].name == "value":
            t = onnx.helper.get_attribute_value(n.attribute[0]); t.name = n.output[0]; m.graph.initializer.append(t)
        else: keep.append(n)
    del m.graph.node[:]; m.graph.node.extend(keep); onnx.save(m, dst); return dst

class Reader(CalibrationDataReader):
    def __init__(s, items): s.it = iter(items)
    def get_next(s): return next(s.it, None)

def states_for(h, qd, qc, n, fs):
    st = np.zeros((100, 66), np.float32); st[:n, :32] = h[:n]; st[:n, 32:64] = h[:n] - h[0]
    st[:n, 64] = (qd[:n] / max(qd[0], 1e-3) - 1) * fs; st[:n, 65] = (qc[:n] / max(qc[0], 1e-3) - 1) * fs
    mk = np.zeros((100,), np.float32); mk[:n] = 1; return st, mk

def st_validate(model, npz, out):
    r = subprocess.run([ST, "validate", "-m", str(model), "--target", "stm32h7", "--input-data-type", "float32", "--output-data-type", "float32", "-vi", str(npz), "--workspace", str(out) + "_ws",
                        "--output", str(out)], capture_output=True, text=True)
    w = [l for l in (r.stdout + r.stderr).splitlines() if "weights (ro)" in l or "ERROR" in l or "NOT IMPL" in l]
    z = np.load(out / "network_val_io.npz") if r.returncode == 0 else None
    return r.returncode, w, (z[[k for k in z.files if k.startswith("c_outputs")][0]] if z is not None else None)

if __name__ == "__main__":
    split = os.environ.get("BL_SPLIT", "2021"); eid = RUNS[split]; rng = np.random.default_rng(0)
    G = json.loads((B / f"{eid}_C_params.json").read_text()); fs = G["fade_scale"]
    d = torch.load(ROOT / f".cache/batterylife_naion_split{split}_v1.pt", weights_only=False); sc = d["label_scaler"]
    sE = ort.InferenceSession(str(B / f"{eid}_E.onnx"), providers=["CPUExecutionProvider"])
    # ---- calibration (train only) ----
    calE, calH = [], []
    for c in d["cells"]["train"]:
        X = c["tensor"].numpy(); nmax = min(100, int(np.ceil(c["eol"])) - 1)
        h = np.stack([sE.run(None, {"curve": X[i:i + 1]})[0][0] for i in range(nmax)])
        qd, qc = X[:, 2, 150:].max(1), X[:, 2, :150].max(1)
        for i in rng.choice(nmax, 12, replace=False): calE.append({"curve": X[i:i + 1]})
        for n in rng.choice(np.arange(1, nmax + 1), 12, replace=False):
            st, mk = states_for(h, qd, qc, int(n), fs); calH.append({"states": st[None], "mask": mk[None]})
    for part, cal in (("E", calE), ("H", calH)):
        quantize_static(const_to_init(str(B / f"{eid}_{part}.onnx"), str(B / f"{eid}_{part}_ci.onnx")), str(B / f"{eid}_{part}_int8.onnx"), Reader(cal), quant_format=QuantFormat.QDQ,
                        per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QInt8, op_types_to_quantize=["Conv", "Gemm", "MatMul"], extra_options={"WeightSymmetric": True, "ActivationSymmetric": False},
                        calibrate_method=getattr(CalibrationMethod, os.environ.get("CAL", "MinMax")),
                        nodes_to_exclude=[x for x in os.environ.get("EXCL", "").split(",") if x and part == "H"])
    # ---- onnxruntime INT8 chain on val ----
    qE = ort.InferenceSession(str(B / f"{eid}_E_int8.onnx"), providers=["CPUExecutionProvider"])
    qH = ort.InferenceSession(str(B / f"{eid}_H_int8.onnx"), providers=["CPUExecutionProvider"])
    ape, curves, metas = [], [], []
    for c in d["cells"]["val"]:
        X = c["tensor"].numpy(); nmax = min(100, int(np.ceil(c["eol"])) - 1); curves.append(X[:nmax])
        h = np.stack([qE.run(None, {"curve": X[i:i + 1]})[0][0] for i in range(nmax)])
        qd, qc = X[:, 2, 150:].max(1), X[:, 2, :150].max(1); metas.append((nmax, qd, qc, float(c["eol"])))
        for n in range(1, nmax + 1):
            st, mk = states_for(h, qd, qc, n, fs)
            y = c_part(qd, n, float(qH.run(None, {"states": st[None], "mask": mk[None]})[0][0, 0]), G)
            ape.append(abs(y * sc["scale"] + sc["mean"] - c["eol"]) / c["eol"])
    print(f"[{eid}] onnxruntime INT8 (E+H) + C fp32: val MAPE {np.mean(ape):.6f}")
    # ---- ST C-model chain on host ----
    np.savez(B / f"{eid}_E_val_int8in.npz", m_inputs_1=np.concatenate(curves).astype(np.float32))
    rcE, wE, hC = st_validate(B / f"{eid}_E_int8.onnx", B / f"{eid}_E_val_int8in.npz", B / f"val_{eid}_E_int8")
    print(f"  E_int8 rc={rcE}", *[x.strip()[:110] for x in wE])
    if rcE == 0:
        hC = hC.reshape(-1, 32); k = 0; S, M, meta2 = [], [], []
        for nmax, qd, qc, eol in metas:
            h = hC[k:k + nmax]; k += nmax
            for n in range(1, nmax + 1):
                st, mk = states_for(h, qd, qc, n, fs); S.append(st); M.append(mk); meta2.append((qd, n, eol))
        np.savez(B / f"{eid}_H_val_int8in.npz", m_inputs_1=np.stack(S), m_inputs_2=np.stack(M))
        rcH, wH, yC = st_validate(B / f"{eid}_H_int8.onnx", B / f"{eid}_H_val_int8in.npz", B / f"val_{eid}_H_int8")
        print(f"  H_int8 rc={rcH}", *[x.strip()[:110] for x in wH])
        if rcH == 0:
            a2 = [abs(c_part(qd, n, float(y), G) * sc["scale"] + sc["mean"] - eol) / eol for (qd, n, eol), y in zip(meta2, yC.reshape(-1))]
            print(f"[{eid}] ST C-model INT8 chain (host): val MAPE {np.mean(a2):.6f}")
