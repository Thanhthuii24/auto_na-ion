"""Phase 3 board, step B (v5): host validation of the ST C-models (E and H) with val data, all splits.
1) build val npz (m_inputs_*/m_outputs_* = PyTorch reference), 2) `stedgeai validate --target stm32h7` on host,
3) take the C-model outputs from <out>/network_val_io.npz, run the C-part reference, compute val MAPE vs labels."""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from st_split_export import Enc, Head, c_part, RUNS
ST = "/Applications/ST/STEdgeAI/4.1/Utilities/macarm/stedgeai"
B = ROOT / "phase3" / "board"

def stedgeai_validate(model, npz, out):
    r = subprocess.run([ST, "validate", "-m", str(model), "--target", "stm32h7", "-vi", str(npz), "--workspace", str(out) + "_ws",
                        "--output", str(out)], capture_output=True, text=True)
    txt = r.stdout + r.stderr
    rows = [l for l in txt.splitlines() if "X-cross" in l or "C-model" in l or "ERROR" in l or "error" in l.lower()]
    return r.returncode, rows, out / "network_val_io.npz"

if __name__ == "__main__":
    import train
    split = os.environ.get("BL_SPLIT", "2021"); eid = RUNS[split]
    ck = torch.load(ROOT / f"experiments/{eid}/best_checkpoint.pt", weights_only=False, map_location="cpu")
    m = train.build_model(ck["model"]); m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.requires_grad_(False)
    G = json.loads((B / f"{eid}_C_params.json").read_text())
    d = torch.load(ROOT / f".cache/batterylife_naion_split{split}_v1.pt", weights_only=False); sc = d["label_scaler"]
    enc, head = Enc(m).eval(), Head(m).eval()
    curves, hs, states, masks, ynet, meta = [], [], [], [], [], []
    with torch.no_grad():
        for c in d["cells"]["val"]:
            X = c["tensor"].numpy(); nmax = min(100, int(np.ceil(c["eol"])) - 1)
            h = enc_all = np.stack([enc(torch.from_numpy(X[i:i + 1])).numpy()[0] for i in range(nmax)])
            curves.append(X[:nmax]); hs.append(h)
            qd, qc = X[:, 2, 150:].max(1), X[:, 2, :150].max(1)
            for n in range(1, nmax + 1):
                mk = np.zeros((100,), np.float32); mk[:n] = 1; st = np.zeros((100, 66), np.float32)
                st[:n, :32] = h[:n]; st[:n, 32:64] = h[:n] - h[0]
                st[:n, 64] = (qd[:n] / max(qd[0], 1e-3) - 1) * G["fade_scale"]; st[:n, 65] = (qc[:n] / max(qc[0], 1e-3) - 1) * G["fade_scale"]
                states.append(st); masks.append(mk)
                ynet.append(float(head(torch.from_numpy(st[None]), torch.from_numpy(mk[None]))[0, 0]))
                meta.append((c["filename"], n, float(c["eol"]), qd))
    np.savez(B / f"{eid}_E_val.npz", m_inputs_1=np.concatenate(curves).astype(np.float32), m_outputs_1=np.concatenate(hs).astype(np.float32))
    np.savez(B / f"{eid}_H_val.npz", m_inputs_1=np.stack(states), m_inputs_2=np.stack(masks), m_outputs_1=np.array(ynet, np.float32)[:, None])
    res = {}
    for part in ("E", "H"):
        rc, rows, io = stedgeai_validate(B / f"{eid}_{part}.onnx", B / f"{eid}_{part}_val.npz", B / f"val_{eid}_{part}")
        res[part] = (rc, rows, io); print(f"[{eid} {part}] rc={rc}"); [print("   ", r[:150]) for r in rows[:6]]
    rc, _, io = res["H"]
    if rc == 0 and io.exists():
        z = np.load(io); key = [k for k in z.files if k.startswith("c_outputs")][0]; yc = z[key].reshape(-1)
        ape_c, ape_t = [], []
        for (fn, n, eol, qd), y_c, y_t in zip(meta, yc, ynet):
            for y, lst in ((y_c, ape_c), (y_t, ape_t)):
                e = c_part(qd, n, float(y), G) * sc["scale"] + sc["mean"]; lst.append(abs(e - eol) / eol)
        print(f"[{eid}] val MAPE: ST C-model (host) {np.mean(ape_c):.6f} | PyTorch {np.mean(ape_t):.6f} | diff {np.mean(ape_c) - np.mean(ape_t):+.2e} ({len(ape_c)} prefixes)")
