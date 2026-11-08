"""Lead helper: run a list of approved proposals sequentially on the remote worker.
For each ID: preflight -> submit -> wait DONE -> fetch -> record recomputed metrics in results.tsv
(decision by the proposal's rule vs the split-matched reference; conclusion marked 'Codex analysis pending')."""
import csv, io, json, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = Path.home() / "Library/CloudStorage/GoogleDrive-thanhthuyytltt@gmail.com/My Drive/v5_auto_remote/out"

def metrics(eid):
    z = np.load(ROOT / f"experiments/{eid}/validation_predictions.npz", allow_pickle=True)
    a, p, t = z["absolute_relative_error"], z["prediction"], z["target"]
    return float(a.mean()), float((a <= 0.15).mean()), float(np.sqrt(((p - t) ** 2).mean()))

def set_state(**kw):
    s = json.loads((ROOT / "state.json").read_text()); s.update(kw)
    (ROOT / "state.json").write_text(json.dumps(s, indent=2, ensure_ascii=False))

for eid in sys.argv[1:]:
    prop = ROOT / f"experiments/.{eid}_proposal.json"; p = json.loads(prop.read_text())
    set_state(next_experiment_id=eid, current_experiment=eid, workflow_status="running_remote")
    subprocess.run([PY, "workflow_preflight.py", eid, str(prop)], cwd=ROOT, check=True)
    subprocess.run([PY, "remote/remote.py", "submit", eid, str(prop), "--split", p["split"]], cwd=ROOT, check=True)
    t0 = time.time()
    while not ((OUT / eid / "DONE").exists() or (OUT / eid / "FAILED").exists()):
        if time.time() - t0 > 3600:
            sys.exit(f"{eid}: not finished after 60 min (is Colab running?)")
        if time.time() - t0 > 600 and not (OUT / eid / "run.log").exists():
            sys.exit(f"{eid}: not picked up after 10 min — Colab worker is not running")
        time.sleep(30)
    subprocess.run([PY, "remote/remote.py", "fetch", eid], cwd=ROOT, check=True)
    m, acc, rmse = metrics(eid); rm, racc, rrmse = metrics(p["reference_experiment"])
    rows = list(csv.DictReader(open(ROOT / "results.tsv"), delimiter="\t"))
    hdr = open(ROOT / "results.tsv").readline().rstrip("\n").split("\t")
    for r in rows:
        if r["experiment"] == eid:
            r["run_validity"] = "VALID"
            r["decision"] = "KEEP" if m < rm else "DISCARD"
            r["conclusion"] = (f"{p['model']['display_name']} split {p['split']}: MAPE {m:.4f} vs parent {p['reference_experiment']} "
                               f"{rm:.4f} (Acc@15 {acc:.3f} vs {racc:.3f}, RMSE {rmse:.1f} vs {rrmse:.1f}). Lead record; Codex analysis pending.")
            r["next_step"] = "Family 3-split attribution after all three splits."
    buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=hdr, delimiter="\t", lineterminator="\n")
    w.writeheader(); w.writerows(rows); (ROOT / "results.tsv").write_text(buf.getvalue())
    n = int(eid.split("E")[-1]) + 1
    set_state(next_experiment_id=f"NA-E{n:03d}", current_experiment=None, workflow_status="proposal_ready")
    print(f"{eid}: MAPE {m:.4f} (parent {rm:.4f}) Acc {acc:.3f} RMSE {rmse:.1f}", flush=True)
    subprocess.run(["git", "add", "-A"], cwd=ROOT)
    subprocess.run(["git", "-c", "user.name=Thanhthuii24", "-c", "user.email=thanhthuyytltt@gmail.com",
                    "commit", "-qm", f"{eid} ablation recorded (lead; Codex analysis pending)"], cwd=ROOT)
