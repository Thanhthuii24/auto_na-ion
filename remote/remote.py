"""Submit experiments to a remote GPU and fetch results back (run on the Mac).

Backend is set in remote/config.json:
  {"backend": "drive",  "drive_dir": "<Google Drive for Desktop path>/v5_auto_remote"}      # Colab
  {"backend": "kaggle", "kaggle_user": "...", "kaggle_config_dir": "~/.kaggle_v6", "slug": "v6-auto"}

Commands (from the project root, with the battery python):
  python remote/remote.py sync-code                     # drive: push code + caches to Drive
  python remote/remote.py init-cache                    # kaggle: upload the 3 locked caches once (dataset <slug>-cache)
  python remote/remote.py submit <ID> <proposal.json> --split 2021
  python remote/remote.py status [<ID>]
  python remote/remote.py fetch <ID>                    # copy results into experiments/<ID>, results.tsv, live.log

Submit always sends the current code snapshot, so the run uses exactly the reviewed train.py.
"""
import argparse, base64, io, json, os, shutil, subprocess, sys, tarfile, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT.parent
CFG = json.loads((ROOT / "remote" / "config.json").read_text())
EXCLUDE_TOP = {".git", "experiments", "handoff", "reports", "live.log", "__pycache__", ".claude"}
BL_PARTS = ["models", "layers", "utils", "data_provider/data_split_recorder.py", "data_provider/__init__.py"]


def code_tar(include_cache):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for p in sorted(ROOT.iterdir()):
            if p.name in EXCLUDE_TOP or (p.name == ".cache" and not include_cache):
                continue
            t.add(p, arcname=f"{ROOT.name}/{p.name}", filter=lambda ti: None if "__pycache__" in ti.name else ti)
        for part in BL_PARTS:
            src = WS / "BatteryLife" / part
            if src.exists():
                t.add(src, arcname=f"BatteryLife/{part}", filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return buf.getvalue()


def job_json(eid, proposal, split):
    job = json.loads(Path(proposal).read_text())
    if job.get("id") not in (None, eid):
        sys.exit(f"proposal id {job.get('id')} != {eid}")
    job["split"] = str(split)
    return json.dumps(job, indent=1)


# ---------------- drive (Colab) ----------------
def drive_dir():
    return Path(os.path.expanduser(CFG["drive_dir"]))


def drive_sync_code():
    d = drive_dir(); (d / "queue").mkdir(parents=True, exist_ok=True); (d / "out").mkdir(exist_ok=True)
    tmp = d / "code.tar.gz.part"
    tmp.write_bytes(code_tar(include_cache=True)); tmp.replace(d / "code.tar.gz")
    (d / "code_version.txt").write_text(time.strftime("%F %T") + "\n")
    print("code + caches ->", d / "code.tar.gz")


def drive_submit(eid, proposal, split):
    drive_sync_code()
    (drive_dir() / "queue" / f"{eid}.json").write_text(job_json(eid, proposal, split))
    print(f"queued {eid} (split {split}) -> {drive_dir() / 'queue'}; the Colab worker picks it up")


def drive_status(eid=None):
    d = drive_dir()
    for q in sorted((d / "queue").glob("*.json")):
        if eid and q.stem != eid:
            continue
        o = d / "out" / q.stem
        st = "DONE" if (o / "DONE").exists() else "FAILED" if (o / "FAILED").exists() else \
             "RUNNING" if (o / "run.log").exists() else "QUEUED"
        print(q.stem, st)


def drive_out(eid):
    return drive_dir() / "out" / eid


# ---------------- kaggle ----------------
def kaggle(*args, capture=True):
    env = dict(os.environ, KAGGLE_CONFIG_DIR=os.path.expanduser(CFG["kaggle_config_dir"]))
    exe = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    r = subprocess.run([exe, *args], env=env, capture_output=capture, text=True)
    if r.returncode:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def kernel_slug(eid):
    return f"{CFG['slug']}-{eid.lower()}"


def kaggle_init_cache():
    lock = json.loads((ROOT / "protocol_lock.json").read_text())
    with tempfile.TemporaryDirectory() as td:
        for s, e in lock["split_caches"].items():
            shutil.copy(ROOT / e["path"], td)
        meta = {"title": f"{CFG['slug']}-cache", "id": f"{CFG['kaggle_user']}/{CFG['slug']}-cache",
                "licenses": [{"name": "CC0-1.0"}]}
        Path(td, "dataset-metadata.json").write_text(json.dumps(meta))
        print(kaggle("datasets", "create", "-p", td, "-r", "skip"))
    print("private dataset created; wait until it shows 'ready':",
          f"kaggle datasets status {CFG['kaggle_user']}/{CFG['slug']}-cache")


KERNEL_TEMPLATE = r'''
import base64, io, json, os, subprocess, sys, tarfile, glob, shutil
CODE_B64 = "{code}"
JOB = {job!r}
EID = {eid!r}
ws = "/kaggle/working/ws"
os.makedirs(ws, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64)), mode="r:gz").extractall(ws)
proj = os.path.join(ws, {proj!r})
os.makedirs(os.path.join(proj, ".cache"), exist_ok=True)
for f in glob.glob("/kaggle/input/**/*.pt", recursive=True):
    dst = os.path.join(proj, ".cache", os.path.basename(f))
    if not os.path.exists(dst):
        os.symlink(f, dst)
q = "/kaggle/working/queue"; os.makedirs(q, exist_ok=True)
open(os.path.join(q, EID + ".json"), "w").write(JOB)
rc = subprocess.call([sys.executable, os.path.join(proj, "remote", "worker.py"), "--code", proj,
                      "--queue", q, "--out", "/kaggle/working/out", "--work", "/kaggle/tmp_work"])
shutil.rmtree(ws, ignore_errors=True)
sys.exit(rc)
'''


def kaggle_submit(eid, proposal, split):
    code = base64.b64encode(code_tar(include_cache=False)).decode()
    src = KERNEL_TEMPLATE.format(code=code, job=job_json(eid, proposal, split), eid=eid, proj=ROOT.name)
    with tempfile.TemporaryDirectory() as td:
        Path(td, "run.py").write_text(src)
        meta = {"id": f"{CFG['kaggle_user']}/{kernel_slug(eid)}", "title": kernel_slug(eid),
                "code_file": "run.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": True, "enable_internet": False,
                "dataset_sources": [f"{CFG['kaggle_user']}/{CFG['slug']}-cache"],
                "competition_sources": [], "kernel_sources": []}
        Path(td, "kernel-metadata.json").write_text(json.dumps(meta))
        print(kaggle("kernels", "push", "-p", td))
    print(f"submitted {eid} (split {split}) as kernel {meta['id']}")


def kaggle_status(eid):
    return kaggle("kernels", "status", f"{CFG['kaggle_user']}/{kernel_slug(eid)}").strip()


def kaggle_out(eid):
    td = Path(tempfile.mkdtemp(prefix=f"kaggle_{eid}_"))
    kaggle("kernels", "output", f"{CFG['kaggle_user']}/{kernel_slug(eid)}", "-p", str(td))
    return td / "out" / eid


# ---------------- fetch (both) ----------------
def fetch(eid):
    if CFG["backend"] == "kaggle":
        st = kaggle_status(eid)
        if "complete" not in st.lower() and "error" not in st.lower():
            sys.exit(f"{eid}: {st}")
        src = kaggle_out(eid)
    else:
        src = drive_out(eid)
    if not src.exists():
        sys.exit(f"no output for {eid} at {src}")
    if (src / "FAILED").exists() or not (src / "DONE").exists():
        print((src / "FAILED").read_text() if (src / "FAILED").exists() else "not finished")
        sys.exit(f"{eid}: FAILED / not done (see {src}/run.log)")
    shutil.copytree(src / eid, ROOT / "experiments" / eid, dirs_exist_ok=True)
    shutil.copy(src / "worker_meta.json", ROOT / "experiments" / eid / "worker_meta.json")
    row = (src / "results_row.tsv").read_text()
    res = ROOT / "results.tsv"
    if row.strip() and not any(l.startswith(eid + "\t") for l in res.read_text().splitlines()):
        with res.open("a") as fh:
            fh.write(row)
    with (ROOT / "live.log").open("a") as fh:
        fh.write(f"\n===== REMOTE {CFG['backend'].upper()} {eid} =====\n")
        fh.write((src / "run.log").read_text())
    print(f"fetched {eid} -> experiments/{eid}; results.tsv and live.log updated")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync-code"); sub.add_parser("init-cache")
    s = sub.add_parser("submit"); s.add_argument("eid"); s.add_argument("proposal"); s.add_argument("--split", required=True, choices=["2021", "42", "2024"])
    s = sub.add_parser("status"); s.add_argument("eid", nargs="?")
    s = sub.add_parser("fetch"); s.add_argument("eid")
    a = ap.parse_args()
    k = CFG["backend"] == "kaggle"
    if a.cmd == "sync-code":
        drive_sync_code()
    elif a.cmd == "init-cache":
        kaggle_init_cache()
    elif a.cmd == "submit":
        (kaggle_submit if k else drive_submit)(a.eid, a.proposal, a.split)
    elif a.cmd == "status":
        print(kaggle_status(a.eid)) if k else drive_status(a.eid)
    elif a.cmd == "fetch":
        fetch(a.eid)


if __name__ == "__main__":
    main()
