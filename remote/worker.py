"""Remote GPU worker (Colab or Kaggle). Runs queued experiments and writes results to --out.

A job is <queue>/<ID>.json = the experiment proposal plus "split": "2021" | "42" | "2024".
For each job not yet in <out>/<ID>/DONE:
  1. copy --code into a fresh work dir (code + .cache, synced from the Mac),
  2. switch required_device to "cuda", set state.next_experiment_id = ID,
  3. run workflow_preflight.py and train.py with BL_SPLIT=<split>,
  4. copy experiments/<ID>/, the results.tsv row and the run log to <out>/<ID>/, then write DONE
     (or FAILED with the error tail). Recording (results.tsv, research_log, state) is done on the
     Mac by the lead agent after remote/fetch.

Usage:
  python worker.py --code CODE --queue QUEUE --out OUT [--loop] [--poll 60] [--idle-hours 11]
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path


def run_job(job_path, code, out, work_root):
    eid = job_path.stem
    dest = out / eid
    if (dest / "DONE").exists():
        return False
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "FAILED").unlink(missing_ok=True)
    job = json.loads(job_path.read_text())
    split = str(job.pop("split", "2021"))
    work = code.parent / f"_run_{eid}"  # sibling of BatteryLife/ (train.py imports ../BatteryLife)
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(code, work, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    job.setdefault("training", {})["required_device"] = "cuda"
    prop = work / "experiments" / f".{eid}_proposal.json"
    prop.parent.mkdir(parents=True, exist_ok=True)
    prop.write_text(json.dumps(job, indent=1))
    state = json.loads((work / "state.json").read_text())
    state["next_experiment_id"] = eid
    (work / "state.json").write_text(json.dumps(state, indent=2))
    env = dict(os.environ, BL_SPLIT=split)
    log = dest / "run.log"
    print(f"[worker] START {eid} split={split}", flush=True)
    t0 = time.time()
    with log.open("w") as fh:
        rc = 0
        for cmd in ([sys.executable, "workflow_preflight.py", eid, str(prop)],
                    [sys.executable, "train.py", "--experiment-id", eid, "--proposal", str(prop)]):
            p = subprocess.Popen(cmd, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                fh.write(line); fh.flush()
                if line.startswith(("epoch=", "Preflight", "Traceback", "[worker]")) or "Error" in line:
                    print(f"[{eid}] {line.rstrip()}", flush=True)
            rc = p.wait()
            if rc:
                break
    if (work / "experiments" / eid).exists():
        shutil.copytree(work / "experiments" / eid, dest / eid, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
    rows = [l for l in (work / "results.tsv").read_text().splitlines() if l.startswith(eid + "\t")] \
        if (work / "results.tsv").exists() else []
    (dest / "results_row.tsv").write_text("\n".join(rows) + ("\n" if rows else ""))
    meta = {"id": eid, "split": split, "exit_code": rc, "wall_seconds": round(time.time() - t0, 1)}
    try:
        import torch
        meta["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        pass
    (dest / "worker_meta.json").write_text(json.dumps(meta, indent=1))
    if rc:
        (dest / "FAILED").write_text("".join(log.read_text().splitlines(True)[-60:]))
        print(f"[worker] FAILED {eid} rc={rc}", flush=True)
    else:
        (dest / "DONE").write_text(json.dumps(meta))
        print(f"[worker] DONE {eid} in {meta['wall_seconds']}s", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", type=Path, help="project dir (Kaggle)")
    ap.add_argument("--code-tar", type=Path, help="code.tar.gz synced from the Mac (Colab/Drive); re-extracted before every job")
    ap.add_argument("--queue", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("/tmp/remote_work"))
    ap.add_argument("--loop", action="store_true", help="keep polling the queue (Colab)")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--idle-hours", type=float, default=11.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); a.work.mkdir(parents=True, exist_ok=True)
    import torch
    print("[worker] GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (CPU only!)",
          "| torch", torch.__version__, flush=True)
    last = time.time()
    while True:
        did = False
        for job in sorted(a.queue.glob("*.json")):
            if (a.out / job.stem / "DONE").exists() or (a.out / job.stem / "FAILED").exists():
                continue
            code = a.code
            if a.code_tar:
                import tarfile
                ws = a.work / "_ws"
                shutil.rmtree(ws, ignore_errors=True); ws.mkdir(parents=True)
                with tarfile.open(a.code_tar) as t:
                    t.extractall(ws)
                code = next(p for p in ws.iterdir() if p.name != "BatteryLife")
            did = run_job(job, code, a.out, a.work) or did
        if did:
            last = time.time()
        if not a.loop:
            break
        if time.time() - last > a.idle_hours * 3600:
            print("[worker] idle limit reached, exiting", flush=True); break
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
