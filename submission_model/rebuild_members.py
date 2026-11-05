"""Rebuild base members from code, each at the git commit that produced it.

For every member in members.json this script:
  1. creates a detached git worktree at the member's commit (output/wt/<commit>)
  2. links data/ and .venv/ from the main checkout into the worktree
  3. copies the member's input files (other submission files it reads) into wt/submission/
  4. runs the original command with the original environment variables
  5. copies the result to output/rebuilt/ and compares it with the archived member

Usage:
    .venv/bin/python submission_model/rebuild_members.py                  # all members
    .venv/bin/python submission_model/rebuild_members.py eof fieldcnn      # selected members
    --inputs DIR [DIR ...]   where to look for input files, in order
                             (default: output/rebuilt, members/, submission/)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from tws.carbon import track  # noqa: E402  CodeCarbon around each member; never fatal

OUT = os.path.join(HERE, "output")
REBUILT = os.path.join(OUT, "rebuilt")
LOGS = os.path.join(OUT, "logs")

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="*")
ap.add_argument("--inputs", nargs="*", default=[REBUILT, os.path.join(HERE, "members"), os.path.join(ROOT, "submission")])
args = ap.parse_args()

table = json.load(open(os.path.join(HERE, "members.json")))["members"]
if args.names:
    table = [m for m in table if m["name"] in args.names]
os.makedirs(REBUILT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)
ids = pd.read_csv(os.path.join(ROOT, "data", "Test.csv"), usecols=["ID"])["ID"].to_numpy()


def worktree(commit):
    """A working copy of the code at `commit`.

    The package ships snapshots/<commit>/ (scripts, src and research at that commit), so no git
    history is needed. A fresh copy is made for every run, so the snapshot itself stays clean.
    Without a snapshot, a git worktree is used instead.
    """
    path = os.path.join(OUT, "wt", commit)
    snap = os.path.join(HERE, "snapshots", commit)
    if os.path.isdir(snap):
        if os.path.lexists(path):
            subprocess.run(["git", "worktree", "remove", "--force", path], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            shutil.rmtree(path, ignore_errors=True)
        shutil.copytree(snap, path)
    elif not os.path.exists(path):
        subprocess.run(["git", "worktree", "add", "--detach", path, commit], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for link in ("data", ".venv"):
        dst = os.path.join(path, link)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(ROOT, link), dst)
    os.makedirs(os.path.join(path, "submission"), exist_ok=True)
    return path


def find_input(name):
    for d in args.inputs:
        p = os.path.join(d, f"{name}.csv")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"input {name}.csv not found in {args.inputs}")


def compare(name):
    new = pd.read_csv(os.path.join(REBUILT, f"{name}.csv")).set_index("ID")["Target"].reindex(ids).to_numpy()
    old = pd.read_csv(os.path.join(HERE, "members", f"{name}.csv")).set_index("ID")["Target"].reindex(ids).to_numpy()
    d = new - old
    return {"member": name, "max_abs_diff": float(np.abs(d).max()), "rms_diff": float(np.sqrt(np.mean(d ** 2))),
            "rows_differing": int((np.abs(d) > 0).sum()), "corr": float(np.corrcoef(new, old)[0, 1]),
            "identical": bool(np.array_equal(new, old))}


results = []
for m in table:
    wt = worktree(m["commit"])
    for name in m["inputs"]:
        shutil.copyfile(find_input(name), os.path.join(wt, "submission", f"{name}.csv"))
    argv = [os.path.join(ROOT, ".venv", "bin", "python")] + [a.replace("@ROOT", ROOT) for a in m["args"]]
    env = dict(os.environ, **m["env"])
    t0 = time.time()
    # The tracker measures the whole machine. On a shared machine it also counts other work.
    with open(os.path.join(LOGS, f"{m['name']}.log"), "w") as log, track(f"rebuild-{m['name']}", OUT):
        rc = subprocess.run(argv, cwd=wt, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    minutes = (time.time() - t0) / 60
    if rc != 0:
        print(f"{m['name']:15s} FAILED rc={rc} after {minutes:.1f} min (see output/logs/{m['name']}.log)", flush=True)
        results.append({"member": m["name"], "failed": rc})
        continue
    for name in m["outputs"]:
        shutil.copyfile(os.path.join(wt, "submission", f"{name}.csv"), os.path.join(REBUILT, f"{name}.csv"))
        if not os.path.exists(os.path.join(HERE, "members", f"{name}.csv")):
            # The public repository does not ship the archived members, so there is nothing to compare.
            results.append({"member": name, "commit": m["commit"], "minutes": round(minutes, 1)})
            print(f"{name:15s} commit {m['commit']}  {minutes:6.1f} min  (no archived member to compare)", flush=True)
            continue
        r = compare(name)
        r.update(commit=m["commit"], minutes=round(minutes, 1))
        results.append(r)
        print(f"{name:15s} commit {m['commit']}  {minutes:6.1f} min  identical={r['identical']}  "
              f"max|diff| {r['max_abs_diff']:.2e}  rms {r['rms_diff']:.2e}  corr {r['corr']:.8f}", flush=True)

path = os.path.join(OUT, "rebuild_members_report.json")
old = json.load(open(path)) if os.path.exists(path) else []
keep = [r for r in old if r["member"] not in {x["member"] for x in results}]
json.dump(keep + results, open(path, "w"), indent=1)
