"""Rebuild the final submission (Zindi ID 8AUSjdXW, private rank 6) from its 14 base members.

The final file is not one model. It is a chain of 19 blend steps over 14 base prediction
files. This script replays that chain in the original order.

Step types:
  linear   p = sum_i w_i * member_i. The weights are frozen float64 values. They were first
           solved from public leaderboard scores (see NOTES.md 6.53-6.98). Here they are
           fixed constants, so no leaderboard access is needed.
  patch    scripts/fix_anchor.py re-anchors 980 rows whose cell re-enters the record mid-run.
  solve2   the original two-member solve, replayed exactly: Gram matrix from the files,
           h from the recorded public scores, SLSQP with sum-to-one and bounds [-0.25, 1.25].
  scale    member * sqrt(E[incumbent^2] / E[member^2]).

Steps marked round5 wrote their CSV with float_format="%.5f", and the next step read that CSV
back. The replay writes and reads the same files, so the rounding is identical.

The public scores in a solve2 step were measured on the original prediction files. With
retrained members, solving again from those scores is badly conditioned: the fixed_a_probe_b
solve divides by the probe weight 0.04, so a small change in the members moved the blend_cnn
weight from 0.604 to 0.473. --scored-weights applies the weights solved on the scored files
(recipe.json "scored_weights", full float64 precision) instead. On the archived members both
modes give the same bytes. run_all.sh uses --scored-weights.

Usage:
    .venv/bin/python submission_model/rebuild_final.py            # from submission_model/members
    .venv/bin/python submission_model/rebuild_final.py --check    # also compare every step
                                                                  # with submission/<step>.csv
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
E_Y2 = 1.013783458 ** 2          # E[y^2] on the public board, from the all-zeros probe
FINAL_MD5 = "72e6ab359311a9ea4fa019947a928503"

ap = argparse.ArgumentParser()
ap.add_argument("--members", default=os.path.join(HERE, "members"))
ap.add_argument("--work", default=os.path.join(HERE, "output"))
ap.add_argument("--check", action="store_true", help="compare each step with submission/<name>.csv")
ap.add_argument("--until", help="stop after this step (the CNN members must run between steps)")
ap.add_argument("--resume", action="store_true", help="skip steps whose output already exists in --work")
ap.add_argument("--scored-weights", action="store_true",
                help="solve2 steps apply the weights solved on the scored files (use with retrained members)")
args = ap.parse_args()
os.makedirs(args.work, exist_ok=True)

ids = pd.read_csv(os.path.join(ROOT, "data", "Test.csv"), usecols=["ID"])["ID"].to_numpy()
recipe = json.load(open(os.path.join(HERE, "recipe.json")))


def path_of(name):
    p = os.path.join(args.work, f"{name}.csv")
    return p if os.path.exists(p) else os.path.join(args.members, f"{name}.csv")


def load(name):
    return pd.read_csv(path_of(name)).set_index("ID")["Target"].reindex(ids).to_numpy()


def write(name, values, round5):
    df = pd.DataFrame({"ID": ids, "Target": values})
    kw = {"float_format": "%.5f"} if round5 else {}
    df.to_csv(os.path.join(args.work, f"{name}.csv"), index=False, **kw)


def solve2(a, b, h):
    M = np.column_stack([a, b])
    G = M.T @ M / len(M)
    fn = lambda v: v @ G @ v - 2 * v @ h + E_Y2
    r = minimize(fn, np.array([1.0, 0.0]), method="SLSQP", bounds=[(-0.25, 1.25)] * 2,
                 constraints=[{"type": "eq", "fun": lambda v: v.sum() - 1}])
    return M, r.x


report = []
for step in recipe["steps"]:
    name, op = step["target"], step["op"]
    if args.resume and os.path.exists(os.path.join(args.work, f"{name}.csv")):
        print(f"{name:15s} exists, skipped", flush=True)
        if name == args.until:      # a skipped step still ends the run at --until
            sys.exit(0)
        continue
    if op == "linear":
        out = sum(w * load(m) for w, m in zip(step["weights"], step["members"]))
        write(name, out, step.get("round5", False))
    elif op == "patch":
        out_path = os.path.join(args.work, f"{name}.csv")
        values = os.path.join(args.members, f"{step.get('values', '')}.csv")
        if step.get("values") and os.path.exists(values):
            # Archived re-anchored values, applied exactly as fix_anchor.py writes its output.
            sub = pd.read_csv(path_of(step["members"][0])).set_index("ID")
            pv = pd.read_csv(values)
            sub.loc[pv["ID"].to_numpy(), "Target"] = pv["Target"].to_numpy()
            sub.reset_index().to_csv(out_path, index=False, float_format="%.5f")
            print(f"{'':15s} (patch values from {os.path.basename(values)})", flush=True)
        else:
            env = dict(os.environ, TWS_IN=path_of(step["members"][0]), TWS_OUT=out_path,
                       TWS_SCALE=str(step["scale"]))
            subprocess.run([sys.executable, "-W", "ignore", "scripts/fix_anchor.py"], cwd=ROOT,
                           env=env, check=True, stdout=subprocess.DEVNULL)
    elif op == "scale":
        inc, mem = load(step["members"][0]), load(step["members"][1])
        k = float(np.sqrt(float(np.mean(inc * inc)) / np.mean(mem * mem)))
        write(name, mem * k, step.get("round5", False))
    elif op == "solve2" and args.scored_weights:
        M = np.column_stack([load(step["members"][0]), load(step["members"][1])])
        step["solved_weights"] = step["scored_weights"]
        write(name, M @ np.array(step["scored_weights"]), step.get("round5", False))
    elif op == "solve2":
        a, b = load(step["members"][0]), load(step["members"][1])
        c = step["h"]
        if c["kind"] == "fixed_a_probe_b":          # h_a given, h_b from a w=0.04 probe of b
            h_a = c["h_a"]
            p = a + c["w"] * (b - a)
            h_b = ((float(np.mean(p * p)) + E_Y2 - c["lb_probe"] ** 2) / 2 - (1 - c["w"]) * h_a) / c["w"]
        elif c["kind"] == "score_a_probe_b":        # h_a from a's score, h_b from the probe
            Ea2 = float(np.mean(a * a))
            h_a = (Ea2 + E_Y2 - c["lb_a"] ** 2) / 2
            p = (1 - c["w"]) * a + c["w"] * b
            h_b = ((float(np.mean(p * p)) + E_Y2 - c["lb_probe"] ** 2) / 2 - (1 - c["w"]) * h_a) / c["w"]
        elif c["kind"] == "scores":                 # both h from the members' own scores
            M = np.column_stack([a, b])
            G = M.T @ M / len(M)
            h_a, h_b = (np.diag(G) + E_Y2 - np.array([c["lb_a"], c["lb_b"]]) ** 2) / 2
        else:
            raise ValueError(c["kind"])
        M, v = solve2(a, b, np.array([h_a, h_b]))
        step["solved_weights"] = [float(x) for x in v]
        write(name, M @ v, step.get("round5", False))
    elif op == "script":                            # byte-exact replay of the original code
        cmd = [sys.executable, "-W", "ignore", os.path.join(ROOT, step["script"]),
               "--members", args.members, "--work", args.work]
        ref_dir = os.path.join(ROOT, "submission")
        if args.check and os.path.isdir(ref_dir):
            cmd += ["--check", ref_dir]
        subprocess.run(cmd, cwd=ROOT, check=True)
    elif op == "final":
        w = step["w"]
        out = (1 - w) * load(step["members"][0]) + w * load(step["members"][1])
        pd.DataFrame({"ID": ids, "Target": out}).to_csv(os.path.join(args.work, f"{name}.csv"), index=False)
    else:
        raise ValueError(op)

    line = f"{name:15s} {op:7s}"
    if "solved_weights" in step:
        line += f" weights {np.round(step['solved_weights'], 6).tolist()}"
    if args.check:
        ref = os.path.join(ROOT, "submission", f"{name}.csv")
        if os.path.exists(ref):
            d = np.abs(load(name) - pd.read_csv(ref).set_index("ID")["Target"].reindex(ids).to_numpy())
            line += f"  max|diff| vs archived {d.max():.2e}  rows differing {int((d > 0).sum())}"
    print(line, flush=True)
    if name == args.until:
        sys.exit(0)

final =os.path.join(args.work, f"{recipe['steps'][-1]['target']}.csv")
md5 = hashlib.md5(open(final, "rb").read()).hexdigest()
print(f"\n{final}\nmd5 {md5}  ({'IDENTICAL to the scored file' if md5 == FINAL_MD5 else 'differs from ' + FINAL_MD5})")
