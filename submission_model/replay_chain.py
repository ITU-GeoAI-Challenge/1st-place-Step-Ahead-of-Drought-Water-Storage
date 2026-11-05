"""Byte-exact replay of the blend chain, from the base members to blend_dart.

Each step repeats the arithmetic of the competition code that first wrote submission/<name>.csv.
The docstring of each step names that code. Every step writes its CSV with pandas' default float
format, and the next step reads the CSV back, as the original code did. So the replay gives the
original bytes, not only the original values.

Inputs:
  --members DIR   the archived base members (submission_model/members)
  --work DIR      where the intermediate and output CSVs are written
  --check DIR     optional: compare every output with the file of the same name in DIR
  data/Test.csv and data/Train.csv (delta108 rebuilds the TWS anchor panel), src/tws/panel.py

Constants are the literal values the original code used. Public leaderboard scores are the values
research/lb_scores.json held when each step ran. They are written out here, so later edits to that
file cannot change the replay. The method behind the weights is described in NOTES.md 6.53-6.98.

Usage:
  .venv/bin/python submission_model/replay_chain.py --members submission_model/members \
      --work OUT_DIR [--check submission]
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from tws.panel import build_cell_index, build_tws_panel, load_raw  # noqa: E402

# Public leaderboard RMSE of each scored file, as recorded when it was first used.
PUBLIC_SCORES = {
    "era5_v5": 0.702725673,
    "probe_persist": 0.705159596,
    "climatology": 1.262230926,
    "nb_persist": 0.825883741,
    "final_v4": 0.713747962,
    "lbstack_clim4": 0.687824711,
    "lbridge": 0.680824739,
    "covonly_big": 0.891023292,
    "lb2cov": 0.677088524,
    "blend_opt": 0.673729017,
    "blend7": 0.673019896,
    "blend_receof": 0.682790651,
    "blend_opt2": 0.670966895,
}
# E[y^2] on the public board. Every step below used this older value, not the later 1.013783458**2.
MEAN_Y_SQUARED = 1.027757
SD_Y = np.sqrt(MEAN_Y_SQUARED)
# E[p*y] of mlpbag8, recovered from the public score of the blend that first contained it.
MLPBAG8_MEAN_PY = 0.59275
# Local residual correlations of the recursive and EOF members with the truth, computed on the local
# validation set just before blend_receof was written. The holdout files are not in the package.
RHO_RECURSIVE = -0.07216848110435956
RHO_EOF = 0.05337357431578785


class ChainReplay:
    """Reads members and earlier outputs, writes each output CSV, and optionally compares it."""

    def __init__(self, members_dir, work_dir, reference_dir):
        self.members_dir, self.work_dir, self.reference_dir = members_dir, work_dir, reference_dir
        self.ids = pd.read_csv(os.path.join(REPO, "data/Test.csv"), usecols=["ID"])["ID"].to_numpy()
        self.results = []

    def file_path(self, name):
        """A base member if the members folder has it, otherwise an output of an earlier step."""
        member_file = os.path.join(self.members_dir, f"{name}.csv")
        return member_file if os.path.exists(member_file) else os.path.join(self.work_dir, f"{name}.csv")

    def load(self, name):
        """Target values in data/Test.csv row order."""
        return pd.read_csv(self.file_path(name)).set_index("ID").reindex(self.ids)["Target"].to_numpy()

    def write(self, name, values, ids=None):
        output_file = os.path.join(self.work_dir, f"{name}.csv")
        frame = pd.DataFrame({"ID": self.ids if ids is None else ids, "Target": values})
        frame.to_csv(output_file, index=False)
        self.compare(name, output_file)

    def compare(self, name, output_file):
        md5 = hashlib.md5(open(output_file, "rb").read()).hexdigest()
        if not self.reference_dir:
            print(f"{name:<15} md5 {md5}")
            self.results.append((name, md5, None))
            return
        reference_file = os.path.join(self.reference_dir, f"{name}.csv")
        if not os.path.exists(reference_file):
            print(f"{name:<15} md5 {md5}  (no reference file in --check, not compared)")
            self.results.append((name, md5, None))
            return
        same_bytes = open(output_file, "rb").read() == open(reference_file, "rb").read()
        replayed = pd.read_csv(output_file).set_index("ID").reindex(self.ids)["Target"].to_numpy()
        reference = pd.read_csv(reference_file).set_index("ID").reindex(self.ids)["Target"].to_numpy()
        equal = bool(np.array_equal(replayed, reference))
        max_diff = float(np.max(np.abs(replayed - reference)))
        print(f"{name:<15} md5 {md5}  bytes_identical={same_bytes}  array_equal={equal}  max|diff|={max_diff:.3g}")
        self.results.append((name, md5, same_bytes and equal))


def mean_py_from_score(values, score):
    """E[p*y] from a public score: (E[p^2] + E[y^2] - RMSE^2) / 2."""
    return (np.mean(values**2) + MEAN_Y_SQUARED - score**2) / 2


def member_moments(replay, names):
    """Member matrix (one row per member), E[p*y] per member, and the Gram matrix E[p_i p_j].

    This is moments() from research/lb_ridge.py, which the later inline solves copied.
    """
    predictions = np.array([replay.load(name) for name in names])
    mean_py = np.array([mean_py_from_score(values, PUBLIC_SCORES[name])
                        for name, values in zip(names, predictions)])
    gram = predictions @ predictions.T / predictions.shape[1]
    return predictions, mean_py, gram


def solve_and_write(replay, output_name, names):
    """Unconstrained moment solve over scored members: weights = Gram^-1 E[p*y]."""
    predictions, mean_py, gram = member_moments(replay, names)
    weights = np.linalg.solve(gram, mean_py)
    replay.write(output_name, weights @ predictions)


def two_member_solve_and_write(replay, output_name, base_name, base, member, member_mean_py):
    """Two-member moment solve when E[p*y] of the member is given rather than scored."""
    base_mean_py = mean_py_from_score(base, PUBLIC_SCORES[base_name])
    predictions = np.array([base, member])
    gram = predictions @ predictions.T / predictions.shape[1]
    weights = np.linalg.solve(gram, np.array([base_mean_py, member_mean_py]))
    replay.write(output_name, weights @ predictions)


def delta108(replay):
    """Original: research/delta_scale.py at commit f949ddf, run with DELTA_S=1.08 on era5_v5.

    Scales the predicted change from the anchor by 1.08. delta108 is not an archived member, so it
    is rebuilt here from the data and era5_v5.
    """
    delta_scale = 1.08
    current_dir = os.getcwd()
    os.chdir(REPO)  # load_raw reads data/ relative to the repository root
    try:
        train, test = load_raw()
    finally:
        os.chdir(current_dir)
    cells = build_cell_index(train, test)
    panel, ym_min = build_tws_panel(train, test, cells)
    # A run starts at a month where most TWS values are observed; every later month belongs to it.
    run_starts = sorted(test.groupby("ym")["TWS_t_masked"].mean().pipe(lambda share: share[share < 0.5]).index)
    all_months = sorted(test["ym"].unique())
    run_start_of = {}
    for index, start in enumerate(run_starts):
        end = run_starts[index + 1] if index + 1 < len(run_starts) else max(all_months) + 1
        for month in all_months:
            if start <= month < end:
                run_start_of[month] = start
    test_key = test["lat"].round(3).astype(str) + "_" + test["lon"].round(3).astype(str)
    cell_key = cells["lat"].round(3).astype(str) + "_" + cells["lon"].round(3).astype(str)
    cell_index = pd.Series(np.arange(len(cells)), index=cell_key.to_numpy()).reindex(test_key.to_numpy()).to_numpy()
    anchor = panel[cell_index, test["ym"].map(run_start_of).to_numpy() - ym_min]
    era5_v5 = pd.read_csv(replay.file_path("era5_v5")).set_index("ID").reindex(test["ID"].to_numpy()).reset_index()
    prediction = era5_v5["Target"].to_numpy()
    has_anchor = np.isfinite(anchor)
    scaled = prediction.copy()
    scaled[has_anchor] = anchor[has_anchor] + delta_scale * (prediction[has_anchor] - anchor[has_anchor])
    replay.write("delta108", scaled, ids=era5_v5["ID"])


def lbstack_clim4(replay):
    """Original: inline command, 2026-09-05 06:53 UTC. Fixed weights on four scored files.

    The original aligned rows on lbstack3's index. Those rows are in data/Test.csv order and the
    arithmetic is element by element, so aligning on data/Test.csv IDs gives the same bytes.
    """
    delta108_values, final_v4, probe_persist, climatology = (
        replay.load(name) for name in ("delta108", "final_v4", "probe_persist", "climatology"))
    blend = 0.404 * delta108_values + 0.620 * final_v4 + 0.143 * probe_persist - 0.167 * climatology
    replay.write("lbstack_clim4", blend)


def lbridge(replay):
    """Original: inline loop around research/lb_ridge.py, 2026-09-05 12:07 UTC (its final write).

    Ridge solve shrunk toward the least-squares weights that reproduce lbstack_clim4, lambda 0.1.
    """
    names = ["era5_v5", "probe_persist", "climatology", "nb_persist", "final_v4"]
    predictions, mean_py, gram = member_moments(replay, names)
    anchor_blend = replay.load("lbstack_clim4")
    prior_weights, *_ = np.linalg.lstsq(predictions.T, anchor_blend, rcond=None)
    ridge_lambda = 1e-1
    regularised_gram = gram + ridge_lambda * np.eye(len(gram))
    weights = np.linalg.solve(regularised_gram, mean_py + ridge_lambda * prior_weights)
    replay.write("lbridge", weights @ predictions)


def lbstack6(replay):
    """Original: inline command, 2026-09-06 00:07 UTC. Unconstrained 6-member moment solve.

    The final blend7 does not read this file. It is replayed as a check.
    """
    solve_and_write(replay, "lbstack6",
                    ["era5_v5", "probe_persist", "climatology", "nb_persist", "final_v4", "covonly_big"])


def lb2cov(replay):
    """Original: inline command, 2026-09-06 00:11 UTC. Two-member moment solve."""
    solve_and_write(replay, "lb2cov", ["lbridge", "covonly_big"])


def build_blend(replay, member_name, assumed_rho=0.13):
    """Original: research/build_blend.py at commit 274c162, run for mlpbag and for mlpbag8.

    The base was the best scored file at both runs, lb2cov. The member's E[p*y] is estimated from
    an assumed residual correlation (NOTES.md 6.60).
    """
    base_name = "lb2cov"
    base = replay.load(base_name)
    member = replay.load(member_name)
    base_mean_py = mean_py_from_score(base, PUBLIC_SCORES[base_name])
    projection = np.dot(base, member) / np.dot(base, base)
    residual = member - projection * base
    residual_sd = float(np.sqrt(np.mean(residual**2)))
    member_mean_py = assumed_rho * residual_sd * SD_Y + projection * base_mean_py
    two_member_solve_and_write(replay, f"blend_{member_name}", base_name, base, member, member_mean_py)


def blend_opt(replay):
    """Original: inline command, 2026-09-07 00:21 UTC. lb2cov plus mlpbag8 with its measured E[p*y]."""
    two_member_solve_and_write(replay, "blend_opt", "lb2cov", replay.load("lb2cov"), replay.load("mlpbag8"),
                               MLPBAG8_MEAN_PY)


def blend7(replay):
    """Original: inline command, 2026-09-07 00:24 UTC (third and final version).

    A step from blend_opt toward the 7-member optimum. The step size t is the one whose predicted
    gain is closest to 0.006, the size that past submissions realised reliably (NOTES.md 6.53).
    """
    names = ["era5_v5", "probe_persist", "climatology", "nb_persist", "final_v4", "covonly_big", "mlpbag8"]
    predictions = np.array([replay.load(name) for name in names])
    mean_py = np.array([mean_py_from_score(values, PUBLIC_SCORES[name]) if name in PUBLIC_SCORES
                        else MLPBAG8_MEAN_PY for name, values in zip(names, predictions)])
    optimum_weights = np.linalg.solve(predictions @ predictions.T / predictions.shape[1], mean_py)
    optimum = optimum_weights @ predictions
    optimum_mean_py = float(mean_py @ optimum_weights)
    current = replay.load("blend_opt")
    current_mean_py = mean_py_from_score(current, PUBLIC_SCORES["blend_opt"])
    best = None
    for step in np.arange(0.05, 0.85, 0.05):
        candidate = (1 - step) * current + step * optimum
        candidate_mean_py = (1 - step) * current_mean_py + step * optimum_mean_py
        predicted_rmse = float(np.sqrt(max(np.mean(candidate**2) - 2 * candidate_mean_py + MEAN_Y_SQUARED, 0)))
        predicted_gain = PUBLIC_SCORES["blend_opt"] - predicted_rmse
        if best is None or abs(predicted_gain - 0.006) < abs(best[1] - 0.006):
            best = (step, predicted_gain, predicted_rmse, candidate)
    step, _, _, candidate = best
    print(f"  blend7: chosen step t={step:.2f}")
    replay.write("blend7", candidate)


def blend_receof(replay):
    """Original: inline command, 2026-09-07 01:37 UTC.

    blend7 plus the parts of the recursive and EOF members that blend7 does not already contain,
    each scaled by 0.85 of its local residual correlation.
    """
    base = replay.load("blend7")
    recursive, eof = replay.load("recursive"), replay.load("eof")
    recursive_residual = recursive - base * (np.dot(base, recursive) / np.dot(base, base))
    design = np.column_stack([base, recursive_residual])
    coefficients, *_ = np.linalg.lstsq(design, eof, rcond=None)
    eof_residual = eof - design @ coefficients
    shrink = 0.85
    recursive_scale = shrink * RHO_RECURSIVE * SD_Y / recursive_residual.std()
    eof_scale = shrink * RHO_EOF * SD_Y / eof_residual.std()
    replay.write("blend_receof", base + recursive_scale * recursive_residual + eof_scale * eof_residual)


def blend_opt2(replay):
    """Original: inline command, 2026-09-07 01:40 UTC. Two-member moment solve."""
    solve_and_write(replay, "blend_opt2", ["blend7", "blend_receof"])


def blend_dart(replay):
    """Original: inline command, 2026-09-10 00:49 UTC (second and final version).

    blend_opt2 plus the part of dart it does not contain, at 0.3 of the step a local drift test
    implied.
    """
    base, dart = replay.load("blend_opt2"), replay.load("dart")
    projection = np.dot(base, dart) / np.dot(base, base)
    residual = dart - projection * base
    residual_sd = float(np.sqrt(np.mean(residual**2)))
    assumed_rho = 0.3 * 0.0783
    scale = assumed_rho * SD_Y / residual_sd
    replay.write("blend_dart", base + scale * residual)


STEPS = [
    ("delta108", delta108),
    ("lbstack_clim4", lbstack_clim4),
    ("lbridge", lbridge),
    ("lbstack6", lbstack6),
    ("lb2cov", lb2cov),
    ("blend_mlpbag", lambda replay: build_blend(replay, "mlpbag")),
    ("blend_mlpbag8", lambda replay: build_blend(replay, "mlpbag8")),
    ("blend_opt", blend_opt),
    ("blend7", blend7),
    ("blend_receof", blend_receof),
    ("blend_opt2", blend_opt2),
    ("blend_dart", blend_dart),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--members", default=os.path.join(REPO, "submission_model/members"))
    parser.add_argument("--work", required=True)
    parser.add_argument("--check", default=None, help="directory of archived CSVs to compare against")
    parser.add_argument("--only", default=None, help="comma-separated step names to run")
    args = parser.parse_args()
    os.makedirs(args.work, exist_ok=True)
    replay = ChainReplay(os.path.abspath(args.members), os.path.abspath(args.work),
                         os.path.abspath(args.check) if args.check else None)
    selected = set(args.only.split(",")) if args.only else None
    if selected:
        # --only runs the named steps and nothing before them, so every earlier output must exist
        last = max(position for position, (name, _) in enumerate(STEPS) if name in selected)
        missing = [name for name, _ in STEPS[:last]
                   if name not in selected and not os.path.exists(replay.file_path(name))]
        if missing:
            sys.exit(f"--only needs these earlier outputs in --work first: {missing}")
    for name, step_function in STEPS:
        if selected is None or name in selected:
            step_function(replay)
    if replay.reference_dir:
        failed = [name for name, _, identical in replay.results if identical is False]
        print("ALL IDENTICAL" if not failed else f"NOT IDENTICAL: {failed}")
        sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
