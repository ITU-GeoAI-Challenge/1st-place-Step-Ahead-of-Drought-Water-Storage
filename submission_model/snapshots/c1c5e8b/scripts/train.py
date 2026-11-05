"""Fit the mask-aware model, validate it against persistence, and write a submission.

Protocol, in the order it runs:

  validate  fit on anchors whose targets land at or before 2009-12, score on the
            era-matched harness folds (anchors 2010-01 onward). Strictly forward in time,
            and the fold set is the one calibrated in scripts/validate_harness.py.
  refit     having decided the model is worth submitting, refit on every anchor in the
            training window and predict the test set. Standard validate-then-refit; the
            submitted model therefore sees more data than the validated one, which is the
            usual and intended asymmetry.

The model is *direct*: lead is a feature and one model covers leads 1-7. Recursive rollout
is permitted too, but direct avoids compounding its own error across six steps and needs
no special casing for the 7-month run.

Training emissions are measured with CodeCarbon because the trustworthiness rubric asks
for emissions measured during training -- it cannot be retrofitted afterwards.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from tws import folds as F
from tws import metrics as M
from tws.features import HistoryStats, build_features, neighbour_matrix
from tws.panel import build_cell_index, build_covariate_panels, build_tws_panel, load_raw, ym_label
import lightgbm as lgb
from codecarbon import EmissionsTracker
import os

TRAIN_END = 2015 * 12 + 9      # last month observable from Train.csv alone
VALID_CUT = 2009 * 12 + 12     # validation model may not see targets after this
ERA_START = 2010 * 12 + 1      # calibrated fold era (see NOTES.md 2.8)

LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.05,
    num_leaves=127,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    num_threads=14,
    verbosity=-1,
)


WITH_PATH = [False]   # set by the caller; keeps the signature stable for older scripts
FEAT_OPTS = [{}]      # extra build_features options (path stats, extra radii)


def training_rows(panel, ym_min, covars, W, cells, max_target: int,
                  cell_frac: float, seed: int = 0, return_anchors: bool = False,
                  min_target: int | None = None):
    """Assemble (anchor, lead) training rows with targets at or before ``max_target``.

    History statistics are rebuilt per anchor from months <= that anchor, so a row never
    sees its own future even though the fit pools every anchor together.

    ``min_target`` additionally drops anchors BEFORE a month, which refitting to the end of
    the window (+0.087, the largest gain in the project) suggests is worth testing as a hard
    cut rather than as the recency weighting that was already rejected. Default None keeps
    the existing behaviour exactly, so no caller changes meaning.
    """
    rng = np.random.default_rng(seed)
    obs = F.observable_months(panel, ym_min)
    obs = obs[obs <= max_target]
    if min_target is not None:
        obs = obs[obs >= int(min_target)]
    n_cells = panel.shape[0]
    X_parts, y_parts, a_parts = [], [], []
    for run in F.contiguous_runs(obs):
        for i, anchor in enumerate(run):
            max_lead = min(F.MAX_LEAD, len(run) - i - 1)
            if max_lead < 1:
                continue
            stats = HistoryStats(panel, ym_min, upto=anchor)
            cache: dict = {}
            for lead in range(1, max_lead + 1):
                tgt = panel[:, anchor + lead - ym_min]
                anc = panel[:, anchor - ym_min]
                ok = np.flatnonzero(np.isfinite(tgt) & np.isfinite(anc))
                if cell_frac < 1.0:
                    ok = ok[rng.random(len(ok)) < cell_frac]
                if not len(ok):
                    continue
                X_parts.append(build_features(ok, np.full(len(ok), lead), anchor,
                                              panel, ym_min, covars, stats, W, cache,
                                              with_path=WITH_PATH[0],
                                              opts=FEAT_OPTS[0]))
                y_parts.append(tgt[ok])
                a_parts.append(np.full(len(ok), anchor, dtype=np.int32))
    X = pd.concat(X_parts, ignore_index=True)
    y = np.concatenate(y_parts)
    if return_anchors:
        return X, y, np.concatenate(a_parts)
    return X, y


def predict_folds(model, feat_names, panel, ym_min, covars, W, era_start):
    flds = F.era_folds(panel, ym_min, era_start=era_start, train_end=TRAIN_END, min_lead=1)
    out = []
    for f in flds:
        h = F.build_holdout(f, panel, ym_min)
        stats = HistoryStats(panel, ym_min, upto=f.anchor)
        cache: dict = {}
        preds = np.empty(len(h), dtype=np.float64)
        for lead, sub in h.groupby("lead", sort=False):
            idx = sub.index.to_numpy()
            X = build_features(sub["cell_id"].to_numpy(), sub["lead"].to_numpy(),
                               f.anchor, panel, ym_min, covars, stats, W, cache)
            preds[idx] = model.predict(X[feat_names])
        h["pred"] = preds
        out.append(h)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    """Command-line entry of the original mask-aware LightGBM pipeline.

    The base members import this module's row and feature builders; they do not call main().
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--cell-frac", type=float, default=0.5)
    ap.add_argument("--radius", type=float, default=500.0)
    ap.add_argument("--out", default="submission/maskaware_lgbm_v1.csv")
    ap.add_argument("--skip-refit", action="store_true")
    args = ap.parse_args()


    t0 = time.time()
    train, test = load_raw()
    cells = build_cell_index(train, test)
    panel, ym_min = build_tws_panel(train, test, cells)
    covars = build_covariate_panels(train, test, cells, ym_min, panel.shape[1])
    W = neighbour_matrix(cells, radius_km=args.radius)
    print(f"panel {panel.shape}  neighbours nnz={W.nnz:,}  ({time.time()-t0:.0f}s)")

    # ---------------- validation model ----------------
    print(f"\n[validate] fitting on targets <= {ym_label(VALID_CUT)}")
    Xv, yv = training_rows(panel, ym_min, covars, W, cells, VALID_CUT, args.cell_frac)
    print(f"  train rows {len(Xv):,}  features {Xv.shape[1]}")
    feat_names = list(Xv.columns)
    tracker = None
    if EmissionsTracker is not None:
        tracker = EmissionsTracker(project_name="tws-maskaware-lgbm", log_level="error",
                                   output_dir="research", save_to_file=True)
        tracker.start()
    m = lgb.train(LGB_PARAMS, lgb.Dataset(Xv, yv), num_boost_round=args.rounds)
    emissions = tracker.stop() if tracker is not None else None
    print(f"  fitted ({time.time()-t0:.0f}s)" +
          (f"  emissions {emissions*1000:.2f} gCO2eq" if emissions else ""))

    df = predict_folds(m, feat_names, panel, ym_min, covars, W, ERA_START)
    df["persistence"] = df["anchor_tws"]
    zw = M.report(df.assign(pred=0.0))["weighted_rmse"]
    rp = M.report(df, name="persistence", pred_col="persistence", zeros_weighted=zw)
    rm = M.report(df, name="model", pred_col="pred", zeros_weighted=zw)
    print()
    for r in (rp, rm):
        print(M.format_report(r)); print()
    b = M.bootstrap_delta(df, "pred", "persistence", n_boot=1000)
    print(f"model - persistence : {b['delta']:+.5f}  95% CI "
          f"[{b['ci_low']:+.5f}, {b['ci_high']:+.5f}]  "
          f"significant: {'yes' if b['ci_high'] < 0 else 'no'}  ({b['n_anchors']} anchors)")

    imp = pd.Series(m.feature_importance("gain"), index=feat_names).sort_values(ascending=False)
    print("\ntop features by gain:")
    for k, v in imp.head(12).items():
        print(f"  {k:<16} {v/imp.sum()*100:5.1f}%")

    res = {"exp": "maskaware-lgbm-v1", "date": "2026-08-25",
           "cv": f"era_folds({ym_label(ERA_START)}..), fit targets<={ym_label(VALID_CUT)}",
           "rounds": args.rounds, "cell_frac": args.cell_frac, "radius_km": args.radius,
           "n_train_rows": int(len(Xv)), "n_val_rows": int(len(df)),
           "persistence": rp["weighted_rmse"], "model": rm["weighted_rmse"],
           "persistence_proj_lb": rp["projected_lb"], "model_proj_lb": rm["projected_lb"],
           "per_lead_model": rm["per_lead"], "per_lead_persistence": rp["per_lead"],
           "bootstrap": b, "emissions_kgco2e": emissions,
           "top_features": {k: round(float(v / imp.sum()), 4) for k, v in imp.head(12).items()}}
    with open("research/results.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")
    print("\nappended research/results.jsonl")

    if args.skip_refit:
        return

    # ---------------- refit on everything, predict test ----------------
    print(f"\n[refit] fitting on targets <= {ym_label(TRAIN_END)}")
    Xf, yf = training_rows(panel, ym_min, covars, W, cells, TRAIN_END, args.cell_frac, seed=1)
    print(f"  train rows {len(Xf):,}")
    if EmissionsTracker is not None:
        tracker = EmissionsTracker(project_name="tws-maskaware-lgbm-final",
                                   log_level="error", output_dir="research", save_to_file=True)
        tracker.start()
    mf = lgb.train(LGB_PARAMS, lgb.Dataset(Xf[feat_names], yf), num_boost_round=args.rounds)
    if EmissionsTracker is not None:
        tracker.stop()

    sub = predict_test(mf, feat_names, test, cells, panel, ym_min, covars, W)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sub.to_csv(args.out, index=False, float_format="%.5f")  # 5dp: well below noise, keeps the file under upload limits
    print(f"\nwrote {args.out}  rows={len(sub):,}")
    print(sub.head().to_string(index=False))
    print(sub["Target"].describe().to_string())


def predict_test(model, feat_names, test, cells, panel, ym_min, covars, W):
    """Predict every test row, grouping by its run anchor so features are built once."""
    te = test.merge(cells, on=["lat", "lon"], how="left")
    months = np.array(sorted(te["ym"].unique()))
    runs = F.contiguous_runs(months)
    anchor_of = {m: run[0] for run in runs for m in run}
    te["anchor_ym"] = te["ym"].map(anchor_of)
    te["lead"] = te["ym"] - te["anchor_ym"] + 1

    parts = []
    for anchor, grp in te.groupby("anchor_ym", sort=True):
        stats = HistoryStats(panel, ym_min, upto=int(anchor))
        cache: dict = {}
        for lead, sub in grp.groupby("lead", sort=False):
            X = build_features(sub["cell_id"].to_numpy(),
                               sub["lead"].to_numpy().astype(np.int64),
                               int(anchor), panel, ym_min, covars, stats, W, cache)
            parts.append(pd.DataFrame({"ID": sub["ID"].to_numpy(),
                                       "Target": model.predict(X[feat_names])}))
    out = pd.concat(parts, ignore_index=True)
    assert len(out) == len(test), f"row count drift: {len(out)} vs {len(test)}"
    assert out["ID"].is_unique, "duplicate IDs in submission"
    return out


if __name__ == "__main__":
    main()
