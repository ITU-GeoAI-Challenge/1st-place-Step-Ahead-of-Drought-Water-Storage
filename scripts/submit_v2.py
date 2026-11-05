"""Production model from the autoresearch champion, refit to the end of training.

The arena runs at a 2012-06 cut and a 15% cell sample to keep rounds fast. What ships must
use everything: targets to 2015-09 and the full cell sample. That is not a free choice --
the first three submissions measured refitting to the end of train as worth 0.087, because
skill from the rich features decays with the gap between training and the predicted month.

Champion (research/champion.json), all confirmed on held-out seeds:
    delta target            +0.0018  (t=11.4, inside path_plus_delta)
    path features           +0.0189  (t=56.0)
    neighbourhood path      +0.0140  (t=50.0)
    rich+sensitivity+recal  +0.0035  (t=18.9)
    session audit           +0.0352 measured on virgin seeds vs claimed +0.0364
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "research")

import lightgbm as lgb

from tws import folds as F
from tws.features import HistoryStats, build_features, neighbour_matrix
from tws.panel import build_cell_index, build_covariate_panels, build_tws_panel, load_raw

import arena_adapter as A
import train as T
from codecarbon import EmissionsTracker

TRAIN_END = 2015 * 12 + 9
CAL_CUT = TRAIN_END - 30
# Budget. Both are pinned in the ARENA (arena_adapter.BUDGET_KEYS) so that candidates compete
# on ideas rather than compute -- correct for comparing ideas, and it means the PRODUCTION
# budget has never been chosen on evidence. 0.55 of cells at 300 rounds is small for 2.1M rows
# and 75 features; the leaderboard can measure whether it is under-trained.
CELL_FRAC = float(os.environ.get("TWS_CELL_FRAC", "0.55"))
ROUNDS = int(os.environ.get("TWS_ROUNDS", "300"))
OUT = os.environ.get("TWS_OUT", "submission/era5_v5.csv")
# Optimal scalar, solved from a submission's OWN leaderboard score via the probe system.
# It is specific to the predictions it was solved for: 0.92663 belongs to submission 4.
# Applying it to a different model is the double-shrink that made SUBMISSION-5 a regression
# (NOTES 6.6). A new model ships unscaled; solve its k from its score, then apply it next.
RESCALE = float(os.environ.get("TWS_RESCALE", "1.0"))


def covars_for(cfg, covars, cells, ym_min, n_months):
    """Delegate to the arena's own _covars, do not reimplement it.

    This was a hand-maintained copy that knew only about the 1-degree ERA5 pull, so
    `era5_subgrid`, `era5_xrank` and `covar_anom` were all silently ignored in production --
    a build with sub-grid path fields produced a model with exactly the champion's 75 features
    and a prediction correlating 1.00000 with it.

    That is the FIFTH instance of this same defect (see NOTES 6.19 for the previous four, all
    in `opts_for`). Copies of the adapter's config handling drift, silently, every time.
    _covars reads its panels from A._STATE, which main() has already populated with the
    production ones; only the anomaly cut has to be overridden, since the arena fits its
    climatology to CUT and production must fit it to TRAIN_END.
    """
    A._STATE["covars"] = covars
    return A._covars({**cfg, "_sens_upto": TRAIN_END, "_svd_upto": TRAIN_END,
                      "_era5_upto": TRAIN_END})


def opts_for(cfg, panel, covars, ym_min, upto):
    """Delegate to the arena's own _opts, do not reimplement it.

    This function used to be a hand-maintained copy, and that duplication is this project's
    recurring defect: it has now silently dropped `smooth_pred`, the whole ERA5 block, and
    the cfg feature-drop spec on three separate occasions, each caught only by inspecting the
    shipped artefact. By the time this was written it was ALSO missing path_local_dev,
    nb_gradient_radius, response_terms, covar_anom_path, era5_nb_radius, extra_radii and
    era5_xrank -- seven keys the arena can promote and the submission would have ignored.

    _opts reads the panel and covariates from A._STATE, which main() populates with the
    PRODUCTION ones, so the only thing that has to be overridden is the cut: statistics that
    the arena fits to 2012-06 must be fitted to the production cut here.
    """
    cfg = {**cfg, "_sens_upto": int(upto), "_svd_upto": int(upto)}
    return A._opts(cfg, phase="eval")


def fit(cut, cfg, state, feats_ref=None):
    """Fit one LightGBM on training rows with targets up to `cut`.

    The target follows the config. `feats_ref` fixes the feature list to match an earlier model.
    Returns the model and its feature names.
    """
    panel, ym_min, covars, cells = state
    W = A._Wc(cfg)
    T.WITH_PATH[0] = bool(cfg.get("path_features"))
    T.FEAT_OPTS[0] = opts_for(cfg, panel, covars, ym_min, cut)
    X, y = T.training_rows(panel, ym_min, covars, W, cells, cut, CELL_FRAC, seed=11)
    if cfg.get("target") == "delta" and cfg.get("anchor_base"):
        # Delta measured from the fitted [anchor, nb, 1] base rather than the raw anchor,
        # whose implicit weight of 1.0 is roughly triple the optimum (REPORT.md 3.3).
        # Coefficients come from the production cut, not the arena's.
        y = y - A._base_values(A._base_coefs(cfg, cut), X["lead"].to_numpy(),
                               X["anchor_tws"].to_numpy(), X["nb_tws"].to_numpy())
    elif cfg.get("target") == "delta":
        y = y - X["anchor_tws"].to_numpy()
    if feats_ref is not None:
        feats = feats_ref
    else:
        spec = cfg.get("features", "all")
        if isinstance(spec, dict):
            drop = set(spec.get("drop", []))
            feats = [c for c in X.columns if c not in drop]
        else:
            feats = list(X.columns)
    # Align the TRAINING lead distribution with the SCORED one. Training rows are nearly
    # uniform across leads (16.1% at lead 1 down to 12.9% at lead 7) because every anchor
    # contributes whatever leads it can reach, but the metric weights lead 1 at 33.3% and
    # leads 5-7 at 5.5% each. So the fit spends ~2.4x too much capacity on the leads that
    # carry least of the score and 2.07x too little on the one that carries most.
    #
    # This is the move rfcx-species-audio-detection and hms-harmful-brain-activity both used:
    # reweight training toward the distribution actually being scored. It costs nothing and
    # needs no new features.
    w = None
    if cfg.get("lead_weights"):
        lead_tr = X["lead"].to_numpy().astype(np.int64)
        share_tr = np.array([(lead_tr == L).mean() for L in range(1, F.MAX_LEAD + 1)])
        share_te = np.array([F.TEST_LEAD_WEIGHTS[L] for L in range(1, F.MAX_LEAD + 1)], float)
        share_te /= share_te.sum()
        ratio = np.where(share_tr > 0, share_te / np.maximum(share_tr, 1e-9), 1.0)
        w = ratio[lead_tr - 1]
        w = w / w.mean()
        print(f"  lead reweighting: min {w.min():.2f} max {w.max():.2f}")

    # SIXTH instance of the duplication defect: a top-level `objective` in the cfg reached
    # the arena (via A._objective) but never reached production, so every objective-diversity
    # submission trained a model identical to the champion -- obj_regression_l1 measured a
    # residual of 0.0010 against the basis, i.e. exactly spanned. Delegate, never restate.
    params = {**A.BASE_PARAMS, **A._objective(cfg), **cfg.get("params", {}),
              "seed": 7, "bagging_seed": 7, "feature_fraction_seed": 7}
    if cfg.get("objective"):
        print(f"  objective: {params.get('objective')} alpha={params.get('alpha')}")
    print(f"  fit cut={cut}  rows={len(X):,}  feats={len(feats)}")
    return lgb.train(params, lgb.Dataset(X[feats], y, weight=w),
                     num_boost_round=ROUNDS), feats


def predict(model, feats, cfg, frame, state, upto_for_opts):
    panel, ym_min, covars, cells = state
    W = A._Wc(cfg)
    out = np.full(len(frame), np.nan)
    pos = {ix: i for i, ix in enumerate(frame.index.to_numpy())}
    for anchor, grp in frame.groupby("anchor_ym", sort=False):
        stats = HistoryStats(panel, ym_min, upto=int(anchor))
        o = opts_for(cfg, panel, covars, ym_min, int(anchor))
        cache: dict = {}
        for lead, g in grp.groupby("lead", sort=False):
            Xf = build_features(g["cell_id"].to_numpy(), g["lead"].to_numpy(), int(anchor),
                                panel, ym_min, covars, stats, W, cache,
                                with_path=bool(cfg.get("path_features")), opts=o)
            p = model.predict(Xf[feats])
            if cfg.get("target") == "delta" and cfg.get("anchor_base"):
                p = p + A._base_values(A._base_coefs(cfg, upto_for_opts),
                                       Xf["lead"].to_numpy(),
                                       Xf["anchor_tws"].to_numpy(),
                                       Xf["nb_tws"].to_numpy())
            elif cfg.get("target") == "delta":
                p = p + Xf["anchor_tws"].to_numpy()
            out[[pos[i] for i in g.index.to_numpy()]] = p
    return out


def main() -> None:
    """Fit the LightGBM described by the config on all training months, predict every test row,
    apply this script's post-processing and write the submission file.
    """
    cfg_path = os.environ.get("TWS_CFG", "research/champion.json")
    cfg = json.load(open(cfg_path))
    print("config:", cfg_path)
    print("champion:", cfg)
    try:
        tracker = EmissionsTracker(project_name="tws-autoresearch-v2", log_level="error",
                                   output_dir="research", save_to_file=True)
        tracker.start()
    except Exception:
        tracker = None

    train, test = load_raw()
    cells = build_cell_index(train, test)
    panel, ym_min = build_tws_panel(train, test, cells)
    covars = build_covariate_panels(train, test, cells, ym_min, panel.shape[1])
    A._STATE.update(panel=panel, ym_min=ym_min, covars=covars, cells=cells,
                    W={(500.0, None, 1.0): neighbour_matrix(cells, 500.0)})
    covars = covars_for(cfg, covars, cells, ym_min, panel.shape[1])
    state = (panel, ym_min, covars, cells)

    model, feats = fit(TRAIN_END, cfg, state)

    coef = {}
    if cfg.get("recalibrate"):
        sub_model, _ = fit(CAL_CUT, cfg, state, feats_ref=feats)
        cal = [f for f in F.era_folds(panel, ym_min, era_start=CAL_CUT + 1,
                                      train_end=TRAIN_END, min_lead=1)
               if f.anchor + f.max_lead <= TRAIN_END]
        fr = pd.concat([F.build_holdout(f, panel, ym_min) for f in cal], ignore_index=True)
        p = predict(sub_model, feats, cfg, fr, state, CAL_CUT)
        for L, g in fr.assign(p=p).groupby("lead"):
            m = np.isfinite(g["p"]) & np.isfinite(g["y"])
            if m.sum() >= 5000:
                b, a = np.polyfit(g["p"][m], g["y"][m], 1)
                coef[int(L)] = (float(b), float(a))
        print("  recalibration slopes:", {k: round(v[0], 3) for k, v in coef.items()})

    te = test.merge(cells, on=["lat", "lon"], how="left")
    months = np.array(sorted(te["ym"].unique()))
    anchor_of = {m: r[0] for r in F.contiguous_runs(months) for m in r}
    te["anchor_ym"] = te["ym"].map(anchor_of)
    te["lead"] = te["ym"] - te["anchor_ym"] + 1
    te = te.reset_index(drop=True)

    pred = predict(model, feats, cfg, te, state, TRAIN_END)
    for L, (b, a) in coef.items():
        m = te["lead"].to_numpy() == L
        pred[m] = b * pred[m] + a

    # 729 test rows sit on a cell absent from its run's anchor month, so anchor_tws is
    # NaN and a delta-target prediction inherits it. Fall back to that cell's most recent
    # observation at or before the anchor -- strictly backward-looking, so legal -- and to
    # the standardised mean of 0 for the handful with no history at all.
    miss = ~np.isfinite(pred)
    if miss.any():
        cid = te.loc[miss, "cell_id"].to_numpy().astype(int)
        anc = te.loc[miss, "anchor_ym"].to_numpy().astype(int)
        fill = np.zeros(miss.sum())
        for i, (c, a) in enumerate(zip(cid, anc)):
            hist = panel[c, : a - ym_min + 1]
            ok = np.flatnonzero(np.isfinite(hist))
            fill[i] = float(hist[ok[-1]]) if len(ok) else 0.0
        pred[miss] = fill
        print(f"  filled {int(miss.sum())} rows with no anchor TWS "
              f"(last observation carried forward)")

    # Spatial smoothing of the prediction field (champion since round 12, +0.00507 local).
    # Operates only on model output, which the 24 Aug ruling explicitly permits.
    if cfg.get("smooth_pred"):
        w = float(cfg["smooth_pred"])
        Wn = A._W(float(cfg.get("radius_km", 500.0)))
        piv = te.assign(p=pred).pivot_table(index=["anchor_ym", "lead"],
                                            columns="cell_id", values="p")
        cols = piv.columns.to_numpy().astype(int)
        Wsub = Wn[cols][:, cols]
        M = piv.to_numpy(dtype=float)
        mask = np.isfinite(M)
        num = (Wsub @ np.where(mask, M, 0.0).T).T
        den = (Wsub @ mask.astype(float).T).T
        nb = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        blend = np.where(np.isfinite(nb) & mask, (1 - w) * M + w * nb, M)
        flat = pd.DataFrame(blend, index=piv.index,
                            columns=piv.columns).stack().rename("ps").reset_index()
        joined = te[["anchor_ym", "lead", "cell_id"]].merge(
            flat, on=["anchor_ym", "lead", "cell_id"], how="left")
        ps = joined["ps"].to_numpy()
        n_sm = int(np.isfinite(ps).sum())
        pred = np.where(np.isfinite(ps), ps, pred)
        print(f"  spatial smoothing w={w} applied to {n_sm:,} of {len(pred):,} rows")

    # Optimal rescale, derived analytically from the probe system rather than fitted:
    # k = E[py]/E[p^2], with E[py] recovered from the previous submission's own score.
    # Worth +0.0023 there, which is more than the whole feature session bought.
    if RESCALE != 1.0:
        pred = RESCALE * pred
        print(f"  rescaled by k={RESCALE}")
    else:
        print("  no rescale: k is unknown for a model that has never been scored")

    sub = pd.DataFrame({"ID": te["ID"].to_numpy(), "Target": pred})
    assert len(sub) == len(test) and sub["ID"].is_unique and np.isfinite(sub["Target"]).all()
    samp = pd.read_csv("data/SampleSubmission.csv")
    assert set(sub["ID"]) == set(samp["ID"]), "ID set mismatch vs SampleSubmission"
    sub.to_csv(OUT, index=False, float_format="%.5f")
    if tracker is not None:
        tracker.stop()
    print(f"\nwrote {OUT}  rows={len(sub):,}")
    print(sub["Target"].describe().to_string())


if __name__ == "__main__":
    main()
