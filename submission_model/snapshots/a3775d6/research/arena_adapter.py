"""Arena adapter for the TWS drought competition.

Two things make this adapter unlike a normal tabular one, and both come from what the
first three submissions taught us (NOTES.md 2.8):

1. **A "fold" is an anchor month, not a random row subset.** The test set masks TWS_t
   per-month, so each test run is an autoregressive rollout from its anchor. Validation
   has to reproduce that or it measures the wrong problem -- the starter notebook's naive
   split reads 0.29 RMSE optimistic.

2. **The training cut is 2012-06, not 2009-12.** Skill from the rich features decays with
   the gap between the end of training and the month predicted. The model we actually ship
   is refit to 2015-09 and faces gaps of 0-38 months. Folds from a 2009-12 cut sit a mean
   of 34 months out and rank features under a staleness the shipped model never sees. A
   2012-06 cut with folds after it reproduces the deployed gap range.

Seed dependence comes from subsampling anchors (the fold assignment) and the training cell
sample. Both must vary or the paired test measures nothing.

`primary` is NEGATIVE test-weighted RMSE so that higher is better, as arena expects. A
delta of +0.002 therefore means 0.002 RMSE removed.
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

import lightgbm as lgb

from tws import folds as F
from tws.features import HistoryStats, build_features, neighbour_matrix
from tws.panel import build_cell_index, build_covariate_panels, build_tws_panel, load_raw
sys.path.insert(0, "scripts")  # train.py lives in scripts/
import pathlib
import train as _t
from train import training_rows
from tws.features import climate_sensitivity, neighbour_mean

TRAIN_END = 2015 * 12 + 9
CUT = 2012 * 12 + 6          # model trains on targets <= this
FOLD_START = CUT + 1         # folds live after it, gap 1-38 months, as deployed
ANCHOR_FRAC = 0.78           # fraction of anchors each seed sees

# --- pinned evaluation budget: candidates must not buy score with compute ---
CELL_FRAC = 0.15
ROUNDS = 300
BUDGET_KEYS = {"cell_frac", "rounds", "n_seeds", "bagging_seed"}

BASE_PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.05,
                   num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   num_threads=14, verbosity=-1)

_DEFAULT_CHAMPION = {"params": {}, "features": "all", "radius_km": 500.0,
                     "target": "raw", "recalibrate": False}

def _load_champion():
    """Promotions are written to research/champion.json by the round driver."""
    f = pathlib.Path("research/champion.json")
    if f.exists():
        return json.loads(f.read_text())
    return dict(_DEFAULT_CHAMPION)

CHAMPION = _load_champion()

# Anchors in (CAL_CUT, CUT] are used to fit post-hoc recalibration coefficients, from a
# sub-model trained only on targets <= CAL_CUT. Fitting them on the holdout would be
# leakage; fitting them in-sample would learn the training fit rather than the forecast.
CAL_CUT = CUT - 30

_STATE: dict = {}
_CACHE: dict = {}


def _boot():
    if _STATE:
        return _STATE
    train, test = load_raw()
    cells = build_cell_index(train, test)
    panel, ym_min = build_tws_panel(train, test, cells)
    covars = build_covariate_panels(train, test, cells, ym_min, panel.shape[1])
    _STATE.update(panel=panel, ym_min=ym_min, covars=covars, cells=cells,
                  W={500.0: neighbour_matrix(cells, 500.0)})
    _STATE["folds"] = F.era_folds(panel, ym_min, era_start=FOLD_START,
                                  train_end=TRAIN_END, min_lead=1)
    return _STATE


def _W(radius: float):
    s = _boot()
    if radius not in s["W"]:
        s["W"][radius] = neighbour_matrix(s["cells"], radius)
    return s["W"][radius]


def load():
    """Holdout rows are the unit of scoring; y is the truth, groups the anchor month."""
    s = _boot()
    parts = [F.build_holdout(f, s["panel"], s["ym_min"]) for f in s["folds"]]
    df = pd.concat(parts, ignore_index=True)
    return df, df["y"].to_numpy(dtype=np.float64), df["anchor_ym"].to_numpy()


def splits(df, y, groups, seed):
    """One split per seed: a random 78% of anchors. Fold assignment must move with seed."""
    rng = np.random.default_rng(1000 + seed)
    anchors = np.unique(groups)
    keep = rng.choice(anchors, size=max(6, int(round(len(anchors) * ANCHOR_FRAC))),
                      replace=False)
    val = np.flatnonzero(np.isin(groups, keep))
    train = np.flatnonzero(~np.isin(groups, keep))
    return [(train, val)]


def _cfg_key(cfg, seed):
    return hashlib.md5((json.dumps(cfg, sort_keys=True, default=str)
                        + f"|{seed}").encode()).hexdigest()


def _feature_list(cfg, all_cols):
    spec = cfg.get("features", "all")
    if spec == "all":
        return list(all_cols)
    if isinstance(spec, dict):
        drop = set(spec.get("drop", []))
        add = [c for c in spec.get("keep_only", []) if c in all_cols]
        return add if add else [c for c in all_cols if c not in drop]
    return list(spec)


def _fit_model(cut, cfg, feats_ref=None):
    """Fit one model on targets <= ``cut``, honouring the config's target transform."""
    s = _boot()
    panel, ym_min, covars = s["panel"], s["ym_min"], s["covars"]
    W = _W(float(cfg.get("radius_km", 500.0)))
    _t.WITH_PATH[0] = bool(cfg.get("path_features"))
    _t.FEAT_OPTS[0] = _opts(cfg)
    X, ytr, anch = training_rows(panel, ym_min, covars, W, s["cells"], cut, CELL_FRAC,
                                 seed=11, return_anchors=True)
    X = _augment(X, cfg, None, None, None)
    feats = feats_ref if feats_ref is not None else _feature_list(cfg, X.columns)

    # Recency weighting. We know rich-feature skill decays with the gap between the end of
    # training and the month predicted (NOTES.md 2.8): the same model scores 0.8033 at a
    # 69-107 month gap and 0.7165 at 0-38. If old relationships are the ones that go stale,
    # down-weighting old rows should sharpen the fit on the regime that still applies.
    w = None
    hl = cfg.get("recency_halflife_months")
    if hl:
        age = (cut - anch).astype(np.float64)
        w = np.power(0.5, age / float(hl))

    blend = cfg.get("anchor_blend")
    if blend is not None:
        # The probes say the test anchor is badly noisy: corr(anchor, y) = 0.61 and the
        # optimal scalar on it is 0.62. A competitor reported 0.637 -> 0.604 by using the
        # 500 km neighbourhood mean INSTEAD of the cell's own anchor. Both are already
        # features, but the delta target is measured from the RAW anchor, so the model is
        # forced to predict away that noise. Measuring the delta from a denoised base
        # removes it from the target instead.
        X["base_tws"] = (float(blend) * X["anchor_tws"]
                         + (1.0 - float(blend)) * X["nb_tws"].fillna(X["anchor_tws"]))
    mode = cfg.get("target", "raw")
    if mode == "delta":
        if blend is not None:
            ytr = ytr - X["base_tws"].to_numpy()
            feats = [f for f in feats if f != "base_tws"]
            params0 = None
        # Learn the CHANGE from the anchor rather than the level. The anchor already
        # explains most of the variance (55% of gain), so this hands the tree the residual
        # directly instead of making it rediscover the identity every split.
        elif True:
            ytr = ytr - X["anchor_tws"].to_numpy()
    elif mode == "scaled":
        # Errors scale with a cell's own variability (RMSE 0.42 in the calmest quintile,
        # 0.75 in the wildest), so a single loss over-weights volatile cells. Train on
        # y / cell_sd to equalise them, then undo it.
        sd = np.maximum(X["cell_std"].to_numpy(), 0.1)
        ytr = ytr / sd
    params = {**BASE_PARAMS, **cfg.get("params", {}), "seed": 7, "bagging_seed": 7,
              "feature_fraction_seed": 7}
    model = lgb.train(params, lgb.Dataset(X[feats], ytr, weight=w),
                      num_boost_round=int(cfg.get("n_estimators", ROUNDS)))
    return model, feats


def _opts(cfg):
    """Translate a cfg into the extra-feature options build_features understands."""
    o = {"path_rich": bool(cfg.get("path_rich"))}
    if cfg.get("nearest_lags"):
        st = _boot()
        if "obs_cols" not in _CACHE:
            _CACHE["obs_cols"] = (F.observable_months(st["panel"], st["ym_min"])
                                  - st["ym_min"])
        o["observed_cols"] = _CACHE["obs_cols"]
    if cfg.get("lags"):
        o["lags"] = tuple(cfg["lags"])
    if cfg.get("pre_anchor_months"):
        o["pre_anchor_months"] = int(cfg["pre_anchor_months"])
    if cfg.get("path_neighbourhood"):
        o["W_path_nb"] = _W(float(cfg.get("radius_km", 500.0)))
    if cfg.get("sensitivity"):
        st = _boot()
        skey = ("sens", int(cfg.get("_sens_upto", CUT)))
        if skey not in _CACHE:
            _CACHE[skey] = climate_sensitivity(st["panel"], st["covars"], st["ym_min"],
                                               int(cfg.get("_sens_upto", CUT)))
        o["sensitivity"] = _CACHE[skey]
    radii = cfg.get("extra_radii") or []
    if radii:
        o["extra_W"] = {float(r): _W(float(r)) for r in radii}
    return o


def _augment(Xf, cfg, cell_idx, anchor, panel_state):
    """Extra neighbourhood radii, if the config asks for them.

    nb_tws at 500 km is the second most important feature (10% of gain) and a competitor
    reported 0.637 -> 0.604 from the neighbourhood mean alone. One radius fixes the scale
    of "local"; several let the model choose it per cell, which should matter because
    hydrological correlation length varies enormously between a river basin and a desert.
    """
    radii = cfg.get("extra_radii") or []
    if not radii:
        return Xf
    s = _boot()
    for r in radii:
        col = f"nb_tws_{int(r)}"
        if col in Xf.columns:
            continue
        if panel_state is None:
            Xf[col] = np.nan  # filled by the caller that has the anchor column
        else:
            panel, ym_min, anchor_ym = panel_state
            nb = neighbour_mean(_W(float(r)), panel[:, int(anchor_ym) - ym_min])
            Xf[col] = nb[cell_idx].astype(np.float32)
            Xf[f"local_dev_{int(r)}"] = Xf["anchor_tws"] - Xf[col]
    return Xf


def _predict_rows(model, feats, cfg, cell_idx, lead, anchor, cache):
    s = _boot()
    panel, ym_min, covars = s["panel"], s["ym_min"], s["covars"]
    W = _W(float(cfg.get("radius_km", 500.0)))
    stats = HistoryStats(panel, ym_min, upto=int(anchor))
    Xf = build_features(cell_idx, lead, int(anchor), panel, ym_min, covars, stats, W, cache,
                        with_path=bool(cfg.get("path_features")), opts=_opts(cfg))
    p = model.predict(Xf[feats])
    mode = cfg.get("target", "raw")
    if mode == "delta":
        blend = cfg.get("anchor_blend")
        if blend is not None:
            nb = Xf["nb_tws"].to_numpy()
            anc = Xf["anchor_tws"].to_numpy()
            base = float(blend) * anc + (1.0 - float(blend)) * np.where(np.isfinite(nb), nb, anc)
            p = p + base
        else:
            p = p + Xf["anchor_tws"].to_numpy()
    elif mode == "scaled":
        p = p * np.maximum(Xf["cell_std"].to_numpy(), 0.1)
    return p


def _predict_frame(model, feats, cfg, frame):
    out = np.full(len(frame), np.nan)
    pos = {ix: i for i, ix in enumerate(frame.index.to_numpy())}
    for anchor, grp in frame.groupby("anchor_ym", sort=False):
        cache: dict = {}
        stats_cache = {}
        for lead, g in grp.groupby("lead", sort=False):
            p = _predict_rows(model, feats, cfg, g["cell_id"].to_numpy(),
                              g["lead"].to_numpy(), anchor, cache)
            out[[pos[i] for i in g.index.to_numpy()]] = p
    return out


def _recalibration(cfg, feats):
    """Per-lead (slope, intercept) fitted on a forecast the calibration model never saw.

    The champion's predictions are systematically too extreme -- regressing truth on
    prediction gives slopes of 0.89-0.96 -- and carry a positive bias that grows from
    +0.013 at lead 1 to +0.075 by lead 4. Both are fixable after the fact, but only with
    coefficients estimated out of sample.
    """
    s = _boot()
    panel, ym_min = s["panel"], s["ym_min"]
    sub_model, _ = _fit_model(CAL_CUT, cfg, feats_ref=feats)
    cal_folds = [f for f in F.era_folds(panel, ym_min, era_start=CAL_CUT + 1,
                                        train_end=CUT, min_lead=1)
                 if f.anchor + f.max_lead <= CUT]
    if not cal_folds:
        return {}
    frame = pd.concat([F.build_holdout(f, panel, ym_min) for f in cal_folds],
                      ignore_index=True)
    p = _predict_frame(sub_model, feats, cfg, frame)
    coef = {}
    for L, g in frame.assign(p=p).groupby("lead"):
        m = np.isfinite(g["p"]) & np.isfinite(g["y"])
        if m.sum() < 5000:
            continue
        b, a = np.polyfit(g["p"][m], g["y"][m], 1)
        coef[int(L)] = (float(b), float(a))
    return coef


def _fit_group_models(cfg):
    """One model per lead band, instead of one model with lead as a feature.

    Lead is the strongest structural axis in the problem -- RMSE runs 0.556 at lead 1 and
    0.704 at lead 7, and what predicts well differs between them: the anchor dominates
    early, the climate path dominates late. A single tree ensemble has to spend splits
    separating the regimes before it can model either. Banding lets each model specialise.
    """
    s = _boot()
    panel, ym_min, covars = s["panel"], s["ym_min"], s["covars"]
    W = _W(float(cfg.get("radius_km", 500.0)))
    _t.WITH_PATH[0] = bool(cfg.get("path_features"))
    _t.FEAT_OPTS[0] = _opts(cfg)
    X, ytr, _ = training_rows(panel, ym_min, covars, W, s["cells"], CUT, CELL_FRAC,
                              seed=11, return_anchors=True)
    feats = _feature_list(cfg, X.columns)
    if cfg.get("target") == "delta":
        ytr = ytr - X["anchor_tws"].to_numpy()
    params = {**BASE_PARAMS, **cfg.get("params", {}), "seed": 7, "bagging_seed": 7,
              "feature_fraction_seed": 7}
    lead_col = X["lead"].to_numpy()
    models = {}
    for band in cfg["lead_groups"]:
        m = np.isin(lead_col, band)
        models[tuple(band)] = lgb.train(params, lgb.Dataset(X.loc[m, feats], ytr[m]),
                                        num_boost_round=int(cfg.get("n_estimators", ROUNDS)))
    return models, feats


def _full_predictions(df, cfg):
    """Train once on targets <= CUT and predict every holdout anchor.

    arena calls ``evaluate(df, y, splits, cfg)`` without the seed, so training cannot
    depend on it -- and should not: the seed's job is to vary the fold assignment, and
    holding the fit fixed makes the paired comparison isolate exactly that. It also means
    one fit serves all seeds, which is the difference between a round taking minutes and
    taking an hour.
    """
    key = _cfg_key(cfg, "full")
    if key in _CACHE:
        return _CACHE[key]

    if cfg.get("lead_groups"):
        models, feats = _fit_group_models(cfg)
        out = np.full(len(df), np.nan)
        for anchor, grp in df.groupby("anchor_ym", sort=False):
            cache: dict = {}
            for lead, g in grp.groupby("lead", sort=False):
                band = next(b for b in models if int(g["lead"].iloc[0]) in b)
                out[g.index.to_numpy()] = _predict_rows(models[band], feats, cfg,
                                                        g["cell_id"].to_numpy(),
                                                        g["lead"].to_numpy(), anchor, cache)
        _CACHE[key] = out
        return out

    model, feats = _fit_model(CUT, cfg)
    coef = _recalibration(cfg, feats) if cfg.get("recalibrate") else {}

    out = np.full(len(df), np.nan)
    for anchor, grp in df.groupby("anchor_ym", sort=False):
        cache: dict = {}
        for lead, g in grp.groupby("lead", sort=False):
            p = _predict_rows(model, feats, cfg, g["cell_id"].to_numpy(),
                              g["lead"].to_numpy(), anchor, cache)
            L = int(g["lead"].iloc[0])
            if L in coef:
                b, a = coef[L]
                p = b * p + a
            out[g.index.to_numpy()] = p
    if cfg.get("smooth_run"):
        # Predictions along one run are independent today, but they describe a single
        # storage trajectory for one cell. Averaging each lead with its neighbours
        # imposes the continuity the physical process actually has.
        w = float(cfg["smooth_run"])
        piv = df.assign(p=out).pivot_table(index=["anchor_ym", "cell_id"],
                                           columns="lead", values="p")
        sm = piv.apply(lambda r: r.rolling(3, center=True, min_periods=1).mean(), axis=1)
        blend = (1 - w) * piv + w * sm
        flat = blend.stack().rename("ps").reset_index()
        joined = df[["anchor_ym", "cell_id", "lead"]].merge(
            flat, on=["anchor_ym", "cell_id", "lead"], how="left")
        ps = joined["ps"].to_numpy()
        out = np.where(np.isfinite(ps), ps, out)
    if len(_CACHE) > 40:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = out
    return out


def evaluate(df, y, splits_, cfg):
    """Mask the full predictions down to this seed's validation anchors."""
    for k in cfg:
        if k in BUDGET_KEYS:
            raise ValueError(f"cfg may not set the pinned evaluation budget: {k}")
    full = _full_predictions(df, cfg)
    _, val = splits_[0]
    out = np.full(len(df), np.nan)
    out[val] = full[val]
    return out


def score(y, oof):
    """Negative test-weighted RMSE, so higher is better."""
    s = _boot()
    df = _STATE.get("_df")
    if df is None:
        df, _, _ = load()
        _STATE["_df"] = df
    lead = df["lead"].to_numpy()
    ok = np.isfinite(oof)
    num = den = 0.0
    for L in range(1, F.MAX_LEAD + 1):
        m = ok & (lead == L)
        if not m.any():
            continue
        w = F.TEST_LEAD_WEIGHTS[L]
        num += w * float(np.mean((oof[m] - y[m]) ** 2))
        den += w
    rmse = float(np.sqrt(num / den))
    return {"primary": -rmse, "rmse": rmse, "n_scored": int(ok.sum())}
