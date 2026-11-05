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
from tws.features import (PATH_COVARIATES as PATH_COVARIATES_, HistoryStats,
                          build_features, neighbour_matrix)
from tws.panel import build_cell_index, build_covariate_panels, build_tws_panel, load_raw
sys.path.insert(0, "scripts")  # train.py lives in scripts/
import pathlib
import train as _t
from train import training_rows
from tws.features import neighbour_mean, climate_sensitivity, PATH_COVARIATES, denoise_panel
from tws.era5 import load_era5_panels, to_anomalies, to_xrank
from tws.era5_subgrid import load_subgrid_panels
import json as _j
from tws.indices import cell_sensitivity, index_panels

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


def _W(radius: float, k: int | None = None, aniso: float = 1.0):
    s = _boot()
    key = (radius, k, aniso)
    if key not in s["W"]:
        s["W"][key] = neighbour_matrix(s["cells"], radius, k=k, aniso=aniso)
    return s["W"][key]


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
    panel, ym_min, covars = s["panel"], s["ym_min"], _covars(cfg)
    W = _Wc(cfg)
    _t.WITH_PATH[0] = bool(cfg.get("path_features"))
    _t.FEAT_OPTS[0] = _opts(cfg, phase="train")
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
    if mode == "delta" and cfg.get("anchor_base"):
        ytr = ytr - _base_values(_base_coefs(cfg), X["lead"].to_numpy(),
                                 X["anchor_tws"].to_numpy(), X["nb_tws"].to_numpy(),
                                 cfg.get("base_shrink", 1.0))
    elif mode == "anom_delta":
        # Predict the change in ANOMALY, not the change in level. The delta target
        # y - anchor still contains the seasonal cycle: a cell that always gains water in
        # March has a positive delta every March, and the model must learn that pattern
        # separately for every cell. Measuring from the climatology strips it, which is the
        # same move that took ERA5 from +0.004 to +0.023 (NOTES 6.8) applied to the target.
        ytr = ytr - X["clim_target"].to_numpy() - X["anom_anchor"].to_numpy()
    elif mode == "delta":
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
    params = {**BASE_PARAMS, **_objective(cfg), **cfg.get("params", {}),
              "seed": 7, "bagging_seed": 7, "feature_fraction_seed": 7}
    model = lgb.train(params, lgb.Dataset(X[feats], ytr, weight=w),
                      num_boost_round=int(cfg.get("n_estimators", ROUNDS)))
    return model, feats


def _objective(cfg):
    """A top-level `objective` in a cfg used to be silently DROPPED.

    Only cfg["params"] ever reached LightGBM, so `{"objective": "huber"}` -- logged in
    results.jsonl as experiment `huber_objective`, screened at +0.000207 -- trained a model
    identical to the champion. Its error correlation with the champion measures 1.000000 and
    its RMSE matches to the sixth decimal. That avenue was never tested.

    It matters because objective diversity is the one ensembling axis the corpus names
    explicitly: allstate's 3rd place trained ~70 XGBs across fairobj, reglinear, gblinear,
    gammareg, logreg and poisson purely as a diversity axis, and prudential read at depth
    concludes "diversity of OBJECTIVE at the base level paid; diversity of regressor over an
    already-stacked feature set did not".
    """
    obj = cfg.get("objective")
    if not obj:
        return {}
    out = {"objective": obj}
    if obj in ("huber", "fair", "quantile") and cfg.get("alpha") is not None:
        out["alpha"] = float(cfg["alpha"])
    return out


def _base_coefs(cfg, cut=None):
    """Per-lead least-squares base for the delta target: y ~ b_a*anchor + b_n*nb + c.

    The delta target measures y from the RAW anchor, i.e. it hands the trees a residual
    that still contains the anchor at weight 1.0. The optimal weight is nothing like 1.0:
    regressing truth on the anchor alone gives 0.67 at lead 1 decaying to 0.46 at lead 7,
    and once the 500 km neighbourhood mean is in the regression the cell's own anchor drops
    to 0.17-0.34 while the neighbourhood takes 0.22-0.55. The cell-specific deviation from
    its region is mostly noise; the region is the signal.

    An earlier `anchor_blend` searched w*anchor + (1-w)*nb at a single global w and measured
    nothing (+0.0004). That family cannot reach this optimum: the weights here sum to
    0.5-0.75, not 1, and they move with lead.

    Coefficients are fitted on folds entirely inside the training window (targets <= cut),
    never on the holdout -- which is also how they would be fitted for a real submission.
    """
    cut = CUT if cut is None else int(cut)
    key = ("base", float(cfg.get("radius_km", 500.0)), str(cfg.get("anchor_base")), cut)
    if key in _CACHE:
        return _CACHE[key]
    st = _boot()
    panel, ym_min = st["panel"], st["ym_min"]
    W = _Wc(cfg)
    fit_folds = [f for f in F.era_folds(panel, ym_min, era_start=cut - 60,
                                        train_end=cut, min_lead=1)
                 if f.anchor + f.max_lead <= cut]
    parts = []
    for f in fit_folds:
        h = F.build_holdout(f, panel, ym_min)
        nb = neighbour_mean(W, panel[:, f.anchor - ym_min])
        h["nb_tws"] = nb[h["cell_id"].to_numpy()]
        parts.append(h)
    d = pd.concat(parts, ignore_index=True)
    d = d[np.isfinite(d["nb_tws"])]

    def lstsq(g):
        M = np.column_stack([g["anchor_tws"].to_numpy(np.float64),
                             g["nb_tws"].to_numpy(np.float64), np.ones(len(g))])
        b, *_ = np.linalg.lstsq(M, g["y"].to_numpy(np.float64), rcond=None)
        return tuple(float(x) for x in b)

    if cfg.get("anchor_base") == "lsq_pooled":
        b = lstsq(d)
        coefs = {L: b for L in range(1, F.MAX_LEAD + 1)}
    else:
        coefs = {int(L): lstsq(g) for L, g in d.groupby("lead")}
    _CACHE[key] = coefs
    return coefs


def _base_values(coefs, lead, anchor_tws, nb_tws, shrink=1.0):
    """Evaluate the per-lead base on a row set. Missing neighbours fall back to the anchor."""
    lead = np.asarray(lead, dtype=np.int64)
    a = np.asarray(anchor_tws, dtype=np.float64)
    n = np.asarray(nb_tws, dtype=np.float64)
    n = np.where(np.isfinite(n), n, a)
    out = a.copy()
    for L, (ba, bn, c) in coefs.items():
        m = lead == L
        if m.any():
            out[m] = ba * a[m] + bn * n[m] + c
    return out * float(shrink) if shrink != 1.0 else out


# ERA5 fields worth a path summary. The full set is 9-10 panels; putting every one through
# the 5-statistic path builder plus a neighbourhood mean would add ~60 columns to a 45-column
# matrix, which buys overfitting more than signal. These four span the balance: net input,
# the input term alone, evaporative demand, and the shallowest store.
ERA5_PATH_DEFAULT = ("era5_water_balance", "era5_precipitation",
                     "era5_temperature", "era5_soilwater_1")


def _era5(anom: bool, corners: str = "mean", upto: int | None = None):
    """ERA5 panels, loaded once and cached. Shape matches the supplied covariates exactly.

    The climatology for the anomaly variant is computed over months <= CUT only, so no
    test-period value reaches a statistic a training row can see.
    """
    upto = CUT if upto is None else int(upto)
    key = ("era5", bool(anom), corners, upto)
    if key in _CACHE:
        return _CACHE[key]
    st = _boot()
    raw = load_era5_panels(st["cells"], st["ym_min"], st["panel"].shape[1],
                           corners=corners)
    out = to_anomalies(raw, st["ym_min"], upto) if anom else raw
    _CACHE[key] = out
    return out


def _covars(cfg):
    """The covariate dict a config sees: the supplied panels, plus ERA5 if it asks."""
    st = _boot()
    out = dict(st["covars"])
    if cfg.get("era5"):
        out.update(_era5(bool(cfg.get("era5_anom")),
                         str(cfg.get("era5_corners", "mean")),
                         cfg.get("_era5_upto")))
    if cfg.get("era5_xrank"):
        key = ("xrank", bool(cfg.get("era5_anom")))
        if key not in _CACHE:
            base = _era5(bool(cfg.get("era5_anom")))
            keep = [k for k in base if any(w in k for w in
                    ("water_balance", "precipitation", "temperature", "soilwater_1"))]
            _CACHE[key] = to_xrank({k: base[k] for k in keep})
        out.update(_CACHE[key])
    if cfg.get("era5_subgrid"):
        # Sub-grid structure of the water balance: sd / range / wetfrac over the 16 quarter-
        # degree cells inside each competition cell. Our 1-degree ERA5 cannot express any of
        # it, and TWS is an integral over the cell, so how the water is distributed inside it
        # governs how the cell responds. Anomalised on the same footing as the rest of ERA5,
        # since NOTES 6.8 found the raw levels nearly useless and the anomalies worth 5x.
        sub_upto = int(cfg.get("_era5_upto") or CUT)
        key = ("era5sub", bool(cfg.get("era5_anom")), sub_upto)
        if key not in _CACHE:
            raw = load_subgrid_panels(st["cells"], st["ym_min"], st["panel"].shape[1])
            _CACHE[key] = (to_anomalies(raw, st["ym_min"], sub_upto)
                           if cfg.get("era5_anom") else raw)
        out.update(_CACHE[key])
    names = cfg.get("covar_anom")
    if names:
        # ERA5 as raw levels bought +0.004; the identical fields as anomalies bought +0.023
        # (NOTES 6.8). The supplied covariates never had that treatment. SPEI is standardised
        # by construction, but SOIL_MOISTURE_t is not, so it may carry the same between-cell
        # level problem that made raw ERA5 nearly useless.
        key = ("covar_anom", tuple(sorted(names)))
        if key not in _CACHE:
            sub = {n: st["covars"][n] for n in names if n in st["covars"]}
            _CACHE[key] = to_anomalies(sub, st["ym_min"], CUT)
        out.update(_CACHE[key])
    return out


def _nbshape(cfg):
    """The neighbourhood operator's shape: fixed-radius disc, k-nearest, or an ellipse."""
    return (cfg.get("nb_k"), float(cfg.get("nb_aniso", 1.0)))


def _Wc(cfg):
    """The primary TWS neighbourhood for this config, whatever shape it asked for."""
    k, an = _nbshape(cfg)
    return _W(float(cfg.get("radius_km", 500.0)), k=k, aniso=an)


_HIST_PROFILES = None


def _hist_profiles():
    global _HIST_PROFILES
    if _HIST_PROFILES is None:
        _HIST_PROFILES = list(_j.load(
            open("research/test_history_profiles.json")).values())
    return _HIST_PROFILES


def _opts(cfg, phase="eval"):
    """Translate a cfg into the extra-feature options build_features understands.

    ``phase`` matters only for the simulated history gap, which has to be switchable
    independently for fitting and for scoring: masking the holdout makes the measurement
    honest but also makes it harder, so a candidate that masks both must be compared against
    a champion that masks the evaluation too, or the delta conflates "the model improved"
    with "the test got harder".
    """
    o = {"path_rich": bool(cfg.get("path_rich"))}
    if cfg.get("hist_mask_" + phase) or cfg.get("hist_mask_both"):
        o["hist_mask_profiles"] = _hist_profiles()
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
        o["W_path_nb"] = _Wc(cfg)
    if cfg.get("sensitivity"):
        st = _boot()
        skey = ("sens", int(cfg.get("_sens_upto", CUT)))
        if skey not in _CACHE:
            _CACHE[skey] = climate_sensitivity(st["panel"], st["covars"], st["ym_min"],
                                               int(cfg.get("_sens_upto", CUT)))
        o["sensitivity"] = _CACHE[skey]
    if cfg.get("era5_path"):
        extra = cfg.get("era5_path_fields") or ERA5_PATH_DEFAULT
        suffix = "_anom" if cfg.get("era5_anom") else ""
        o["path_names"] = tuple(PATH_COVARIATES) + tuple(f + suffix for f in extra)
    if cfg.get("path_local_dev"):
        o["path_local_dev"] = True
    r_g = cfg.get("nb_gradient_radius")
    if r_g:
        o["nb_gradient_radius"] = float(r_g)
        o["nb_gradient_W"] = _W(float(r_g))
    if cfg.get("response_terms"):
        o["response_terms"] = tuple(cfg["response_terms"])
    if cfg.get("covar_anom_path"):
        o["path_names"] = tuple(o.get("path_names") or PATH_COVARIATES_) + \
                          tuple(n + "_anom" for n in cfg["covar_anom"])
    r_era5 = cfg.get("era5_nb_radius")
    if r_era5:
        names = o.get("path_names") or ()
        o["W_path_nb_for"] = {n: _W(float(r_era5)) for n in names if n.startswith("era5_")}
    if cfg.get("era5_cum"):
        suffix = "_anom" if cfg.get("era5_anom") else ""
        fields = cfg.get("era5_cum_fields") or ("era5_water_balance",)
        o["cum_names"] = tuple(f + suffix for f in fields)
    if cfg.get("indices"):
        st = _boot()
        names = list(cfg["indices"]) if isinstance(cfg["indices"], (list, tuple)) else None
        key = ("idx", tuple(names) if names else "all", int(cfg.get("_sens_upto", CUT)))
        if key not in _CACHE:
            pan = index_panels(st["ym_min"], st["panel"].shape[1], names)
            upto = int(cfg.get("_sens_upto", CUT)) - st["ym_min"]
            sens = {n: cell_sensitivity(st["panel"], v, upto) for n, v in pan.items()}
            _CACHE[key] = (pan, sens)
        o["indices"], o["index_sens"] = _CACHE[key]
    if cfg.get("svd_rank"):
        st = _boot()
        key = ("svd", int(cfg["svd_rank"]), int(cfg.get("_svd_upto", CUT)))
        if key not in _CACHE:
            _CACHE[key] = denoise_panel(st["panel"], st["ym_min"],
                                        int(cfg.get("_svd_upto", CUT)),
                                        rank=int(cfg["svd_rank"]))
        o["svd_panel"] = _CACHE[key]
    if cfg.get("tws_trend_windows"):
        o["tws_trend_windows"] = tuple(int(w) for w in cfg["tws_trend_windows"])
    ea = cfg.get("extra_aniso") or []
    if ea:
        r0 = float(cfg.get("radius_km", 500.0))
        o["extra_aniso_W"] = {str(a).replace(".", "p"): _W(r0, aniso=float(a))
                              for a in ea}
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
    """Predict one group of rows that share an anchor month and a lead.

    Builds the features for the given cells, runs the model, and turns the output back into a
    TWS value according to the config's target mode.
    """
    s = _boot()
    panel, ym_min, covars = s["panel"], s["ym_min"], _covars(cfg)
    W = _Wc(cfg)
    stats = HistoryStats(panel, ym_min, upto=int(anchor))
    Xf = build_features(cell_idx, lead, int(anchor), panel, ym_min, covars, stats, W, cache,
                        with_path=bool(cfg.get("path_features")), opts=_opts(cfg))
    p = model.predict(Xf[feats])
    mode = cfg.get("target", "raw")
    if mode == "anom_delta":
        p = p + Xf["clim_target"].to_numpy() + Xf["anom_anchor"].to_numpy()
    elif mode == "delta" and cfg.get("anchor_base"):
        p = p + _base_values(_base_coefs(cfg), Xf["lead"].to_numpy(),
                             Xf["anchor_tws"].to_numpy(), Xf["nb_tws"].to_numpy(),
                             cfg.get("base_shrink", 1.0))
    elif mode == "delta":
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
    panel, ym_min, covars = s["panel"], s["ym_min"], _covars(cfg)
    W = _Wc(cfg)
    _t.WITH_PATH[0] = bool(cfg.get("path_features"))
    _t.FEAT_OPTS[0] = _opts(cfg, phase="train")
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


def _recursive_predictions(df, cfg):
    """One-month-ahead model rolled forward, instead of one direct model per lead.

    The organizers permit this explicitly: "recursive forecasting, feeding your own
    predictions forward, is permitted -- the restriction is on future *observations*, not
    model output."

    Why it might beat the direct model: at lead 1 the model reaches RMSE 0.556 against
    0.704 at lead 7, so the one-step map is by far the best-estimated object we have.
    Direct prediction asks a single model to learn seven different maps; recursion applies
    the good one seven times. It also consumes each intervening month's climate AT ITS OWN
    STEP rather than as an aggregate over the whole interval -- strictly more information,
    and every covariate used at step j is dated a+j-1 <= t, so it stays legal.

    The risk is error compounding, which is exactly what the paired test is for.
    """
    s = _boot()
    panel, ym_min, covars = s["panel"], s["ym_min"], s["covars"]
    W = _Wc(cfg)
    _t.WITH_PATH[0] = False
    _t.FEAT_OPTS[0] = {}

    # lead-1 rows only: the one-step transition map
    X, ytr, _ = training_rows(panel, ym_min, covars, W, s["cells"], CUT, CELL_FRAC,
                              seed=11, return_anchors=True)
    m1 = X["lead"].to_numpy() == 1
    feats = [c for c in X.columns if c != "lead"]
    y1 = ytr[m1] - X.loc[m1, "anchor_tws"].to_numpy()
    params = {**BASE_PARAMS, **cfg.get("params", {}), "seed": 7, "bagging_seed": 7,
              "feature_fraction_seed": 7}
    step = lgb.train(params, lgb.Dataset(X.loc[m1, feats], y1),
                     num_boost_round=int(cfg.get("n_estimators", ROUNDS)))

    out = np.full(len(df), np.nan)
    for anchor, grp in df.groupby("anchor_ym", sort=False):
        anchor = int(anchor)
        cells_all = np.arange(panel.shape[0])
        state = panel[:, anchor - ym_min].astype(np.float64).copy()
        max_lead = int(grp["lead"].max())
        want = {int(L): g for L, g in grp.groupby("lead", sort=False)}
        for j in range(1, max_lead + 1):
            month = anchor + j - 1          # the month the step is launched from
            stats = HistoryStats(panel, ym_min, upto=min(month, anchor))
            work = panel[:, anchor - ym_min].copy()
            Xf = build_features(cells_all, np.ones(len(cells_all), dtype=np.int64),
                                month, panel, ym_min, covars, stats, W, {})
            # overwrite the anchor with the rolled-forward state
            Xf["anchor_tws"] = state.astype(np.float32)
            state = state + step.predict(Xf[feats])
            if j in want:
                g = want[j]
                out[g.index.to_numpy()] = state[g["cell_id"].to_numpy()]
    return out


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

    if cfg.get("recursive"):
        out = _recursive_predictions(df, cfg)
        _CACHE[key] = out
        return out

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
    if cfg.get("smooth_pred"):
        # SPATIAL smoothing of the prediction field -- as opposed to smooth_run's
        # temporal smoothing along a run. GRACE's effective resolution is ~300 km against
        # a 1 degree grid, so adjacent cells carry nearly the same signal with largely
        # independent noise. That is exactly why a 500 km neighbourhood mean of the
        # ANCHOR beats the cell's own anchor. The same argument applies to our output:
        # averaging each prediction with its neighbours' should cancel independent error
        # while preserving the shared signal.
        #
        # Legal: operates only on model output. The 24 Aug ruling restricts future
        # OBSERVATIONS, not model output, and explicitly permits spatial filtering.
        w = float(cfg["smooth_pred"])
        Wn = _Wc(cfg)
        piv = df.assign(p=out).pivot_table(index=["anchor_ym", "lead"],
                                           columns="cell_id", values="p")
        cols = piv.columns.to_numpy()
        Wsub = Wn[cols][:, cols]                      # stays sparse; dense would be ~2 GB
        M = piv.to_numpy(dtype=float)
        mask = np.isfinite(M)
        filled = np.where(mask, M, 0.0)
        # (sparse @ dense.T).T keeps the product dense-thin rather than materialising W
        num = (Wsub @ filled.T).T
        den = (Wsub @ mask.astype(float).T).T
        nb = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        blend = np.where(np.isfinite(nb) & mask, (1 - w) * M + w * nb, M)
        sm = pd.DataFrame(blend, index=piv.index, columns=piv.columns)
        flat = sm.stack().rename("ps").reset_index()
        joined = df[["anchor_ym", "lead", "cell_id"]].merge(
            flat, on=["anchor_ym", "lead", "cell_id"], how="left")
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
