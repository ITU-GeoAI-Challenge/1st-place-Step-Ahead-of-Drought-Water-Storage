"""Recursive one-step forecasting for the TEST set — a genuinely different predictor.

Explicitly permitted (ruling #34403): "Recursive forecasting that feeds your own predictions
forward -- permitted; the restriction is on future *observations*, not model output."

Why it should be decorrelated from the direct model, which is what the LB-moment stack needs:
the direct model learns seven different anchor->target maps at once and is scored worst at the
long leads (local RMSE 0.556 at lead 1 against 0.704 at lead 7). Recursion learns only the
one-step map -- by far the best-estimated object available -- and applies it repeatedly. It
also consumes each intervening month's climate AT ITS OWN STEP rather than as an aggregate
over the whole anchor->row interval, which is strictly more information. Every covariate used
at step j is dated a+j-1 <= t, so it stays legal.

The failure mode is error compounding. That is precisely what the leaderboard measurement is
for; local CV has mispredicted sign twice (NOTES 6.23) and is not trusted here.
"""
import json, os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path[:0] = ["src", "research", "scripts"]
import arena_adapter as A
import train as T
from tws.features import HistoryStats, build_features, neighbour_matrix
from tws.panel import build_cell_index, build_covariate_panels, build_tws_panel, load_raw
from submit_v2 import covars_for, opts_for

TRAIN_END = 2015 * 12 + 9
CELL_FRAC = float(os.environ.get("TWS_CELL_FRAC", "0.55"))
ROUNDS = int(os.environ.get("TWS_ROUNDS", "300"))
OUT = os.environ.get("TWS_OUT", "submission/recursive.csv")

cfg = json.load(open(os.environ.get("TWS_CFG", "research/champion.json")))
print("cfg:", cfg)

train, test = load_raw()
cells = build_cell_index(train, test)
panel, ym_min = build_tws_panel(train, test, cells)
covars = build_covariate_panels(train, test, cells, ym_min, panel.shape[1])
A._STATE.update(panel=panel, ym_min=ym_min, covars=covars, cells=cells,
                W={(500.0, None, 1.0): neighbour_matrix(cells, 500.0)})
covars = covars_for(cfg, covars, cells, ym_min, panel.shape[1])
W = A._Wc(cfg)

# ---- the one-step map, fitted on lead-1 rows only ------------------------------------
T.WITH_PATH[0] = bool(cfg.get("path_features"))
T.FEAT_OPTS[0] = opts_for(cfg, panel, covars, ym_min, TRAIN_END)
X, y = T.training_rows(panel, ym_min, covars, W, cells, TRAIN_END, CELL_FRAC, seed=11)
m1 = X["lead"].to_numpy() == 1
feats = [c for c in X.columns if c != "lead"]
y1 = y[m1] - X.loc[m1, "anchor_tws"].to_numpy()
params = {**A.BASE_PARAMS, **cfg.get("params", {}), "seed": 7, "bagging_seed": 7,
          "feature_fraction_seed": 7}
print(f"one-step fit: {int(m1.sum()):,} lead-1 rows, {len(feats)} features")
step = lgb.train(params, lgb.Dataset(X.loc[m1, feats], y1), num_boost_round=ROUNDS)

# ---- roll forward from each test anchor ----------------------------------------------
anch = sorted(test.groupby("ym")["TWS_t_masked"].mean().pipe(lambda s: s[s < 0.5]).index)
allm = sorted(test["ym"].unique())
run_of = {}
for i, a in enumerate(anch):
    end = anch[i + 1] if i + 1 < len(anch) else max(allm) + 1
    for m in allm:
        if a <= m < end:
            run_of[m] = a
key = test["lat"].round(3).astype(str) + "_" + test["lon"].round(3).astype(str)
ckey = cells["lat"].round(3).astype(str) + "_" + cells["lon"].round(3).astype(str)
cidx = pd.Series(np.arange(len(cells)), index=ckey.to_numpy()).reindex(key.to_numpy()).to_numpy()
test = test.assign(_anchor=test["ym"].map(run_of), _cell=cidx)
test["_lead"] = test["ym"] - test["_anchor"] + 1

out = np.full(len(test), np.nan)
all_cells = np.arange(panel.shape[0])
for a, grp in test.groupby("_anchor", sort=True):
    a = int(a)
    state = panel[:, a - ym_min].astype(np.float64).copy()
    bad = ~np.isfinite(state)
    if bad.any():
        state[bad] = np.nanmean(state)
    stats = HistoryStats(panel, ym_min, upto=a)
    o = opts_for(cfg, panel, covars, ym_min, a)
    want = {int(L): g for L, g in grp.groupby("_lead", sort=False)}
    for j in range(1, int(grp["_lead"].max()) + 1):
        month = a + j - 1                      # the month this step launches from
        Xf = build_features(all_cells, np.ones(len(all_cells), dtype=np.int64), month,
                            panel, ym_min, covars, stats, W, {},
                            with_path=bool(cfg.get("path_features")), opts=o)
        Xf["anchor_tws"] = state.astype(np.float32)
        state = state + step.predict(Xf[feats])
        if j in want:
            g = want[j]
            out[g.index.to_numpy()] = state[g["_cell"].to_numpy()]
    print(f"  anchor {a//12}-{a%12+1:02d}: rolled {int(grp['_lead'].max())} steps, "
          f"{len(grp):,} rows", flush=True)

print("non-finite:", int((~np.isfinite(out)).sum()))
out = np.where(np.isfinite(out), out, 0.0)
pd.DataFrame({"ID": test["ID"], "Target": out}).to_csv(OUT, index=False)
print("wrote", OUT, "sd", round(float(np.std(out)), 4))
