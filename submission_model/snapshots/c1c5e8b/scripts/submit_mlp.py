"""Production MLP submission: a different function class as a stack member.

Screened at own corr 0.6403 and corr(residual, y) +0.0598 on the holdout, which the calibrated
x2.18 offset (NOTES 6.48/6.54) puts near +0.130 on the board -- the strongest residual signal of
any member built here, covonly included, and with far better own skill.

Everything about the data path delegates to arena_adapter so the config cannot silently drift;
that duplication has caused six defects (NOTES 6.47). Only the learner differs:

  * trees ignore NaN and monotone transforms; an MLP needs imputation and standardisation,
    which is the actual source of the different inductive bias, not an implementation detail
  * trained to TRAIN_END like every other submission, not to the arena's CUT
"""
import os, sys, time
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "research")
import json
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import arena_adapter as A
import tws.folds as F
from tws.panel import load_raw, build_cell_index, build_tws_panel, build_covariate_panels, to_ym
from tws.features import build_features, HistoryStats

CFG = json.load(open(os.environ.get("TWS_CFG", "research/champion.json")))
OUT = os.environ.get("TWS_OUT", "submission/mlp.csv")
TRAIN_END = 2015 * 12 + 9
ITERS = int(os.environ.get("MLP_ITERS", "60"))
NROWS = int(os.environ.get("MLP_ROWS", "2500000"))

train, test = load_raw()
cells = build_cell_index(train, test)
panel, ym_min = build_tws_panel(train, test, cells)
A._STATE.update(panel=panel, ym_min=ym_min, cells=cells,
                covars=build_covariate_panels(train, test, cells, ym_min, panel.shape[1]),
                W={500.0: A.neighbour_matrix(cells, 500.0)})
A._STATE["folds"] = F.era_folds(panel, ym_min, era_start=A.FOLD_START,
                                train_end=TRAIN_END, min_lead=1)
covars = A._covars({**CFG, "_sens_upto": TRAIN_END, "_svd_upto": TRAIN_END,
                    "_era5_upto": TRAIN_END})
W = A._Wc(CFG)

import train as _t
_t.WITH_PATH[0] = bool(CFG.get("path_features"))
_t.FEAT_OPTS[0] = A._opts({**CFG, "_sens_upto": TRAIN_END, "_svd_upto": TRAIN_END},
                          phase="train")
from train import training_rows
t0 = time.time()
X, ytr, _ = training_rows(panel, ym_min, covars, W, cells, TRAIN_END, 0.55,
                          seed=11, return_anchors=True)
feats = A._feature_list(CFG, X.columns)
ytr = ytr - X["anchor_tws"].to_numpy()          # champion's delta target
Xtr = X[feats].to_numpy(np.float32)
print(f"train {Xtr.shape} in {time.time()-t0:.0f}s  feats={len(feats)}", flush=True)

rng = np.random.default_rng(0)
if len(Xtr) > NROWS:
    keep = rng.choice(len(Xtr), size=NROWS, replace=False)
    Xtr, ytr = Xtr[keep], ytr[keep]
med = np.nanmedian(Xtr, axis=0)
prep = lambda M: np.where(np.isfinite(M), M, med)
sc = StandardScaler().fit(prep(Xtr))
Z = sc.transform(prep(Xtr))

# Seed-bagging, measured: averaging seeds lifts BOTH axes -- own corr 0.640 -> 0.659 and
# corr(residual, y) 0.060 -> 0.071 (NOTES 6.57). That is not just variance smoothing; a bag
# whose prediction is less noisy leaves a residual that is smaller AND purer signal, which is
# exactly what a stack member is wanted for. mlp_wide is included despite being the weakest
# member alone: its val R2 is the highest of any net (0.712) and its holdout skill the lowest,
# so it contributes decorrelated error rather than accuracy.
SPECS = [((256, 128), 7), ((256, 128), 8), ((256, 128), 9), ((512, 256, 128), 7)]
nets = []
for hls, seed in SPECS:
    t0 = time.time()
    m = MLPRegressor(hidden_layer_sizes=hls, activation="relu", solver="adam",
                     alpha=1e-4, batch_size=8192, learning_rate_init=1e-3,
                     max_iter=ITERS, early_stopping=True, n_iter_no_change=6,
                     validation_fraction=0.05, random_state=seed)
    m.fit(Z, ytr)
    nets.append(m)
    print(f"  net {hls} seed {seed}: {time.time()-t0:.0f}s iters={m.n_iter_} "
          f"val R2={m.best_validation_score_:.5f}", flush=True)
print(f"bagged {len(nets)} nets", flush=True)

te = test.copy(); te["ym"] = to_ym(te["time"])
te = te.merge(cells, on=["lat", "lon"], how="left")
months = np.array(sorted(te["ym"].unique()))
anchor_of = {m: r[0] for r in F.contiguous_runs(months) for m in r}
te["anchor_ym"] = te["ym"].map(anchor_of)
te["lead"] = te["ym"] - te["anchor_ym"] + 1
te = te.reset_index(drop=True)
pred = np.full(len(te), np.nan)
for anchor, grp in te.groupby("anchor_ym", sort=False):
    stats = HistoryStats(panel, ym_min, upto=int(anchor))
    o = A._opts({**CFG, "_sens_upto": int(anchor), "_svd_upto": int(anchor)}, phase="eval")
    cache = {}
    for lead, g in grp.groupby("lead", sort=False):
        Xf = build_features(g["cell_id"].to_numpy(), g["lead"].to_numpy(), int(anchor),
                            panel, ym_min, covars, stats, W, cache,
                            with_path=bool(CFG.get("path_features")), opts=o)
        M = sc.transform(prep(Xf[feats].to_numpy(np.float32)))
        avg = np.mean([m.predict(M) for m in nets], axis=0)
        pred[g.index.to_numpy()] = avg + Xf["anchor_tws"].to_numpy()

miss = ~np.isfinite(pred)
if miss.any():
    print(f"  filling {miss.sum()} rows with 0 (no anchor)")
    pred[miss] = 0.0
sub = pd.DataFrame({"ID": te["ID"].to_numpy(), "Target": pred})
assert len(sub) == len(test) and sub["ID"].is_unique
sub.to_csv(OUT, index=False, float_format="%.5f")
print(f"wrote {OUT} rows={len(sub):,}  mean={pred.mean():+.4f} sd={pred.std():.4f}")
